# train.py
# - Stable Diffusion (frozen backbone)
# - DualImageAttnProcessor injected into ALL cross-attn (attn2) blocks
# - Two external condition streams: ID (ArcFace tokens) and Hair (Parsing + CLIP)
# - Text stream ("text") is Arc2Face-style: ArcFace(512) is injected into the CLIP text prompt token "id"
# - Qualitative samples saved as 4-up grid: both_on / id_only / hair_only / both_off
#
# Expected repo structure:
#   src/data/images.py                      -> ImageFolderDataset (returns pixel_values, pil, path)
#   src/model/dual_ip_attention.py          -> DualImageAttnProcessor
#   src/model/id_conditioner_insightface.py -> IDArcFaceConditioner (with extract_arcface_embs)
#   src/model/hair_conditioner_parsing.py   -> HairConditioner
#   src/utils/repro.py                      -> set_seed
#   src/utils/project_face_embs.py          -> project_face_embs (ArcFace->CLIP prompt embeds)
#
# Run:
#   python train.py --cfg config.yaml

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml
import torchvision

from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers import DPMSolverMultistepScheduler

from src.utils.repro import set_seed
from src.data.images import ImageFolderDataset
from src.model.dual_ip_attention import DualImageAttnProcessor
from src.model.clip_text_model_wrapper import CLIPTextModelWrapper
from src.model.id_conditioner_insightface import IDArcFaceConditioner
from src.model.hair_conditioner_parsing import HairConditioner
from src.utils.project_face_embs import project_face_embs


# Data collation (keep PIL)
def collate_keep_pil(batch_list):
    pixel_values = torch.stack([b["pixel_values"] for b in batch_list], dim=0)  # [B,3,H,W] float
    pil = [b["pil"] for b in batch_list]  # list[PIL.Image]
    path = [b["path"] for b in batch_list]
    return {"pixel_values": pixel_values, "pil": pil, "path": path}

@torch.no_grad()
def sanity_check_tokens(pipe, id_cond, hair_cond, dl, dtype_unet, n_samples=4):
    """
    Проверяем, что:
    1) ID токены ненулевые и frozen
    2) Hair токены ненулевые и trainable
    3) Выводим mean и norm для проверки
    """
    batch = next(iter(dl))
    pil_images = batch["pil"][:n_samples]
    B = len(pil_images)

    # ID tokens
    face_embs_512, face_mask = id_cond.extract_arcface_embs(pil_images, return_mask=True)
    id_tokens = id_cond.embs_to_tokens(face_embs_512, out_dtype=dtype_unet)

    # Hair tokens
    hair_tokens = hair_cond(pil_images, out_dtype=dtype_unet)
    hair_tokens = hair_tokens / (hair_tokens.norm(dim=-1, keepdim=True) + 1e-6)

    print("\n=== Sanity Check Tokens ===")
    print(f"face_mask: {face_mask.tolist()}")
    print(f"ID tokens mean abs: {id_tokens.abs().mean().item():.4f}, norm mean: {id_tokens.norm(dim=-1).mean().item():.4f}")
    print(f"Hair tokens mean abs: {hair_tokens.abs().mean().item():.4f}, norm mean: {hair_tokens.norm(dim=-1).mean().item():.4f}")

    # Проверка заморозки
    id_grad = any(p.requires_grad for p in id_cond.parameters())
    hair_grad = any(p.requires_grad for p in hair_cond.parameters())
    print(f"ID conditioner trainable? {id_grad} (должно быть False)")
    print(f"Hair conditioner trainable? {hair_grad} (должно быть True)")

    # Пример: нулевые ID для отсутствующих лиц
    if (~face_mask).any():
        print("Нулевая fallback подставлена для отсутствующих лиц:", (~face_mask).any())


@torch.no_grad()
def _vae_decode_to_01(pipe: StableDiffusionPipeline, latents: torch.Tensor, dtype_unet: torch.dtype):
    """
    latents: [B,4,h,w] in latent space
    return:  [B,3,H,W] in [0,1] fp32
    """
    latents = latents.to(dtype_unet)
    imgs = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample  # [-1, 1]
    imgs_01 = (imgs.float() * 0.5 + 0.5).clamp(0, 1)  # fp32 [0,1]
    return imgs_01


@torch.no_grad()
def _save_row(images_01: torch.Tensor, path: str):
    """
    images_01: (N,3,H,W) in [0,1]
    Saves one row grid.
    """
    grid = torchvision.utils.make_grid(images_01, nrow=images_01.shape[0])
    torchvision.utils.save_image(grid, path)


@torch.no_grad()
def sample_with_cfg(
    pipe,
    scheduler,
    latents,          # (B,4,h,w) стандартный N(0,1)
    enc_cond: dict,
    enc_uncond: dict,
    num_steps: int,
    cfg_scale: float = 7.0,
):
    device = latents.device
    scheduler.set_timesteps(num_steps, device=device)

    # для DPMSolver latents должны быть умножены на init_noise_sigma
    latents = latents * scheduler.init_noise_sigma
    x = latents

    for t in scheduler.timesteps:
        t_int = int(t.item())
        t_batch = torch.full((x.shape[0],), t_int, device=device, dtype=torch.long)

        # scale_model_input нужен для DPMSolver
        x_in = scheduler.scale_model_input(x, t)

        eps_u = pipe.unet(x_in, t_batch, encoder_hidden_states=enc_uncond).sample
        eps_c = pipe.unet(x_in, t_batch, encoder_hidden_states=enc_cond).sample

        eps = eps_u + cfg_scale * (eps_c - eps_u)
        x = scheduler.step(eps, t, x).prev_sample

    return x


@torch.no_grad()
def qualitative_check(
    *,
    step: int,
    run_dir: Path,
    pipe: StableDiffusionPipeline,
    scheduler,
    pixel_values: torch.Tensor,  # [B,3,H,W] in [-1,1] (dtype_unet)
    pil_images,                  # list[PIL]
    text_emb: torch.Tensor,      # [B,T,D] (dtype_unet)
    id_cond,
    hair_cond,
    dtype_unet: torch.dtype,
    num_steps: int,
    seed: int,
):
    """
    Saves: runs/<exp>/samples/step_XXXXXXX.png
    Row: [orig | both_on | id_only | hair_only | both_off]
    """
    out_dir = run_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    B, _, H, W = pixel_values.shape
    vae_sf = pipe.vae_scale_factor if hasattr(pipe, "vae_scale_factor") else 8

    face_mask = torch.ones(len(pil_images), device=pixel_values.device, dtype=torch.bool)

    hair_tokens = hair_cond(pil_images, out_dtype=dtype_unet)
    hair_tokens = hair_tokens / (hair_tokens.norm(dim=-1, keepdim=True) + 1e-6)

    # ID goes only through Arc2Face text stream in this setup.
    with torch.no_grad():
        _, face_mask = id_cond.extract_arcface_embs(pil_images, return_mask=True)
    id_tokens = torch.zeros_like(hair_tokens)
    
    print("has_face:", face_mask.tolist())

    print(
        f"[qual diag] mean|text|={text_emb.detach().float().abs().mean().item():.4f}  "
        f"mean|id|={id_tokens.detach().float().abs().mean().item():.4f}  "
        f"mean|hair|={hair_tokens.detach().float().abs().mean().item():.4f}"
    )
    print(
        f"[qual diag] mean_norm text={text_emb.detach().float().norm(dim=-1).mean().item():.4f}  "
        f"id={id_tokens.detach().float().norm(dim=-1).mean().item():.4f}  "
        f"hair={hair_tokens.detach().float().norm(dim=-1).mean().item():.4f}"
    )

    # Fixed noise
    gen = torch.Generator(device=pixel_values.device)
    gen.manual_seed(int(seed))
    latents0 = torch.randn(
        (B, 4, H // vae_sf, W // vae_sf),
        device=pixel_values.device,
        dtype=dtype_unet,
        generator=gen,
    )

    # unconditional text (для CFG)
    tok_uc = pipe.tokenizer(
        [""] * B,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pixel_values.device)

    with torch.no_grad():
        text_emb_uc = pipe.text_encoder(**tok_uc).last_hidden_state.to(dtype_unet)
        tok_plain = pipe.tokenizer(
            ["a portrait photo of a person"] * B,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(pixel_values.device)
        text_emb_plain = pipe.text_encoder(**tok_plain).last_hidden_state.to(dtype_unet)

    variants = [
        ("both_on",   text_emb,       id_tokens,                   hair_tokens,                   3.0),
        ("id_only",   text_emb,       id_tokens,                   torch.zeros_like(hair_tokens), 3.0),
        ("hair_only", text_emb_plain, id_tokens,                   hair_tokens,                   3.0),
        ("both_off",  text_emb_plain, id_tokens,                   torch.zeros_like(hair_tokens), 3.0),
    ]

    rows = []
    for tag, txt_t, id_t, hair_t, cfg_s in variants:
        enc_cond   = {"text": txt_t, "id": id_t, "hair": hair_t}
        enc_uncond = {"text": text_emb_uc, "id": torch.zeros_like(id_t), "hair": torch.zeros_like(hair_t)}

        lat = sample_with_cfg(
            pipe=pipe,
            scheduler=scheduler,
            latents=latents0.clone(),
            enc_cond=enc_cond,
            enc_uncond=enc_uncond,
            num_steps=num_steps,
            cfg_scale=float(cfg_s),
        )

        img_01 = _vae_decode_to_01(pipe, lat, dtype_unet)  # [B,3,H,W]
        rows.append(img_01[:1])

    orig_01 = (pixel_values[:1].float() * 0.5 + 0.5).clamp(0, 1)
    row = torch.cat([orig_01] + rows, dim=0)

    path = out_dir / f"step_{step:07d}.png"
    _save_row(row, str(path))
    print(f"[qual] saved {path}")


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(cfg_path: str):
    cfg = load_cfg(cfg_path)

    exp_name = cfg["exp_name"]
    seed = int(cfg["seed"])
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "GPU is required for this training script."

    run_dir = Path("runs") / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config + env info
    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "config.yaml").write_text(Path(cfg_path).read_text())

    # Load base model; optionally replace text encoder + UNet with Arc2Face weights.
    base_model_id = cfg["models"]["sd_model_id"]
    arc2face_repo_id = cfg["models"].get("arc2face_repo_id")

    text_encoder = None
    unet_override = None
    if arc2face_repo_id:
        print(f"[init] loading Arc2Face modules from {arc2face_repo_id}")
        text_encoder = CLIPTextModelWrapper.from_pretrained(
            arc2face_repo_id,
            subfolder="encoder",
            torch_dtype=torch.float16,
        )
        unet_override = UNet2DConditionModel.from_pretrained(
            arc2face_repo_id,
            subfolder="arc2face",
            torch_dtype=torch.float16,
        )

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        text_encoder=text_encoder,
        unet=unet_override,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    unet = pipe.unet
    dtype_unet = next(unet.parameters()).dtype  # fp16
    cross_dim = unet.config.cross_attention_dim

    # Inject DualImageAttnProcessor into ALL cross-attn blocks (attn2)
    base_procs = unet.attn_processors
    attn_procs = {}
    n_cross = 0

    for name, base_proc in base_procs.items():
        if name.endswith("attn2.processor"):
            # locate the attention module to read hidden size from its to_q
            m = unet
            for key in name.split(".")[:-1]:
                m = getattr(m, key)
            hidden_size = m.to_q.in_features

            proc = DualImageAttnProcessor(
                base_processor=base_proc,
                hidden_size=hidden_size,
                cross_attention_dim=cross_dim,
                scale_id=float(cfg["cond"]["scale_id"]),
                scale_hair=float(cfg["cond"]["scale_hair"]),
                attn_fp32=True,
            ).to(device=device, dtype=torch.float32)  # keep this module stable in fp32

            # Frozen ID branch should start from meaningful SD projections, not zeros.
            with torch.no_grad():
                proc.to_k_id.weight.copy_(m.to_k.weight.detach().to(proc.to_k_id.weight.dtype))
                proc.to_v_id.weight.copy_(m.to_v.weight.detach().to(proc.to_v_id.weight.dtype))

            attn_procs[name] = proc
            n_cross += 1
        else:
            attn_procs[name] = base_proc

    unet.set_attn_processor(attn_procs)
    print(f"[init] injected DualImageAttnProcessor into {n_cross} cross-attn blocks")

    # Freeze SD backbone
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Build conditioners
    n_tokens = int(cfg["cond"]["n_tokens"])
    clip_id = cfg["models"]["clip_vision_id"]
    hair_w = cfg["models"]["hair_parsing_weights"]

    id_cond = IDArcFaceConditioner(
        n_tokens=n_tokens,
        cross_dim=cross_dim,
        device=device,
        proj_dtype=torch.float32,
        model_root="/content",
    ).to(device)

    hair_cond = HairConditioner(
        clip_vision_id=clip_id,
        n_tokens=n_tokens,
        cross_dim=cross_dim,
        hair_weights_path=hair_w,
        hair_class=int(cfg["cond"].get("hair_class", 17)),
        device=device,
        clip_dtype=torch.float16,
        proj_dtype=torch.float32,
        bg_value=float(cfg["cond"].get("hair_bg_value", 0.0)),
    ).to(device)

    # Shortcut exp: FREEZE ID branch completely
    id_cond.eval()
    id_cond.requires_grad_(False)

    # Data
    ds = ImageFolderDataset(cfg["data"]["train_dir"], image_size=int(cfg["data"]["image_size"]))
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 2)),
        pin_memory=True,
        collate_fn=collate_keep_pil,
        drop_last=True,
    )
    # Пример вызова после инициализации моделей и DataLoader:
    sanity_check_tokens(pipe, id_cond, hair_cond, dl, dtype_unet)

    # Train only hair projection + hair branch inside DualImageAttnProcessor
    hair_cond.requires_grad_(False)
    hair_cond.proj.requires_grad_(True)

    for proc in unet.attn_processors.values():
        if isinstance(proc, DualImageAttnProcessor):
            proc.to_k_id.requires_grad_(False)
            proc.to_v_id.requires_grad_(False)
            proc.to_k_hair.requires_grad_(True)
            proc.to_v_hair.requires_grad_(True)

    dual_params = []
    for proc in unet.attn_processors.values():
        if isinstance(proc, DualImageAttnProcessor):
            dual_params += [p for p in proc.parameters() if p.requires_grad]

    train_params = list(hair_cond.proj.parameters()) + dual_params

    opt = torch.optim.AdamW(
        [
            {"params": list(hair_cond.proj.parameters()), "lr": float(cfg["train"]["lr"])},
            {"params": dual_params, "lr": float(cfg["train"]["lr"]) * 0.05},
        ],
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    def count_trainable(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("trainable id_cond:", count_trainable(id_cond))            # 0
    print("trainable hair_cond:", count_trainable(hair_cond))
    print("trainable hair_proj:", count_trainable(hair_cond.proj))   # > 0

    scaler = torch.amp.GradScaler("cuda", enabled=True)

    fixed_batch = None
    if cfg.get("eval", {}).get("enabled", False):
        fixed_batch = next(iter(dl))

    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    eval_scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Optional sanity compare (text-only standard)
    if cfg.get("eval", {}).get("sanity_compare", False):
        pipe.scheduler = eval_scheduler
        prompt_s = cfg.get("eval", {}).get("prompt", "a portrait photo of a person")
        steps_s = int(cfg["eval"].get("num_inference_steps", 30))
        seed_s = int(cfg["eval"].get("seed", 123))

        tok = pipe.tokenizer([prompt_s], padding="max_length",
                             max_length=pipe.tokenizer.model_max_length,
                             return_tensors="pt").to(device)
        text_emb_s = pipe.text_encoder(**tok).last_hidden_state.to(dtype_unet)

        tok_uc = pipe.tokenizer([""], padding="max_length",
                                max_length=pipe.tokenizer.model_max_length,
                                return_tensors="pt").to(device)
        text_emb_uc_s = pipe.text_encoder(**tok_uc).last_hidden_state.to(dtype_unet)

        id0 = torch.zeros((1, 1, cross_dim), device=device, dtype=dtype_unet)
        h0 = torch.zeros((1, 1, cross_dim), device=device, dtype=dtype_unet)

        enc_c = {"text": text_emb_s, "id": id0, "hair": h0}
        enc_u = {"text": text_emb_uc_s, "id": id0, "hair": h0}

        gen = torch.Generator(device=device).manual_seed(seed_s)
        lat0 = torch.randn((1, 4, 64, 64), device=device, dtype=dtype_unet, generator=gen)

        lat = sample_with_cfg(pipe, eval_scheduler, lat0, enc_c, enc_u, num_steps=steps_s, cfg_scale=7.0)
        img01 = _vae_decode_to_01(pipe, lat, dtype_unet)

        torchvision.utils.save_image(img01, str(run_dir / "sanity_text_only.png"))
        print("[sanity] saved", run_dir / "sanity_text_only.png")

    max_steps = int(cfg["train"]["max_steps"])
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])

    # Modes
    unet.train()
    id_cond.eval()
    hair_cond.train()

    step = 0
    it = iter(dl)

    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        pixel_values = batch["pixel_values"].to(device=device, dtype=dtype_unet)  # [-1,1], fp16
        pil_images = batch["pil"]  # list[PIL]
        B = pixel_values.shape[0]

        # ---- Arc2Face text embedding ----
        with torch.no_grad():
            face_embs_512, face_mask = id_cond.extract_arcface_embs(pil_images, return_mask=True)  # (B,512), (B,)
            text_emb = project_face_embs(pipe, face_embs_512).to(dtype_unet)  # (B,T,H)

            if text_emb.shape[0] != B:
                if text_emb.shape[0] == 1:
                    text_emb = text_emb.repeat(B, 1, 1)
                else:
                    raise RuntimeError(f"text_emb batch mismatch: got {text_emb.shape[0]} vs B={B}")

            # fallback: если лица нет — подставим обычный текст только для этих элементов
            if (~face_mask).any():
                prompt_fb = cfg.get("train", {}).get("prompt", "a portrait photo of a person")
                tok_fb = pipe.tokenizer(
                    [prompt_fb] * B,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(device)
                text_fb = pipe.text_encoder(**tok_fb).last_hidden_state.to(dtype_unet)
                text_emb = text_emb.clone()
                text_emb[~face_mask] = text_fb[~face_mask]

        # VAE encode -> latents
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor  # [B,4,h,w]

        # Add diffusion noise
        noise = torch.randn_like(latents)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
        noisy = scheduler.add_noise(latents, noise, t).to(dtype=dtype_unet)

        # Hair tokens
        hair_tokens = hair_cond(pil_images, out_dtype=dtype_unet)  # [B, n_tokens, cross_dim]
        hair_tokens = hair_tokens / (hair_tokens.norm(dim=-1, keepdim=True) + 1e-6)
        id_tokens = torch.zeros_like(hair_tokens)
        
        # Сборка conditioning
        enc = {"text": text_emb, "id": id_tokens, "hair": hair_tokens}

        # Train step (predict noise)
        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            noise_pred = pipe.unet(noisy, t, encoder_hidden_states=enc).sample
            loss = F.mse_loss(noise_pred.float(), noise.float())

        if not torch.isfinite(loss):
            print(f"[step {step}] loss non-finite -> skipping")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(train_params, float(cfg["train"].get("grad_clip", 1.0)))
        scaler.step(opt)
        scaler.update()

        if step % log_every == 0:
            print(f"[step {step}/{max_steps}] loss={loss.item():.6f}")

        # Qualitative sampling
        if cfg.get("eval", {}).get("enabled", False):
            every = int(cfg["eval"].get("every_steps", 200))
            if step % every == 0:
                qb = fixed_batch if fixed_batch is not None else batch
                q_pixel = qb["pixel_values"].to(device=device, dtype=dtype_unet)
                q_pil = qb["pil"]
                qB = q_pixel.shape[0]

                with torch.no_grad():
                    q_face_embs_512, q_face_mask = id_cond.extract_arcface_embs(q_pil, return_mask=True)
                    q_text_emb = project_face_embs(pipe, q_face_embs_512).to(dtype_unet)

                    if q_text_emb.shape[0] != qB:
                        if q_text_emb.shape[0] == 1:
                            q_text_emb = q_text_emb.repeat(qB, 1, 1)
                        else:
                            raise RuntimeError(f"q_text_emb batch mismatch: got {q_text_emb.shape[0]} vs qB={qB}")

                    if (~q_face_mask).any():
                        eval_prompt = cfg.get("eval", {}).get("prompt", "a portrait photo of a person")
                        q_tok_fb = pipe.tokenizer(
                            [eval_prompt] * qB,
                            padding="max_length",
                            max_length=pipe.tokenizer.model_max_length,
                            return_tensors="pt",
                        ).to(device)
                        q_text_fb = pipe.text_encoder(**q_tok_fb).last_hidden_state.to(dtype_unet)
                        q_text_emb = q_text_emb.clone()
                        q_text_emb[~q_face_mask] = q_text_fb[~q_face_mask]

                # switch to eval for sampling
                was_unet_train = unet.training
                was_id_train = id_cond.training
                was_hair_train = hair_cond.training
                unet.eval()
                id_cond.eval()
                hair_cond.eval()

                qualitative_check(
                    step=step,
                    run_dir=run_dir,
                    pipe=pipe,
                    scheduler=eval_scheduler,
                    pixel_values=q_pixel,
                    pil_images=q_pil,
                    text_emb=q_text_emb,
                    id_cond=id_cond,
                    hair_cond=hair_cond,
                    dtype_unet=dtype_unet,
                    num_steps=int(cfg["eval"].get("num_inference_steps", 50)),
                    seed=int(cfg["eval"].get("seed", 123)),
                )

                # restore modes
                if was_unet_train:
                    unet.train()
                if was_id_train:
                    id_cond.train()
                if was_hair_train:
                    hair_cond.train()

        # Save checkpoint
        if step % save_every == 0 or step == max_steps:
            ckpt = {
                "step": step,
                "id_proj": id_cond.proj.state_dict(),
                "hair_proj": hair_cond.proj.state_dict(),
                "dual_attn": {
                    k: v.state_dict()
                    for k, v in unet.attn_processors.items()
                    if isinstance(v, DualImageAttnProcessor)
                },
                "cfg": cfg,
                "meta": meta,
            }
            out = run_dir / f"ckpt_step{step}.pt"
            torch.save(ckpt, out)
            print("Saved:", out)

        step += 1

    print("Done. Run dir:", run_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml")
    args = ap.parse_args()
    main(args.cfg)
