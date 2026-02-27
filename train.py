# train.py
# Full, corrected training script for:
# - Stable Diffusion (frozen backbone)
# - DualImageAttnProcessor injected into ALL cross-attn (attn2) blocks
# - Two external condition streams: ID (ArcFace) and Hair (Parsing + CLIP)
# - Qualitative samples saved as 4-up grid: both_on / id_only / hair_only / both_off
#
# Expected repo structure:
#   src/data/images.py                      -> ImageFolderDataset (returns pixel_values, pil, path)
#   src/model/dual_ip_attention.py          -> DualImageAttnProcessor
#   src/model/id_conditioner_insightface.py -> IDArcFaceConditioner
#   src/model/hair_conditioner_parsing.py   -> HairConditioner
#   src/utils/repro.py                      -> set_seed
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

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers import DPMSolverMultistepScheduler

from src.utils.repro import set_seed
from src.data.images import ImageFolderDataset
from src.model.dual_ip_attention import DualImageAttnProcessor
from src.model.id_conditioner_insightface import IDArcFaceConditioner
from src.model.hair_conditioner_parsing import HairConditioner

def _maybe_display(path: str):
    try:
        from IPython.display import display
        from PIL import Image
        display(Image.open(path))
    except Exception as e:
        print(f"[qual] (display skipped) {e}")
def _print_png_base64(path: str, max_kb: int = 800):
    import base64
    data = open(path, "rb").read()
    if len(data) > max_kb * 1024:
        print(f"[qual] png too big ({len(data)/1024:.1f} KB), not printing base64")
        return
    b64 = base64.b64encode(data).decode("utf-8")
    print("data:image/png;base64," + b64)
# -------------------------
# Data collation (keep PIL)
# -------------------------
def collate_keep_pil(batch_list):
    pixel_values = torch.stack([b["pixel_values"] for b in batch_list], dim=0)  # [B,3,H,W] float
    pil = [b["pil"] for b in batch_list]  # list[PIL.Image]
    path = [b["path"] for b in batch_list]
    return {"pixel_values": pixel_values, "pil": pil, "path": path}


# -------------------------
# Helpers: decode/save/sample
# -------------------------
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
def ddim_sample_with_enc(
    pipe: StableDiffusionPipeline,
    noise_scheduler: DDPMScheduler,
    latents: torch.Tensor,
    enc: dict,
    num_steps: int,
):
    """
    Minimal sampler using DDPMScheduler timesteps + step().
    (Not "true DDIM", but OK for qualitative comparisons.)

    latents: start noise [B,4,h,w]
    enc: {"text": ..., "id": ..., "hair": ...}  (dict consumed by DualImageAttnProcessor)
    """
    noise_scheduler.set_timesteps(num_steps, device=latents.device)
    x = latents

    for t in noise_scheduler.timesteps:
        # UNet expects batch timesteps
        t_batch = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            eps = pipe.unet(x, t_batch, encoder_hidden_states=enc).sample

        step_out = noise_scheduler.step(eps, t, x)
        x = step_out.prev_sample

    return x


@torch.no_grad()
def qualitative_check(
    *,
    step: int,
    run_dir: Path,
    pipe: StableDiffusionPipeline,
    noise_scheduler: DDPMScheduler,
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
    Row: [both_on | id_only | hair_only | both_off]
    """
    out_dir = run_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    B, _, H, W = pixel_values.shape
    vae_sf = pipe.vae_scale_factor if hasattr(pipe, "vae_scale_factor") else 8

    # 1) Compute conditioning tokens ONCE
    id_tokens = id_cond(pil_images, out_dtype=dtype_unet)
    hair_tokens = hair_cond(pil_images, out_dtype=dtype_unet)

    # 2) Fixed noise
    gen = torch.Generator(device=pixel_values.device)
    gen.manual_seed(int(seed))
    latents0 = torch.randn(
        (B, 4, H // vae_sf, W // vae_sf),
        device=pixel_values.device,
        dtype=dtype_unet,
        generator=gen,
    )

    # 3) 4 variants (same noise, same text; only toggling id/hair)
    variants = [
        ("both_on",   id_tokens,                    hair_tokens),
        ("id_only",   id_tokens,                    torch.zeros_like(hair_tokens)),
        ("hair_only", torch.zeros_like(id_tokens),  hair_tokens),
        ("both_off",  torch.zeros_like(id_tokens),  torch.zeros_like(hair_tokens)),
    ]

    rows = []
    for tag, id_t, hair_t in variants:
        enc = {"text": text_emb, "id": id_t, "hair": hair_t}
        lat = ddim_sample_with_enc(pipe, noise_scheduler, latents0.clone(), enc, num_steps=num_steps)
        img_01 = _vae_decode_to_01(pipe, lat, dtype_unet)  # [B,3,H,W]
        rows.append(img_01[:1])  # take first for a clean 4-up compare

    row = torch.cat(rows, dim=0)  # [4,3,H,W]
    path = out_dir / f"step_{step:07d}.png"
    _save_row(row, str(path))
    print(f"[qual] saved {path}")
    _maybe_display(str(path))
    _print_png_base64(str(path), max_kb=800)


# -------------------------
# Config
# -------------------------
def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------
# Main
# -------------------------
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

    # ---- Load Stable Diffusion
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["models"]["sd_model_id"],
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    unet = pipe.unet
    dtype_unet = next(unet.parameters()).dtype  # fp16
    cross_dim = unet.config.cross_attention_dim

    # ---- Inject DualImageAttnProcessor into ALL cross-attn blocks (attn2)
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

            attn_procs[name] = DualImageAttnProcessor(
                base_processor=base_proc,
                hidden_size=hidden_size,
                cross_attention_dim=cross_dim,
                scale_id=float(cfg["cond"]["scale_id"]),
                scale_hair=float(cfg["cond"]["scale_hair"]),
                attn_fp32=True,
            ).to(device=device, dtype=torch.float32)  # keep this module stable in fp32
            n_cross += 1
        else:
            attn_procs[name] = base_proc

    unet.set_attn_processor(attn_procs)
    print(f"[init] injected DualImageAttnProcessor into {n_cross} cross-attn blocks")

    # ---- Freeze SD backbone
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # ---- Build conditioners (created ONCE)
    n_tokens = int(cfg["cond"]["n_tokens"])
    clip_id = cfg["models"]["clip_vision_id"]
    hair_w = cfg["models"]["hair_parsing_weights"]  # path to .pth

    id_cond = IDArcFaceConditioner(
        n_tokens=n_tokens,
        cross_dim=cross_dim,
        device=device,
        proj_dtype=torch.float32,
    ).to(device)

    hair_cond = HairConditioner(
        clip_vision_id=clip_id,
        n_tokens=n_tokens,
        cross_dim=cross_dim,
        hair_weights_path=hair_w,
        device=device,
        clip_dtype=torch.float16,
        proj_dtype=torch.float32,
        bg_value=float(cfg["cond"].get("hair_bg_value", 0.0)),
    ).to(device)

    # ---- Trainable params: conditioner projections + dual attention K/V projections
    train_params = list(id_cond.proj.parameters()) + list(hair_cond.proj.parameters())

    for proc in unet.attn_processors.values():
        if isinstance(proc, DualImageAttnProcessor):
            for p in proc.parameters():
                p.requires_grad = True
            train_params += list(proc.parameters())

    opt = torch.optim.AdamW(
        train_params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    # AMP scaler (use torch.cuda.amp for broad compatibility)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # ---- Data
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

    # Fixed batch for eval
    fixed_batch = None
    if cfg.get("eval", {}).get("enabled", False):
        fixed_batch = next(iter(dl))

    # Scheduler for training noise / for eval sampling
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    eval_scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    max_steps = int(cfg["train"]["max_steps"])
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])

    # ---- Modes
    unet.train()      # only attn_processor params have requires_grad=True
    id_cond.train()   # projection trainable
    hair_cond.train() # projection trainable

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

        # ---- Text embedding (prompt)
        prompt = cfg.get("train", {}).get("prompt", "a portrait photo of a person")
        tok = pipe.tokenizer(
            [prompt] * B,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_emb = pipe.text_encoder(**tok).last_hidden_state.to(dtype_unet)  # [B,T,D]

        # ---- VAE encode -> latents
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor  # [B,4,h,w]

        # ---- Add diffusion noise
        noise = torch.randn_like(latents)
        t = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()
        noisy = noise_scheduler.add_noise(latents, noise, t).to(dtype=dtype_unet)

        # ---- Conditioning tokens (from PIL)
        id_tokens = id_cond(pil_images, out_dtype=dtype_unet)
        hair_tokens = hair_cond(pil_images, out_dtype=dtype_unet)
        enc = {"text": text_emb, "id": id_tokens, "hair": hair_tokens}

        # ---- Train step (predict noise)
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
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

        step += 1

        if step % log_every == 0:
            print(f"[step {step}/{max_steps}] loss={loss.item():.6f}")

        # ---- Qualitative sampling
        if cfg.get("eval", {}).get("enabled", False):
            every = int(cfg["eval"].get("every_steps", 200))
            if step % every == 0:
                qb = fixed_batch if fixed_batch is not None else batch
                q_pixel = qb["pixel_values"].to(device=device, dtype=dtype_unet)
                q_pil = qb["pil"]
                qB = q_pixel.shape[0]

                eval_prompt = cfg.get("eval", {}).get("prompt", prompt)
                q_tok = pipe.tokenizer(
                    [eval_prompt] * qB,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    q_text_emb = pipe.text_encoder(**q_tok).last_hidden_state.to(dtype_unet)

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
                    noise_scheduler=eval_scheduler,
                    pixel_values=q_pixel,
                    pil_images=q_pil,
                    text_emb=q_text_emb,
                    id_cond=id_cond,
                    hair_cond=hair_cond,
                    dtype_unet=dtype_unet,
                    num_steps=int(cfg["eval"].get("num_inference_steps", 30)),
                    seed=int(cfg["eval"].get("seed", 123)),
                )

                # restore modes
                if was_unet_train:
                    unet.train()
                if was_id_train:
                    id_cond.train()
                if was_hair_train:
                    hair_cond.train()

        # ---- Save checkpoint
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

    print("Done. Run dir:", run_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml")
    args = ap.parse_args()
    main(args.cfg)
