import torchvision
from PIL import Image

import argparse, os, time, json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml
from diffusers import StableDiffusionPipeline, DDPMScheduler

from src.utils.repro import set_seed
from src.data.images import ImageFolderDataset
from src.model.dual_ip_attention import DualImageAttnProcessor
from src.model.clip_conditioner import CLIPTokenConditioner

@torch.no_grad()
def _vae_decode_to_01(pipe, latents, dtype_unet):
    latents = latents.to(dtype_unet)
    imgs = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample  # [-1,1]
    imgs_01 = (imgs.float() * 0.5 + 0.5).clamp(0, 1)  # fp32 [0,1]
    return imgs_01

@torch.no_grad()
def _save_row(images_01: torch.Tensor, path: str):
    """
    images_01: (N,3,H,W) in [0,1]
    Сохраняет одной строкой
    """
    grid = torchvision.utils.make_grid(images_01, nrow=images_01.shape[0])
    torchvision.utils.save_image(grid, path)

@torch.no_grad()
def ddim_sample_with_enc(pipe, noise_scheduler, latents, enc, num_steps: int, generator: torch.Generator):
    """
    Мини-DDIM sampler на базе DDPM scheduler, но с set_timesteps (diffusers).
    latents: (B,4,H/8,W/8) стартовый шум
    enc: dict {"text":..., "id":..., "hair":...}
    """
    # инференсные таймстепы
    noise_scheduler.set_timesteps(num_steps, device=latents.device)
    x = latents

    for t in noise_scheduler.timesteps:
        # t в diffusers часто scalar tensor; unet ожидает batch timesteps
        t_batch = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            eps = pipe.unet(x, t_batch, encoder_hidden_states=enc).sample

        step_out = noise_scheduler.step(eps, t, x, generator=generator)
        x = step_out.prev_sample

    return x

@torch.no_grad()
def qualitative_check(
    *,
    step: int,
    run_dir,
    pipe,
    noise_scheduler,
    pixel_values,     # (B,3,H,W) fp16 in [-1,1]
    text_emb,         # (B,T,D)
    id_cond,
    hair_cond,
    dtype_unet,
    num_steps: int,
    seed: int,
):
    out_dir = run_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    B = pixel_values.shape[0]

    # 1) готовим pil_images из pixel_values
    imgs_01 = (pixel_values.float() * 0.5 + 0.5).clamp(0, 1)
    pil_images = []
    for i in range(B):
        arr = (imgs_01[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pil_images.append(Image.fromarray(arr))

    # 2) токены условий
    id_tokens = id_cond(pil_images, out_dtype=dtype_unet)
    hair_tokens = hair_cond(pil_images, out_dtype=dtype_unet)

    # 3) фиксированный шум
    gen = torch.Generator(device=pixel_values.device)
    gen.manual_seed(int(seed))
    latents0 = torch.randn((B, 4, pixel_values.shape[-2] // 8, pixel_values.shape[-1] // 8),
                           device=pixel_values.device, dtype=dtype_unet, generator=gen)

    # 4) 4 режима
    variants = [
        ("both_on",  id_tokens,                         hair_tokens),
        ("id_only",  id_tokens,                         torch.zeros_like(hair_tokens)),
        ("hair_only",torch.zeros_like(id_tokens),       hair_tokens),
        ("both_off", torch.zeros_like(id_tokens),       torch.zeros_like(hair_tokens)),
    ]

    rows = []
    for tag, id_t, hair_t in variants:
        enc = {"text": text_emb, "id": id_t, "hair": hair_t}
        lat = ddim_sample_with_enc(pipe, noise_scheduler, latents0.clone(), enc, num_steps=num_steps, generator=gen)
        img_01 = _vae_decode_to_01(pipe, lat, dtype_unet)  # (B,3,H,W)
        # если B>1, можно брать только первую картинку, чтобы сравнение было проще:
        rows.append(img_01[:1])

    # склеиваем 4 картинки в строку
    row = torch.cat(rows, dim=0)  # (4,3,H,W)
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

    # Load SD
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["models"]["sd_model_id"],
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    unet = pipe.unet
    dtype_unet = next(unet.parameters()).dtype  # fp16
    cross_dim = unet.config.cross_attention_dim

    base_procs = unet.attn_processors
    attn_procs = {}
    n_cross = 0

    for name, base_proc in base_procs.items():
        if name.endswith("attn2.processor"):
            m = unet
            for key in name.split(".")[:-1]:
                m = getattr(m, key)
            hidden_size = m.to_q.in_features

            # keep DualImageAttnProcessor in fp32
            attn_procs[name] = DualImageAttnProcessor(
                base_processor=base_proc,
                hidden_size=hidden_size,
                cross_attention_dim=cross_dim,
                scale_id=float(cfg["cond"]["scale_id"]),
                scale_hair=float(cfg["cond"]["scale_hair"]),
                attn_fp32=True,
            ).to(device=device, dtype=torch.float32)

            n_cross += 1
        else:
            attn_procs[name] = base_proc

    unet.set_attn_processor(attn_procs)

    # Freeze SD backbone
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Conditioners (Method 1: y1=y2=x)
    n_tokens = int(cfg["cond"]["n_tokens"])
    clip_id = cfg["models"]["clip_vision_id"]

    id_cond = CLIPTokenConditioner(
        clip_id, n_tokens, cross_dim, device=device, clip_dtype=torch.float16, proj_dtype=torch.float32
    )
    hair_cond = CLIPTokenConditioner(
        clip_id, n_tokens, cross_dim, device=device, clip_dtype=torch.float16, proj_dtype=torch.float32
    )

    # Trainable params: projections + dual K/V
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

    # fp16 training -> use GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Data 
    ds = ImageFolderDataset(cfg["data"]["train_dir"], image_size=int(cfg["data"]["image_size"]))
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    # fixed batch for qualitative check
    fixed_batch = None
    if cfg.get("eval", {}).get("enabled", False):
        fixed_batch = next(iter(dl))
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    max_steps = int(cfg["train"]["max_steps"])
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])

    step = 0
    unet.train()  # only processor params are trainable
    id_cond.train()
    hair_cond.train()

    it = iter(dl)
    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        pixel_values = batch["pixel_values"].to(device=device, dtype=dtype_unet)  # [B,3,H,W] fp16

        # text emb (empty prompt)
        tok = pipe.tokenizer(
            [""] * pixel_values.shape[0],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_emb = pipe.text_encoder(**tok).last_hidden_state.to(dtype_unet)

        # VAE encode -> latents
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor

        # noise
        noise = torch.randn_like(latents)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, t)
        noisy = noisy.to(dtype=dtype_unet) 

        # Method 1 conditioning: use same image as both ID and Hair source
        with torch.no_grad():
            imgs_01 = (pixel_values.float() * 0.5 + 0.5).clamp(0, 1)  # fp32 [0,1]

        pil_images = []
        from PIL import Image

        for i in range(imgs_01.shape[0]):
            arr = (imgs_01[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            pil_images.append(Image.fromarray(arr))

        id_tokens = id_cond(pil_images, out_dtype=dtype_unet)
        hair_tokens = hair_cond(pil_images, out_dtype=dtype_unet)

        enc = {"text": text_emb, "id": id_tokens, "hair": hair_tokens}

        # AMP + GradScaler training step
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            noise_pred = pipe.unet(noisy, t, encoder_hidden_states=enc).sample
            loss = F.mse_loss(noise_pred.float(), noise.float())

        if not torch.isfinite(loss):
            print(f"[step {step}] loss is non-finite -> skipping step")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        scaler.step(opt)
        scaler.update()

        step += 1

        if step % log_every == 0:
            print(f"[step {step}/{max_steps}] loss={loss.item():.6f}")
        # qualitative sampling check
        if cfg.get("eval", {}).get("enabled", False):
            every = int(cfg["eval"].get("every_steps", 200))
            if step % every == 0:
                qb = fixed_batch if fixed_batch is not None else batch
                q_pixel = qb["pixel_values"].to(device=device, dtype=dtype_unet)
        
                # text emb for the fixed batch
                q_tok = pipe.tokenizer(
                    [""] * q_pixel.shape[0],
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(device)
        
                with torch.no_grad():
                    q_text_emb = pipe.text_encoder(**q_tok).last_hidden_state.to(dtype_unet)
        
                qualitative_check(
                    step=step,
                    run_dir=run_dir,
                    pipe=pipe,
                    noise_scheduler=noise_scheduler,
                    pixel_values=q_pixel,
                    text_emb=q_text_emb,
                    id_cond=id_cond,
                    hair_cond=hair_cond,
                    dtype_unet=dtype_unet,
                    num_steps=int(cfg["eval"].get("num_inference_steps", 30)),
                    seed=int(cfg["eval"].get("seed", 123)),
                )
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
