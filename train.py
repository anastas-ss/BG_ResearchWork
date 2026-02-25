
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

    # Save config + env info for reproducibility
    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "config.yaml").write_text(Path(cfg_path).read_text())

    # --- Load SD ---
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["models"]["sd_model_id"],
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    unet = pipe.unet
    dtype_unet = next(unet.parameters()).dtype  # fp16
    cross_dim = unet.config.cross_attention_dim

    # --- Install Dual processors on top of CLEAN base processors ---
    base_procs = unet.attn_processors
    attn_procs = {}
    n_cross = 0

    for name, base_proc in base_procs.items():
        if name.endswith("attn2.processor"):
            m = unet
            for key in name.split(".")[:-1]:
                m = getattr(m, key)
            hidden_size = m.to_q.in_features

            # IMPORTANT: keep DualImageAttnProcessor in fp32 for stability
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

    # --- Freeze SD backbone ---
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # --- Conditioners (Method 1: y1=y2=x) ---
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

    # IMPORTANT: fp16 training -> use GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # --- Data ---
    ds = ImageFolderDataset(cfg["data"]["train_dir"], image_size=int(cfg["data"]["image_size"]))
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

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
        noisy = noisy.to(dtype=dtype_unet)  # IMPORTANT: match UNet dtype

        # Method 1 conditioning: use SAME image as both ID and Hair source
        with torch.no_grad():
            imgs_01 = (pixel_values.float() * 0.5 + 0.5).clamp(0, 1)  # fp32 [0,1]

        pil_images = []
        from PIL import Image

        for i in range(imgs_01.shape[0]):
            arr = (imgs_01[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            pil_images.append(Image.fromarray(arr))

        id_tokens = id_cond(pil_images, out_dtype=dtype_unet)
        hair_tokens = hair_cond(pil_images, out_dtype=dtype_unet)

        # Optional sanity checks (uncomment for debugging)
        # assert torch.isfinite(id_tokens).all(), "id_tokens has NaN/Inf"
        # assert torch.isfinite(hair_tokens).all(), "hair_tokens has NaN/Inf"
        # assert torch.isfinite(text_emb).all(), "text_emb has NaN/Inf"

        enc = {"text": text_emb, "id": id_tokens, "hair": hair_tokens}

        # ----- AMP + GradScaler training step -----
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
        # -----------------------------------------

        step += 1

        if step % log_every == 0:
            print(f"[step {step}/{max_steps}] loss={loss.item():.6f}")

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
