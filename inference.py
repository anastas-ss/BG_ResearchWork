import argparse
import csv
from pathlib import Path

import torch
import torchvision
from PIL import Image

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from src.model.dual_ip_attention import DualImageAttnProcessor
from src.model.id_conditioner_insightface import IDArcFaceConditioner
from src.model.hair_conditioner_parsing import HairConditioner, remove_hair_from_pil
from src.utils.project_face_embs import project_face_embs


@torch.no_grad()
def load_pil(path: str, size: int = 512) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize((size, size), Image.BILINEAR)
    return im


def inject_dual_attn(pipe, scale_id: float, scale_hair: float, attn_fp32: bool = True):
    unet = pipe.unet
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

            proc = DualImageAttnProcessor(
                base_processor=base_proc,
                hidden_size=hidden_size,
                cross_attention_dim=cross_dim,
                scale_id=float(scale_id),
                scale_hair=float(scale_hair),
                attn_fp32=attn_fp32,
            ).to(device=pipe.device, dtype=torch.float32)

            with torch.no_grad():
                proc.to_k_id.weight.copy_(m.to_k.weight.detach().to(proc.to_k_id.weight.dtype))
                proc.to_v_id.weight.copy_(m.to_v.weight.detach().to(proc.to_v_id.weight.dtype))

            attn_procs[name] = proc
            n_cross += 1
        else:
            attn_procs[name] = base_proc

    unet.set_attn_processor(attn_procs)
    print(f"[init] injected DualImageAttnProcessor into {n_cross} cross-attn blocks")


def load_ckpt_into_modules(pipe, id_cond, hair_cond, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # projections
    if "id_proj" in ckpt:
        id_cond.proj.load_state_dict(ckpt["id_proj"], strict=True)
    if "hair_proj" in ckpt:
        hair_cond.proj.load_state_dict(ckpt["hair_proj"], strict=True)

    # dual attention processors
    dual = ckpt.get("dual_attn", {})
    for k, proc in pipe.unet.attn_processors.items():
        if isinstance(proc, DualImageAttnProcessor) and k in dual:
            proc.load_state_dict(dual[k], strict=True)

    print("[ckpt] loaded:", ckpt_path)


@torch.no_grad()
def get_text_emb(pipe, prompt: str, device: str, dtype: torch.dtype):
    tok = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    return pipe.text_encoder(**tok).last_hidden_state.to(dtype)


@torch.no_grad()
def generate_one(
    pipe,
    scheduler,
    prompt: str,
    pil_id: Image.Image,
    pil_hair: Image.Image,
    id_cond,
    hair_cond,
    num_steps: int,
    guidance_scale: float,
    seed: int,
):
    device = pipe.device
    dtype_unet = next(pipe.unet.parameters()).dtype

    hair_mask = hair_cond.get_hair_masks([pil_id])[0]
    pil_id_no_hair = remove_hair_from_pil(pil_id, hair_mask, fill=0.5)

    face_embs_512, _ = id_cond.extract_arcface_embs([pil_id_no_hair], return_mask=True)
    text_emb = project_face_embs(pipe, face_embs_512).to(dtype_unet)
    text_emb_uc = get_text_emb(pipe, "", device, dtype_unet)

    id_tokens = id_cond.embs_to_tokens(face_embs_512, out_dtype=dtype_unet)
    hair_tokens = hair_cond([pil_hair], out_dtype=dtype_unet)
    hair_tokens = hair_tokens / (hair_tokens.norm(dim=-1, keepdim=True) + 1e-6)

    enc_cond = {"text": text_emb, "id": id_tokens, "hair": hair_tokens}
    enc_uncond = {"text": text_emb_uc, "id": torch.zeros_like(id_tokens), "hair": torch.zeros_like(hair_tokens)}

    # initial noise
    vae_sf = pipe.vae_scale_factor if hasattr(pipe, "vae_scale_factor") else 8
    H, W = pil_id.size[1], pil_id.size[0]
    gen = torch.Generator(device=device).manual_seed(int(seed))
    latents = torch.randn((1, 4, H // vae_sf, W // vae_sf), device=device, dtype=dtype_unet, generator=gen)

    scheduler.set_timesteps(num_steps, device=device)
    x = latents

    for t in scheduler.timesteps:
        t_batch = torch.tensor([int(t)], device=device, dtype=torch.long)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            eps_u = pipe.unet(x, t_batch, encoder_hidden_states=enc_uncond).sample
            eps_c = pipe.unet(x, t_batch, encoder_hidden_states=enc_cond).sample
        eps = eps_u + guidance_scale * (eps_c - eps_u)
        x = scheduler.step(eps, t, x).prev_sample

    # decode
    imgs = pipe.vae.decode(x / pipe.vae.config.scaling_factor).sample
    imgs_01 = (imgs.float() * 0.5 + 0.5).clamp(0, 1)  # [1,3,H,W]
    return imgs_01[0].cpu()  # [3,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs_infer/out")
    ap.add_argument("--sd_model_id", type=str, required=True)
    ap.add_argument("--clip_vision_id", type=str, required=True)
    ap.add_argument("--hair_weights", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--prompt", type=str, default="a portrait photo of a person")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--scale_id", type=float, default=1.0)
    ap.add_argument("--scale_hair", type=float, default=1.0)
    ap.add_argument("--hair_class", type=int, default=17)

    ap.add_argument("--n_tokens", type=int, default=4)
    ap.add_argument("--max_items", type=int, default=0, help="0 = all")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Need GPU"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    inject_dual_attn(pipe, scale_id=args.scale_id, scale_hair=args.scale_hair, attn_fp32=True)

    cross_dim = pipe.unet.config.cross_attention_dim

    id_cond = IDArcFaceConditioner(
        n_tokens=args.n_tokens,
        cross_dim=cross_dim,
        device=device,
        proj_dtype=torch.float32,
    ).to(device).eval()

    hair_cond = HairConditioner(
        clip_vision_id=args.clip_vision_id,
        n_tokens=args.n_tokens,
        cross_dim=cross_dim,
        hair_weights_path=args.hair_weights,
        hair_class=args.hair_class,
        device=device,
        clip_dtype=torch.float16,
        proj_dtype=torch.float32,
        bg_value=0.0,
    ).to(device).eval()

    # load ckpt
    load_ckpt_into_modules(pipe, id_cond, hair_cond, args.ckpt)

    # run
    rows = []
    with open(args.pairs_csv, "r") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if args.max_items and i >= args.max_items:
                break

            pair_id = r["pair_id"]
            ref_id = r["ref_id"]
            ref_hair = r["ref_hair"]

            pil_id = load_pil(ref_id, size=args.image_size)
            pil_hair = load_pil(ref_hair, size=args.image_size)

            img = generate_one(
                pipe=pipe,
                scheduler=pipe.scheduler,
                prompt=args.prompt,
                pil_id=pil_id,
                pil_hair=pil_hair,
                id_cond=id_cond,
                hair_cond=hair_cond,
                num_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed,
            )

            sample_dir = out_dir / f"{int(pair_id):06d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # save refs
            pil_id.save(sample_dir / "ref_id.png")
            pil_hair.save(sample_dir / "ref_hair.png")
            torchvision.utils.save_image(img, sample_dir / "gen.png")

            rows.append([pair_id, str(sample_dir / "gen.png"), ref_id, ref_hair])

            if (i + 1) % 20 == 0:
                print(f"[infer] {i+1} done")

    # save manifest
    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "gen_path", "ref_id", "ref_hair"])
        w.writerows(rows)

    print("Done. manifest:", manifest)


if __name__ == "__main__":
    main()
