import argparse
import csv
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel

from src.model.clip_text_model_wrapper import CLIPTextModelWrapper
from src.model.dual_ip_attention import DualImageAttnProcessor
from src.model.id_conditioner_insightface import IDArcFaceConditioner
from src.model.hair_conditioner_parsing import HairConditioner
from src.utils.project_face_embs import project_face_embs


@torch.no_grad()
def load_pil(path: str, size: int = 512) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize((size, size), Image.BILINEAR)
    return im


def is_valid_image(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def inject_dual_attn(
    pipe,
    scale_hair: float,
    attn_fp32: bool = True,
    spatial_gate_hair: bool = False,
):
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
                scale_hair=float(scale_hair),
                attn_fp32=attn_fp32,
                processor_name=name,
                spatial_gate_hair=spatial_gate_hair,
            ).to(device=pipe.device, dtype=torch.float32)

            attn_procs[name] = proc
            n_cross += 1
        else:
            attn_procs[name] = base_proc

    unet.set_attn_processor(attn_procs)
    print(f"[init] injected DualImageAttnProcessor into {n_cross} cross-attn blocks")


def load_ckpt_into_modules(pipe, hair_cond, ckpt_path: str, ckpt=None):
    if ckpt is None:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if hair_cond is not None and "hair_proj" in ckpt:
        hair_cond.proj.load_state_dict(ckpt["hair_proj"], strict=True)

    # dual attention processors
    dual = ckpt.get("dual_attn", {})
    for k, proc in pipe.unet.attn_processors.items():
        if isinstance(proc, DualImageAttnProcessor) and k in dual:
            missing, unexpected = proc.load_state_dict(dual[k], strict=False)
            if missing or unexpected:
                print(
                    f"[ckpt] dual_attn {k}: missing={list(missing)} unexpected={list(unexpected)}"
                )

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


def set_hair_attention_callback(pipe, callback):
    for proc in pipe.unet.attn_processors.values():
        if isinstance(proc, DualImageAttnProcessor):
            proc.hair_attention_callback = callback


def _normalize_map(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    value = value - value.min()
    return value / value.max().clamp_min(1e-8)


def _colorize_map(value: torch.Tensor) -> torch.Tensor:
    value = _normalize_map(value)
    red = (1.5 - (4.0 * value - 3.0).abs()).clamp(0, 1)
    green = (1.5 - (4.0 * value - 2.0).abs()).clamp(0, 1)
    blue = (1.5 - (4.0 * value - 1.0).abs()).clamp(0, 1)
    return torch.stack([red, green, blue], dim=0)


def save_hair_attention_diagnostics(diagnostics, generated_image, sample_dir: Path):
    if not diagnostics:
        return

    debug_dir = sample_dir / "hair_attention"
    debug_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for step_index, step_data in sorted(diagnostics.items()):
        layer_maps = []
        for layer in step_data["layers"]:
            height = layer["spatial_height"]
            width = layer["spatial_width"]
            contribution = layer["contribution_norm"][0]
            if height is None or width is None or contribution.numel() != height * width:
                continue
            contribution = contribution.reshape(1, 1, height, width)
            contribution = F.interpolate(
                contribution,
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            layer_maps.append(_normalize_map(contribution))
            summary_rows.append(
                [
                    step_index,
                    layer["processor_name"],
                    height,
                    width,
                    float(layer["contribution_norm"].mean()),
                    float(layer["contribution_norm"].max()),
                    float(layer["token_attention_max"].mean()),
                    float(layer["token_attention_entropy"].mean()),
                ]
            )

        if layer_maps:
            aggregate = torch.stack(layer_maps).mean(dim=0)
            heatmap = _colorize_map(aggregate)
            heatmap_full = F.interpolate(
                heatmap.unsqueeze(0),
                size=generated_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0]
            overlay = (0.55 * generated_image.float() + 0.45 * heatmap_full).clamp(0, 1)
            torchvision.utils.save_image(
                aggregate.unsqueeze(0),
                debug_dir / f"hair_contribution_step_{step_index:02d}.png",
            )
            torchvision.utils.save_image(
                overlay,
                debug_dir / f"hair_contribution_overlay_step_{step_index:02d}.png",
            )

        noise_delta = step_data.get("noise_delta")
        if noise_delta is not None:
            noise_delta = _normalize_map(noise_delta[0])
            torchvision.utils.save_image(
                noise_delta.unsqueeze(0),
                debug_dir / f"hair_noise_delta_step_{step_index:02d}.png",
            )

    with (debug_dir / "layer_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step_index",
                "processor_name",
                "height",
                "width",
                "contribution_mean",
                "contribution_max",
                "token_attention_max_mean",
                "token_attention_entropy_mean",
            ]
        )
        writer.writerows(summary_rows)


@torch.no_grad()
def generate_one(
    pipe,
    scheduler,
    prompt: str,
    pil_id: Image.Image,
    pil_hair: Image.Image,
    id_cond,
    hair_cond,
    n_tokens: int,
    cross_dim: int,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    use_dual_attn: bool,
    attention_debug_steps: set[int] | None = None,
    cfg_mode: str = "joint",
    guidance_id: float | None = None,
    guidance_hair: float | None = None,
):
    device = pipe.device
    dtype_unet = next(pipe.unet.parameters()).dtype

    face_embs_512, _ = id_cond.extract_arcface_embs([pil_id], return_mask=True)
    text_emb = project_face_embs(pipe, face_embs_512).to(dtype_unet)
    text_emb_uc = get_text_emb(pipe, "", device, dtype_unet)
    H, W = pil_id.size[1], pil_id.size[0]

    if use_dual_attn:
        if hair_cond is None:
            hair_tokens = torch.zeros((1, n_tokens, cross_dim), device=device, dtype=dtype_unet)
            hair_token_mask = torch.zeros(
                hair_tokens.shape[:2],
                device=device,
                dtype=torch.float32,
            )
            hair_spatial_mask = torch.zeros(
                (1, H, W),
                device=device,
                dtype=torch.float32,
            )
        else:
            hair_tokens, hair_token_mask, hair_spatial_mask = hair_cond(
                [pil_hair],
                out_dtype=dtype_unet,
                return_masks=True,
            )

        enc_cond = {
            "text": text_emb,
            "hair": hair_tokens,
            "hair_token_mask": hair_token_mask,
            "hair_spatial_mask": hair_spatial_mask,
        }
        enc_uncond = {
            "text": text_emb_uc,
            "hair": torch.zeros_like(hair_tokens),
            "hair_token_mask": torch.zeros_like(hair_token_mask),
            "hair_spatial_mask": torch.zeros_like(hair_spatial_mask),
        }
        enc_id_only = {
            "text": text_emb,
            "hair": torch.zeros_like(hair_tokens),
            "hair_token_mask": torch.zeros_like(hair_token_mask),
            "hair_spatial_mask": torch.zeros_like(hair_spatial_mask),
        }
    else:
        enc_cond = text_emb
        enc_uncond = text_emb_uc
        enc_id_only = None

    if cfg_mode == "factorized" and not use_dual_attn:
        raise ValueError("factorized CFG requires DualImageAttnProcessor")
    guidance_id = guidance_scale if guidance_id is None else float(guidance_id)
    guidance_hair = guidance_scale if guidance_hair is None else float(guidance_hair)

    # initial noise
    vae_sf = pipe.vae_scale_factor if hasattr(pipe, "vae_scale_factor") else 8
    gen = torch.Generator(device=device).manual_seed(int(seed))
    latents = torch.randn((1, 4, H // vae_sf, W // vae_sf), device=device, dtype=dtype_unet, generator=gen)

    scheduler.set_timesteps(num_steps, device=device)
    x = latents * scheduler.init_noise_sigma
    attention_diagnostics = {}

    for step_index, t in enumerate(scheduler.timesteps):
        t_batch = torch.tensor([int(t)], device=device, dtype=torch.long)
        x_in = scheduler.scale_model_input(x, t)
        capture_attention = bool(
            use_dual_attn
            and attention_debug_steps
            and step_index in attention_debug_steps
        )
        captured_layers = []

        def capture_callback(data):
            captured_layers.append(
                {
                    key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in data.items()
                }
            )

        with torch.amp.autocast("cuda", dtype=torch.float16):
            eps_u = pipe.unet(x_in, t_batch, encoder_hidden_states=enc_uncond).sample
            eps_id = None
            if cfg_mode == "factorized":
                eps_id = pipe.unet(
                    x_in,
                    t_batch,
                    encoder_hidden_states=enc_id_only,
                ).sample
            if capture_attention:
                set_hair_attention_callback(pipe, capture_callback)
            try:
                eps_c = pipe.unet(x_in, t_batch, encoder_hidden_states=enc_cond).sample
            finally:
                if capture_attention:
                    set_hair_attention_callback(pipe, None)

            if capture_attention:
                if eps_id is None:
                    eps_id = pipe.unet(
                        x_in,
                        t_batch,
                        encoder_hidden_states=enc_id_only,
                    ).sample
                noise_delta = (eps_c.float() - eps_id.float()).pow(2).mean(
                    dim=1
                ).sqrt().cpu()
                attention_diagnostics[step_index] = {
                    "layers": captured_layers,
                    "noise_delta": noise_delta,
                }
        if cfg_mode == "factorized":
            eps = (
                eps_u
                + guidance_id * (eps_id - eps_u)
                + guidance_hair * (eps_c - eps_id)
            )
        else:
            eps = eps_u + guidance_scale * (eps_c - eps_u)
        x = scheduler.step(eps, t, x).prev_sample

    # decode
    imgs = pipe.vae.decode(x / pipe.vae.config.scaling_factor).sample
    imgs_01 = (imgs.float() * 0.5 + 0.5).clamp(0, 1)  # [1,3,H,W]
    return imgs_01[0].cpu(), attention_diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs_infer/out")
    ap.add_argument("--sd_model_id", type=str, required=True)
    ap.add_argument("--arc2face_repo_id", type=str, default="")
    ap.add_argument("--clip_vision_id", type=str, required=True)
    ap.add_argument("--hair_weights", type=str, default="")
    ap.add_argument("--ckpt", type=str, default="")

    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--prompt", type=str, default="a portrait photo of a person")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--cfg_mode", choices=["joint", "factorized"], default="joint")
    ap.add_argument("--guidance_id", type=float, default=None)
    ap.add_argument("--guidance_hair", type=float, default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--save_refs", type=int, default=1)
    ap.add_argument("--skip_existing", type=int, default=0)

    ap.add_argument("--scale_hair", type=float, default=1.0)
    ap.add_argument("--hair_class", type=int, default=17)
    ap.add_argument("--hair_classes", type=str, default="17")
    ap.add_argument("--hair_mask_dilate_kernel", type=int, default=1)
    ap.add_argument("--hair_mask_dilate_iters", type=int, default=1)
    ap.add_argument("--hair_focus_crop", type=int, default=0)
    ap.add_argument("--hair_focus_crop_margin", type=float, default=0.20)
    ap.add_argument("--hair_focus_crop_square", type=int, default=1)
    ap.add_argument("--insightface_root", type=str, default=None)

    ap.add_argument("--n_tokens", type=int, default=4)
    ap.add_argument("--hair_token_mode", choices=["global", "patch"], default=None)
    ap.add_argument("--hair_patch_mask_threshold", type=float, default=None)
    ap.add_argument("--hair_max_patch_tokens", type=int, default=None)
    ap.add_argument("--hair_patch_post_layernorm", type=int, default=None)
    ap.add_argument("--hair_patch_binary_mask", type=int, default=None)
    ap.add_argument("--hair_apply_token_mask_to_values", type=int, default=None)
    ap.add_argument(
        "--hair_token_normalization",
        choices=["l2", "layernorm", "none"],
        default=None,
    )
    ap.add_argument("--hair_spatial_gate", type=int, default=None)
    ap.add_argument("--max_items", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--disable_dual_attn",
        action="store_true",
        help="Skip DualImageAttnProcessor entirely; intended for pure Arc2Face no-hair checks.",
    )
    ap.add_argument(
        "--attention_debug",
        action="store_true",
        help="Save spatial diagnostics for the hair cross-attention branch.",
    )
    ap.add_argument(
        "--attention_debug_steps",
        type=str,
        default="0,12,24",
        help="Comma-separated denoising step indices to capture.",
    )
    ap.add_argument(
        "--attention_debug_max_items",
        type=int,
        default=4,
        help="Capture diagnostics only for the first N pairs.",
    )

    args = ap.parse_args()
    checkpoint = None
    checkpoint_cond = {}
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        checkpoint_cond = checkpoint.get("cfg", {}).get("cond", {})
    if args.hair_token_mode is None:
        args.hair_token_mode = str(checkpoint_cond.get("hair_token_mode", "global"))
    if args.hair_patch_mask_threshold is None:
        args.hair_patch_mask_threshold = float(
            checkpoint_cond.get("hair_patch_mask_threshold", 0.05)
        )
    if args.hair_max_patch_tokens is None:
        args.hair_max_patch_tokens = int(checkpoint_cond.get("hair_max_patch_tokens", 64))
    if args.hair_patch_post_layernorm is None:
        args.hair_patch_post_layernorm = int(
            bool(checkpoint_cond.get("hair_patch_post_layernorm", False))
        )
    if args.hair_patch_binary_mask is None:
        args.hair_patch_binary_mask = int(
            bool(checkpoint_cond.get("hair_patch_binary_mask", False))
        )
    if args.hair_apply_token_mask_to_values is None:
        args.hair_apply_token_mask_to_values = int(
            bool(checkpoint_cond.get("hair_apply_token_mask_to_values", True))
        )
    if args.hair_token_normalization is None:
        args.hair_token_normalization = str(
            checkpoint_cond.get("hair_token_normalization", "l2")
        )
    if args.hair_spatial_gate is None:
        args.hair_spatial_gate = int(bool(checkpoint_cond.get("hair_spatial_gate", False)))
    print(
        "[hair conditioning]",
        f"token_mode={args.hair_token_mode}",
        f"patch_mask_threshold={args.hair_patch_mask_threshold}",
        f"max_patch_tokens={args.hair_max_patch_tokens}",
        f"patch_post_layernorm={bool(args.hair_patch_post_layernorm)}",
        f"patch_binary_mask={bool(args.hair_patch_binary_mask)}",
        f"mask_values={bool(args.hair_apply_token_mask_to_values)}",
        f"token_normalization={args.hair_token_normalization}",
        f"spatial_gate={bool(args.hair_spatial_gate)}",
    )
    if args.cfg_mode == "factorized":
        guidance_id = args.guidance if args.guidance_id is None else args.guidance_id
        guidance_hair = args.guidance if args.guidance_hair is None else args.guidance_hair
        print(
            f"[cfg] mode=factorized guidance_id={guidance_id} "
            f"guidance_hair={guidance_hair}"
        )
    else:
        print(f"[cfg] mode=joint guidance={args.guidance}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Need GPU"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # pipe (optionally Arc2Face encoder+UNet on top of SD1.5 base)
    text_encoder = None
    unet_override = None
    if args.arc2face_repo_id:
        print(f"[init] loading Arc2Face modules from {args.arc2face_repo_id}")
        text_encoder = CLIPTextModelWrapper.from_pretrained(
            args.arc2face_repo_id, subfolder="encoder", torch_dtype=torch.float16
        )
        unet_override = UNet2DConditionModel.from_pretrained(
            args.arc2face_repo_id, subfolder="arc2face", torch_dtype=torch.float16
        )

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_id,
        text_encoder=text_encoder,
        unet=unet_override,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    use_dual_attn = not args.disable_dual_attn
    if use_dual_attn:
        inject_dual_attn(
            pipe,
            scale_hair=args.scale_hair,
            attn_fp32=True,
            spatial_gate_hair=bool(args.hair_spatial_gate),
        )
    else:
        if float(args.scale_hair) != 0.0:
            raise ValueError("--disable_dual_attn is only supported with --scale_hair 0.0")
        print("[init] DualImageAttnProcessor disabled; running base Arc2Face cross-attention")

    cross_dim = pipe.unet.config.cross_attention_dim

    id_cond = IDArcFaceConditioner(
        n_tokens=args.n_tokens,
        cross_dim=cross_dim,
        device=device,
        proj_dtype=torch.float32,
        model_root=args.insightface_root,
    ).to(device).eval()

    hair_cond = None
    if float(args.scale_hair) != 0.0:
        if not args.hair_weights:
            raise ValueError("--hair_weights is required when --scale_hair is non-zero")
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
            token_mode=args.hair_token_mode,
            patch_mask_threshold=args.hair_patch_mask_threshold,
            max_patch_tokens=args.hair_max_patch_tokens,
            patch_post_layernorm=bool(args.hair_patch_post_layernorm),
            patch_binary_mask=bool(args.hair_patch_binary_mask),
            apply_token_mask_to_values=bool(args.hair_apply_token_mask_to_values),
            token_normalization=args.hair_token_normalization,
        ).to(device).eval()
    else:
        print("[init] scale_hair=0.0: skip HairConditioner and use zero hair tokens")

    if args.ckpt:
        load_ckpt_into_modules(pipe, hair_cond, args.ckpt, ckpt=checkpoint)
    else:
        print("[ckpt] skipped: running initialized model without training checkpoint")

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
            sample_dir = out_dir / f"{int(pair_id):06d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            gen_path = sample_dir / "gen.png"

            if args.skip_existing and is_valid_image(gen_path):
                rows.append([pair_id, str(gen_path), ref_id, ref_hair])
                if (i + 1) % 20 == 0:
                    print(f"[infer] {i+1} done (existing outputs reused)")
                continue
            if gen_path.exists():
                gen_path.unlink()

            pil_id = load_pil(ref_id, size=args.image_size)
            pil_hair = load_pil(ref_hair, size=args.image_size)

            debug_steps = None
            if args.attention_debug and i < args.attention_debug_max_items:
                debug_steps = {
                    int(value)
                    for value in re.split(r"[,:\s]+", args.attention_debug_steps)
                    if value.strip()
                }

            img, attention_diagnostics = generate_one(
                pipe=pipe,
                scheduler=pipe.scheduler,
                prompt=args.prompt,
                pil_id=pil_id,
                pil_hair=pil_hair,
                id_cond=id_cond,
                hair_cond=hair_cond,
                n_tokens=args.n_tokens,
                cross_dim=cross_dim,
                num_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed,
                use_dual_attn=use_dual_attn,
                attention_debug_steps=debug_steps,
                cfg_mode=args.cfg_mode,
                guidance_id=args.guidance_id,
                guidance_hair=args.guidance_hair,
            )

            if args.save_refs:
                pil_id.save(sample_dir / "ref_id.png")
                pil_hair.save(sample_dir / "ref_hair.png")
            torchvision.utils.save_image(img, gen_path)
            save_hair_attention_diagnostics(
                attention_diagnostics,
                img,
                sample_dir,
            )

            rows.append([pair_id, str(gen_path), ref_id, ref_hair])

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
