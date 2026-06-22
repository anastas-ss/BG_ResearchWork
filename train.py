# train.py
# - Stable Diffusion (frozen backbone)
# - DualImageAttnProcessor injected into ALL cross-attn (attn2) blocks
# - External condition stream: Hair (Parsing + CLIP)
# - Identity is provided only through the Arc2Face text stream:
#   ArcFace(512) is injected into the CLIP text prompt token "id"
# - Qualitative samples compare hair_on / hair_off / empty_text variants
#
# Expected repo structure:
#   src/data/images.py                      -> ImageFolderDataset (returns pixel_values, pil, path)
#   src/model/dual_ip_attention.py          -> DualImageAttnProcessor
#   src/model/id_conditioner_insightface.py -> IDArcFaceConditioner (ArcFace extractor for Arc2Face text)
#   src/model/hair_conditioner_parsing.py   -> HairConditioner
#   src/utils/repro.py                      -> set_seed
#   src/utils/project_face_embs.py          -> project_face_embs (ArcFace->CLIP prompt embeds)
#
# Run:
#   python train.py --cfg config.yaml

import argparse
import inspect
import json
import time
from collections import defaultdict, OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml
import torchvision
from torchvision.transforms import functional as TVF

from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers import DPMSolverMultistepScheduler

from src.utils.repro import set_seed
from src.data.images import ImageFolderDataset, PairedImageDataset
from src.model.dual_ip_attention import DualImageAttnProcessor
from src.model.clip_text_model_wrapper import CLIPTextModelWrapper
from src.model.id_conditioner_insightface import IDArcFaceConditioner
from src.model.hair_conditioner_parsing import HairConditioner, apply_mask_to_pil
from src.utils.project_face_embs import project_face_embs


# Data collation (keep PIL)
def collate_keep_pil(batch_list):
    pixel_values = torch.stack([b["pixel_values"] for b in batch_list], dim=0)  # [B,3,H,W] float
    pil = [b["pil"] for b in batch_list]  # list[PIL.Image]
    path = [b["path"] for b in batch_list]
    out = {"pixel_values": pixel_values, "pil": pil, "path": path}
    optional_keys = [
        "id_pil",
        "id_path",
        "hair_pil",
        "hair_path",
        "pair_id",
        "target_cluster",
        "hair_cluster",
    ]
    for key in optional_keys:
        if key in batch_list[0]:
            out[key] = [b[key] for b in batch_list]
    return out

@torch.no_grad()
def sanity_check_tokens(pipe, id_cond, hair_cond, dl, dtype_unet, n_samples=4):
    """
    Проверяем, что:
    1) ArcFace extractor видит лица для Arc2Face text stream
    2) Hair токены ненулевые и trainable
    3) Отдельной ID-token ветки больше нет
    """
    batch = next(iter(dl))
    pil_images = batch.get("id_pil", batch["pil"])[:n_samples]
    hair_images = batch.get("hair_pil", batch["pil"])[:n_samples]
    B = len(pil_images)

    face_embs_512, face_mask = id_cond.extract_arcface_embs(pil_images, return_mask=True)

    # Hair tokens
    hair_tokens = hair_cond(hair_images, out_dtype=dtype_unet)
    hair_tokens = hair_tokens / (hair_tokens.norm(dim=-1, keepdim=True) + 1e-6)

    print("\n=== Sanity Check Tokens ===")
    print(f"face_mask: {face_mask.tolist()}")
    print(f"ArcFace emb mean abs: {face_embs_512.abs().mean().item():.4f}, norm mean: {face_embs_512.norm(dim=-1).mean().item():.4f}")
    print(f"Hair tokens mean abs: {hair_tokens.abs().mean().item():.4f}, norm mean: {hair_tokens.norm(dim=-1).mean().item():.4f}")

    # Проверка заморозки
    id_grad = any(p.requires_grad for p in id_cond.parameters())
    hair_grad = any(p.requires_grad for p in hair_cond.parameters())
    print(f"ArcFace extractor trainable? {id_grad} (должно быть False)")
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
def _save_hair_debug_triplet(
    *,
    run_dir: Path,
    step: int,
    pil_images,
    hair_masks: torch.Tensor,
    hair_cond,
):
    """
    Save: original | hair_mask | hair_masked
    Useful for verifying that selected hair_class highlights actual hair.
    """
    out_dir = run_dir / "hair_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    im = pil_images[0].convert("RGB").resize((512, 512))
    m = hair_masks[0].detach().float().cpu().clamp(0, 1)  # [512,512]
    mask_rgb = m.unsqueeze(0).repeat(3, 1, 1)
    masked = apply_mask_to_pil(im, hair_masks[0], bg=hair_cond.bg_value)

    t_orig = TVF.to_tensor(im)
    t_masked = TVF.to_tensor(masked)
    row = torch.stack([t_orig, mask_rgb, t_masked], dim=0)

    path = out_dir / f"step_{step:07d}.png"
    _save_row(row, str(path))
    print(f"[hair debug] saved {path} mask_coverage={m.mean().item():.4f}")

@torch.no_grad()
def _img_pair_metrics(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    """
    a,b: [1,3,H,W] in [0,1]
    returns: (l2_per_pixel, cosine_similarity)
    """
    av = a.float().reshape(a.shape[0], -1)
    bv = b.float().reshape(b.shape[0], -1)
    l2 = torch.sqrt(((av - bv) ** 2).mean(dim=1)).mean().item()
    cos = F.cosine_similarity(av, bv, dim=1, eps=eps).mean().item()
    return l2, cos


@torch.no_grad()
def select_cross_source_indices(
    hair_tokens: torch.Tensor,
    hair_masks: torch.Tensor | None = None,
    min_hair_coverage: float = 0.02,
) -> torch.Tensor:
    """
    Pick source-B index for each sample:
    prefer most dissimilar hair token among candidates with enough hair coverage.
    """
    bsz = hair_tokens.shape[0]
    device = hair_tokens.device
    if bsz < 2:
        return torch.arange(bsz, device=device, dtype=torch.long)

    flat = hair_tokens.detach().float().reshape(bsz, -1)
    flat = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
    sim = flat @ flat.t()
    sim.fill_diagonal_(2.0)

    if hair_masks is not None:
        cov = hair_masks.detach().float().mean(dim=(1, 2))
        eligible = cov >= float(min_hair_coverage)
    else:
        cov = None
        eligible = torch.ones(bsz, device=device, dtype=torch.bool)

    src = []
    for i in range(bsz):
        cand = eligible.clone()
        cand[i] = False
        if cand.any():
            row = sim[i].clone()
            row[~cand] = 2.0
            j = int(row.argmin().item())
        else:
            if cov is not None:
                cov_row = cov.clone()
                cov_row[i] = -1.0
                j = int(cov_row.argmax().item())
            else:
                j = (i + 1) % bsz
        src.append(j)
    return torch.tensor(src, device=device, dtype=torch.long)


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
    hair_pil_images=None,        # list[PIL] for hair condition; defaults to pil_images
    text_emb: torch.Tensor,      # [B,T,D] (dtype_unet)
    id_cond,
    hair_cond,
    dtype_unet: torch.dtype,
    num_steps: int,
    seed: int,
    save_hair_debug: bool = True,
    cross_min_hair_coverage: float = 0.02,
):
    """
    Saves: runs/<exp>/samples/step_XXXXXXX.png
    Row: [orig | hair_on | hair_off | empty_text_hair | empty_text_off | cross_hair(optional) | hair_source_B(optional) | hair_source_B_masked(optional)]
    """
    out_dir = run_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    B, _, H, W = pixel_values.shape
    vae_sf = pipe.vae_scale_factor if hasattr(pipe, "vae_scale_factor") else 8
    hair_pil_images = hair_pil_images if hair_pil_images is not None else pil_images

    face_mask = torch.ones(len(pil_images), device=pixel_values.device, dtype=torch.bool)

    hair_tokens = hair_cond(hair_pil_images, out_dtype=dtype_unet)
    hair_tokens = hair_tokens / (hair_tokens.norm(dim=-1, keepdim=True) + 1e-6)
    hair_masks = hair_cond.get_hair_masks(hair_pil_images)
    if save_hair_debug:
        _save_hair_debug_triplet(
            run_dir=run_dir,
            step=step,
            pil_images=hair_pil_images,
            hair_masks=hair_masks,
            hair_cond=hair_cond,
        )

    # Arc2Face text stream is always present. For empty-text ablations we feed
    # an explicit empty ArcFace embedding, not plain text.
    with torch.no_grad():
        face_embs_512, face_mask = id_cond.extract_arcface_embs(pil_images, return_mask=True)
        text_emb_empty = project_face_embs(pipe, torch.zeros_like(face_embs_512)).to(dtype_unet)
    
    print("has_face:", face_mask.tolist())

    print(
        f"[qual diag] mean|text|={text_emb.detach().float().abs().mean().item():.4f}  "
        f"mean|hair|={hair_tokens.detach().float().abs().mean().item():.4f}"
    )
    print(
        f"[qual diag] mean_norm text={text_emb.detach().float().norm(dim=-1).mean().item():.4f}  "
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

    variants = [
        ("hair_on",         text_emb,       hair_tokens,                   3.0),
        ("hair_off",        text_emb,       torch.zeros_like(hair_tokens), 3.0),
        ("empty_text_hair", text_emb_empty, hair_tokens,                   3.0),
        ("empty_text_off",  text_emb_empty, torch.zeros_like(hair_tokens), 3.0),
    ]
    cross_src_idx0 = None
    src_img_b = None
    src_img_b_masked = None
    if B >= 2:
        src_idx = select_cross_source_indices(
            hair_tokens=hair_tokens,
            hair_masks=hair_masks,
            min_hair_coverage=float(cross_min_hair_coverage),
        )
        hair_tokens_cross = hair_tokens[src_idx]
        cross_src_idx0 = int(src_idx[0].item())
        pil_b = hair_pil_images[cross_src_idx0].convert("RGB").resize((W, H))
        src_img_b = TVF.to_tensor(pil_b).unsqueeze(0).to(device=pixel_values.device, dtype=torch.float32)
        pil_b_masked = apply_mask_to_pil(pil_b, hair_masks[cross_src_idx0], bg=hair_cond.bg_value)
        src_img_b_masked = TVF.to_tensor(pil_b_masked).unsqueeze(0)
        variants.append(("cross_hair", text_emb, hair_tokens_cross, 3.0))

    rows = []
    row_by_tag = {}
    for tag, txt_t, hair_t, cfg_s in variants:
        enc_cond   = {"text": txt_t, "hair": hair_t}
        enc_uncond = {"text": text_emb_uc, "hair": torch.zeros_like(hair_t)}

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
        row_by_tag[tag] = img_01[:1]

    # Numeric diagnostics: if these stay ~0 over training, branch effect is weak.
    if "hair_on" in row_by_tag and "hair_off" in row_by_tag:
        l2_bi, cos_bi = _img_pair_metrics(row_by_tag["hair_on"], row_by_tag["hair_off"])
        print(f"[qual diff] hair_on vs hair_off: l2={l2_bi:.6f}, cos={cos_bi:.6f}")
    if "empty_text_hair" in row_by_tag and "empty_text_off" in row_by_tag:
        l2_hb, cos_hb = _img_pair_metrics(row_by_tag["empty_text_hair"], row_by_tag["empty_text_off"])
        print(f"[qual diff] empty_text_hair vs empty_text_off: l2={l2_hb:.6f}, cos={cos_hb:.6f}")
    if "cross_hair" in row_by_tag and "hair_on" in row_by_tag:
        l2_ch, cos_ch = _img_pair_metrics(row_by_tag["cross_hair"], row_by_tag["hair_on"])
        print(f"[qual diff] cross_hair vs hair_on: l2={l2_ch:.6f}, cos={cos_ch:.6f} src_idx0={cross_src_idx0}")

    orig_01 = (pixel_values[:1].float() * 0.5 + 0.5).clamp(0, 1)
    row_items = [orig_01] + rows
    if src_img_b is not None:
        row_items.append(src_img_b)
    if src_img_b_masked is not None:
        row_items.append(src_img_b_masked.to(dtype=orig_01.dtype, device=orig_01.device))
    row = torch.cat(row_items, dim=0)

    path = out_dir / f"step_{step:07d}.png"
    _save_row(row, str(path))
    print(f"[qual] saved {path}")


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_hair_classes(cond_cfg: dict):
    vals = cond_cfg.get("hair_classes", None)
    if vals is None:
        return [int(cond_cfg.get("hair_class", 17))]
    if isinstance(vals, int):
        return [int(vals)]
    if isinstance(vals, str):
        return [int(x.strip()) for x in vals.split(",") if x.strip()]
    return [int(v) for v in vals]


def build_hair_conditioner_compat(**kwargs):
    """
    Backward-compatible HairConditioner constructor.
    If running with an older HairConditioner signature, extra kwargs are dropped.
    """
    sig = inspect.signature(HairConditioner.__init__)
    supported = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted([k for k in kwargs.keys() if k not in supported])
    if dropped:
        print(f"[warn] HairConditioner ignores unsupported args on this code version: {dropped}")
    return HairConditioner(**filtered)


@torch.no_grad()
def extract_arcface_embs_cached(
    *,
    id_cond,
    pil_images,
    image_paths,
    cache: OrderedDict,
    max_items: int,
    use_cache: bool,
    stats: dict | None = None,
):
    """
    Caches ArcFace embeddings by image path in CPU RAM.
    Returns:
      embs: (B,512) float32 on id_cond.device
      mask: (B,) bool on id_cond.device
    """
    if (not use_cache) or (image_paths is None) or (len(image_paths) != len(pil_images)):
        return id_cond.extract_arcface_embs(pil_images, return_mask=True)

    bsz = len(pil_images)
    embs_cpu = [None] * bsz
    mask_list = [False] * bsz
    miss_idx = []
    miss_pil = []
    miss_paths = []

    for i, p in enumerate(image_paths):
        key = str(p)
        item = cache.get(key)
        if item is None:
            miss_idx.append(i)
            miss_pil.append(pil_images[i])
            miss_paths.append(key)
            if stats is not None:
                stats["miss"] += 1
        else:
            emb_cpu, has_face = item
            embs_cpu[i] = emb_cpu
            mask_list[i] = bool(has_face)
            cache.move_to_end(key)
            if stats is not None:
                stats["hit"] += 1

    if miss_idx:
        miss_embs, miss_mask = id_cond.extract_arcface_embs(miss_pil, return_mask=True)
        miss_embs_cpu = miss_embs.detach().cpu()
        miss_mask_cpu = miss_mask.detach().cpu()

        for local_i, key, emb_cpu, mk in zip(miss_idx, miss_paths, miss_embs_cpu, miss_mask_cpu):
            has_face = bool(mk.item())
            embs_cpu[local_i] = emb_cpu
            mask_list[local_i] = has_face
            cache[key] = (emb_cpu, has_face)
            if max_items > 0 and len(cache) > max_items:
                cache.popitem(last=False)

    embs = torch.stack(embs_cpu, dim=0).to(id_cond.device, dtype=torch.float32)
    mask = torch.tensor(mask_list, dtype=torch.bool, device=id_cond.device)
    return embs, mask


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
                scale_hair=float(cfg["cond"]["scale_hair"]),
                attn_fp32=True,
            ).to(device=device, dtype=torch.float32)  # keep this module stable in fp32

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

    insightface_device = str(cfg["models"].get("insightface_device", device)).lower()
    if insightface_device not in {"cuda", "cpu"}:
        raise ValueError(f"models.insightface_device must be 'cuda' or 'cpu', got: {insightface_device}")
    print(f"[init] insightface_device={insightface_device}")
    id_cond = IDArcFaceConditioner(
        n_tokens=n_tokens,
        cross_dim=cross_dim,
        device=insightface_device,
        proj_dtype=torch.float32,
        model_root=cfg["models"].get("insightface_root"),
    ).to(device)

    hair_classes = resolve_hair_classes(cfg["cond"])
    hair_mask_dilate_kernel = int(cfg["cond"].get("hair_mask_dilate_kernel", 3))
    hair_mask_dilate_iters = int(cfg["cond"].get("hair_mask_dilate_iters", 1))
    hair_focus_crop = bool(cfg["cond"].get("hair_focus_crop", False))
    hair_focus_crop_margin = float(cfg["cond"].get("hair_focus_crop_margin", 0.2))
    hair_focus_crop_square = bool(cfg["cond"].get("hair_focus_crop_square", True))
    hair_cond = build_hair_conditioner_compat(
        clip_vision_id=clip_id,
        n_tokens=n_tokens,
        cross_dim=cross_dim,
        hair_weights_path=hair_w,
        hair_class=int(hair_classes[0]),
        hair_classes=hair_classes,
        hair_mask_dilate_kernel=hair_mask_dilate_kernel,
        hair_mask_dilate_iters=hair_mask_dilate_iters,
        hair_focus_crop=hair_focus_crop,
        hair_focus_crop_margin=hair_focus_crop_margin,
        hair_focus_crop_square=hair_focus_crop_square,
        device=device,
        clip_dtype=torch.float16,
        proj_dtype=torch.float32,
        bg_value=float(cfg["cond"].get("hair_bg_value", 0.0)),
    ).to(device)
    enc_h = getattr(hair_cond, "enc_h", None)
    classes_logged = list(getattr(enc_h, "hair_classes", hair_classes))
    dilate_kernel_logged = getattr(enc_h, "dilate_kernel", hair_mask_dilate_kernel)
    dilate_iters_logged = getattr(enc_h, "dilate_iters", hair_mask_dilate_iters)
    focus_enabled_logged = getattr(hair_cond, "hair_focus_crop", hair_focus_crop)
    focus_margin_logged = getattr(hair_cond, "hair_focus_crop_margin", hair_focus_crop_margin)
    focus_square_logged = getattr(hair_cond, "hair_focus_crop_square", hair_focus_crop_square)

    print(f"[init] hair_classes={classes_logged}")
    print(
        "[init] hair_mask_dilate:",
        f"kernel={dilate_kernel_logged}",
        f"iters={dilate_iters_logged}",
    )
    print(
        "[init] hair_focus_crop:",
        f"enabled={focus_enabled_logged}",
        f"margin={focus_margin_logged}",
        f"square={focus_square_logged}",
    )
    print("[init] separate ID-token branch: disabled/removed")

    # ArcFace extractor is frozen; identity enters only through Arc2Face text embeddings.
    id_cond.eval()
    id_cond.requires_grad_(False)

    eval_cfg = cfg.get("eval", {})
    eval_enabled = bool(eval_cfg.get("enabled", False))
    eval_every = int(eval_cfg.get("every_steps", 200))

    # Data
    data_cfg = cfg["data"]
    image_size = int(data_cfg["image_size"])
    train_pairs_csv = data_cfg.get("train_pairs_csv")
    val_pairs_csv = data_cfg.get("val_pairs_csv")
    if train_pairs_csv:
        ds = PairedImageDataset(train_pairs_csv, image_size=image_size)
        print(f"[data] train paired csv={train_pairs_csv} rows={len(ds)}")
    else:
        ds = ImageFolderDataset(data_cfg["train_dir"], image_size=image_size)
        print(f"[data] train image dir={data_cfg['train_dir']} images={len(ds)}")
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 2)),
        pin_memory=True,
        collate_fn=collate_keep_pil,
        drop_last=True,
    )
    eval_dl = dl
    if eval_enabled:
        if val_pairs_csv:
            val_ds = PairedImageDataset(val_pairs_csv, image_size=image_size)
            eval_dl = DataLoader(
                val_ds,
                batch_size=int(cfg["train"]["batch_size"]),
                shuffle=False,
                num_workers=int(cfg["train"].get("num_workers", 2)),
                pin_memory=True,
                collate_fn=collate_keep_pil,
                drop_last=False,
            )
            print(f"[data] val paired csv={val_pairs_csv} rows={len(val_ds)}")
        elif data_cfg.get("val_dir") and data_cfg.get("val_dir") != data_cfg.get("train_dir"):
            val_ds = ImageFolderDataset(data_cfg["val_dir"], image_size=image_size)
            eval_dl = DataLoader(
                val_ds,
                batch_size=int(cfg["train"]["batch_size"]),
                shuffle=False,
                num_workers=int(cfg["train"].get("num_workers", 2)),
                pin_memory=True,
                collate_fn=collate_keep_pil,
                drop_last=False,
            )
            print(f"[data] val image dir={data_cfg['val_dir']} images={len(val_ds)}")
    if bool(cfg["train"].get("run_sanity_check", True)):
        sanity_check_tokens(pipe, id_cond, hair_cond, dl, dtype_unet)
    else:
        print("[sanity] skipped (train.run_sanity_check=false)")

    # Train only hair projection + hair branch inside DualImageAttnProcessor
    hair_cond.requires_grad_(False)
    hair_cond.proj.requires_grad_(True)

    for proc in unet.attn_processors.values():
        if isinstance(proc, DualImageAttnProcessor):
            proc.to_k_hair.requires_grad_(True)
            proc.to_v_hair.requires_grad_(True)

    dual_params = []
    for proc in unet.attn_processors.values():
        if isinstance(proc, DualImageAttnProcessor):
            dual_params += [p for p in proc.parameters() if p.requires_grad]

    train_params = list(hair_cond.proj.parameters()) + dual_params

    param_groups = [
        {"params": list(hair_cond.proj.parameters()), "lr": float(cfg["train"]["lr"])},
    ]
    param_groups.append(
        {
            "params": dual_params,
            "lr": float(cfg["train"]["lr"]) * float(cfg["train"].get("dual_lr_mult", 1.0)),
        }
    )

    opt = torch.optim.AdamW(
        param_groups,
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    def count_trainable(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("trainable arcface_extractor:", count_trainable(id_cond))
    print("trainable hair_cond:", count_trainable(hair_cond))
    print("trainable hair_proj:", count_trainable(hair_cond.proj))
    print(
        "lr hair_proj=", float(cfg["train"]["lr"]),
        "lr dual=",
        float(cfg["train"]["lr"]) * float(cfg["train"].get("dual_lr_mult", 1.0)),
        "hair_aux_weight=",
        float(cfg["train"].get("hair_aux_weight", 0.0)),
        "cross_hair_clip_weight=",
        float(cfg["train"].get("cross_hair_clip_weight", 0.0)),
        "cross_hair_contrast_weight=",
        float(cfg["train"].get("cross_hair_contrast_weight", 0.0)),
        "cross_hair_margin=",
        float(cfg["train"].get("cross_hair_margin", 0.1)),
        "cross_hair_clip_every=",
        int(cfg["train"].get("cross_hair_clip_every", 1)),
        "cross_hair_clip_batch=",
        int(cfg["train"].get("cross_hair_clip_batch", 2)),
        "cross_hair_decode_size=",
        int(cfg["train"].get("cross_hair_decode_size", 256)),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=True)

    fixed_batch = None
    if eval_enabled:
        fixed_batch = next(iter(eval_dl))

    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    eval_scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Optional sanity compare (text-only standard)
    if eval_cfg.get("sanity_compare", False):
        pipe.scheduler = eval_scheduler
        prompt_s = eval_cfg.get("prompt", "a portrait photo of a person")
        steps_s = int(eval_cfg.get("num_inference_steps", 30))
        seed_s = int(eval_cfg.get("seed", 123))

        tok = pipe.tokenizer([prompt_s], padding="max_length",
                             max_length=pipe.tokenizer.model_max_length,
                             return_tensors="pt").to(device)
        text_emb_s = pipe.text_encoder(**tok).last_hidden_state.to(dtype_unet)

        tok_uc = pipe.tokenizer([""], padding="max_length",
                                max_length=pipe.tokenizer.model_max_length,
                                return_tensors="pt").to(device)
        text_emb_uc_s = pipe.text_encoder(**tok_uc).last_hidden_state.to(dtype_unet)

        h0 = torch.zeros((1, 1, cross_dim), device=device, dtype=dtype_unet)

        enc_c = {"text": text_emb_s, "hair": h0}
        enc_u = {"text": text_emb_uc_s, "hair": h0}

        gen = torch.Generator(device=device).manual_seed(seed_s)
        lat0 = torch.randn((1, 4, 64, 64), device=device, dtype=dtype_unet, generator=gen)

        lat = sample_with_cfg(pipe, eval_scheduler, lat0, enc_c, enc_u, num_steps=steps_s, cfg_scale=7.0)
        img01 = _vae_decode_to_01(pipe, lat, dtype_unet)

        torchvision.utils.save_image(img01, str(run_dir / "sanity_text_only.png"))
        print("[sanity] saved", run_dir / "sanity_text_only.png")

    max_steps = int(cfg["train"]["max_steps"])
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])
    cache_arcface_embs = bool(cfg["train"].get("cache_arcface_embs", False))
    disable_arcface_runtime = bool(cfg["train"].get("disable_arcface_runtime", False))
    arcface_cache_max_items = int(cfg["train"].get("arcface_cache_max_items", 50000))
    arcface_cache = OrderedDict()
    arcface_cache_stats = {"hit": 0, "miss": 0}
    profile_timing = bool(cfg["train"].get("profile_timing", False))
    profile_sync_cuda = bool(cfg["train"].get("profile_sync_cuda", False))
    profile_every = int(cfg["train"].get("profile_every", log_every))
    only_both_face = bool(cfg["train"].get("only_both_face", False))
    timing_acc = defaultdict(float)
    timing_steps = 0
    face_filter_stats = {"kept": 0, "dropped": 0, "skipped_batches": 0}
    print(
        "arcface_cache=",
        cache_arcface_embs,
        "arcface_cache_max_items=",
        arcface_cache_max_items,
        "disable_arcface_runtime=",
        disable_arcface_runtime,
    )
    print("only_both_face=", only_both_face)
    if only_both_face and disable_arcface_runtime:
        raise ValueError("train.only_both_face=true requires disable_arcface_runtime=false")
    print(
        "[eval]",
        f"enabled={eval_enabled}",
        f"every_steps={eval_every}",
        f"samples_dir={run_dir / 'samples'}",
        f"hair_debug={bool(eval_cfg.get('debug_hair_masks', True))}",
    )

    def _sync_if_needed():
        if profile_timing and profile_sync_cuda and device == "cuda":
            torch.cuda.synchronize()

    def _t_start():
        if not profile_timing:
            return None
        _sync_if_needed()
        return time.perf_counter()

    def _t_stop(key: str, t0):
        if not profile_timing or t0 is None:
            return
        _sync_if_needed()
        timing_acc[key] += time.perf_counter() - t0

    # Modes
    unet.train()
    id_cond.eval()
    hair_cond.train()

    step = 0
    it = iter(dl)

    while step < max_steps:
        t_step_total = _t_start()

        t0 = _t_start()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        _t_stop("dataloader", t0)

        t0 = _t_start()
        pixel_values = batch["pixel_values"].to(device=device, dtype=dtype_unet)  # [-1,1], fp16
        pil_images = batch.get("id_pil", batch["pil"])  # identity/target images
        hair_pil_images = batch.get("hair_pil", batch["pil"])  # hair-condition images
        image_paths = batch.get("id_path", batch.get("path", None))
        hair_paths = batch.get("hair_path", None)
        B = pixel_values.shape[0]
        _t_stop("host_to_device", t0)

        # ---- Arc2Face text embedding ----
        t0 = _t_start()
        face_embs_512 = None
        with torch.no_grad():
            if disable_arcface_runtime:
                prompt_fb = cfg.get("train", {}).get("prompt", "a portrait photo of a person")
                tok_fb = pipe.tokenizer(
                    [prompt_fb] * B,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(device)
                text_emb = pipe.text_encoder(**tok_fb).last_hidden_state.to(dtype_unet)
                tok_empty = pipe.tokenizer(
                    [""] * B,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(device)
                text_emb_empty = pipe.text_encoder(**tok_empty).last_hidden_state.to(dtype_unet)
            else:
                face_embs_512, face_mask = extract_arcface_embs_cached(
                    id_cond=id_cond,
                    pil_images=pil_images,
                    image_paths=image_paths,
                    cache=arcface_cache,
                    max_items=arcface_cache_max_items,
                    use_cache=cache_arcface_embs,
                    stats=arcface_cache_stats,
                )  # (B,512), (B,)
                text_emb = project_face_embs(pipe, face_embs_512).to(dtype_unet)  # (B,T,H)
                text_emb_empty = project_face_embs(pipe, torch.zeros_like(face_embs_512)).to(dtype_unet)

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
        _t_stop("id_text_embed", t0)

        # Strict face-only training mode: keep only samples with detected face.
        if only_both_face:
            keep = face_mask
            keep_idx = keep.nonzero(as_tuple=False).squeeze(1)
            n_keep = int(keep_idx.numel())
            face_filter_stats["kept"] += n_keep
            face_filter_stats["dropped"] += int(B - n_keep)
            if n_keep == 0:
                face_filter_stats["skipped_batches"] += 1
                if step % log_every == 0:
                    print(f"[step {step}/{max_steps}] skip batch: no face-detected samples")
                continue
            if n_keep < B:
                pixel_values = pixel_values[keep_idx]
                pil_images = [pil_images[i] for i in keep_idx.tolist()]
                hair_pil_images = [hair_pil_images[i] for i in keep_idx.tolist()]
                if image_paths is not None:
                    image_paths = [image_paths[i] for i in keep_idx.tolist()]
                if hair_paths is not None:
                    hair_paths = [hair_paths[i] for i in keep_idx.tolist()]
                face_embs_512 = face_embs_512[keep_idx]
                face_mask = face_mask[keep_idx]
                text_emb = text_emb[keep_idx]
                text_emb_empty = text_emb_empty[keep_idx]
                B = n_keep

        # VAE encode -> latents
        t0 = _t_start()
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor  # [B,4,h,w]

        # Add diffusion noise
        noise = torch.randn_like(latents)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
        noisy = scheduler.add_noise(latents, noise, t).to(dtype=dtype_unet)
        _t_stop("vae_noise", t0)

        # Hair tokens
        t0 = _t_start()
        hair_tokens = hair_cond(hair_pil_images, out_dtype=dtype_unet)  # [B, n_tokens, cross_dim]
        hair_tokens = hair_tokens / (hair_tokens.norm(dim=-1, keepdim=True) + 1e-6)
        src_idx = None
        hair_tokens_cross = None
        _t_stop("hair_tokens", t0)
        
        # Сборка conditioning
        t0 = _t_start()
        enc = {"text": text_emb, "hair": hair_tokens}
        enc_hair_only = {"text": text_emb_empty, "hair": hair_tokens}
        hair_aux_weight = float(cfg["train"].get("hair_aux_weight", 0.0))
        cross_hair_clip_weight = float(cfg["train"].get("cross_hair_clip_weight", 0.0))
        cross_hair_contrast_weight = float(cfg["train"].get("cross_hair_contrast_weight", 0.0))
        cross_hair_margin = float(cfg["train"].get("cross_hair_margin", 0.1))
        cross_hair_clip_every = int(cfg["train"].get("cross_hair_clip_every", 1))
        cross_hair_clip_batch = int(cfg["train"].get("cross_hair_clip_batch", 2))
        cross_hair_decode_size = int(cfg["train"].get("cross_hair_decode_size", 256))
        cross_min_hair_coverage = float(cfg["train"].get("cross_min_hair_coverage", 0.02))
        need_cross_clip = (
            ((cross_hair_clip_weight > 0.0) or (cross_hair_contrast_weight > 0.0))
            and (B >= 2)
            and (step % max(1, cross_hair_clip_every) == 0)
        )

        # reference hair CLIP embeddings from source B (detached target)
        if need_cross_clip:
            with torch.no_grad():
                # memory saver: compute cross loss only on a random subset of batch
                cross_bs = min(B, max(1, cross_hair_clip_batch))
                if cross_bs < B:
                    cross_idx = torch.randperm(B, device=device)[:cross_bs]
                else:
                    cross_idx = torch.arange(B, device=device)

                hair_masks_all = hair_cond.get_hair_masks(hair_pil_images).detach()    # [B,512,512]
                src_idx = select_cross_source_indices(
                    hair_tokens=hair_tokens,
                    hair_masks=hair_masks_all,
                    min_hair_coverage=cross_min_hair_coverage,
                )
                hair_tokens_cross = hair_tokens[src_idx]

                noisy_cross = noisy[cross_idx]
                t_cross = t[cross_idx]
                enc_cross = {
                    "text": text_emb[cross_idx],
                    "hair": hair_tokens_cross[cross_idx],
                }

                ref_pooled_all = hair_cond._pooled_hair(hair_pil_images).detach().float()  # [B, D]
                # positive target: source-B hair embedding
                ref_pooled_pos = ref_pooled_all[src_idx][cross_idx]                    # [Bc, D]
                ref_pooled_pos = ref_pooled_pos / (ref_pooled_pos.norm(dim=-1, keepdim=True) + 1e-6)
                # negative target: original source-A hair embedding
                ref_pooled_neg = hair_cond._pooled_hair(pil_images).detach().float()[cross_idx]  # [Bc, D]
                ref_pooled_neg = ref_pooled_neg / (ref_pooled_neg.norm(dim=-1, keepdim=True) + 1e-6)
                hair_masks = hair_masks_all[src_idx][cross_idx].unsqueeze(1).to(device=device, dtype=torch.float32)
        _t_stop("cross_prep", t0)

        # Train step (predict noise)
        opt.zero_grad(set_to_none=True)

        t0 = _t_start()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            noise_pred = pipe.unet(noisy, t, encoder_hidden_states=enc).sample
            loss_main = F.mse_loss(noise_pred.float(), noise.float())
            loss = loss_main
            if hair_aux_weight > 0.0:
                noise_pred_h = pipe.unet(noisy, t, encoder_hidden_states=enc_hair_only).sample
                loss_hair = F.mse_loss(noise_pred_h.float(), noise.float())
                loss = loss + hair_aux_weight * loss_hair
            else:
                loss_hair = torch.zeros((), device=device, dtype=loss_main.dtype)

            if need_cross_clip:
                noise_pred_cross = pipe.unet(noisy_cross, t_cross, encoder_hidden_states=enc_cross).sample
            else:
                noise_pred_cross = None
        _t_stop("unet_forward", t0)

        t0 = _t_start()
        if need_cross_clip:
            # Predict x0 from eps-prediction and compute CLIP similarity on hair-masked generated image.
            b_cross = noisy_cross.shape[0]
            alpha_t = scheduler.alphas_cumprod.to(device=noisy.device, dtype=torch.float32)[t_cross].view(b_cross, 1, 1, 1)
            x0_cross = (noisy_cross.float() - (1.0 - alpha_t).sqrt() * noise_pred_cross.float()) / alpha_t.sqrt()
            x0_cross = x0_cross.to(dtype=dtype_unet)

            decode_lat = x0_cross
            if 64 <= cross_hair_decode_size < 512 and (cross_hair_decode_size % 8 == 0):
                lat_sz = cross_hair_decode_size // 8
                decode_lat = F.interpolate(decode_lat, size=(lat_sz, lat_sz), mode="bilinear", align_corners=False)

            img_cross = pipe.vae.decode(decode_lat / pipe.vae.config.scaling_factor).sample  # [-1,1]
            img_cross_01 = (img_cross.float() * 0.5 + 0.5).clamp(0, 1)

            if hair_masks.shape[-2:] != img_cross_01.shape[-2:]:
                hair_masks = F.interpolate(hair_masks, size=img_cross_01.shape[-2:], mode="nearest")
            bg = float(cfg["cond"].get("hair_bg_value", 0.0))
            img_cross_masked = img_cross_01 * hair_masks + bg * (1.0 - hair_masks)

            clip_size = int(hair_cond.clip.config.image_size)
            clip_in = F.interpolate(img_cross_masked, size=(clip_size, clip_size), mode="bicubic", align_corners=False)
            clip_mean = torch.tensor(hair_cond.proc.image_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
            clip_std = torch.tensor(hair_cond.proc.image_std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
            clip_in = (clip_in - clip_mean) / clip_std

            pooled_cross = hair_cond.clip(pixel_values=clip_in).pooler_output.float()
            pooled_cross = pooled_cross / (pooled_cross.norm(dim=-1, keepdim=True) + 1e-6)
            d_pos = 1.0 - (pooled_cross * ref_pooled_pos).sum(dim=-1)
            d_neg = 1.0 - (pooled_cross * ref_pooled_neg).sum(dim=-1)

            loss_cross_clip = d_pos.mean()
            loss_cross_contrast = F.relu(cross_hair_margin + d_pos - d_neg).mean()

            if cross_hair_clip_weight > 0.0:
                loss = loss + cross_hair_clip_weight * loss_cross_clip
            if cross_hair_contrast_weight > 0.0:
                loss = loss + cross_hair_contrast_weight * loss_cross_contrast
        else:
            loss_cross_clip = torch.zeros((), device=device, dtype=loss_main.dtype)
            loss_cross_contrast = torch.zeros((), device=device, dtype=loss_main.dtype)
        _t_stop("cross_loss", t0)

        if not torch.isfinite(loss):
            print(f"[step {step}] loss non-finite -> skipping")
            continue

        t0 = _t_start()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(train_params, float(cfg["train"].get("grad_clip", 1.0)))
        scaler.step(opt)
        scaler.update()
        _t_stop("backward_opt", t0)

        if step % log_every == 0:
            print(
                f"[step {step}/{max_steps}] loss={loss.item():.6f} "
                f"(main={loss_main.item():.6f}, hair_aux={loss_hair.item():.6f}, "
                f"cross_clip={loss_cross_clip.item():.6f}, cross_ctr={loss_cross_contrast.item():.6f})"
            )
            if only_both_face:
                print(
                    "[face_filter] kept_total=",
                    face_filter_stats["kept"],
                    "dropped_total=",
                    face_filter_stats["dropped"],
                    "skipped_batches=",
                    face_filter_stats["skipped_batches"],
                )
            if cache_arcface_embs:
                req = arcface_cache_stats["hit"] + arcface_cache_stats["miss"]
                if req > 0:
                    hit_rate = 100.0 * arcface_cache_stats["hit"] / req
                    print(
                        f"[arcface_cache] size={len(arcface_cache)} "
                        f"hit={arcface_cache_stats['hit']} miss={arcface_cache_stats['miss']} "
                        f"hit_rate={hit_rate:.1f}%"
                    )

        # Qualitative sampling
        if eval_enabled:
            if step % eval_every == 0:
                t0 = _t_start()
                qb = fixed_batch if fixed_batch is not None else batch
                q_pixel = qb["pixel_values"].to(device=device, dtype=dtype_unet)
                q_pil = qb.get("id_pil", qb["pil"])
                q_hair_pil = qb.get("hair_pil", qb["pil"])
                q_paths = qb.get("id_path", qb.get("path", None))
                q_hair_paths = qb.get("hair_path", None)
                qB = q_pixel.shape[0]

                with torch.no_grad():
                    q_face_embs_512, q_face_mask = extract_arcface_embs_cached(
                        id_cond=id_cond,
                        pil_images=q_pil,
                        image_paths=q_paths,
                        cache=arcface_cache,
                        max_items=arcface_cache_max_items,
                        use_cache=cache_arcface_embs,
                        stats=arcface_cache_stats,
                    )
                    q_text_emb = project_face_embs(pipe, q_face_embs_512).to(dtype_unet)

                    if q_text_emb.shape[0] != qB:
                        if q_text_emb.shape[0] == 1:
                            q_text_emb = q_text_emb.repeat(qB, 1, 1)
                        else:
                            raise RuntimeError(f"q_text_emb batch mismatch: got {q_text_emb.shape[0]} vs qB={qB}")

                    if (~q_face_mask).any():
                        eval_prompt = eval_cfg.get("prompt", "a portrait photo of a person")
                        q_tok_fb = pipe.tokenizer(
                            [eval_prompt] * qB,
                            padding="max_length",
                            max_length=pipe.tokenizer.model_max_length,
                            return_tensors="pt",
                        ).to(device)
                        q_text_fb = pipe.text_encoder(**q_tok_fb).last_hidden_state.to(dtype_unet)
                        q_text_emb = q_text_emb.clone()
                        q_text_emb[~q_face_mask] = q_text_fb[~q_face_mask]

                    if only_both_face:
                        q_keep_idx = q_face_mask.nonzero(as_tuple=False).squeeze(1)
                        q_keep = int(q_keep_idx.numel())
                        if q_keep == 0:
                            print("[eval] skip qualitative_check: no face-detected samples in eval batch")
                            _t_stop("eval_sample", t0)
                            continue
                        if q_keep < qB:
                            q_pixel = q_pixel[q_keep_idx]
                            q_pil = [q_pil[i] for i in q_keep_idx.tolist()]
                            q_hair_pil = [q_hair_pil[i] for i in q_keep_idx.tolist()]
                            if q_paths is not None:
                                q_paths = [q_paths[i] for i in q_keep_idx.tolist()]
                            if q_hair_paths is not None:
                                q_hair_paths = [q_hair_paths[i] for i in q_keep_idx.tolist()]
                            q_face_embs_512 = q_face_embs_512[q_keep_idx]
                            q_face_mask = q_face_mask[q_keep_idx]
                            q_text_emb = q_text_emb[q_keep_idx]
                            qB = int(q_pixel.shape[0])

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
                    hair_pil_images=q_hair_pil,
                    text_emb=q_text_emb,
                    id_cond=id_cond,
                    hair_cond=hair_cond,
                    dtype_unet=dtype_unet,
                    num_steps=int(eval_cfg.get("num_inference_steps", 50)),
                    seed=int(eval_cfg.get("seed", 123)),
                    save_hair_debug=bool(eval_cfg.get("debug_hair_masks", True)),
                    cross_min_hair_coverage=float(cfg["train"].get("cross_min_hair_coverage", 0.02)),
                )

                # restore modes
                if was_unet_train:
                    unet.train()
                if was_id_train:
                    id_cond.train()
                if was_hair_train:
                    hair_cond.train()
                _t_stop("eval_sample", t0)

        # Save checkpoint
        if step % save_every == 0 or (step + 1) == max_steps:
            t0 = _t_start()
            ckpt = {
                "step": step,
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
            _t_stop("checkpoint_save", t0)

        _t_stop("step_total", t_step_total)
        if profile_timing:
            timing_steps += 1
            if step % max(1, profile_every) == 0:
                total = sum(timing_acc.values())
                if total > 0 and timing_steps > 0:
                    ordered = sorted(timing_acc.items(), key=lambda kv: kv[1], reverse=True)
                    parts = [
                        f"{k}={v/timing_steps:.3f}s ({100.0*v/total:.1f}%)"
                        for k, v in ordered
                    ]
                    print(f"[timing avg/{timing_steps} steps] " + " | ".join(parts))
                timing_acc.clear()
                timing_steps = 0

        step += 1

    print("Done. Run dir:", run_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml")
    args = ap.parse_args()
    main(args.cfg)
