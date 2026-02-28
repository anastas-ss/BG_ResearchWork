```python
# metrics.py
# Evaluation for A2F project (SD1.5 + DualImageAttnProcessor):
# - ID preservation: ArcFace cosine similarity
# - Hair mask overlap: IoU / Dice
# - Hair perceptual distances on hair region: CLIP cosine distance, DINO cosine distance
# - Image quality: FID (Inception-v3 pool3) and FID-CLIP
#
# Assumptions:
# - You have pairs.csv with columns: pair_id, ref_id, ref_hair
#   ref_id   = identity reference image (source_img_id)
#   ref_hair = hair reference image     (source_img_hair)
# - You have generated images saved in a folder, e.g.:
#   out_dir/{pair_id}_{seed}.png  (you can change pattern with --gen_pattern)
#
# Usage examples:
# 1) Basic metrics (IDSim + hair IoU/Dice + CLIP/DINO hair):
#   python metrics.py \
#     --pairs_csv /content/pairs.csv \
#     --gen_dir /content/runs/exp/infer/images \
#     --hair_weights /content/bisenet.pth \
#     --device cuda
#
# 2) Add FID/FID-CLIP (needs enough images, recommend >= 100):
#   python metrics.py \
#     --pairs_csv /content/pairs.csv \
#     --gen_dir /content/runs/exp/infer/images \
#     --hair_weights /content/bisenet.pth \
#     --compute_fid 1 \
#     --fid_preprocess face_parsing \
#     --face_parsing_weights /content/bisenet.pth
#
# Notes:
# - FID protocol “face detect + align + remove background via parsing” is implemented as an option.
#   If face detection fails, sample is skipped for FID (but still can be used for other metrics).
#
# Dependencies:
#   pip install -U transformers insightface opencv-python scipy tqdm pandas
#
import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------- InsightFace (ArcFace + optional align) ----------
from insightface.app import FaceAnalysis

try:
    from insightface.utils.face_align import norm_crop
except Exception:
    norm_crop = None

# ---------- CLIP / DINO ----------
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import AutoModel, AutoImageProcessor

# ---------- Inception for FID ----------
import torchvision.transforms as T
from torchvision.models import inception_v3

try:
    from scipy import linalg
except Exception:
    linalg = None


# =========================
# Utils
# =========================
def load_pil(path: str, rgb=True) -> Image.Image:
    im = Image.open(path)
    if rgb:
        im = im.convert("RGB")
    return im


def pil_to_numpy_uint8(im: Image.Image) -> np.ndarray:
    return np.array(im.convert("RGB"))


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps=1e-8) -> torch.Tensor:
    # a,b: [B,D] or [D]
    if a.dim() == 1:
        a = a[None, :]
    if b.dim() == 1:
        b = b[None, :]
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


# =========================
# ArcFace embedder + face detection
# =========================
class ArcFaceEmbedder:
    """
    Uses InsightFace 'buffalo_l' (det+rec).
    - embed(): returns (emb, has_face, face_obj)
    - align(): if possible returns aligned face crop for FID preprocessing
    """

    def __init__(self, device="cuda", det_size=(640, 640)):
        ctx_id = 0 if device.startswith("cuda") else -1
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, pil: Image.Image):
        img = pil_to_numpy_uint8(pil)[:, :, ::-1]  # RGB->BGR
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        return faces[0]

    @torch.no_grad()
    def embed(self, pil: Image.Image) -> Tuple[torch.Tensor, bool]:
        face = self.detect(pil)
        if face is None:
            emb = torch.zeros(512, dtype=torch.float32)
            return emb, False
        emb = torch.from_numpy(face.embedding.astype(np.float32))
        return emb, True

    def align_face(self, pil: Image.Image, image_size=256) -> Optional[Image.Image]:
        """
        Returns aligned face as PIL.Image if possible.
        Uses norm_crop (5-point align) if available and face has kps.
        """
        face = self.detect(pil)
        if face is None:
            return None
        if norm_crop is None:
            return None
        if getattr(face, "kps", None) is None:
            return None

        img = pil_to_numpy_uint8(pil)[:, :, ::-1]  # BGR
        try:
            crop = norm_crop(img, face.kps, image_size=image_size)  # BGR uint8
            crop = crop[:, :, ::-1]  # RGB
            return Image.fromarray(crop)
        except Exception:
            return None


# =========================
# Hair segmentation (BiSeNet parsing)
# We reuse your BiSeNet wrapper if available.
# =========================
def load_bisenet_from_repo(weights_path: str, device="cuda", hair_class: int = 17):
    """
    Tries to import your repo BiSeNet + the same weight remapping logic.
    If you want 100% consistency with training, point to the same file.
    """
    from src.model.bisenet import BiSeNet  # your file

    net = BiSeNet(n_classes=19).to(device).eval()

    sd = torch.load(weights_path, map_location="cpu")
    if isinstance(sd, dict):
        for key in ("state_dict", "model", "net", "params"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    if not isinstance(sd, dict):
        raise ValueError(f"Unsupported weights format: {type(sd)}")

    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}

    def _remap_key(k: str) -> str:
        nk = k
        for prefix in ("model.", "bisenet.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]

        if nk.startswith("context_path.resnet."):
            nk = "cp.backbone." + nk[len("context_path.resnet.") :]
        elif nk.startswith("context_path.backbone."):
            nk = "cp.backbone." + nk[len("context_path.backbone.") :]

        if nk.startswith("cp.resnet."):
            nk = "cp.backbone." + nk[len("cp.resnet.") :]
        elif nk.startswith("resnet."):
            nk = "cp.backbone." + nk[len("resnet.") :]
        elif nk.startswith("backbone."):
            nk = "cp.backbone." + nk[len("backbone.") :]

        if nk == "cp.backbone.conv1.weight":
            nk = "cp.backbone.conv1.0.weight"
        if nk.startswith("cp.backbone.bn1."):
            nk = "cp.backbone.conv1.1." + nk[len("cp.backbone.bn1.") :]

        return nk

    remapped = {_remap_key(k): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(remapped, strict=False)
    if len(missing) or len(unexpected):
        print("[BiSeNet] load_state_dict strict=False")
        if len(missing):
            print("  missing keys:", missing[:10], "... total:", len(missing))
        if len(unexpected):
            print("  unexpected keys:", unexpected[:10], "... total:", len(unexpected))

    for p in net.parameters():
        p.requires_grad = False

    tf = T.Compose(
        [
            T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    @torch.no_grad()
    def hair_mask(pil: Image.Image) -> torch.Tensor:
        x = tf(pil.convert("RGB")).unsqueeze(0).to(device)  # [1,3,512,512]
        out = net(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out  # [1,19,512,512]
        parsing = logits.argmax(dim=1)[0]  # [512,512]
        mask = (parsing == int(hair_class)).float()
        return mask

    @torch.no_grad()
    def parsing_mask(pil: Image.Image) -> torch.Tensor:
        # full parsing map: [512,512] long
        x = tf(pil.convert("RGB")).unsqueeze(0).to(device)
        out = net(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        parsing = logits.argmax(dim=1)[0]
        return parsing

    return hair_mask, parsing_mask


def compute_iou_dice(mask_a: torch.Tensor, mask_b: torch.Tensor, eps=1e-8) -> Tuple[float, float]:
    # masks: [H,W] float {0,1}
    a = (mask_a > 0.5).float()
    b = (mask_b > 0.5).float()
    inter = (a * b).sum().item()
    union = (a + b - a * b).sum().item()
    iou = inter / (union + eps)
    dice = (2.0 * inter) / (a.sum().item() + b.sum().item() + eps)
    return float(iou), float(dice)


def apply_mask_to_pil(im: Image.Image, mask_512: torch.Tensor, bg=0.0) -> Image.Image:
    im = im.convert("RGB").resize((512, 512), Image.BILINEAR)
    arr = np.array(im).astype(np.float32) / 255.0
    m = mask_512.detach().float().cpu().numpy().astype(np.float32)
    m3 = np.stack([m, m, m], axis=-1)
    out = arr * m3 + float(bg) * (1.0 - m3)
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)


# =========================
# CLIP + DINO embedders
# =========================
class VisionEmbedder:
    def __init__(
        self,
        device="cuda",
        clip_id="openai/clip-vit-base-patch32",
        dino_id="facebook/dinov2-base",  # you can swap to a DINOv3 id if you have one
        clip_dtype=torch.float16,
        dino_dtype=torch.float16,
    ):
        self.device = device

        self.clip_proc = CLIPImageProcessor.from_pretrained(clip_id)
        self.clip = CLIPVisionModel.from_pretrained(clip_id, torch_dtype=clip_dtype).to(device).eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self.dino_proc = AutoImageProcessor.from_pretrained(dino_id)
        self.dino = AutoModel.from_pretrained(dino_id, torch_dtype=dino_dtype).to(device).eval()
        for p in self.dino.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def clip_embed(self, pil: Image.Image) -> torch.Tensor:
        inputs = self.clip_proc(images=pil, return_tensors="pt").to(self.device)
        out = self.clip(**inputs).pooler_output  # [1,D]
        out = out[0].float()
        return out

    @torch.no_grad()
    def dino_embed(self, pil: Image.Image) -> torch.Tensor:
        inputs = self.dino_proc(images=pil, return_tensors="pt").to(self.device)
        out = self.dino(**inputs)
        # Prefer pooler_output if exists; else CLS token
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output[0].float()
        else:
            feat = out.last_hidden_state[:, 0, :][0].float()
        return feat


# =========================
# FID implementation
# =========================
class InceptionPool3:
    """
    Torchvision inception_v3, returns pool3 features (2048).
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.model = inception_v3(weights="DEFAULT", transform_input=False, aux_logits=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.tf = T.Compose(
            [
                T.Resize((299, 299), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),  # [0,1]
            ]
        )

    @torch.no_grad()
    def __call__(self, pil_images: List[Image.Image], batch_size=32) -> np.ndarray:
        feats = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            x = torch.stack([self.tf(im.convert("RGB")) for im in batch], dim=0).to(self.device)
            # normalize like torchvision Inception expects: [-1,1]
            x = x * 2.0 - 1.0
            # forward up to pooling
            # We can get features by hooking, but simplest: use model without final FC by editing forward:
            # torchvision inception_v3 returns logits; we’ll re-run its internal blocks.
            m = self.model
            x = m.Conv2d_1a_3x3(x)
            x = m.Conv2d_2a_3x3(x)
            x = m.Conv2d_2b_3x3(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = m.Conv2d_3b_1x1(x)
            x = m.Conv2d_4a_3x3(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = m.Mixed_5b(x)
            x = m.Mixed_5c(x)
            x = m.Mixed_5d(x)
            x = m.Mixed_6a(x)
            x = m.Mixed_6b(x)
            x = m.Mixed_6c(x)
            x = m.Mixed_6d(x)
            x = m.Mixed_6e(x)
            x = m.Mixed_7a(x)
            x = m.Mixed_7b(x)
            x = m.Mixed_7c(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)  # [B,2048]
            feats.append(x.float().cpu().numpy())
        return np.concatenate(feats, axis=0)


def _compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    """
    sqrtm for PSD matrices.
    Prefer scipy if available; else eigen decomposition.
    """
    if linalg is not None:
        s = linalg.sqrtm(mat)
        if np.iscomplexobj(s):
            s = s.real
        return s
    # eigen fallback
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)[None, :]) @ v.T


def frechet_distance(mu1, s1, mu2, s2, eps=1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    s1 = np.atleast_2d(s1)
    s2 = np.atleast_2d(s2)

    diff = mu1 - mu2
    covmean = _sqrtm_psd(s1 @ s2)

    # numeric stability
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = _sqrtm_psd((s1 + off) @ (s2 + off))

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2.0 * tr_covmean
    return float(np.real(fid))


# =========================
# FID preprocessing (face detect + align + background remove)
# =========================
@dataclass
class FIDPreprocessConfig:
    mode: str = "none"  # none | face_align | face_parsing
    face_size: int = 256
    bg_value: float = 0.0
    # parsing classes to keep (non-background); for 19-class face parsing, 0 is often background.
    keep_nonzero: bool = True


def preprocess_for_fid(
    pil: Image.Image,
    arcface: ArcFaceEmbedder,
    fid_cfg: FIDPreprocessConfig,
    parsing_fn=None,  # parsing_mask(pil)->[512,512] long
) -> Optional[Image.Image]:
    """
    Returns PIL image used for FID feature extraction, or None if can't preprocess (e.g., no face for align modes).
    """
    if fid_cfg.mode == "none":
        return pil.convert("RGB")

    if fid_cfg.mode in ("face_align", "face_parsing"):
        aligned = arcface.align_face(pil, image_size=fid_cfg.face_size)
        if aligned is None:
            return None
        pil = aligned.convert("RGB")

    if fid_cfg.mode == "face_align":
        return pil

    if fid_cfg.mode == "face_parsing":
        if parsing_fn is None:
            raise ValueError("fid_preprocess=face_parsing requires parsing_fn (BiSeNet parsing).")
        # compute parsing on original-size expectation; we’ll just run parsing on resized to 512 inside parsing_fn
        parsing = parsing_fn(pil)  # [512,512] long
        if fid_cfg.keep_nonzero:
            mask = (parsing != 0).float()
        else:
            # if you want custom keep set, change here
            mask = (parsing != 0).float()
        return apply_mask_to_pil(pil, mask, bg=fid_cfg.bg_value)

    raise ValueError(f"Unknown fid preprocess mode: {fid_cfg.mode}")


# =========================
# Main evaluation
# =========================
def resolve_generated_path(gen_dir: str, pair_id: int, seed: int, pattern: str) -> str:
    # pattern can use {pair_id} and {seed}
    return str(Path(gen_dir) / pattern.format(pair_id=pair_id, seed=seed))


def parse_seeds(seeds_str: str) -> List[int]:
    # "0,1,2" or "0-3"
    seeds_str = seeds_str.strip()
    if "-" in seeds_str:
        a, b = seeds_str.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in seeds_str.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--gen_dir", type=str, required=True)
    ap.add_argument("--gen_pattern", type=str, default="{pair_id}_{seed}.png")
    ap.add_argument("--seeds", type=str, default="0")  # "0" or "0,1,2" or "0-3"

    ap.add_argument("--device", type=str, default="cuda")

    # Hair parsing (for hair IoU/Dice + hair-region masking)
    ap.add_argument("--hair_weights", type=str, required=True)
    ap.add_argument("--hair_class", type=int, default=17)
    ap.add_argument("--hair_bg", type=float, default=0.0)

    # ArcFace
    ap.add_argument("--arcface_det", type=int, default=640)

    # CLIP / DINO models
    ap.add_argument("--clip_id", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--dino_id", type=str, default="facebook/dinov2-base")

    # FID options
    ap.add_argument("--compute_fid", type=int, default=0)
    ap.add_argument("--fid_preprocess", type=str, default="none", choices=["none", "face_align", "face_parsing"])
    ap.add_argument("--fid_face_size", type=int, default=256)
    ap.add_argument("--fid_bg", type=float, default=0.0)

    # output
    ap.add_argument("--out_csv", type=str, default="metrics_results.csv")

    args = ap.parse_args()

    device = args.device
    seeds = parse_seeds(args.seeds)

    pairs = pd.read_csv(args.pairs_csv)
    # allow either pair_id column or implicit row index
    if "pair_id" not in pairs.columns:
        pairs = pairs.reset_index().rename(columns={"index": "pair_id"})
    assert "ref_id" in pairs.columns and "ref_hair" in pairs.columns, "pairs.csv must have ref_id and ref_hair"

    # init models
    arcface = ArcFaceEmbedder(device=device, det_size=(args.arcface_det, args.arcface_det))
    hair_mask_fn, parsing_fn = load_bisenet_from_repo(args.hair_weights, device=device, hair_class=args.hair_class)
    vis = VisionEmbedder(device=device, clip_id=args.clip_id, dino_id=args.dino_id)

    fid_cfg = FIDPreprocessConfig(
        mode=args.fid_preprocess,
        face_size=args.fid_face_size,
        bg_value=args.fid_bg,
        keep_nonzero=True,
    )

    # for FID feature extraction
    inception = InceptionPool3(device=device) if args.compute_fid else None

    rows = []
    # For FID we’ll collect two sets: generated and real (targets)
    fid_gen_imgs: List[Image.Image] = []
    fid_real_imgs: List[Image.Image] = []
    # For FID-CLIP we’ll collect CLIP embeddings
    fidclip_gen: List[np.ndarray] = []
    fidclip_real: List[np.ndarray] = []

    missing_gen = 0

    for _, r in tqdm(pairs.iterrows(), total=len(pairs), desc="pairs"):
        pair_id = int(r["pair_id"])
        ref_id_path = str(r["ref_id"])
        ref_hair_path = str(r["ref_hair"])

        ref_id_pil = load_pil(ref_id_path)
        ref_hair_pil = load_pil(ref_hair_path)

        # reference embeddings / masks
        ref_id_emb, ref_has_face = arcface.embed(ref_id_pil)
        ref_hair_m = hair_mask_fn(ref_hair_pil)  # [512,512]
        ref_hair_masked_for_clip = apply_mask_to_pil(ref_hair_pil, ref_hair_m, bg=args.hair_bg)
        ref_clip_h = vis.clip_embed(ref_hair_masked_for_clip)
        ref_dino_h = vis.dino_embed(ref_hair_masked_for_clip)

        # For each seed: evaluate generated image
        for seed in seeds:
            gen_path = resolve_generated_path(args.gen_dir, pair_id, seed, args.gen_pattern)
            if not os.path.exists(gen_path):
                missing_gen += 1
                continue

            gen_pil = load_pil(gen_path)

            # IDSim
            gen_id_emb, gen_has_face = arcface.embed(gen_pil)
            idsim = float(cosine_sim(gen_id_emb, ref_id_emb).item())

            # Hair IoU/Dice (mask from generated vs ref_hair)
            gen_hair_m = hair_mask_fn(gen_pil)
            iou, dice = compute_iou_dice(gen_hair_m, ref_hair_m)

            # Hair perceptual distances on masked region
            gen_hair_masked = apply_mask_to_pil(gen_pil, gen_hair_m, bg=args.hair_bg)
            gen_clip_h = vis.clip_embed(gen_hair_masked)
            gen_dino_h = vis.dino_embed(gen_hair_masked)

            d_clip = float(1.0 - cosine_sim(gen_clip_h, ref_clip_h).item())
            d_dino = float(1.0 - cosine_sim(gen_dino_h, ref_dino_h).item())

            rows.append(
                {
                    "pair_id": pair_id,
                    "seed": seed,
                    "gen_path": gen_path,
                    "ref_id": ref_id_path,
                    "ref_hair": ref_hair_path,
                    "ref_has_face": int(ref_has_face),
                    "gen_has_face": int(gen_has_face),
                    "IDSim_arcface": idsim,
                    "Hair_IoU": iou,
                    "Hair_Dice": dice,
                    "dCLIP_hair": d_clip,
                    "dDINO_hair": d_dino,
                }
            )

            # collect for FID / FID-CLIP
            if args.compute_fid:
                g = preprocess_for_fid(gen_pil, arcface, fid_cfg, parsing_fn=parsing_fn)
                rr = preprocess_for_fid(ref_hair_pil, arcface, fid_cfg, parsing_fn=parsing_fn)
                # Here for "real" we use ref_hair as target distribution (you can switch to true dataset split if needed)
                if g is not None and rr is not None:
                    fid_gen_imgs.append(g)
                    fid_real_imgs.append(rr)

                    # FID-CLIP embeddings on same preprocessed images
                    g_clip = vis.clip_embed(g).cpu().numpy()
                    r_clip = vis.clip_embed(rr).cpu().numpy()
                    fidclip_gen.append(g_clip)
                    fidclip_real.append(r_clip)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[metrics] saved per-sample metrics -> {args.out_csv}")
    if missing_gen:
        print(f"[metrics] WARNING: missing generated files: {missing_gen}")

    # Aggregate summary
    if len(df) > 0:
        agg = df.groupby("seed")[["IDSim_arcface", "Hair_IoU", "Hair_Dice", "dCLIP_hair", "dDINO_hair"]].mean()
        print("\n[metrics] mean by seed:\n", agg)

        agg_all = df[["IDSim_arcface", "Hair_IoU", "Hair_Dice", "dCLIP_hair", "dDINO_hair"]].mean()
        print("\n[metrics] overall mean:\n", agg_all)

        # how often face fails
        print("\n[metrics] face detection fail rate:")
        print("  ref_has_face=0:", float((df["ref_has_face"] == 0).mean()))
        print("  gen_has_face=0:", float((df["gen_has_face"] == 0).mean()))

    # FID / FID-CLIP
    if args.compute_fid:
        if len(fid_gen_imgs) < 10 or len(fid_real_imgs) < 10:
            print("[FID] Not enough samples collected for FID (need more).")
            return

        # Inception features
        gen_feats = inception(fid_gen_imgs, batch_size=32)
        real_feats = inception(fid_real_imgs, batch_size=32)
        mu_g, sg_g = _compute_stats(gen_feats)
        mu_r, sg_r = _compute_stats(real_feats)
        fid = frechet_distance(mu_g, sg_g, mu_r, sg_r)
        print(f"\n[FID] FID (Inception pool3), preprocess={args.fid_preprocess}: {fid:.4f}")

        # FID-CLIP
        fidclip_gen_np = np.stack(fidclip_gen, axis=0)
        fidclip_real_np = np.stack(fidclip_real, axis=0)
        mu_g2, sg_g2 = _compute_stats(fidclip_gen_np)
        mu_r2, sg_r2 = _compute_stats(fidclip_real_np)
        fid_clip = frechet_distance(mu_g2, sg_g2, mu_r2, sg_r2)
        print(f"[FID-CLIP] FID-CLIP, preprocess={args.fid_preprocess}: {fid_clip:.4f}")


if __name__ == "__main__":
    main()
```
