import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor

from src.model.bisenet import BiSeNet


class HairSegmentationEncoder(nn.Module):
    """
    Enc_H: PIL -> hair mask (binary) using BiSeNet face parsing.
    Returns: (B, 512, 512) float mask in {0,1}
    """
    def __init__(self, weights_path: str, device="cuda", hair_class: int = 17):
        super().__init__()
        self.device = device
        self.hair_class = int(hair_class)

        self.net = BiSeNet(n_classes=19).to(device).eval()

        sd = torch.load(weights_path, map_location="cpu")
        if isinstance(sd, dict):
            for key in ("state_dict", "model", "net", "params"):
                if key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
                    break

        if not isinstance(sd, dict):
            raise ValueError(f"Unsupported weights format: {type(sd)}")

        # убираем DataParallel префикс
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
        
        def _remap_key(k: str) -> str:
            nk = k
        
            # 0) убираем частые верхние префиксы (разные repo сохраняют как model./net./bisenet.)
            for prefix in ("model.", "bisenet.", "net."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
        
            # нормализуем backbone / context_path
            if nk.startswith("context_path.resnet."):
                nk = "cp.backbone." + nk[len("context_path.resnet."):]
            elif nk.startswith("context_path.backbone."):
                nk = "cp.backbone." + nk[len("context_path.backbone."):]
        
            if nk.startswith("cp.resnet."):
                nk = "cp.backbone." + nk[len("cp.resnet."):]
            elif nk.startswith("resnet."):
                nk = "cp.backbone." + nk[len("resnet."):]
            elif nk.startswith("backbone."):
                nk = "cp.backbone." + nk[len("backbone."):]
        
            # приводим conv1/bn1 к твоей структуре (Sequential(conv,bn,relu))
            if nk == "cp.backbone.conv1.weight":
                nk = "cp.backbone.conv1.0.weight"
        
            if nk.startswith("cp.backbone.bn1."):
                nk = "cp.backbone.conv1.1." + nk[len("cp.backbone.bn1."):]
        
            return nk

        # ремап
        remapped = {}
        for k, v in sd.items():
            remapped[_remap_key(k)] = v
        sd = remapped

        # загрузить
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        if len(missing) or len(unexpected):
            print("[HairSegEnc] load_state_dict strict=False")
            if len(missing):
                print("  missing keys:", missing[:10], "... total:", len(missing))
            if len(unexpected):
                print("  unexpected keys:", unexpected[:10], "... total:", len(unexpected))

        for p in self.net.parameters():
            p.requires_grad = False

        self.tf = T.Compose([
            T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
    
    @torch.no_grad()
    def forward(self, pil_images):
        xs = [self.tf(im.convert("RGB")) for im in pil_images]
        x = torch.stack(xs, dim=0).to(self.device)  # (B,3,512,512)

        out = self.net(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out  # (B,19,512,512)

        parsing = logits.argmax(dim=1)  # (B,512,512)
        hair = (parsing == self.hair_class).float()  # (B,512,512)

        # лёгкая дилатация, чтобы маска была чуть “плотнее”
        hair = F.max_pool2d(hair.unsqueeze(1), kernel_size=3, stride=1, padding=1)[:, 0]
        return hair


def apply_mask_to_pil(pil: Image.Image, mask_512: torch.Tensor, bg=0.0):
    im = pil.convert("RGB").resize((512, 512), Image.BILINEAR)
    arr = np.array(im).astype(np.float32) / 255.0
    m = mask_512.detach().cpu().numpy().astype(np.float32)
    m3 = np.stack([m, m, m], axis=-1)
    out = arr * m3 + float(bg) * (1.0 - m3)
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)
    
def remove_hair_from_pil(pil: Image.Image, mask_512: torch.Tensor, fill=0.5):
    """
    mask_512: (512,512) 1=hair, 0=not hair
    Возвращает изображение, где hair регион закрашен fill (0..1).
    """
    im = pil.convert("RGB").resize((512, 512), Image.BILINEAR)
    arr = np.array(im).astype(np.float32) / 255.0

    m = mask_512.detach().cpu().numpy().astype(np.float32)  # 1=hair
    m3 = np.stack([m, m, m], axis=-1)

    out = arr * (1.0 - m3) + float(fill) * m3
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)

class HairConditioner(nn.Module):
    """
    Hair branch:
    y2 -> Enc_H -> hair_image -> CLIPVision -> proj -> tokens
    """
    def __init__(
        self,
        clip_vision_id: str,
        n_tokens: int,
        cross_dim: int,
        hair_weights_path: str,
        device="cuda",
        clip_dtype=torch.float16,
        proj_dtype=torch.float32,
        bg_value=0.0,
        hair_class: int = 17,
        debug_save: bool = False,
        token_mode: str = "global",
        patch_mask_threshold: float = 0.05,
        max_patch_tokens: int = 64,
        patch_post_layernorm: bool = False,
        patch_binary_mask: bool = False,
        apply_token_mask_to_values: bool = True,
        token_normalization: str = "l2",
    ):
        super().__init__()
        self.device = device
        self.n_tokens = int(n_tokens)
        self.cross_dim = int(cross_dim)
        self.bg_value = float(bg_value)
        self.debug_save = bool(debug_save)
        self.token_mode = str(token_mode)
        self.patch_mask_threshold = float(patch_mask_threshold)
        self.max_patch_tokens = int(max_patch_tokens)
        self.patch_post_layernorm = bool(patch_post_layernorm)
        self.patch_binary_mask = bool(patch_binary_mask)
        self.apply_token_mask_to_values = bool(apply_token_mask_to_values)
        self.token_normalization = str(token_normalization)
        self.last_debug_stats = {}
        if self.token_mode not in {"global", "patch"}:
            raise ValueError(f"token_mode must be 'global' or 'patch', got: {self.token_mode}")
        if self.token_normalization not in {"l2", "layernorm", "none"}:
            raise ValueError(
                "token_normalization must be 'l2', 'layernorm', or 'none', "
                f"got: {self.token_normalization}"
            )
        if self.token_mode == "global" and self.token_normalization == "layernorm":
            raise ValueError("layernorm token normalization is currently supported only in patch mode")

        self.enc_h = HairSegmentationEncoder(
            hair_weights_path, device=device, hair_class=hair_class
        )

        self.clip = CLIPVisionModel.from_pretrained(clip_vision_id, torch_dtype=clip_dtype).to(device).eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self.proc = CLIPImageProcessor.from_pretrained(clip_vision_id)

        in_dim = self.clip.config.hidden_size
        if self.token_mode == "global":
            self.proj = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, self.n_tokens * self.cross_dim),
            ).to(device=device, dtype=proj_dtype)
        else:
            projection_layers = [
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, self.cross_dim),
            ]
            if self.token_normalization == "layernorm":
                projection_layers.append(nn.LayerNorm(self.cross_dim))
            self.proj = nn.Sequential(*projection_layers).to(
                device=device,
                dtype=proj_dtype,
            )

    @torch.no_grad()
    def get_hair_masks(self, pil_images):
        return self.enc_h(pil_images)  # (B,512,512) float {0,1}
    
    @torch.no_grad()
    def _pooled_hair(self, pil_images, masks=None):
        if masks is None:
            masks = self.enc_h(pil_images)  # (B,512,512)
        hair_pil = [apply_mask_to_pil(im, masks[i], bg=self.bg_value) for i, im in enumerate(pil_images)]

        if self.debug_save and len(hair_pil) > 0:
            hair_pil[0].save("debug_hair.png")

        inputs = self.proc(images=hair_pil, return_tensors="pt").to(self.device)
        pooled = self.clip(**inputs).pooler_output  # (B,in_dim)
        return pooled

    @torch.no_grad()
    def _patch_hair(self, pil_images):
        masks = self.enc_h(pil_images)
        hair_pil = [apply_mask_to_pil(im, masks[i], bg=self.bg_value) for i, im in enumerate(pil_images)]

        if self.debug_save and len(hair_pil) > 0:
            hair_pil[0].save("debug_hair.png")

        inputs = self.proc(images=hair_pil, return_tensors="pt").to(self.device)
        patch_tokens = self.clip(**inputs).last_hidden_state[:, 1:]
        raw_patch_norm = patch_tokens.detach().float().norm(dim=-1)
        if self.patch_post_layernorm:
            patch_tokens = self.clip.vision_model.post_layernorm(patch_tokens)
        normalized_patch_norm = patch_tokens.detach().float().norm(dim=-1)
        patch_count = patch_tokens.shape[1]
        patch_grid = int(round(patch_count ** 0.5))
        if patch_grid * patch_grid != patch_count:
            raise RuntimeError(f"CLIP patch count must form a square grid, got: {patch_count}")

        patch_mask = F.interpolate(
            masks.unsqueeze(1),
            size=(patch_grid, patch_grid),
            mode="area",
        )[:, 0].flatten(1)
        patch_mask = torch.where(
            patch_mask >= self.patch_mask_threshold,
            patch_mask,
            torch.zeros_like(patch_mask),
        )
        if 0 < self.max_patch_tokens < patch_count:
            patch_mask, patch_indices = patch_mask.topk(
                self.max_patch_tokens,
                dim=1,
                largest=True,
                sorted=False,
            )
            patch_tokens = torch.gather(
                patch_tokens,
                dim=1,
                index=patch_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1]),
            )
        if self.patch_binary_mask:
            patch_mask = (patch_mask > 0).to(dtype=patch_mask.dtype)
        active = patch_mask > 0
        self.last_debug_stats = {
            "raw_patch_norm_mean": raw_patch_norm.mean().item(),
            "post_layernorm_patch_norm_mean": normalized_patch_norm.mean().item(),
            "active_patch_count_mean": active.float().sum(dim=1).mean().item(),
            "active_patch_weight_mean": (
                patch_mask[active].mean().item() if active.any() else 0.0
            ),
            "spatial_mask_coverage": masks.float().mean().item(),
        }
        return patch_tokens, patch_mask, masks

    def forward(
        self,
        pil_images,
        out_dtype: torch.dtype,
        return_masks: bool = False,
    ):
        if self.token_mode == "global":
            masks = self.enc_h(pil_images)
            pooled = self._pooled_hair(pil_images, masks=masks)
            pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-6)
            tokens = self.proj(pooled.float()).view(-1, self.n_tokens, self.cross_dim)
            token_mask = torch.ones(
                tokens.shape[:2],
                device=tokens.device,
                dtype=torch.float32,
            )
        else:
            patch_tokens, token_mask, masks = self._patch_hair(pil_images)
            tokens = self.proj(patch_tokens.float())

        projected_token_norm = tokens.detach().float().norm(dim=-1)
        if self.token_normalization == "l2":
            tokens = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-6)
        if self.apply_token_mask_to_values:
            tokens = tokens * token_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        active = token_mask > 0
        self.last_debug_stats.update(
            {
                "projected_token_norm_mean": projected_token_norm.mean().item(),
                "final_active_token_norm_mean": (
                    tokens.detach().float().norm(dim=-1)[active].mean().item()
                    if active.any()
                    else 0.0
                ),
            }
        )
        tokens = tokens.to(dtype=out_dtype)
        if return_masks:
            return tokens, token_mask, masks
        return tokens
