import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image

from transformers import CLIPVisionModel, CLIPImageProcessor


# BiSeNet (face parsing)
# Это минимальная реализация "обёртки".
# Сам BiSeNet-архитектуру - отдельным файлом,
from src.model.bisenet import BiSeNet


class HairSegmentationEncoder(nn.Module):
    """
    Enc_H: PIL -> hair mask (binary)
    """
    def __init__(self, weights_path: str, device="cuda"):
        super().__init__()
        self.device = device
        self.net = BiSeNet(n_classes=19).to(device).eval()
        sd = torch.load(weights_path, map_location=device)
        self.net.load_state_dict(sd, strict=True)

        for p in self.net.parameters():
            p.requires_grad = False

        self.tf = T.Compose([
            T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])

        # Для самых популярных весов face-parsing.PyTorch: hair = 17.
        self.hair_class = 17

    @torch.no_grad()
    def forward(self, pil_images):
        masks = []
        for im in pil_images:
            x = self.tf(im.convert("RGB")).unsqueeze(0).to(self.device)  # (1,3,512,512)
            logits = self.net(x)[0]  # (1,19,512,512) или (1,19,...) в зависимости от реализации
            parsing = logits.argmax(dim=1)[0]  # (512,512)
            hair = (parsing == self.hair_class).float()  # (512,512)
            masks.append(hair)
        return torch.stack(masks, dim=0)  # (B,512,512)


def apply_mask_to_pil(pil: Image.Image, mask_512: torch.Tensor, bg=0):
    """
    pil: исходная картинка любого размера
    mask_512: (512,512) float {0,1}
    возвращает PIL с выделенными волосами
    """
    im = pil.convert("RGB").resize((512, 512), Image.BILINEAR)
    arr = np.array(im).astype(np.float32) / 255.0  # (512,512,3)
    m = mask_512.cpu().numpy().astype(np.float32)  # (512,512)

    m3 = np.stack([m, m, m], axis=-1)
    out = arr * m3 + bg * (1.0 - m3)

    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)


class HairConditioner(nn.Module):
    """
    Полная hair ветка как на схеме:
    y2 -> Enc_H -> hair_image -> F() -> proj -> tokens
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
    ):
        super().__init__()
        self.device = device
        self.n_tokens = n_tokens
        self.cross_dim = cross_dim
        self.bg_value = float(bg_value)

        # Enc_H
        self.enc_h = HairSegmentationEncoder(hair_weights_path, device=device)

        # F(): CLIPVision pooled
        self.clip = CLIPVisionModel.from_pretrained(clip_vision_id, torch_dtype=clip_dtype).to(device).eval()
        for p in self.clip.parameters():
            p.requires_grad = False
        self.proc = CLIPImageProcessor.from_pretrained(clip_vision_id)

        in_dim = self.clip.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, n_tokens * cross_dim),
        ).to(device=device, dtype=proj_dtype)

        self.clip_dtype = clip_dtype
        self.proj_dtype = proj_dtype

    @torch.no_grad()
    def _pooled_hair(self, pil_images):
        masks = self.enc_h(pil_images)  # (B,512,512)
        hair_pil = [
            apply_mask_to_pil(im, masks[i], bg=self.bg_value) for i, im in enumerate(pil_images)
        ]
        inputs = self.proc(images=hair_pil, return_tensors="pt").to(self.device)
        pooled = self.clip(**inputs).pooler_output  # (B,in_dim)
        return pooled

    def forward(self, pil_images, out_dtype: torch.dtype):
        pooled = self._pooled_hair(pil_images)
        tokens_fp32 = self.proj(pooled.float()).view(-1, self.n_tokens, self.cross_dim)
        return tokens_fp32.to(dtype=out_dtype)
