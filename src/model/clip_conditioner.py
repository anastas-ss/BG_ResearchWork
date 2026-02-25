
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

class CLIPTokenConditioner(nn.Module):
    """
    Frozen CLIP vision -> pooled embedding -> trainable projection -> N tokens (cross_dim).
    Designed so projection can be fp32 while CLIP can be fp16.
    """
    def __init__(self, clip_vision_id: str, n_tokens: int, cross_dim: int, device="cuda", clip_dtype=torch.float16, proj_dtype=torch.float32):
        super().__init__()
        self.device = device
        self.n_tokens = n_tokens
        self.cross_dim = cross_dim

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
    def _pooled(self, pil_images):
        inputs = self.proc(images=pil_images, return_tensors="pt").to(self.device)
        pooled = self.clip(**inputs).pooler_output  # clip_dtype
        return pooled

    def forward(self, pil_images, out_dtype: torch.dtype):
        pooled = self._pooled(pil_images)
        pooled_fp32 = pooled.float()  # чтобы совпало с fp32 проекцией
        tokens_fp32 = self.proj(pooled_fp32).view(-1, self.n_tokens, self.cross_dim)
        return tokens_fp32.to(dtype=out_dtype)
