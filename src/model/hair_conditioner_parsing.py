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

        # 1) убрать DataParallel префикс
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
        
        def _remap_key(k: str) -> str:
            nk = k
        
            # 0) убрать частые верхние префиксы (разные repo сохраняют как model./net./bisenet.)
            for prefix in ("model.", "bisenet.", "net."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
        
            # A) нормализуем backbone / context_path
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
        
            # B) привести conv1/bn1 к твоей структуре (Sequential(conv,bn,relu))
            if nk == "cp.backbone.conv1.weight":
                nk = "cp.backbone.conv1.0.weight"
        
            if nk.startswith("cp.backbone.bn1."):
                nk = "cp.backbone.conv1.1." + nk[len("cp.backbone.bn1."):]
        
            return nk

        # 2) применить ремап
        remapped = {}
        for k, v in sd.items():
            remapped[_remap_key(k)] = v
        sd = remapped

        # 3) загрузить
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
    ):
        super().__init__()
        self.device = device
        self.n_tokens = int(n_tokens)
        self.cross_dim = int(cross_dim)
        self.bg_value = float(bg_value)
        self.debug_save = bool(debug_save)

        self.enc_h = HairSegmentationEncoder(
            hair_weights_path, device=device, hair_class=hair_class
        )

        self.clip = CLIPVisionModel.from_pretrained(clip_vision_id, torch_dtype=clip_dtype).to(device).eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self.proc = CLIPImageProcessor.from_pretrained(clip_vision_id)

        in_dim = self.clip.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, self.n_tokens * self.cross_dim),
        ).to(device=device, dtype=proj_dtype)

    @torch.no_grad()
    def _pooled_hair(self, pil_images):
        masks = self.enc_h(pil_images)  # (B,512,512)
        hair_pil = [apply_mask_to_pil(im, masks[i], bg=self.bg_value) for i, im in enumerate(pil_images)]

        if self.debug_save and len(hair_pil) > 0:
            hair_pil[0].save("debug_hair.png")

        inputs = self.proc(images=hair_pil, return_tensors="pt").to(self.device)
        pooled = self.clip(**inputs).pooler_output  # (B,in_dim)
        return pooled

    def forward(self, pil_images, out_dtype: torch.dtype):
        pooled = self._pooled_hair(pil_images)
        pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-6)
        tokens = self.proj(pooled.float()).view(-1, self.n_tokens, self.cross_dim)
        tokens = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-6)
        tokens = tokens * 6.0
        return tokens.to(dtype=out_dtype)
