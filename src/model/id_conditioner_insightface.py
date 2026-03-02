import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from insightface.app import FaceAnalysis
import numpy as np
import torch


class InsightFaceArcFaceEmbedder:
    def __init__(self, device="cuda", min_size=256, max_size=640, step=64, model_root="./"):
        ctx_id = 0 if device.startswith("cuda") else -1
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )

        # Грузим ТОЛЬКО нужные модули
        self.app = FaceAnalysis(
            name="antelopev2",
            root=model_root,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )

        self.app.prepare(ctx_id=ctx_id, det_size=(max_size, max_size))

        # sanity check — чтобы точно грузился arcface
        print("Recognition model:", self.app.models["recognition"].model_file)

        self.det_sizes = [(size, size) for size in range(max_size, min_size - 1, -step)]

    def __call__(self, pil_images, return_mask=False):
        embs = []
        has_face = []

        for im in pil_images:
            img = np.array(im.convert("RGB"))[:, :, ::-1]

            faces = []
            for ds in self.det_sizes:
                self.app.det_model.input_size = ds
                faces = self.app.get(img)
                if len(faces) > 0:
                    break

            if len(faces) == 0:
                emb = np.zeros((512,), dtype=np.float32)
                has_face.append(False)
            else:
                faces = sorted(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True,
                )
                emb = faces[0].embedding.astype(np.float32)
                has_face.append(True)

            embs.append(emb)

        embs = torch.from_numpy(np.stack(embs, axis=0))
        mask = torch.tensor(has_face, dtype=torch.bool)

        if return_mask:
            return embs, mask
        return embs


class IDArcFaceConditioner(nn.Module):
    def __init__(self, n_tokens: int, cross_dim: int, device="cuda", proj_dtype=torch.float32):
        super().__init__()
        self.device = device
        self.n_tokens = n_tokens
        self.cross_dim = cross_dim

        self.embedder = InsightFaceArcFaceEmbedder(device=device, model_root="./")
        in_dim = 512

        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, n_tokens * cross_dim),
        ).to(device=device, dtype=proj_dtype)

    @torch.no_grad()
    def _emb(self, pil_images, return_mask: bool = False):
        out = self.embedder(pil_images, return_mask=return_mask)
        if return_mask:
            emb, mask = out
            return emb.to(self.device), mask.to(self.device)
        return out.to(self.device)

    def forward(self, pil_images, out_dtype: torch.dtype, return_mask: bool = False):
        if return_mask:
            emb, mask = self._emb(pil_images, return_mask=True)
            tokens = self.proj(emb.float()).view(-1, self.n_tokens, self.cross_dim)
            return tokens.to(dtype=out_dtype), mask

        emb = self._emb(pil_images, return_mask=False).float()
        tokens = self.proj(emb).view(-1, self.n_tokens, self.cross_dim)
        return tokens.to(dtype=out_dtype)
