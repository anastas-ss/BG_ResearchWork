import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# InsightFace
from insightface.app import FaceAnalysis


import numpy as np
import torch
from insightface.app import FaceAnalysis


class InsightFaceArcFaceEmbedder:
    def __init__(self, device="cuda", min_size=256, max_size=640, step=64):
        """
        det_size будет перебираться:
        640, 576, 512, ..., 256
        """
        ctx_id = 0 if device.startswith("cuda") else -1
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=(max_size, max_size))

        # список размеров для перебора
        self.det_sizes = [(size, size) for size in range(max_size, min_size - 1, -step)]

    def __call__(self, pil_images):
        embs = []
        has_face = []

        for im in pil_images:
            img = np.array(im.convert("RGB"))[:, :, ::-1]  # RGB -> BGR

            faces = []
            used_size = None

            # Перебор det_size
            for ds in self.det_sizes:
                self.app.det_model.input_size = ds
                faces = self.app.get(img)
                if len(faces) > 0:
                    used_size = ds
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

        embs = np.stack(embs, axis=0)
        return torch.from_numpy(embs), torch.tensor(has_face, dtype=torch.bool)


class IDArcFaceConditioner(nn.Module):
    """
    ID shortcut ветка:
    y1 -> (det+align+ArcFace) -> f_id -> proj -> tokens -> в cross-attn
    """
    def __init__(
        self,
        n_tokens: int,
        cross_dim: int,
        device="cuda",
        proj_dtype=torch.float32,
    ):
        super().__init__()
        self.device = device
        self.n_tokens = n_tokens
        self.cross_dim = cross_dim

        self.embedder = InsightFaceArcFaceEmbedder(device=device)
        in_dim = 512

        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, n_tokens * cross_dim),
        ).to(device=device, dtype=proj_dtype)

    @torch.no_grad()
    def _emb(self, pil_images):
        emb = self.embedder(pil_images).to(self.device)  # fp32
        return emb

    def forward(self, pil_images, out_dtype: torch.dtype):
        emb = self._emb(pil_images).float()
        tokens = self.proj(emb).view(-1, self.n_tokens, self.cross_dim)
        return tokens.to(dtype=out_dtype)
