
from pathlib import Path
from typing import List
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageFolderDataset(Dataset):
    def __init__(self, root: str, image_size: int = 512):
        self.root = Path(root)
        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        paths: List[Path] = []
        for e in exts:
            paths += list(self.root.glob(e))
        self.paths = sorted(paths)

        self.tf = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),   # -> [-1,1]
        ])

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        pil = Image.open(p).convert("RGB")
        x = self.tf(pil)  # [-1,1]
    
        return {
            "pixel_values": x,
            "pil": pil,
            "path": str(p),
        }
