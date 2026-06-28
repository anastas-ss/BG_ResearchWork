
import csv
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


class PairedImageDataset(Dataset):
    """
    Dataset for identity-disjoint conditional training/evaluation rows.

    Expected CSV columns:
      - target or ref_id: image used as denoising target and identity condition
      - ref_hair: image used as hair condition

    The training split builder uses single-image self-reconstruction, so target,
    ref_id and ref_hair contain the same path. Inference CSV files may use
    different ref_id and ref_hair images.

    Optional columns are preserved only as paths/metadata in downstream logs.
    """

    def __init__(self, csv_path: str, image_size: int = 512):
        self.csv_path = Path(csv_path)
        self.root = self.csv_path.parent
        self.rows = []

        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise RuntimeError(f"Empty pairs CSV: {self.csv_path}")
            fieldnames = set(reader.fieldnames)
            if "ref_hair" not in fieldnames:
                raise RuntimeError(f"{self.csv_path} must contain ref_hair column")
            if ("target" not in fieldnames) and ("ref_id" not in fieldnames):
                raise RuntimeError(f"{self.csv_path} must contain target or ref_id column")

            for row in reader:
                target = row.get("target") or row.get("ref_id")
                ref_id = row.get("ref_id") or target
                ref_hair = row.get("ref_hair")
                if not target or not ref_hair:
                    continue
                self.rows.append(
                    {
                        "pair_id": row.get("pair_id", str(len(self.rows))),
                        "target": self._resolve(target),
                        "ref_id": self._resolve(ref_id),
                        "ref_hair": self._resolve(ref_hair),
                        "target_cluster": row.get("target_cluster", ""),
                        "hair_cluster": row.get("hair_cluster", ""),
                    }
                )

        self.tf = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

        if len(self.rows) == 0:
            raise RuntimeError(f"No usable rows found in {self.csv_path}")

    def _resolve(self, value: str) -> str:
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = self.root / p
        return str(p)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        target_pil = Image.open(row["target"]).convert("RGB")
        id_pil = Image.open(row["ref_id"]).convert("RGB")
        hair_pil = Image.open(row["ref_hair"]).convert("RGB")
        x = self.tf(target_pil)

        return {
            "pixel_values": x,
            "pil": target_pil,
            "path": row["target"],
            "id_pil": id_pil,
            "id_path": row["ref_id"],
            "hair_pil": hair_pil,
            "hair_path": row["ref_hair"],
            "pair_id": row["pair_id"],
            "target_cluster": row["target_cluster"],
            "hair_cluster": row["hair_cluster"],
        }
