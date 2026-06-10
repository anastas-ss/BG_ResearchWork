#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def make_rel_or_abs(path: Path, relative_to: Path | None) -> str:
    if relative_to is None:
        return str(path)
    try:
        return str(path.relative_to(relative_to))
    except ValueError:
        return str(path)


def choose_hair_source(
    *,
    row_idx: int,
    split_indices: list[int],
    labels: np.ndarray,
    rng: random.Random,
) -> int:
    target_label = labels[row_idx]
    candidates = [idx for idx in split_indices if idx != row_idx and labels[idx] != target_label]
    if not candidates:
        candidates = [idx for idx in split_indices if idx != row_idx]
    if not candidates:
        raise RuntimeError("Need at least two images to build paired CSV")
    return rng.choice(candidates)


def write_pairs(
    *,
    path: Path,
    split_name: str,
    indices: list[int],
    paths: list[Path],
    labels: np.ndarray,
    rng: random.Random,
    relative_to: Path | None,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id",
            "target",
            "ref_id",
            "ref_hair",
            "target_cluster",
            "hair_cluster",
            "split",
        ])
        for pair_id, idx in enumerate(indices):
            hair_idx = choose_hair_source(row_idx=idx, split_indices=indices, labels=labels, rng=rng)
            writer.writerow([
                pair_id,
                make_rel_or_abs(paths[idx], relative_to),
                make_rel_or_abs(paths[idx], relative_to),
                make_rel_or_abs(paths[hair_idx], relative_to),
                int(labels[idx]),
                int(labels[hair_idx]),
                split_name,
            ])
    return len(indices)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build ArcFace/DBSCAN identity clusters, identity-disjoint train/val split, "
            "and paired target/ref_hair CSV files."
        )
    )
    parser.add_argument("--image-root", required=True, help="Directory with input face images.")
    parser.add_argument("--out-dir", required=True, help="Output directory for manifest/splits/pairs.")
    parser.add_argument("--insightface-root", default=None, help="Directory containing models/antelopev2.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps under cosine distance.")
    parser.add_argument("--min-samples", type=int, default=4, help="DBSCAN min_samples.")
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Discard identity clusters smaller than this when building pairs.",
    )
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of identity clusters for validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--relative-paths",
        type=int,
        default=0,
        help="Write paths relative to image root instead of absolute paths.",
    )
    args = parser.parse_args()

    image_root = Path(args.image_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    paths = list_images(image_root)
    if not paths:
        raise SystemExit(f"No images found in {image_root}")

    from src.model.id_conditioner_insightface import InsightFaceArcFaceEmbedder

    embedder = InsightFaceArcFaceEmbedder(
        device=args.device,
        model_root=args.insightface_root,
    )

    embs = []
    has_face = []
    for i, path in enumerate(paths):
        if i % 100 == 0:
            print(f"[embed] {i}/{len(paths)}")
        pil = Image.open(path).convert("RGB")
        emb, mask = embedder([pil], return_mask=True)
        embs.append(emb.numpy()[0])
        has_face.append(bool(mask.numpy()[0]))

    embs_np = np.stack(embs, axis=0).astype(np.float32)
    norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
    embs_np = embs_np / np.clip(norms, 1e-12, None)

    try:
        from sklearn.cluster import DBSCAN
    except Exception as exc:
        raise SystemExit("scikit-learn is required: pip install scikit-learn") from exc

    face_indices = np.array([i for i, ok in enumerate(has_face) if ok], dtype=np.int64)
    labels = np.full((len(paths),), -1, dtype=np.int64)
    if len(face_indices) == 0:
        raise SystemExit("No detected faces; cannot cluster identities")

    clustering = DBSCAN(eps=float(args.eps), min_samples=int(args.min_samples), metric="cosine")
    labels_face = clustering.fit_predict(embs_np[face_indices])
    labels[face_indices] = labels_face

    cluster_sizes = Counter(int(x) for x in labels if int(x) >= 0)
    usable_clusters = sorted(
        cluster_id for cluster_id, size in cluster_sizes.items() if size >= int(args.min_cluster_size)
    )
    if len(usable_clusters) < 2:
        raise SystemExit(
            f"Need at least 2 usable clusters for cross-identity hair pairs; got {len(usable_clusters)}"
        )

    rng.shuffle(usable_clusters)
    n_val = max(1, int(round(len(usable_clusters) * float(args.val_frac))))
    n_val = min(n_val, len(usable_clusters) - 1)
    val_clusters = set(usable_clusters[:n_val])
    train_clusters = set(usable_clusters[n_val:])

    split_by_idx = []
    split_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label in train_clusters:
            split = "train"
        elif label in val_clusters:
            split = "val"
        else:
            split = "discard"
        split_by_idx.append(split)
        if split in {"train", "val"}:
            split_to_indices[split].append(idx)

    relative_to = image_root if int(args.relative_paths) else None
    manifest_path = out_dir / "identity_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "has_face", "cluster_id", "cluster_size", "split"])
        for path, ok, label, split in zip(paths, has_face, labels, split_by_idx):
            size = cluster_sizes.get(int(label), 0) if int(label) >= 0 else 0
            writer.writerow([make_rel_or_abs(path, relative_to), int(ok), int(label), int(size), split])

    train_pairs = out_dir / "train_pairs.csv"
    val_pairs = out_dir / "val_pairs.csv"
    n_train = write_pairs(
        path=train_pairs,
        split_name="train",
        indices=split_to_indices["train"],
        paths=paths,
        labels=labels,
        rng=rng,
        relative_to=relative_to,
    )
    n_val_pairs = write_pairs(
        path=val_pairs,
        split_name="val",
        indices=split_to_indices["val"],
        paths=paths,
        labels=labels,
        rng=rng,
        relative_to=relative_to,
    )

    summary = {
        "image_root": str(image_root),
        "n_images": len(paths),
        "n_detected_faces": int(sum(has_face)),
        "dbscan": {
            "eps": float(args.eps),
            "min_samples": int(args.min_samples),
            "metric": "cosine",
        },
        "n_clusters_total": len(cluster_sizes),
        "n_clusters_usable": len(usable_clusters),
        "min_cluster_size": int(args.min_cluster_size),
        "train_clusters": len(train_clusters),
        "val_clusters": len(val_clusters),
        "train_pairs": n_train,
        "val_pairs": n_val_pairs,
        "discarded_images": int(sum(1 for s in split_by_idx if s == "discard")),
        "outputs": {
            "manifest": str(manifest_path),
            "train_pairs": str(train_pairs),
            "val_pairs": str(val_pairs),
        },
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("[done] manifest:", manifest_path)
    print("[done] train_pairs:", train_pairs, "rows=", n_train)
    print("[done] val_pairs:", val_pairs, "rows=", n_val_pairs)
    print("[done] summary:", out_dir / "split_summary.json")


if __name__ == "__main__":
    main()
