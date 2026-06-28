#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def convert_file(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty pairs CSV: {input_path}")
        required = {"target", "ref_id", "ref_hair", "target_cluster", "hair_cluster"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise RuntimeError(f"{input_path} is missing columns: {sorted(missing)}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            image_path = row["target"]
            row["ref_id"] = image_path
            row["ref_hair"] = image_path
            row["hair_cluster"] = row["target_cluster"]
            writer.writerow(row)

    with output_path.open("r", newline="", encoding="utf-8") as f:
        converted_rows = list(csv.DictReader(f))
    for row in converted_rows:
        if not (row["target"] == row["ref_id"] == row["ref_hair"]):
            raise RuntimeError(f"Conversion failed for pair {row.get('pair_id', '')}")
        if row["target_cluster"] != row["hair_cluster"]:
            raise RuntimeError(f"Cluster conversion failed for pair {row.get('pair_id', '')}")
    return len(converted_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert identity-disjoint pair CSV files to single-image self-reconstruction rows."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in ("train", "val"):
        input_path = input_dir / f"{split}_pairs.csv"
        output_path = out_dir / f"{split}_pairs.csv"
        counts[split] = convert_file(input_path, output_path)
        print(f"[done] {output_path} rows={counts[split]}")

    source_manifest = input_dir / "identity_manifest.csv"
    if source_manifest.exists():
        target_manifest = out_dir / "identity_manifest.csv"
        target_manifest.write_bytes(source_manifest.read_bytes())

    summary = {
        "pairing_mode": "single_image_self_reconstruction",
        "source_dir": str(input_dir),
        "train_pairs": counts["train"],
        "val_pairs": counts["val"],
    }
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
