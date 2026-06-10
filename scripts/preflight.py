#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
import yaml


def check_path(label: str, p: str, expect_dir: bool):
    path = Path(os.path.expanduser(p))
    ok = path.is_dir() if expect_dir else path.is_file()
    kind = "dir" if expect_dir else "file"
    print(f"[{'OK' if ok else 'MISS'}] {label} ({kind}): {path}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.cluster.yaml")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checks = []
    train_pairs_csv = cfg["data"].get("train_pairs_csv")
    val_pairs_csv = cfg["data"].get("val_pairs_csv")
    if train_pairs_csv:
        checks.append(check_path("data.train_pairs_csv", train_pairs_csv, expect_dir=False))
    else:
        checks.append(check_path("data.train_dir", cfg["data"]["train_dir"], expect_dir=True))
    if val_pairs_csv:
        checks.append(check_path("data.val_pairs_csv", val_pairs_csv, expect_dir=False))
    else:
        checks.append(check_path("data.val_dir", cfg["data"]["val_dir"], expect_dir=True))
    checks.append(check_path("models.hair_parsing_weights", cfg["models"]["hair_parsing_weights"], expect_dir=False))

    insightface_root = cfg["models"].get("insightface_root", os.environ.get("INSIGHTFACE_MODEL_ROOT", "."))
    antelope = Path(os.path.expanduser(insightface_root)) / "models" / "antelopev2"
    checks.append(check_path("insightface antelopev2 dir", str(antelope), expect_dir=True))

    print(f"[INFO] torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    missing = len([x for x in checks if not x])
    if missing:
        raise SystemExit(f"Preflight failed: missing checks={missing}")
    print("Preflight OK")


if __name__ == "__main__":
    main()
