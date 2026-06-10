#!/bin/bash
# Fast end-to-end smoke test:
# 1) build small pairs csv from val dir
# 2) run inference on a small subset
# 3) run metrics with robust/offline-friendly options
#
# Example:
#   sbatch -p rocky --export=ALL,CFG_FILE=config.cluster.smoke.yaml,CKPT_PATH=/home/.../ckpt_step119.pt scripts/smoke_e2e.sbatch

#SBATCH --job-name=bg-smoke-e2e
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs eval

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export ORT_DISABLE_CPU_AFFINITY=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

export INSIGHTFACE_MODEL_ROOT="${INSIGHTFACE_MODEL_ROOT:-${HOME}/.cache/insightface}"
export CUDNN_LIB="${CUDNN_LIB:-${HOME}/.conda/envs/bg-research/lib/python3.10/site-packages/nvidia/cudnn/lib}"
export CUBLAS_LIB="${CUBLAS_LIB:-${HOME}/.conda/envs/bg-research/lib/python3.10/site-packages/nvidia/cublas/lib}"
export LD_LIBRARY_PATH="${CUDNN_LIB}:${CUBLAS_LIB}:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/arobryadchikova/.conda/envs/bg-research/bin/python}"
CFG_FILE="${CFG_FILE:-config.cluster.smoke.yaml}"
CKPT_PATH="${CKPT_PATH:-}"
PAIRS_N="${PAIRS_N:-8}"

if [ ! -f "${CFG_FILE}" ]; then
  echo "[FATAL] cfg not found: ${CFG_FILE}"
  exit 2
fi
if [ -z "${CKPT_PATH}" ]; then
  echo "[FATAL] set CKPT_PATH to a checkpoint path"
  exit 2
fi
if [ ! -f "${CKPT_PATH}" ]; then
  echo "[FATAL] checkpoint not found: ${CKPT_PATH}"
  exit 2
fi

SMOKE_TAG="smoke_${SLURM_JOB_ID}"
PAIRS_CSV="eval/pairs_${SMOKE_TAG}.csv"
OUT_DIR="${HOME}/projects/BG_ResearchWork/runs/${SMOKE_TAG}"
METRICS_CSV="${OUT_DIR}/metrics_results.csv"

echo "[env] python=${PYTHON_BIN}"
echo "[env] cfg=${CFG_FILE}"
echo "[env] ckpt=${CKPT_PATH}"
echo "[env] out_dir=${OUT_DIR}"

"${PYTHON_BIN}" - <<PY
import csv
from pathlib import Path
import yaml

cfg = yaml.safe_load(open("${CFG_FILE}", "r", encoding="utf-8"))
val = Path(cfg["data"]["val_dir"])
exts = {".jpg", ".jpeg", ".png", ".webp"}
imgs = [p for p in sorted(val.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
n = min(int("${PAIRS_N}"), len(imgs))
if n < 2:
    raise SystemExit(f"Need at least 2 images in val_dir, got {len(imgs)}")
sel = imgs[:n]
out = Path("${PAIRS_CSV}")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["pair_id", "ref_id", "ref_hair"])
    for i, p in enumerate(sel):
        w.writerow([i, str(p), str(sel[(i + 1) % n])])
print("saved", out, "rows=", n)
PY

HAIR_W=$("${PYTHON_BIN}" - <<PY
import yaml
cfg=yaml.safe_load(open("${CFG_FILE}", "r", encoding="utf-8"))
print(cfg["models"]["hair_parsing_weights"])
PY
)

HAIR_CLASSES=$("${PYTHON_BIN}" - <<PY
import yaml
cfg=yaml.safe_load(open("${CFG_FILE}", "r", encoding="utf-8"))
cond = cfg.get("cond", {})
vals = cond.get("hair_classes", None)
if vals is None:
    vals = [int(cond.get("hair_class", 17))]
elif isinstance(vals, str):
    vals = [int(x.strip()) for x in vals.split(",") if x.strip()]
elif isinstance(vals, int):
    vals = [int(vals)]
else:
    vals = [int(x) for x in vals]
vals = list(dict.fromkeys(vals))
if not vals:
    vals = [17]
print(",".join(str(v) for v in vals))
PY
)
HAIR_CLASS_PRIMARY="${HAIR_CLASSES%%,*}"
if [ -z "${HAIR_CLASS_PRIMARY}" ]; then
  HAIR_CLASS_PRIMARY="17"
fi
echo "[env] hair_classes=${HAIR_CLASSES}"
SCALE_ID="${SCALE_ID:-0.0}"
SCALE_HAIR="${SCALE_HAIR:-0.65}"
USE_ID_TOKENS="${USE_ID_TOKENS:-0}"
ID_TOKEN_MODE="${ID_TOKEN_MODE:-auto}"
ID_TOKEN_MIX="${ID_TOKEN_MIX:-0.5}"
HAIR_MASK_DILATE_KERNEL="${HAIR_MASK_DILATE_KERNEL:-1}"
HAIR_MASK_DILATE_ITERS="${HAIR_MASK_DILATE_ITERS:-1}"
ARCFACE_NAME="${ARCFACE_NAME:-antelopev2}"
ARCFACE_DET="${ARCFACE_DET:-640}"
ARCFACE_MIN_DET="${ARCFACE_MIN_DET:-256}"
ARCFACE_DET_STEP="${ARCFACE_DET_STEP:-64}"
ARCFACE_RETRY_UPSCALE="${ARCFACE_RETRY_UPSCALE:-1}"
ONLY_BOTH_FACE="${ONLY_BOTH_FACE:-1}"
HAIR_FOCUS_CROP="${HAIR_FOCUS_CROP:-0}"
HAIR_FOCUS_CROP_MARGIN="${HAIR_FOCUS_CROP_MARGIN:-0.20}"
HAIR_FOCUS_CROP_SQUARE="${HAIR_FOCUS_CROP_SQUARE:-1}"

"${PYTHON_BIN}" -u inference.py \
  --pairs_csv "${PAIRS_CSV}" \
  --out_dir "${OUT_DIR}" \
  --sd_model_id runwayml/stable-diffusion-v1-5 \
  --arc2face_repo_id FoivosPar/Arc2Face \
  --clip_vision_id openai/clip-vit-large-patch14 \
  --hair_weights "${HAIR_W}" \
  --ckpt "${CKPT_PATH}" \
  --insightface_root "${INSIGHTFACE_MODEL_ROOT}" \
  --steps 20 \
  --guidance 7.0 \
  --seed 123 \
  --scale_id "${SCALE_ID}" \
  --scale_hair "${SCALE_HAIR}" \
  --use_id_tokens "${USE_ID_TOKENS}" \
  --id_token_mode "${ID_TOKEN_MODE}" \
  --id_token_mix "${ID_TOKEN_MIX}" \
  --hair_class "${HAIR_CLASS_PRIMARY}" \
  --hair_classes "${HAIR_CLASSES}" \
  --hair_mask_dilate_kernel "${HAIR_MASK_DILATE_KERNEL}" \
  --hair_mask_dilate_iters "${HAIR_MASK_DILATE_ITERS}" \
  --hair_focus_crop "${HAIR_FOCUS_CROP}" \
  --hair_focus_crop_margin "${HAIR_FOCUS_CROP_MARGIN}" \
  --hair_focus_crop_square "${HAIR_FOCUS_CROP_SQUARE}"

"${PYTHON_BIN}" -u metrics.py \
  --pairs_csv "${PAIRS_CSV}" \
  --gen_dir "${OUT_DIR}" \
  --gen_pattern "{pair_id_int:06d}/gen.png" \
  --hair_weights "${HAIR_W}" \
  --insightface_root "${INSIGHTFACE_MODEL_ROOT}" \
  --arcface_name "${ARCFACE_NAME}" \
  --arcface_det "${ARCFACE_DET}" \
  --arcface_min_det "${ARCFACE_MIN_DET}" \
  --arcface_det_step "${ARCFACE_DET_STEP}" \
  --arcface_retry_upscale "${ARCFACE_RETRY_UPSCALE}" \
  --clip_id openai/clip-vit-large-patch14 \
  --dino_id facebook/dinov2-base \
  --local_files_only 1 \
  --allow_missing_dino 1 \
  --hair_class "${HAIR_CLASS_PRIMARY}" \
  --hair_classes "${HAIR_CLASSES}" \
  --hair_mask_dilate_kernel "${HAIR_MASK_DILATE_KERNEL}" \
  --hair_mask_dilate_iters "${HAIR_MASK_DILATE_ITERS}" \
  --device cuda \
  --seeds 123 \
  --only_both_face "${ONLY_BOTH_FACE}" \
  --out_csv "${METRICS_CSV}"

echo "[done] smoke metrics: ${METRICS_CSV}"
ls -lh "${METRICS_CSV}" "${OUT_DIR}"/metrics_results_summary.csv "${OUT_DIR}"/metrics_results_groups.csv
