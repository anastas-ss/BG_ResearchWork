#!/bin/bash
#SBATCH --job-name=bg-infer
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Optional: activate your conda env if needed.
# source "${HOME}/miniconda3/etc/profile.d/conda.sh"
# conda activate bg-research

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export ORT_DISABLE_CPU_AFFINITY=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export INSIGHTFACE_MODEL_ROOT="${INSIGHTFACE_MODEL_ROOT:-${HOME}/.cache/insightface}"
export CONDA_PREFIX="${CONDA_PREFIX:-${HOME}/.conda/envs/bg-research}"
export CUDNN_LIB="${CUDNN_LIB:-${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn/lib}"
export CUBLAS_LIB="${CUBLAS_LIB:-${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/lib}"
export CUDA_RUNTIME_LIB="${CUDA_RUNTIME_LIB:-${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDNN_LIB}:${CUBLAS_LIB}:${CUDA_RUNTIME_LIB}:${LD_LIBRARY_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"

# Update these variables before submission.
PAIRS_CSV="${PAIRS_CSV:-/path/to/pairs.csv}"
OUT_DIR="${OUT_DIR:-/path/to/infer_out}"
CKPT_PATH="${CKPT_PATH:-}"
HAIR_WEIGHTS="${HAIR_WEIGHTS:-}"
HAIR_CLASS="${HAIR_CLASS:-17}"
HAIR_CLASSES="${HAIR_CLASSES:-17}"
SCALE_HAIR="${SCALE_HAIR:-0.65}"
STEPS="${STEPS:-25}"
GUIDANCE="${GUIDANCE:-3.0}"
CFG_MODE="${CFG_MODE:-joint}"
GUIDANCE_ID="${GUIDANCE_ID:-}"
GUIDANCE_HAIR="${GUIDANCE_HAIR:-}"
SAVE_REFS="${SAVE_REFS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MAX_ITEMS="${MAX_ITEMS:-0}"
ATTENTION_DEBUG="${ATTENTION_DEBUG:-0}"
ATTENTION_DEBUG_STEPS="${ATTENTION_DEBUG_STEPS:-0:12:24}"
ATTENTION_DEBUG_MAX_ITEMS="${ATTENTION_DEBUG_MAX_ITEMS:-4}"
HAIR_TOKEN_MODE="${HAIR_TOKEN_MODE:-}"
HAIR_PATCH_MASK_THRESHOLD="${HAIR_PATCH_MASK_THRESHOLD:-}"
HAIR_MAX_PATCH_TOKENS="${HAIR_MAX_PATCH_TOKENS:-}"
HAIR_SPATIAL_GATE="${HAIR_SPATIAL_GATE:-}"
HAIR_MASK_DILATE_KERNEL="${HAIR_MASK_DILATE_KERNEL:-1}"
HAIR_MASK_DILATE_ITERS="${HAIR_MASK_DILATE_ITERS:-1}"
HAIR_FOCUS_CROP="${HAIR_FOCUS_CROP:-0}"
HAIR_FOCUS_CROP_MARGIN="${HAIR_FOCUS_CROP_MARGIN:-0.20}"
HAIR_FOCUS_CROP_SQUARE="${HAIR_FOCUS_CROP_SQUARE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

nvidia-smi || true
echo "[env] python_bin=${PYTHON_BIN}"

cmd=(
  "${PYTHON_BIN}" -u inference.py
  --pairs_csv "${PAIRS_CSV}" \
  --out_dir "${OUT_DIR}" \
  --sd_model_id runwayml/stable-diffusion-v1-5 \
  --arc2face_repo_id FoivosPar/Arc2Face \
  --clip_vision_id openai/clip-vit-large-patch14 \
  --insightface_root "${INSIGHTFACE_MODEL_ROOT}" \
  --steps "${STEPS}" \
  --guidance "${GUIDANCE}" \
  --cfg_mode "${CFG_MODE}" \
  --seed 123 \
  --save_refs "${SAVE_REFS}" \
  --skip_existing "${SKIP_EXISTING}" \
  --max_items "${MAX_ITEMS}" \
  --scale_hair "${SCALE_HAIR}" \
  --hair_class "${HAIR_CLASS}" \
  --hair_classes "${HAIR_CLASSES}" \
  --hair_mask_dilate_kernel "${HAIR_MASK_DILATE_KERNEL}" \
  --hair_mask_dilate_iters "${HAIR_MASK_DILATE_ITERS}" \
  --hair_focus_crop "${HAIR_FOCUS_CROP}" \
  --hair_focus_crop_margin "${HAIR_FOCUS_CROP_MARGIN}" \
  --hair_focus_crop_square "${HAIR_FOCUS_CROP_SQUARE}"
)

if [ -n "${GUIDANCE_ID}" ]; then
  cmd+=(--guidance_id "${GUIDANCE_ID}")
fi

if [ -n "${GUIDANCE_HAIR}" ]; then
  cmd+=(--guidance_hair "${GUIDANCE_HAIR}")
fi

if [ -n "${HAIR_TOKEN_MODE}" ]; then
  cmd+=(--hair_token_mode "${HAIR_TOKEN_MODE}")
fi

if [ -n "${HAIR_PATCH_MASK_THRESHOLD}" ]; then
  cmd+=(--hair_patch_mask_threshold "${HAIR_PATCH_MASK_THRESHOLD}")
fi

if [ -n "${HAIR_MAX_PATCH_TOKENS}" ]; then
  cmd+=(--hair_max_patch_tokens "${HAIR_MAX_PATCH_TOKENS}")
fi

if [ -n "${HAIR_SPATIAL_GATE}" ]; then
  cmd+=(--hair_spatial_gate "${HAIR_SPATIAL_GATE}")
fi

if [ -n "${HAIR_WEIGHTS}" ]; then
  cmd+=(--hair_weights "${HAIR_WEIGHTS}")
fi

if [ -n "${CKPT_PATH}" ]; then
  cmd+=(--ckpt "${CKPT_PATH}")
fi

if [ "${ATTENTION_DEBUG}" = "1" ]; then
  cmd+=(
    --attention_debug
    --attention_debug_steps "${ATTENTION_DEBUG_STEPS}"
    --attention_debug_max_items "${ATTENTION_DEBUG_MAX_ITEMS}"
  )
fi

if [ -n "${EXTRA_ARGS}" ]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
  cmd+=("${extra_args[@]}")
fi

printf '[cmd]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
