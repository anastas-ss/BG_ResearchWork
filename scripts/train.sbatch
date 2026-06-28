#!/bin/bash
#SBATCH --job-name=bg-train
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

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
CFG_FILE="${CFG_FILE:-config.cluster.fullclean.long.yaml}"

nvidia-smi || true
echo "[env] python_bin=${PYTHON_BIN}"
"${PYTHON_BIN}" -V

if [ ! -f "${CFG_FILE}" ]; then
  echo "[FATAL] config file not found: ${CFG_FILE}"
  exit 1
fi

echo "[env] cfg_file=${CFG_FILE}"
"${PYTHON_BIN}" -u train.py --cfg "${CFG_FILE}"
