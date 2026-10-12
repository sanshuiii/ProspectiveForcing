#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_DIR=${PF_STAGE2_RUN_DIR:-${ROOT}/runs/stage2}
CONFIG=${PF_STAGE2_CONFIG:-${ROOT}/configs/training/stage2_m1.yaml}
NNODES=${PF_NNODES:-1}
NPROC_PER_NODE=${PF_NPROC_PER_NODE:-8}
export PF_STAGE1_CHECKPOINT=${PF_STAGE1_CHECKPOINT:-${ROOT}/checkpoints/stage1/prospective_forcing_stage1.pt}
export PF_TRAIN_PROMPTS=${PF_TRAIN_PROMPTS:-${ROOT}/data/train_prompts.txt}
export PF_EMBEDDINGS_DIR=${PF_EMBEDDINGS_DIR:-${ROOT}/data/embeddings}

test -s "${PF_STAGE1_CHECKPOINT}"
test -f "${PF_EMBEDDINGS_DIR}/manifest.json"
if test -e "${RUN_DIR}" && test -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)"; then
  echo "Refusing non-empty output directory: ${RUN_DIR}" >&2
  exit 3
fi
mkdir -p "${RUN_DIR}/logs"
cd "${ROOT}"
export PYTHONPATH=${ROOT}
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}

if test "$((NNODES * NPROC_PER_NODE))" -ne 8; then
  echo "The reference contract requires exactly 8 total ranks." >&2
  exit 2
fi
if test "${NNODES}" -eq 1; then
  LAUNCH_ARGS=(--standalone --nproc_per_node="${NPROC_PER_NODE}")
else
  : "${PF_NODE_RANK:?Set PF_NODE_RANK for multi-node training}"
  : "${PF_MASTER_ADDR:?Set PF_MASTER_ADDR for multi-node training}"
  : "${PF_MASTER_PORT:?Set PF_MASTER_PORT for multi-node training}"
  LAUNCH_ARGS=(
    --nnodes="${NNODES}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --node_rank="${PF_NODE_RANK}"
    --master_addr="${PF_MASTER_ADDR}"
    --master_port="${PF_MASTER_PORT}"
  )
fi

exec "${PYTHON_BIN}" -m torch.distributed.run "${LAUNCH_ARGS[@]}" \
  "${ROOT}/train.py" --config_path "${CONFIG}" \
  --logdir "${RUN_DIR}/logs" --disable-wandb
