#!/usr/bin/env bash
set -euo pipefail

if test "$#" -lt 3 || test "$#" -gt 4; then
  echo "Usage: $0 <b0|a1|m1> <prompts.txt> <output-dir> [extended-prompts.txt]" >&2
  exit 2
fi
MODE=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
PROMPTS=$2
OUTPUT=$3
EXTENDED=${4:-}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${PF_GPU_IDS:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
WORLD_SIZE=${#GPUS[@]}

case "${MODE}" in
  b0)
    CONFIG=${ROOT}/configs/inference/b0.yaml
    CHECKPOINT=${PF_STAGE1_CHECKPOINT:-${ROOT}/checkpoints/stage1/prospective_forcing_stage1.pt}
    AUX=(--auxiliary_window_heads 0)
    ;;
  a1)
    CONFIG=${ROOT}/configs/inference/a1.yaml
    CHECKPOINT=${PF_STAGE1_CHECKPOINT:-${ROOT}/checkpoints/stage1/prospective_forcing_stage1.pt}
    AUX=(--auxiliary_window_heads 1 --draft_commit_policy last_chunk_only_rolling_stitch_v1 --expected_draft_output_num_chunks 1)
    ;;
  m1)
    CONFIG=${ROOT}/configs/inference/m1.yaml
    CHECKPOINT=${PF_STAGE2_CHECKPOINT:-${ROOT}/checkpoints/stage2/prospective_forcing_stage2_m1.pt}
    AUX=(--auxiliary_window_heads 1 --draft_commit_policy last_chunk_only_rolling_stitch_v1 --expected_draft_output_num_chunks 1)
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac

test -s "${CHECKPOINT}"
test -s "${PROMPTS}"
if test -n "${EXTENDED}"; then test -s "${EXTENDED}"; fi
if test -e "${OUTPUT}" && test -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)"; then
  echo "Refusing non-empty output directory: ${OUTPUT}" >&2
  exit 3
fi
mkdir -p "${OUTPUT}/videos" "${OUTPUT}/instrumentation" "${OUTPUT}/logs"
OUTPUT=$(cd "${OUTPUT}" && pwd)
PROMPTS=$(cd "$(dirname "${PROMPTS}")" && pwd)/$(basename "${PROMPTS}")
if test -n "${EXTENDED}"; then
  EXTENDED=$(cd "$(dirname "${EXTENDED}")" && pwd)/$(basename "${EXTENDED}")
  EXTENDED_ARGS=(--extended_prompt_path "${EXTENDED}")
else
  EXTENDED_ARGS=()
fi

cd "${ROOT}"
export PYTHONPATH=${ROOT}
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
pids=()
for rank in "${!GPUS[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES=${GPUS[$rank]}
    export INFERENCE_RANK=${rank} INFERENCE_WORLD_SIZE=${WORLD_SIZE} INSTRUMENTATION_RANK=${rank}
    "${PYTHON_BIN}" "${ROOT}/inference.py" \
      --config_path "${CONFIG}" --checkpoint_path "${CHECKPOINT}" \
      --data_path "${PROMPTS}" "${EXTENDED_ARGS[@]}" \
      --output_folder "${OUTPUT}/videos" \
      --num_output_frames "${PF_NUM_LATENT_FRAMES:-126}" \
      --num_save_frames "${PF_NUM_RGB_FRAMES:-480}" \
      --num_samples 1 --seed "${PF_SEED:-20260720}" --use_ema \
      --save_with_index --deterministic_by_prompt --independent_sharding \
      --require_prospective --instrumentation_folder "${OUTPUT}/instrumentation" \
      "${AUX[@]}" > "${OUTPUT}/logs/rank_${rank}.out" 2> "${OUTPUT}/logs/rank_${rank}.err"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if test "${failed}" -ne 0; then
  echo "Inference failed. Inspect ${OUTPUT}/logs; the directory is record-only." >&2
  exit 4
fi
