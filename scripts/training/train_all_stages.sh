#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
bash "${ROOT}/scripts/training/train_stage1.sh"
export PF_STAGE1_CHECKPOINT=${PF_STAGE1_CHECKPOINT:-${PF_STAGE1_RUN_DIR:-${ROOT}/runs/stage1}/logs/checkpoint_model_003000/model.pt}
bash "${ROOT}/scripts/training/train_stage2.sh"
