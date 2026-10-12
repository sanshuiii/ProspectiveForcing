#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
command -v hf >/dev/null || {
  echo "Install huggingface_hub[cli] so the 'hf' command is available." >&2
  exit 2
}
mkdir -p "${ROOT}/wan_models" "${ROOT}/checkpoints/base"
if test -e "${ROOT}/checkpoints/base/causal_cd.pt"; then
  echo "Refusing to overwrite checkpoints/base/causal_cd.pt" >&2
  exit 3
fi

hf download Wan-AI/Wan2.1-T2V-1.3B \
  --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
  --local-dir "${ROOT}/wan_models/Wan2.1-T2V-1.3B"
hf download Wan-AI/Wan2.1-T2V-14B \
  --revision a064a6c71f5be440641209c07bf2a5ce7a2ff5e4 \
  --local-dir "${ROOT}/wan_models/Wan2.1-T2V-14B"
hf download zhuhz22/Causal-Forcing chunkwise/causal_cd.pt \
  --revision a6a8f0e3bbdea1044fc6fef09c9cb9f648bf1bc3 \
  --local-dir "${ROOT}/checkpoints/base/.download"
mv -n "${ROOT}/checkpoints/base/.download/chunkwise/causal_cd.pt" \
  "${ROOT}/checkpoints/base/causal_cd.pt"

printf '%s  %s\n' \
  3fff3d62a8245ff4693fbaa91ce6f2bcf6848c94965b84d16d91f3e7664f763b \
  "${ROOT}/checkpoints/base/causal_cd.pt" | sha256sum --check --status
