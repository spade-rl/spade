#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

HF_DATASET="${HF_DATASET:-spare-rl/spare-gpt55-static-corpus}"
HF_DATASET_REVISION="${HF_DATASET_REVISION:-e179a371bc7764dacf0bcee1f808100beb463137}"
DATASET_ROOT="${DATASET_ROOT:-/scratch/spare-gpt55-static-corpus}"

if [[ "${SKIP_DATASET_DOWNLOAD:-0}" == "1" ]]; then
  if [[ "${PREPARE_ONLY:-0}" != "1" ]]; then
    echo "ERROR: SKIP_DATASET_DOWNLOAD=1 is allowed only with PREPARE_ONLY=1." >&2
    exit 1
  fi
  export DATASET_ROOT
  export STATIC_POOL_DIR="${STATIC_POOL_DIR:-${DATASET_ROOT}/games}"
  exec "${SCRIPT_DIR}/_train_fixed_gpt55.sh"
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: install huggingface_hub so the 'hf' CLI is available." >&2
  exit 1
fi

mkdir -p "${DATASET_ROOT}"
hf download "${HF_DATASET}" \
  --repo-type dataset \
  --revision "${HF_DATASET_REVISION}" \
  --local-dir "${DATASET_ROOT}" \
  --include 'games/*' \
  --include 'manifest.jsonl' \
  --include 'validation_manifest.jsonl' \
  --include 'validation_summary.json' \
  --include 'snapshot_summary.json' \
  --include 'checksums.sha256' \
  --include 'README.md'

if [[ -d "${DATASET_ROOT}/games" ]]; then
  game_count="$(find "${DATASET_ROOT}/games" -maxdepth 1 -type f -name 'game_*.py' | wc -l | tr -d ' ')"
else
  game_count=0
fi
if [[ "${game_count}" -ne 7872 ]]; then
  echo "ERROR: downloaded ${game_count} games; expected 7872." >&2
  exit 1
fi

(
  cd "${DATASET_ROOT}"
  sha256sum -c checksums.sha256
)

export DATASET_ROOT
export STATIC_POOL_DIR="${STATIC_POOL_DIR:-${DATASET_ROOT}/games}"
exec "${SCRIPT_DIR}/_train_fixed_gpt55.sh"
