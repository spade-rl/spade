#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export FIXED_MODEL_SIZE=8b
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/spare_paper_fixed_gpt55_8b/$(date +%Y%m%d_%H%M%S)}"
export WANDB_GROUP="${WANDB_GROUP:-paper-fixed-gpt55-8b}"

exec "${SCRIPT_DIR}/_download_gpt55_pool.sh"
