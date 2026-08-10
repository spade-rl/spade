#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export NO_PROPOSER_TRAIN=1
export NO_ENV_MEMORY=1
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/spare_paper_no_ed_no_memory_30b/$(date +%Y%m%d_%H%M%S)}"
export WANDB_GROUP="${WANDB_GROUP:-paper-games-30b-no-ed-no-memory}"

exec "${SCRIPT_DIR}/../games/train_spade_30b.sh"
