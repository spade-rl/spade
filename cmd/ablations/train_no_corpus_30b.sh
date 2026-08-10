#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export CORPUS_FILE=""
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/spare_paper_no_corpus_30b/$(date +%Y%m%d_%H%M%S)}"
export WANDB_GROUP="${WANDB_GROUP:-paper-games-30b-no-corpus}"

exec "${SCRIPT_DIR}/../games/train_spade_30b.sh"
