#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export REWARD_VARIANT=learning_potential
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/spare_paper_learning_potential_30b/$(date +%Y%m%d_%H%M%S)}"
export WANDB_GROUP="${WANDB_GROUP:-paper-games-30b-learning-potential}"

exec "${SCRIPT_DIR}/../games/train_spade_30b.sh"
