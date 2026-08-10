#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export SPARE_SKILLS="Mathematical_Reasoning Pattern_Recognition"
export SKILLS_PER_REGEN=2
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/spare_paper_two_skill_30b/$(date +%Y%m%d_%H%M%S)}"
export WANDB_GROUP="${WANDB_GROUP:-paper-games-30b-two-skill}"

exec "${SCRIPT_DIR}/../games/train_spade_30b.sh"
