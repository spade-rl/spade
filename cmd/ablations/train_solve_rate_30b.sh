#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export REWARD_VARIANT=solve_rate
export PLATEAU_LO="${PLATEAU_LO:-0.4}"
export PLATEAU_HI="${PLATEAU_HI:-0.6}"
export PLATEAU_RAMP="${PLATEAU_RAMP:-0.25}"
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/spare_paper_solve_rate_30b/$(date +%Y%m%d_%H%M%S)}"
export WANDB_GROUP="${WANDB_GROUP:-paper-games-30b-solve-rate}"

exec "${SCRIPT_DIR}/../games/train_spade_30b.sh"
