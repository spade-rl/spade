#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export TOOL_MODEL_SIZE=30b
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/spare_paper_tool_use_30b_blend/$(date +%Y%m%d_%H%M%S)}"
export WANDB_GROUP="${WANDB_GROUP:-paper-tool-use-30b-blend}"

exec "${SCRIPT_DIR}/_train_spade_blend.sh"
