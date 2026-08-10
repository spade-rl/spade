#!/bin/bash
# Convert a slime/Megatron torch_dist checkpoint to HuggingFace format.
#
# Run inside a Slime container with Megatron-LM and Transformers available.
#
# Usage:
#   bash scripts/convert_ckpt_to_hf.sh \
#       <iter_dir> <output_dir> [origin_hf_dir]
#
#   iter_dir       Iteration directory containing common.pt
#   output_dir     Where to write the HF-format weights
#   origin_hf_dir  (optional) HF model dir to copy tokenizer/config from.
#                  Defaults to ${WORKSPACE_DIR}/Qwen3-4B-Instruct-2507 if unset.
#
# Example:
#   bash scripts/convert_ckpt_to_hf.sh \
#       /path/to/checkpoint/iter_0000064 \
#       /path/to/converted_hf
#
# Reference: slime/docs/en/get_started/quick_start.md

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <iter_dir> <output_dir> [origin_hf_dir]" >&2
    exit 1
fi

ITER_DIR="$1"
OUTPUT_DIR="$2"
ORIGIN_HF_DIR="${3:-${WORKSPACE_DIR:-/workspace/spare-workspace}/Qwen3-4B-Instruct-2507}"

if [ ! -f "${ITER_DIR}/common.pt" ]; then
    echo "ERROR: ${ITER_DIR}/common.pt not found. Pass the specific iter_NNNNNNN directory." >&2
    exit 1
fi
if [ ! -d "${ORIGIN_HF_DIR}" ]; then
    echo "ERROR: origin HF dir ${ORIGIN_HF_DIR} not found. Needed for tokenizer/config." >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime"

mkdir -p "${OUTPUT_DIR}"

echo "[CONVERT] iter_dir   = ${ITER_DIR}"
echo "[CONVERT] output_dir = ${OUTPUT_DIR}"
echo "[CONVERT] origin_hf  = ${ORIGIN_HF_DIR}"

PYTHONPATH=/root/Megatron-LM:${PYTHONPATH:-} \
    python3 "${SLIME_DIR}/tools/convert_torch_dist_to_hf.py" \
    --input-dir "${ITER_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --origin-hf-dir "${ORIGIN_HF_DIR}" \
    --force

echo "[CONVERT] done -> ${OUTPUT_DIR}"
