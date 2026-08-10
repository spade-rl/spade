#!/bin/bash
# Set up LiveCodeBench-v6 eval dependencies on a fresh cluster.
#
# Downloads the pinned official runner and the release_v6 test data.
#
# Usage:
#   bash scripts/setup_lcb.sh
#   WORKSPACE_DIR=/path/to/spare-workspace bash scripts/setup_lcb.sh
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:?Set WORKSPACE_DIR to a writable data directory.}"

LCB_REPO="https://github.com/LiveCodeBench/LiveCodeBench.git"
LCB_COMMIT="28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"   # pinned 2025-07-15 (#109)
LCB_DIR="${WORKSPACE_DIR}/LiveCodeBench-official"

JSONL_URL="https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl"
JSONL_DIR="${WORKSPACE_DIR}/livecodebench"
JSONL="${JSONL_DIR}/test6.jsonl"
JSONL_BYTES=134303240

mkdir -p "${WORKSPACE_DIR}" "${JSONL_DIR}"

# Pinned lcb_runner clone
if [ -d "${LCB_DIR}/lcb_runner" ]; then
    have="$(git -C "${LCB_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "[lcb] lcb_runner already present at ${LCB_DIR} (HEAD=${have:0:8})"
    if [ "${have}" != "${LCB_COMMIT}" ]; then
        echo "[lcb]   note: HEAD != pinned ${LCB_COMMIT:0:8}; 'git -C ${LCB_DIR} checkout ${LCB_COMMIT}' to pin."
    fi
else
    echo "[lcb] cloning ${LCB_REPO} and pinning to ${LCB_COMMIT:0:8} ..."
    git clone "${LCB_REPO}" "${LCB_DIR}"
    git -C "${LCB_DIR}" checkout "${LCB_COMMIT}"
fi

# release_v6 dataset
if [ -f "${JSONL}" ] && [ "$(stat -c%s "${JSONL}")" = "${JSONL_BYTES}" ]; then
    echo "[lcb] test6.jsonl present and correct size (${JSONL_BYTES} bytes)"
else
    echo "[lcb] downloading test6.jsonl (${JSONL_BYTES} bytes) from HF ..."
    wget --continue -O "${JSONL}" "${JSONL_URL}"
    got="$(stat -c%s "${JSONL}")"
    if [ "${got}" != "${JSONL_BYTES}" ]; then
        echo "[lcb] WARNING: size mismatch (got ${got}, expected ${JSONL_BYTES}) — re-download." >&2
        exit 1
    fi
    echo "[lcb] test6.jsonl OK (${got} bytes)"
fi

echo
echo "[lcb] setup complete. The eval runners already point at:"
echo "    LCB_OFFICIAL_DIR / LCB_ROOT = ${LCB_DIR}"
echo "    LCB_V6_JSONL                = ${JSONL}"
echo "  (inside the container these map to /workspace/spare-workspace/... via the bind)"
