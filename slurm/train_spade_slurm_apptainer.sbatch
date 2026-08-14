#!/bin/bash
#SBATCH --job-name=spade-train
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1900G
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Run a SPADE training recipe on a SLURM cluster inside an Apptainer/Singularity
# container. This is the reference cluster launcher: it wraps the ordinary
# `bash cmd/<family>/<script>.sh` entry point, so anything runnable locally is
# runnable here without modification.
#
# ---------------------------------------------------------------------------
# QUICK START
#   sbatch --export=ALL,\
# SPADE_DIR=/path/to/spade,\
# CONTAINER=/path/to/slime.sif,\
# MODEL_ROOT=/path/to/models,\
# TRAIN_SCRIPT=cmd/games/train_spade_4b.sh \
#     slurm/train_spade_slurm_apptainer.sbatch
#
# REQUIREMENTS
#   * Apptainer (or Singularity) with --nv, and a container carrying slime +
#     SGLang + Megatron. The pinned image is `slimerl/slime:v0.3.0`; build with:
#         apptainer build slime.sif docker://slimerl/slime:v0.3.0
#   * The `slime` submodule must be initialised (the training scripts source
#     ${SPADE_DIR}/slime/scripts/models/*.sh):
#         git submodule update --init --recursive
#   * Base model weights under ${MODEL_ROOT} (e.g. Qwen3-4B-Instruct-2507),
#     matching the --hf-checkpoint path in the chosen recipe.
#   * WANDB_API_KEY exported (or set --use-wandb off in the recipe).
#   * WANDB_ENTITY (your W&B team) and CORPUS_FILE (the grounding corpus JSONL) are
#     REQUIRED by the games recipes; they exit with an explicit message if unset.
#   * Ray's temp dir MUST be on node-local disk: Ray places unix domain sockets there
#     and its dashboard (which `ray job submit` needs) fails on GPFS/NFS with
#     'Module MetricsHead failed to start. Received EOF from pipe.' Defaults to /tmp;
#     override with RAY_TMPDIR if your node uses a different local scratch.
#   * The recipes pip-install extras at runtime (math_verify, weave). Debian/Ubuntu
#     images mark system Python 'externally managed' (PEP 668), so that pip call
#     fails unless PIP_BREAK_SYSTEM_PACKAGES=1 is set (this script sets it) or the
#     extras are baked into the image.
# ---------------------------------------------------------------------------
set -uo pipefail

SPADE_DIR="${SPADE_DIR:?set SPADE_DIR to the spade checkout}"
CONTAINER="${CONTAINER:?set CONTAINER to the .sif image}"
MODEL_ROOT="${MODEL_ROOT:?set MODEL_ROOT to the directory holding base model weights}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-cmd/games/train_spade_4b.sh}"
# Writable scratch for caches, games, checkpoints (NOT the repo, which may be read-only
# and is usually on a small quota).
WORK_DIR="${WORK_DIR:-${SPADE_DIR}/.spade_runs}"

cd "${SPADE_DIR}"
mkdir -p "${WORK_DIR}"/{cache,games,ckpt,tmp}
# The recipes default OUTPUT_DIR to /scratch/..., which is site-specific and read-only
# inside the container. Point it at the writable WORK_DIR unless the caller overrides.
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/ckpt/$(basename "${TRAIN_SCRIPT}" .sh)_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================================="
echo " SPADE @ $(git -C "${SPADE_DIR}" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo " node=$(hostname)  gpus=${SLURM_GPUS_ON_NODE:-?}  script=${TRAIN_SCRIPT}"
echo " container=${CONTAINER}"
echo "=========================================================="

# Fail early with a clear message rather than a confusing source error mid-run.
if [ ! -f "${SPADE_DIR}/slime/scripts/models/qwen3-4B-Instruct-2507.sh" ]; then
    echo "ERROR: slime submodule not initialised — run: git submodule update --init --recursive" >&2
    exit 2
fi

# The recipes reference the model via ${MODEL_ROOT}; export so they resolve.
export MODEL_ROOT

apptainer exec --nv \
    --bind "${SPADE_DIR}":/workspace \
    --bind "${MODEL_ROOT}":"${MODEL_ROOT}":ro \
    --bind "${WORK_DIR}":"${WORK_DIR}" \
    ${CORPUS_FILE:+--bind "$(dirname "${CORPUS_FILE}")":"$(dirname "${CORPUS_FILE}")":ro} \
    --env HF_HOME="${WORK_DIR}/cache/hf" \
    --env HF_HUB_CACHE="${WORK_DIR}/cache/hf/hub" \
    --env XDG_CACHE_HOME="${WORK_DIR}/cache" \
    --env TRITON_CACHE_DIR="${WORK_DIR}/cache/triton" \
    --env TMPDIR="${RAY_TMPDIR:-/tmp}" \
    --env RAY_TMPDIR="${RAY_TMPDIR:-/tmp}" \
    --env PYTHONPATH=/workspace \
    --env PYTHONUNBUFFERED=1 \
    --env PIP_BREAK_SYSTEM_PACKAGES=1 \
    --env MODEL_ROOT="${MODEL_ROOT}" \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env WANDB_ENTITY="${WANDB_ENTITY:-}" \
    --env WANDB_PROJECT="${WANDB_PROJECT:-spade}" \
    --env CORPUS_FILE="${CORPUS_FILE:-}" \
    --env OUTPUT_DIR="${OUTPUT_DIR}" \
    --env GAMES_DIR="${GAMES_DIR:-}" \
    --env WORKSPACE_DIR="${WORKSPACE_DIR:-${WORK_DIR}}" \
    --env OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
    --env TAU2_DATA_DIR="${TAU2_DATA_DIR:-}" \
    --pwd /workspace \
    "${CONTAINER}" \
    bash -lc "pip install --no-deps -e /workspace >/dev/null 2>&1 || true; bash ${TRAIN_SCRIPT}"

rc=$?
echo "[spade] exit rc=${rc}"
exit ${rc}
