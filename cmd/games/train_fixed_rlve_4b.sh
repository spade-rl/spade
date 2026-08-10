#!/usr/bin/env bash

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
sleep 3
pkill -9 ray

set -ex

export PYTHONUNBUFFERED=1
export WEAVE_PRINT_CALL_LINK=false
pip install math_verify weave

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime"

RLVE_ENV_SET="${RLVE_ENV_SET:-official16}"
if [ "${RLVE_ENV_SET}" = "official16" ]; then
   RLVE_SOURCES=(
      rlve:Division rlve:EuclidGame rlve:GCDOne_Counting rlve:HamiltonianPath
      rlve:LampChanging rlve:LargestConvexPolygon rlve:Multiplication
      rlve:PCPPermutation rlve:Path_NoGoingBack_Counting rlve:SAT
      rlve:ShortestPath rlve:Sorting rlve:SpiralMatrix
      rlve:SubsequenceReversalLNDS rlve:UndamagedSubmatrixCounting
      rlve:WYRLevelingGround
   )
elif [ "${RLVE_ENV_SET}" = "all" ]; then
   RLVE_SOURCES=( "rlve:*" )
else
   echo "ERROR: RLVE_ENV_SET must be 'official16' or 'all', got '${RLVE_ENV_SET}'" >&2
   exit 1
fi
echo "[RLVE] Env set: ${RLVE_ENV_SET} (${#RLVE_SOURCES[@]} source specs)"

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/fixed_rlve_4b_${RLVE_ENV_SET}/$(date +%Y%m%d_%H%M%S)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/spare-workspace}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/spare-workspace}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${WORKSPACE_DIR}/hf_cache/datasets}"
if [ -z "${WANDB_ENTITY:-}" ]; then
   echo "ERROR: Set WANDB_ENTITY to your Weights & Biases team/entity." >&2
   exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [ ! -d "${PROJECT_ROOT}/spare/external/rlve/Gym" ]; then
   echo "ERROR: vendored RLVE not found at spare/external/rlve/Gym." >&2
   exit 1
fi

DIFFICULTY_VARIANT="${DIFFICULTY_VARIANT:-sliding_window}"

source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

export WORKSPACE_DIR
EVAL_CONFIG_FILE="${OUTPUT_DIR}/eval_aime.yaml"
envsubst < "${PROJECT_ROOT}/eval_configs/eval_aime_avg32.yaml" > "${EVAL_CONFIG_FILE}"

GEM_EVAL_CONFIG_FILE="${PROJECT_ROOT}/eval_configs/gem_eval_inloop.yaml"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/Qwen3-4B-Instruct-2507"
   --ref-load "${MODEL_ROOT}/Qwen3-4B-Instruct-2507_torch_dist"
   ${LOAD_DIR:+--load "${LOAD_DIR}"}
   --save "${OUTPUT_DIR}/Qwen3-4B-Instruct-2507_fixed_rlve_${RLVE_ENV_SET}/"
   --save-interval 16
)

ROLLOUT_ARGS=(
   --data-source-path spare.slime.data_source.SpareDataSource
   --rollout-function-path spare.slime.fixed_env_rollout.spare_fixed_env_rollout
   --num-rollout "${NUM_ROLLOUT:-400}"
   --rollout-batch-size 16
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --global-batch-size 256
   --use-dynamic-global-batch-size
   --balance-data
   --apply-chat-template-kwargs '{"enable_thinking":false}'
)

SPARE_ARGS=(
   --spare-mode fixed_env
   --spare-fixed-env-source "${RLVE_SOURCES[@]}"
   --spare-game-regeneration-interval 1
   --spare-fixed-env-same-problem
   --spare-trajectories-per-game 16
   --spare-difficulty-variant "${DIFFICULTY_VARIANT}"
   --spare-difficulty-tau-acc 0.9
   --spare-difficulty-tau-num 128
   --spare-difficulty-d-delta 4
   --spare-lp-gamma-fast 0.35
   --spare-lp-gamma-slow 0.15
   --spare-actor-template qwen3_game
   --spare-actor-temperature 1.0
   --spare-actor-max-tokens "${ACTOR_MAX_TOKENS:-8192}"
   --spare-max-context-length "${MAX_CONTEXT_LENGTH:-32768}"
   --spare-max-turns "${MAX_TURNS:-25}"
   --spare-gamma 0.99
   --spare-reward-normalization grpo
   --spare-game-baseline-decay 0.5
   --spare-gem-eval-config "${GEM_EVAL_CONFIG_FILE}"
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-grpo-std-normalization
   --disable-rewards-normalization
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

WANDB_ARGS=(
   --use-wandb
   --wandb-team "${WANDB_ENTITY}"
   --wandb-project "${WANDB_PROJECT:-spade}"
   --wandb-group "${WANDB_GROUP:-fixed-rlve-4b-${RLVE_ENV_SET}}"
   --wandb-key "${WANDB_API_KEY:?Set WANDB_API_KEY.}"
)

EVAL_ARGS=(
   --eval-interval ${EVAL_INTERVAL:-100000}
   --eval-config ${EVAL_CONFIG_FILE}
   --skip-eval-before-train
   --apply-chat-template
   --rm-type math
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-tool-call-parser qwen
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"FLASHINFER_DISABLE_VERSION_CHECK\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"OPENROUTER_API_KEY\": \"${OPENROUTER_API_KEY}\",
    \"TAU2_DATA_DIR\": \"${TAU2_DATA_DIR}\",
    \"LITELLM_LOCAL_MODEL_COST_MAP\": \"True\",
    \"LITELLM_LOG\": \"WARNING\",
    \"LCB_OFFICIAL_DIR\": \"${LCB_OFFICIAL_DIR:-${WORKSPACE_DIR}/LiveCodeBench-official}\",
    \"LCB_V6_JSONL\": \"${LCB_V6_JSONL:-${WORKSPACE_DIR}/livecodebench/test6.jsonl}\",
    \"HF_DATASETS_CACHE\": \"${HF_DATASETS_CACHE}\"
  }
}"

RAY_JOB_ID="fixed_rlve_4b_${RLVE_ENV_SET}_$$"

ray job submit --address="http://127.0.0.1:8265" \
   --submission-id="${RAY_JOB_ID}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --no-wait \
   -- python3 -m train_spare_slime \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${SPARE_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${WANDB_ARGS[@]}"

ray job logs --address="http://127.0.0.1:8265" --follow "${RAY_JOB_ID}"
