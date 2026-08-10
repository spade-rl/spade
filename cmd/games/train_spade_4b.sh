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

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/spade_games_4b/$(date +%Y%m%d_%H%M%S)}"
GAMES_DIR="${GAMES_DIR:-${OUTPUT_DIR}/spare_games}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/spare-workspace}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/spare-workspace}"
if [ -z "${WANDB_ENTITY:-}" ]; then
   echo "ERROR: Set WANDB_ENTITY to your Weights & Biases team/entity." >&2
   exit 1
fi
if [ "${CORPUS_FILE+x}" != x ]; then
   echo "ERROR: Set CORPUS_FILE to the paper games corpus JSONL." >&2
   exit 1
fi
CORPUS_FILE="${CORPUS_FILE:-}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${GAMES_DIR}"

source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

if [ -n "${LOAD_DIR:-}" ]; then
   OLD_MEM="$(dirname "${LOAD_DIR}")/spare_games_cache/env_memory.json"
   if [ -f "${OLD_MEM}" ]; then
      mkdir -p "${OUTPUT_DIR}/spare_games_cache"
      cp "${OLD_MEM}" "${OUTPUT_DIR}/spare_games_cache/env_memory.json"
      echo "[MEMORY] Seeded env memory from ${OLD_MEM}"
   fi
fi

export WORKSPACE_DIR
EVAL_CONFIG_FILE="${OUTPUT_DIR}/eval_aime.yaml"
envsubst < "${PROJECT_ROOT}/eval_configs/eval_aime_avg32.yaml" > "${EVAL_CONFIG_FILE}"

GEM_EVAL_CONFIG_FILE="${PROJECT_ROOT}/eval_configs/gem_eval_inloop.yaml"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/Qwen3-4B-Instruct-2507"
   --ref-load "${MODEL_ROOT}/Qwen3-4B-Instruct-2507_torch_dist"
   ${LOAD_DIR:+--load "${LOAD_DIR}"}
   --save "${OUTPUT_DIR}/Qwen3-4B-Instruct-2507_games_blend/"
   --save-interval 16
)

ROLLOUT_ARGS=(
   --data-source-path spare.slime.data_source.SpareDataSource
   --rollout-function-path spare.slime.spare_rollout.spare_generate_rollout
   --num-rollout 400
   --rollout-batch-size 24
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --global-batch-size 192
   --use-dynamic-global-batch-size
   --balance-data
)

SPARE_ARGS=(
   --spare-gamma1 0.98
   --spare-gamma2 0.85
   --spare-env-temperature 0.6
   --spare-actor-temperature 0.6
   --spare-actor-max-tokens 8192
   --spare-env-max-tokens 16384
   --spare-max-context-length "${MAX_CONTEXT_LENGTH:-32768}"
   --spare-max-turns 25
   --spare-gamma 0.99
   --spare-game-regeneration-interval 4
   --spare-skills Mathematical_Reasoning Logical_Deduction Spatial_Reasoning Pattern_Recognition Optimization Causal_Inference
   --spare-skills-per-regen 3
   --spare-num-games-per-rollout 24
   --spare-games-dir "${GAMES_DIR}"
   --spare-game-difficulty medium
   --spare-trajectories-per-game 16
   --spare-cache-dir "${OUTPUT_DIR}/spare_games_cache"
   --spare-env-generation-template qwen3_multiturn_game_generation
   --spare-actor-template qwen3_game
   --spare-reward-normalization grpo
   --spare-env-reward-variant blend
   --spare-plateau-weight "${PLATEAU_WEIGHT:-0.6}"
   --spare-regret-weight "${REGRET_WEIGHT:-0.4}"
   --spare-regret-floor
   --spare-regret-scale 0.15
   --spare-micro-lp-weight 0.0
   --spare-frontier-weight 0.0
   --spare-plateau-lo "${PLATEAU_LO:-0.4}"
   --spare-plateau-hi "${PLATEAU_HI:-0.6}"
   --spare-plateau-ramp "${PLATEAU_RAMP:-0.25}"
   --spare-hint-mode self
   --spare-hint-temperature 0.3
   --spare-hint-max-tokens 4096
   --spare-hint-plays-per-game 16   # matched to trajectories-per-game (16) for clean regret
   --spare-proposer-training-delay 4   # == regen (matched-timing regret at offset 0)
)

if [ -n "${CORPUS_FILE:-}" ]; then
   if [ ! -f "${CORPUS_FILE}" ]; then
      echo "ERROR: CORPUS_FILE=${CORPUS_FILE} not found." >&2
      exit 1
   fi
   echo "[CORPUS] Grounding proposer on ${CORPUS_FILE}"
   SPARE_ARGS+=(
      --spare-corpus-file "${CORPUS_FILE}"
      --spare-corpus-max-doc-tokens 6000
      --spare-corpus-seed 42
   )
fi

if [ "${NO_PROPOSER_TRAIN:-0}" = "1" ]; then
   echo "[NO-PROPOSER] Proposer training disabled (--spare-no-train-on-env-trajectories)"
   SPARE_ARGS+=( --spare-no-train-on-env-trajectories )
fi

if [ "${NO_ENV_MEMORY:-0}" = "1" ]; then
   echo "[NO-ENV-MEMORY] Env-memory disabled"
else
   SPARE_ARGS+=( --spare-use-env-memory --spare-env-memory-max-size 200 )
fi

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
   --wandb-group "${WANDB_GROUP:-spade-games-4b}"
   --wandb-key "${WANDB_API_KEY:?Set WANDB_API_KEY.}"
)

EVAL_ARGS=(
   --eval-interval 100000   # eval REMOVED: never fires within num-rollout (training-dynamics focus)
   --eval-config ${EVAL_CONFIG_FILE}
   --apply-chat-template
   --rm-type math
   --skip-eval-before-train
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
    \"LITELLM_LOG\": \"WARNING\"
  }
}"

RAY_JOB_ID="spade_games_4b_$$"

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
