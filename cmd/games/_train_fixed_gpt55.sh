#!/usr/bin/env bash
set -euo pipefail

FIXED_MODEL_SIZE="${FIXED_MODEL_SIZE:?Set FIXED_MODEL_SIZE to 4b, 8b, or 30b.}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime"

STATIC_POOL_DIR="${STATIC_POOL_DIR:-/scratch/spare-gpt55-static-corpus/games}"
MIN_POOL_GAMES="${MIN_POOL_GAMES:-7872}"
NUM_ROLLOUT="${NUM_ROLLOUT:-400}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-24}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-192}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-32768}"
MAX_TURNS="${MAX_TURNS:-25}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/spare-workspace}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/spare-workspace}"

if [[ -z "${WANDB_ENTITY:-}" ]]; then
   echo "ERROR: Set WANDB_ENTITY to your Weights & Biases team/entity." >&2
   exit 1
fi

if (( GLOBAL_BATCH_SIZE % ROLLOUT_BATCH_SIZE != 0 )); then
   echo "ERROR: GLOBAL_BATCH_SIZE must be divisible by ROLLOUT_BATCH_SIZE." >&2
   exit 1
fi

if [[ -d "${STATIC_POOL_DIR}" ]]; then
   pool_count="$(find "${STATIC_POOL_DIR}" -maxdepth 1 -type f -name 'game_*.py' | wc -l | tr -d ' ')"
else
   pool_count=0
fi
if [[ "${pool_count}" -lt "${MIN_POOL_GAMES}" ]]; then
   echo "ERROR: ${STATIC_POOL_DIR} has ${pool_count} games; need at least ${MIN_POOL_GAMES}." >&2
   exit 1
fi

case "${FIXED_MODEL_SIZE}" in
   4b)
      MODEL_CONFIG="${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"
      MODEL_NAME="Qwen3-4B-Instruct-2507"
      KL_COEF="${KL_COEF:-0.00}"
      ACTOR_THINKING=0
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
      SGLANG_ARGS=(
         --rollout-num-gpus-per-engine 1
         --sglang-mem-fraction-static 0.7
         --sglang-tool-call-parser qwen
      )
      OPTIMIZER_EXTRA_ARGS=()
      ;;
   8b)
      MODEL_CONFIG="${PROJECT_ROOT}/cmd/models/qwen3-8B.sh"
      MODEL_NAME="Qwen3-8B"
      KL_COEF="${KL_COEF:-0.005}"
      ACTOR_THINKING=1
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
      SGLANG_ARGS=(
         --rollout-num-gpus-per-engine 1
         --sglang-mem-fraction-static 0.7
         --sglang-tool-call-parser qwen
      )
      OPTIMIZER_EXTRA_ARGS=()
      ;;
   30b)
      MODEL_CONFIG="${PROJECT_ROOT}/cmd/models/qwen3-30B-A3B.sh"
      MODEL_NAME="Qwen3-30B-A3B-Instruct-2507"
      KL_COEF="${KL_COEF:-0.00}"
      ACTOR_THINKING=0
      PERF_ARGS=(
         --tensor-model-parallel-size 4
         --sequence-parallel
         --pipeline-model-parallel-size 1
         --context-parallel-size 1
         --expert-model-parallel-size 8
         --expert-tensor-parallel-size 1
         --recompute-granularity full
         --recompute-method uniform
         --recompute-num-layers 1
         --use-dynamic-batch-size
         --max-tokens-per-gpu 8192
      )
      SGLANG_ARGS=(
         --rollout-num-gpus-per-engine 8
         --sglang-mem-fraction-static 0.7
         --sglang-ep-size 8
         --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
         --sglang-tool-call-parser qwen
         --sglang-moe-runner-backend triton
         --sglang-disable-custom-all-reduce
      )
      OPTIMIZER_EXTRA_ARGS=(
         --optimizer-cpu-offload
         --overlap-cpu-optimizer-d2h-h2d
         --use-precision-aware-optimizer
      )
      ;;
   *)
      echo "ERROR: FIXED_MODEL_SIZE must be 4b, 8b, or 30b; got ${FIXED_MODEL_SIZE}." >&2
      exit 1
      ;;
esac

TRAJECTORIES_PER_STEP=$((GLOBAL_BATCH_SIZE / ROLLOUT_BATCH_SIZE))
echo "[FIXED-ENV GRPO ${FIXED_MODEL_SIZE}]"
echo "  model: ${MODEL_NAME}"
echo "  pool: ${STATIC_POOL_DIR} (${pool_count} GPT-5.5 curated games)"
echo "  budget: ${NUM_ROLLOUT} rollouts"
echo "  batch: ${ROLLOUT_BATCH_SIZE} games x ${TRAJECTORIES_PER_STEP} training trajectories"
echo "  actor thinking: ${ACTOR_THINKING}; KL: ${KL_COEF}"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
   exit 0
fi

if [[ ! -f "${MODEL_CONFIG}" ]]; then
   echo "ERROR: model config not found: ${MODEL_CONFIG}. Initialize the Slime submodule if required." >&2
   exit 1
fi

# shellcheck source=/dev/null
source "${MODEL_CONFIG}"

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/spare_paper_fixed_gpt55_${FIXED_MODEL_SIZE}/$(date +%Y%m%d_%H%M%S)}"
GAMES_DIR="${STATIC_POOL_DIR}"

for process_name in sglang ray; do
   pkill -9 "${process_name}" 2>/dev/null || true
done
ray stop --force >/dev/null 2>&1 || true
sleep 3

export PYTHONUNBUFFERED=1
export WEAVE_PRINT_CALL_LINK=false
pip install math_verify weave

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l | tr -d ' ' || true)"
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi

mkdir -p "${OUTPUT_DIR}"
export WORKSPACE_DIR
EVAL_CONFIG_FILE="${OUTPUT_DIR}/eval_aime.yaml"
envsubst < "${PROJECT_ROOT}/eval_configs/eval_aime_avg32.yaml" > "${EVAL_CONFIG_FILE}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/${MODEL_NAME}"
   --ref-load "${MODEL_ROOT}/${MODEL_NAME}_torch_dist"
)
if [[ -n "${LOAD_DIR:-}" ]]; then
   CKPT_ARGS+=( --load "${LOAD_DIR}" )
fi
CKPT_ARGS+=(
   --save "${OUTPUT_DIR}/${MODEL_NAME}_fixed_gpt55/"
   --save-interval "${SAVE_INTERVAL:-16}"
)

ROLLOUT_ARGS=(
   --data-source-path spare.slime.data_source.SpareDataSource
   --rollout-function-path spare.slime.spare_rollout.spare_generate_rollout
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --use-dynamic-global-batch-size
   --balance-data
   --apply-chat-template-kwargs '{"enable_thinking":false}'
)

SPARE_ARGS=(
   --spare-gamma1 0.98
   --spare-gamma2 0.85
   --spare-actor-temperature 0.6
   --spare-actor-max-tokens "${ACTOR_MAX_TOKENS:-8192}"
   --spare-max-context-length "${MAX_CONTEXT_LENGTH}"
   --spare-max-turns "${MAX_TURNS}"
   --spare-gamma 0.99
   --spare-game-regeneration-interval 0
   --spare-skills Mathematical_Reasoning Logical_Deduction Spatial_Reasoning Pattern_Recognition Optimization Causal_Inference
   --spare-skills-per-regen 0
   --spare-num-games-per-rollout 24
   --spare-games-dir "${GAMES_DIR}"
   --spare-game-difficulty medium
   --spare-trajectories-per-game 16
   --spare-cache-dir "${OUTPUT_DIR}/spare_games_cache"
   --spare-env-generation-template qwen3_multiturn_game_generation
   --spare-actor-template qwen3_game
   --spare-reward-normalization grpo
   --spare-no-train-on-env-trajectories
   --spare-proposer-training-delay 0
   --spare-static-game-pool
   --spare-no-replacement
)
if [[ "${ACTOR_THINKING}" == "1" ]]; then
   SPARE_ARGS+=( --spare-actor-enable-thinking )
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-grpo-std-normalization
   --disable-rewards-normalization
   --use-kl-loss
   --kl-loss-coef "${KL_COEF}"
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
   --wandb-group "${WANDB_GROUP:-paper-fixed-gpt55-${FIXED_MODEL_SIZE}}"
   --wandb-key "${WANDB_API_KEY:?Set WANDB_API_KEY.}"
)

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL:-100000}"
   --eval-config "${EVAL_CONFIG_FILE}"
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
   "${OPTIMIZER_EXTRA_ARGS[@]}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"FLASHINFER_DISABLE_VERSION_CHECK\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"LITELLM_LOCAL_MODEL_COST_MAP\": \"True\",
    \"LITELLM_LOG\": \"WARNING\"
  }
}"

RAY_JOB_ID="spare_fixed_gpt55_${FIXED_MODEL_SIZE}_$$"
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
