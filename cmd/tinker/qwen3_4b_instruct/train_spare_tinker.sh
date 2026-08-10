#!/bin/bash

set -e  # Exit on error

if [ -z "${WANDB_ENTITY:-}" ]; then
    echo "ERROR: Set WANDB_ENTITY to your Weights & Biases team/entity." >&2
    exit 1
fi
export WANDB_ENTITY

echo "============================================"
echo "SPARE Training (Tinker Backend)"
echo "Model: Qwen3-4B-Instruct"
echo "============================================"
echo ""

pip install math_verify weave

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
RENDERER_NAME="${RENDERER_NAME:-qwen3_instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments/spare_tinker/qwen3_4b_instruct_$(date +%Y%m%d_%H%M%S)}"
GAMES_DIR="${GAMES_DIR:-${OUTPUT_DIR}/spare_games}"

GAME_DIFFICULTY="${GAME_DIFFICULTY:-hard}"
NUM_GAMES_PER_ROLLOUT="${NUM_GAMES_PER_ROLLOUT:-16}"
GAME_REGENERATION_INTERVAL="${GAME_REGENERATION_INTERVAL:-4}"
TRAJECTORIES_PER_GAME="${TRAJECTORIES_PER_GAME:-8}"

BATCH_SIZE="${BATCH_SIZE:-128}"  # global_batch_size from slime
LEARNING_RATE="${LEARNING_RATE:-1e-6}"  # matched from slime optimizer
MAX_TOKENS="${MAX_TOKENS:-32768}"
LORA_RANK="${LORA_RANK:-32}"

ENV_TEMPERATURE="${ENV_TEMPERATURE:-0.6}"
ENV_MAX_TOKENS="${ENV_MAX_TOKENS:-16000}"

ACTOR_TEMPERATURE="${ACTOR_TEMPERATURE:-0.6}"
ACTOR_MAX_TOKENS="${ACTOR_MAX_TOKENS:-8192}"
MAX_TURNS="${MAX_TURNS:-3}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-32768}"

GAMMA="${GAMMA:-0.99}"
USE_SOLVER_VARIANCE_REWARD="${USE_SOLVER_VARIANCE_REWARD:-true}"

GAMMA1="${GAMMA1:-0.98}"
GAMMA2="${GAMMA2:-0.85}"

ENV_REWARD_SCALING_VARIANT="${ENV_REWARD_SCALING_VARIANT:-1}"
MAX_ENV_REWARD_SCALE="${MAX_ENV_REWARD_SCALE:-50.0}"

USE_SELF_JUDGE="${USE_SELF_JUDGE:-false}"

EVAL_EVERY="${EVAL_EVERY:-16}"
SAVE_EVERY="${SAVE_EVERY:-32}"

FIXED_EVAL_INTERVAL="${FIXED_EVAL_INTERVAL:-16}"
FIXED_EVAL_MODEL="${FIXED_EVAL_MODEL:-gpt-5-mini}"
FIXED_EVAL_API_KEY_ENV="${FIXED_EVAL_API_KEY_ENV:-OPENAI_API_KEY}"
FIXED_EVAL_MAX_CONCURRENT="${FIXED_EVAL_MAX_CONCURRENT:-128}"

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/spare-workspace}"
AIME_2024_PATH="${AIME_2024_PATH:-${WORKSPACE_DIR}/aime-2024/aime-2024.jsonl}"
AIME_2025_PATH="${AIME_2025_PATH:-${WORKSPACE_DIR}/aime-2025/aime-2025.jsonl}"
AIME_2026_PATH="${AIME_2026_PATH:-${WORKSPACE_DIR}/aime-2026/aime-2026.jsonl}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-4}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-16384}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"

BASE_URL="${BASE_URL:-}"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Renderer: $RENDERER_NAME"
echo "  Games directory: $GAMES_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Game difficulty: $GAME_DIFFICULTY"
echo "  Num games per rollout: $NUM_GAMES_PER_ROLLOUT"
echo "  Game regeneration interval: $GAME_REGENERATION_INTERVAL"
echo "  Trajectories per game: $TRAJECTORIES_PER_GAME"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Env temperature: $ENV_TEMPERATURE"
echo "  Actor temperature: $ACTOR_TEMPERATURE"
echo "  Max turns: $MAX_TURNS"
echo "  Learning potential: gamma1=$GAMMA1, gamma2=$GAMMA2"
echo "  Env reward scaling variant: $ENV_REWARD_SCALING_VARIANT"
echo "  Self-judge: $USE_SELF_JUDGE"
echo "  Fixed eval: interval=$FIXED_EVAL_INTERVAL, model=$FIXED_EVAL_MODEL"
echo ""

SELF_JUDGE_ARGS=""
if [ "$USE_SELF_JUDGE" = "true" ]; then
    SELF_JUDGE_ARGS="use_self_judge=true self_judge_temperature=0.3 self_judge_max_tokens=2048 self_judge_penalty=-0.5"
fi

VARIANCE_ARGS=""
if [ "$USE_SOLVER_VARIANCE_REWARD" = "true" ]; then
    VARIANCE_ARGS="use_solver_variance_reward=true"
fi

AIME_ARGS=""
if [ -f "$AIME_2024_PATH" ]; then
    AIME_ARGS="$AIME_ARGS aime_2024_path=$AIME_2024_PATH"
    echo "  AIME 2024: $AIME_2024_PATH"
fi
if [ -f "$AIME_2025_PATH" ]; then
    AIME_ARGS="$AIME_ARGS aime_2025_path=$AIME_2025_PATH"
    echo "  AIME 2025: $AIME_2025_PATH"
fi
if [ -f "$AIME_2026_PATH" ]; then
    AIME_ARGS="$AIME_ARGS aime_2026_path=$AIME_2026_PATH"
    echo "  AIME 2026: $AIME_2026_PATH"
fi
if [ -n "$AIME_ARGS" ]; then
    AIME_ARGS="$AIME_ARGS eval_n_samples=$EVAL_N_SAMPLES eval_max_tokens=$EVAL_MAX_TOKENS eval_temperature=$EVAL_TEMPERATURE"
fi

echo "Starting SPARE training with Tinker..."
echo "Games will be generated on-the-fly during training."
echo ""

python3 train_spare_tinker.py \
    model_name="$MODEL_NAME" \
    renderer_name="$RENDERER_NAME" \
    lora_rank=$LORA_RANK \
    \
    games_dir="$GAMES_DIR" \
    game_difficulty="$GAME_DIFFICULTY" \
    num_games_per_rollout=$NUM_GAMES_PER_ROLLOUT \
    game_regeneration_interval=$GAME_REGENERATION_INTERVAL \
    trajectories_per_game=$TRAJECTORIES_PER_GAME \
    max_context_length=$MAX_CONTEXT_LENGTH \
    \
    batch_size=$BATCH_SIZE \
    learning_rate=$LEARNING_RATE \
    max_tokens=$MAX_TOKENS \
    \
    env_temperature=$ENV_TEMPERATURE \
    env_max_tokens=$ENV_MAX_TOKENS \
    actor_temperature=$ACTOR_TEMPERATURE \
    actor_max_tokens=$ACTOR_MAX_TOKENS \
    max_turns=$MAX_TURNS \
    \
    gamma=$GAMMA \
    gamma1=$GAMMA1 \
    gamma2=$GAMMA2 \
    env_reward_scaling_variant=$ENV_REWARD_SCALING_VARIANT \
    max_env_reward_scale=$MAX_ENV_REWARD_SCALE \
    $VARIANCE_ARGS \
    $SELF_JUDGE_ARGS \
    \
    eval_every=$EVAL_EVERY \
    save_every=$SAVE_EVERY \
    $AIME_ARGS \
    \
    fixed_eval_interval=$FIXED_EVAL_INTERVAL \
    fixed_eval_model="$FIXED_EVAL_MODEL" \
    fixed_eval_api_key_env="$FIXED_EVAL_API_KEY_ENV" \
    fixed_eval_max_concurrent=$FIXED_EVAL_MAX_CONCURRENT \
    \
    log_path="$OUTPUT_DIR" \
    wandb_entity="$WANDB_ENTITY" \
    wandb_project="spare" \
    wandb_name="spare_tinker_qwen3_4b_instruct_$(date +%Y%m%d_%H%M%S)" \
    use_weave=true \
    ${BASE_URL:+base_url="$BASE_URL"}

echo ""
echo "============================================"
echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"
