#!/usr/bin/env python3
"""
SPARE training script using Tinker framework.

This is the main entry point for SPARE training with the Tinker backend.
Uses SpareOrchestrator for unified dual-role training.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import chz
from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.rl import train
from tinker_cookbook.tokenizer_utils import get_tokenizer, Tokenizer
from tinker_cookbook.renderers import get_renderer

from spare.tinker.rollout import spare_generate_rollout
from spare.tinker.train_step import train_step
from spare.tinker.evaluators import create_aime_evaluators
from spare.core.eval.fixed_model_eval import run_fixed_model_evaluation
import time
import tinker
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

RENDERER_NAME_MAP = {
    "qwen3_game": "qwen3_instruct",
    "qwen3_game_generation": "qwen3_instruct",
}

@chz.chz
class SpareConfig(train.Config):
    """Configuration for SPARE training with Tinker."""

    # Model settings
    model_name: str = "Qwen/Qwen3-8B-Base"
    renderer_name: str = "qwen3"
    lora_rank: int = 32

    # Game settings
    games_dir: str = "./generated_games"
    max_context_length: int = 16384
    game_difficulty: str = "hard"
    num_games_per_rollout: int = 4
    game_regeneration_interval: int = 50
    trajectories_per_game: int = 1
    cache_dir: str | None = None  # Optional directory to cache generated games

    # Training settings
    batch_size: int = 32
    num_train_datapoints: int = 12800
    num_test_datapoints: int = 128
    learning_rate: float = 1e-4
    max_tokens: int = 8192
    num_substeps: int = 1

    # Environment generation settings
    env_temperature: float = 0.7
    env_max_tokens: int = 8192
    env_generation_template: str = "qwen3_game_generation"

    # Actor settings
    actor_temperature: float = 1.0
    actor_max_tokens: int = 8192
    actor_template: str = "qwen3_game"
    max_turns: int = 30

    # Reward settings
    gamma: float = 0.99
    use_format_reward: bool = False
    format_reward_value: float = 1.0
    use_solver_variance_reward: bool = False

    # Learning potential settings
    gamma1: float = 0.99  # Fast moving average
    gamma2: float = 0.95  # Slow moving average

    # Actor reward normalization settings
    reward_normalization: str = "ema_baseline"  # "ema_baseline" or "grpo"
    game_baseline_decay: float = 0.5  # Decay rate for per-game baseline EMA (ema_baseline only)

    # Environment reward scaling settings
    env_reward_scaling_variant: int = 1  # 0=none, 1=simple scaling
    max_env_reward_scale: float = 50.0  # Cap for auto-computed scale
    auto_compute_env_reward_scale: bool = True  # Auto-compute scale from trajectory counts
    train_on_env_trajectories: bool = True  # Include env trajectories in training

    # Self-judge settings
    use_self_judge: bool = False
    self_judge_temperature: float = 0.3
    self_judge_max_tokens: int = 2048
    self_judge_penalty: float = -0.5
    self_judge_max_turns_to_show: int = 5

    # Environment validator settings (closed-source LLM rejection sampling)
    use_env_validator: bool = False
    env_validator_model: str = "google/gemini-3-flash-preview"
    env_validator_api_key_env: str = "OPENROUTER_API_KEY"
    env_validator_api_base_url: str | None = "https://openrouter.ai/api/v1"
    env_validator_temperature: float = 0.3
    env_validator_max_tokens: int = 16384

    # Environment reward variant
    env_reward_variant: str = "learning_potential"  # "learning_potential" or "regret_based"

    # Regret-based env reward parameters
    hint_model: str = "gpt-5.1-mini"
    hint_api_key_env: str = "OPENAI_API_KEY"
    hint_api_base_url: str | None = None
    hint_temperature: float = 0.3
    hint_max_tokens: int = 8192
    hint_plays_per_game: int = 4

    # Evaluation
    eval_every: int = 16
    save_every: int = 20
    compute_post_kl: bool = False

    # AIME evaluation settings
    aime_2024_path: str | None = None
    aime_2025_path: str | None = None
    eval_n_samples: int = 4  # Pass@K for evaluation
    eval_max_tokens: int = 16384
    eval_temperature: float = 0.6

    # Fixed model evaluation settings
    fixed_eval_interval: int = 0  # Batches between fixed model evaluations (0=disabled)
    fixed_eval_model: str = "gpt-5-mini"  # Model ID for evaluation
    fixed_eval_api_base_url: str | None = None  # API base URL (None for OpenAI default)
    fixed_eval_api_key_env: str = "OPENAI_API_KEY"  # Env var for API key
    fixed_eval_temperature: float = 0.7  # Temperature for fixed model
    fixed_eval_max_tokens: int = 16384  # Max tokens for fixed model response
    fixed_eval_max_concurrent: int = 128  # Max concurrent game plays

    # Loss function
    loss_fn: str = "importance_sampling"

    # Logging
    wandb_entity: str | None = None
    wandb_project: str | None = "spade"
    wandb_name: str | None = None
    log_path: str = ""

    # Tinker service
    base_url: str | None = None

    # Checkpoint resumption
    resume_from_checkpoint: str | None = None

    # Weave tracing
    use_weave: bool = False


async def do_spare_training_loop(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: SpareConfig,
    training_client,
    _service_client,
    evaluators: list,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    renderer,
):
    """
    Main SPARE training loop using SpareOrchestrator.

    Args:
        start_batch: Starting batch index
        end_batch: Ending batch index
        num_batches: Total number of batches
        cfg: Training configuration
        training_client: Tinker training client
        _service_client: Tinker service client (unused, kept for future use)
        evaluators: List of evaluators
        ml_logger: Logger for metrics
        tokenizer: Tokenizer
        renderer: Renderer for prompt formatting
    """

    games_dir = Path(cfg.games_dir)
    games_dir.mkdir(parents=True, exist_ok=True)

    for i_batch in range(start_batch, end_batch):
        t_start = time.time()

        sampling_client, _ = await train.save_checkpoint_and_get_sampling_client(
            training_client, i_batch + 1, cfg.log_path, cfg.save_every
        )

        if i_batch % cfg.save_every == 0:
            logger.info(f"Saved checkpoint at batch {i_batch}")

        metrics: Dict[str, Any] = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }

        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0 and evaluators:
            logger.info(f"[EVAL] Running {len(evaluators)} evaluator(s) at batch {i_batch}...")
            for evaluator in evaluators:
                eval_metrics = await evaluator(sampling_client)
                metrics.update(eval_metrics)
                logger.info(f"[EVAL] Batch {i_batch}: {eval_metrics}")

        if cfg.fixed_eval_interval > 0 and i_batch % cfg.fixed_eval_interval == 0:
            api_key = os.environ.get(cfg.fixed_eval_api_key_env)
            if api_key:
                try:
                    logger.info(f"[FIXED_EVAL] Running fixed model evaluation at batch {i_batch}...")
                    fixed_eval_result = await run_fixed_model_evaluation(
                        games_dir=games_dir,
                        model=cfg.fixed_eval_model,
                        plays_per_game=cfg.trajectories_per_game,
                        max_turns=cfg.max_turns,
                        max_concurrent=cfg.fixed_eval_max_concurrent,
                        temperature=cfg.fixed_eval_temperature,
                        max_tokens=cfg.fixed_eval_max_tokens,
                        api_base_url=cfg.fixed_eval_api_base_url,
                        api_key_env=cfg.fixed_eval_api_key_env,
                    )
                    fixed_eval_metrics = fixed_eval_result.to_metrics_dict()
                    metrics.update(fixed_eval_metrics)
                    logger.info(
                        f"[FIXED_EVAL] Batch {i_batch}: pass_rate={fixed_eval_metrics.get('fixed_eval/overall_pass_rate', 0):.2f}, "
                        f"difficulty_score={fixed_eval_metrics.get('fixed_eval/difficulty_score', 0):.2f}"
                    )
                except Exception as e:
                    logger.error(f"[FIXED_EVAL] Fixed model evaluation failed: {e}")
                    metrics["fixed_eval/error"] = 1.0
            else:
                logger.warning(f"[FIXED_EVAL] Skipped: {cfg.fixed_eval_api_key_env} not set")

        logger.info(f"[BATCH {i_batch}] Starting rollout...")
        traj_groups, rollout_metrics = await spare_generate_rollout(
            sampling_client=sampling_client,
            current_step=i_batch,
            cfg=cfg,
            renderer=renderer,
            tokenizer=tokenizer,
        )

        metrics.update(rollout_metrics)

        train_metrics = await train_step(
            cfg=cfg,
            i_batch=i_batch,
            training_client=training_client,
            tokenizer=tokenizer,
            trajectory_groups_P=traj_groups,
            sampling_client=sampling_client,
        )
        metrics.update(train_metrics)

        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)

        logger.info(
            f"[BATCH {i_batch}/{num_batches}] "
            f"Actor: {metrics.get('rollout/num_actor_trajectories', 0)}, "
            f"Env: {metrics.get('rollout/num_env_trajectories', 0)}, "
            f"Mean return: {metrics.get('rollout/mean_return', 0):.3f}, "
            f"Avg ρ_t: {metrics.get('learning_potential/avg_rho_t', 0):.4f}, "
            f"Loss: {metrics.get('train/loss', 0):.4f}"
        )


async def create_spare_train_loop(cfg: SpareConfig):
    """
    Create and run SPARE training loop.

    This is the main entry point that sets up all components and runs training.
    """
    if not cfg.log_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg = chz.replace(cfg, log_path=f"./experiments/spare_tinker/{timestamp}")
    os.makedirs(cfg.log_path, exist_ok=True)

    if cfg.wandb_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg = chz.replace(cfg, wandb_name=f"spare_tinker_{timestamp}")

    logger.info("=" * 80)
    logger.info("SPARE Training Configuration (Tinker Backend)")
    logger.info("=" * 80)
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Games directory: {cfg.games_dir}")
    logger.info(f"Game difficulty: {cfg.game_difficulty}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"Learning rate: {cfg.learning_rate}")
    logger.info(f"Learning potential: γ1={cfg.gamma1}, γ2={cfg.gamma2}")
    logger.info(f"Self-judge: {cfg.use_self_judge}")
    logger.info(f"Eval every: {cfg.eval_every} batches")
    logger.info(f"AIME 2024: {cfg.aime_2024_path}")
    logger.info(f"AIME 2025: {cfg.aime_2025_path}")
    logger.info(f"Log path: {cfg.log_path}")
    logger.info("=" * 80)

    tokenizer = get_tokenizer(cfg.model_name)
    renderer_name = RENDERER_NAME_MAP.get(cfg.renderer_name, cfg.renderer_name)
    renderer = get_renderer(renderer_name, tokenizer)

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        config=cfg,
    )

    service_client = tinker.ServiceClient(base_url=cfg.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"]
            )
        )
        logger.info(f"Resumed training from {resume_info['state_path']} at batch {start_batch}")
    elif cfg.resume_from_checkpoint:
        training_client = await service_client.create_training_client_from_state_async(
            cfg.resume_from_checkpoint
        )
        start_batch = 0
        logger.info(f"Loaded weights from {cfg.resume_from_checkpoint}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )
        start_batch = 0

    num_batches = cfg.num_train_datapoints // cfg.batch_size

    evaluators = create_aime_evaluators(
        renderer=renderer,
        aime_2024_path=cfg.aime_2024_path,
        aime_2025_path=cfg.aime_2025_path,
        n_samples=cfg.eval_n_samples,
        max_tokens=cfg.eval_max_tokens,
        temperature=cfg.eval_temperature,
    )
    if evaluators:
        logger.info(f"Created {len(evaluators)} AIME evaluator(s)")
    else:
        logger.info("No AIME evaluators configured (set aime_2024_path or aime_2025_path)")

    await do_spare_training_loop(
        start_batch,
        num_batches,
        num_batches,
        cfg,
        training_client,
        service_client,
        evaluators,
        ml_logger,
        tokenizer,
        renderer,
    )

    logger.info("Training complete!")


def main():
    """Main entry point for SPARE Tinker training."""
    cli_config = chz.entrypoint(SpareConfig)

    cli_utils.check_log_dir(cli_config.log_path, behavior_if_exists="ask")

    asyncio.run(create_spare_train_loop(cli_config))


if __name__ == "__main__":
    main()
