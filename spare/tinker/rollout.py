"""Thin wrapper for SPARE rollouts using Tinker backend.

This module provides a simple interface to SpareOrchestrator for Tinker training.
All core logic lives in spare/core/orchestrator.py.
"""

import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from tinker import SamplingClient
from tinker_cookbook.rl.types import TrajectoryGroup

from spare.core.orchestrator import SpareOrchestrator
from spare.core.types import SpareConfig, Trajectory
from spare.core.learning_potential import LearningPotential
from spare.core.game_policy import GamePolicy
from spare.core.game_generator import SyntheticGameGenerator
from spare.core.utils.game_utils import (
    cleanup_old_games,
    normalize_rewards_per_game,
    upsample_trajectories,
    compute_env_reward_scale,
    assign_trajectory_weights,
    assign_env_rewards,
    assign_env_rewards_regret,
)
from spare.tinker.model_adapter import TinkerModelAdapter
from spare.tinker.trajectory_converter import spare_trajectory_to_tinker_trajectory

if TYPE_CHECKING:
    from train_spare_tinker import SpareConfig as TinkerSpareConfig

logger = logging.getLogger(__name__)


# Global state (persists across rollouts within a training run)
_LEARNING_POTENTIALS: Optional[Dict[str, LearningPotential]] = None
_GAME_POLICY: Optional[GamePolicy] = None
_ORCHESTRATOR: Optional[SpareOrchestrator] = None


def get_learning_potentials(cfg: "TinkerSpareConfig") -> Dict[str, LearningPotential]:
    """Get or create per-skill learning potential trackers."""
    global _LEARNING_POTENTIALS
    if _LEARNING_POTENTIALS is None:
        skills = list(SyntheticGameGenerator.COGNITIVE_SKILLS.keys())
        _LEARNING_POTENTIALS = {
            skill: LearningPotential(gamma_fast=cfg.gamma1, gamma_slow=cfg.gamma2)
            for skill in skills
        }
    return _LEARNING_POTENTIALS


def get_game_policy() -> GamePolicy:
    """Get or create the game policy."""
    global _GAME_POLICY
    if _GAME_POLICY is None:
        _GAME_POLICY = GamePolicy()
    return _GAME_POLICY


def get_orchestrator(
    sampling_client: SamplingClient,
    tokenizer,
    renderer,
    cfg: "TinkerSpareConfig",
) -> SpareOrchestrator:
    """Get or create the SpareOrchestrator (reuses across rollouts)."""
    global _ORCHESTRATOR

    if _ORCHESTRATOR is None:
        # Create model adapter
        model_adapter = TinkerModelAdapter(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            renderer=renderer,
            use_weave=cfg.use_weave,
        )

        # Create config
        spare_config = SpareConfig(
            env_temperature=cfg.env_temperature,
            actor_temperature=cfg.actor_temperature,
            max_turns=cfg.max_turns,
            gamma=cfg.gamma,
            env_max_tokens=cfg.env_max_tokens,
            actor_max_tokens=cfg.actor_max_tokens,
            max_context_length=cfg.max_context_length,
            env_generation_template=cfg.env_generation_template,
            actor_template=cfg.actor_template,
            use_format_reward=cfg.use_format_reward,
            format_reward_value=cfg.format_reward_value,
            use_solver_variance_reward=cfg.use_solver_variance_reward,
            use_self_judge=cfg.use_self_judge,
            self_judge_temperature=cfg.self_judge_temperature,
            self_judge_max_tokens=cfg.self_judge_max_tokens,
            self_judge_penalty=cfg.self_judge_penalty,
            self_judge_max_turns_to_show=cfg.self_judge_max_turns_to_show,
            reward_normalization=cfg.reward_normalization,
            game_baseline_decay=cfg.game_baseline_decay,
            env_reward_scaling_variant=cfg.env_reward_scaling_variant,
            max_env_reward_scale=cfg.max_env_reward_scale,
            auto_compute_env_reward_scale=cfg.auto_compute_env_reward_scale,
            train_on_env_trajectories=cfg.train_on_env_trajectories,
            # Environment reward variant
            env_reward_variant=cfg.env_reward_variant,
            hint_model=cfg.hint_model,
            hint_api_key_env=cfg.hint_api_key_env,
            hint_api_base_url=cfg.hint_api_base_url,
            hint_temperature=cfg.hint_temperature,
            hint_max_tokens=cfg.hint_max_tokens,
            hint_plays_per_game=cfg.hint_plays_per_game,
            # Environment validator parameters
            use_env_validator=cfg.use_env_validator,
            env_validator_model=cfg.env_validator_model,
            env_validator_api_key_env=cfg.env_validator_api_key_env,
            env_validator_api_base_url=cfg.env_validator_api_base_url,
            env_validator_temperature=cfg.env_validator_temperature,
            env_validator_max_tokens=cfg.env_validator_max_tokens,
        )

        _ORCHESTRATOR = SpareOrchestrator(
            model=model_adapter,
            config=spare_config,
            learning_potentials=get_learning_potentials(cfg),
            game_policy=get_game_policy(),
        )

    return _ORCHESTRATOR


async def spare_generate_rollout(
    sampling_client: SamplingClient,
    current_step: int,
    cfg: "TinkerSpareConfig",
    renderer,
    tokenizer,
) -> Tuple[List[TrajectoryGroup], Dict[str, Any]]:
    """Perform SPARE rollout and return list of Tinker TrajectoryGroups with metrics.

    This is the main entry point for Tinker training. It:
    1. Gets/creates the orchestrator
    2. Generates games and plays them (async)
    3. Converts results to Tinker format (one TrajectoryGroup per trajectory)
    4. Computes rollout metrics

    Args:
        sampling_client: Tinker SamplingClient for generation
        current_step: Current training step
        cfg: Training config
        renderer: Tinker renderer
        tokenizer: Tokenizer

    Returns:
        Tuple of (List[TrajectoryGroup], metrics_dict):
        - Each env trajectory → one TrajectoryGroup with single transition
        - Each actor trajectory → one TrajectoryGroup with multiple transitions
    """
    orchestrator = get_orchestrator(sampling_client, tokenizer, renderer, cfg)

    # Get config values
    games_dir = Path(cfg.games_dir)
    games_dir.mkdir(parents=True, exist_ok=True)

    should_regenerate = (
        cfg.game_regeneration_interval > 0 and
        current_step % cfg.game_regeneration_interval == 0
    )

    skills = list(SyntheticGameGenerator.COGNITIVE_SKILLS.keys())
    existing_games = list(games_dir.glob("game_*.py"))

    # Collect trajectories using async methods directly
    env_trajectories, actor_trajectories, collect_info = await collect_trajectories_async(
        orchestrator=orchestrator,
        should_regenerate=should_regenerate,
        skills=skills,
        difficulty=cfg.game_difficulty,
        game_files=existing_games,
        games_dir=games_dir,
        min_games=cfg.num_games_per_rollout,
        max_games=cfg.num_games_per_rollout,
        global_batch_size=cfg.batch_size,
        rollout_id=current_step,
        use_solver_variance_reward=cfg.use_solver_variance_reward,
        regeneration_interval=cfg.game_regeneration_interval,
    )

    # Convert to Tinker format (uses token IDs directly, no renderer needed)
    # Each trajectory is its own TrajectoryGroup:
    # - env: single transition (one-shot generation)
    # - actor: multiple transitions (multi-turn gameplay)
    trajectory_groups: List[TrajectoryGroup] = []

    for traj in env_trajectories:
        tinker_traj = spare_trajectory_to_tinker_trajectory(traj, role="environment")
        trajectory_groups.append(TrajectoryGroup(
            trajectories_G=[tinker_traj],
            final_rewards_G=[traj.reward or 0.0],
            metrics_G=[{"role": "environment", "skill": traj.metadata.get("skill", "unknown")}],
        ))

    for traj in actor_trajectories:
        tinker_traj = spare_trajectory_to_tinker_trajectory(traj, role="actor")
        trajectory_groups.append(TrajectoryGroup(
            trajectories_G=[tinker_traj],
            final_rewards_G=[traj.reward or 0.0],
            metrics_G=[{"role": "actor", "game_file": traj.metadata.get("game_file", "")}],
        ))

    # Compute metrics (matching Slime's format)
    actor_rewards = [t.reward for t in actor_trajectories if t.reward is not None]
    env_rewards = [t.reward for t in env_trajectories if t.reward is not None]
    all_rewards = actor_rewards + env_rewards

    trajectories_per_game = cfg.batch_size // cfg.num_games_per_rollout if cfg.num_games_per_rollout > 0 else 1

    metrics: Dict[str, Any] = {
        # Rollout counts
        "rollout/num_actor_trajectories": len(actor_trajectories),
        "rollout/num_env_trajectories": len(env_trajectories),
        "rollout/num_games": collect_info.get("num_games_attempted", 0) // max(trajectories_per_game, 1),
        "rollout/mode": "async",

        # Rewards
        "rollout/mean_return": sum(all_rewards) / max(len(all_rewards), 1),
        "rollout/actor_mean_return": sum(actor_rewards) / max(len(actor_rewards), 1) if actor_rewards else 0.0,
        "rollout/env_mean_return": sum(env_rewards) / max(len(env_rewards), 1) if env_rewards else 0.0,

        # From collect_info (replace spaces with underscores)
        **{k.replace(" ", "_"): v for k, v in collect_info.items()},
    }

    if all_rewards:
        metrics["rollout/max_return"] = max(all_rewards)
        metrics["rollout/min_return"] = min(all_rewards)

    logger.info(
        f"[ROLLOUT {current_step}] {len(actor_trajectories)} actor, {len(env_trajectories)} env, "
        f"mean_return={metrics['rollout/mean_return']:.3f}"
    )

    return trajectory_groups, metrics


async def collect_trajectories_async(
    orchestrator: SpareOrchestrator,
    should_regenerate: bool,
    skills: List[str],
    difficulty: str,
    game_files: List[Path],
    games_dir: Path,
    min_games: int = 1,
    max_games: int = 10,
    global_batch_size: int = 128,
    rollout_id: int = 0,
    cache_dir: Optional[Path] = None,
    max_attempts: int = 5,
    use_solver_variance_reward: bool = False,
    regeneration_interval: int = 50,
) -> Tuple[List[Trajectory], List[Trajectory], Dict[str, Any]]:
    """Async version of collect_trajectories for Tinker backend.

    This calls the orchestrator's async methods directly to avoid
    event loop nesting issues.
    """
    info: Dict[str, Any] = {}
    env2trajectories: Dict[str, Trajectory] = {}
    games_dir = Path(games_dir)
    games_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Cleanup old games if regenerating
    if should_regenerate and game_files:
        cleanup_old_games(game_files)
        updated_game_files: List[Path] = []
    else:
        updated_game_files = list(game_files)

    # Stage 2: Generate new games if needed (ASYNC)
    if should_regenerate or len(game_files) < min_games:
        num_to_generate = min_games - (0 if should_regenerate else len(game_files))
        base_index = 0 if should_regenerate else len(game_files)

        new_files, new_env_trajs = await orchestrator.generate_games_async(
            skills=skills,
            difficulty=difficulty,
            games_dir=games_dir,
            num_games=num_to_generate,
            base_index=base_index,
            rollout_id=rollout_id,
            cache_dir=cache_dir,
            max_attempts=max_attempts,
        )

        # Register with policy and update tracking
        for game_file, env_traj in zip(new_files, new_env_trajs.values()):
            updated_game_files.append(game_file)
            skill = env_traj.metadata["skill"]
            orchestrator.game_policy.register_game(str(game_file), skill, difficulty)

        for game_file, env_traj in new_env_trajs.items():
            env2trajectories[str(game_file)] = env_traj
        info["num_games_generated"] = len(new_files)

    # Stage 3: Select games to play
    selected_game_files = (
        random.sample(updated_game_files, max_games)
        if len(updated_game_files) > max_games
        else updated_game_files
    )

    # Stage 4: Play games (ASYNC)
    trajectories_per_game = math.ceil(global_batch_size / len(selected_game_files)) if selected_game_files else 1
    actor_trajectories, play_info = await orchestrator.play_games_async(
        game_files=selected_game_files,
        game_policy=orchestrator.game_policy,
        trajectories_per_game=trajectories_per_game,
    )

    # Stage 5: Normalize rewards per game
    actor_trajectories, stats = normalize_rewards_per_game(
        actor_trajectories,
        selected_game_files,
        game_baseline_tracker=orchestrator.game_baseline_tracker,
        reward_normalization=orchestrator.config.reward_normalization,
    )
    actor_info = {"actor/all_zero_std": 0.0}
    for skill, rewards in stats["skill"].items():
        actor_info[f"actor/skill_{skill}/avg_reward"] = float(np.mean(rewards))
        actor_info[f"actor/skill_{skill}/std_reward"] = float(np.std(rewards))

    for game_file, rewards in stats["game"].items():
        if float(np.std(rewards)) == 0.0:
            actor_info["actor/all_zero_std"] += 1
    actor_info["actor/all_zero_std"] /= max(len(stats["game"]), 1)

    # Add per-game baseline stats if enabled
    if orchestrator.game_baseline_tracker is not None:
        baseline_stats = orchestrator.game_baseline_tracker.get_stats()
        for k, v in baseline_stats.items():
            actor_info[f"actor/game_baseline/{k}"] = v

    # Stage 6: Compute environment rewards
    config = orchestrator.config
    if config.auto_compute_env_reward_scale and config.env_reward_scaling_variant == 1:
        env_reward_scale = compute_env_reward_scale(
            num_env_trajectories=len(env2trajectories),
            num_actor_trajectories=len(actor_trajectories),
            regeneration_interval=regeneration_interval,
            max_scale=config.max_env_reward_scale,
        )
    else:
        env_reward_scale = 1.0

    if config.env_reward_variant == "regret_based" and env2trajectories:
        # Regret-based: call external LLM for hints + dual gameplay
        if should_regenerate:
            game2regret, game2hint_stats = await orchestrator._compute_regrets_batched_async(
                env_trajectories=env2trajectories,
                game2rewards=stats["game"],
            )
            orchestrator._cached_regret = game2regret
            orchestrator._cached_hint_stats = game2hint_stats
        else:
            game2regret = orchestrator._cached_regret
            game2hint_stats = getattr(orchestrator, '_cached_hint_stats', {})

        env_trajectories, env_info = assign_env_rewards_regret(
            env_trajectories=env2trajectories,
            game2regret=game2regret,
            game2rewards=stats["game"],
            game2hint_stats=game2hint_stats,
            env_reward_scale=env_reward_scale,
        )
    else:
        # Default: learning potential variant
        env_trajectories, env_info = assign_env_rewards(
            env_trajectories=env2trajectories,
            skill2rewards=stats["skill"],
            game2rewards=stats["game"],
            learning_potentials=orchestrator.learning_potentials,
            use_solver_variance_reward=use_solver_variance_reward,
            env_reward_scale=env_reward_scale,
        )

    # Stage 6.5: Self-judge (ASYNC)
    if config.use_self_judge and env2trajectories:
        logger.info(f"[SELF-JUDGE] Running self-judge on {len(env2trajectories)} games")
        judge_results = await orchestrator.self_judge_games_async(
            env_trajectories=env2trajectories,
            actor_trajectories=actor_trajectories,
            max_concurrent=config.max_concurrent_games,
        )

        num_valid = 0
        num_invalid = 0
        for env_traj in env_trajectories:
            game_file = env_traj.metadata.get("game_file", "")
            if game_file in judge_results:
                is_valid, reasoning = judge_results[game_file]
                env_traj.metadata["self_judge_valid"] = is_valid
                env_traj.metadata["self_judge_reasoning"] = reasoning[:500]

                if is_valid:
                    num_valid += 1
                else:
                    num_invalid += 1
                    env_traj.reward += config.self_judge_penalty
                    logger.info(f"[SELF-JUDGE] Game {game_file} marked invalid, penalty applied")

        env_info["self_judge/num_valid"] = num_valid
        env_info["self_judge/num_invalid"] = num_invalid
        env_info["self_judge/valid_ratio"] = num_valid / (num_valid + num_invalid) if (num_valid + num_invalid) > 0 else 1.0

    # Stage 6.75: Assign trajectory weights (Variant 1)
    if config.env_reward_scaling_variant == 1:
        assign_trajectory_weights(
            env_trajectories=env_trajectories,
            actor_trajectories=actor_trajectories,
            regeneration_interval=regeneration_interval,
            max_sample_weight=config.max_env_reward_scale,
            auto_compute=config.auto_compute_env_reward_scale,
        )

    # Stage 7: Upsample if needed
    num_env = len(env_trajectories) if should_regenerate else 0
    min_actor_needed = global_batch_size - num_env
    info["batch/variant"] = config.env_reward_scaling_variant
    info["batch/num_env"] = num_env

    if len(actor_trajectories) < min_actor_needed:
        logger.warning(
            f"[COLLECT] Upsampling actor trajectories: {len(actor_trajectories)} -> {min_actor_needed}"
        )
        actor_trajectories = upsample_trajectories(actor_trajectories, min_actor_needed)

    info.update(**play_info, **actor_info, **env_info)

    # Stage 8: Optionally exclude env trajectories
    if not config.train_on_env_trajectories:
        logger.info(
            f"[COLLECT] Excluding {len(env_trajectories)} env trajectories from training "
            "(train_on_env_trajectories=False)"
        )
        env_trajectories = []

    # Final logging
    total_trajectories = len(env_trajectories) + len(actor_trajectories)
    logger.info(
        f"[COLLECT] Complete: {len(actor_trajectories)} actor, {len(env_trajectories)} env, "
        f"{total_trajectories} total (batch_size={global_batch_size})"
    )

    return env_trajectories, actor_trajectories, info
