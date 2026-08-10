"""Trajectory construction, normalization, and role weighting."""

from collections import defaultdict
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING

import numpy as np
from spare.core.types import Trajectory, TrajectoryStatus
from spare.core.utils.rewards import (
    compute_env_reward_scale,
    episode_reward,
)

if TYPE_CHECKING:
    from spare.core.learning_potential import GameBaselineTracker

logger = logging.getLogger(__name__)

def normalize_rewards_per_game(
    actor_trajectories: List[Trajectory],
    selected_game_files: List[Path],
    game_baseline_tracker: Optional["GameBaselineTracker"] = None,
    reward_normalization: str = "ema_baseline",
) -> Tuple[List[Trajectory], Dict[str, Any]]:
    """Normalize actor rewards per game.

    Two variants controlled by reward_normalization:

    "ema_baseline": Subtract per-game EMA baseline.
        normalized = raw_reward - EMA_mean(game)
        Provides temporal improvement signal: positive when the actor does
        better than historical average, negative when worse. Works well with
        small plays_per_game and binary outcomes.

    "grpo": Per-game z-score normalization (GRPO-style).
        normalized = (raw_reward - batch_mean) / (batch_std + eps)
        Compares plays within the same game in the current batch. Requires
        sufficient plays_per_game for meaningful variance.

    Args:
        actor_trajectories: All actor trajectories
        selected_game_files: Game files that were played
        game_baseline_tracker: Tracker for per-game EMA baselines (ema_baseline mode)
        reward_normalization: "ema_baseline" or "grpo"

    Returns:
        Tuple of (trajectories with normalized rewards, stats dict)
    """
    if not actor_trajectories:
        return actor_trajectories, {}

    # Group trajectories by game file
    game_trajectories: Dict[str, List[Trajectory]] = {str(gf): [] for gf in selected_game_files}
    game2rewards: Dict[str, List[float]] = {}
    skill2rewards: Dict[str, List[float]] = defaultdict(list)

    for traj in actor_trajectories:
        game_trajectories[str(traj.metadata["game_file"])].append(traj)

    # Normalize rewards per game
    result = []

    for game_file, trajectories in game_trajectories.items():
        if not trajectories:
            continue

        # Get raw rewards for this game
        raw_rewards = [t.reward if t.status != TrajectoryStatus.FAILED else -1.0 for t in trajectories]

        if reward_normalization == "ema_baseline":
            # EMA baseline: normalized = raw - EMA_mean(game)
            baseline = 0.0
            if game_baseline_tracker is not None:
                baseline = game_baseline_tracker.get_baseline(game_file)
            normalized_rewards = [r - baseline for r in raw_rewards]
            # Update EMA ONCE with batch average of RAW rewards
            if game_baseline_tracker is not None:
                avg_raw = sum(raw_rewards) / len(raw_rewards)
                game_baseline_tracker.update_baseline(game_file, avg_raw)
        elif reward_normalization == "grpo":
            # GRPO: per-game z-score
            mean_reward = np.mean(raw_rewards)
            std_reward = np.std(raw_rewards)
            normalized_rewards = [(r - float(mean_reward)) / (float(std_reward) + 1e-8) for r in raw_rewards]
        else:
            raise ValueError(f"Unknown reward_normalization: {reward_normalization}")

        # Store ORIGINAL rewards for learning potential computation
        game2rewards[game_file] = list(raw_rewards)
        skill2rewards[trajectories[0].metadata["skill"]].extend(raw_rewards)

        for i, traj in enumerate(trajectories):
            metadata = {
                **traj.metadata,
                "original_reward": traj.reward,
                "normalized_reward": normalized_rewards[i],
            }
            if reward_normalization == "ema_baseline" and game_baseline_tracker is not None:
                metadata["game_baseline"] = baseline

            normalized_traj = Trajectory(
                index=traj.index,
                prompt=traj.prompt,
                messages=traj.messages,
                tokens=traj.tokens,
                loss_mask=traj.loss_mask,
                response=traj.response,
                response_length=traj.response_length,
                rollout_log_probs=traj.rollout_log_probs,
                reward=float(normalized_rewards[i]),
                status=traj.status,
                metadata=metadata,
                turn_count=traj.turn_count,
            )
            result.append(normalized_traj)

    stats: Dict[str, Any] = {
        "game": game2rewards,
        "skill": dict(skill2rewards),
    }

    return result, stats


def upsample_trajectories(
    trajectories: List[Trajectory],
    min_samples: int,
) -> List[Trajectory]:
    """Upsample trajectories to meet minimum sample count.

    When some games/trajectories fail, we may not have enough samples.
    This duplicates existing trajectories (with new indices) to fill the gap.

    Args:
        trajectories: List of collected trajectories
        min_samples: Minimum number of samples required

    Returns:
        List of trajectories with at least min_samples entries
    """
    if not trajectories or len(trajectories) >= min_samples:
        return trajectories

    original_count = len(trajectories)
    result = list(trajectories)

    # Duplicate trajectories round-robin until we have enough
    idx = 0
    while len(result) < min_samples:
        # Create a copy with updated index
        original = trajectories[idx % original_count]
        duplicate = Trajectory(
            index=len(result),
            prompt=original.prompt,
            messages=original.messages,
            tokens=original.tokens,
            loss_mask=original.loss_mask,
            response=original.response,
            response_length=original.response_length,
            rollout_log_probs=original.rollout_log_probs,
            reward=original.reward,
            status=original.status,
            metadata={**original.metadata, "upsampled": True, "original_index": original.index},
            turn_count=original.turn_count,
        )
        result.append(duplicate)
        idx += 1

    logger.info(f"[UPSAMPLE] Upsampled {original_count} -> {len(result)} trajectories")
    return result


# =============================================================================
# TRAJECTORY BUILDING
# =============================================================================

def build_env_trajectory(
    messages: List[Dict[str, str]],
    all_tokens: List[int],
    all_masks: List[int],
    all_logprobs: List[float],
    assistant_responses: List[str],
    skill: str,
    difficulty: str,
    game_code: str,
    index: int = 0,
) -> Trajectory:
    """Build environment trajectory from generation result.

    This is the shared logic for building env trajectories, used by both
    batched and async generation paths.

    Args:
        model: Model adapter with tokenizer and apply_template
        messages: Input messages for generation
        skill: Cognitive skill name
        difficulty: Difficulty level
        game_code: Generated game code
        index: Trajectory index

    Returns:
        Environment trajectory
    """
    initial_prompt = messages[0]["content"] if messages else ""

    return Trajectory(
        index=index,
        prompt=initial_prompt,
        messages=messages,
        tokens=all_tokens,
        loss_mask=all_masks,
        response="".join(assistant_responses),
        response_length=len(all_logprobs),
        rollout_log_probs=all_logprobs,
        reward=0.0, # Reward is delayed until gameplay are collected
        status=TrajectoryStatus.COMPLETED,
        metadata={"skill": skill, "difficulty": difficulty, "game_code": game_code},
        turn_count=1,
    )


def build_actor_trajectory(
    messages: List[Dict[str, str]],
    all_tokens: List[int],
    all_masks: List[int],
    all_logprobs: List[float],
    assistant_responses: List[str],
    rewards: List[float],
    game_file_path: str,
    skill: str,
    turn: int,
    terminated: bool,
    truncated: bool,
    index: int = 0,
) -> Trajectory:
    """Build actor trajectory from gameplay session.

    This is the shared logic for building actor trajectories, used by both
    sync, async, and batched gameplay paths.

    Args:
        messages: Full conversation messages
        all_tokens: All token IDs
        all_masks: Loss masks for tokens
        all_logprobs: Log probabilities for tokens
        assistant_responses: List of assistant responses
        rewards: List of rewards per turn
        game_file_path: Path to the game file
        skill: Cognitive skill name
        turn: Number of turns played
        terminated: Whether game terminated normally
        truncated: Whether game was truncated
        index: Trajectory index

    Returns:
        Actor trajectory
    """
    # Outcome reward: terminal reward if the goal was reached, else 0 (clamped). Intermediate
    # shaping is intentionally ignored — see episode_reward.
    reward = episode_reward(rewards, terminated)

    if terminated:
        status = TrajectoryStatus.COMPLETED
    elif truncated:
        status = TrajectoryStatus.TRUNCATED
    else:
        status = TrajectoryStatus.TIMEOUT

    initial_prompt = messages[0]["content"] if messages else ""

    return Trajectory(
        index=index,
        prompt=initial_prompt,
        messages=messages,
        tokens=all_tokens,
        loss_mask=all_masks,
        response="".join(assistant_responses),
        response_length=len(all_logprobs),
        rollout_log_probs=all_logprobs,
        reward=reward,
        status=status,
        metadata={"game_file": game_file_path, "turns": turn, "skill": skill},
        turn_count=turn,
    )


def assign_trajectory_weights(
    env_trajectories: List[Trajectory],
    actor_trajectories: List[Trajectory],
    regeneration_interval: int = 50,
    max_sample_weight: float = 50.0,
    auto_compute: bool = True,
) -> Dict[str, float]:
    """Assign importance weights to balance role frequencies in training.

    Environment trajectories are upweighted to compensate for their lower
    frequency (generated every regeneration_interval steps vs every step).
    This helps balance gradient contribution between roles.

    Args:
        env_trajectories: Environment generation trajectories (rare)
        actor_trajectories: Actor gameplay trajectories (frequent)
        regeneration_interval: Steps between environment regenerations
        max_sample_weight: Maximum weight cap to prevent instability
        auto_compute: If True, compute weights from trajectory counts

    Returns:
        Dict with computed weights: {"env_weight": float, "actor_weight": float, "computed_scale": float}
    """
    num_env = len(env_trajectories)
    num_actor = len(actor_trajectories)

    if auto_compute and num_env > 0 and num_actor > 0:
        # Auto-compute: actor-to-env ratio per regeneration window
        env_weight = compute_env_reward_scale(
            num_env_trajectories=num_env,
            num_actor_trajectories=num_actor,
            regeneration_interval=regeneration_interval,
            max_scale=max_sample_weight,
        )
        actor_weight = 1.0
    else:
        env_weight = 1.0
        actor_weight = 1.0

    for env_traj in env_trajectories:
        env_traj.metadata["sample_weight"] = env_weight
        env_traj.metadata["role"] = "environment"

    for actor_traj in actor_trajectories:
        actor_traj.metadata["sample_weight"] = actor_weight
        actor_traj.metadata["role"] = "actor"

    logger.info(
        f"[WEIGHTS] Assigned weights: env={env_weight:.1f} ({num_env} trajs), "
        f"actor={actor_weight:.1f} ({num_actor} trajs), "
        f"regen_interval={regeneration_interval}"
    )

    return {
        "env_weight": env_weight,
        "actor_weight": actor_weight,
        "computed_scale": env_weight,  # For logging
    }
