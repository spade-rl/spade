"""Adaptive environment-reward assignment for proposer trajectories."""

from collections import defaultdict
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from spare.core.types import Trajectory
from spare.core.learning_potential import LearningPotential
from spare.core.utils.rewards import (
    STEP_TIMEOUT_PENALTY,
    compute_variance_reward,
)

logger = logging.getLogger(__name__)


def _mean_or_zero(values) -> float:
    return float(np.mean(values)) if values else 0.0


def _center_skill_rewards(
    skill_trajectories,
    skill_raw_rewards,
    env_reward_scale: float,
    skills=None,
) -> List[Trajectory]:
    result = []
    for skill in skills if skills is not None else skill_trajectories:
        trajectories = skill_trajectories[skill]
        raw_rewards = skill_raw_rewards[skill]
        if not trajectories:
            continue
        skill_mean = _mean_or_zero(raw_rewards)
        for trajectory, raw_reward in zip(trajectories, raw_rewards):
            trajectory.reward = (raw_reward - skill_mean) * env_reward_scale
            result.append(trajectory)
    return result


def _skill_game_stds(env_trajectories, game2rewards):
    skill_game_stds = defaultdict(list)
    for game_file, trajectory in env_trajectories.items():
        if game_file in game2rewards and len(game2rewards[game_file]) > 1:
            skill_game_stds[trajectory.metadata["skill"]].append(
                float(np.std(game2rewards[game_file]))
            )
    return skill_game_stds


def _lp_summary(learning_potentials) -> Dict[str, float]:
    trackers = list(learning_potentials.values())
    fast_values = [tracker.mu_fast for tracker in trackers if tracker.mu_fast is not None]
    slow_values = [tracker.mu_slow for tracker in trackers if tracker.mu_slow is not None]
    return {
        "env/avg_signed_lp": _mean_or_zero(
            [tracker.get_signed_potential() for tracker in trackers]
        ),
        "env/avg_learning_potential": float(
            np.mean([tracker.get_current_potential() for tracker in trackers])
        ),
        "env/avg_mu_fast": _mean_or_zero(fast_values),
        "env/avg_mu_slow": _mean_or_zero(slow_values),
    }


def _difficulty_metrics(game2rewards) -> Dict[str, float]:
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for rewards in game2rewards.values():
        win_rate = float(np.mean(rewards))
        if win_rate > 0.8:
            counts["easy"] += 1
        elif win_rate < 0.2:
            counts["hard"] += 1
        else:
            counts["medium"] += 1
    total = len(game2rewards)
    return {
        f"env/game_difficulty/{difficulty}_pct": count / total if total else 0.0
        for difficulty, count in counts.items()
    }


def assign_env_rewards(
    env_trajectories: Dict[str, Trajectory],
    skill2rewards: Dict[str, List[float]],
    game2rewards: Dict[str, List[float]],
    learning_potentials: Dict[str, LearningPotential],
    use_solver_variance_reward: bool,
    env_reward_scale: float = 10.0,
    skip_lp_update: bool = False,
) -> Tuple[List[Trajectory], Dict[str, Any]]:
    """Assign baseline-distance rewards and center them within each skill."""
    if not env_trajectories:
        return [], {}

    game_variance_rewards = defaultdict(list)
    skill_trajectories = defaultdict(list)
    skill_raw_rewards = defaultdict(list)

    for game_file, trajectory in env_trajectories.items():
        trajectory.metadata["game_file"] = game_file
        skill = trajectory.metadata["skill"]
        if game_file not in game2rewards and str(game_file) not in game2rewards:
            logger.warning(
                "[ASSIGN_ENV_REWARDS] No actor rewards for %s — skipping env trajectory",
                Path(game_file).name,
            )
            continue

        game_key = game_file if game_file in game2rewards else str(game_file)
        raw_reward = abs(
            np.mean(game2rewards[game_key]) - learning_potentials[skill].get_baseline()
        )
        if use_solver_variance_reward:
            variance_reward = compute_variance_reward(game2rewards[game_key])
            game_variance_rewards[skill].append(variance_reward)
            raw_reward += variance_reward
        if trajectory.metadata.get("turn1_valid", True) is False:
            raw_reward = 0.0

        trajectory.metadata["env_raw_reward"] = raw_reward
        skill_raw_rewards[skill].append(raw_reward)
        skill_trajectories[skill].append(trajectory)

    for skill, rewards in skill2rewards.items():
        if not skip_lp_update:
            learning_potentials[skill].update(_mean_or_zero(rewards))

    result = _center_skill_rewards(
        skill_trajectories,
        skill_raw_rewards,
        env_reward_scale,
        skills=skill2rewards,
    )
    skill_env_rewards = defaultdict(list)
    for trajectory in result:
        skill_env_rewards[trajectory.metadata["skill"]].append(trajectory.reward)
    skill_game_stds = _skill_game_stds(env_trajectories, game2rewards)

    stats: Dict[str, Any] = {}
    for skill, tracker in learning_potentials.items():
        skill_rewards = skill2rewards.get(skill, [])
        game_stds = skill_game_stds.get(skill, [])
        stats.update(
            {
                f"env/skill_{skill}/actor_win_rate": _mean_or_zero(skill_rewards),
                f"env/skill_{skill}/baseline": tracker.get_baseline(),
                f"env/skill_{skill}/signed_lp": tracker.get_signed_potential(),
                f"env/skill_{skill}/learning_potential": tracker.get_current_potential(),
                f"env/skill_{skill}/mu_fast": tracker.mu_fast
                if tracker.mu_fast is not None
                else 0.0,
                f"env/skill_{skill}/mu_slow": tracker.mu_slow
                if tracker.mu_slow is not None
                else 0.0,
                f"env/skill_{skill}/variance_reward": _mean_or_zero(
                    game_variance_rewards[skill]
                ),
                f"env/skill_{skill}/env_reward_std": float(
                    np.std(skill_env_rewards[skill])
                )
                if skill_env_rewards[skill]
                else 0.0,
                f"env/skill_{skill}/game_std_reward": _mean_or_zero(game_stds),
            }
        )

    game_win_rates = [float(np.mean(rewards)) for rewards in game2rewards.values()]
    stats["env/avg_actor_win_rate"] = _mean_or_zero(game_win_rates)
    stats.update(_lp_summary(learning_potentials))
    stats["env/avg_game_std_reward"] = _mean_or_zero(
        [value for values in skill_game_stds.values() for value in values]
    )
    stats.update(_difficulty_metrics(game2rewards))
    return result, stats


def assign_env_rewards_regret(
    env_trajectories: Dict[str, Trajectory],
    game2regret: Dict[str, float],
    game2rewards: Dict[str, List[float]],
    game2hint_stats: Dict[str, Dict[str, Any]],
    learning_potentials: Dict[str, LearningPotential],
    skill2rewards: Dict[str, List[float]],
    env_reward_scale: float = 10.0,
) -> Tuple[List[Trajectory], Dict[str, Any]]:
    """Assign centered regret rewards while preserving bounded failure signals."""
    if not env_trajectories:
        return [], {}

    skill_trajectories = defaultdict(list)
    skill_raw_rewards = defaultdict(list)
    floored_count = 0
    timeout_count = 0

    for game_file, trajectory in env_trajectories.items():
        trajectory.metadata["game_file"] = game_file
        skill = trajectory.metadata["skill"]
        if trajectory.metadata.get("step_timeout"):
            raw_reward = STEP_TIMEOUT_PENALTY
            timeout_count += 1
        else:
            raw_reward = game2regret.get(game_file, 0.0)
            if trajectory.metadata.get("turn1_valid", True) is False:
                raw_reward = 0.0
            if raw_reward < 0:
                floored_count += 1
                raw_reward = 0.0

        trajectory.metadata["env_raw_reward"] = raw_reward
        trajectory.metadata["regret"] = raw_reward
        skill_raw_rewards[skill].append(raw_reward)
        skill_trajectories[skill].append(trajectory)

    result = _center_skill_rewards(
        skill_trajectories,
        skill_raw_rewards,
        env_reward_scale,
    )
    for skill, rewards in skill2rewards.items():
        learning_potentials[skill].update(_mean_or_zero(rewards))

    skill_env_rewards = defaultdict(list)
    skill_regrets = defaultdict(list)
    for trajectory in result:
        skill = trajectory.metadata["skill"]
        skill_env_rewards[skill].append(trajectory.reward)
        skill_regrets[skill].append(trajectory.metadata.get("regret", 0.0))

    skill_hint_rewards = defaultdict(list)
    skill_no_hint_rewards = defaultdict(list)
    for game_file, trajectory in env_trajectories.items():
        hint_stats = game2hint_stats.get(game_file, {})
        skill = trajectory.metadata["skill"]
        if "r_hint" in hint_stats:
            skill_hint_rewards[skill].append(hint_stats["r_hint"])
        if "r_no_hint" in hint_stats:
            skill_no_hint_rewards[skill].append(hint_stats["r_no_hint"])

    skill_actor_rewards = defaultdict(list)
    for game_file, rewards in game2rewards.items():
        if game_file in env_trajectories:
            skill_actor_rewards[env_trajectories[game_file].metadata["skill"]].extend(
                rewards
            )
    skill_game_stds = _skill_game_stds(env_trajectories, game2rewards)

    stats: Dict[str, Any] = {}
    for skill in skill_trajectories:
        tracker = learning_potentials[skill]
        env_rewards = skill_env_rewards.get(skill, [])
        stats.update(
            {
                f"env/skill_{skill}/actor_win_rate": _mean_or_zero(
                    skill_actor_rewards.get(skill, [])
                ),
                f"env/skill_{skill}/baseline": tracker.get_baseline(),
                f"env/skill_{skill}/signed_lp": tracker.get_signed_potential(),
                f"env/skill_{skill}/learning_potential": tracker.get_current_potential(),
                f"env/skill_{skill}/mu_fast": tracker.mu_fast
                if tracker.mu_fast is not None
                else 0.0,
                f"env/skill_{skill}/mu_slow": tracker.mu_slow
                if tracker.mu_slow is not None
                else 0.0,
                f"env/skill_{skill}/env_reward_std": float(np.std(env_rewards))
                if env_rewards
                else 0.0,
                f"env/skill_{skill}/game_std_reward": _mean_or_zero(
                    skill_game_stds.get(skill, [])
                ),
                f"env/skill_{skill}/mean_regret": _mean_or_zero(
                    skill_regrets.get(skill, [])
                ),
                f"env/skill_{skill}/env_reward": _mean_or_zero(env_rewards),
                f"env/skill_{skill}/r_hint": _mean_or_zero(
                    skill_hint_rewards.get(skill, [])
                ),
                f"env/skill_{skill}/r_no_hint": _mean_or_zero(
                    skill_no_hint_rewards.get(skill, [])
                ),
            }
        )

    stats["env/avg_actor_win_rate"] = _mean_or_zero(
        [reward for rewards in game2rewards.values() for reward in rewards]
    )
    stats.update(_lp_summary(learning_potentials))
    stats["env/avg_game_std_reward"] = _mean_or_zero(
        [value for values in skill_game_stds.values() for value in values]
    )
    stats["env/avg_regret"] = _mean_or_zero(
        [game2regret.get(game_file, 0.0) for game_file in env_trajectories]
    )
    stats["env/avg_r_hint"] = _mean_or_zero(
        [data["r_hint"] for data in game2hint_stats.values() if "r_hint" in data]
    )
    stats["env/avg_r_no_hint"] = _mean_or_zero(
        [data["r_no_hint"] for data in game2hint_stats.values() if "r_no_hint" in data]
    )
    count = len(env_trajectories) or 1
    stats["env/proposer_floored_frac"] = floored_count / count
    stats["env/proposer_step_timeout_frac"] = timeout_count / count
    stats.update(_difficulty_metrics(game2rewards))
    return result, stats
