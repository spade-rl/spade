"""Pure reward calculations shared by rollout backends."""

import numpy as np

from spare.core.utils.parsing import extract_boxed_answer

# Bounded penalty for generated games whose env.step() times out.
STEP_TIMEOUT_PENALTY = -0.5


def compute_variance_reward(rewards: list[float]) -> float:
    variance = np.var(rewards)
    return np.exp(-((variance - 0.25) ** 2) / (2 * 0.01))


def plateau_reward(
    win_rate: float,
    lo: float = 0.4,
    hi: float = 0.6,
    ramp: float = 0.25,
) -> float:
    if win_rate < lo:
        return max(0.0, 1.0 - (lo - win_rate) / ramp)
    if win_rate > hi:
        return max(0.0, 1.0 - (win_rate - hi) / ramp)
    return 1.0


def compute_format_reward(
    response: str,
    format_reward_value: float,
    no_format_penalty: float,
) -> float:
    extracted = extract_boxed_answer(response)
    return format_reward_value if extracted is not None and extracted.strip() else no_format_penalty


def compute_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = np.zeros_like(rewards, dtype=np.float32)
    current = 0.0
    for index in reversed(range(len(rewards))):
        current = rewards[index] + gamma * current
        returns[index] = current
    return returns.tolist()


def episode_reward(rewards: list[float], terminated: bool) -> float:
    if not rewards or not terminated:
        return 0.0
    return max(-1.0, min(1.0, float(rewards[-1])))


def compute_env_reward_scale(
    num_env_trajectories: int,
    num_actor_trajectories: int,
    regeneration_interval: int,
    max_scale: float = 50.0,
) -> float:
    if num_env_trajectories <= 0 or num_actor_trajectories <= 0:
        return 1.0
    scale = num_actor_trajectories / num_env_trajectories * regeneration_interval
    return min(scale, max_scale)
