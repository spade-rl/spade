"""Converter between SpareOrchestrator's episode Trajectory and Tinker's Transition/Trajectory format.

This module converts trajectories using token IDs directly from the orchestrator,
avoiding redundant re-tokenization. The orchestrator tracks tokens continuously
via token-in-token-out, so we use those directly.
"""

import logging
from typing import List

from tinker import ModelInput
from tinker_cookbook.rl.types import Transition, Trajectory as TinkerTrajectory, TokensWithLogprobs

from spare.core.types import Trajectory as SpareTrajectory

logger = logging.getLogger(__name__)


def spare_trajectory_to_tinker_trajectory(
    trajectory: SpareTrajectory,
    role: str = "actor",
) -> TinkerTrajectory:
    """Convert a SPARE episode Trajectory to a Tinker Trajectory.

    Uses token IDs directly from the trajectory (token-in-token-out tracking).
    No re-tokenization needed.

    Args:
        trajectory: Episode Trajectory from orchestrator with tokens and loss_mask
        renderer: Unused (kept for backward compatibility)
        role: Role identifier ("actor" or "environment")

    Returns:
        Tinker Trajectory object
    """
    if role == "environment":
        return _env_trajectory_to_tinker(trajectory)
    else:
        return _actor_trajectory_to_tinker(trajectory)


def _env_trajectory_to_tinker(trajectory: SpareTrajectory) -> TinkerTrajectory:
    """Convert an environment generation trajectory to Tinker format.

    Environment generation is single-turn:
    - obs: prompt tokens (loss_mask=0)
    - action: response tokens (loss_mask=1)

    Args:
        trajectory: Episode Trajectory for environment generation

    Returns:
        Tinker Trajectory with single transition
    """
    tokens = list(trajectory.tokens)
    loss_mask = list(trajectory.loss_mask)

    if not tokens or not loss_mask:
        # Empty trajectory
        empty_ob = ModelInput.from_ints([])
        return TinkerTrajectory(transitions=[], final_ob=empty_ob)

    # Find where response starts (first loss_mask=1)
    response_start = len(loss_mask)

    # Split into observation and action
    obs_tokens = tokens[:-response_start]
    action_tokens = tokens[-response_start:]

    # Get logprobs for action tokens
    # rollout_log_probs only contains response tokens, not full sequence
    action_logprobs = list(trajectory.rollout_log_probs)
    assert len(action_logprobs) == len(action_tokens), \
        f"Logprobs length mismatch: {len(action_logprobs)} vs {len(action_tokens)}, obs_tokens: {len(obs_tokens)}, action_tokens: {len(action_tokens)}"

    # Create observation and action
    ob = ModelInput.from_ints(obs_tokens)
    ac_with_logprobs = TokensWithLogprobs(
        tokens=action_tokens,
        maybe_logprobs=action_logprobs,
    )

    # Create transition
    transition = Transition(
        ob=ob,
        ac=ac_with_logprobs,
        reward=trajectory.reward,  # Learning potential for environment
        episode_done=True,  # Environment generation is one-shot
        metrics={
            "skill": trajectory.metadata.get("skill", "unknown"),
            "difficulty": trajectory.metadata.get("difficulty", "unknown"),
            "turn_count": trajectory.turn_count,
        },
    )

    # Final observation is empty (generation is one-shot)
    final_ob = ModelInput.from_ints([])

    return TinkerTrajectory(transitions=[transition], final_ob=final_ob)


def _actor_trajectory_to_tinker(trajectory: SpareTrajectory) -> TinkerTrajectory:
    """Convert an actor gameplay trajectory to Tinker format.

    Multi-turn gameplay:
    - Turn N obs: all tokens from start up to where Nth response begins
    - Turn N action: the Nth response tokens

    Pattern in tokens/loss_mask:
    [obs1 tokens (0s)] [resp1 tokens (1s)] [obs2 tokens (0s)] [resp2 tokens (1s)] ...

    Args:
        trajectory: Episode Trajectory for actor gameplay

    Returns:
        Tinker Trajectory with multiple transitions (one per turn)
    """
    tokens = list(trajectory.tokens)
    loss_mask = list(trajectory.loss_mask)
    rollout_log_probs = list(trajectory.rollout_log_probs)

    if not tokens or not loss_mask:
        empty_ob = ModelInput.from_ints([])
        return TinkerTrajectory(transitions=[], final_ob=empty_ob)

    boundaries = [0] + [i for i in range(1, len(loss_mask)) if loss_mask[i] != loss_mask[i-1]] + [len(loss_mask)]
    actions = [(boundaries[i], boundaries[i+1]) for i in range(0, len(boundaries)-1, 2)]

    if not actions:
        # No actions found
        ob = ModelInput.from_ints(tokens)
        return TinkerTrajectory(transitions=[], final_ob=ob)

    base_offset = len(tokens) - len(loss_mask)
    transitions = []

    for turn_idx, (action_start, action_end) in enumerate(actions):
        obs = tokens[:base_offset + action_start]
        # Action: tokens for this assistant response
        action_tokens = tokens[base_offset + action_start:base_offset + action_end]
        action_len = len(action_tokens)

        logprobs = rollout_log_probs[action_start:action_end]
        assert len(logprobs) == action_len, \
            f"Logprobs length mismatch: {len(logprobs)} vs {action_len}"

        # Create observation and action
        ob = ModelInput.from_ints(obs)
        ac_with_logprobs = TokensWithLogprobs(
            tokens=action_tokens,
            maybe_logprobs=logprobs,
        )

        # Reward: only final turn gets the episode reward
        is_final_turn = (turn_idx == len(actions) - 1)
        reward = trajectory.reward if is_final_turn else 0.0

        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=reward,
            episode_done=is_final_turn,
            metrics={
                "turn": turn_idx + 1,
                "turn_count": trajectory.turn_count,
            },
        )
        transitions.append(transition)

    final_ob = ob

    return TinkerTrajectory(transitions=transitions, final_ob=final_ob)


# Legacy aliases for backward compatibility
def spare_trajectory_to_tinker_trajectories(
    trajectory: SpareTrajectory,
    _learning_potential: float,
    _renderer=None,
) -> List[TinkerTrajectory]:
    """Legacy function - converts a single trajectory to a list."""
    role = trajectory.metadata.get("role", "actor")
    tinker_traj = spare_trajectory_to_tinker_trajectory(trajectory, role=role)
    return [tinker_traj]


def env_trajectory_to_tinker(
    trajectory: SpareTrajectory,
    _learning_potential: float,
    _renderer=None,
) -> TinkerTrajectory:
    """Legacy wrapper for _env_trajectory_to_tinker."""
    return _env_trajectory_to_tinker(trajectory)


def actor_trajectory_to_tinker(
    trajectory: SpareTrajectory,
    _learning_potential: float,
    _renderer=None,
) -> TinkerTrajectory:
    """Legacy wrapper for _actor_trajectory_to_tinker."""
    return _actor_trajectory_to_tinker(trajectory)
