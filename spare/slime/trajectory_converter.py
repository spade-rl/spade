"""Converter between SpareOrchestrator's Trajectory and Slime's Sample format.

This module provides utilities to convert between the orchestrator's episode Trajectory
format and Slime's Sample format used for training.
"""

import logging
from typing import List

from slime.agent.trajectory import (
    TurnRecord,
    fan_out_sample_segments,
    make_turn_segment,
    merge_turn_segments,
)
from slime.utils.types import Sample
from spare.core.types import Trajectory, TrajectoryStatus

logger = logging.getLogger(__name__)


def trajectory_to_slime_sample(
    trajectory: Trajectory,
    index: int = 0,
    role: str = "actor",
) -> Sample:
    """Convert an episode Trajectory to a Slime Sample.

    With the new episode-based Trajectory structure, this conversion is simpler
    since the fields align more closely with Slime's Sample format.

    Args:
        trajectory: Episode Trajectory from orchestrator
        index: Sample index
        role: Role identifier ("actor" or "environment")

    Returns:
        Sample object for Slime training
    """
    # Map SPARE TrajectoryStatus to Slime Sample.Status
    status_map = {
        TrajectoryStatus.COMPLETED: Sample.Status.COMPLETED,
        TrajectoryStatus.TRUNCATED: Sample.Status.TRUNCATED,
        TrajectoryStatus.ABORTED: Sample.Status.ABORTED,
        TrajectoryStatus.FAILED: Sample.Status.FAILED,
        TrajectoryStatus.TIMEOUT: Sample.Status.TRUNCATED,
        TrajectoryStatus.PENDING: Sample.Status.PENDING,
        TrajectoryStatus.RUNNING: Sample.Status.PENDING,
    }
    slime_status = status_map.get(trajectory.status, Sample.Status.PENDING)

    # Build metadata, ensuring sample_weight is preserved for loss weighting
    metadata = {
        "role": role,
        "turn_count": trajectory.turn_count,
        "sample_weight": trajectory.metadata.get("sample_weight", 1.0),
        **trajectory.metadata,
    }

    # Determine response_length from loss_mask if not set
    response_length = trajectory.response_length
    if response_length == 0 and trajectory.loss_mask:
        response_length = sum(trajectory.loss_mask)

    return Sample(
        index=index,
        prompt=trajectory.prompt,
        tokens=trajectory.tokens,
        response=trajectory.response,
        response_length=response_length,
        reward=trajectory.reward,
        loss_mask=trajectory.loss_mask,
        rollout_log_probs=trajectory.rollout_log_probs,
        status=slime_status,
        metadata=metadata,
        group_index=trajectory.metadata.get("group_index"),
    )


def fan_out_thinking_sample(sample: Sample, tokenizer) -> List[Sample]:
    """Fan an actor-thinking episode Sample out into one Sample per turn.

    Uses slime/agent's own trajectory machinery end to end, mirroring the
    coding_agent_rl consumer (``examples/coding_agent_rl/generate.py``):
    raw turn records -> ``TurnRecord`` -> ``make_turn_segment`` ->
    ``merge_turn_segments`` -> ``fan_out_sample_segments``.

    Segmentation granularity deviates from that consumer: ONE TurnSegment PER
    TURN instead of one segment per compaction chain. Qwen3's chat template
    strips prior turns' ``<think>`` blocks on re-render, so a whole-episode
    ``merge_turns`` would detect prefix drift inside every previous output and
    mask those outputs out (training only the final turn's thinking). Per-turn
    segments keep every turn's think+answer trainable: for turn k the segment
    is exactly [canonical stripped-history prompt (mask 0)] + [that turn's
    generated tokens incl. its think block (mask 1, rollout logprobs
    preserved)].

    Post-fan-out fix-ups (deviations from ``fan_out_sample_segments``
    defaults, both required by SPARE's precomputed reward pipeline):
    - Reward: the FULL episode reward is broadcast to every turn sample
      instead of the uniform reward/K split. SPARE precomputes the GRPO
      advantage per game BEFORE fan-out (normalize_rewards_per_game); a 1/K
      split would rescale advantages by each episode's turn count, distorting
      cross-play comparisons within a game group. The group-level loss
      denominator (slime's group_mask_sums) already prevents over-counting.
    - Status: the episode status (e.g. TRUNCATED) is restored on every
      sibling; ``write_segment_to_sample`` unconditionally sets COMPLETED,
      which would defeat spare_compact_filter and truncation metrics.
    - loss_mask: if the episode sample's mask was zeroed upstream (compact
      filter / inert padding), every turn sample's mask is zeroed too.

    All siblings share ``group_id`` (slime's contract for multi-sample
    training groups) so build_dp_schedule counts the episode as ONE group and
    the loss reducer averages it once.

    Args:
        sample: Episode-level Sample whose metadata carries "turn_records"
            (list of dicts with prompt_ids/output_ids/output_log_probs/
            finish_reason, produced by the orchestrator thinking path).
            The sample is MUTATED (reused as the first sibling), matching
            fan_out_sample_segments semantics.
        tokenizer: Tokenizer used by slime to decode segment responses.

    Returns:
        List of per-turn Samples (or [sample] unchanged when the sample has
        no turn records).
    """
    records = (sample.metadata or {}).get("turn_records")
    if not records:
        return [sample]

    episode_reward = float(sample.reward)
    episode_status = sample.status
    episode_mask_zeroed = bool(sample.loss_mask) and sum(sample.loss_mask) == 0

    turns = [
        TurnRecord(
            prompt_ids=r["prompt_ids"],
            output_ids=r["output_ids"],
            finish_reason=r["finish_reason"],
            output_log_probs=r["output_log_probs"],
        )
        for r in records
    ]
    segments = merge_turn_segments(
        [
            make_turn_segment([t], kind="turn", metadata={"turn_idx": i})
            for i, t in enumerate(turns)
        ]
    )
    if not segments:
        return [sample]

    fanned = fan_out_sample_segments(
        sample,
        segments,
        episode_reward,
        tokenizer,
    )
    for sub in fanned:
        sub.reward = episode_reward
        sub.status = episode_status
        if episode_mask_zeroed:
            sub.loss_mask = [0] * len(sub.loss_mask)
        # Drop the heavy raw token payload from training metadata.
        sub.metadata.pop("turn_records", None)
    return fanned


def trajectories_to_samples(
    trajectories: List[Trajectory],
    role: str = "actor",
    learning_potential: float = 0.0,
    base_index: int = 0,
) -> List[Sample]:
    """Convert a list of episode Trajectories to Slime Samples.

    Args:
        trajectories: List of episode Trajectory objects
        role: Role identifier ("actor" or "environment")
        learning_potential: Current learning potential value
        base_index: Starting index for samples

    Returns:
        List of Sample objects
    """
    samples = []
    for idx, traj in enumerate(trajectories):
        sample = trajectory_to_slime_sample(
            trajectory=traj,
            index=base_index + idx,
            role=role,
            learning_potential=learning_potential,
        )
        samples.append(sample)
    return samples


def trajectories_to_grouped_samples(
    env_trajectories: List[Trajectory],
    actor_trajectories: List[Trajectory],
    learning_potential: float = 0.0,
    base_index: int = 0,
) -> List[List[Sample]]:
    """Convert environment and actor episode trajectories to grouped Slime Samples.

    Slime expects list[list[Sample]] where inner lists are n_samples_per_prompt groups.
    For SPARE, we treat each sample as its own group.

    Args:
        env_trajectories: Environment generation episode trajectories
        actor_trajectories: Actor gameplay episode trajectories
        learning_potential: Current learning potential value
        base_index: Starting index for samples

    Returns:
        List of Sample groups (each group is a single-element list)
    """
    grouped_samples = []
    current_index = base_index

    # Convert environment trajectories
    for traj in env_trajectories:
        sample = trajectory_to_slime_sample(
            trajectory=traj,
            index=current_index,
            role="environment",
            learning_potential=learning_potential,
        )
        grouped_samples.append([sample])
        current_index += 1

    # Convert actor trajectories
    for traj in actor_trajectories:
        sample = trajectory_to_slime_sample(
            trajectory=traj,
            index=current_index,
            role="actor",
            learning_potential=learning_potential,
        )
        grouped_samples.append([sample])
        current_index += 1

    return grouped_samples
