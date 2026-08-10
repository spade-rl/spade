"""Characterization tests for the delayed proposer-reward variants.

The four ``recompute_delayed_env_rewards*`` functions are selected at runtime by
``args.spare_env_reward_variant`` in ``spare/slime/spare_rollout.py`` and mutate
slime ``Sample`` objects by duck typing (``Sample`` stays under ``TYPE_CHECKING``
in ``game_utils``). ``FakeSample`` below replicates exactly the contract they
touch:

* read  ``sample.metadata["game_file"]``    (default ``""``)
* read  ``sample.metadata["skill"]``        (default ``"unknown"``)
* read  ``sample.metadata["step_timeout"]`` (regret variant only)
* write ``sample.reward``
* write ``sample.metadata["regret" | "micro_lp" | "win_rate" |
  "frontier_penalty" | "plateau" | "blend_raw" | "proposer_masked"]``
  (blend variant only)

These are behavior locks, not derived ground truth: every asserted number is
what the code produces today, so a later refactor cannot silently change it.
Simple arithmetic is derived by hand in the comments; the intricate blend
components are snapshots verified for sanity against the documented formula.
"""

from typing import Any

import pytest

from spare.core.learning_potential import LearningPotential
from spare.core.utils.game_utils import (
    recompute_delayed_env_rewards,
    recompute_delayed_env_rewards_blend,
    recompute_delayed_env_rewards_micro_lp,
    recompute_delayed_env_rewards_regret,
)


class FakeSample:
    """Minimal stand-in for ``slime.utils.types.Sample`` (no slime import)."""

    def __init__(self, game_file: str, skill: str = "logic", **metadata: Any) -> None:
        self.reward: float | None = None
        self.metadata: dict[str, Any] = {
            "game_file": game_file,
            "skill": skill,
            **metadata,
        }


def _tracker(seed: float | None = None) -> LearningPotential:
    tracker = LearningPotential(gamma_fast=0.8, gamma_slow=0.2)
    if seed is not None:
        tracker.update(seed)
    return tracker


# =============================================================================
# recompute_delayed_env_rewards (game_utils.py:704) — baseline-distance variant
# =============================================================================


def test_delayed_baseline_rewards_are_centered_per_skill() -> None:
    # Two skills x two games. raw = |mean(accumulated) - mu_slow|; g4 has no
    # accumulated plays at all, which is the raw_reward = 0.0 branch (L773-774).
    trackers = {"logic": _tracker(0.5), "math": _tracker(0.2)}
    logic_a, logic_b = FakeSample("g1.py"), FakeSample("g2.py")
    math_a, math_b = FakeSample("g3.py", skill="math"), FakeSample("g4.py", skill="math")

    metrics = recompute_delayed_env_rewards(
        env_samples=[logic_a, logic_b, math_a, math_b],
        accumulated_game_rewards={
            "g1.py": [0.0, 0.4],  # mean 0.2 -> raw |0.2 - 0.5| = 0.3
            "g2.py": [1.0, 1.0],  # mean 1.0 -> raw |1.0 - 0.5| = 0.5
            "g3.py": [0.2, 0.6],  # mean 0.4 -> raw |0.4 - 0.2| = 0.2
        },
        learning_potentials=trackers,
        env_reward_scale=10.0,
    )

    # logic mean raw = 0.4, math mean raw = 0.1; reward = (raw - mean) * scale.
    assert logic_a.reward == pytest.approx(-1.0)
    assert logic_b.reward == pytest.approx(1.0)
    assert math_a.reward == pytest.approx(1.0)
    assert math_b.reward == pytest.approx(-1.0)
    # Per-skill centering: each skill's rewards sum to zero.
    assert logic_a.reward + logic_b.reward == pytest.approx(0.0, abs=1e-10)
    assert math_a.reward + math_b.reward == pytest.approx(0.0, abs=1e-10)

    assert set(metrics) == {
        "delayed_env/skill_logic/mean_raw_reward",
        "delayed_env/skill_logic/num_accumulated_plays",
        "delayed_env/skill_logic/mu_fast",
        "delayed_env/skill_logic/mu_slow",
        "delayed_env/skill_logic/signed_lp",
        "delayed_env/skill_logic/baseline",
        "delayed_env/skill_math/mean_raw_reward",
        "delayed_env/skill_math/num_accumulated_plays",
        "delayed_env/skill_math/mu_fast",
        "delayed_env/skill_math/mu_slow",
        "delayed_env/skill_math/signed_lp",
        "delayed_env/skill_math/baseline",
    }
    assert metrics["delayed_env/skill_logic/mean_raw_reward"] == pytest.approx(0.4)
    assert metrics["delayed_env/skill_logic/num_accumulated_plays"] == 4
    assert metrics["delayed_env/skill_logic/baseline"] == pytest.approx(0.5)
    assert metrics["delayed_env/skill_logic/signed_lp"] == pytest.approx(0.0)
    assert metrics["delayed_env/skill_math/mean_raw_reward"] == pytest.approx(0.1)
    assert metrics["delayed_env/skill_math/num_accumulated_plays"] == 2
    assert metrics["delayed_env/skill_math/mu_fast"] == pytest.approx(0.2)
    assert metrics["delayed_env/skill_math/mu_slow"] == pytest.approx(0.2)


def test_delayed_baseline_update_lp_advances_trackers_before_baselines() -> None:
    # update_lp=True folds every accumulated play into one LP update (L737-755)
    # BEFORE the baselines are read, so the recomputed raws see the new mu_slow.
    trackers = {"logic": _tracker(0.0)}
    high, low, mid = FakeSample("g1.py"), FakeSample("g2.py"), FakeSample("g3.py")

    metrics = recompute_delayed_env_rewards(
        env_samples=[high, low, mid],
        accumulated_game_rewards={
            "g1.py": [1.0, 1.0],
            "g2.py": [0.0, 0.0],
            "g3.py": [0.5, 0.5],
        },
        learning_potentials=trackers,
        env_reward_scale=10.0,
        update_lp=True,
    )

    # Accumulated skill mean = 0.5 -> mu_fast = 0.8*0.5 = 0.4, mu_slow = 0.2*0.5 = 0.1.
    assert trackers["logic"].mu_fast == pytest.approx(0.4)
    assert trackers["logic"].mu_slow == pytest.approx(0.1)
    assert len(trackers["logic"].history) == 2  # seed + the one delayed update

    # raws vs the fresh baseline 0.1: 0.9, 0.1, 0.4 -> mean 1.4/3.
    assert high.reward == pytest.approx(4.333333333333333)
    assert low.reward == pytest.approx(-3.6666666666666665)
    assert mid.reward == pytest.approx(-0.6666666666666666)
    assert high.reward + low.reward + mid.reward == pytest.approx(0.0, abs=1e-10)

    assert metrics["delayed_env/skill_logic/mean_raw_reward"] == pytest.approx(0.4666666666666667)
    assert metrics["delayed_env/skill_logic/num_accumulated_plays"] == 6
    assert metrics["delayed_env/skill_logic/mu_fast"] == pytest.approx(0.4)
    assert metrics["delayed_env/skill_logic/mu_slow"] == pytest.approx(0.1)
    assert metrics["delayed_env/skill_logic/baseline"] == pytest.approx(0.1)
    # signed_lp = (mu_fast - mu_slow) * 1/((0.8 - 0.2) * 2) = 0.3 / 1.2.
    assert metrics["delayed_env/skill_logic/signed_lp"] == pytest.approx(0.25)


def test_delayed_baseline_variance_reward_needs_more_than_one_play() -> None:
    # use_solver_variance_reward only fires when len(game_rewards) > 1 (L776).
    trackers = {"logic": _tracker(0.5)}
    two_plays, one_play = FakeSample("g1.py"), FakeSample("g2.py")

    metrics = recompute_delayed_env_rewards(
        env_samples=[two_plays, one_play],
        accumulated_game_rewards={"g1.py": [0.0, 1.0], "g2.py": [0.5]},
        learning_potentials=trackers,
        env_reward_scale=2.0,
        use_solver_variance_reward=True,
    )

    # Both games have mean 0.5 -> distance raw 0.0. g1 additionally earns
    # compute_variance_reward([0.0, 1.0]) = exp(0) = 1.0; g2 (one play) does not.
    assert two_plays.reward == pytest.approx(1.0)
    assert one_play.reward == pytest.approx(-1.0)
    assert metrics["delayed_env/skill_logic/mean_raw_reward"] == pytest.approx(0.5)
    assert metrics["delayed_env/skill_logic/num_accumulated_plays"] == 3


def test_delayed_baseline_skips_samples_whose_skill_has_no_tracker() -> None:
    # Unknown skill -> warned and dropped (L764-766): no reward, no metrics.
    trackers = {"logic": _tracker(0.5)}
    known, unknown = FakeSample("g1.py"), FakeSample("g2.py", skill="mystery")

    metrics = recompute_delayed_env_rewards(
        env_samples=[known, unknown],
        accumulated_game_rewards={"g1.py": [0.5], "g2.py": [1.0]},
        learning_potentials=trackers,
        env_reward_scale=10.0,
    )

    assert known.reward == pytest.approx(0.0)
    assert unknown.reward is None
    assert not any("mystery" in key for key in metrics)
    # update_lp defaults to False, so the tracker is untouched.
    assert len(trackers["logic"].history) == 1
    assert trackers["logic"].mu_slow == pytest.approx(0.5)


# =============================================================================
# recompute_delayed_env_rewards_micro_lp (game_utils.py:803)
# =============================================================================


def test_micro_lp_rewards_split_the_window_into_early_and_late_halves() -> None:
    trackers_free_scale = 4.0
    # logic: both games improve; math: one regresses (masked to 0) and one has
    # no accumulated data at all.
    logic_a, logic_b = FakeSample("g1.py"), FakeSample("g2.py")
    math_a, math_b = FakeSample("g3.py", skill="math"), FakeSample("g4.py", skill="math")

    metrics = recompute_delayed_env_rewards_micro_lp(
        env_samples=[logic_a, logic_b, math_a, math_b],
        accumulated_game_rewards={
            "g1.py": {0: [0.0, 0.0], 2: [1.0, 1.0]},  # micro_lp = 1.0
            "g2.py": {0: [0.25, 0.25], 1: [0.75, 0.75]},  # micro_lp = 0.5
            "g3.py": {0: [1.0, 1.0], 1: [0.0, 0.0]},  # max(0, -1.0) = 0.0
        },
        env_reward_scale=trackers_free_scale,
    )

    # logic raws [1.0, 0.5] -> mean 0.75; math raws [0.0, 0.0] -> mean 0.0.
    assert logic_a.reward == pytest.approx(1.0)
    assert logic_b.reward == pytest.approx(-1.0)
    assert math_a.reward == pytest.approx(0.0)
    assert math_b.reward == pytest.approx(0.0)
    assert logic_a.reward + logic_b.reward == pytest.approx(0.0, abs=1e-10)

    assert set(metrics) == {
        "delayed_env/skill_logic/mean_micro_lp",
        "delayed_env/skill_logic/early_mean",
        "delayed_env/skill_logic/late_mean",
        "delayed_env/skill_logic/improvement",
        "delayed_env/skill_logic/first_step_mean",
        "delayed_env/skill_logic/last_step_mean",
        "delayed_env/skill_logic/endpoint_diff",
        "delayed_env/skill_logic/num_games",
        "delayed_env/skill_logic/num_step_offsets",
        "delayed_env/skill_logic/num_accumulated_plays",
        "delayed_env/skill_math/mean_micro_lp",
        "delayed_env/skill_math/early_mean",
        "delayed_env/skill_math/late_mean",
        "delayed_env/skill_math/improvement",
        "delayed_env/skill_math/first_step_mean",
        "delayed_env/skill_math/last_step_mean",
        "delayed_env/skill_math/endpoint_diff",
        "delayed_env/skill_math/num_games",
        "delayed_env/skill_math/num_step_offsets",
        "delayed_env/skill_math/num_accumulated_plays",
        "delayed_env/avg_early_mean",
        "delayed_env/avg_late_mean",
        "delayed_env/avg_improvement",
        "delayed_env/avg_endpoint_diff",
        "delayed_env/improving_skill_pct",
    }
    assert metrics["delayed_env/skill_logic/mean_micro_lp"] == pytest.approx(0.75)
    assert metrics["delayed_env/skill_logic/early_mean"] == pytest.approx(0.125)
    assert metrics["delayed_env/skill_logic/late_mean"] == pytest.approx(0.875)
    assert metrics["delayed_env/skill_logic/improvement"] == pytest.approx(0.75)
    assert metrics["delayed_env/skill_logic/first_step_mean"] == pytest.approx(0.125)
    assert metrics["delayed_env/skill_logic/last_step_mean"] == pytest.approx(0.875)
    assert metrics["delayed_env/skill_logic/endpoint_diff"] == pytest.approx(0.75)
    assert metrics["delayed_env/skill_logic/num_games"] == 2
    assert metrics["delayed_env/skill_logic/num_step_offsets"] == pytest.approx(2.0)
    assert metrics["delayed_env/skill_logic/num_accumulated_plays"] == 8

    # math: the regressing game keeps its true early/late means in the metrics
    # even though its micro_lp reward is floored at 0.
    assert metrics["delayed_env/skill_math/mean_micro_lp"] == pytest.approx(0.0)
    assert metrics["delayed_env/skill_math/early_mean"] == pytest.approx(0.5)
    assert metrics["delayed_env/skill_math/late_mean"] == pytest.approx(0.0)
    assert metrics["delayed_env/skill_math/improvement"] == pytest.approx(-0.5)
    assert metrics["delayed_env/skill_math/endpoint_diff"] == pytest.approx(-0.5)
    assert metrics["delayed_env/skill_math/num_step_offsets"] == pytest.approx(1.0)
    assert metrics["delayed_env/skill_math/num_accumulated_plays"] == 4

    assert metrics["delayed_env/avg_early_mean"] == pytest.approx(0.3125)
    assert metrics["delayed_env/avg_late_mean"] == pytest.approx(0.4375)
    assert metrics["delayed_env/avg_improvement"] == pytest.approx(0.125)
    assert metrics["delayed_env/avg_endpoint_diff"] == pytest.approx(0.125)
    assert metrics["delayed_env/improving_skill_pct"] == pytest.approx(50.0)


def test_micro_lp_odd_window_puts_the_middle_step_in_the_late_half() -> None:
    # mid = len // 2, so 3 offsets split 1 early / 2 late (L848-850).
    sample = FakeSample("g1.py")

    metrics = recompute_delayed_env_rewards_micro_lp(
        env_samples=[sample],
        accumulated_game_rewards={"g1.py": {0: [0.0], 1: [0.5], 2: [1.0]}},
        env_reward_scale=10.0,
    )

    # early = [0.0] -> 0.0; late = [0.5, 1.0] -> 0.75; micro_lp = 0.75.
    # A lone game is its own skill mean, so the centered reward is exactly 0.
    assert sample.reward == pytest.approx(0.0)
    assert metrics["delayed_env/skill_logic/mean_micro_lp"] == pytest.approx(0.75)
    assert metrics["delayed_env/skill_logic/early_mean"] == pytest.approx(0.0)
    assert metrics["delayed_env/skill_logic/late_mean"] == pytest.approx(0.75)
    assert metrics["delayed_env/skill_logic/first_step_mean"] == pytest.approx(0.0)
    assert metrics["delayed_env/skill_logic/last_step_mean"] == pytest.approx(1.0)
    assert metrics["delayed_env/skill_logic/endpoint_diff"] == pytest.approx(1.0)
    assert metrics["delayed_env/skill_logic/num_step_offsets"] == pytest.approx(3.0)
    assert metrics["delayed_env/improving_skill_pct"] == pytest.approx(100.0)


def test_micro_lp_needs_at_least_two_step_offsets() -> None:
    # Fewer than 2 temporal steps zeroes every per-game stat (L839-845).
    single_offset, no_data = FakeSample("g1.py"), FakeSample("g2.py")

    metrics = recompute_delayed_env_rewards_micro_lp(
        env_samples=[single_offset, no_data],
        accumulated_game_rewards={"g1.py": {0: [0.7]}},
        env_reward_scale=10.0,
    )

    assert single_offset.reward == pytest.approx(0.0)
    assert no_data.reward == pytest.approx(0.0)
    assert metrics["delayed_env/skill_logic/mean_micro_lp"] == pytest.approx(0.0)
    assert metrics["delayed_env/skill_logic/early_mean"] == pytest.approx(0.0)
    assert metrics["delayed_env/skill_logic/late_mean"] == pytest.approx(0.0)
    assert metrics["delayed_env/skill_logic/num_games"] == 2
    assert metrics["delayed_env/skill_logic/num_step_offsets"] == pytest.approx(0.5)
    assert metrics["delayed_env/skill_logic/num_accumulated_plays"] == 1
    assert metrics["delayed_env/improving_skill_pct"] == pytest.approx(0.0)


# =============================================================================
# recompute_delayed_env_rewards_regret (game_utils.py:921)
# =============================================================================


def test_regret_floors_negatives_and_penalizes_step_timeouts() -> None:
    helpful = FakeSample("g1.py")  # regret 0.8 - 0.2 = 0.6
    unhelpful = FakeSample("g2.py")  # regret 0.1 - 0.5 = -0.4 -> floored to 0.0
    timed_out = FakeSample("g3.py", skill="math", step_timeout=True)
    no_data = FakeSample("g4.py", skill="math")  # no hint, no plays -> regret 0.0

    metrics = recompute_delayed_env_rewards_regret(
        env_samples=[helpful, unhelpful, timed_out, no_data],
        accumulated_game_rewards={"g1.py": [0.2, 0.2], "g2.py": [0.5, 0.5]},
        cached_hint_rewards={"g1.py": 0.8, "g2.py": 0.1},
        env_reward_scale=10.0,
    )

    # logic raws [0.6, 0.0] -> mean 0.3; math raws [-0.5, 0.0] -> mean -0.25.
    assert helpful.reward == pytest.approx(3.0)
    assert unhelpful.reward == pytest.approx(-3.0)
    assert timed_out.reward == pytest.approx(-2.5)  # STEP_TIMEOUT_PENALTY = -0.5
    assert no_data.reward == pytest.approx(2.5)
    assert helpful.reward + unhelpful.reward == pytest.approx(0.0, abs=1e-10)
    assert timed_out.reward + no_data.reward == pytest.approx(0.0, abs=1e-10)

    assert set(metrics) == {
        "delayed_env/skill_logic/mean_regret",
        "delayed_env/skill_logic/num_accumulated_plays",
        "delayed_env/skill_math/mean_regret",
        "delayed_env/skill_math/num_accumulated_plays",
        "delayed_env/proposer_floored_frac",
        "delayed_env/proposer_step_timeout_frac",
        "delayed_env/proposer_masked_negative_frac",
    }
    assert metrics["delayed_env/skill_logic/mean_regret"] == pytest.approx(0.3)
    assert metrics["delayed_env/skill_logic/num_accumulated_plays"] == 4
    assert metrics["delayed_env/skill_math/mean_regret"] == pytest.approx(-0.25)
    assert metrics["delayed_env/skill_math/num_accumulated_plays"] == 0
    assert metrics["delayed_env/proposer_floored_frac"] == pytest.approx(0.25)
    assert metrics["delayed_env/proposer_step_timeout_frac"] == pytest.approx(0.25)
    # Historical alias: same value as the floored fraction.
    assert metrics["delayed_env/proposer_masked_negative_frac"] == pytest.approx(0.25)


def test_regret_reports_zero_fractions_when_every_game_is_positive() -> None:
    strong, weak = FakeSample("g1.py"), FakeSample("g2.py")

    metrics = recompute_delayed_env_rewards_regret(
        env_samples=[strong, weak],
        accumulated_game_rewards={"g1.py": [0.1, 0.1], "g2.py": [0.3]},
        cached_hint_rewards={"g1.py": 0.9, "g2.py": 0.5},
        env_reward_scale=2.0,
    )

    # raws [0.8, 0.2] -> mean 0.5.
    assert strong.reward == pytest.approx(0.6)
    assert weak.reward == pytest.approx(-0.6)
    assert metrics["delayed_env/skill_logic/mean_regret"] == pytest.approx(0.5)
    assert metrics["delayed_env/skill_logic/num_accumulated_plays"] == 3
    assert metrics["delayed_env/proposer_floored_frac"] == pytest.approx(0.0)
    assert metrics["delayed_env/proposer_step_timeout_frac"] == pytest.approx(0.0)
    assert metrics["delayed_env/proposer_masked_negative_frac"] == pytest.approx(0.0)


# =============================================================================
# recompute_delayed_env_rewards_blend (game_utils.py:994)
# =============================================================================


def test_blend_default_weights_mask_negative_games() -> None:
    # Default blend: 0.5*(regret/0.15) + 0.5*(micro_lp/0.10), signed slope
    # micro-LP, no frontier/plateau. Snapshot values below are behavior locks.
    teachable = FakeSample("g1.py")  # regret 0.3, micro_lp 0.6 -> blend 4.0
    saturated = FakeSample("g2.py")  # regret 0.0, micro_lp 0.0 -> blend 0.0
    harmful = FakeSample("g3.py", skill="math")  # blend -3.666... -> masked
    no_window = FakeSample("g4.py", skill="math")  # no plays -> blend 0.5

    metrics = recompute_delayed_env_rewards_blend(
        env_samples=[teachable, saturated, harmful, no_window],
        accumulated_game_rewards={
            "g1.py": {0: [0.0, 0.0], 1: [0.6, 0.6]},
            "g2.py": {0: [1.0, 1.0], 1: [1.0, 1.0]},
            "g3.py": {0: [0.5, 0.5], 1: [0.1, 0.1]},
        },
        cached_hint_rewards={"g1.py": 0.3, "g2.py": 1.0, "g4.py": 0.15},
        env_reward_scale=10.0,
    )

    # Per-sample component breadcrumbs written before the mask check (L1145-1150).
    assert teachable.metadata["regret"] == pytest.approx(0.3)
    assert teachable.metadata["micro_lp"] == pytest.approx(0.6)
    assert teachable.metadata["win_rate"] == pytest.approx(0.3)
    assert teachable.metadata["frontier_penalty"] == pytest.approx(0.04)
    assert teachable.metadata["plateau"] == pytest.approx(0.6)
    assert teachable.metadata["blend_raw"] == pytest.approx(4.0)

    assert harmful.metadata["blend_raw"] == pytest.approx(-3.6666666666666665)
    assert harmful.metadata["proposer_masked"] is True
    assert harmful.reward is None  # masked samples are never scored
    assert "proposer_masked" not in teachable.metadata

    # logic blends [4.0, 0.0] -> mean 2.0; math keeps only g4 -> mean 0.5.
    assert teachable.reward == pytest.approx(20.0)
    assert saturated.reward == pytest.approx(-20.0)
    assert no_window.reward == pytest.approx(0.0)
    assert teachable.reward + saturated.reward == pytest.approx(0.0, abs=1e-10)

    assert metrics["delayed_env/skill_logic/mean_blend"] == pytest.approx(2.0)
    assert metrics["delayed_env/skill_logic/num_games"] == 2
    assert metrics["delayed_env/skill_math/mean_blend"] == pytest.approx(0.5)
    assert metrics["delayed_env/skill_math/num_games"] == 1

    assert metrics["delayed_env/blend_regret_mean"] == pytest.approx(-0.0125)
    # Population std snapshots (behavior lock, pinned to 1e-10).
    assert metrics["delayed_env/blend_regret_std"] == pytest.approx(
        0.3007802353878991, abs=1e-10
    )
    assert metrics["delayed_env/blend_micro_lp_mean"] == pytest.approx(0.05)
    assert metrics["delayed_env/blend_micro_lp_std"] == pytest.approx(
        0.3570714214271424, abs=1e-10
    )
    assert metrics["delayed_env/blend_win_rate_mean"] == pytest.approx(0.4)
    assert metrics["delayed_env/blend_frontier_penalty_mean"] == pytest.approx(0.145)
    assert metrics["delayed_env/comp_regret_norm"] == pytest.approx(0.7916666666666666)
    assert metrics["delayed_env/comp_micro_lp_norm"] == pytest.approx(1.25)
    assert metrics["delayed_env/comp_frontier_norm"] == pytest.approx(0.0)
    assert metrics["delayed_env/blend_plateau_mean"] == pytest.approx(0.3)
    assert metrics["delayed_env/comp_plateau_norm"] == pytest.approx(0.0)
    assert metrics["delayed_env/blend_win_rate_var_mean"] == pytest.approx(0.0325)
    # Only games with >= 2 offsets contribute early/late means (g4 is excluded).
    assert metrics["delayed_env/blend_early_mean"] == pytest.approx(0.5)
    assert metrics["delayed_env/blend_late_mean"] == pytest.approx(0.5666666666666667)
    assert metrics["delayed_env/proposer_masked_negative_frac"] == pytest.approx(0.25)


def test_blend_plateau_config_keeps_every_game_non_negative() -> None:
    # Intended plateau config: plateau + floored regret, no micro-LP/frontier.
    # regret_floor clamps regret/regret_scale into [0, 1] (L1135-1136), so the
    # blend can never go negative and nothing is masked.
    easy = FakeSample("g1.py")  # win 1.0 -> plateau 0.0; regret -1.0 -> 0.0
    in_band = FakeSample("g2.py")  # win 0.5 -> plateau 1.0; regret 0.3 -> clip 1.0
    hard = FakeSample("g3.py", skill="math")  # win 0.2 -> plateau 0.2; regret 0.5

    metrics = recompute_delayed_env_rewards_blend(
        env_samples=[easy, in_band, hard],
        accumulated_game_rewards={
            "g1.py": {0: [1.0, 1.0], 1: [1.0, 1.0]},
            "g2.py": {0: [0.5, 0.5], 1: [0.5, 0.5]},
            "g3.py": {0: [0.2, 0.2], 1: [0.2, 0.2]},
        },
        cached_hint_rewards={"g2.py": 0.8, "g3.py": 0.275},
        env_reward_scale=1.0,
        regret_weight=1.0,
        micro_lp_weight=0.0,
        plateau_weight=1.0,
        regret_floor=True,
    )

    assert easy.metadata["plateau"] == pytest.approx(0.0)
    assert in_band.metadata["plateau"] == pytest.approx(1.0)
    assert hard.metadata["plateau"] == pytest.approx(0.2)
    assert easy.metadata["blend_raw"] == pytest.approx(0.0)  # floored regret + plateau 0
    assert in_band.metadata["blend_raw"] == pytest.approx(2.0)  # clipped regret 1.0 + 1.0
    assert hard.metadata["blend_raw"] == pytest.approx(0.7)  # 0.5 + 0.2

    # logic blends [0.0, 2.0] -> mean 1.0; math has a single game -> 0.
    assert easy.reward == pytest.approx(-1.0)
    assert in_band.reward == pytest.approx(1.0)
    assert hard.reward == pytest.approx(0.0)
    assert metrics["delayed_env/skill_logic/mean_blend"] == pytest.approx(1.0)
    assert metrics["delayed_env/skill_math/mean_blend"] == pytest.approx(0.7)
    assert metrics["delayed_env/proposer_masked_negative_frac"] == pytest.approx(0.0)
    assert not any("proposer_masked" in sample.metadata for sample in (easy, in_band, hard))
    assert metrics["delayed_env/blend_plateau_mean"] == pytest.approx(0.4)
    assert metrics["delayed_env/comp_plateau_norm"] == pytest.approx(0.4)


def test_blend_frontier_penalty_masks_games_that_drift_off_fifty_percent() -> None:
    # frontier_weight isolated: blend = -(win_rate - 0.5)^2 / frontier_scale.
    eased = FakeSample("g1.py")  # win 1.0 -> penalty 0.25 -> blend -3.125 -> masked
    centered = FakeSample("g2.py")  # win 0.5 -> penalty 0.0 -> blend 0.0 -> kept

    metrics = recompute_delayed_env_rewards_blend(
        env_samples=[eased, centered],
        accumulated_game_rewards={
            "g1.py": {0: [1.0, 1.0], 1: [1.0, 1.0]},
            "g2.py": {0: [0.5, 0.5], 1: [0.5, 0.5]},
        },
        cached_hint_rewards={},
        env_reward_scale=10.0,
        regret_weight=0.0,
        micro_lp_weight=0.0,
        frontier_weight=1.0,
    )

    assert eased.metadata["frontier_penalty"] == pytest.approx(0.25)
    assert eased.metadata["blend_raw"] == pytest.approx(-3.125)
    assert eased.metadata["proposer_masked"] is True
    assert eased.reward is None
    assert centered.metadata["frontier_penalty"] == pytest.approx(0.0)
    assert centered.reward == pytest.approx(0.0)

    assert metrics["delayed_env/skill_logic/num_games"] == 1
    assert metrics["delayed_env/blend_frontier_penalty_mean"] == pytest.approx(0.125)
    assert metrics["delayed_env/comp_frontier_norm"] == pytest.approx(1.5625)
    assert metrics["delayed_env/comp_regret_norm"] == pytest.approx(0.0)
    assert metrics["delayed_env/proposer_masked_negative_frac"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("micro_lp_slope", "expected"),
    [
        # Slope estimator: perfectly linear window, fitted end-minus-start
        # = slope 0.5 * (3 - 1) = 1.0.
        (True, 1.0),
        # Two-bucket estimator: mean([0.5, 1.0]) - mean([0.0]) = 0.75.
        (False, 0.75),
    ],
)
def test_blend_micro_lp_estimator_switch(micro_lp_slope: bool, expected: float) -> None:
    sample = FakeSample("g1.py")

    recompute_delayed_env_rewards_blend(
        env_samples=[sample],
        accumulated_game_rewards={"g1.py": {0: [0.0], 1: [0.5], 2: [1.0]}},
        cached_hint_rewards={},
        env_reward_scale=1.0,
        regret_weight=0.0,
        micro_lp_weight=1.0,
        micro_lp_scale=1.0,
        micro_lp_slope=micro_lp_slope,
    )

    assert sample.metadata["micro_lp"] == pytest.approx(expected)


def test_blend_slope_estimator_weights_steps_by_rollout_count() -> None:
    # Weighted least squares with w = sqrt(n) minimizes sum n_i * residual^2.
    # xs = [0, 1, 2], ys = [1.0, 0.0, 1.0], n = [4, 1, 1]:
    #   x_bar = 3/6 = 0.5, y_bar = 5/6
    #   Sxy = -0.5, Sxx = 3.5  ->  slope = -1/7
    #   micro_lp = slope * (3 - 1) = -2/7
    # The unweighted two-bucket estimator would instead give 0.5 - 1.0 = -0.5.
    sample = FakeSample("g1.py")

    recompute_delayed_env_rewards_blend(
        env_samples=[sample],
        accumulated_game_rewards={"g1.py": {0: [1.0, 1.0, 1.0, 1.0], 1: [0.0], 2: [1.0]}},
        cached_hint_rewards={},
        env_reward_scale=1.0,
        regret_weight=0.0,
        micro_lp_weight=1.0,
        micro_lp_scale=1.0,
    )

    assert sample.metadata["micro_lp"] == pytest.approx(-0.2857142857142857)
    # Negative blend -> masked before centering.
    assert sample.metadata["proposer_masked"] is True
    assert sample.reward is None


@pytest.mark.parametrize(
    ("micro_lp_signed", "expected_micro_lp", "expected_masked"),
    [
        (True, -1.0, True),  # signed: a regressing game is a negative signal
        (False, 0.0, False),  # unsigned: floored to 0, so nothing is masked
    ],
)
def test_blend_unsigned_micro_lp_floors_regressions(
    micro_lp_signed: bool,
    expected_micro_lp: float,
    expected_masked: bool,
) -> None:
    sample = FakeSample("g1.py")

    metrics = recompute_delayed_env_rewards_blend(
        env_samples=[sample],
        accumulated_game_rewards={"g1.py": {0: [1.0, 1.0], 1: [0.0, 0.0]}},
        cached_hint_rewards={},
        env_reward_scale=1.0,
        regret_weight=0.0,
        micro_lp_weight=1.0,
        micro_lp_scale=1.0,
        micro_lp_signed=micro_lp_signed,
    )

    assert sample.metadata["micro_lp"] == pytest.approx(expected_micro_lp)
    assert sample.metadata.get("proposer_masked", False) is expected_masked
    assert metrics["delayed_env/proposer_masked_negative_frac"] == pytest.approx(
        1.0 if expected_masked else 0.0
    )


# =============================================================================
# Shared empty-input contract
# =============================================================================


def test_every_variant_returns_empty_metrics_without_samples() -> None:
    assert recompute_delayed_env_rewards([], {}, {}, 10.0) == {}
    assert recompute_delayed_env_rewards_micro_lp([], {}, 10.0) == {}
    assert recompute_delayed_env_rewards_regret([], {}, {}, 10.0) == {}
    assert recompute_delayed_env_rewards_blend([], {}, {}, 10.0) == {}


def test_blend_masking_boundary_is_strictly_negative() -> None:
    """Pin the mask threshold at exactly ``blend < 0``: an infinitesimally
    negative blend must mask (the blend == 0.0 case is pinned not-masked by
    test_blend_default_weights_mask_negative_games). Guards against loosening
    the comparison to ``blend < -eps``."""
    barely_negative = FakeSample("g_tiny.py")
    # micro_lp is exactly 0 (flat window); regret = cached - mean(offset 0)
    # = -1.5e-13, so blend = 0.5 * (-1.5e-13 / 0.15) ~ -5e-13.
    metrics = recompute_delayed_env_rewards_blend(
        env_samples=[barely_negative],
        accumulated_game_rewards={"g_tiny.py": {0: [0.5, 0.5], 1: [0.5, 0.5]}},
        cached_hint_rewards={"g_tiny.py": 0.5 - 1.5e-13},
        env_reward_scale=10.0,
    )
    assert -1e-9 < barely_negative.metadata["blend_raw"] < 0
    assert barely_negative.metadata["proposer_masked"] is True
    assert barely_negative.reward is None
    assert metrics["delayed_env/proposer_masked_negative_frac"] == pytest.approx(1.0)
