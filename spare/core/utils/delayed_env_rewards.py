"""Delayed proposer reward recomputation over accumulated actor plays.

Every variant follows one pipeline: bucket the buffered env samples by skill,
score each with a variant-specific raw signal, then center per skill and scale
onto ``sample.reward``. Only the raw signal (and the metrics justifying it)
differs; the shared tail is ``_apply_centered_rewards``.
"""

from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import List, Dict, Tuple, TYPE_CHECKING

import numpy as np
from spare.core.learning_potential import LearningPotential
from spare.core.utils.env_rewards import _mean_or_zero
from spare.core.utils.rewards import (
    STEP_TIMEOUT_PENALTY,
    compute_variance_reward,
    plateau_reward,
)

if TYPE_CHECKING:
    from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Per-skill buckets of (env sample, raw proposer signal).
_SkillItems = Dict[str, List[Tuple["Sample", float]]]


# -----------------------------------------------------------------------------
# Shared pipeline pieces
# -----------------------------------------------------------------------------


def _game_file(sample: "Sample") -> str:
    return sample.metadata.get("game_file", "")


def _skill(sample: "Sample") -> str:
    return sample.metadata.get("skill", "unknown")


def _apply_centered_rewards(
    skill_items: _SkillItems,
    env_reward_scale: float,
    mean_metric: str,
) -> Dict[str, float]:
    """Center each skill's raw signal on its own mean and scale it onto the samples.

    Shared tail of all four variants. Mean subtraction only, no std division (a
    small-batch std blows the rewards up), and each skill logs the raw mean it
    was centered on as ``delayed_env/skill_<skill>/<mean_metric>``.
    """
    metrics: Dict[str, float] = {}
    for skill, items in skill_items.items():
        skill_mean = float(np.mean([raw_reward for _, raw_reward in items]))
        for sample, raw_reward in items:
            sample.reward = (raw_reward - skill_mean) * env_reward_scale
        metrics[f"delayed_env/skill_{skill}/{mean_metric}"] = skill_mean
    return metrics


def _accumulated_play_count(
    accumulated_game_rewards: Dict[str, List[float]],
    items: List[Tuple["Sample", float]],
) -> int:
    """Actor plays accumulated across the delay window behind one skill's games."""
    return sum(
        len(accumulated_game_rewards.get(_game_file(sample), [])) for sample, _ in items
    )


def _early_late_means(
    step_rewards: Dict[int, List[float]],
    offsets: List[int],
) -> Tuple[float, float]:
    """Mean actor reward over the early vs late half of a step-keyed window.

    ``mid = len // 2``, so an odd window's middle step lands in the late half.
    """
    mid = len(offsets) // 2
    early = [r for offset in offsets[:mid] for r in step_rewards[offset]]
    late = [r for offset in offsets[mid:] for r in step_rewards[offset]]
    return _mean_or_zero(early), _mean_or_zero(late)


# -----------------------------------------------------------------------------
# Variants, selected at runtime by --spare-env-reward-variant
# -----------------------------------------------------------------------------


def recompute_delayed_env_rewards(
    env_samples: List["Sample"],
    accumulated_game_rewards: Dict[str, List[float]],
    learning_potentials: Dict[str, LearningPotential],
    env_reward_scale: float,
    use_solver_variance_reward: bool = False,
    update_lp: bool = False,
) -> Dict[str, float]:
    """Recompute env sample rewards using accumulated actor data.

    Called at delayed training time when we have multiple rollouts of actor
    gameplay data on the same games.

    Raw signal: |accumulated win rate - mu_slow| (+ optional solver variance).

    When update_lp=True (delayed LP mode), this function also updates the LP
    trackers using the full accumulated data — the per-rollout LP updates are
    skipped (via skip_lp_update on SpareConfig) so this is the only place
    LP gets updated.

    Args:
        env_samples: Buffered Slime Sample objects with env trajectory data
        accumulated_game_rewards: {game_file: [raw_rewards]} across delay window
        learning_potentials: Per-skill LP trackers
        env_reward_scale: Scale factor for centered env rewards
        use_solver_variance_reward: Whether to add variance reward component
        update_lp: If True, update LP trackers with accumulated skill rewards

    Returns:
        Dict of metrics for logging
    """
    if not env_samples:
        return {}

    # Collect per-skill mean rewards from accumulated data for LP update
    if update_lp:
        skill_accumulated_rewards: Dict[str, List[float]] = defaultdict(list)
        for sample in env_samples:
            skill_accumulated_rewards[_skill(sample)].extend(
                accumulated_game_rewards.get(_game_file(sample), [])
            )

        # Update LP trackers with accumulated data before computing baselines
        for skill, rewards in skill_accumulated_rewards.items():
            if skill in learning_potentials and rewards:
                avg_reward = float(np.mean(rewards))
                learning_potentials[skill].update(avg_reward)
                logger.info(
                    f"[DELAYED_LP] Updated LP for skill '{skill}': "
                    f"mu_fast={learning_potentials[skill].mu_fast:.4f}, "
                    f"mu_slow={learning_potentials[skill].mu_slow:.4f}, "
                    f"avg_reward={avg_reward:.4f} (from {len(rewards)} plays)"
                )

    # Group by skill, compute raw rewards
    skill_data: _SkillItems = defaultdict(list)

    for sample in env_samples:
        skill = _skill(sample)

        if skill not in learning_potentials:
            logger.warning(f"[DELAYED_RECOMPUTE] Skill '{skill}' not in learning_potentials, skipping")
            continue

        baseline = learning_potentials[skill].get_baseline()
        game_rewards = accumulated_game_rewards.get(_game_file(sample), [])

        if game_rewards:
            raw_reward = abs(float(np.mean(game_rewards)) - baseline)
        else:
            raw_reward = 0.0

        if use_solver_variance_reward and len(game_rewards) > 1:
            raw_reward += compute_variance_reward(game_rewards)

        skill_data[skill].append((sample, raw_reward))

    metrics = _apply_centered_rewards(skill_data, env_reward_scale, "mean_raw_reward")
    for skill, items in skill_data.items():
        lp = learning_potentials[skill]
        prefix = f"delayed_env/skill_{skill}"
        metrics[f"{prefix}/num_accumulated_plays"] = _accumulated_play_count(
            accumulated_game_rewards, items
        )
        # Log LP state (always available, but especially useful when update_lp=True)
        metrics[f"{prefix}/mu_fast"] = lp.mu_fast if lp.mu_fast is not None else 0.0
        metrics[f"{prefix}/mu_slow"] = lp.mu_slow if lp.mu_slow is not None else 0.0
        metrics[f"{prefix}/signed_lp"] = lp.get_signed_potential()
        metrics[f"{prefix}/baseline"] = lp.get_baseline()

    return metrics


@dataclass
class _MicroLPGame:
    """One env sample's micro-LP signal plus the window stats logged beside it."""

    sample: "Sample"
    micro_lp: float
    early_mean: float
    late_mean: float
    first_step_mean: float
    last_step_mean: float


def recompute_delayed_env_rewards_micro_lp(
    env_samples: List["Sample"],
    accumulated_game_rewards: Dict[str, Dict[int, List[float]]],
    env_reward_scale: float,
) -> Dict[str, float]:
    """Recompute env rewards using micro learning potential within the delay window.

    Measures actual actor improvement on each game: did the actor get better
    at this game over the delay window? This avoids stale baselines and external
    proxies by directly comparing early vs late performance.

    Formula: micro_lp = max(0, mean(late_rewards) - mean(early_rewards))
    Then center per skill and scale (same pipeline as other variants).

    Args:
        env_samples: Buffered Slime Sample objects with env trajectory data
        accumulated_game_rewards: {game_file: {step_offset: [raw_rewards]}}
        env_reward_scale: Scale factor for centered env rewards

    Returns:
        Dict of metrics for logging
    """
    if not env_samples:
        return {}

    skill_games: Dict[str, List[_MicroLPGame]] = defaultdict(list)

    for sample in env_samples:
        step_rewards = accumulated_game_rewards.get(_game_file(sample), {})
        offsets = sorted(step_rewards.keys())

        if len(offsets) < 2:
            # Can't compute improvement with fewer than 2 temporal steps
            game = _MicroLPGame(sample, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            early_mean, late_mean = _early_late_means(step_rewards, offsets)
            game = _MicroLPGame(
                sample=sample,
                micro_lp=max(0.0, late_mean - early_mean),
                early_mean=early_mean,
                late_mean=late_mean,
                # First and last step means (exact endpoints of the delay window)
                first_step_mean=float(np.mean(step_rewards[offsets[0]])),
                last_step_mean=float(np.mean(step_rewards[offsets[-1]])),
            )

        skill_games[_skill(sample)].append(game)

    metrics = _apply_centered_rewards(
        {
            skill: [(game.sample, game.micro_lp) for game in games]
            for skill, games in skill_games.items()
        },
        env_reward_scale,
        "mean_micro_lp",
    )

    all_early_means: List[float] = []
    all_late_means: List[float] = []
    all_improvements: List[float] = []
    all_endpoint_diffs: List[float] = []

    for skill, games in skill_games.items():
        skill_early = float(np.mean([game.early_mean for game in games]))
        skill_late = float(np.mean([game.late_mean for game in games]))
        skill_first = float(np.mean([game.first_step_mean for game in games]))
        skill_last = float(np.mean([game.last_step_mean for game in games]))
        skill_improvement = skill_late - skill_early
        skill_endpoint_diff = skill_last - skill_first

        prefix = f"delayed_env/skill_{skill}"
        metrics[f"{prefix}/early_mean"] = skill_early
        metrics[f"{prefix}/late_mean"] = skill_late
        metrics[f"{prefix}/improvement"] = skill_improvement
        metrics[f"{prefix}/first_step_mean"] = skill_first
        metrics[f"{prefix}/last_step_mean"] = skill_last
        metrics[f"{prefix}/endpoint_diff"] = skill_endpoint_diff
        metrics[f"{prefix}/num_games"] = len(games)
        metrics[f"{prefix}/num_step_offsets"] = float(np.mean([
            len(accumulated_game_rewards.get(_game_file(game.sample), {}))
            for game in games
        ]))
        metrics[f"{prefix}/num_accumulated_plays"] = sum(
            sum(len(r) for r in accumulated_game_rewards.get(_game_file(game.sample), {}).values())
            for game in games
        )

        all_early_means.append(skill_early)
        all_late_means.append(skill_late)
        all_improvements.append(skill_improvement)
        all_endpoint_diffs.append(skill_endpoint_diff)

    # Aggregate across skills
    if all_improvements:
        metrics["delayed_env/avg_early_mean"] = float(np.mean(all_early_means))
        metrics["delayed_env/avg_late_mean"] = float(np.mean(all_late_means))
        metrics["delayed_env/avg_improvement"] = float(np.mean(all_improvements))
        metrics["delayed_env/avg_endpoint_diff"] = float(np.mean(all_endpoint_diffs))
        metrics["delayed_env/improving_skill_pct"] = (
            sum(1 for x in all_improvements if x > 0) / len(all_improvements) * 100.0
        )

    return metrics


def recompute_delayed_env_rewards_regret(
    env_samples: List["Sample"],
    accumulated_game_rewards: Dict[str, List[float]],
    cached_hint_rewards: Dict[str, float],
    env_reward_scale: float,
) -> Dict[str, float]:
    """Recompute regret-based env rewards using accumulated R_no_hint.

    Regret = cached_R_hint - mean(accumulated_R_no_hint). The R_hint values
    are cached from the regen step (hint plays don't change), while R_no_hint
    improves with more accumulated actor plays.

    Args:
        env_samples: Buffered Slime Sample objects with env trajectory data
        accumulated_game_rewards: {game_file: [raw_rewards]} from actor plays
        cached_hint_rewards: {game_file: R_hint} from regen step
        env_reward_scale: Scale factor for centered env rewards

    Returns:
        Dict of metrics for logging
    """
    if not env_samples:
        return {}

    # Group by skill, compute regret
    skill_data: _SkillItems = defaultdict(list)
    floored_n = 0
    step_timeout_n = 0

    for sample in env_samples:
        skill = _skill(sample)

        if sample.metadata.get("step_timeout"):
            # A generated-code timeout overrides regret with a bounded penalty.
            skill_data[skill].append((sample, STEP_TIMEOUT_PENALTY))
            step_timeout_n += 1
            continue

        game_file = _game_file(sample)
        r_hint = cached_hint_rewards.get(game_file, 0.0)
        r_no_hint = _mean_or_zero(accumulated_game_rewards.get(game_file, []))

        regret = r_hint - r_no_hint
        # Negative hint regret is unreliable, so floor it without dropping the sample.
        raw = max(0.0, regret)
        if regret < 0:
            floored_n += 1
        skill_data[skill].append((sample, raw))

    metrics = _apply_centered_rewards(skill_data, env_reward_scale, "mean_regret")
    for skill, items in skill_data.items():
        metrics[f"delayed_env/skill_{skill}/num_accumulated_plays"] = _accumulated_play_count(
            accumulated_game_rewards, items
        )

    # Preserve the historical metric name as an alias for the floored fraction.
    _n = len(env_samples) if env_samples else 1
    metrics["delayed_env/proposer_floored_frac"] = floored_n / _n
    metrics["delayed_env/proposer_step_timeout_frac"] = step_timeout_n / _n
    metrics["delayed_env/proposer_masked_negative_frac"] = floored_n / _n

    return metrics


def recompute_delayed_env_rewards_blend(
    env_samples: List["Sample"],
    accumulated_game_rewards: Dict[str, Dict[int, List[float]]],
    cached_hint_rewards: Dict[str, float],
    env_reward_scale: float,
    regret_weight: float = 0.5,
    micro_lp_weight: float = 0.5,
    regret_scale: float = 0.15,
    micro_lp_scale: float = 0.10,
    micro_lp_signed: bool = True,
    micro_lp_slope: bool = True,
    frontier_weight: float = 0.0,
    frontier_scale: float = 0.08,
    plateau_weight: float = 0.0,
    plateau_lo: float = 0.4,
    plateau_hi: float = 0.6,
    plateau_ramp: float = 0.25,
    regret_floor: bool = False,
) -> Dict[str, float]:
    """Proposer reward = blend of matched-timing regret + micro learning potential.

    Two complementary per-game signals, computed from the SAME step-keyed buffer
    (so no extra rollouts):

    - regret (matched timing): R_hint - R_no_hint, BOTH measured at the regen step
      (step_offset 0). The hint plays and the offset-0 actor plays see the same
      actor state, so this is the clean "does privileged info help here" signal,
      free of the temporal bias that the flattened-accumulated R_no_hint introduced
      (accumulating over the improving actor dragged regret negative on learnable
      games). Match the play counts via --spare-hint-plays-per-game == trajectories
      per game (e.g. 16 vs 16).
    - micro_lp: did the actor actually improve on this game over the delay window.
      With micro_lp_slope=True (default) this is the WLS slope of per-step win-rate
      across the FULL window (weighted by #rollouts/step), expressed as fitted
      end-minus-start so micro_lp_scale stays interpretable as a win-rate delta. Uses
      every step -> ~2-3x lower variance than the 2-bucket late_mean - early_mean
      (micro_lp_slope=False). signed by default so a game that made the actor WORSE
      is a negative signal, not just masked to 0.

    - frontier anchor (optional, frontier_weight>0): subtract a penalty for games
      whose overall win-rate strays from 0.5 (max learnability). Holds the curriculum
      near 50% so micro_lp has headroom; an eased game (win_rate -> 1) earns a large
      negative -> masked, instead of being re-floated to positive by per-skill
      centering (the difficulty-easing death spiral). frontier_penalty = (wr-0.5)**2.

    - plateau anchor (optional, plateau_weight>0): ADD plateau_reward(win_rate) — a
      flat-top [0,1] difficulty reward, 1.0 in-band (~50% win) ramping to 0 at the
      extremes. This is the additive, non-negative alternative to the frontier
      penalty: instead of punishing drift with a negative term (which masks and can
      be reward-hacked around a threshold), it pays in-band games and pays easy/hard
      games 0, so the easy-drift deterrent is pure opportunity cost via the per-skill
      advantage baseline. Pair with regret_floor=True so the whole blend is >= 0
      (no masking, no extreme-negative for the gate to filter). The intended config
      is plateau_weight + regret_weight with micro_lp_weight=frontier_weight=0:
      difficulty (plateau) regulates *which hardness*, floored regret refines *which
      in-band game is most teachable*. See plateau_reward() for the bimodal caveat.

    Each component is divided by a FIXED nominal scale (NOT a per-batch std, which
    this codebase avoids to prevent small-std blowup) so the *_weight knobs are
    interpretable importance knobs rather than confounded with raw magnitude.
    blend = regret_weight*(regret/regret_scale [floored >=0 if regret_floor])
            + micro_lp_weight*(micro_lp/micro_lp_scale)
            + plateau_weight*plateau_reward(win_rate)
            - frontier_weight*(frontier_penalty/frontier_scale).

    Games with blend < 0 are masked from the proposer update (generalizes the
    neg-regret mask): keeping them would drag the per-skill mean below zero and hand
    blend~=0 unsolvable games positive advantage after centering -> death spiral.
    With the plateau config (regret_floor + no frontier) blend is always >= 0, so
    nothing is masked.

    Returns a metrics dict (per-skill + aggregate component breakdown) for logging.
    """
    if not env_samples:
        return {}

    skill_data: _SkillItems = defaultdict(list)
    all_regret: List[float] = []
    all_micro_lp: List[float] = []
    all_win_rate: List[float] = []
    all_frontier: List[float] = []
    all_plateau: List[float] = []        # flat-top difficulty reward per game
    all_win_rate_var: List[float] = []   # within-game play variance (bimodal-hack monitor)
    all_early: List[float] = []   # absolute early-half win rate (for early-vs-late tracking)
    all_late: List[float] = []    # absolute late-half win rate
    masked_negative = 0

    for sample in env_samples:
        game_file = _game_file(sample)

        step_rewards = accumulated_game_rewards.get(game_file, {})
        offsets = sorted(step_rewards.keys())

        # Matched-timing regret: hint vs no-hint, both at the regen step (offset 0).
        r_hint = cached_hint_rewards.get(game_file, 0.0)
        r_no_hint_0 = _mean_or_zero(step_rewards.get(offsets[0], []) if offsets else [])
        regret = r_hint - r_no_hint_0

        # Micro-LP: actor improvement over the window (signed by default).
        if len(offsets) >= 2:
            # Absolute early/late within-window win rates (tracked regardless of estimator).
            early_m, late_m = _early_late_means(step_rewards, offsets)
            all_early.append(early_m)
            all_late.append(late_m)
            if micro_lp_slope:
                # WLS slope of per-step mean win-rate across the FULL window,
                # weighted by sqrt(#rollouts) per step, as fitted end-minus-start
                # (same units as late-early). Every step contributes -> lower variance.
                xs = np.arange(len(offsets), dtype=float)
                ys = np.array([
                    float(np.mean(step_rewards[o])) if step_rewards[o] else 0.0
                    for o in offsets
                ])
                ns = np.array([len(step_rewards[o]) for o in offsets], dtype=float)
                w = np.sqrt(np.maximum(ns, 1.0))
                slope = float(np.polyfit(xs, ys, 1, w=w)[0])
                micro_lp = slope * (len(offsets) - 1)
            else:
                micro_lp = late_m - early_m
            if not micro_lp_signed:
                micro_lp = max(0.0, micro_lp)
        else:
            micro_lp = 0.0

        # Window win rate (used by frontier penalty AND the plateau anchor).
        all_window_r = [r for o in offsets for r in step_rewards[o]]
        win_rate = _mean_or_zero(all_window_r)
        win_rate_var = float(np.var(all_window_r)) if all_window_r else 0.0
        frontier_penalty = (win_rate - 0.5) ** 2

        # Plateau anchor: additive flat-top difficulty reward (>=0), peaking in-band.
        plateau = plateau_reward(win_rate, plateau_lo, plateau_hi, plateau_ramp)

        # Optional flooring keeps unreliable negative regret on the plateau's [0, 1] scale.
        regret_norm = regret / regret_scale
        if regret_floor:
            regret_norm = min(1.0, max(0.0, regret_norm))

        blend = (
            regret_weight * regret_norm
            + micro_lp_weight * (micro_lp / micro_lp_scale)
            + plateau_weight * plateau
            - frontier_weight * (frontier_penalty / frontier_scale)
        )

        sample.metadata["regret"] = regret
        sample.metadata["micro_lp"] = micro_lp
        sample.metadata["win_rate"] = win_rate
        sample.metadata["frontier_penalty"] = frontier_penalty
        sample.metadata["plateau"] = plateau
        sample.metadata["blend_raw"] = blend
        all_regret.append(regret)
        all_micro_lp.append(micro_lp)
        all_win_rate.append(win_rate)
        all_frontier.append(frontier_penalty)
        all_plateau.append(plateau)
        all_win_rate_var.append(win_rate_var)

        if blend < 0:
            sample.metadata["proposer_masked"] = True
            masked_negative += 1
            continue
        skill_data[_skill(sample)].append((sample, blend))

    metrics = _apply_centered_rewards(skill_data, env_reward_scale, "mean_blend")
    for skill, items in skill_data.items():
        metrics[f"delayed_env/skill_{skill}/num_games"] = len(items)

    # Aggregate component breakdown: lets us see which signal carries the curriculum
    # and tune the weights. std here is just for monitoring (NOT used in the reward).
    if all_regret:
        metrics["delayed_env/blend_regret_mean"] = float(np.mean(all_regret))
        metrics["delayed_env/blend_regret_std"] = float(np.std(all_regret))
        metrics["delayed_env/blend_micro_lp_mean"] = float(np.mean(all_micro_lp))
        metrics["delayed_env/blend_micro_lp_std"] = float(np.std(all_micro_lp))
        metrics["delayed_env/blend_win_rate_mean"] = float(np.mean(all_win_rate))
        metrics["delayed_env/blend_frontier_penalty_mean"] = float(np.mean(all_frontier))
        # Normalized component magnitudes — these three should be the same order of
        # magnitude; if one swamps the others, retune its *_scale (live calibration).
        metrics["delayed_env/comp_regret_norm"] = (
            regret_weight * float(np.mean(np.abs(all_regret))) / regret_scale
        )
        metrics["delayed_env/comp_micro_lp_norm"] = (
            micro_lp_weight * float(np.mean(np.abs(all_micro_lp))) / micro_lp_scale
        )
        metrics["delayed_env/comp_frontier_norm"] = (
            frontier_weight * float(np.mean(all_frontier)) / frontier_scale
        )
        # Expose the plateau contribution for reward-balance monitoring.
        metrics["delayed_env/blend_plateau_mean"] = float(np.mean(all_plateau))
        metrics["delayed_env/comp_plateau_norm"] = (
            plateau_weight * float(np.mean(all_plateau))
        )
        # Within-game play variance: a high mean at an off-0.5 win rate flags
        # possible bimodal (trivial-on-some-seeds) games gaming plateau(mean_win).
        # Monitoring only (per-seed structure needed to defend it in the reward).
        metrics["delayed_env/blend_win_rate_var_mean"] = float(np.mean(all_win_rate_var))
    # Absolute window endpoints complement the late-minus-early difference.
    if all_early:
        metrics["delayed_env/blend_early_mean"] = float(np.mean(all_early))
        metrics["delayed_env/blend_late_mean"] = float(np.mean(all_late))
    metrics["delayed_env/proposer_masked_negative_frac"] = (
        masked_negative / len(env_samples) if env_samples else 0.0
    )
    return metrics
