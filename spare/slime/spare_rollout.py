"""SPARE rollout function for Slime backend using SpareOrchestrator.

This module provides a custom rollout function that integrates SpareOrchestrator
with Slime's training pipeline for dual-role RL training.
"""

import asyncio
import copy
import logging
import os
import random
import resource
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.types import Sample
from spare.core.corpus import CorpusLoader
from spare.core.corpus_orchestrator import CorpusGroundedOrchestrator
from spare.core.env_memory import EnvironmentMemory
from spare.core.eval.fixed_model_eval import run_fixed_model_evaluation
from spare.core.game_generator import SyntheticGameGenerator
from spare.core.game_policy import GamePolicy
from spare.core.learning_potential import LearningPotential
from spare.core.orchestrator import SpareOrchestrator
from spare.core.prompts.tool_use_template import TOOL_USE_SKILLS
from spare.core.types import SpareConfig
from spare.core.utils.game_utils import compute_env_reward_scale, recompute_delayed_env_rewards, recompute_delayed_env_rewards_blend, recompute_delayed_env_rewards_micro_lp, recompute_delayed_env_rewards_regret
from spare.slime.model_adapter import create_slime_model_adapter
from spare.slime.trajectory_converter import fan_out_thinking_sample, trajectory_to_slime_sample

logger = logging.getLogger(__name__)


def _maybe_cap_rollout_memory() -> None:
    """Cap this process's address space via SPARE_ROLLOUT_RLIMIT_GB (0/unset = off).

    This bounds memory leaked by generated games whose executor threads cannot
    be cancelled. Generation runs in separate SGLang processes.
    """
    gb = int(os.environ.get("SPARE_ROLLOUT_RLIMIT_GB", "0") or 0)
    if gb <= 0:
        return
    cap = gb * 1024**3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_hard = cap if hard == resource.RLIM_INFINITY else min(cap, hard)
    if soft != resource.RLIM_INFINITY and soft <= new_hard:
        return  # already capped at least this tightly
    resource.setrlimit(resource.RLIMIT_AS, (new_hard, new_hard))
    logger.info("[ROLLOUT-GUARD] RLIMIT_AS capped at %d GB for pid %d", gb, os.getpid())


_maybe_cap_rollout_memory()

# Global state for learning potential tracker
_LEARNING_POTENTIALS = None
_GAME_POLICY = None

# Buffer for delayed proposer training: rollout_id -> list of env samples
_ENV_SAMPLE_BUFFER: Dict[int, List[Sample]] = {}

# First sample of the last non-empty rollout — fallback template for
# inert-padding a rollout that produced 0 samples.
_LAST_VALID_SAMPLE: Optional[Sample] = None

# Accumulated actor rewards for delayed env reward recomputation
# Key: regen epoch rollout_id, Value: {game_file: {step_offset: [raw_rewards]}}
_ACCUMULATED_GAME_REWARDS: Dict[int, Dict[str, Dict[int, List[float]]]] = {}
# For regret variant: cached hint rewards from regen step
# Key: regen epoch rollout_id, Value: {game_file: R_hint}
_CACHED_HINT_REWARDS: Dict[int, Dict[str, float]] = {}

_WANDB_METRICS_DEFINED = False

# Cached corpus loader (loaded once, reused across rollouts)
_CORPUS = None
_ENV_MEMORY: Optional[EnvironmentMemory] = None
_STATIC_GAME_POOL_STATE: Dict[str, Tuple[Tuple[str, ...], List[Path], random.Random]] = {}


def _select_static_games_without_replacement(
    game_files: List[Path],
    num_games: int,
    pool_key: str,
    seed: int,
) -> List[Path]:
    """Select a deterministic batch while exhausting the full pool before reuse."""
    if not game_files:
        return []

    canonical_files = tuple(sorted(str(path.resolve()) for path in game_files))
    state = _STATIC_GAME_POOL_STATE.get(pool_key)
    if state is None or state[0] != canonical_files:
        rng = random.Random(seed)
        remaining = [Path(path) for path in canonical_files]
        rng.shuffle(remaining)
        state = (canonical_files, remaining, rng)
        _STATIC_GAME_POOL_STATE[pool_key] = state

    canonical_files, remaining, rng = state
    selected = remaining[:num_games]
    remaining = remaining[num_games:]

    if len(selected) < num_games:
        refill = [Path(path) for path in canonical_files]
        rng.shuffle(refill)
        selected_set = {str(path) for path in selected}
        refill = [path for path in refill if str(path) not in selected_set]
        needed = num_games - len(selected)
        selected.extend(refill[:needed])
        remaining = refill[needed:]

    _STATIC_GAME_POOL_STATE[pool_key] = (canonical_files, remaining, rng)
    return selected


def _get_or_create_corpus(args: Namespace):
    """Get or create the global corpus loader (lazy singleton)."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = CorpusLoader(
            corpus_path=args.spare_corpus_file,
            max_doc_tokens=args.spare_corpus_max_doc_tokens,
            seed=args.spare_corpus_seed,
        )
        num_loaded = _CORPUS.load()
        if num_loaded == 0:
            raise RuntimeError(
                f"[SPARE] Corpus file contains no usable documents: "
                f"{args.spare_corpus_file}"
            )
        logger.info(
            f"[SPARE] Corpus loaded: {num_loaded} documents "
            f"from {args.spare_corpus_file}"
        )
    return _CORPUS


def _define_spare_wandb_metrics():
    """Register SPARE metrics on the appropriate WandB step axes."""
    global _WANDB_METRICS_DEFINED
    if _WANDB_METRICS_DEFINED:
        return
    _WANDB_METRICS_DEFINED = True
    try:
        import wandb
        if wandb.run is not None:
            wandb.define_metric("gem_eval/*", step_metric="eval/step")
            # Earlier Slime patterns take precedence over this catch-all.
            wandb.define_metric("*", step_metric="rollout/step")
            logger.info("[SPARE] Registered gem_eval/* (eval/step) + catch-all * (rollout/step)")
    except Exception as e:
        logger.warning("[SPARE] Failed to define wandb metrics: %s", e)


def get_game_policy(args: Namespace) -> GamePolicy:  # noqa: ARG001
    global _GAME_POLICY
    if _GAME_POLICY is None:
        _GAME_POLICY = GamePolicy()
    return _GAME_POLICY

DEFAULT_SKILLS_COGNITIVE = ["Pattern Recognition", "Mathematical Reasoning"]
DEFAULT_SKILLS_TOOL_USE = list(TOOL_USE_SKILLS.keys())[:2]

# Map game_type -> (default_skills, available_skills_dict)
_GAME_TYPE_SKILLS = {
    "cognitive": (DEFAULT_SKILLS_COGNITIVE, SyntheticGameGenerator.COGNITIVE_SKILLS),
    "tool_use": (DEFAULT_SKILLS_TOOL_USE, TOOL_USE_SKILLS),
}


def get_configured_skills(args: Namespace) -> List[str]:
    """Get the list of skills to use for game generation.

    Respects --spare-game-type and validates skills against the correct skill set.
    If --spare-skills is not specified, uses default skills for the game type.
    """
    game_type = getattr(args, "spare_game_type", "cognitive")
    defaults, available = _GAME_TYPE_SKILLS.get(game_type, _GAME_TYPE_SKILLS["cognitive"])

    if args.spare_skills:
        # Accept underscore aliases that survive shell and Ray argument forwarding.
        requested = [s.replace("_", " ").strip() for s in args.spare_skills]
        for skill in requested:
            if skill not in available:
                raise ValueError(
                    f"Unknown skill '{skill}' for game_type='{game_type}'. "
                    f"Available: {sorted(available)}"
                )
        return requested
    return defaults


def select_active_skills(
    all_skills: List[str], rollout_id: int, regen_interval: int, skills_per_regen: int
) -> List[str]:
    """Round-robin a subset of skills per regen epoch (skill-rotation, Option B).

    With skills_per_regen < len(all_skills), each regen epoch activates a disjoint
    window of `skills_per_regen` skills, advancing by that many each epoch. Keeps
    per-rollout cost flat while giving each active skill more games, and bounds how long
    any skill stays idle (gap = ceil(n/k) - 1 regen cycles). 0 or >= n disables rotation.
    The window is keyed on the regen EPOCH so it's constant across the replay steps within
    a regen interval (game generation + its replays use the same active skills).
    """
    n = len(all_skills)
    if skills_per_regen <= 0 or skills_per_regen >= n or regen_interval <= 0:
        return all_skills
    regen_epoch = rollout_id // regen_interval
    start = (regen_epoch * skills_per_regen) % n
    return [all_skills[(start + j) % n] for j in range(skills_per_regen)]


def get_learning_potentials(args: Namespace) -> Dict[str, LearningPotential]:
    """Get or create the global learning potential tracker."""
    global _LEARNING_POTENTIALS
    if _LEARNING_POTENTIALS is None:
        gamma_fast = args.spare_gamma1
        gamma_slow = args.spare_gamma2

        skills = get_configured_skills(args)
        _LEARNING_POTENTIALS = {skill: LearningPotential(gamma_fast=gamma_fast, gamma_slow=gamma_slow) for skill in skills}
    return _LEARNING_POTENTIALS

def _flatten_accumulated_rewards(
    accumulated: Dict[str, Dict[int, List[float]]],
) -> Dict[str, List[float]]:
    """Flatten per-step accumulated rewards into flat lists for LP/regret variants."""
    return {
        gf: [r for offset in sorted(step_dict) for r in step_dict[offset]]
        for gf, step_dict in accumulated.items()
    }


def _inert_copy(template: Sample, index: int) -> Sample:
    """Zero-loss-mask copy of `template`: fills the batch (slime requires exactly
    global_batch_size groups) but contributes no gradient. Tagged spare_is_pad so the
    denominator correction excludes it from n_real. DISTINCT object per call — aliasing
    one sample would collapse to a single group_id and trip slime's num_groups assert."""
    dup = copy.deepcopy(template)
    dup.loss_mask = [0] * (len(dup.loss_mask) if dup.loss_mask is not None else dup.response_length)
    dup.index = index
    dup.metadata = {**(dup.metadata or {}), "spare_is_pad": True}
    return dup


def spare_generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Callable,
    evaluation: bool = False,
) -> RolloutFnTrainOutput:
    """SPARE rollout function using SpareOrchestrator for dual-role training.

    This function replaces Slime's standard rollout with SPARE's dual-role approach:
    1. Generates games using the model (environment role)
    2. Plays all games concurrently using async HTTP (optimal for Slime)
    3. Returns samples for both roles with appropriate rewards

    Args:
        args: Slime Namespace with all configuration
        rollout_id: Current rollout ID for deterministic generation
        data_source: Data source callable for fetching prompts (not used in SPARE)
        evaluation: Whether this is an evaluation rollout

    Returns:
        RolloutFnTrainOutput with samples from both actor and environment roles
    """
    global _LAST_VALID_SAMPLE
    _define_spare_wandb_metrics()

    # Route to fixed-env mode if configured
    if args.spare_mode == "fixed_env":
        from spare.slime.fixed_env_rollout import spare_fixed_env_rollout
        return spare_fixed_env_rollout(args, rollout_id, data_source, evaluation)

    if evaluation:
        # Check for GEM eval
        if args.spare_gem_eval_config:
            from spare.slime.gem_eval_rollout import spare_gem_eval_rollout
            return spare_gem_eval_rollout(args, rollout_id, data_source)
        from slime.rollout.sglang_rollout import generate_rollout
        return generate_rollout(args, rollout_id, data_source, evaluation=True)

    del data_source  # data_source is not used in SPARE, but Slime requires it to be provided.

    # Validate delayed proposer training config
    if args.spare_proposer_training_delay > 0 and not args.use_tis:
        logger.warning(
            "[SPARE] proposer-training-delay > 0 but --use-tis not enabled. "
            "IS correction recommended for delayed training."
        )
    if args.spare_env_reward_variant in ("micro_lp", "blend") and args.spare_proposer_training_delay == 0:
        raise ValueError(
            f"[SPARE] {args.spare_env_reward_variant} env reward variant requires --spare-proposer-training-delay > 0."
        )

    # Thinking fan-out varies sample counts, so SPARE computes group advantages.
    if args.spare_actor_enable_thinking and args.rewards_normalization:
        raise ValueError(
            "[SPARE] --spare-actor-enable-thinking requires "
            "--disable-rewards-normalization (advantages are precomputed "
            "per game; slime's group norm assumes a fixed sample count per group)."
        )

    # Create Slime model adapter
    model_adapter = create_slime_model_adapter(args)

    # Create SpareConfig from args
    spare_config = SpareConfig(
        env_temperature=args.spare_env_temperature,
        env_top_p=args.spare_env_top_p,
        env_top_k=args.spare_env_top_k,
        actor_temperature=args.spare_actor_temperature,
        actor_top_p=args.spare_actor_top_p,
        actor_top_k=args.spare_actor_top_k,
        max_turns=args.spare_max_turns,
        gamma=args.spare_gamma,
        env_generation_template=args.spare_env_generation_template,
        env_max_tokens=args.spare_env_max_tokens,
        env_enable_thinking=True if args.spare_env_enable_thinking else None,
        # Proposer-only thinking explicitly keeps the actor non-thinking.
        actor_enable_thinking=(
            True if args.spare_actor_enable_thinking
            else (False if args.spare_env_enable_thinking else None)
        ),
        env_repair_turns=args.spare_env_repair_turns,
        persist_rejected=args.spare_persist_rejected,
        actor_template=args.spare_actor_template,
        actor_max_tokens=args.spare_actor_max_tokens,
        max_context_length=args.spare_max_context_length,
        use_format_reward=args.spare_use_format_reward,
        format_reward_value=args.spare_format_reward_value,
        use_solver_variance_reward=args.spare_use_solver_variance_reward,
        # Self-judge parameters
        use_self_judge=args.spare_use_self_judge,
        self_judge_temperature=args.spare_self_judge_temperature,
        self_judge_max_tokens=args.spare_self_judge_max_tokens,
        self_judge_penalty=args.spare_self_judge_penalty,
        self_judge_max_turns_to_show=args.spare_self_judge_max_turns_to_show,
        # Environment reward scaling parameters
        env_reward_scaling_variant=args.spare_env_reward_scaling_variant,
        max_env_reward_scale=args.spare_max_env_reward_scale,
        auto_compute_env_reward_scale=args.spare_auto_compute_env_reward_scale,
        # Training control
        train_on_env_trajectories=args.spare_train_on_env_trajectories,
        # Actor reward normalization
        reward_normalization=args.spare_reward_normalization,
        game_baseline_decay=args.spare_game_baseline_decay,
        # Environment reward variant
        env_reward_variant=args.spare_env_reward_variant,
        hint_mode=args.spare_hint_mode,
        hint_model=args.spare_hint_model,
        hint_api_key_env=args.spare_hint_api_key_env,
        hint_api_base_url=args.spare_hint_api_base_url,
        hint_temperature=args.spare_hint_temperature,
        hint_max_tokens=args.spare_hint_max_tokens,
        hint_plays_per_game=args.spare_hint_plays_per_game,
        # Fixed model evaluation parameters
        fixed_eval_model=args.spare_fixed_eval_model,
        fixed_eval_plays_per_game=args.spare_trajectories_per_game,
        fixed_eval_max_concurrent=args.spare_fixed_eval_max_concurrent,
        fixed_eval_temperature=args.spare_fixed_eval_temperature,
        fixed_eval_max_tokens=args.spare_fixed_eval_max_tokens,
        fixed_eval_api_base_url=args.spare_fixed_eval_api_base_url,
        fixed_eval_api_key_env=args.spare_fixed_eval_api_key_env,
        # Environment validator parameters
        use_env_validator=args.spare_use_env_validator,
        env_validator_model=args.spare_env_validator_model,
        env_validator_api_key_env=args.spare_env_validator_api_key_env,
        env_validator_api_base_url=args.spare_env_validator_api_base_url,
        env_validator_temperature=args.spare_env_validator_temperature,
        env_validator_max_tokens=args.spare_env_validator_max_tokens,
        # Delayed LP: skip per-rollout LP updates when delay > 0 with LP variant
        skip_lp_update=(
            args.spare_proposer_training_delay > 0
            and args.spare_env_reward_variant == "learning_potential"
        ),
        # Corpus-grounded generation
        corpus_file=args.spare_corpus_file,
        corpus_max_doc_tokens=args.spare_corpus_max_doc_tokens,
        corpus_seed=args.spare_corpus_seed,
        # Action format
        action_format=args.spare_action_format,
        # Game type for self-play generation
        game_type=args.spare_game_type,
    )

    # Environment memory (optional, for memory-augmented generation): high-regret
    # past environments are injected as few-shot seeds into the generation prompt.
    # Works with BOTH the cold-start and corpus-grounded proposers.
    env_memory = None
    if args.spare_use_env_memory:
        global _ENV_MEMORY
        if _ENV_MEMORY is None:
            _ENV_MEMORY = EnvironmentMemory(max_size=args.spare_env_memory_max_size)
            memory_path = Path(args.spare_cache_dir) / "env_memory.json"
            _ENV_MEMORY.load(memory_path)
        env_memory = _ENV_MEMORY

    # Create orchestrator (corpus-grounded if corpus file provided)
    if args.spare_corpus_file:
        corpus = _get_or_create_corpus(args)
        orchestrator = CorpusGroundedOrchestrator(
            model=model_adapter,
            config=spare_config,
            learning_potentials=get_learning_potentials(args),
            game_policy=get_game_policy(args),
            corpus=corpus,
            env_memory=env_memory,
        )
    else:
        orchestrator = SpareOrchestrator(
            model=model_adapter,
            config=spare_config,
            learning_potentials=get_learning_potentials(args),
            game_policy=get_game_policy(args),
            env_memory=env_memory,
        )

    # Configuration
    game_regeneration_interval = args.spare_game_regeneration_interval
    should_regenerate = (
        game_regeneration_interval > 0 and
        rollout_id % game_regeneration_interval == 0
    )
    games_dir = Path(args.spare_games_dir)
    games_dir.mkdir(parents=True, exist_ok=True)

    # Cache directory for generated games (optional)
    cache_dir = args.spare_cache_dir
    if cache_dir:
        cache_dir = Path(cache_dir)

    skills = get_configured_skills(args)
    skills = select_active_skills(
        skills, rollout_id, game_regeneration_interval, args.spare_skills_per_regen
    )
    if args.spare_skills_per_regen > 0:
        logger.info("[SPARE] rollout %d active skills (round-robin %d/%d): %s",
                    rollout_id, args.spare_skills_per_regen,
                    len(get_configured_skills(args)), skills)
    num_games = args.rollout_batch_size
    difficulty = args.spare_game_difficulty
    use_solver_variance_reward = args.spare_use_solver_variance_reward
    # Calculate trajectories_per_game from global_batch_size
    # We generate exactly global_batch_size samples: num_games * trajectories_per_game = global_batch_size
    global_batch_size = args.global_batch_size
    trajectories_per_game = max(1, global_batch_size // num_games)

    logger.info(
        f"[SPARE] Batch calculation: global_batch_size={global_batch_size}, "
        f"num_games={num_games}, trajectories_per_game={trajectories_per_game}"
    )

    # Get existing game files
    existing_game_files = list(games_dir.glob("game_*.py"))
    if getattr(args, "spare_static_game_pool", False):
        if not existing_game_files:
            raise RuntimeError(f"[SPARE] Static game pool is empty: {games_dir}")
        if getattr(args, "spare_no_replacement", False):
            existing_game_files = _select_static_games_without_replacement(
                existing_game_files,
                num_games=num_games,
                pool_key=str(games_dir.resolve()),
                seed=getattr(args, "spare_fixed_pool_seed", 42),
            )
        logger.info(
            "[SPARE] Static pool selected %d/%d games for rollout %d",
            len(existing_game_files), num_games, rollout_id,
        )

    # Use collect_trajectories with async mode (optimal for Slime's HTTP backend)
    env_trajectories, actor_trajectories, collect_info = orchestrator.collect_trajectories(
        should_regenerate=should_regenerate,
        skills=skills,
        difficulty=difficulty,
        game_files=existing_game_files,
        games_dir=games_dir,
        min_games=num_games,
        max_games=num_games,
        global_batch_size=global_batch_size,
        mode="async",
        rollout_id=rollout_id,
        cache_dir=cache_dir,
        max_attempts=5,
        use_solver_variance_reward=use_solver_variance_reward,
        regeneration_interval=game_regeneration_interval,
        allow_generation=not getattr(args, "spare_static_game_pool", False),
    )

    # Convert trajectories to Slime samples
    actor_samples: List[Sample] = []
    env_samples: List[Sample] = []

    # Convert environment trajectories
    for traj in env_trajectories:
        sample = trajectory_to_slime_sample(
            trajectory=traj,
            index=0,  # re-indexed below
            role="environment",
        )
        env_samples.append(sample)

    # Convert actor trajectories
    for traj in actor_trajectories:
        sample = trajectory_to_slime_sample(
            trajectory=traj,
            index=0,  # re-indexed below
            role="actor",
        )
        actor_samples.append(sample)

    # --- Delayed proposer training ---
    delay = args.spare_proposer_training_delay

    # Tag ALL samples with train_metadata (needed for batch["metadata"] to exist)
    for sample in actor_samples:
        sample.train_metadata = {"role": "actor", "delayed_by": 0}
    for sample in env_samples:
        sample.train_metadata = {"role": "environment", "delayed_by": delay}

    # --- Accumulate actor rewards for delayed reward recomputation ---
    if delay > 0:
        # Determine which regen epoch these games belong to
        regen_epoch = (
            (rollout_id // game_regeneration_interval) * game_regeneration_interval
            if game_regeneration_interval > 0 else 0
        )

        # Extract raw actor rewards per game_file, keyed by step offset within regen epoch
        if regen_epoch not in _ACCUMULATED_GAME_REWARDS:
            _ACCUMULATED_GAME_REWARDS[regen_epoch] = defaultdict(lambda: defaultdict(list))
        step_offset = rollout_id - regen_epoch
        for sample in actor_samples:
            gf = sample.metadata.get("game_file", "")
            raw_r = sample.metadata.get("original_reward", sample.reward)
            _ACCUMULATED_GAME_REWARDS[regen_epoch][gf][step_offset].append(raw_r)

        # For regret / blend variants: cache hint rewards at regen step
        if should_regenerate and args.spare_env_reward_variant in ("regret_based", "blend"):
            _CACHED_HINT_REWARDS[regen_epoch] = {}
            for sample in env_samples:
                gf = sample.metadata.get("game_file", "")
                hint_stats = sample.metadata.get("hint_stats", {})
                if "r_hint" in hint_stats:
                    _CACHED_HINT_REWARDS[regen_epoch][gf] = hint_stats["r_hint"]

    if delay > 0 and env_samples:
        # Buffer current env samples
        _ENV_SAMPLE_BUFFER[rollout_id] = list(env_samples)

        # Retrieve delayed env samples
        env_samples_for_training: List[Sample] = []
        target_id = rollout_id - delay
        if target_id in _ENV_SAMPLE_BUFFER:
            env_samples_for_training = _ENV_SAMPLE_BUFFER.pop(target_id)

        # Flush at final rollout to avoid losing env samples
        if rollout_id == args.num_rollout - 1 and _ENV_SAMPLE_BUFFER:
            for buf_id in sorted(_ENV_SAMPLE_BUFFER.keys()):
                env_samples_for_training.extend(_ENV_SAMPLE_BUFFER[buf_id])
            _ENV_SAMPLE_BUFFER.clear()
    else:
        env_samples_for_training = env_samples

    # --- Recompute delayed env rewards using accumulated actor data ---
    recompute_metrics: Dict[str, float] = {}
    if delay > 0 and env_samples_for_training:
        # Collect all accumulated rewards for the delayed samples.
        # Normal case: target_id is the regen epoch for the delayed batch.
        # Flush case: multiple epochs may be in the buffer — merge them all.
        is_flush = rollout_id == args.num_rollout - 1
        accumulated: Dict[str, Dict[int, List[float]]] = {}
        cached_hints: Dict[str, float] = {}

        if is_flush:
            # Merge all remaining accumulated data (per-step structure)
            for epoch_id in sorted(_ACCUMULATED_GAME_REWARDS.keys()):
                for gf, step_dict in _ACCUMULATED_GAME_REWARDS[epoch_id].items():
                    if gf not in accumulated:
                        accumulated[gf] = {}
                    for offset, rewards in step_dict.items():
                        accumulated[gf].setdefault(offset, []).extend(rewards)
            _ACCUMULATED_GAME_REWARDS.clear()
            for epoch_id in sorted(_CACHED_HINT_REWARDS.keys()):
                cached_hints.update(_CACHED_HINT_REWARDS[epoch_id])
            _CACHED_HINT_REWARDS.clear()
        else:
            # Normal delayed retrieval: pop the target regen epoch
            delayed_regen_epoch = (
                (target_id // game_regeneration_interval) * game_regeneration_interval
                if game_regeneration_interval > 0 else target_id
            )
            if delayed_regen_epoch in _ACCUMULATED_GAME_REWARDS:
                accumulated = dict(_ACCUMULATED_GAME_REWARDS.pop(delayed_regen_epoch))
            cached_hints = _CACHED_HINT_REWARDS.pop(delayed_regen_epoch, {})

        if accumulated:
            # Compute env_reward_scale
            env_reward_scale = compute_env_reward_scale(
                num_env_trajectories=len(env_samples_for_training),
                num_actor_trajectories=len(actor_samples),
                regeneration_interval=game_regeneration_interval,
                max_scale=args.spare_max_env_reward_scale,
            ) if args.spare_auto_compute_env_reward_scale else 1.0

            if args.spare_env_reward_variant == "blend":
                # Needs the step-keyed accumulated buffer (step-0 for matched regret,
                # early/late split for micro-LP) AND the cached hint rewards.
                recompute_metrics = recompute_delayed_env_rewards_blend(
                    env_samples=env_samples_for_training,
                    accumulated_game_rewards=accumulated,
                    cached_hint_rewards=cached_hints,
                    env_reward_scale=env_reward_scale,
                    regret_weight=args.spare_regret_weight,
                    micro_lp_weight=args.spare_micro_lp_weight,
                    regret_scale=args.spare_regret_scale,
                    micro_lp_scale=args.spare_micro_lp_scale,
                    micro_lp_signed=not args.spare_micro_lp_unsigned,
                    micro_lp_slope=(args.spare_micro_lp_estimator == "slope"),
                    frontier_weight=args.spare_frontier_weight,
                    frontier_scale=args.spare_frontier_scale,
                    plateau_weight=args.spare_plateau_weight,
                    plateau_lo=args.spare_plateau_lo,
                    plateau_hi=args.spare_plateau_hi,
                    plateau_ramp=args.spare_plateau_ramp,
                    regret_floor=args.spare_regret_floor,
                )
            elif args.spare_env_reward_variant == "micro_lp":
                recompute_metrics = recompute_delayed_env_rewards_micro_lp(
                    env_samples=env_samples_for_training,
                    accumulated_game_rewards=accumulated,
                    env_reward_scale=env_reward_scale,
                )
            elif args.spare_env_reward_variant == "regret_based":
                flat = _flatten_accumulated_rewards(accumulated)
                recompute_metrics = recompute_delayed_env_rewards_regret(
                    env_samples=env_samples_for_training,
                    accumulated_game_rewards=flat,
                    cached_hint_rewards=cached_hints,
                    env_reward_scale=env_reward_scale,
                )
            else:
                flat = _flatten_accumulated_rewards(accumulated)
                recompute_metrics = recompute_delayed_env_rewards(
                    env_samples=env_samples_for_training,
                    accumulated_game_rewards=flat,
                    learning_potentials=get_learning_potentials(args),
                    env_reward_scale=env_reward_scale,
                    use_solver_variance_reward=args.spare_use_solver_variance_reward,
                    update_lp=(args.spare_env_reward_variant == "learning_potential"),
                )

            total_plays = sum(len(r) for sd in accumulated.values() for r in sd.values())
            logger.info(
                f"[SPARE] Recomputed delayed env rewards "
                f"using {total_plays} accumulated actor plays"
            )

    # Drop proposer samples masked for negative regret (see assign_env_rewards_regret).
    env_samples_for_training = [
        s for s in env_samples_for_training if not s.metadata.get("proposer_masked")
    ]

    # Re-index and combine. ENV (proposer) samples FIRST: the actor batch alone fills
    # global_batch_size, so the [:global_batch_size] trim below would otherwise shed the
    # whole env tail (zeroing the proposer gradient). env-first trims surplus actor instead.
    all_samples: List[Sample] = env_samples_for_training + list(actor_samples)
    for i, sample in enumerate(all_samples):
        sample.index = i

    # Compute metrics
    trained_actor = [s for s in all_samples if s.metadata.get('role') == 'actor']
    trained_env = [s for s in all_samples if s.metadata.get('role') == 'environment']

    # Compute separate reward metrics for actor and env (avoid `if s.reward` which filters out 0s)
    actor_rewards = [s.reward for s in trained_actor]
    env_rewards = [s.reward for s in trained_env]
    all_rewards = [s.reward for s in all_samples]
    original_actor_rewards = [
        s.metadata.get("original_reward", s.reward) for s in trained_actor
    ]

    # Per-game grouping for GRPO diagnostics
    from collections import defaultdict as _defaultdict
    game_groups = _defaultdict(list)
    for s in trained_actor:
        gf = s.metadata.get("game_file", "unknown")
        game_groups[gf].append(s.reward)

    # Zero-std groups: games where all 16 plays got the same reward (no variance for GRPO)
    num_zero_std_groups = sum(
        1 for rewards in game_groups.values()
        if len(rewards) > 1 and len(set(rewards)) == 1
    )
    num_all_zero_groups = sum(
        1 for rewards in game_groups.values()
        if len(rewards) > 1 and all(r == 0.0 for r in rewards)
    )
    num_all_positive_groups = sum(
        1 for rewards in game_groups.values()
        if len(rewards) > 1 and all(r > 0.0 for r in rewards)
    )
    num_groups_total = len(game_groups)

    # Update environment memory with game quality data (memory-augmented generation).
    # Records each regenerated game's code + actor win-rate + per-game regret so the
    # proposer can retrieve high-regret seeds next regen step (env_memory.high_regret_seeds).
    if _ENV_MEMORY is not None and should_regenerate:
        _orig_game_groups = _defaultdict(list)
        for s in trained_actor:
            gf = s.metadata.get("game_file", "unknown")
            _orig_game_groups[gf].append(s.metadata.get("original_reward", s.reward))
        seen_games = set()
        for gf, plays in _orig_game_groups.items():
            if gf in seen_games or gf == "unknown":
                continue
            seen_games.add(gf)
            game_path = Path(gf)
            game_code = ""
            if game_path.exists():
                try:
                    game_code = game_path.read_text()
                except Exception:
                    pass
            skill = ""
            for s in trained_actor:
                if s.metadata.get("game_file") == gf:
                    skill = s.metadata.get("skill", "")
                    break
            win_rate = sum(1 for r in plays if r > 0) / max(len(plays), 1)
            # Real per-game regret cached on the orchestrator this regen step
            # (getattr default: orchestrator may not cache it outside the regret variant).
            game_regret = float(getattr(orchestrator, "_cached_regret", {}).get(gf, 0.0))
            if game_code and gf:
                _ENV_MEMORY.add(
                    game_file=gf, skill=skill, game_code=game_code,
                    actor_win_rate=win_rate, regret=game_regret, rollout_id=rollout_id,
                )
        memory_path = Path(args.spare_cache_dir) / "env_memory.json"
        _ENV_MEMORY.save(memory_path)

    # Replace spaces in all dict keys with underscores, including collect_info
    metrics = {
        k.replace(" ", "_"): v for k, v in {
            **collect_info,
            **recompute_metrics,
            "rollout/num_actor_samples": len(trained_actor),
            "rollout/num_env_samples": len(trained_env),
            # Overall mean (includes both actor z-scored and env scaled rewards)
            "rollout/mean_return": sum(all_rewards) / max(len(all_rewards), 1),
            # Separate actor metrics (z-score normalized, expect mean ~0)
            "rollout/actor_mean_return": sum(actor_rewards) / max(len(actor_rewards), 1),
            # Separate env metrics (scaled learning potential)
            "rollout/env_mean_return": sum(env_rewards) / max(len(env_rewards), 1) if env_rewards else 0.0,
            # Original (pre-normalized) actor rewards from metadata
            "rollout/actor_mean_original_reward": sum(original_actor_rewards) / max(len(original_actor_rewards), 1),
            # Actor reward std (should be >0 for meaningful GRPO updates)
            "rollout/actor_reward_std": float(np.std(actor_rewards)) if actor_rewards else 0.0,
            "rollout/actor_original_reward_std": float(np.std(original_actor_rewards)) if original_actor_rewards else 0.0,
            # Win rate: fraction of actor plays with positive reward
            "rollout/actor_win_rate": sum(1 for r in original_actor_rewards if r > 0) / max(len(original_actor_rewards), 1),
            # Per-game GRPO diagnostics
            "rollout/num_games": num_groups_total,
            "rollout/zero_std_group_pct": num_zero_std_groups / max(num_groups_total, 1),
            "rollout/all_zero_group_pct": num_all_zero_groups / max(num_groups_total, 1),
            "rollout/all_positive_group_pct": num_all_positive_groups / max(num_groups_total, 1),
            # Num failed trajectories
            "rollout/num_failed": collect_info.get("num_failed", 0),
            "rollout/num_timeout": collect_info.get("num_timeout", 0),
            "rollout/mode": "async",
            # Delayed proposer metrics
            "rollout/proposer_delay_buffered": sum(len(v) for v in _ENV_SAMPLE_BUFFER.values()),
            "rollout/proposer_delay_num_trained": len(env_samples_for_training) if delay > 0 else 0,
            # Batch composition
            "rollout/env_batch_fraction": len(trained_env) / max(len(all_samples), 1),
            "rollout/env_actual_delay_steps": float(delay) if delay > 0 and trained_env else 0.0,
            # Mean and maximum lengths expose role-specific token-cap pressure.
            "rollout/actor_mean_response_len": float(np.mean([s.response_length for s in trained_actor])) if trained_actor else 0.0,
            "rollout/env_mean_response_len": float(np.mean([s.response_length for s in trained_env])) if trained_env else 0.0,
            "rollout/actor_response_len_max": float(max((s.response_length for s in trained_actor), default=0)),
            "rollout/env_response_len_max": float(max((s.response_length for s in trained_env), default=0)),
            # Environment truncation metadata is populated by repair mode.
            "rollout/actor_truncated_pct": sum(1 for s in trained_actor if s.status == Sample.Status.TRUNCATED) / max(len(trained_actor), 1),
            "rollout/env_truncated_pct": sum(1 for s in trained_env if s.metadata.get("env_truncated")) / max(len(trained_env), 1),
        }.items()
    }

    # Inert-pad empty rollouts because Slime requires at least one sample.
    if not all_samples:
        logger.critical(
            f"[SPARE] Rollout {rollout_id} produced 0 samples "
            f"(failed={collect_info.get('num_failed', 0)}, "
            f"timeout={collect_info.get('num_timeout', 0)}) — inert-padding to skip it."
        )
        fallback = _LAST_VALID_SAMPLE
        for buf_id in sorted(_ENV_SAMPLE_BUFFER.keys(), reverse=True):
            if _ENV_SAMPLE_BUFFER[buf_id]:
                fallback = _ENV_SAMPLE_BUFFER[buf_id][0]
                break
        if fallback is None:
            raise ValueError(
                f"[SPARE] Rollout {rollout_id}: 0 samples and no fallback (first rollout, "
                f"all trajectories failed). Check game generation/execution logs."
            )
        all_samples = [_inert_copy(fallback, i) for i in range(global_batch_size)]

    # Update last-valid-sample for future empty-rollout fallbacks
    if all_samples:
        _LAST_VALID_SAMPLE = all_samples[0]

    # Zero loss_mask on truncated trajectories: hitting max_turns is the env running out
    # of budget, not a policy failure — training on them as reward=0 teaches early submit.
    if args.spare_compact_filter:
        n_compacted = 0
        for sample in all_samples:
            if sample.status == Sample.Status.TRUNCATED:
                sample.loss_mask = [0] * len(sample.loss_mask)
                n_compacted += 1
        metrics["rollout/compact_filtered_count"] = float(n_compacted)
        metrics["rollout/compact_filtered_pct"] = n_compacted / max(len(all_samples), 1)

    # Inert-pad partial batches to satisfy Slime's fixed group count.
    if 0 < len(all_samples) < global_batch_size:
        n_real = len(all_samples)
        shortfall = global_batch_size - n_real
        all_samples += [_inert_copy(all_samples[i % n_real], n_real + i) for i in range(shortfall)]
        for i, sample in enumerate(all_samples):
            sample.index = i
        logger.warning(
            f"[SPARE] Rollout {rollout_id}: partial batch ({n_real} < {global_batch_size}); "
            f"inert-padded with {shortfall} copies."
        )
        metrics["rollout/num_padded_samples"] = float(shortfall)
        metrics["rollout/padded_pct"] = shortfall / global_batch_size

    # Group samples for Slime (each sample is its own group)
    grouped_samples = [[sample] for sample in all_samples]
    if len(grouped_samples) > global_batch_size:
        n_trim = len(grouped_samples) - global_batch_size
        grouped_samples = grouped_samples[:global_batch_size]
        logger.warning(
            f"[SPARE] Trimmed {n_trim} surplus (now actor) samples to meet global_batch_size"
        )
        metrics["rollout/num_trimmed_samples"] = n_trim

    # Slime divides by global_batch_size, so supported partial-batch configurations
    # scale real rewards by global_batch_size / n_real to preserve their mean loss.
    _scale_supported = (
        not args.normalize_advantages
        and args.entropy_coef == 0.0
        and args.kl_loss_coef == 0.0
        and args.advantage_estimator == "grpo"
    )
    final_samples = [grp[0] for grp in grouped_samples]
    n_real = sum(1 for s in final_samples if not (s.metadata or {}).get("spare_is_pad"))
    if 0 < n_real < global_batch_size:
        if _scale_supported:
            reward_scale = global_batch_size / n_real
            for s in final_samples:
                if not (s.metadata or {}).get("spare_is_pad"):
                    s.reward = s.reward * reward_scale
            metrics["rollout/reward_scale"] = float(reward_scale)
            metrics["rollout/n_real_samples"] = float(n_real)
            logger.info(
                f"[SPARE] Rollout {rollout_id}: {n_real}/{global_batch_size} real "
                f"samples; scaled real-sample rewards by {reward_scale:.4f} so the "
                f"loss divides by n_real (matches --use-dynamic-global-batch-size)."
            )
        else:
            metrics["rollout/reward_scale"] = 1.0
            metrics["rollout/n_real_samples"] = float(n_real)
            logger.warning(
                f"[SPARE] Rollout {rollout_id}: partial batch "
                f"({n_real}/{global_batch_size} real) but reward-scaling is DISABLED "
                f"for this config (normalize_advantages={args.normalize_advantages}, "
                f"entropy_coef={args.entropy_coef}, kl_loss_coef={args.kl_loss_coef}, "
                f"advantage_estimator={args.advantage_estimator}). The padded loss is "
                f"attenuated by {n_real / global_batch_size:.3f}x on this step."
            )

    # Fan out thinking episodes last so earlier accounting stays episode-based;
    # all turn samples retain one shared scheduler group.
    if args.spare_actor_enable_thinking:
        n_turn_samples = 0
        n_fanned_episodes = 0
        fanned_groups: List[List[Sample]] = []
        for grp in grouped_samples:
            episode = grp[0]
            is_pad = bool((episode.metadata or {}).get("spare_is_pad"))
            if is_pad or not (episode.metadata or {}).get("turn_records"):
                # Pads stay singleton (zero-gradient filler; group count is
                # what matters). Drop any copied raw-token payload.
                if episode.metadata:
                    episode.metadata.pop("turn_records", None)
                fanned_groups.append(grp)
                continue
            fanned = fan_out_thinking_sample(episode, model_adapter.tokenizer)
            n_fanned_episodes += 1
            n_turn_samples += len(fanned)
            fanned_groups.append(fanned)
        grouped_samples = fanned_groups
        # Explicit group IDs avoid collisions after turn fan-out.
        running_index = 0
        for gi, grp in enumerate(grouped_samples):
            for s in grp:
                s.group_id = gi
                s.index = running_index
                running_index += 1
        metrics["rollout/actor_num_turn_samples"] = float(n_turn_samples)
        metrics["rollout/actor_thinking_episodes"] = float(n_fanned_episodes)
        metrics["rollout/actor_turns_per_episode"] = (
            n_turn_samples / n_fanned_episodes if n_fanned_episodes else 0.0
        )

    logger.info(
        f"[SPARE] Rollout {rollout_id} complete: "
        f"{len(trained_actor)} actor, {len(trained_env)} env samples"
        f"{f' (delay={delay}, buffered={sum(len(v) for v in _ENV_SAMPLE_BUFFER.values())})' if delay > 0 else ''}"
    )

    # Fixed model evaluation (if enabled)
    fixed_eval_interval = args.spare_fixed_eval_interval

    if fixed_eval_interval > 0:
        # Validate API key is available upfront — fail loud, not silent
        _api_key_env = spare_config.fixed_eval_api_key_env
        if not os.environ.get(_api_key_env):
            raise RuntimeError(
                f"[SPARE] Fixed model evaluation is enabled (interval={fixed_eval_interval}) "
                f"but {_api_key_env} environment variable is not set. "
                f"Either set the API key or disable fixed eval (--spare-fixed-eval-interval 0)."
            )

        if rollout_id % fixed_eval_interval == 0:
            try:
                fixed_eval_metrics = _run_fixed_model_evaluation(spare_config, games_dir)
                # Merge fixed eval metrics into rollout metrics
                metrics.update(fixed_eval_metrics)
                logger.info(
                    "[SPARE] Fixed model evaluation complete: pass_rate=%.2f, difficulty_score=%.2f",
                    fixed_eval_metrics.get("fixed_eval/overall_pass_rate", 0.0),
                    fixed_eval_metrics.get("fixed_eval/difficulty_score", 0.0),
                )
            except Exception as e:
                logger.error("[SPARE] Fixed model evaluation failed: %s", e)
                metrics["fixed_eval/error"] = 1.0

    return RolloutFnTrainOutput(samples=grouped_samples, metrics=metrics)


def _run_fixed_model_evaluation(config: SpareConfig, games_dir: Path) -> Dict[str, float]:
    """Run fixed model evaluation and return metrics dict.

    Args:
        config: SpareConfig with evaluation parameters
        games_dir: Directory containing game files

    Returns:
        Dictionary of metrics for logging
    """
    result = asyncio.run(
        run_fixed_model_evaluation(
            games_dir=games_dir,
            model=config.fixed_eval_model,
            plays_per_game=config.fixed_eval_plays_per_game,
            max_turns=config.max_turns,
            max_concurrent=config.fixed_eval_max_concurrent,
            temperature=config.fixed_eval_temperature,
            max_tokens=config.fixed_eval_max_tokens,
            api_base_url=config.fixed_eval_api_base_url,
            api_key_env=config.fixed_eval_api_key_env,
        )
    )

    return result.to_metrics_dict()
