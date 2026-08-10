"""Fixed-environment rollout for Slime backend.

Called when --spare-mode fixed_env. Uses external environments (GEM/RLVE)
or pre-generated synthetic games with adaptive difficulty instead of
self-generated games. Actor-only training.
"""

import asyncio
import copy
import logging
import random
from argparse import Namespace
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.types import Sample
from spare.core.difficulty_controller import create_difficulty_controller
from spare.core.envs.env_adapter import EnvironmentAdapter
from spare.core.envs.synthetic_game_adapter import SyntheticGameAdapter
from spare.core.fixed_env_orchestrator import FixedEnvOrchestrator
from spare.core.learning_potential import GameBaselineTracker
from spare.core.types import SpareConfig
from spare.slime.model_adapter import create_slime_model_adapter
from spare.slime.trajectory_converter import trajectory_to_slime_sample

logger = logging.getLogger(__name__)

# Global state for persistent objects
_FIXED_ENV_ORCHESTRATOR = None


def _parse_env_sources(source_specs: List[str]) -> List[EnvironmentAdapter]:
    """Parse --spare-fixed-env-source specs into environment adapters.

    Formats:
    - "rlve:Sorting" — single RLVE environment
    - "rlve:*" — all RLVE environments
    - "gem:game:Sudoku-v0-easy" — single GEM task
    - "gem:game:*" — all GEM tasks in category
    - "gem:*" — all default GEM tasks
    - "game_file:/path/to/dir" — synthetic game .py files from directory

    Args:
        source_specs: List of source specification strings

    Returns:
        List of configured EnvironmentAdapter instances
    """
    adapters: List[EnvironmentAdapter] = []
    gem_tasks: List[str] = []
    rlve_names: List[str] = []

    for spec in source_specs:
        spec = spec.strip()
        if not spec:
            continue

        if spec.startswith("rlve:"):
            name = spec[5:]
            if name == "*":
                rlve_names = ["*"]
            else:
                rlve_names.append(name)
        elif spec.startswith("gem:"):
            task_id = spec[4:]  # Everything after "gem:"
            gem_tasks.append(task_id)
        elif spec.startswith("game_file:"):
            games_dir = spec[len("game_file:"):]
            adapter = SyntheticGameAdapter(games_dir=games_dir)
            adapters.append(adapter)
            logger.info(
                "[FIXED-ENV] Synthetic game adapter: %d games from %s",
                len(adapter.list_environments()), games_dir,
            )
        else:
            logger.warning(
                "Unknown env source format: '%s'. "
                "Expected 'rlve:Name', 'gem:category:TaskName', or 'game_file:/path'.",
                spec,
            )

    # Create RLVE adapter if needed
    if rlve_names:
        try:
            from spare.core.envs.rlve_adapter import RlveEnvironmentAdapter
            adapter = RlveEnvironmentAdapter(
                env_names=None if rlve_names == ["*"] else rlve_names,
            )
            adapters.append(adapter)
            logger.info(
                "[FIXED-ENV] RLVE adapter: %d environments",
                len(adapter.list_environments()),
            )
        except ImportError as e:
            logger.error("[FIXED-ENV] Failed to create RLVE adapter: %s", e)

    # Create GEM adapter if needed
    if gem_tasks:
        try:
            from spare.core.envs.gem_adapter import GemEnvironmentAdapter
            adapter = GemEnvironmentAdapter(task_ids=gem_tasks)
            adapters.append(adapter)
            logger.info(
                "[FIXED-ENV] GEM adapter: %d tasks",
                len(adapter.list_environments()),
            )
        except ImportError as e:
            logger.error("[FIXED-ENV] Failed to create GEM adapter: %s", e)

    return adapters


def _validate_env_entry(
    adapter: EnvironmentAdapter,
    env_id: str,
    difficulty: int,
    num_attempts: int = 3,
) -> bool:
    """Test whether an (env_id, difficulty) pair can generate problems.

    Tries reset() with a few seeds. Returns True if at least one succeeds.
    """
    for attempt_seed in range(num_attempts):
        try:
            instance = adapter.create_instance(env_id, difficulty=difficulty)
            instance.reset(seed=attempt_seed + 1000)
            instance.close()
            return True
        except Exception:
            continue
    return False


def _create_fixed_pool(
    adapters: List[EnvironmentAdapter],
    pool_size: int,
    max_difficulty: int,
    seed: int,
) -> List[Tuple[str, int]]:
    """Create a fixed pool of validated (env_id, difficulty) pairs.

    Randomly samples env_id and difficulty (up to max_difficulty), validates
    each entry with reset(), and replaces failures with new samples.

    Args:
        adapters: Environment adapters to sample from
        pool_size: Number of entries in the pool
        max_difficulty: Maximum difficulty cap
        seed: RNG seed for reproducibility

    Returns:
        List of (env_id, difficulty) tuples (all validated)
    """
    rng = random.Random(seed)

    # Build env_id -> adapter mapping and collect difficulty ranges
    env_to_adapter: dict = {}
    env_info: List[Tuple[str, int, int]] = []  # (env_id, min_d, capped_max_d)
    for adapter in adapters:
        for env_id in adapter.list_environments():
            env_to_adapter[env_id] = adapter
            min_d, max_d = adapter.get_difficulty_range(env_id)
            capped_max_d = min(max_d, max_difficulty)
            if min_d <= capped_max_d:
                env_info.append((env_id, min_d, capped_max_d))

    if not env_info:
        raise ValueError("No environments available for fixed pool sampling")

    logger.info(
        "[FIXED-POOL] Sampling from %d eligible environments (max_difficulty=%d)...",
        len(env_info), max_difficulty,
    )

    # Sample and validate entries, replacing failures
    pool: List[Tuple[str, int]] = []
    num_rejected = 0
    max_total_attempts = pool_size * 10  # safety limit
    total_attempts = 0

    while len(pool) < pool_size and total_attempts < max_total_attempts:
        total_attempts += 1
        env_id, min_d, capped_max_d = rng.choice(env_info)
        difficulty = rng.randint(min_d, capped_max_d)

        adapter = env_to_adapter[env_id]
        if _validate_env_entry(adapter, env_id, difficulty):
            pool.append((env_id, difficulty))
        else:
            num_rejected += 1

    if len(pool) < pool_size:
        logger.warning(
            "[FIXED-POOL] Could only create %d/%d valid entries after %d attempts",
            len(pool), pool_size, total_attempts,
        )

    # Log pool composition for reproducibility
    env_counts: dict = {}
    for env_id, diff in pool:
        key = f"{env_id}@d{diff}"
        env_counts[key] = env_counts.get(key, 0) + 1

    logger.info(
        "[FIXED-POOL] Created pool of %d entries (seed=%d, max_difficulty=%d, "
        "%d rejected during validation):",
        len(pool), seed, max_difficulty, num_rejected,
    )
    for key in sorted(env_counts.keys()):
        logger.info("[FIXED-POOL]   %s: %d entries", key, env_counts[key])

    return pool


def _pre_generate_games(args: Namespace) -> SyntheticGameAdapter:
    """Pre-generate synthetic games at init using the model.

    Creates a SpareOrchestrator, generates N games, discards env trajectories,
    and returns a SyntheticGameAdapter wrapping the saved game files.

    Args:
        args: Slime Namespace with SPARE configuration

    Returns:
        SyntheticGameAdapter wrapping the pre-generated game files
    """
    from spare.core.game_generator import SyntheticGameGenerator
    from spare.core.game_policy import GamePolicy
    from spare.core.learning_potential import LearningPotential
    from spare.core.orchestrator import SpareOrchestrator

    num_games = args.spare_pre_generate_games
    difficulty = args.spare_pre_generate_difficulty

    # Determine skills
    skills = args.spare_pre_generate_skills
    if not skills:
        skills = ["Pattern Recognition", "Mathematical Reasoning"]

    # Validate skills
    available_skills = set(SyntheticGameGenerator.COGNITIVE_SKILLS.keys())
    for skill in skills:
        if skill not in available_skills:
            raise ValueError(
                f"Unknown pre-generate skill: {skill}. "
                f"Available: {sorted(available_skills)}"
            )

    # Create temporary orchestrator for generation only
    model_adapter = create_slime_model_adapter(args)

    gen_config = SpareConfig(
        env_temperature=args.spare_env_temperature,
        env_top_p=args.spare_env_top_p,
        env_top_k=args.spare_env_top_k,
        env_max_tokens=args.spare_env_max_tokens,
        actor_temperature=args.spare_actor_temperature,
        actor_top_p=args.spare_actor_top_p,
        actor_top_k=args.spare_actor_top_k,
        actor_max_tokens=args.spare_actor_max_tokens,
        max_turns=args.spare_max_turns,
        max_context_length=args.spare_max_context_length,
        gamma=args.spare_gamma,
    )

    learning_potentials = {
        skill: LearningPotential(
            gamma_fast=args.spare_gamma1,
            gamma_slow=args.spare_gamma2,
        )
        for skill in skills
    }

    orchestrator = SpareOrchestrator(
        model=model_adapter,
        config=gen_config,
        learning_potentials=learning_potentials,
        game_policy=GamePolicy(),
    )

    # Generate games into a dedicated directory
    games_dir = Path(args.spare_games_dir)
    games_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[PRE-GEN] Generating %d synthetic games (difficulty=%s, skills=%s) into %s",
        num_games, difficulty, skills, games_dir,
    )

    game_files, _ = asyncio.run(
        orchestrator.generate_games_async(
            skills=skills,
            difficulty=difficulty,
            games_dir=games_dir,
            num_games=num_games,
            validate=True,
            rollout_id=0,
            max_attempts=5,
        )
    )

    logger.info(
        "[PRE-GEN] Successfully generated %d/%d games",
        len(game_files), num_games,
    )

    if not game_files:
        raise RuntimeError(
            f"Pre-generation failed: 0/{num_games} games were valid. "
            "Check model and generation settings."
        )

    return SyntheticGameAdapter(games_dir=str(games_dir))


def _get_orchestrator(args: Namespace) -> FixedEnvOrchestrator:
    """Get or create the global FixedEnvOrchestrator."""
    global _FIXED_ENV_ORCHESTRATOR

    if _FIXED_ENV_ORCHESTRATOR is not None:
        return _FIXED_ENV_ORCHESTRATOR

    # Pre-generate games if requested (runs once at init)
    if args.spare_pre_generate_games > 0:
        synthetic_adapter = _pre_generate_games(args)
        # Override env source to use the pre-generated games
        adapters = [synthetic_adapter]
        logger.info(
            "[FIXED-ENV] Using %d pre-generated synthetic games",
            len(synthetic_adapter.list_environments()),
        )
    else:
        # Parse env sources from --spare-fixed-env-source
        adapters = _parse_env_sources(args.spare_fixed_env_source)

    if not adapters:
        raise ValueError(
            "No valid environment adapters found. "
            "Check --spare-fixed-env-source configuration."
        )

    # Create config
    spare_config = SpareConfig(
        actor_temperature=args.spare_actor_temperature,
        actor_top_p=args.spare_actor_top_p,
        actor_top_k=args.spare_actor_top_k,
        actor_max_tokens=args.spare_actor_max_tokens,
        max_turns=args.spare_max_turns,
        max_context_length=args.spare_max_context_length,
        gamma=args.spare_gamma,
        reward_normalization=args.spare_reward_normalization,
        game_baseline_decay=args.spare_game_baseline_decay,
    )

    # Create model adapter
    model_adapter = create_slime_model_adapter(args)

    # Create baseline tracker (needed for ema_baseline)
    baseline_tracker = None
    if spare_config.reward_normalization == "ema_baseline":
        baseline_tracker = GameBaselineTracker(decay=spare_config.game_baseline_decay)

    # Choose between fixed pool (ablation) and adaptive difficulty
    fixed_pool = None
    if args.spare_fixed_pool_size > 0:
        fixed_pool = _create_fixed_pool(
            adapters=adapters,
            pool_size=args.spare_fixed_pool_size,
            max_difficulty=args.spare_fixed_pool_max_difficulty,
            seed=args.spare_fixed_pool_seed,
        )
        logger.info(
            "[FIXED-ENV] Using FIXED POOL of %d entries (no adaptive difficulty)",
            len(fixed_pool),
        )

    # Create difficulty controller
    difficulty_controller = create_difficulty_controller(
        variant=args.spare_difficulty_variant,
        tau_acc=args.spare_difficulty_tau_acc,
        tau_num=args.spare_difficulty_tau_num,
        d_delta=args.spare_difficulty_d_delta,
        gamma_fast=args.spare_lp_gamma_fast,
        gamma_slow=args.spare_lp_gamma_slow,
    )

    _FIXED_ENV_ORCHESTRATOR = FixedEnvOrchestrator(
        model=model_adapter,
        config=spare_config,
        adapters=adapters,
        game_baseline_tracker=baseline_tracker,
        difficulty_controller=difficulty_controller,
        env_regeneration_interval=args.spare_game_regeneration_interval,
        fixed_pool=fixed_pool,
        no_replacement=True,  # Always use no-replacement for fixed pools
        same_problem_groups=args.spare_fixed_env_same_problem,
    )

    return _FIXED_ENV_ORCHESTRATOR


def spare_fixed_env_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Callable,
    evaluation: bool = False,
) -> RolloutFnTrainOutput:
    """Fixed-environment rollout function for Slime.

    Args:
        args: Slime Namespace with SPARE configuration
        rollout_id: Current rollout ID
        data_source: Slime data source (not used for fixed-env)
        evaluation: Whether this is an evaluation rollout

    Returns:
        RolloutFnTrainOutput with actor-only samples
    """
    if evaluation:
        # For eval, check if GEM eval is configured
        if args.spare_gem_eval_config:
            from spare.slime.gem_eval_rollout import spare_gem_eval_rollout
            return spare_gem_eval_rollout(args, rollout_id, data_source)
        # Otherwise, fall back to Slime's built-in eval
        from slime.rollout.sglang_rollout import generate_rollout
        return generate_rollout(args, rollout_id, data_source, evaluation=True)

    del data_source  # Not used in fixed-env mode

    orchestrator = _get_orchestrator(args)

    global_batch_size = args.global_batch_size
    trajectories_per_env = args.spare_trajectories_per_game

    # Overprovision by 25% to account for failures, then trim to exact batch size
    overprovision = int(global_batch_size * 1.25) + 2

    # Collect trajectories
    _, actor_trajectories, collect_info = asyncio.run(
        orchestrator.collect_trajectories_async(
            num_trajectories=overprovision,
            trajectories_per_env=trajectories_per_env,
        )
    )

    # Convert to Slime samples
    all_samples: List[Sample] = []
    for traj in actor_trajectories:
        sample = trajectory_to_slime_sample(
            trajectory=traj,
            index=len(all_samples),
            role="actor",
        )
        all_samples.append(sample)

    # Trim or pad to exact batch size (Slime requires exactly global_batch_size samples)
    if 0 < len(all_samples) < global_batch_size:
        deficit = global_batch_size - len(all_samples)
        n_orig = len(all_samples)
        # Fresh indices make each padded copy a distinct scheduler group.
        for k in range(deficit):
            dup = copy.copy(random.choice(all_samples))
            dup.index = n_orig + k
            dup.group_id = None
            all_samples.append(dup)
        logger.warning(
            "[FIXED-ENV] Upsampling %d -> %d samples (added %d fresh-index copies to fill batch)",
            n_orig, global_batch_size, deficit,
        )
    elif len(all_samples) > global_batch_size:
        all_samples = all_samples[:global_batch_size]

    # Compute metrics
    actor_rewards = [s.reward for s in all_samples]
    original_actor_rewards = [
        s.metadata.get("original_reward", s.reward) for s in all_samples
    ]

    # Per-game grouping for GRPO diagnostics
    from collections import defaultdict as _defaultdict
    game_groups = _defaultdict(list)
    for s in all_samples:
        gf = s.metadata.get("game_file", "unknown")
        game_groups[gf].append(s.reward)

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

    metrics = {
        k.replace(" ", "_"): v for k, v in {
            **collect_info,
            "rollout/num_actor_samples": len(all_samples),
            "rollout/num_env_samples": 0,
            "rollout/mean_return": float(np.mean(actor_rewards)) if actor_rewards else 0.0,
            "rollout/actor_mean_return": float(np.mean(actor_rewards)) if actor_rewards else 0.0,
            "rollout/env_mean_return": 0.0,
            "rollout/actor_mean_original_reward": float(np.mean(original_actor_rewards)) if original_actor_rewards else 0.0,
            "rollout/actor_reward_std": float(np.std(actor_rewards)) if actor_rewards else 0.0,
            "rollout/actor_original_reward_std": float(np.std(original_actor_rewards)) if original_actor_rewards else 0.0,
            "rollout/actor_win_rate": sum(1 for r in original_actor_rewards if r > 0) / max(len(original_actor_rewards), 1),
            "rollout/num_games": num_groups_total,
            "rollout/zero_std_group_pct": num_zero_std_groups / max(num_groups_total, 1),
            "rollout/all_zero_group_pct": num_all_zero_groups / max(num_groups_total, 1),
            "rollout/all_positive_group_pct": num_all_positive_groups / max(num_groups_total, 1),
            "rollout/num_failed": collect_info.get("num_failed", 0),
            "rollout/actor_mean_response_len": float(np.mean([s.response_length for s in all_samples])) if all_samples else 0.0,
            "rollout/actor_truncated_pct": sum(1 for s in all_samples if s.status == Sample.Status.TRUNCATED) / max(len(all_samples), 1),
            "rollout/mode": "fixed_env",
        }.items()
    }

    # Group samples
    grouped_samples = [[sample] for sample in all_samples]

    logger.info(
        "[FIXED-ENV] Rollout %d complete: %d actor samples (batch_size=%d)",
        rollout_id, len(actor_trajectories), global_batch_size,
    )

    return RolloutFnTrainOutput(samples=grouped_samples, metrics=metrics)
