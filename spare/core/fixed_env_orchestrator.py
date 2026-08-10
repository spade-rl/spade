"""Orchestrator for fixed-environment training mode.

Replaces self-generated games with external GEM/RLVE environments or
pre-generated synthetic games. Actor-only training (no env role trajectories).

Pipeline:
1. Select environments (cached at regen intervals, or random each step)
2. Sample difficulty fresh each step (matches RLVE per-prompt sampling)
3. Create env instances with sampled difficulty
4. Play games (reuses play_game_async pattern from SpareOrchestrator)
5. Update difficulty controller with results
6. Normalize actor rewards per-environment
7. Return ([], actor_trajs, info) -- empty env trajs
"""

import asyncio
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm.asyncio import tqdm as atqdm

from spare.core.difficulty_controller import DifficultyController
from spare.core.envs.env_adapter import EnvInstance, EnvironmentAdapter
from spare.core.learning_potential import GameBaselineTracker
from spare.core.model_adapter import ModelAdapter
from spare.core.types import Trajectory, TrajectoryStatus, SpareConfig
from spare.core.utils import (
    get_token_delta,
    normalize_rewards_per_game,
    parse_action,
)

logger = logging.getLogger(__name__)


def _build_initial_prompt(category: str, observation: str) -> str:
    """Build category-appropriate initial prompt."""
    if category == "code":
        return (
            "You are solving a coding problem. Write a Python solution.\n"
            f"Problem: {observation}\n"
            "Please put your solution in a Python code block inside \\boxed{}, "
            "for example: \\boxed{{```python\nyour code\n```}}"
        )
    if category == "math":
        return (
            f"Problem: {observation}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}."
        )
    if category == "eval":
        return (
            f"Question: {observation}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}."
        )
    if category == "rlve":
        return (
            f"{observation}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}."
        )
    if category == "synthetic":
        # Match SpareOrchestrator._apply_game_template("qwen3_game") exactly
        return (
            "You are playing a language game. Make valid actions to win.\n"
            f"Observation: {observation}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}."
        )
    return (
        "You are playing a game. Make valid actions to win.\n"
        f"Observation: {observation}\n"
        "Please reason step by step, and put your final answer within \\boxed{{}}."
    )


def _parse_action_passthrough(response: str) -> str:
    """Pass raw model response as action.

    GEM's concat wrapper and RLVE's verifier handle \\boxed{} extraction
    internally, so we pass the full response through.
    """
    return response


class FixedEnvOrchestrator:
    """Orchestrator for fixed-environment actor-only training.

    Unlike SpareOrchestrator, this does not generate games. It uses
    pre-existing environments from GEM/RLVE or pre-generated synthetic games.

    When a difficulty_controller is provided:
    - Environment IDs are cached at regen intervals (env_regeneration_interval)
    - Difficulties are sampled fresh each step (matches RLVE per-prompt sampling)
    - Controller is updated with results after gameplay

    When no difficulty_controller:
    - Environments are randomly sampled each step
    - No difficulty control (adapters use default difficulty)

    Args:
        model: ModelAdapter for generation
        config: SpareConfig for training parameters
        adapters: List of environment adapters (GEM, RLVE, synthetic, etc.)
        game_baseline_tracker: Optional per-env reward baseline tracker
        difficulty_controller: Optional controller for adaptive difficulty
        env_regeneration_interval: Steps between env re-selection (default 8)
        fixed_pool: Optional pre-sampled list of (env_id, difficulty) pairs.
            When provided, bypasses adaptive difficulty control entirely.
            Each step samples from this pool instead of using the controller.
        no_replacement: If True, sample environments without replacement.
            Once the pool is exhausted, refill from the full list and reshuffle.
        same_problem_groups: If True, the trajectories_per_env plays of each
            selected env share ONE generated problem (per-prompt GRPO group,
            matching RLVE's n-samples-per-prompt semantics). Requires adapter
            support (create_instances_same_problem); adapters without it fall
            back to independent problems.
    """

    def __init__(
        self,
        model: ModelAdapter,
        config: SpareConfig,
        adapters: List[EnvironmentAdapter],
        game_baseline_tracker: Optional[GameBaselineTracker] = None,
        difficulty_controller: Optional[DifficultyController] = None,
        env_regeneration_interval: int = 8,
        fixed_pool: Optional[List[Tuple[str, int]]] = None,
        no_replacement: bool = False,
        same_problem_groups: bool = False,
    ):
        self.model = model
        self.config = config
        self.adapters = adapters
        self.game_baseline_tracker = game_baseline_tracker
        self.difficulty_controller = difficulty_controller
        self.env_regeneration_interval = env_regeneration_interval
        self._fixed_pool = fixed_pool
        self.same_problem_groups = same_problem_groups

        # Build env_id -> adapter mapping
        self._env_to_adapter: Dict[str, EnvironmentAdapter] = {}
        self._all_env_ids: List[str] = []
        for adapter in adapters:
            for env_id in adapter.list_environments():
                self._env_to_adapter[env_id] = adapter
                self._all_env_ids.append(env_id)

        # Initialize difficulty ranges from adapters
        if self.difficulty_controller is not None:
            for env_id, adapter in self._env_to_adapter.items():
                min_d, max_d = adapter.get_difficulty_range(env_id)
                self.difficulty_controller.set_difficulty_range(env_id, min_d, max_d)

        # Caching state for env selection
        self._step_count: int = 0
        self._cached_env_ids: Optional[List[str]] = None

        # No-replacement pool: when enabled, games are removed after sampling
        if no_replacement:
            self._remaining_env_ids = list(self._all_env_ids)
            random.shuffle(self._remaining_env_ids)
            logger.info(
                "[FIXED-ENV] No-replacement mode: pool of %d environments",
                len(self._remaining_env_ids),
            )
        else:
            self._remaining_env_ids = None

        if self._fixed_pool is not None:
            logger.info(
                "[FIXED-ENV] Initialized with FIXED POOL of %d entries "
                "(no adaptive difficulty), %d adapters",
                len(self._fixed_pool), len(adapters),
            )
        else:
            logger.info(
                "[FIXED-ENV] Initialized with %d environments from %d adapters"
                " (difficulty_controller=%s, regen_interval=%d, same_problem_groups=%s)",
                len(self._all_env_ids), len(adapters),
                type(self.difficulty_controller).__name__ if self.difficulty_controller else "None",
                self.env_regeneration_interval,
                self.same_problem_groups,
            )

    def _format_observation(self, observation: str, category: str = "") -> List[Dict[str, str]]:
        """Format observation into OpenAI messages."""
        return [
            {
                "role": "user",
                "content": _build_initial_prompt(category, observation),
            }
        ]

    def _select_env_ids(self, num_envs: int) -> List[str]:
        """Select environment IDs, using cache when difficulty controller is active.

        With a difficulty controller, env_ids are cached and refreshed every
        env_regeneration_interval steps. Without one, random sample each step.

        When no_replacement=True, sampled env IDs are removed from the pool.
        Once the pool is exhausted, it is refilled from the original list.
        """
        if self.difficulty_controller is not None:
            # Re-select on first step or at regen intervals
            if (
                self._cached_env_ids is None
                or (self.env_regeneration_interval > 0
                    and self._step_count % self.env_regeneration_interval == 0)
            ):
                self._cached_env_ids = self.difficulty_controller.select_environments(
                    self._all_env_ids, num_envs
                )
                logger.info(
                    "[FIXED-ENV] Step %d: selected %d environments for next %d steps",
                    self._step_count, len(self._cached_env_ids),
                    self.env_regeneration_interval,
                )
            return self._cached_env_ids

        # No-replacement mode
        if self._remaining_env_ids is not None:
            if len(self._remaining_env_ids) < num_envs:
                # Pool exhausted — refill and reshuffle
                logger.info(
                    "[FIXED-ENV] Pool exhausted (%d remaining, need %d). "
                    "Refilling from %d total environments.",
                    len(self._remaining_env_ids), num_envs, len(self._all_env_ids),
                )
                self._remaining_env_ids = list(self._all_env_ids)
                random.shuffle(self._remaining_env_ids)

            selected = self._remaining_env_ids[:num_envs]
            self._remaining_env_ids = self._remaining_env_ids[num_envs:]
            logger.info(
                "[FIXED-ENV] No-replacement: selected %d envs, %d remaining in pool",
                len(selected), len(self._remaining_env_ids),
            )
            return selected

        # Default: random sample with replacement each step
        if len(self._all_env_ids) <= num_envs:
            return list(self._all_env_ids)
        return random.sample(self._all_env_ids, num_envs)

    def _select_pool_entries(self, num_envs: int) -> List[Tuple[str, int]]:
        """Sample entries from the fixed pool.

        Args:
            num_envs: Number of (env_id, difficulty) entries to sample

        Returns:
            List of (env_id, difficulty) tuples
        """
        k = min(num_envs, len(self._fixed_pool))
        return random.sample(self._fixed_pool, k)


    async def play_env_async(
        self,
        env_instance: EnvInstance,
    ) -> Trajectory:
        """Play a single environment instance asynchronously.

        Same gameplay loop pattern as SpareOrchestrator.play_game_async()
        but without token-in-token-out tracking (simplified for eval-like play).
        We still build full Trajectory with tokens for training.

        Args:
            env_instance: Environment instance to play

        Returns:
            Trajectory from the gameplay
        """
        # GEM/RLVE adapters may own sockets or subprocesses, so reset failures close them.
        try:
            obs, _ = env_instance.reset()
        except Exception as e:
            logger.error("[FIXED-ENV] Reset failed for %s: %s", env_instance.env_id, e)
            try:
                env_instance.close()
            except Exception:
                pass
            return Trajectory(
                status=TrajectoryStatus.FAILED,
                metadata={
                    "error": str(e),
                    "env_id": env_instance.env_id,
                    "skill": env_instance.category,
                },
            )

        messages = self._format_observation(obs, env_instance.category)
        all_tokens: List[int] = self.model.apply_template(messages)
        all_masks: List[int] = []
        all_logprobs: List[float] = []
        assistant_responses: List[str] = []
        rewards: List[float] = []

        # Session ID for consistent hashing: all turns of the same episode
        # route to the same SGLang engine for prefix cache reuse. Mirrors
        # SpareOrchestrator.play_game_async.
        session_id = str(uuid.uuid4())

        # Missing EOS marks a truncated response and prevents invalid turn alignment.
        eos_token_id = self.model.tokenizer.eos_token_id

        terminated = False
        truncated = False
        generation_error = False
        turn = 0

        try:
            for turn in range(self.config.max_turns):
                # Token limit check BEFORE appending new observation
                min_generation_tokens = 64
                tokens_remaining = self.config.max_context_length - len(all_tokens)
                if tokens_remaining <= min_generation_tokens:
                    truncated = True
                    break

                if turn > 0:
                    messages.append({"role": "user", "content": obs})
                    obs_tokens, loss_mask = get_token_delta(
                        tokenizer=self.model.tokenizer,
                        messages=messages,
                    )
                    all_tokens.extend(obs_tokens)
                    all_masks.extend(loss_mask)
                    all_logprobs.extend([0.0] * len(obs_tokens))

                    # Re-check context budget AFTER observation append. Large
                    # late-turn observations can push past max_context_length
                    # on their own.
                    tokens_remaining = self.config.max_context_length - len(all_tokens)
                    if tokens_remaining <= min_generation_tokens:
                        truncated = True
                        break

                # Dynamic max_tokens: never exceed remaining budget.
                effective_max_tokens = min(
                    self.config.actor_max_tokens,
                    tokens_remaining - min_generation_tokens,
                )

                # Generate action
                results = await self.model.generate_async(
                    messages=messages,
                    input_ids=all_tokens,
                    temperature=self.config.actor_temperature,
                    top_p=self.config.actor_top_p,
                    max_tokens=effective_max_tokens,
                    session_id=session_id,
                    role="actor",
                )

                result = results[0]

                # Adapter exceptions are returned as error dictionaries.
                if result.get("error"):
                    logger.warning(
                        "[FIXED-ENV] SGLang error on turn %d for %s: %s",
                        turn, env_instance.env_id, result["error"],
                    )
                    generation_error = True
                    break

                raw_action = result['text']
                response_tokens = result['token_ids']

                messages.append({"role": "assistant", "content": raw_action})
                all_tokens.extend(response_tokens)
                all_masks.extend([1] * len(response_tokens))
                all_logprobs.extend(result['logprobs'])
                assistant_responses.append(raw_action)

                # Detect max_tokens truncation: if the last generated token
                # isn't EOS, the response is incomplete. Mark the trajectory
                # truncated and bail out — no further turns can be appended.
                hit_eos = (
                    len(response_tokens) > 0
                    and eos_token_id is not None
                    and response_tokens[-1] == eos_token_id
                )
                if not hit_eos:
                    logger.info(
                        "[FIXED-ENV] Turn %d for %s hit max_tokens (%d) "
                        "without EOS — truncating trajectory",
                        turn, env_instance.env_id, effective_max_tokens,
                    )
                    truncated = True
                    break

                # Step environment
                # Synthetic games expect \boxed{}-extracted actions (like SpareOrchestrator);
                # GEM/RLVE handle extraction internally and need the full response.
                if env_instance.source == "synthetic":
                    action = parse_action(raw_action, self.config.action_format)
                else:
                    action = _parse_action_passthrough(raw_action)
                obs, reward, terminated, truncated, _ = env_instance.step(action)
                rewards.append(reward)

                if terminated or truncated:
                    break

        except Exception as e:
            logger.error("[FIXED-ENV] Error during gameplay: %s", e)
            return Trajectory(
                status=TrajectoryStatus.FAILED,
                metadata={
                    "error": str(e),
                    "env_id": env_instance.env_id,
                    "skill": env_instance.category,
                },
            )
        finally:
            env_instance.close()

        # If generation failed mid-rollout, return FAILED so downstream
        # filtering drops the trajectory instead of training on a partial batch.
        if generation_error:
            return Trajectory(
                status=TrajectoryStatus.FAILED,
                metadata={
                    "error": "sglang_generation_error",
                    "env_id": env_instance.env_id,
                    "skill": env_instance.category,
                },
            )

        turn += 1
        # Current adapters emit terminal-only rewards, so the final reward is sufficient.
        final_reward = rewards[-1] if rewards else 0.0
        status = (
            TrajectoryStatus.COMPLETED if terminated
            else TrajectoryStatus.TRUNCATED if truncated
            else TrajectoryStatus.COMPLETED
        )

        traj = Trajectory(
            tokens=all_tokens,
            loss_mask=all_masks,
            rollout_log_probs=all_logprobs,
            response="\n".join(assistant_responses),
            response_length=len(all_masks),
            reward=final_reward,
            status=status,
            turn_count=turn,
            messages=messages,
            metadata={
                "game_file": env_instance.env_id,
                "skill": env_instance.category,
                "env_id": env_instance.env_id,
                "difficulty": env_instance.difficulty,
                "source": env_instance.source,
                "rewards": rewards,
            },
        )

        logger.info(
            "[FIXED-ENV] %s: %d turns, reward=%.2f, difficulty=%d",
            env_instance.env_id, turn, final_reward, env_instance.difficulty,
        )
        return traj

    async def collect_trajectories_async(
        self,
        num_trajectories: int,
        trajectories_per_env: int = 1,
    ) -> Tuple[List[Trajectory], List[Trajectory], Dict[str, Any]]:
        """Collect actor trajectories from fixed environments.

        When a difficulty controller is present:
        - Env IDs are cached at regen intervals
        - Difficulties are sampled fresh each step
        - Controller is updated with results after gameplay

        Args:
            num_trajectories: Total number of trajectories to collect
            trajectories_per_env: Number of plays per environment instance

        Returns:
            Tuple of ([], actor_trajectories, info) -- empty env trajs
        """
        info: Dict[str, Any] = {}
        use_fixed_pool = self._fixed_pool is not None

        # Select which environments to play
        num_envs = max(1, num_trajectories // trajectories_per_env)

        # Create env instances
        instances: List[Tuple[EnvInstance, str]] = []
        num_create_failed = 0

        if use_fixed_pool:
            # Fixed pool mode: sample (env_id, difficulty) pairs from pool
            pool_entries = self._select_pool_entries(num_envs)
            for env_id, difficulty in pool_entries:
                adapter = self._env_to_adapter[env_id]
                for _ in range(trajectories_per_env):
                    try:
                        instance = adapter.create_instance(env_id, difficulty=difficulty)
                        instances.append((instance, env_id))
                    except Exception as e:
                        logger.error(
                            "[FIXED-ENV] Failed to create instance for %s (difficulty=%d): %s",
                            env_id, difficulty, e,
                        )
                        num_create_failed += 1
        else:
            # Select environments (cached or fresh)
            selected_env_ids = self._select_env_ids(num_envs)

            for env_id in selected_env_ids:
                adapter = self._env_to_adapter[env_id]
                # Sample difficulty fresh each step (matches RLVE per-prompt sampling)
                if self.difficulty_controller is not None:
                    difficulty = self.difficulty_controller.sample_difficulty(env_id)
                else:
                    difficulty = 0

                if self.same_problem_groups:
                    # One shared problem per env group (per-prompt GRPO,
                    # RLVE n-samples-per-prompt semantics)
                    try:
                        group = adapter.create_instances_same_problem(
                            env_id, difficulty=difficulty, n=trajectories_per_env
                        )
                        instances.extend((inst, env_id) for inst in group)
                    except Exception as e:
                        logger.warning(
                            "[FIXED-ENV] Failed to create same-problem group for %s (d=%d): %s",
                            env_id, difficulty, e,
                        )
                        num_create_failed += trajectories_per_env
                    continue

                for _ in range(trajectories_per_env):
                    try:
                        instance = adapter.create_instance(env_id, difficulty=difficulty)
                        instances.append((instance, env_id))
                    except Exception as e:
                        logger.warning(
                            "[FIXED-ENV] Failed to create instance for %s (d=%d): %s",
                            env_id, difficulty, e,
                        )
                        num_create_failed += 1

        logger.info(
            "[FIXED-ENV] Playing %d instances across %d environments (%d creation failures, pool=%s)",
            len(instances), num_envs, num_create_failed, use_fixed_pool,
        )

        # Play all instances concurrently with progress bar
        tasks = [self.play_env_async(inst) for inst, _ in instances]
        results = await atqdm.gather(
            *tasks, desc="[FIXED-ENV] Collecting trajectories",
        )

        # Collect results and blacklist games that timed out
        actor_trajectories: List[Trajectory] = []
        num_succeeded = 0
        num_failed = num_create_failed

        for i, traj in enumerate(results):
            if traj.status == TrajectoryStatus.FAILED:
                num_failed += 1
                # Blacklist timed-out games so they're never sampled again
                error = traj.metadata.get("error", "")
                env_id = traj.metadata.get("env_id", "")
                if "timed out" in error and env_id:
                    adapter = self._env_to_adapter.get(env_id)
                    if adapter and hasattr(adapter, "blacklist"):
                        adapter.blacklist(env_id)
                continue

            actor_trajectories.append(traj)
            num_succeeded += 1

            if use_fixed_pool:
                # Override game_file to encode (env_id, difficulty) for per-entry
                # reward normalization grouping
                env_id = traj.metadata.get("env_id", "")
                difficulty = traj.metadata.get("difficulty", 0)
                traj.metadata["game_file"] = f"{env_id}@d{difficulty}"

        # Update difficulty controller with results (adaptive mode only)
        if not use_fixed_pool and self.difficulty_controller is not None:
            for traj in actor_trajectories:
                # Use bare env_id (without prefix) for controller tracking
                env_id = traj.metadata.get("env_id", "")
                # Strip prefix (e.g., "rlve:Sorting" -> "Sorting")
                bare_env_id = env_id.split(":", 1)[-1] if ":" in env_id else env_id
                difficulty = traj.metadata.get("difficulty", 0)
                raw_reward = sum(traj.metadata.get("rewards", []))
                self.difficulty_controller.update(bare_env_id, difficulty, raw_reward)

        # Build game_files list for reward normalization
        game_files = list({
            Path(t.metadata.get("game_file", t.metadata.get("env_id", "unknown")))
            for t in actor_trajectories
        })

        # Normalize rewards
        actor_trajectories, stats = normalize_rewards_per_game(
            actor_trajectories,
            game_files,
            game_baseline_tracker=self.game_baseline_tracker,
            reward_normalization=self.config.reward_normalization,
        )

        # Merge reward normalization stats
        for k, v in stats.items():
            if k.startswith("rollout/"):
                info[k] = v

        # Compute zero-std group percentage from per-game reward stats
        zero_std_groups = 0
        total_groups = max(len(stats.get("game", {})), 1)
        for game_file, game_rewards in stats.get("game", {}).items():
            if float(np.std(game_rewards)) == 0.0:
                zero_std_groups += 1
        zero_std_pct = zero_std_groups / total_groups

        # Build metrics
        actor_rewards = [t.reward for t in actor_trajectories]
        raw_rewards = [t.metadata.get("original_reward", t.reward) for t in actor_trajectories]
        difficulties = [t.metadata.get("difficulty", 0) for t in actor_trajectories]
        info.update({
            "num_succeeded": num_succeeded,
            "num_failed": num_failed,
            "num_instances": len(instances),
            "rollout/num_actor_samples": len(actor_trajectories),
            "rollout/num_env_samples": 0,
            "rollout/actor_mean_return": float(np.mean(actor_rewards)) if actor_rewards else 0.0,
            "rollout/avg_raw_reward": float(np.mean(raw_rewards)) if raw_rewards else 0.0,
            "rollout/env_mean_return": 0.0,
            "rollout/mode": "fixed_env",
            "rollout/zero_std_group_pct": zero_std_pct,
            "rollout/zero_std_groups": zero_std_groups,
            "rollout/total_groups": total_groups,
        })

        # Add per-rollout difficulty stats
        if difficulties:
            info["rollout/difficulty_mean"] = float(np.mean(difficulties))
            info["rollout/difficulty_max"] = float(np.max(difficulties))
            info["rollout/difficulty_min"] = float(np.min(difficulties))

        # Add difficulty controller metrics (adaptive mode only)
        if not use_fixed_pool and self.difficulty_controller is not None:
            dc_metrics = self.difficulty_controller.get_metrics()
            info.update(dc_metrics)

        # Add per-env baseline stats if enabled
        if self.game_baseline_tracker is not None:
            baseline_stats = self.game_baseline_tracker.get_stats()
            for k, v in baseline_stats.items():
                info[f"actor/game_baseline/{k}"] = v

        # Compute actor generation entropy (mean negative log prob)
        actor_logprobs = [lp for t in actor_trajectories for lp in t.rollout_log_probs]
        if actor_logprobs:
            info["entropy/actor_mean_neg_logprob"] = -sum(actor_logprobs) / len(actor_logprobs)

        # Increment step counter
        self._step_count += 1

        logger.info(
            "[FIXED-ENV] Step %d complete: %d actor trajectories (%d succeeded, %d failed)",
            self._step_count - 1, len(actor_trajectories), num_succeeded, num_failed,
        )

        return [], actor_trajectories, info

    def collect_trajectories(
        self,
        num_trajectories: int,
        trajectories_per_env: int = 1,
    ) -> Tuple[List[Trajectory], List[Trajectory], Dict[str, Any]]:
        """Synchronous wrapper for collect_trajectories_async."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.collect_trajectories_async(
                num_trajectories=num_trajectories,
                trajectories_per_env=trajectories_per_env,
            )
        )
