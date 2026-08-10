"""SpareOrchestrator: Unified dual-role orchestration for SPARE training.

This orchestrator manages the dual-role interaction pattern:
1. Environment role: Generates game code, receives learning potential as reward
2. Actor role: Plays games, receives game rewards

Architecture:
- Async-first: Core logic is async, with sync wrappers for compatibility
- Shared helpers: Trajectory building, validation, file I/O in utils/game_utils.py
- Batched mode: turn-level generation through the backend adapter
"""

import asyncio
import concurrent.futures
import logging
import math
import os
import random
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Size the environment pool for play concurrency because hung workers cannot be killed.
_ENV_STEP_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=128)
ENV_STEP_TIMEOUT = 30  # seconds — kill games whose step() takes longer
ACTOR_TURN_TIMEOUT = 300  # seconds — bound one generate_async turn (matches env-gen 5-min cap)

from spare.core.types import (
    Trajectory,
    TrajectoryStatus,
    SpareConfig,
)
from spare.core.model_adapter import ModelAdapter
from spare.core.learning_potential import LearningPotential, GameBaselineTracker
from spare.core.game_policy import GamePolicy
from spare.core.envs.synthetic_game_env import make_synthetic_env, criteria_throw_at_reset
from spare.core.prompts import (
    generate_game_generation_prompt,
    generate_multiturn_game_generation_prompt,
    GAME_GENERATION_SYSTEM_PROMPT,
    MULTITURN_GAME_GENERATION_SYSTEM_PROMPT,
    SELF_JUDGE_SYSTEM_PROMPT,
    format_trajectory_for_judge,
    generate_self_judge_prompt,
)
from spare.core.prompts.tool_use_template import (
    TOOL_USE_SKILLS,
    generate_tool_use_prompt,
    get_env_gen_prompt_fn,
)
from spare.core.env_memory import EnvironmentMemory
from spare.core.game_generator import SyntheticGameGenerator
from spare.core.utils import (
    get_token_delta,
    parse_action,
    validate_game,
    save_game_file,
    save_rejected_game,
    extract_game_code,
    cleanup_old_games,
    normalize_rewards_per_game,
    upsample_trajectories,
    build_env_trajectory,
    build_actor_trajectory,
    episode_reward,
    assign_env_rewards,
    assign_env_rewards_regret,
    assign_trajectory_weights,
    compute_env_reward_scale,
)
from spare.core.utils.tool_call_parser import parse_tool_calls
from spare.core.batched import (
    generate_games_batched,
    play_games_batched,
)
from spare.core.hint_generator import HintGenerator, SelfHintGenerator
from spare.core.openrouter_adapter import create_openai_adapter
from spare.core.env_validator import EnvironmentValidator
import numpy as np
logger = logging.getLogger(__name__)

# validate_game() runs the generated game's reset()/step() — give it longer than
# ENV_STEP_TIMEOUT since validation replays several steps back-to-back.
VALIDATE_GAME_TIMEOUT = 60  # seconds — legit games validate in <1s; this only bounds hangs


async def validate_game_async(game_file) -> bool:
    """Run validate_game() in a thread with a hard timeout.

    validate_game() is sync and runs the generated game's reset()/step()
    calls. If the proposer produces a game with an infinite loop in those
    methods, calling validate_game() directly inside an async function
    blocks the entire event loop — preventing outer asyncio.wait_for
    timers from firing. Wrapping it in run_in_executor isolates it on a
    thread; the asyncio.wait_for here aborts after VALIDATE_GAME_TIMEOUT
    so the rollout never stalls on a single bad game.
    """
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_ENV_STEP_EXECUTOR, validate_game, game_file),
            timeout=VALIDATE_GAME_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"validate_game timed out after {VALIDATE_GAME_TIMEOUT}s on "
            f"{getattr(game_file, 'name', game_file)} — treating as invalid"
        )
        return False


class SpareOrchestrator:
    """Orchestrates dual-role SPARE training with unified logic for all backends.

    Architecture:
    - Async-first: play_game_async and generate_games_async are the core implementations
    - Sync wrappers: play_game() and generate_games() wrap async for compatibility
    - Batched mode: Uses batched.py functions for vLLM turn-level batching
    - Shared helpers: Uses utils/game_utils.py for trajectory building, validation, etc.
    """

    def __init__(
        self,
        model: ModelAdapter,
        config: SpareConfig,
        learning_potentials: Dict[str, LearningPotential],
        game_policy: GamePolicy,
        env_memory: Optional[EnvironmentMemory] = None,
    ):
        """Initialize the orchestrator.

        Args:
            model: Model adapter for text generation
            config: Configuration for dual-role orchestration
            learning_potentials: Per-skill learning potential trackers
            game_policy: Game selection policy
            env_memory: Optional environment memory; when set, high-regret
                past games are injected as few-shot seeds into generation
        """
        self.model = model
        self.config = config
        self.learning_potentials = learning_potentials
        self.game_policy = game_policy
        self.env_memory = env_memory

        # Per-role chat template kwargs overrides (e.g., enable_thinking)
        self._env_template_kwargs = (
            {"enable_thinking": config.env_enable_thinking}
            if config.env_enable_thinking is not None else None
        )
        self._actor_template_kwargs = (
            {"enable_thinking": config.actor_enable_thinking}
            if config.actor_enable_thinking is not None else None
        )

        # Per-game baseline tracker for actor rewards (used in ema_baseline mode)
        if config.reward_normalization == "ema_baseline":
            self.game_baseline_tracker: Optional[GameBaselineTracker] = GameBaselineTracker(
                decay=config.game_baseline_decay
            )
        else:
            self.game_baseline_tracker = None

        # Hint generator for regret-based env reward (also used by the blend variant)
        if config.env_reward_variant in ("regret_based", "blend"):
            if config.hint_mode == "self":
                # Self-hint: use the training model itself
                self.hint_generator = SelfHintGenerator(
                    model_adapter=model,
                    temperature=config.hint_temperature,
                    max_tokens=config.hint_max_tokens,
                    game_type=config.game_type,
                )
                logger.info(
                    f"[HINT] Initialized self-hint generator (training model), "
                    f"game_type={config.game_type}"
                )
            else:
                # External hint: use OpenAI/OpenRouter API
                hint_adapter = create_openai_adapter(
                    model=config.hint_model,
                    api_key_env=config.hint_api_key_env,
                    base_url=config.hint_api_base_url,
                    temperature=config.hint_temperature,
                    max_tokens=config.hint_max_tokens,
                )
                self.hint_generator = HintGenerator(
                    model_adapter=hint_adapter,
                    temperature=config.hint_temperature,
                    max_tokens=config.hint_max_tokens,
                    game_type=config.game_type,
                )
                logger.info(
                    f"[HINT] Initialized external hint generator ({config.hint_model}), "
                    f"game_type={config.game_type}"
                )
            self._cached_regret: Dict[str, float] = {}
        else:
            self.hint_generator = None
            self._cached_regret = {}

        # Track timed-out games so their environment trajectories can be penalized.
        self._step_timeout_games: set = set()

        # Blacklist games whose executor threads hang; cancellation cannot stop them.
        self._hung_game_blacklist: set = set()

        # Environment validator for rejection sampling
        # "self" mode: uses the training model; otherwise: uses external model via OpenAI/OpenRouter
        if config.use_env_validator:
            if config.env_validator_model == "self":
                validator_adapter = model  # Use the training model
                logger.info("[ENV-VALIDATOR] Initialized with training model (self)")
            else:
                validator_adapter = create_openai_adapter(
                    model=config.env_validator_model,
                    api_key_env=config.env_validator_api_key_env,
                    base_url=config.env_validator_api_base_url,
                    temperature=config.env_validator_temperature,
                    max_tokens=config.env_validator_max_tokens,
                )
                logger.info(
                    f"[ENV-VALIDATOR] Initialized with external model={config.env_validator_model}"
                )
            self.env_validator: Optional[EnvironmentValidator] = EnvironmentValidator(
                model_adapter=validator_adapter,
                temperature=config.env_validator_temperature,
                max_tokens=config.env_validator_max_tokens,
                game_type=config.game_type,
            )
            # Per-rollout counters (reset each rollout in collect_trajectories)
            self._env_validator_accepted = 0
            self._env_validator_rejected = 0
            self._env_validator_errors = 0
        else:
            self.env_validator = None
            self._env_validator_accepted = 0
            self._env_validator_rejected = 0
            self._env_validator_errors = 0
    # =========================================================================
    # ASYNC CORE - Primary implementations (Slime/Tinker backends)
    # =========================================================================

    def _apply_game_template(
        self, messages: List[Dict[str, str]], template_name: str = "qwen3_game"
    ) -> List[Dict[str, str]]:
        user_content = None
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # Build messages for tokenizer based on template
        if template_name == "qwen3_game":
            # Actor template: Add game-playing instructions
            chat_messages = [
                {
                    "role": "user",
                    "content": (
                        f"You are playing a language game. Make valid actions to win.\n"
                        f"Observation: {user_content}\n"
                        f"Please reason step by step, and put your final answer within \\boxed{{}}."
                    )
                }
            ]
        elif template_name == "qwen3_game_generation":
            # Environment generation template: Add system prompt
            chat_messages = [
                {"role": "system", "content": GAME_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        elif template_name == "qwen3_multiturn_game_generation":
            # Multi-turn environment generation template
            chat_messages = [
                {"role": "system", "content": MULTITURN_GAME_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        else:
            # Default: pass through as user message
            chat_messages = [{"role": "user", "content": user_content}]
        return chat_messages

    async def play_game_async(
        self,
        game_file_path: str,
        skill: str,
        env=None,
    ) -> Trajectory:
        """Play a single game asynchronously (actor role).

        This is the core gameplay implementation. Uses generate_async for
        true async execution, optimal for HTTP backends like Slime.

        Args:
            game_file_path: Path to game file to play
            env: Optional pre-initialized environment

        Returns:
            Episode Trajectory containing the full gameplay interaction
        """
        should_close_env = env is None

        # Fail fast on games that already hung once — loading them again would
        # leak another unkillable executor thread (see _hung_game_blacklist).
        if game_file_path in self._hung_game_blacklist:
            return Trajectory(
                status=TrajectoryStatus.FAILED,
                metadata={
                    "error": "blacklisted after earlier hang",
                    "game_file": game_file_path,
                    "skill": skill,
                },
            )

        # Read game code once for Weave tracing metadata
        try:
            game_code = Path(game_file_path).read_text()
        except Exception:
            game_code = None

        try:
            if env is None:
                logger.info(f"[GAME] Loading: {Path(game_file_path).name}")
                loop = asyncio.get_event_loop()

                def _load_and_reset():
                    # respect_game_max_turns: use the game's own designed pacing, capped
                    # at config.max_turns (the budget). See SyntheticGameEnv.
                    e = make_synthetic_env(
                        game_file_path, max_turns=self.config.max_turns,
                        respect_game_max_turns=True,
                    )
                    o, _ = e.reset()
                    return e, o

                env, obs = await asyncio.wait_for(
                    loop.run_in_executor(_ENV_STEP_EXECUTOR, _load_and_reset),
                    timeout=ENV_STEP_TIMEOUT,
                )
            else:
                obs, _ = env.reset()
        except asyncio.TimeoutError:
            logger.error(
                f"[GAME] Loading/reset timed out after {ENV_STEP_TIMEOUT}s "
                f"on {Path(game_file_path).name} — aborting game and blacklisting"
            )
            self._hung_game_blacklist.add(game_file_path)
            self._step_timeout_games.add(game_file_path)
            return Trajectory(status=TrajectoryStatus.FAILED, metadata={"error": "load/reset timeout", "game_file": game_file_path, "skill": skill})
        except Exception as e:
            logger.error(f"Failed to create/reset game environment: {e}")
            return Trajectory(status=TrajectoryStatus.FAILED, metadata={"error": str(e), "game_file": game_file_path, "skill": skill})

        # Dispatch to tool-use path if environment provides tool schemas
        if hasattr(env, 'get_tools') and env.get_tools():
            return await self._play_game_tool_use_async(
                env=env,
                obs=obs,
                game_file_path=game_file_path,
                skill=skill,
                game_code=game_code,
                should_close_env=should_close_env,
            )

        # Actor thinking uses per-turn records and canonical prompt re-rendering.
        if self.config.actor_enable_thinking:
            return await self._play_game_thinking_async(
                env=env,
                obs=obs,
                game_file_path=game_file_path,
                skill=skill,
                game_code=game_code,
                should_close_env=should_close_env,
            )

        # Initialize tracking
        messages: List[Dict[str, str]] = self._apply_game_template(
            messages=[{"role": "user", "content": obs}],
            template_name="qwen3_game"
        )
        all_tokens: List[int] = self.model.apply_template(
            messages,
            chat_template_kwargs_override=self._actor_template_kwargs,
        )
        # Initialize masks and logprobs with zeros for prompt tokens
        # This marks them as non-trainable (loss_mask=0) with no logprobs
        all_masks: List[int] = []
        all_logprobs: List[float] = []
        assistant_responses: List[str] = []
        rewards: List[float] = []

        # Session ID for consistent hashing: all turns of the same game
        # route to the same SGLang engine for prefix cache reuse
        session_id = str(uuid.uuid4())

        # Missing EOS marks a truncated response and ends the episode.
        eos_token_id = self.model.tokenizer.eos_token_id

        terminated = False
        truncated = False
        generation_error = False
        turn = 0

        try:
            for turn in range(self.config.max_turns):
                # Check remaining context budget before generation
                min_generation_tokens = 64
                tokens_remaining = self.config.max_context_length - len(all_tokens)
                if tokens_remaining <= min_generation_tokens:
                    logger.warning(
                        f"[GAME] Context full at turn {turn}: {len(all_tokens)} tokens, "
                        f"only {tokens_remaining} remaining (need {min_generation_tokens}), truncating"
                    )
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

                    # Re-check after adding observation tokens
                    tokens_remaining = self.config.max_context_length - len(all_tokens)
                    if tokens_remaining <= min_generation_tokens:
                        logger.warning(
                            f"[GAME] Context full after obs at turn {turn}: {len(all_tokens)} tokens, truncating"
                        )
                        truncated = True
                        break

                # Dynamic max_tokens: use remaining budget, capped at actor_max_tokens
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
                    top_k=self.config.actor_top_k,
                    max_tokens=effective_max_tokens,
                    session_id=session_id,
                    game_code=game_code,
                    role="actor",
                )

                if not results:
                    logger.warning(f"[GAME] Empty results on turn {turn} — aborting")
                    generation_error = True
                    break
                result = results[0]

                # Adapter exceptions are returned as error dictionaries.
                if result.get("error"):
                    logger.warning(
                        f"[GAME] SGLang error on turn {turn}: {result['error']}"
                    )
                    generation_error = True
                    break

                raw_action = result['text']
                response_tokens = result['token_ids']

                # Track assistant response
                messages.append({"role": "assistant", "content": raw_action})
                all_tokens.extend(response_tokens)
                all_masks.extend([1] * len(response_tokens))
                all_logprobs.extend(result['logprobs'])
                assistant_responses.append(raw_action)

                # Missing EOS marks truncation and prevents invalid turn alignment.
                hit_eos = (
                    len(response_tokens) > 0
                    and eos_token_id is not None
                    and response_tokens[-1] == eos_token_id
                )
                if not hit_eos:
                    logger.info(
                        f"[GAME] Turn {turn} hit max_tokens "
                        f"({effective_max_tokens}) without EOS — truncating"
                    )
                    truncated = True
                    break

                # Step environment (with timeout to catch infinite loops)
                action = parse_action(raw_action, self.config.action_format)
                loop = asyncio.get_event_loop()
                try:
                    obs, reward, terminated, truncated, _ = await asyncio.wait_for(
                        loop.run_in_executor(_ENV_STEP_EXECUTOR, env.step, action),
                        timeout=ENV_STEP_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"[GAME] env.step() timed out after {ENV_STEP_TIMEOUT}s "
                        f"on {Path(game_file_path).name} turn {turn} — aborting game"
                    )
                    # Degenerate game code (step() hangs): flag for the bounded -0.5
                    # proposer penalty in the reward stage.
                    self._step_timeout_games.add(game_file_path)
                    self._hung_game_blacklist.add(game_file_path)
                    return Trajectory(
                        status=TrajectoryStatus.FAILED,
                        metadata={"error": "env.step() timeout", "game_file": game_file_path, "skill": skill},
                    )
                rewards.append(reward)

                if terminated or truncated:
                    break

        except Exception as e:
            logger.error(f"Error during game play: {e}")
            return Trajectory(status=TrajectoryStatus.FAILED, metadata={"error": str(e), "game_file": game_file_path, "skill": skill})
        finally:
            if should_close_env:
                env.close()

        # If the SGLang adapter reported an error mid-rollout, drop the
        # trajectory so downstream filters don't treat partial data as valid.
        if generation_error:
            return Trajectory(
                status=TrajectoryStatus.FAILED,
                metadata={
                    "error": "sglang_generation_error",
                    "game_file": game_file_path,
                    "skill": skill,
                },
            )

        turn += 1
        traj = build_actor_trajectory(
            messages=messages,
            all_tokens=all_tokens,
            all_masks=all_masks,
            all_logprobs=all_logprobs,
            assistant_responses=assistant_responses,
            rewards=rewards,
            game_file_path=game_file_path,
            skill=skill,
            turn=turn,
            terminated=terminated,
            truncated=truncated,
        )

        logger.info(f"[GAME] Ended: {turn} turns, reward={traj.reward:.2f}")
        return traj

    async def _play_game_thinking_async(
        self,
        env,
        obs: str,
        game_file_path: str,
        skill: str,
        game_code: Optional[str],
        should_close_env: bool,
    ) -> Trajectory:
        """Play a game with a THINKING actor (per-turn re-render + TurnRecord capture).

        Hybrid Qwen3 chat templates STRIP ``<think>...</think>`` blocks from
        prior assistant turns when re-rendering history, so the legacy
        accumulated token-in-token-out path diverges from the canonical
        rendering after turn 1. This path mirrors slime/agent's adapter flow
        (``slime.agent.adapters.common.render_token_ids`` +
        ``call_sglang_generate``) instead:

        - Each turn's prompt is re-rendered from scratch with
          ``apply_template`` on the full message history. The template itself
          strips prior turns' think blocks, so generation is conditioned on
          exactly the canonical deployment-time context.
        - Each generation is captured as a raw turn record
          (prompt_ids / output_ids / output_log_probs / finish_reason —
          the exact fields of ``slime.agent.trajectory.TurnRecord``, stored
          as plain dicts because spare/core must not import slime).
        - Generation stays token-in-token-out WITHIN a turn: the recorded
          prompt ids are the exact ids sent to the engine and the output ids
          come back verbatim with their logprobs, so training rows match the
          generation context exactly. Only the (loss-mask-0) history region
          is re-tokenized across turns.

        The returned episode Trajectory carries the per-turn records in
        ``metadata["turn_records"]``; the slime backend fans them out into
        one training Sample per turn (all sharing the episode's reward and
        group). The trajectory-level token fields hold the FINAL turn's
        (prompt + output) as a well-formed single-sequence fallback view.

        Returns:
            Episode Trajectory with per-turn records in metadata.
        """
        messages: List[Dict[str, str]] = self._apply_game_template(
            messages=[{"role": "user", "content": obs}],
            template_name="qwen3_game",
        )
        turn_records: List[Dict[str, Any]] = []
        assistant_responses: List[str] = []
        rewards: List[float] = []

        # Session ID for consistent hashing: all turns of the same game
        # route to the same SGLang engine for prefix cache reuse (the
        # rendered history prefix is stable across turns).
        session_id = str(uuid.uuid4())
        eos_token_id = self.model.tokenizer.eos_token_id

        terminated = False
        truncated = False
        generation_error = False
        turn = 0
        prompt_ids: List[int] = []

        try:
            for turn in range(self.config.max_turns):
                if turn > 0:
                    messages.append({"role": "user", "content": obs})

                # Canonical re-render: prior assistant turns are rendered
                # WITHOUT their think blocks by the chat template itself.
                prompt_ids = self.model.apply_template(
                    messages,
                    chat_template_kwargs_override=self._actor_template_kwargs,
                )

                # Check remaining context budget before generation
                min_generation_tokens = 64
                tokens_remaining = self.config.max_context_length - len(prompt_ids)
                if tokens_remaining <= min_generation_tokens:
                    logger.warning(
                        f"[GAME-THINK] Context full at turn {turn}: {len(prompt_ids)} prompt tokens, "
                        f"only {tokens_remaining} remaining (need {min_generation_tokens}), truncating"
                    )
                    truncated = True
                    break

                # Dynamic max_tokens: per-turn thinking spends the existing
                # actor_max_tokens budget (no separate thinking knob).
                effective_max_tokens = min(
                    self.config.actor_max_tokens,
                    tokens_remaining - min_generation_tokens,
                )

                # Generate action (TITO within the turn: input is prompt_ids)
                results = await self.model.generate_async(
                    messages=messages,
                    input_ids=prompt_ids,
                    temperature=self.config.actor_temperature,
                    top_p=self.config.actor_top_p,
                    top_k=self.config.actor_top_k,
                    max_tokens=effective_max_tokens,
                    session_id=session_id,
                    game_code=game_code,
                    role="actor",
                )

                if not results:
                    logger.warning(f"[GAME-THINK] Empty results on turn {turn} — aborting")
                    generation_error = True
                    break
                result = results[0]
                if result.get("error"):
                    logger.warning(
                        f"[GAME-THINK] SGLang error on turn {turn}: {result['error']}"
                    )
                    generation_error = True
                    break

                raw_action = result['text']
                response_tokens = result['token_ids']

                # A missing EOS marks the turn truncated, including mid-thought
                # responses; compact filtering can later mask the record.
                hit_eos = (
                    len(response_tokens) > 0
                    and eos_token_id is not None
                    and response_tokens[-1] == eos_token_id
                )

                # Strip textual EOS before templating; token records retain the real EOS.
                msg_content = raw_action
                eos_text = self.model.tokenizer.eos_token
                if hit_eos and eos_text and msg_content.endswith(eos_text):
                    msg_content = msg_content[: -len(eos_text)]

                messages.append({"role": "assistant", "content": msg_content})
                turn_records.append({
                    "prompt_ids": list(prompt_ids),
                    "output_ids": list(response_tokens),
                    "output_log_probs": list(result['logprobs']),
                    "finish_reason": "stop" if hit_eos else "length",
                })
                assistant_responses.append(raw_action)

                if not hit_eos:
                    logger.info(
                        f"[GAME-THINK] Turn {turn} hit max_tokens "
                        f"({effective_max_tokens}) without EOS — truncating"
                    )
                    truncated = True
                    break

                # Parse only visible content after any Qwen thinking block.
                if '</think>' in raw_action:
                    visible_action = raw_action.split('</think>')[-1]
                else:
                    visible_action = raw_action
                action = parse_action(visible_action, self.config.action_format)

                # Step environment (with timeout to catch infinite loops)
                loop = asyncio.get_event_loop()
                try:
                    obs, reward, terminated, truncated, _ = await asyncio.wait_for(
                        loop.run_in_executor(_ENV_STEP_EXECUTOR, env.step, action),
                        timeout=ENV_STEP_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"[GAME-THINK] env.step() timed out after {ENV_STEP_TIMEOUT}s "
                        f"on {Path(game_file_path).name} turn {turn} — aborting game"
                    )
                    # Penalize the proposer and prevent further plays of a hung game.
                    self._step_timeout_games.add(game_file_path)
                    self._hung_game_blacklist.add(game_file_path)
                    return Trajectory(
                        status=TrajectoryStatus.FAILED,
                        metadata={"error": "env.step() timeout", "game_file": game_file_path, "skill": skill},
                    )
                rewards.append(reward)

                if terminated or truncated:
                    break

        except Exception as e:
            logger.error(f"Error during thinking game play: {e}")
            return Trajectory(status=TrajectoryStatus.FAILED, metadata={"error": str(e), "game_file": game_file_path, "skill": skill})
        finally:
            if should_close_env:
                env.close()

        if generation_error:
            return Trajectory(
                status=TrajectoryStatus.FAILED,
                metadata={
                    "error": "sglang_generation_error",
                    "game_file": game_file_path,
                    "skill": skill,
                },
            )

        turn += 1

        # Keep a valid final-turn fallback; Slime reconstructs per-turn samples.
        if turn_records:
            last = turn_records[-1]
            all_tokens = list(last["prompt_ids"]) + list(last["output_ids"])
            all_masks = [1] * len(last["output_ids"])
            all_logprobs = list(last["output_log_probs"])
        else:
            all_tokens = list(prompt_ids)
            all_masks = []
            all_logprobs = []

        traj = build_actor_trajectory(
            messages=messages,
            all_tokens=all_tokens,
            all_masks=all_masks,
            all_logprobs=all_logprobs,
            assistant_responses=assistant_responses,
            rewards=rewards,
            game_file_path=game_file_path,
            skill=skill,
            turn=turn,
            terminated=terminated,
            truncated=truncated,
        )
        traj.metadata["turn_records"] = turn_records
        traj.metadata["actor_thinking"] = True

        logger.info(
            f"[GAME-THINK] Ended: {turn} turns, reward={traj.reward:.2f}, "
            f"{len(turn_records)} turn records"
        )
        return traj

    async def _play_game_tool_use_async(
        self,
        env,
        obs: str,
        game_file_path: str,
        skill: str,
        game_code: Optional[str],
        should_close_env: bool,
    ) -> Trajectory:
        """Play a tool-use game using native tool calling.

        Uses apply_chat_template(tools=...) for proper tool schema injection,
        parse_tool_calls() for response parsing, and role="tool" messages for
        observations. This aligns with tool-use evaluation harnesses such as BFCL.
        """
        tools = env.get_tools()
        tool_kwargs = {"tools": tools}

        # Build initial messages (no game template wrapper — tools handle formatting)
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": obs},
        ]
        all_tokens: List[int] = self.model.apply_template(
            messages,
            chat_template_kwargs_override={
                # Actor template overrides are optional for non-thinking runs.
                **(self._actor_template_kwargs or {}),
                **tool_kwargs,
            },
        )
        all_masks: List[int] = []
        all_logprobs: List[float] = []
        assistant_responses: List[str] = []
        rewards: List[float] = []

        session_id = str(uuid.uuid4())
        eos_token_id = self.model.tokenizer.eos_token_id

        terminated = False
        truncated = False
        generation_error = False
        turn = 0

        try:
            for turn in range(self.config.max_turns):
                # Check remaining context budget
                min_generation_tokens = 64
                tokens_remaining = self.config.max_context_length - len(all_tokens)
                if tokens_remaining <= min_generation_tokens:
                    truncated = True
                    break

                if turn > 0:
                    # Observation tokens already added after tool execution below
                    tokens_remaining = self.config.max_context_length - len(all_tokens)
                    if tokens_remaining <= min_generation_tokens:
                        truncated = True
                        break

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
                    top_k=self.config.actor_top_k,
                    max_tokens=effective_max_tokens,
                    session_id=session_id,
                    game_code=game_code,
                    role="actor",
                )

                if not results:
                    generation_error = True
                    break
                result = results[0]
                if result.get("error"):
                    generation_error = True
                    break

                raw_action = result['text']
                response_tokens = result['token_ids']

                # Track assistant response tokens (trainable)
                messages.append({"role": "assistant", "content": raw_action})
                all_tokens.extend(response_tokens)
                all_masks.extend([1] * len(response_tokens))
                all_logprobs.extend(result['logprobs'])
                assistant_responses.append(raw_action)

                # Check for max_tokens truncation
                hit_eos = (
                    len(response_tokens) > 0
                    and eos_token_id is not None
                    and response_tokens[-1] == eos_token_id
                )
                if not hit_eos:
                    truncated = True
                    break

                # Parse tool calls from response
                parsed = parse_tool_calls(raw_action, tools)

                # Check for <answer> tag (tool-use envs still support answer submission)
                import re as _re
                answer_match = _re.search(r'<answer>(.*?)</answer>', raw_action, _re.DOTALL)
                if answer_match:
                    # Direct answer submission — step the env with raw action
                    loop = asyncio.get_event_loop()
                    try:
                        obs, reward, terminated, truncated, _ = await asyncio.wait_for(
                            loop.run_in_executor(_ENV_STEP_EXECUTOR, env.step, raw_action),
                            timeout=ENV_STEP_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        # Degenerate game code (step() hangs): flag for the -0.5 penalty.
                        self._step_timeout_games.add(game_file_path)
                        self._hung_game_blacklist.add(game_file_path)
                        return Trajectory(
                            status=TrajectoryStatus.FAILED,
                            metadata={"error": "env.step() timeout", "game_file": game_file_path, "skill": skill},
                        )
                    rewards.append(reward)
                    if terminated or truncated:
                        break
                    # Add observation as user message
                    messages.append({"role": "user", "content": obs})
                    obs_tokens, loss_mask = get_token_delta(
                        tokenizer=self.model.tokenizer,
                        messages=messages,
                        tools=tools,
                    )
                    all_tokens.extend(obs_tokens)
                    all_masks.extend(loss_mask)
                    all_logprobs.extend([0.0] * len(obs_tokens))
                    continue

                if not parsed["calls"]:
                    # Optionally mask malformed tool-call tokens while retaining context.
                    _malformed_tc = (
                        os.getenv("SPARE_MASK_MALFORMED_TOOLCALL") == "1"
                        and "<tool_call>" in raw_action
                    )
                    if _malformed_tc:
                        for _i in range(1, len(response_tokens) + 1):
                            all_masks[-_i] = 0
                    # Step env with raw action so it can provide guidance
                    loop = asyncio.get_event_loop()
                    try:
                        obs, reward, terminated, truncated, _ = await asyncio.wait_for(
                            loop.run_in_executor(_ENV_STEP_EXECUTOR, env.step, raw_action),
                            timeout=ENV_STEP_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        # Degenerate game code (step() hangs): flag for the -0.5 penalty.
                        self._step_timeout_games.add(game_file_path)
                        self._hung_game_blacklist.add(game_file_path)
                        return Trajectory(
                            status=TrajectoryStatus.FAILED,
                            metadata={"error": "env.step() timeout", "game_file": game_file_path, "skill": skill},
                        )
                    rewards.append(reward)
                    if terminated or truncated:
                        break
                    if _malformed_tc:
                        # Error-as-feedback (ToRL): surface the canonical schema so the
                        # actor can recover next turn; the recovered well-formed call is
                        # masked=1 and reinforced normally (positive pull to canonical).
                        obs = (
                            f"{obs}\n\n[FORMAT] Your previous <tool_call> could not be "
                            'parsed. Emit exactly: <tool_call>{"name": "<tool_name>", '
                            '"arguments": {<args>}}</tool_call>'
                        )
                    messages.append({"role": "user", "content": obs})
                    obs_tokens, loss_mask = get_token_delta(
                        tokenizer=self.model.tokenizer,
                        messages=messages,
                        tools=tools,
                    )
                    all_tokens.extend(obs_tokens)
                    all_masks.extend(loss_mask)
                    all_logprobs.extend([0.0] * len(obs_tokens))
                    continue

                # Execute tool calls via env.execute_tool()
                for call in parsed["calls"]:
                    loop = asyncio.get_event_loop()
                    try:
                        tool_result = await asyncio.wait_for(
                            loop.run_in_executor(
                                _ENV_STEP_EXECUTOR,
                                env.execute_tool,
                                call["name"],
                                call["arguments"],
                            ),
                            timeout=ENV_STEP_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        return Trajectory(
                            status=TrajectoryStatus.FAILED,
                            metadata={"error": "execute_tool timeout", "game_file": game_file_path, "skill": skill},
                        )

                    # Add tool result as role="tool" message
                    tool_content = tool_result.get("output", "")
                    messages.append({"role": "tool", "content": tool_content})
                    obs_tokens, loss_mask = get_token_delta(
                        tokenizer=self.model.tokenizer,
                        messages=messages,
                        tools=tools,
                    )
                    all_tokens.extend(obs_tokens)
                    all_masks.extend(loss_mask)
                    all_logprobs.extend([0.0] * len(obs_tokens))

                # No reward until answer is submitted (tool calls are intermediate)
                rewards.append(0.0)

        except Exception as e:
            logger.error(f"Error during tool-use game play: {e}")
            return Trajectory(status=TrajectoryStatus.FAILED, metadata={"error": str(e), "game_file": game_file_path, "skill": skill})
        finally:
            if should_close_env:
                env.close()

        if generation_error:
            return Trajectory(
                status=TrajectoryStatus.FAILED,
                metadata={"error": "generation_error", "game_file": game_file_path, "skill": skill},
            )

        turn += 1
        traj = build_actor_trajectory(
            messages=messages,
            all_tokens=all_tokens,
            all_masks=all_masks,
            all_logprobs=all_logprobs,
            assistant_responses=assistant_responses,
            rewards=rewards,
            game_file_path=game_file_path,
            skill=skill,
            turn=turn,
            terminated=terminated,
            truncated=truncated,
        )
        logger.info(f"[GAME-TOOLS] Ended: {turn} turns, reward={traj.reward:.2f}")
        return traj

    async def _generate_single_game_async(
        self,
        skill: str,
        difficulty: str,
        games_dir: Path,
        rollout_id: int,
        index: int,
        validate: bool,
        cache_step_dir: Optional[Path],
        max_attempts: int = 8,
    ) -> Optional[Tuple[Path, Trajectory]]:
        """Generate a single game with retry logic.

        Args:
            skill: Cognitive skill to target
            difficulty: Difficulty level
            games_dir: Directory to save games
            rollout_id: Current rollout ID
            index: Index for file naming
            validate: Whether to validate game
            cache_step_dir: Optional cache directory
            max_attempts: Maximum generation attempts

        Returns:
            Tuple of (game_file, Trajectory) or None if all attempts fail
        """
        game_file = None

        # Repair mode trains only the initial generation turn.
        if self.config.env_repair_turns > 0:
            return await self._generate_single_game_with_repair(
                skill, difficulty, games_dir, rollout_id, index, validate, cache_step_dir,
            )

        for attempt in range(max_attempts):

            game_type = getattr(self.config, "game_type", "cognitive")
            template_name = self.config.env_generation_template

            if game_type == "tool_use":
                skill_info = TOOL_USE_SKILLS[skill]
                prompt_content = generate_tool_use_prompt(skill, skill_info, difficulty)
                # Default to Qwen3 tool_use template unless overridden
                if template_name == "qwen3_game_generation":
                    template_name = "qwen3_tool_use_game_generation"
            elif template_name == "qwen3_multiturn_game_generation":
                skill_info = SyntheticGameGenerator.COGNITIVE_SKILLS[skill]
                prompt_content = generate_multiturn_game_generation_prompt(
                    skill, skill_info, difficulty, max_turns=self.config.max_turns,
                )
            else:
                skill_info = SyntheticGameGenerator.COGNITIVE_SKILLS[skill]
                prompt_content = generate_game_generation_prompt(skill, skill_info, difficulty)
            prompt_content = self._augment_prompt_with_memory(prompt_content, skill)
            messages: List[Dict[str, str]] = self._apply_game_template(
                messages=[{"role": "user", "content": prompt_content}],
                template_name=template_name,
            )
            all_tokens: List[int] = self.model.apply_template(
                messages,
                chat_template_kwargs_override=self._env_template_kwargs,
            )
            # Initialize masks and logprobs with zeros for prompt tokens
            # This marks them as non-trainable (loss_mask=0) with no logprobs
            all_masks: List[int] = []
            all_logprobs: List[float] = []
            assistant_responses: List[str] = []

            try:
                result = await asyncio.wait_for(
                    self.model.generate_async(
                        messages=messages,
                        input_ids=all_tokens,
                        temperature=self.config.env_temperature,
                        top_p=self.config.env_top_p,
                        top_k=self.config.env_top_k,
                        max_tokens=self.config.env_max_tokens,
                        role="environment",
                    ),
                    timeout=300,  # 5 min hard timeout per env generation
                )
                result = result[0] if isinstance(result, list) else result

                # Adapter returns an error dict on exception — treat as a
                # failed attempt so the retry loop can try again with a
                # fresh request.
                if result.get("error"):
                    raise RuntimeError(
                        f"SGLang generation error: {result['error']}"
                    )

                response_tokens = result['token_ids']
                all_tokens.extend(response_tokens)
                all_masks.extend([1] * len(response_tokens))
                all_logprobs.extend(result['logprobs'])
                assistant_responses.append(result['text'])
                response_text = result['text']

                messages.append({"role": "assistant", "content": response_text})

                game_code = extract_game_code(response_text)

                # Save game file
                game_file = save_game_file(
                    game_code, games_dir, rollout_id, index, skill
                )

                # Validation is executor-bounded because generated code may hang.
                if validate and not await validate_game_async(game_file):
                    raise RuntimeError("Game validation failed")

                # Reject criteria that raise at reset for every tested seed.
                # Rollout zero skips this opt-in gate because no fallback exists yet.
                _reset_gate_on = (
                    os.environ.get("SPARE_RESET_GATE", "") not in ("", "0", "false", "False")
                    and rollout_id > 0
                )
                if _reset_gate_on and criteria_throw_at_reset(str(game_file), self.config.max_turns):
                    self._env_validator_rejected += 1
                    _reason = ("Reset-solvability gate: a success criterion raises on the "
                               "initial state at every tested seed (unsolvable step).")
                    save_rejected_game(
                        game_code, games_dir, rollout_id, index, skill,
                        reject_stage="criteria_throw_at_reset", reasoning=_reason,
                    )
                    raise RuntimeError(_reason)

                # LLM-based environment validation (rejection sampling)
                if self.env_validator is not None:
                    is_valid, reasoning = await asyncio.wait_for(
                        self.env_validator.validate_async(game_code),
                        timeout=120,  # 2 min timeout for validator API call
                    )
                    if not is_valid:
                        self._env_validator_rejected += 1
                        # Persist the rejected game + FULL reasoning so a sample can be
                        # re-probed offline to disambiguate validator-over-reject vs
                        # generator-makes-impossible. Best-effort; never aborts training.
                        save_rejected_game(
                            game_code, games_dir, rollout_id, index, skill,
                            reject_stage="env_validator", reasoning=reasoning,
                        )
                        reason_preview = reasoning[:200].replace('\n', ' ')
                        raise RuntimeError(
                            f"Environment validator rejected game: {reason_preview}"
                        )
                    self._env_validator_accepted += 1
                    logger.info(f"[ENV-VALIDATOR] Game {index} (skill={skill}) passed validation")

                # Build trajectory
                env_traj = build_env_trajectory(
                    messages=messages,
                    all_tokens=all_tokens,
                    all_masks=all_masks,
                    all_logprobs=all_logprobs,
                    assistant_responses=assistant_responses,
                    skill=skill,
                    difficulty=difficulty,
                    game_code=game_code,
                    index=index,
                )

                # Cache if enabled
                if cache_step_dir is not None:
                    shutil.copy2(str(game_file), str(cache_step_dir / game_file.name))

                return game_file, env_traj

            except Exception as e:
                logger.warning(f"[GEN] Game {index} (skill={skill}) attempt {attempt + 1}/{max_attempts} failed: {e}")
                if game_file and game_file.exists():
                    game_file.unlink(missing_ok=True)
                continue

        logger.error(f"[GEN] Game {index} (skill={skill}) failed after {max_attempts} attempts")
        return None

    def _augment_prompt_with_memory(self, prompt_content: str, skill: str) -> str:
        """Append memory-derived few-shot examples to a generation prompt.

        High-regret past environments are injected as positive seeds and one
        low-quality environment as a negative example. No-op when memory is
        absent or empty, so all call sites can apply it unconditionally.
        """
        if not self.env_memory or len(self.env_memory) == 0:
            return prompt_content
        seeds = self.env_memory.high_regret_seeds(n=2, skill=skill)
        if seeds:
            seeds_text = self.env_memory.format_seeds_for_prompt(seeds)
            prompt_content += (
                "\n\n<REFERENCE_ENVIRONMENTS>\n"
                "Here are examples of effective training environments with "
                "good difficulty calibration. Use them as inspiration for "
                "structure and complexity, but create a DIFFERENT game:\n\n"
                f"{seeds_text}\n"
                "</REFERENCE_ENVIRONMENTS>\n"
            )
        negatives = self.env_memory.low_quality_examples(n=1, skill=skill)
        if negatives:
            neg_text = self.env_memory.format_negative_examples(negatives)
            prompt_content += f"\n{neg_text}\n"
        return prompt_content

    def _build_env_gen_prompt(self, skill: str, difficulty: str) -> Tuple[str, str]:
        """Build the (prompt_content, template_name) for env generation.
        Same dispatch as the legacy loop; factored so the repair path reuses it."""
        game_type = getattr(self.config, "game_type", "cognitive")
        template_name = self.config.env_generation_template
        if game_type == "tool_use":
            skill_info = TOOL_USE_SKILLS[skill]
            prompt_content = get_env_gen_prompt_fn()(skill, skill_info, difficulty)
            if template_name == "qwen3_game_generation":
                template_name = "qwen3_tool_use_game_generation"
        elif template_name == "qwen3_multiturn_game_generation":
            skill_info = SyntheticGameGenerator.COGNITIVE_SKILLS[skill]
            prompt_content = generate_multiturn_game_generation_prompt(
                skill, skill_info, difficulty, max_turns=self.config.max_turns,
            )
        else:
            skill_info = SyntheticGameGenerator.COGNITIVE_SKILLS[skill]
            prompt_content = generate_game_generation_prompt(skill, skill_info, difficulty)
        return self._augment_prompt_with_memory(prompt_content, skill), template_name

    async def _validate_game_with_error(
        self, game_file: Path, game_code: str, validate: bool,
    ) -> Tuple[bool, str]:
        """Validate a generated game; return (is_valid, error_message). The error
        is fed back to the proposer for repair, so make it specific."""
        if validate and not await validate_game_async(game_file):
            # Pin down a useful reason for the repair feedback.
            try:
                compile(game_code, "<game>", "exec")
                err = ("the game raised at reset()/step() or has a broken interface "
                       "(missing reset(seed=None)/step(action) or wrong return signature)")
            except SyntaxError as se:
                err = f"SyntaxError: {se.msg} (line {se.lineno})"
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            return False, err
        if self.env_validator is not None:
            try:
                is_valid, reasoning = await asyncio.wait_for(
                    self.env_validator.validate_async(game_code), timeout=120,
                )
            except Exception as e:  # noqa: BLE001
                return False, f"env_validator error: {e}"
            if not is_valid:
                self._env_validator_rejected += 1
                return False, f"environment validator rejected it: {reasoning[:200].strip()}"
            self._env_validator_accepted += 1
        return True, ""

    def _persist_rejected(
        self, games_dir: Path, rollout_id: int, index: int, skill: str,
        code: str, error: str, attempt: int,
    ) -> None:
        """Save a validation-failed generation for inspection (opt-in)."""
        if not self.config.persist_rejected:
            return
        try:
            rej_dir = Path(games_dir) / "rejected"
            rej_dir.mkdir(parents=True, exist_ok=True)
            fn = rej_dir / f"rej_r{rollout_id}_g{index}_a{attempt}_{skill.replace(' ', '_')}.py"
            fn.write_text(f"# REJECTED (attempt {attempt}): {error}\n\n{code}")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[GEN] persist_rejected failed: {e}")

    @staticmethod
    def _repair_prompt(error: str) -> str:
        return (
            f"The game code you wrote failed validation with this error:\n\n"
            f"    {error}\n\n"
            f"Return a COMPLETE, corrected version of the game in a single "
            f"```python code block. Fix the error, keep the same game idea, and "
            f"ensure the class defines reset(seed=None) and step(action)."
        )

    async def _generate_one_env_response(self, messages: List[Dict[str, str]]):
        """One env generation turn. Returns the (single) result dict or raises."""
        tokens = self.model.apply_template(
            messages, chat_template_kwargs_override=self._env_template_kwargs,
        )
        result = await asyncio.wait_for(
            self.model.generate_async(
                messages=messages, input_ids=tokens,
                temperature=self.config.env_temperature, top_p=self.config.env_top_p,
                top_k=self.config.env_top_k, max_tokens=self.config.env_max_tokens,
                role="environment",
            ),
            timeout=300,
        )
        result = result[0] if isinstance(result, list) else result
        if result.get("error"):
            raise RuntimeError(f"SGLang generation error: {result['error']}")
        return tokens, result

    async def _save_validate_game(
        self, code: str, games_dir: Path, rollout_id: int, index: int, skill: str,
        validate: bool, attempt: int, cache_step_dir: Optional[Path],
    ) -> Tuple[Optional[Path], str]:
        """Save + validate one generated game. On success, copy it into the step
        cache and return (file, ""). On failure, persist the rejected code for
        inspection, delete the file, and return (None, error)."""
        game_file = save_game_file(code, games_dir, rollout_id, index, skill)
        valid, error = await self._validate_game_with_error(game_file, code, validate)
        if valid:
            if cache_step_dir is not None:
                shutil.copy2(str(game_file), str(cache_step_dir / game_file.name))
            return game_file, ""
        self._persist_rejected(games_dir, rollout_id, index, skill, code, error, attempt)
        game_file.unlink(missing_ok=True)
        return None, error

    async def _generate_single_game_with_repair(
        self, skill: str, difficulty: str, games_dir: Path, rollout_id: int,
        index: int, validate: bool, cache_step_dir: Optional[Path],
    ) -> Optional[Tuple[Path, Trajectory]]:
        """Train the proposer on its turn-1 generation; if turn-1 is broken, run an
        inference-only repair loop (validation error fed back) to salvage a valid env
        for the actor. The trained proposer trajectory is always turn-1."""
        prompt_content, template_name = self._build_env_gen_prompt(skill, difficulty)
        messages = self._apply_game_template(
            messages=[{"role": "user", "content": prompt_content}],
            template_name=template_name,
        )

        # Turn 1 — the trajectory the proposer is trained on.
        try:
            prompt_tokens, result = await self._generate_one_env_response(messages)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[GEN] Game {index} (skill={skill}) turn-1 generation failed: {e}")
            return None

        resp_tokens = result["token_ids"]
        turn1_text = result["text"]
        turn1_code = extract_game_code(turn1_text)
        turn1_messages = messages + [{"role": "assistant", "content": turn1_text}]
        env_traj = build_env_trajectory(
            messages=turn1_messages,
            all_tokens=list(prompt_tokens) + list(resp_tokens),
            all_masks=[1] * len(resp_tokens),
            all_logprobs=list(result["logprobs"]),
            assistant_responses=[turn1_text],
            skill=skill, difficulty=difficulty, game_code=turn1_code, index=index,
        )
        env_traj.metadata["env_truncated"] = len(resp_tokens) >= self.config.env_max_tokens

        game_file, val_error = await self._save_validate_game(
            turn1_code, games_dir, rollout_id, index, skill, validate, 0, cache_step_dir,
        )
        env_traj.metadata["turn1_valid"] = game_file is not None
        if game_file is not None:
            return game_file, env_traj

        # turn-1 broken → repair (inference-only) so the actor still gets a valid env.
        repair_messages = list(turn1_messages)
        for r in range(self.config.env_repair_turns):
            repair_messages.append({"role": "user", "content": self._repair_prompt(val_error)})
            try:
                _, rres = await self._generate_one_env_response(repair_messages)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[GEN] Game {index} (skill={skill}) repair turn {r + 1} failed: {e}")
                break
            repair_file, val_error = await self._save_validate_game(
                extract_game_code(rres["text"]), games_dir, rollout_id, index, skill,
                validate, r + 1, cache_step_dir,
            )
            if repair_file is not None:
                logger.info(f"[GEN] Game {index} (skill={skill}) repaired on turn {r + 1} (proposer trains on turn-1).")
                return repair_file, env_traj
            repair_messages.append({"role": "assistant", "content": rres["text"]})

        logger.warning(
            f"[GEN] Game {index} (skill={skill}) turn-1 broken and "
            f"{self.config.env_repair_turns} repair(s) failed — dropping."
        )
        return None

    async def generate_games_async(
        self,
        skills: List[str],
        difficulty: str,
        games_dir: Path,
        num_games: int,
        base_index: int = 0,
        validate: bool = True,
        rollout_id: int = 0,
        cache_dir: Optional[Path] = None,
        max_attempts: int = 8,
    ) -> Tuple[List[Path], Dict[Path, Trajectory]]:
        """Generate multiple games concurrently (environment role).

        This is the core generation implementation. Uses asyncio.gather for
        concurrent generation, optimal for HTTP backends.

        Args:
            skills: List of skills to cycle through
            difficulty: Difficulty level
            games_dir: Directory to save games
            num_games: Number of games to generate
            base_index: Starting index for file naming
            validate: Whether to validate games
            rollout_id: Current rollout ID
            cache_dir: Optional cache directory

        Returns:
            Tuple of (game file paths, path -> trajectory mapping)
        """
        games_dir = Path(games_dir)
        games_dir.mkdir(parents=True, exist_ok=True)

        # Setup cache directory
        cache_step_dir = None
        if cache_dir is not None:
            cache_step_dir = Path(cache_dir) / f"step_{rollout_id:05d}"
            cache_step_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[GEN] Generating {num_games} games concurrently")

        # Generate all games concurrently with retry logic
        tasks = [
            self._generate_single_game_async(
                skill=skills[i % len(skills)],
                difficulty=difficulty,
                games_dir=games_dir,
                rollout_id=rollout_id,
                index=base_index + i,
                validate=validate,
                cache_step_dir=cache_step_dir,
                max_attempts=max_attempts,
            )
            for i in range(num_games)
        ]

        # Run all generations concurrently with overall timeout
        # Each attempt: up to 5 min generation + 2 min validation = 7 min
        # With max_attempts retries, cap at 15 min total to avoid blocking training
        overall_timeout = 900  # 15 minutes
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=overall_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[GEN] Game generation timed out after {overall_timeout}s — "
                f"this likely indicates SGLang requests are hanging"
            )
            results = []

        # Collect successful results
        game_files: List[Path] = []
        env_trajectories: Dict[Path, Trajectory] = {}

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[GEN] Game {i} raised exception: {result}")
                continue
            if result is not None:
                game_file, env_traj = result
                game_files.append(game_file)
                env_trajectories[game_file] = env_traj
                logger.info(f"[GEN] Saved: {game_file.name}")

        logger.info(f"[GEN] Generated {len(game_files)}/{num_games} games")
        return game_files, env_trajectories

    async def play_games_async(
        self,
        game_files: List[Path],
        game_policy: GamePolicy,
        trajectories_per_game: int = 1,
        max_concurrent: int = 128,
        max_attempts: int = 2,  # 1 retry for a transient blip; timeouts don't retry (break).
    ) -> Tuple[List[Trajectory], Dict[str, Any]]:
        """Play multiple games concurrently.

        Uses asyncio.gather with semaphore for controlled concurrency.
        Each trajectory gets its own environment instance.

        Args:
            game_files: List of game files to play
            trajectories_per_game: Number of trajectories per game
            max_concurrent: Maximum concurrent games
            max_attempts: Maximum number of attempts to play a game
        Returns:
            Tuple of (trajectories, info dict)
        """
        info: Dict[str, Any] = {
            "num_instances": len(game_files) * trajectories_per_game,
            "num_succeeded": 0,
            "num_failed": 0,
        }

        semaphore = asyncio.Semaphore(max_concurrent)

        per_game_timeout = 300  # 5 min per single game play

        async def run_single_game(game_file: Path, skill: str, group_index: int = 0) -> Trajectory:
            # Convert per-play failures into FAILED trajectories.
            async with semaphore:
                for _ in range(max_attempts):
                    try:
                        traj = await asyncio.wait_for(
                            self.play_game_async(str(game_file), skill),
                            timeout=per_game_timeout,
                        )
                        if traj is not None and traj.status != TrajectoryStatus.FAILED:
                            traj.metadata["group_index"] = group_index
                            return traj
                    except asyncio.TimeoutError:
                        # Timeouts are not retried because executor threads cannot be killed.
                        logger.warning(
                            f"[PLAY] {Path(game_file).name} timed out after "
                            f"{per_game_timeout}s — dropping this play (no retry on timeout)"
                        )
                        break
                    except Exception as e:
                        logger.warning(
                            f"[PLAY] {Path(game_file).name} play errored "
                            f"({type(e).__name__}: {str(e)[:100]}) — retry if attempts remain"
                        )
                return Trajectory(status=TrajectoryStatus.FAILED)

        async def run_game_group(game_file: Path, skill: str, group_index: int) -> List:
            """Run one canary play before launching the rest of a game group.

            A timed-out executor thread cannot be killed, so canary gating limits
            each hung game to one stranded thread. Set SPARE_PLAY_CANARY=0 to
            disable the gate.
            """
            first = await run_single_game(game_file, skill, group_index=group_index)
            group: List = [first]
            remaining = trajectories_per_game - 1
            if remaining <= 0:
                return group
            if str(game_file) in self._hung_game_blacklist:
                logger.warning(
                    f"[PLAY] Canary flagged {Path(game_file).name} as hung — "
                    f"skipping its remaining {remaining} plays"
                )
                group.extend(
                    Trajectory(
                        status=TrajectoryStatus.FAILED,
                        metadata={"error": "skipped: canary hang", "game_file": str(game_file), "skill": skill},
                    )
                    for _ in range(remaining)
                )
                return group
            rest = await asyncio.gather(
                *[run_single_game(game_file, skill, group_index=group_index) for _ in range(remaining)],
                return_exceptions=True,
            )
            group.extend(rest)
            return group

        # Build per-game group tasks. group_index groups all plays of the same
        # game for GRPO advantage computation. Canary gating is ON by default;
        # SPARE_PLAY_CANARY=0 restores the flat all-at-once launch.
        use_canary = os.environ.get("SPARE_PLAY_CANARY", "1") != "0"
        tasks = []
        for group_idx, game_file in enumerate(game_files):
            metadata = game_policy.get_game_metadata(str(game_file))
            skill = metadata.skill if metadata is not None else "unknown"
            if use_canary:
                tasks.append(run_game_group(game_file, skill, group_index=group_idx))
            else:
                for _ in range(trajectories_per_game):
                    tasks.append(run_single_game(game_file, skill, group_index=group_idx))

        logger.info(
            f"[PLAY] Starting {len(game_files) * trajectories_per_game} game instances "
            f"(max {max_concurrent} concurrent, canary={'on' if use_canary else 'off'})"
        )

        # Per-play timeouts preserve completed trajectories when another play stalls.
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten group results (canary mode returns a list per game)
        results = []
        for r in raw_results:
            if isinstance(r, list):
                results.extend(r)
            else:
                results.append(r)

        # Collect results
        all_trajectories: List[Trajectory] = []
        for result in results:
            # BaseException (not just Exception) so a CancelledError from a cancelled
            # task can't slip through and AttributeError on result.status below.
            if isinstance(result, BaseException):
                logger.error(f"[PLAY] Exception: {type(result).__name__}: {result}")
                info["num_failed"] += 1
            elif result and getattr(result, "status", None) != TrajectoryStatus.FAILED:
                all_trajectories.append(result)
                info["num_succeeded"] += 1
            else:
                info["num_failed"] += 1

        logger.info(f"[PLAY] Done: {info['num_succeeded']} succeeded, {info['num_failed']} failed")

        if all_trajectories:
            info["avg_turns"] = sum(t.turn_count for t in all_trajectories) / len(all_trajectories)
            info["avg_reward"] = sum(t.reward for t in all_trajectories) / len(all_trajectories)

        return all_trajectories, info

    # =========================================================================
    # SELF-JUDGE - Validate generated environments using the model itself
    # =========================================================================

    async def self_judge_game_async(
        self,
        game_code: str,
        actor_trajectories: List[Trajectory],
    ) -> Tuple[bool, str]:
        """Use the model to judge if a generated game and its trajectories make sense.

        Args:
            game_code: The Python code of the generated game
            actor_trajectories: List of trajectories from playing this game

        Returns:
            Tuple of (is_valid: bool, reasoning: str)
        """
        if not actor_trajectories:
            return True, "No trajectories to judge"

        # Pick one trajectory to judge (prefer completed ones)
        traj = None
        for t in actor_trajectories:
            if t.status == TrajectoryStatus.COMPLETED:
                traj = t
                break
        if traj is None:
            traj = actor_trajectories[0]

        # Extract observations, actions, rewards from the trajectory
        observations = []
        actions = []
        rewards = []

        # Parse from messages
        for i, msg in enumerate(traj.messages):
            if msg.get("role") == "user":
                observations.append(msg.get("content", ""))
            elif msg.get("role") == "assistant":
                actions.append(msg.get("content", ""))

        # Get rewards from metadata or use final reward
        if "rewards" in traj.metadata:
            rewards = traj.metadata["rewards"]
        else:
            # Use final reward for all turns
            rewards = [0.0] * (len(actions) - 1) + [traj.reward] if actions else []

        # Format trajectory for judging
        trajectory_str = format_trajectory_for_judge(
            observations=observations,
            actions=actions,
            rewards=rewards,
            max_turns_to_show=self.config.self_judge_max_turns_to_show,
        )

        # Generate the self-judge prompt
        judge_prompt = generate_self_judge_prompt(game_code, trajectory_str)

        # Create messages for the judge call
        messages = [
            {"role": "system", "content": SELF_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_prompt},
        ]

        try:
            # Call the model to judge
            results = await self.model.generate_async(
                messages=messages,
                input_ids=None,  # Let model tokenize
                temperature=self.config.self_judge_temperature,
                top_p=self.config.actor_top_p,
                top_k=self.config.actor_top_k,
                max_tokens=self.config.self_judge_max_tokens,
                role="self_judge",
            )

            response = results[0]["text"] if results else ""

            # Parse the verdict from \boxed{}
            from spare.core.utils import extract_boxed_answer
            verdict = extract_boxed_answer(response)

            if verdict is None:
                logger.warning("[SELF-JUDGE] No boxed answer found in response")
                return True, response  # Default to valid if can't parse

            is_valid = verdict.lower().strip() in ["yes", "y", "true", "valid", "correct"]
            return is_valid, response

        except Exception as e:
            logger.error(f"[SELF-JUDGE] Error during evaluation: {e}")
            return True, str(e)  # Default to valid on error

    async def self_judge_games_async(
        self,
        env_trajectories: Dict[str, Trajectory],
        actor_trajectories: List[Trajectory],
        max_concurrent: int = 8,
    ) -> Dict[str, Tuple[bool, str]]:
        """Run self-judge on multiple games concurrently.

        Args:
            env_trajectories: Mapping from game file to env trajectory (contains game code)
            actor_trajectories: All actor trajectories
            max_concurrent: Maximum concurrent judge calls

        Returns:
            Dict mapping game file to (is_valid, reasoning)
        """
        # Group actor trajectories by game file
        game_to_actor_trajs: Dict[str, List[Trajectory]] = {}
        for traj in actor_trajectories:
            game_file = traj.metadata.get("game_file", "")
            if game_file not in game_to_actor_trajs:
                game_to_actor_trajs[game_file] = []
            game_to_actor_trajs[game_file].append(traj)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def judge_single_game(game_file: str, env_traj: Trajectory) -> Tuple[str, bool, str]:
            async with semaphore:
                game_code = env_traj.metadata.get("game_code", "")
                actor_trajs = game_to_actor_trajs.get(game_file, [])

                if not game_code:
                    return game_file, True, "No game code found"

                is_valid, reasoning = await self.self_judge_game_async(game_code, actor_trajs)
                return game_file, is_valid, reasoning

        # Run all judgments concurrently
        tasks = [
            judge_single_game(game_file, env_traj)
            for game_file, env_traj in env_trajectories.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        judge_results: Dict[str, Tuple[bool, str]] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[SELF-JUDGE] Exception: {result}")
                continue
            game_file, is_valid, reasoning = result
            judge_results[game_file] = (is_valid, reasoning)

        return judge_results

    # =========================================================================
    # REGRET-BASED ENV REWARD - Hint generation + dual gameplay
    # =========================================================================

    async def _play_game_with_hint_async(
        self,
        game_file_path: str,
        hint: str,
        seed: int,
    ) -> float:
        """Play a game with hint injected into the first observation.

        Simplified version of play_game_async - only returns the episode reward,
        no trajectory (tokens/masks/logprobs) tracking needed.

        Pattern follows FixedModelEvaluator._play_game_once().

        Args:
            game_file_path: Path to game file
            hint: Hint text to inject into first observation
            seed: Random seed for game reset

        Returns:
            Bounded cumulative episode reward (same episode_reward aggregation as
            build_actor_trajectory, so regret r_hint - r_no_hint is apples-to-apples).
        """
        # Fail fast on games that already hung (see _hung_game_blacklist).
        if game_file_path in self._hung_game_blacklist:
            return 0.0

        try:
            loop = asyncio.get_event_loop()

            def _load_and_reset_hint():
                # Match the no-hint arm's environment-internal turn cap.
                e = make_synthetic_env(
                    game_file_path, max_turns=self.config.max_turns,
                    respect_game_max_turns=True,
                )
                o, _ = e.reset(seed=seed)
                return e, o

            env, obs = await asyncio.wait_for(
                loop.run_in_executor(_ENV_STEP_EXECUTOR, _load_and_reset_hint),
                timeout=ENV_STEP_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(f"[HINT-PLAY] Load/reset timed out on {Path(game_file_path).name}")
            self._hung_game_blacklist.add(game_file_path)
            self._step_timeout_games.add(game_file_path)
            return 0.0
        except Exception as e:
            logger.error(f"[HINT-PLAY] Failed to load game {game_file_path}: {e}")
            return 0.0

        # Inject hint into first observation
        obs_with_hint = self.config.hint_injection_template.format(
            observation=obs, hint=hint
        )

        # Hinted tool-use plays must follow the native tool-call path so both
        # regret arms execute the same environment interface.
        if hasattr(env, "get_tools") and env.get_tools():
            traj = await self._play_game_tool_use_async(
                env,
                obs_with_hint,
                game_file_path,
                skill="hint",
                game_code=None,
                should_close_env=True,
            )
            return float(traj.reward or 0.0)

        messages: List[Dict[str, str]] = self._apply_game_template(
            messages=[{"role": "user", "content": obs_with_hint}],
            template_name="qwen3_game",
        )
        hint_rewards: List[float] = []
        hint_terminated = False
        hint_session_id = str(uuid.uuid4())

        try:
            for turn in range(self.config.max_turns):
                if turn > 0:
                    messages.append({"role": "user", "content": obs})

                # Bound generation like the main actor loop: an unbounded stuck SGLang
                # request here would hang the whole regret computation indefinitely.
                try:
                    results = await asyncio.wait_for(
                        self.model.generate_async(
                            messages=messages,
                            input_ids=None,  # No token tracking needed
                            temperature=self.config.actor_temperature,
                            top_p=self.config.actor_top_p,
                            top_k=self.config.actor_top_k,
                            max_tokens=self.config.actor_max_tokens,
                            session_id=hint_session_id,
                            role="hint_actor",
                        ),
                        timeout=ACTOR_TURN_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"[HINT-PLAY] generate_async timed out after {ACTOR_TURN_TIMEOUT}s — aborting"
                    )
                    break

                raw_action = results[0]["text"]
                messages.append({"role": "assistant", "content": raw_action})

                action = parse_action(raw_action, self.config.action_format)
                loop = asyncio.get_event_loop()
                try:
                    obs, reward, terminated, truncated, _ = await asyncio.wait_for(
                        loop.run_in_executor(_ENV_STEP_EXECUTOR, env.step, action),
                        timeout=ENV_STEP_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"[HINT-PLAY] env.step() timed out after {ENV_STEP_TIMEOUT}s — aborting"
                    )
                    # Degenerate game code (step() hangs): flag for the -0.5 penalty.
                    self._step_timeout_games.add(game_file_path)
                    self._hung_game_blacklist.add(game_file_path)
                    break
                hint_rewards.append(reward)
                hint_terminated = terminated

                if terminated or truncated:
                    break
        except Exception as e:
            logger.error(f"[HINT-PLAY] Error during hint gameplay: {e}")
        finally:
            env.close()

        # Same outcome-only aggregation as the no-hint arm (build_actor_trajectory) so the
        # regret r_hint - r_no_hint compares like with like (solve-rate gap).
        return episode_reward(hint_rewards, hint_terminated)

    async def _compute_hint_phase_for_game_async(
        self,
        game_file: str,
        game_code: str,
    ) -> Tuple[str, Optional[float], Dict[str, Any]]:
        """Hint phase for one game: generate the hint, then play with it.

        Depends only on the game CODE — not on the no-hint plays — so it can run
        concurrently with the Stage-4 actor plays (the regret subtraction happens
        later, in _finalize_regrets). The hint plays are mutually independent
        (deterministic per-play seeds), so they also run concurrently.

        Returns:
            Tuple of (game_file, r_with_hint or None on hint failure, stats_dict)
        """
        # Fail fast on games that already hung (see _hung_game_blacklist) —
        # skip the hint-gen model call and all plays.
        if game_file in self._hung_game_blacklist:
            return game_file, None, {"hint_error": "blacklisted: earlier hang"}

        # Generate hint from game code ("self": training model; "external": API)
        try:
            hint = await self.hint_generator.generate_hint_with_retry(game_code)
        except Exception as e:
            logger.error(f"[REGRET] Hint generation failed for {game_file}: {e}")
            return game_file, None, {"hint_error": str(e)}

        # Probe one hint play before fan-out so hung games are blacklisted early.
        n_plays = self.config.hint_plays_per_game
        use_canary = os.environ.get("SPARE_PLAY_CANARY", "1") != "0"
        if use_canary and n_plays > 1:
            first = await self._play_game_with_hint_async(
                game_file, hint, hash(game_file + "0") % (2**31)
            )
            if game_file in self._hung_game_blacklist:
                logger.warning(
                    f"[HINT-PLAY] Canary hint play flagged {Path(game_file).name} "
                    f"as hung — skipping its remaining {n_plays - 1} hint plays"
                )
                return game_file, None, {"hint_error": "canary hang"}
            rest = await asyncio.gather(
                *(
                    self._play_game_with_hint_async(
                        game_file, hint, hash(game_file + str(i)) % (2**31)
                    )
                    for i in range(1, n_plays)
                )
            )
            hint_rewards = [first] + list(rest)
        else:
            hint_rewards = list(
                await asyncio.gather(
                    *(
                        self._play_game_with_hint_async(
                            game_file, hint, hash(game_file + str(i)) % (2**31)
                        )
                        for i in range(n_plays)
                    )
                )
            )

        r_with_hint = float(np.mean(hint_rewards)) if hint_rewards else 0.0
        stats = {
            "hint": hint[:200],  # Truncate for logging/wandb
            "hint_full": hint,  # Full hint for cache saving
            "r_hint": r_with_hint,
        }
        return game_file, r_with_hint, stats

    async def _compute_hint_phases_batched_async(
        self,
        env_trajectories: Dict[str, Trajectory],
    ) -> Dict[str, Tuple[Optional[float], Dict[str, Any]]]:
        """Run the hint phase (hint-gen + hint plays) for all games in parallel.

        Independent of the no-hint plays, so collect_trajectories can run this
        concurrently with Stage 4 and finish the regret math afterwards via
        _finalize_regrets.

        Returns:
            game_file -> (r_with_hint or None on hint failure, partial stats)
        """
        tasks = []
        for game_file, env_traj in env_trajectories.items():
            game_code = env_traj.metadata.get("game_code", "")
            if not game_code:
                try:
                    game_code = Path(game_file).read_text()
                except Exception:
                    logger.warning(f"[REGRET] Could not read game code for {game_file}")
                    continue
            tasks.append(self._compute_hint_phase_for_game_async(game_file, game_code))

        if not tasks:
            return {}

        logger.info(f"[REGRET] Computing hint phase for {len(tasks)} games in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        hint_phases: Dict[str, Tuple[Optional[float], Dict[str, Any]]] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[REGRET] Exception: {result}")
                continue
            game_file, r_with_hint, stats = result
            hint_phases[game_file] = (r_with_hint, stats)
        return hint_phases

    def _finalize_regrets(
        self,
        hint_phases: Dict[str, Tuple[Optional[float], Dict[str, Any]]],
        game2rewards: Dict[str, List[float]],
        cache_dir: Optional[Path] = None,
        rollout_id: int = 0,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        """Subtract R(no_hint) from the hint-phase results and save hint cache.

        Split from the hint phase so the (slow, model-bound) hint phase can
        overlap Stage 4 while this (instant) subtraction runs after both finish.
        """
        game2regret: Dict[str, float] = {}
        game2hint_stats: Dict[str, Dict[str, Any]] = {}
        for game_file, (r_with_hint, stats) in hint_phases.items():
            if r_with_hint is None:  # hint generation failed -> regret 0
                game2regret[game_file] = 0.0
                game2hint_stats[game_file] = stats
                continue
            no_hint_rewards = game2rewards.get(game_file, [0.0])
            r_without_hint = float(np.mean(no_hint_rewards)) if no_hint_rewards else 0.0
            regret = r_with_hint - r_without_hint
            stats["r_no_hint"] = r_without_hint
            stats["regret"] = regret
            logger.info(
                f"[REGRET] {Path(game_file).name}: "
                f"R(hint)={r_with_hint:.3f}, R(no_hint)={r_without_hint:.3f}, "
                f"regret={regret:.3f}"
            )
            game2regret[game_file] = regret
            game2hint_stats[game_file] = stats

        # Save hints to cache directory if enabled
        if cache_dir is not None and game2hint_stats:
            cache_step_dir = Path(cache_dir) / f"step_{rollout_id:05d}"
            cache_step_dir.mkdir(parents=True, exist_ok=True)
            for game_file, hint_stats in game2hint_stats.items():
                hint_full = hint_stats.get("hint_full", "")
                if not hint_full:
                    continue
                game_stem = Path(game_file).stem
                hint_path = cache_step_dir / f"{game_stem}_hint.txt"
                try:
                    hint_path.write_text(hint_full)
                except Exception as e:
                    logger.warning(f"[REGRET] Failed to save hint to {hint_path}: {e}")
            logger.info(f"[REGRET] Saved {len(game2hint_stats)} hints to {cache_step_dir}")

        return game2regret, game2hint_stats

    # =========================================================================
    # SYNC WRAPPERS - For compatibility with sync backends
    # =========================================================================

    def _get_or_create_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get existing event loop or create a new one."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def play_game(self, game_file_path: str) -> Trajectory:
        """Play a single game synchronously (wraps async)."""
        loop = self._get_or_create_event_loop()
        return loop.run_until_complete(self.play_game_async(game_file_path))

    def generate_environment(self, skill: str, difficulty: str) -> Trajectory:
        """Generate a single game environment synchronously (environment role).

        This is a convenience method for generating a single game. For batch
        generation, use generate_games() or generate_games_async().

        Args:
            skill: Cognitive skill to test (e.g., "Pattern Recognition")
            difficulty: Difficulty level ("easy", "medium", "hard")

        Returns:
            Trajectory containing the generated game code in metadata["game_code"]
        """
        # Create temp directory for the game file
        with tempfile.TemporaryDirectory() as tmpdir:
            games_dir = Path(tmpdir)
            game_files, env_trajectories = self.generate_games(
                model=self.model,
                config=self.config,
                skills=[skill],
                difficulty=difficulty,
                games_dir=games_dir,
                num_games=1,
                validate=False,  # Caller can validate if needed
            )

            if game_files and env_trajectories:
                # Return the trajectory (game file is in temp dir, but code is in metadata)
                return list(env_trajectories.values())[0]

            # Return failed trajectory if generation failed
            return Trajectory(status=TrajectoryStatus.FAILED)

    def generate_and_save_game(
        self,
        skill: str,
        difficulty: str,
        games_dir: Path,
        game_index: int = 0,
        max_attempts: int = 5,
        validate: bool = True,
    ) -> Tuple[Optional[Path], Optional[Trajectory]]:
        """Generate a game, save it to disk, and optionally validate.

        This is a convenience method for generating a single game with retries.

        Args:
            skill: Cognitive skill to target
            difficulty: Difficulty level
            games_dir: Directory to save game
            game_index: Index for naming the file
            max_attempts: Maximum generation attempts
            validate: Whether to validate game before returning

        Returns:
            Tuple of (game_file_path, Trajectory), or (None, None) if failed
        """
        for attempt in range(max_attempts):
            try:
                logger.info(f"[GAME] Generating {skill} game (attempt {attempt + 1}/{max_attempts})")

                game_files, env_trajectories = self.generate_games(
                    model=self.model,
                    config=self.config,
                    skills=[skill],
                    difficulty=difficulty,
                    games_dir=games_dir,
                    num_games=1,
                    base_index=game_index,
                    validate=validate,
                    rollout_id=0,
                )

                if game_files and env_trajectories:
                    game_file = game_files[0]
                    trajectory = list(env_trajectories.values())[0]
                    logger.info(f"[GAME] Saved: {game_file.name}")
                    return game_file, trajectory

            except Exception as e:
                logger.warning(f"[GAME] Generation attempt {attempt + 1} failed: {e}")
                continue

        logger.error(f"[GAME] Failed to generate {skill} game after {max_attempts} attempts")
        return None, None

    def generate_games(
        self,
        model: ModelAdapter,
        config: SpareConfig,
        skills: List[str],
        difficulty: str,
        games_dir: Path,
        num_games: int,
        base_index: int = 0,
        validate: bool = True,
        rollout_id: int = 0,
        cache_dir: Optional[Path] = None,
        max_attempts: int = 5,
    ) -> Tuple[List[Path], Dict[Path, Trajectory]]:
        """Generate games synchronously (wraps async)."""
        loop = self._get_or_create_event_loop()
        return loop.run_until_complete(
            self.generate_games_async(
                skills=skills,
                difficulty=difficulty,
                games_dir=games_dir,
                num_games=num_games,
                base_index=base_index,
                validate=validate,
                rollout_id=rollout_id,
                cache_dir=cache_dir,
                max_attempts=max_attempts,
            )
        )

    def play_games(
        self,
        model: ModelAdapter,
        config: SpareConfig,
        game_files: List[Path],
        game_policy: GamePolicy,
        trajectories_per_game: int = 1,
        max_concurrent: int = 128,
    ) -> Tuple[List[Trajectory], Dict[str, Any]]:
        """Play games synchronously (wraps async)."""
        loop = self._get_or_create_event_loop()
        return loop.run_until_complete(
            self.play_games_async(
                game_files=game_files,
                trajectories_per_game=trajectories_per_game,
                max_concurrent=max_concurrent,
                game_policy=game_policy,
            )
        )

    # =========================================================================
    # HIGH-LEVEL ROLLOUT API
    # =========================================================================

    def collect_trajectories(
        self,
        should_regenerate: bool,
        skills: List[str],
        difficulty: str,
        game_files: List[Path],
        games_dir: Path,
        min_games: int = 1,
        max_games: int = 10,
        global_batch_size: int = 128,
        mode: str = "batched",
        rollout_id: int = 0,
        cache_dir: Optional[Path] = None,
        max_attempts: int = 8,
        use_solver_variance_reward: bool = False,
        regeneration_interval: int = 50,
        allow_generation: bool = True,
    ) -> Tuple[List[Trajectory], List[Trajectory], Dict[str, Any]]:
        """High-level rollout loop shared across all backends.

        Pipeline stages:
        1. Cleanup old games (if regenerating)
        2. Generate new games (if needed)
        3. Select games to play
        4. Play games and collect actor trajectories
        5. Compute environment rewards from actor performance
        6. Assign trajectory weights for balanced training

        Args:
            should_regenerate: Whether to force regeneration of new games
            skills: Available cognitive skills to sample from
            difficulty: Current difficulty level
            game_files: List of valid existing game files
            games_dir: Directory to save generated games
            game_policy: Optional GamePolicy for game lifecycle management
            min_games: Minimum number of games to collect
            max_games: Maximum number of games to collect
            global_batch_size: Global batch size
            mode: "batched" (vLLM) or "async" (HTTP)
            rollout_id: Current rollout ID
            cache_dir: Optional cache directory
            max_attempts: Maximum generation attempts per game
            use_solver_variance_reward: Whether to use solver variance reward
            regeneration_interval: Steps between game regeneration (for weight computation)

        Returns:
            Tuple of (env_trajectories, actor_trajectories, info_dict)
        """
        info: Dict[str, Any] = {}
        env2trajectories: Dict[str, Trajectory] = {}
        games_dir = Path(games_dir)
        games_dir.mkdir(parents=True, exist_ok=True)

        # Reset the env.step()-timeout capture for this rollout (populated during plays,
        # consumed after Stage 4 to stamp metadata["step_timeout"] for the -0.5 penalty).
        self._step_timeout_games.clear()

        # Stage 1: Cleanup old games if regenerating
        if should_regenerate and game_files:
            cleanup_old_games(game_files)
            updated_game_files: List[Path] = []
        else:
            updated_game_files = list(game_files)

        # Stage 2: Generate new games if needed
        if allow_generation and (should_regenerate or len(game_files) < min_games):
            num_to_generate = min_games - (0 if should_regenerate else len(game_files))
            base_index = 0 if should_regenerate else len(game_files)

            # Reset env validator counters for this rollout
            self._env_validator_accepted = 0
            self._env_validator_rejected = 0

            generate_games_func = self.generate_games if mode != "batched" else generate_games_batched
            new_files, new_env_trajs = generate_games_func(
                model=self.model,
                config=self.config,
                skills=skills,
                difficulty=difficulty,
                games_dir=games_dir,
                num_games=num_to_generate,
                base_index=base_index,
                rollout_id=rollout_id,
                cache_dir=cache_dir,
                max_attempts=max_attempts,
            )

            # Register with policy and update tracking
            for game_file, env_traj in zip(new_files, new_env_trajs.values()):
                updated_game_files.append(game_file)
                skill = env_traj.metadata["skill"]
                self.game_policy.register_game(str(game_file), skill, difficulty)

            for game_file, env_traj in new_env_trajs.items():
                env2trajectories[str(game_file)] = env_traj
            info["num_games_generated"] = len(new_files)
            # Fraction of requested games that passed validation; a declining
            # pass-rate signals the proposer collapsing into invalid game code.
            info["env_validation_pass_rate"] = len(new_files) / max(num_to_generate, 1)

            # Log env validator metrics
            if self.env_validator is not None:
                total_checked = self._env_validator_accepted + self._env_validator_rejected
                rejection_rate = (
                    self._env_validator_rejected / total_checked
                    if total_checked > 0 else 0.0
                )
                info["env_validator/accepted"] = self._env_validator_accepted
                info["env_validator/rejected"] = self._env_validator_rejected
                info["env_validator/total_checked"] = total_checked
                info["env_validator/rejection_rate"] = rejection_rate
                info["env_validator/games_requested"] = num_to_generate
                info["env_validator/games_produced"] = len(new_files)
                logger.info(
                    f"[ENV-VALIDATOR] Rollout stats: {self._env_validator_accepted} accepted, "
                    f"{self._env_validator_rejected} rejected "
                    f"(rejection_rate={rejection_rate:.2%}), "
                    f"{len(new_files)}/{num_to_generate} games produced"
                )

        # Stage 3: Select games to play, excluding blacklisted hangs.
        if self._hung_game_blacklist:
            n_before = len(updated_game_files)
            updated_game_files = [
                g for g in updated_game_files
                if str(g) not in self._hung_game_blacklist
            ]
            if len(updated_game_files) < n_before:
                logger.info(
                    "[COLLECT] Excluded %d hung-blacklisted games from sampling "
                    "(%d blacklisted total this run)",
                    n_before - len(updated_game_files),
                    len(self._hung_game_blacklist),
                )
            info["rollout/hung_blacklist_size"] = len(self._hung_game_blacklist)

        if not updated_game_files:
            logger.error("[COLLECT] No valid game files available — returning empty trajectories")
            return [], [], info

        selected_game_files = (
            random.sample(updated_game_files, max_games)
            if len(updated_game_files) > max_games
            else updated_game_files
        )

        # Stage 4: Run actor and independent hint phases concurrently.
        trajectories_per_game = math.ceil(global_batch_size / len(selected_game_files))
        compute_hints = (
            self.config.env_reward_variant in ("regret_based", "blend")
            and should_regenerate
            and env2trajectories
        )
        if compute_hints:
            loop = self._get_or_create_event_loop()
            (actor_trajectories, play_info), hint_phases = loop.run_until_complete(
                asyncio.gather(
                    self.play_games_async(
                        game_files=selected_game_files,
                        trajectories_per_game=trajectories_per_game,
                        game_policy=self.game_policy,
                    ),
                    self._compute_hint_phases_batched_async(env2trajectories),
                )
            )
        else:
            play_games_func = self.play_games if mode != "batched" else play_games_batched
            actor_trajectories, play_info = play_games_func(
                model=self.model,
                config=self.config,
                game_files=selected_game_files,
                trajectories_per_game=trajectories_per_game,
                game_policy=self.game_policy,
            )

        # Stamp games whose env.step() hung during play so the proposer reward stage
        # applies a bounded -0.5 penalty (degenerate game code). The flag rides the env
        # trajectory -> Slime Sample (converter copies metadata) -> regret reward fns.
        if self._step_timeout_games:
            timeout_keys = {str(g) for g in self._step_timeout_games}
            n_flagged = 0
            for gf, env_traj in env2trajectories.items():
                if str(gf) in timeout_keys:
                    env_traj.metadata["step_timeout"] = True
                    n_flagged += 1
            logger.warning(
                f"[COLLECT] {n_flagged} game(s) hit env.step() timeout -> proposer "
                f"step_timeout penalty (-0.5)"
            )

        # Stage 5: Normalize rewards per game (so rewards are comparable across different games)
        actor_trajectories, stats = normalize_rewards_per_game(
            actor_trajectories,
            selected_game_files,
            game_baseline_tracker=self.game_baseline_tracker,
            reward_normalization=self.config.reward_normalization,
        )
        if not actor_trajectories:
            # Let the caller inert-pad a rollout in which every play failed.
            logger.critical(
                "[COLLECT] 0 actor trajectories survived play — returning empty "
                "batch for inert-pad fallback"
            )
            return [], [], info
        actor_info = {
            "actor/all_zero_std": 0.0,
        }
        # Defensive access keeps partial batches observable instead of fatal.
        for skill, rewards in stats.get("skill", {}).items():
            actor_info[f"actor/skill_{skill}/avg_reward"] = float(np.mean(rewards))
            actor_info[f"actor/skill_{skill}/std_reward"] = float(np.std(rewards))

        # Per-game mean and std of raw rewards (8 plays each), then average across all games
        game_means = []
        game_stds = []
        game_stats = stats.get("game", {})
        for game_file, rewards in game_stats.items():
            if float(np.std(rewards)) == 0.0:
                actor_info["actor/all_zero_std"] += 1
            game_means.append(float(np.mean(rewards)))
            game_stds.append(float(np.std(rewards)))
        actor_info["actor/all_zero_std"] /= max(len(game_stats), 1)
        actor_info["actor/batch_mean_reward"] = float(np.mean(game_means)) if game_means else 0.0
        actor_info["actor/batch_mean_std_reward"] = float(np.mean(game_stds)) if game_stds else 0.0

        # Add per-game baseline stats if enabled
        if self.game_baseline_tracker is not None:
            baseline_stats = self.game_baseline_tracker.get_stats()
            for k, v in baseline_stats.items():
                actor_info[f"actor/game_baseline/{k}"] = v


        print(f"[COLLECT] stats: {stats}")

        # Stage 6: Compute environment rewards by aggregating actor rewards
        # Compute env_reward_scale if auto-compute is enabled (Variant 1: simple scaling)
        if self.config.auto_compute_env_reward_scale and self.config.env_reward_scaling_variant == 1:
            env_reward_scale = compute_env_reward_scale(
                num_env_trajectories=len(env2trajectories),
                num_actor_trajectories=len(actor_trajectories),
                regeneration_interval=regeneration_interval,
                max_scale=self.config.max_env_reward_scale,
            )
        else:
            env_reward_scale = 1.0  # No scaling

        if self.config.env_reward_variant in ("regret_based", "blend") and env2trajectories:
            # Regret-based / blend: hint phase ran concurrently with Stage 4 — subtract
            # R(no_hint) and write the hint cache.
            if should_regenerate:
                game2regret, game2hint_stats = self._finalize_regrets(
                    hint_phases,
                    # .get(): stats["game"] is absent when every actor play failed.
                    stats.get("game", {}),
                    cache_dir=cache_dir,
                    rollout_id=rollout_id,
                )
                self._cached_regret = game2regret
                self._cached_hint_stats = game2hint_stats
                # Preserve hint statistics for delayed reward recomputation.
                for game_file, hint_stats in game2hint_stats.items():
                    if game_file in env2trajectories:
                        env2trajectories[game_file].metadata["hint_stats"] = hint_stats
            else:
                # Non-regeneration step: use cached regret values
                game2regret = self._cached_regret
                game2hint_stats = getattr(self, '_cached_hint_stats', {})

            env_trajectories, env_info = assign_env_rewards_regret(
                env_trajectories=env2trajectories,
                game2regret=game2regret,
                game2rewards=stats.get("game", {}),
                game2hint_stats=game2hint_stats,
                learning_potentials=self.learning_potentials,
                skill2rewards=stats.get("skill", {}),
                env_reward_scale=env_reward_scale,
            )
        else:
            # Default: learning potential variant (existing code)
            env_trajectories, env_info = assign_env_rewards(
                env_trajectories=env2trajectories,
                # .get(): empty-batch safe — see the regret call above.
                skill2rewards=stats.get("skill", {}),
                game2rewards=stats.get("game", {}),
                learning_potentials=self.learning_potentials,
                use_solver_variance_reward=use_solver_variance_reward,
                env_reward_scale=env_reward_scale,
                skip_lp_update=self.config.skip_lp_update,
            )

        # Stage 6.5: Self-judge to validate generated environments (if enabled)
        if self.config.use_self_judge and env2trajectories:
            logger.info(f"[SELF-JUDGE] Running self-judge on {len(env2trajectories)} games")
            loop = self._get_or_create_event_loop()
            judge_results = loop.run_until_complete(
                self.self_judge_games_async(
                    env_trajectories=env2trajectories,
                    actor_trajectories=actor_trajectories,
                    max_concurrent=self.config.max_concurrent_games,
                )
            )

            # Apply penalties and collect stats
            num_valid = 0
            num_invalid = 0
            for env_traj in env_trajectories:
                game_file = env_traj.metadata.get("game_file", "")
                if game_file in judge_results:
                    is_valid, reasoning = judge_results[game_file]
                    env_traj.metadata["self_judge_valid"] = is_valid
                    env_traj.metadata["self_judge_reasoning"] = reasoning[:500]  # Truncate for storage

                    if is_valid:
                        num_valid += 1
                    else:
                        num_invalid += 1
                        # Apply penalty to env reward
                        env_traj.reward += self.config.self_judge_penalty
                        logger.info(f"[SELF-JUDGE] Game {game_file} marked invalid, penalty applied")

            env_info["self_judge/num_valid"] = num_valid
            env_info["self_judge/num_invalid"] = num_invalid
            env_info["self_judge/valid_ratio"] = num_valid / (num_valid + num_invalid) if (num_valid + num_invalid) > 0 else 1.0
            logger.info(f"[SELF-JUDGE] Results: {num_valid} valid, {num_invalid} invalid")

        # Stage 6.75: Assign trajectory weights for balanced training (Variant 1)
        if self.config.env_reward_scaling_variant == 1:
            assign_trajectory_weights(
                env_trajectories=env_trajectories,
                actor_trajectories=actor_trajectories,
                regeneration_interval=regeneration_interval,
                max_sample_weight=self.config.max_env_reward_scale,
                auto_compute=self.config.auto_compute_env_reward_scale,
            )

        # Remove empty responses and trajectories missing token log probabilities.
        def _has_valid_log_probs(traj: Trajectory) -> bool:
            if traj.response_length == 0:
                return False
            return len(traj.rollout_log_probs) == traj.response_length

        valid_actor = [t for t in actor_trajectories if _has_valid_log_probs(t)]
        valid_env = [t for t in env_trajectories if _has_valid_log_probs(t)]

        num_filtered_actor = len(actor_trajectories) - len(valid_actor)
        num_filtered_env = len(env_trajectories) - len(valid_env)

        if num_filtered_actor > 0:
            logger.warning(
                f"[COLLECT] Filtered {num_filtered_actor} actor trajectories with invalid rollout_log_probs"
            )
        if num_filtered_env > 0:
            logger.warning(
                f"[COLLECT] Filtered {num_filtered_env} env trajectories with invalid rollout_log_probs"
            )

        info["rollout/num_filtered_actor_trajectories"] = num_filtered_actor
        info["rollout/num_filtered_env_trajectories"] = num_filtered_env

        actor_trajectories = valid_actor
        env_trajectories = valid_env

        # Stage 6.95: Compute per-role generation entropy (mean negative log prob)
        # Higher entropy = model more uncertain/diverse in its generation
        actor_logprobs = [lp for t in actor_trajectories for lp in t.rollout_log_probs]
        env_logprobs = [lp for t in env_trajectories for lp in t.rollout_log_probs]

        if actor_logprobs:
            info["entropy/actor_mean_neg_logprob"] = -sum(actor_logprobs) / len(actor_logprobs)
        if env_logprobs:
            info["entropy/env_mean_neg_logprob"] = -sum(env_logprobs) / len(env_logprobs)

        # Stage 7: Upsample actor trajectories to fill the batch.
        if not self.config.train_on_env_trajectories:
            num_env = 0
        else:
            num_env = len(env_trajectories) if should_regenerate else 0
        min_actor_needed = global_batch_size - num_env
        info["batch/variant"] = self.config.env_reward_scaling_variant
        info["batch/num_env"] = num_env

        if len(actor_trajectories) < min_actor_needed:
            logger.warning(
                f"[COLLECT] Upsampling actor trajectories: {len(actor_trajectories)} -> {min_actor_needed}"
            )
            actor_trajectories = upsample_trajectories(actor_trajectories, min_actor_needed)

        info.update(**play_info, **actor_info, **env_info)

        # Stage 8: Optionally exclude env trajectories from training batch
        # When train_on_env_trajectories=False, env trajectories are still generated
        # (for game creation) but not included in the returned training batch
        if not self.config.train_on_env_trajectories:
            logger.info(
                f"[COLLECT] Excluding {len(env_trajectories)} env trajectories from training "
                "(train_on_env_trajectories=False)"
            )
            env_trajectories = []  # Clear env trajectories from batch

        # Final logging
        total_trajectories = len(env_trajectories) + len(actor_trajectories)
        logger.info(
            f"[COLLECT] Complete: {len(actor_trajectories)} actor, {len(env_trajectories)} env, "
            f"{total_trajectories} total (batch_size={global_batch_size})"
        )

        return env_trajectories, actor_trajectories, info
