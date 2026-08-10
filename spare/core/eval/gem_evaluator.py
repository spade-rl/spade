"""GEM benchmark evaluator for measuring model capabilities during training.

Evaluates the model being trained on GEM environments (game, code, math,
reasoning_gym). Results are logged to W&B. Eval-only -- no training on GEM tasks.

GEM environments are multi-turn interactive (Gym-style reset()/step()),
unlike single-turn JSONL eval. This evaluator runs the full gameplay loop.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

import gem

import spare.core.envs.gem_adapter  # noqa: F401 — registers custom envs before gem.make()
from spare.core.eval.gem_tasks import GemTaskSpec, rg_golden_goose_category
from spare.core.model_adapter import ModelAdapter
from spare.core.utils import get_token_delta

logger = logging.getLogger(__name__)


@dataclass
class GemTaskResult:
    """Result of evaluating a single GEM task."""
    task_id: str
    category: str
    num_episodes: int
    num_wins: int
    win_rate: float
    mean_reward: float
    mean_turns: float
    errors: int = 0
    elapsed_sec: float = 0.0
    # Distinguishes metrics when one task is evaluated with multiple prompts.
    metric_suffix: str = ""


@dataclass
class GemEvalResult:
    """Aggregate results from GEM evaluation."""
    overall_win_rate: float
    overall_mean_reward: float
    overall_mean_turns: float
    total_episodes: int
    total_tasks: int
    per_task_results: List[GemTaskResult] = field(default_factory=list)
    per_category_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    errors: int = 0

    def to_metrics_dict(self, prefix: str = "gem_eval") -> Dict[str, float]:
        """Convert to flat metrics dictionary for W&B logging."""
        metrics = {
            f"{prefix}/overall_win_rate": self.overall_win_rate,
            f"{prefix}/overall_mean_reward": self.overall_mean_reward,
            f"{prefix}/overall_mean_turns": self.overall_mean_turns,
            f"{prefix}/total_episodes": float(self.total_episodes),
            f"{prefix}/total_tasks": float(self.total_tasks),
            f"{prefix}/errors": float(self.errors),
        }

        # Per-category metrics
        for category, cat_metrics in self.per_category_metrics.items():
            for metric_name, value in cat_metrics.items():
                metrics[f"{prefix}/category_{category}/{metric_name}"] = value

        # Per-task metrics
        for result in self.per_task_results:
            task_key = result.task_id.replace(":", "_").replace("-", "_").lower()
            task_key = f"{task_key}{result.metric_suffix}"
            metrics[f"{prefix}/task_{task_key}/win_rate"] = result.win_rate
            metrics[f"{prefix}/task_{task_key}/mean_reward"] = result.mean_reward
            metrics[f"{prefix}/task_{task_key}/mean_turns"] = result.mean_turns
            metrics[f"{prefix}/task_{task_key}/elapsed_sec"] = result.elapsed_sec

        return metrics


def _build_initial_prompt(
    category: str,
    observation: str,
    task_id: Optional[str] = None,
    prompt_style: str = "default",
) -> str:
    """Build category-appropriate initial prompt.

    For `rg` (reasoning_gym) tasks, mirrors the canonical RG paper prompt
    (`reasoning_gym/utils.py:SYSTEM_PROMPTS["default"]`) which uses
    `<answer></answer>` tags — paired with `RG_STRICT_EXTRACT=1` in
    SpareReasoningGymEnv.step() this gives published-comparable scores.

    Set env var `RG_USE_BOXED_PROMPT=1` to fall back to the older
    `\\boxed{}` prompt (legacy / lenient mode).

    For multiple-choice tasks (e.g., GPQA-Diamond) Qwen3-4B-Instruct-2507
    model card recommends MCQ JSON format (`"answer": "C"`). Two ways to opt-in:
      (1) per-task: set `prompt_style: mcq_json` in the yaml for that task
      (2) globally: set env var `GPQA_PROMPT_STYLE=mcq_json`
    Default ("default" / "boxed") keeps the legacy `\\boxed{}` prompt.
    """
    import os

    _env_style = os.environ.get("GPQA_PROMPT_STYLE", "boxed")
    _effective_style = prompt_style if prompt_style != "default" else (
        "mcq_json" if _env_style == "mcq_json" else "default"
    )
    if (
        _effective_style == "mcq_json"
        and task_id is not None
        and "GPQA" in task_id
        and category != "mcq"  # mcq-category envs bake their own prompt (see passthrough below); never re-wrap them
    ):
        return (
            f"Question: {observation}\n"
            'Please show your choice in the "answer" field with only the choice letter, '
            'e.g., "answer": "C".'
        )

    # The `mcq:` category (McqJsonQaEnv) already applies the full MCQ JSON
    # prompt to the observation inside the env; pass it through unchanged to
    # avoid double-wrapping with a conflicting answer-format instruction.
    if category == "mcq":
        return observation

    # LiveCodeBench env already returns the full LCB-faithful prompt (stdin vs
    # functional instruction baked in to match official). Pass through unchanged
    # so we don't double-wrap with a conflicting answer-format instruction.
    if task_id and "LiveCodeBench" in task_id:
        return observation

    # CodeContest expects executable Python using stdin/stdout, not a boxed answer.
    if task_id and "CodeContest" in task_id:
        return (
            "You are solving a competitive programming problem. Read the input "
            "from standard input (stdin) and write the answer to standard output "
            "(stdout).\n"
            f"Problem:\n{observation}\n"
            "Write a complete Python 3 solution inside a single ```python code "
            "block."
        )

    if category == "code":
        return (
            "You are solving a coding problem. Write a Python solution.\n"
            f"Problem: {observation}\n"
            "Please put your solution in a Python code block inside \\boxed{}, "
            "for example: \\boxed{{```python\nyour code\n```}}"
        )
    if category == "math":
        return (
            "You are solving a math problem.\n"
            f"Problem: {observation}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}."
        )
    if category == "eval":
        return (
            "You are answering a question.\n"
            f"Question: {observation}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}."
        )
    if category == "rg" and os.environ.get("RG_USE_BOXED_PROMPT", "0") in ("0", "false", "False", ""):
        return (
            "Given a problem, your task is to answer the question by thinking "
            "step-by-step in a clear and specific manner.\n"
            f"Problem: {observation}\n"
            "Once you have thought about the reasoning process, provide the "
            "answer in the following format:\n"
            "<answer>answer here</answer>\n"
            "Do not explain your reasoning inside the answer tags, provide only "
            "the final answer. When an example is provided, you should strictly "
            "follow the format of the output/answer in that example."
        )
    return (
        "You are playing a game. Make valid actions to win.\n"
        f"Observation: {observation}\n"
        "Please reason step by step, and put your final answer within \\boxed{{}}."
    )


def _parse_action(response: str) -> str:
    """Pass raw model response as action.

    GEM's concat wrapper handles \\boxed{} extraction internally,
    so we pass the full response through.
    """
    return response


class GemEvaluator:
    """Evaluates model on GEM benchmark environments.

    Uses the same gameplay loop pattern as actor training (play_game_async
    in SpareOrchestrator) but without loss_mask/logprob tracking.

    Per-task settings (max_turns, temperature, max_tokens, episodes) come
    from GemTaskSpec, loaded from the YAML config.

    Args:
        model: ModelAdapter for generation (the model being trained)
        max_concurrent: Maximum concurrent episodes across all tasks
    """

    def __init__(
        self,
        model: ModelAdapter,
        max_concurrent: int = 16,
    ):
        self.model = model
        self.max_concurrent = max_concurrent

    async def _play_episode(
        self, task_spec: GemTaskSpec, episode_idx: int
    ) -> Tuple[bool, float, int]:
        """Play a single episode of a GEM task.

        Uses per-task settings from the task spec for max_turns,
        temperature, and max_tokens.

        Args:
            task_spec: Task spec with per-task settings
            episode_idx: Episode index (used as seed)

        Returns:
            Tuple of (won, total_reward, num_turns)
        """
        try:
            # The evaluator owns conversation history; the environment does not concatenate it.
            env = gem.make(task_spec.task_id)
        except Exception as e:
            logger.error("[GEM-EVAL] Failed to create env for %s: %s", task_spec.task_id, e)
            raise

        max_context = 32768  # Hard cap to prevent context explosion

        try:
            obs, info = env.reset(seed=episode_idx)
            # Append suffix (e.g., board state) if the env provides one
            suffix = info.get("suffix", "")
            if suffix:
                obs = obs + "\n" + suffix
            total_reward = 0.0

            messages: List[Dict[str, str]] = [
                {
                    "role": "user",
                    "content": _build_initial_prompt(
                        task_spec.category, obs,
                        task_id=task_spec.task_id,
                        prompt_style=task_spec.prompt_style,
                    ),
                }
            ]

            # Token tracking for efficient prefix caching
            all_tokens: List[int] = self.model.apply_template(messages)

            # Session ID for consistent hashing: all turns of the same episode
            # route to the same SGLang engine for prefix cache reuse
            session_id = str(uuid.uuid4())

            for turn in range(task_spec.max_turns):
                # Append observation for subsequent turns
                if turn > 0:
                    messages.append({"role": "user", "content": obs})
                    obs_tokens, _ = get_token_delta(
                        tokenizer=self.model.tokenizer,
                        messages=messages,
                    )
                    all_tokens.extend(obs_tokens)

                # Fit generation into the remaining context instead of rejecting
                # tasks whose configured output cap equals the context window.
                _MIN_GEN = 256
                gen_max_tokens = min(task_spec.max_tokens, max_context - len(all_tokens) - 8)
                if gen_max_tokens < _MIN_GEN:
                    logger.warning(
                        "[GEM-EVAL] %s ep %d: prompt %d tokens leaves <%d for output "
                        "(max_context %d), stopping",
                        task_spec.task_id, episode_idx, len(all_tokens), _MIN_GEN, max_context,
                    )
                    return False, total_reward, turn + 1

                # Bound each generation so one stalled request cannot block the suite.
                results = await asyncio.wait_for(self.model.generate_async(
                    messages=messages,
                    input_ids=all_tokens,
                    temperature=task_spec.temperature,
                    # Qwen3 non-thinking sampling from the model card.
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    presence_penalty=0.0,
                    max_tokens=gen_max_tokens,
                    session_id=session_id,
                ), timeout=600)

                result = results[0] if results else {}
                response_text = result.get("text", "")

                # Track assistant response in conversation history
                messages.append({"role": "assistant", "content": response_text})
                response_tokens = result.get("token_ids", [])
                all_tokens.extend(response_tokens)

                action = _parse_action(response_text)
                obs, reward, terminated, truncated, info = env.step(action)
                # Append suffix (e.g., board state) from env
                suffix = info.get("suffix", "")
                if suffix:
                    obs = obs + "\n" + suffix
                total_reward += reward

                if terminated:
                    # Only full-credit terminal rewards count as wins.
                    return reward >= 1.0 - 1e-9, total_reward, turn + 1
                if truncated:
                    return False, total_reward, turn + 1

            return False, total_reward, task_spec.max_turns

        except Exception as e:
            logger.error(
                "[GEM-EVAL] Error playing %s episode %d: %s",
                task_spec.task_id, episode_idx, e,
            )
            raise

    async def evaluate_all(
        self,
        task_specs: List[GemTaskSpec],
    ) -> GemEvalResult:
        """Evaluate all specified GEM tasks.

        Flattens all (task, episode) pairs and runs them in parallel
        with a single tqdm progress bar. Each task uses its own
        episodes/max_turns/temperature from the spec.

        Args:
            task_specs: List of task specifications (with per-task config)

        Returns:
            GemEvalResult with aggregate statistics
        """
        if not task_specs:
            return GemEvalResult(
                overall_win_rate=0.0,
                overall_mean_reward=0.0,
                overall_mean_turns=0.0,
                total_episodes=0,
                total_tasks=0,
            )

        # Flatten all (task, episode) pairs — each task has its own episode count
        all_jobs: List[Tuple[GemTaskSpec, int]] = [
            (spec, ep_idx)
            for spec in task_specs
            for ep_idx in range(spec.episodes)
        ]

        semaphore = asyncio.Semaphore(self.max_concurrent)
        pbar = tqdm(total=len(all_jobs), desc="[GEM-EVAL]", unit="ep")

        # Per-task results accumulator. Keyed by (task_id, metric_suffix) so
        # the same task_id can appear twice with different prompt_style
        # (e.g., GPQA boxed + GPQA MCQ JSON in one eval run).
        task_wins: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        task_rewards: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        task_turns: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        task_errors: Dict[Tuple[str, str], int] = defaultdict(int)
        task_start_times: Dict[Tuple[str, str], float] = {}
        task_end_times: Dict[Tuple[str, str], float] = {}

        # Newline-terminated progress log (tqdm uses \r and is swallowed when
        # the stream is piped through grep / written to a non-tty log; this
        # guarantees visible progress in slurm logs and piped dry-runs).
        total_jobs = len(all_jobs)
        done_count = {"n": 0}
        log_every = max(1, total_jobs // 20)

        async def play_one(spec: GemTaskSpec, ep_idx: int) -> None:
            key = (spec.task_id, spec.metric_suffix)
            async with semaphore:
                # Track first-episode start time per task
                if key not in task_start_times:
                    task_start_times[key] = time.monotonic()
                try:
                    won, reward, num_turns = await self._play_episode(spec, ep_idx)
                    task_wins[key].append(1 if won else 0)
                    task_rewards[key].append(reward)
                    task_turns[key].append(num_turns)
                except Exception as e:
                    logger.error(
                        "[GEM-EVAL] Episode %d of %s%s failed: %s",
                        ep_idx, spec.task_id, spec.metric_suffix, e,
                    )
                    task_errors[key] += 1
                finally:
                    task_end_times[key] = time.monotonic()
                    pbar.update(1)
                    done_count["n"] += 1
                    if done_count["n"] % log_every == 0 or done_count["n"] == total_jobs:
                        logger.info(
                            "[GEM-EVAL] progress: %d/%d episodes done",
                            done_count["n"], total_jobs,
                        )

        eval_start = time.monotonic()
        await asyncio.gather(*[play_one(spec, ep) for spec, ep in all_jobs])
        pbar.close()
        eval_elapsed = time.monotonic() - eval_start

        # Log per-task timing
        for key in sorted(task_end_times.keys()):
            t_start = task_start_times.get(key, eval_start)
            t_end = task_end_times[key]
            logger.info(
                "[GEM-EVAL] %s%s: %.1fs",
                key[0], key[1], t_end - t_start,
            )
        logger.info("[GEM-EVAL] Total eval time: %.1fs", eval_elapsed)

        # Build per-task results. Iterate over specs directly to preserve
        # duplicate-task_id entries that differ only in metric_suffix.
        per_task_results: List[GemTaskResult] = []
        total_errors = 0

        for spec in task_specs:
            key = (spec.task_id, spec.metric_suffix)
            wins = task_wins.get(key, [])
            rewards = task_rewards.get(key, [])
            turns = task_turns.get(key, [])
            errors = task_errors.get(key, 0)
            total_errors += errors
            t_start = task_start_times.get(key, eval_start)
            t_end = task_end_times.get(key, eval_start)
            elapsed = t_end - t_start

            num_played = len(wins)
            if num_played == 0:
                per_task_results.append(GemTaskResult(
                    task_id=spec.task_id,
                    category=spec.category,
                    num_episodes=0, num_wins=0,
                    win_rate=0.0, mean_reward=0.0, mean_turns=0.0,
                    errors=errors,
                    elapsed_sec=elapsed,
                    metric_suffix=spec.metric_suffix,
                ))
            else:
                per_task_results.append(GemTaskResult(
                    task_id=spec.task_id,
                    category=spec.category,
                    num_episodes=num_played,
                    num_wins=sum(wins),
                    win_rate=sum(wins) / num_played,
                    mean_reward=sum(rewards) / num_played,
                    mean_turns=sum(turns) / num_played,
                    errors=errors,
                    elapsed_sec=elapsed,
                    metric_suffix=spec.metric_suffix,
                ))

        if not any(r.num_episodes > 0 for r in per_task_results):
            return GemEvalResult(
                overall_win_rate=0.0,
                overall_mean_reward=0.0,
                overall_mean_turns=0.0,
                total_episodes=0,
                total_tasks=len(task_specs),
                errors=total_errors,
            )

        # Aggregate
        total_wins = sum(r.num_wins for r in per_task_results)
        total_episodes = sum(r.num_episodes for r in per_task_results)
        overall_win_rate = total_wins / total_episodes if total_episodes > 0 else 0.0
        overall_mean_reward = (
            sum(r.mean_reward * r.num_episodes for r in per_task_results) / total_episodes
            if total_episodes > 0 else 0.0
        )
        overall_mean_turns = (
            sum(r.mean_turns * r.num_episodes for r in per_task_results) / total_episodes
            if total_episodes > 0 else 0.0
        )

        # Per-category aggregation
        per_category_metrics: Dict[str, Dict[str, float]] = {}
        category_wins: Dict[str, int] = {}
        category_episodes: Dict[str, int] = {}
        category_rewards: Dict[str, List[float]] = {}

        for result in per_task_results:
            cat = result.category
            if cat not in category_wins:
                category_wins[cat] = 0
                category_episodes[cat] = 0
                category_rewards[cat] = []
            category_wins[cat] += result.num_wins
            category_episodes[cat] += result.num_episodes
            category_rewards[cat].append(result.mean_reward)

        for cat in category_wins:
            eps = category_episodes[cat]
            if eps > 0:
                per_category_metrics[cat] = {
                    "win_rate": category_wins[cat] / eps,
                    "mean_reward": sum(category_rewards[cat]) / len(category_rewards[cat]),
                    "num_tasks": float(len(category_rewards[cat])),
                    "num_episodes": float(eps),
                }

        # Report Golden Goose reasoning_gym category rollups alongside the aggregate.
        gg_wins: Dict[str, int] = defaultdict(int)
        gg_eps: Dict[str, int] = defaultdict(int)
        gg_rewards: Dict[str, List[float]] = defaultdict(list)
        for result in per_task_results:
            gg = rg_golden_goose_category(result.task_id)
            if gg is None or result.num_episodes == 0:
                continue
            gkey = f"rg_{gg}"
            gg_wins[gkey] += result.num_wins
            gg_eps[gkey] += result.num_episodes
            gg_rewards[gkey].append(result.mean_reward)
        for gkey, eps in gg_eps.items():
            if eps > 0:
                per_category_metrics[gkey] = {
                    "win_rate": gg_wins[gkey] / eps,
                    "mean_reward": sum(gg_rewards[gkey]) / len(gg_rewards[gkey]),
                    "num_tasks": float(len(gg_rewards[gkey])),
                    "num_episodes": float(eps),
                }

        return GemEvalResult(
            overall_win_rate=overall_win_rate,
            overall_mean_reward=overall_mean_reward,
            overall_mean_turns=overall_mean_turns,
            total_episodes=total_episodes,
            total_tasks=len(per_task_results),
            per_task_results=per_task_results,
            per_category_metrics=per_category_metrics,
            errors=total_errors,
        )
