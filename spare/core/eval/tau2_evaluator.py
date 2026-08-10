"""tau2-bench evaluator for measuring model capability during training.

Evaluates the model being trained on tau2-bench domains (retail, telecom,
...). Results are logged to W&B under the ``gem_eval/tau2_*`` prefix so
both GEM and tau2 metrics cluster in one W&B group.

Unlike the GEM evaluator (which drives reset()/step() directly via our
ModelAdapter), this evaluator delegates to tau2's native ``run_tasks``
loop. The trained model is served over HTTP by sglang (behind the slime
router) and is called by tau2's ``LLMAgent`` via LiteLLM's ``openai/``
provider path. Tool calls are parsed server-side by sglang's Qwen tool
parser and returned as standard OpenAI ``message.tool_calls``. The user
simulator is routed to OpenRouter via LiteLLM.

Requires:
    pip install --ignore-requires-python \
        "git+https://github.com/sierra-research/tau2-bench.git"
    # Plus:
    #   git clone https://github.com/sierra-research/tau2-bench.git /path/to/data
    #   export TAU2_DATA_DIR=/path/to/data/data
    #   export OPENROUTER_API_KEY=sk-or-v1-...
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from spare.core.eval.tau2_tasks import (
    DEFAULT_OPENROUTER_API_BASE,
    Tau2TaskSpec,
)

logger = logging.getLogger(__name__)


# tau2-bench is optional so other GEM evaluations can run without it.
try:
    from tau2.data_model.simulation import Results, SimulationRun
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.run import get_tasks, run_tasks

    # Route tau2's module-level NL-assertion judge through the configured provider.
    try:
        import tau2.evaluator.evaluator_nl_assertions as _tau2_nl
        # Preserve tau2's default judge unless explicitly overridden.
        _NL_JUDGE_MODEL = os.environ.get(
            "TAU2_NL_ASSERTIONS_LLM", "openrouter/openai/gpt-4o-mini"
        )
        _tau2_nl.DEFAULT_LLM_NL_ASSERTIONS = _NL_JUDGE_MODEL
        _tau2_nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0}
        logger.info(
            "[TAU2-EVAL] NL-assertion judge model -> %s", _NL_JUDGE_MODEL,
        )
    except Exception as _nl_exc:  # pragma: no cover
        logger.warning(
            "[TAU2-EVAL] Could not patch NL-assertion judge model: %s", _nl_exc
        )

    _TAU2_AVAILABLE = True
    _TAU2_IMPORT_ERROR: Optional[Exception] = None
except Exception as _exc:  # pragma: no cover - import paths tested separately
    _TAU2_AVAILABLE = False
    _TAU2_IMPORT_ERROR = _exc
    Results = None  # type: ignore[assignment]
    SimulationRun = None  # type: ignore[assignment]
    EvaluationType = None  # type: ignore[assignment]
    get_tasks = None  # type: ignore[assignment]
    run_tasks = None  # type: ignore[assignment]


def _override_tau2_nl_judge(model: str) -> None:
    """Force tau2-bench's NL-assertion judge to route via the given model.

    Both the config and the evaluator's import-time binding must be updated.
    """
    if not _TAU2_AVAILABLE:
        return
    import tau2.config as _tau2_config
    import tau2.evaluator.evaluator_nl_assertions as _nl_eval_mod
    _tau2_config.DEFAULT_LLM_NL_ASSERTIONS = model
    _nl_eval_mod.DEFAULT_LLM_NL_ASSERTIONS = model
    logger.info("[TAU2-EVAL] NL-assertion judge overridden to: %s", model)


# LiteLLM label for our sglang-served model. SGLang ignores the request-body
# ``model`` field (one model per instance); LiteLLM uses the ``openai/`` prefix
# for provider routing. Keeping this out of LiteLLM's cost registry avoids
# noisy fake-cost log lines.
SGLANG_MODEL_NAME = "openai/sglang-local"

# The default matches tau2's no-retry evaluation protocol.
_TAU2_NUM_RETRIES = int(os.environ.get("TAU2_NUM_RETRIES", "0"))
# Per-call timeout bounds stalled requests without shortening long generations.
_TAU2_TIMEOUT = int(os.environ.get("TAU2_TIMEOUT", "600"))

# Hard cap on the tau2 branch inside spare_gem_eval_rollout. Beyond this,
# cancel the branch and return partial metrics rather than block training.
DEFAULT_BRANCH_TIMEOUT_SEC = int(os.environ.get("TAU2_BRANCH_TIMEOUT", "1800"))

# Infrastructure failures are errors; normal unsuccessful trials count against pass@1.
_ERROR_TERMINATION_REASONS = {
    "infrastructure_error",   # API disconnect, LLM provider down
    "unexpected_error",       # Unhandled exception in tau2 orchestrator
    "too_many_errors",        # Cumulative error threshold in run_tasks
    "timeout",                # Episode exceeded tau2's timeout
    "agent_error",            # Agent-side LLM repeatedly failed
    "user_error",             # User-sim LLM repeatedly failed
}


@dataclass
class Tau2SpecResult:
    """Result of evaluating a single Tau2TaskSpec."""
    label: str                   # e.g. "retail_test"
    domain: str
    split: str
    num_tasks: int               # expected tasks for this spec
    num_simulations: int         # actual simulations attempted (num_tasks * num_trials)
    num_passed: int              # simulations with reward ≈ 1.0 (official is_successful, 1e-6 tol)
    num_failed: int              # simulations not passed AND not an error
    errors: int                  # simulations that ended with an error termination_reason
    pass_at_1: float             # strict: num_passed / num_simulations (option B)
    pass_at_1_clean: float       # forensic: num_passed / (num_simulations - errors)
    mean_num_turns: float        # average ticks/turns across simulations that completed
    elapsed_sec: float = 0.0


@dataclass
class Tau2EvalResult:
    """Aggregate result from one tau2 evaluation rollout."""
    total_tasks: int
    total_simulations: int
    total_passed: int
    total_errors: int
    overall_pass_at_1: float
    per_spec_results: List[Tau2SpecResult] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def to_metrics_dict(self, prefix: str = "gem_eval") -> Dict[str, float]:
        """Flat metrics dict for W&B logging.

        All keys nest under ``<prefix>/tau2_*`` so tau2 metrics cluster
        with GEM metrics under a single W&B group without colliding with
        GEM's own ``category_*``/``task_*``/``overall_*`` keys.
        """
        metrics: Dict[str, float] = {
            f"{prefix}/tau2_overall_pass_at_1": self.overall_pass_at_1,
            f"{prefix}/tau2_total_tasks": float(self.total_tasks),
            f"{prefix}/tau2_total_simulations": float(self.total_simulations),
            f"{prefix}/tau2_total_passed": float(self.total_passed),
            f"{prefix}/tau2_total_errors": float(self.total_errors),
            f"{prefix}/tau2_elapsed_sec": self.elapsed_sec,
        }
        for spec in self.per_spec_results:
            base = f"{prefix}/tau2_{spec.label}"
            metrics[f"{base}/pass_at_1"] = spec.pass_at_1
            metrics[f"{base}/pass_at_1_clean"] = spec.pass_at_1_clean
            metrics[f"{base}/num_tasks"] = float(spec.num_tasks)
            metrics[f"{base}/num_simulations"] = float(spec.num_simulations)
            metrics[f"{base}/num_passed"] = float(spec.num_passed)
            metrics[f"{base}/errors"] = float(spec.errors)
            metrics[f"{base}/mean_num_turns"] = spec.mean_num_turns
            metrics[f"{base}/elapsed_sec"] = spec.elapsed_sec
        return metrics


def _quiet_noisy_loggers() -> None:
    """Suppress chatty upstream loggers that would flood training logs."""
    for name in ("litellm", "LiteLLM", "httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("tau2").setLevel(logging.INFO)


class Tau2Evaluator:
    """Evaluates the trained model on tau2-bench domains via tau2.run.run_tasks.

    The evaluator hits our sglang HTTP endpoint (via the slime router)
    as the agent and OpenRouter as the user simulator. It computes
    strict Pass@1 per spec and aggregates into ``Tau2EvalResult``.

    Args:
        sglang_base_url: Base URL of the slime router exposing sglang's
            OpenAI-compat chat/completions endpoint. Must already
            include the ``/v1`` suffix (e.g. ``http://10.0.0.1:8000/v1``).
            The slime router's catch-all proxy forwards this to sglang.
        openrouter_api_key: OpenRouter API key. If ``None``, reads from
            ``$OPENROUTER_API_KEY``. Missing key raises ``RuntimeError``.
        openrouter_api_base: Override for OpenRouter base URL (useful
            for testing against mock providers).
        branch_timeout_sec: Hard cap on ``evaluate_all`` as a whole.
            On timeout, returns partial metrics so training continues.
    """

    def __init__(
        self,
        sglang_base_url: str,
        openrouter_api_key: Optional[str] = None,
        openrouter_api_base: str = DEFAULT_OPENROUTER_API_BASE,
        branch_timeout_sec: float = DEFAULT_BRANCH_TIMEOUT_SEC,
    ) -> None:
        self.sglang_base_url = sglang_base_url
        self.openrouter_api_base = openrouter_api_base
        self.branch_timeout_sec = branch_timeout_sec

        # OpenRouter key (for openrouter/* user LLMs)
        self.openrouter_api_key = (
            openrouter_api_key
            or os.environ.get("OPENROUTER_API_KEY", "")
        )
        # OpenAI key (for openai/* user LLMs)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

        if not self.openrouter_api_key and not self.openai_api_key:
            raise RuntimeError(
                "[TAU2-EVAL] Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set. "
                "tau2 eval requires at least one to run the user simulator."
            )

        _quiet_noisy_loggers()

        if not _TAU2_AVAILABLE:
            logger.error(
                "[TAU2-EVAL] tau2-bench is not installed: %s. "
                "Install via: pip install --ignore-requires-python "
                "git+https://github.com/sierra-research/tau2-bench.git",
                _TAU2_IMPORT_ERROR,
            )

    # ------------------------------------------------------------------
    # User LLM routing

    def _resolve_user_llm_args(self, spec: Tau2TaskSpec) -> Dict[str, Any]:
        """Build llm_args_user dict based on the user_llm provider prefix.

        - ``openai/gpt-4o`` → uses OPENAI_API_KEY, hits api.openai.com
        - ``openrouter/openai/gpt-4o`` → uses OPENROUTER_API_KEY, hits openrouter
        - No prefix / other → falls back to OpenRouter
        """
        args: Dict[str, Any] = {
            "temperature": spec.user_temperature,
            "max_tokens": spec.user_max_tokens,
            "num_retries": _TAU2_NUM_RETRIES,
            "timeout": _TAU2_TIMEOUT,
        }

        user_llm = spec.user_llm
        if user_llm.startswith("openai/"):
            # Direct OpenAI — use OPENAI_API_KEY, no custom api_base
            args["api_key"] = self.openai_api_key
            # Don't set api_base — litellm defaults to api.openai.com
        else:
            # OpenRouter (default) — use OPENROUTER_API_KEY + openrouter base
            args["api_base"] = self.openrouter_api_base
            args["api_key"] = self.openrouter_api_key

        return args

    # ------------------------------------------------------------------
    # Per-spec evaluation

    def _load_spec_tasks(self, spec: Tau2TaskSpec) -> List[Any]:
        """Load the task list for a single spec using tau2.run.get_tasks.

        Raises:
            RuntimeError: if tau2 is not importable.
        """
        if not _TAU2_AVAILABLE:
            raise RuntimeError("tau2-bench is not installed")
        assert get_tasks is not None  # narrowing for the type checker
        return get_tasks(
            task_set_name=spec.domain,
            task_split_name=spec.split,
            num_tasks=spec.truncate,
        )

    def _run_spec_sync(
        self, spec: Tau2TaskSpec, tasks: List[Any]
    ) -> Any:
        """Call ``tau2.run.run_tasks`` synchronously with our LLM wiring.

        Returns the raw ``Results`` object (or raises). This is isolated in
        its own function so tests can patch it.
        """
        assert run_tasks is not None  # narrowing
        # Forward sampling overrides only when they are explicitly positive.
        agent_llm_args: Dict[str, Any] = {
            "api_base": self.sglang_base_url,
            "api_key": "dummy",
            "temperature": spec.agent_temperature,
            "max_tokens": spec.agent_max_tokens,
            "num_retries": _TAU2_NUM_RETRIES,
            "timeout": _TAU2_TIMEOUT,
        }
        if getattr(spec, "agent_top_p", 0.0) and spec.agent_top_p > 0.0:
            agent_llm_args["top_p"] = spec.agent_top_p
        if getattr(spec, "agent_top_k", 0) and spec.agent_top_k > 0:
            agent_llm_args["top_k"] = spec.agent_top_k

        return run_tasks(
            domain=spec.domain,
            tasks=tasks,
            agent="llm_agent",
            user="user_simulator",
            llm_agent=SGLANG_MODEL_NAME,
            llm_args_agent=agent_llm_args,
            llm_user=spec.user_llm,
            llm_args_user=self._resolve_user_llm_args(spec),
            num_trials=spec.num_trials,
            max_steps=spec.max_steps,
            max_concurrency=spec.max_concurrency,
            console_display=True,
            # Some task rewards require natural-language assertion evaluation.
            evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
        )

    def _aggregate_spec_result(
        self,
        spec: Tau2TaskSpec,
        num_tasks_expected: int,
        results: Any,
        elapsed_sec: float,
    ) -> Tau2SpecResult:
        """Convert a raw tau2 ``Results`` object into a Tau2SpecResult.

        Handles the case where results is None or has fewer/more sims
        than expected (e.g. tau2 bailed mid-run with max_errors).
        """
        simulations: List[Any] = []
        if results is not None and hasattr(results, "simulations"):
            simulations = list(results.simulations or [])

        num_simulations = len(simulations)
        expected_simulations = num_tasks_expected * spec.num_trials

        num_passed = 0
        num_failed = 0
        errors = 0
        turns_accum = 0.0
        turns_count = 0

        for sim in simulations:
            termination = getattr(sim, "termination_reason", None)
            if termination in _ERROR_TERMINATION_REASONS:
                errors += 1
                continue
            reward_info = getattr(sim, "reward_info", None)
            reward = getattr(reward_info, "reward", 0.0) if reward_info is not None else 0.0
            # Match tau2's 1e-6 success tolerance around a reward of 1.0.
            if (1 - 1e-6) <= reward <= (1 + 1e-6):
                num_passed += 1
            else:
                num_failed += 1
            ticks = getattr(sim, "ticks", None)
            if ticks is not None:
                turns_accum += float(ticks)
                turns_count += 1

        # Missing simulations count as errors to preserve the expected denominator.
        missing = max(0, expected_simulations - num_simulations)
        if missing > 0:
            logger.warning(
                "[TAU2-EVAL] %s: expected %d simulations, got %d — counting %d as errors",
                spec.label, expected_simulations, num_simulations, missing,
            )
            errors += missing

        total_sims = expected_simulations  # strict denominator
        pass_at_1 = (num_passed / total_sims) if total_sims > 0 else 0.0
        clean_denom = max(1, total_sims - errors)
        pass_at_1_clean = num_passed / clean_denom if (total_sims - errors) > 0 else 0.0
        mean_turns = (turns_accum / turns_count) if turns_count > 0 else 0.0

        return Tau2SpecResult(
            label=spec.label,
            domain=spec.domain,
            split=spec.split,
            num_tasks=num_tasks_expected,
            num_simulations=total_sims,
            num_passed=num_passed,
            num_failed=num_failed,
            errors=errors,
            pass_at_1=pass_at_1,
            pass_at_1_clean=pass_at_1_clean,
            mean_num_turns=mean_turns,
            elapsed_sec=elapsed_sec,
        )

    async def _evaluate_spec(self, spec: Tau2TaskSpec) -> Tau2SpecResult:
        """Run one spec and return its aggregated result.

        Catches any exception raised by ``run_tasks`` and counts all
        expected simulations as errors for that spec. Training continues.
        """
        t0 = time.monotonic()
        try:
            tasks = self._load_spec_tasks(spec)
        except Exception as exc:
            logger.error(
                "[TAU2-EVAL] %s: failed to load tasks: %s", spec.label, exc,
            )
            return Tau2SpecResult(
                label=spec.label, domain=spec.domain, split=spec.split,
                num_tasks=0, num_simulations=0, num_passed=0,
                num_failed=0, errors=0,
                pass_at_1=0.0, pass_at_1_clean=0.0, mean_num_turns=0.0,
                elapsed_sec=time.monotonic() - t0,
            )

        num_tasks = len(tasks)
        logger.info(
            "[TAU2-EVAL] %s: running %d tasks × %d trials (max_concurrency=%d)",
            spec.label, num_tasks, spec.num_trials, spec.max_concurrency,
        )

        # Per-spec timeout: max_steps * 30 seconds per step per simulation.
        # Cap generous enough to handle full-batch async execution.
        spec_timeout = max(300.0, spec.max_steps * 30.0)

        def _sync_call() -> Any:
            return self._run_spec_sync(spec, tasks)

        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(_sync_call),
                timeout=spec_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                "[TAU2-EVAL] %s: spec timed out after %.0fs — counting all as errors",
                spec.label, spec_timeout,
            )
            return Tau2SpecResult(
                label=spec.label, domain=spec.domain, split=spec.split,
                num_tasks=num_tasks,
                num_simulations=num_tasks * spec.num_trials,
                num_passed=0, num_failed=0,
                errors=num_tasks * spec.num_trials,
                pass_at_1=0.0, pass_at_1_clean=0.0, mean_num_turns=0.0,
                elapsed_sec=time.monotonic() - t0,
            )
        except Exception as exc:
            logger.error(
                "[TAU2-EVAL] %s: run_tasks crashed: %s — counting all as errors",
                spec.label, exc,
            )
            return Tau2SpecResult(
                label=spec.label, domain=spec.domain, split=spec.split,
                num_tasks=num_tasks,
                num_simulations=num_tasks * spec.num_trials,
                num_passed=0, num_failed=0,
                errors=num_tasks * spec.num_trials,
                pass_at_1=0.0, pass_at_1_clean=0.0, mean_num_turns=0.0,
                elapsed_sec=time.monotonic() - t0,
            )

        elapsed = time.monotonic() - t0
        result = self._aggregate_spec_result(spec, num_tasks, results, elapsed)
        logger.info(
            "[TAU2-EVAL] %s: pass_at_1=%.3f (%d/%d passed, %d errors) in %.1fs",
            spec.label, result.pass_at_1, result.num_passed,
            result.num_simulations, result.errors, elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Public API

    async def evaluate_all(
        self, specs: List[Tau2TaskSpec]
    ) -> Tau2EvalResult:
        """Run every spec sequentially and aggregate.

        Specs run in-order (not in parallel across specs) so that the
        slime router isn't oversubscribed. Within a spec, tau2 manages
        its own concurrency via ``max_concurrency``.

        If tau2 is not importable, returns an empty result and logs a
        single warning — training continues.
        """
        if not specs:
            return Tau2EvalResult(
                total_tasks=0, total_simulations=0, total_passed=0,
                total_errors=0, overall_pass_at_1=0.0,
                per_spec_results=[], elapsed_sec=0.0,
            )

        if not _TAU2_AVAILABLE:
            logger.warning(
                "[TAU2-EVAL] tau2-bench not installed — skipping %d specs",
                len(specs),
            )
            return Tau2EvalResult(
                total_tasks=0, total_simulations=0, total_passed=0,
                total_errors=0, overall_pass_at_1=0.0,
                per_spec_results=[], elapsed_sec=0.0,
            )

        total_tasks = 0
        if _TAU2_AVAILABLE:
            for s in specs:
                try:
                    total_tasks += len(self._load_spec_tasks(s)) * s.num_trials
                except Exception:
                    pass  # will be caught again in _evaluate_spec
        t0 = time.monotonic()
        per_spec: List[Tau2SpecResult] = []
        pbar = tqdm(total=total_tasks, desc="[TAU2-EVAL]", unit="task")

        async def _run_all() -> None:
            for spec in specs:
                result = await self._evaluate_spec(spec)
                per_spec.append(result)
                pbar.update(result.num_simulations)
                pbar.set_postfix(
                    spec=spec.label,
                    pass_at_1=f"{result.pass_at_1:.2f}",
                    errors=result.errors,
                )

        try:
            await asyncio.wait_for(_run_all(), timeout=self.branch_timeout_sec)
        except asyncio.TimeoutError:
            logger.error(
                "[TAU2-EVAL] Branch hit hard timeout of %.0fs — returning partial results (%d/%d specs done)",
                self.branch_timeout_sec, len(per_spec), len(specs),
            )
        finally:
            pbar.close()

        elapsed = time.monotonic() - t0

        total_tasks = sum(r.num_tasks for r in per_spec)
        total_simulations = sum(r.num_simulations for r in per_spec)
        total_passed = sum(r.num_passed for r in per_spec)
        total_errors = sum(r.errors for r in per_spec)
        overall_pass_at_1 = (
            total_passed / total_simulations if total_simulations > 0 else 0.0
        )

        if total_errors > 0.1 * max(total_simulations, 1):
            logger.warning(
                "[TAU2-EVAL] High error rate: %d/%d simulations errored (%.1f%%)",
                total_errors, total_simulations,
                100.0 * total_errors / max(total_simulations, 1),
            )

        logger.info(
            "[TAU2-EVAL] Complete: overall_pass_at_1=%.3f (%d/%d passed, %d errors, %d specs) in %.1fs",
            overall_pass_at_1, total_passed, total_simulations,
            total_errors, len(per_spec), elapsed,
        )

        return Tau2EvalResult(
            total_tasks=total_tasks,
            total_simulations=total_simulations,
            total_passed=total_passed,
            total_errors=total_errors,
            overall_pass_at_1=overall_pass_at_1,
            per_spec_results=per_spec,
            elapsed_sec=elapsed,
        )
