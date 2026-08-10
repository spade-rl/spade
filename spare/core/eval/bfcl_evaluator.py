"""BFCL V4 evaluator for measuring function-calling accuracy during training.

Evaluates the model on Berkeley Function Calling Leaderboard V4 problems.
Uses AST-based scoring for single-turn (copied from official BFCL repo)
and execution-based scoring for multi-turn (imported from official BFCL repo).
Results are logged to W&B under the gem_eval/bfcl_* prefix.

This implementation reproduces the Qwen3-4B-Instruct-2507 official numbers
(live_simple 79.07%, live_multiple 76.16%, multi_turn_base 26.5%).
"""

import asyncio
import copy
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from spare.core.eval.bfcl_ast_checker import ast_checker, Language
from spare.core.model_adapter import ModelAdapter
from spare.core.utils.tool_call_parser import parse_tool_calls

logger = logging.getLogger(__name__)


def _purge_bfcl_globals(run_id: str) -> int:
    """Drop all cached backend instances belonging to ``run_id`` from
    ``bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils.globals()``.

    The official BFCL backend caches Python class instances (GorillaFileSystem,
    MessageAPI, …) in its own module-level globals() keyed by
    ``f"{model_name}_{test_entry_id}_{class_name}_instance"``. We thread a
    unique ``run_id`` into every ``model_name`` we pass so each evaluator
    invocation gets a fresh namespace. This helper purges that namespace once
    the eval is complete so we don't leak memory across many evals in a long
    training loop. Returns # of entries purged (for logging / tests).
    """
    try:
        import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mt
    except ImportError:
        return 0
    g = mt.__dict__
    needles = (
        f"spare_gen_{run_id}_",
        f"spare_eval_{run_id}_",
        f"spare_eval_{run_id}_ground_truth_",
    )
    victims = [k for k in g if isinstance(k, str) and any(n in k for n in needles)]
    for k in victims:
        try:
            del g[k]
        except KeyError:
            pass
    return len(victims)

# Default path to official BFCL repo (inside apptainer container). Override
# with BFCL_REPO_PATH env var when running outside the image.
_BFCL_REPO_PATH = "/workspace/spare-workspace/bfcl/gorilla/berkeley-function-call-leaderboard"


def _ensure_bfcl_importable() -> None:
    """Add the official BFCL repo to sys.path so the multi_turn execution
    checker / utils can be imported on first use."""
    repo = os.environ.get("BFCL_REPO_PATH", _BFCL_REPO_PATH)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _convert_to_function_call(function_call_list: list) -> List[str]:
    """Convert [{"func_name": {"param": value}}] to ["func_name(param=value)"].

    Mirrors official BFCL model_handler/utils.py:convert_to_function_call so
    that decoded calls can be fed to execute_multi_turn_func_call.
    """
    if isinstance(function_call_list, dict):
        function_call_list = [function_call_list]
    execution_list = []
    for function_call in function_call_list:
        for key, value in function_call.items():
            if isinstance(value, str):
                # value may be a JSON-encoded args dict (most cases) OR an already-
                # decoded plain value (rare, e.g. when parse_tool_calls returned a
                # string verbatim). Don't crash on the latter.
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    logger.debug(
                        f"[BFCL] _convert_to_function_call: value not JSON for "
                        f"key={key!r}, treating as empty args. raw={value[:100]!r}"
                    )
                    value = {}
            if not isinstance(value, dict):
                # Defensive: BFCL expects dict args; coerce non-dict to empty so
                # downstream eval() doesn't get a malformed expression.
                value = {}
            execution_list.append(
                f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})"
            )
    return execution_list


@dataclass
class BfclEvalResult:
    """Results from BFCL evaluation."""
    total: int = 0
    correct: int = 0
    per_category: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    elapsed_sec: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)

    # BFCL v3 Overall equally weights Non-Live, Live, and Multi-Turn groups.
    # Non-Live and Multi-Turn use category means; Live uses a micro-average.
    #   Non-Live   = UNWEIGHTED mean of per-category acc
    #   Live       = ENTRY-COUNT-WEIGHTED mean (micro within the live group)
    #   Multi-Turn = UNWEIGHTED mean of per-category acc
    _V3_NON_LIVE = (
        "simple", "multiple", "parallel", "parallel_multiple",
        "java", "javascript", "irrelevance",
    )
    _V3_LIVE = (
        "live_simple", "live_multiple", "live_parallel",
        "live_parallel_multiple", "live_relevance", "live_irrelevance",
    )
    _V3_MULTI_TURN = (
        "multi_turn_base", "multi_turn_miss_func",
        "multi_turn_miss_param", "multi_turn_long_context",
    )

    def _group_acc(self, cats: tuple, weighted: bool) -> Optional[float]:
        """Group accuracy over the categories present in per_category.
        weighted=True → entry-count-weighted (micro within group, used for Live);
        weighted=False → unweighted mean of per-category accuracies (macro)."""
        present = [c for c in cats if c in self.per_category]
        if not present:
            return None
        if weighted:
            tot = sum(self.per_category[c]["total"] for c in present)
            cor = sum(self.per_category[c]["correct"] for c in present)
            return cor / tot if tot else None
        accs = [
            self.per_category[c]["correct"] / max(self.per_category[c]["total"], 1)
            for c in present
        ]
        return sum(accs) / len(accs)

    def official_overall_v3(self) -> Optional[float]:
        """Official gorilla v3 Overall = mean(Non-Live, Live, Multi-Turn), equal
        1/3 per group. Returns None if no group ran. NOTE: only meaningful when
        all three groups' categories are configured (the Non-Live group needs
        simple/multiple/parallel/parallel_multiple/java/javascript/irrelevance)."""
        groups = [
            self._group_acc(self._V3_NON_LIVE, weighted=False),
            self._group_acc(self._V3_LIVE, weighted=True),
            self._group_acc(self._V3_MULTI_TURN, weighted=False),
        ]
        present = [g for g in groups if g is not None]
        return sum(present) / len(present) if present else None

    def to_metrics_dict(
        self, prefix: str = "gem_eval", version_tag: str = ""
    ) -> Dict[str, float]:
        """Flat metrics for W&B. ``version_tag`` (e.g. ``"v3"``) infixes the
        key so multiple BFCL versions can be logged side-by-side without
        collision. Empty tag (the default) preserves the original ``bfcl_*``
        keys used by the existing v4 pass."""
        tag = f"{version_tag}_" if version_tag else ""
        metrics = {
            f"{prefix}/bfcl_{tag}accuracy": self.accuracy,
            f"{prefix}/bfcl_{tag}total": float(self.total),
            f"{prefix}/bfcl_{tag}correct": float(self.correct),
            f"{prefix}/bfcl_{tag}elapsed_sec": self.elapsed_sec,
        }
        for cat, stats in self.per_category.items():
            metrics[f"{prefix}/bfcl_{tag}{cat}/accuracy"] = stats["correct"] / max(stats["total"], 1)
        # Official gorilla v3 group accuracies + Overall (1/3 per group). This,
        # NOT `accuracy` (micro-avg), is the number comparable to the published
        # BFCL-v3 Overall (Qwen3-4B-Inst-2507 = 61.9).
        _nl = self._group_acc(self._V3_NON_LIVE, weighted=False)
        _lv = self._group_acc(self._V3_LIVE, weighted=True)
        _mt = self._group_acc(self._V3_MULTI_TURN, weighted=False)
        _ov = self.official_overall_v3()
        if _nl is not None:
            metrics[f"{prefix}/bfcl_{tag}non_live_acc"] = _nl
        if _lv is not None:
            metrics[f"{prefix}/bfcl_{tag}live_acc"] = _lv
        if _mt is not None:
            metrics[f"{prefix}/bfcl_{tag}multi_turn_acc"] = _mt
        if _ov is not None:
            metrics[f"{prefix}/bfcl_{tag}official_overall_v3"] = _ov
        return metrics


class BfclEvaluator:
    """Evaluates function-calling using official BFCL V4 scoring."""

    def __init__(
        self,
        model: ModelAdapter,
        max_concurrent: int = 64,
        temperature: float = 0.001,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        max_context_length: Optional[int] = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Over-window prompts score as failed without reaching the model server.
        # The check measures tokens but never mutates the submitted prompt.
        self.max_context_length = max_context_length

    async def evaluate_all(
        self,
        dataset_dir: str,
        categories: Optional[List[str]] = None,
        version: str = "v4",
    ) -> BfclEvalResult:
        """Run BFCL evaluation with AST scoring (single-turn) and execution scoring (multi-turn).

        ``version`` selects the data-file prefix (``BFCL_{version}_{cat}.json``).
        Default ``v4`` preserves the original behavior. Set ``v3`` (with a
        ``dataset_dir`` pointing at a BFCL-v3 data tree) to reproduce the
        BFCL-v3 multi-turn numbers reported by Qwen3 / EnvScaler / SEAL. The
        multi-turn execution backend (GorillaFileSystem etc.) is supplied by
        the installed ``bfcl_eval`` package and is version-independent; only
        the problem data, possible_answer, and multi_turn_func_doc are read
        from ``dataset_dir``.
        """
        if categories is None:
            categories = [
                "live_simple", "live_multiple", "live_parallel",
                "live_parallel_multiple", "live_relevance", "live_irrelevance",
                "multi_turn_base", "multi_turn_miss_func",
                "multi_turn_miss_param", "multi_turn_long_context",
            ]

        start = time.time()
        result = BfclEvalResult()
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Isolate BFCL's process-global multi-turn backend cache across runs.
        self._run_id = uuid.uuid4().hex[:8]
        logger.debug(f"[BFCL] evaluate_all run_id={self._run_id}")

        # Keep canonical category keys while resolving BFCL v4 filenames.
        _v4_file_alias = {
            "simple": "simple_python",
            "java": "simple_java",
            "javascript": "simple_javascript",
        }

        for cat in categories:
            file_cat = _v4_file_alias.get(cat, cat) if version == "v4" else cat
            path = Path(dataset_dir) / f"BFCL_{version}_{file_cat}.json"
            if not path.exists():
                logger.warning(f"[BFCL] Missing dataset: {path}")
                continue

            # Load problems
            problems = []
            with open(path) as f:
                for line in f:
                    problems.append(json.loads(line))

            # Load possible answers (ground truth)
            pa_path = Path(dataset_dir) / "possible_answer" / f"BFCL_{version}_{file_cat}.json"
            possible_answers = {}
            if pa_path.exists():
                with open(pa_path) as f:
                    for line in f:
                        pa = json.loads(line)
                        possible_answers[pa["id"]] = pa["ground_truth"]

            is_multi_turn = cat.startswith("multi_turn")
            is_irrelevance = "irrelevance" in cat
            logger.info(f"[BFCL] Evaluating {cat}: {len(problems)} problems")

            if is_multi_turn:
                cat_correct = await self._eval_multi_turn_category(
                    problems, possible_answers, cat, semaphore, dataset_dir
                )
            else:
                cat_correct = 0
                tasks = []
                for problem in problems:
                    pa = possible_answers.get(problem.get("id", ""))
                    tasks.append(self._eval_single_turn_problem(
                        problem, pa, cat, is_irrelevance, semaphore
                    ))
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        continue
                    if r:
                        cat_correct += 1

            result.total += len(problems)
            result.correct += cat_correct
            result.per_category[cat] = {"total": len(problems), "correct": cat_correct}
            acc = cat_correct / max(len(problems), 1)
            logger.info(f"[BFCL] {cat}: {cat_correct}/{len(problems)} ({acc:.1%})")

        result.elapsed_sec = time.time() - start
        logger.info(f"[BFCL] Overall: {result.correct}/{result.total} ({result.accuracy:.1%}), elapsed={result.elapsed_sec:.1f}s")
        # Avoid retaining BFCL backend state across training-loop evaluations.
        purged = _purge_bfcl_globals(self._run_id)
        if purged:
            logger.debug(f"[BFCL] purged {purged} cached backend instances for run_id={self._run_id}")
        return result

    # ── Single-turn evaluation (AST-based) ──────────────────────────────

    async def _eval_single_turn_problem(
        self, problem, possible_answer, category, is_irrelevance, semaphore,
    ) -> bool:
        """Evaluate one single-turn problem using AST checker."""
        async with semaphore:
            try:
                content = await self._generate_single_turn(problem)

                functions = problem.get("function", [])
                tools = [{"type": "function", "function": f} for f in functions]
                parsed = parse_tool_calls(content, tools if tools else None)
                has_calls = bool(parsed["calls"])

                if is_irrelevance:
                    return not has_calls

                if not possible_answer:
                    return has_calls

                if not has_calls:
                    return False

                model_output = []
                for call in parsed["calls"]:
                    model_output.append({call["name"]: call["arguments"]})

                # Check JavaScript before Java because the category names overlap.
                _lang = (
                    Language.JAVASCRIPT if "javascript" in category
                    else Language.JAVA if "java" in category
                    else Language.PYTHON
                )
                result = ast_checker(
                    func_description=functions,
                    model_output=model_output,
                    possible_answer=possible_answer,
                    language=_lang,
                    test_category=category,
                    model_name=f"spare_eval_{self._run_id}",
                )
                return result["valid"]

            except Exception as e:
                logger.debug(f"[BFCL] Single-turn eval failed: {e}")
                return False

    async def _generate_single_turn(self, problem: Dict) -> str:
        question = problem["question"][0][0]["content"]
        functions = problem.get("function", [])

        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})
        input_ids = self.model.apply_template(
            messages,
            # HF chat templates and the official Qwen handler expect raw schemas.
            chat_template_kwargs_override={"tools": functions} if functions else None,
        )
        # Skip a generation that would overflow the model's context window (only
        # measures length; see __init__). Avoids the circuit-breaker storm.
        if (
            self.max_context_length is not None
            and len(input_ids) + self.max_tokens > self.max_context_length
        ):
            logger.warning(
                f"[BFCL] single-turn prompt {len(input_ids)} + {self.max_tokens} "
                f"tokens exceeds model context {self.max_context_length}; skipping "
                f"generation (scored as empty, avoids SGLang circuit-breaker)."
            )
            return ""
        results = await self.model.generate_async(
            messages=messages, input_ids=input_ids,
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        content = results[0].get("text", "") if results else ""
        if "</think>" in content:
            content = content[content.index("</think>") + 8:].strip()
        return content

    # ── Multi-turn evaluation (execution-based, official BFCL checker) ──

    async def _eval_multi_turn_category(
        self, problems, possible_answers, category, semaphore, dataset_dir,
    ) -> int:
        """Evaluate all problems in a multi-turn category using official execution-based checker.

        Generation uses real tool execution (not stubs) to match official inference.
        Scoring uses official multi_turn_checker (execution + state comparison).
        """
        _ensure_bfcl_importable()
        try:
            from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
                multi_turn_checker,
                multi_turn_irrelevance_checker,
            )
            from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
                execute_multi_turn_func_call,
            )
        except ImportError as e:
            logger.error(f"[BFCL] Cannot import multi_turn eval modules: {e}")
            return 0

        is_miss_func = "miss_func" in category
        is_miss_param = "miss_param" in category

        tasks = []
        for problem in problems:
            pa = possible_answers.get(problem.get("id", ""))
            tasks.append(self._eval_multi_turn_problem(
                problem, pa, category, semaphore, dataset_dir,
                multi_turn_checker, multi_turn_irrelevance_checker,
                execute_multi_turn_func_call,
                is_miss_func, is_miss_param,
            ))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        correct = 0
        for r in results:
            if isinstance(r, Exception):
                logger.debug(f"[BFCL] Multi-turn exception: {r}")
                continue
            if r:
                correct += 1
        return correct

    async def _eval_multi_turn_problem(
        self, problem, ground_truth_list, category, semaphore, dataset_dir,
        multi_turn_checker, multi_turn_irrelevance_checker,
        execute_multi_turn_func_call,
        is_miss_func, is_miss_param,
    ) -> bool:
        """Evaluate one multi-turn problem using official execution-based checker."""
        async with semaphore:
            try:
                # Generate per-turn model responses with real tool execution
                per_turn_decoded = await self._generate_multi_turn_decoded(
                    problem, dataset_dir, execute_multi_turn_func_call
                )

                if not ground_truth_list:
                    return False

                if len(per_turn_decoded) != len(ground_truth_list):
                    return False

                # Official BFCL uses only multi_turn_checker for multi-turn tasks,
                # including miss_func and miss_param categories.

                test_entry = copy.deepcopy(problem)
                result = multi_turn_checker(
                    multi_turn_model_result_list_decoded=per_turn_decoded,
                    multi_turn_ground_truth_list=ground_truth_list,
                    test_entry=test_entry,
                    test_category=category,
                    model_name=f"spare_eval_{self._run_id}",
                )
                return result["valid"]

            except Exception as e:
                logger.debug(f"[BFCL] Multi-turn eval failed for {problem.get('id', '?')}: {e}")
                return False

    async def _generate_multi_turn_decoded(
        self, problem: Dict, dataset_dir: str,
        execute_multi_turn_func_call,
    ) -> List[List[List[str]]]:
        """Generate multi-turn responses with real tool execution.

        Instead of returning stub tool results, actually executes the model's
        function calls against the official BFCL Python class instances
        (GorillaFileSystem, TwitterAPI, etc.) and feeds real results back
        to the model. This matches the official BFCL inference pipeline.

        Returns per-turn decoded results:
            result[turn_index][step_index] = ["func_name(param=value)", ...]
        """
        turns = problem.get("question", [])
        all_functions = self._load_multi_turn_functions(problem, dataset_dir)

        # Some functions are revealed only at designated turns; excluded
        # functions are never exposed to the model.
        missed_function = problem.get("missed_function", {}) or {}
        excluded_function = set(problem.get("excluded_function", []) or [])
        # Functions that will be revealed later (held out from initial tool set).
        held_out_names = {n for fns in missed_function.values() for n in fns}
        # Initial tools = all - excluded - all_held_out
        always_excluded = excluded_function | held_out_names
        functions = [f for f in all_functions if f["name"] not in always_excluded]
        tools = [{"type": "function", "function": f} for f in functions]
        # Map name -> function spec for fast lookup when revealing held-out funcs.
        all_funcs_by_name = {f["name"]: f for f in all_functions}

        initial_config = problem.get("initial_config", {})
        involved_classes = problem.get("involved_classes", [])
        test_entry_id = problem.get("id", "unknown")
        is_long_context = "long_context" in test_entry_id

        max_steps_per_turn = 20  # Official BFCL MAXIMUM_STEP_LIMIT
        mt_max_tokens = max(self.max_tokens, 4096)
        messages: List[Dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        per_turn_decoded: List[List[List[str]]] = []

        # Placeholder shown to the agent when held-out functions are revealed.
        # Matches BFCL's DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC.
        HOLDOUT_REVEAL_MSG = (
            "I have updated some more functions you can choose from. What about now?"
        )

        for turn_idx, turn_msgs in enumerate(turns):
            # Reveal held-out functions for this turn if any.
            new_funcs = missed_function.get(str(turn_idx), [])
            if new_funcs:
                for n in new_funcs:
                    f = all_funcs_by_name.get(n)
                    if f and not any(t["function"]["name"] == n for t in tools):
                        tools.append({"type": "function", "function": f})

            # If question is empty AND this is a holdout-reveal turn, inject the
            # placeholder user message so the agent can use the new function.
            if not turn_msgs:
                if new_funcs:
                    turn_msgs = [{"role": "user", "content": HOLDOUT_REVEAL_MSG}]
                else:
                    per_turn_decoded.append([])
                    continue

            for msg in turn_msgs:
                messages.append(msg)

            # Multi-step loop within each turn: keep generating until model
            # stops producing tool calls, matching official BFCL inference.
            turn_step_results: List[List[str]] = []
            for step in range(max_steps_per_turn):
                input_ids = self.model.apply_template(
                    messages,
                    # The template uses raw schemas; holdout checks retain wrapped tools.
                    chat_template_kwargs_override=(
                        {"tools": [t["function"] for t in tools]} if tools else None
                    ),
                )
                # Skip (don't fire) a generation that would overflow the model's
                # context window — a doomed request trips SGLang's circuit breaker
                # and takes the colocated trainer down. Measures len(input_ids) only.
                if (
                    self.max_context_length is not None
                    and len(input_ids) + mt_max_tokens > self.max_context_length
                ):
                    logger.warning(
                        f"[BFCL] {test_entry_id} turn={turn_idx} step={step}: prompt "
                        f"{len(input_ids)} + {mt_max_tokens} tokens exceeds model "
                        f"context {self.max_context_length}; skipping generation "
                        f"(task scored as failed, avoids SGLang circuit-breaker)."
                    )
                    break
                results = await self.model.generate_async(
                    messages=messages, input_ids=input_ids,
                    temperature=self.temperature, max_tokens=mt_max_tokens,
                )
                raw_content = results[0].get("text", "") if results else ""
                content = raw_content
                if "</think>" in content:
                    content = content[content.index("</think>") + 8:].strip()

                messages.append({"role": "assistant", "content": content})

                # Parse tool calls
                parsed = parse_tool_calls(content, tools if tools else None)

                if not parsed["calls"]:
                    # No tool calls → end of this turn
                    break

                model_output = [{c["name"]: c["arguments"]} for c in parsed["calls"]]
                decoded = _convert_to_function_call(model_output)
                turn_step_results.append(decoded)

                # Execute function calls and feed real results back. model_name
                # carries run_id so per-evaluate_all() cache slots stay isolated.
                try:
                    exec_results, _ = execute_multi_turn_func_call(
                        func_call_list=decoded,
                        initial_config=initial_config,
                        involved_classes=involved_classes,
                        model_name=f"spare_gen_{self._run_id}_{test_entry_id}",
                        test_entry_id=test_entry_id,
                        long_context=is_long_context,
                    )
                    for exec_result in exec_results:
                        messages.append({"role": "tool", "content": str(exec_result)})
                except Exception as exec_err:
                    # Preserve tool failures in both logs and model-visible history.
                    logger.warning(
                        f"[BFCL] {test_entry_id} turn={turn_idx} step={step}: "
                        f"execute_multi_turn_func_call failed ({type(exec_err).__name__}: "
                        f"{str(exec_err)[:200]}); feeding error back to model."
                    )
                    err_payload = json.dumps({
                        "status": "error",
                        "error_type": type(exec_err).__name__,
                        "error_message": str(exec_err)[:500],
                    })
                    for c in parsed["calls"]:
                        messages.append({"role": "tool", "content": err_payload})

                if test_entry_id.endswith(("_0", "_1", "_2")):
                    logger.info(
                        f"[BFCL-DBG] {test_entry_id} turn={turn_idx} step={step} "
                        f"n_calls={len(parsed['calls'])} decoded={decoded}"
                    )

            per_turn_decoded.append(turn_step_results)

        return per_turn_decoded

    # ── Shared helpers ──────────────────────────────────────────────────

    def _load_multi_turn_functions(self, problem: Dict, dataset_dir: str) -> List[Dict]:
        """Load function definitions for multi-turn problems from doc files."""
        CLASS_TO_FILE = {
            "GorillaFileSystem": "gorilla_file_system.json",
            "MathAPI": "math_api.json",
            "MessageAPI": "message_api.json",
            "TicketAPI": "ticket_api.json",
            "TradingBot": "trading_bot.json",
            "TravelAPI": "travel_booking.json",
            "TwitterAPI": "posting_api.json",
            "VehicleControlAPI": "vehicle_control.json",
        }
        functions = problem.get("function", [])
        if functions:
            return functions
        involved_classes = problem.get("involved_classes", [])
        func_doc_dir = Path(dataset_dir) / "multi_turn_func_doc"
        for cls_name in involved_classes:
            fname = CLASS_TO_FILE.get(cls_name)
            if not fname:
                continue
            fpath = func_doc_dir / fname
            if not fpath.exists():
                continue
            with open(fpath) as f:
                for line in f:
                    func = json.loads(line)
                    functions.append({
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    })
        return functions


def load_bfcl_eval_config(config_path: str) -> Optional[Dict]:
    """Load BFCL eval config from the unified GEM/tau2/BFCL YAML."""
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("bfcl_eval")
    except Exception as e:
        logger.debug(f"[BFCL] No bfcl_eval config: {e}")
        return None
