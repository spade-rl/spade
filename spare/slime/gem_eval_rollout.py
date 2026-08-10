"""GEM + tau2 + BFCL evaluation rollout for Slime backend.

Called during evaluation rollouts when GEM, tau2, and/or BFCL eval is configured.
Creates a SlimeModelAdapter (for GEM/BFCL), runs GemEvaluator, then runs
Tau2Evaluator against the slime router + OpenRouter, then runs BfclEvaluator
for function-calling accuracy, and returns merged metrics in Slime's
RolloutFnEvalOutput format.

Also runs Slime's built-in eval if eval datasets are configured.

All sections are optional (per-section opt-in via the YAML file passed to
--spare-gem-eval-config). If a section is absent, that branch silently skips.
"""

import asyncio
import json
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Callable, Dict, Optional

from slime.rollout.base_types import RolloutFnEvalOutput

from spare.core.eval.gem_evaluator import GemEvaluator, GemEvalResult
from spare.core.eval.gem_tasks import load_gem_eval_config
from spare.core.eval.tau2_evaluator import Tau2Evaluator, Tau2EvalResult
from spare.core.eval.tau2_tasks import load_tau2_eval_config
from spare.core.eval.bfcl_evaluator import BfclEvaluator, BfclEvalResult, load_bfcl_eval_config
from spare.slime.model_adapter import create_slime_model_adapter

logger = logging.getLogger(__name__)


def _resolve_slime_router_url(args: Namespace) -> str:
    """Return the OpenAI-compat base URL of the slime router.

    The slime router is a fastapi catch-all proxy (router.py line 73) that
    forwards any path, including /v1/chat/completions, to a backend sglang
    worker. We hit it as the agent LLM for tau2.
    """
    ip = args.sglang_router_ip
    port = args.sglang_router_port
    return f"http://{ip}:{port}/v1"


def _run_gem_branch(
    args: Namespace,
) -> Dict[str, float]:
    """Run the GEM evaluation branch. Returns metrics dict (empty on skip)."""
    if not args.spare_gem_eval_config:
        return {}

    defaults, task_specs = load_gem_eval_config(args.spare_gem_eval_config)
    if not task_specs:
        logger.info("[GEM-EVAL] No GEM tasks in config, skipping GEM branch")
        return {}

    model_adapter = create_slime_model_adapter(args)
    evaluator = GemEvaluator(
        model=model_adapter,
        max_concurrent=defaults.max_concurrent,
    )

    total_episodes = sum(spec.episodes for spec in task_specs)
    logger.info(
        "[GEM-EVAL] Starting evaluation: %d tasks, %d total episodes",
        len(task_specs), total_episodes,
    )

    result: GemEvalResult = asyncio.run(
        evaluator.evaluate_all(task_specs=task_specs)
    )

    logger.info(
        "[GEM-EVAL] Evaluation complete: overall_win_rate=%.3f, "
        "total_episodes=%d, tasks=%d",
        result.overall_win_rate,
        result.total_episodes,
        result.total_tasks,
    )

    return result.to_metrics_dict()


def _run_tau2_branch(
    args: Namespace,
) -> Dict[str, float]:
    """Run the tau2 evaluation branch. Returns metrics dict (empty on skip).

    Swallows any exception raised by tau2 so a failing tau2 eval cannot
    crash the training loop. Errors are logged and the GEM branch's
    metrics (if any) still propagate to W&B.
    """
    if not args.spare_gem_eval_config:
        return {}

    try:
        defaults, task_specs = load_tau2_eval_config(args.spare_gem_eval_config)
    except Exception as exc:
        logger.error("[TAU2-EVAL] Failed to load tau2 eval config: %s", exc)
        return {}

    if not task_specs:
        # Graceful skip: no tau2_eval section or empty tasks list.
        return {}

    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error(
            "[TAU2-EVAL] OPENROUTER_API_KEY is not set in the training env. "
            "Skipping tau2 eval branch. Set it before launching training to "
            "run tau2 eval."
        )
        return {}

    try:
        sglang_url = _resolve_slime_router_url(args)
        logger.info(
            "[TAU2-EVAL] Starting evaluation: %d specs, slime router @ %s",
            len(task_specs), sglang_url,
        )
        evaluator = Tau2Evaluator(sglang_base_url=sglang_url)
        result: Tau2EvalResult = asyncio.run(evaluator.evaluate_all(task_specs))
    except Exception as exc:
        logger.exception("[TAU2-EVAL] tau2 eval branch crashed: %s", exc)
        return {}

    logger.info(
        "[TAU2-EVAL] Complete: overall_pass_at_1=%.3f, total_sims=%d, errors=%d",
        result.overall_pass_at_1, result.total_simulations, result.total_errors,
    )
    return result.to_metrics_dict()


def _resolve_model_context_length(args: Namespace) -> Optional[int]:
    """Return the model's hard context window (max_position_embeddings).

    Read straight from the HF checkpoint's config.json so it's exact per model
    (Qwen3-8B → 40960, Qwen3-30B-A3B-2507 → 262144). The BFCL evaluator uses
    this to MEASURE (not mutate) prompt length and skip generations that would
    overflow the window — a doomed over-window request trips SGLang's circuit
    breaker and takes the colocated training run down with it. Returns None if
    the config can't be read, which disables the guard (current behavior).
    """
    config_path = Path(args.hf_checkpoint) / "config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(
            f"[BFCL] Could not read {config_path} for context-length guard ({e}); "
            f"the over-window skip guard is disabled for this run."
        )
        return None
    # Some configs nest it under text_config (multimodal); check both.
    max_pos = config.get("max_position_embeddings")
    if max_pos is None:
        max_pos = config.get("text_config", {}).get("max_position_embeddings")
    if isinstance(max_pos, int) and max_pos > 0:
        return max_pos
    logger.warning(
        f"[BFCL] No usable max_position_embeddings in {config_path}; "
        f"over-window skip guard disabled."
    )
    return None


def _run_bfcl_branch(
    args: Namespace,
) -> Dict[str, float]:
    """Run the BFCL evaluation branch. Returns metrics dict (empty on skip)."""
    if not args.spare_gem_eval_config:
        return {}

    bfcl_config = load_bfcl_eval_config(args.spare_gem_eval_config)
    if not bfcl_config:
        logger.info("[BFCL] No bfcl_eval section in config, skipping")
        return {}

    dataset_dir = bfcl_config.get("dataset_dir", "")
    if not dataset_dir or not Path(dataset_dir).exists():
        logger.warning(f"[BFCL] dataset_dir not found: {dataset_dir}")
        return {}

    try:
        model_adapter = create_slime_model_adapter(args)
        # BFCL comparability requires the official prompt without a custom system message.
        evaluator = BfclEvaluator(
            model=model_adapter,
            max_concurrent=bfcl_config.get("max_concurrent", 64),
            # Defaults match the official Qwen3 BFCL sampling configuration.
            temperature=bfcl_config.get("temperature", 0.001),
            max_tokens=bfcl_config.get("max_tokens", 2048),
            system_prompt=None,
            # Skip over-window prompts using the YAML override or model config.
            max_context_length=(
                bfcl_config.get("max_context_length")
                or _resolve_model_context_length(args)
            ),
        )
        categories = bfcl_config.get("categories")

        result: BfclEvalResult = asyncio.run(
            evaluator.evaluate_all(dataset_dir=dataset_dir, categories=categories)
        )
        metrics = result.to_metrics_dict()

        # Log optional BFCL versions under separate metric prefixes.
        for extra in bfcl_config.get("additional_versions", []) or []:
            ver = extra.get("version")
            ver_dir = extra.get("dataset_dir", "")
            if not ver or not ver_dir or not Path(ver_dir).exists():
                logger.warning(
                    "[BFCL] additional_versions entry skipped (version=%s dir=%s)",
                    ver, ver_dir,
                )
                continue
            logger.info("[BFCL] Running additional version=%s from %s", ver, ver_dir)
            ver_result: BfclEvalResult = asyncio.run(
                evaluator.evaluate_all(
                    dataset_dir=ver_dir, categories=categories, version=ver,
                )
            )
            metrics.update(ver_result.to_metrics_dict(version_tag=ver))

        return metrics
    except Exception as exc:
        logger.exception("[BFCL] BFCL eval branch crashed: %s", exc)
        return {}


def _run_slime_builtin_eval(
    args: Namespace,
    rollout_id: int,
    data_source: Callable,
    metrics: Dict[str, float],
) -> Dict:
    """Run Slime's built-in eval if eval datasets are configured.

    Merges any returned metrics into the passed-in metrics dict in place
    and returns the data field from Slime's eval output.
    """
    eval_datasets = getattr(args, "eval_datasets", None)
    if not eval_datasets:
        return {}
    try:
        from slime.rollout.sglang_rollout import generate_rollout
        slime_result = generate_rollout(
            args, rollout_id, data_source, evaluation=True
        )
        if slime_result.metrics:
            metrics.update(slime_result.metrics)
        return slime_result.data or {}
    except Exception as exc:
        logger.error("[GEM-EVAL] Slime built-in eval failed: %s", exc)
        return {}


def spare_gem_eval_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Callable,
) -> RolloutFnEvalOutput:
    """Run the SPARE eval rollout (GEM + tau2 + Slime built-in).

    Args:
        args: Slime Namespace with SPARE configuration.
        rollout_id: Current rollout ID.
        data_source: Slime data source (used only by Slime built-in eval).

    Returns:
        RolloutFnEvalOutput with merged metrics from all three branches.
    """
    metrics: Dict[str, float] = {}

    bfcl_metrics = _run_bfcl_branch(args)
    metrics.update(bfcl_metrics)

    gem_metrics = _run_gem_branch(args)
    metrics.update(gem_metrics)

    tau2_metrics = _run_tau2_branch(args)
    metrics.update(tau2_metrics)

    slime_data = _run_slime_builtin_eval(args, rollout_id, data_source, metrics)

    if not metrics and not slime_data:
        logger.warning(
            "[GEM-EVAL] No eval branches produced metrics. Verify "
            "--spare-gem-eval-config points to a file with gem_eval or "
            "tau2_eval sections, or that eval_datasets is configured."
        )

    return RolloutFnEvalOutput(data=slime_data, metrics=metrics)
