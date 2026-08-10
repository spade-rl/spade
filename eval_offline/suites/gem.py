"""GEM suite adapter for the retained Reasoning-Gym evaluations.

Reads the same YAML schema as eval_configs/gem_eval_*.yaml, builds an
OfflineModelAdapter, calls GemEvaluator.evaluate_all, writes scores.json.

Config:
    suites:
      gem:
        config_path: eval_offline/configs/_games_gem.yaml

Output:
    <out>/scores.json   — flat metric dict (gem_eval/... keys)
    <out>/raw_result.json — full GemEvalResult (per-task results, errors)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval_offline.suites.gem")


def _resolve_config_path(p: str) -> Path:
    if not p:
        raise ValueError("gem suite needs `config_path`.")
    pp = Path(p)
    if pp.is_absolute() and pp.is_file():
        return pp
    for root in ("/workspace", os.getcwd()):
        candidate = Path(root) / p
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"gem config_path {p!r} not found")


def run(client, cfg: dict, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    from spare.core.eval.gem_evaluator import GemEvaluator
    from spare.core.eval.gem_tasks import load_gem_eval_config
    from eval_offline.model_adapter_shim import OfflineModelAdapter

    config_path = _resolve_config_path(cfg.get("config_path", ""))
    logger.info("[gem] loading config from %s", config_path)
    defaults, task_specs = load_gem_eval_config(str(config_path))

    if not task_specs:
        logger.warning("[gem] no tasks in config — nothing to run")
        return {"gem_n_tasks": 0}

    adapter = OfflineModelAdapter(client, model_path=client.model)

    max_concurrent = cfg.get("max_concurrent", defaults.max_concurrent)
    evaluator = GemEvaluator(model=adapter, max_concurrent=max_concurrent)

    logger.info(
        "[gem] running %d tasks (defaults episodes=%d max_turns=%d, "
        "max_concurrent=%d)",
        len(task_specs), defaults.episodes, defaults.max_turns, max_concurrent,
    )
    eval_result = asyncio.run(evaluator.evaluate_all(task_specs))

    metrics: dict[str, Any] = dict(eval_result.to_metrics_dict(prefix="gem_eval"))
    (out_dir / "scores.json").write_text(json.dumps(metrics, indent=2))

    try:
        (out_dir / "raw_result.json").write_text(
            json.dumps(asdict(eval_result), indent=2, default=str)
        )
    except Exception as e:  # pragma: no cover
        logger.warning("[gem] could not serialize raw result: %s", e)

    return metrics
