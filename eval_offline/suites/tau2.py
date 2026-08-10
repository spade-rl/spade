"""tau2-bench suite — shim around spare.core.eval.tau2_evaluator.

The tau2 evaluator hits SGLang via OpenAI-compatible HTTP for the agent
side, and OpenRouter (or OpenAI directly) for the user simulator.

Config:
    suites:
      tau2:
        config_path: eval_offline/configs/_tool_use_tau2.yaml
        concurrent: true   # run alongside other suites — tau2 is API-bound

Skip-with-warning policy: if neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set,
this suite logs a warning and returns {"skipped": True} so the rest of the eval
still runs.

Output:
    <out>/scores.json    — flat metric dict (gem_eval/tau2_*/pass_at_1, etc.)
    <out>/raw_result.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval_offline.suites.tau2")


def _resolve_config_path(p: str) -> Path:
    if not p:
        raise ValueError("tau2 suite needs `config_path`.")
    pp = Path(p)
    if pp.is_absolute() and pp.is_file():
        return pp
    for root in ("/workspace", os.getcwd()):
        candidate = Path(root) / p
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"tau2 config_path {p!r} not found")


def run(client, cfg: dict, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        logger.warning(
            "[tau2] skipped — neither OPENROUTER_API_KEY nor OPENAI_API_KEY in env"
        )
        return {"skipped": True}

    config_path = _resolve_config_path(cfg.get("config_path", ""))
    logger.info("[tau2] loading config from %s", config_path)

    from spare.core.eval.tau2_evaluator import Tau2Evaluator
    from spare.core.eval.tau2_tasks import load_tau2_eval_config

    defaults, specs = load_tau2_eval_config(str(config_path))
    if not specs:
        logger.warning("[tau2] config has no tau2_eval section / no specs — skipped")
        return {"skipped": True}

    sglang_base_url = f"{client.base_url}/v1"
    evaluator = Tau2Evaluator(sglang_base_url=sglang_base_url)

    logger.info(
        "[tau2] running %d specs against %s (user sim via OpenRouter/OpenAI)",
        len(specs), sglang_base_url,
    )
    eval_result = asyncio.run(evaluator.evaluate_all(specs))

    metrics: dict[str, Any] = dict(eval_result.to_metrics_dict(prefix="gem_eval"))
    (out_dir / "scores.json").write_text(json.dumps(metrics, indent=2))
    try:
        (out_dir / "raw_result.json").write_text(
            json.dumps(asdict(eval_result), indent=2, default=str)
        )
    except Exception as e:  # pragma: no cover
        logger.warning("[tau2] could not serialize raw result: %s", e)

    return metrics
