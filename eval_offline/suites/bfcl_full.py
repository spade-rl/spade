"""Full BFCL v4 eval suite — drives bfcl_eval's NATIVE generation + scoring
against our already-running SGLang, so we get the categories the single-turn
``bfcl`` suite can't: multi-turn + agentic (web_search).

Why this exists (vs ``bfcl.py``):
    ``bfcl.py`` does CUSTOM single-turn generation (one chat call with tools).
    Multi-turn + agentic categories need BFCL's own stateful interaction harness
    (``execute_multi_turn_func_call``, the web_search tool, etc.). Rather than
    reimplement those, we run the official ``bfcl generate`` / ``bfcl evaluate``
    CLI, pointing BFCL's OpenAI-compatible handler (qwen3-4b-FC -> QwenAPIHandler
    -> OpenAICompletionsHandler, which honours OPENAI_BASE_URL) at our SGLang.

Prereqs handled by the wrapper/sbatch:
    - SGLang served under the name BFCL sends (qwen3-4b-FC -> model="qwen3-4b"):
      set SGLANG_SERVED_MODEL_NAME=qwen3-4b (server.py passes --served-model-name).
    - SERPER_API_KEY in env for web_search (web_search.py patched SerpAPI->Serper).
    - bfcl_eval source on PYTHONPATH (run_offline_eval.sh).

Scope: non_live + live + multi_turn + web_search. ``memory`` is intentionally
excluded (memory_vector needs a vector/embedding backend we don't run).
"""
from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval_offline.suites.bfcl_full")

# OSS/local FC handle (QwenFCHandler -> OSSHandler): hits LOCAL_SERVER_ENDPOINT/PORT
# (/v1/completions) with model="Qwen/Qwen3-4B-Instruct-2507". NOT the API handle
# qwen3-4b-FC (QwenAPIHandler routes to DashScope cloud, ignores our endpoint).
_DEFAULT_MODEL_HANDLE = "Qwen/Qwen3-4B-Instruct-2507-FC"
# BFCL category group keywords (see constants/category_mapping.py TEST_COLLECTION_MAPPING).
_DEFAULT_CATEGORIES = ["non_live", "live", "multi_turn", "web_search"]


def _model_handle(cfg: dict) -> str:
    """BFCL model handle to generate and score under.

    Set ``BFCL_MODEL_HANDLE`` to evaluate a checkpoint whose size differs from
    the one pinned in the YAML (e.g. the 8B / 30B runs) — the environment
    variable wins over the config value, which in turn wins over the default.
    """
    return (
        os.environ.get("BFCL_MODEL_HANDLE", "").strip()
        or cfg.get("model_handle")
        or _DEFAULT_MODEL_HANDLE
    )


def _setup_bfcl_project_root(out_dir: Path) -> Path:
    """Point bfcl_eval at a writable project root (its default lives in
    site-packages and isn't writable). Must be absolute — BFCL runs with
    cwd=project_root and computes more paths relative to it."""
    project_root = (out_dir / "bfcl_workdir").resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    return project_root


def _pct(v: Any) -> float | None:
    if v is None or v in ("N/A", ""):
        return None
    try:
        return float(str(v).rstrip("%")) / 100.0
    except ValueError:
        return None


def _read_row(score_dir: Path, fname: str) -> dict:
    fp = score_dir / fname
    if not fp.is_file():
        logger.warning("[bfcl_full] missing score csv: %s", fp)
        return {}
    with fp.open() as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def _parse_scores(proj: Path) -> dict[str, float]:
    """Parse the v4 aggregate CSVs into a flat metrics dict (fractions 0-1)."""
    score_dir = proj / "score"
    metrics: dict[str, Any] = {}
    overall = _read_row(score_dir, "data_overall.csv")
    non_live = _read_row(score_dir, "data_non_live.csv")
    live = _read_row(score_dir, "data_live.csv")
    multi_turn = _read_row(score_dir, "data_multi_turn.csv")

    # Group-level accuracies straight off the overall row.
    for key, col in (
        ("overall", "Overall Acc"),
        ("non_live", "Non-Live AST Acc"),
        ("live", "Live Acc"),
        ("multi_turn", "Multi Turn Acc"),
        ("web_search", "Web Search Acc"),
    ):
        v = _pct(overall.get(col))
        if v is not None:
            metrics[f"bfcl_{key}_acc"] = v

    # Section overalls as a fallback / cross-check.
    for key, row, col in (
        ("non_live_overall", non_live, "Non-Live Overall Acc"),
        ("live_overall", live, "Live Overall Acc"),
        ("multi_turn_overall", multi_turn, "Multi Turn Overall Acc"),
    ):
        v = _pct(row.get(col))
        if v is not None:
            metrics.setdefault(f"bfcl_{key}_acc", v)

    # Per-multi-turn-category (the informative slices).
    for key, col in (
        ("multi_turn_base", "Base"),
        ("multi_turn_miss_func", "Miss Func"),
        ("multi_turn_miss_param", "Miss Param"),
        ("multi_turn_long_context", "Long Context"),
    ):
        v = _pct(multi_turn.get(col))
        if v is not None:
            metrics[f"bfcl_{key}_acc"] = v

    if not metrics:
        logger.warning("[bfcl_full] no scores parsed from %s", score_dir)
    return metrics


def _run(cmd: list[str], proj: Path, log_path: Path, env: dict) -> int:
    logger.info("[bfcl_full] $ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=proj, env=env, capture_output=True, text=True)
    log_path.write_text((proc.stdout or "") + "\n===STDERR===\n" + (proc.stderr or ""))
    return proc.returncode


def run(client, cfg: dict, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proj = _setup_bfcl_project_root(out_dir)

    model_handle = _model_handle(cfg)
    categories = cfg.get("categories") or _DEFAULT_CATEGORIES
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]
    cat_arg = ",".join(categories)
    num_threads = int(cfg.get("num_threads", 16))
    temperature = float(cfg.get("temperature", 0.0))

    # Point BFCL's OpenAI-compatible handler at our SGLang. OPENAI_BASE_URL is
    # read by OpenAICompletionsHandler; LOCAL_SERVER_* covers the OSS path too.
    env = os.environ.copy()
    sglang_v1 = f"{client.base_url}/v1"
    env["OPENAI_BASE_URL"] = sglang_v1
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY") or "sglang-local"
    host = client.base_url.split("://", 1)[-1].rsplit(":", 1)[0]
    port = client.base_url.rsplit(":", 1)[-1]
    env["LOCAL_SERVER_ENDPOINT"] = host
    env["LOCAL_SERVER_PORT"] = port

    if "web_search" in categories and not env.get("SERPER_API_KEY"):
        logger.warning("[bfcl_full] SERPER_API_KEY not set — web_search will error")

    logger.info("[bfcl_full] model=%s base_url=%s categories=%s",
                model_handle, sglang_v1, cat_arg)

    # 1) Native generation (multi-turn + agentic harness) against our SGLang.
    gen_cmd = [
        sys.executable, "-m", "bfcl_eval", "generate",
        "--model", model_handle,
        "--test-category", cat_arg,
        "--num-threads", str(num_threads),
        "--temperature", str(temperature),
        "--skip-server-setup",
        "--allow-overwrite",
    ]
    rc = _run(gen_cmd, proj, out_dir / "bfcl_generate.log", env)
    if rc != 0:
        raise RuntimeError(
            f"bfcl generate exited {rc}; see {out_dir/'bfcl_generate.log'}"
        )

    # 2) Score.
    eval_cmd = [
        sys.executable, "-m", "bfcl_eval", "evaluate",
        "--model", model_handle,
        "--test-category", cat_arg,
    ]
    rc = _run(eval_cmd, proj, out_dir / "bfcl_evaluate.log", env)
    if rc != 0:
        raise RuntimeError(
            f"bfcl evaluate exited {rc}; see {out_dir/'bfcl_evaluate.log'}"
        )

    metrics = _parse_scores(proj)
    metrics["bfcl_model_handle"] = model_handle
    metrics["bfcl_categories"] = cat_arg
    return metrics
