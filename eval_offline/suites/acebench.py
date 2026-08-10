"""ACEBench offline-eval suite.

Wraps the official ACEBench harness (https://github.com/ACEBench/ACEBench)
to evaluate a served OpenAI-compatible endpoint on ACEBench-en test_all,
and reports Normal / Special / Agent / Overall scores aligned with the
ACEBench paper (arXiv 2509.13311 / AgentScaler).

Prerequisites
-------------
Set the ``ACEBENCH_DIR`` environment variable to the root of a local clone
of ACEBench (``git clone https://github.com/ACEBench/ACEBench``).  The suite
skips silently when the variable is absent.

Also set ``GPT_BASE_URL`` / ``GPT_AGENT_API_KEY`` (or ``OPENAI_API_KEY``) so
that the ACEBench user-simulator (default: gpt-4o) can call the OpenAI API.

Config example
--------------
::

    suites:
      acebench:
        concurrent: true
        language: en
        category: test_all
        temperature: 0.7
        top_p: 1.0
        max_tokens: 1200
        num_threads: 16
        max_dialog_turns: 40
        user_model: gpt-4o

Output keys
-----------
``acebench/en/normal``, ``acebench/en/special``, ``acebench/en/agent``,
``acebench/en/overall`` — all fractions in the range 0–1.

Routing patch
-------------
ACEBench's ``APIModelInference.__init__`` has no ``else`` branch for unknown
model names, causing ``UnboundLocalError`` for non-GPT/DeepSeek/o1 models.
The suite applies an idempotent in-place patch the first time it runs against
a given ACEBench clone. The patch adds an ``else`` block that reads
``ACEBENCH_AGENT_BASE_URL`` / ``ACEBENCH_AGENT_API_KEY``, and wraps
``inference_map`` in a ``__missing__``-based dict that falls back to
``APIModelInference`` for unknown names.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval_offline.suites._acebench_patches import (
    _apply_routing_patch,
    _patch_disable_thinking,
)

logger = logging.getLogger("eval_offline.suites.acebench")

# ---------------------------------------------------------------------------
# Category → group mapping (from ACEBench category.py test_all)
# ---------------------------------------------------------------------------

_NORMAL_LEAVES: list[str] = [
    "normal_single_turn_single_function",
    "normal_single_turn_parallel_function",
    "normal_multi_turn_user_adjust",
    "normal_multi_turn_user_switch",
    "normal_similar_api",
    "normal_preference",
    "normal_atom_bool",
    "normal_atom_enum",
    "normal_atom_number",
    "normal_atom_list",
    "normal_atom_object_deep",
    "normal_atom_object_short",
]

_SPECIAL_LEAVES: list[str] = [
    "special_incomplete",
    "special_error_param",
    "special_irrelevant",
]

_AGENT_LEAVES: list[str] = [
    "agent_multi_step",
    "agent_multi_turn",
]

# Mapping from category name → group name
_CATEGORY_TO_GROUP: dict[str, str] = {}
for _cat in _NORMAL_LEAVES:
    _CATEGORY_TO_GROUP[_cat] = "normal"
for _cat in _SPECIAL_LEAVES:
    _CATEGORY_TO_GROUP[_cat] = "special"
for _cat in _AGENT_LEAVES:
    _CATEGORY_TO_GROUP[_cat] = "agent"

# Summary weights (from eval_main.py):
#   summary = special_acc * 0.2676 + normal_acc * 0.578 + agent_acc * 0.1545
_W_NORMAL = 0.578
_W_SPECIAL = 0.2676
_W_AGENT = 0.1545


# ---------------------------------------------------------------------------
# Public pure functions (unit-tested)
# ---------------------------------------------------------------------------


def _overall(normal: float, special: float, agent: float) -> float:
    """Weighted ACEBench Summary score.

    Inputs are percentages (0–100). Returns percentage.
    Weights from eval_main.py: normal*0.578 + special*0.2676 + agent*0.1545.
    Verified against the paper's Gemini row (76.7/90.0/63.4 → 78.2).
    """
    return normal * _W_NORMAL + special * _W_SPECIAL + agent * _W_AGENT


def _acebench_python() -> str:
    """Python interpreter used to run ACEBench's generate.py / eval_main.py.

    Set ``ACEBENCH_PYTHON`` to a dedicated environment when ACEBench's pinned
    dependencies conflict with the serving environment. The current interpreter
    is used when the variable is unset.
    """
    return os.environ.get("ACEBENCH_PYTHON") or sys.executable


def _build_generate_cmd(cfg: dict, model: str) -> list[str]:
    """Build the generate.py subprocess command list.

    Uses confirmed hyphenated flag names from generate.py argparse spec.
    """
    lang = cfg.get("language", "en")
    category = cfg.get("category", "test_all")
    temperature = cfg.get("temperature", 0.7)
    top_p = cfg.get("top_p", 1.0)
    max_tokens = cfg.get("max_tokens", 1200)
    num_threads = cfg.get("num_threads", 16)
    max_dialog_turns = cfg.get("max_dialog_turns", 40)
    user_model = cfg.get("user_model", "gpt-4o")

    return [
        _acebench_python(), "generate.py",
        "--model", model,
        "--category", category,
        "--language", lang,
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--max-tokens", str(max_tokens),
        "--num-threads", str(num_threads),
        "--max-dialog-turns", str(max_dialog_turns),
        "--user-model", user_model,
    ]


def _build_eval_cmd(cfg: dict, model: str) -> list[str]:
    """Build the eval_main.py subprocess command list.

    eval_main.py only accepts --language, --model, --category.
    """
    lang = cfg.get("language", "en")
    category = cfg.get("category", "test_all")

    return [
        _acebench_python(), "eval_main.py",
        "--model", model,
        "--category", category,
        "--language", lang,
    ]


def _parse_scores(score_dir: Path) -> dict[str, float]:
    """Parse per-category score JSONs and return group scores as percentages.

    ``score_dir`` is ``<acebench_root>/score_all/score_<lang>/<model>``.

    Returns ``{"normal": float, "special": float, "agent": float}`` where
    each value is in percentage points (0–100).

    Agent categories use ``end_to_end_accuracy``; all others use ``accuracy``.
    Values already in [0,1] are multiplied by 100.
    """
    group_values: dict[str, list[float]] = {"normal": [], "special": [], "agent": []}

    for json_file in sorted(score_dir.glob("data_*_score.json")):
        # Extract category from filename: data_<category>_score.json
        name = json_file.stem  # e.g. data_normal_atom_bool_score
        # Strip leading "data_" and trailing "_score"
        if not name.startswith("data_") or not name.endswith("_score"):
            continue
        category = name[len("data_"):-len("_score")]

        group = _CATEGORY_TO_GROUP.get(category)
        if group is None:
            logger.warning("[acebench] unknown category %r in %s — skipping", category, json_file)
            continue

        try:
            first_line = json_file.read_text().splitlines()[0].strip()
            row = json.loads(first_line)
        except Exception as exc:
            logger.warning("[acebench] could not parse %s: %s", json_file, exc)
            continue

        if group == "agent":
            acc = row.get("end_to_end_accuracy")
        else:
            acc = row.get("accuracy")

        if acc is None:
            logger.warning("[acebench] no accuracy key in %s", json_file)
            continue

        # Normalise to percentage
        if acc <= 1.0:
            acc *= 100.0

        group_values[group].append(acc)

    result: dict[str, float] = {}
    for grp, vals in group_values.items():
        if not vals:
            logger.warning("[acebench] no scores found for group %r", grp)
            result[grp] = 0.0
        else:
            result[grp] = sum(vals) / len(vals)

    return result


def _acebench_dir() -> Path | None:
    """Return the ACEBench root directory from ``ACEBENCH_DIR``, or None."""
    val = os.environ.get("ACEBENCH_DIR", "").strip()
    if not val:
        return None
    p = Path(val)
    if not p.is_dir():
        logger.warning("[acebench] ACEBENCH_DIR=%r is not a directory", val)
        return None
    return p


# ---------------------------------------------------------------------------
# run() — the suite entry point
# ---------------------------------------------------------------------------


def run(client: Any, cfg: dict, out_dir: Path) -> dict[str, Any]:
    """Run ACEBench-en test_all against the served model.

    Parameters
    ----------
    client:
        ``OfflineClient`` — provides ``.base_url`` (no trailing /v1) and
        ``.model``.
    cfg:
        Suite config dict from the eval YAML.
    out_dir:
        Directory for raw outputs + scores JSON.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    acebench = _acebench_dir()
    if acebench is None:
        logger.warning(
            "[acebench] ACEBENCH_DIR not set or invalid — skipping. "
            "Set ACEBENCH_DIR to the root of a local ACEBench clone."
        )
        return {"skipped": True}

    # Apply routing patch (idempotent)
    _apply_routing_patch(acebench)

    # Thinking mode relies on the server's reasoning parser to keep answers strict.
    if not cfg.get("enable_thinking", True):
        _patch_disable_thinking(acebench)
        logger.info("[acebench] enable_thinking=false — disabling Qwen3 thinking")
    else:
        logger.info(
            "[acebench] enable_thinking=true — model reasons; ensure the server "
            "runs with --reasoning-parser qwen3 and a large max_tokens"
        )

    model = client.model
    lang = cfg.get("language", "en")

    # Build env for subprocesses
    env = dict(os.environ)
    # Preserve the exact served model ID across ACEBench's role-specific clients.
    agent_base_url = f"{client.base_url}/v1"
    env["ACEBENCH_AGENT_BASE_URL"] = agent_base_url
    env["ACEBENCH_AGENT_API_KEY"] = "EMPTY"
    env["ACEBENCH_SERVED_MODEL"] = model
    # The Agent user-simulator uses OpenAI while the evaluated model stays local.
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        env.setdefault("GPT_API_KEY", openai_key)
        env.setdefault("GPT_AGENT_API_KEY", openai_key)
    else:
        logger.warning(
            "[acebench] OPENAI_API_KEY not set — the Agent category's gpt-4o "
            "user-simulator cannot run; Agent scores will be unreliable."
        )

    # --- generate step ---
    gen_cmd = _build_generate_cmd(cfg, model)
    logger.info("[acebench] running generate: %s", " ".join(gen_cmd))
    gen_result = subprocess.run(
        gen_cmd,
        cwd=str(acebench),
        env=env,
        capture_output=True,
        text=True,
    )
    if gen_result.returncode != 0:
        logger.error(
            "[acebench] generate failed (rc=%d):\n%s",
            gen_result.returncode,
            gen_result.stderr[-2000:],
        )
        return {"skipped": True, "error": f"generate failed rc={gen_result.returncode}"}

    # --- eval step ---
    # Agent-only evaluation expects its score directory to exist.
    (acebench / "score_all" / f"score_{lang}" / model).mkdir(parents=True, exist_ok=True)
    eval_cmd = _build_eval_cmd(cfg, model)
    logger.info("[acebench] running eval: %s", " ".join(eval_cmd))
    eval_result = subprocess.run(
        eval_cmd,
        cwd=str(acebench),
        env=env,
        capture_output=True,
        text=True,
    )
    if eval_result.returncode != 0:
        logger.error(
            "[acebench] eval failed (rc=%d):\n%s",
            eval_result.returncode,
            eval_result.stderr[-2000:],
        )
        return {"skipped": True, "error": f"eval failed rc={eval_result.returncode}"}

    # --- parse scores ---
    score_dir = acebench / "score_all" / f"score_{lang}" / model
    if not score_dir.is_dir():
        logger.error("[acebench] score dir not found: %s", score_dir)
        return {"skipped": True, "error": f"score dir not found: {score_dir}"}

    groups = _parse_scores(score_dir)
    normal = groups.get("normal", 0.0)
    special = groups.get("special", 0.0)
    agent = groups.get("agent", 0.0)
    overall = _overall(normal, special, agent)

    metrics: dict[str, Any] = {
        f"acebench/{lang}/normal": round(normal / 100.0, 5),
        f"acebench/{lang}/special": round(special / 100.0, 5),
        f"acebench/{lang}/agent": round(agent / 100.0, 5),
        f"acebench/{lang}/overall": round(overall / 100.0, 5),
    }

    # --- copy raw outputs ---
    raw_out = out_dir / "acebench_raw"
    raw_out.mkdir(parents=True, exist_ok=True)
    # Copy score JSONs
    _copy_tree_if_exists(score_dir, raw_out / "scores")
    # Copy result files
    result_dir = acebench / "result_all" / f"result_{lang}" / model
    _copy_tree_if_exists(result_dir, raw_out / "results")

    # Persist metrics
    (out_dir / "scores.json").write_text(json.dumps(metrics, indent=2))
    logger.info("[acebench] scores: %s", metrics)
    return metrics


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    """Copy a directory tree if the source exists."""
    if src.is_dir():
        try:
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
        except Exception as exc:
            logger.warning("[acebench] could not copy %s → %s: %s", src, dst, exc)
