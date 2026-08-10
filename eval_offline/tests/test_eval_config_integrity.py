"""Integrity checks for the evaluation YAML configs and the results-table map.

Two failure modes are guarded here, both of which produce silently empty table
cells rather than a crash:

1. A config lists a reasoning_gym task_id that is not a registered environment
   (e.g. the Golden Goose rollup names ``rg:math-hard`` / ``rg:logic-hard``,
   which are bucket labels, not envs). Only the 100 ``rg:<name>-hard`` ids
   backed by ``A3_HARD_CONFIGS`` can actually be run.
2. ``table_mapping.RQ1_COLUMNS`` points at a metric key that the evaluator never
   emits. The expected keys are derived from ``gem_evaluator``'s own source so
   the two stay pinned together.

Kept dependency-light on purpose: yaml + source parsing only, no ``gem`` import
and no network.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from eval_offline.table_mapping import RQ1_COLUMNS
from spare.core.eval.gem_tasks import RG_GOLDEN_GOOSE, rg_golden_goose_category
from spare.core.eval.rg_a3_hard import A3_HARD_CONFIGS

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRS = (
    REPO_ROOT / "eval_offline" / "configs",
    REPO_ROOT / "eval_configs",
)
GEM_EVALUATOR_SRC = REPO_ROOT / "spare" / "core" / "eval" / "gem_evaluator.py"
GAMES_GEM_CONFIG = REPO_ROOT / "eval_offline" / "configs" / "_games_gem.yaml"

# Paper column -> Golden Goose bucket for the four RG cells of the RQ1 table.
RQ1_RG_BUCKETS = {
    "RG:Math": "math",
    "RG:Algorithmic": "algorithmic",
    "RG:Cognition": "cognition",
    "RG:Logic": "logic",
}


def _config_files() -> list[Path]:
    """Every YAML config shipped under the two eval config directories."""
    found: list[Path] = []
    for directory in CONFIG_DIRS:
        found.extend(sorted(directory.glob("*.yaml")))
        found.extend(sorted(directory.glob("*.yml")))
    return found


def _iter_task_ids(node: Any) -> Iterator[str]:
    """Yield every task id in a parsed config, dict-style or bare-string style."""
    if isinstance(node, dict):
        task_id = node.get("task_id")
        if isinstance(task_id, str):
            yield task_id
        for key, value in node.items():
            if key == "tasks" and isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        yield item
            yield from _iter_task_ids(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_task_ids(item)


def _rg_task_ids(config_path: Path) -> list[str]:
    raw = yaml.safe_load(config_path.read_text())
    return [tid for tid in _iter_task_ids(raw) if tid.startswith("rg:")]


def _extract_f_string(pattern: str, source: str, what: str) -> str:
    """Pull a single f-string literal out of ``source`` via ``pattern``."""
    match = re.search(pattern, source)
    assert match is not None, (
        f"could not locate {what} in {GEM_EVALUATOR_SRC}; the metric-key format "
        "changed and this test needs updating alongside table_mapping"
    )
    return match.group(1)


def _emitted_category_metric_key(bucket: str, metric_name: str = "win_rate") -> str:
    """Rebuild the metric key gem_evaluator emits for a Golden Goose rollup.

    Derived from gem_evaluator's own literals (the default ``prefix``, the
    ``rg_<bucket>`` category key, and the per-category metric template) so this
    test tracks the emitter instead of restating it.
    """
    source = GEM_EVALUATOR_SRC.read_text()
    prefix = _extract_f_string(
        r'def to_metrics_dict\(self, prefix: str = "([^"]+)"\)',
        source,
        "the default metrics prefix",
    )
    category_template = _extract_f_string(
        r'gkey = f"([^"]+)"', source, "the Golden Goose category key template"
    )
    metric_template = _extract_f_string(
        r'metrics\[f"([^"]*category[^"]*)"\]',
        source,
        "the per-category metric key template",
    )
    assert "{gg}" in category_template, category_template
    for placeholder in ("{prefix}", "{category}", "{metric_name}"):
        assert placeholder in metric_template, metric_template
    return metric_template.format(
        prefix=prefix,
        category=category_template.format(gg=bucket),
        metric_name=metric_name,
    )


def test_config_directories_are_not_empty():
    """Guards the parametrised tests below against silently collecting nothing."""
    assert _config_files(), f"no YAML configs found under {CONFIG_DIRS}"


@pytest.mark.parametrize("config_path", _config_files(), ids=lambda p: p.name)
def test_rg_task_ids_are_registered_hard_split_envs(config_path: Path):
    """Every rg: task in a config must be a real registered hard-split env."""
    for task_id in _rg_task_ids(config_path):
        name = task_id.split(":", 1)[1].rsplit("-", 1)[0]
        assert name in A3_HARD_CONFIGS, (
            f"{config_path.name}: '{task_id}' is not a registered reasoning_gym "
            f"hard-split env ('{name}' missing from A3_HARD_CONFIGS). Golden "
            "Goose bucket names are rollups, not task ids."
        )
        assert rg_golden_goose_category(task_id) is not None, (
            f"{config_path.name}: '{task_id}' has no Golden Goose category, so "
            "it would never reach an RG column of the results table."
        )


def test_games_gem_config_covers_all_golden_goose_buckets():
    """The RQ1 games config must feed all four RG columns."""
    task_ids = _rg_task_ids(GAMES_GEM_CONFIG)
    buckets = {rg_golden_goose_category(tid) for tid in task_ids}
    assert buckets == set(RG_GOLDEN_GOOSE.values())
    # The paper config runs every Golden Goose env, not a subset: dropping
    # tasks would silently change the RQ1 rollup numbers.
    assert len(set(task_ids)) == len(RG_GOLDEN_GOOSE), (
        f"expected one task per RG_GOLDEN_GOOSE env ({len(RG_GOLDEN_GOOSE)}), "
        f"got {len(set(task_ids))}"
    )


def test_rq1_rg_columns_match_emitted_metric_keys():
    """RQ1's RG cells must read the rollup keys gem_evaluator actually emits."""
    assert set(RQ1_RG_BUCKETS.values()) == set(RG_GOLDEN_GOOSE.values())
    for column, bucket in RQ1_RG_BUCKETS.items():
        expected = _emitted_category_metric_key(bucket)
        assert RQ1_COLUMNS[column] == expected, (
            f"RQ1 column {column!r} maps to {RQ1_COLUMNS[column]!r} but "
            f"gem_evaluator emits {expected!r}"
        )


def test_rq1_rg_columns_are_rollups_not_per_task_metrics():
    """Regression: the RG columns are per-category rollups, never per-task keys."""
    for column in RQ1_RG_BUCKETS:
        assert "task_rg_" not in RQ1_COLUMNS[column], (
            f"RQ1 column {column!r} points at a per-task metric; the four RG "
            "columns aggregate 100 tasks and must use the category rollup."
        )
