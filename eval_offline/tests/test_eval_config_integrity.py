"""Integrity checks for the evaluation YAML configs and the results-table map.

Guarded here: ``table_mapping.RQ1_COLUMNS`` points at a metric key that the
evaluator never emits, which produces silently empty table cells rather than a
crash. The expected keys are derived from ``gem_evaluator``'s own source so the
two stay pinned together.

Note: there is no longer a check that every ``rg:`` task_id in a config is a
real registered env. That check needed the A.3 split, which now lives in the
private ``spade-rl/rg-golden-goose`` dataset and so cannot be read on a runner
without HF credentials.

Kept dependency-light on purpose: yaml + source parsing only, no ``gem`` import
and no network.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import yaml

from eval_offline.table_mapping import RQ1_COLUMNS
from spade.core.eval.gem_tasks import RG_GOLDEN_GOOSE, rg_golden_goose_category

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRS = (
    REPO_ROOT / "eval_offline" / "configs",
    REPO_ROOT / "eval_configs",
)
GEM_EVALUATOR_SRC = REPO_ROOT / "spade" / "core" / "eval" / "gem_evaluator.py"
GAMES_GEM_CONFIG = REPO_ROOT / "eval_offline" / "configs" / "games.yaml"

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
