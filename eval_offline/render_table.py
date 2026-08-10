"""Render an offline-eval `results.json` into one markdown / csv / latex row.

Usage:
    python -m eval_offline.render_table \\
        --results /scratch/offline_eval/<ckpt>/<ts>/results.json \\
        --rq RQ1 \\
        --method-name "Spare" \\
        --format markdown
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval_offline.table_mapping import COLUMN_MAPS


def _load_metrics(results_path: Path) -> dict[str, Any]:
    """Read `results.json` produced by run_offline_eval and return its 'metrics' dict."""
    data = json.loads(results_path.read_text())
    return data.get("metrics", data)  # tolerate flat or nested


def _format_cell(value: Any) -> str:
    """Format a metric value to a 1-decimal percentage string. None -> em-dash."""
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.1f}"
    except (TypeError, ValueError):
        return "—"


def render_markdown_row(*, results_path: Path, rq: str, method_name: str) -> str:
    """Return a 3-line markdown table: header | separator | one content row."""
    if rq not in COLUMN_MAPS:
        raise ValueError(f"Unknown RQ {rq!r}; known: {list(COLUMN_MAPS)}")
    columns = COLUMN_MAPS[rq]
    metrics = _load_metrics(results_path)

    header_cells = ["Method"] + list(columns.keys())
    sep_cells = ["---"] * len(header_cells)
    content_cells = [method_name] + [
        _format_cell(metrics.get(key)) for key in columns.values()
    ]
    return (
        "| " + " | ".join(header_cells) + " |\n"
        + "|" + "|".join(sep_cells) + "|\n"
        + "| " + " | ".join(content_cells) + " |\n"
    )


def render_csv_row(*, results_path: Path, rq: str, method_name: str) -> str:
    columns = COLUMN_MAPS[rq]
    metrics = _load_metrics(results_path)
    header = ["Method"] + list(columns.keys())
    content = [method_name] + [_format_cell(metrics.get(k)) for k in columns.values()]
    return ",".join(header) + "\n" + ",".join(content) + "\n"


def render_latex_row(*, results_path: Path, rq: str, method_name: str) -> str:
    columns = COLUMN_MAPS[rq]
    metrics = _load_metrics(results_path)
    cells = [method_name] + [_format_cell(metrics.get(k)) for k in columns.values()]
    return " & ".join(cells) + " \\\\\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--rq", required=True, choices=list(COLUMN_MAPS.keys()))
    p.add_argument("--method-name", required=True)
    p.add_argument("--format", default="markdown",
                   choices=["markdown", "csv", "latex"])
    args = p.parse_args()

    renderers = {
        "markdown": render_markdown_row,
        "csv": render_csv_row,
        "latex": render_latex_row,
    }
    sys.stdout.write(renderers[args.format](
        results_path=args.results, rq=args.rq, method_name=args.method_name,
    ))


if __name__ == "__main__":
    main()
