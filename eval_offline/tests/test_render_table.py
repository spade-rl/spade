"""Tests for eval_offline.render_table and table_mapping."""
from __future__ import annotations

import json

from eval_offline import render_table, table_mapping


def test_rq1_columns_cover_expected_cells():
    """RQ1 contains only the eight games benchmarks in the paper release."""
    cols = list(table_mapping.RQ1_COLUMNS.keys())
    assert cols == [
        "AIME-25",
        "AIME-26",
        "GPQA-Diamond",
        "LiveCodeBench",
        "RG:Math",
        "RG:Algorithmic",
        "RG:Cognition",
        "RG:Logic",
    ]


def test_render_emits_markdown_row(tmp_path):
    """Given a results.json, produce a single-row markdown table."""
    results = {
        "metrics": {
            "aime2025_pass1": 0.474,
            "aime2026_pass1": 0.500,
            "gpqa_diamond_avg_at_n": 0.620,
            "livecodebench_pass_at_1": 0.351,
        }
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results))

    md = render_table.render_markdown_row(
        results_path=results_path,
        rq="RQ1",
        method_name="Qwen3-4B-Instruct-2507 baseline",
    )
    # Header row + separator row + content row
    assert md.count("\n") >= 2
    # Method name in content row
    assert "Qwen3-4B-Instruct-2507 baseline" in md
    # Values appear (formatted to 1 decimal, % scale)
    assert "47.4" in md  # aime25
    assert "62.0" in md  # gpqa


def test_render_handles_missing_metrics(tmp_path):
    """Missing metrics render as '—' (em-dash) without erroring."""
    results = {"metrics": {"aime2025_pass1": 0.5}}
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results))
    md = render_table.render_markdown_row(
        results_path=results_path,
        rq="RQ1",
        method_name="partial",
    )
    assert "—" in md
    assert "50.0" in md  # only metric that exists


def test_rq2_scores_render_as_percentages(tmp_path):
    results = {
        "metrics": {
            "bfcl_overall_acc": 0.404,
            "gem_eval/tau2_retail_base/pass_at_1": 0.250,
            "acebench/en/overall": 0.782,
        }
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results))

    md = render_table.render_markdown_row(
        results_path=results_path,
        rq="RQ2",
        method_name="SPADE",
    )
    assert "40.4" in md
    assert "25.0" in md
    assert "78.2" in md
