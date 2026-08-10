"""Tests for eval_offline.suites.livecodebench (wraps official LCB package)."""
from __future__ import annotations

import asyncio
import json

from eval_offline.suites import livecodebench
from eval_offline.suites.tests.conftest import MockOfflineClient


def _fake_problems():
    """Two LCB-shaped problems."""
    return [
        {"question_id": "p1", "question_content": "Write add.", "starter_code": ""},
        {"question_id": "p2", "question_content": "Write square.", "starter_code": ""},
    ]


def test_livecodebench_rollout_then_grade(tmp_path, monkeypatch):
    """End-to-end: load problems → roll out N samples → grade via mocked LCB evaluator."""
    monkeypatch.setattr(livecodebench, "_load_problems", _fake_problems)

    # Mock the LCB evaluator: it returns a metrics dict like the real one.
    captured: dict = {}

    def fake_invoke(*, generations_path, scenario, release_version, work_dir):
        captured["generations_path"] = generations_path
        captured["scenario"] = scenario
        captured["release_version"] = release_version
        # LCB's evaluator output shape: {"pass@1": float, ...}
        return {"pass@1": 0.5, "easy_pass@1": 1.0, "medium_pass@1": 0.0, "hard_pass@1": 0.0}

    monkeypatch.setattr(livecodebench, "_invoke_lcb_evaluator", fake_invoke)

    client = MockOfflineClient(lambda m, n: ["```python\ndef solve(): return 1\n```"] * n)
    cfg = {
        "n_samples_per_prompt": 1,
        "max_tokens": 1024,
        "temperature": 0.2,
        "max_concurrent": 2,
        "release_version": "release_v6",
    }
    out = tmp_path / "lcb"
    metrics = asyncio.run(livecodebench._run_async(client, cfg, out))

    # The suite forwards the LCB-reported pass@1 verbatim.
    assert metrics["livecodebench_pass_at_1"] == 0.5
    assert metrics["livecodebench_n_problems"] == 2
    # Suite persists the generations payload sent to the evaluator
    assert (out / "lcb_generations.json").exists()
    payload = json.loads((out / "lcb_generations.json").read_text())
    # Each entry must have code_list + question_id (LCB's required schema)
    assert all("code_list" in entry and "question_id" in entry for entry in payload)
    # Captured invocation args
    assert captured["scenario"] == "codegeneration"
    assert captured["release_version"] == "release_v6"


def test_livecodebench_code_extraction(tmp_path, monkeypatch):
    """Code is extracted from the LAST ```python ... ``` fence in the response."""
    monkeypatch.setattr(livecodebench, "_load_problems",
                        lambda: [{"question_id": "p1", "question_content": "Q", "starter_code": ""}])
    monkeypatch.setattr(livecodebench, "_invoke_lcb_evaluator",
                        lambda **kw: {"pass@1": 0.0})

    response = (
        "Here's a draft:\n```python\ndef wrong(): pass\n```\n"
        "Let me fix it:\n```python\ndef final(): return 42\n```"
    )
    client = MockOfflineClient(lambda m, n: [response] * n)
    cfg = {"n_samples_per_prompt": 1, "max_tokens": 1024, "temperature": 0.2,
           "max_concurrent": 1}
    out = tmp_path / "lcb"
    asyncio.run(livecodebench._run_async(client, cfg, out))
    payload = json.loads((out / "lcb_generations.json").read_text())
    # Should pick the LAST fence ("def final(): return 42")
    assert "def final" in payload[0]["code_list"][0]
    assert "def wrong" not in payload[0]["code_list"][0]
