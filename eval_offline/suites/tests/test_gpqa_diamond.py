"""Tests for eval_offline.suites.gpqa_diamond."""
from __future__ import annotations

import asyncio

from eval_offline.suites import gpqa_diamond
from eval_offline.suites.tests.conftest import MockOfflineClient


def test_gpqa_diamond_avg_at_8(tmp_path, monkeypatch):
    """avg@8 = mean of correct across all (problem, sample) pairs."""
    monkeypatch.setattr(
        gpqa_diamond, "_load_problems",
        lambda: [
            {"question_id": "d1", "question": "Q1", "options": ["w", "x", "y", "z"], "answer": "B"},
            {"question_id": "d2", "question": "Q2", "options": ["w", "x", "y", "z"], "answer": "C"},
        ],
    )
    # Model always answers B. So d1 (answer B) is 8/8 correct, d2 (answer C) is 0/8.
    # avg@8 = (8 + 0) / (2 * 8) = 0.5
    client = MockOfflineClient(lambda m, n: [r"\boxed{B}"] * n)
    cfg = {"n_samples_per_prompt": 8, "max_tokens": 256, "temperature": 0.7, "max_concurrent": 1}
    metrics = asyncio.run(gpqa_diamond._run_async(client, cfg, tmp_path / "out"))
    assert metrics["gpqa_diamond_avg_at_8"] == 0.5
    assert metrics["gpqa_diamond_avg_at_n"] == metrics["gpqa_diamond_avg_at_8"]
    assert metrics["gpqa_diamond_n_problems"] == 2
    assert metrics["gpqa_diamond_n_samples"] == 16


def test_gpqa_diamond_pass_at_n(tmp_path, monkeypatch):
    """pass@n = fraction of problems where ANY sample was correct."""
    monkeypatch.setattr(
        gpqa_diamond, "_load_problems",
        lambda: [
            {"question_id": "d1", "question": "Q1", "options": ["w","x","y","z"], "answer": "B"},
            {"question_id": "d2", "question": "Q2", "options": ["w","x","y","z"], "answer": "C"},
        ],
    )
    client = MockOfflineClient(lambda m, n: [r"\boxed{B}"] * n)
    cfg = {"n_samples_per_prompt": 8, "max_tokens": 256, "temperature": 0.7, "max_concurrent": 1}
    metrics = asyncio.run(gpqa_diamond._run_async(client, cfg, tmp_path / "out"))
    # d1: any-correct = 1; d2: any-correct = 0; pass@8 = 0.5
    assert metrics["gpqa_diamond_pass_at_8"] == 0.5
    assert metrics["gpqa_diamond_pass_at_n"] == metrics["gpqa_diamond_pass_at_8"]
