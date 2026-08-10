"""Shared pytest fixtures for eval_offline suite tests."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest


@dataclass
class FakeCompletion:
    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 16
    completion_tokens: int = 32
    raw: dict[str, Any] = None  # type: ignore[assignment]


class MockOfflineClient:
    """Stand-in for eval_offline.client.OfflineClient.

    The test passes a `responder: Callable[[list[dict], int], list[str]]`
    that returns N completion strings per call. The mock wraps each in a
    FakeCompletion. Tracks every call so tests can assert on prompts/N.
    """

    def __init__(self, responder: Callable[[list[dict], int], list[str]]):
        self.responder = responder
        self.model = "mock-model"
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        n: int = 1,
        stop: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> list[FakeCompletion]:
        self.calls.append({
            "messages": messages, "temperature": temperature, "top_p": top_p,
            "max_tokens": max_tokens, "n": n, "stop": stop, "extra_body": extra_body,
        })
        texts = self.responder(messages, n)
        return [FakeCompletion(text=t) for t in texts]


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
