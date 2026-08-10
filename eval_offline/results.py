"""Dataclasses + JSON/JSONL writers for offline eval results."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SuiteResult:
    name: str
    metrics: dict[str, Any]
    elapsed_sec: float
    skipped: bool = False
    error: str | None = None


@dataclass
class RunResult:
    ckpt: str                        # original arg
    ckpt_resolved: str               # local path
    started_at: str                  # ISO 8601
    elapsed_sec: float = 0.0
    suites: dict[str, SuiteResult] = field(default_factory=dict)

    def add(self, suite: SuiteResult) -> None:
        self.suites[suite.name] = suite

    def to_dict(self) -> dict[str, Any]:
        return {
            "ckpt": self.ckpt,
            "ckpt_resolved": self.ckpt_resolved,
            "started_at": self.started_at,
            "elapsed_sec": self.elapsed_sec,
            "per_suite_elapsed_sec": {
                n: s.elapsed_sec for n, s in self.suites.items()
            },
            "metrics": {n: s.metrics for n, s in self.suites.items()},
            "skipped": [n for n, s in self.suites.items() if s.skipped],
            "errors": {n: s.error for n, s in self.suites.items() if s.error},
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
