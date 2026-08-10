"""Registry for the benchmark suites retained in the paper release."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Protocol


class SuiteFn(Protocol):
    def __call__(self, client: Any, cfg: dict, out_dir: Path) -> dict[str, Any]: ...


BUILTIN_SUITES = {
    "aime": "eval_offline.suites.aime",
    "gem": "eval_offline.suites.gem",
    "gpqa_diamond": "eval_offline.suites.gpqa_diamond",
    "livecodebench": "eval_offline.suites.livecodebench",
    "bfcl_full": "eval_offline.suites.bfcl_full",
    "tau2": "eval_offline.suites.tau2",
    "acebench": "eval_offline.suites.acebench",
}


def load_suite(name: str) -> SuiteFn:
    """Resolve `name` to a callable `run(client, cfg, out_dir)`."""
    if name not in BUILTIN_SUITES:
        raise KeyError(f"Unknown suite {name!r}. Available: {list(BUILTIN_SUITES)}.")
    mod = importlib.import_module(BUILTIN_SUITES[name])
    if not hasattr(mod, "run"):
        raise AttributeError(f"Suite module {mod.__name__} is missing `run(...)`.")
    return mod.run
