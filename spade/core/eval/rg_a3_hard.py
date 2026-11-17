"""reasoning_gym splits for the RG eval, loaded from a gated Hugging Face dataset.

The split definitions live in ``spade-rl/rg-golden-goose`` (private), not in
this repo: running the RG eval requires access to that dataset. Two
configurations are derived from it:

* ``hard`` -- the reasoning_gym paper (arXiv 2505.24760) Appendix A.3 hard
  configs, transcribed from the A.3 "# hard" column: the params behind that
  paper's zero-shot leaderboard (Fig 3a) and difficulty cliff (Fig 3b).
  100 datasets, validated to construct.
* ``golden_goose`` / ``gg`` -- ``hard`` restricted to the 79 tasks that remain
  once the saturated ones are dropped.

See eval_offline/configs/rg_hard_split.README.md for how the subset was
selected and what may not be claimed from a result measured on it.

Nothing is fetched at import. The dataset is downloaded (and thereafter served
from the local HF cache) the first time a split is actually read; without
access that read raises :class:`RgSplitUnavailable`. Set
``SPADE_RG_SPLIT_PATH`` to a local JSONL to bypass the download, e.g. in CI.
"""

import json
import logging
import os
from typing import Any, Dict, Set

logger = logging.getLogger(__name__)

HF_REPO_ID = "spade-rl/rg-golden-goose"
HF_FILENAME = "rg_eval_splits.jsonl"


class RgSplitUnavailable(RuntimeError):
    """The reasoning_gym split definitions could not be read."""


_HELP = (
    f"Could not read the reasoning_gym eval splits from the {HF_REPO_ID} "
    f"dataset. It is private, so running the RG eval needs access to it:\n"
    f"  1. ask a SPADE maintainer for read access to "
    f"https://huggingface.co/datasets/{HF_REPO_ID}\n"
    f"  2. authenticate with `huggingface-cli login`, or set HF_TOKEN\n"
    f"Alternatively point SPADE_RG_SPLIT_PATH at a local copy of "
    f"{HF_FILENAME}."
)

_ROWS: "list[dict[str, Any]] | None" = None


def _rows() -> "list[dict[str, Any]]":
    """Read the split definitions, downloading them on first use."""
    global _ROWS
    if _ROWS is not None:
        return _ROWS
    local = os.environ.get("SPADE_RG_SPLIT_PATH")
    if not local:
        try:
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(HF_REPO_ID, HF_FILENAME, repo_type="dataset")
        except Exception as e:  # noqa: BLE001 -- auth, network and 404 all land here
            raise RgSplitUnavailable(f"{_HELP}\n\ncause: {e!r}") from e
    try:
        with open(local) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except OSError as e:
        raise RgSplitUnavailable(f"{_HELP}\n\ncause: {e!r}") from e
    if not rows:
        raise RgSplitUnavailable(f"{_HELP}\n\ncause: {local} is empty")
    _ROWS = rows
    return _ROWS


def load_split(name: str) -> Dict[str, Dict[str, Any]]:
    """Return task -> ``reasoning_gym.create_dataset`` kwargs for a split.

    Raises :class:`RgSplitUnavailable` without access to the dataset, and
    ``ValueError`` for an unknown split name.
    """
    if name == "hard":
        return {r["task"]: r["config"] for r in _rows()}
    if name in ("golden_goose", "gg"):
        return {r["task"]: r["config"] for r in _rows() if r["kept"]}
    raise ValueError(
        f"unknown rg split {name!r}; expected 'hard' or 'golden_goose'"
    )


# Module-level names kept for existing callers. PEP 562: resolved on first
# attribute access, so importing this module still costs nothing and needs no
# network.
_LAZY = {
    "A3_HARD_CONFIGS": lambda: load_split("hard"),
    "GG_SUBSET": lambda: {r["task"] for r in _rows() if r["kept"]},
}


def __getattr__(name: str) -> Any:
    try:
        build = _LAZY[name]
    except KeyError:
        raise AttributeError(name) from None
    return build()


def __dir__() -> "list[str]":
    return sorted(list(globals()) + list(_LAZY))


if False:  # type-checker only; resolved at runtime by __getattr__ above
    A3_HARD_CONFIGS: Dict[str, Dict[str, Any]]
    GG_SUBSET: Set[str]
