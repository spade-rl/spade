"""Resolve a checkpoint argument to a local directory.

Accepts either a local path containing config.json + safetensors, or a
HuggingFace Hub repo id (e.g. `<org>/<model>`). Hub repos are
auto-downloaded to ${HF_HOME}/hub/<repo>.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_ckpt(ckpt: str) -> Path:
    """Resolve `ckpt` (local path or `org/name` hub id) to a local dir.

    Local paths must already contain `config.json` and at least one
    `*.safetensors` shard (or pytorch_model.bin). Hub repos are downloaded
    via `huggingface_hub.snapshot_download` into the HF cache (controlled
    by HF_HOME).

    Returns the resolved directory as a Path. Raises ValueError on
    structural issues, FileNotFoundError on missing local paths.
    """
    p = Path(ckpt)
    if p.exists() and p.is_dir():
        _validate_hf_dir(p)
        logger.info("[ckpt] using local path: %s", p)
        return p.resolve()

    if "/" in ckpt and not ckpt.startswith("/"):
        # Looks like an org/name hub id.
        return _download_hub_repo(ckpt)

    raise FileNotFoundError(
        f"Checkpoint {ckpt!r} is not a local directory and doesn't look like "
        f"a hub repo id (expected `org/name`)."
    )


def _validate_hf_dir(path: Path) -> None:
    if not (path / "config.json").is_file():
        raise ValueError(f"{path} is missing config.json — not an HF model dir.")
    has_weights = (
        any(path.glob("*.safetensors"))
        or (path / "pytorch_model.bin").is_file()
        or (path / "model.safetensors.index.json").is_file()
    )
    if not has_weights:
        raise ValueError(f"{path} has no safetensors / pytorch weight files.")


def _download_hub_repo(repo_id: str) -> Path:
    """Download a Hub repo to ${HF_HOME}/hub via snapshot_download."""
    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        cache_dir = str(Path(cache_dir) / "hub")
    token = os.environ.get("HF_TOKEN")
    logger.info(
        "[ckpt] downloading hub repo %s (cache_dir=%s, token=%s)",
        repo_id, cache_dir, "set" if token else "unset",
    )
    local_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        token=token,
        # Skip *.bin; safetensors only (slime conversion always emits safetensors).
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "tokenizer*",
            "*.txt",
            "*.model",
        ],
    )
    p = Path(local_path)
    _validate_hf_dir(p)
    logger.info("[ckpt] downloaded to: %s", p)
    return p
