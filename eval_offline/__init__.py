"""Offline evaluation pipeline.

Runs the retained paper benchmark suites against a local Hugging Face
checkpoint or an OpenAI-compatible endpoint. See `eval_offline/README.md`.
"""

# Older datasets releases expect this tokenizer class at Transformers' top level.
try:
    import transformers as _transformers  # type: ignore[import-untyped]
    if not hasattr(_transformers, "PreTrainedTokenizerBase"):
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB
        _transformers.PreTrainedTokenizerBase = _PTB  # type: ignore[attr-defined]
except Exception:
    # Individual suites report missing optional dependencies when invoked.
    pass
