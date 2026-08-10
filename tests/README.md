# Tests

Run the offline suite from the repository root:

```bash
python -m pytest
```

Tests that need optional packages or local model assets skip when those
dependencies are unavailable. `test_actor_thinking.py` can use a local Qwen3
tokenizer through `QWEN3_TOKENIZER`; the remaining tests run without a GPU.

Generated games, API experiments, and one-off validation scripts belong under
`scripts/` or an ignored output directory, not in this test suite.
