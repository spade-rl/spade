# Tinker backend

The Tinker launchers are retained as a supported alternative backend. Paper
result commands use Slime and live in the adjacent experiment directories.

## Installation

The Tinker backend needs Python 3.11 or newer — `tinker`, `chz`, and
`tinker_cookbook` all declare `requires-python >= 3.11`. Combined with the
`gem-llm` cap described in the top-level [`README.md`](../../README.md), a
Tinker environment is easiest on Python 3.11.

Install the PyPI dependencies through the extra:

```bash
python -m pip install -e ".[tinker]"
```

That extra covers the [Tinker SDK](https://github.com/thinking-machines-lab/tinker)
(`tinker`), the [chz](https://github.com/openai/chz) configuration library used
by `train_spare_tinker.py`, and `torch`.

`tinker_cookbook` is **not** installed from PyPI. This repository pins a
specific [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
revision as a git submodule, and the launchers are written against that
revision, so install it from the checkout:

```bash
git submodule update --init tinker-cookbook
python -m pip install -e ./tinker-cookbook
```

Finally, export a Tinker API key before launching:

```bash
export TINKER_API_KEY=...
```

## Running

```bash
export WANDB_ENTITY=your-wandb-entity
bash cmd/tinker/qwen3_8b/train_spare_tinker.sh
```

Each launcher reads its knobs from environment variables (`MODEL_NAME`,
`BATCH_SIZE`, `LEARNING_RATE`, ...); see the top of the script for the full
list and defaults.
