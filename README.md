<div align="center">

# SPADE ♠

### Self-Play in Adaptive Synthetic Executable Environments

[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗%20Dataset-GPT--5.5%20Static%20Corpus-FFD21E)](https://huggingface.co/datasets/spare-rl/spare-gpt55-static-corpus)

[Installation](#installation) · [Configuration](#configuration) · [Paper recipes](#paper-recipes) · [Evaluation](#evaluation) · [Citation](#citation)

</div>

SPADE is a self-play reinforcement-learning framework in which one language
model learns in two roles. An **environment designer** writes executable Python
environments with `reset()` and `step()` interfaces, and a **reasoning agent**
learns by interacting with those environments. Both roles update the same model.

The designer is trained with **hint-based regret**: the gap between the agent's
return with and without a privileged hint. That signal steers generation toward
environments near the agent's capability frontier while keeping them feasible.
The released recipes cover cognitive games and multi-turn tool use at 4B, 8B,
and 30B-A3B scale.

> SPADE is the paper and project name; the earlier `spare` name is retained
> where renaming would break existing artifacts. The Python package is still
> imported as `spare`, and the released dataset still lives under the
> `spare-rl` Hugging Face organization, while the code is published at
> [github.com/spade-rl](https://github.com/spade-rl).

<p align="center">
  <img src="assets/spade_framework.png" alt="SPADE framework overview" width="100%">
</p>

<p align="center"><em>One shared policy learns to design executable environments and solve them with hint-based regret.</em></p>

## How it works

1. The designer samples grounding context and generates a complete executable
   environment, including a privileged hint.
2. Generated code passes structural and runtime validation before entering the
   training pool.
3. The reasoning agent plays the environment with and without the hint.
4. Task return trains the reasoning role, hint-based regret trains the designer
   role, and accumulated rollout memory guides later environment generation.

The backend-independent orchestration lives in `spare/core/`. Distributed
training uses the Slime/SGLang integration under `spare/slime/`; a retained
Tinker integration lives under `spare/tinker/`, with setup and launchers
documented in [`cmd/tinker/README.md`](cmd/tinker/README.md).

To reproduce a paper run, work in this order: install the package
([Installation](#installation)), export the configuration variables
([Configuration](#configuration)), satisfy the checkpoint and data
prerequisites in [`cmd/README.md`](cmd/README.md#prerequisites), then launch a
recipe ([Paper recipes](#paper-recipes)).

## Installation

Clone the repository with its pinned submodules — `slime` for distributed
training and `tinker-cookbook` for the Tinker backend — then install SPADE:

```bash
git clone --recurse-submodules https://github.com/spade-rl/spade.git
cd spade

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional extras add the local test and lint tooling (`dev`) and the Tinker
backend (`tinker`); the offline evaluation extra is covered under
[Evaluation](#evaluation).

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[tinker]"
```

Python 3.10 through 3.12 are supported. GEM environment support comes from
`gem-llm`, whose published metadata caps at Python 3.12.0, so on 3.12.1 and
newer it is skipped by the base install and has to be added manually:

```bash
python -m pip install --ignore-requires-python gem-llm
```

Python 3.10 or 3.11 avoids that step and is the easiest choice for the full
stack. The Tinker backend is the exception — it needs 3.11 or newer, plus the
`tinker-cookbook` submodule installed from the checkout; see
[`cmd/tinker/README.md`](cmd/tinker/README.md).

## Configuration

The launchers under `cmd/` take their paths and credentials from the
environment. Export these before running any recipe:

```bash
# Holds the HF checkpoints (<Model>) and their Megatron conversions
# (<Model>_torch_dist) that launchers pass as --hf-checkpoint / --ref-load.
export MODEL_ROOT=/path/to/model/checkpoints

# Root of the external eval data the launchers read from disk:
# aime-2025/ and aime-2026/ for AIME (the Tinker launchers also read
# aime-2024/), bfcl/ for the tool-use recipes.
export WORKSPACE_DIR=/path/to/spade/workspace

# JSONL grounding corpus the designer samples context from; required by the
# adaptive games and tool-use recipes. Schema in cmd/README.md.
export CORPUS_FILE=/path/to/paper-grounding-corpus.jsonl

# Weights & Biases logging.
export WANDB_API_KEY=...                     # required
export WANDB_ENTITY=your-wandb-entity        # required: your W&B team
export WANDB_PROJECT=spade                   # optional, defaults to "spade"
export WANDB_GROUP=my-run-group              # optional, overrides the per-recipe group
```

Paths alone are not enough: Slime training also requires its GPU environment,
model checkpoints, and an 8×GPU node. Building the `_torch_dist` reference
checkpoint and laying out the AIME and BFCL data under `WORKSPACE_DIR` are
documented in [`cmd/README.md`](cmd/README.md#prerequisites).

## Paper recipes

Paper commands are organized by setting and intentionally exclude historical
internal experiments. Work through
[Prerequisites](cmd/README.md#prerequisites) in `cmd/README.md` before the first
launch; the launchers assume that setup exists and fail fast when it does not.

| Setting | Models | Commands |
|---|---|---|
| SPADE games | 4B, 8B, 30B-A3B | `cmd/games/train_spade_{4b,8b,30b}.sh` |
| Fixed-env GRPO, GPT-5.5 corpus | 4B, 8B, 30B-A3B | `cmd/games/train_fixed_gpt55_{4b,8b,30b}.sh` |
| Fixed-env RLVE | 4B, 8B, 30B-A3B | `cmd/games/train_fixed_rlve_{4b,8b,30b}.sh` |
| SPADE tool use | 4B, 8B, 30B-A3B | `cmd/tool_use/train_spade_{4b,8b,30b}.sh` |
| Paper ablations | 30B-A3B | `cmd/ablations/*.sh` |

For example:

```bash
# Adaptive games self-play, both roles trained
bash cmd/games/train_spade_8b.sh

# Actor-only GRPO on the pinned GPT-5.5 environment corpus
bash cmd/games/train_fixed_gpt55_8b.sh

# Tool-use self-play with the matched blend reward
bash cmd/tool_use/train_spade_30b.sh

# Reward control: solve-rate plateau instead of hint-based regret
bash cmd/ablations/train_solve_rate_30b.sh

# Curriculum control: two cognitive skills instead of the full set
bash cmd/ablations/train_two_skill_30b.sh
```

[`cmd/README.md`](cmd/README.md) has the full command matrix; the
setting-specific READMEs under `cmd/games/`, `cmd/tool_use/`, and
`cmd/ablations/` document the corpora and overrides each recipe expects.

`CORPUS_FILE` must be set explicitly for the adaptive games and tool-use
recipes; the no-corpus ablation is the sole exception. Those grounding corpora
are not bundled because their authoritative public snapshots, checksums, and
redistribution records remain unresolved.

## Static GPT-5.5 corpus

All three fixed-env GRPO launchers train on the same public corpus, so they
need no `CORPUS_FILE`:

- **Dataset:** [`spare-rl/spare-gpt55-static-corpus`](https://huggingface.co/datasets/spare-rl/spare-gpt55-static-corpus)
- **Pinned revision:** `e179a371bc7764dacf0bcee1f808100beb463137`
- **Contents:** 7,872 validated Python environments across six cognitive skills
- **Integrity:** one SHA-256 entry per environment in `checksums.sha256`
- **Dataset license:** Apache-2.0

The launcher downloads this exact revision and verifies all checksums before
training. Override `HF_DATASET_REVISION` only when intentionally testing a
different snapshot.

## Evaluation

`eval_offline/` scores a trained checkpoint, or an existing OpenAI-compatible
endpoint, against the benchmark suites retained for the paper.
`run_offline_eval` runs them and `render_table` formats the results:

```bash
python -m eval_offline.run_offline_eval --help
python -m eval_offline.render_table --help
```

The runner needs the `[eval]` extra and per-benchmark data setup. Both, plus
the retained benchmark matrix, are documented in
[`eval_offline/README.md`](eval_offline/README.md).

`eval_configs/` is separate: those YAML files drive the in-loop evaluations the
training launchers run during a job, not the offline driver.

## Citation

```bibtex
@misc{liu2026spade,
  title  = {SPADE: Self-Play in Adaptive Synthetic Executable Environments},
  author = {Bo Liu and Simon Yu and Yiding Jiang and Ao Qu and Andrew Zhao and
            Zichen Liu and Junsu Kim and Zijian Zhou and Seungone Kim and
            Tongzheng Ren and Mickel Liu and Hanfei Yu and Zhaorun Chen and
            Weiyan Shi and Paul Pu Liang and Luke Zettlemoyer and Yejin Choi and
            Natasha Jaques},
  year   = {2026}
}
```

## Acknowledgments

- [Slime](https://github.com/THUDM/slime), the distributed RL training backbone
  this release builds on, and [SGLang](https://github.com/sgl-project/sglang)
  for inference.
- The [Miles](https://github.com/radixark/miles) team, whose RL post-training
  framework informed this project's training infrastructure.
- [Tinker](https://thinkingmachines.ai/tinker) and
  [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook),
  the retained alternative training backend.
- [Modal](https://modal.com) for compute and model serving during development.
- RLVE, the Berkeley Function Calling Leaderboard, and PRIME evaluation
  utilities.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for retained licenses and
provenance.

## License

SPADE source code is released under the [MIT License](LICENSE). Datasets,
vendored code, and adapted evaluation components retain their respective
licenses as documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
