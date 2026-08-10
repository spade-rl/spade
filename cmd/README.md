# Release commands

The public command surface is organized by experiment type:

```text
cmd/
├── games/       # SPADE games and fixed-environment game baselines
├── tool_use/    # SPADE tool-use training
├── ablations/   # Paper ablations expressed as thin wrappers
├── eval/        # Evaluation entry points and preparation helpers
└── tinker/      # Retained Tinker training backend
```

Only commands tied to the active paper are kept here. Historical sweeps,
date-stamped trials, cluster-specific submission wrappers, terminal experiments,
and unsupported model variants are intentionally excluded from the release.

## Prerequisites

None of the training launchers are self-contained: each one expects the Slime
GPU environment, a converted reference checkpoint, and external evaluation data.

### Training environment

Every launcher starts a local Ray head with `--num-gpus 8` and submits a job
with `--actor-num-gpus-per-node 8`, so a single 8-GPU node with Ray available is
required. The submitted runtime environment sets
`PYTHONPATH=/root/Megatron-LM/`, which only resolves inside Slime's GPU
container; upstream Slime publishes that environment as the `slimerl/slime`
image family (`slimerl/slime:latest`), documented in
`slime/docs/en/get_started/quick_start.md` of the pinned submodule.

Initialize the submodule before launching — the 4B commands source their model
configuration from it:

```bash
git submodule update --init --recursive slime
```

### Reference checkpoints (`--ref-load`)

Every launcher passes `--hf-checkpoint ${MODEL_ROOT}/<Model>` together with
`--ref-load ${MODEL_ROOT}/<Model>_torch_dist`. The `_torch_dist` directory is a
Megatron checkpoint and is not downloadable — build it from the HF checkpoint
with the converter in the pinned Slime submodule, after sourcing the matching
model configuration:

```bash
cd slime
source ../cmd/models/qwen3-8B.sh          # defines MODEL_ARGS
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ${MODEL_ROOT}/Qwen3-8B \
    --save ${MODEL_ROOT}/Qwen3-8B_torch_dist
```

Model configurations: `cmd/models/qwen3-8B.sh` and
`cmd/models/qwen3-30B-A3B.sh` in this repository,
`slime/scripts/models/qwen3-4B-Instruct-2507.sh` in the submodule.

### AIME evaluation data

The in-loop eval configurations (`eval_configs/eval_aime.yaml`,
`eval_configs/eval_aime_avg32.yaml`) are expanded with `envsubst` and read
`${WORKSPACE_DIR}/aime-2025/aime-2025.jsonl` and
`${WORKSPACE_DIR}/aime-2026/aime-2026.jsonl`. Each line is
`{"prompt": [{"role": "user", "content": "..."}], "label": "73"}`. The files
are not bundled. The Tinker launchers (`cmd/tinker/*/train_spare_tinker.sh`)
additionally read `${WORKSPACE_DIR}/aime-2024/aime-2024.jsonl` in the same
format.

`cmd/eval/prepare_aime_datasets.py` post-processes existing files; it does not
download them. It appends the step-by-step `\boxed{}` instruction to every user
message and, by default, rewrites each input in place after saving a
`<name>.jsonl.backup` next to it:

```bash
python cmd/eval/prepare_aime_datasets.py \
    ${WORKSPACE_DIR}/aime-2025/aime-2025.jsonl \
    ${WORKSPACE_DIR}/aime-2026/aime-2026.jsonl
```

Use `--no-backup` to skip the backup, or `-o DIR` to write
`<name>_eval.jsonl` into `DIR` instead of editing in place.

### BFCL data (tool-use recipes)

`tool_use/_train_spade_blend.sh` aborts unless both `${WORKSPACE_DIR}/bfcl/data`
and `${WORKSPACE_DIR}/bfcl/data_v3` exist. They back the in-loop BFCL eval in
`eval_configs/gem_eval_bfcl_only.yaml`: `bfcl/data` is read with the `v4` file
prefix, `bfcl/data_v3` with `v3`. Both are data trees from the Berkeley Function
Calling Leaderboard in the gorilla repository
(<https://github.com/ShishirPatil/gorilla>, `berkeley-function-call-leaderboard`)
and are not redistributed here. Each tree must provide, per evaluated category:

```text
<tree>/BFCL_<v4|v3>_<category>.json               # one JSON object per line
<tree>/possible_answer/BFCL_<v4|v3>_<category>.json
<tree>/multi_turn_func_doc/<api>.json             # gorilla_file_system.json, ...
```

Under `v4` three categories use renamed files: `simple` → `simple_python`,
`java` → `simple_java`, `javascript` → `simple_javascript`. The multi-turn
execution backend is imported from a gorilla checkout;
`spare/core/eval/bfcl_evaluator.py` defaults to
`/workspace/spare-workspace/bfcl/gorilla/berkeley-function-call-leaderboard`
and honours `BFCL_REPO_PATH`.

### `CORPUS_FILE` schema

The adaptive games and tool-use recipes require `CORPUS_FILE`, a JSONL
grounding corpus consumed by `spare/core/corpus.py`. The paper corpora are not
distributed; supply your own file in this format.

One JSON object per line. Each line needs exactly one text field:

| Key | Type | Required | Meaning |
|---|---|---|---|
| `orig_doc` | string | one of the two | Document text (SPICE format); wins if both are present |
| `text` | string | one of the two | Document text (generic format) |
| any other key | any | no | Retained as document metadata; not shown to the model |

Blank lines, unparseable lines, and lines whose text is empty are skipped; if no
document loads, the run aborts. Each document is truncated to
`--spare-corpus-max-doc-tokens` whitespace-separated words (6000 for games,
4000 for tool use) and sampled uniformly with `--spare-corpus-seed`.

```json
{"text": "Ferry schedules on the lower river are published as a table of departure times ...", "source": "wikipedia:River_ferry"}
```

### Weights & Biases

`WANDB_ENTITY` (your W&B team) and `WANDB_API_KEY` are required.
`WANDB_PROJECT` is optional and defaults to `spade`; `WANDB_GROUP` overrides the
per-recipe group name.

## Paper command coverage

| Paper experiment | Public command coverage |
|---|---|
| SPADE games, 4B / 8B / 30B | `games/train_spade_{4b,8b,30b}.sh` |
| Fixed-env RLVE, 4B / 8B / 30B | `games/train_fixed_rlve_{4b,8b,30b}.sh` |
| Fixed-env GRPO on GPT-5.5 curated games, 4B / 8B / 30B | `games/train_fixed_gpt55_{4b,8b,30b}.sh` |
| Tool-use SPADE, 4B | `tool_use/train_spade_4b.sh` |
| Tool-use SPADE, 8B / 30B | `tool_use/train_spade_{8b,30b}.sh` |
| 30B corpus, memory, ED-training, GPT-5.5, and two-skill controls | `ablations/` |
| Learning-potential / solve-rate reward controls | `ablations/train_{learning_potential,solve_rate}_30b.sh` |

These launchers encode the user-confirmed paper recipes; they do not claim
historical run provenance. Required external checkpoints and result artifacts
are called out explicitly instead of being inferred from nearby experiments.

Adaptive games and tool-use launchers require an explicit `CORPUS_FILE` path.
Those grounding corpora are not redistributed because an authoritative public
snapshot, checksum, and redistribution record have not been established. The
fixed GPT-5.5 environment corpus is the separately pinned Hugging Face dataset
documented in `games/README.md`.
