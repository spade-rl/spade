# Games training

| Command | Paper setting | Provenance |
|---|---|---|
| `train_spade_4b.sh` | SPADE, Qwen3-4B-Instruct-2507 | Initial defaults match; later phase requires the overrides below |
| `train_spade_8b.sh` | SPADE, Qwen3-8B | Defaults match the paper: both roles thinking, KL `0.005` |
| `train_spade_30b.sh` | SPADE, Qwen3-30B-A3B-Instruct-2507 | Strong command match |
| `train_fixed_rlve_{4b,8b,30b}.sh` | Fixed-env RLVE baselines | Commands recovered; result JSONs still required |
| `train_fixed_gpt55_{4b,8b,30b}.sh` | Fixed-env GRPO on GPT-5.5 curated games | Shared 7,872-game pool and `400/24/192` budget |

Fixed-env GRPO means actor-only GRPO on the shared GPT-5.5 curated-game pool;
it is distinct from the RLVE baseline. The three model entry points download
and verify the same pool before invoking the common trainer. Set
`HF_DATASET_REVISION` to override the pinned paper snapshot
`e179a371bc7764dacf0bcee1f808100beb463137`.

The adaptive SPADE launchers require `CORPUS_FILE=/path/to/games-corpus.jsonl`.
The grounding corpus is not bundled because its authoritative public snapshot,
checksum, license, and redistribution terms remain unresolved. The
`train_no_corpus_30b.sh` ablation explicitly sets `CORPUS_FILE` to an empty
value and is the only paper command that intentionally runs without it.

The paper records a later 4B phase with a 49,152-token context and plateau band
`[0.2, 0.4]`. Resume that phase with
`MAX_CONTEXT_LENGTH=49152 PLATEAU_LO=0.2 PLATEAU_HI=0.4 PLATEAU_RAMP=0.2`.
The exact transition checkpoint is not encoded in the paper or launcher, so it
remains a required external input rather than an invented default.

Files beginning with `_` are shared implementation helpers and are not separate
experiments.

## Baseline settings

The fixed-RLVE baseline does not share the SPADE hyperparameters; it reproduces
the paper's baseline configuration, so these differences are deliberate rather
than drift:

| Setting | `train_spade_{4b,8b,30b}.sh` | `train_fixed_rlve_{4b,8b,30b}.sh` |
|---|---|---|
| `--rollout-batch-size` / `--global-batch-size` | 24 / 192 | 16 / 256 |
| `--spare-actor-temperature` | 0.6 | 1.0 |
| `--kl-loss-coef`, 8B | 0.005 | 0.00 |

At 4B and 30B both settings use `--kl-loss-coef 0.00`; the 8B SPADE run is the
only KL-anchored one. Rollout budget (`--num-rollout 400`) and rollout sampling
temperature (`--rollout-temperature 1.0`) match across both.
