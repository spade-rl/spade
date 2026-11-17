# reasoning_gym eval splits

Split definitions live in the private dataset
[`spade-rl/rg-golden-goose`](https://huggingface.co/datasets/spade-rl/rg-golden-goose),
not in this repo; `spade/core/eval/rg_a3_hard.py` downloads them on first use
and raises `RgSplitUnavailable` without access. `SPADE_RG_SPLIT_PATH=<file>`
bypasses the download (CI, offline).

It holds the 100 reasoning_gym tasks at the paper's Appendix A.3 *hard* configs
(arXiv 2505.24760), one JSON object per line:

    {"task", "category", "config", "kept"}

`config` is the kwargs passed to `reasoning_gym.create_dataset(task, size=500,
seed=42, **config)`. `category` is the Golden Goose 4-way grouping (arXiv
2601.22975 Table 3): math = algebra+arithmetic+geometry+graphs, algorithmic =
algorithmic+code, cognition = arc+games+cognition, logic = logic+induction.

Both the A.3 configs and the grouping come from public papers, and the
NeMo-Gym answer extraction this eval reuses is Apache-2.0. Nothing here is
under a private sharing agreement.

Two configs, both `RG_PROTOCOL=boxed`, 16 samples/task, T 0.6, top_p 0.95, 32K
response: `rg_gg_hard.yaml` scores all 100 `rg:*-hard` tasks and
`rg_gg_subset.yaml` the 79-task `rg:*-gg` subset. `games.yaml` also runs the
100 `rg:*-hard` tasks, at the games-eval sampling settings.

## `kept`: the 79-task subset

`kept: false` marks 21 tasks removed as saturated. The cutoff is the
per-category win rate of **base Qwen3-4B-Instruct-2507** on this eval:

| category | cutoff | kept |
|---|---|---|
| math | 7/8 | 22/31 |
| algorithmic | 3/4 | 32/36 |
| cognition | 3/4 | 19/24 |
| logic | 15/16 | 6/9 |

Those four cutoffs were chosen to land the subset on previously reported
per-category numbers (31.6 / 11.8 / 16.4 / 54.7); it scores
31.8 / 11.5 / 16.4 / 57.3. That is four free parameters fitted to four
targets, so the agreement is a property of the selection and not evidence
about the protocol or the models. It is not a replication of those numbers.

Two constraints when reporting:

1. Not comparable to reasoning_gym's or Golden Goose's published numbers,
   which score all 100 tasks.
2. **Base-vs-finetune deltas on this subset are inflated.** Selection used
   base 4B's own scores, so the dropped tasks are exactly where base sat at
   ceiling. SPADE-4B vs base is **+11.4** RG win rate over all 100 tasks and
   **+15.6** over the subset. Report all-100 as primary, or re-select using a
   model outside the comparison.

Logic cannot go below 57.3 under this rule: dropping its three easiest tasks
(knights_knaves, syllogism, zebra_puzzles) leaves 6, and any lower cutoff
leaves fewer than 4 tasks in the category.
