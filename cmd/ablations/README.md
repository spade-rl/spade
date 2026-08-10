# Games ablations

These thin wrappers make each paper intervention explicit while reusing the
canonical 30B games command.

| Command | Intervention | Status |
|---|---|---|
| `train_no_corpus_30b.sh` | Disable corpus grounding | Strong command match |
| `train_no_memory_30b.sh` | Disable environment memory | Strong command match |
| `train_no_ed_training_no_memory_30b.sh` | Disable ED updates and memory | Reconstructed; original wrapper/run ID missing |
| `train_fixed_gpt55_30b.sh` | Train on the frozen GPT-5.5 game pool | Paper fixed-env control |
| `train_two_skill_30b.sh` | Restrict the curriculum to Mathematical Reasoning and Pattern Recognition | Matched paper control |
| `train_learning_potential_30b.sh` | Select the slow-EMA-distance `learning_potential` mode | Paper reward control |
| `train_solve_rate_30b.sh` | Use only the solve-rate plateau reward over `[0.4, 0.6]` | Paper reward control |

The solve-rate command selects the existing plateau component with weight `1.0`
and sets regret, micro-LP, and frontier weights to zero.

The existing `learning_potential` mode maintains fast and slow performance EMAs,
but its environment reward is the absolute distance from the slow EMA baseline.
The fast-minus-slow gap is recorded as a monitoring metric rather than used
directly as the reward; this is the intended paper-control implementation.
