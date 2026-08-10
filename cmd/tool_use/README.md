# Tool-use training

The paper tool-use commands are:

- `train_spade_4b.sh`: the original 4B KL-anchored tool-use run.
- `train_spade_8b.sh`: the 30B blend recipe mirrored onto Qwen3-8B, with both
  roles thinking.
- `train_spade_30b.sh`: blend reward, regeneration/delay 8, 16 hint plays,
  rollout batch 24, global batch 192, and KL `0.005`.

The 8B and 30B entry points share `_train_spade_blend.sh`. The shared paper
recipe keeps the reward, batch, regeneration, hint, and KL settings matched;
model parallelism, environment token budget, and thinking mode remain
model-specific. The 8B command enables thinking for both roles, while 30B is
non-thinking.

Set `CORPUS_FILE=/path/to/tool-use-corpus.jsonl` before launching any of these
commands. The corpus is not bundled because its authoritative public snapshot,
checksum, license, and redistribution terms remain unresolved.
