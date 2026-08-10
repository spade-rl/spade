# Evaluation

Offline evaluation is implemented in `eval_offline/`; reusable suite
configuration lives in `eval_offline/configs/` and `eval_configs/`.

```bash
python -m eval_offline.run_offline_eval --help
python -m eval_offline.render_table --help
```

`prepare_aime_datasets.py` is the preprocessing helper used by retained game
launchers. `eval_tinker_single_agent_games.py` is the retained Tinker evaluator.
