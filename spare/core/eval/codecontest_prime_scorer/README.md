# CodeContests scorer — vendored from PRIME-RL/PRIME

Verbatim copy of [PRIME-RL/PRIME](https://github.com/PRIME-RL/PRIME)'s code-execution
scorer, vendored so our CodeContests reproduction is self-contained (no external scratch
path). Used to reproduce the **Qwen3-4B-Instruct-2507 CodeContests baseline = 42.08%**
(Golden Goose, arXiv:2601.22975) on PRIME's validation split.

**Reproduced: base-4B = 152/377 = 40.32%** (within serving noise of the paper's 42.08% —
sglang greedy versus the paper's vLLM setup).

## Contents (all copied, not reimplemented)

- `code_util/` — copied verbatim from PRIME's `eval/.../code_util/` (itself derived from the
  APPS metric, https://huggingface.co/spaces/codeparrot/apps_metric). `evaluate_code(completion,
  test_cases)` is the lenient APPS-style scorer: it extracts the ```` ```python ```` block, runs
  all tests once, and on any failure re-checks **only the first 10 test cases**
  (`__init__.py`: `if test_case_id >= 9: break`). Per-test comparison (`testing_util.custom_compare_`
  + `run_test`) falls back exact → per-line strip → float via `np.allclose` → set-unordered. Each
  test runs in a `multiprocessing.Process` with a timeout (`utils.check_correctness`).

- `pyext/` — verbatim copy of `pyext-0.6` (PyPI) `pyext.py`, which `testing_util.py:21` imports for
  `RuntimeModule`. pyext is uninstallable on Python ≥ 3.11 (it references the removed
  `inspect.getargspec`; the pip wheel build also fails on it). The **only** change is a 17-line
  compat preamble that aliases `inspect.getargspec` / `IPython.core.oinspect.getargspec` to
  `getfullargspec` so the genuine pyext imports — pyext's logic is untouched (453 lines verbatim).
  Proven to give identical `run_test` verdicts to a direct `exec(src, module.__dict__)`.

## Why our old in-loop scorer read 19% (not a bug, a strictness/data difference)

`SpareCodeContestEnv` (axon-rl/CodeContest, DeepMind merged-test split) runs **every** ~203
tests/problem with near-exact comparison → far stricter than PRIME (first-10 tests + lenient
compare on a different ~100-test split). 19% is the honest strict number; 40–42% is the
field-standard PRIME number. They measure different things.
