"""Subprocess CLI for the PRIME/Golden-Goose CodeContests scorer.

Run as a CLEAN child process so PRIME's `evaluate_code` (which internally uses
multiprocessing.Process / fork for per-test isolation) never forks from inside
the CUDA-initialized, multithreaded slime training process (fork-after-CUDA /
fork-in-multithread → deadlock/corruption). The parent (SpareCodeContestEnv in
PRIME mode) invokes this via subprocess.run, so all of PRIME's forking happens
in this small clean process instead.

Protocol: read one JSON object {"completion": str, "ground_truth": <test_cases>}
from stdin, print one JSON object {"success": bool} to stdout. Any error →
{"success": false}, plus an "error" string for debugging (callers should ignore
unknown fields). Scoring uses the vendored code_util.evaluate_code implementation.
"""
import json
import os
import sys

# code_util + pyext are vendored next to this file; one dir on sys.path resolves
# both `import code_util` and pyext's `from pyext import RuntimeModule`.
_VENDOR = os.path.dirname(os.path.abspath(__file__))
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)


def main() -> None:
    out = {"success": False}
    try:
        from code_util import evaluate_code  # PRIME's exact scorer (vendored verbatim)

        payload = json.loads(sys.stdin.read())
        completion = payload["completion"]
        ground_truth = payload["ground_truth"]
        success, _meta = evaluate_code(completion, ground_truth)
        out["success"] = bool(success)
    except Exception as e:  # never let the child crash the parent's parsing
        out["success"] = False
        out["error"] = repr(e)
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
