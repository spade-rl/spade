"""GEM-compatible LiveCodeBench v6 environment using the official runner.

The environment uses official prompts, extraction, and correctness checks for
both stdin and call-based problems. Reward is one only when all tests pass.
"""

import logging
import os
import random
import sys
from typing import Any, List, Optional, Tuple

from gem.core import Env

logger = logging.getLogger(__name__)

TERMINAL_STATE = ""

_LCB_OFFICIAL_DIR = os.environ.get(
    "LCB_OFFICIAL_DIR",
    "/opt/lcb",
)


def _ensure_lcb_on_path() -> None:
    if _LCB_OFFICIAL_DIR not in sys.path:
        if not os.path.isdir(_LCB_OFFICIAL_DIR):
            raise FileNotFoundError(
                f"Official LiveCodeBench repo not found at {_LCB_OFFICIAL_DIR}. "
                "git clone https://github.com/LiveCodeBench/LiveCodeBench.git "
                "and set LCB_OFFICIAL_DIR."
            )
        sys.path.insert(0, _LCB_OFFICIAL_DIR)


class LiveCodeBenchEnv(Env):
    """Single-turn LiveCodeBench v6 env (contextual bandit), GEM-compatible.

    Thin wrapper over the official lcb_runner harness — all prompting,
    extraction, and execution/scoring is official code.
    """

    def __init__(
        self,
        release_version: str = "release_v6",
        jsonl_path: Optional[str] = None,
        seed: int = 0,
        timeout: int = 6,
        **_: Any,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.timeout = timeout
        _ensure_lcb_on_path()
        # Build official problem objects directly from the release-v6 JSONL.
        import json as _json
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

        path = (
            os.environ.get("LCB_V6_JSONL")
            or jsonl_path
            or "/workspace/spare-workspace/livecodebench/test6.jsonl"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"LiveCodeBench {release_version} data not found: {path}. "
                "Download test6.jsonl from livecodebench/code_generation_lite."
            )
        self.problems: List[Any] = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    self.problems.append(CodeGenerationProblem(**_json.loads(line)))
        rng = random.Random(seed)
        rng.shuffle(self.problems)
        self._iter = iter(self.problems)
        self._cur: Optional[Any] = None
        logger.info("[LCB] Loaded %d official LiveCodeBench problems (%s) from %s",
                    len(self.problems), release_version, path)

    def _build_prompt(self, problem: Any) -> str:
        # Official LCB prompt, inlined verbatim to avoid its cwd-relative asset load.
        SYSTEM = (
            "You are an expert Python programmer. You will be given a question "
            "(problem specification) and will generate a correct Python program "
            "that matches the specification and passes all tests."
        )
        FMT_STARTER = (
            "You will use the following starter code to write the solution to "
            "the problem and enclose your code within delimiters."
        )
        FMT_STDIN = (
            "Read the inputs from stdin solve the problem and write the answer "
            "to stdout (do not directly test on the sample inputs). Enclose your "
            "code within delimiters as follows. Ensure that when the python "
            "program runs, it reads the inputs, runs the algorithm and writes "
            "output to STDOUT."
        )
        p = f"### Question:\n{problem.question_content}\n\n"
        if problem.starter_code:
            p += f"### Format: {FMT_STARTER}\n```python\n{problem.starter_code}\n```\n\n"
        else:
            p += f"### Format: {FMT_STDIN}\n```python\n# YOUR CODE HERE\n```\n\n"
        p += "### Answer: (use the provided format with backticks)\n\n"
        return SYSTEM + "\n\n" + p

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict]:
        super().reset(seed)
        if seed is not None:
            # Episode-index seeds provide deterministic full-dataset coverage.
            self._cur = self.problems[seed % len(self.problems)]
        else:
            try:
                self._cur = next(self._iter)
            except StopIteration:
                self._iter = iter(self.problems)
                self._cur = next(self._iter)
        return self._build_prompt(self._cur), {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        _ensure_lcb_on_path()
        from lcb_runner.evaluation.compute_code_generation_metrics import (
            check_correctness,
        )
        from lcb_runner.utils.extraction_utils import extract_code
        from lcb_runner.lm_styles import LMStyle

        code = extract_code(action, LMStyle.OpenAIChat)  # official ``` extraction
        sample = self._cur.get_evaluation_sample()
        try:
            result, _meta = check_correctness(
                sample, code, timeout=self.timeout, debug=False
            )
            passed = bool(result) and all(r is True or r == True for r in result)
            reward = 1.0 if passed else 0.0
        except Exception as e:
            logger.debug("[LCB] check_correctness failed: %s", e)
            reward = 0.0
        return TERMINAL_STATE, reward, True, True, {}

    def get_state(self) -> dict:
        return {"cur": self._cur}

    def set_state(self, state: dict) -> None:
        self._cur = state["cur"]
