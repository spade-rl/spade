"""CodeContest evaluation over the full ``axon-rl/CodeContest`` test split.

Episodes cover each problem deterministically and require the extracted Python
solution to pass every provided stdin/stdout test case.
"""

import json
import logging
from typing import Any, Optional, SupportsFloat, Tuple

from datasets import Dataset, DatasetDict, load_dataset

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_code_from_model
from gem.utils.sandbox import run_python

logger = logging.getLogger(__name__)


def _parse_test(test: Any) -> str:
    """Normalize a stdin/stdout test field to a string (lists → newline-joined)."""
    if isinstance(test, str):
        return test
    if isinstance(test, list):
        return "\n".join(map(str, test))
    return str(test)


class SpareCodeContestEnv(Env):
    """Single-turn CodeContest env (full coverage + all tests), GEM-compatible."""

    def __init__(
        self,
        dataset_name: str = "axon-rl/CodeContest",
        split: str = "test",
        dataset: Optional[Dataset] = None,
        question_key: str = "problem",
        test_key: str = "tests",
        seed: int = 0,
        timeout: int = 6,
        sandbox_type: str = "none",
        **_: Any,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.question_key = question_key
        self.test_key = test_key
        self.timeout = timeout
        self.sandbox_type = sandbox_type

        if dataset is None:
            dataset = load_dataset(dataset_name)
        if isinstance(dataset, DatasetDict):
            if split is not None:
                dataset = dataset[split]
            elif len(list(dataset.keys())) == 1:
                dataset = dataset[list(dataset.keys())[0]]
            else:
                raise ValueError(
                    f"Dataset {dataset_name} has multiple splits; specify one: "
                    f"{list(dataset.keys())}"
                )
        assert isinstance(dataset, Dataset), f"Expected a Dataset, got {type(dataset)}"
        self.dataset = dataset
        self.idx = 0
        self.epoch = 0
        logger.info(
            "[CodeContest] Loaded %d problems (%s/%s) — full coverage, all tests",
            len(self.dataset), dataset_name, split,
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            # Episode-index seeds provide deterministic full-dataset coverage.
            self.idx = seed % len(self.dataset)
        else:
            if self.idx >= len(self.dataset):
                self.epoch += 1
                self.idx = 0
        data = self.dataset[self.idx]
        self.first_obs = data[self.question_key]
        tests = data[self.test_key]
        if isinstance(tests, str):
            tests = json.loads(tests)
        self.tests = tests
        self.idx += 1
        return self.first_obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        code = extract_code_from_model(action)
        if code is None:
            return TERMINAL_STATE, 0.0, True, True, {"correct": False}
        passed = self._check_all_tests(code)
        return TERMINAL_STATE, (1.0 if passed else 0.0), True, True, {"correct": passed}

    def _check_all_tests(self, code: str) -> bool:
        """Run the model code on EVERY test (stdin→stdout), early-exit on first
        failure. No max_tests cap — this is true pass@1."""
        inputs = self.tests.get("inputs", [])
        outputs = self.tests.get("outputs", [])
        if not inputs:
            return False
        for stdin, expected in zip(inputs, outputs):
            try:
                success, stdout, _stderr = run_python(
                    code, self.sandbox_type, _parse_test(stdin), timeout=self.timeout
                )
            except Exception as e:  # subprocess timeout / sandbox error
                logger.debug("[CodeContest] run failed: %s", e)
                return False
            if not success:
                return False
            if str(stdout).strip() != _parse_test(expected).strip():
                return False
        return True

    def get_state(self) -> dict[str, Any]:
        return {"first_obs": self.first_obs, "tests": self.tests}

    def set_state(self, state: dict[str, Any]) -> None:
        self.first_obs = state["first_obs"]
        self.tests = state["tests"]

    def sample_random_action(self) -> str:
        return "```python\nprint()\n```"
