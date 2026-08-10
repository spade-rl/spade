"""GEM environment adapter for fixed-env training.

Thin wrapper around GEM's Gym-compatible environments.
GEM already matches the Gym API (reset/step/close), so this
adapter primarily handles listing and instance creation.

Requires: pip install gem-llm
"""

import logging
import random
import re
import warnings
from typing import Any, List, Optional, SupportsFloat, Tuple

import reasoning_gym as rg

import gem
from gem.core import Env
from gem.envs.reasoning_gym import ReasoningGymEnv
from gem.envs.registration import ENV_REGISTRY, EnvSpec
from gem.envs.registration import register as gem_register
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_boxed_answer

from spare.core.envs.env_adapter import EnvironmentAdapter, EnvInstance

logger = logging.getLogger(__name__)


class SpareReasoningGymEnv(ReasoningGymEnv):
    """ReasoningGymEnv that forwards extra kwargs to rg.create_dataset().

    GEM's ReasoningGymEnv.__init__ uses ``**_: Any`` which silently discards
    extra kwargs before they reach ``rg.create_dataset()``.  This subclass
    captures those kwargs and forwards them so that custom dataset parameters
    (e.g. ``min_num_vertices``, ``num_colors`` for graph_color) are respected.
    """

    def __init__(self, name: str, size: int = 500, seed: int = 42, **kwargs: Any) -> None:
        # Bypass ReasoningGymEnv.__init__ which discards kwargs via **_
        Env.__init__(self)
        self.idx = 0
        self.name = name
        self.size = size
        self.seed = seed
        self._extra_kwargs = kwargs
        self.ds = rg.create_dataset(name, size=size, seed=seed, **kwargs)
        self.ds_iter = iter(self.ds)
        self.reward_fn = self.ds.score_answer

    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        """Sample a question, forwarding extra kwargs on dataset re-creation."""
        # Call Env.reset (grandparent), not ReasoningGymEnv.reset
        Env.reset(self, seed)
        if seed is not None:
            data = random.choice(self.ds)
            if (self.idx + 1) % self.size == 0:
                self.ds = rg.create_dataset(
                    self.name, size=self.size, seed=self.seed + self.idx,
                    **self._extra_kwargs,
                )
        else:
            try:
                data = next(self.ds_iter)
            except StopIteration:
                self.ds = rg.create_dataset(
                    self.name, size=self.size, seed=self.seed + self.idx,
                    **self._extra_kwargs,
                )
                self.ds_iter = iter(self.ds)
                data = next(self.ds_iter)

        # reasoning_gym's scorer expects boxed or tagged final answers.
        question = data["question"] + "\n\nPut your final answer within \\boxed{}."
        self.idx += 1
        self.data = data
        return question, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Score the rollout against reasoning_gym's ``score_answer``.

        GEM's base ``ReasoningGymEnv.step`` extracts the answer with
        ``extract_last_boxed_answer`` (i.e. ``\\boxed{}``), but the rg eval
        prompt (and the reasoning_gym SYSTEM_PROMPTS convention) instructs the
        model to answer inside ``<answer>...</answer>`` tags — so the boxed
        extractor never matched and EVERY rg task scored 0.0 regardless of
        correctness. We extract the ``<answer>`` payload (the format the prompt
        asks for; e.g. the JSON coloring map for graph_color) and pass it to
        ``score_answer``. Falls back to ``\\boxed{}`` for the
        RG_USE_BOXED_PROMPT path.
        """
        matches = re.findall(
            r"<answer>\s*(.*?)\s*</answer>", action, re.DOTALL | re.IGNORECASE
        )
        if matches:
            clean_action = matches[-1].strip()
        else:
            clean_action = extract_last_boxed_answer(action)
        reward = self.reward_fn(answer=clean_action, entry=self.data)
        return TERMINAL_STATE, reward, True, True, {}

    def spawn(self, same_state: bool = False, **kwargs: Any) -> Env:
        merged_kwargs = {**self._extra_kwargs, **kwargs}
        if same_state:
            child = SpareReasoningGymEnv(
                name=self.name, size=self.size, seed=self.seed,
                **self._extra_kwargs,
            )
            child.set_state(self.get_state())
        else:
            child = SpareReasoningGymEnv(
                name=self.name, size=self.size, **merged_kwargs,
            )
            if child.seed == self.seed:
                warnings.warn(
                    "same_state is False but the seed is not changed, "
                    "which may lead to the same sequence of questions."
                )
        return child


# Register custom difficulty variants missing from gem-llm
# Sudoku: easy=4x4/10clues, medium=9x9/50clues(=Sudoku-v0), hard=9x9/30clues
_SPARE_CUSTOM_ENVS = {
    "game:Sudoku-v0-medium": (
        "gem.envs.game_env.sudoku:SudokuEnv",
        {"scale": 9, "clues": 50, "max_turns": 50},
    ),
    "rg:graph_color-easy": (
        "spare.core.envs.gem_adapter:SpareReasoningGymEnv",
        {"name": "graph_color", "seed": 42, "min_num_vertices": 10, "max_num_vertices": 10, "num_colors": 3},
    ),
    # The hard graph-color split is registered from the paper's A.3 settings.
    "eval:GPQA-Diamond": (
        "spare.core.eval.boxed_qa_env:BoxedQaEnv",
        {"dataset_name": "fingertap/GPQA-Diamond", "split": "test",
         "question_key": "question", "answer_key": "answer"},
    ),
    # MCQ JSON variant — Qwen3-4B-Inst-2507 model card answer format, used to
    # reproduce the official GPQA 62.0 (measured with JSON MCQ, not \boxed{}).
    "mcq:GPQA-Diamond": (
        "spare.core.eval.boxed_qa_env:McqJsonQaEnv",
        {"dataset_name": "fingertap/GPQA-Diamond", "split": "test",
         "question_key": "question", "answer_key": "answer"},
    ),
    # Official GPQA-Diamond: raw Idavidrein/gpqa + per-question A/B/C/D shuffle
    # + verbatim Qwen model-card MCQ instruction (the fingertap variant above
    # bakes a fixed lowercase option order, diverging from official → ~7pt low).
    "mcq:GPQA-Diamond-official": (
        "spare.core.eval.boxed_qa_env:GpqaOfficialMcqEnv",
        {"dataset_name": "Idavidrein/gpqa", "config_name": "gpqa_diamond",
         "split": "train", "shuffle_seed": 42},
    ),
    # GPQA using the verbatim OpenAI simple-evals prompt and extraction protocol.
    "mcq:GPQA-Diamond-scieval": (
        "spare.core.eval.boxed_qa_env:GpqaScievalMcqEnv",
        {"dataset_name": "Idavidrein/gpqa", "config_name": "gpqa_diamond",
         "split": "train", "shuffle_seed": 42},
    ),
    # LiveCodeBench v6 — Qwen3's OFFICIAL coding benchmark (4B-Inst-2507=35.1,
    # 30B-A3B-Inst-2507=43.2). LiveCodeBenchEnv wraps the OFFICIAL lcb_runner
    # harness (prompt + extraction + execution) — repo via LCB_OFFICIAL_DIR.
    "eval:LiveCodeBench-v6": (
        "spare.core.eval.livecodebench_env:LiveCodeBenchEnv",
        {"release_version": "release_v6"},
    ),
}

for _env_id, (_entry_point, _kwargs) in _SPARE_CUSTOM_ENVS.items():
    try:
        gem_register(_env_id, _entry_point, **_kwargs)
    except Exception:
        pass  # Already registered


def _register_rg_hard_split() -> int:
    """Register Reasoning-Gym's Appendix A.3 hard configuration."""
    from spare.core.eval.rg_a3_hard import A3_HARD_CONFIGS
    n = 0
    for name, params in A3_HARD_CONFIGS.items():
        env_id = f"rg:{name}-hard"
        if env_id in ENV_REGISTRY:
            continue
        try:
            gem_register(
                env_id, "spare.core.envs.gem_adapter:SpareReasoningGymEnv",
                name=name, seed=42, **params,
            )
            n += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("[RG] could not register %s at A.3 hard config: %s", env_id, e)
    logger.info("[RG] registered %d rg:<task>-hard envs (reasoning_gym paper A.3 hard configs)", n)
    return n


_register_rg_hard_split()

# Use no-replacement sampling and complete test execution for CodeContest.
# Direct assignment preserves the existing task ID and metric key.
ENV_REGISTRY["eval:CodeContest"] = EnvSpec(
    id="eval:CodeContest",
    entry_point="spare.core.eval.codecontest_env:SpareCodeContestEnv",
    kwargs={"dataset_name": "axon-rl/CodeContest", "split": "test"},
)


class GemEnvironmentAdapter(EnvironmentAdapter):
    """Adapter for GEM benchmark environments.

    GEM environments are already Gym-compatible, so this adapter
    is mostly pass-through with metadata tracking.

    Args:
        task_ids: List of GEM task IDs to make available.
                  If None, uses a default set.
    """

    def __init__(self, task_ids: Optional[List[str]] = None):
        self._task_ids = task_ids or []

    def list_environments(self) -> List[str]:
        return list(self._task_ids)

    def create_instance(
        self, env_id: str, difficulty: int = 0
    ) -> EnvInstance:
        """Create a GEM environment instance.

        Args:
            env_id: Full GEM task ID (e.g., "game:GuessTheNumber-v0-easy")
            difficulty: Not used for GEM (difficulty is baked into task ID)

        Returns:
            EnvInstance wrapping the GEM environment
        """
        env = gem.make(env_id)
        category = self.get_category(env_id)

        return EnvInstance(
            env=env,
            env_id=env_id,
            category=category,
            difficulty=difficulty,
            source="gem",
            metadata={"gem_task_id": env_id},
        )

    def get_difficulty_range(self, env_id: str) -> Tuple[int, int]:
        # GEM tasks have fixed difficulty (baked into the task ID)
        return (0, 0)

    def get_category(self, env_id: str) -> str:
        if ":" in env_id:
            return env_id.split(":")[0]
        return "unknown"
