"""RLVE environment adapter for fixed-env training.

Wraps RLVE's single-turn (generator/scorer) environments as Gym-compatible
multi-turn environments. Uses RLVE's ParameterController to produce correct
parameter dicts for each environment at each difficulty level.

Difficulty is an integer starting at 0. To get parameters for difficulty d,
we create a fresh ParameterController and call update() d times (matching
RLVE's generate_problem() logic).

RLVE is vendored in-tree at spare/external/rlve/ (Gym/ only, no benchmarks).
Source: https://github.com/Zhiyuan-Zeng/RLVE
Import path: spare.external.rlve.Gym.environments.identifier2environment

RLVE VerifiableEnvironment API:
  - generator(seed: int, parameter: dict) -> bool  # generates problem
  - prompt_generator() -> str                       # returns problem text
  - scorer(output: str) -> float                    # scores answer in [-1, +1]
  - verifier(output: str) -> dict                   # {reward, accuracy, format_score}
  - parameter["reference_answer"] / parameter["gold_answer"]  # ground truth
  - passing_reward_threshold: float                 # typically 1.0
"""

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from spare.core.envs.env_adapter import EnvironmentAdapter, EnvInstance
from spare.core.utils.game_utils import extract_boxed_answer

logger = logging.getLogger(__name__)

# Add RLVE to path if needed
_RLVE_ROOT = Path(__file__).resolve().parents[2] / "external" / "rlve"
if _RLVE_ROOT.exists() and str(_RLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RLVE_ROOT))


def _get_rlve_environments() -> Dict[str, Any]:
    """Import and return RLVE's identifier2environment mapping."""
    try:
        from Gym.environments import identifier2environment  # type: ignore[import-untyped]
        return identifier2environment
    except ImportError as e:
        logger.error(
            "Failed to import RLVE. Expected vendored copy at %s",
            _RLVE_ROOT,
        )
        raise ImportError(
            "RLVE not found at spare/external/rlve/. Re-vendor Gym/ from "
            "https://github.com/Zhiyuan-Zeng/RLVE"
        ) from e


def _get_rlve_controllers() -> Dict[str, Any]:
    """Import and return RLVE's identifier2controller mapping."""
    try:
        from Gym.parameter_controllers import identifier2controller  # type: ignore[import-untyped]
        return identifier2controller
    except ImportError as e:
        logger.error(
            "Failed to import RLVE parameter controllers from %s",
            _RLVE_ROOT,
        )
        raise ImportError(
            "RLVE parameter controllers not found at spare/external/rlve/Gym/."
        ) from e


def _get_parameter_for_difficulty(controller_class: Any, difficulty: int) -> Dict:
    """Get a random parameter dict for a given difficulty level.

    Creates a fresh ParameterController, calls update() `difficulty` times
    to advance to the target level, then picks a random parameter from the
    controller's parameter list. This matches RLVE's generate_problem() logic.

    Args:
        controller_class: RLVE ParameterController class
        difficulty: Target difficulty level (0-indexed)

    Returns:
        Parameter dict to pass to env.generator()
    """
    controller = controller_class()
    for _ in range(difficulty):
        controller.update()
    param_list = controller.get_parameter_list()
    return random.choice(param_list)


class RlveGymWrapper:
    """Wraps a single-turn RLVE environment as a Gym-compatible env.

    Single-turn: reset() generates a problem, step() scores the answer
    and terminates.

    Args:
        rlve_env_class: RLVE environment class from identifier2environment
        parameter: Parameter dict to pass to generator()
        difficulty: Difficulty level (for metadata only)
        seed: Optional fixed generator seed. When set, reset() without an
            explicit seed reproduces the same problem — used by
            create_instances_same_problem() so a GRPO group shares one problem.
    """

    def __init__(
        self,
        rlve_env_class: Any,
        parameter: Dict,
        difficulty: int = 0,
        seed: Optional[int] = None,
    ):
        self._env_class = rlve_env_class
        self._parameter = parameter
        self._difficulty = difficulty
        self._fixed_seed = seed
        self._env_instance = None
        self._current_answer = None

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is None:
            seed = (
                self._fixed_seed
                if self._fixed_seed is not None
                else random.randint(0, 2**31 - 1)
            )

        self._env_instance = self._env_class()
        success = self._env_instance.generator(
            seed=seed, parameter=self._parameter
        )
        if not success:
            raise RuntimeError(
                f"RLVE generator failed for {self._env_class.__name__} "
                f"with parameter={self._parameter}, seed={seed}"
            )

        problem_text = self._env_instance.prompt_generator()
        params = self._env_instance.parameter or {}
        self._current_answer = params.get(
            "reference_answer", params.get("gold_answer")
        )

        return problem_text, {
            "difficulty": self._difficulty,
            "has_answer": self._current_answer is not None,
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._env_instance is None:
            return "No problem generated. Call reset() first.", 0.0, True, False, {}

        # RLVE verifier expects <answer>...</answer> tags.
        # Convert \boxed{} to <answer> tags so SPARE can use \boxed{} uniformly.
        boxed = extract_boxed_answer(action)
        if boxed is not None:
            action = action + f"\n<answer>{boxed}</answer>"

        try:
            result = self._env_instance.verifier(action)
            correct = bool(result["accuracy"])
        except Exception as e:
            logger.warning("RLVE verifier failed: %s", e)
            correct = False

        # Binary reward: 1.0 if fully correct, 0.0 otherwise
        reward = 1.0 if correct else 0.0

        obs = "Correct!" if correct else f"Incorrect. The answer was: {self._current_answer}"

        return obs, reward, True, False, {
            "correct": correct,
            "difficulty": self._difficulty,
            "expected_answer": str(self._current_answer) if self._current_answer else None,
        }

    def close(self) -> None:
        self._env_instance = None
        self._current_answer = None


class RlveEnvironmentAdapter(EnvironmentAdapter):
    """Adapter for RLVE environments using ParameterController.

    Filters to environments that have both an env class in
    identifier2environment AND a controller in identifier2controller.
    Uses the controller to produce correct parameter dicts at each
    difficulty level, matching RLVE's generate_problem() logic.

    Args:
        env_names: List of RLVE environment names. Use ["*"] or None for all.
        max_difficulty: Maximum difficulty level (default 20)
    """

    def __init__(
        self,
        env_names: Optional[List[str]] = None,
        max_difficulty: int = 20,
    ):
        self._rlve_envs = _get_rlve_environments()
        self._rlve_controllers = _get_rlve_controllers()
        self._max_difficulty = max_difficulty

        if env_names is None or env_names == ["*"]:
            candidates = list(self._rlve_envs.keys())
        else:
            candidates = []
            for name in env_names:
                if name in self._rlve_envs:
                    candidates.append(name)
                else:
                    logger.warning(
                        "RLVE environment '%s' not found. Available: %s",
                        name, sorted(self._rlve_envs.keys())[:10],
                    )

        # Filter to envs that have both an env class and a controller
        self._env_names: List[str] = []
        num_no_controller = 0

        for name in candidates:
            if name in self._rlve_controllers:
                self._env_names.append(name)
            else:
                num_no_controller += 1
                logger.debug(
                    "RLVE env '%s' has no ParameterController, skipping", name
                )

        logger.info(
            "[RLVE] %d environments available (%d skipped: no controller)",
            len(self._env_names), num_no_controller,
        )

    def list_environments(self) -> List[str]:
        return list(self._env_names)

    def create_instance(self, env_id: str, difficulty: int = 0) -> EnvInstance:
        """Create an RLVE environment instance at the given difficulty.

        Uses the env's ParameterController to produce the correct parameter
        dict for the requested difficulty level.

        Args:
            env_id: RLVE environment name
            difficulty: Difficulty level (0-indexed, clamped to [0, max_difficulty])
        """
        if env_id not in self._rlve_envs:
            raise ValueError(
                f"Unknown RLVE environment: {env_id}. "
                f"Available: {sorted(self._rlve_envs.keys())[:10]}..."
            )

        if env_id not in self._rlve_controllers:
            raise ValueError(
                f"No ParameterController for RLVE environment: {env_id}"
            )

        # Clamp difficulty to valid range
        effective_difficulty = max(0, min(difficulty, self._max_difficulty))

        # Get parameter dict from controller at the target difficulty
        controller_class = self._rlve_controllers[env_id]
        parameter = _get_parameter_for_difficulty(controller_class, effective_difficulty)

        env_class = self._rlve_envs[env_id]
        env = RlveGymWrapper(env_class, parameter=parameter, difficulty=effective_difficulty)

        return EnvInstance(
            env=env,
            env_id=f"rlve:{env_id}",
            category="rlve",
            difficulty=effective_difficulty,
            source="rlve",
            metadata={
                "rlve_name": env_id,
                "difficulty": effective_difficulty,
                "parameter": parameter,
            },
        )

    def create_instances_same_problem(
        self, env_id: str, difficulty: int = 0, n: int = 1
    ) -> List[EnvInstance]:
        """Create n instances that share ONE generated problem.

        Draws (parameter, seed) once and validates the generator on it, then
        builds n wrappers with the same pair — every reset() reproduces the
        identical problem, so the n plays form a per-prompt GRPO group
        (RLVE's n-samples-per-prompt semantics).
        """
        if env_id not in self._rlve_envs:
            raise ValueError(
                f"Unknown RLVE environment: {env_id}. "
                f"Available: {sorted(self._rlve_envs.keys())[:10]}..."
            )
        if env_id not in self._rlve_controllers:
            raise ValueError(
                f"No ParameterController for RLVE environment: {env_id}"
            )

        effective_difficulty = max(0, min(difficulty, self._max_difficulty))
        controller_class = self._rlve_controllers[env_id]
        env_class = self._rlve_envs[env_id]

        # Validate (parameter, seed) once so a generator failure doesn't kill
        # all n plays of the group. Generators are cheap pure Python.
        parameter = None
        seed = None
        max_attempts = 10
        for _ in range(max_attempts):
            cand_parameter = _get_parameter_for_difficulty(
                controller_class, effective_difficulty
            )
            cand_seed = random.randint(0, 2**31 - 1)
            probe = env_class()
            if probe.generator(seed=cand_seed, parameter=cand_parameter):
                parameter, seed = cand_parameter, cand_seed
                break
        if parameter is None:
            raise RuntimeError(
                f"RLVE generator failed {max_attempts} times for {env_id} "
                f"at difficulty={effective_difficulty}"
            )

        instances = []
        for _ in range(n):
            env = RlveGymWrapper(
                env_class,
                parameter=parameter,
                difficulty=effective_difficulty,
                seed=seed,
            )
            instances.append(
                EnvInstance(
                    env=env,
                    env_id=f"rlve:{env_id}",
                    category="rlve",
                    difficulty=effective_difficulty,
                    source="rlve",
                    metadata={
                        "rlve_name": env_id,
                        "difficulty": effective_difficulty,
                        "parameter": parameter,
                        "problem_seed": seed,
                    },
                )
            )
        return instances

    def get_difficulty_range(self, env_id: str) -> Tuple[int, int]:
        return (0, self._max_difficulty)

    def get_category(self, env_id: str) -> str:
        return "rlve"
