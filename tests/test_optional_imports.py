"""Regression tests for optional dependency boundaries."""

import subprocess
import sys


def test_eval_submodule_does_not_import_gem() -> None:
    code = "import sys; import spare.core.eval.tau2_tasks; assert 'gem' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_slime_namespace_does_not_load_backend() -> None:
    code = (
        "import sys; import spare.slime; " "assert 'spare.slime.model_adapter' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_top_level_package_is_lightweight() -> None:
    code = (
        "import sys; import spare; "
        "assert 'spare.core.orchestrator' not in sys.modules; "
        "assert 'spare.core.game_generator' not in sys.modules; "
        "assert spare.LearningPotential.__name__ == 'LearningPotential'; "
        "assert 'spare.core.game_generator' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_core_exports_remain_available() -> None:
    code = (
        "from spare.core import LearningPotential, SpareConfig; "
        "assert LearningPotential.__name__ == 'LearningPotential'; "
        "assert SpareConfig.__name__ == 'SpareConfig'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_token_utility_does_not_load_rollout_utilities() -> None:
    code = (
        "import sys; from spare.core.utils import get_token_delta; "
        "assert callable(get_token_delta); "
        "assert 'spare.core.utils.game_utils' not in sys.modules; "
        "assert 'spare.core.utils.env_rewards' not in sys.modules; "
        "assert 'spare.core.utils.delayed_env_rewards' not in sys.modules; "
        "assert 'spare.core.utils.trajectory_build' not in sys.modules; "
        "assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
