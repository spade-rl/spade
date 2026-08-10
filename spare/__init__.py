"""SPADE: Self-Play in Adaptive Synthetic Executable Environments (package name retains the historical 'spare' module name)."""

from spare.__about__ import __version__
from spare._lazy import lazy_exports

_EXPORTS = {
    "LearningPotential": ("spare.core.learning_potential", "LearningPotential"),
    "SyntheticGameEnv": ("spare.core.envs.synthetic_game_env", "SyntheticGameEnv"),
    "SyntheticGameGenerator": ("spare.core.game_generator", "SyntheticGameGenerator"),
    "core": ("spare.core", None),
    "make_synthetic_env": ("spare.core.envs.synthetic_game_env", "make_synthetic_env"),
    "tinker": ("spare.tinker", None),
}

__all__ = ["__version__", *_EXPORTS]
__getattr__ = lazy_exports(__name__, globals(), _EXPORTS)
