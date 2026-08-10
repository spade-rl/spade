"""Tinker backend integration for SPARE."""

from spare._lazy import lazy_exports

_EXPORTS = {
    "TinkerModelAdapter": ("spare.tinker.model_adapter", "TinkerModelAdapter"),
    "get_game_policy": ("spare.tinker.rollout", "get_game_policy"),
    "get_learning_potentials": ("spare.tinker.rollout", "get_learning_potentials"),
    "spare_generate_rollout": ("spare.tinker.rollout", "spare_generate_rollout"),
    "spare_trajectory_to_tinker_trajectory": (
        "spare.tinker.trajectory_converter",
        "spare_trajectory_to_tinker_trajectory",
    ),
    "train_step": ("spare.tinker.train_step", "train_step"),
}

__all__ = list(_EXPORTS)
__getattr__ = lazy_exports(__name__, globals(), _EXPORTS)
