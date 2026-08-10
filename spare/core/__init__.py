"""Framework-independent SPARE components."""

from spare._lazy import lazy_exports

_EXPORTS = {
    "EMA": ("spare.core.learning_potential", "EMA"),
    "ExploitabilityBasedPotential": (
        "spare.core.learning_potential",
        "ExploitabilityBasedPotential",
    ),
    "GameBaselineTracker": ("spare.core.learning_potential", "GameBaselineTracker"),
    "LearningPotential": ("spare.core.learning_potential", "LearningPotential"),
    "ModelAdapter": ("spare.core.model_adapter", "ModelAdapter"),
    "MultiAgentLearningPotential": (
        "spare.core.learning_potential",
        "MultiAgentLearningPotential",
    ),
    "SpareConfig": ("spare.core.types", "SpareConfig"),
    "SpareOrchestrator": ("spare.core.orchestrator", "SpareOrchestrator"),
    "SyntheticGameEnv": ("spare.core.envs.synthetic_game_env", "SyntheticGameEnv"),
    "SyntheticGameGenerator": ("spare.core.game_generator", "SyntheticGameGenerator"),
    "calculate_game_progress": (
        "spare.core.learning_potential",
        "calculate_game_progress",
    ),
    "make_synthetic_env": ("spare.core.envs.synthetic_game_env", "make_synthetic_env"),
}

__all__ = list(_EXPORTS)
__getattr__ = lazy_exports(__name__, globals(), _EXPORTS)
