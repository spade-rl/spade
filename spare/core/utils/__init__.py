"""Public utility exports grouped behind lazy module boundaries."""

from spare._lazy import lazy_exports

_EXPORTS = {
    "LanguageGameReward": ("spare.core.utils.parsing", "LanguageGameReward"),
    "assign_env_rewards": ("spare.core.utils.env_rewards", "assign_env_rewards"),
    "recompute_delayed_env_rewards": (
        "spare.core.utils.delayed_env_rewards",
        "recompute_delayed_env_rewards",
    ),
    "recompute_delayed_env_rewards_micro_lp": (
        "spare.core.utils.delayed_env_rewards",
        "recompute_delayed_env_rewards_micro_lp",
    ),
    "recompute_delayed_env_rewards_regret": (
        "spare.core.utils.delayed_env_rewards",
        "recompute_delayed_env_rewards_regret",
    ),
    "recompute_delayed_env_rewards_blend": (
        "spare.core.utils.delayed_env_rewards",
        "recompute_delayed_env_rewards_blend",
    ),
    "assign_env_rewards_regret": (
        "spare.core.utils.env_rewards",
        "assign_env_rewards_regret",
    ),
    "assign_trajectory_weights": (
        "spare.core.utils.trajectory_build",
        "assign_trajectory_weights",
    ),
    "build_actor_trajectory": (
        "spare.core.utils.trajectory_build",
        "build_actor_trajectory",
    ),
    "build_env_trajectory": (
        "spare.core.utils.trajectory_build",
        "build_env_trajectory",
    ),
    "cleanup_old_games": ("spare.core.utils.game_files", "cleanup_old_games"),
    "compute_env_reward_scale": (
        "spare.core.utils.rewards",
        "compute_env_reward_scale",
    ),
    "compute_format_reward": ("spare.core.utils.rewards", "compute_format_reward"),
    "compute_returns": ("spare.core.utils.rewards", "compute_returns"),
    "episode_reward": ("spare.core.utils.rewards", "episode_reward"),
    "extract_boxed_answer": ("spare.core.utils.parsing", "extract_boxed_answer"),
    "extract_game_code": ("spare.core.utils.parsing", "extract_game_code"),
    "format_error_response": ("spare.core.utils.parsing", "format_error_response"),
    "get_token_delta": ("spare.core.utils.token_utils", "get_token_delta"),
    "normalize_rewards_per_game": (
        "spare.core.utils.trajectory_build",
        "normalize_rewards_per_game",
    ),
    "parse_action": ("spare.core.utils.parsing", "parse_action"),
    "save_game_file": ("spare.core.utils.game_files", "save_game_file"),
    "save_rejected_game": ("spare.core.utils.game_files", "save_rejected_game"),
    "upsample_trajectories": (
        "spare.core.utils.trajectory_build",
        "upsample_trajectories",
    ),
    "validate_boxed_format": ("spare.core.utils.parsing", "validate_boxed_format"),
    "validate_game": ("spare.core.utils.game_files", "validate_game"),
}

__all__ = list(_EXPORTS)
__getattr__ = lazy_exports(__name__, globals(), _EXPORTS)
