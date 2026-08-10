"""Compatibility facade re-exporting the rollout utility helpers.

The implementations live in focused sibling modules — ``game_files``,
``parsing``, ``rewards``, ``token_utils``, ``trajectory_build``,
``env_rewards`` and ``delayed_env_rewards``. This module keeps the historical
``spare.core.utils.game_utils`` import path working unchanged.
"""

from spare.core.utils.delayed_env_rewards import (
    recompute_delayed_env_rewards,
    recompute_delayed_env_rewards_blend,
    recompute_delayed_env_rewards_micro_lp,
    recompute_delayed_env_rewards_regret,
)
from spare.core.utils.env_rewards import (
    assign_env_rewards,
    assign_env_rewards_regret,
)
from spare.core.utils.game_files import (
    cleanup_old_games,
    save_game_file,
    save_rejected_game,
    validate_game,
)
from spare.core.utils.parsing import (
    LanguageGameReward,
    extract_boxed_answer,
    extract_command,
    extract_game_code,
    extract_tool_call,
    format_error_response,
    parse_action,
    repair_fstring_braces,
    validate_boxed_format,
)
from spare.core.utils.rewards import (
    # Not in __all__, but re-exported because it used to be defined here.
    STEP_TIMEOUT_PENALTY as STEP_TIMEOUT_PENALTY,
    compute_env_reward_scale,
    compute_format_reward,
    compute_returns,
    compute_variance_reward,
    episode_reward,
    plateau_reward,
)
from spare.core.utils.token_utils import get_token_delta
from spare.core.utils.trajectory_build import (
    assign_trajectory_weights,
    build_actor_trajectory,
    build_env_trajectory,
    normalize_rewards_per_game,
    upsample_trajectories,
)

__all__ = [
    "LanguageGameReward",
    "assign_env_rewards",
    "assign_env_rewards_regret",
    "assign_trajectory_weights",
    "build_actor_trajectory",
    "build_env_trajectory",
    "cleanup_old_games",
    "compute_env_reward_scale",
    "compute_format_reward",
    "compute_returns",
    "compute_variance_reward",
    "episode_reward",
    "extract_boxed_answer",
    "extract_command",
    "extract_game_code",
    "extract_tool_call",
    "format_error_response",
    "get_token_delta",
    "normalize_rewards_per_game",
    "parse_action",
    "plateau_reward",
    "recompute_delayed_env_rewards",
    "recompute_delayed_env_rewards_blend",
    "recompute_delayed_env_rewards_micro_lp",
    "recompute_delayed_env_rewards_regret",
    "repair_fstring_braces",
    "save_game_file",
    "save_rejected_game",
    "upsample_trajectories",
    "validate_boxed_format",
    "validate_game",
]
