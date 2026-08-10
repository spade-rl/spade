"""Prompt templates for game generation and self-judge."""

from .template import (
    GAME_GENERATION_SYSTEM_PROMPT,
    generate_game_generation_prompt,
    MULTITURN_GAME_GENERATION_SYSTEM_PROMPT,
    generate_multiturn_game_generation_prompt,
    TOOL_USE_GAME_GENERATION_SYSTEM_PROMPT,
    SELF_JUDGE_SYSTEM_PROMPT,
    SELF_JUDGE_PROMPT_TEMPLATE,
    format_trajectory_for_judge,
    generate_self_judge_prompt,
)

__all__ = [
    'GAME_GENERATION_SYSTEM_PROMPT',
    'generate_game_generation_prompt',
    'MULTITURN_GAME_GENERATION_SYSTEM_PROMPT',
    'generate_multiturn_game_generation_prompt',
    'TOOL_USE_GAME_GENERATION_SYSTEM_PROMPT',
    'SELF_JUDGE_SYSTEM_PROMPT',
    'SELF_JUDGE_PROMPT_TEMPLATE',
    'format_trajectory_for_judge',
    'generate_self_judge_prompt',
]
