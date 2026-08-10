"""GEM task registry for evaluation and fixed-env training.

Provides default task sets organized by category, per-task evaluation
configuration via YAML, and parsing utilities.

YAML config format:
    gem_eval:
      defaults:
        episodes: 8
        max_turns: 40
        temperature: 1.0
        max_tokens: 8192
        max_concurrent: 16

      tasks:
        - task_id: "game:Sudoku-v0-easy"
          episodes: 16
          max_turns: 20

        - task_id: "rg:graph_color-hard"
          max_turns: 60
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# Default GEM tasks organized by category
# Format: "category:TaskName-version-difficulty"
DEFAULT_GEM_TASKS: Dict[str, List[str]] = {
    "game": [
        "game:Mastermind-v0-hard",
        "game:Wordle-v0-hard",
        "game:Sokoban-v0-hard",
        "game:TowerofHanoi-v0-hard",
        "game:Crosswords-v0-easy",
    ],
    "code": [
        "code:HumanEval-v0",
        "code:MBPP-v0",
    ],
    "math": [
        "math:GSM8K-v0",
        "math:MATH-v0",
    ],
    "reasoning_gym": [
        "reasoning_gym:ARC-v0",
        "reasoning_gym:LogicGrid-v0",
    ],
}

# Flat list of all default tasks
ALL_DEFAULT_TASKS: List[str] = [
    task for tasks in DEFAULT_GEM_TASKS.values() for task in tasks
]


@dataclass
class GemEvalDefaults:
    """Default settings for GEM evaluation."""
    episodes: int = 8
    max_turns: int = 40
    temperature: float = 1.0
    max_tokens: int = 8192
    max_concurrent: int = 16


@dataclass
class GemTaskSpec:
    """Parsed GEM task specification with per-task config."""
    category: str
    task_id: str
    episodes: int = 8
    max_turns: int = 40
    temperature: float = 1.0
    max_tokens: int = 8192
    # `mcq_json` uses the Qwen3 model-card format for multiple-choice tasks.
    prompt_style: str = "default"
    # Distinguishes metrics when one task is evaluated with multiple prompts.
    metric_suffix: str = ""

    @property
    def short_name(self) -> str:
        """Task name without category prefix."""
        if ":" in self.task_id:
            return self.task_id.split(":", 1)[1]
        return self.task_id

    @property
    def metric_name(self) -> str:
        """Wandb metric base name (task_id with optional suffix)."""
        base = self.task_id.replace(":", "_").lower()
        return f"{base}{self.metric_suffix}"


def _extract_category(task_id: str) -> str:
    """Extract category from task_id (e.g., 'game' from 'game:Sudoku-v0-easy')."""
    if ":" in task_id:
        return task_id.split(":")[0]
    return "unknown"


# Golden Goose (arXiv:2601.22975, Sec. 3/Table 3) category rollups:
#   Math       = algebra + arithmetic + geometry + graphs
#   Algorithmic= algorithmic + code
#   Cognition  = arc + games + cognition
#   Logic      = logic + induction
RG_GOLDEN_GOOSE: Dict[str, str] = {
    "complex_arithmetic": "math", "intermediate_integration": "math", "polynomial_equations": "math", "polynomial_multiplication": "math", "simple_equations": "math", "simple_integration": "math", "basic_arithmetic": "math", "bitwise_arithmetic": "math", "calendar_arithmetic": "math", "chain_sum": "math", "count_bits": "math", "decimal_arithmetic": "math", "decimal_chain_sum": "math", "dice": "math", "fraction_simplification": "math", "gcd": "math", "gsm_symbolic": "math", "lcm": "math", "leg_counting": "math", "number_format": "math", "power_function": "math", "prime_factorization": "math", "products": "math", "time_intervals": "math", "advanced_geometry": "math", "simple_geometry": "math", "course_schedule": "math", "family_relationships": "math", "largest_island": "math", "quantum_lock": "math", "shortest_path": "math",
    "ab": "algorithmic", "base_conversion": "algorithmic", "binary_alternation": "algorithmic", "binary_matrix": "algorithmic", "caesar_cipher": "algorithmic", "count_primes": "algorithmic", "cryptarithm": "algorithmic", "game_of_life": "algorithmic", "game_of_life_halting": "algorithmic", "graph_color": "algorithmic", "group_anagrams": "algorithmic", "isomorphic_strings": "algorithmic", "jugs": "algorithmic", "letter_counting": "algorithmic", "letter_jumble": "algorithmic", "manipulate_matrix": "algorithmic", "number_filtering": "algorithmic", "number_sorting": "algorithmic", "palindrome_generation": "algorithmic", "palindrome_partitioning": "algorithmic", "pool_matrix": "algorithmic", "ransom_note": "algorithmic", "rotate_matrix": "algorithmic", "rotten_oranges": "algorithmic", "sentence_reordering": "algorithmic", "spell_backward": "algorithmic", "spiral_matrix": "algorithmic", "string_insertion": "algorithmic", "string_manipulation": "algorithmic", "string_splitting": "algorithmic", "string_synthesis": "algorithmic", "word_ladder": "algorithmic", "word_sequence_reversal": "algorithmic", "word_sorting": "algorithmic", "bf": "algorithmic", "codeio": "algorithmic",
    "arc_1d": "cognition", "arc_agi": "cognition", "rearc": "cognition", "color_cube_rotation": "cognition", "figlet_font": "cognition", "modulo_grid": "cognition", "needle_haystack": "cognition", "number_sequence": "cognition", "rectangle_count": "cognition", "rubiks_cube": "cognition", "countdown": "cognition", "emoji_mystery": "cognition", "futoshiki": "cognition", "knight_swap": "cognition", "mahjong_puzzle": "cognition", "maze": "cognition", "mini_sudoku": "cognition", "n_queens": "cognition", "puzzle24": "cognition", "rush_hour": "cognition", "sokoban": "cognition", "sudoku": "cognition", "tower_of_hanoi": "cognition", "tsumego": "cognition",
    "acre": "logic", "list_functions": "logic", "aiw": "logic", "circuit_logic": "logic", "knights_knaves": "logic", "propositional_logic": "logic", "self_reference": "logic", "syllogism": "logic", "zebra_puzzles": "logic",
}


def rg_golden_goose_category(task_id: str) -> Optional[str]:
    """For an 'rg:<dataset>-<difficulty>' id, return its Golden Goose bucket
    ('math'/'algorithmic'/'cognition'/'logic'), or None if not a known rg task."""
    if not task_id.startswith("rg:"):
        return None
    name = task_id.split(":", 1)[1].rsplit("-", 1)[0]
    return RG_GOLDEN_GOOSE.get(name)


def load_gem_eval_config(config_path: str) -> tuple[GemEvalDefaults, List[GemTaskSpec]]:
    """Load GEM evaluation config from a YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Tuple of (defaults, task_specs)
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"GEM eval config not found: {config_path}")

    with open(path) as f:
        content = f.read()
    # Expand environment variables (e.g., ${WORKSPACE_DIR})
    content = os.path.expandvars(content)
    raw = yaml.safe_load(content)

    gem_eval = raw.get("gem_eval", raw)

    # Parse defaults
    defaults_dict = gem_eval.get("defaults", {})
    defaults = GemEvalDefaults(
        episodes=defaults_dict.get("episodes", 8),
        max_turns=defaults_dict.get("max_turns", 40),
        temperature=defaults_dict.get("temperature", 1.0),
        max_tokens=defaults_dict.get("max_tokens", 8192),
        max_concurrent=defaults_dict.get("max_concurrent", 16),
    )

    # Parse tasks
    tasks_list = gem_eval.get("tasks", [])
    specs: List[GemTaskSpec] = []

    for task_entry in tasks_list:
        if isinstance(task_entry, str):
            task_id = task_entry
            overrides: Dict[str, Any] = {}
        elif isinstance(task_entry, dict):
            task_id = task_entry["task_id"]
            overrides = {k: v for k, v in task_entry.items() if k != "task_id"}
        else:
            logger.warning("[GEM-EVAL] Skipping invalid task entry: %s", task_entry)
            continue

        # Auto-derive metric_suffix from prompt_style if not explicitly set
        # so users don't have to specify both.
        _prompt_style = overrides.get("prompt_style", "default")
        _metric_suffix = overrides.get("metric_suffix", "")
        if not _metric_suffix and _prompt_style == "mcq_json":
            _metric_suffix = "_mcq"

        specs.append(GemTaskSpec(
            category=_extract_category(task_id),
            task_id=task_id,
            episodes=overrides.get("episodes", defaults.episodes),
            max_turns=overrides.get("max_turns", defaults.max_turns),
            temperature=overrides.get("temperature", defaults.temperature),
            max_tokens=overrides.get("max_tokens", defaults.max_tokens),
            prompt_style=_prompt_style,
            metric_suffix=_metric_suffix,
        ))

    logger.info(
        "[GEM-EVAL] Loaded config: %d tasks, defaults(episodes=%d, max_turns=%d)",
        len(specs), defaults.episodes, defaults.max_turns,
    )

    return defaults, specs
