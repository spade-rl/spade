"""Environment wrapper for LLM-generated language games."""

import collections
import copy
import functools
import heapq
import importlib.util
import itertools
import json
import logging
import math
import operator
import os
import random
import re
import statistics
import string
import sys
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Names auto-injected into every generated game's exec namespace so that
# proposer code referencing common stdlib without `import` statements still
# loads. Maps the bare name the code might use to the actual object.
_COMMON_STDLIB_INJECTIONS: Dict[str, Any] = {
    # Whole modules
    "itertools": itertools,
    "math": math,
    "random": random,
    "re": re,
    "json": json,
    "collections": collections,
    "functools": functools,
    "string": string,
    "copy": copy,
    "heapq": heapq,
    "operator": operator,
    "statistics": statistics,
    "np": np,
    "numpy": np,
    # Frequent collection types proposer references unimported
    "deque": collections.deque,
    "defaultdict": collections.defaultdict,
    "Counter": collections.Counter,
    "OrderedDict": collections.OrderedDict,
    "namedtuple": collections.namedtuple,
    # Frequent itertools names
    "combinations": itertools.combinations,
    "permutations": itertools.permutations,
    "product": itertools.product,
    "chain": itertools.chain,
    "accumulate": itertools.accumulate,
    # Frequent functools names
    "reduce": functools.reduce,
    "lru_cache": functools.lru_cache,
    # Frequent copy/heapq names
    "deepcopy": copy.deepcopy,
    "heappush": heapq.heappush,
    "heappop": heapq.heappop,
    # Generated games may use common annotations without importing them.
    "Tuple": Tuple,
    "Dict": Dict,
    "List": List,
    "Optional": Optional,
    "Union": Union,
    "Any": Any,
    "Set": Set,
    "FrozenSet": FrozenSet,
    "Callable": Callable,
    "Iterable": Iterable,
    "Sequence": Sequence,
    "Mapping": Mapping,
}


def _inject_common_stdlib(namespace: Dict[str, Any]) -> None:
    """Pre-populate a module namespace with common stdlib names.

    Generated games sometimes use names like `itertools.combinations` or
    `deque(...)` without importing them; rather than rejecting an otherwise
    valid game, we make the names resolvable. Explicit imports in the game
    code still work (they just rebind the same object).
    """
    for name, obj in _COMMON_STDLIB_INJECTIONS.items():
        namespace.setdefault(name, obj)


class SyntheticGameEnv:
    """Wrapper for dynamically generated language game code."""

    def _sanitize_info_dict(self, info: Dict) -> Dict:
        """Remove string values from info before numeric aggregation."""
        if not isinstance(info, dict):
            return {}

        sanitized = {}
        for key, value in info.items():
            # Only keep numeric values and booleans
            if isinstance(value, (int, float, bool, type(None))):
                sanitized[key] = value
            # Skip string values which cause numpy.mean() to fail
            elif isinstance(value, str):
                continue  # Silently drop string values
            else:
                # For other types, try to convert to float
                try:
                    sanitized[key] = float(value)
                except (ValueError, TypeError):
                    continue  # Skip non-numeric values

        return sanitized

    def __init__(self, game_spec_file: str = None, game_code: str = None, max_steps: int = 1000, max_turns: int = None, use_conversation_history: bool = False, respect_game_max_turns: bool = False):
        """Initialize synthetic game environment.

        Args:
            game_spec_file: Path to game specification file (.py or .json)
            game_code: Python code string for the game
            max_steps: Maximum steps per episode (used as max_turns for the game class)
            max_turns: If provided, overrides max_steps for the game class
            use_conversation_history: If True, return full conversation history. If False, return only current observation.
            respect_game_max_turns: If True, use the game class's OWN designed max_turns
                (the proposer sets the pacing), capped at max_steps as the budget; the
                default (False) forces max_steps as the game's max_turns (legacy behavior).
        """
        if max_turns is not None:
            max_steps = max_turns
        self.max_steps = max_steps
        self.current_step = 0
        self.conversation_history = []
        self.use_conversation_history = use_conversation_history
        logger.info(f"Initializing SyntheticGameEnv with game_spec_file={game_spec_file}")

        if game_spec_file:
            # Load game from file with error handling
            try:
                if game_spec_file.endswith('.json'):
                    # Load spec and code from JSON
                    with open(game_spec_file, 'r') as f:
                        json.load(f)
                    code_file = game_spec_file.replace('.json', '.py')
                    if not os.path.exists(code_file):
                        raise FileNotFoundError(f"Python file not found: {code_file}")
                    with open(code_file, 'r') as f:
                        game_code = f.read()
                else:
                    # Load code directly from Python file
                    if not os.path.exists(game_spec_file):
                        raise FileNotFoundError(f"Game file not found: {game_spec_file}")
                    with open(game_spec_file, 'r') as f:
                        game_code = f.read()
            except FileNotFoundError as e:
                logger.error(f"File not found: {e}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {game_spec_file}: {e}")
                raise ValueError(f"Failed to parse JSON: {e}")
            except Exception as e:
                logger.error(f"Failed to load game from {game_spec_file}: {e}")
                raise RuntimeError(f"Failed to load game: {e}")

        # Validate game_code
        if not game_code:
            raise ValueError("No game code provided")

        # Load game dynamically
        self.game = self._load_game_from_code(game_code, max_steps=max_steps, respect_game_max_turns=respect_game_max_turns)

    def _load_game_from_code(self, code: str, max_steps: int = 1000, respect_game_max_turns: bool = False):
        """Dynamically load game from code string with robust validation.

        Raises exception if game is invalid - no fallback!
        """
        # Step 1: Create module and execute code
        spec = importlib.util.spec_from_loader("dynamic_game", loader=None)
        module = importlib.util.module_from_spec(spec)
        sys.modules["dynamic_game"] = module

        # Tolerate generated games that omit common standard-library imports.
        _inject_common_stdlib(module.__dict__)

        # Inject ToolUseBaseEnv so generated subclasses can reference it by
        # direct name or import it as ``tool_use_base_env``.
        try:
            from spare.core.envs import tool_use_base_env as _tu_mod
            module.__dict__["ToolUseBaseEnv"] = _tu_mod.ToolUseBaseEnv
            sys.modules["tool_use_base_env"] = _tu_mod
        except ImportError:
            pass

        try:
            exec(code, module.__dict__)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in generated game code: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to execute game code: {e}")

        # Step 2: Find the game class (should end with 'Env')
        # Skip base classes that were injected, not generated
        _BASE_CLASS_NAMES = {'Env', 'ToolUseBaseEnv'}
        game_class = None
        for name, obj in module.__dict__.items():
            if name.endswith('Env') and name not in _BASE_CLASS_NAMES and callable(obj):
                if hasattr(obj, '__init__'):
                    game_class = obj
                    break

        if game_class is None:
            raise ValueError("No valid game class found in code (expected class ending with 'Env')")

        # Step 3: Instantiate the game
        try:
            if respect_game_max_turns:
                # Use the game's OWN designed max_turns, capped at max_steps (the budget).
                # Instantiate with the game's default; fall back to forcing max_steps only
                # if the class requires max_turns as a positional/required arg.
                try:
                    game_instance = game_class()
                except TypeError:
                    game_instance = game_class(max_turns=max_steps)
                designed = getattr(game_instance, 'max_turns', None)
                if isinstance(designed, int) and designed > 0:
                    game_instance.max_turns = min(designed, max_steps)
                elif hasattr(game_instance, 'max_turns'):
                    game_instance.max_turns = max_steps
            else:
                try:
                    game_instance = game_class(max_turns=max_steps)
                except TypeError:
                    game_instance = game_class()
                    if hasattr(game_instance, 'max_turns'):
                        game_instance.max_turns = max_steps
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate game class: {e}")

        # Step 4: Validate required methods exist
        required_methods = ['reset', 'step']
        for method in required_methods:
            if not hasattr(game_instance, method):
                raise ValueError(f"Game class missing required method: {method}")

        # Step 5: ROBUST VALIDATION - Actually test the game works!
        self._validate_game_execution(game_instance)

        logger.info(f"Successfully loaded and validated game class: {game_class.__name__}")
        return game_instance

    def _validate_game_execution(self, game_instance):
        """Validate that game actually runs without errors.

        Tests:
        1. reset() returns valid (obs, info) tuple
        2. step() returns valid 5-tuple with correct types
        3. Game can handle a few test actions without crashing
        4. Rewards are in expected range
        """
        # Test 1: reset() works
        try:
            result = game_instance.reset(seed=42)
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            elif isinstance(result, str):
                obs, info = result, {}
            else:
                raise ValueError(f"reset() returned invalid type: {type(result)}")

            if not isinstance(obs, str) or len(obs) == 0:
                raise ValueError("reset() returned empty or non-string observation")

            if not isinstance(info, dict):
                raise ValueError(f"reset() info must be dict, got {type(info)}")

        except Exception as e:
            raise RuntimeError(f"Game reset() failed: {e}")

        # Test 2: step() works with test actions
        test_actions = [
            "\\boxed{1}",
            "\\boxed{test}",
            "\\boxed{42}",
        ]

        for i, action in enumerate(test_actions):
            try:
                result = game_instance.step(action)

                if not isinstance(result, tuple) or len(result) != 5:
                    raise ValueError(f"step() must return 5-tuple, got {type(result)} with length {len(result) if isinstance(result, tuple) else 'N/A'}")

                obs, reward, terminated, truncated, info = result

                # Validate types
                if not isinstance(obs, str):
                    raise ValueError(f"step() observation must be string, got {type(obs)}")
                if not isinstance(reward, (int, float)):
                    raise ValueError(f"step() reward must be numeric, got {type(reward)}")
                if not isinstance(terminated, bool):
                    raise ValueError(f"step() terminated must be bool, got {type(terminated)}")
                if not isinstance(truncated, bool):
                    raise ValueError(f"step() truncated must be bool, got {type(truncated)}")
                if not isinstance(info, dict):
                    raise ValueError(f"step() info must be dict, got {type(info)}")

                # Validate reward range
                if reward < -10 or reward > 10:
                    raise ValueError(f"step() reward {reward} outside expected range [-10, 10]")

                # If game ended, break
                if terminated or truncated:
                    break

            except Exception as e:
                raise RuntimeError(f"Game step() failed on action {i+1} '{action}': {e}")

        # Test 3: Reset again to ensure game is replayable
        try:
            game_instance.reset(seed=123)
        except Exception as e:
            raise RuntimeError(f"Game reset() failed on second call: {e}")

        logger.info("Game validation passed: reset(), step(), and replay all work correctly")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[str, Dict]:
        """Reset the environment with error handling."""
        self.current_step = 0
        self.conversation_history = []
        try:
            result = self.game.reset(seed=seed)
            # Ensure proper return format
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            elif isinstance(result, str):
                obs, info = result, {}
            else:
                logger.warning(f"Unexpected reset return format: {type(result)}")
                obs, info = str(result), {}

            # Filter out string values from info dict to prevent numpy.mean() errors
            info = self._sanitize_info_dict(info)

            # Start conversation history
            self.conversation_history.append(f"Game: {obs}")
            return obs, info
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            return "Error during reset. Starting new game.", {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict]:
        """Execute one step in the environment with robust error handling."""
        self.current_step += 1

        # Validate input
        if action is None:
            action = ""
        if not isinstance(action, str):
            action = str(action)
        # Safety cap against pathological inputs only — tool-use games submit
        # the full raw response to env.step, so keep it well above a real answer.
        if len(action) > 100000:
            action = action[:100000]

        # Add player action to conversation history (cleaned version)
        # Extract just the answer part if it's in boxed format
        # Try multiple patterns to handle different brace levels
        patterns = [
            r'\\boxed\{([^{}]+)\}',        # \boxed{28}
            r'\\boxed\{\{([^{}]+)\}\}',    # \boxed{{28}}
        ]

        answer_found = None
        for pattern in patterns:
            match = re.search(pattern, action)
            if match:
                answer_found = match.group(1).strip()
                break

        if answer_found:
            # Found boxed answer - store just the answer
            self.conversation_history.append(f"Player: {answer_found}")
        elif len(action) > 100:
            # Long response - truncate to avoid clutter
            self.conversation_history.append(f"Player: {action[:100]}...")
        elif action.strip() == "":
            self.conversation_history.append("Player: [No response]")
        else:
            # Short response - store as is
            self.conversation_history.append(f"Player: {action}")

        try:
            result = self.game.step(action)

            # Ensure proper 5-tuple return format
            if isinstance(result, tuple) and len(result) == 5:
                obs, reward, terminated, truncated, info = result
                # Validate types
                obs = str(obs) if obs is not None else "No observation"
                reward = float(reward) if reward is not None else 0.0
                terminated = bool(terminated) if terminated is not None else False
                truncated = bool(truncated) if truncated is not None else False
                info = info if isinstance(info, dict) else {}

                # Filter out string values from info dict to prevent numpy.mean() errors
                info = self._sanitize_info_dict(info)

                # Add game response to conversation history
                self.conversation_history.append(f"Game: {obs}")

                # Return either full history or just current observation based on setting
                if self.use_conversation_history:
                    final_obs = "\n".join(self.conversation_history)
                else:
                    final_obs = obs  # Just return the current game observation
                return final_obs, reward, terminated, truncated, info
            else:
                logger.warning(f"Unexpected step return format: {type(result)}, length: {len(result) if isinstance(result, tuple) else 'N/A'}")
                return "Invalid game response", -1.0, True, False, {}

        except Exception as e:
            logger.error(f"Error in step: {e}")
            # Return empty dict to avoid numpy errors - don't put strings in info
            return f"Error: {str(e)}", -1.0, True, False, {}

    def render(self) -> None:
        """Render the game state."""
        if hasattr(self.game, 'render'):
            self.game.render()

    def get_tools(self) -> list:
        """Delegate to underlying game's get_tools() if available.

        Returns OpenAI-format tool schemas for native tool calling.
        Returns empty list if game doesn't support tool-use.
        """
        if hasattr(self.game, 'get_tools'):
            return self.game.get_tools()
        return []

    def execute_tool(self, name: str, arguments: dict) -> dict:
        """Delegate to underlying game's execute_tool() if available.

        Args:
            name: Tool name
            arguments: Tool arguments dict

        Returns:
            {"output": str, "returncode": int}
        """
        if hasattr(self.game, 'execute_tool'):
            return self.game.execute_tool(name, arguments)
        return {"output": f"execute_tool not supported by {type(self.game).__name__}", "returncode": 1}

    def close(self):
        """Clean up resources."""
        if hasattr(self.game, 'close'):
            self.game.close()


def make_synthetic_env(game_spec_file: str, use_string_obs: bool = True, use_conversation_history: bool = False, **kwargs) -> SyntheticGameEnv:
    """
    Create a synthetic game environment.

    Args:
        game_spec_file: Path to the game specification file
        use_string_obs: Always True for LLM games (kept for compatibility)
        use_conversation_history: If True, return full conversation history. If False (default), return only current observation.
        **kwargs: Additional arguments passed to the environment (e.g., max_steps)

    Returns:
        SyntheticGameEnv instance
    """
    logger.debug(f"Creating synthetic environment for {game_spec_file}")
    return SyntheticGameEnv(game_spec_file=game_spec_file, use_conversation_history=use_conversation_history, **kwargs)

def criteria_throw_at_reset(game_spec_file: str, max_turns: int = 25,
                            seeds: tuple = (0, 1, 2)) -> bool:
    """Deterministic reset-solvability check for multi-turn tool_use games.

    Loads the game, reset(seed)s it across SEVERAL seeds, and evaluates EACH
    ``_message_criteria`` lambda on the initial ``_state``. Returns True (REJECT) iff a
    criterion RAISES (KeyError / TypeError / IndexError) at EVERY tested seed - i.e. the
    criterion can never even be evaluated REGARDLESS of the random data, so it is a
    genuine structural bug and that step is UNSOLVABLE. Both ``validate_game()`` and the
    LLM env-validator miss this (the exception is swallowed as a "Error calling <tool>"
    string inside the tool result).

    WHY MULTI-SEED: a single fixed seed gives FALSE positives - a criterion may throw on
    seed-0's particular random data yet be fine at the seed the game is actually PLAYED
    with. Single-seed rejection once starved a rollout-0 to 0 games and crashed the run.
    Requiring the throw at ALL tested seeds keeps only data-independent (real) bugs.

    Difficulty-NEUTRAL: hard-but-solvable criteria evaluate to False cleanly and pass.
    Pre-satisfied criteria (True at reset) are intentionally NOT flagged. Single-task /
    cognitive games (no ``_message_criteria``) return False. Any load/reset failure is
    skipped. Best-effort, never raises.
    """
    try:
        checked = 0
        threw = 0
        for seed in seeds:
            try:
                env = make_synthetic_env(game_spec_file, max_turns=max_turns)
                env.reset(seed=seed)
            except Exception:
                continue  # load/reset failures are caught by validate_game; skip this seed
            game = getattr(env, "game", env)
            crits = getattr(game, "_message_criteria", None)
            state = getattr(game, "_state", None)
            if not crits or state is None:
                return False  # not a criteria-gated multi-turn game -> never reject here
            checked += 1
            for c in crits:
                try:
                    c(state)
                except Exception:
                    threw += 1
                    break  # this seed has a throwing criterion
        # Reject ONLY if a criterion threw at EVERY seed we could evaluate (consistent,
        # data-independent breakage). Thrown-on-some-seeds-only = seed-sensitive, keep.
        return checked > 0 and threw == checked
    except Exception:
        return False
