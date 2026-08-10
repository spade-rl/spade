"""Environment adapter interface for fixed-env training.

Provides a uniform interface for creating Gym-compatible environment instances
from different sources (GEM, RLVE, etc.).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


class GymEnv(Protocol):
    """Protocol for Gym-compatible environments."""

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        ...

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        ...

    def close(self) -> None:
        ...


ENV_CALL_TIMEOUT_SEC = 30


def _run_with_alarm(fn, timeout_sec: int = ENV_CALL_TIMEOUT_SEC):
    """Run a synchronous function with SIGALRM timeout.

    SIGALRM interrupts any Python code (including infinite loops) when
    called from the main thread. Falls back to no timeout in non-main
    threads (e.g., when called from tests).
    """
    import signal

    def _handler(signum, frame):
        raise TimeoutError(f"env call timed out after {timeout_sec}s")

    try:
        old = signal.signal(signal.SIGALRM, _handler)
    except (ValueError, OSError):
        # Not main thread — run without timeout
        return fn()

    signal.alarm(timeout_sec)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@dataclass
class EnvInstance:
    """A concrete environment instance with metadata.

    Wraps a Gym-compatible environment with identification and
    difficulty information for the difficulty controller.

    All env calls (reset, step) are protected by SIGALRM timeout
    to prevent hanging on games with infinite loops.
    """
    env: GymEnv
    env_id: str          # Unique identifier (e.g., "gem:game:Sudoku-v0-easy")
    category: str        # Category (e.g., "game", "code", "math")
    difficulty: int = 0  # Current difficulty level (for parameterized envs)
    source: str = ""     # Source adapter name (e.g., "gem", "rlve")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        return _run_with_alarm(lambda: self.env.reset(seed=seed))

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        return _run_with_alarm(lambda: self.env.step(action))

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()


class EnvironmentAdapter(ABC):
    """Abstract interface for creating environment instances from a source.

    Each adapter knows how to list available environments from its source
    and create instances at specified difficulty levels.
    """

    @abstractmethod
    def list_environments(self) -> List[str]:
        """List available environment IDs from this source.

        Returns:
            List of environment ID strings
        """
        ...

    @abstractmethod
    def create_instance(
        self, env_id: str, difficulty: int = 0
    ) -> EnvInstance:
        """Create a concrete environment instance.

        Args:
            env_id: Environment identifier
            difficulty: Difficulty level (interpretation depends on adapter)

        Returns:
            EnvInstance wrapping the created environment
        """
        ...

    def create_instances_same_problem(
        self, env_id: str, difficulty: int = 0, n: int = 1
    ) -> List[EnvInstance]:
        """Create n instances that all present the SAME problem.

        Used for per-prompt GRPO grouping: all n plays of the group score
        one shared problem, matching RLVE's n-samples-per-prompt semantics.

        The base implementation cannot guarantee a shared problem — it falls
        back to n independent instances — so adapters that randomize the
        problem per instance (e.g., RLVE) must override this.

        Args:
            env_id: Environment identifier
            difficulty: Difficulty level
            n: Number of instances sharing the problem

        Returns:
            List of n EnvInstance objects
        """
        return [self.create_instance(env_id, difficulty=difficulty) for _ in range(n)]

    @abstractmethod
    def get_difficulty_range(self, env_id: str) -> Tuple[int, int]:
        """Get the valid difficulty range for an environment.

        Args:
            env_id: Environment identifier

        Returns:
            Tuple of (min_difficulty, max_difficulty)
        """
        ...

    @abstractmethod
    def get_category(self, env_id: str) -> str:
        """Get the category for an environment.

        Args:
            env_id: Environment identifier

        Returns:
            Category string (e.g., "game", "math", "code")
        """
        ...
