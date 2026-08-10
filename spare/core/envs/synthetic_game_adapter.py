"""Synthetic game adapter for fixed-env training.

Wraps .py game files from a directory as an EnvironmentAdapter, allowing
pre-generated synthetic games to be used in fixed-env mode.
"""

import logging
from pathlib import Path
from typing import List, Set, Tuple

from spare.core.envs.env_adapter import ENV_CALL_TIMEOUT_SEC, EnvironmentAdapter, EnvInstance, _run_with_alarm
from spare.core.envs.synthetic_game_env import make_synthetic_env

logger = logging.getLogger(__name__)


class SyntheticGameAdapter(EnvironmentAdapter):
    """Adapter for pre-generated synthetic game .py files.

    Scans a directory for game_*.py files and wraps each as an environment.
    Synthetic games have no difficulty parameter (fixed at 1).

    Maintains an in-memory blacklist of games that hang or error during
    loading. Blacklisted games are excluded from list_environments() and
    rejected immediately by create_instance().

    Args:
        games_dir: Directory containing game_*.py files
    """

    def __init__(self, games_dir: str):
        self._games_dir = Path(games_dir)
        if not self._games_dir.is_dir():
            raise ValueError(f"Games directory does not exist: {games_dir}")

        self._game_files = sorted(self._games_dir.glob("game_*.py"))
        if not self._game_files:
            raise ValueError(f"No game_*.py files found in {games_dir}")

        # Use absolute path strings as env IDs
        self._env_ids = [str(f) for f in self._game_files]
        self._blacklisted: Set[str] = set()
        logger.info(
            "[SYNTHETIC] Loaded %d game files from %s",
            len(self._env_ids), games_dir,
        )

    def list_environments(self) -> List[str]:
        return [eid for eid in self._env_ids if eid not in self._blacklisted]

    def blacklist(self, env_id: str) -> None:
        """Blacklist a game so it is never sampled again this run."""
        if env_id not in self._blacklisted:
            self._blacklisted.add(env_id)
            logger.warning(
                "[SYNTHETIC] Blacklisted %s (total blacklisted: %d / %d)",
                Path(env_id).name, len(self._blacklisted), len(self._env_ids),
            )

    def create_instance(self, env_id: str, difficulty: int = 1) -> EnvInstance:
        """Create a synthetic game environment instance.

        Args:
            env_id: Absolute path to the game .py file
            difficulty: Ignored (synthetic games have no difficulty parameter)

        Returns:
            EnvInstance wrapping the synthetic game

        Raises:
            RuntimeError: If the game is blacklisted or fails to load
        """
        if env_id in self._blacklisted:
            raise RuntimeError(f"Game is blacklisted: {Path(env_id).name}")

        try:
            env = _run_with_alarm(lambda: make_synthetic_env(env_id), ENV_CALL_TIMEOUT_SEC)
        except Exception:
            self.blacklist(env_id)
            raise

        game_name = Path(env_id).stem

        return EnvInstance(
            env=env,
            env_id=env_id,
            category="synthetic",
            difficulty=1,
            source="synthetic",
            metadata={"game_file": env_id, "game_name": game_name},
        )

    def get_difficulty_range(self, env_id: str) -> Tuple[int, int]:
        return (1, 1)

    def get_category(self, env_id: str) -> str:
        return "synthetic"
