"""Game lifecycle policy for SPARE training."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class GameMetadata:
    """Metadata for a tracked game."""
    path: Path
    skill: str
    difficulty: str
    created_at: float

    reward_history: List[float] = field(default_factory=list)
    last_played_at: Optional[float] = None

    # External GEM/RLVE environments are immutable.
    immutable: bool = False

    @property
    def play_count(self) -> int:
        """Number of times this game was played."""
        return len(self.reward_history)

    @property
    def mean_reward(self) -> float:
        """Average reward across all plays."""
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)

    @property
    def min_reward(self) -> float:
        """Minimum reward across all plays."""
        if not self.reward_history:
            return 0.0
        return min(self.reward_history)

    @property
    def max_reward(self) -> float:
        """Maximum reward across all plays."""
        if not self.reward_history:
            return 0.0
        return max(self.reward_history)

    @property
    def latest_reward(self) -> float:
        """Most recent play reward."""
        if not self.reward_history:
            return 0.0
        return self.reward_history[-1]


class GamePolicy:
    """Manages game lifecycle and tracks performance.

    Responsibilities:
    - Register new games when generated
    - Record play results and track reward history
    - Delete games that are too easy (>upper) or too hard (<lower)
    - Cache learning potential (rho_t) from learner

    Usage:
        policy = GamePolicy(actor_id=0, games_dir=Path("./games"))

        # When generating a new game
        policy.register_game(game_path, skill="PatternRecognition", difficulty="hard")

        # After playing a game
        should_delete = policy.record_play(game_path, mean_reward=0.85)
        if should_delete:
            # Game was deleted (too easy or too hard)
            pass

        # Get cached rho_t for environment reward
        env_reward = policy.get_env_reward()
    """

    def __init__(
        self,
        env_lower_threshold: float = 0.1,
        env_upper_threshold: float = 0.9,
    ):
        self.env_lower_threshold = env_lower_threshold
        self.env_upper_threshold = env_upper_threshold

        self.games: Dict[str, GameMetadata] = {}

        self.total_deletions: int = 0
        self.deletions_too_easy: int = 0
        self.deletions_too_hard: int = 0

    def register_game(
        self,
        game_file: str,
        skill: str,
        difficulty: str,
        immutable: bool = False,
    ) -> GameMetadata:
        """Register a new game for tracking.

        Args:
            game_file: Path to the game file
            skill: Cognitive skill category
            difficulty: Difficulty level
            immutable: If True, prevents this game from being deleted
                       (used for external GEM/RLVE environments)
        """
        metadata = GameMetadata(
            path=game_file,
            skill=skill,
            difficulty=difficulty,
            created_at=time.time(),
            immutable=immutable,
        )
        self.games[str(game_file)] = metadata
        logging.info(
            f"[GamePolicy] Registered: {str(game_file)} "
            f"(skill={skill}, difficulty={difficulty})"
        )
        return metadata

    def record_play(self, game_file: str, mean_reward: float) -> bool:
        """Record a play result. Returns True if game was deleted."""
        if game_file not in self.games:
            self.games[game_file] = GameMetadata(
                path=game_file,
                skill="unknown",
                difficulty="unknown",
                created_at=time.time(),
            )

        metadata = self.games[game_file]
        metadata.reward_history.append(mean_reward)
        metadata.last_played_at = time.time()

        if metadata.immutable:
            pass
        elif mean_reward > self.env_upper_threshold:
            self._delete_game(game_file, "too_easy", mean_reward)
            self.deletions_too_easy += 1
            return True
        elif mean_reward < self.env_lower_threshold:
            self._delete_game(game_file, "too_hard", mean_reward)
            self.deletions_too_hard += 1
            return True

        logging.debug(
            f"[GamePolicy] Played: {game_file} "
            f"(reward={mean_reward:.3f}, plays={metadata.play_count}, "
            f"mean={metadata.mean_reward:.3f})"
        )
        return False

    def _delete_game(self, game_file: str, reason: str, reward: float):
        """Delete a game file and remove from tracking."""
        self.total_deletions += 1

        if game_file in self.games:
            metadata = self.games[game_file]
            logging.info(
                f"[GamePolicy] Deleting: {game_file} ({reason}: {reward:.3f}) "
                f"history={[f'{r:.2f}' for r in metadata.reward_history]}"
            )
            del self.games[game_file]

        try:
            if Path(game_file).exists():
                Path(game_file).unlink()
        except Exception as e:
            logging.warning(f"[GamePolicy] Failed to delete {game_file}: {e}")

    def get_game_metadata(self, game_file: str) -> Optional[GameMetadata]:
        """Get metadata for a specific game."""
        if game_file not in self.games:
            return None
        return self.games[game_file]

    def get_statistics(self) -> Dict:
        """Get policy statistics for logging."""
        # Compute aggregate stats across all games
        all_rewards = []
        for metadata in self.games.values():
            all_rewards.extend(metadata.reward_history)

        return {
            "game_policy/total_games": len(self.games),
            "game_policy/total_plays": len(all_rewards),
            "game_policy/mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "game_policy/total_deletions": self.total_deletions,
            "game_policy/deletions_too_easy": self.deletions_too_easy,
            "game_policy/deletions_too_hard": self.deletions_too_hard,
        }

    def dump_game_history(self) -> Dict[str, Dict]:
        """Dump all game histories for debugging."""
        result = {}
        for game_file, metadata in self.games.items():
            result[game_file] = {
                "skill": metadata.skill,
                "difficulty": metadata.difficulty,
                "play_count": metadata.play_count,
                "reward_history": metadata.reward_history,
                "mean_reward": metadata.mean_reward,
                "min_reward": metadata.min_reward,
                "max_reward": metadata.max_reward,
            }
        return result
