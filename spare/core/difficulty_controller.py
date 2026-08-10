"""Difficulty controllers for fixed-env adaptive training.

Two variants for controlling environment difficulty:

Variant A — RLVE-style sliding window:
    Per env, maintain window [l, h]:
      Track: a = successes at difficulty h, b = attempts at difficulty h
      If b >= tau_num AND a/b >= tau_acc: h += 1, l = max(l, h - d_delta + 1)
      Sample difficulty uniformly from [l, h]

Variant B — SPARE LP-based:
    Per env, track mu_fast / mu_slow via EMA:
      Select environments with highest |mu_fast - mu_slow| (learning frontier)
      Continuous signal vs binary threshold
"""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

from spare.core.learning_potential import LearningPotential

logger = logging.getLogger(__name__)


class DifficultyController(ABC):
    """Abstract base for difficulty control strategies."""

    @abstractmethod
    def sample_difficulty(self, env_id: str) -> int:
        """Sample a difficulty level for the given environment.

        Args:
            env_id: Environment identifier

        Returns:
            Sampled difficulty level
        """
        ...

    @abstractmethod
    def update(self, env_id: str, difficulty: int, reward: float) -> None:
        """Update the controller with the result of playing at a difficulty.

        Args:
            env_id: Environment identifier
            difficulty: Difficulty level that was played
            reward: Reward obtained (typically 0 or 1 for win/loss)
        """
        ...

    @abstractmethod
    def select_environments(
        self, env_ids: List[str], n: int
    ) -> List[str]:
        """Select which environments to play next.

        Args:
            env_ids: Available environment IDs
            n: Number to select

        Returns:
            List of selected environment IDs
        """
        ...

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get metrics for logging."""
        ...

    def set_difficulty_range(self, env_id: str, min_d: int, max_d: int) -> None:
        """Set the valid difficulty range for an environment.

        Called during initialization to inform the controller of each
        environment's difficulty bounds. Default is a no-op; subclasses
        override if they need range information.

        Args:
            env_id: Environment identifier
            min_d: Minimum difficulty level
            max_d: Maximum difficulty level
        """


# =========================================================================
# Variant A: RLVE-style Sliding Window
# =========================================================================

@dataclass
class _WindowState:
    """Per-environment state for sliding window controller."""
    low: int = 0
    high: int = 0
    successes_at_high: int = 0
    attempts_at_high: int = 0


class SlidingWindowController(DifficultyController):
    """RLVE-style sliding window difficulty controller.

    Per environment, maintains a window [low, high]:
    - Tracks successes and attempts at the current high difficulty
    - When enough attempts with high success rate: promote (h += 1)
    - Window width bounded by d_delta

    Args:
        tau_acc: Success rate threshold for promotion (default 0.9)
        tau_num: Minimum attempts before promotion check (default 8)
        d_delta: Maximum window width (default 4)
        max_difficulty: Global maximum difficulty (default 20)
    """

    def __init__(
        self,
        tau_acc: float = 0.9,
        tau_num: int = 8,
        d_delta: int = 4,
        max_difficulty: int = 20,
    ):
        self.tau_acc = tau_acc
        self.tau_num = tau_num
        self.d_delta = d_delta
        self.max_difficulty = max_difficulty
        self._states: Dict[str, _WindowState] = {}

    def _get_state(self, env_id: str) -> _WindowState:
        if env_id not in self._states:
            self._states[env_id] = _WindowState()
        return self._states[env_id]

    def sample_difficulty(self, env_id: str) -> int:
        state = self._get_state(env_id)
        return random.randint(state.low, state.high)

    def update(self, env_id: str, difficulty: int, reward: float) -> None:
        state = self._get_state(env_id)

        # Only track attempts at the current high difficulty
        if difficulty == state.high:
            state.attempts_at_high += 1
            if reward > 0:
                state.successes_at_high += 1

            # Check for promotion when enough attempts
            if state.attempts_at_high >= self.tau_num:
                success_rate = state.successes_at_high / state.attempts_at_high
                if success_rate >= self.tau_acc and state.high < self.max_difficulty:
                    state.high += 1
                    state.low = max(state.low, state.high - self.d_delta + 1)
                    logger.info(
                        "[SLIDING-WINDOW] %s promoted: [%d, %d]",
                        env_id, state.low, state.high,
                    )
                # ALWAYS reset after check (matches RLVE behavior)
                state.successes_at_high = 0
                state.attempts_at_high = 0

    def select_environments(
        self, env_ids: List[str], n: int
    ) -> List[str]:
        # Sliding window doesn't prioritize — sample uniformly
        if len(env_ids) <= n:
            return list(env_ids)
        return random.sample(env_ids, n)

    def set_difficulty_range(self, env_id: str, min_d: int, max_d: int) -> None:
        """Initialize difficulty window from adapter's range.

        Ensures the window starts at min_d (e.g., RLVE starts at 0).
        """
        state = self._get_state(env_id)
        state.low = max(state.low, min_d)
        state.high = max(state.high, min_d)

    def get_metrics(self) -> Dict[str, float]:
        if not self._states:
            return {}
        highs = [s.high for s in self._states.values()]
        lows = [s.low for s in self._states.values()]
        return {
            "difficulty/avg_high": sum(highs) / len(highs),
            "difficulty/avg_low": sum(lows) / len(lows),
            "difficulty/max_high": float(max(highs)),
            "difficulty/num_envs": float(len(self._states)),
        }


# =========================================================================
# Variant B: LP-based Difficulty Controller
# =========================================================================

class LPDifficultyController(DifficultyController):
    """Learning-potential-based difficulty controller.

    Per environment, tracks mu_fast and mu_slow via EMA. Selects
    environments at the learning frontier (highest |mu_fast - mu_slow|).

    This provides a continuous signal compared to the binary threshold
    of the sliding window approach.

    Args:
        gamma_fast: EMA gamma for fast tracker (default 0.35)
        gamma_slow: EMA gamma for slow tracker (default 0.15)
        default_difficulty: Fixed difficulty when LP doesn't control it (default 1)
    """

    def __init__(
        self,
        gamma_fast: float = 0.35,
        gamma_slow: float = 0.15,
        default_difficulty: int = 1,
    ):
        self.gamma_fast = gamma_fast
        self.gamma_slow = gamma_slow
        self.default_difficulty = default_difficulty
        self._trackers: Dict[str, LearningPotential] = {}

    def _get_tracker(self, env_id: str) -> LearningPotential:
        if env_id not in self._trackers:
            self._trackers[env_id] = LearningPotential(
                gamma_fast=self.gamma_fast,
                gamma_slow=self.gamma_slow,
            )
        return self._trackers[env_id]

    def sample_difficulty(self, env_id: str) -> int:
        # LP controller doesn't vary difficulty — it selects environments
        return self.default_difficulty

    def update(self, env_id: str, difficulty: int, reward: float) -> None:
        tracker = self._get_tracker(env_id)
        tracker.update(reward)

    def select_environments(
        self, env_ids: List[str], n: int
    ) -> List[str]:
        """Select environments with highest learning potential (frontier).

        Environments with high |mu_fast - mu_slow| are at the learning
        frontier — where the model is actively improving or struggling.

        Args:
            env_ids: Available environment IDs
            n: Number to select

        Returns:
            Top-n environments by absolute learning potential
        """
        if len(env_ids) <= n:
            return list(env_ids)

        # Score each environment by absolute LP
        scored = []
        for env_id in env_ids:
            tracker = self._get_tracker(env_id)
            lp = tracker.get_current_potential()
            scored.append((env_id, lp))

        # Sort by LP descending, take top n
        scored.sort(key=lambda x: x[1], reverse=True)

        # Mix: top n//2 by LP + random n//2 from remaining (exploration)
        n_exploit = max(1, n // 2)
        n_explore = n - n_exploit

        selected = [env_id for env_id, _ in scored[:n_exploit]]
        remaining = [env_id for env_id, _ in scored[n_exploit:]]
        if remaining and n_explore > 0:
            selected.extend(random.sample(remaining, min(n_explore, len(remaining))))

        return selected

    def get_metrics(self) -> Dict[str, float]:
        if not self._trackers:
            return {}
        lps = [t.get_current_potential() for t in self._trackers.values()]
        signed_lps = [t.get_signed_potential() for t in self._trackers.values()]
        mu_fasts = [t.mu_fast for t in self._trackers.values() if t.mu_fast is not None]
        mu_slows = [t.mu_slow for t in self._trackers.values() if t.mu_slow is not None]

        metrics = {
            "difficulty/avg_lp": sum(lps) / len(lps),
            "difficulty/avg_signed_lp": sum(signed_lps) / len(signed_lps),
            "difficulty/max_lp": float(max(lps)),
            "difficulty/num_envs": float(len(self._trackers)),
        }
        if mu_fasts:
            metrics["difficulty/avg_mu_fast"] = sum(mu_fasts) / len(mu_fasts)
        if mu_slows:
            metrics["difficulty/avg_mu_slow"] = sum(mu_slows) / len(mu_slows)

        return metrics


def create_difficulty_controller(
    variant: str,
    tau_acc: float = 0.9,
    tau_num: int = 8,
    d_delta: int = 4,
    gamma_fast: float = 0.35,
    gamma_slow: float = 0.15,
) -> DifficultyController:
    """Factory function to create a difficulty controller.

    Args:
        variant: "sliding_window" or "lp"
        tau_acc: Success threshold for sliding window
        tau_num: Minimum attempts for sliding window
        d_delta: Window width for sliding window
        gamma_fast: Fast EMA gamma for LP controller
        gamma_slow: Slow EMA gamma for LP controller

    Returns:
        Configured DifficultyController instance
    """
    if variant == "sliding_window":
        return SlidingWindowController(
            tau_acc=tau_acc,
            tau_num=tau_num,
            d_delta=d_delta,
        )
    elif variant == "lp":
        return LPDifficultyController(
            gamma_fast=gamma_fast,
            gamma_slow=gamma_slow,
        )
    else:
        raise ValueError(
            f"Unknown difficulty variant: {variant}. "
            "Choose 'sliding_window' or 'lp'."
        )
