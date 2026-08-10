"""Learning-potential and reward-baseline trackers."""

from typing import Any


class LearningPotential:
    """Track the gap between fast and slow reward moving averages."""

    def __init__(self, gamma_fast: float = 0.99, gamma_slow: float = 0.95):
        assert gamma_fast > gamma_slow, "gamma_fast must be > gamma_slow"
        self.gamma_fast = gamma_fast
        self.gamma_slow = gamma_slow
        self.mu_fast: float | None = None
        self.mu_slow: float | None = None
        self.scale = 1 / ((gamma_fast - gamma_slow) * 2)
        self.history: list[dict[str, float]] = []

    def update(self, r_t: float) -> float:
        if self.mu_fast is None:
            self.mu_fast = self.mu_slow = r_t
            rho_t = 0.0
        else:
            self.mu_fast = (1 - self.gamma_fast) * self.mu_fast + self.gamma_fast * r_t
            self.mu_slow = (1 - self.gamma_slow) * self.mu_slow + self.gamma_slow * r_t
            rho_t = abs(self.mu_fast - self.mu_slow) * self.scale

        self.history.append(
            {
                "r_t": r_t,
                "mu_fast": self.mu_fast,
                "mu_slow": self.mu_slow,
                "rho_t": rho_t,
            }
        )
        return rho_t

    def get_current_potential(self) -> float:
        if self.mu_fast is None or self.mu_slow is None:
            return 0.0
        return abs(self.mu_fast - self.mu_slow) * self.scale

    def get_signed_potential(self) -> float:
        if self.mu_fast is None or self.mu_slow is None:
            return 0.0
        return (self.mu_fast - self.mu_slow) * self.scale

    def get_baseline(self) -> float:
        return self.mu_slow if self.mu_slow is not None else 0.0

    def reset(self) -> None:
        self.mu_fast = None
        self.mu_slow = None
        self.history = []

    def get_statistics(self) -> dict[str, float]:
        return {
            "mu_fast": self.mu_fast if self.mu_fast is not None else 0.0,
            "mu_slow": self.mu_slow if self.mu_slow is not None else 0.0,
            "rho": self.get_current_potential(),
            "gamma_fast": self.gamma_fast,
            "gamma_slow": self.gamma_slow,
            "history_length": len(self.history),
        }

    def to_dict(self) -> dict[str, float | None]:
        return {
            "gamma_fast": self.gamma_fast,
            "gamma_slow": self.gamma_slow,
            "mu_fast": self.mu_fast,
            "mu_slow": self.mu_slow,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningPotential":
        tracker = cls(gamma_fast=data["gamma_fast"], gamma_slow=data["gamma_slow"])
        tracker.mu_fast = data.get("mu_fast")
        tracker.mu_slow = data.get("mu_slow")
        return tracker


class MultiAgentLearningPotential:
    """Maintain an independent learning-potential tracker per agent."""

    def __init__(
        self,
        num_agents: int = 2,
        gamma_fast: float = 0.99,
        gamma_slow: float = 0.95,
    ):
        self.num_agents = num_agents
        self.agents = {
            agent_id: LearningPotential(gamma_fast, gamma_slow) for agent_id in range(num_agents)
        }

    def update(self, agent_id: int, r_t: float) -> float:
        return self.agents[agent_id].update(r_t)

    def update_all(self, rewards: dict[int, float]) -> dict[int, float]:
        return {agent_id: self.update(agent_id, reward) for agent_id, reward in rewards.items()}

    def get_all_potentials(self) -> dict[int, float]:
        return {
            agent_id: tracker.get_current_potential() for agent_id, tracker in self.agents.items()
        }

    def get_statistics(self) -> dict[int, dict[str, float]]:
        return {agent_id: tracker.get_statistics() for agent_id, tracker in self.agents.items()}

    def reset(self, agent_id: int | None = None) -> None:
        if agent_id is not None:
            self.agents[agent_id].reset()
            return
        for tracker in self.agents.values():
            tracker.reset()


class ExploitabilityBasedPotential(LearningPotential):
    """Convert lower exploitability into a higher progress signal."""

    def __init__(
        self,
        gamma_fast: float = 0.99,
        gamma_slow: float = 0.95,
        normalize: bool = True,
    ):
        super().__init__(gamma_fast, gamma_slow)
        self.normalize = normalize
        self.max_exploitability = 1.0

    def update_from_exploitability(self, exploitability: float) -> float:
        if self.normalize:
            self.max_exploitability = max(self.max_exploitability, exploitability)
            progress = 1.0 - exploitability / self.max_exploitability
        else:
            progress = -exploitability
        return self.update(progress)


def calculate_game_progress(game_result: dict[str, Any]) -> float:
    if "winner" in game_result:
        if game_result["winner"] == 0:
            return 1.0
        if game_result["winner"] == 1:
            return -1.0
        return 0.0
    return game_result.get("score", 0.0)


class EMA:
    """Exponential moving average used for per-game reward baselines."""

    def __init__(self, decay: float = 0.99):
        assert 0.0 < decay < 1.0, "Decay must be between 0 and 1"
        self.decay = decay
        self.value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.decay * self.value + (1 - self.decay) * x
        return self.value

    def get(self) -> float:
        return self.value if self.value is not None else 0.0


class GameBaselineTracker:
    """Maintain one EMA reward baseline per game."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self._baselines: dict[str, EMA] = {}

    def get_baseline(self, game_file: str) -> float:
        tracker = self._baselines.get(game_file)
        return tracker.get() if tracker is not None else 0.0

    def update_baseline(self, game_file: str, avg_reward: float) -> float:
        tracker = self._baselines.setdefault(game_file, EMA(self.decay))
        return tracker.update(avg_reward)

    def get_stats(self) -> dict[str, Any]:
        if not self._baselines:
            return {}
        baselines = [tracker.get() for tracker in self._baselines.values()]
        return {
            "num_games_tracked": len(self._baselines),
            "avg_baseline": sum(baselines) / len(baselines),
            "min_baseline": min(baselines),
            "max_baseline": max(baselines),
        }
