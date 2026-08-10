import pytest

from spare.core.learning_potential import LearningPotential
from spare.core.types import Trajectory
from spare.core.utils.game_utils import assign_env_rewards, assign_env_rewards_regret


def _environment(skill: str = "logic", **metadata) -> Trajectory:
    return Trajectory(metadata={"skill": skill, **metadata})


def _trackers() -> dict[str, LearningPotential]:
    return {"logic": LearningPotential(gamma_fast=0.8, gamma_slow=0.2)}


def test_learning_potential_rewards_are_centered_per_skill() -> None:
    environments = {"g1.py": _environment(), "g2.py": _environment()}
    trackers = _trackers()
    result, stats = assign_env_rewards(
        env_trajectories=environments,
        skill2rewards={"logic": [0.0, 0.4, 0.8, 0.8]},
        game2rewards={"g1.py": [0.0, 0.4], "g2.py": [0.8, 0.8]},
        learning_potentials=trackers,
        use_solver_variance_reward=False,
        env_reward_scale=10.0,
    )

    assert [trajectory.reward for trajectory in result] == pytest.approx([-3.0, 3.0])
    assert [trajectory.metadata["env_raw_reward"] for trajectory in result] == pytest.approx(
        [0.2, 0.8]
    )
    assert trackers["logic"].get_baseline() == 0.5
    assert stats["env/skill_logic/actor_win_rate"] == 0.5
    assert stats["env/game_difficulty/medium_pct"] == 1.0
    assert stats["env/avg_game_std_reward"] == pytest.approx(0.1)


def test_lp_assignment_handles_missing_repair_and_variance_paths() -> None:
    trackers = _trackers()
    environments = {
        "missing.py": _environment(),
        "repaired.py": _environment(turn1_valid=False),
        "valid.py": _environment(),
    }
    result, _ = assign_env_rewards(
        env_trajectories=environments,
        skill2rewards={"logic": [0.0, 1.0]},
        game2rewards={"repaired.py": [0.0, 1.0], "valid.py": [0.0, 1.0]},
        learning_potentials=trackers,
        use_solver_variance_reward=True,
        env_reward_scale=2.0,
        skip_lp_update=True,
    )

    assert len(result) == 2
    assert result[0].metadata["env_raw_reward"] == 0.0
    assert result[1].metadata["env_raw_reward"] == pytest.approx(1.5)
    assert [trajectory.reward for trajectory in result] == pytest.approx([-1.5, 1.5])
    assert trackers["logic"].mu_fast is None
    assert assign_env_rewards({}, {}, {}, trackers, False) == ([], {})


def test_regret_rewards_floor_negative_values_and_report_metrics() -> None:
    environments = {"g1.py": _environment(), "g2.py": _environment()}
    trackers = _trackers()
    result, stats = assign_env_rewards_regret(
        env_trajectories=environments,
        game2regret={"g1.py": -0.2, "g2.py": 0.8},
        game2rewards={"g1.py": [0.0], "g2.py": [1.0]},
        game2hint_stats={
            "g1.py": {"r_hint": 0.1, "r_no_hint": 0.3},
            "g2.py": {"r_hint": 1.0, "r_no_hint": 0.2},
        },
        learning_potentials=trackers,
        skill2rewards={"logic": [0.0, 1.0]},
        env_reward_scale=10.0,
    )

    assert [trajectory.reward for trajectory in result] == pytest.approx([-4.0, 4.0])
    assert [trajectory.metadata["regret"] for trajectory in result] == [0.0, 0.8]
    assert stats["env/proposer_floored_frac"] == 0.5
    assert stats["env/proposer_step_timeout_frac"] == 0.0
    assert stats["env/avg_r_hint"] == 0.55
    assert stats["env/game_difficulty/easy_pct"] == 0.5
    assert stats["env/game_difficulty/hard_pct"] == 0.5
    assert trackers["logic"].get_baseline() == 0.5


def test_regret_step_timeout_and_failed_first_turn_keep_bounded_signals() -> None:
    environments = {
        "timeout.py": _environment(step_timeout=True),
        "repaired.py": _environment(turn1_valid=False),
        "valid.py": _environment(),
    }
    result, stats = assign_env_rewards_regret(
        env_trajectories=environments,
        game2regret={"timeout.py": 1.0, "repaired.py": 1.0, "valid.py": 0.5},
        game2rewards={},
        game2hint_stats={},
        learning_potentials=_trackers(),
        skill2rewards={"logic": []},
        env_reward_scale=3.0,
    )

    assert [trajectory.metadata["env_raw_reward"] for trajectory in result] == [
        -0.5,
        0.0,
        0.5,
    ]
    assert [trajectory.reward for trajectory in result] == pytest.approx([-1.5, 0.0, 1.5])
    assert stats["env/proposer_step_timeout_frac"] == pytest.approx(1 / 3)
    assert assign_env_rewards_regret({}, {}, {}, {}, _trackers(), {}) == ([], {})
