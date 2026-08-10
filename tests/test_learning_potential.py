import pytest

from spare.core.learning_potential import (
    EMA,
    ExploitabilityBasedPotential,
    GameBaselineTracker,
    LearningPotential,
    MultiAgentLearningPotential,
    calculate_game_progress,
)


def test_learning_potential_updates_and_serializes() -> None:
    tracker = LearningPotential(gamma_fast=0.8, gamma_slow=0.2)
    assert tracker.update(0.25) == 0.0
    assert tracker.update(0.75) == pytest.approx(0.25)
    assert tracker.get_signed_potential() == pytest.approx(0.25)
    assert tracker.get_baseline() == pytest.approx(0.35)

    restored = LearningPotential.from_dict(tracker.to_dict())
    assert restored.get_current_potential() == tracker.get_current_potential()
    assert restored.get_statistics()["history_length"] == 0
    restored.reset()
    assert restored.get_current_potential() == 0.0


def test_invalid_learning_potential_gammas_are_rejected() -> None:
    with pytest.raises(AssertionError):
        LearningPotential(gamma_fast=0.2, gamma_slow=0.8)


def test_multi_agent_trackers_are_independent() -> None:
    tracker = MultiAgentLearningPotential(num_agents=2, gamma_fast=0.8, gamma_slow=0.2)
    tracker.update(0, 0.0)
    tracker.update(0, 1.0)
    assert tracker.get_all_potentials()[0] > 0
    assert tracker.get_all_potentials()[1] == 0
    tracker.reset(0)
    assert tracker.get_all_potentials()[0] == 0


def test_exploitability_is_converted_to_progress() -> None:
    tracker = ExploitabilityBasedPotential(gamma_fast=0.8, gamma_slow=0.2)
    assert tracker.update_from_exploitability(1.0) == 0.0
    assert tracker.update_from_exploitability(0.5) > 0


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"winner": 0}, 1.0),
        ({"winner": 1}, -1.0),
        ({"winner": 2}, 0.0),
        ({"score": 0.4}, 0.4),
        ({}, 0.0),
    ],
)
def test_calculate_game_progress(result: dict, expected: float) -> None:
    assert calculate_game_progress(result) == expected


def test_ema_and_game_baselines() -> None:
    ema = EMA(decay=0.5)
    assert ema.get() == 0.0
    assert ema.update(1.0) == 1.0
    assert ema.update(0.0) == 0.5

    baselines = GameBaselineTracker(decay=0.5)
    assert baselines.get_baseline("game.py") == 0.0
    assert baselines.update_baseline("game.py", 1.0) == 1.0
    assert baselines.update_baseline("game.py", 0.0) == 0.5
    assert baselines.get_stats() == {
        "num_games_tracked": 1,
        "avg_baseline": 0.5,
        "min_baseline": 0.5,
        "max_baseline": 0.5,
    }
