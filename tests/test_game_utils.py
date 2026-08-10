from pathlib import Path

import pytest

from spare.core.learning_potential import GameBaselineTracker
from spare.core.types import Trajectory, TrajectoryStatus
from spare.core.utils.game_utils import (
    assign_trajectory_weights,
    build_actor_trajectory,
    build_env_trajectory,
    cleanup_old_games,
    compute_env_reward_scale,
    compute_format_reward,
    compute_returns,
    episode_reward,
    extract_boxed_answer,
    extract_command,
    extract_game_code,
    extract_tool_call,
    normalize_rewards_per_game,
    parse_action,
    plateau_reward,
    repair_fstring_braces,
    save_game_file,
    save_rejected_game,
    upsample_trajectories,
    validate_boxed_format,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"answer: \boxed{42}", "42"),
        (r"answer: \boxed{{42}}", "42"),
        (r"answer: \boxed{{{x + 1}}}", "x + 1"),
        (r"answer: \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        ("no answer", None),
        ("", None),
    ],
)
def test_boxed_answer_extraction(text: str, expected: str | None) -> None:
    assert extract_boxed_answer(text) == expected
    assert validate_boxed_format(text) is (expected is not None)


def test_action_parsing() -> None:
    assert parse_action(r"reasoning \boxed{7}") == r"\boxed{7}"
    assert parse_action("  raw  ", "tool_call") == "raw"
    assert parse_action("  raw  ", "command") == "raw"
    assert extract_tool_call('<tool_call>{"name":"x"}</tool_call>') == '{"name":"x"}'
    assert extract_tool_call("<answer>done</answer>") == "ANSWER: done"
    assert extract_command("<command>pwd</command>") == "pwd"
    assert extract_command("<answer>done</answer>") == "ANSWER: done"


def test_basic_reward_math() -> None:
    assert compute_format_reward(r"\boxed{ok}", 2.0, -1.0) == 2.0
    assert compute_format_reward("missing", 2.0, -1.0) == -1.0
    assert compute_returns([1.0, 2.0, 3.0], 0.5) == pytest.approx([2.75, 3.5, 3.0])
    assert plateau_reward(0.5) == 1.0
    assert plateau_reward(0.15) == pytest.approx(0.0)
    assert plateau_reward(0.85) == pytest.approx(0.0)
    assert episode_reward([0.2, 2.0], terminated=True) == 1.0
    assert episode_reward([-2.0], terminated=True) == -1.0
    assert episode_reward([1.0], terminated=False) == 0.0


def test_generated_code_file_helpers(tmp_path: Path) -> None:
    game = save_game_file("class Demo: pass\n", tmp_path, 3, 2, "Logical Deduction")
    assert game.name == "game_00003_002_logical_deduction.py"
    assert "class Demo" in game.read_text()

    rejected = save_rejected_game(
        "broken",
        tmp_path,
        3,
        2,
        "Logical Deduction",
        "syntax",
        "invalid code",
    )
    assert rejected is not None and rejected.exists()
    assert rejected.with_suffix(".reason.txt").read_text() == "invalid code"

    cleanup_old_games([game])
    assert not game.exists()


def test_generated_code_extraction_and_repair() -> None:
    valid = 'message = f"value={value}"'
    assert repair_fstring_braces(valid) == valid

    broken = 'message = f"use \\boxed{<answer>}"'
    repaired = repair_fstring_braces(broken)
    compile(repaired, "<generated>", "exec")
    assert "{{<answer>}}" in repaired
    assert extract_game_code(f"```python\n{broken}\n```") == repaired


def _trajectory(index: int, reward: float, game: Path) -> Trajectory:
    return Trajectory(
        index=index,
        reward=reward,
        status=TrajectoryStatus.COMPLETED,
        metadata={"game_file": str(game), "skill": "logic"},
    )


def test_reward_normalization_and_upsampling(tmp_path: Path) -> None:
    game = tmp_path / "game.py"
    trajectories = [_trajectory(0, 0.0, game), _trajectory(1, 1.0, game)]
    baseline = GameBaselineTracker(decay=0.5)

    normalized, stats = normalize_rewards_per_game(
        trajectories,
        [game],
        game_baseline_tracker=baseline,
    )
    assert [trajectory.reward for trajectory in normalized] == [0.0, 1.0]
    assert baseline.get_baseline(str(game)) == 0.5
    assert stats["game"][str(game)] == [0.0, 1.0]

    grpo, _ = normalize_rewards_per_game(trajectories, [game], reward_normalization="grpo")
    assert [trajectory.reward for trajectory in grpo] == pytest.approx([-1.0, 1.0])
    with pytest.raises(ValueError, match="Unknown reward_normalization"):
        normalize_rewards_per_game(trajectories, [game], reward_normalization="unknown")

    upsampled = upsample_trajectories(trajectories, min_samples=5)
    assert len(upsampled) == 5
    assert upsampled[-1].metadata["upsampled"] is True
    assert upsample_trajectories([], min_samples=5) == []


def test_trajectory_builders_and_weights() -> None:
    env = build_env_trajectory(
        messages=[{"role": "user", "content": "prompt"}],
        all_tokens=[1, 2],
        all_masks=[0, 1],
        all_logprobs=[-0.2],
        assistant_responses=["code"],
        skill="logic",
        difficulty="hard",
        game_code="pass",
    )
    actor = build_actor_trajectory(
        messages=[{"role": "user", "content": "play"}],
        all_tokens=[1, 2],
        all_masks=[0, 1],
        all_logprobs=[-0.3],
        assistant_responses=["answer"],
        rewards=[0.0, 1.0],
        game_file_path="game.py",
        skill="logic",
        turn=2,
        terminated=True,
        truncated=False,
    )
    assert env.status is TrajectoryStatus.COMPLETED and env.reward == 0.0
    assert actor.status is TrajectoryStatus.COMPLETED and actor.reward == 1.0
    assert compute_env_reward_scale(2, 8, 3) == 12.0
    assert compute_env_reward_scale(0, 8, 3) == 1.0

    weights = assign_trajectory_weights([env], [actor], regeneration_interval=4)
    assert weights == {"env_weight": 4.0, "actor_weight": 1.0, "computed_scale": 4.0}
    assert env.metadata["role"] == "environment"
    assert actor.metadata["role"] == "actor"
