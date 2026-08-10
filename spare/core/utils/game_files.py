"""Generated-game filesystem operations."""

import logging
from pathlib import Path

from spare.core.envs.synthetic_game_env import make_synthetic_env

logger = logging.getLogger(__name__)


def validate_game(game_file: Path) -> bool:
    try:
        env = make_synthetic_env(str(game_file))
        env.reset()
        for action in ["1", "2", "test"]:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()
        return True
    except Exception as exc:
        logger.warning("Game validation failed: %s", exc)
        return False


def save_game_file(
    game_code: str,
    games_dir: Path,
    rollout_id: int,
    index: int,
    skill: str,
) -> Path:
    slug = skill.lower().replace(" ", "_")
    game_file = games_dir / f"game_{rollout_id:05d}_{index:03d}_{slug}.py"
    game_file.write_text('"""SPARE self-play generated game"""\n\n' + game_code)
    return game_file


def save_rejected_game(
    game_code: str,
    games_dir: Path,
    rollout_id: int,
    index: int,
    skill: str,
    reject_stage: str,
    reasoning: str,
) -> Path | None:
    try:
        rejects_dir = Path(games_dir).parent / "spare_games_rejected"
        rejects_dir.mkdir(parents=True, exist_ok=True)
        slug = skill.lower().replace(" ", "_")
        stem = f"reject_{rollout_id:05d}_{index:03d}_{slug}_{reject_stage}"
        game_file = rejects_dir / f"{stem}.py"
        game_file.write_text(
            f'"""SPARE self-play REJECTED game (stage={reject_stage})"""\n\n{game_code}'
        )
        (rejects_dir / f"{stem}.reason.txt").write_text(reasoning or "")
        return game_file
    except Exception:
        return None


def cleanup_old_games(game_files: list[Path]) -> None:
    for game_file in game_files:
        try:
            Path(game_file).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("[COLLECT] Failed to delete %s: %s", game_file, exc)
    logger.info("[COLLECT] Deleted %d old games", len(game_files))
