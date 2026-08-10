"""Backend-neutral turn-level batching helpers.

These functions use generate_batch() for turn-level batching, which is
optimal for vLLM where batching across multiple prompts in a single GPU
call provides significant throughput improvements.

For HTTP backends (Slime, Tinker), use the async methods in orchestrator.py
which provide concurrent execution via asyncio.gather().
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from spare.core.types import Trajectory, SpareConfig
from spare.core.model_adapter import ModelAdapter
from spare.core.game_generator import SyntheticGameGenerator
from spare.core.prompts import generate_game_generation_prompt
from spare.core.envs.synthetic_game_env import make_synthetic_env
from spare.core.utils import (
    validate_game,
    save_game_file,
    extract_game_code,
    build_env_trajectory,
    build_actor_trajectory,
    parse_action,
)

logger = logging.getLogger(__name__)


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get existing event loop or create a new one."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


async def _generate_single_game_async(
    model: ModelAdapter,
    config: SpareConfig,
    skill: str,
    difficulty: str,
    games_dir: Path,
    rollout_id: int,
    index: int,
    validate: bool,
    cache_step_dir: Optional[Path],
    max_attempts: int = 8,
) -> Optional[Tuple[Path, Trajectory]]:
    """Generate a single game with retry logic (for retry fallback).

    This is used when batched generation fails for some games and we need
    to retry them individually.
    """
    skill_info = SyntheticGameGenerator.COGNITIVE_SKILLS[skill]
    prompt_content = generate_game_generation_prompt(skill, skill_info, difficulty)
    messages = [{"role": "user", "content": prompt_content}]
    game_file = None

    for attempt in range(max_attempts):
        try:
            result = await model.generate_async(
                messages=messages,
                temperature=config.env_temperature,
                top_p=config.env_top_p,
                max_tokens=config.env_max_tokens,
            )
            result = result[0] if isinstance(result, list) else result

            response_text = result['text']
            game_code = extract_game_code(response_text)

            # Save game file
            game_file = save_game_file(
                game_code, games_dir, rollout_id, index, skill
            )

            # Validate if requested
            if validate and not validate_game(game_file):
                raise RuntimeError("Game validation failed")

            # Build trajectory
            env_traj = build_env_trajectory(
                model=model,
                messages=messages,
                result=result,
                skill=skill,
                difficulty=difficulty,
                game_code=game_code,
                index=index,
            )

            # Cache if enabled
            if cache_step_dir is not None:
                shutil.copy2(str(game_file), str(cache_step_dir / game_file.name))

            return game_file, env_traj

        except Exception as e:
            logger.warning(f"[GEN] Game {index} (skill={skill}) attempt {attempt + 1}/{max_attempts} failed: {e}")
            if game_file and game_file.exists():
                game_file.unlink(missing_ok=True)
            continue

    logger.error(f"[GEN] Game {index} (skill={skill}) failed after {max_attempts} attempts")
    return None


def generate_games_batched(
    model: ModelAdapter,
    config: SpareConfig,
    skills: List[str],
    difficulty: str,
    games_dir: Path,
    num_games: int,
    base_index: int = 0,
    validate: bool = True,
    rollout_id: int = 0,
    cache_dir: Optional[Path] = None,
    max_attempts: int = 8,
) -> Tuple[List[Path], Dict[Path, Trajectory]]:
    """Generate multiple games with batched inference - optimal for vLLM.

    Uses generate_batch() to generate all games in one GPU call.

    Args:
        model: Model adapter with generate_batch method
        config: SpareConfig with temperature/top_p/max_tokens settings
        skills: List of skills to cycle through
        difficulty: Difficulty level
        games_dir: Directory to save games
        num_games: Number of games to generate
        base_index: Starting index for file naming
        validate: Whether to validate games
        rollout_id: Current rollout ID
        cache_dir: Optional cache directory
        max_attempts: Maximum generation attempts per game

    Returns:
        Tuple of (game file paths, path -> trajectory mapping)
    """
    games_dir = Path(games_dir)
    games_dir.mkdir(parents=True, exist_ok=True)

    cache_step_dir = None
    if cache_dir is not None:
        cache_step_dir = Path(cache_dir) / f"step_{rollout_id:05d}"
        cache_step_dir.mkdir(parents=True, exist_ok=True)

    # Build prompts
    messages_batch = []
    skill_list = []
    for i in range(num_games):
        skill = skills[i % len(skills)]
        skill_info = SyntheticGameGenerator.COGNITIVE_SKILLS[skill]
        prompt_content = generate_game_generation_prompt(skill, skill_info, difficulty)
        messages_batch.append([{"role": "user", "content": prompt_content}])
        skill_list.append(skill)

    logger.info(f"[GEN-BATCH] Generating {num_games} games")

    # Batch generate
    results = model.generate_batch(
        messages_batch,
        temperature=config.env_temperature,
        top_p=config.env_top_p,
        max_tokens=config.env_max_tokens,
    )

    # Process results with retry
    game_files: List[Path] = []
    env_trajectories: Dict[Path, Trajectory] = {}
    failed_indices = []

    for i, (result, skill) in enumerate(zip(results, skill_list)):
        game_file = None
        try:
            response_text = result['text']
            game_code = extract_game_code(response_text)

            game_file = save_game_file(
                game_code, games_dir, rollout_id, base_index + i, skill
            )

            if validate and not validate_game(game_file):
                raise RuntimeError("Game validation failed")

            env_traj = build_env_trajectory(
                model=model,
                messages=messages_batch[i],
                result=result,
                skill=skill,
                difficulty=difficulty,
                game_code=game_code,
                index=i,
            )

            game_files.append(game_file)
            env_trajectories[game_file] = env_traj

            if cache_step_dir is not None:
                shutil.copy2(str(game_file), str(cache_step_dir / game_file.name))

            logger.info(f"[GEN-BATCH] Saved: {game_file.name}")

        except Exception as e:
            logger.warning(f"[GEN-BATCH] Game {i} failed: {e}")
            if game_file is not None and game_file.exists():
                game_file.unlink(missing_ok=True)
            failed_indices.append((i, skill))

    # Retry failed games individually using async fallback
    for retry_i, (i, skill) in enumerate(failed_indices, start=1):
        logger.info(f"[GEN-BATCH] Retrying game {i} ({retry_i}/{len(failed_indices)})")
        try:
            loop = _get_or_create_event_loop()
            result = loop.run_until_complete(_generate_single_game_async(
                model=model,
                config=config,
                skill=skill,
                difficulty=difficulty,
                games_dir=games_dir,
                rollout_id=rollout_id,
                index=base_index + i,
                validate=validate,
                cache_step_dir=cache_step_dir,
                max_attempts=3,
            ))

            if result is not None:
                game_file, env_traj = result
                game_files.append(game_file)
                env_trajectories[game_file] = env_traj
                logger.info(f"[GEN-BATCH] Retry succeeded: {game_file.name}")

        except Exception as e:
            logger.error(f"[GEN-BATCH] Retry for game {i} failed: {e}")

    logger.info(f"[GEN-BATCH] Generated {len(game_files)}/{num_games} games")
    return game_files, env_trajectories


def play_games_batched(
    model: ModelAdapter,
    config: SpareConfig,
    game_files: List[Path],
    trajectories_per_game: int = 1,
) -> Tuple[List[Trajectory], Dict[str, Any]]:
    """Play multiple games with turn-level batching - optimal for vLLM.

    This batches across turns (not games): each turn, all active games
    submit observations → single generate_batch call → step all envs.

    Args:
        model: Model adapter with generate_batch method
        config: SpareConfig with temperature/top_p/max_tokens/max_turns settings
        game_files: List of game files to play
        trajectories_per_game: Number of trajectories per game

    Returns:
        Tuple of (trajectories, info dict)
    """
    info: Dict[str, Any] = {
        "num_instances": 0,
        "num_succeeded": 0,
        "num_failed": 0,
        "total_turns": 0,
    }

    # Initialize all game instances
    active_instances: List[Dict[str, Any]] = []

    for game_file in game_files:
        for _ in range(trajectories_per_game):
            try:
                env = make_synthetic_env(str(game_file))
                obs, _ = env.reset()
                active_instances.append({
                    "env": env,
                    "obs": obs,
                    "messages": [{"role": "user", "content": obs}],
                    "all_tokens": model.tokenizer(obs, add_special_tokens=False)["input_ids"],
                    "all_masks": [],
                    "all_logprobs": [],
                    "assistant_responses": [],
                    "rewards": [],
                    "done": False,
                    "terminated": False,
                    "truncated": False,
                    "game_file": game_file,
                    "turn": 0,
                })
                info["num_instances"] += 1
            except Exception as e:
                logger.error(f"[BATCH] Failed to init: {game_file.name}: {e}")
                info["num_failed"] += 1

    logger.info(f"[BATCH] Initialized {len(active_instances)} instances")

    # Turn-level batched execution
    for turn in range(config.max_turns):
        if turn > 0:
            for inst in active_instances:
                if not inst["done"]:
                    inst["messages"].append({"role": "user", "content": inst["obs"]})
                    obs_tokens = model.tokenizer(inst["obs"], add_special_tokens=False)["input_ids"]
                    inst["all_tokens"].extend(obs_tokens)
                    inst["all_masks"].extend([0] * len(obs_tokens))
                    inst["all_logprobs"].extend([0.0] * len(obs_tokens))

        active_indices = [i for i, inst in enumerate(active_instances) if not inst["done"]]
        if not active_indices:
            break

        messages_batch = [active_instances[i]["messages"] for i in active_indices]

        try:
            results = model.generate_batch(
                messages_batch,
                temperature=config.actor_temperature,
                top_p=config.actor_top_p,
                max_tokens=config.actor_max_tokens,
            )
        except Exception as e:
            logger.error(f"[BATCH] Generation failed at turn {turn}: {e}")
            break

        for result_idx, instance_idx in enumerate(active_indices):
            inst = active_instances[instance_idx]
            result = results[result_idx]
            raw_action = result['text']

            inst["messages"].append({"role": "assistant", "content": raw_action})
            response_tokens = result['token_ids']
            inst["all_tokens"].extend(response_tokens)
            inst["all_masks"].extend([1] * len(response_tokens))
            inst["all_logprobs"].extend(result['logprobs'])
            inst["assistant_responses"].append(raw_action)

            try:
                action = parse_action(raw_action, config.action_format)
                obs, reward, terminated, truncated, _ = inst["env"].step(action)
                inst["turn"] += 1
                inst["rewards"].append(reward)
                inst["obs"] = obs
                inst["terminated"] = terminated
                inst["truncated"] = truncated
                inst["done"] = terminated or truncated

            except Exception as e:
                logger.error(f"[BATCH] Step failed: {e}")
                inst["done"] = True

        info["total_turns"] += 1

        if turn % 5 == 0:
            num_active = sum(1 for inst in active_instances if not inst["done"])
            logger.info(f"[BATCH] Turn {turn + 1}: {num_active}/{len(active_instances)} active")

    # Build trajectories
    all_trajectories: List[Trajectory] = []
    for inst in active_instances:
        inst["env"].close()

        if inst["messages"]:
            traj = build_actor_trajectory(
                messages=inst["messages"],
                all_tokens=inst["all_tokens"],
                all_masks=inst["all_masks"],
                all_logprobs=inst["all_logprobs"],
                assistant_responses=inst["assistant_responses"],
                rewards=inst["rewards"],
                game_file_path=str(inst["game_file"]),
                turn=inst["turn"],
                terminated=inst["terminated"],
                truncated=inst["truncated"],
                index=len(all_trajectories),
            )
            all_trajectories.append(traj)
            info["num_succeeded"] += 1
        else:
            info["num_failed"] += 1

    logger.info(f"[BATCH] Done: {info['num_succeeded']} succeeded, {info['num_failed']} failed")

    if all_trajectories:
        info["avg_turns"] = sum(t.turn_count for t in all_trajectories) / len(all_trajectories)
        info["avg_reward"] = sum(t.reward for t in all_trajectories) / len(all_trajectories)

    return all_trajectories, info
