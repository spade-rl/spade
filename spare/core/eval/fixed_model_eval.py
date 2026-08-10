"""Fixed model evaluation for measuring game difficulty.

This module provides functionality to evaluate environment-generated games
using a fixed external model (e.g., Gemini 3 Flash via OpenRouter) to measure
game difficulty objectively.

Key Metrics:
- pass_rate: Percentage of games won by the fixed model (~0.5 is optimal)
- variance: Variance of binary outcomes (~0.25 is optimal for pass_rate=0.5)
- difficulty_score: 1 - |pass_rate - 0.5| / 0.5 (1.0 is optimal)
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.asyncio import tqdm_asyncio

from spare.core.envs.synthetic_game_env import make_synthetic_env
from spare.core.openrouter_adapter import OpenAIModelAdapter, create_openai_adapter

logger = logging.getLogger(__name__)


@dataclass
class GameEvalResult:
    """Result of evaluating a single game with the fixed model."""

    game_file: str
    skill: str
    num_plays: int
    num_wins: int
    pass_rate: float
    variance: float
    mean_reward: float
    mean_turns: int
    errors: int = 0

    @property
    def difficulty_score(self) -> float:
        """Compute difficulty score: 1.0 when pass_rate=0.5, 0.0 when pass_rate=0 or 1."""
        return 1.0 - abs(self.pass_rate - 0.5) / 0.5


@dataclass
class FixedModelEvalResult:
    """Aggregate results from evaluating all games."""

    overall_pass_rate: float
    overall_variance: float
    difficulty_score: float
    num_optimal_games: int  # Games with pass_rate in [0.3, 0.7]
    total_games: int
    total_plays: int
    avg_reward: float = 0.0
    std_reward: float = 0.0
    per_game_results: List[GameEvalResult] = field(default_factory=list)
    per_skill_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    errors: int = 0

    def to_metrics_dict(self, prefix: str = "fixed_eval") -> Dict[str, float]:
        """Convert results to a flat metrics dictionary for logging.

        Args:
            prefix: Prefix for metric names

        Returns:
            Dictionary of metrics suitable for W&B logging
        """
        metrics = {
            "{}/overall_pass_rate".format(prefix): self.overall_pass_rate,
            "{}/overall_variance".format(prefix): self.overall_variance,
            "{}/difficulty_score".format(prefix): self.difficulty_score,
            "{}/num_optimal_games".format(prefix): float(self.num_optimal_games),
            "{}/total_games".format(prefix): float(self.total_games),
            "{}/total_plays".format(prefix): float(self.total_plays),
            "{}/avg_reward".format(prefix): self.avg_reward,
            "{}/std_reward".format(prefix): self.std_reward,
            "{}/errors".format(prefix): float(self.errors),
        }

        # Add per-skill metrics
        for skill, skill_metrics in self.per_skill_metrics.items():
            # Replace spaces with underscores in skill name
            skill_key = skill.replace(" ", "_").lower()
            for metric_name, value in skill_metrics.items():
                metrics["{}/skill_{}/{}".format(prefix, skill_key, metric_name)] = value

        return metrics


def _extract_skill_from_game_file(game_file: str) -> str:
    """Extract skill name from game file path.

    Game files are typically named: game_{step}_{index}_{skill}.py
    e.g., game_00000_002_pattern_recognition.py
    e.g., game_00048_001_mathematical_reasoning.py

    The skill is everything after the second numeric part.
    """
    path = Path(game_file)
    name = path.stem  # Remove .py extension

    # Try to extract skill from filename pattern
    if name.startswith("game_"):
        # Remove "game_" prefix
        rest = name[5:]
        # Split by underscore
        parts = rest.split("_")

        # Find where the skill name starts (after numeric parts)
        # Pattern: {step}_{index}_{skill_part1}_{skill_part2}...
        # e.g., 00000_002_pattern_recognition -> skill starts at index 2
        skill_start_idx = 0
        for i, part in enumerate(parts):
            if part.isdigit() or (len(part) > 0 and part[0].isdigit()):
                skill_start_idx = i + 1
            else:
                # Found non-numeric part, this is where skill starts
                skill_start_idx = i
                break

        if skill_start_idx < len(parts):
            # Join remaining parts as the skill name
            skill_parts = parts[skill_start_idx:]
            skill = "_".join(skill_parts).replace("_", " ").title()
            return skill

    return "Unknown"


def _parse_action_from_response(response: str) -> Optional[str]:
    """Parse action from model response.

    Extracts content from \\boxed{} format.
    """
    patterns = [
        r"\\boxed\{([^{}]+)\}",  # \boxed{answer}
        r"\\boxed\{\{([^{}]+)\}\}",  # \boxed{{answer}}
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()

    return None


class FixedModelEvaluator:
    """Evaluator that uses a fixed external model to measure game difficulty.

    The evaluator plays each game multiple times with a fixed model and computes
    statistics on pass rates and variance to determine if games are appropriately
    challenging.
    """

    def __init__(
        self,
        model_adapter: OpenAIModelAdapter,
        plays_per_game: int = 8,
        max_turns: int = 30,
        max_concurrent: int = 4,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """Initialize the fixed model evaluator.

        Args:
            model_adapter: OpenRouter model adapter for generation
            plays_per_game: Number of times to play each game
            max_turns: Maximum turns per game
            max_concurrent: Maximum concurrent game plays (rate limiting)
            temperature: Temperature for generation
            max_tokens: Maximum tokens per response
        """
        self.model = model_adapter
        self.plays_per_game = plays_per_game
        self.max_turns = max_turns
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _format_observation(self, observation: str) -> List[Dict[str, str]]:
        """Format observation into OpenAI message format for the model."""
        return [
            {
                "role": "user",
                "content": (
                    "You are playing a language game. Make valid actions to win.\n"
                    "Observation: {}\n"
                    "Please reason step by step, and put your final answer within \\boxed{{}}.".format(
                        observation
                    )
                ),
            }
        ]

    async def _play_game_once(
        self, game_file: str, seed: int
    ) -> Tuple[bool, float, int]:
        """Play a game once and return (won, total_reward, num_turns).

        Args:
            game_file: Path to game file
            seed: Random seed for game reset

        Returns:
            Tuple of (won: bool, total_reward: float, num_turns: int)
        """
        try:
            env = make_synthetic_env(game_file, use_conversation_history=True)
            obs, _ = env.reset(seed=seed)

            total_reward = 0.0
            won = False

            for turn in range(self.max_turns):
                # Format observation and generate response
                messages = self._format_observation(obs)
                results = await self.model.generate_async(
                    messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                response_text = results[0].get("text", "") if results else ""

                # Parse action from response
                action = _parse_action_from_response(response_text)
                if not action:
                    # If no boxed answer found, use the raw response
                    action = response_text

                # Format action in expected format
                action_formatted = "\\boxed{{{}}}".format(action)

                # Take step in environment
                obs, reward, terminated, truncated, _ = env.step(action_formatted)
                total_reward += reward

                if terminated:
                    won = reward > 0
                    env.close()
                    return won, total_reward, turn + 1

                if truncated:
                    env.close()
                    return False, total_reward, turn + 1

            env.close()
            return False, total_reward, self.max_turns

        except Exception as e:
            logger.error("Error playing game %s: %s", game_file, e)
            return False, 0.0, 0

    async def evaluate_game(self, game_file: str) -> GameEvalResult:
        """Evaluate a single game by playing it multiple times.

        Args:
            game_file: Path to game file

        Returns:
            GameEvalResult with statistics
        """
        skill = _extract_skill_from_game_file(game_file)

        wins = []
        rewards = []
        turns = []
        errors = 0

        # Play game multiple times with different seeds
        for play_idx in range(self.plays_per_game):
            seed = play_idx * 1000 + hash(game_file) % 10000
            try:
                won, reward, num_turns = await self._play_game_once(game_file, seed)
                wins.append(1 if won else 0)
                rewards.append(reward)
                turns.append(num_turns)
            except Exception as e:
                logger.error("Error in play %d of game %s: %s", play_idx, game_file, e)
                errors += 1

        # Compute statistics
        num_plays = len(wins)
        if num_plays == 0:
            return GameEvalResult(
                game_file=game_file,
                skill=skill,
                num_plays=0,
                num_wins=0,
                pass_rate=0.0,
                variance=0.0,
                mean_reward=0.0,
                mean_turns=0,
                errors=errors,
            )

        num_wins = sum(wins)
        pass_rate = num_wins / num_plays
        variance = pass_rate * (1 - pass_rate)  # Variance of Bernoulli
        mean_reward = sum(rewards) / num_plays
        mean_turns = int(sum(turns) / num_plays)

        return GameEvalResult(
            game_file=game_file,
            skill=skill,
            num_plays=num_plays,
            num_wins=num_wins,
            pass_rate=pass_rate,
            variance=variance,
            mean_reward=mean_reward,
            mean_turns=mean_turns,
            errors=errors,
        )

    async def evaluate_all_games(
        self, game_files: List[str]
    ) -> FixedModelEvalResult:
        """Evaluate all games and compute aggregate statistics.

        Args:
            game_files: List of paths to game files

        Returns:
            FixedModelEvalResult with aggregate statistics
        """
        if not game_files:
            return FixedModelEvalResult(
                overall_pass_rate=0.0,
                overall_variance=0.0,
                difficulty_score=0.0,
                num_optimal_games=0,
                total_games=0,
                total_plays=0,
            )

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def eval_with_semaphore(game_file: str) -> Optional[GameEvalResult]:
            """Evaluate game with semaphore for rate limiting, catching exceptions."""
            async with semaphore:
                try:
                    return await self.evaluate_game(game_file)
                except Exception as e:
                    logger.error("Game evaluation failed for %s: %s", game_file, e)
                    return None

        # Evaluate all games concurrently (with rate limiting) with progress bar
        tasks = [eval_with_semaphore(gf) for gf in game_files]
        results: List[Optional[GameEvalResult]] = await tqdm_asyncio.gather(
            *tasks,
            desc="[fixed_eval] Evaluating games",
            unit="game",
        )

        # Filter out None results (failed evaluations) and collect valid results
        per_game_results = []
        total_errors = 0
        for result in results:
            if result is None:
                total_errors += 1
            else:
                per_game_results.append(result)
                total_errors += result.errors

        if not per_game_results:
            return FixedModelEvalResult(
                overall_pass_rate=0.0,
                overall_variance=0.0,
                difficulty_score=0.0,
                num_optimal_games=0,
                total_games=len(game_files),
                total_plays=0,
                errors=total_errors,
            )

        # Compute aggregate statistics
        total_wins = sum(r.num_wins for r in per_game_results)
        total_plays = sum(r.num_plays for r in per_game_results)
        overall_pass_rate = total_wins / total_plays if total_plays > 0 else 0.0
        overall_variance = overall_pass_rate * (1 - overall_pass_rate)
        difficulty_score = 1.0 - abs(overall_pass_rate - 0.5) / 0.5

        # Compute reward statistics
        all_rewards = [r.mean_reward for r in per_game_results]
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        if len(all_rewards) > 1:
            variance = sum((r - avg_reward) ** 2 for r in all_rewards) / len(all_rewards)
            std_reward = variance ** 0.5
        else:
            std_reward = 0.0

        # Count optimal games (pass rate in [0.3, 0.7])
        num_optimal = sum(1 for r in per_game_results if 0.3 <= r.pass_rate <= 0.7)

        # Compute per-skill metrics
        per_skill_metrics: Dict[str, Dict[str, Any]] = {}
        skill_wins: Dict[str, int] = {}
        skill_plays: Dict[str, int] = {}
        skill_games: Dict[str, int] = {}
        skill_rewards: Dict[str, List[float]] = {}

        for result in per_game_results:
            skill = result.skill
            if skill not in skill_wins:
                skill_wins[skill] = 0
                skill_plays[skill] = 0
                skill_games[skill] = 0
                skill_rewards[skill] = []

            skill_wins[skill] += result.num_wins
            skill_plays[skill] += result.num_plays
            skill_games[skill] += 1
            skill_rewards[skill].append(result.mean_reward)

        for skill in skill_wins:
            plays = skill_plays[skill]
            rewards = skill_rewards[skill]
            if plays > 0:
                pass_rate = skill_wins[skill] / plays
                skill_avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
                if len(rewards) > 1:
                    skill_var = sum((r - skill_avg_reward) ** 2 for r in rewards) / len(rewards)
                    skill_std_reward = skill_var ** 0.5
                else:
                    skill_std_reward = 0.0
                per_skill_metrics[skill] = {
                    "pass_rate": pass_rate,
                    "variance": pass_rate * (1 - pass_rate),
                    "difficulty_score": 1.0 - abs(pass_rate - 0.5) / 0.5,
                    "num_games": float(skill_games[skill]),
                    "avg_reward": skill_avg_reward,
                    "std_reward": skill_std_reward,
                }

        return FixedModelEvalResult(
            overall_pass_rate=overall_pass_rate,
            overall_variance=overall_variance,
            difficulty_score=difficulty_score,
            num_optimal_games=num_optimal,
            total_games=len(per_game_results),
            total_plays=total_plays,
            avg_reward=avg_reward,
            std_reward=std_reward,
            per_game_results=per_game_results,
            per_skill_metrics=per_skill_metrics,
            errors=total_errors,
        )


async def run_fixed_model_evaluation(
    games_dir: Path,
    model: str = "gpt-5-mini",
    plays_per_game: int = 8,
    max_turns: int = 30,
    max_concurrent: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    api_base_url: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> FixedModelEvalResult:
    """Run fixed model evaluation on all games in a directory.

    Convenience function that creates the evaluator and runs evaluation.

    Args:
        games_dir: Directory containing game files
        model: Model ID (e.g., "gpt-5-mini" for OpenAI)
        plays_per_game: Number of times to play each game
        max_turns: Maximum turns per game
        max_concurrent: Maximum concurrent game plays
        temperature: Generation temperature
        max_tokens: Maximum tokens per response
        api_base_url: API base URL (None for OpenAI default)
        api_key_env: Environment variable name for API key (default: OPENAI_API_KEY)

    Returns:
        FixedModelEvalResult with aggregate statistics
    """
    # Find all game files
    game_files = [str(f) for f in games_dir.glob("game_*.py")]

    if not game_files:
        logger.warning("No game files found in %s", games_dir)
        return FixedModelEvalResult(
            overall_pass_rate=0.0,
            overall_variance=0.0,
            difficulty_score=0.0,
            num_optimal_games=0,
            total_games=0,
            total_plays=0,
        )

    logger.info("Found %d games to evaluate in %s", len(game_files), games_dir)

    # Create adapter and evaluator
    adapter = create_openai_adapter(
        model=model,
        api_key_env=api_key_env,
        base_url=api_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    evaluator = FixedModelEvaluator(
        model_adapter=adapter,
        plays_per_game=plays_per_game,
        max_turns=max_turns,
        max_concurrent=max_concurrent,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Run evaluation
    result = await evaluator.evaluate_all_games(game_files)

    logger.info(
        "Fixed model evaluation complete: pass_rate=%.2f, difficulty_score=%.2f, "
        "optimal_games=%d/%d",
        result.overall_pass_rate,
        result.difficulty_score,
        result.num_optimal_games,
        result.total_games,
    )

    return result
