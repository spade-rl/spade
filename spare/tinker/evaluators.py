"""Evaluators for SPARE Tinker training.

Includes AIME evaluation for math competition problems.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import tinker
from tinker import SamplingParams
from tqdm.asyncio import tqdm_asyncio
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer

logger = logging.getLogger(__name__)


class AIMEEvaluator(SamplingClientEvaluator):
    """Evaluator for AIME math competition problems.

    Supports Pass@K evaluation where K samples are generated per problem
    and the problem is considered solved if any sample is correct.
    """

    def __init__(
        self,
        dataset_path: str,
        renderer: renderers.Renderer,
        name: str = "aime",
        n_samples: int = 4,
        max_tokens: int = 16384,
        temperature: float = 0.6,
        prompt_key: str = "prompt",
        label_key: str = "label",
    ):
        """Initialize AIME evaluator.

        Args:
            dataset_path: Path to JSONL file with problems
            renderer: Renderer for building prompts and parsing responses
            name: Name for this evaluator (used in metric keys)
            n_samples: Number of samples to generate per problem (Pass@K)
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            prompt_key: Key for prompt in JSONL
            label_key: Key for label/answer in JSONL
        """
        self.dataset_path = Path(dataset_path)
        self.renderer = renderer
        self.name = name
        self.n_samples = n_samples
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_key = prompt_key
        self.label_key = label_key
        self.problems: List[Dict[str, Any]] = []

        if self.dataset_path.exists():
            self.problems = self._load_problems()
            logger.info(f"[{self.name}] Loaded {len(self.problems)} problems from {self.dataset_path}")
        else:
            logger.warning(f"[{self.name}] Dataset not found: {self.dataset_path}")

    def _load_problems(self) -> List[Dict[str, Any]]:
        """Load problems from JSONL file."""
        problems = []
        with open(self.dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
        return problems

    async def _evaluate_problem(
        self,
        sampling_client: tinker.SamplingClient,
        problem: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate a single problem with n_samples attempts."""
        prompt = problem[self.prompt_key]
        label = str(problem[self.label_key])

        # Build generation prompt using renderer
        # prompt can be either a string or a list of messages (OpenAI format)
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            # Already in message format (list of dicts)
            messages = prompt
        model_input = self.renderer.build_generation_prompt(messages)

        # Create sampling params
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.renderer.get_stop_sequences(),
        )

        # Generate all n_samples in parallel
        sample_tasks = [
            sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            for _ in range(self.n_samples)
        ]
        responses = await asyncio.gather(*sample_tasks, return_exceptions=True)

        # Grade all responses
        correct_count = 0
        for response in responses:
            if isinstance(response, Exception):
                logger.warning(f"Error generating response: {response}")
                continue

            try:
                # Parse response tokens to get text
                tokens = response.sequences[0].tokens
                parsed_message = self.renderer.parse_response(tokens)[0]
                response_text = renderers.get_text_content(parsed_message)

                # Extract and grade answer
                try:
                    model_answer = extract_boxed(response_text)
                    is_correct = grade_answer(model_answer, label)
                    if is_correct:
                        correct_count += 1
                except ValueError:
                    # No boxed answer found
                    pass
                except Exception as e:
                    logger.debug(f"Error grading answer: {e}")

            except Exception as e:
                logger.warning(f"Error processing response: {e}")

        return {
            "correct_count": correct_count,
            "total_samples": self.n_samples,
            "any_correct": correct_count > 0,
        }

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """Run evaluation on AIME problems.

        Args:
            sampling_client: Tinker sampling client for generation

        Returns:
            Dictionary of metrics
        """
        if not self.problems:
            logger.warning(f"[{self.name}] No problems loaded, skipping evaluation")
            return {}

        total_problems = len(self.problems)
        logger.info(f"[{self.name}] Evaluating {total_problems} problems with Pass@{self.n_samples}...")

        # Evaluate all problems concurrently with progress bar
        tasks = [
            self._evaluate_problem(sampling_client, problem)
            for problem in self.problems
        ]
        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"[{self.name}] Evaluating",
            unit="problem",
        )

        # Aggregate results
        pass_at_k_correct = sum(1 for r in results if r["any_correct"])
        total_correct_samples = sum(r["correct_count"] for r in results)
        total_samples = sum(r["total_samples"] for r in results)

        # Calculate metrics
        pass_at_k = pass_at_k_correct / total_problems if total_problems > 0 else 0.0
        sample_accuracy = total_correct_samples / total_samples if total_samples > 0 else 0.0

        metrics = {
            f"{self.name}/pass@{self.n_samples}": pass_at_k,
            f"{self.name}/sample_accuracy": sample_accuracy,
            f"{self.name}/problems_solved": pass_at_k_correct,
            f"{self.name}/total_problems": total_problems,
        }

        logger.info(
            f"[{self.name}] Pass@{self.n_samples}: {pass_at_k_correct}/{total_problems} "
            f"({pass_at_k:.2%}), Sample accuracy: {sample_accuracy:.2%}"
        )

        return metrics


def create_aime_evaluators(
    renderer: renderers.Renderer,
    aime_2024_path: Optional[str] = None,
    aime_2025_path: Optional[str] = None,
    n_samples: int = 4,
    max_tokens: int = 16384,
    temperature: float = 0.6,
) -> List[SamplingClientEvaluator]:
    """Create AIME evaluators for 2024 and 2025 datasets.

    Args:
        renderer: Renderer for building prompts and parsing responses
        aime_2024_path: Path to AIME 2024 JSONL file
        aime_2025_path: Path to AIME 2025 JSONL file
        n_samples: Number of samples per problem (Pass@K)
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature

    Returns:
        List of evaluators
    """
    evaluators = []

    if aime_2024_path and Path(aime_2024_path).exists():
        evaluators.append(AIMEEvaluator(
            dataset_path=aime_2024_path,
            renderer=renderer,
            name="aime2024",
            n_samples=n_samples,
            max_tokens=max_tokens,
            temperature=temperature,
        ))

    if aime_2025_path and Path(aime_2025_path).exists():
        evaluators.append(AIMEEvaluator(
            dataset_path=aime_2025_path,
            renderer=renderer,
            name="aime2025",
            n_samples=n_samples,
            max_tokens=max_tokens,
            temperature=temperature,
        ))

    return evaluators
