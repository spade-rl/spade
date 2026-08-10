"""Corpus-grounded orchestrator: generates games from corpus documents.

Subclass of SpareOrchestrator that overrides _generate_single_game_async
to inject a sampled corpus document into the game generation prompt.
All other behavior (gameplay, rewards, validation) is inherited unchanged.
"""

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from spare.core.corpus import CorpusLoader
from spare.core.env_memory import EnvironmentMemory
from spare.core.game_generator import SyntheticGameGenerator
from spare.core.game_policy import GamePolicy
from spare.core.learning_potential import LearningPotential
from spare.core.model_adapter import ModelAdapter
from spare.core.orchestrator import SpareOrchestrator
from spare.core.prompts.corpus_template import generate_corpus_grounded_prompt
from spare.core.prompts.tool_use_template import (
    TOOL_USE_SKILLS,
    get_env_gen_prompt_fn,
)
from spare.core.types import SpareConfig, Trajectory
from spare.core.orchestrator import validate_game_async
from spare.core.utils import (
    build_env_trajectory,
    extract_game_code,
    save_game_file,
    save_rejected_game,
)
from spare.core.envs.synthetic_game_env import criteria_throw_at_reset

logger = logging.getLogger(__name__)


class CorpusGroundedOrchestrator(SpareOrchestrator):
    """Orchestrator that grounds game generation in corpus documents.

    For each game generation attempt, a document is sampled from the corpus
    and included in the generation prompt. The LLM is instructed to create
    a game that tests understanding of the document's content.

    All other orchestration logic (gameplay, rewards, validation, self-judge)
    is inherited from SpareOrchestrator.
    """

    def __init__(
        self,
        model: ModelAdapter,
        config: SpareConfig,
        learning_potentials: Dict[str, LearningPotential],
        game_policy: GamePolicy,
        corpus: CorpusLoader,
        env_memory: Optional[EnvironmentMemory] = None,
    ):
        super().__init__(model, config, learning_potentials, game_policy, env_memory=env_memory)
        self.corpus = corpus

    async def _generate_single_game_async(
        self,
        skill: str,
        difficulty: str,
        games_dir: Path,
        rollout_id: int,
        index: int,
        validate: bool,
        cache_step_dir: Optional[Path],
        max_attempts: int = 3,
    ) -> Optional[Tuple[Path, Trajectory]]:
        """Generate a single game grounded in a corpus document.

        Overrides the parent method to inject a sampled document into the
        game generation prompt. All other logic (validation, retry, caching)
        is identical to the parent.
        """
        game_file = None

        # Match the dispatch in SpareOrchestrator._generate_single_game_async
        # so the corpus path supports tool-use in addition to cognitive games.
        game_type = getattr(self.config, "game_type", "cognitive")
        template_name = "qwen3_game_generation"

        for attempt in range(max_attempts):
            # Sample a fresh document for each attempt
            doc = self.corpus.sample(1)[0]

            game_type = getattr(self.config, "game_type", "cognitive")
            template_name = self.config.env_generation_template

            if game_type == "tool_use":
                skill_info = TOOL_USE_SKILLS[skill]
                # Dispatch single-task vs multi-turn variant via SPARE_MULTITURN_ENV_GEN env var
                prompt_content = get_env_gen_prompt_fn()(skill, skill_info, difficulty, doc.text)
                if template_name == "qwen3_game_generation":
                    template_name = "qwen3_tool_use_game_generation"
            else:
                skill_info = SyntheticGameGenerator.COGNITIVE_SKILLS[skill]
                prompt_content = generate_corpus_grounded_prompt(
                    skill, skill_info, difficulty, doc.text
                )

            # Memory augmentation: inject high-regret seeds as few-shot examples
            prompt_content = self._augment_prompt_with_memory(prompt_content, skill)

            messages: List[Dict[str, str]] = self._apply_game_template(
                messages=[{"role": "user", "content": prompt_content}],
                template_name=template_name,
            )
            all_tokens: List[int] = self.model.apply_template(messages)
            all_masks: List[int] = []
            all_logprobs: List[float] = []
            assistant_responses: List[str] = []

            try:
                result = await asyncio.wait_for(
                    self.model.generate_async(
                        messages=messages,
                        input_ids=all_tokens,
                        temperature=self.config.env_temperature,
                        top_p=self.config.env_top_p,
                        top_k=self.config.env_top_k,
                        max_tokens=self.config.env_max_tokens,
                        role="environment",
                    ),
                    timeout=300,
                )
                result = result[0] if isinstance(result, list) else result

                response_tokens = result["token_ids"]
                all_tokens.extend(response_tokens)
                all_masks.extend([1] * len(response_tokens))
                all_logprobs.extend(result["logprobs"])
                assistant_responses.append(result["text"])
                response_text = result["text"]
                game_code = extract_game_code(response_text)

                game_file = save_game_file(
                    game_code, games_dir, rollout_id, index, skill
                )

                if validate and not await validate_game_async(game_file):
                    raise RuntimeError("Game validation failed")

                # Optional gate for criteria that fail on every tested reset state.
                # Rollout zero skips it because no fallback pool exists yet.
                _reset_gate_on = (
                    os.environ.get("SPARE_RESET_GATE", "") not in ("", "0", "false", "False")
                    and rollout_id > 0
                )
                if _reset_gate_on and criteria_throw_at_reset(str(game_file), self.config.max_turns):
                    self._env_validator_rejected += 1
                    _reason = ("Reset-solvability gate: a success criterion raises on the "
                               "initial state at every tested seed (unsolvable step).")
                    save_rejected_game(
                        game_code, games_dir, rollout_id, index, skill,
                        reject_stage="criteria_throw_at_reset", reasoning=_reason,
                    )
                    raise RuntimeError(_reason)

                # LLM-based environment validation (rejection sampling)
                if self.env_validator is not None:
                    is_valid, reasoning = await asyncio.wait_for(
                        self.env_validator.validate_async(game_code),
                        timeout=120,
                    )
                    if not is_valid:
                        self._env_validator_rejected += 1
                        # Persist rejected game + FULL reasoning for offline audit
                        # (validator-over-reject vs generator-makes-impossible).
                        save_rejected_game(
                            game_code, games_dir, rollout_id, index, skill,
                            reject_stage="env_validator", reasoning=reasoning,
                        )
                        reason_preview = reasoning[:200].replace("\n", " ")
                        raise RuntimeError(
                            f"Environment validator rejected game: {reason_preview}"
                        )
                    self._env_validator_accepted += 1
                    logger.info(
                        f"[ENV-VALIDATOR] Game {index} (skill={skill}) passed validation"
                    )

                env_traj = build_env_trajectory(
                    messages=messages,
                    all_tokens=all_tokens,
                    all_masks=all_masks,
                    all_logprobs=all_logprobs,
                    assistant_responses=assistant_responses,
                    skill=skill,
                    difficulty=difficulty,
                    game_code=game_code,
                    index=index,
                )

                # Add corpus metadata for logging / debugging
                env_traj.metadata["corpus_doc_id"] = doc.doc_id
                env_traj.metadata["corpus_doc_snippet"] = doc.text[:200]

                if cache_step_dir is not None:
                    shutil.copy2(
                        str(game_file), str(cache_step_dir / game_file.name)
                    )

                logger.info(
                    f"[GEN-CORPUS] Game {index} (skill={skill}) generated from "
                    f"doc {doc.doc_id} (attempt {attempt + 1})"
                )
                return game_file, env_traj

            except Exception as e:
                logger.warning(
                    f"[GEN-CORPUS] Game {index} (skill={skill}) attempt "
                    f"{attempt + 1}/{max_attempts} failed: {e}"
                )
                if game_file and game_file.exists():
                    game_file.unlink(missing_ok=True)
                continue

        logger.error(
            f"[GEN-CORPUS] Game {index} (skill={skill}) failed after "
            f"{max_attempts} attempts"
        )
        return None
