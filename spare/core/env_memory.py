"""Environment memory buffer for memory-augmented generation.

Stores previously generated environments annotated with quality metrics
(regret scores, actor win rates, skill tags). During generation, high-quality
environments are sampled as few-shot seeds to guide the generator.
"""

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentRecord:
    """A single environment record in memory."""
    game_file: str              # Path to game .py file
    skill: str                  # Cognitive skill tag
    code_snippet: str           # Truncated game code (for prompt injection)
    actor_win_rate: float       # Mean actor win rate on this game
    regret: float = 0.0        # Hint-based regret score (if available)
    rollout_id: int = 0        # When this was generated
    metadata: Dict = field(default_factory=dict)


class EnvironmentMemory:
    """Buffer of past environments for memory-augmented generation.

    Provides two sampling strategies:
    - high_regret_seeds(): High-regret envs as positive few-shot examples
    - low_quality_examples(): Low-regret/too-easy/too-hard envs as negative examples
    """

    def __init__(
        self,
        max_size: int = 200,
        code_snippet_max_chars: int = 2000,
        seed: Optional[int] = None,
    ):
        self.max_size = max_size
        self.code_snippet_max_chars = code_snippet_max_chars
        self.records: List[EnvironmentRecord] = []
        self.rng = random.Random(seed)

    def add(
        self,
        game_file: str,
        skill: str,
        game_code: str,
        actor_win_rate: float,
        regret: float = 0.0,
        rollout_id: int = 0,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add an environment to memory."""
        snippet = game_code[:self.code_snippet_max_chars]
        record = EnvironmentRecord(
            game_file=game_file,
            skill=skill,
            code_snippet=snippet,
            actor_win_rate=actor_win_rate,
            regret=regret,
            rollout_id=rollout_id,
            metadata=metadata or {},
        )
        self.records.append(record)

        # Evict oldest if over capacity
        if len(self.records) > self.max_size:
            self.records = self.records[-self.max_size:]

    def high_regret_seeds(self, n: int = 2, skill: Optional[str] = None) -> List[EnvironmentRecord]:
        """Sample high-regret environments as positive generation seeds.

        Args:
            n: Number of seeds to sample
            skill: If provided, filter by skill

        Returns:
            List of high-regret EnvironmentRecords
        """
        candidates = self.records
        if skill:
            candidates = [r for r in candidates if r.skill == skill]

        # Filter to "good difficulty" range: win_rate in [0.15, 0.85]
        candidates = [r for r in candidates if 0.15 <= r.actor_win_rate <= 0.85]

        if not candidates:
            return []

        # Sort by regret descending, take top pool, sample from it
        candidates.sort(key=lambda r: r.regret, reverse=True)
        pool = candidates[:max(n * 3, 10)]
        return self.rng.sample(pool, min(n, len(pool)))

    def low_quality_examples(self, n: int = 1, skill: Optional[str] = None) -> List[EnvironmentRecord]:
        """Sample low-quality environments as negative examples.

        Low quality = too easy (win_rate > 0.9) or too hard (win_rate < 0.1).

        Args:
            n: Number of examples to sample
            skill: If provided, filter by skill

        Returns:
            List of low-quality EnvironmentRecords
        """
        candidates = self.records
        if skill:
            candidates = [r for r in candidates if r.skill == skill]

        # Filter to extremes
        bad = [r for r in candidates if r.actor_win_rate > 0.9 or r.actor_win_rate < 0.1]

        if not bad:
            return []

        return self.rng.sample(bad, min(n, len(bad)))

    def format_seeds_for_prompt(self, seeds: List[EnvironmentRecord]) -> str:
        """Format seed environments for injection into generation prompt."""
        if not seeds:
            return ""

        parts = []
        for i, seed in enumerate(seeds, 1):
            parts.append(
                f"### Example {i} (skill: {seed.skill}, "
                f"win_rate: {seed.actor_win_rate:.2f}, regret: {seed.regret:.2f})\n"
                f"```python\n{seed.code_snippet}\n```"
            )
        return "\n\n".join(parts)

    def format_negative_examples(self, examples: List[EnvironmentRecord]) -> str:
        """Format negative examples for injection into generation prompt."""
        if not examples:
            return ""

        parts = []
        for ex in examples:
            reason = "too easy" if ex.actor_win_rate > 0.9 else "too hard"
            parts.append(
                f"- {reason} (win_rate={ex.actor_win_rate:.2f}): "
                f"class {Path(ex.game_file).stem}"
            )
        return "Avoid generating environments like:\n" + "\n".join(parts)

    def save(self, path: Path) -> None:
        """Save memory buffer to JSON file."""
        data = [asdict(r) for r in self.records]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[MEMORY] Saved {len(self.records)} records to {path}")

    def load(self, path: Path) -> None:
        """Load memory buffer from JSON file."""
        if not path.exists():
            logger.info(f"[MEMORY] No existing memory at {path}")
            return
        with open(path) as f:
            data = json.load(f)
        self.records = [EnvironmentRecord(**r) for r in data]
        logger.info(f"[MEMORY] Loaded {len(self.records)} records from {path}")

    def __len__(self) -> int:
        return len(self.records)
