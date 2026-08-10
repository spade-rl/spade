"""Corpus loader and document sampler for corpus-grounded game generation.

Loads JSONL corpus files (SPICE-compatible format) and provides uniform
random sampling of documents for grounding game generation prompts.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class CorpusDocument:
    """A single document from the corpus."""
    text: str
    doc_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class CorpusLoader:
    """Loads and samples documents from a JSONL corpus file.

    Supports two JSONL formats:
    - {"orig_doc": "text..."} (SPICE / data.chunk format)
    - {"text": "text..."} (Nemotron / generic format)

    Documents are truncated to max_doc_tokens words on load.
    Sampling is uniform random with a configurable seed.
    """

    def __init__(
        self,
        corpus_path: str,
        max_doc_tokens: int = 6000,
        seed: int = 42,
    ):
        self.corpus_path = Path(corpus_path)
        self.max_doc_tokens = max_doc_tokens
        self.seed = seed
        self.documents: List[CorpusDocument] = []
        self._rng = random.Random(seed)

    def load(self) -> int:
        """Load corpus from JSONL file.

        Returns:
            Number of documents successfully loaded.
        """
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        loaded = 0
        skipped = 0

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                # Extract text from either format
                text = obj.get("orig_doc") or obj.get("text") or ""
                if not isinstance(text, str):
                    text = str(text)
                text = text.strip()

                if not text:
                    skipped += 1
                    continue

                # Truncate to max word count
                text = self._truncate(text)

                # Preserve metadata (everything except the text field)
                metadata = {k: v for k, v in obj.items() if k not in ("orig_doc", "text")}

                self.documents.append(
                    CorpusDocument(text=text, doc_id=loaded, metadata=metadata)
                )
                loaded += 1

        logger.info(
            f"[CORPUS] Loaded {loaded} documents from {self.corpus_path}"
            f" (skipped {skipped} invalid lines)"
        )
        return loaded

    def sample(self, n: int = 1) -> List[CorpusDocument]:
        """Sample n documents uniformly at random.

        Uses replacement if n > len(documents), otherwise without replacement.
        """
        if not self.documents:
            raise RuntimeError("Corpus is empty. Call load() first.")

        if n > len(self.documents):
            return self._rng.choices(self.documents, k=n)
        return self._rng.sample(self.documents, n)

    def _truncate(self, text: str) -> str:
        """Truncate text to approximately max_doc_tokens (word-level proxy)."""
        words = text.split()
        if len(words) <= self.max_doc_tokens:
            return text
        truncated = " ".join(words[: self.max_doc_tokens])
        return truncated + f"\n[Document truncated at ~{self.max_doc_tokens} words]"

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return (
            f"CorpusLoader(path={self.corpus_path}, "
            f"docs={len(self.documents)}, "
            f"max_tokens={self.max_doc_tokens})"
        )
