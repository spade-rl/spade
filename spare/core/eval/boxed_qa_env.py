"""QaEnv variant that uses \\boxed{} format instead of <answer> tags.

Subclasses GEM's QaEnv to override the prompt template and answer
extraction to use \\boxed{LETTER} format for multiple-choice QA.
"""

import logging
import random
import re
from functools import partial
from typing import Any, Optional, SupportsFloat, Tuple

from datasets import Dataset, DatasetDict, load_dataset

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_boxed_answer
from gem.utils.qa_em import em_check

logger = logging.getLogger(__name__)


def _boxed_prompt(example, question_key: str = "question"):
    """Apply \\boxed{} prompt template to a QA example."""
    example[question_key] = (
        f"Question: {example[question_key]}\n"
        "Please reason step by step, and put your final answer within \\boxed{}. "
        "Your final answer should be of the following format: "
        "\\boxed{LETTER} where LETTER is one of ABCD."
    )
    return example


def _mcq_json_prompt(example, question_key: str = "question"):
    """Apply the Qwen3-4B-Instruct-2507 model card MCQ JSON template.

    The model card recommends, for multiple-choice questions:
      'Please show your choice in the answer field with only the choice
       letter, e.g., "answer": "C".'
    This is the format used to reproduce the official GPQA 62.0 number.
    """
    example[question_key] = (
        f"Question: {example[question_key]}\n"
        "Please reason step by step, then give your final choice. Show your "
        'choice in the "answer" field with only the choice letter (one of '
        'A, B, C, D), e.g., {"answer": "C"}.'
    )
    return example


def _extract_boxed_mcq_letter(action: str) -> Optional[str]:
    """Extract the A/B/C/D choice letter from a ``\\boxed{}`` MCQ answer,
    ROBUST to the LaTeX wrappers the model actually emits.

    The bare ``extract_last_boxed_answer`` returns the raw box content, so
    ``\\boxed{\\text{C}}`` → ``"\\text{C}"`` and ``\\boxed{C: 2.4}`` → ``"C: 2.4"``;
    ``em_check`` then compares that whole string to the gold letter ``"C"`` and
    FAILS — a false negative that scored a correct letter as wrong. (Measured:
    this cost GPQA-Diamond ~16pt on the 30B proxy — 52% → 68%, vs official 70.4.)
    We strip the common LaTeX text wrappers (\\text/\\mathrm/\\mathbf/…) and
    braces, then take the first A–D letter — which is the choice letter for every
    format the model emits (``C``, ``\\text{C}``, ``C: value``, ``\\text{D: }v``).
    """
    boxed = extract_last_boxed_answer(action)
    if not boxed:
        return None
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|bf|rm|textbf|textrm)\s*", "", boxed)
    s = s.replace("{", "").replace("}", "")
    m = re.search(r"[A-Da-d]", s)
    return m.group(0).upper() if m else None


def _extract_mcq_json_answer(action: str) -> Optional[str]:
    """Extract the choice letter from a model card MCQ JSON response.

    Robust to surrounding prose / fenced code: looks for the last
    ``"answer": "X"`` pattern, falling back to a bare ``\\boxed{X}`` if the
    model emitted that instead. Returns the uppercase letter or None.
    """
    # Robust to the formats the model actually emits: {"answer": "C"},
    # "answer": "C", answer: "B", answer: C (the word "answer" may be unquoted
    # and the letter may be bare). Take the LAST occurrence (final choice).
    matches = re.findall(r'["\']?answer["\']?\s*:\s*["\']?\s*([A-Da-d])\b', action, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    # Fallback: model emitted \boxed{X} despite the JSON instruction.
    boxed = extract_last_boxed_answer(action)
    if boxed:
        m = re.search(r"[A-Da-d]", boxed)
        if m:
            return m.group(0).upper()
    return None


class BoxedQaEnv(Env):
    """QA environment using \\boxed{} answer format.

    Same as GEM's QaEnv but uses \\boxed{LETTER} prompt and extraction
    instead of <answer> tags. Designed for multiple-choice benchmarks
    like GPQA, MMLU, etc.

    Subclasses override PROMPT_FN (dataset prompt template) and
    _extract_answer (response parsing) to support alternative formats
    such as the model card MCQ JSON (see McqJsonQaEnv).
    """

    # Class-level prompt template applied to each dataset row in __init__.
    # Subclasses override to use a different answer format.
    PROMPT_FN = staticmethod(_boxed_prompt)

    def __init__(
        self,
        dataset_name: Optional[str] = "",
        split: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        question_key: str = "question",
        answer_key: str = "answer",
        seed: int = 0,
        load_from_cache_file: bool = True,
        **_,
    ):
        super().__init__()
        self.seed = seed
        self.question_key = question_key
        self.answer_key = answer_key

        if dataset is None:
            dataset = load_dataset(dataset_name)
            logger.info("Loaded dataset: %s", dataset_name)
        if isinstance(dataset, DatasetDict):
            if split is not None:
                dataset = dataset[split]
            elif len(list(dataset.keys())) == 1:
                dataset = dataset[list(dataset.keys())[0]]
            else:
                raise ValueError(
                    f"Dataset {dataset_name} has multiple splits. "
                    f"Please specify a split: {list(dataset.keys())}"
                )
        assert isinstance(dataset, Dataset), f"Expected a Dataset, got {type(dataset)}"

        apply_fn = partial(self.PROMPT_FN, question_key=question_key)
        dataset = dataset.map(apply_fn, load_from_cache_file=load_from_cache_file)
        self.dataset = dataset.shuffle(seed=self.seed)
        self.idx = 0
        self.epoch = 0

    def _extract_answer(self, action: str) -> Optional[str]:
        """Extract the MCQ choice letter from a ``\\boxed{}`` response.

        BoxedQaEnv's prompt is multiple-choice ("LETTER is one of ABCD"), so the
        answer is always an A–D letter. Use the robust letter extractor (strips
        ``\\text{}`` etc.) — NOT the bare boxed content, which produced false
        negatives on ``\\boxed{\\text{C}}`` / ``\\boxed{C: value}``. Override in a
        subclass for non-boxed formats (see McqJsonQaEnv)."""
        return _extract_boxed_mcq_letter(action)

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        model_answer = self._extract_answer(action)
        if model_answer is None:
            is_correct = False
            reward = 0.0
        else:
            is_correct = em_check(model_answer, self._correct_answers())
            reward = 1.0 if is_correct else 0.0
        return TERMINAL_STATE, reward, True, True, {"correct": bool(is_correct)}

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            # Episode-index seeds provide deterministic full-dataset coverage.
            self.idx = seed % len(self.dataset)
        else:
            if self.idx == len(self.dataset):
                self.epoch += 1
                self.dataset = self.dataset.shuffle(seed=self.seed + self.epoch)
                self.idx = 0

        data = self.dataset[self.idx]
        self.first_obs = data[self.question_key]
        self.answer = data[self.answer_key]
        self.idx += 1
        return self.first_obs, {}

    def _correct_answers(self):
        if isinstance(self.answer, (str, float, int)):
            return [str(self.answer)]
        elif isinstance(self.answer, list):
            return [str(a) for a in self.answer]
        raise ValueError(f"Unexpected answer type: {type(self.answer)}")

    def sample_random_action(self) -> str:
        return "\\boxed{A}"

    def get_state(self) -> dict[str, Any]:
        return {"first_obs": self.first_obs, "answer": self.answer}

    def set_state(self, state: dict[str, Any]) -> None:
        self.first_obs = state["first_obs"]
        self.answer = state["answer"]


_GPQA_LETTERS = ["A", "B", "C", "D"]

# General-Reasoner's GPQA query template, preserved verbatim for comparability.
_GR_GPQA_QUERY_TEMPLATE = (
    "{question}\n\n"
    "A: {a}\n"
    "B: {b}\n"
    "C: {c}\n"
    "D: {d}\n\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
    "Please only provide the letter of the answer in the box."
)


def _build_gpqa_official_example(row, idx, shuffle_seed: int = 42):
    """Build one GPQA-Diamond MCQ from raw Idavidrein/gpqa columns using the
    verbatim General-Reasoner template (boxed-letter answer + step-by-step CoT).
    The Correct + 3 Incorrect answers are shuffled into A/B/C/D deterministically
    per question (General-Reasoner permutes per example; we seed per index).
    """
    correct = str(row["Correct Answer"]).strip()
    options = [
        correct,
        str(row["Incorrect Answer 1"]).strip(),
        str(row["Incorrect Answer 2"]).strip(),
        str(row["Incorrect Answer 3"]).strip(),
    ]
    rng = random.Random(shuffle_seed + idx)
    rng.shuffle(options)
    correct_letter = _GPQA_LETTERS[options.index(correct)]
    question = _GR_GPQA_QUERY_TEMPLATE.format(
        question=str(row["Question"]).strip(),
        a=options[0], b=options[1], c=options[2], d=options[3],
    )
    return {"question": question, "answer": correct_letter}


class GpqaOfficialMcqEnv(BoxedQaEnv):
    """GPQA-Diamond MCQ env using the General-Reasoner eval protocol verbatim:
    loads raw Idavidrein/gpqa, shuffles the four answers into A/B/C/D, and uses
    the General-Reasoner QUERY_TEMPLATE (0-shot CoT + \\boxed{LETTER} answer).
    Answer extraction is the inherited \\boxed{} parser. This matches the eval
    SPICE/SPARE aligns to (TIGER-AI-Lab/General-Reasoner gpqa_eval_qwen.py).
    """

    def __init__(
        self,
        dataset_name: str = "Idavidrein/gpqa",
        config_name: str = "gpqa_diamond",
        split: str = "train",
        shuffle_seed: int = 42,
        seed: int = 0,
        load_from_cache_file: bool = False,  # rebuild (198 rows, trivial); avoids stale prompt cache
        **_,
    ):
        # Build the official MCQ dataset ourselves (bypass BoxedQaEnv.__init__'s
        # PROMPT_FN.map path, which assumes a pre-formed question column).
        Env.__init__(self)
        self.seed = seed
        self.question_key = "question"
        self.answer_key = "answer"
        raw = load_dataset(dataset_name, config_name, split=split)
        logger.info("Loaded official GPQA dataset: %s/%s (%d)",
                    dataset_name, config_name, len(raw))
        prep = partial(_build_gpqa_official_example, shuffle_seed=shuffle_seed)
        dataset = raw.map(
            prep, with_indices=True, remove_columns=raw.column_names,
            load_from_cache_file=load_from_cache_file,
        )
        self.dataset = dataset.shuffle(seed=self.seed)
        self.idx = 0
        self.epoch = 0

    # Uses the inherited BoxedQaEnv extraction (extract_last_boxed_answer +
    # case-insensitive em_check), matching General-Reasoner's \boxed{LETTER}
    # answer format — no JSON override.


# OpenAI simple-evals GPQA query template, preserved verbatim for comparability.
_SCIEVAL_GPQA_QUERY_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before answering."
    "\n\n"
    "{question}\n\n"
    "A) {a}\n"
    "B) {b}\n"
    "C) {c}\n"
    "D) {d}"
)

# OpenAI simple-evals common.py ANSWER_PATTERN_MULTICHOICE (verbatim): tolerant
# of case, surrounding "$", and tabs around the colon — more lenient than the
# strict \boxed{} parser, so it recovers letters from "Answer: C" prose.
_SCIEVAL_ANSWER_PATTERN = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"


def _extract_scieval_answer(action: str) -> Optional[str]:
    """Extract the choice letter via the OpenAI simple-evals answer regex.

    Takes the LAST ``Answer: X`` match (the model's final line, per the
    instruction). Falls back to a bare ``\\boxed{X}`` if the model boxed
    instead. Returns the uppercase letter or None.
    """
    matches = re.findall(_SCIEVAL_ANSWER_PATTERN, action)
    if matches:
        return matches[-1].upper()
    boxed = extract_last_boxed_answer(action)
    if boxed:
        m = re.search(r"[A-Da-d]", boxed)
        if m:
            return m.group(0).upper()
    return None


def _build_gpqa_scieval_example(row, idx, shuffle_seed: int = 42):
    """Build one GPQA-Diamond MCQ from raw Idavidrein/gpqa columns using the
    verbatim OpenAI simple-evals template (0-shot CoT + 'Answer: $LETTER').
    Uses the SAME per-question A/B/C/D shuffle (seeded per index) as
    _build_gpqa_official_example, so this variant differs from the official
    (GR-boxed) variant ONLY in prompt/extraction, not in option order.
    """
    correct = str(row["Correct Answer"]).strip()
    options = [
        correct,
        str(row["Incorrect Answer 1"]).strip(),
        str(row["Incorrect Answer 2"]).strip(),
        str(row["Incorrect Answer 3"]).strip(),
    ]
    rng = random.Random(shuffle_seed + idx)
    rng.shuffle(options)
    correct_letter = _GPQA_LETTERS[options.index(correct)]
    question = _SCIEVAL_GPQA_QUERY_TEMPLATE.format(
        question=str(row["Question"]).strip(),
        a=options[0], b=options[1], c=options[2], d=options[3],
    )
    return {"question": question, "answer": correct_letter}


class GpqaScievalMcqEnv(BoxedQaEnv):
    """GPQA-Diamond MCQ env using the OpenAI simple-evals protocol verbatim:
    raw Idavidrein/gpqa + per-question A/B/C/D shuffle + the simple-evals
    QUERY_TEMPLATE_MULTICHOICE (0-shot CoT, 'Answer: $LETTER') with the
    simple-evals ANSWER_PATTERN_MULTICHOICE extraction.

    This is the "science leaderboard" triangulation variant — a more
    reasoning-eliciting prompt and a more lenient extractor than the \\boxed{}
    variants — and shares the SAME shuffle seed as GpqaOfficialMcqEnv so the
    measured delta isolates prompt/extraction (not option order).
    """

    def __init__(
        self,
        dataset_name: str = "Idavidrein/gpqa",
        config_name: str = "gpqa_diamond",
        split: str = "train",
        shuffle_seed: int = 42,
        seed: int = 0,
        load_from_cache_file: bool = False,  # rebuild (198 rows); avoid stale cache
        **_,
    ):
        Env.__init__(self)
        self.seed = seed
        self.question_key = "question"
        self.answer_key = "answer"
        raw = load_dataset(dataset_name, config_name, split=split)
        logger.info("Loaded official GPQA dataset (scieval): %s/%s (%d)",
                    dataset_name, config_name, len(raw))
        prep = partial(_build_gpqa_scieval_example, shuffle_seed=shuffle_seed)
        dataset = raw.map(
            prep, with_indices=True, remove_columns=raw.column_names,
            load_from_cache_file=load_from_cache_file,
        )
        self.dataset = dataset.shuffle(seed=self.seed)
        self.idx = 0
        self.epoch = 0

    def _extract_answer(self, action: str) -> Optional[str]:
        return _extract_scieval_answer(action)


class McqJsonQaEnv(BoxedQaEnv):
    """Multiple-choice QA using the Qwen model-card JSON answer format."""

    PROMPT_FN = staticmethod(_mcq_json_prompt)

    def _extract_answer(self, action: str) -> Optional[str]:
        return _extract_mcq_json_answer(action)

    def sample_random_action(self) -> str:
        return '{"answer": "A"}'
