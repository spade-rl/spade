"""GPQA-Diamond multiple-choice eval (avg@8, T=0.7).

Loads `Idavidrein/gpqa` (`gpqa_diamond` config) from HuggingFace. Each
problem has 4 options (A–D). Samples N completions per problem (default
N=8) at T=0.7 and reports avg@N + pass@N.

Default config:
    suites:
      gpqa_diamond:
        n_samples_per_prompt: 8
        temperature: 0.7
        top_p: 0.95
        max_tokens: 4096
        max_concurrent: 32
        # Optional: limit for smoke runs
        limit: null
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval_offline.suites.gpqa_diamond")

_BOXED_RE = re.compile(r"\\boxed\{([A-D])\}")
_LETTER_RE = re.compile(r"(?:answer is|Answer:)\s*([A-D])\b", re.IGNORECASE)


def _extract_letter(text: str) -> str | None:
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].upper()
    m2 = _LETTER_RE.search(text)
    if m2:
        return m2.group(1).upper()
    return None


def _format_prompt(question: str, options: list[str]) -> list[dict]:
    # Canonical prompt — matches spare/core/eval/boxed_qa_env.py.
    letters = "ABCD"
    opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
    user = (
        f"{question}\n\n{opts}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}. "
        "Specifically, you should provide your final answer in the format "
        "\\boxed{LETTER} where LETTER is one of ABCD."
    )
    return [{"role": "user", "content": user}]


def _load_problems() -> list[dict]:
    """Load the gpqa_diamond split (198 questions)."""
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out = []
    rng = random.Random(42)
    for i, row in enumerate(ds):
        # GPQA stores correct + 3 incorrects; shuffle deterministically.
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[j] for j in order]
        # Correct answer's new position determines the letter.
        correct_letter = "ABCD"[order.index(0)]
        out.append({
            "question_id": f"d{i}",
            "question": row["Question"],
            "options": shuffled,
            "answer": correct_letter,
        })
    return out


async def _run_async(client, cfg: dict, out_dir: Path) -> dict[str, Any]:
    problems = _load_problems()
    limit = cfg.get("limit")
    if limit:
        problems = problems[: int(limit)]
    n_samples = int(cfg.get("n_samples_per_prompt", 8))
    logger.info("[gpqa_diamond] n_problems=%d n_samples=%d", len(problems), n_samples)

    sem = asyncio.Semaphore(int(cfg.get("max_concurrent", 32)))

    async def one(i, row):
        async with sem:
            completions = await client.chat(
                messages=_format_prompt(row["question"], row["options"]),
                temperature=float(cfg.get("temperature", 0.7)),
                top_p=float(cfg.get("top_p", 0.95)),
                max_tokens=int(cfg.get("max_tokens", 4096)),
                n=n_samples,
            )
            samples = []
            for s_idx, c in enumerate(completions):
                pred = _extract_letter(c.text)
                samples.append({
                    "problem_idx": i,
                    "question_id": row["question_id"],
                    "sample_idx": s_idx,
                    "label": row["answer"],
                    "predicted": pred,
                    "correct": pred == row["answer"],
                    "response": c.text,
                    "finish_reason": c.finish_reason,
                })
            return samples

    per_problem = await asyncio.gather(*(one(i, r) for i, r in enumerate(problems)))
    rows: list[dict] = []
    for pr in per_problem:
        rows.extend(pr)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "gpqa_diamond_responses.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # avg@N = mean over (problem × sample) pairs
    avg_at_n = sum(r["correct"] for r in rows) / len(rows) if rows else 0.0
    # pass@N = fraction of problems where any sample is correct
    by_problem: dict[int, list[bool]] = {}
    for r in rows:
        by_problem.setdefault(r["problem_idx"], []).append(r["correct"])
    pass_at_n = sum(any(v) for v in by_problem.values()) / len(by_problem) if by_problem else 0.0
    return {
        f"gpqa_diamond_avg_at_{n_samples}": avg_at_n,
        f"gpqa_diamond_pass_at_{n_samples}": pass_at_n,
        # Canonical aliases (n-independent) for the table renderer.
        "gpqa_diamond_avg_at_n": avg_at_n,
        "gpqa_diamond_pass_at_n": pass_at_n,
        "gpqa_diamond_n_problems": len(by_problem),
        "gpqa_diamond_n_samples": len(rows),
    }


def run(client, cfg: dict, out_dir: Path) -> dict[str, Any]:
    return asyncio.run(_run_async(client, cfg, out_dir))
