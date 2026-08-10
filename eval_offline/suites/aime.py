"""AIME math eval (2024 / 2025 / 2026).

Reads JSONL with `{"prompt": [...messages...], "label": "<int>"}` rows,
samples N completions per problem at temp=1.0, scores via math_verify, and
reports pass@1 + pass@k per year and overall.

Default config:
    suites:
      aime:
        years: [2024, 2025, 2026]
        n_samples_per_prompt: 64
        temperature: 1.0
        top_p: 0.95
        top_k: 20
        max_tokens: 32768
        # Optional: override path template. Defaults to
        #   ${WORKSPACE_DIR}/aime-<year>/aime-<year>.jsonl
        path_template: null
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval_offline.suites.aime")


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def _extract_boxed(text: str) -> str | None:
    """Pull the contents of the last \\boxed{...} in `text`, or None."""
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _verify_answer(predicted: str | None, label: str) -> bool:
    """Compare predicted vs label. Try math_verify (LaTeX-aware) first;
    fall back to string-normalised int compare."""
    if predicted is None:
        return False
    try:
        from math_verify import parse, verify
        gold = parse(f"\\boxed{{{label}}}")
        pred = parse(f"\\boxed{{{predicted}}}")
        return bool(verify(gold, pred))
    except Exception:
        pass
    # Fallback: strip whitespace + try int compare.
    try:
        return int(predicted.strip()) == int(label.strip())
    except (ValueError, TypeError):
        return predicted.strip() == label.strip()


def _default_path(year: int) -> Path:
    workspace = os.environ.get("WORKSPACE_DIR", "/workspace/spare-workspace")
    return Path(workspace) / f"aime-{year}" / f"aime-{year}.jsonl"


def _load_problems(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def _score_one_year(
    client,
    year: int,
    path: Path,
    *,
    n_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    out_dir: Path,
) -> dict[str, Any]:
    problems = _load_problems(path)
    logger.info("[aime] year=%s path=%s n_problems=%d", year, path, len(problems))

    async def one_problem(idx, row):
        # Pass top_k via extra_body since OpenAI client doesn't accept it natively.
        completions = await client.chat(
            messages=row["prompt"],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n_samples,
            extra_body={"top_k": top_k},
        )
        results = []
        for s_idx, c in enumerate(completions):
            pred = _extract_boxed(c.text)
            correct = _verify_answer(pred, str(row["label"]))
            results.append({
                "year": year,
                "problem_idx": idx,
                "sample_idx": s_idx,
                "label": row["label"],
                "predicted": pred,
                "correct": correct,
                "response": c.text,
                "finish_reason": c.finish_reason,
                "completion_tokens": c.completion_tokens,
            })
        return results

    tasks = [one_problem(i, row) for i, row in enumerate(problems)]
    per_problem = await asyncio.gather(*tasks)

    rows: list[dict] = []
    for pr in per_problem:
        rows.extend(pr)

    # Persist per-sample JSONL for re-scoring/inspection.
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"aime{year}_responses.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Metrics: pass@1 = mean correct across all samples;
    # pass@k = mean over problems of (any sample correct).
    pass_at_1 = sum(r["correct"] for r in rows) / len(rows) if rows else 0.0
    by_problem = {}
    for r in rows:
        by_problem.setdefault(r["problem_idx"], []).append(r["correct"])
    pass_at_k = (
        sum(any(v) for v in by_problem.values()) / len(by_problem)
        if by_problem else 0.0
    )
    return {
        f"aime{year}_pass1": pass_at_1,
        f"aime{year}_pass_at_{n_samples}": pass_at_k,
        f"aime{year}_n_problems": len(by_problem),
        f"aime{year}_n_samples": len(rows),
    }


def run(client, cfg: dict, out_dir: Path) -> dict[str, Any]:
    years = cfg.get("years", [2024, 2025, 2026])
    n_samples = int(cfg.get("n_samples_per_prompt", 64))
    temperature = float(cfg.get("temperature", 1.0))
    top_p = float(cfg.get("top_p", 0.95))
    top_k = int(cfg.get("top_k", 20))
    max_tokens = int(cfg.get("max_tokens", 32768))
    path_template = cfg.get("path_template")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve which years to actually score (skip missing files).
    year_paths: list[tuple[int, Path]] = []
    metrics: dict[str, Any] = {}
    for year in years:
        if path_template:
            path = Path(path_template.format(year=year))
        else:
            path = _default_path(year)
        if not path.is_file():
            logger.warning("[aime] year %s skipped — file not found: %s", year, path)
            metrics[f"aime{year}_skipped"] = 1
            continue
        year_paths.append((year, path))

    # Run all years concurrently within one event loop.
    async def _score_all() -> list[dict[str, Any]]:
        tasks = [
            _score_one_year(
                client, year, path,
                n_samples=n_samples,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                out_dir=out_dir,
            )
            for year, path in year_paths
        ]
        return await asyncio.gather(*tasks)

    if year_paths:
        t0 = time.time()
        per_year_metrics = asyncio.run(_score_all())
        total_elapsed = time.time() - t0
        pass1_per_year = []
        for (year, _path), year_metrics in zip(year_paths, per_year_metrics):
            year_metrics[f"aime{year}_elapsed_sec"] = total_elapsed
            metrics.update(year_metrics)
            if f"aime{year}_pass1" in year_metrics:
                pass1_per_year.append(year_metrics[f"aime{year}_pass1"])
            logger.info("[aime] year=%s metrics=%s", year, year_metrics)
        logger.info("[aime] all years done in %.1fs (parallel)", total_elapsed)
    else:
        pass1_per_year = []

    if pass1_per_year:
        metrics["aime_overall_pass1"] = sum(pass1_per_year) / len(pass1_per_year)

    (out_dir / "scores.json").write_text(json.dumps(metrics, indent=2))
    return metrics
