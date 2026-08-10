"""LiveCodeBench code-generation eval — wraps the official LCB package.

The OFFICIAL `livecodebench` package handles problem loading + sandbox +
grading. We do only:
1. Rollout: sample `n_samples_per_prompt` completions per problem via our client.
2. Format-convert: write a JSON in LCB's `custom_output_file` schema.
3. Invoke LCB's evaluator (`lcb_runner.runner.custom_evaluator`) as a
   subprocess and parse its metrics.

Default config:
    suites:
      livecodebench:
        n_samples_per_prompt: 1
        temperature: 0.2
        top_p: 0.95
        max_tokens: 16384
        max_concurrent: 16
        release_version: "release_v6"   # passed straight to LCB
        # Optional: limit for smoke
        limit: null

Install the official package at runtime:
    pip install --ignore-requires-python git+https://github.com/LiveCodeBench/LiveCodeBench.git
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval_offline.suites.livecodebench")

_FENCE_RE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Pull the LAST ```python ... ``` block from text. Fallback: return text."""
    matches = _FENCE_RE.findall(text)
    if matches:
        return matches[-1]
    return text


def _format_prompt(question_content: str, starter_code: str) -> list[dict]:
    user_parts = [question_content]
    if starter_code:
        user_parts.append("\nStarter code:\n```python\n" + starter_code + "\n```")
    user_parts.append(
        "\nWrite the complete solution in a single ```python ... ``` block."
    )
    return [{"role": "user", "content": "\n".join(user_parts)}]


def _load_problems(release_version: str = "release_v6") -> list[dict]:
    """Load LCB problems via the official package.

    Returns a list of dicts with at minimum: question_id, question_content,
    starter_code. Additional fields from LCB's benchmark are preserved
    untouched — the LCB evaluator pulls what it needs from `question_id`.
    """
    # Import lazily so missing-dep failure surfaces here, not at module load.
    from lcb_runner.benchmarks import code_generation  # type: ignore[import-untyped]

    problems = code_generation.load_code_generation_dataset(
        release_version=release_version,
    )
    out: list[dict] = []
    for p in problems:
        # `p` is an LCB CodeGenerationProblem dataclass; pull the fields we need
        # for our prompt rendering, but the evaluator will pull from its own
        # benchmark cache by question_id.
        out.append({
            "question_id": p.question_id,
            "question_content": p.question_content,
            "starter_code": getattr(p, "starter_code", "") or "",
        })
    return out


def _invoke_lcb_evaluator(
    *,
    generations_path: Path,
    scenario: str,
    release_version: str,
    work_dir: Path,
) -> dict[str, Any]:
    """Run `python -m lcb_runner.runner.custom_evaluator ...` and parse its metrics.

    The LCB runner writes a metrics JSON into `work_dir`. We read it back.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "lcb_runner.runner.custom_evaluator",
        "--custom_output_file", str(generations_path),
        "--scenario", scenario,
        "--release_version", release_version,
    ]
    # Run from the LCB source root so its relative prompt assets resolve.
    import os as _os
    lcb_root = _os.environ.get("LCB_ROOT")
    if not lcb_root:
        for p in ("/opt/lcb", "/workspace/lcb"):
            if Path(p).is_dir():
                lcb_root = p
                break
    if not lcb_root:
        try:
            import lcb_runner as _lcb  # type: ignore
            lcb_root = str(Path(_lcb.__file__).resolve().parent.parent)
        except Exception:
            lcb_root = str(work_dir)
    logger.info("[livecodebench] invoking LCB: %s  (cwd=%s)", " ".join(cmd), lcb_root)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=lcb_root)
    if proc.returncode != 0:
        raise RuntimeError(
            f"LCB evaluator failed (exit {proc.returncode}):\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        )
    # LCB writes its score JSON as `<output_stem>_<scenario>_output_eval.json`
    # alongside the generations file. Older versions used `*_metrics.json`;
    # check both for safety.
    candidates: list[Path] = []
    for d in (work_dir, generations_path.parent):
        candidates += sorted(d.glob(f"*_{scenario}_output_eval.json"))
        candidates += sorted(d.glob("*_metrics.json"))
    if lcb_root and Path(lcb_root).is_dir():
        candidates += sorted(Path(lcb_root).glob(f"*_{scenario}_output_eval.json"))
        candidates += sorted(Path(lcb_root).glob("*_metrics.json"))
    # Dedup while preserving order
    seen, dedup = set(), []
    for p in candidates:
        if p not in seen:
            seen.add(p); dedup.append(p)
    candidates = dedup
    if not candidates:
        raise RuntimeError(
            f"LCB evaluator produced no *_metrics.json (checked work_dir={work_dir}, "
            f"generations_dir={generations_path.parent}, lcb_root={lcb_root}); "
            f"stdout tail:\n{proc.stdout[-500:]}"
        )
    metrics_path = candidates[-1]
    data = json.loads(metrics_path.read_text())
    # Accept both list-wrapped and flat legacy metrics payloads.
    if isinstance(data, list):
        # Accept the official metrics-first payload and older single-item forms.
        if len(data) > 1:
            head = data[0] if data else {}
            if isinstance(head, dict) and "pass@1" in head:
                data = head
            else:
                raise RuntimeError(
                    f"LCB metrics file has {len(data)} entries and head lacks "
                    f"'pass@1'; unrecognized payload. File: {metrics_path}"
                )
        else:
            data = data[0] if data else {}
    if not isinstance(data, dict):
        raise RuntimeError(
            f"LCB metrics file has unexpected shape: {type(data)} from {metrics_path}"
        )
    return data


async def _run_async(client: Any, cfg: dict, out_dir: Path) -> dict[str, Any]:
    release_version = cfg.get("release_version", "release_v6")
    problems = _load_problems()
    limit = cfg.get("limit")
    if limit:
        problems = problems[: int(limit)]
    n_samples = int(cfg.get("n_samples_per_prompt", 1))
    logger.info(
        "[livecodebench] release=%s n_problems=%d n_samples=%d",
        release_version, len(problems), n_samples,
    )

    sem = asyncio.Semaphore(int(cfg.get("max_concurrent", 16)))

    async def one(row: dict) -> dict:
        async with sem:
            completions = await client.chat(
                messages=_format_prompt(row["question_content"], row["starter_code"]),
                temperature=float(cfg.get("temperature", 0.2)),
                top_p=float(cfg.get("top_p", 0.95)),
                max_tokens=int(cfg.get("max_tokens", 16384)),
                n=n_samples,
            )
            code_list = [_extract_code(c.text) for c in completions]
            return {
                "question_id": row["question_id"],
                "code_list": code_list,
                "responses": [c.text for c in completions],
            }

    payloads = await asyncio.gather(*(one(r) for r in problems))

    out_dir.mkdir(parents=True, exist_ok=True)
    generations_path = out_dir / "lcb_generations.json"
    # LCB's custom_evaluator expects entries containing `code_list` + `question_id`.
    # We drop `responses` from the JSON we send to LCB (not in their schema),
    # but keep a separate file for inspection.
    eval_input = [{"question_id": p["question_id"], "code_list": p["code_list"]}
                  for p in payloads]
    generations_path.write_text(json.dumps(eval_input))
    (out_dir / "lcb_responses.jsonl").open("w").writelines(
        json.dumps(p) + "\n" for p in payloads
    )

    lcb_metrics = _invoke_lcb_evaluator(
        generations_path=generations_path,
        scenario="codegeneration",
        release_version=release_version,
        work_dir=out_dir,
    )

    # Map LCB's metric keys into our flat dict.
    out: dict[str, Any] = {
        "livecodebench_pass_at_1": float(lcb_metrics.get("pass@1", 0.0)),
        "livecodebench_n_problems": len(problems),
        "livecodebench_n_samples": len(problems) * n_samples,
    }
    # Forward per-difficulty pass@1 if LCB reported them.
    for difficulty in ("easy", "medium", "hard"):
        key = f"{difficulty}_pass@1"
        if key in lcb_metrics:
            out[f"livecodebench_{difficulty}_pass_at_1"] = float(lcb_metrics[key])
    return out


def run(client: Any, cfg: dict, out_dir: Path) -> dict[str, Any]:
    return asyncio.run(_run_async(client, cfg, out_dir))
