"""Offline-eval driver: spin up SGLang once, run all configured suites,
aggregate results.

Usage (inside container):
    python -m eval_offline.run_offline_eval \\
        --ckpt /path/to/checkpoint \\
        --config eval_offline/configs/games.yaml \\
        --output-dir /scratch/offline_eval/<run>

See `eval_offline/README.md` for usage and benchmark coverage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import yaml

from eval_offline.ckpt_resolver import resolve_ckpt
from eval_offline.client import OfflineClient
from eval_offline.results import RunResult, SuiteResult, now_iso
from eval_offline.server import sglang_server
from eval_offline.suites import load_suite


@contextmanager
def _remote_server(base_url: str, model_name: str):
    """Use a remote endpoint without starting or stopping a local server."""
    yield base_url.rstrip("/"), model_name

logger = logging.getLogger("eval_offline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline eval driver")
    p.add_argument("--ckpt", required=False, default=None,
                   help="Local HF dir OR Hub repo id (org/name). "
                        "Required unless --base-url is given.")
    p.add_argument("--base-url", default=None,
                   help="Use an existing OpenAI-compatible endpoint instead "
                        "of launching SGLang locally. When "
                        "set, --ckpt is optional and only used as a label.")
    p.add_argument("--served-model-name", default=None,
                   help="model field to send in chat requests when using "
                        "--base-url. Defaults to the --ckpt string.")
    p.add_argument("--config", required=True,
                   help="Path to a YAML config (see eval_offline/configs/)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write results. Default: "
                        "$SPARE_EVAL_ROOT/<model>/<timestamp> or runs/offline_eval/...")
    p.add_argument("--suites", default=None,
                   help="Comma-separated list overriding the config's enabled suites")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dp", type=int, default=None,
                   help="Default = $SLURM_GPUS_ON_NODE / tp, else 1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-group", default="offline-eval")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--max-concurrent", type=int, default=64)
    p.add_argument("--server-startup-timeout", type=int, default=600)
    p.add_argument(
        "--mem-fraction-static", type=float, default=0.7,
        help="SGLang --mem-fraction-static")
    p.add_argument("--smoke", action="store_true",
                   help="Just spin up the server, hit /v1/models, exit 0")
    return p.parse_args()


def _default_dp(tp: int) -> int:
    n = os.environ.get("SLURM_GPUS_ON_NODE")
    if n and n.isdigit():
        return max(1, int(n) // tp)
    return 1


def _default_output_dir(ckpt: str) -> Path:
    name = ckpt.replace("/", "__").replace(":", "_")
    root = Path(os.environ.get("SPARE_EVAL_ROOT", "runs/offline_eval"))
    return root / name / time.strftime("%Y%m%d_%H%M%S")


def _read_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "suites" not in cfg:
        raise ValueError(f"{path} is missing top-level `suites:` key")
    return cfg


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    config_path = Path(args.config).resolve()
    cfg = _read_config(config_path)

    enabled = list(cfg["suites"].keys())
    if args.suites:
        enabled = [s.strip() for s in args.suites.split(",") if s.strip()]
    logger.info("[driver] enabled suites: %s", enabled)

    if args.base_url is None and args.ckpt is None:
        raise SystemExit("[driver] must pass either --ckpt or --base-url")

    ckpt_label = args.ckpt or args.base_url
    out_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(ckpt_label)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[driver] output_dir: %s", out_dir)

    # Resolve checkpoint only when launching SGLang locally.
    if args.base_url:
        ckpt_path = Path(args.ckpt) if args.ckpt else None
        logger.info("[driver] using remote server %s (model=%s)",
                    args.base_url, args.served_model_name or ckpt_label)
    else:
        ckpt_path = resolve_ckpt(args.ckpt)

    # WandB (optional).
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "spade"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                group=args.wandb_group,
                name=args.wandb_name or ckpt_label.replace("/", "__"),
                config={
                    "ckpt": args.ckpt,
                    "ckpt_resolved": str(ckpt_path),
                    "config_path": str(config_path),
                    "suites": enabled,
                    "tp": args.tp,
                    "dp": args.dp or _default_dp(args.tp),
                },
                reinit=True,
            )
            # Save the YAML config as an artifact.
            wandb.save(str(config_path), policy="now")
        except Exception as e:
            logger.warning("[driver] WandB init failed: %s — continuing without it", e)
            wandb_run = None

    run = RunResult(
        ckpt=ckpt_label,
        ckpt_resolved=str(ckpt_path) if ckpt_path else (args.base_url or ""),
        started_at=now_iso(),
    )
    t_total_start = time.time()

    server_log = out_dir / "server.log"
    dp = args.dp if args.dp is not None else _default_dp(args.tp)

    if args.base_url:
        server_ctx = _remote_server(
            args.base_url, args.served_model_name or ckpt_label
        )
    else:
        server_ctx = sglang_server(
            ckpt_path,
            host=args.host,
            port=args.port,
            tp=args.tp,
            dp=dp,
            mem_fraction_static=args.mem_fraction_static,
            log_path=server_log,
            startup_timeout=args.server_startup_timeout,
        )

    with server_ctx as (base_url, model_name):
        if args.smoke:
            logger.info("[driver] --smoke: server reachable at %s, model=%s. Exiting.",
                        base_url, model_name)
            return 0

        client = OfflineClient(
            base_url=base_url,
            model=model_name,
            max_concurrent=args.max_concurrent,
        )

        # API-bound suites may run concurrently with GPU-heavy suites.
        sequential: list[str] = []
        concurrent_ones: list[str] = []
        for s in enabled:
            sc = cfg["suites"].get(s) or {}
            if isinstance(sc, dict) and sc.get("concurrent", False):
                concurrent_ones.append(s)
            else:
                sequential.append(s)
        if concurrent_ones:
            logger.info("[driver] concurrent suites (background): %s", concurrent_ones)
            logger.info("[driver] sequential suites (foreground): %s", sequential)

        def _run_one(suite_name: str) -> SuiteResult:
            suite_out = out_dir / suite_name
            suite_cfg = cfg["suites"].get(suite_name) or {}
            t0 = time.time()
            try:
                fn = load_suite(suite_name)
                logger.info("[driver] running suite: %s", suite_name)
                metrics = fn(client, suite_cfg, suite_out)
                if not isinstance(metrics, dict):
                    raise TypeError(
                        f"suite {suite_name} returned {type(metrics)}, expected dict"
                    )
                skipped = bool(metrics.pop("skipped", False))
                err = None
            except Exception as e:
                metrics = {}
                err = "".join(traceback.format_exception(e))
                suite_out.mkdir(parents=True, exist_ok=True)
                (suite_out / "error.log").write_text(err)
                skipped = False
                logger.exception("[driver] suite %s raised", suite_name)
            elapsed = time.time() - t0
            return SuiteResult(
                name=suite_name, metrics=metrics, elapsed_sec=elapsed,
                skipped=skipped, error=err,
            )

        def _record(sr: SuiteResult) -> None:
            run.add(sr)
            tag = " [SKIPPED]" if sr.skipped else (" [ERROR]" if sr.error else "")
            logger.info(
                "[driver] %s done in %.1fs%s metrics=%s",
                sr.name, sr.elapsed_sec, tag, sr.metrics,
            )
            if wandb_run:
                try:
                    flat = {f"eval/{sr.name}/{k}": v for k, v in sr.metrics.items()}
                    flat[f"eval/{sr.name}/elapsed_sec"] = sr.elapsed_sec
                    if sr.skipped:
                        flat[f"eval/{sr.name}/skipped"] = 1
                    wandb_run.log(flat)
                except Exception as e:
                    logger.warning("[driver] WandB log failed for %s: %s", sr.name, e)

        # Kick off concurrent suites in background threads (suites do their own
        # asyncio.run internally, so threads are correct).
        executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(concurrent_ones)),
                thread_name_prefix="bg-suite",
            )
            if concurrent_ones else None
        )
        bg_futures = {
            executor.submit(_run_one, s): s for s in concurrent_ones
        } if executor else {}

        for suite_name in sequential:
            _record(_run_one(suite_name))

        for fut, suite_name in bg_futures.items():
            try:
                _record(fut.result())
            except Exception:
                logger.exception("[driver] background suite %s crashed", suite_name)
        if executor:
            executor.shutdown(wait=False)

    run.elapsed_sec = time.time() - t_total_start
    results_path = out_dir / "results.json"
    run.write(results_path)
    logger.info("[driver] wrote %s (total %.1fs)", results_path, run.elapsed_sec)

    if wandb_run:
        try:
            import wandb
            wandb.save(str(results_path), policy="now")
            wandb_run.summary.update({
                "total_elapsed_sec": run.elapsed_sec,
                "num_suites_run": len(run.suites),
                "num_suites_errored": sum(1 for s in run.suites.values() if s.error),
                "num_suites_skipped": sum(1 for s in run.suites.values() if s.skipped),
            })
            wandb_run.finish()
        except Exception as e:
            logger.warning("[driver] WandB finalize failed: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
