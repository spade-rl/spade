"""Manage a single SGLang HTTP server subprocess for offline eval.

Spawns `python -m sglang.launch_server`, waits for /health, exposes the URL,
and shuts it down on context exit.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


@contextmanager
def sglang_server(
    model_path: Path,
    *,
    port: int = 30000,
    host: str = "127.0.0.1",
    tp: int = 1,
    dp: int = 1,
    mem_fraction_static: float = 0.7,
    tool_call_parser: str | None = "qwen",
    log_path: Path | None = None,
    startup_timeout: int = 600,
    extra_args: list[str] | None = None,
):
    """Yield (base_url, model_name) once the server is ready.

    On exit (success or exception), terminates the subprocess.
    """
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", str(model_path),
        "--host", host, "--port", str(port),
        "--tp", str(tp), "--dp", str(dp),
        "--mem-fraction-static", str(mem_fraction_static),
        "--trust-remote-code",
    ]
    # Optionally pin the OpenAI `model` name SGLang serves under. Needed when a
    # downstream harness sends a fixed model id (e.g. BFCL's qwen3-4b-FC handle
    # sends model="qwen3-4b") that must match the served name. Default behaviour
    # (unset) leaves it as the model-path string, which the driver then queries.
    _served_name = os.getenv("SGLANG_SERVED_MODEL_NAME")
    if _served_name:
        cmd += ["--served-model-name", _served_name]
    if tool_call_parser:
        cmd += ["--tool-call-parser", tool_call_parser]
    if extra_args:
        cmd += list(extra_args)

    logger.info("[server] launching: %s", " ".join(cmd))
    log_fp = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        # New process group so we can clean up children too.
        start_new_session=True,
    )

    base_url = f"http://{host}:{port}"
    try:
        _wait_for_health(base_url, timeout=startup_timeout, proc=proc)
        # served_model_name defaults to the model-path string; query it.
        model_name = _query_model_name(base_url)
        logger.info(
            "[server] ready at %s (served_model_name=%s)", base_url, model_name
        )
        yield base_url, model_name
    finally:
        logger.info("[server] terminating SGLang (pid=%s)", proc.pid)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("[server] SGLang didn't exit in 30s; SIGKILL")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        if log_path:
            log_fp.close()


def _wait_for_health(base_url: str, *, timeout: int, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout
    last_err = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"SGLang exited early with code {proc.returncode} before health OK; "
                f"see server log."
            )
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(2)
    raise TimeoutError(
        f"SGLang /health did not return 200 within {timeout}s. Last error: {last_err}"
    )


def _query_model_name(base_url: str) -> str:
    r = requests.get(f"{base_url}/v1/models", timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise RuntimeError("/v1/models returned empty data list")
    return data[0]["id"]
