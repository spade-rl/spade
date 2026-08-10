"""Async OpenAI-compatible client for the SGLang server."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx
import openai

logger = logging.getLogger(__name__)


@dataclass
class CompletionResult:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    raw: dict[str, Any]


class OfflineClient:
    """Thin async wrapper around an OpenAI-compatible HTTP server.

    Used by every eval suite. Adds:
    - default model name pinned at construction (no per-call repetition)
    - structured CompletionResult return
    - bounded concurrency via a semaphore
    - graceful retry on transient HTTP errors
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        max_concurrent: int = 64,
        max_retries: int = 10,
        max_retry_delay: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.max_retry_delay = max_retry_delay
        self._max_concurrent = max_concurrent
        # Semaphores are loop-bound, so concurrent suites need per-loop instances.
        self._sems_by_loop_id: dict[int, asyncio.Semaphore] = {}
        http_client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=60.0),
        )
        # Keyless SGLang endpoints ignore the token ("EMPTY" is fine), but
        # authenticated APIs need a real bearer key. When the caller left the
        # default placeholder, pull the right key from the env based on the host.
        if api_key == "EMPTY":
            if "openrouter" in self.base_url:
                api_key = os.environ.get("OPENROUTER_API_KEY") or api_key
            elif "openai.com" in self.base_url:
                api_key = os.environ.get("OPENAI_API_KEY") or api_key
        self._client = openai.AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=api_key,
            http_client=http_client,
        )

    def _get_sem(self) -> asyncio.Semaphore:
        """Return the Semaphore bound to the currently-running event loop,
        creating one on first use. Each loop gets `max_concurrent` slots —
        the per-suite max_concurrent caps already enforce the global ceiling."""
        loop = asyncio.get_running_loop()
        sem = self._sems_by_loop_id.get(id(loop))
        if sem is None:
            sem = asyncio.Semaphore(self._max_concurrent)
            self._sems_by_loop_id[id(loop)] = sem
        return sem

    @property
    def openai_client(self) -> openai.AsyncOpenAI:
        """Expose the underlying client for code that wants to call it
        directly (e.g. `tau2_evaluator` which constructs its own clients)."""
        return self._client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        n: int = 1,
        stop: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> list[CompletionResult]:
        """Chat completion. Returns one CompletionResult per `n` samples."""
        async with self._get_sem():
            return await self._with_retry(
                self._chat_once,
                messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=n,
                stop=stop,
                extra_body=extra_body,
            )

    async def _chat_once(
        self,
        messages,
        *,
        temperature,
        top_p,
        max_tokens,
        n,
        stop,
        extra_body,
    ):
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
        )
        if stop is not None:
            kwargs["stop"] = stop
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = await self._client.chat.completions.create(**kwargs)
        return [
            CompletionResult(
                text=c.message.content or "",
                finish_reason=c.finish_reason or "",
                prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
                raw=c.model_dump(),
            )
            for c in resp.choices
        ]

    async def _with_retry(self, fn, *args, **kwargs):
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                return await fn(*args, **kwargs)
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                if attempt + 1 == self.max_retries:
                    raise
                logger.warning(
                    "[client] %s on attempt %d, retrying in %.1fs: %s",
                    type(e).__name__, attempt + 1, delay, e,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_retry_delay)
            except openai.APIStatusError as e:
                # Retry transient server errors and rate-limit backpressure.
                # 4xx other than 429 are client bugs — don't retry.
                if e.status_code not in (429, 500, 502, 503, 504):
                    raise
                if attempt + 1 == self.max_retries:
                    raise
                logger.warning(
                    "[client] HTTP %s on attempt %d, retrying in %.1fs",
                    e.status_code, attempt + 1, delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_retry_delay)
        raise RuntimeError("unreachable")
