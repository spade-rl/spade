"""ModelAdapter shim for offline eval.

Adapts our `OfflineClient` (SGLang HTTP) to the `spare.core.model_adapter.ModelAdapter`
protocol so the existing `GemEvaluator` and `tau2_evaluator` can be reused without
modification.

The shim:
- Loads the HF tokenizer from the local model dir.
- `apply_template`: uses tokenizer.apply_chat_template (matches SGLang's
  server-side tokenization).
- `generate_async`: calls /v1/chat/completions on the OfflineClient, then
  re-tokenizes the text locally to populate `token_ids`. Logprobs are
  returned empty (GEM/tau2 evaluators don't use them).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from transformers import AutoTokenizer

from eval_offline.client import OfflineClient

logger = logging.getLogger("eval_offline.model_adapter_shim")


class OfflineModelAdapter:
    """Implements `spare.core.model_adapter.ModelAdapter` against an
    OpenAI-compatible HTTP server."""

    def __init__(self, client: OfflineClient, model_path: Path | str):
        self._client = client
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        logger.info("[shim] tokenizer loaded from %s (vocab=%d)",
                    model_path, self._tokenizer.vocab_size)

    @property
    def tokenizer(self):
        return self._tokenizer

    def apply_template(
        self,
        messages: list[dict],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        chat_template_kwargs_override: Optional[dict] = None,
    ) -> list[int]:
        kwargs = chat_template_kwargs_override or {}
        out = self._tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        # apply_chat_template returns either a string (tokenize=False) or a
        # list of ints (tokenize=True). The protocol expects List[int].
        if not tokenize and isinstance(out, str):
            out = self._tokenizer.encode(out, add_special_tokens=False)
        return list(out)

    def generate(self, messages, input_ids=None, temperature=1.0, top_p=0.9,
                 max_tokens=512, **kwargs):
        # GEM/tau2 evaluators always go through generate_async; sync path is a
        # convenience for callers that don't have an event loop.
        import asyncio
        return asyncio.run(self.generate_async(
            messages, input_ids=input_ids, temperature=temperature,
            top_p=top_p, max_tokens=max_tokens, **kwargs,
        ))

    async def generate_async(
        self,
        messages,
        input_ids: Optional[list[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> list[dict]:
        """Call /v1/chat/completions and shape the response into the dict
        format expected by spare.core.eval.* (text, token_ids, logprobs,
        prompt_token_ids)."""
        # Pull SGLang-specific kwargs into extra_body and drop kwargs that
        # the OpenAI client doesn't accept.
        extra_body: dict[str, Any] = {}
        if "top_k" in kwargs:
            extra_body["top_k"] = kwargs.pop("top_k")
        # session_id, role, game_code etc. are training-only orchestrator
        # hints; safe to drop for offline eval.
        for k in ("session_id", "role", "game_code"):
            kwargs.pop(k, None)

        completions = await self._client.chat(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
            extra_body=extra_body or None,
        )

        prompt_token_ids = (
            list(input_ids) if input_ids is not None else
            self.apply_template(messages)
        )

        results = []
        for c in completions:
            response_token_ids = self._tokenizer.encode(
                c.text, add_special_tokens=False,
            )
            results.append({
                "text": c.text,
                "token_ids": response_token_ids,
                "logprobs": [],
                "prompt_token_ids": prompt_token_ids,
                "finish_reason": c.finish_reason,
            })
        return results
