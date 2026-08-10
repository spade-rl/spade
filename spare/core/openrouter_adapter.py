"""External model adapter for fixed model evaluation.

This module provides a standalone ModelAdapter implementation that uses OpenAI-compatible APIs
(OpenAI, OpenRouter, etc.) for external model evaluation. It does NOT import from OpenEnv.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Default API URLs
OPENAI_API_URL = "https://api.openai.com/v1"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"


class OpenAIModelAdapter:
    """Model adapter for OpenAI-compatible APIs implementing the ModelAdapter protocol.

    Uses OpenAI SDK to access OpenAI, OpenRouter, or any OpenAI-compatible API.
    Designed for fixed model evaluation where we use an external model to measure
    game difficulty.

    Note: Some APIs don't provide token-level logprobs, so placeholder values
    are returned for logprobs fields.
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120,
        tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_retries: int = 4,
    ):
        """Initialize OpenAI-compatible model adapter.

        Args:
            model: Model ID (e.g., "gpt-5-mini", "google/gemini-3-flash-preview")
            api_key: API key. If None, reads from api_key_env environment variable
            api_key_env: Environment variable name for API key (default: OPENAI_API_KEY)
            base_url: API base URL. If None, uses OpenAI default. Set to OPENROUTER_API_URL for OpenRouter.
            temperature: Default sampling temperature
            max_tokens: Default maximum tokens in response
            timeout: Request timeout in seconds
            tokenizer_name: HuggingFace tokenizer for template formatting
        """
        self.model = model
        self.api_key_env = api_key_env
        self.api_key = api_key or os.environ.get(api_key_env)
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                f"API key required. Set {api_key_env} environment variable "
                "or pass api_key parameter."
            )

        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        # Optional sampling controls are omitted when unset.
        self.default_top_p = top_p
        self.default_top_k = top_k
        self.default_min_p = min_p
        self.default_presence_penalty = presence_penalty
        self.max_retries = max_retries
        self.timeout = timeout

        # Detect if using OpenRouter (for custom headers)
        is_openrouter = base_url and "openrouter" in base_url.lower()

        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        if is_openrouter:
            client_kwargs["default_headers"] = {
                "HTTP-Referer": "https://github.com/spade-rl/spade",
                "X-Title": "SPADE",
            }

        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

        # Initialize tokenizer for template formatting
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )

        # Detect if model is a newer OpenAI model with different API requirements
        self._is_newer_openai_model = self._check_newer_openai_model(model)

    def _check_newer_openai_model(self, model: str) -> bool:
        """Check if model is a newer OpenAI model with different API requirements.

        Newer OpenAI models (gpt-4o, gpt-5, o1, o3, etc.):
        - Require max_completion_tokens instead of max_tokens
        - Don't support temperature parameter (only default 1)
        """
        newer_model_prefixes = ("gpt-4o", "gpt-5", "o1", "o3", "chatgpt-4o")
        return any(model.startswith(prefix) for prefix in newer_model_prefixes)

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer used for template formatting.

        Returns:
            HuggingFace tokenizer instance
        """
        return self._tokenizer

    def _inject_sampling(self, request_kwargs: Dict[str, Any]) -> None:
        """Forward the official sampling controls onto the request when set.

        top_p / presence_penalty are standard OpenAI params; top_k / min_p are
        non-standard and passed via ``extra_body`` (OpenRouter forwards them to
        the provider). No-op for newer OpenAI models (gpt-4o/5/o1/o3) which
        reject these. None defaults are omitted so the provider uses its own.
        """
        if self._is_newer_openai_model:
            return
        if self.default_top_p is not None:
            request_kwargs["top_p"] = self.default_top_p
        if self.default_presence_penalty is not None:
            request_kwargs["presence_penalty"] = self.default_presence_penalty
        extra_body: Dict[str, Any] = {}
        if self.default_top_k is not None:
            extra_body["top_k"] = self.default_top_k
        if self.default_min_p is not None:
            extra_body["min_p"] = self.default_min_p
        if extra_body:
            request_kwargs["extra_body"] = extra_body

    def apply_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """Apply chat template to convert OpenAI messages to token IDs.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            tokenize: Whether to return token IDs (always True for this adapter)
            add_generation_prompt: Whether to add generation prompt

        Returns:
            List of token IDs
        """
        if tokenize:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            # Return text if tokenize=False (for compatibility)
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            # Still need to return list for protocol, encode the text
            return self._tokenizer.encode(text, add_special_tokens=False)

    def generate(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Synchronously generate response for given messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            input_ids: Not used for API-based generation (included for protocol compliance)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter (may not be supported by all models)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (ignored)

        Returns:
            List with single generation result dict containing:
                - 'text': Generated text
                - 'token_ids': Token IDs of response (encoded locally)
                - 'logprobs': Placeholder list of zeros
                - 'prompt_token_ids': Input token IDs
        """
        del input_ids, top_p, kwargs  # Not used for API-based generation

        try:
            # Build request kwargs
            request_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }

            # Newer OpenAI models don't support temperature, use max_completion_tokens
            if self._is_newer_openai_model:
                request_kwargs["max_completion_tokens"] = max_tokens
            else:
                request_kwargs["temperature"] = temperature
                request_kwargs["max_tokens"] = max_tokens
            self._inject_sampling(request_kwargs)

            response = self.client.chat.completions.create(**request_kwargs)

            response_text = response.choices[0].message.content or ""

            # Encode response text to get token IDs
            response_tokens = self._tokenizer.encode(
                response_text, add_special_tokens=False
            )

            # Get prompt tokens
            prompt_tokens = self.apply_template(
                messages, tokenize=True, add_generation_prompt=True
            )

            return [
                {
                    "text": response_text,
                    "token_ids": response_tokens,
                    "logprobs": [0.0] * len(response_tokens),  # Placeholder
                    "prompt_token_ids": prompt_tokens,
                }
            ]

        except Exception as e:
            logger.error("API generation failed: %s", e)
            return [
                {
                    "text": "",
                    "token_ids": [],
                    "logprobs": [],
                    "prompt_token_ids": [],
                    "error": str(e),
                }
            ]

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Asynchronously generate response for given messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            input_ids: Not used for API-based generation (included for protocol compliance)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter (may not be supported by all models)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (ignored)

        Returns:
            List with single generation result dict (same format as generate())
        """
        del input_ids, top_p, kwargs  # Not used for API-based generation

        request_kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
        if self._is_newer_openai_model:
            request_kwargs["max_completion_tokens"] = max_tokens
        else:
            request_kwargs["temperature"] = temperature
            request_kwargs["max_tokens"] = max_tokens
        self._inject_sampling(request_kwargs)

        # Retry with exponential backoff on transient errors (rate limits,
        # disconnects) AND empty completions, so a flaky call is not silently
        # scored as a wrong answer — which would deflate eval numbers.
        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.chat.completions.create(**request_kwargs)
                response_text = response.choices[0].message.content or ""
                if not response_text.strip():
                    last_err = "empty completion"
                    raise ValueError(last_err)
                response_tokens = self._tokenizer.encode(
                    response_text, add_special_tokens=False
                )
                prompt_tokens = self.apply_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                return [
                    {
                        "text": response_text,
                        "token_ids": response_tokens,
                        "logprobs": [0.0] * len(response_tokens),
                        "prompt_token_ids": prompt_tokens,
                    }
                ]
            except Exception as e:
                last_err = str(e)
                if attempt < self.max_retries - 1:
                    backoff = 2.0 ** attempt
                    logger.warning(
                        "API async gen attempt %d/%d failed (%s); retry in %.0fs",
                        attempt + 1, self.max_retries, last_err, backoff,
                    )
                    await asyncio.sleep(backoff)

        logger.error("API async generation failed after %d retries: %s",
                     self.max_retries, last_err)
        return [
            {"text": "", "token_ids": [], "logprobs": [],
             "prompt_token_ids": [], "error": last_err}
        ]


# Backwards compatibility alias
OpenRouterModelAdapter = OpenAIModelAdapter


def create_openai_adapter(
    model: str = "gpt-5-mini",
    api_key: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 16384,
    timeout: int = 120,
) -> OpenAIModelAdapter:
    """Create an OpenAI-compatible model adapter for fixed model evaluation.

    Convenience function for creating the adapter with common settings.

    Args:
        model: Model ID (e.g., "gpt-5-mini" for OpenAI, "google/gemini-3-flash-preview" for OpenRouter)
        api_key: API key (or uses api_key_env environment variable)
        api_key_env: Environment variable name for API key (default: OPENAI_API_KEY)
        base_url: API base URL. None for OpenAI, OPENROUTER_API_URL for OpenRouter.
        temperature: Default sampling temperature
        max_tokens: Default max tokens
        timeout: Request timeout in seconds

    Returns:
        Configured OpenAIModelAdapter instance
    """
    return OpenAIModelAdapter(
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


# Backwards compatibility alias
def create_openrouter_adapter(
    model: str = "google/gemini-3-flash-preview",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 16384,
    timeout: int = 120,
) -> OpenAIModelAdapter:
    """Create an OpenRouter model adapter (backwards compatibility).

    Args:
        model: OpenRouter model ID
        api_key: OpenRouter API key (or uses OPENROUTER_API_KEY env var)
        temperature: Default sampling temperature
        max_tokens: Default max tokens
        timeout: Request timeout in seconds

    Returns:
        Configured OpenAIModelAdapter instance for OpenRouter
    """
    return OpenAIModelAdapter(
        model=model,
        api_key=api_key,
        api_key_env="OPENROUTER_API_KEY",
        base_url=OPENROUTER_API_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
