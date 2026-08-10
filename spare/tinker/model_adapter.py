"""Tinker-specific ModelAdapter implementation for Tinker service-based generation.

This adapter wraps Tinker's SamplingClient to provide a unified interface for SpareOrchestrator.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from tinker import ModelInput, SamplingParams, SamplingClient
from tinker_cookbook.tokenizer_utils import Tokenizer

from spare.core.model_adapter import ModelAdapter
from tinker_cookbook.renderers import Renderer
import weave

logger = logging.getLogger(__name__)


class TinkerModelAdapter(ModelAdapter):
    """ModelAdapter for Tinker backend using SamplingClient async generation.

    This adapter wraps Tinker's SamplingClient to provide
    the standard ModelAdapter interface. It inherits from ModelAdapter
    which provides sync wrapper functionality.
    """

    def __init__(
        self,
        sampling_client: SamplingClient,
        tokenizer: Tokenizer,
        renderer: Renderer,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        use_weave: bool = False,
    ):
        """Initialize the Tinker model adapter.

        Args:
            sampling_client: Tinker SamplingClient for async generation
            tokenizer: Tokenizer instance (from tinker_cookbook)
            renderer: Renderer for formatting prompts
            max_tokens: Default maximum tokens for generation
            temperature: Default temperature for generation
            use_weave: Whether to enable Weave tracing for LLM calls
        """
        self.sampling_client = sampling_client
        self._tokenizer = tokenizer  # Store as private for property access
        self.renderer = renderer
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.use_weave = use_weave

        # Conditionally wrap generate_async method with weave.op if enabled
        if self.use_weave:
            self._generate_async_impl = weave.op(self._generate_async_impl)

    def apply_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """Apply chat template to convert OpenAI messages to token IDs.

        Uses the Tinker renderer to convert messages directly to tokens.
        This is more efficient than returning a string that needs re-tokenization.

        Args:
            messages: List of message dicts with 'role' and 'content' keys,
                     e.g., [{'role': 'user', 'content': 'Hello'}]
            tokenize: Whether to tokenize the messages
            add_generation_prompt: Whether to add generation prompt (assistant header)

        Returns:
            List of token IDs ready for generation
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        model_input = self.renderer.build_generation_prompt(messages)

        return model_input.to_ints()

    def generate(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        template_name: str = "default",
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Synchronously generate responses (wraps async method).

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            input_ids: List of input token IDs, used for token-in-token-out tracking
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            template_name: Template to apply
            **kwargs: Additional parameters

        Returns:
            List of generation results with standardized format
        """
        # Run async method in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.generate_async(messages, input_ids, temperature, top_p, max_tokens, template_name, **kwargs)
        )

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        template_name: str = "default",
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Generate responses using Tinker SamplingClient asynchronously.

        This is the unified interface matching SlimeModelAdapter's signature.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            input_ids: List of input token IDs, used for token-in-token-out tracking
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            template_name: Template to apply
            **kwargs: Additional Tinker parameters

        Returns:
            List of generation results with standardized format
        """
        return await self._generate_async_impl(
            messages, input_ids, temperature, top_p, max_tokens, template_name, **kwargs
        )

    async def _generate_async_impl(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        template_name: str = "default",
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Internal implementation of async generation.

        Token-in-token-out: The orchestrator handles all tokenization and passes
        input_ids directly. This adapter just forwards tokens to Tinker.

        This method can be wrapped by weave.op for tracing.
        """
        try:
            stop_condition = self.renderer.get_stop_sequences()

            # Orchestrator always provides input_ids (token-in-token-out)
            if input_ids is None:
                raise ValueError("input_ids is required - orchestrator should always provide tokens")

            model_input = ModelInput.from_ints(input_ids)

            # Create sampling params
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_condition,
            )

            # Generate with Tinker SamplingClient
            sample_response = await self.sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )

            # Extract tokens and logprobs from the first sample
            sequence = sample_response.sequences[0]
            response_token_ids = list(sequence.tokens)
            response_logprobs = list(sequence.logprobs) if sequence.logprobs is not None else []

            # Handle logprobs length mismatch (common in LLM APIs)
            # Some APIs return N-1 logprobs for N tokens (first token has no logprob)
            if len(response_logprobs) != len(response_token_ids):
                logger.debug(
                    f"Logprobs length mismatch from Tinker: {len(response_logprobs)} logprobs vs "
                    f"{len(response_token_ids)} tokens. Padding with 0.0."
                )
                if len(response_logprobs) < len(response_token_ids):
                    # Pad with 0.0 at the beginning (first token typically missing)
                    padding_needed = len(response_token_ids) - len(response_logprobs)
                    response_logprobs = [0.0] * padding_needed + response_logprobs
                else:
                    # Truncate if somehow we have more logprobs
                    response_logprobs = response_logprobs[:len(response_token_ids)]

            # Decode response text from tokens
            response_text = self._tokenizer.decode(response_token_ids)

            return [{
                'text': response_text,
                'token_ids': response_token_ids,
                'logprobs': response_logprobs,
            }]

        except Exception as e:
            logger.error(f"Tinker generation failed: {e}")
            return [{
                'text': "",
                'token_ids': [],
                'logprobs': [],
                'error': str(e),
            }]

    @property
    def tokenizer(self) -> Tokenizer:
        """Get the tokenizer instance for token-in-token-out tracking."""
        return self._tokenizer


def create_tinker_model_adapter(
    sampling_client: SamplingClient,
    tokenizer: Tokenizer,
    renderer: Renderer,
    max_tokens: int = 8192,
    temperature: float = 1.0,
    use_weave: bool = False,
) -> TinkerModelAdapter:
    """Factory function to create TinkerModelAdapter.

    Args:
        sampling_client: Tinker SamplingClient for async generation
        tokenizer: Tokenizer instance
        renderer: Renderer for formatting prompts
        max_tokens: Default maximum tokens for generation
        temperature: Default temperature for generation
        use_weave: Whether to enable Weave tracing

    Returns:
        Configured TinkerModelAdapter instance
    """
    return TinkerModelAdapter(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        renderer=renderer,
        max_tokens=max_tokens,
        temperature=temperature,
        use_weave=use_weave,
    )
