"""Slime-specific ModelAdapter implementation using SGLang HTTP endpoints.

This adapter wraps Slime's SGLang HTTP client to provide a unified interface for SpareOrchestrator.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from argparse import Namespace

import weave  # Weave tracing always enabled

from spare.core.model_adapter import ModelAdapter
from slime.utils.http_utils import post
from slime.utils.processing_utils import load_tokenizer, load_processor

# Header key used by SGLang router for consistent hashing session affinity
_ROUTING_KEY_HEADER = "X-SMG-Routing-Key"

logger = logging.getLogger(__name__)


class SlimeModelAdapter(ModelAdapter):
    """ModelAdapter for Slime backend using SGLang HTTP generation.

    This adapter wraps Slime's SGLang router to provide the standard
    ModelAdapter interface for SpareOrchestrator. It uses async HTTP calls
    to SGLang for efficient generation.
    """

    def __init__(
        self,
        router_ip: str,
        router_port: int,
        tokenizer,
        processor=None,
        default_sampling_params: Optional[Dict[str, Any]] = None,
        router_policy: Optional[str] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Slime model adapter.

        Args:
            router_ip: SGLang router IP address
            router_port: SGLang router port
            tokenizer: Tokenizer instance (from transformers)
            processor: Optional processor for multimodal inputs
            default_sampling_params: Default sampling parameters for generation
            router_policy: SGLang router policy (e.g., "consistent_hashing")
            chat_template_kwargs: Extra kwargs passed to tokenizer.apply_chat_template()
                (e.g., {"enable_thinking": False} for Qwen3.5 models)
        """
        self.router_ip = router_ip
        self.router_port = router_port
        self._tokenizer = tokenizer  # Store as private for property access
        self.processor = processor
        self.router_policy = router_policy
        self.chat_template_kwargs = chat_template_kwargs or {}

        # Default sampling params following Slime conventions
        self.default_sampling_params = default_sampling_params or {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 8192,
            "stop": [],
            "stop_token_ids": [],
            "skip_special_tokens": False,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        }

    def apply_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        chat_template_kwargs_override: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """Apply chat template to convert OpenAI messages to token IDs.

        This uses the tokenizer's chat template system to format and tokenize prompts.
        Returns tokens directly for efficient token-in-token-out generation.

        Args:
            messages: List of message dicts with 'role' and 'content' keys,
                     e.g., [{'role': 'user', 'content': 'Hello'}]
            add_generation_prompt: Whether to add assistant generation prompt
            chat_template_kwargs_override: Per-call override merged on top of
                the adapter's default chat_template_kwargs. Use this for
                per-role thinking control, e.g. {"enable_thinking": True}.

        Returns:
            List of token IDs ready for generation
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        kwargs = {**self.chat_template_kwargs}
        if chat_template_kwargs_override:
            kwargs.update(chat_template_kwargs_override)

        result = self._tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        if tokenize and not isinstance(result, list):
            # transformers >=5.0 returns BatchEncoding; extract plain list
            return list(result["input_ids"])
        return result

    @weave.op
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
            **kwargs: Additional SGLang parameters

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

    @weave.op
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
        """Generate responses using Slime's SGLang HTTP endpoint asynchronously.

        Args:
            messages: List of message dicts with 'role' and 'content' keys, ignore if input_ids is provided
            input_ids: List of input token IDs, used for token-in-token-out tracking
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            template_name: Template to apply
            **kwargs: Additional SGLang parameters

        Returns:
            List of generation results with standardized format
        """
        # Update sampling params with provided values
        sampling_params = self.default_sampling_params.copy()
        sampling_params.update({
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
        })
        # Extract metadata/routing kwargs before passing remaining to sampling params
        session_id = kwargs.pop("session_id", None)
        kwargs.pop("game_code", None)  # Weave tracing only, not sent to SGLang
        kwargs.pop("role", None)  # Weave tracing only, not sent to SGLang
        sampling_params.update(kwargs)

        url = f"http://{self.router_ip}:{self.router_port}/generate"

        try:
            if input_ids is not None:
                prompt_ids = input_ids
            else:
                prompt_ids = self.apply_template(messages)

            # Prepare payload for SGLang server
            payload = {
                "input_ids": prompt_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }

            # Build routing headers for session affinity (consistent hashing)
            headers = None
            if self.router_policy == "consistent_hashing" and session_id:
                headers = {_ROUTING_KEY_HEADER: session_id}

            # Generate with SGLang
            output = await post(url, payload, headers=headers)

            # Extract response tokens and logprobs
            if "output_token_logprobs" in output["meta_info"]:
                response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                response_logprobs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            else:
                response_tokens = []
                response_logprobs = []

            # Decode response
            response_text = output["text"]

            return [{
                'text': response_text,
                'token_ids': response_tokens,
                'logprobs': response_logprobs,
            }]

        except Exception as e:
            logger.error(f"Slime generation failed: {e}")
            return [{
                'text': "",
                'token_ids': [],
                'logprobs': [],
                'error': str(e),
            }]

    @property
    def tokenizer(self):
        """Get the tokenizer instance for token-in-token-out tracking."""
        return self._tokenizer


def create_slime_model_adapter(args: Namespace) -> SlimeModelAdapter:
    """Factory function to create SlimeModelAdapter from Slime args.

    Args:
        args: Slime Namespace with sglang_router_ip, sglang_router_port, etc.

    Returns:
        Configured SlimeModelAdapter instance
    """
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

    # Build sampling params from args
    default_sampling_params = {
        "temperature": getattr(args, "rollout_temperature", 1.0),
        "top_p": getattr(args, "rollout_top_p", 0.95),
        "top_k": getattr(args, "rollout_top_k", 20),
        "max_new_tokens": getattr(args, "rollout_max_response_len", 512),
        "stop": getattr(args, "rollout_stop", []),
        "stop_token_ids": getattr(args, "rollout_stop_token_ids", []),
        "skip_special_tokens": getattr(args, "rollout_skip_special_tokens", False),
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    return SlimeModelAdapter(
        router_ip=args.sglang_router_ip,
        router_port=args.sglang_router_port,
        tokenizer=tokenizer,
        processor=processor,
        default_sampling_params=default_sampling_params,
        router_policy=getattr(args, "sglang_router_policy", None),
        chat_template_kwargs=args.apply_chat_template_kwargs,
    )
