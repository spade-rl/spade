"""Protocol for model adapters across different RL backends.

This defines the interface that all framework-specific adapters must implement
to work with SpareOrchestrator. Each backend (Tinker and Slime) provides its
own adapter that translates between the backend's model interface and this protocol.
"""

from typing import Protocol, List, Dict, Any, runtime_checkable, Optional


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol defining the interface for generating text with a model.

    Each RL backend implements this protocol to provide
    a unified interface for the SpareOrchestrator.

    The adapter abstracts framework-specific operations:
    1. Template application: Converting OpenAI messages to framework-specific format
    2. Generation: Calling the framework's LLM interface
    3. Token tracking: Providing tokenizer for incremental token-in-token-out

    Implementations must support both synchronous and asynchronous generation:
    - Tinker: Natively async with Tinker's TokenCompleter
    - Slime: HTTP-based async generation
    """

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer used by this model.

        Required for incremental token-in-token-out tracking during rollout.
        The tokenizer must implement apply_chat_template() method.

        Returns:
            Tokenizer object (typically HuggingFace transformers.tokenization_utils_base.PreTrainedTokenizerBase)
        """
        ...

    def apply_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        chat_template_kwargs_override: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """Apply chat template to convert OpenAI messages to formatted prompt.

        This abstracts framework-specific template application. Each backend
        can implement its own template system while orchestrator uses standard
        OpenAI message format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys,
                     e.g., [{'role': 'user', 'content': 'Hello'}]
            tokenize: Whether to tokenize the messages
            add_generation_prompt: Whether to add generation prompt
            chat_template_kwargs_override: Per-call override for chat template
                kwargs (e.g., {"enable_thinking": True} for env generation).
                Merged on top of the adapter's default chat_template_kwargs.
        Returns:
            List of token IDs ready for generation
        """
        ...

    def generate(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Synchronously generate responses for a batch of prompts.

        Args:
            messages: List of message dicts with 'role' and 'content' keys,
                     e.g., [{'role': 'user', 'content': 'Hello'}]
            input_ids: List of input token IDs, used for token-in-token-out tracking
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional backend-specific parameters

        Returns:
            List of generation results, one per prompt. Each result is a dict with:
                - 'text': Generated text (str)
                - 'token_ids': List of token IDs (List[int])
                - 'logprobs': List of log probabilities (List[float])
                - 'prompt_token_ids': Token IDs of the prompt (List[int])
                - Any additional backend-specific fields
        """
        raise NotImplementedError

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Asynchronously generate responses for a batch of prompts.

        Same interface as generate() but async for better GPU utilization.

        Args:
            messages: List of message dicts with 'role' and 'content' keys,
                     e.g., [{'role': 'user', 'content': 'Hello'}]
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional backend-specific parameters

        Returns:
            List of generation results (same format as generate())
        """
        raise NotImplementedError
