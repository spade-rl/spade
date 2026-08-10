"""Token accounting helpers."""

from typing import Any


def get_token_delta(
    tokenizer: Any,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Return tokens and loss mask contributed by the newest message."""
    if not messages:
        return [], []
    is_assistant = messages[-1]["role"] == "assistant"
    extra_kwargs = {}
    if tools:
        extra_kwargs["tools"] = tools
    if chat_template_kwargs:
        extra_kwargs.update(chat_template_kwargs)

    current = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not is_assistant,
        **extra_kwargs,
    )
    previous = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=is_assistant,
        **extra_kwargs,
    )
    tokens = tokenizer.encode(current[len(previous) :], add_special_tokens=False)
    return tokens, [int(is_assistant)] * len(tokens)
