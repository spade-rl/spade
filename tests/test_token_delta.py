"""Unit tests for multi-turn token accounting.

Verifies that after a simulated multi-turn rollout:
1. The per-turn deltas from ``get_token_delta`` correctly include the
   generation prompt when the last message is from the user. (Regression test
   for the ``compute_token_delta`` bug that caused the model to generate
   ``<|im_start|>Assistant`` mid-response.)
2. ``loss_mask`` is ``1`` exactly on assistant response tokens (including
   ``<|im_end|>``) and ``0`` everywhere else.
3. Lengths of ``all_tokens``, ``all_masks``, ``all_logprobs`` are consistent
   with Slime's ``prompt_length = total_length - response_length`` convention.
"""

import pytest

pytest.importorskip("jinja2")
transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer

from spare.core.utils import get_token_delta


# Cover both Qwen2.5 (which injects a default system prompt) and
# Qwen3-Instruct-2507 (which starts directly at <|im_start|>user). Any
# tokenizer not installed locally will be skipped by importorskip.
QWEN_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]


@pytest.fixture(scope="module", params=QWEN_MODELS)
def tokenizer(request):
    try:
        return AutoTokenizer.from_pretrained(
            request.param,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as e:  # pragma: no cover — env-dependent
        pytest.skip(f"Tokenizer {request.param} unavailable: {e}")


@pytest.fixture(scope="module")
def im_end_id(tokenizer):
    return tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]


def _simulate_sglang_response(tokenizer, text, im_end_id):
    """Return (tokens, logprobs) as SGLang would: text + <|im_end|>, no trailing newline."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens.append(im_end_id)
    logprobs = [-0.5] * len(tokens)
    return tokens, logprobs


def _simulate_rollout(tokenizer, im_end_id, num_turns):
    """Replay the orchestrator's multi-turn token accounting.

    Mirrors the production loop in spare/core/orchestrator.py — accumulates
    SGLang's exact response tokens for assistant turns and uses get_token_delta
    to encode new user observations. No sync_to_chat_template call (deleted).
    """
    messages = [{"role": "user", "content": "obs0"}]
    all_tokens = list(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    )
    all_masks: list[int] = []
    all_logprobs: list[float] = []

    for turn in range(num_turns):
        if turn > 0:
            messages.append({"role": "user", "content": f"obs{turn}"})
            obs_tokens, obs_mask = get_token_delta(tokenizer, messages)
            all_tokens.extend(obs_tokens)
            all_masks.extend(obs_mask)
            all_logprobs.extend([0.0] * len(obs_tokens))

        resp_text = f"resp{turn}"
        sglang_tokens, sglang_logprobs = _simulate_sglang_response(
            tokenizer, resp_text, im_end_id
        )

        messages.append({"role": "assistant", "content": resp_text})
        all_tokens.extend(sglang_tokens)
        all_masks.extend([1] * len(sglang_tokens))
        all_logprobs.extend(sglang_logprobs)

    return messages, all_tokens, all_masks, all_logprobs


def test_get_token_delta_user_message_includes_generation_prompt(tokenizer):
    """Regression test: user-message delta must include ``<|im_start|>assistant\\n``."""
    messages = [
        {"role": "user", "content": "obs1"},
        {"role": "assistant", "content": "resp1"},
        {"role": "user", "content": "obs2"},
    ]
    tokens, mask = get_token_delta(tokenizer, messages)
    decoded = tokenizer.decode(tokens)
    assert "<|im_start|>assistant" in decoded, (
        f"user-message delta is missing the generation prompt: {decoded!r}"
    )
    assert all(m == 0 for m in mask), "user-message delta must not be trainable"


def test_get_token_delta_assistant_message(tokenizer):
    """Assistant delta should be trainable and not include a new generation prompt."""
    messages = [
        {"role": "user", "content": "obs1"},
        {"role": "assistant", "content": "resp1"},
    ]
    tokens, mask = get_token_delta(tokenizer, messages)
    decoded = tokenizer.decode(tokens)
    assert "resp1" in decoded
    assert all(m == 1 for m in mask), "assistant delta must be fully trainable"


@pytest.mark.parametrize("num_turns", [1, 2, 3])
def test_buffer_lengths_consistent(tokenizer, im_end_id, num_turns):
    """all_masks and all_logprobs must have the same length and must equal
    the response portion (total - prompt)."""
    _, all_tokens, all_masks, all_logprobs = _simulate_rollout(
        tokenizer, im_end_id, num_turns
    )
    assert len(all_masks) == len(all_logprobs), (
        "mask/logprob length must match for Slime's response_length convention"
    )
    assert len(all_tokens) >= len(all_masks), (
        "all_tokens must contain at least the response portion"
    )


@pytest.mark.parametrize("num_turns", [1, 2, 3])
def test_loss_mask_covers_only_assistant_responses(tokenizer, im_end_id, num_turns):
    """mask=1 runs must decode to assistant response content followed by <|im_end|>."""
    _, all_tokens, all_masks, _ = _simulate_rollout(tokenizer, im_end_id, num_turns)

    prompt_length = len(all_tokens) - len(all_masks)
    trainable_ids = [
        all_tokens[prompt_length + i]
        for i, m in enumerate(all_masks)
        if m == 1
    ]
    # Every assistant response should end with <|im_end|>
    assert trainable_ids.count(im_end_id) == num_turns, (
        f"expected {num_turns} <|im_end|> tokens inside loss_mask=1 runs, "
        f"got {trainable_ids.count(im_end_id)}"
    )

    # Collect contiguous runs of mask=1 and verify each decodes to
    # "resp{k}<|im_end|>"
    runs = []
    start = None
    for i, m in enumerate(all_masks):
        if m == 1 and start is None:
            start = i
        elif m == 0 and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(all_masks)))

    assert len(runs) == num_turns, (
        f"expected {num_turns} trainable runs, got {len(runs)}"
    )
    for turn_idx, (rs, re) in enumerate(runs):
        decoded = tokenizer.decode(
            all_tokens[prompt_length + rs: prompt_length + re]
        )
        assert f"resp{turn_idx}" in decoded, (
            f"run {turn_idx} doesn't contain response content: {decoded!r}"
        )
        assert "<|im_end|>" in decoded, (
            f"run {turn_idx} doesn't end at <|im_end|>: {decoded!r}"
        )
