"""Tests for flag-gated ACTOR-THINKING multi-turn gameplay.

Uses the REAL Qwen3-8B tokenizer + chat template (no GPU / no torch needed)
to verify:

  (a) per-turn sample token_ids exactly match the apply_chat_template
      rendering of the think-stripped history + the generated turn;
  (b) loss masks cover exactly the current turn's think+answer, nothing else;
  (c) rollout logprob array lengths align with the mask-1 spans;
  (d) flag OFF reproduces the legacy accumulated single-sequence trajectory
      byte-identically (reference reimplementation of the legacy TITO
      accumulation), with no turn-record metadata;
  (e) truncated thinking (no closing </think>, no EOS) marks the episode
      TRUNCATED and still yields consistent per-turn samples.

Tokenizer path override: QWEN3_TOKENIZER=/path/to/Qwen3-8B
"""

import asyncio
import os
import sys
import time
import types
from typing import Any, Dict, List, Optional

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "slime"))

try:
    import weave  # noqa: F401
except ImportError:
    _weave_stub = types.ModuleType("weave")

    def _op(func=None, **_kwargs):
        if callable(func):
            return func
        return lambda g: g

    _weave_stub.op = _op
    sys.modules["weave"] = _weave_stub

transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer

TOKENIZER_DIR = os.environ.get("QWEN3_TOKENIZER")
if not TOKENIZER_DIR:
    pytest.skip("Set QWEN3_TOKENIZER to run actor-thinking tests", allow_module_level=True)
try:
    TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
except (OSError, ValueError) as exc:
    pytest.skip(f"Qwen3 tokenizer unavailable: {exc}", allow_module_level=True)

EOS_TEXT = TOKENIZER.eos_token
EOS_ID = TOKENIZER.eos_token_id

import spare.core.orchestrator as orchestrator_module
from spare.core.orchestrator import SpareOrchestrator
from spare.core.game_policy import GamePolicy
from spare.core.types import SpareConfig, TrajectoryStatus
from spare.core.utils import get_token_delta
from spare.slime.trajectory_converter import (
    fan_out_thinking_sample,
    trajectory_to_slime_sample,
)
from slime.utils.types import Sample

def render_ids(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    **kwargs: Any,
) -> List[int]:
    """apply_chat_template -> plain list of ids (transformers 4.x and 5.x)."""
    result = TOKENIZER.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    if not isinstance(result, list):
        return list(result["input_ids"])
    return list(result)


class FakeAdapter:
    """ModelAdapter double: real tokenizer/template, scripted generations.

    Mirrors SlimeModelAdapter.apply_template (default kwargs + per-call
    override) and generate_async's return contract:
    text INCLUDES the trailing EOS text (skip_special_tokens=False +
    no_stop_trim, like SGLang), token_ids include the EOS id.
    """

    def __init__(self, scripted_responses: List[Dict[str, Any]]):
        self._tokenizer = TOKENIZER
        self.scripted = list(scripted_responses)
        self.calls: List[Dict[str, Any]] = []

    @property
    def tokenizer(self):
        return self._tokenizer

    def apply_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        chat_template_kwargs_override: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        kwargs = dict(chat_template_kwargs_override or {})
        result = self._tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        if tokenize and not isinstance(result, list):
            return list(result["input_ids"])
        return result

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        spec = self.scripted[len(self.calls)]
        text: str = spec["text"]
        with_eos: bool = spec.get("with_eos", True)

        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if with_eos:
            token_ids = token_ids + [EOS_ID]
            text = text + EOS_TEXT
        logprobs = [-0.5 - 0.001 * i - 0.1 * len(self.calls) for i in range(len(token_ids))]

        self.calls.append(
            {
                "input_ids": list(input_ids) if input_ids is not None else None,
                "token_ids": list(token_ids),
                "logprobs": list(logprobs),
            }
        )
        return [{"text": text, "token_ids": token_ids, "logprobs": logprobs}]


class FakeGameEnv:
    """Minimal env double for play_game_async(env=...)."""

    def __init__(self, observations: List[str], final_reward: float = 1.0):
        self.observations = list(observations)
        self.final_reward = final_reward
        self.actions: List[str] = []
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        self.actions = []
        return self.observations[0], {}

    def step(self, action: str):
        self.actions.append(action)
        self.step_count += 1
        if self.step_count >= len(self.observations):
            return "done", self.final_reward, True, False, {}
        return self.observations[self.step_count], 0.0, False, False, {}

    def close(self):
        pass


def make_orchestrator(adapter: FakeAdapter, actor_enable_thinking: Optional[bool]) -> SpareOrchestrator:
    config = SpareConfig(
        actor_enable_thinking=actor_enable_thinking,
        max_turns=6,
        max_context_length=8192,
        actor_max_tokens=512,
        action_format="boxed",
    )
    return SpareOrchestrator(
        model=adapter,
        config=config,
        learning_potentials={},
        game_policy=GamePolicy(),
    )


def wrapped_first_user_message(obs: str) -> Dict[str, str]:
    """The qwen3_game template wrapper applied by _apply_game_template."""
    return {
        "role": "user",
        "content": (
            f"You are playing a language game. Make valid actions to win.\n"
            f"Observation: {obs}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}."
        ),
    }


THINKING_RESPONSES = [
    {"text": "<think>\nTHINK_SECRET_0 decoy answer \\boxed{9}\n</think>\n\nFirst guess: \\boxed{7}"},
    {"text": "<think>\nTHINK_SECRET_1 reconsidering\n</think>\n\nSecond guess: \\boxed{5}"},
    {"text": "<think>\nTHINK_SECRET_2 it must be four\n</think>\n\nFinal answer: \\boxed{4}"},
]

OBSERVATIONS = ["What is 2+2? Guess a digit.", "Wrong. Try again (turn 2).", "Wrong. Last chance (turn 3)."]


def run_thinking_episode(responses=THINKING_RESPONSES, observations=OBSERVATIONS):
    adapter = FakeAdapter(responses)
    orch = make_orchestrator(adapter, actor_enable_thinking=True)
    env = FakeGameEnv(observations)
    traj = asyncio.run(
        orch.play_game_async("/nonexistent/game_test.py", skill="Test Skill", env=env)
    )
    return adapter, env, traj


def test_a_per_turn_prompts_match_canonical_stripped_rendering():
    adapter, env, traj = run_thinking_episode()
    records = traj.metadata["turn_records"]
    assert len(records) == 3, f"expected 3 turn records, got {len(records)}"

    # Reconstruct the message history exactly as the orchestrator builds it:
    # assistant content = generated text WITHOUT the trailing EOS text.
    full_messages = [wrapped_first_user_message(OBSERVATIONS[0])]
    stripped_messages = [wrapped_first_user_message(OBSERVATIONS[0])]

    for k, record in enumerate(records):
        # (a) generation-time prompt ids == canonical rendering of history
        expected_full = render_ids(full_messages, enable_thinking=True)
        assert record["prompt_ids"] == expected_full, (
            f"turn {k}: prompt_ids diverge from apply_chat_template rendering "
            f"of the message history"
        )

        # Same rendering when assistant contents are MANUALLY think-stripped:
        # proves the template dropped prior turns' think blocks.
        expected_stripped = render_ids(stripped_messages, enable_thinking=True)
        assert record["prompt_ids"] == expected_stripped, (
            f"turn {k}: rendering differs between raw and manually think-"
            f"stripped histories — think blocks leaked into the prompt"
        )

        # No prior think content in the decoded prompt.
        decoded_prompt = TOKENIZER.decode(record["prompt_ids"])
        for j in range(k):
            assert f"THINK_SECRET_{j}" not in decoded_prompt, (
                f"turn {k}: turn {j}'s think content leaked into the prompt"
            )
        # But the generated turn itself keeps its think content.
        decoded_output = TOKENIZER.decode(record["output_ids"])
        assert f"THINK_SECRET_{k}" in decoded_output

        raw_text = THINKING_RESPONSES[k]["text"]
        visible_text = raw_text.split("</think>")[-1]
        full_messages.append({"role": "assistant", "content": raw_text})
        stripped_messages.append(
            {"role": "assistant", "content": visible_text.lstrip("\n")}
        )
        if k + 1 < len(records):
            full_messages.append({"role": "user", "content": OBSERVATIONS[k + 1]})
            stripped_messages.append({"role": "user", "content": OBSERVATIONS[k + 1]})

    # Actions parsed from the VISIBLE part only: turn-0 decoy \boxed{9}
    # inside <think> must not become the action.
    assert env.actions[0] == "\\boxed{7}", env.actions[0]
    assert env.actions[-1] == "\\boxed{4}", env.actions[-1]
    assert traj.status == TrajectoryStatus.COMPLETED
    assert traj.reward == 1.0
    print("PASS: (a) per-turn prompts match canonical stripped rendering")


def test_b_c_fanout_samples_masks_and_logprobs():
    adapter, env, traj = run_thinking_episode()
    records = traj.metadata["turn_records"]

    episode_sample = trajectory_to_slime_sample(traj, index=17, role="actor")
    episode_sample.reward = 0.625  # normalized episode reward
    fanned = fan_out_thinking_sample(episode_sample, TOKENIZER)

    assert len(fanned) == len(records) == 3

    group_ids = {s.group_id for s in fanned}
    assert len(group_ids) == 1, f"siblings must share group_id, got {group_ids}"

    for k, (record, sample) in enumerate(zip(records, fanned)):
        prompt_ids = record["prompt_ids"]
        output_ids = record["output_ids"]

        # (a cont.) sample tokens == canonical stripped-history rendering +
        # the generated turn (verbatim engine ids).
        assert sample.tokens == prompt_ids + output_ids, f"turn {k}: token mismatch"

        # (b) loss mask covers exactly the current turn's think+answer.
        assert sample.response_length == len(output_ids)
        assert sample.loss_mask == [1] * len(output_ids), f"turn {k}: mask wrong"
        implied_prompt_len = len(sample.tokens) - sample.response_length
        assert implied_prompt_len == len(prompt_ids)

        # (c) logprobs align with the mask-1 span, values preserved verbatim.
        assert len(sample.rollout_log_probs) == sum(sample.loss_mask)
        assert sample.rollout_log_probs == adapter.calls[k]["logprobs"], (
            f"turn {k}: rollout logprobs not preserved"
        )

        # Episode reward broadcast (NOT split 1/K), status preserved,
        # heavy payload dropped.
        assert sample.reward == 0.625
        assert sample.status == Sample.Status.COMPLETED
        assert "turn_records" not in sample.metadata
        assert sample.metadata["turn_idx"] == k
        assert sample.metadata["num_segments"] == 3

    print("PASS: (b)+(c) fan-out tokens/masks/logprob alignment")


def test_d_flag_off_reproduces_legacy_trajectory():
    legacy_responses = [
        {"text": "First guess: \\boxed{7}"},
        {"text": "Second guess: \\boxed{5}"},
        {"text": "Final answer: \\boxed{4}"},
    ]
    adapter = FakeAdapter(legacy_responses)
    orch = make_orchestrator(adapter, actor_enable_thinking=None)
    env = FakeGameEnv(OBSERVATIONS)
    traj = asyncio.run(
        orch.play_game_async("/nonexistent/game_test.py", skill="Test Skill", env=env)
    )

    # No thinking artifacts on the legacy path.
    assert "turn_records" not in traj.metadata
    assert "actor_thinking" not in traj.metadata

    # Reference reimplementation of the legacy accumulated TITO sequence:
    # prompt tokens, then per turn [response tokens (mask 1, logprobs)] and
    # [obs delta via get_token_delta (mask 0, logprob 0.0)].
    messages = [wrapped_first_user_message(OBSERVATIONS[0])]
    expected_tokens = render_ids(messages)  # adapter default kwargs (none)
    expected_masks: List[int] = []
    expected_logprobs: List[float] = []

    for k in range(3):
        call = adapter.calls[k]
        # Legacy passes the ACCUMULATED tokens as input_ids each turn.
        assert call["input_ids"] == expected_tokens, f"turn {k}: legacy TITO input drifted"
        expected_tokens = expected_tokens + call["token_ids"]
        expected_masks = expected_masks + [1] * len(call["token_ids"])
        expected_logprobs = expected_logprobs + call["logprobs"]
        # raw text including EOS text, exactly as legacy appends it
        messages.append(
            {"role": "assistant", "content": legacy_responses[k]["text"] + EOS_TEXT}
        )
        if k + 1 < 3:
            messages.append({"role": "user", "content": OBSERVATIONS[k + 1]})
            obs_tokens, obs_mask = get_token_delta(tokenizer=TOKENIZER, messages=messages)
            expected_tokens = expected_tokens + obs_tokens
            expected_masks = expected_masks + obs_mask
            expected_logprobs = expected_logprobs + [0.0] * len(obs_tokens)

    assert traj.tokens == expected_tokens, "flag OFF: tokens diverged from legacy accumulation"
    assert traj.loss_mask == expected_masks, "flag OFF: loss mask diverged from legacy"
    assert traj.rollout_log_probs == expected_logprobs, "flag OFF: logprobs diverged from legacy"
    assert traj.response_length == len(expected_logprobs)
    assert traj.status == TrajectoryStatus.COMPLETED
    assert traj.reward == 1.0
    print("PASS: (d) flag OFF reproduces legacy single-sequence trajectory")


def test_e_truncated_think_no_closing_tag():
    responses = [
        {"text": "<think>\nTHINK_SECRET_0 ok\n</think>\n\nGuess: \\boxed{7}"},
        # Truncated mid-think: no closing tag, no EOS (max_tokens cut-off).
        {"text": "<think>\nTHINK_SECRET_1 this thought never end", "with_eos": False},
    ]
    adapter, env, traj = run_thinking_episode(responses)

    assert traj.status == TrajectoryStatus.TRUNCATED
    records = traj.metadata["turn_records"]
    assert len(records) == 2
    assert records[0]["finish_reason"] == "stop"
    assert records[1]["finish_reason"] == "length"
    # Env only stepped once — the truncated turn never reached env.step.
    assert len(env.actions) == 1

    episode_sample = trajectory_to_slime_sample(traj, index=3, role="actor")
    fanned = fan_out_thinking_sample(episode_sample, TOKENIZER)
    assert len(fanned) == 2
    for k, (record, sample) in enumerate(zip(records, fanned)):
        assert sample.tokens == record["prompt_ids"] + record["output_ids"]
        assert sample.loss_mask == [1] * len(record["output_ids"])
        assert len(sample.rollout_log_probs) == sum(sample.loss_mask)
        # Episode status (TRUNCATED) restored on every sibling so
        # spare_compact_filter / truncation metrics see it.
        assert sample.status == Sample.Status.TRUNCATED, f"turn {k}: status not preserved"
    # The truncated turn's tokens end WITHOUT eos.
    assert fanned[1].tokens[-1] != EOS_ID
    print("PASS: (e) truncated-think episode handled")


def test_f_compact_filtered_episode_zeroes_turn_masks():
    responses = [
        {"text": "<think>\nT0\n</think>\n\n\\boxed{7}"},
        {"text": "<think>\nT1 runs out of budget", "with_eos": False},
    ]
    adapter, env, traj = run_thinking_episode(responses)
    episode_sample = trajectory_to_slime_sample(traj, index=0, role="actor")
    # Simulate spare_compact_filter zeroing the truncated episode.
    episode_sample.loss_mask = [0] * len(episode_sample.loss_mask)
    fanned = fan_out_thinking_sample(episode_sample, TOKENIZER)
    for sample in fanned:
        assert sum(sample.loss_mask) == 0, "compacted episode must stay zero-loss after fan-out"
        assert len(sample.loss_mask) == sample.response_length
    print("PASS: (f) compact-filtered episode stays zero-loss through fan-out")


def test_g_non_thinking_sample_passthrough():
    """Samples without turn_records (env/proposer, legacy) pass through unchanged."""
    sample = Sample(
        index=5,
        tokens=[1, 2, 3, 4],
        response_length=2,
        loss_mask=[1, 1],
        rollout_log_probs=[-0.1, -0.2],
        reward=0.5,
        status=Sample.Status.COMPLETED,
        metadata={"role": "environment"},
    )
    out = fan_out_thinking_sample(sample, TOKENIZER)
    assert out == [sample]
    print("PASS: (g) non-thinking sample passthrough")


def test_h_blacklist_fail_fast_precedes_thinking_dispatch():
    """Hung-game guard interaction: the persistent blacklist fail-fast sits at
    the top of play_game_async, BEFORE the thinking dispatch — a blacklisted
    game must fail fast without any generation, exactly as on the legacy path.
    This is also what makes the play canary work unchanged in thinking mode:
    the canary IS a normal play_game_async call, and run_game_group's
    post-canary blacklist check sees the same blacklist."""
    adapter = FakeAdapter(THINKING_RESPONSES)
    orch = make_orchestrator(adapter, actor_enable_thinking=True)
    orch._hung_game_blacklist.add("/games/hung_game.py")

    traj = asyncio.run(
        orch.play_game_async("/games/hung_game.py", skill="Test Skill")
    )
    assert traj.status == TrajectoryStatus.FAILED
    assert traj.metadata["error"] == "blacklisted after earlier hang"
    assert adapter.calls == [], "blacklisted game must not reach generation"
    print("PASS: (h) blacklist fail-fast precedes thinking dispatch")


class HangingStepEnv(FakeGameEnv):
    """Env whose step() hangs long enough to trip ENV_STEP_TIMEOUT."""

    def __init__(self, hang_seconds: float):
        super().__init__(["obs turn 1"])
        self.hang_seconds = hang_seconds

    def step(self, action: str):
        time.sleep(self.hang_seconds)
        return "never", 0.0, True, False, {}


def test_i_thinking_step_timeout_blacklists_game():
    """A hanging env.step() inside the THINKING loop must blacklist the game
    (canary semantics: remaining plays of the group get skipped) and flag it
    for the step-timeout proposer penalty — same guards as the legacy loop."""
    original_timeout = orchestrator_module.ENV_STEP_TIMEOUT
    orchestrator_module.ENV_STEP_TIMEOUT = 1
    try:
        adapter = FakeAdapter(THINKING_RESPONSES)
        orch = make_orchestrator(adapter, actor_enable_thinking=True)
        env = HangingStepEnv(hang_seconds=5.0)
        traj = asyncio.run(
            orch.play_game_async("/games/step_hang.py", skill="Test Skill", env=env)
        )
    finally:
        orchestrator_module.ENV_STEP_TIMEOUT = original_timeout

    assert traj.status == TrajectoryStatus.FAILED
    assert traj.metadata["error"] == "env.step() timeout"
    assert "/games/step_hang.py" in orch._hung_game_blacklist, (
        "thinking path must blacklist step-timeout games (canary gating relies on it)"
    )
    assert "/games/step_hang.py" in orch._step_timeout_games, (
        "thinking path must flag step-timeout games for the proposer penalty"
    )
    print("PASS: (i) thinking-loop step timeout blacklists the game")


def main() -> int:
    tests = [
        test_a_per_turn_prompts_match_canonical_stripped_rendering,
        test_b_c_fanout_samples_masks_and_logprobs,
        test_d_flag_off_reproduces_legacy_trajectory,
        test_e_truncated_think_no_closing_tag,
        test_f_compact_filtered_episode_zeroes_turn_masks,
        test_g_non_thinking_sample_passthrough,
        test_h_blacklist_fail_fast_precedes_thinking_dispatch,
        test_i_thinking_step_timeout_blacklists_game,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {test.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR: {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
