"""Offline unit tests for spare.core.eval.tau2_evaluator.

No network, no sglang, no tau2-bench required. Mocks ``tau2.run.run_tasks``
and ``tau2.run.get_tasks`` via direct monkeypatching of the module-level
symbols so tests run even in environments where tau2 is not installed.

Exercises:
- Env var enforcement (OPENROUTER_API_KEY must be set)
- Graceful degradation when tau2 is not importable
- Pass@1 strict denominator (option B)
- Error accounting (errors include denominator, drop pass rate)
- Per-spec sequential execution
- Empty specs returns empty result
- Metric key naming convention (gem_eval/tau2_* prefix)
- Timeout path
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import patch

import pytest

import spare.core.eval.tau2_evaluator as te
from spare.core.eval.tau2_evaluator import (
    SGLANG_MODEL_NAME,
    Tau2EvalResult,
    Tau2Evaluator,
)
from spare.core.eval.tau2_tasks import Tau2TaskSpec


# ---------------------------------------------------------------------------
# Fake tau2 data objects
#
# We do NOT import tau2 here — the tests should run even when the package
# is missing. Instead, we build duck-typed objects that expose the same
# attribute paths our evaluator reads:
#   - results.simulations : list[SimulationRun]
#   - sim.reward_info.reward : float
#   - sim.termination_reason : Optional[str]
#   - sim.ticks : int


@dataclass
class _FakeRewardInfo:
    reward: float


@dataclass
class _FakeSimulation:
    task_id: str
    termination_reason: Optional[str] = "agent_stop"
    ticks: int = 5
    reward_info: _FakeRewardInfo = field(default_factory=lambda: _FakeRewardInfo(reward=0.0))


@dataclass
class _FakeResults:
    simulations: List[_FakeSimulation]


def _make_results(reward_list: List[float], termination: str = "agent_stop") -> _FakeResults:
    """Build a fake Results object with N simulations having the given rewards."""
    sims = []
    for i, r in enumerate(reward_list):
        sims.append(_FakeSimulation(
            task_id=f"t{i}",
            termination_reason=termination,
            ticks=4 + i,
            reward_info=_FakeRewardInfo(reward=r),
        ))
    return _FakeResults(simulations=sims)


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture
def openrouter_env(monkeypatch):
    """Ensure OPENROUTER_API_KEY is set for tests that need the evaluator."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake-key")


@pytest.fixture
def retail_spec() -> Tau2TaskSpec:
    return Tau2TaskSpec(
        domain="retail",
        split="test",
        max_concurrency=4,
        num_trials=1,
        max_steps=50,
        truncate=4,  # pretend we're only running 4 tasks for the test
    )


@pytest.fixture
def telecom_spec() -> Tau2TaskSpec:
    return Tau2TaskSpec(
        domain="telecom",
        split="test",
        max_concurrency=4,
        num_trials=1,
        max_steps=50,
        truncate=4,
    )


@pytest.fixture
def force_tau2_available(monkeypatch):
    """Force the module-level _TAU2_AVAILABLE flag to True so tests
    bypass the degraded-no-tau2 branch even if the package is missing."""
    monkeypatch.setattr(te, "_TAU2_AVAILABLE", True)
    # The evaluator calls tau2.run.get_tasks via a method we can patch,
    # so we don't need to assign a real get_tasks import here.


# ---------------------------------------------------------------------------
# OPENROUTER_API_KEY enforcement


def test_missing_openrouter_key_raises(monkeypatch):
    """Instantiating Tau2Evaluator without OPENROUTER_API_KEY must raise."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        Tau2Evaluator(sglang_base_url="http://localhost:30000/v1")


def test_explicit_key_overrides_env(monkeypatch):
    """Passing openrouter_api_key arg lets us instantiate without env var."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    ev = Tau2Evaluator(
        sglang_base_url="http://localhost:30000/v1",
        openrouter_api_key="sk-explicit",
    )
    assert ev.openrouter_api_key == "sk-explicit"


# ---------------------------------------------------------------------------
# tau2 not installed branch


def test_evaluate_all_graceful_when_tau2_missing(openrouter_env, monkeypatch, retail_spec):
    """When _TAU2_AVAILABLE is False, evaluate_all returns empty result,
    does not raise, and logs a warning."""
    monkeypatch.setattr(te, "_TAU2_AVAILABLE", False)
    ev = Tau2Evaluator(sglang_base_url="http://localhost:30000/v1")
    result = asyncio.run(ev.evaluate_all([retail_spec]))
    assert isinstance(result, Tau2EvalResult)
    assert result.total_simulations == 0
    assert result.total_errors == 0
    assert result.overall_pass_at_1 == 0.0


def test_evaluate_all_empty_specs(openrouter_env, force_tau2_available):
    """Empty spec list returns empty result without touching tau2."""
    ev = Tau2Evaluator(sglang_base_url="http://localhost:30000/v1")
    result = asyncio.run(ev.evaluate_all([]))
    assert result.total_tasks == 0
    assert result.total_simulations == 0
    assert result.per_spec_results == []


# ---------------------------------------------------------------------------
# Happy path: mocked tau2 returns clean results


def test_pass_at_1_computation(openrouter_env, force_tau2_available, retail_spec):
    """Mock run_tasks to return 4 simulations with rewards [1, 1, 0, 0].
    Expect pass_at_1 == 0.5, errors == 0, num_tasks == 4.
    """
    ev = Tau2Evaluator(sglang_base_url="http://localhost:30000/v1")

    fake_tasks = [f"task_{i}" for i in range(4)]
    fake_results = _make_results([1.0, 1.0, 0.0, 0.0])

    with patch.object(ev, "_load_spec_tasks", return_value=fake_tasks), \
         patch.object(ev, "_run_spec_sync", return_value=fake_results):
        result = asyncio.run(ev.evaluate_all([retail_spec]))

    assert len(result.per_spec_results) == 1
    spec_result = result.per_spec_results[0]
    assert spec_result.num_simulations == 4
    assert spec_result.num_passed == 2
    assert spec_result.num_failed == 2
    assert spec_result.errors == 0
    assert spec_result.pass_at_1 == 0.5
    assert spec_result.pass_at_1_clean == 0.5  # no errors → same as strict

    assert result.total_passed == 2
    assert result.total_simulations == 4
    assert result.overall_pass_at_1 == 0.5
    assert result.total_errors == 0


def test_multiple_specs_aggregation(
    openrouter_env, force_tau2_available, retail_spec, telecom_spec
):
    """Two specs with different pass rates aggregate to overall_pass_at_1
    weighted by total simulation count."""
    ev = Tau2Evaluator(sglang_base_url="http://localhost:30000/v1")

    fake_tasks = [f"t_{i}" for i in range(4)]
    # retail: 4 sims, 2 pass → 0.5
    # telecom: 4 sims, 1 pass → 0.25
    # overall: 3/8 = 0.375
    retail_results = _make_results([1.0, 1.0, 0.0, 0.0])
    telecom_results = _make_results([1.0, 0.0, 0.0, 0.0])

    call_count = {"n": 0}
    def _fake_run(spec, tasks):
        call_count["n"] += 1
        return retail_results if spec.domain == "retail" else telecom_results

    with patch.object(ev, "_load_spec_tasks", return_value=fake_tasks), \
         patch.object(ev, "_run_spec_sync", side_effect=_fake_run):
        result = asyncio.run(ev.evaluate_all([retail_spec, telecom_spec]))

    assert len(result.per_spec_results) == 2
    assert result.per_spec_results[0].pass_at_1 == 0.5
    assert result.per_spec_results[1].pass_at_1 == 0.25
    assert result.total_simulations == 8
    assert result.total_passed == 3
    assert result.overall_pass_at_1 == pytest.approx(3 / 8)


# ---------------------------------------------------------------------------
# Error accounting (option B — strict denominator)


def test_errors_included_in_strict_denominator(
    openrouter_env, force_tau2_available, retail_spec
):
    """With 4 tasks expected but run_tasks raising, all 4 become errors and
    pass_at_1 == 0 (strict option B)."""
    ev = Tau2Evaluator(sglang_base_url="http://localhost:30000/v1")

    fake_tasks = [f"t_{i}" for i in range(4)]

    with patch.object(ev, "_load_spec_tasks", return_value=fake_tasks), \
         patch.object(ev, "_run_spec_sync", side_effect=RuntimeError("boom")):
        result = asyncio.run(ev.evaluate_all([retail_spec]))

    spec_result = result.per_spec_results[0]
    assert spec_result.num_simulations == 4
    assert spec_result.errors == 4
    assert spec_result.num_passed == 0
    assert spec_result.pass_at_1 == 0.0
    # pass_at_1_clean divides by (4 - 4) = 0 → fallback to 0.0, not NaN
    assert spec_result.pass_at_1_clean == 0.0


@pytest.mark.parametrize("err_termination", [
    "infrastructure_error",
    "unexpected_error",
    "too_many_errors",
    "timeout",
    "agent_error",
    "user_error",
])
def test_error_termination_reason_counts_as_error(
    openrouter_env, force_tau2_available, retail_spec, err_termination
):
    """Simulations with termination_reason in the error set count as errors,
    not as failures. Covers every enum value we treat as infra/error."""
    ev = Tau2Evaluator(sglang_base_url="http://localhost:30000/v1")

    fake_tasks = [f"t_{i}" for i in range(4)]
    # 2 pass, 1 error, 1 fail → strict = 2/4 = 0.5, clean = 2/3
    sims = [
        _FakeSimulation("t0", "agent_stop", 4, _FakeRewardInfo(1.0)),
        _FakeSimulation("t1", "agent_stop", 4, _FakeRewardInfo(1.0)),
        _FakeSimulation("t2", err_termination, 0, _FakeRewardInfo(0.0)),
        _FakeSimulation("t3", "agent_stop", 4, _FakeRewardInfo(0.0)),
    ]
    fake_results = _FakeResults(simulations=sims)

    with patch.object(ev, "_load_spec_tasks", return_value=fake_tasks), \
         patch.object(ev, "_run_spec_sync", return_value=fake_results):
        result = asyncio.run(ev.evaluate_all([retail_spec]))

    spec_result = result.per_spec_results[0]
    assert spec_result.num_simulations == 4
    assert spec_result.num_passed == 2
    assert spec_result.num_failed == 1
    assert spec_result.errors == 1, f"{err_termination} should count as error"
    assert spec_result.pass_at_1 == 0.5
    assert spec_result.pass_at_1_clean == pytest.approx(2 / 3)


@pytest.mark.parametrize("normal_termination", [
    "user_stop",
    "agent_stop",
    "max_steps",
    "context_window_exceeded",
])
def test_normal_termination_reason_counts_as_failure(
    openrouter_env, force_tau2_available, retail_spec, normal_termination
):
    """Normal terminations (user_stop, agent_stop, max_steps,
    context_window_exceeded) with reward < 1.0 must count as failures,
    NOT errors. They should hurt pass@1 the same way the tau2 paper
    reports it."""
    ev = Tau2Evaluator(sglang_base_url="http://localhost:30000/v1")

    fake_tasks = [f"t_{i}" for i in range(4)]
    # 1 pass (agent_stop + reward 1.0), 3 normal failures with varying reasons
    sims = [
        _FakeSimulation("t0", "agent_stop", 4, _FakeRewardInfo(1.0)),
        _FakeSimulation("t1", normal_termination, 4, _FakeRewardInfo(0.0)),
        _FakeSimulation("t2", normal_termination, 5, _FakeRewardInfo(0.0)),
        _FakeSimulation("t3", normal_termination, 6, _FakeRewardInfo(0.0)),
    ]
    fake_results = _FakeResults(simulations=sims)

    with patch.object(ev, "_load_spec_tasks", return_value=fake_tasks), \
         patch.object(ev, "_run_spec_sync", return_value=fake_results):
        result = asyncio.run(ev.evaluate_all([retail_spec]))

    spec_result = result.per_spec_results[0]
    assert spec_result.num_simulations == 4
    assert spec_result.num_passed == 1
    assert spec_result.num_failed == 3
    assert spec_result.errors == 0, f"{normal_termination} must NOT count as error"
    assert spec_result.pass_at_1 == 0.25
    assert spec_result.pass_at_1_clean == 0.25  # no errors → same as strict


def test_missing_simulations_counted_as_errors(
    openrouter_env, force_tau2_available, retail_spec
):
    """If tau2 returns fewer simulations than expected (max_errors bailout),
    the missing ones are counted as errors so denominators stay consistent."""
    ev = Tau2Evaluator(sglang_base_url="http://localhost:30000/v1")

    fake_tasks = [f"t_{i}" for i in range(4)]
    # Only 2 simulations returned instead of 4
    fake_results = _make_results([1.0, 0.0])

    with patch.object(ev, "_load_spec_tasks", return_value=fake_tasks), \
         patch.object(ev, "_run_spec_sync", return_value=fake_results):
        result = asyncio.run(ev.evaluate_all([retail_spec]))

    spec_result = result.per_spec_results[0]
    assert spec_result.num_simulations == 4  # expected, not 2
    assert spec_result.num_passed == 1
    assert spec_result.errors == 2  # 2 missing
    assert spec_result.pass_at_1 == 0.25  # 1/4


def test_load_tasks_failure_is_isolated(
    openrouter_env, force_tau2_available, retail_spec, telecom_spec
):
    """If get_tasks fails for one spec, the other specs still run."""
    ev = Tau2Evaluator(sglang_base_url="http://localhost:30000/v1")

    fake_tasks = [f"t_{i}" for i in range(4)]
    fake_results = _make_results([1.0, 1.0, 1.0, 0.0])

    def _fake_load(spec):
        if spec.domain == "retail":
            raise FileNotFoundError("no retail data")
        return fake_tasks

    with patch.object(ev, "_load_spec_tasks", side_effect=_fake_load), \
         patch.object(ev, "_run_spec_sync", return_value=fake_results):
        result = asyncio.run(ev.evaluate_all([retail_spec, telecom_spec]))

    assert len(result.per_spec_results) == 2
    retail_result, telecom_result = result.per_spec_results
    # retail failed to load — zero sims, zero errors, not a disaster
    assert retail_result.num_simulations == 0
    assert retail_result.num_passed == 0
    # telecom ran normally
    assert telecom_result.num_simulations == 4
    assert telecom_result.num_passed == 3
    assert telecom_result.pass_at_1 == 0.75


# ---------------------------------------------------------------------------
# Metric key naming


def test_metrics_dict_uses_gem_eval_prefix(openrouter_env, force_tau2_available, retail_spec):
    """All metric keys must nest under `gem_eval/tau2_*` so they cluster
    with GEM metrics in the same W&B group."""
    ev = Tau2Evaluator(sglang_base_url="http://localhost:30000/v1")

    fake_tasks = [f"t_{i}" for i in range(4)]
    fake_results = _make_results([1.0, 0.0, 0.0, 0.0])

    with patch.object(ev, "_load_spec_tasks", return_value=fake_tasks), \
         patch.object(ev, "_run_spec_sync", return_value=fake_results):
        result = asyncio.run(ev.evaluate_all([retail_spec]))

    metrics = result.to_metrics_dict(prefix="gem_eval")
    # Every key must start with gem_eval/tau2_
    for k in metrics:
        assert k.startswith("gem_eval/tau2_"), f"unexpected key: {k}"
    # Specific keys the hook relies on
    assert "gem_eval/tau2_overall_pass_at_1" in metrics
    assert "gem_eval/tau2_total_simulations" in metrics
    assert "gem_eval/tau2_total_errors" in metrics
    assert "gem_eval/tau2_retail_test/pass_at_1" in metrics
    assert "gem_eval/tau2_retail_test/pass_at_1_clean" in metrics
    assert "gem_eval/tau2_retail_test/errors" in metrics
    assert "gem_eval/tau2_retail_test/mean_num_turns" in metrics
    assert metrics["gem_eval/tau2_retail_test/pass_at_1"] == 0.25


def test_metrics_dict_custom_prefix(openrouter_env, force_tau2_available, retail_spec):
    """Non-default prefix still produces tau2_ sub-prefix keys."""
    ev = Tau2Evaluator(sglang_base_url="http://localhost:30000/v1")
    fake_tasks = [f"t_{i}" for i in range(2)]
    fake_results = _make_results([1.0, 1.0])
    # Shrink retail spec to truncate=2 for this test so expectations match
    retail_spec.truncate = 2
    with patch.object(ev, "_load_spec_tasks", return_value=fake_tasks), \
         patch.object(ev, "_run_spec_sync", return_value=fake_results):
        result = asyncio.run(ev.evaluate_all([retail_spec]))
    metrics = result.to_metrics_dict(prefix="custom_eval")
    assert "custom_eval/tau2_overall_pass_at_1" in metrics


# ---------------------------------------------------------------------------
# Consistency across repeated calls


def test_evaluate_all_is_idempotent_on_repeat(
    openrouter_env, force_tau2_available, retail_spec
):
    """Calling evaluate_all twice with the same mocks gives the same result.
    Guards against hidden global state in the evaluator."""
    ev = Tau2Evaluator(sglang_base_url="http://localhost:30000/v1")
    fake_tasks = [f"t_{i}" for i in range(4)]
    fake_results = _make_results([1.0, 1.0, 0.0, 0.0])

    with patch.object(ev, "_load_spec_tasks", return_value=fake_tasks), \
         patch.object(ev, "_run_spec_sync", return_value=fake_results):
        r1 = asyncio.run(ev.evaluate_all([retail_spec]))
        r2 = asyncio.run(ev.evaluate_all([retail_spec]))

    assert r1.overall_pass_at_1 == r2.overall_pass_at_1
    assert r1.total_passed == r2.total_passed
    assert r1.total_errors == r2.total_errors


# ---------------------------------------------------------------------------
# Constant sanity


def test_sglang_model_name_constant():
    """SGLang model name constant uses openai/ prefix (LiteLLM routing)."""
    assert SGLANG_MODEL_NAME.startswith("openai/")
    # Should not be a well-known model name (cost registry hit)
    assert "gpt-" not in SGLANG_MODEL_NAME
