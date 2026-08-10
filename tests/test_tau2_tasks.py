"""Unit tests for spare.core.eval.tau2_tasks.

Tests the YAML config loader independently of tau2-bench itself — no
tau2 import required. Exercises:
- Graceful skip when tau2_eval section is absent
- Per-spec override propagation
- Coexistence with gem_eval section in the same file
- ValueError on malformed entries
- FileNotFoundError on missing path
- Empty file handling
"""

from pathlib import Path

import pytest

from spare.core.eval.tau2_tasks import (
    DEFAULT_USER_LLM,
    Tau2EvalDefaults,
    Tau2TaskSpec,
    load_tau2_eval_config,
)


FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Graceful skip paths


def test_load_skips_when_tau2_section_missing():
    """No tau2_eval section → return empty specs without raising."""
    defaults, specs = load_tau2_eval_config(str(FIXTURES / "gem_eval_only.yaml"))
    assert isinstance(defaults, Tau2EvalDefaults)
    assert specs == []


def test_load_skips_for_empty_file(tmp_path):
    """YAML file with no content → return empty defaults + []."""
    f = tmp_path / "empty.yaml"
    f.write_text("")
    defaults, specs = load_tau2_eval_config(str(f))
    assert isinstance(defaults, Tau2EvalDefaults)
    assert specs == []


def test_load_skips_when_tau2_tasks_empty(tmp_path):
    """tau2_eval section present but with no tasks → return empty specs."""
    f = tmp_path / "empty_tasks.yaml"
    f.write_text("tau2_eval:\n  defaults:\n    max_concurrency: 8\n  tasks: []\n")
    defaults, specs = load_tau2_eval_config(str(f))
    assert defaults.max_concurrency == 8
    assert specs == []


# ---------------------------------------------------------------------------
# Standalone tau2_eval file


def test_load_tau2_only_file_parses_defaults_and_specs():
    """tau2_eval-only fixture loads both defaults and per-spec overrides."""
    defaults, specs = load_tau2_eval_config(str(FIXTURES / "gem_eval_tau2_only.yaml"))

    # Defaults from the fixture
    assert defaults.max_concurrency == 16
    assert defaults.num_trials == 1
    assert defaults.max_steps == 50
    assert defaults.agent_temperature == 1.0
    assert defaults.agent_max_tokens == 2048
    assert defaults.user_llm == "openrouter/openai/gpt-5-mini"
    assert defaults.user_temperature == 0.0

    assert len(specs) == 2

    # First spec inherits all defaults.
    retail = specs[0]
    assert retail.domain == "retail"
    assert retail.split == "test"
    assert retail.max_concurrency == 16  # from defaults
    assert retail.max_steps == 50
    assert retail.truncate is None
    assert retail.label == "retail_test"

    # Second spec has per-spec overrides.
    telecom = specs[1]
    assert telecom.domain == "telecom"
    assert telecom.split == "test"
    assert telecom.max_concurrency == 8   # overridden
    assert telecom.max_steps == 50        # from defaults
    assert telecom.truncate == 2          # overridden
    assert telecom.label == "telecom_test"


# ---------------------------------------------------------------------------
# Unified file: both gem_eval and tau2_eval sections


def test_load_unified_file_parses_tau2_independently():
    """When gem_eval and tau2_eval coexist, tau2 loader reads only its own."""
    defaults, specs = load_tau2_eval_config(
        str(FIXTURES / "gem_eval_unified_smoke.yaml"),
    )
    assert defaults.max_concurrency == 4
    assert len(specs) == 1
    assert specs[0].domain == "retail"
    assert specs[0].split == "test"
    assert specs[0].truncate == 1


def test_unified_file_does_not_affect_gem_loader():
    """Smoke: the unified fixture should also load cleanly via gem loader.

    This is a cheap regression guard — we don't assert on gem internals here,
    only that importing the gem loader and calling it on the unified file
    does not raise. Deeper gem tests live in tests/test_gem_tasks.py.
    """
    from spare.core.eval.gem_tasks import load_gem_eval_config

    defaults, specs = load_gem_eval_config(
        str(FIXTURES / "gem_eval_unified_smoke.yaml"),
    )
    assert len(specs) == 1
    assert specs[0].task_id == "rg:graph_color-easy"


# ---------------------------------------------------------------------------
# Error paths


def test_load_missing_file_raises():
    """Non-existent path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_tau2_eval_config("/nonexistent/path/to/config.yaml")


def test_load_malformed_entry_raises_value_error():
    """Missing `split` field on a task entry must raise ValueError."""
    with pytest.raises(ValueError, match="split"):
        load_tau2_eval_config(str(FIXTURES / "gem_eval_malformed_tau2.yaml"))


def test_load_non_list_tasks_raises(tmp_path):
    """tau2_eval.tasks must be a list, not a mapping."""
    f = tmp_path / "bad.yaml"
    f.write_text(
        "tau2_eval:\n"
        "  defaults:\n"
        "    max_concurrency: 8\n"
        "  tasks:\n"
        "    domain: retail\n"
        "    split: test\n"
    )
    with pytest.raises(ValueError, match="list"):
        load_tau2_eval_config(str(f))


def test_load_non_dict_task_entry_raises(tmp_path):
    """Each task entry must be a mapping, not a string/scalar."""
    f = tmp_path / "bad2.yaml"
    f.write_text(
        "tau2_eval:\n"
        "  tasks:\n"
        "    - retail:test\n"
    )
    with pytest.raises(ValueError, match="mapping"):
        load_tau2_eval_config(str(f))


# ---------------------------------------------------------------------------
# Dataclass behaviour


def test_spec_label_property():
    """Tau2TaskSpec.label combines domain + split."""
    spec = Tau2TaskSpec(domain="retail", split="test")
    assert spec.label == "retail_test"


def test_defaults_include_documented_constants():
    """Defaults match the spec's documented values so the schema is stable."""
    d = Tau2EvalDefaults()
    assert d.max_concurrency == 48
    assert d.num_trials == 1
    assert d.max_steps == 100
    assert d.user_llm == DEFAULT_USER_LLM
    assert d.user_llm == "openrouter/openai/gpt-5-mini"
    assert d.user_temperature == 0.0
