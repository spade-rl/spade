"""tau2-bench task registry for evaluation.

Provides per-spec evaluation configuration for running tau2-bench domains
(retail, telecom, ...) against the model being trained.

YAML config format lives as an optional ``tau2_eval:`` sibling section in
the same file consumed by ``load_gem_eval_config``. Both loaders can read
the same file independently; either section may be omitted.

Example::

    tau2_eval:
      defaults:
        max_concurrency: 48
        num_trials: 1
        max_steps: 100
        agent_temperature: 1.0
        agent_max_tokens: 4096
        user_llm: "openrouter/openai/gpt-5-mini"
        user_temperature: 0.0
        user_max_tokens: 2048

      tasks:
        - domain: retail
          split: test
        - domain: telecom
          split: test
          max_concurrency: 32   # per-spec override
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


DEFAULT_USER_LLM = "openrouter/openai/gpt-5-mini"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


@dataclass
class Tau2EvalDefaults:
    """Default settings for tau2 evaluation."""
    max_concurrency: int = 48
    num_trials: int = 1
    max_steps: int = 100
    agent_temperature: float = 1.0
    agent_max_tokens: int = 4096
    # Optional sampling controls forwarded to the agent LLM. <=0 means "do not
    # send", letting the server use its own default. Both default to 0 so
    # existing configs that only set agent_temperature keep their behavior.
    agent_top_p: float = 0.0
    agent_top_k: int = 0
    user_llm: str = DEFAULT_USER_LLM
    user_temperature: float = 0.0
    user_max_tokens: int = 2048


@dataclass
class Tau2TaskSpec:
    """Parsed tau2 task specification — one (domain, split) pair to evaluate.

    Every setting from ``Tau2EvalDefaults`` can be overridden per-spec.
    """
    domain: str                  # "retail", "telecom", ...
    split: str                   # "test", "train", "base", "small", "full"
    max_concurrency: int = 48
    num_trials: int = 1
    max_steps: int = 100
    agent_temperature: float = 1.0
    agent_max_tokens: int = 4096
    agent_top_p: float = 0.0
    agent_top_k: int = 0
    user_llm: str = DEFAULT_USER_LLM
    user_temperature: float = 0.0
    user_max_tokens: int = 2048
    # Testing/debugging: run only the first N tasks of the loaded split.
    # Production runs should leave this as None.
    truncate: Optional[int] = None

    @property
    def label(self) -> str:
        """W&B-safe label e.g. ``retail_test``, ``telecom_test``."""
        return f"{self.domain}_{self.split}"


def _require(entry: Dict[str, Any], key: str, context: str) -> Any:
    """Raise ValueError if a required key is missing from a task entry."""
    if key not in entry:
        raise ValueError(
            f"[TAU2-EVAL] Invalid task entry (missing '{key}') in {context}: {entry}"
        )
    return entry[key]


def load_tau2_eval_config(
    config_path: str,
) -> Tuple[Tau2EvalDefaults, List[Tau2TaskSpec]]:
    """Load the ``tau2_eval:`` section from a SPARE eval YAML.

    The same file can also contain a ``gem_eval:`` section; they are
    parsed independently by their respective loaders.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Tuple of ``(defaults, task_specs)``. If the file has no
        ``tau2_eval:`` section (or the section has no tasks), returns
        ``(Tau2EvalDefaults(), [])`` so callers can gracefully skip tau2
        evaluation.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If a task entry is malformed (missing ``domain`` or
            ``split``, or has an unexpected shape).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"tau2 eval config not found: {config_path}")

    with open(path) as f:
        content = f.read()
    # Expand environment variables (e.g. ${WORKSPACE_DIR})
    content = os.path.expandvars(content)
    raw = yaml.safe_load(content)

    if raw is None:
        logger.info("[TAU2-EVAL] Empty config file: %s — skipping tau2 eval", config_path)
        return Tau2EvalDefaults(), []

    tau2_section = raw.get("tau2_eval")
    if not tau2_section:
        logger.info(
            "[TAU2-EVAL] No 'tau2_eval' section in %s — skipping tau2 eval",
            config_path,
        )
        return Tau2EvalDefaults(), []

    defaults_dict: Dict[str, Any] = tau2_section.get("defaults", {}) or {}
    defaults = Tau2EvalDefaults(
        max_concurrency=defaults_dict.get("max_concurrency", 48),
        num_trials=defaults_dict.get("num_trials", 1),
        max_steps=defaults_dict.get("max_steps", 100),
        agent_temperature=defaults_dict.get("agent_temperature", 1.0),
        agent_max_tokens=defaults_dict.get("agent_max_tokens", 4096),
        agent_top_p=defaults_dict.get("agent_top_p", 0.0),
        agent_top_k=defaults_dict.get("agent_top_k", 0),
        user_llm=defaults_dict.get("user_llm", DEFAULT_USER_LLM),
        user_temperature=defaults_dict.get("user_temperature", 0.0),
        user_max_tokens=defaults_dict.get("user_max_tokens", 2048),
    )

    tasks_list = tau2_section.get("tasks", []) or []
    if not isinstance(tasks_list, list):
        raise ValueError(
            f"[TAU2-EVAL] Expected 'tau2_eval.tasks' to be a list in {config_path}, "
            f"got {type(tasks_list).__name__}"
        )

    specs: List[Tau2TaskSpec] = []
    for i, entry in enumerate(tasks_list):
        if not isinstance(entry, dict):
            raise ValueError(
                f"[TAU2-EVAL] Task entry #{i} in {config_path} must be a mapping, "
                f"got {type(entry).__name__}: {entry}"
            )
        context = f"{config_path} (task #{i})"
        domain = _require(entry, "domain", context)
        split = _require(entry, "split", context)
        specs.append(Tau2TaskSpec(
            domain=str(domain),
            split=str(split),
            max_concurrency=entry.get("max_concurrency", defaults.max_concurrency),
            num_trials=entry.get("num_trials", defaults.num_trials),
            max_steps=entry.get("max_steps", defaults.max_steps),
            agent_temperature=entry.get("agent_temperature", defaults.agent_temperature),
            agent_max_tokens=entry.get("agent_max_tokens", defaults.agent_max_tokens),
            agent_top_p=entry.get("agent_top_p", defaults.agent_top_p),
            agent_top_k=entry.get("agent_top_k", defaults.agent_top_k),
            user_llm=entry.get("user_llm", defaults.user_llm),
            user_temperature=entry.get("user_temperature", defaults.user_temperature),
            user_max_tokens=entry.get("user_max_tokens", defaults.user_max_tokens),
            truncate=entry.get("truncate"),
        ))

    logger.info(
        "[TAU2-EVAL] Loaded tau2 config: %d specs, defaults(max_concurrency=%d, num_trials=%d)",
        len(specs), defaults.max_concurrency, defaults.num_trials,
    )

    return defaults, specs
