"""Evaluation modules for SPARE training.

Evaluators have independent optional dependencies, so public exports are
resolved lazily instead of making every evaluation dependency mandatory.
"""

from spare._lazy import lazy_exports

_EXPORTS = {
    "DEFAULT_GEM_TASKS": ("spare.core.eval.gem_tasks", "DEFAULT_GEM_TASKS"),
    "GemEvalDefaults": ("spare.core.eval.gem_tasks", "GemEvalDefaults"),
    "GemEvalResult": ("spare.core.eval.gem_evaluator", "GemEvalResult"),
    "GemEvaluator": ("spare.core.eval.gem_evaluator", "GemEvaluator"),
    "GemTaskResult": ("spare.core.eval.gem_evaluator", "GemTaskResult"),
    "GemTaskSpec": ("spare.core.eval.gem_tasks", "GemTaskSpec"),
    "load_gem_eval_config": ("spare.core.eval.gem_tasks", "load_gem_eval_config"),
    "run_fixed_model_evaluation": (
        "spare.core.eval.fixed_model_eval",
        "run_fixed_model_evaluation",
    ),
}

__all__ = list(_EXPORTS)
__getattr__ = lazy_exports(__name__, globals(), _EXPORTS)
