"""Monkeypatches applied to the upstream ACEBench harness to run it offline
against a served OpenAI-compatible endpoint. Moved verbatim out of
``acebench.py`` — see that module's docstring for the routing-patch
rationale.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger("eval_offline.suites.acebench")

# Marker to detect an already-patched file
_PATCH_MARKER = "SPARE_PATCH_MARKER"


def _apply_routing_patch(acebench: Path) -> None:
    """Idempotently route unrecognized ACEBench models to the configured API."""
    api_file = acebench / "model_inference" / "apimodel_inference.py"
    map_file = acebench / "model_inference" / "inference_map.py"
    infer_file = acebench / "model_inference" / "model_infer.py"

    # These independent patches are safe to apply on every invocation.
    _patch_model_infer_lazy_vllm(infer_file)
    _patch_agent_routing(acebench)
    _patch_strip_reasoning(acebench)

    if not api_file.exists():
        logger.warning("[acebench] apimodel_inference.py not found — skipping patch")
        return

    content = api_file.read_text()
    if _PATCH_MARKER in content:
        logger.debug("[acebench] routing patch already applied")
        return

    patch_file = Path(__file__).parents[1] / "patches" / "acebench_route.patch"
    patched_via_git = False
    if patch_file.exists():
        result = subprocess.run(
            ["git", "apply", "--check", str(patch_file)],
            cwd=str(acebench),
            capture_output=True,
        )
        if result.returncode == 0:
            subprocess.run(
                ["git", "apply", str(patch_file)],
                cwd=str(acebench),
                capture_output=True,
            )
            # Verify
            if _PATCH_MARKER in api_file.read_text():
                patched_via_git = True
                logger.info("[acebench] routing patch applied via git apply")

    if not patched_via_git:
        # Fall back to in-place edits
        _patch_apimodel_inference(api_file)
        _patch_inference_map(map_file)
        logger.info("[acebench] routing patch applied via in-place edit")


def _patch_apimodel_inference(api_file: Path) -> None:
    """Add the else branch to APIModelInference.__init__."""
    content = api_file.read_text()

    # Pattern: the closing of the elif "o1" block just before `self.client = ...`
    # We insert an else branch after the last elif block.
    old = '        elif "o1" in self.model_name:\n            api_key = os.getenv("GPT_AGENT_API_KEY")\n            base_url = os.getenv("GPT_BASE_URL")'
    new = (
        '        elif "o1" in self.model_name:\n'
        '            api_key = os.getenv("GPT_AGENT_API_KEY")\n'
        '            base_url = os.getenv("GPT_BASE_URL")\n'
        '        else:  # SPARE_PATCH_MARKER\n'
        '            api_key = os.getenv("ACEBENCH_AGENT_API_KEY", "EMPTY")\n'
        '            base_url = os.getenv("ACEBENCH_AGENT_BASE_URL")'
    )

    if old in content:
        content = content.replace(old, new, 1)
    else:
        # Fallback: find the line `self.client = OpenAI(` and insert before it
        # using a regex that handles variable indentation
        insert_block = (
            "        else:  # SPARE_PATCH_MARKER\n"
            '            api_key = os.getenv("ACEBENCH_AGENT_API_KEY", "EMPTY")\n'
            '            base_url = os.getenv("ACEBENCH_AGENT_BASE_URL")\n'
        )
        content = re.sub(
            r"(\n)([ \t]+self\.client\s*=\s*OpenAI\()",
            r"\n" + insert_block + r"\2",
            content,
            count=1,
        )

    api_file.write_text(content)


def _patch_model_infer_lazy_vllm(infer_file: Path) -> None:
    """Keep ACEBench's API path independent of its optional vLLM runtime."""
    if not infer_file.exists():
        return

    content = infer_file.read_text()
    if _PATCH_MARKER in content:
        return

    top_import = "from vllm import LLM, SamplingParams\n"
    if top_import not in content:
        # An unknown upstream layout should fail through its normal import path.
        return

    content = content.replace(
        top_import,
        f"# {_PATCH_MARKER}: vllm import moved into LLMInfer.__init__ (lazy)\n",
        1,
    )

    lazy = "        from vllm import LLM, SamplingParams  # lazy: keep vllm off the import path\n"
    anchor = "        gpu_ids = get_free_gpu(use_gpu_num=tensor_parallel_size)\n"
    if anchor in content:
        content = content.replace(anchor, lazy + anchor, 1)
    else:
        # Fallback: inject right after the LLMInfer.__init__ signature line.
        content = re.sub(
            r"(\n)([ \t]+)(def __init__\(self, model_path,[^\n]*\) -> None:\n)",
            r"\1\2\3\2    from vllm import LLM, SamplingParams  # lazy\n",
            content,
            count=1,
        )

    infer_file.write_text(content)


def _patch_strip_reasoning(acebench: Path) -> None:
    """Strip ``<think>...</think>`` from the served model's responses in-place.

    This is ACEBench's OWN mechanism for reasoning models: keep thinking ON, let
    the model emit ``<think>reasoning</think>[func(...)]`` in ``message.content``
    (NO server-side reasoning parser, so ``content`` is always a string — never
    ``None``), then drop everything up to and including the last ``</think>``
    before the strict ``[func(...)]`` parse. ACEBench already does this, but only
    for model names containing ``deepseek-r1`` and only in the single-turn path.

    We rewrite every served-model content read

        response.choices[0].message.content
    ->  (response.choices[0].message.content or "").split("</think>")[-1].strip()

    which (a) coerces a None/empty content to ``""`` (a turn with no post-think
    answer no longer crashes ``re.match``), (b) keeps only the post-``</think>``
    answer, and (c) ``.strip()`` leading whitespace so the agent's
    ``re.match(r"\\[.*?\\]", ...)`` anchors on the function call. Applied to the
    three served-model modules (single-turn + both agent roles); the gpt-4o
    user-simulator and local-infer path are left untouched. Idempotent.
    """
    targets = [
        acebench / "model_inference" / "apimodel_inference.py",
        acebench / "model_inference" / "multi_step" / "APIModel_agent.py",
        acebench / "model_inference" / "multi_turn" / "APIModel_agent.py",
    ]
    needle = "response.choices[0].message.content"
    inject = '(response.choices[0].message.content or "").split("</think>")[-1].strip()'
    for target in targets:
        if not target.exists():
            continue
        content = target.read_text()
        if 'split("</think>")' in content:
            continue
        if needle in content:
            content = content.replace(needle, inject)
            target.write_text(content)


def _patch_disable_thinking(acebench: Path) -> None:
    """Disable thinking for the served model without changing the user simulator."""
    targets = [
        acebench / "model_inference" / "apimodel_inference.py",
        acebench / "model_inference" / "multi_step" / "APIModel_agent.py",
        acebench / "model_inference" / "multi_turn" / "APIModel_agent.py",
    ]
    needle = "self.client.chat.completions.create("
    inject = needle + 'extra_body={"chat_template_kwargs": {"enable_thinking": False}}, '
    for target in targets:
        if not target.exists():
            continue
        content = target.read_text()
        if "enable_thinking" in content:
            continue
        if needle in content:
            content = content.replace(needle, inject)
            target.write_text(content)


def _patch_agent_routing(acebench: Path) -> None:
    """Route non-GPT Agent clients to the exact configured model endpoint."""
    targets = [
        acebench / "model_inference" / "multi_step" / "APIModel_agent.py",
        acebench / "model_inference" / "multi_turn" / "APIModel_agent.py",
    ]
    anchor = "        self.client = OpenAI(base_url=base_url, api_key=api_key)\n"
    override = (
        "        # SPARE_PATCH_MARKER: route the served (non-gpt) model to our\n"
        "        # endpoint with its exact served id (family branches above read\n"
        "        # unset envs -> OpenAI fallback, and lowercase the model name).\n"
        '        _ace_base = os.getenv("ACEBENCH_AGENT_BASE_URL")\n'
        '        if _ace_base and "gpt" not in self.model_name:\n'
        "            base_url = _ace_base\n"
        '            api_key = os.getenv("ACEBENCH_AGENT_API_KEY", "EMPTY")\n'
        '            self.model_name = os.getenv("ACEBENCH_SERVED_MODEL", self.model_name)\n'
    )
    for target in targets:
        if not target.exists():
            continue
        content = target.read_text()
        if _PATCH_MARKER in content:
            continue
        if anchor in content:
            content = content.replace(anchor, override + anchor, 1)
            target.write_text(content)


def _patch_inference_map(map_file: Path) -> None:
    """Wrap inference_map in a __missing__-based dict."""
    if not map_file.exists():
        return

    content = map_file.read_text()
    if _PATCH_MARKER in content:
        return

    append_block = (
        "\n\n"
        "# SPARE_PATCH_MARKER — added by eval_offline.suites.acebench\n"
        "class _DefaultAPIInferenceMap(dict):\n"
        '    """Falls back to APIModelInference for unknown model names."""\n'
        "\n"
        "    def __missing__(self, key):\n"
        "        from model_inference.apimodel_inference import APIModelInference\n"
        "        return APIModelInference\n"
        "\n"
        "\n"
        "inference_map = _DefaultAPIInferenceMap(inference_map)\n"
    )
    map_file.write_text(content + append_block)
