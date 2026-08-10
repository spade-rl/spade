"""Parse <tool_call> blocks from model responses.

Supports both Qwen3.5 XML format and Qwen2.5 JSON format.
Copied from usim_rl/usim/core/coding_orchestrator.py (lines 37-153).
"""

import html
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Regex patterns for Qwen3.5 XML-style tool calls:
#   <tool_call>
#   <function=name>
#   <parameter=key>value</parameter>
#   </function>
#   </tool_call>
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>", re.DOTALL
)
_FUNCTION_RE = re.compile(
    r"<function=(\w+)>(.*?)</function>", re.DOTALL
)
_PARAMETER_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL
)

# Also support JSON-style tool calls (Qwen2.5 format):
#   <tool_call>
#   {"name": "bash", "arguments": {"command": "ls"}}
#   </tool_call>
_JSON_TOOL_CALL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', re.DOTALL
)


def _parse_xml_block(block_content: str) -> Optional[Dict[str, Any]]:
    """Parse a single <tool_call> block (XML format: Qwen3.5/qwen3_coder)."""
    func_match = _FUNCTION_RE.search(block_content)
    if func_match:
        func_name = func_match.group(1)
        func_body = func_match.group(2)
        params = {}
        for param_match in _PARAMETER_RE.finditer(func_body):
            key = param_match.group(1)
            val = html.unescape(param_match.group(2).strip())
            # Try to parse as JSON (for numeric/bool values)
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            params[key] = val
        return {"name": func_name, "arguments": params, "id": ""}
    return None


def _parse_json_block(block_content: str) -> Optional[Dict[str, Any]]:
    """Parse a single <tool_call> block (JSON format: Qwen2.5/qwen25)."""
    match = _JSON_TOOL_CALL_RE.search(block_content)
    if match:
        func_name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except (json.JSONDecodeError, ValueError):
            args = {}
        return {"name": func_name, "arguments": args, "id": ""}
    # Also try raw JSON parse
    try:
        data = json.loads(block_content.strip())
        if isinstance(data, dict) and "name" in data:
            args = data.get("arguments", data.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            return {"name": data["name"], "arguments": args, "id": ""}
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def parse_tool_calls(
    response_text: str,
    tools_schema: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Parse tool calls from agent response text.

    Self-contained parser — no sglang dependency. Supports both XML format
    (Qwen3.5/qwen3_coder) and JSON format (Qwen2.5/qwen25) inside <tool_call> blocks.

    Args:
        response_text: Raw model response text
        tools_schema: Optional list of tool schemas (if empty/None, returns no calls)

    Returns:
        {"normal_text": str, "calls": list} where each call is
        {"name": str, "arguments": dict, "id": str}.
    """
    if response_text.endswith("<|im_end|>"):
        response_text = response_text[:-10]
    response_text = response_text.strip()

    if tools_schema is not None and not tools_schema:
        return {"normal_text": response_text, "calls": []}

    calls = []
    normal_parts = []
    cursor = 0

    for match in _TOOL_CALL_BLOCK_RE.finditer(response_text):
        # Collect text before this block
        normal_parts.append(response_text[cursor:match.start()])
        cursor = match.end()

        block_content = match.group(1)
        # Try XML format first (Qwen3.5), then JSON (Qwen2.5)
        parsed = _parse_xml_block(block_content)
        if parsed is None:
            parsed = _parse_json_block(block_content)
        if parsed is not None:
            calls.append(parsed)
        else:
            logger.warning(
                f"Could not parse tool_call block: {block_content[:200]!r}"
            )

    # Remaining text after last block
    normal_parts.append(response_text[cursor:])
    normal_text = "".join(normal_parts).strip()

    if not calls:
        logger.debug(
            f"No <tool_call> blocks found. "
            f"Response preview: {response_text[:300]!r}"
        )

    return {"normal_text": normal_text, "calls": calls}
