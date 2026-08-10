"""Base class for tool-use environments.

Handles the <tool_call> JSON protocol, tool dispatch, <answer> submission,
and format error handling. Subclasses only need to implement reset(),
solution(), and tool_xxx() methods.

Tools are defined as OpenAI-format schemas via get_tools(). The orchestrator
passes these to apply_chat_template(tools=...) for native tool calling.
"""

import json
import re
from typing import Any, Dict, List, Tuple


class ToolUseBaseEnv:
    """Base environment for tool-use tasks with <tool_call> protocol.

    The Reasoning Agent interacts via:
      <tool_call>{"name": "tool_name", "arguments": {"key": "value"}}</tool_call>
      <answer>final result</answer>

    Subclasses must implement:
      - reset(seed=None) -> Tuple[str, dict]: set up self._tools, self._state,
        self._expected_answer, return (observation, {})
      - solution() -> str: return correct tool call sequence + answer
      - tool_xxx(**kwargs) -> str: one method per simulated tool

    The base class handles step(), parsing, dispatch, answer checking,
    and observation formatting. Tools are also exposed via get_tools() in
    OpenAI JSON schema format for native tool calling integration.
    """

    def __init__(self, max_turns=10, **kwargs):
        self.max_turns = max_turns
        self.turn_count = 0
        self._tools = {}           # {name: {"description": str, "parameters": dict}}
        self._state = {}           # simulated backend state
        self._call_history = []    # track agent's tool calls
        self._expected_answer = None
        self.reset()

    def reset(self, seed=None) -> Tuple[str, dict]:
        """Generate task, data, tools, and expected answer.

        Subclasses MUST override this method.
        """
        raise NotImplementedError("Subclass must implement reset()")

    def solution(self) -> str:
        """Return the correct tool call sequence and final answer.

        Subclasses MUST override this method.
        """
        raise NotImplementedError("Subclass must implement solution()")

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return tool schemas in OpenAI function-calling format.

        Used by the orchestrator to pass tools via
        apply_chat_template(messages, tools=env.get_tools()).

        Returns a list of tool dicts, each with:
          {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

        Default implementation converts self._tools (set in reset()) to OpenAI format.
        Subclasses may override for custom schemas.
        """
        tools = []
        for name, info in self._tools.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info.get("description", ""),
                    "parameters": info.get("parameters", {
                        "type": "object",
                        "properties": {},
                    }),
                },
            })
        return tools

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call (pre-parsed by orchestrator).

        Args:
            name: Tool name
            arguments: Tool arguments dict

        Returns:
            {"output": str, "returncode": int} where returncode 0 = success
        """
        method = getattr(self, f"tool_{name}", None)
        if method is None:
            return {"output": f"Unknown tool '{name}'", "returncode": 1}
        try:
            result = method(**arguments)
            self._call_history.append({
                "name": name,
                "arguments": arguments,
                "result": str(result),
            })
            return {"output": str(result), "returncode": 0}
        except Exception as e:
            return {"output": f"Error calling {name}: {e}", "returncode": 1}

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        """Parse <tool_call> or <answer> from action, dispatch accordingly."""
        self.turn_count += 1
        truncated = self.turn_count >= self.max_turns

        # Try <answer> first
        answer_match = re.search(r'<answer>(.*?)</answer>', action, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            if self._check_answer(answer):
                return ("Correct!", 1.0, True, False, {})
            if truncated:
                return (f"Wrong answer. Solution:\n{self.solution()}", 0.0, False, True, {})
            return (self._incorrect_answer_feedback(), 0.0, False, False, {})

        # Try <tool_call>
        tc_match = re.search(r'<tool_call>(.*?)</tool_call>', action, re.DOTALL)
        if not tc_match:
            if truncated:
                return (f"Time up. Solution:\n{self.solution()}", 0.0, False, True, {})
            return (
                "Format error: use tools to call a function, "
                "or <answer>result</answer> to submit.",
                -0.1, False, False, {}
            )

        # Parse JSON
        try:
            call = json.loads(tc_match.group(1))
            tool_name = call.get("name", "")
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
        except (json.JSONDecodeError, AttributeError):
            if truncated:
                return (f"Time up. Solution:\n{self.solution()}", 0.0, False, True, {})
            return (
                "Format error: invalid tool call arguments.",
                -0.1, False, False, {}
            )

        # Dispatch to tool_xxx method
        method = getattr(self, f"tool_{tool_name}", None)
        if method is None:
            avail = ", ".join(self._tools.keys())
            if truncated:
                return (f"Time up. Solution:\n{self.solution()}", 0.0, False, True, {})
            return (
                f"Unknown tool '{tool_name}'. Available: {avail}",
                0.0, False, False, {}
            )

        try:
            result = method(**arguments)
            self._call_history.append({
                "name": tool_name,
                "arguments": arguments,
                "result": str(result),
            })
        except Exception as e:
            result = f"Error calling {tool_name}: {e}"

        obs = f"Tool result:\n{result}"
        if truncated:
            obs += f"\n\nTime up. Solution:\n{self.solution()}"
            return (obs, 0.0, False, True, {})
        return (obs, 0.0, False, False, {})

    def _obs_suffix(self) -> str:
        """Deprecated: previously embedded tool definitions in observations.

        Now a no-op — tools are passed via apply_chat_template(tools=...)
        instead. Kept for backward compatibility with previously generated
        game subclasses that call self._obs_suffix() in reset().
        """
        return ""

    def _incorrect_answer_feedback(self) -> str:
        """Feedback string for a wrong <answer> submission.

        For MULTI-TURN games (those that expose _user_messages + _current_msg),
        name the step still pending so the agent acts on the unmet instruction
        instead of re-submitting <answer>done</answer> into an information-free
        "Incorrect" loop (the dominant deadlock failure mode). Single-task games
        that don't set these attrs fall back to the original message unchanged.
        """
        base = "Incorrect. Keep using tools."
        msgs = getattr(self, "_user_messages", None)
        idx = getattr(self, "_current_msg", None)
        if isinstance(msgs, (list, tuple)) and isinstance(idx, int) and 0 <= idx < len(msgs):
            return (
                f"{base} Not all steps are complete — you are still on step "
                f"{idx + 1}/{len(msgs)}: \"{msgs[idx]}\" Complete this step with the "
                f"tools before submitting <answer>done</answer>."
            )
        return base

    def _check_answer(self, answer: str) -> bool:
        """Tolerant answer comparison. Subclasses can override for custom logic."""
        a = str(answer).strip()
        e = str(self._expected_answer).strip()
        # Exact match (case-insensitive)
        if a.lower() == e.lower():
            return True
        # Numeric comparison with tolerance
        try:
            a_clean = a.replace("$", "").replace(",", "").strip()
            e_clean = e.replace("$", "").replace(",", "").strip()
            return abs(float(a_clean) - float(e_clean)) < 0.01
        except (ValueError, TypeError):
            return False

    def close(self):
        pass
