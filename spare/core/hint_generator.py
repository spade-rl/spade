"""Hint generator for regret-based environment reward.

Generates hints from game source code using an LLM.
The hint is used to compute regret: R(with_hint) - R(without_hint).

Two modes:
  - External (HintGenerator): Uses OpenAI-compatible API (GPT-4.1-mini, etc.)
  - Self (SelfHintGenerator): Uses the training model's own ModelAdapter
"""

import asyncio
import json
import logging
import re
from typing import Any, Optional

from spare.core.openrouter_adapter import OpenAIModelAdapter

logger = logging.getLogger(__name__)

# Prompts are selected by game type: cognitive or tool-use.

HINT_SYSTEM_PROMPT = """You are analyzing game source code to generate a solving hint for an LLM player.

CONTEXT: The player sees only text observations and responds with \\boxed{answer}. It CANNOT see the source code - only you can. Your hint will be appended to the player's first observation.

TASK: Read the code carefully. Identify:
1. What puzzle the game presents (from the player's perspective, not the code's)
2. How the correct answer is derived (trace through reset() and the checking logic in step())
3. What answer format is expected (a number, a word, a list, etc.)

WRITE a hint that includes:
- The key insight or strategy for solving this specific puzzle
- The expected answer format (e.g., "answer with a single integer", "answer with a comma-separated list")

RULES:
- Keep it to 1-3 sentences (under 100 words)
- Reference only what the player can SEE in the observation, not code internals
- Be specific to this game - no generic advice like "think carefully" or "consider all possibilities"
- Do NOT reveal the exact answer, but DO explain the approach

GOOD hint example: "The sequence follows a rule where each term is the sum of the two preceding terms. Count the pattern and predict the next value. Answer with a single integer."
BAD hint example: "The self.solution variable stores the answer computed in reset(). Try to think about what it could be."

Output ONLY the hint text. No JSON, no markdown, no labels."""


TOOL_USE_HINT_SYSTEM_PROMPT = """You are analyzing a MULTI-TURN tool-use environment's source code to write a PROCEDURAL strategy hint for an LLM agent that will play it.

CONTEXT: The agent acts turn-by-turn. It sees ONE user instruction at a time, calls tools with <tool_call>{"name": "...", "arguments": {...}}</tool_call>, reads each tool result, and submits <answer>done</answer> ONLY after every instruction has been satisfied. It CANNOT see the source code - only you can. Your hint is appended to the agent's FIRST observation.

GOAL: The hint must help the agent get UNSTUCK without SOLVING the task for it. Summarize the approach (a lossy plan), never the answer. Discovering the specific values by calling tools is the SKILL being tested - a hint the agent can win by copy-pasting is worthless, and it trains the generator to make games that are unsolvable WITHOUT the leaked answer.

TASK: Read the code and identify the high-level STRATEGY only: which KINDS of tools/operations are involved, the general order of PHASES (which tool's output feeds the next), and which tools are distractors / out-of-scope.

WRITE a hint that:
- Describes the general approach and the order of PHASES in terms of tool ROLES (e.g. "first look up the relevant record, then update it, then organize the files") - NOT a fully filled-in call sequence.
- Tells the agent to act turn-by-turn and submit <answer>done</answer> ONLY after all steps are complete - never jump straight to a final answer.
- Tells the agent to DISCOVER the specific arguments by calling lookup/list tools and reading each user instruction - it must not assume them.
- If an instruction omits a required argument, say to ASK for it (do not guess); if a request is out of scope of the available tools, say to DECLINE.

HARD RULES (these prevent answer leakage - the #1 failure mode):
- NEVER write concrete data values read from the code or hidden state: no exact IDs, names, file names, paths, amounts, dates, or status strings. Use placeholders like <order_id>, <amount>, or phrases like "the order the user names". If you are about to type a specific value (e.g. 'ORD104', 'savings_67890', 'final_report.pdf'), replace it with a placeholder.
- Do NOT emit a complete, copy-pasteable call-by-call walkthrough with all arguments filled in.
- Reference only tool-level behavior the agent can observe - never reveal code internals (state-variable names, exact criteria expressions).
- Keep it to 2-3 sentences (under 100 words). Specific to THIS environment's tools/goal, but VALUE-FREE.
- Do NOT tell the agent to "answer with a single integer/word" or submit a final answer immediately. This is a multi-turn EXECUTION task.

GOOD hint (strategy + placeholders; the agent still must discover specifics):
"Work one instruction at a time. First use the order-lookup tool to find the order the user refers to, then the order-update tool to change its status; afterwards use the file tools to create the destination folder and move the named file into it. Read each tool result and the next instruction to get the exact arguments - don't assume IDs or paths. Submit <answer>done</answer> only after the final instruction is done."

BAD hint (LEAKS THE ANSWER - never do this): "call find_order(order_id='ORD104'), then update_order_status(order_id='ORD104', status='shipped'), then move_file(src='order_summary.txt', dst='/customer/orders/tracking'). Submit <answer>done</answer>." - it hardcodes exact IDs/paths so the agent wins by copying instead of doing the task.

Output ONLY the hint text. No JSON, no markdown, no labels."""



def get_hint_system_prompt(game_type: str) -> str:
    """Select the tool-use prompt or the default cognitive prompt."""
    if game_type == "tool_use":
        return TOOL_USE_HINT_SYSTEM_PROMPT
    return HINT_SYSTEM_PROMPT


class HintGenerator:
    """Generates hints for games using an external LLM (e.g., GPT-5.1)."""

    def __init__(
        self,
        model_adapter: OpenAIModelAdapter,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        game_type: str = "cognitive",
    ):
        """Initialize the hint generator.

        Args:
            model_adapter: OpenAI-compatible model adapter for hint generation
            temperature: Temperature for hint generation
            max_tokens: Max tokens for hint response
            game_type: "tool_use" selects the multi-turn tool-use hint prompt;
                any other value (default) uses the cognitive \\boxed{} hint prompt
        """
        self.model = model_adapter
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._system_prompt = get_hint_system_prompt(game_type)

    def _parse_hint(self, response_text: str) -> str:
        """Parse hint from model response.

        The prompt asks for plain text output. Falls back to JSON extraction
        for backward compatibility.

        Args:
            response_text: Raw response from the model

        Returns:
            Extracted hint text
        """
        text = response_text.strip()

        # If the model wrapped in JSON despite instructions, extract it
        json_match = re.search(r'```json\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1).strip())
                if isinstance(data, dict) and "hint" in data:
                    return data["hint"]
            except json.JSONDecodeError:
                pass

        return text

    async def generate_hint_async(self, game_code: str) -> str:
        """Generate a hint for a game using the external LLM.

        Args:
            game_code: Python source code of the game

        Returns:
            Hint text string
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Environment code:\n```python\n{game_code}\n```"},
        ]

        results = await asyncio.wait_for(
            self.model.generate_async(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
            timeout=120,  # 2 min hard timeout per hint
        )

        response_text = results[0].get("text", "") if results else ""
        if not response_text:
            raise ValueError("Empty response from hint model")

        return self._parse_hint(response_text)

    async def generate_hint_with_retry(
        self,
        game_code: str,
        max_retries: int = 3,
    ) -> str:
        """Generate a hint with exponential backoff retry on failure.

        Args:
            game_code: Python source code of the game
            max_retries: Maximum number of retry attempts

        Returns:
            Hint text string

        Raises:
            RuntimeError: If all retries fail
        """
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                return await self.generate_hint_async(game_code)
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(
                    f"[HINT] Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

        raise RuntimeError(
            f"Hint generation failed after {max_retries} attempts: {last_error}"
        )


class SelfHintGenerator:
    """Generates hints using the training model itself (self-hint).

    Instead of calling an external API, this uses the same ModelAdapter that
    powers the training rollout. The model reads its own game code and produces
    a hint - matching the paper's claim that "the generator (same LLM, different
    prompt) produces privileged information."
    """

    def __init__(
        self,
        model_adapter: Any,  # ModelAdapter protocol
        temperature: float = 0.3,
        max_tokens: int = 512,
        game_type: str = "cognitive",
    ):
        self.model = model_adapter
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._system_prompt = get_hint_system_prompt(game_type)

    def _parse_hint(self, response_text: str) -> str:
        """Parse hint from model response (same logic as HintGenerator)."""
        text = response_text.strip()

        json_match = re.search(r'```json\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1).strip())
                if isinstance(data, dict) and "hint" in data:
                    return data["hint"]
            except json.JSONDecodeError:
                pass

        return text

    async def generate_hint_async(self, game_code: str) -> str:
        """Generate a hint using the training model.

        Args:
            game_code: Python source code of the game

        Returns:
            Hint text string
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Environment code:\n```python\n{game_code}\n```"},
        ]

        results = await asyncio.wait_for(
            self.model.generate_async(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
            timeout=120,
        )

        response_text = results[0].get("text", "") if results else ""
        if not response_text:
            raise ValueError("Empty response from self-hint model")

        return self._parse_hint(response_text)

    async def generate_hint_with_retry(
        self,
        game_code: str,
        max_retries: int = 3,
    ) -> str:
        """Generate a hint with exponential backoff retry on failure."""
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                return await self.generate_hint_async(game_code)
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(
                    f"[SELF-HINT] Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

        raise RuntimeError(
            f"Self-hint generation failed after {max_retries} attempts: {last_error}"
        )
