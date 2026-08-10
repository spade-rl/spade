
"""Environment validator for rejection sampling during game generation.

Validates generated game environments by sending the code to a model
(either the training model itself or an external LLM via OpenAI/OpenRouter)
to check for solvability and reward correctness. Invalid games are rejected
and regenerated.

Note: Basic structural checks (reset/step exist, no crashes) are handled
programmatically by validate_game(). This validator focuses on semantic
issues that code execution cannot catch.

Game-type-aware prompts (mirrors hint_generator.get_hint_system_prompt):
- "cognitive" (default): the original lenient playable/winnable check.
- "tool_use": multi-turn tool-use games — reject IMPOSSIBLE (unreachable /
  type-mismatched / underivable success criteria) while KEEPING hard-but-
  feasible games (difficulty is good curriculum; impossibility is fatal).
"""

import logging
import re
from typing import Any, Tuple

logger = logging.getLogger(__name__)

ENV_VALIDATION_SYSTEM_PROMPT = (
    "You review auto-generated Python game environments. "
    "Only reject games with clear, fatal bugs. When in doubt, accept."
)

ENV_VALIDATION_PROMPT_TEMPLATE = """Is this game environment playable? Only reject if there is a clear fatal bug.

```python
{game_code}
```

Check these two things:

1. **Solvable**: Can a player who understands the puzzle submit a correct answer? (Reject only if the solution is impossible to derive — e.g., answer depends on hidden randomness, or the puzzle description doesn't match the checking logic.)

2. **Winnable**: Does submitting the correct answer actually give a positive reward and end the game? (Reject only if there's an obvious bug — e.g., type mismatch in comparison that always fails, or success path is unreachable.)

Minor issues like imperfect error messages, unusual formatting, or edge cases are fine — do NOT reject for those.

Answer \\boxed{{yes}} unless there is a clear fatal bug, then \\boxed{{no}}."""


# ----- Tool-use game validation (game_type="tool_use") ----------------------
TOOL_USE_ENV_VALIDATION_SYSTEM_PROMPT = (
    "You rigorously audit auto-generated Python tool-use game environments for "
    "SOLVABILITY. Your one job: REJECT games that are IMPOSSIBLE to solve (no tool-call "
    "path can satisfy the success criteria), while KEEPING games that are merely HARD (a "
    "path exists but the agent must discover hidden values by calling tools). Difficulty "
    "is GOOD curriculum; impossibility is fatal. Trace every success criterion to the tool "
    "that produces it — if any required state is unreachable, reject."
)

TOOL_USE_ENV_VALIDATION_PROMPT_TEMPLATE = """Audit this multi-turn tool-use game for SOLVABILITY. The agent calls tools turn-by-turn and submits <answer>done</answer> after satisfying every step. The game is SOLVABLE iff there EXISTS a sequence of tool calls that makes every success/termination criterion True.

```python
{game_code}
```

Trace it, criterion by criterion:

1. **List the success criteria** — the per-step checks (e.g. the lambdas in self._message_criteria, the _advance_if_done / _check_answer / reward / termination logic). Note the EXACT state each one requires (e.g. `state['order_status'] == 'completed'`, `state['last_action'] == 'generated_invoice'`).

2. **For EACH required state value, find the tool that PRODUCES it.** Scan every tool method for an assignment setting that state to the required value (e.g. `self._state['order_status'] = 'completed'`).
   - If EVERY criterion's required state IS set by some tool → SOLVABLE (keep it, even if very hard).
   - If ANY criterion requires a state value that NO tool ever assigns (unreachable), OR two criteria are mutually contradictory → the success path is IMPOSSIBLE → REJECT.
   - NOTE on randomness (read carefully — this is the most common FALSE-reject): a random value is FINE and DISCOVERABLE as long as some tool RETURNS it to the agent. Generated games routinely create random IDs / codes / dates either at reset() OR inside a tool (e.g. `flight_id = f'FL{{random.randint(1000,9999)}}'` is generated inside tool_search_flights and ECHOED in its result, then the success criterion checks a state flag the agent set using that discovered id). The agent reads the value from the tool result and uses it on the next call → SOLVABLE, do NOT reject. Only reject for randomness if (a) a required value is NEVER surfaced in ANY tool result and cannot be queried — the agent has literally no way to ever observe it; or (b) a tool returns a DIFFERENT random value on every call for the SAME input, so the agent can never act on a stable value. Random-at-reset that is later returned by a query tool is always fine.

   GROUNDED example to REJECT: a criterion checks `state['order_status'] == 'completed'`, but no tool anywhere assigns `order_status = 'completed'` (tools only set it to 'viewing'/'shipped'). The agent can NEVER satisfy it → impossible → reject.

2b. **Also catch these IMPOSSIBLE / unwinnable criteria (common bugs that look fine at a glance):**
   - TYPE-MISMATCH: a criterion whose Python can never be True because it reads state with the wrong shape — e.g. `'laptop' in s['order_history']` when order_history is a list of DICTS (membership of a string in a list of dicts is ALWAYS False), or `s['x'] == 5` when the tool stores `s['x']` as a string '5'. Trace the actual type the tool writes vs the type the criterion compares. If they can never be equal → impossible → REJECT.
   - UNREACHABLE TOOL FIELD: a criterion's required state is "set" only by a tool that itself can never succeed — e.g. tool_process_refund looks up a refund by `r['refund_id']`, but the tool that creates refunds never stores a 'refund_id' key, so process_refund always errors and the criterion's state is never written → impossible → REJECT.
   - UNDERIVABLE VALUE: a criterion requires a SPECIFIC hidden value (an exact date / id / amount / threshold) that NEITHER the user instruction states NOR any tool result reveals — so the agent can only GUESS. E.g. instruction "schedule a visit next week" but criterion `s['last_updated'] >= '2024-01-22'`; or criterion needs order id '1024' that no instruction names and no tool returns. The agent cannot reliably reach it → treat as unwinnable → REJECT. (Contrast with the randomness note above: a value a tool ECHOES back IS derivable — keep it.)
   - PRE-SATISFIED STEP: a criterion that is already True on the state reset() builds (e.g. `s['inventory']['monitor'] >= 5` when reset stocks 8, or a verify-step criterion `s['line_status'] == 'active'` when it starts 'active'). That step auto-completes on the agent's FIRST tool call without doing its instruction. Trace each criterion against the reset() state. REJECT only if the MAJORITY of steps are pre-satisfied (the game degenerates — the agent wins without doing the task); a single pre-satisfied step in an otherwise sound game is a KEEP.

3. **Do NOT reject for difficulty.** KEEP these: hidden values the agent must discover by calling lookup/list tools (e.g. a specific id `T104` that a search tool returns), long strict multi-step sequences, distractor/out-of-scope tools, needing to read each user instruction carefully. As long as a tool-call path to every criterion exists AND every required value is stated-or-discoverable, it is GOOD hard curriculum — keep it.

4. Ignore cosmetic issues (error-message wording, formatting, harmless edge cases).

Reject ONLY when a required success state is genuinely unreachable / contradictory / type-impossible / set by a tool that can never succeed / dependent on an underivable hidden value. Otherwise accept (difficulty is GOOD).

Answer \\boxed{{no}} if any success criterion is impossible or unwinnable by the above tests; else \\boxed{{yes}}."""



def get_env_validation_prompts(game_type: str) -> Tuple[str, str]:
    """Select (system_prompt, prompt_template) for a game type, mirroring
    hint_generator.get_hint_system_prompt.

    "tool_use" -> the multi-turn tool-use solvability auditor;
    anything else (default, incl. "cognitive") -> the original lenient
    playable/winnable check, preserving prior behavior.
    """
    if game_type == "tool_use":
        return TOOL_USE_ENV_VALIDATION_SYSTEM_PROMPT, TOOL_USE_ENV_VALIDATION_PROMPT_TEMPLATE
    return ENV_VALIDATION_SYSTEM_PROMPT, ENV_VALIDATION_PROMPT_TEMPLATE


class EnvironmentValidator:
    """Validates generated game environments using an LLM.

    Supports two modes:
    - Training model ("self"): Uses the ModelAdapter passed to the orchestrator
    - External model: Uses an OpenAIModelAdapter wrapping OpenAI/OpenRouter APIs

    Both adapters share the same generate_async interface, so the validator
    is agnostic to which backend is used.
    """

    def __init__(
        self,
        model_adapter: Any,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        game_type: str = "cognitive",
    ):
        """Initialize the environment validator.

        Args:
            model_adapter: Any object with a generate_async(messages, ...) method
                that returns List[Dict] with a "text" key. Can be either:
                - ModelAdapter (training model, when env_validator_model="self")
                - OpenAIModelAdapter (external model via OpenAI/OpenRouter)
            temperature: Sampling temperature (low for deterministic)
            max_tokens: Max tokens for validation response
            game_type: "tool_use" selects the game-type-aware
                solvability auditor; any other value (default) keeps the
                original lenient playable/winnable prompt.
        """
        self.model_adapter = model_adapter
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Game-type-aware prompt selection (tool_use / generic).
        self._system_prompt, self._template = get_env_validation_prompts(game_type)

    async def validate_async(self, game_code: str) -> Tuple[bool, str]:
        """Validate a generated game environment.

        Args:
            game_code: Python source code of the generated game

        Returns:
            Tuple of (is_valid, reasoning)
        """
        prompt = self._template.format(game_code=game_code)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            results = await self.model_adapter.generate_async(
                messages=messages,
                input_ids=None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                role="env_validator",
            )

            reasoning = results[0].get("text", "") if results else ""
            if not reasoning:
                logger.warning("[ENV-VALIDATOR] Empty response, defaulting to valid")
                return True, "Empty response from validator"

            # Parse verdict from \boxed{}
            match = re.search(r"\\boxed\{(.+?)\}", reasoning)
            if match:
                verdict = match.group(1).strip().lower()
                is_valid = verdict in ("yes", "y", "true")
            else:
                # If no boxed answer, be conservative and accept
                logger.warning("[ENV-VALIDATOR] No boxed verdict found, defaulting to valid")
                is_valid = True

            return is_valid, reasoning

        except Exception as e:
            logger.error(f"[ENV-VALIDATOR] Validation failed: {e}")
            # On error, don't block training — accept the game
            return True, f"Validation error: {e}"
