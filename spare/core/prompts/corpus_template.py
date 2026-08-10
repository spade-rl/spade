"""Prompt template for corpus-grounded MULTI-TURN game generation.

Instructs the model to create an INTERACTIVE, multi-turn Python environment grounded
in a reference document's concepts. Each turn the agent submits an action wrapped as
\\boxed{<action>}; the pipeline's parse_action normalizes the actor output to that form,
and the env's step() extracts the command, updates state, and returns a NEW observation
plus partial reward — NOT single-shot \\boxed{answer} grading.

Replaces the prior single-shot-QA contract: that produced games whose step() graded one
\\boxed answer and terminated, which (a) are not interactive and (b) give zero within-game
reward variance (every play of an instance returns the same reward) so GRPO filters them
out. The interactive contract below was validated on the local 30B (dp=8): ~92% valid,
~68% of valid envs genuinely multi-turn (higher on math/code grounding).
"""

# Interactive multi-turn spec (interface contract + rules + robustness). \boxed wraps
# each turn's COMMAND (not a final answer); step() dispatches it and evolves state.
INTERACTIVE_SPEC = r'''
This MUST be a genuine multi-turn INTERACTIVE environment, NOT a one-shot question-and-answer quiz.

HOW THE AGENT ACTS (this matches the training pipeline — follow it exactly):
- Every turn the agent submits ONE action wrapped as \boxed{<action>}, e.g. \boxed{measure node 3},
  \boxed{move north}, \boxed{open valve A}. step() receives that string; extract the action with
  re.search(r'\\boxed\{(.+?)\}', action) and treat the extracted text as an interactive COMMAND.
- The extracted text is a COMMAND / move / query that CHANGES the environment — it is NOT a final
  answer to grade once. step() must UPDATE the environment's state from it and return a NEW
  observation reflecting the changed state.

WHAT MAKES IT MULTI-TURN (required):
- The goal CANNOT be reached in one action. It needs a SEQUENCE of actions — explore to uncover
  hidden information, then act on it; or manipulate state step-by-step toward a target.
- HIDDEN STATE: the agent does NOT see everything at reset(); it must act to reveal/probe state
  across turns (the observation grows/changes as it acts).
- PARTIAL reward for progress toward the goal; terminated=True only when the goal is reached.
- Each turn's observation must (a) reflect the updated state and (b) remind the agent to reply
  with its next action as \boxed{<action>}.

DO NOT (these collapse it back to single-shot QA):
- Do NOT extract \boxed{answer}, grade it once, and terminate. \boxed carries a per-turn COMMAND,
  not a final answer.
- Do NOT state the full problem at reset() and just check one submitted value.
- Do NOT repeat the SAME static task every turn with only 'wrong, try again' feedback.

GROUNDING: the interaction must require understanding a concept/technique from the document
(e.g. if it describes an algorithm, the agent EXECUTES it step-by-step interactively; if it
describes a system, the agent OPERATES it over turns). Never reference the document in the game
text (no 'according to the passage'); write it as a standalone environment.

INTERFACE CONTRACT (interactive; \boxed wraps each turn's COMMAND):
```python
import random, re
from typing import Tuple, Dict

class DescriptiveInteractiveEnv:
    def __init__(self, max_turns=12, **kwargs):
        self.reset()
    def reset(self, seed=None) -> Tuple[str, dict]:
        self.turn_count = 0
        # set up HIDDEN state + a goal needing several actions; randomize by seed.
        # return (observation: situation + AVAILABLE ACTIONS + 'reply with \boxed{<action>}',
        #         but NOT the full solution), {}
        ...
    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        self.turn_count += 1
        truncated = self.turn_count >= self.max_turns
        m = re.search(r'\\boxed\{(.+?)\}', action)
        cmd = (m.group(1) if m else action).strip()      # the per-turn COMMAND
        # 1) PARSE cmd into an operation; 2) UPDATE self state; 3) build a NEW observation
        #    reflecting the updated state; 4) reward = partial progress in [0,1],
        #    terminated=True only when the goal is reached.
        # every code path returns (str, float, bool, bool, dict)
        ...
    def solution(self) -> str:
        # a reference action-sequence (or goal description) that solves it
        ...
    def close(self): pass
```

REWARD SCALE: keep total reward in [0, 1]; partial progress < 1.0, full success = 1.0.

ROBUSTNESS (the most common failures — follow ALL of these or the env won't load):
- BRACE ESCAPING: in f-strings and .format(), {word} evaluates word; to write a LITERAL brace you
  must DOUBLE it. To print \boxed{<action>} inside an f-string, write \boxed{{<action>}}.
- Initialize ALL instance variables in __init__ BEFORE calling reset() (no AttributeError mid-episode).
- Imports: use ONLY the Python standard library plus `math` and `random`. Import everything you use;
  never call an unimported name (no bare `integrate`, no `np.`, no `x.cos()` — use `math.cos(x)`).
- Every step() code path returns the 5-tuple (str, float, bool, bool, dict); info is always a dict {}.
- Never crash on an unrecognized/garbage command — return a helpful observation + reward 0.0 instead.
- Never divide by a value that could be zero; never store data in self.solution (use self._solution).

SELF-VERIFICATION before finalizing:
- Trace TWO different action sequences from reset(seed=0): does the observation CHANGE based on the
  actions? (If it never changes, it is NOT interactive — fix it.)
- Does \boxed carry a per-turn COMMAND that evolves state, rather than a final answer that ends it?
- Is partial reward given for progress, kept in [0,1]?
'''


def generate_corpus_grounded_prompt(
    skill_name: str,
    skill_info: dict,
    difficulty: str,
    document_text: str,
) -> str:
    """Build a prompt for corpus-grounded MULTI-TURN interactive game generation.

    Args:
        skill_name: Cognitive skill to target (e.g., "Mathematical Reasoning")
        skill_info: Skill metadata dict from COGNITIVE_SKILLS (needs 'description')
        difficulty: "easy", "medium", or "hard"
        document_text: Full (or truncated) document text from the corpus

    Returns:
        Complete prompt string for interactive multi-turn game generation.
    """
    steps = 3 if difficulty != "easy" else 2
    header = (
        "You are given a reference document. Create an INTERACTIVE, MULTI-TURN Python game\n"
        "environment grounded in a concept/technique from this document.\n\n"
        "<REFERENCE_DOCUMENT>\n" + document_text + "\n</REFERENCE_DOCUMENT>\n"
    )
    footer = (
        f"\nTarget skill: {skill_name} ({skill_info.get('description', skill_name)}). "
        f"DIFFICULTY: {difficulty} — the goal should need {steps}+ interaction steps.\n\n"
        "Generate the complete Python code in a ```python block."
    )
    return header + INTERACTIVE_SPEC + footer
