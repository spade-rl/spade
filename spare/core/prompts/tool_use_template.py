"""Prompt template for tool-use environment generation (subclass-based).

Instructs the LLM to generate a subclass of ToolUseBaseEnv. The base class
handles <tool_call> JSON parsing, tool dispatch, <answer> checking, and
observation formatting. The LLM only generates reset(), solution(), and
tool_xxx() methods.
"""


# Skill definitions for tool-use environments.
TOOL_USE_SKILLS = {
    "API Orchestration": {
        "category": "Stateful Operations",
        "description": "Chain multiple tool calls that mutate state sequentially",
        "complexity": "medium",
        "example_tasks": [
            "Navigate to a directory, create a subdirectory, move files into it",
            "Check account balance, transfer funds, verify new balance",
            "Search inventory, reserve items, confirm reservation, generate invoice",
            "List available flights, book one, add passenger details, confirm booking",
        ],
    },
    "Data Retrieval": {
        "category": "Query Then Act",
        "description": "First discover information via query tools, then act on it precisely",
        "complexity": "low",
        "example_tasks": [
            "List directory contents, find target file, read its content, extract specific data",
            "Search for user by name, get their order history, find the most recent order",
            "Check current settings, identify which ones need change, update them",
            "Query database for records matching criteria, compute aggregate, report result",
        ],
    },
    "State Modification": {
        "category": "Mutation Chains",
        "description": "Execute a sequence of state-changing operations where each depends on prior state",
        "complexity": "medium",
        "example_tasks": [
            "Create user account, set permissions, add to group, verify membership",
            "Navigate file tree, create directory structure, move and rename files",
            "Initialize vehicle systems (engine, doors, fuel), then execute driving commands",
            "Set up trading portfolio: check prices, place orders, verify positions",
        ],
    },
    "Error Recovery": {
        "category": "Robustness",
        "description": "Handle tool failures, verify state before acting, retry with corrections",
        "complexity": "high",
        "example_tasks": [
            "Try to move a file, handle 'not found' error, search for correct path, retry",
            "Attempt operation, detect permission error, elevate access, retry",
            "Check if directory exists before creating, handle 'already exists' gracefully",
            "Validate inputs by querying current state before calling destructive operation",
        ],
    },
    "Tool Selection": {
        "category": "Decision Making",
        "description": "Choose correct tool from alternatives based on current state",
        "complexity": "low",
        "example_tasks": [
            "Check current location first, then decide whether to navigate or act locally",
            "Query item type, then pick appropriate processing tool for that type",
            "Read file format, then choose correct parser/converter tool",
            "Check resource status, decide between create vs update operation",
        ],
    },
    "Multi-Step Workflows": {
        "category": "Sequential Planning",
        "description": "Execute ordered sequences where later steps depend on results of earlier steps",
        "complexity": "high",
        "example_tasks": [
            "File management: list → navigate → create dir → move files → verify → report",
            "Order processing: search catalog → check stock → reserve → pay → confirm → notify",
            "System setup: check status → configure → enable → test → verify → log result",
            "Data pipeline: connect → query → filter → transform → write → validate output",
        ],
    },
}

# Difficulty levels for tool-use tasks.
TOOL_USE_DIFFICULTY = {
    "easy": (
        "DIFFICULTY: easy\n"
        "- 1-2 tool calls needed. Single tool or simple two-step chain.\n"
        "- 3-4 total tools available (1 unused).\n"
        "- Answer derivable from one or two tool call results.\n"
    ),
    "medium": (
        "DIFFICULTY: medium\n"
        "- 2-4 tool calls needed. Must chain results across calls.\n"
        "- 4-6 total tools available (1-2 unused).\n"
        "- Answer requires combining information from multiple tool results.\n"
    ),
    "hard": (
        "DIFFICULTY: hard\n"
        "- 4+ tool calls with dependencies. Correct ordering matters.\n"
        "- 5-8 total tools available (2-3 unused).\n"
        "- Answer requires multi-step reasoning across chained tool outputs.\n"
        "- Include scenarios where one tool may return partial or paginated data.\n"
    ),
}


def generate_tool_use_prompt(
    skill_name: str,
    skill_info: dict,
    difficulty: str = "medium",
    document_text: str = "",
) -> str:
    """Generate a prompt for tool-use environment generation.

    The LLM generates a subclass of ToolUseBaseEnv with reset(), solution(),
    and tool_xxx() methods. The base class handles parsing and dispatch.

    Args:
        skill_name: Tool-use skill to target (e.g., "API Orchestration")
        skill_info: Skill metadata dict from TOOL_USE_SKILLS
        difficulty: "easy", "medium", or "hard"
        document_text: Optional reference document for corpus-grounded generation

    Returns:
        Complete prompt string for tool-use environment generation
    """
    import random as _rng

    difficulty_text = TOOL_USE_DIFFICULTY.get(difficulty, TOOL_USE_DIFFICULTY["hard"])

    examples = skill_info.get("example_tasks", [])
    if len(examples) > 4:
        examples = _rng.sample(examples, 4)
    examples_str = (
        "\n  ".join(f"- {e}" for e in examples)
        if examples
        else "- tool-calling tasks"
    )

    corpus_section = ""
    if document_text:
        corpus_section = (
            "You are given a reference document. Create a tool-use environment\n"
            "in the DOMAIN described by this document.\n"
            "\n"
            "<REFERENCE_DOCUMENT>\n"
            f"{document_text}\n"
            "</REFERENCE_DOCUMENT>\n"
            "\n"
            "GROUNDING REQUIREMENTS:\n"
            "- Tools and data must relate to the document's domain.\n"
            "- The game must be SELF-CONTAINED: the agent will NOT see the document.\n"
            "- NEVER reference the source document in game text.\n"
            "\n"
        )

    return (
        f"{corpus_section}"
        "Create a tool-use environment by subclassing ToolUseBaseEnv.\n"
        f"The environment tests: {skill_name} ({skill_info['description']}).\n"
        "\n"
        "TASK CONCEPT IDEAS (pick one or invent your own):\n"
        f"  {examples_str}\n"
        "\n"
        "WHAT YOU IMPLEMENT (the base class handles everything else):\n"
        "  - reset(seed) -- generate data, define self._tools, set self._expected_answer\n"
        "  - tool_xxx(**kwargs) -> str -- one method per simulated tool (prefix: tool_)\n"
        "  - solution() -> str -- correct tool call sequence + final answer\n"
        "  - get_tools() is auto-generated from self._tools by the base class (no override needed)\n"
        "\n"
        "THE BASE CLASS ALREADY HANDLES:\n"
        "  - <tool_call> JSON parsing and dispatch to your tool_xxx() methods\n"
        "  - <answer> submission and checking via _check_answer()\n"
        "  - Format error handling (returns -0.1 reward on bad format)\n"
        "  - Available tools display in every observation\n"
        "  - Turn counting, truncation, solution reveal on timeout\n"
        "  Do NOT reimplement step(). Do NOT parse <tool_call> yourself.\n"
        "\n"
        "AGENT ACTION FORMAT (handled by base class, for your reference):\n"
        '  <tool_call>{"name": "tool_name", "arguments": {"key": "value"}}</tool_call>\n'
        "  <answer>final result</answer>\n"
        "\n"
        "RULES:\n"
        "- Your class MUST inherit from ToolUseBaseEnv.\n"
        "- Do NOT import ToolUseBaseEnv. It is already available in the namespace.\n"
        "  Just write: class MyEnv(ToolUseBaseEnv):\n"
        "- Only import standard library modules (random, json, re, math, etc.).\n"
        "  Do NOT import any custom modules, utils, helpers, or external packages.\n"
        "- Tools are SIMULATED as methods. No real API calls, no network, no subprocess.\n"
        "- One task per episode. Task and data are fixed at reset().\n"
        "- Give your class a descriptive, unique name ending with Env.\n"
        "- The task must be solvable from the tool results alone.\n"
        "- The observation from reset() must tell the agent what to find and the answer format.\n"
        "\n"
        "INFORMATION HIDING:\n"
        "- The observation describes the TASK but NOT the answer.\n"
        "- The agent must call tools to discover information.\n"
        "- NO SINGLE TOOL should return enough to solve the task alone.\n"
        "  The agent MUST call 2-3 different tools and combine outputs.\n"
        "\n"
        "STATEFUL ENVIRONMENT (CRITICAL):\n"
        "- The environment MUST have mutable state (e.g., current_dir, balance, inventory,\n"
        "  active_records, status flags) stored in self._state.\n"
        "- Provide BOTH query tools (read state) AND mutation tools (modify state).\n"
        "- Mutation tools MUST modify self._state and return success/error messages.\n"
        "- Examples of stateful operations to model:\n"
        "  * filesystem: cd, mkdir, mv, rm, touch, echo (changes current_dir + files)\n"
        "  * trading: buy, sell, deposit, withdraw (changes balance + positions)\n"
        "  * vehicle: startEngine, lockDoors, fillFuelTank (changes status flags)\n"
        "  * messaging: send, delete, archive (changes message list)\n"
        "  * booking: book, cancel, modify (changes reservations)\n"
        "- The task MUST require AT LEAST 3 sequential dependent mutations:\n"
        "  call_A → call_B (uses result of A) → call_C (uses state set by B)\n"
        "- Mutation tools should ERROR or fail gracefully when called with wrong\n"
        "  preconditions (e.g., mv before cd into right dir → 'file not found').\n"
        "  This forces the agent to query state BEFORE mutating.\n"
        "\n"
        "PRECISE EXECUTION (CRITICAL):\n"
        "- Tool arguments must match EXACTLY. No extra content beyond what's requested.\n"
        "- If task says 'write \"hello\" to file.txt', the agent MUST write exactly 'hello',\n"
        "  not 'hello world' or 'Greetings: hello'. Verify content match in solution().\n"
        "- The answer (or final state) is checked for EXACT match, not approximate.\n"
        "\n"
        f"{difficulty_text}"
        "\n"
        "TOOL DESIGN:\n"
        "- Define 3-8 tools as tool_xxx() methods returning strings.\n"
        "- Each tool takes simple args (str, int, float, bool).\n"
        "- Tools must be DETERMINISTIC (no random inside tool methods).\n"
        "- Include 1-2 tools that are NOT needed for the solution.\n"
        "  Do NOT label them as unused in their descriptions.\n"
        "- CRITICAL: tool method parameter names MUST EXACTLY MATCH the names\n"
        "  declared in self._tools[name]['parameters']['properties']. If the\n"
        "  schema says {'account': {...}}, the method MUST be\n"
        "  `def tool_check(self, account=...)`. NOT `account_type`, NOT a\n"
        "  renamed alias. Mismatched names cause every tool call to fail.\n"
        "- Register tools in self._tools dict with description and parameters.\n"
        "  Use OpenAI JSON Schema format for parameters:\n"
        '    self._tools = {"search": {"description": "...", "parameters": {\n'
        '      "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]\n'
        "    }}}\n"
        "\n"
        "SIMULATED DATA:\n"
        "- Generate realistic mock data in reset() (10-50 records, not 2-3).\n"
        "- Use random to vary data across seeds.\n"
        "- The answer must be COMPUTED from data, never hardcoded.\n"
        "\n"
        "ANSWER FORMAT:\n"
        "- Prefer simple answers: a number, an ID, a comma-separated list.\n"
        "- _check_answer() uses numeric tolerance (0.01) and case-insensitive match.\n"
        "- Override _check_answer() if you need custom comparison logic.\n"
        "\n"
        "ROBUSTNESS:\n"
        "- Initialize ALL instance variables in __init__ before calling reset().\n"
        "  Actually, ToolUseBaseEnv.__init__ does this. Just call super().__init__().\n"
        "- Use self._expected_answer (not self.solution, which is a method).\n"
        "- solution() is called on timeout. It MUST NOT crash.\n"
        "- All randomness in reset() only, never in tool methods.\n"
        "- When generating random-length data, use modular indexing to prevent IndexError.\n"
        "\n"
        "TURN BUDGET:\n"
        "- Task must be solvable within (max_turns - 2) tool calls.\n"
        "- Do NOT require iterating over every record individually.\n"
        "\n"
        "SELF-VERIFICATION:\n"
        "- Trace reset(seed=42): what data is generated?\n"
        "- What tools must be called to solve the task?\n"
        "- Does solution() return the correct sequence + answer?\n"
        "\n"
        "EXAMPLE STRUCTURE (stateful filesystem-like task):\n"
        "```python\n"
        "import random\n"
        "import json\n"
        "from typing import Tuple\n"
        "# Do NOT import ToolUseBaseEnv -- it is pre-loaded in the namespace\n"
        "\n"
        "class FileOrganizerEnv(ToolUseBaseEnv):\n"
        "    def reset(self, seed=None) -> Tuple[str, dict]:\n"
        "        self.turn_count = 0\n"
        "        self._call_history = []\n"
        "        if seed is not None:\n"
        "            random.seed(seed)\n"
        "        # Stateful environment: directory tree + current location\n"
        "        self._state = {\n"
        "            'cwd': '/home',\n"
        "            'tree': {\n"
        "                '/home': ['docs', 'temp.txt'],\n"
        "                '/home/docs': ['report.txt', 'notes.txt'],\n"
        "            },\n"
        "        }\n"
        "        self._tools = {\n"
        "            'pwd': {'description': 'Get current directory', 'parameters': {'type':'object','properties':{}}},\n"
        "            'ls': {'description': 'List current directory', 'parameters': {'type':'object','properties':{}}},\n"
        "            'cd': {'description': 'Change directory', 'parameters': {'type':'object','properties':{'path':{'type':'string'}},'required':['path']}},\n"
        "            'mkdir': {'description': 'Create directory in current location', 'parameters': {'type':'object','properties':{'name':{'type':'string'}},'required':['name']}},\n"
        "            'mv': {'description': 'Move file from current dir to subdir', 'parameters': {'type':'object','properties':{'src':{'type':'string'},'dst':{'type':'string'}},'required':['src','dst']}},\n"
        "        }\n"
        "        # Task: requires sequential mutations (cd → mkdir → mv)\n"
        "        self._expected_answer = 'archive/report.txt'\n"
        "        return ('TASK: Move report.txt from /home/docs into a new subdirectory '\n"
        "                'called \"archive\". Submit final path with <answer>PATH</answer>.', {})\n"
        "\n"
        "    def tool_pwd(self) -> str:\n"
        "        return self._state['cwd']\n"
        "\n"
        "    def tool_ls(self) -> str:\n"
        "        return json.dumps(self._state['tree'].get(self._state['cwd'], []))\n"
        "\n"
        "    def tool_cd(self, path='') -> str:\n"
        "        target = path if path.startswith('/') else f\"{self._state['cwd']}/{path}\"\n"
        "        if target not in self._state['tree']:\n"
        "            return f'Error: directory {target} not found'  # forces query first\n"
        "        self._state['cwd'] = target\n"
        "        return f'Changed to {target}'\n"
        "\n"
        "    def tool_mkdir(self, name='') -> str:\n"
        "        new_path = f\"{self._state['cwd']}/{name}\"\n"
        "        if new_path in self._state['tree']:\n"
        "            return f'Error: {new_path} already exists'\n"
        "        self._state['tree'][new_path] = []\n"
        "        self._state['tree'][self._state['cwd']].append(name)\n"
        "        return f'Created {new_path}'\n"
        "\n"
        "    def tool_mv(self, src='', dst='') -> str:\n"
        "        cwd_files = self._state['tree'].get(self._state['cwd'], [])\n"
        "        if src not in cwd_files:\n"
        "            return f'Error: {src} not in current dir'  # forces cd first\n"
        "        dst_path = f\"{self._state['cwd']}/{dst}\"\n"
        "        if dst_path not in self._state['tree']:\n"
        "            return f'Error: destination {dst_path} not found'  # forces mkdir first\n"
        "        cwd_files.remove(src)\n"
        "        self._state['tree'][dst_path].append(src)\n"
        "        return f'Moved {src} to {dst}'\n"
        "\n"
        "    def solution(self) -> str:\n"
        "        # 3 dependent mutations: cd → mkdir → mv\n"
        "        return ('1. cd(path=\"docs\") -> Changed to /home/docs '\n"
        "                '2. mkdir(name=\"archive\") -> Created /home/docs/archive '\n"
        "                '3. mv(src=\"report.txt\", dst=\"archive\") -> Moved '\n"
        "                'ANSWER: archive/report.txt')\n"
        "```\n"
        "\n"
        "Notice: each mutation depends on prior state (cd before mv, mkdir before mv).\n"
        "Tools return errors when preconditions aren't met, forcing query-then-act.\n"
        "\n"
        "Generate the complete Python code in a ```python block.\n"
    )


# ── Multi-turn variant: BFCL-multi-turn-structured task synthesis ──────────
#
# Generates tasks that mirror the structure of BFCL multi_turn_base: a
# sequence of 3-5 atomic user instructions, each with its own success
# criterion (state match), revealed PROGRESSIVELY by the env as previous
# ones are completed (rather than dumped up front).
#
# Hypothesis: matching training task structure to eval task structure
# prevents the policy from learning single-turn solving patterns that
# fail on multi_turn_base's multi-message format.
#
# Use via os.environ["SPARE_MULTITURN_ENV_GEN"] = "1" — orchestrators
# dispatch to this function instead of the single-task one.

_V_MULTITURN_PROMPT = """\
Create a MULTI-TURN tool-use environment by subclassing ToolUseBaseEnv.
The environment tests: __SKILL_NAME__ (__SKILL_DESCRIPTION__).

THIS TASK MUST MIRROR THE STRUCTURE OF A REAL MULTI-TURN INTERACTION.
A real user does NOT describe the entire workflow up front. Instead, they
issue ONE atomic instruction, wait for it to be done, then issue the next.
Your env must reproduce this turn-by-turn structure.

──────────────────────────────────────────────────────────────────────────
WHAT YOU IMPLEMENT (the base class handles parsing/dispatch):

  - reset(seed) -- generate state, define self._tools, define
        self._user_messages (list of 3-5 atomic instruction strings),
        self._message_criteria (list of callables: state -> bool),
        self._current_msg = 0, self._expected_answer = "done".
        Return (self._user_messages[0], {})

  - tool_xxx(**kwargs) -> str -- one method per tool. After mutating
        state, EACH tool method must call self._advance_if_done() and
        APPEND its output to the tool result, like:
          base_result = "Moved file.pdf to /temp"
          progress = self._advance_if_done()  # appends next instruction or completion marker
          return base_result + progress

  - solution() -> str -- describe the full multi-turn solution

  - _advance_if_done(self) -> str -- helper you write yourself:
      ```
      def _advance_if_done(self):
          if self._current_msg >= len(self._user_messages):
              return ""
          criterion = self._message_criteria[self._current_msg]
          if criterion(self._state):
              self._current_msg += 1
              if self._current_msg >= len(self._user_messages):
                  return "\\n\\n[ALL STEPS COMPLETE] Submit <answer>done</answer>."
              next_msg = self._user_messages[self._current_msg]
              return f"\\n\\n[STEP {self._current_msg} COMPLETE — NEW INSTRUCTION] {next_msg}"
          return ""
      ```

  - Override _check_answer(answer) so it returns True only if
        answer == "done" AND self._current_msg == len(self._user_messages).

──────────────────────────────────────────────────────────────────────────
USER MESSAGE STRUCTURE (THE CORE OF MULTI-TURN):

self._user_messages must be 3-5 ATOMIC instructions, e.g.:
  [
    "Navigate to the document folder.",
    "Move final_report.pdf to a new 'temp' subdirectory.",
    "Now create an 'archive' folder and move all .txt files there.",
    "Finally, list the contents of the archive folder.",
  ]

self._message_criteria are callables that read self._state and return True
when the criterion is met:
  [
    lambda s: s["cwd"] == "/document",
    lambda s: "final_report.pdf" in s["tree"].get("/document/temp", []),
    lambda s: all(f in s["tree"].get("/document/archive", []) for f in s["txt_files"]),
    lambda s: s["last_ls_path"] == "/document/archive",
  ]

──────────────────────────────────────────────────────────────────────────
DOMAIN RULES (mirror BFCL multi_turn_base):

- Each user message is a SHORT, SPECIFIC, ATOMIC instruction (1-2 sentences).
- Do NOT describe the whole workflow in advance. The user reveals
  instructions ONE AT A TIME as previous ones are completed.
- Do NOT mention tool names, file structures, or implementation details
  in user messages. Use natural-language goals only.
- Each instruction's success criterion must be checkable from self._state.
- The actor must complete the CURRENT message before progressing. Trying
  to skip ahead won't reveal future messages.
- Allow 4-8 tool calls per message.

──────────────────────────────────────────────────────────────────────────
CRITICAL — SOLVABILITY & CRITERION CORRECTNESS (the single biggest source of BROKEN games):

A game is BROKEN if a success criterion cannot be satisfied by following its
instruction, or is already satisfied before the agent acts. Broken games waste
training and produce deadlocks. Obey ALL of these:

1. FALSE AT RESET / NOT FREE. Every criterion MUST require the agent to actually
   perform its instruction's action. For a WRITE instruction, the criterion must
   check the CHANGE it makes and be False on the state reset() returns — a
   criterion already True at reset (e.g. `s['inventory']['monitor'] >= 5` when
   monitor starts at 8, or `any('phone' in i for i in s['electronics'])` when a
   phone is already stocked) lets the agent skip the step for free. For a READ /
   VERIFY instruction ("check the line is active", "confirm the balance"), do NOT
   use a criterion that is already True at reset (e.g. `s['line_status']=='active'`
   when it starts 'active') — that advances on ANY tool call without the agent
   doing the read. Instead have the read tool record that it ran (e.g. set
   `s['last_checked']='line_status'`) and make the criterion check THAT flag, so
   the step requires actually calling the read tool.

2. TYPE-CORRECT. The criterion must read self._state with the SAME shape the
   tools write it. If order_history is a list of DICTS, check
   `any(o['item']=='laptop' for o in s['order_history'])`, NOT
   `'laptop' in s['order_history']` — string-in-list-of-dicts is ALWAYS False, so
   the step can NEVER complete and the game deadlocks. Type mismatches are the
   most common unsolvable-game bug.

3. DERIVABLE BY THE AGENT. Every value a criterion requires must be obtainable by
   the agent from EITHER (a) the words of its instruction, OR (b) a tool result it
   can read. NEVER hide an exact date / id / amount / threshold the instruction
   does not state and no tool reveals. BAD: instruction "schedule a visit next
   week" but criterion `s['last_updated'] >= '2024-01-22'` — the agent cannot know
   the threshold, must guess, and usually fails. FIX: state it in the instruction
   ("schedule for 2024-01-22 or later"), OR accept ANY valid action
   (`s['last_updated'] != '2024-01-15'`, i.e. any new date), OR have a tool reveal
   the value. Same for ids: if a criterion needs order '1024', the instruction
   must name it or a tool must return it.

4. EVERY REQUIRED TOOL CAN SUCCEED. For each instruction, the tool that satisfies
   its criterion must have a reachable success path. Trace the field the tool
   looks up — it must be a key some tool (or reset) actually SETS. BAD:
   tool_process_refund finds a refund by `r['refund_id']` but tool_create_refund
   never stores a 'refund_id' key → process_refund can NEVER succeed. If a tool
   reads an id/field, an earlier tool or reset MUST write that exact key.

5. INSTRUCTION MATCHES CRITERION. Each instruction must describe exactly what its
   criterion checks — no more, no less. Do NOT write "after the refund is
   processed, list inventory" if the criterion only checks that inventory was
   listed (and process_refund is broken/unneeded): the agent chases the
   irrelevant step and stalls.

6. SELF-TRACE BEFORE YOU FINISH (REQUIRED). Mentally run reset(seed=0), then
   execute your own solution() sequence step by step. After EACH solution step the
   matching criterion must flip False→True, IN ORDER, and no later criterion may be
   True yet; at the end self._current_msg must equal len(self._user_messages) and
   _check_answer("done") must be True. If any criterion is True at reset, never
   becomes True, or throws (KeyError/IndexError), the game is BROKEN — fix it
   before returning.

──────────────────────────────────────────────────────────────────────────
DIFFICULTY: __DIFFICULTY__

__DOMAIN_BLOCK__
__TASK_TYPE_BLOCK__
SKILL FOCUS: __SKILL_CATEGORY__
EXAMPLE TASKS (pick one or invent your own):
  __EXAMPLES__

__CORPUS_SECTION__

──────────────────────────────────────────────────────────────────────────
RULES:
- Class MUST inherit from ToolUseBaseEnv.
- Do NOT import ToolUseBaseEnv. Just write: class MyEnv(ToolUseBaseEnv):
- Only standard library imports (random, json, re, math). No custom packages.
- Tools simulated as methods. No real APIs, network, or subprocess.
- Tools must be DETERMINISTIC. All randomness in reset() only.
- 3-8 tools as tool_xxx() methods returning strings.
- Give at least one tool a typed/constrained parameter using 'enum' and 'required' in its
  schema (e.g. status: {'type':'string','enum':['pending','shipped','cancelled']}).
- Include 1-2 DISTRACTOR tools: plausible but wrong for the task, so the agent must select
  the correct one from alternatives.
- Each tool ENDS with `return base_result + self._advance_if_done()`.
- Mutation tools must error gracefully on bad preconditions
  (e.g., mv without target dir → "Error: target dir not found").
- CRITICAL: tool method parameter names MUST EXACTLY MATCH the names declared
  in self._tools[name]['parameters']['properties']. If schema says
  {'account': {...}}, the method MUST be `def tool_check(self, account=...)`.
  NOT `account_type`, NOT `acct`, NOT a renamed alias. Mismatched names
  cause every tool call to fail with "got an unexpected keyword argument".

──────────────────────────────────────────────────────────────────────────
EXAMPLE STRUCTURE (multi-turn filesystem task):

```python
import random
import json
from typing import Tuple

class FileWorkflowMTEnv(ToolUseBaseEnv):
    def reset(self, seed=None) -> Tuple[str, dict]:
        self.turn_count = 0
        self._call_history = []
        if seed is not None:
            random.seed(seed)
        self._state = {
            'cwd': '/home',
            'tree': {
                '/home': ['document', 'temp.txt'],
                '/home/document': ['report.txt', 'notes.txt', 'log.txt'],
            },
            'last_ls_path': None,
        }
        self._tools = {
            'pwd': {'description': 'Show current dir', 'parameters': {'type':'object','properties':{}}},
            'ls': {'description': 'List a directory', 'parameters': {'type':'object','properties':{'path':{'type':'string'}}, 'required':['path']}},
            'cd': {'description': 'Change dir', 'parameters': {'type':'object','properties':{'path':{'type':'string'}}, 'required':['path']}},
            'mkdir': {'description': 'Create a subdirectory in current dir', 'parameters': {'type':'object','properties':{'name':{'type':'string'}}, 'required':['name']}},
            'mv': {'description': 'Move file from current dir to a destination dir', 'parameters': {'type':'object','properties':{'src':{'type':'string'},'dst':{'type':'string'}}, 'required':['src','dst']}},
        }
        self._user_messages = [
            "Navigate to the document folder.",
            "Create a 'temp' subdirectory and move report.txt into it.",
            "Now list what's in the temp folder.",
        ]
        self._message_criteria = [
            lambda s: s['cwd'] == '/home/document',
            lambda s: 'report.txt' in s['tree'].get('/home/document/temp', []),
            lambda s: s['last_ls_path'] == '/home/document/temp',
        ]
        self._current_msg = 0
        self._expected_answer = "done"
        return (self._user_messages[0], {})

    def _advance_if_done(self) -> str:
        if self._current_msg >= len(self._user_messages):
            return ""
        if self._message_criteria[self._current_msg](self._state):
            self._current_msg += 1
            if self._current_msg >= len(self._user_messages):
                return "\\n\\n[ALL STEPS COMPLETE] Submit <answer>done</answer>."
            next_msg = self._user_messages[self._current_msg]
            return f"\\n\\n[STEP {self._current_msg} COMPLETE — NEW INSTRUCTION] {next_msg}"
        return ""

    def _check_answer(self, answer: str) -> bool:
        return answer.strip().lower() == 'done' and self._current_msg >= len(self._user_messages)

    def tool_pwd(self) -> str:
        return self._state['cwd'] + self._advance_if_done()

    def tool_ls(self, path='') -> str:
        if path not in self._state['tree']:
            return f"Error: {path} not found" + self._advance_if_done()
        self._state['last_ls_path'] = path
        out = json.dumps(self._state['tree'][path])
        return out + self._advance_if_done()

    def tool_cd(self, path='') -> str:
        if path not in self._state['tree']:
            return f"Error: {path} not found" + self._advance_if_done()
        self._state['cwd'] = path
        return f"Changed to {path}" + self._advance_if_done()

    def tool_mkdir(self, name='') -> str:
        new_path = f"{self._state['cwd']}/{name}"
        if new_path in self._state['tree']:
            return f"Error: {new_path} exists" + self._advance_if_done()
        self._state['tree'][new_path] = []
        self._state['tree'][self._state['cwd']].append(name)
        return f"Created {new_path}" + self._advance_if_done()

    def tool_mv(self, src='', dst='') -> str:
        cwd_files = self._state['tree'].get(self._state['cwd'], [])
        if src not in cwd_files:
            return f"Error: {src} not in current dir" + self._advance_if_done()
        dst_path = f"{self._state['cwd']}/{dst}"
        if dst_path not in self._state['tree']:
            return f"Error: target {dst} not found" + self._advance_if_done()
        cwd_files.remove(src)
        self._state['tree'][dst_path].append(src)
        return f"Moved {src} to {dst}" + self._advance_if_done()

    def solution(self) -> str:
        return ("1. cd(path='/home/document') 2. mkdir(name='temp') "
                "3. mv(src='report.txt', dst='temp') 4. ls(path='/home/document/temp') "
                "5. <answer>done</answer>")
```

Notice:
- self._user_messages defines the multi-turn STRUCTURE (BFCL-style)
- Each tool_xxx() ends with self._advance_if_done() so when criterion is met,
  the actor sees "[STEP N COMPLETE — NEW INSTRUCTION] ..." inline in the result
- _check_answer requires "done" AND all messages completed
- Reset returns ONLY the first instruction; the rest are revealed progressively

Generate the complete Python code in a ```python block.
"""


# --- Realistic agentic domains + benchmark-style task-type variants. One domain and one
# task-type are sampled PER generation call (call-level cluster rotation, not within-call
# "be diverse") so the generated SET spans the OOD tool-use benchmark distribution:
# tau2-bench / BFCL / ACEBench / VitaBench / MCP-Mark / MCP-Universe. ---
_TOOL_USE_DOMAINS = [
    "retail order management (orders, items, refunds, shipping)",
    "flight and hotel booking (search, reserve, pay, cancel)",
    "telecom account management (plans, lines, billing, data add-ons)",
    "personal banking and payments (accounts, transfers, cards, invoices)",
    "a calendar and reminders assistant (events, invites, reminders)",
    "a file system (files, folders, move/copy/search)",
    "an e-commerce shopping cart (catalog, cart, checkout, coupons)",
    "customer-support tickets / CRM (tickets, status, assignment, notes)",
    "smart-home device control (lights, thermostat, locks, scenes)",
    "a project / task tracker (tasks, assignees, status, due dates)",
]

_DOMAIN_BLOCK = (
    "REAL-WORLD DOMAIN (build the tools, state, and user goal for THIS domain; do NOT\n"
    "use the reference snippet's academic subject as the domain):\n"
    "  __DOMAIN__\n"
    "The task must be a realistic user accomplishing a goal here (search, book, order,\n"
    "update, cancel, transfer, schedule), like tau2-bench / BFCL / ACEBench / VitaBench\n"
    "tasks - NOT an exam or puzzle about any document.\n"
)

_TASK_TYPE_BLOCKS = {
    "execute": (
        "TASK VARIANT: execute (happy-path). The user issues atomic instructions; the\n"
        "agent calls the right tools in order. Each criterion checks the resulting state.\n"
        "(BFCL multi_turn_base / tau2-bench)\n"
    ),
    "read_then_write": (
        "TASK VARIANT: read_then_write. At least ONE instruction must force the agent to\n"
        "FIRST query state with a read tool, branch on a predicate over the result (e.g.\n"
        "'for every order over $100', 'all lines past their data cap'), THEN make the\n"
        "dependent write. Its criterion must check the final state reflects the\n"
        "predicate-conditioned writes - not merely that one tool was called.\n"
        "(MCP-Mark / MCP-Universe long-horizon CRUD)\n"
    ),
    "miss_param": (
        "TASK VARIANT: missing_value. At least ONE instruction references a value the user\n"
        "does NOT state (e.g. 'cancel my most recent order' with no order_id; 'pay the\n"
        "overdue invoice' with no invoice id). The agent must DISCOVER it via a read tool\n"
        "before the write. The write tool MUST return an error if given a wrong/guessed\n"
        "value, so a fabricated argument cannot satisfy the criterion.\n"
        "(BFCL multi_turn_miss_param / ACEBench incomplete)\n"
    ),
    "irrelevant": (
        "TASK VARIANT: with_irrelevant. Include EXACTLY ONE instruction that NONE of the\n"
        "domain tools can satisfy (e.g. in a retail env, 'what will the weather be\n"
        "tomorrow?'). Add a tool named 'decline_request' (one required string param\n"
        "'reason') whose method sets a state flag and then advances; that instruction's\n"
        "criterion is met ONLY by calling decline_request. Provide NO other tool that\n"
        "could plausibly fulfill the irrelevant request.\n"
        "(BFCL irrelevance / ACEBench irrelevant)\n"
    ),
}

# Weighted rotation: execute / read_then_write are the common case; the abstention and
# missing-value variants are rarer (as in the benchmarks), one per call.
_TASK_TYPE_ROTATION = (
    ["execute"] * 3 + ["read_then_write"] * 2 + ["miss_param"] * 1 + ["irrelevant"] * 1
)


def generate_tool_use_prompt_multiturn(
    skill_name: str,
    skill_info: dict,
    difficulty: str = "medium",
    document_text: str = "",
) -> str:
    """Multi-turn variant of generate_tool_use_prompt — BFCL-style structure.

    Generates tasks where:
    - 3-5 atomic user messages are revealed sequentially
    - Each message has its own state-based success criterion
    - Final reward only when all messages are completed and "done" submitted

    Mirrors BFCL multi_turn_base structure to reduce train-eval distribution
    mismatch on the multi-turn axis.
    """
    import random as _rng

    examples = skill_info.get("example_tasks", [])
    if len(examples) > 4:
        examples = _rng.sample(examples, 4)
    examples_str = (
        "\n  ".join(f"- {e}" for e in examples)
        if examples
        else "- multi-turn workflows"
    )

    corpus_section = ""
    if document_text:
        corpus_section = (
            "OPTIONAL FLAVOR (you MAY borrow surface names/numbers/entities from this\n"
            "snippet, but the DOMAIN above is fixed - do NOT make the task about this\n"
            "snippet, and never reference it in user messages):\n"
            f"<REFERENCE_SNIPPET>\n{document_text}\n</REFERENCE_SNIPPET>\n"
        )

    domain = _rng.choice(_TOOL_USE_DOMAINS)
    task_type = _rng.choice(_TASK_TYPE_ROTATION)

    return (
        _V_MULTITURN_PROMPT
        .replace("__SKILL_NAME__", skill_name)
        .replace("__SKILL_DESCRIPTION__", skill_info.get("description", ""))
        .replace("__SKILL_CATEGORY__", skill_info.get("category", ""))
        .replace("__DIFFICULTY__", difficulty)
        .replace("__EXAMPLES__", examples_str)
        .replace("__DOMAIN_BLOCK__", _DOMAIN_BLOCK.replace("__DOMAIN__", domain))
        .replace("__TASK_TYPE_BLOCK__", _TASK_TYPE_BLOCKS[task_type])
        .replace("__CORPUS_SECTION__", corpus_section)
    )


def get_env_gen_prompt_fn():
    """Return the env-gen prompt function based on SPARE_MULTITURN_ENV_GEN env var.

    Default: generate_tool_use_prompt (single-task structure, original).
    SPARE_MULTITURN_ENV_GEN=1: generate_tool_use_prompt_multiturn (BFCL-style
        multi-turn structure with 3-5 atomic user messages revealed progressively).
    """
    import os
    if os.environ.get("SPARE_MULTITURN_ENV_GEN") == "1":
        return generate_tool_use_prompt_multiturn
    return generate_tool_use_prompt
