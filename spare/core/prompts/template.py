"""Prompt templates for language game generation."""
from typing import Optional


# Base templates (originally from spiral)
def apply_qwen3_template(observation: str, system_prompt: Optional[str] = None) -> str:
    del system_prompt
    return (
        f"<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_qwen3_general_template(
    question: str, system_prompt: Optional[str] = None
) -> str:
    del system_prompt
    return (
        f"<|im_start|>user\nQuestion: {question}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_octothinker_template(
    observation: str, system_prompt: Optional[str] = None
) -> str:
    """OctoThinker template for game-based tasks."""
    del system_prompt
    return (
        f"A conversation between User and Assistant. The User presents the observation of a zero-sum game, and the Assistant makes a valid action in order to win. "
        f"The Assistant first thinks about the reasoning process in the mind and then provides the action. "
        f"User: You must put your answer inside \\boxed{{}} "
        f"and your final answer will be extracted automatically by the \\boxed{{}} tag.\n"
        f"Observation: {observation}\n"
        f"Assistant:"
    )


def apply_llama_instruct_template(
    observation: str, system_prompt: Optional[str] = None
) -> str:
    system_message = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are playing a two-player zero-sum game. Make valid actions to win.<|eot_id|>"
    user_message = f"<|start_header_id|>user<|end_header_id|>\n\nCurrent Observation: {observation}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>\n"
    assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
    return system_message + user_message + assistant_start


# SPARE-specific templates
def apply_qwen3_game_generation_template(prompt: str) -> str:
    """Apply Qwen3 template for game generation (environment role)."""
    return (
        f"<|im_start|>system\nYou are an expert Python programmer and game designer. "
        f"You create educational language games for training Large Language Models.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def apply_qwen3_game_template(observation: str) -> str:
    """Apply Qwen3 template for language games - SPARE-specific variant."""
    return (
        f"<|im_start|>user\nYou are playing a language game. Make valid actions to win.\n"
        f"Observation: {observation}\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def apply_qwen3_multiturn_game_generation_template(prompt: str) -> str:
    """Apply Qwen3 template for multi-turn agentic game generation (environment role)."""
    return (
        f"<|im_start|>system\n{MULTITURN_GAME_GENERATION_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


TEMPLATE_FACTORY = {
    # Base templates
    "qwen3": apply_qwen3_template,
    "qwen3_general": apply_qwen3_general_template,
    "octothinker": apply_octothinker_template,
    "r1": apply_octothinker_template,
    "llama_instruct": apply_llama_instruct_template,
    # SPARE-specific templates
    "qwen3_game_generation": apply_qwen3_game_generation_template,
    "qwen3_multiturn_game_generation": apply_qwen3_multiturn_game_generation_template,
    "qwen3_game": apply_qwen3_game_template,
    # Tool-use templates are registered after their function definitions.
}

GAME_GENERATION_SYSTEM_PROMPT = "You are an expert game designer. Generate diverse, creative Python game environments."

# Self-judge prompt template for validating generated environments
SELF_JUDGE_SYSTEM_PROMPT = """You are a code reviewer and game quality evaluator. Your task is to analyze game code and gameplay trajectories to determine if the game is working correctly."""

SELF_JUDGE_PROMPT_TEMPLATE = """You are reviewing a generated game environment and a trajectory from playing it.

## Game Code:
```python
{game_code}
```

## Trajectory from playing this game:
{trajectory}

## Your Task:
Analyze whether this trajectory makes sense given the game code. Consider:

1. **Observation Consistency**: Are the observations returned by the game consistent with the game's logic and state?
2. **Reward Correctness**: Are the rewards appropriate for the actions taken? (Success should give positive reward, failures negative)
3. **Game Logic**: Does the game logic work correctly? Can the player actually win?
4. **Code Bugs**: Are there obvious bugs in the code (e.g., wrong conditions, unreachable success states)?
5. **Termination**: Does the game terminate correctly on success/max turns?

Please reason step by step about each point, then give your final verdict.

Put your final answer in \\boxed{{yes}} if the trajectory makes sense and the game is working correctly.
Put your final answer in \\boxed{{no}} if there are issues with the game or trajectory.
"""


def format_trajectory_for_judge(
    observations: list,
    actions: list,
    rewards: list,
    max_turns_to_show: int = 5
) -> str:
    """Format a trajectory for self-judge evaluation.

    Args:
        observations: List of observations from the game
        actions: List of actions taken
        rewards: List of rewards received
        max_turns_to_show: Maximum number of turns to include (to limit context)

    Returns:
        Formatted trajectory string
    """
    lines = []
    num_turns = min(len(actions), max_turns_to_show)

    # Show first observation
    if observations:
        lines.append("**Initial Observation (Turn 0):**")
        obs_preview = observations[0][:500] + "..." if len(observations[0]) > 500 else observations[0]
        lines.append(obs_preview)
        lines.append("")

    # Show turns
    for i in range(num_turns):
        lines.append("**Turn {}:**".format(i + 1))

        # Action
        if i < len(actions):
            action_preview = actions[i][:300] + "..." if len(actions[i]) > 300 else actions[i]
            lines.append("- Action: {}".format(action_preview))

        # Reward
        if i < len(rewards):
            lines.append("- Reward: {}".format(rewards[i]))

        # Next observation (if not last turn)
        if i + 1 < len(observations):
            obs_preview = observations[i + 1][:300] + "..." if len(observations[i + 1]) > 300 else observations[i + 1]
            lines.append("- Next Observation: {}".format(obs_preview))

        lines.append("")

    if len(actions) > max_turns_to_show:
        lines.append("... ({} more turns not shown)".format(len(actions) - max_turns_to_show))

    # Summary
    lines.append("")
    lines.append("**Summary:**")
    lines.append("- Total turns: {}".format(len(actions)))
    lines.append("- Total reward: {}".format(sum(rewards) if rewards else 0))
    lines.append("- Final reward: {}".format(rewards[-1] if rewards else "N/A"))

    return "\n".join(lines)


def generate_self_judge_prompt(game_code: str, trajectory_str: str) -> str:
    """Generate the full self-judge prompt.

    Args:
        game_code: The Python code of the generated game
        trajectory_str: Formatted trajectory string (from format_trajectory_for_judge)

    Returns:
        Complete prompt for self-judge evaluation
    """
    return SELF_JUDGE_PROMPT_TEMPLATE.format(
        game_code=game_code,
        trajectory=trajectory_str
    )


def generate_game_generation_prompt(skill_name: str, skill_info: dict, difficulty: str = "medium") -> str:
    """Generate a prompt for LLM game creation.

    Args:
        skill_name: Name of the cognitive skill to target
        skill_info: Dictionary containing skill metadata (category, description, complexity, example_games)
        difficulty: Game difficulty level ("easy", "medium", "hard")

    Returns:
        Complete prompt string for game generation
    """
    import random as _rng
    # Sample 3-4 example games to show variety without overwhelming the prompt
    examples = skill_info.get("example_games", [])
    if len(examples) > 4:
        examples = _rng.sample(examples, 4)
    examples_str = ", ".join(examples) if examples else "puzzles, logic problems, reasoning challenges"

    return (
        f"Create a challenging single-player text-based game as a Python class"
        f" that tests: {skill_name} ({skill_info['description']}).\n"
        "\n"
        f"GAME CONCEPT IDEAS (pick one or invent your own):\n"
        f"  {examples_str}\n"
        "\n"
        "RULES:\n"
        "- Be creative with the game concept — it can be a puzzle, riddle, logic problem,"
        " code tracing, word game, optimization task, or any reasoning challenge.\n"
        "- One task per episode. Player answers with \\boxed{answer}.\n"
        "- Give your class a descriptive, unique name.\n"
        "- The task must be DETERMINISTICALLY SOLVABLE from the observation alone —"
        " all information needed to derive the answer must be explicitly stated."
        " No hidden state the player cannot see.\n"
        "- The observation should be clearly written so a careful reader can follow"
        " the logic to the answer.\n"
        "\n"
        "INFORMATION HIDING (critical for RL training):\n"
        "- The observation must NOT contain the answer. The player must DERIVE it.\n"
        "- For sequence/pattern games: show only partial data, never the target value.\n"
        "- Never state the pattern rule directly — the player must discover it.\n"
        "- Wrong-answer feedback must give hints (e.g., higher/lower, which part is wrong),"
        " never reveal the full solution.\n"
        "- The player must use \\boxed{answer} format — remind them in the observation.\n"
        "\n"
        "DIFFICULTY:\n"
        "- The task MUST require at least 3 distinct reasoning steps — not solvable"
        " in one or two arithmetic operations.\n"
        "- Random guessing should succeed < 1% of the time — use large answer spaces.\n"
        "- Use randomized parameters large enough that the answer is not obvious.\n"
        "- Do NOT generate simple formula-substitution or single-operation problems.\n"
        "- Compute the solution from the generated puzzle; never hardcode it.\n"
        "  WRONG: return 'recursive'  # same answer regardless of puzzle\n"
        "  WRONG: return str(random.randint(1,10))  # not derived from the actual task\n"
        "  RIGHT: generate puzzle first, then compute self._solution from it\n"
        "\n"
        "INTERFACE CONTRACT:\n"
        "```python\n"
        "import random\n"
        "import re\n"
        "from typing import Any, Optional, Tuple, Dict, List\n"
        "from math_verify import parse, verify\n"
        "\n"
        "def verify_answer(player_answer, solution):\n"
        '    """Check if answer is correct: string match first, then math equivalence."""\n'
        "    if str(player_answer).strip() == str(solution).strip():\n"
        "        return True\n"
        "    try:\n"
        "        return verify(parse(player_answer), parse(solution))\n"
        "    except:\n"
        "        return False\n"
        "\n"
        "class DescriptiveNameEnv:\n"
        "    def __init__(self, max_turns=10, **kwargs):\n"
        "        ...\n"
        "        self.reset()\n"
        "\n"
        "    def reset(self, seed=None) -> Tuple[str, dict]:\n"
        "        # Generate a new task. Returns (observation_with_instructions, {})\n"
        "        ...\n"
        "\n"
        "    def solution(self) -> str:\n"
        "        # REQUIRED: Return the exact answer string that solves the current task.\n"
        "        ...\n"
        "\n"
        "    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:\n"
        "        self.turn_count += 1\n"
        "        truncated = self.turn_count >= self.max_turns\n"
        r"        match = re.search(r'\\boxed\{([^}]+)\}', action)" + "\n"
        "        if match and verify_answer(match.group(1), self.solution()):\n"
        '            return ("Correct! ...", 1.0, True, False, {})\n'
        "        elif truncated:\n"
        "            return (f\"Time's up. The answer was {self.solution()}. [task]\", 0.0, False, True, {})\n"
        "        else:\n"
        '            return ("Wrong. [specific hint about why]. Try again. [task description]", 0.0, False, False, {})\n'
        "        # EVERY code path must return (str, float, bool, bool, dict) — no exceptions.\n"
        "\n"
        "    def close(self): pass\n"
        "```\n"
        "\n"
        "EPISODE STRUCTURE (critical for RL training):\n"
        "1. reset(): Generate ONE task/puzzle for this episode\n"
        "2. step(): Player tries to solve that SAME task multiple times\n"
        "3. Correct answer → terminated=True (episode ends with success)\n"
        "4. Max turns reached → truncated=True, reveal the solution so the model learns from failure\n"
        "5. NEVER generate a new task inside step() — the task is fixed for the whole episode\n"
        "\n"
        "OBSERVATION REQUIREMENTS:\n"
        "- Turn 1 (reset): Show instructions + task + remind player to use \\boxed{answer} format\n"
        "- Turn 2+ (step): Show specific feedback on the attempt + the SAME task (no instructions)\n"
        "- The task/puzzle MUST be visible every turn or the model won't know what to solve\n"
        "- Feedback must be specific to the attempted answer, not generic ('Wrong. Try again.' is not enough)\n"
        "- No \\boxed{} in action: remind the player of the format + show task (do not terminate)\n"
        "\n"
        "ROBUSTNESS:\n"
        "- Never divide by a value that could be zero — regenerate if needed\n"
        "- Initialize ALL instance variables in __init__ before calling reset()\n"
        "- Return empty dict {} for info (never strings)\n"
        "- Every code path in step() must return a 5-tuple (str, float, bool, bool, dict).\n"
        "- Naming: never store data in self.solution — that shadows the required solution()"
        " method. Use self._solution or self._answer instead.\n"
        "- Brace escaping: in f-strings, {word} evaluates word as a Python expression."
        " To write a literal \\boxed{answer} escape it as \\boxed{{answer}}."
        " Same rule with .format(): any literal brace must be doubled.\n"
        "\n"
        "Generate the complete Python code in a ```python block."
    )


# --- Multi-Turn Agentic Game Generation ---

MULTITURN_GAME_GENERATION_SYSTEM_PROMPT = (
    "You are an expert Python programmer and game designer specializing in "
    "interactive, multi-turn text-based games. You create environments where "
    "the player must make sequential decisions across multiple turns, with "
    "each action changing the game state."
)

_MULTITURN_SKILL_EXAMPLES = {
    "Pattern Recognition": [
        "grid exploration where rooms contain pattern fragments — collect enough fragments to decode a message",
        "Mastermind: guess a hidden color code, get feedback (correct position, wrong position, absent)",
        "mine sweeper on a small grid — uncover cells, flag mines, use number clues to deduce safe cells",
        "decode a cipher by exploring a map — each location reveals a letter mapping",
    ],
    "Logical Deduction": [
        "detective investigation: interview suspects, search crime scenes, examine evidence, then accuse",
        "Clue/Cluedo: move between rooms, gather evidence cards, deduce who/what/where",
        "logic grid puzzle: ask yes/no questions to fill a constraint grid across turns",
        "escape room: examine objects in sequence, each reveals a clue for the next puzzle",
    ],
    "Mathematical Reasoning": [
        "balance a set of equations by adjusting coefficients across turns",
        "navigate a number maze — each cell has a math operation, choose path to reach target value",
        "auction game: bid on items with hidden values, manage budget to maximize total worth",
        "build a tower of blocks with weight constraints — choose placement order carefully",
    ],
    "Spatial Reasoning": [
        "navigate a 2D grid maze — find the exit while avoiding dead ends and traps",
        "Sokoban: push boxes onto target positions in a grid warehouse",
        "arrange furniture in rooms to satisfy spatial constraints (next to window, not blocking door)",
        "navigate a robot through obstacles — choose direction each turn, limited fuel",
    ],
    "Strategic Planning": [
        "tower defense: place 3 turrets on a grid, then simulate waves of enemies",
        "city builder: allocate budget across turns to build infrastructure and reach population target",
        "supply chain: route goods through a network, manage inventory at each node",
        "chess endgame: move pieces on a small board to reach checkmate",
    ],
    "Causal Inference": [
        "scientific experiment: choose which variables to test each turn, deduce the hidden causal rule",
        "troubleshooting: diagnose a broken machine by testing components in sequence",
        "epidemic simulation: choose interventions each turn, observe effects, find the transmission source",
        "circuit debugging: toggle switches and measure outputs to find the faulty gate",
    ],
    "Memory Recall": [
        "memory palace: visit rooms and memorize items, then answer recall questions at the end",
        "witness interview: hear testimony across turns, then spot contradictions to find the liar",
        "spy decoder: read intercepted messages across turns, then reconstruct the full secret",
        "shopping list: items are revealed one at a time — at the end, buy exactly what was listed",
    ],
    "Optimization": [
        "knapsack: items arrive one by one — decide to take or skip each to maximize value within weight limit",
        "scheduling: assign jobs to machines across turns to minimize total completion time",
        "portfolio: allocate investments each round with changing market conditions",
        "factory: configure production lines across turns to maximize output before deadline",
    ],
    "Language Understanding": [
        "text adventure: explore a mansion, read notes, interpret riddles to find hidden treasure",
        "translation puzzle: decode foreign phrases by exploring a village and talking to NPCs",
        "crossword builder: get clues across turns, fill in a word grid incrementally",
        "story detective: read chapters of a story across turns, answer comprehension questions to progress",
    ],
}


def generate_multiturn_game_generation_prompt(
    skill_name: str, skill_info: dict, difficulty: str = "medium",
    max_turns: int = 20,
) -> str:
    """Generate a prompt for LLM creation of multi-turn agentic games.

    These games require sequential decision-making across turns, with
    evolving state — unlike single-turn puzzle games.

    Args:
        skill_name: Name of the cognitive skill to target
        skill_info: Dictionary with skill metadata (category, description, example_games)
        difficulty: Game difficulty level ("easy", "medium", "hard")
        max_turns: Maximum turns the game will be played for (passed to game's __init__)

    Returns:
        Complete prompt string for multi-turn game generation
    """
    import random as _rng
    # Use multi-turn specific examples if available, fall back to skill_info
    examples = _MULTITURN_SKILL_EXAMPLES.get(skill_name, skill_info.get("example_games", []))
    if len(examples) > 3:
        examples = _rng.sample(examples, 3)
    examples_str = "\n".join(f"  - {e}" for e in examples)

    return (
        f"Create an interactive multi-turn text-based game as a Python class"
        f" that tests: {skill_name} ({skill_info['description']}).\n"
        "\n"
        f"GAME CONCEPT IDEAS for {skill_name} (pick one or invent similar):\n"
        f"{examples_str}\n"
        "\n"
        "WHAT MAKES A GOOD MULTI-TURN GAME:\n"
        "Think of classic text adventures, board games, or strategy games. The player\n"
        "navigates a world that changes with every action. Good examples:\n"
        "  - Grid exploration: move through rooms, find keys to unlock doors, reach the exit\n"
        "  - Trading: buy low / sell high across rounds with fluctuating prices\n"
        "  - Survival: manage health, food, tools while exploring — wrong choices kill you\n"
        "  - Investigation: question suspects, search locations, piece together clues\n"
        "  - Tower defense: place defenses, then waves arrive — adapt strategy each wave\n"
        "  - Crafting: gather materials, combine them in the right order to build something\n"
        "\n"
        "WHAT TO AVOID (common failure modes):\n"
        "  BAD: A math puzzle where the player guesses the answer and retries on failure.\n"
        "       That is a single-turn puzzle with retry, NOT a multi-turn game.\n"
        "  BAD: Increment/decrement a variable until it matches a target.\n"
        "       That is a linear search, not a game — there are no decisions.\n"
        "  BAD: The player has only one reasonable action each turn (e.g., always 'allocate').\n"
        "       If the optimal path is obvious, there is no strategic depth.\n"
        "  BAD: The game can be solved in 1-3 turns. Too short for multi-turn training.\n"
        "  BAD: Embedding single-turn math/logic puzzles inside a multi-turn frame.\n"
        "       E.g., 'go to forest, solve 2+3, collect wood' — the actual decisions are trivial.\n"
        "  BAD: Using input() to get player answers. NEVER use input(). The action parameter\n"
        "       to step() IS the player's full response. Parse everything from it.\n"
        "\n"
        "DESIGN REQUIREMENTS:\n"
        "- The game world has STATE that changes on every action (positions, inventories,\n"
        "  health, money, unlocked areas, NPC attitudes, etc.).\n"
        "- Each turn, the player chooses from 2+ meaningfully different actions.\n"
        "- Some actions are BETTER than others — wrong choices waste turns or cause harm.\n"
        f"- The game will be played for at most {max_turns} turns. Design difficulty,\n"
        f"  resource budgets, and pacing accordingly. Optimal play should win in\n"
        f"  roughly {max_turns // 2}–{max_turns * 3 // 4} turns.\n"
        "- Random play should lose most of the time.\n"
        "- The game must have a clear WIN condition and ideally a LOSE condition too.\n"
        "\n"
        "ACTION FORMAT:\n"
        "- The player wraps every action in \\boxed{action}. The step() method extracts\n"
        "  the content inside \\boxed{} and interprets it.\n"
        "- Show available actions clearly in every observation, e.g.:\n"
        "    Actions: \\boxed{go north}, \\boxed{go south}, \\boxed{pick up key}, \\boxed{rest}\n"
        "- If the player's input has no \\boxed{}, return a format reminder. Do NOT terminate.\n"
        "\n"
        "OBSERVATION FORMAT:\n"
        "- Each observation must show: (1) current state, (2) result of last action,\n"
        "  (3) available actions, (4) progress (e.g., 'Turn 3/12 | HP: 7/10 | Items: [key]').\n"
        "- The initial observation (from reset) includes rules, goal, and starting state.\n"
        "\n"
        "REWARD:\n"
        "- Win: reward = 1.0, terminated = True\n"
        "- Lose: reward = 0.0, terminated = True\n"
        "- Intermediate turns: reward = 0.0\n"
        "- Max turns without winning: reward = 0.0, truncated = True\n"
        "\n"
        "INTERFACE:\n"
        "```python\n"
        "import random\n"
        "import re\n"
        "from typing import Tuple\n"
        "from math_verify import parse, verify\n"
        "\n"
        "def verify_answer(player_answer, solution):\n"
        '    """Check if answer is correct: string match first, then math equivalence."""\n'
        "    if str(player_answer).strip() == str(solution).strip():\n"
        "        return True\n"
        "    try:\n"
        "        return verify(parse(player_answer), parse(solution))\n"
        "    except:\n"
        "        return False\n"
        "\n"
        "class DescriptiveNameEnv:\n"
        f"    def __init__(self, max_turns={max_turns}, **kwargs):\n"
        "        self.max_turns = max_turns\n"
        "        self.turn_count = 0\n"
        "        # ALL state variables initialized here\n"
        "        self.reset()\n"
        "\n"
        "    def reset(self, seed=None) -> Tuple[str, dict]:\n"
        "        if seed is not None:\n"
        "            random.seed(seed)\n"
        "        self.turn_count = 0\n"
        "        # Randomize the game world\n"
        "        return observation, {}\n"
        "\n"
        "    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:\n"
        "        self.turn_count += 1\n"
        r"        match = re.search(r'\\boxed\{([^}]*)\}', action)" + "\n"
        "        if not match:\n"
        "            return ('Use \\\\boxed{action} format.', 0.0, False, False, {})\n"
        "        cmd = match.group(1).strip().lower()\n"
        "        # Update state, check win/lose\n"
        "        # Use verify_answer(cmd, solution) for numeric comparisons\n"
        "        truncated = self.turn_count >= self.max_turns\n"
        "        return (observation, reward, terminated, truncated, {})\n"
        "\n"
        "    def close(self): pass\n"
        "```\n"
        "\n"
        "REFERENCE EXAMPLE — Trading Game (shows the pattern; yours must be DIFFERENT):\n"
        "```python\n"
        "class TradingGameEnv:\n"
        '    """Buy low, sell high across rounds with fluctuating prices. Reach target gold."""\n'
        "    def __init__(self, max_turns=12, **kwargs):\n"
        "        self.max_turns = max_turns\n"
        "        self.turn_count = 0\n"
        "        self.gold = 0\n"
        "        self.stock = 0\n"
        "        self.price = 0\n"
        "        self.price_history = []\n"
        "        self.target_gold = 0\n"
        "        self.trend = 0  # hidden: +1 rising, -1 falling\n"
        "        self.reset()\n"
        "\n"
        "    def reset(self, seed=None):\n"
        "        if seed is not None:\n"
        "            random.seed(seed)\n"
        "        self.turn_count = 0\n"
        "        self.gold = 100\n"
        "        self.stock = 0\n"
        "        self.price = random.randint(8, 15)\n"
        "        self.price_history = [self.price]\n"
        "        self.target_gold = 200\n"
        "        self.trend = random.choice([-1, 1])\n"
        "        return (f'Welcome to the Trading Game! Reach {self.target_gold} gold to win.\\n'\n"
        "                f'Price: {self.price} | Gold: {self.gold} | Stock: {self.stock}\\n'\n"
        "                f'History: {self.price_history}\\n'\n"
        "                f'Turn 0/{self.max_turns}\\n'\n"
        "                f'Actions: \\\\boxed{{buy N}}, \\\\boxed{{sell N}}, \\\\boxed{{wait}}'), {}\n"
        "\n"
        "    def step(self, action):\n"
        "        self.turn_count += 1\n"
        r"        match = re.search(r'\\boxed\{([^}]*)\}', action)" + "\n"
        "        if not match:\n"
        "            return ('Use \\\\boxed{action} format.', 0.0, False, False, {})\n"
        "        cmd = match.group(1).strip().lower()\n"
        "        truncated = self.turn_count >= self.max_turns\n"
        "        msg = ''\n"
        "        # Parse action\n"
        "        if cmd.startswith('buy '):\n"
        "            try:\n"
        "                n = int(cmd[4:])\n"
        "                cost = n * self.price\n"
        "                if cost <= self.gold and n > 0:\n"
        "                    self.gold -= cost\n"
        "                    self.stock += n\n"
        "                    msg = f'Bought {n} at {self.price}.'\n"
        "                else:\n"
        "                    msg = f'Cannot buy {n} (need {cost} gold, have {self.gold}).'\n"
        "            except ValueError:\n"
        "                msg = 'Invalid number.'\n"
        "        elif cmd.startswith('sell '):\n"
        "            try:\n"
        "                n = int(cmd[5:])\n"
        "                if n <= self.stock and n > 0:\n"
        "                    self.gold += n * self.price\n"
        "                    self.stock -= n\n"
        "                    msg = f'Sold {n} at {self.price}.'\n"
        "                else:\n"
        "                    msg = f'Cannot sell {n} (have {self.stock}).'\n"
        "            except ValueError:\n"
        "                msg = 'Invalid number.'\n"
        "        elif cmd == 'wait':\n"
        "            msg = 'You wait.'\n"
        "        else:\n"
        "            msg = 'Unknown action.'\n"
        "        # Update price with trend + noise\n"
        "        if random.random() < 0.3:  # trend reversal\n"
        "            self.trend *= -1\n"
        "        self.price = max(1, self.price + self.trend * random.randint(1, 4))\n"
        "        self.price_history.append(self.price)\n"
        "        # Check win\n"
        "        total = self.gold + self.stock * self.price\n"
        "        if self.gold >= self.target_gold:\n"
        "            return (f'{msg} Gold: {self.gold}. You win!', 1.0, True, False, {})\n"
        "        if truncated:\n"
        "            return (f'{msg} Time up. Gold: {self.gold}, Stock: {self.stock}. Needed {self.target_gold}.', 0.0, False, True, {})\n"
        "        obs = (f'{msg}\\nPrice: {self.price} | Gold: {self.gold} | Stock: {self.stock}\\n'\n"
        "               f'History: {self.price_history[-5:]}\\n'\n"
        "               f'Turn {self.turn_count}/{self.max_turns}\\n'\n"
        "               f'Actions: \\\\boxed{{buy N}}, \\\\boxed{{sell N}}, \\\\boxed{{wait}}')\n"
        "        return (obs, 0.0, False, False, {})\n"
        "\n"
        "    def close(self): pass\n"
        "```\n"
        "\n"
        "ROBUSTNESS:\n"
        "- NEVER use input(). The step(action) parameter IS the player's response.\n"
        "- Never divide by a value that could be zero.\n"
        "- Initialize ALL instance variables in __init__ before calling reset().\n"
        "- Return empty dict {} for info (never strings).\n"
        "- Every code path in step() must return a 5-tuple (str, float, bool, bool, dict).\n"
        "- Handle unexpected player input gracefully (show valid actions, don't crash).\n"
        "- Use only the Python standard library (random, re, collections, etc.).\n"
        "- Brace escaping: in f-strings, literal braces must be doubled {{ }}.\n"
        "- NEVER put backslashes inside f-string expressions. Use a variable instead:\n"
        "    BAD:  f'{\"\\n\".join(items)}'\n"
        "    GOOD: sep = '\\n'; f'{sep.join(items)}'  OR  '\\n'.join(items)\n"
        "\n"
        "Generate the complete Python code in a ```python block.\n"
        "Remember: the game must have BRANCHING DECISIONS where different choices\n"
        "lead to different outcomes. A game with only one reasonable action per turn is not acceptable."
    )


# --- Tool-Use Game Generation (Qwen3 / Qwen3.5 optimized) ---
#
# Used when --spare-game-type tool_use.
# The model generates ToolUseBaseEnv subclasses where the agent
# calls structured API tools (search, filter, query, etc.) to solve tasks.

TOOL_USE_GAME_GENERATION_SYSTEM_PROMPT = (
    "You are an expert API and tool-calling environment designer. "
    "You create structured tool-use task environments for training language "
    "models to call functions correctly. Each environment defines custom "
    "tools (search, filter, lookup, compute, etc.) and a task that requires "
    "calling these tools in the right sequence to arrive at an answer. "
    "The agent uses native tool calling via OpenAI-format tool schemas and "
    "submits its final answer with <answer>result</answer>."
)


def apply_qwen3_tool_use_game_generation_template(prompt: str) -> str:
    """Apply Qwen3/Qwen3.5 template for tool-use game generation.

    Used when --spare-game-type tool_use. Optimized for Qwen3's
    <|im_start|>/<|im_end|> format with a tool-design system prompt.
    """
    return (
        f"<|im_start|>system\n{TOOL_USE_GAME_GENERATION_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# Register the tool-use template after its function definition
TEMPLATE_FACTORY["qwen3_tool_use_game_generation"] = apply_qwen3_tool_use_game_generation_template
