"""LLM-based synthetic language game generator for LLM training."""

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import openai
except ImportError:
    openai = None  # Make openai optional for testing/deployment without it

from .prompts import GAME_GENERATION_SYSTEM_PROMPT, generate_game_generation_prompt


@dataclass
class GameSpec:
    """Specification for a generated LLM game."""
    skill_name: str
    skill_category: str
    skill_description: str
    game_name: str
    game_description: str
    instructions: str
    action_format: str
    reward_structure: Dict
    termination_conditions: List[str]
    code: str
    metadata: Dict = field(default_factory=dict)


class SyntheticGameGenerator:
    """Generate synthetic language games for LLM training."""

    COGNITIVE_SKILLS = {
        "Pattern Recognition": {
            "category": "Core Reasoning",
            "description": "Identifying patterns in sequences, numbers, or text",
            "complexity": "low",
            "example_games": [
                "Caesar/substitution cipher decoding",
                "matrix transformation rule discovery (what operation maps grid A to grid B?)",
                "regex pattern inference from string examples",
                "number base conversion patterns",
                "modular arithmetic cycles (clock arithmetic)",
                "sequence with hidden piecewise rules (different rules for odd/even indices)",
                "ASCII art pattern completion (rotate, reflect, tile)",
                "binary encoding/decoding puzzles",
                "function composition rule discovery (given input-output pairs, find the function)",
            ]
        },
        "Logical Deduction": {
            "category": "Core Reasoning",
            "description": "Drawing conclusions from given premises and rules",
            "complexity": "medium",
            "example_games": ["logic puzzles", "syllogism solving", "rule-based games"]
        },
        "Strategic Planning": {
            "category": "Executive Skills",
            "description": "Multi-step planning and resource management",
            "complexity": "high",
            "example_games": ["path planning", "resource allocation", "scheduling puzzles"]
        },
        "Mathematical Reasoning": {
            "category": "Analytical Skills",
            "description": "Solving mathematical problems and equations",
            "complexity": "medium",
            "example_games": [
                "system of nonlinear equations",
                "combinatorial counting with constraints (permutations, partitions)",
                "modular inverse / Chinese Remainder Theorem",
                "probability of compound events",
                "geometric area/volume with composite shapes",
                "integer partition or Diophantine equation problems",
                "graph-based shortest path or connectivity",
                "polynomial root finding or factoring",
                "expected value or conditional probability calculations",
                "cryptarithmetic (SEND + MORE = MONEY style)",
            ]
        },
        "Spatial Reasoning": {
            "category": "Visual Skills",
            "description": "Understanding spatial relationships and mental rotation",
            "complexity": "medium",
            "example_games": ["grid navigation", "direction puzzles", "position tracking"]
        },
        "Causal Inference": {
            "category": "Core Reasoning",
            "description": "Understanding cause-effect relationships",
            "complexity": "high",
            "example_games": ["cause-effect chains", "system dynamics", "prediction games"]
        },
        "Memory Recall": {
            "category": "Cognitive Skills",
            "description": "Remembering and utilizing past information",
            "complexity": "low",
            "example_games": ["memory games", "sequence recall", "information tracking"]
        },
        "Optimization": {
            "category": "Executive Skills",
            "description": "Finding optimal solutions under constraints",
            "complexity": "high",
            "example_games": ["knapsack problems", "scheduling optimization", "route planning"]
        },
        "Language Understanding": {
            "category": "Language Skills",
            "description": "Understanding and manipulating language",
            "complexity": "medium",
            "example_games": ["word puzzles", "anagrams",  "riddles", "semantic games"]
        }
    }

    def __init__(
        self,
        model: str = "qwen/qwen3-coder",
        use_tinker: bool = False,
        tinker_sampling_client=None,
        tinker_renderer=None,
    ):
        """Initialize the game generator.

        Args:
            model: Model to use for generation
            use_tinker: Whether to use Tinker backend instead of OpenRouter
            tinker_sampling_client: Tinker SamplingClient (required if use_tinker=True)
            tinker_renderer: Tinker Renderer (required if use_tinker=True)
        """
        self.model = model
        self.use_tinker = use_tinker

        if use_tinker:
            if tinker_sampling_client is None or tinker_renderer is None:
                raise ValueError("tinker_sampling_client and tinker_renderer required when use_tinker=True")
            self.tinker_sampling_client = tinker_sampling_client
            self.tinker_renderer = tinker_renderer
            self.client = None
        else:
            if openai is None:
                raise ImportError("openai package required when use_tinker=False. Install with: pip install openai")
            assert os.environ.get('OPENROUTER_API_KEY'), 'Please include OpenRouter API key in the environment variable'
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get('OPENROUTER_API_KEY')
            )
            self.tinker_sampling_client = None
            self.tinker_renderer = None

    def parse_generation_response(self, response: str) -> str:
        """Parse the LLM response to extract game code."""
        # Remove thinking tags if present
        if '<think>' in response and '</think>' in response:
            # Remove everything between <think> and </think>
            think_start = response.find('<think>')
            think_end = response.find('</think>')
            if think_end > think_start:
                response = response[:think_start] + response[think_end + 8:]

        # Extract code from markdown blocks
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            # If no markdown blocks, assume entire response is code
            # But remove any remaining think tags
            response = response.replace('<think>', '').replace('</think>', '')
            code = response.strip()

        # Post-process to fix common issues
        code = self._fix_common_issues(code)
        return code

    def _fix_common_issues(self, code: str) -> str:
        """Automatically fix common issues in generated code."""

        # Preserve f-strings because generated games use meaningful brace syntax.

        # Ensure max_turns check exists
        if 'def step' in code and 'truncated = self.turn_count >= self.max_turns' not in code:
            # Add it before the return statement in step()
            lines = code.split('\n')
            in_step = False
            for i, line in enumerate(lines):
                if 'def step' in line:
                    in_step = True
                elif in_step and 'return ' in line and 'truncated' in line:
                    # Insert the check before the return
                    indent = len(line) - len(line.lstrip())
                    lines.insert(i, ' ' * indent + '# Check max turns')
                    lines.insert(i + 1, ' ' * indent + 'truncated = self.turn_count >= self.max_turns')
                    break
            code = '\n'.join(lines)

        return code

    async def _call_llm_async(self, system_message: str, user_message: str) -> str:
        """Call LLM asynchronously using the configured backend.

        Args:
            system_message: System message for the LLM
            user_message: User message/prompt for the LLM

        Returns:
            LLM response text
        """
        if self.use_tinker:
            messages = [
                {"role": "user", "content": user_message},
            ]
            model_input = self.tinker_renderer.build_generation_prompt(messages)

            import tinker
            response = await self.tinker_sampling_client.sample_async(
                model_input,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    temperature=0.3,  # Lower temperature for more reliable code generation
                    max_tokens=16384,
                ),
            )

            # Parse response
            parsed_message, _ = self.tinker_renderer.parse_response(response.sequences[0].tokens)
            return parsed_message['content']
        else:
            # Use OpenRouter backend (sync)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,  # Lower temperature for more reliable code generation
                max_tokens=16384,
            )
            return response.choices[0].message.content

    def _call_llm(self, system_message: str, user_message: str) -> str:
        """Call LLM using the configured backend (sync wrapper).

        Args:
            system_message: System message for the LLM
            user_message: User message/prompt for the LLM

        Returns:
            LLM response text
        """
        if self.use_tinker:
            # Tinker requires async, so run in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(
                self._call_llm_async(system_message, user_message)
            )
        else:
            # OpenRouter is sync
            return asyncio.run(self._call_llm_async(system_message, user_message))

    def generate_game(self, skill_name: Optional[str] = None, difficulty: str = "medium") -> GameSpec:
        """Generate a complete language game.

        Args:
            skill_name: Specific skill to target, or None for random
            difficulty: Game difficulty level

        Returns:
            GameSpec object with complete game definition
        """
        if skill_name is None:
            skill_name = random.choice(list(self.COGNITIVE_SKILLS.keys()))

        skill_info = self.COGNITIVE_SKILLS[skill_name]
        prompt = generate_game_generation_prompt(skill_name, skill_info, difficulty)

        # Generate game using LLM (via backend-agnostic helper)
        response_text = self._call_llm(
            system_message=GAME_GENERATION_SYSTEM_PROMPT,
            user_message=prompt
        )

        code = self.parse_generation_response(response_text)

        # Extract game name from code
        game_name_match = re.search(r'class\s+(\w+)Env', code)
        game_name = game_name_match.group(1) if game_name_match else f"{skill_name.replace(' ', '')}Game"

        return GameSpec(
            skill_name=skill_name,
            skill_category=skill_info['category'],
            skill_description=skill_info['description'],
            game_name=game_name,
            game_description=f"A language game that trains {skill_name}",
            instructions=f"Game training {skill_name}",
            action_format="\\boxed{answer}",
            reward_structure={
                "success": 1.0,
                "format_error": -1.0,
                "invalid_action": -0.5,
                "step": -0.01
            },
            termination_conditions=["max_turns", "game_won", "invalid_format"],
            code=code,
            metadata={
                "difficulty": difficulty,
                "complexity": skill_info['complexity'],
                "generation_model": self.model,
                "timestamp": time.time()
            }
        )

    def generate_game_for_player(self, learning_potential: float) -> str:
        """Generate a game adapted to player's learning potential.

        Args:
            learning_potential: Player's current learning potential (ρ_t)

        Returns:
            Path to generated game file
        """
        # Map learning potential to difficulty and skill complexity
        if learning_potential > 0.5:
            difficulty = "hard"
            skill_complexity = "high"
        elif learning_potential > 0.1:
            difficulty = "medium"
            skill_complexity = "medium"
        else:
            difficulty = "easy"
            skill_complexity = "low"

        # Select appropriate skill based on complexity
        appropriate_skills = [
            skill for skill, info in self.COGNITIVE_SKILLS.items()
            if info['complexity'] == skill_complexity
        ]

        skill_name = random.choice(appropriate_skills)

        # Generate game
        game_spec = self.generate_game(skill_name, difficulty)

        # Save game to file
        timestamp = int(time.time())
        game_file = f"generated_games/game_{timestamp}_{game_spec.game_name.lower()}.py"
        os.makedirs("generated_games", exist_ok=True)

        with open(game_file, 'w') as f:
            f.write(game_spec.code)

        # Save game spec
        spec_file = game_file.replace('.py', '.json')
        with open(spec_file, 'w') as f:
            json.dump({
                'skill_name': game_spec.skill_name,
                'skill_category': game_spec.skill_category,
                'game_name': game_spec.game_name,
                'difficulty': difficulty,
                'learning_potential': learning_potential,
                'metadata': game_spec.metadata
            }, f, indent=2)

        return game_file

    def save_game(self, game_spec: GameSpec, filepath: str):
        """Save game specification to file."""
        # Save Python code
        with open(filepath, 'w') as f:
            f.write(f'"""Generated game for {game_spec.skill_name}"""\n\n')
            f.write(game_spec.code)

        # Save JSON spec
        json_path = filepath.replace('.py', '.json')
        spec_dict = {
            "skill_name": game_spec.skill_name,
            "skill_category": game_spec.skill_category,
            "skill_description": game_spec.skill_description,
            "game_name": game_spec.game_name,
            "game_description": game_spec.game_description,
            "instructions": game_spec.instructions,
            "action_format": game_spec.action_format,
            "reward_structure": game_spec.reward_structure,
            "termination_conditions": game_spec.termination_conditions,
            "metadata": game_spec.metadata
        }

        with open(json_path, 'w') as f:
            json.dump(spec_dict, f, indent=2)
