"""Shared types for SPARE dual-role orchestration across all backends."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class TrajectoryStatus(str, Enum):
    """Status of a trajectory episode.

    Similar to Slime's Sample.Status and τ²-bench's InteractionResult status.
    """
    PENDING = "pending"       # Created but not executed
    RUNNING = "running"       # Currently executing
    COMPLETED = "completed"   # Completed successfully
    TRUNCATED = "truncated"   # Hit max turns/length limit
    TIMEOUT = "timeout"       # Took too long
    FAILED = "failed"         # Failed with recoverable error
    ABORTED = "aborted"       # Aborted (critical failure)


@dataclass
class Trajectory:
    """Complete episode trajectory for SPARE dual-role training.

    Similar to Slime's Sample and τ²-bench's InteractionResult.
    Holds entire multi-turn conversation history with token-level learning control
    via loss_mask.

    This replaces the old per-step Trajectory format with an episode-based format.
    """
    # Core identification
    index: int = 0                         # Unique index

    # Prompt data (initial state)
    prompt: str = ""                       # Initial prompt text
    messages: List[Dict[str, Any]] = field(default_factory=list)  # Full OpenAI-format conversation

    # Token data (accumulated during rollout)
    tokens: List[int] = field(default_factory=list)          # ALL tokens (prompt + all responses)
    loss_mask: List[int] = field(default_factory=list)       # 1=learn, 0=don't learn (matches tokens)

    # Response data
    response: str = ""                     # Concatenated assistant responses (for compatibility)
    response_length: int = 0               # Number of response tokens (sum of loss_mask)

    # Training data
    rollout_log_probs: List[float] = field(default_factory=list)  # Log probs for response tokens

    # Reward data
    reward: float = 0.0                    # Episode-level reward

    # Status
    status: TrajectoryStatus = TrajectoryStatus.PENDING

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional: Per-turn data (for analysis/debugging)
    turn_count: int = 0                    # Number of turns executed


@dataclass
class SpareConfig:
    """Configuration for SpareOrchestrator.

    This centralizes all configuration needed for dual-role orchestration,
    making it easy to share across different backends.
    """
    # Model generation parameters
    env_temperature: float = 0.7  # Temperature for environment generation
    env_top_p: float = 0.95
    env_top_k: int = 20
    env_max_tokens: int = 8192

    actor_temperature: float = 1.0  # Temperature for actor gameplay
    actor_top_p: float = 0.95
    actor_top_k: int = 20
    actor_max_tokens: int = 8192

    # Prompt templates
    env_generation_template: str = "qwen3_game_generation"  # Template name for env generation
    actor_template: str = "qwen3_game"  # Template name for actor gameplay

    # Per-role thinking control (for models like Qwen3 that support enable_thinking)
    env_enable_thinking: Optional[bool] = None  # None = use adapter default; True/False = override
    actor_enable_thinking: Optional[bool] = None  # None = use adapter default; True/False = override

    # Repair invalid generated games while training only the initial generation.
    env_repair_turns: int = 0
    # Persist rejected generations under <games_dir>/rejected/.
    persist_rejected: bool = False

    # Game parameters
    max_turns: int = 30  # Maximum turns per game
    max_context_length: int = 16384  # Maximum context characters
    plays_per_game: int = 8  # Number of times to play each game (produces N trajectories per game)

    # RL parameters
    gamma: float = 0.99  # Discount factor for actor returns

    # Format reward (spurious reward) parameters
    use_format_reward: bool = False
    format_reward_value: float = 1.0
    no_format_penalty: float = -1.0

    # Async optimization parameters
    max_concurrent_games: int = 8  # Max concurrent games in async mode
    batch_generation_size: int = 4  # Batch size for environment generation

    # Learning potential parameters (for reward computation)
    gamma1: float = 0.99  # Slow moving average (gamma_fast in code)
    gamma2: float = 0.95  # Fast moving average (gamma_slow in code)
    use_solver_variance_reward: bool = False

    # Actor normalization: per-game EMA centering or within-batch GRPO z-scores.
    reward_normalization: str = "ema_baseline"
    game_baseline_decay: float = 0.5  # Decay rate for per-game baseline EMA (ema_baseline only)

    # Self-judge parameters (for verifier robustness)
    use_self_judge: bool = False  # Enable self-judge to validate generated environments
    self_judge_temperature: float = 0.3  # Low temperature for more deterministic judgments
    self_judge_max_tokens: int = 2048  # Max tokens for self-judge response
    self_judge_penalty: float = -0.5  # Penalty applied to env reward if self-judge says "no"
    self_judge_max_turns_to_show: int = 5  # Max turns to include in trajectory for judging

    # Environment reward scaling balances the two roles' update frequencies.
    env_reward_scaling_variant: int = 1  # 0=none, 1=simple scaling
    max_env_reward_scale: float = 50.0  # Maximum scale cap to prevent training instability
    auto_compute_env_reward_scale: bool = True  # If True, compute scale from trajectory counts

    # Generation continues when environment trajectories are excluded from training.
    train_on_env_trajectories: bool = True

    # Fixed model evaluation parameters
    # Uses an external model (e.g., GPT-5-mini via OpenAI) to evaluate game difficulty
    fixed_eval_model: str = "gpt-5-mini"  # Model ID (OpenAI or OpenRouter)
    fixed_eval_plays_per_game: int = 8  # Number of times fixed model plays each game
    fixed_eval_max_concurrent: int = 128  # Max concurrent game plays (rate limiting)
    fixed_eval_temperature: float = 0.7  # Temperature for fixed model generation
    fixed_eval_max_tokens: int = 16384  # Max tokens per fixed model response
    fixed_eval_api_base_url: Optional[str] = None  # API base URL (None for OpenAI default)
    fixed_eval_api_key_env: str = "OPENAI_API_KEY"  # Env var for API key

    # Environment reward variant: "learning_potential" (default), "regret_based", or "micro_lp"
    # learning_potential: raw_reward = |mean(game_rewards) - mu_slow|
    # regret_based: raw_reward = R(with_hint) - R(without_hint), where hint is from an external LLM
    # micro_lp: raw_reward = max(0, mean(late_rewards) - mean(early_rewards)) within delay window
    env_reward_variant: str = "learning_potential"

    # Regret-based env reward parameters (only used when env_reward_variant="regret_based")
    hint_mode: str = "self"                            # "self" = training model, "external" = OpenAI/OpenRouter API
    hint_model: str = "gpt-5.1-mini"                  # Model for hint generation (external mode only)
    hint_api_key_env: str = "OPENAI_API_KEY"           # Env var for API key
    hint_api_base_url: Optional[str] = None            # API base URL (None = OpenAI default)
    hint_temperature: float = 0.3                       # Temperature for hint generation
    hint_max_tokens: int = 2048                         # Max tokens for hint response
    hint_plays_per_game: int = 4                        # Plays with hint per game for regret
    hint_injection_template: str = "{observation}\n\nHINT: {hint}"

    # Cognitive skills for game generation
    # If empty, uses all available skills in the game generator
    # Available skills: Pattern Recognition, Mathematical Reasoning, Logical Deduction,
    #                   Strategic Planning, Spatial Reasoning, Causal Inference,
    #                   Memory Recall, Optimization, Language Understanding
    skills: List[str] = field(default_factory=list)

    # GEM evaluation parameters
    gem_eval_tasks: Optional[List[str]] = None  # GEM task IDs for evaluation
    gem_eval_episodes: int = 4  # Episodes per GEM task
    gem_eval_max_concurrent: int = 16  # Max concurrent episodes during GEM eval

    # Fixed environment mode parameters
    fixed_env_mode: bool = False  # Whether to use fixed environments instead of self-play
    fixed_env_source: Optional[List[str]] = None  # Environment source specs
    difficulty_variant: str = "sliding_window"  # "sliding_window" or "lp"
    difficulty_tau_acc: float = 0.9  # Success rate threshold (sliding window)
    difficulty_tau_num: int = 8  # Min attempts before promotion (sliding window)
    difficulty_d_delta: int = 4  # Max window width (sliding window)
    lp_gamma_fast: float = 0.35  # Fast EMA gamma (LP controller)
    lp_gamma_slow: float = 0.15  # Slow EMA gamma (LP controller)

    # Environment validator parameters (LLM rejection sampling during generation)
    # When enabled, generated games are validated by an LLM before gameplay.
    # "self" uses the training model; otherwise uses external model via OpenAI/OpenRouter.
    use_env_validator: bool = False
    env_validator_model: str = "self"
    env_validator_api_key_env: str = "OPENROUTER_API_KEY"
    env_validator_api_base_url: Optional[str] = "https://openrouter.ai/api/v1"
    env_validator_temperature: float = 0.3
    env_validator_max_tokens: int = 16384

    # Delayed LP: skip per-rollout LP updates, defer to delayed training time
    # Set automatically when proposer_training_delay > 0 with LP variant
    skip_lp_update: bool = False

    # Corpus-grounded generation
    corpus_file: Optional[str] = None
    corpus_max_doc_tokens: int = 6000
    corpus_seed: int = 42

    # Action format: how the Reasoning Agent wraps actions in its responses.
    # "boxed" = \boxed{answer} (cognitive games, default)
    # "tool_call" = <tool_call>JSON</tool_call> + <answer>result</answer> (tool-use)
    action_format: str = "boxed"

    # Game type for self-play environment generation.
    # "cognitive" = ToolUseBaseEnv/logic/math games (default)
    # "tool_use" = ToolUseBaseEnv subclasses (multi-tool API games)
    game_type: str = "cognitive"

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
