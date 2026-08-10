"""SPARE-specific arguments for Slime training.

This module provides functions to add SPARE-related command-line arguments
to Slime's argument parser.
"""

def add_spare_arguments(parser):
    """Add SPARE-specific arguments to the argument parser.

    Args:
        parser: argparse.ArgumentParser to add arguments to
    """
    spare_args = parser.add_argument_group("SPARE Arguments")

    # Learning potential arguments
    spare_args.add_argument(
        "--spare-gamma1",
        type=float,
        default=0.99,
        help="Fast moving average gamma for learning potential computation",
    )
    spare_args.add_argument(
        "--spare-gamma2",
        type=float,
        default=0.95,
        help="Slow moving average gamma for learning potential computation",
    )

    # Environment generation arguments
    spare_args.add_argument(
        "--spare-env-temperature",
        type=float,
        default=0.7,
        help="Temperature for environment generation",
    )
    spare_args.add_argument(
        "--spare-env-top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus sampling) for environment generation",
    )
    spare_args.add_argument(
        "--spare-env-top-k",
        type=int,
        default=20,
        help="Top-k sampling for environment generation",
    )
    spare_args.add_argument(
        "--spare-env-generation-template",
        type=str,
        default="qwen3_game_generation",
        help="Template name for environment generation",
    )

    spare_args.add_argument(
        "--spare-env-max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for environment generation",
    )

    # Actor gameplay arguments
    spare_args.add_argument(
        "--spare-actor-temperature",
        type=float,
        default=1.0,
        help="Temperature for actor gameplay",
    )
    spare_args.add_argument(
        "--spare-actor-top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus sampling) for actor gameplay",
    )
    spare_args.add_argument(
        "--spare-actor-top-k",
        type=int,
        default=20,
        help="Top-k sampling for actor gameplay",
    )
    spare_args.add_argument(
        "--spare-actor-template",
        type=str,
        default="qwen3_game",
        help="Template name for actor gameplay",
    )
    spare_args.add_argument(
        "--spare-env-enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode for env generation (proposer). "
             "Passes enable_thinking=True to chat template for env role only.",
    )
    spare_args.add_argument(
        "--spare-actor-enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode for the actor in multi-turn games. "
             "Each turn's prompt is re-rendered from the canonical (think-"
             "stripped) chat history, the actor generates <think>...</think> "
             "before its answer, and every turn is emitted as its own per-"
             "turn training sample (slime TurnRecord/fan-out machinery); all "
             "turn samples of one episode share a group_id and the episode "
             "reward. Default off = legacy accumulated single-sequence path.",
    )
    spare_args.add_argument(
        "--spare-env-repair-turns",
        type=int,
        default=0,
        help="Multi-turn game-gen repair (Option A). 0 = legacy independent "
             "re-sampling. N>0 = feed the validation error back and ask the "
             "proposer to fix it (inference-only, up to N turns) so the actor "
             "gets a valid env; the proposer is trained on its TURN-1 generation "
             "(valid->regret, broken->0).",
    )
    spare_args.add_argument(
        "--spare-persist-rejected",
        action="store_true",
        default=False,
        help="Persist validation-failed game generations to "
             "<games_dir>/rejected/ for inspection (otherwise discarded).",
    )
    spare_args.add_argument(
        "--spare-max-turns",
        type=int,
        default=50,
        help="Maximum number of turns per game",
    )
    spare_args.add_argument(
        "--spare-actor-max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for actor gameplay",
    )

    spare_args.add_argument(
        "--spare-use-solver-variance-reward",
        action="store_true",
        default=False,
        help="Enable solver variance reward (reward for solver variance)",
    )
    spare_args.add_argument(
        "--spare-max-context-length",
        type=int,
        default=32768,
        help="Maximum context length for game history",
    )

    # Reward computation arguments
    spare_args.add_argument(
        "--spare-gamma",
        type=float,
        default=0.99,
        help="Discount factor for reward computation",
    )
    spare_args.add_argument(
        "--spare-use-format-reward",
        action="store_true",
        help="Enable spurious format reward (reward for \\boxed{} format)",
    )
    spare_args.add_argument(
        "--spare-format-reward-value",
        type=float,
        default=1.0,
        help="Reward value for using correct format",
    )

    # Game generation arguments
    spare_args.add_argument(
        "--spare-game-regeneration-interval",
        type=int,
        default=50,
        help="Rollout interval between game regeneration (0 to disable)",
    )
    spare_args.add_argument(
        "--spare-num-games-per-rollout",
        type=int,
        default=32,
        help="Number of games to generate per rollout",
    )
    spare_args.add_argument(
        "--spare-games-dir",
        type=str,
        default="./spare_games",
        help="Directory to store generated games",
    )
    spare_args.add_argument(
        "--spare-game-difficulty",
        type=str,
        default="hard",
        choices=["easy", "medium", "hard"],
        help="Difficulty level for generated games",
    )
    spare_args.add_argument(
        "--spare-game-type",
        type=str,
        default="cognitive",
        choices=["cognitive", "tool_use"],
        help="Type of games to generate: 'cognitive' (default, logic/math games), "
             "'tool_use' (API/data tool-calling tasks using ToolUseBaseEnv)",
    )
    spare_args.add_argument(
        "--spare-skills",
        type=str,
        nargs="+",
        default=None,
        help="List of skills to use for game generation. "
             "For cognitive games: 'Pattern Recognition', 'Mathematical Reasoning', etc. "
             "For tool_use games: 'Information Retrieval', 'Data Analysis', etc. "
             "Default: top-2 skills for the selected game type.",
    )
    spare_args.add_argument(
        "--spare-skills-per-regen",
        type=int,
        default=0,
        help="Round-robin only this many of the configured --spare-skills per regen epoch "
             "(0 or >= #skills = use all skills every regen). Rotates a disjoint window of "
             "skills each regen so per-rollout cost stays flat while skill breadth scales; "
             "each active skill gets more games and idle gaps are bounded.",
    )
    spare_args.add_argument(
        "--spare-trajectories-per-game",
        type=int,
        default=1,
        help="Number of trajectories to collect per game",
    )
    spare_args.add_argument(
        "--spare-cache-dir",
        type=str,
        default=None,
        help="Directory to cache generated games (e.g., ./cached_games/). "
             "When set, games are saved to cache_dir/step_{rollout_id}/ on creation, "
             "allowing inspection even after regeneration deletes them from games_dir.",
    )

    # Environment reward scaling arguments
    spare_args.add_argument(
        "--spare-env-reward-scaling-variant",
        type=int,
        default=1,
        choices=[0, 1],
        help="Variant for env reward scaling: 0=no reweighting, 1=simple scaling",
    )
    spare_args.add_argument(
        "--spare-max-env-reward-scale",
        type=float,
        default=50.0,
        help="Maximum scale factor for environment rewards (caps the auto-computed scale)",
    )
    spare_args.add_argument(
        "--spare-auto-compute-env-reward-scale",
        action="store_true",
        default=True,
        help="Auto-compute env reward scale from trajectory counts and regeneration interval",
    )
    spare_args.add_argument(
        "--spare-no-auto-compute-env-reward-scale",
        action="store_false",
        dest="spare_auto_compute_env_reward_scale",
        help="Disable auto-compute of env reward scale",
    )
    spare_args.add_argument(
        "--spare-train-on-env-trajectories",
        action="store_true",
        default=True,
        help="Include environment trajectories in training batch (default: True)",
    )
    spare_args.add_argument(
        "--spare-no-train-on-env-trajectories",
        action="store_false",
        dest="spare_train_on_env_trajectories",
        help="Exclude environment trajectories from training batch (for ablation)",
    )

    # Delayed proposer training
    spare_args.add_argument(
        "--spare-proposer-training-delay",
        type=int,
        default=0,
        help="Delay proposer training by N rollout steps. 0=immediate (default). "
             "Env trajectories from step K are buffered and trained at step K+N. "
             "Requires --use-tis for importance ratio correction.",
    )

    # Actor reward normalization arguments
    spare_args.add_argument(
        "--spare-reward-normalization",
        type=str,
        default="ema_baseline",
        choices=["ema_baseline", "grpo"],
        help="Actor reward normalization: 'ema_baseline' (subtract per-game EMA mean) or 'grpo' (per-game z-score)",
    )
    spare_args.add_argument(
        "--spare-game-baseline-decay",
        type=float,
        default=0.5,
        help="Decay rate for per-game baseline EMA (ema_baseline mode only, default: 0.5)",
    )

    # Self-judge arguments (for verifier robustness)
    spare_args.add_argument(
        "--spare-use-self-judge",
        action="store_true",
        default=False,
        help="Enable self-judge to validate generated environments",
    )
    spare_args.add_argument(
        "--spare-self-judge-temperature",
        type=float,
        default=0.3,
        help="Temperature for self-judge (low for deterministic judgments)",
    )
    spare_args.add_argument(
        "--spare-self-judge-max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for self-judge response",
    )
    spare_args.add_argument(
        "--spare-self-judge-penalty",
        type=float,
        default=-0.5,
        help="Penalty applied to env reward if self-judge says 'no'",
    )
    spare_args.add_argument(
        "--spare-self-judge-max-turns-to-show",
        type=int,
        default=5,
        help="Max turns to include in trajectory for self-judge evaluation",
    )

    # Environment reward variant arguments
    spare_args.add_argument(
        "--spare-env-reward-variant",
        type=str,
        default="learning_potential",
        choices=["learning_potential", "regret_based", "micro_lp", "blend"],
        help="Environment reward variant: 'learning_potential' (LP-based), 'regret_based' (hint-based regret), "
             "'micro_lp' (game-level improvement within delay window, requires delay > 0), or 'blend' "
             "(matched-timing regret + micro_lp, fixed-scale normalized then weighted; requires delay > 0)",
    )
    spare_args.add_argument(
        "--spare-regret-weight",
        type=float,
        default=0.5,
        help="[blend] weight on the (scale-normalized) matched-timing regret component",
    )
    spare_args.add_argument(
        "--spare-micro-lp-weight",
        type=float,
        default=0.5,
        help="[blend] weight on the (scale-normalized) micro-LP (actor improvement) component",
    )
    spare_args.add_argument(
        "--spare-regret-scale",
        type=float,
        default=0.15,
        help="[blend] fixed nominal scale the regret component is divided by before weighting "
             "(NOT a per-batch std; keeps weights interpretable and avoids small-std blowup)",
    )
    spare_args.add_argument(
        "--spare-micro-lp-scale",
        type=float,
        default=0.10,
        help="[blend] fixed nominal scale the micro-LP component is divided by before weighting",
    )
    spare_args.add_argument(
        "--spare-micro-lp-unsigned",
        action="store_true",
        default=False,
        help="[blend] clamp micro-LP to max(0, late-early) instead of the signed default. "
             "Signed (default) lets a game that made the actor WORSE be a negative signal.",
    )
    spare_args.add_argument(
        "--spare-micro-lp-estimator",
        type=str,
        default="slope",
        choices=["slope", "twobucket"],
        help="[blend] micro-LP estimator. 'slope' (default): WLS slope of per-step "
             "win-rate across the full window (uses every step, ~2-3x lower variance). "
             "'twobucket': legacy late_mean - early_mean.",
    )
    spare_args.add_argument(
        "--spare-frontier-weight",
        type=float,
        default=0.0,
        help="[blend] weight on the frontier anchor: subtract frontier_weight*"
             "((win_rate-0.5)^2 / frontier_scale) from blend. >0 holds the curriculum "
             "near 50% win-rate (max learnability) and breaks the difficulty-easing "
             "death spiral. 0 disables.",
    )
    spare_args.add_argument(
        "--spare-frontier-scale",
        type=float,
        default=0.08,
        help="[blend] fixed nominal scale the frontier penalty (win_rate-0.5)^2 is "
             "divided by before weighting. Tune so comp_frontier_norm matches the "
             "regret/micro-LP component magnitudes.",
    )
    spare_args.add_argument(
        "--spare-plateau-weight",
        type=float,
        default=0.0,
        help="[blend] weight on the flat-top difficulty reward plateau_reward(win_rate). "
             ">0 ADDS a non-negative anchor that is 1.0 in-band (~50% win) and 0 at the "
             "extremes — the additive alternative to the frontier penalty. Intended use: "
             "--spare-plateau-weight 0.6 --spare-regret-weight 0.4 --spare-regret-floor "
             "with micro-LP and frontier off (0). Difficulty regulates which hardness; "
             "floored regret refines which in-band game is most teachable.",
    )
    spare_args.add_argument(
        "--spare-plateau-lo",
        type=float,
        default=0.4,
        help="[blend] low edge of the plateau flat top (full reward for win_rate>=lo).",
    )
    spare_args.add_argument(
        "--spare-plateau-hi",
        type=float,
        default=0.6,
        help="[blend] high edge of the plateau flat top (full reward for win_rate<=hi).",
    )
    spare_args.add_argument(
        "--spare-plateau-ramp",
        type=float,
        default=0.25,
        help="[blend] linear ramp width on each side of the flat top; plateau hits 0 at "
             "lo-ramp and hi+ramp (defaults => 0 at win_rate 0.15 / 0.85).",
    )
    spare_args.add_argument(
        "--spare-regret-floor",
        action="store_true",
        default=False,
        help="[blend] floor the (scaled) regret component at 0 (max(0,.) then clip to 1). "
             "Neutralizes hint-play-timeout / misleading-hint negatives without masking, "
             "and keeps regret on the same [0,1] scale as the plateau. Pair with "
             "--spare-plateau-weight so the whole blend is >=0 (no masking, no death-spiral).",
    )
    spare_args.add_argument(
        "--spare-hint-mode",
        type=str,
        default="self",
        choices=["self", "external"],
        help="Hint generation mode: 'self' (training model) or 'external' (OpenAI/OpenRouter API)",
    )
    spare_args.add_argument(
        "--spare-hint-model",
        type=str,
        default="gpt-5.1-mini",
        help="Model for hint generation in external mode (e.g., openai/gpt-4.1-mini)",
    )
    spare_args.add_argument(
        "--spare-hint-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for hint model API key",
    )
    spare_args.add_argument(
        "--spare-hint-api-base-url",
        type=str,
        default=None,
        help="API base URL for hint model (None for OpenAI default)",
    )
    spare_args.add_argument(
        "--spare-hint-temperature",
        type=float,
        default=0.3,
        help="Temperature for hint generation",
    )
    spare_args.add_argument(
        "--spare-hint-max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for hint model response",
    )
    spare_args.add_argument(
        "--spare-hint-plays-per-game",
        type=int,
        default=4,
        help="Number of times to play each game with hint (for regret computation)",
    )
    spare_args.add_argument(
        "--spare-compact-filter",
        action="store_true",
        default=False,
        help="Zero loss_mask on truncated (max-turns) trajectories so they carry no "
             "gradient: running out of turn budget is not a policy failure.",
    )

    # Environment memory arguments (memory-augmented generation)
    spare_args.add_argument(
        "--spare-use-env-memory",
        action="store_true",
        default=False,
        help="Enable environment memory buffer for memory-augmented generation. "
             "High-regret past environments are used as few-shot seeds.",
    )
    spare_args.add_argument(
        "--spare-env-memory-max-size",
        type=int,
        default=200,
        help="Maximum number of environments to keep in memory buffer",
    )

    # Environment validator arguments (rejection sampling during game generation)
    spare_args.add_argument(
        "--spare-use-env-validator",
        action="store_true",
        default=False,
        help="Enable environment validation via LLM rejection sampling during game generation",
    )
    spare_args.add_argument(
        "--spare-env-validator-model",
        type=str,
        default="self",
        help='Model for environment validation. "self" uses the training model (no API key needed). '
             'Otherwise specify a model ID (e.g., google/gemini-3-flash-preview for OpenRouter)',
    )
    spare_args.add_argument(
        "--spare-env-validator-api-key-env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable name for validator API key",
    )
    spare_args.add_argument(
        "--spare-env-validator-api-base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="API base URL for environment validator",
    )
    spare_args.add_argument(
        "--spare-env-validator-temperature",
        type=float,
        default=0.3,
        help="Temperature for environment validator (low for deterministic)",
    )
    spare_args.add_argument(
        "--spare-env-validator-max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens for environment validator response",
    )

    # Weave tracking (optional)
    spare_args.add_argument(
        "--spare-use-weave",
        action="store_true",
        help="Enable Weave tracing for experiments",
    )
    spare_args.add_argument(
        "--spare-wandb-project",
        type=str,
        default="spare",
        help="W&B project name for Weave tracking",
    )

    # Fixed model evaluation arguments
    spare_args.add_argument(
        "--spare-fixed-eval-interval",
        type=int,
        default=0,
        help="Rollouts between fixed model evaluations (0=disabled)",
    )
    spare_args.add_argument(
        "--spare-fixed-eval-model",
        type=str,
        default="gpt-5-mini",
        help="Model ID for fixed evaluation (e.g., gpt-5-mini for OpenAI)",
    )
    spare_args.add_argument(
        "--spare-fixed-eval-api-base-url",
        type=str,
        default=None,
        help="API base URL for fixed evaluation (None for OpenAI default)",
    )
    spare_args.add_argument(
        "--spare-fixed-eval-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for API key (default: OPENAI_API_KEY)",
    )
    spare_args.add_argument(
        "--spare-fixed-eval-plays-per-game",
        type=int,
        default=None,
        help="Number of times fixed model plays each game (default: uses --spare-trajectories-per-game)",
    )
    spare_args.add_argument(
        "--spare-fixed-eval-max-concurrent",
        type=int,
        default=128,
        help="Maximum concurrent game plays for rate limiting",
    )
    spare_args.add_argument(
        "--spare-fixed-eval-temperature",
        type=float,
        default=0.7,
        help="Temperature for fixed model generation",
    )
    spare_args.add_argument(
        "--spare-fixed-eval-max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens for fixed model response",
    )

    # GEM evaluation
    gem_args = parser.add_argument_group("SPARE GEM Evaluation")

    gem_args.add_argument(
        "--spare-gem-eval-config",
        type=str,
        default=None,
        help="Path to GEM evaluation YAML config file. "
             "Defines tasks with per-task episodes, max_turns, temperature, max_tokens.",
    )

    # Fixed environments
    fixed_env_args = parser.add_argument_group("SPARE Fixed Environment")

    fixed_env_args.add_argument(
        "--spare-mode",
        type=str,
        default="self_play",
        choices=["self_play", "fixed_env"],
        help='Training mode: "self_play" (default, generate games) or "fixed_env" (use external envs)',
    )
    fixed_env_args.add_argument(
        "--spare-fixed-env-source",
        type=str,
        nargs="+",
        default=None,
        help='Environment sources for fixed_env mode. '
             'Formats: "rlve:Sorting", "rlve:*", "gem:game:Sudoku-v0-easy", "gem:*", '
             '"game_file:/path/to/dir"',
    )
    fixed_env_args.add_argument(
        "--spare-fixed-env-same-problem",
        action="store_true",
        default=False,
        help="All trajectories-per-game plays of an env share ONE generated problem "
             "(per-prompt GRPO grouping, matching RLVE's n-samples-per-prompt). "
             "Requires adapter support; adapters without it fall back to "
             "independent problems.",
    )
    fixed_env_args.add_argument(
        "--spare-difficulty-variant",
        type=str,
        default="sliding_window",
        choices=["sliding_window", "lp"],
        help='Difficulty control variant: "sliding_window" (RLVE-style) or "lp" (LP-based)',
    )
    fixed_env_args.add_argument(
        "--spare-difficulty-tau-acc",
        type=float,
        default=0.9,
        help="Success rate threshold for sliding window promotion (default: 0.9)",
    )
    fixed_env_args.add_argument(
        "--spare-difficulty-tau-num",
        type=int,
        default=8,
        help="Minimum attempts before sliding window promotion check (default: 8)",
    )
    fixed_env_args.add_argument(
        "--spare-difficulty-d-delta",
        type=int,
        default=4,
        help="Maximum sliding window width (default: 4)",
    )
    fixed_env_args.add_argument(
        "--spare-lp-gamma-fast",
        type=float,
        default=0.35,
        help="Fast EMA gamma for LP difficulty controller (default: 0.35)",
    )
    fixed_env_args.add_argument(
        "--spare-lp-gamma-slow",
        type=float,
        default=0.15,
        help="Slow EMA gamma for LP difficulty controller (default: 0.15)",
    )

    # Fixed pool ablation (no adaptive difficulty)
    fixed_env_args.add_argument(
        "--spare-fixed-pool-size",
        type=int,
        default=0,
        help="Fixed pool size. 0 = disabled (use adaptive difficulty). "
             "When > 0, pre-samples a fixed pool of (env_id, difficulty) pairs at init "
             "and uses them for the entire run instead of adaptive difficulty control.",
    )
    fixed_env_args.add_argument(
        "--spare-fixed-pool-seed",
        type=int,
        default=42,
        help="RNG seed for reproducible fixed pool sampling (default: 42)",
    )
    fixed_env_args.add_argument(
        "--spare-fixed-pool-max-difficulty",
        type=int,
        default=5,
        help="Maximum difficulty cap when sampling fixed pool entries (default: 5)",
    )

    fixed_env_args.add_argument(
        "--spare-no-replacement",
        action="store_true",
        default=False,
        help="Sample environments without replacement. Once the pool is exhausted, "
             "refill and reshuffle. Ensures each game is seen before any is repeated.",
    )
    fixed_env_args.add_argument(
        "--spare-static-game-pool",
        action="store_true",
        default=False,
        help="Use pre-generated files from --spare-games-dir without generating "
             "replacement games. This keeps the normal native tool-use rollout path.",
    )

    # Pre-generation of synthetic games (for no-proposer ablation)
    fixed_env_args.add_argument(
        "--spare-pre-generate-games",
        type=int,
        default=0,
        help="Number of synthetic games to pre-generate at init (0=disabled). "
             "When > 0, generates N games using the model at init, saves them, "
             "and creates a SyntheticGameAdapter for fixed-env training.",
    )
    fixed_env_args.add_argument(
        "--spare-pre-generate-difficulty",
        type=str,
        default="hard",
        choices=["easy", "medium", "hard"],
        help="Difficulty level for pre-generated games (default: hard)",
    )
    fixed_env_args.add_argument(
        "--spare-pre-generate-skills",
        type=str,
        nargs="+",
        default=None,
        help="Skills for pre-generated games. Default: Pattern Recognition, Mathematical Reasoning",
    )

    # Corpus-grounded generation
    corpus_args = parser.add_argument_group("SPARE Corpus-Grounded Generation")
    corpus_args.add_argument(
        "--spare-corpus-file",
        type=str,
        default=None,
        help="Path to JSONL corpus file for corpus-grounded game generation. "
             "Each line: {\"text\": \"...\"} or {\"orig_doc\": \"...\"}. "
             "When set, games are generated from sampled corpus documents.",
    )
    corpus_args.add_argument(
        "--spare-corpus-max-doc-tokens",
        type=int,
        default=6000,
        help="Maximum number of tokens per corpus document (word-level approximation, truncated if longer). Default: 6000.",
    )
    corpus_args.add_argument(
        "--spare-corpus-seed",
        type=int,
        default=42,
        help="Random seed for corpus document sampling. Default: 42.",
    )

    # Action format for different environment types
    action_args = parser.add_argument_group("SPARE action format")
    action_args.add_argument(
        "--spare-action-format",
        type=str,
        default="boxed",
        choices=["boxed", "tool_call"],
        help=(
            "Action format the Reasoning Agent uses in responses. "
            "'boxed' = \\boxed{answer} (cognitive games, default). "
            "'tool_call' = <tool_call>JSON</tool_call> (tool-use). "
            "Default: boxed."
        ),
    )

    # Override Slime defaults for SPARE runs
    parser.set_defaults(wandb_always_use_train_step=True)

    return parser
