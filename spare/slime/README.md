# SPARE Slime Backend Integration

This document explains how SPARE integrates with the Slime training backend.

## Overview

SPARE uses Slime as a distributed training backend with SGLang HTTP inference. The integration follows a **non-invasive approach** - we don't modify Slime's core code, instead using its extension points.

## Key Components

### 1. Custom Data Source (`spare/slime/data_source.py`)

**Purpose**: Bypass Slime's requirement for pre-existing training data files.

SPARE generates games on-the-fly, so we provide `SpareDataSource` that implements Slime's `DataSource` interface but doesn't require data files.

```python
class SpareDataSource(DataSource):
    def get_samples(self, num_samples: int) -> List[List[Sample]]:
        return []  # Rollout function generates actual data

    def add_samples(self, samples: List[List[Sample]]):
        pass  # Samples handled by rollout function

    def save(self, rollout_id):
        pass  # No state to save

    def load(self, rollout_id=None):
        pass  # No state to load
```

**Usage in shell script**:
```bash
ROLLOUT_ARGS=(
   --data-source-path spare.slime.data_source.SpareDataSource
   --rollout-function-path spare.slime.spare_rollout.spare_generate_rollout
   ...
)
```

### 2. Custom Rollout Function (`spare/slime/spare_rollout.py`)

**Purpose**: Implement SPARE's dual-role training loop.

The `spare_generate_rollout()` function:
1. Creates a `SpareOrchestrator` with the Slime model adapter
2. Calls `collect_trajectories(mode="async")` to generate games and play them
3. Converts `Trajectory` objects to Slime's `Sample` format
4. Returns `RolloutFnTrainOutput(samples, metrics)`

**Key points**:
- Uses `async` mode for optimal HTTP inference performance
- Generates games concurrently using `generate_and_save_games_async()`
- Plays games concurrently using `play_games_async()`
- Handles both environment and actor trajectories

### 3. Model Adapter (`spare/slime/model_adapter.py`)

**Purpose**: Adapt SGLang HTTP API to SPARE's `ModelAdapter` interface.

`SlimeModelAdapter` wraps SGLang's HTTP client:
- `generate()`: Synchronous HTTP call
- `generate_async()`: Native async HTTP call
- `generate_batch()`: Batches synchronous HTTP calls
- `apply_template()`: Uses tokenizer's chat template

### 4. Trajectory Converter (`spare/slime/trajectory_converter.py`)

**Purpose**: Convert SPARE `Trajectory` to Slime `Sample`.

`trajectory_to_slime_sample()` maps:
- `observation` → `Sample.prompt`
- `prompt_token_ids + response_token_ids` → `Sample.tokens`
- `reward` → `Sample.reward`
- `metadata` (role, turn, skill, etc.) → `Sample.metadata`

### 5. Arguments (`spare/slime/arguments.py`)

**Purpose**: Add SPARE-specific command-line arguments.

`add_spare_arguments(parser)` adds all SPARE args (returns `parser`):
- Learning potential: `--spare-gamma1`, `--spare-gamma2`
- Environment: `--spare-env-temperature`, `--spare-env-generation-template`
- Actor: `--spare-actor-temperature`, `--spare-actor-template`
- Game generation: `--spare-games-dir`, `--spare-num-games-per-rollout`

### 6. Training Wrapper (`train_spare_slime.py`)

**Purpose**: Entry point that properly initializes SPARE arguments.

```python
def main():
    from slime.utils.arguments import parse_args
    from spare.slime.arguments import add_spare_arguments

    args = parse_args(add_custom_arguments=add_spare_arguments)
    train(args)
```

## Data Flow

```
1. Slime calls spare_generate_rollout(args, rollout_id)
   ↓
2. Create SlimeModelAdapter (SGLang HTTP client)
   ↓
3. Create SpareOrchestrator(config, model, learning_potential)
   ↓
4. orchestrator.collect_trajectories(mode="async"):
   a. generate_and_save_games_async() → List[Path], Dict[Path, Trajectory]
   b. play_games_async() → List[List[Trajectory]]
   c. Process returns and compute environment rewards
   ↓
5. Convert Trajectories to Slime Samples:
   - Environment trajectories → Samples with role="environment"
   - Actor trajectories → Samples with role="actor"
   ↓
6. Return RolloutFnTrainOutput(samples, metrics)
```

## Logging

The orchestrator provides detailed logging:

```
[GEN-ASYNC] Generating 32 games concurrently
[GEN-ASYNC] Generated 32/32 games
[ASYNC] Statistics: 32 games, avg 12.5 turns (min=5, max=20), avg final reward 0.750, success rate 75.0%
[COLLECT] Final: 400 actor steps, 32 env trajectories
[SPARE] Rollout 0 complete: 400 actor, 32 env samples
```

## Configuration

See `cmd/games/train_spade_4b.sh` for a released configuration example.

Key parameters:
- `--rollout-batch-size`: Number of games per rollout (matches `--spare-num-games-per-rollout`)
- `--spare-game-regeneration-interval`: Regenerate games every N rollouts
- `--spare-games-dir`: Directory for generated game files
- `--spare-gamma1`, `--spare-gamma2`: Learning potential parameters

## Non-Invasive Integration

We follow these principles:

1. **No modifications to Slime core**: All SPARE code lives in `spare/slime/` (outside `slime/slime/`)
2. **Use extension points**:
   - `--data-source-path` for custom data sources
   - `--rollout-function-path` for custom rollout functions
   - `add_custom_arguments` for custom CLI arguments
3. **Wrapper scripts** instead of modifying Slime's entry points

This allows updating Slime independently without merge conflicts.
