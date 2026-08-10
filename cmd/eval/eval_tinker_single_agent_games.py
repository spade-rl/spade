#!/usr/bin/env python3
"""Evaluate models on single-agent synthetic games using Tinker inference."""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict

import tinker
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tqdm.asyncio import tqdm

sys.path.append('.')
from spare.core.envs.synthetic_game_env import make_synthetic_env


async def test_single_episode(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    game_file: str,
    episode_seed: int,
    max_turns: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
) -> Dict:
    """Test a single episode of a game."""
    result = {
        'success': False,
        'failure': False,
        'timeout': False,
        'invalid_format': False,
        'reward': 0.0,
        'error': None,
    }

    try:
        env = make_synthetic_env(game_file)
        obs, info = env.reset(seed=episode_seed)

        done = False
        turn = 0
        episode_reward = 0.0

        while not done and turn < max_turns:
            messages = [{'role': 'user', 'content': obs}]
            model_input = renderer.build_generation_prompt(messages)

            response = await sampling_client.sample_async(
                model_input,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                ),
            )

            parsed_message, _ = renderer.parse_response(response.sequences[0].tokens)
            action = parsed_message['content']

            match = re.search(r'\\boxed\{([^}]+)\}', action)
            if not match:
                match = re.search(r'\\\\boxed\{([^}]+)\}', action)

            if match:
                action = f'\\boxed{{{match.group(1)}}}'
            else:
                result['invalid_format'] = True
                result['reward'] = episode_reward
                return result

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            turn += 1

        result['reward'] = episode_reward

        if episode_reward > 0:
            result['success'] = True
        elif turn >= max_turns:
            result['timeout'] = True
        else:
            result['failure'] = True

    except Exception as e:
        result['error'] = str(e)
        result['failure'] = True

    return result


def aggregate_episode_results(episode_results) -> Dict:
    """Aggregate results from multiple episodes."""
    aggregated = {
        'success': 0,
        'failure': 0,
        'timeout': 0,
        'invalid_format': 0,
        'total_rewards': 0.0,
    }

    for result in episode_results:
        aggregated['total_rewards'] += result['reward']
        if result['success']:
            aggregated['success'] += 1
        if result['failure']:
            aggregated['failure'] += 1
        if result['timeout']:
            aggregated['timeout'] += 1
        if result['invalid_format']:
            aggregated['invalid_format'] += 1

    return aggregated


async def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a model on single-agent games using Tinker'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name to evaluate (e.g., Qwen/Qwen3-8B-Base)'
    )
    parser.add_argument(
        '--games_dir',
        type=str,
        required=True,
        help='Directory containing game files (.py)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to save results JSON'
    )
    parser.add_argument(
        '--renderer',
        type=str,
        default='qwen3',
        help='Renderer name (default: qwen3)'
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10,
        help='Number of episodes per game (default: 10)'
    )
    parser.add_argument(
        '--max_turns',
        type=int,
        default=20,
        help='Max turns per episode (default: 20)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (default: 1.0)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=1.0,
        help='Sampling top_p (default: 1.0)'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=4096,
        help='Max tokens per generation (default: 4096)'
    )
    parser.add_argument(
        '--base_url',
        type=str,
        default=None,
        help='Tinker service URL (optional)'
    )

    args = parser.parse_args()

    print('='*70)
    print(f'Evaluating: {args.model}')
    print('='*70)
    print()

    games_dir = Path(args.games_dir)
    game_files = sorted(list(games_dir.glob('*.py')))

    if not game_files:
        print(f'ERROR: No game files found in {games_dir}!')
        sys.exit(1)

    print(f'Found {len(game_files)} games to test')
    print(f'Episodes per game: {args.num_episodes}')
    print(f'Max turns per episode: {args.max_turns}')
    print(f'Temperature: {args.temperature}, Top-p: {args.top_p}')
    print()

    service_client = tinker.ServiceClient(base_url=args.base_url)

    model_name = args.model

    try:
        print('Setting up tokenizer and renderer...')
        tokenizer = get_tokenizer(model_name)
        renderer = get_renderer(args.renderer, tokenizer=tokenizer)

        print(f'Creating sampling client for {model_name}...')
        sampling_client = service_client.create_sampling_client(base_model=model_name)
        print()

        model_results = {}

        total_tasks = len(game_files) * args.num_episodes
        print(f'Running {total_tasks} episodes ({len(game_files)} games × {args.num_episodes} episodes) in parallel...')
        print()

        tasks = []
        task_metadata = []

        for game_file in game_files:
            game_name = game_file.stem
            for episode in range(args.num_episodes):
                task = test_single_episode(
                    sampling_client,
                    renderer,
                    str(game_file),
                    episode_seed=episode,
                    max_turns=args.max_turns,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                tasks.append(task)
                task_metadata.append(game_name)

        all_results = await tqdm.gather(
            *tasks,
            desc=f'Testing {model_name}',
        )

        game_results = {}
        for game_name, result in zip(task_metadata, all_results):
            if game_name not in game_results:
                game_results[game_name] = []
            game_results[game_name].append(result)

        for game_name, episode_results in game_results.items():
            aggregated = aggregate_episode_results(episode_results)
            model_results[game_name] = aggregated

            success_rate = aggregated['success'] / args.num_episodes
            avg_reward = aggregated['total_rewards'] / args.num_episodes
            print(f'  {game_name}: Success: {success_rate:.1%} | Avg reward: {avg_reward:.2f}')

        print()

    except Exception as e:
        print(f'ERROR testing {model_name}: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[model_name] = model_results

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print('='*70)
    print(f'RESULTS FOR {model_name}')
    print('='*70)
    print()

    total_success = sum(r['success'] for r in model_results.values())
    total_episodes = len(model_results) * args.num_episodes
    total_rewards = sum(r['total_rewards'] for r in model_results.values())

    success_rate = total_success / total_episodes if total_episodes > 0 else 0
    avg_reward = total_rewards / total_episodes if total_episodes > 0 else 0

    print(f"Overall Success Rate: {success_rate*100:.1f}%")
    print(f"Overall Avg Reward: {avg_reward:.2f}")
    print()
    print(f'Results saved to: {output_path}')
    print('='*70)


if __name__ == '__main__':
    asyncio.run(main())
