"""Dual-role SPARE training step."""

import logging
from typing import Any, List, Sequence

import tinker
import torch
from tinker import AdamParams, TensorData
from tinker_cookbook.rl.metrics import compute_post_kl
from tinker_cookbook.rl.types import (EnvGroupBuilder, TrajectoryGroup, Transition)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils.misc_utils import timed

logger = logging.getLogger(__name__)


async def train_step(
    cfg: Any,
    i_batch: int,
    training_client: tinker.TrainingClient,
    tokenizer: Tokenizer,
    trajectory_groups_P: list[TrajectoryGroup],
    sampling_client: tinker.SamplingClient | None = None,
    env_group_builders_P: Sequence[EnvGroupBuilder] | None = None,
) -> dict[str, Any]:
    """Train on actor and environment trajectories in one optimizer step."""
    metrics = {}
    actor_transitions: List[Transition] = []
    env_transitions: List[Transition] = []

    with timed("compute_turn_advantages", metrics):
        for traj_group in trajectory_groups_P:
            total_rewards = traj_group.get_total_rewards()

            for reward, trajectory in zip(total_rewards, traj_group.trajectories_G):
                num_turns = len(trajectory.transitions)

                role = None
                if trajectory.transitions and hasattr(trajectory.transitions[0], 'info'):
                    role = trajectory.transitions[0].info.get('role')

                for turn_idx, transition in enumerate(trajectory.transitions):
                    transition.reward = reward * (cfg.gamma ** (num_turns - turn_idx - 1))

                    if hasattr(cfg, "filter_zero_adv") and cfg.filter_zero_adv and transition.reward == 0:
                        continue

                    if role == "environment":
                        env_transitions.append(transition)
                    else:
                        actor_transitions.append(transition)

    logger.info(f"Actor transitions: {len(actor_transitions)}, Env transitions: {len(env_transitions)}")

    all_transitions = actor_transitions + env_transitions
    total_transitions = len(all_transitions)

    advantage_scale = cfg.batch_size / total_transitions if total_transitions > 0 else 1.0
    logger.info(f"Advantage scale: {advantage_scale:.4f} (batch_size={cfg.batch_size}, total={total_transitions})")

    with timed("prepare_datums", metrics):
        training_datums = []
        for transition in all_transitions:
            ob_tokens = transition.ob.to_ints()
            ac_tokens = transition.ac.tokens
            ob_len = len(ob_tokens) - 1
            tokens = ob_tokens + ac_tokens
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]

            all_logprobs = [0.0] * ob_len + transition.ac.logprobs
            assert len(all_logprobs) == len(input_tokens), \
                f"Logprobs length mismatch: {len(all_logprobs)} vs {len(input_tokens)}"

            advantages = [0.0] * ob_len + [transition.reward * advantage_scale] * len(ac_tokens)

            training_datums.append(
                tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs, dtype=torch.float32)),
                        "advantages": TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
                    },
                )
            )

    logger.info(f"Created {len(training_datums)} training datums")

    adam_params = AdamParams(
        learning_rate=cfg.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-12,
    )

    with timed("forward_backward", metrics):
        fwd_bwd_result = await training_client.forward_backward_async(
            training_datums,
            loss_fn=cfg.loss_fn,
        )
        fwd_bwd_result = await fwd_bwd_result.result_async()

    with timed("optim_step", metrics):
        optim_result = await training_client.optim_step_async(adam_params)
        optim_result = await optim_result.result_async()

    if sampling_client and getattr(cfg, 'compute_post_kl', False):
        with timed("compute_post_kl", metrics):
            metrics["train/post_kl"] = await compute_post_kl(
                tokenizer, training_datums, sampling_client, cfg.batch_size
            )

    metrics.update({
        f"train/{k}": v for k, v in fwd_bwd_result.metrics.items()
    })
    metrics["train/num_actor_transitions"] = len(actor_transitions)
    metrics["train/num_env_transitions"] = len(env_transitions)
    metrics["train/total_transitions"] = total_transitions
    metrics["train/num_datums"] = len(training_datums)
    metrics["train/advantage_scale"] = advantage_scale

    if total_transitions > 0:
        metrics["train/env_transition_ratio"] = len(env_transitions) / total_transitions

    logger.info(f"Training complete: "
               f"actor_trans={len(actor_transitions)}, env_trans={len(env_transitions)}, "
               f"loss={fwd_bwd_result.metrics.get('loss:sum', 0.0):.4f}, "
               f"metrics={', '.join([f'{k}: {v:.4f}' for k, v in fwd_bwd_result.metrics.items()])}")

    return metrics
