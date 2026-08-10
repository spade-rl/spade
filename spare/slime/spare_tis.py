"""Custom TIS function for SPARE delayed proposer training.

Vanilla TIS (clipped importance weighting) without env-specific metrics.
Loaded via:
    --custom-tis-function-path spare.slime.spare_tis.spare_delayed_tis
"""

from typing import Any

import torch


def spare_delayed_tis(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """Vanilla TIS with clipped importance weighting.

    Args:
        args: Slime args namespace (must have ``tis_clip_low``, ``tis_clip``).
        pg_loss: Per-token policy gradient loss (1-D, concatenated).
        train_log_probs: Per-sample list of train log-prob tensors.
        rollout_log_probs: Per-sample list of rollout log-prob tensors.
        loss_masks: Per-sample list of loss mask tensors.
        **kwargs: Extra keys (ignored).

    Returns:
        Tuple of (modified pg_loss, loss_masks, metrics dict).
    """
    cat_rollout = torch.cat(rollout_log_probs, dim=0)
    cat_train = torch.cat(train_log_probs, dim=0)

    tis = torch.exp(cat_train - cat_rollout)
    tis_abs = (tis - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()

    pg_loss = pg_loss * tis_weights

    metrics: dict[str, torch.Tensor] = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }

    return pg_loss, loss_masks, metrics
