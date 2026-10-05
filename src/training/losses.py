from __future__ import annotations

import torch
import torch.nn.functional as F


def critic_loss(q1: torch.Tensor, q2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(q1, target) + F.mse_loss(q2, target)


def actor_loss(log_prob: torch.Tensor, min_q: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return (alpha * log_prob - min_q).mean()


def entropy_temperature_loss(
    log_alpha: torch.Tensor,
    log_prob: torch.Tensor,
    target_entropy: float,
) -> torch.Tensor:
    return -(log_alpha * (log_prob + target_entropy).detach()).mean()
