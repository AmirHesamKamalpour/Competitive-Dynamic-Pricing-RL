from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def build_mlp(
    sizes: Sequence[int],
    *,
    activation: type[nn.Module] = nn.ReLU,
    output_activation: type[nn.Module] = nn.Identity,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layer_activation = activation if index < len(sizes) - 2 else output_activation
        layers.extend((nn.Linear(sizes[index], sizes[index + 1]), layer_activation()))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")
        self.backbone = build_mlp(
            [obs_dim, *hidden_sizes],
            activation=nn.ReLU,
            output_activation=nn.ReLU,
        )
        self.mean = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std = nn.Linear(hidden_sizes[-1], act_dim)

    def distribution_parameters(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a reparameterized action in normalized ``[-1, 1]`` coordinates."""

        mean, log_std = self.distribution_parameters(obs)
        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        pre_tanh = distribution.rsample()
        normalized_action = torch.tanh(pre_tanh)

        log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1.0 - normalized_action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return normalized_action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution_parameters(obs)
        return torch.tanh(mean)


class QCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.network = build_mlp([obs_dim + act_dim, *hidden_sizes, 1], activation=nn.ReLU)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((obs, action), dim=-1))
