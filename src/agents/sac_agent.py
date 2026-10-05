from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.training.losses import actor_loss, critic_loss, entropy_temperature_loss

from .base_agent import BaseAgent
from .networks import QCritic, SquashedGaussianActor


@dataclass(slots=True)
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    batch_size: int = 256
    replay_size: int = 1_000_000
    hidden_sizes: tuple[int, ...] = (256, 256)
    auto_entropy_tuning: bool = True
    initial_alpha: float = 0.2
    target_entropy: float | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SACConfig":
        payload = dict(data)
        payload.pop("device", None)
        if "hidden_sizes" in payload:
            payload["hidden_sizes"] = tuple(int(value) for value in payload["hidden_sizes"])
        return cls(**payload)


class SACAgent(BaseAgent):
    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        config: SACConfig,
        device: torch.device,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.config = config
        self.device = device

        self.action_low = torch.as_tensor(act_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(act_high, dtype=torch.float32, device=device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0
        if torch.any(self.action_scale <= 0):
            raise ValueError("Action bounds must have positive width.")

        hidden_sizes: Sequence[int] = config.hidden_sizes
        self.actor = SquashedGaussianActor(self.obs_dim, self.act_dim, hidden_sizes).to(device)
        self.q1 = QCritic(self.obs_dim, self.act_dim, hidden_sizes).to(device)
        self.q2 = QCritic(self.obs_dim, self.act_dim, hidden_sizes).to(device)
        self.target_q1 = QCritic(self.obs_dim, self.act_dim, hidden_sizes).to(device)
        self.target_q2 = QCritic(self.obs_dim, self.act_dim, hidden_sizes).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            [*self.q1.parameters(), *self.q2.parameters()],
            lr=config.learning_rate,
        )

        self.target_entropy = (
            -float(self.act_dim) if config.target_entropy is None else float(config.target_entropy)
        )
        if config.auto_entropy_tuning:
            self.log_alpha = torch.tensor(
                np.log(config.initial_alpha),
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            self.alpha_optimizer: torch.optim.Optimizer | None = torch.optim.Adam(
                [self.log_alpha], lr=config.learning_rate
            )
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self._fixed_alpha = torch.tensor(config.initial_alpha, dtype=torch.float32, device=device)

    def _scale_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        return normalized_action * self.action_scale + self.action_bias

    def alpha(self) -> torch.Tensor:
        if self.config.auto_entropy_tuning:
            assert self.log_alpha is not None
            return self.log_alpha.exp()
        return self._fixed_alpha

    @torch.no_grad()
    def act(self, observation: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            normalized_action = self.actor.deterministic(obs)
        else:
            normalized_action, _ = self.actor.sample(obs)
        action = self._scale_action(normalized_action)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        terminated = batch["terminated"]

        with torch.no_grad():
            next_normalized_action, next_log_prob = self.actor.sample(next_obs)
            next_action = self._scale_action(next_normalized_action)
            next_q = torch.minimum(
                self.target_q1(next_obs, next_action),
                self.target_q2(next_obs, next_action),
            )
            target = reward + self.config.gamma * (1.0 - terminated) * (
                next_q - self.alpha().detach() * next_log_prob
            )

        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        q_loss = critic_loss(q1, q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.critic_optimizer.step()

        for parameter in [*self.q1.parameters(), *self.q2.parameters()]:
            parameter.requires_grad_(False)
        normalized_action, log_prob = self.actor.sample(obs)
        policy_action = self._scale_action(normalized_action)
        min_q = torch.minimum(self.q1(obs, policy_action), self.q2(obs, policy_action))
        pi_loss = actor_loss(log_prob, min_q, self.alpha().detach())
        self.actor_optimizer.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.actor_optimizer.step()
        for parameter in [*self.q1.parameters(), *self.q2.parameters()]:
            parameter.requires_grad_(True)

        alpha_loss_value = float("nan")
        if self.config.auto_entropy_tuning:
            assert self.log_alpha is not None and self.alpha_optimizer is not None
            alpha_loss = entropy_temperature_loss(self.log_alpha, log_prob, self.target_entropy)
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.item())

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

        return {
            "q_loss": float(q_loss.item()),
            "pi_loss": float(pi_loss.item()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha().detach().item()),
        }

    @torch.no_grad()
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.config.tau
        for source_param, target_param in zip(source.parameters(), target.parameters(), strict=True):
            target_param.lerp_(source_param, tau)

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        if self.config.auto_entropy_tuning:
            assert self.log_alpha is not None and self.alpha_optimizer is not None
            state["log_alpha"] = self.log_alpha.detach().clone()
            state["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.target_q1.load_state_dict(state["target_q1"])
        self.target_q2.load_state_dict(state["target_q2"])
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if self.config.auto_entropy_tuning and "log_alpha" in state:
            assert self.log_alpha is not None
            with torch.no_grad():
                self.log_alpha.copy_(state["log_alpha"].to(self.device))
            if self.alpha_optimizer is not None and "alpha_optimizer" in state:
                self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
