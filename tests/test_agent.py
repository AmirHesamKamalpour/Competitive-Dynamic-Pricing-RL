from __future__ import annotations

import numpy as np
import torch

from src.agents.sac_agent import SACAgent, SACConfig


def make_agent() -> SACAgent:
    return SACAgent(
        obs_dim=4,
        act_dim=1,
        act_low=np.array([1.0], dtype=np.float32),
        act_high=np.array([5.0], dtype=np.float32),
        config=SACConfig(batch_size=8, replay_size=64, hidden_sizes=(32, 32)),
        device=torch.device("cpu"),
    )


def test_action_respects_environment_bounds() -> None:
    agent = make_agent()
    action = agent.act(np.zeros(4, dtype=np.float32), deterministic=True)
    assert action.shape == (1,)
    assert 1.0 <= action[0] <= 5.0


def test_update_returns_finite_metrics() -> None:
    torch.manual_seed(0)
    agent = make_agent()
    batch_size = 8
    batch = {
        "obs": torch.randn(batch_size, 4),
        "action": 1.0 + 4.0 * torch.rand(batch_size, 1),
        "reward": torch.randn(batch_size, 1),
        "next_obs": torch.randn(batch_size, 4),
        "terminated": torch.zeros(batch_size, 1),
    }
    metrics = agent.update(batch)
    assert {"q_loss", "pi_loss", "alpha", "alpha_loss"} <= metrics.keys()
    assert all(np.isfinite(value) for value in metrics.values())
