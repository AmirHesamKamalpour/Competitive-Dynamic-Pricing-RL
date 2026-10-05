from __future__ import annotations

import numpy as np
import torch

from src.agents.sac_agent import SACAgent, SACConfig
from src.utils.checkpoint import load_checkpoint, save_checkpoint


def _agent() -> SACAgent:
    return SACAgent(
        obs_dim=3,
        act_dim=1,
        act_low=np.array([1.0], dtype=np.float32),
        act_high=np.array([4.0], dtype=np.float32),
        config=SACConfig(hidden_sizes=(16, 16), batch_size=4, replay_size=32),
        device=torch.device("cpu"),
    )


def test_checkpoint_roundtrip(tmp_path) -> None:
    agent = _agent()
    obs = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    before = agent.act(obs, deterministic=True)

    path = tmp_path / "agent.pt"
    save_checkpoint(agent, path, step=12, metadata={"name": "test"})

    restored = _agent()
    payload = load_checkpoint(restored, path)
    after = restored.act(obs, deterministic=True)

    assert payload["step"] == 12
    np.testing.assert_allclose(before, after, rtol=0, atol=1e-6)
