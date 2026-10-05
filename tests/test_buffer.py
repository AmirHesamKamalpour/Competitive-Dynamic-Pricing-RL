from __future__ import annotations

import numpy as np
import torch

from src.buffers.replay_buffer import ReplayBuffer


def test_replay_buffer_shapes_and_capacity() -> None:
    buffer = ReplayBuffer(obs_dim=3, act_dim=1, capacity=4, device=torch.device("cpu"), seed=7)
    for index in range(6):
        obs = np.full(3, index, dtype=np.float32)
        buffer.add(obs, np.array([index], dtype=np.float32), float(index), obs + 1, terminated=False)

    assert len(buffer) == 4
    batch = buffer.sample(3)
    assert batch["obs"].shape == (3, 3)
    assert batch["action"].shape == (3, 1)
    assert batch["reward"].shape == (3, 1)
    assert batch["terminated"].shape == (3, 1)
