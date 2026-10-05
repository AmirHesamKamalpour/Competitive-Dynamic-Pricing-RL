from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size replay buffer with an isolated NumPy RNG.

    ``terminated`` is stored separately from episode truncation so SAC can still
    bootstrap through exogenous time limits.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        capacity: int,
        device: torch.device,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, act_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.float32)
        self.capacity = int(capacity)
        self.device = device
        self._rng = np.random.default_rng(seed)
        self._cursor = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        *,
        terminated: bool,
    ) -> None:
        index = self._cursor
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index, 0] = float(reward)
        self.next_observations[index] = next_observation
        self.terminated[index, 0] = float(terminated)

        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = self._rng.integers(0, self._size, size=int(batch_size))
        return {
            "obs": torch.as_tensor(self.observations[indices], device=self.device),
            "action": torch.as_tensor(self.actions[indices], device=self.device),
            "reward": torch.as_tensor(self.rewards[indices], device=self.device),
            "next_obs": torch.as_tensor(self.next_observations[indices], device=self.device),
            "terminated": torch.as_tensor(self.terminated[indices], device=self.device),
        }
