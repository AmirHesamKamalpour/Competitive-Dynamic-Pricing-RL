from __future__ import annotations

import gymnasium as gym
import numpy as np


class RewardScale(gym.RewardWrapper):
    """Scale rewards and optionally clip the scaled value."""

    def __init__(self, env: gym.Env, *, scale: float = 1.0, clip: float | None = None) -> None:
        super().__init__(env)
        if scale <= 0:
            raise ValueError("scale must be positive")
        if clip is not None and clip <= 0:
            raise ValueError("clip must be positive when provided")
        self.scale = float(scale)
        self.clip = None if clip is None else float(clip)

    def reward(self, reward: float) -> float:
        scaled = float(reward) * self.scale
        if self.clip is not None:
            scaled = float(np.clip(scaled, -self.clip, self.clip))
        return scaled
