from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from src.agents.base_agent import BaseAgent


@dataclass(frozen=True, slots=True)
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def episode_done(self) -> bool:
        return self.terminated or self.truncated


def collect_training_transition(
    env: gym.Env,
    agent: BaseAgent,
    observation: np.ndarray,
    *,
    random_action: bool,
) -> Transition:
    action = env.action_space.sample() if random_action else agent.act(observation, deterministic=False)
    next_observation, reward, terminated, truncated, info = env.step(action)
    return Transition(
        observation=np.asarray(observation, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        reward=float(reward),
        next_observation=np.asarray(next_observation, dtype=np.float32),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=dict(info),
    )
