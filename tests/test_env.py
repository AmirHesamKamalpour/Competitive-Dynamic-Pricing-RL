from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from src.environments.custom_env import CompetitivePricingEnv


def test_environment_passes_gym_checker() -> None:
    env = CompetitivePricingEnv(horizon=5, k=3, m=1)
    gym.utils.env_checker.check_env(env, skip_render_check=True)


def test_truncates_at_horizon() -> None:
    env = CompetitivePricingEnv(horizon=3, k=2, m=1)
    observation, _ = env.reset(seed=123)
    assert observation.shape == env.observation_space.shape

    truncated = False
    for _ in range(3):
        observation, reward, terminated, truncated, info = env.step(np.array([10.0], dtype=np.float32))
        assert np.isfinite(reward)
        assert not terminated
        assert "profit" in info
    assert truncated
