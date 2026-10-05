from __future__ import annotations

from typing import Any

import gymnasium as gym

from src.agents.base_agent import BaseAgent
from src.evaluation.metrics import summarize_returns


def evaluate_policy(
    env: gym.Env,
    agent: BaseAgent,
    *,
    episodes: int = 5,
    seed: int | None = None,
) -> dict[str, float]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    returns: list[float] = []
    profits: list[float] = []
    for episode in range(episodes):
        episode_seed = None if seed is None else seed + episode
        observation, _ = env.reset(seed=episode_seed)
        episode_return = 0.0
        episode_profit = 0.0
        done = False
        while not done:
            action = agent.act(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_return += float(reward)
            episode_profit += float(info.get("profit", 0.0))
        returns.append(episode_return)
        profits.append(episode_profit)

    stats = summarize_returns(returns)
    stats["profit_mean"] = float(sum(profits) / len(profits))
    return stats


def collect_diagnostic_rollout(
    env: gym.Env,
    agent: BaseAgent,
    *,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    observation, _ = env.reset(seed=seed)
    rows: list[dict[str, Any]] = []
    done = False
    step = 0
    while not done:
        action = agent.act(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        rows.append(
            {
                "step": step,
                "price": info.get("price"),
                "competitor_price": info.get("competitor_price"),
                "market_size": info.get("market_size"),
                "demand": info.get("demand"),
                "profit": info.get("profit"),
                "adjustment_cost": info.get("adjustment_cost"),
                "reward": float(reward),
            }
        )
        step += 1
    return rows
