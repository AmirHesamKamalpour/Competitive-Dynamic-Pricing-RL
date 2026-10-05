from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .reward import compute_pricing_reward


class CompetitivePricingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-product dynamic pricing environment with a reactive competitor.

    Observation
    -----------
    Concatenated history of agent prices, competitor prices and demand. Optional
    sine/cosine seasonality features can be appended.

    Action
    ------
    A one-dimensional continuous price bounded by ``[p_min, p_max]``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        horizon: int = 1000,
        k: int = 10,
        m: int = 1,
        p_min: float = 1.0,
        p_max: float = 20.0,
        p_min_comp: float | None = None,
        p_max_comp: float | None = None,
        unit_cost: float = 8.0,
        adjustment_penalty: float = 0.1,
        alpha0: float = 50.0,
        alpha_growth: float = 5.0,
        alpha_seasonality: float = 0.1,
        alpha_period: int = 52,
        alpha_phase: float = 0.0,
        beta: float = 2.0,
        eta: float = 1.0,
        demand_noise_std: float = 2.0,
        competitor_intercept: float = 5.0,
        competitor_own_price_weight: float = 0.2,
        competitor_lag_weight: float = 0.7,
        competitor_noise_std: float = 0.5,
        include_time_features: bool = True,
        initial_price: float | None = None,
        initial_competitor_price: float | None = None,
    ) -> None:
        super().__init__()

        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        if not 1 <= m <= k:
            raise ValueError("m must satisfy 1 <= m <= k")
        if p_min >= p_max:
            raise ValueError("p_min must be strictly less than p_max")
        if alpha_period <= 0:
            raise ValueError("alpha_period must be positive")
        if demand_noise_std < 0 or competitor_noise_std < 0:
            raise ValueError("noise standard deviations must be non-negative")
        if adjustment_penalty < 0:
            raise ValueError("adjustment_penalty must be non-negative")

        self.horizon = int(horizon)
        self.k = int(k)
        self.m = int(m)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.p_min_comp = float(p_min if p_min_comp is None else p_min_comp)
        self.p_max_comp = float(p_max if p_max_comp is None else p_max_comp)
        self.unit_cost = float(unit_cost)
        self.adjustment_penalty = float(adjustment_penalty)

        self.alpha0 = float(alpha0)
        self.alpha_growth = float(alpha_growth)
        self.alpha_seasonality = float(alpha_seasonality)
        self.alpha_period = int(alpha_period)
        self.alpha_phase = float(alpha_phase)
        self.beta = float(beta)
        self.eta = float(eta)
        self.demand_noise_std = float(demand_noise_std)

        self.competitor_intercept = float(competitor_intercept)
        self.competitor_noise_std = float(competitor_noise_std)
        self.competitor_own_price_weights = np.full(m, competitor_own_price_weight, dtype=np.float32)
        self.competitor_lag_weights = np.full(m, competitor_lag_weight, dtype=np.float32)

        self.include_time_features = bool(include_time_features)
        midpoint = 0.5 * (self.p_min + self.p_max)
        self.initial_price = float(midpoint if initial_price is None else initial_price)
        self.initial_competitor_price = float(
            midpoint if initial_competitor_price is None else initial_competitor_price
        )

        self.action_space = spaces.Box(
            low=np.array([self.p_min], dtype=np.float32),
            high=np.array([self.p_max], dtype=np.float32),
            dtype=np.float32,
        )
        obs_dim = 3 * self.k + (2 if self.include_time_features else 0)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._t = 0
        self._price_history = np.zeros(self.k, dtype=np.float32)
        self._competitor_history = np.zeros(self.k, dtype=np.float32)
        self._demand_history = np.zeros(self.k, dtype=np.float32)

    def _season_angle(self, t: int) -> float:
        return float(2.0 * np.pi * (t / self.alpha_period) + self.alpha_phase)

    def _market_size(self, t: int) -> float:
        trend = np.log1p(t / self.alpha_period)
        baseline = self.alpha0 + self.alpha_growth * trend
        seasonal_multiplier = 1.0 + self.alpha_seasonality * np.sin(self._season_angle(t))
        return float(max(0.0, baseline * seasonal_multiplier))

    def _time_features(self, t: int) -> np.ndarray:
        angle = self._season_angle(t)
        return np.asarray([np.sin(angle), np.cos(angle)], dtype=np.float32)

    def _sample_competitor_price(self) -> float:
        agent_lags = self._price_history[-self.m :][::-1]
        competitor_lags = self._competitor_history[-self.m :][::-1]
        mean = (
            self.competitor_intercept
            + float(np.dot(self.competitor_own_price_weights, agent_lags))
            + float(np.dot(self.competitor_lag_weights, competitor_lags))
        )
        sampled = mean + float(self.np_random.normal(0.0, self.competitor_noise_std))
        return float(np.clip(sampled, self.p_min_comp, self.p_max_comp))

    def _sample_demand(self, price: float, competitor_price: float, market_size: float) -> float:
        noise = float(self.np_random.normal(0.0, self.demand_noise_std))
        demand = market_size - self.beta * price + self.eta * (competitor_price - price) + noise
        return float(max(0.0, demand))

    def _observation(self) -> np.ndarray:
        parts: list[np.ndarray] = [
            self._price_history,
            self._competitor_history,
            self._demand_history,
        ]
        if self.include_time_features:
            parts.append(self._time_features(self._t))
        return np.concatenate(parts).astype(np.float32, copy=False)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self._t = 0
        price = self.initial_price
        competitor_price = self.initial_competitor_price
        demand = self._sample_demand(price, competitor_price, self._market_size(0))

        self._price_history.fill(price)
        self._competitor_history.fill(competitor_price)
        self._demand_history.fill(demand)
        return self._observation(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != 1:
            raise ValueError(f"Expected a single price action, got shape {np.asarray(action).shape}.")

        current_t = self._t
        price = float(np.clip(action_array[0], self.p_min, self.p_max))
        competitor_price = self._sample_competitor_price()
        market_size = self._market_size(current_t)
        demand = self._sample_demand(price, competitor_price, market_size)
        previous_price = float(self._price_history[-1])

        reward_terms = compute_pricing_reward(
            price=price,
            unit_cost=self.unit_cost,
            demand=demand,
            previous_price=previous_price,
            adjustment_penalty=self.adjustment_penalty,
        )

        self._price_history = np.roll(self._price_history, -1)
        self._competitor_history = np.roll(self._competitor_history, -1)
        self._demand_history = np.roll(self._demand_history, -1)
        self._price_history[-1] = price
        self._competitor_history[-1] = competitor_price
        self._demand_history[-1] = demand
        self._t += 1

        terminated = False
        truncated = self._t >= self.horizon
        info = {
            "profit": reward_terms.profit,
            "adjustment_cost": reward_terms.adjustment_cost,
            "price": price,
            "competitor_price": competitor_price,
            "market_size": market_size,
            "demand": demand,
            "raw_reward": reward_terms.reward,
        }
        return self._observation(), reward_terms.reward, terminated, bool(truncated), info
