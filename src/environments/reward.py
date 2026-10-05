from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PricingReward:
    reward: float
    profit: float
    adjustment_cost: float


def compute_pricing_reward(
    *,
    price: float,
    unit_cost: float,
    demand: float,
    previous_price: float,
    adjustment_penalty: float,
) -> PricingReward:
    """Return profit minus a penalty for changing price between periods."""

    profit = (price - unit_cost) * demand
    adjustment_cost = adjustment_penalty * abs(price - previous_price)
    return PricingReward(
        reward=float(profit - adjustment_cost),
        profit=float(profit),
        adjustment_cost=float(adjustment_cost),
    )
