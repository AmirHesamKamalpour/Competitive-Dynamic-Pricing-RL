from __future__ import annotations

import numpy as np


def summarize_returns(returns: list[float]) -> dict[str, float]:
    values = np.asarray(returns, dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one return is required.")
    return {
        "return_mean": float(values.mean()),
        "return_std": float(values.std()),
        "return_min": float(values.min()),
        "return_max": float(values.max()),
    }
