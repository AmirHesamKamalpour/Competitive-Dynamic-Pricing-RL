from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np
import torch


class BaseAgent(ABC):
    """Minimal interface shared by agents used by trainers/evaluators."""

    @abstractmethod
    def act(self, observation: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def update(self, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        raise NotImplementedError
