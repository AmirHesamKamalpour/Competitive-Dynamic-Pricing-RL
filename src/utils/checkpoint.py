from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from src.agents.base_agent import BaseAgent


def save_checkpoint(
    agent: BaseAgent,
    path: str | Path,
    *,
    step: int,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "agent": agent.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)


def load_checkpoint(agent: BaseAgent, path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "agent" not in payload:
        raise ValueError(f"Invalid checkpoint: {path}")
    agent.load_state_dict(payload["agent"])
    return dict(payload)
