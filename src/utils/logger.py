from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


@dataclass(frozen=True, slots=True)
class RunPaths:
    results_root: Path
    run_name: str
    checkpoint_dir: Path
    log_dir: Path
    figure_dir: Path

    @classmethod
    def create(cls, results_root: Path, run_name: str) -> "RunPaths":
        results_root = Path(results_root)
        checkpoint_dir = results_root / "checkpoints" / run_name
        log_dir = results_root / "logs" / run_name
        figure_dir = results_root / "figures" / run_name
        for path in (checkpoint_dir, log_dir, figure_dir):
            path.mkdir(parents=True, exist_ok=True)
        return cls(results_root, run_name, checkpoint_dir, log_dir, figure_dir)

    @property
    def training_csv(self) -> Path:
        return self.log_dir / "training.csv"

    @property
    def rollout_csv(self) -> Path:
        return self.log_dir / "rollout.csv"

    @property
    def run_config_json(self) -> Path:
        return self.log_dir / "run_config.json"

    @property
    def final_checkpoint(self) -> Path:
        return self.checkpoint_dir / "agent_final.pt"


def save_records_csv(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(records)).to_csv(path, index=False)


def save_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
