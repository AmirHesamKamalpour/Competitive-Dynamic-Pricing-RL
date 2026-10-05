from __future__ import annotations

from pathlib import Path

from src.config import load_experiment_config


def test_default_config_resolves() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs" / "experiment.yaml")
    assert config.algorithm == "sac"
    assert config.environment["horizon"] > 0
    assert config.agent["batch_size"] > 0
