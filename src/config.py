from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    """Resolved experiment configuration assembled from the YAML files."""

    source_path: Path
    algorithm: str
    seed: int
    run_name: str
    environment: dict[str, Any]
    reward: dict[str, Any]
    agent: dict[str, Any]
    training: dict[str, Any]
    output: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top level of {path}.")
    return data


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load the experiment file and resolve its environment/agent config references."""

    path = Path(path).expanduser().resolve()
    experiment = _load_yaml(path)

    project = experiment.get("project", {})
    refs = experiment.get("configs", {})
    if not isinstance(project, dict) or not isinstance(refs, dict):
        raise ValueError("'project' and 'configs' must be YAML mappings.")

    env_name = refs.get("environment")
    agent_name = refs.get("agent")
    if not env_name or not agent_name:
        raise ValueError("Experiment config must reference environment and agent YAML files.")

    env_data = _load_yaml(path.parent / str(env_name))
    agent_data = _load_yaml(path.parent / str(agent_name))

    algorithm = str(project.get("algorithm", "sac")).lower()
    if algorithm != "sac":
        raise ValueError(f"Unsupported algorithm {algorithm!r}; this project currently implements SAC.")

    return ExperimentConfig(
        source_path=path,
        algorithm=algorithm,
        seed=int(project.get("seed", 0)),
        run_name=str(project.get("run_name", "sac_pricing")),
        environment=dict(env_data.get("environment", {})),
        reward=dict(env_data.get("reward", {})),
        agent=dict(agent_data.get("agent", {})),
        training=dict(experiment.get("training", {})),
        output=dict(experiment.get("output", {})),
    )
