from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from src.agents.sac_agent import SACAgent, SACConfig
from src.config import ExperimentConfig
from src.environments.custom_env import CompetitivePricingEnv
from src.environments.wrappers import RewardScale
from src.training.trainer import TrainConfig, train_sac
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import RunPaths


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_device(value: str | None) -> torch.device:
    requested = (value or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_environment(config: ExperimentConfig) -> gym.Env:
    env = CompetitivePricingEnv(**config.environment)
    return RewardScale(
        env,
        scale=float(config.reward.get("scale", 1.0)),
        clip=config.reward.get("clip"),
    )


def build_agent(config: ExperimentConfig, env: gym.Env) -> SACAgent:
    if config.algorithm != "sac":
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")
    agent_cfg = SACConfig.from_mapping(config.agent)
    device = resolve_device(config.agent.get("device"))
    return SACAgent(
        obs_dim=int(env.observation_space.shape[0]),
        act_dim=int(env.action_space.shape[0]),
        act_low=env.action_space.low,
        act_high=env.action_space.high,
        config=agent_cfg,
        device=device,
    )


def run_experiment(config: ExperimentConfig) -> RunPaths:
    env = build_environment(config)
    eval_env = build_environment(config)
    agent = build_agent(config, env)
    train_config = TrainConfig.from_mapping(config.training)

    root = project_root()
    results_dir = Path(config.output.get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    paths = RunPaths.create(results_dir, config.run_name)

    metadata: dict[str, Any] = {
        "algorithm": config.algorithm,
        "environment": config.environment,
        "reward": config.reward,
        "device": str(agent.device),
        "source_config": str(config.source_path),
    }
    return train_sac(
        env,
        eval_env,
        agent,
        config=train_config,
        paths=paths,
        seed=config.seed,
        run_metadata=metadata,
    )


def load_agent_for_evaluation(
    config: ExperimentConfig,
    checkpoint: str | Path,
) -> tuple[gym.Env, SACAgent, dict[str, Any]]:
    env = build_environment(config)
    agent = build_agent(config, env)
    payload = load_checkpoint(agent, checkpoint)
    return env, agent, payload
