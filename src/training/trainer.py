from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import gymnasium as gym

from src.agents.sac_agent import SACAgent
from src.buffers.replay_buffer import ReplayBuffer
from src.evaluation.evaluator import collect_diagnostic_rollout, evaluate_policy
from src.evaluation.plots import plot_rollout, plot_training_log
from src.training.collector import collect_training_transition
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import RunPaths, save_json, save_records_csv
from src.utils.seeding import seed_environment, set_global_seed


@dataclass(slots=True)
class TrainConfig:
    total_steps: int = 300_000
    start_steps: int = 10_000
    update_after: int = 2_000
    update_every: int = 1
    updates_per_step: int = 1
    eval_every: int = 25_000
    eval_episodes: int = 5
    checkpoint_every: int = 25_000

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrainConfig":
        return cls(**dict(data))

    def validate(self) -> None:
        for name in (
            "total_steps",
            "update_every",
            "updates_per_step",
            "eval_every",
            "eval_episodes",
            "checkpoint_every",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.start_steps < 0 or self.update_after < 0:
            raise ValueError("start_steps and update_after cannot be negative")


def train_sac(
    env: gym.Env,
    eval_env: gym.Env,
    agent: SACAgent,
    *,
    config: TrainConfig,
    paths: RunPaths,
    seed: int,
    run_metadata: Mapping[str, Any] | None = None,
) -> RunPaths:
    config.validate()
    set_global_seed(seed)
    seed_environment(env, seed)
    seed_environment(eval_env, seed + 10_000)

    replay = ReplayBuffer(
        obs_dim=agent.obs_dim,
        act_dim=agent.act_dim,
        capacity=agent.config.replay_size,
        device=agent.device,
        seed=seed,
    )

    observation, _ = env.reset(seed=seed)
    latest_update: dict[str, float] = {}
    training_rows: list[dict[str, float | int]] = []
    start_time = time.perf_counter()

    for step in range(1, config.total_steps + 1):
        transition = collect_training_transition(
            env,
            agent,
            observation,
            random_action=step <= config.start_steps,
        )
        replay.add(
            transition.observation,
            transition.action,
            transition.reward,
            transition.next_observation,
            terminated=transition.terminated,
        )
        observation = transition.next_observation
        if transition.episode_done:
            observation, _ = env.reset()

        if (
            step >= config.update_after
            and len(replay) >= agent.config.batch_size
            and step % config.update_every == 0
        ):
            for _ in range(config.updates_per_step):
                latest_update = agent.update(replay.sample(agent.config.batch_size))

        if step % config.eval_every == 0:
            stats = evaluate_policy(
                eval_env,
                agent,
                episodes=config.eval_episodes,
                seed=seed + 20_000,
            )
            row: dict[str, float | int] = {
                "step": step,
                "eval_return_mean": stats["return_mean"],
                "eval_return_std": stats["return_std"],
                "eval_profit_mean": stats["profit_mean"],
                "alpha": latest_update.get("alpha", float("nan")),
                "q_loss": latest_update.get("q_loss", float("nan")),
                "pi_loss": latest_update.get("pi_loss", float("nan")),
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            training_rows.append(row)
            print(
                f"[step={step}] return={stats['return_mean']:.3f}±{stats['return_std']:.3f} "
                f"profit={stats['profit_mean']:.2f} alpha={row['alpha']:.4f}"
            )

        if step % config.checkpoint_every == 0:
            save_checkpoint(
                agent,
                paths.checkpoint_dir / f"agent_step_{step}.pt",
                step=step,
                metadata={"seed": seed, "run_name": paths.run_name},
            )

    if not training_rows or training_rows[-1]["step"] != config.total_steps:
        stats = evaluate_policy(
            eval_env,
            agent,
            episodes=config.eval_episodes,
            seed=seed + 20_000,
        )
        training_rows.append(
            {
                "step": config.total_steps,
                "eval_return_mean": stats["return_mean"],
                "eval_return_std": stats["return_std"],
                "eval_profit_mean": stats["profit_mean"],
                "alpha": latest_update.get("alpha", float("nan")),
                "q_loss": latest_update.get("q_loss", float("nan")),
                "pi_loss": latest_update.get("pi_loss", float("nan")),
                "elapsed_seconds": time.perf_counter() - start_time,
            }
        )

    save_checkpoint(
        agent,
        paths.final_checkpoint,
        step=config.total_steps,
        metadata={"seed": seed, "run_name": paths.run_name},
    )
    save_records_csv(training_rows, paths.training_csv)

    rollout_rows = collect_diagnostic_rollout(eval_env, agent, seed=seed + 30_000)
    save_records_csv(rollout_rows, paths.rollout_csv)

    snapshot = {
        "seed": seed,
        "run_name": paths.run_name,
        "training": asdict(config),
        "agent": asdict(agent.config),
        **dict(run_metadata or {}),
    }
    save_json(snapshot, paths.run_config_json)

    plot_training_log(paths.training_csv, paths.figure_dir / "training.png")
    plot_rollout(paths.rollout_csv, paths.figure_dir / "rollout.png")
    return paths
