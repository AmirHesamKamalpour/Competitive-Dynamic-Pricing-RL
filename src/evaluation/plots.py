from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_log(log_csv: Path, output_path: Path) -> None:
    frame = pd.read_csv(log_csv)
    if frame.empty:
        return

    steps = frame["step"].to_numpy()
    returns = frame["eval_return_mean"].to_numpy()
    smooth = pd.Series(returns).rolling(5, min_periods=1).mean()

    figure, axes = plt.subplots(3, 1, figsize=(9, 12))
    axes[0].plot(steps, returns, label="Evaluation return")
    axes[0].plot(steps, smooth, linewidth=2, label="5-point moving average")
    axes[0].set_title("Learning curve")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Scaled return")
    axes[0].legend()

    axes[1].plot(steps, frame["alpha"].to_numpy())
    axes[1].set_title("Entropy temperature")
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("alpha")

    axes[2].plot(steps, frame["q_loss"].to_numpy(), label="Critic loss")
    axes[2].plot(steps, frame["pi_loss"].to_numpy(), label="Actor loss")
    axes[2].set_title("Optimization losses")
    axes[2].set_xlabel("Environment steps")
    axes[2].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_rollout(log_csv: Path, output_path: Path) -> None:
    frame = pd.read_csv(log_csv)
    if frame.empty:
        return

    steps = frame["step"].to_numpy()
    figure, axes = plt.subplots(2, 1, figsize=(9, 8))

    axes[0].plot(steps, frame["price"].to_numpy(), label="Agent price")
    axes[0].plot(steps, frame["competitor_price"].to_numpy(), label="Competitor price")
    axes[0].set_title("Price dynamics")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Price")
    axes[0].legend()

    axes[1].plot(steps, frame["demand"].to_numpy(), label="Demand")
    if "market_size" in frame:
        axes[1].plot(steps, frame["market_size"].to_numpy(), label="Market size", linestyle="--")
    axes[1].set_title("Demand dynamics")
    axes[1].set_xlabel("Step")
    axes[1].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
