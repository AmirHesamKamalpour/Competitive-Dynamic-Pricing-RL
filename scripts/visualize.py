from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_experiment_config
from src.evaluation.plots import plot_rollout, plot_training_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate figures from saved CSV logs.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    run_name = args.run_name or config.run_name
    log_dir = ROOT / "results" / "logs" / run_name
    figure_dir = ROOT / "results" / "figures" / run_name
    plot_training_log(log_dir / "training.csv", figure_dir / "training.png")
    plot_rollout(log_dir / "rollout.csv", figure_dir / "rollout.png")
    print(f"Saved figures to {figure_dir}")


if __name__ == "__main__":
    main()
