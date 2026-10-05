from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_experiment_config
from src.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SAC pricing agent.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override project.seed")
    parser.add_argument("--run-name", type=str, default=None, help="Override project.run_name")
    parser.add_argument("--total-steps", type=int, default=None, help="Override training.total_steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.total_steps is not None:
        config.training["total_steps"] = args.total_steps

    paths = run_experiment(config)
    print(f"Final checkpoint: {paths.final_checkpoint}")
    print(f"Training log: {paths.training_csv}")


if __name__ == "__main__":
    main()
