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
    parser = argparse.ArgumentParser(description="Run a simple multi-seed SAC sweep.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_experiment_config(args.config)
    base_name = base.run_name.removesuffix(f"_seed{base.seed}")
    for seed in args.seeds:
        config = load_experiment_config(args.config)
        config.seed = seed
        config.run_name = f"{base_name}_seed{seed}"
        run_experiment(config)


if __name__ == "__main__":
    main()
