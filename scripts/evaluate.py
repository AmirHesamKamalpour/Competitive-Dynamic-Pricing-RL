from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_experiment_config
from src.evaluation.evaluator import evaluate_policy
from src.experiment import load_agent_for_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved SAC checkpoint.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    checkpoint = args.checkpoint or (
        ROOT / "results" / "checkpoints" / config.run_name / "agent_final.pt"
    )
    env, agent, payload = load_agent_for_evaluation(config, checkpoint)
    stats = evaluate_policy(env, agent, episodes=args.episodes, seed=config.seed + 40_000)
    print(f"Loaded checkpoint at step {payload.get('step', 'unknown')}")
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
