"""Train the PPO signal controller.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\train_rl.py --timesteps 150000
    .\\.venv\\Scripts\\python.exe scripts\\train_rl.py --timesteps 20000   # quick check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.rl.train import train_ppo  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on the SUMO intersection.")
    p.add_argument("--timesteps", type=int, default=150_000)
    p.add_argument("--duration", type=float, default=900.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=config.PPO_MODEL)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"timesteps : {args.timesteps}")
    print(f"duration  : {args.duration}s  device={args.device} (CPU recommended)")
    path = train_ppo(
        timesteps=args.timesteps,
        duration=args.duration,
        seed=args.seed,
        out=args.out,
        device=args.device,
    )
    print(f"model -> {path}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
