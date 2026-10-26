"""Compare Fixed-Time, Rule-Based (actuated) and PPO controllers on SUMO.

Runs each controller on a normal and a highly-unbalanced (stress) demand,
reports avg delay / queue / throughput, and writes a comparison table + chart.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\evaluate_controllers.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.rl.controllers import (  # noqa: E402
    ActuatedController,
    FixedTimeController,
    PPOController,
)
from src.rl.evaluate import evaluate_controllers  # noqa: E402
from src.sumo.network import DEFAULT_DEMAND  # noqa: E402

STRESS_DEMAND = {"N": 2400, "S": 400, "E": 600, "W": 300}  # highly unbalanced


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare signal controllers.")
    p.add_argument("--model", type=Path, default=config.PPO_MODEL.with_suffix(".zip"))
    p.add_argument("--duration", type=float, default=1200.0)
    p.add_argument("--episodes", type=int, default=2)
    return p.parse_args()


def _controllers(model_path: Path):
    controllers = [FixedTimeController(green=30.0), ActuatedController()]
    if model_path.exists():
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path), device="cpu")
        controllers.append(PPOController(model))
    return controllers


def _print_table(title: str, results: list[dict]) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)
    print(f"{'controller':<14}{'avg_queue':>11}{'avg_delay(s)':>14}{'throughput':>12}")
    for r in results:
        print(f"{r['controller']:<14}{r['avg_queue']:>11.2f}{r['avg_delay']:>14.2f}{r['throughput']:>12}")


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        print(f"WARNING: PPO model not found at {args.model}; comparing 2 controllers only.")

    seeds = list(range(42, 42 + args.episodes))
    controllers = _controllers(args.model)

    normal = evaluate_controllers(
        _controllers(args.model), demand=DEFAULT_DEMAND,
        duration=args.duration, seeds=seeds,
    )
    stress = evaluate_controllers(
        _controllers(args.model), demand=STRESS_DEMAND,
        duration=args.duration, seeds=seeds,
    )

    _print_table("NORMAL DEMAND", normal)
    _print_table("STRESS DEMAND (highly unbalanced)", stress)

    # improvement of PPO vs fixed-time
    def pct(base, new, key):
        b = next((r[key] for r in base if r["controller"] == "Fixed-Time"), None)
        n = next((r[key] for r in new if r["controller"] == "PPO"), None)
        return None if not b or n is None else (b - n) / b * 100.0

    ppo_normal = pct(normal, normal, "avg_delay")
    ppo_stress = pct(stress, stress, "avg_delay")
    if ppo_normal is not None:
        print(f"\nPPO avg-delay reduction vs Fixed-Time: "
              f"normal {ppo_normal:.1f}%  |  stress {ppo_stress:.1f}%")

    # chart
    controllers_order = [r["controller"] for r in normal]
    x = np.arange(len(controllers_order))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, [next(r["avg_delay"] for r in normal if r["controller"] == c) for c in controllers_order],
           width, label="normal")
    ax.bar(x + width / 2, [next(r["avg_delay"] for r in stress if r["controller"] == c) for c in controllers_order],
           width, label="stress (unbalanced)")
    ax.set_xticks(x)
    ax.set_xticklabels(controllers_order)
    ax.set_ylabel("avg delay (s/vehicle)")
    ax.set_title("Controller comparison: Fixed-Time vs Rule-Based vs PPO")
    ax.legend()
    fig.tight_layout()
    out_png = config.FIGURES_DIR / "rl_comparison.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    payload = {
        "normal": normal,
        "stress": stress,
        "ppo_delay_reduction_normal_pct": ppo_normal,
        "ppo_delay_reduction_stress_pct": ppo_stress,
        "normal_demand": DEFAULT_DEMAND,
        "stress_demand": STRESS_DEMAND,
    }
    out_json = config.FIGURES_DIR / "rl_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nchart   -> {out_png}")
    print(f"summary -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
