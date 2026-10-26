"""Generate the system architecture / pipeline diagram (required by the report).

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\make_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

STAGES = [
    ("Intersection video / synthetic frames", "input", "#9aa0a6"),
    ("YOLO detector - 7 vehicle classes\n(car, motorcycle, autorickshaw,\nvan, tractor, bus, truck)", "Model", "#4C78A8"),
    ("ByteTrack multi-object tracking\n(stable IDs -> no double counting)", "Model", "#4C78A8"),
    ("Lane ROI assignment\n(ray-casting + hysteresis)", "Analysis", "#54A24B"),
    ("Per-lane class counts", "Analysis", "#54A24B"),
    ("PCU weighting\n(motorcycle 0.5 ... bus/truck 3.5)", "Analysis", "#54A24B"),
    ("Queue + arrival-demand analysis\n(per approach, over a time window)", "Analysis", "#54A24B"),
    ("SUMO micro-simulation (TraCI)\nfour-approach intersection", "Simulation", "#B279A2"),
    ("PPO reinforcement-learning agent\nobserves queues -> allocates green time", "RL", "#E45756"),
    ("Baselines\nFixed-Time vs Rule-Based (Webster/actuated)", "Decision", "#F58518"),
    ("Controller comparison\ndelay / queue / throughput", "Decision", "#F58518"),
    ("Streamlit dashboard\nvideo -> detections -> plan -> RL comparison", "UI", "#B279A2"),
]


def main() -> int:
    fig, ax = plt.subplots(figsize=(9.5, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(STAGES) * 1.15 + 0.5)
    ax.axis("off")

    box_w, box_h = 7.6, 0.82
    x = (10 - box_w) / 2

    coords: list[tuple[float, float]] = []
    for i, (text, category, color) in enumerate(STAGES):
        y = (len(STAGES) - i) * 1.15
        ax.add_patch(FancyBboxPatch(
            (x, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.5, edgecolor=color, facecolor=color + "22",
        ))
        ax.text(x + 0.25, y, text, va="center", ha="left", fontsize=10.5, wrap=True)
        ax.text(x + box_w - 0.15, y, category, va="center", ha="right",
                fontsize=8.5, color=color, style="italic")
        coords.append((x + box_w / 2, y))

    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        ax.add_patch(FancyArrowPatch(
            (x0, y0 - box_h / 2 - 0.02), (x1, y1 + box_h / 2 + 0.02),
            arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color="#444444",
        ))

    ax.set_title("Traffic Signal Timing Optimizer - system pipeline",
                 fontsize=14, fontweight="bold", pad=16)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "pipeline.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"pipeline diagram -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
