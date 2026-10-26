"""Ablation: does PCU weighting actually change the recommended plan?

Compares two plans on the same measured scenario:
  A) naive  - every vehicle counts as 1 unit
  B) PCU    - each class weighted by its Passenger Car Unit factor

If the plans and delays differ, it proves PCU weighting is not cosmetic.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\ablation_pcu.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.analysis import flow as flow_mod  # noqa: E402
from src.analysis.roi import load_rois  # noqa: E402
from src.decision.baseline import evaluate  # noqa: E402
from src.decision.intersection import load_intersection  # noqa: E402
from src.inference.analyze_video import (  # noqa: E402
    counts_from_frames,
    synthetic_frames,
)

SPEC = {
    "A1": {"car": 40, "motorcycle": 25, "autorickshaw": 10},
    "A2": {"car": 30, "motorcycle": 20},
    "B1": {"car": 20, "bus": 4, "truck": 6},
    "B2": {"car": 15, "motorcycle": 10, "van": 3},
}


def _naive_flows(counts, intersection, window: float) -> dict[str, float]:
    """Treat every vehicle as 1 PCU (ignores class)."""
    approach_flow = {}
    for approach_id, classes in counts.approach_counts.items():
        vehicles = sum(classes.values())
        approach_flow[approach_id] = flow_mod.pcu_per_hour(float(vehicles), window)
    for a in intersection.approaches.values():
        approach_flow.setdefault(a.id, 0.0)
    return flow_mod.flows_by_phase(intersection, approach_flow)


def main() -> int:
    rois = load_rois(config.ROOT / "configs" / "rois" / "example.json")
    intersection = load_intersection(config.ROOT / "configs" / "intersections" / "default.json")

    frames = synthetic_frames(rois, SPEC, fps=4.0)
    counts = counts_from_frames(frames, rois, fps=4.0)
    window = counts.duration_s  # use the observed clip duration, not an arbitrary window

    pcu_flows = flow_mod.flows_by_phase(
        intersection,
        {a: flow_mod.pcu_per_hour(p, window) for a, p in counts.pcu_by_approach.items()},
    )
    naive = _naive_flows(counts, intersection, window)

    plan_pcu, cmp_pcu = evaluate(intersection, pcu_flows)
    plan_naive, cmp_naive = evaluate(intersection, naive)

    print("=" * 60)
    print(f"PCU ABLATION  (window = {window:.0f}s)")
    print("=" * 60)
    print(f"{'':<22}{'naive':>12}{'PCU':>12}")
    print(f"{'flow P1 (PCU/h)':<22}{naive['P1']:>12.0f}{pcu_flows['P1']:>12.0f}")
    print(f"{'flow P2 (PCU/h)':<22}{naive['P2']:>12.0f}{pcu_flows['P2']:>12.0f}")
    print(f"{'cycle (s)':<22}{plan_naive.cycle:>12.1f}{plan_pcu.cycle:>12.1f}")
    print(f"{'green P1 (s)':<22}{plan_naive.green['P1']:>12.1f}{plan_pcu.green['P1']:>12.1f}")
    print(f"{'green P2 (s)':<22}{plan_naive.green['P2']:>12.1f}{plan_pcu.green['P2']:>12.1f}")
    print(f"{'avg delay (s/veh)':<22}{cmp_naive.optimized_avg_delay:>12.2f}{cmp_pcu.optimized_avg_delay:>12.2f}")

    green_gap = abs(plan_naive.green["P1"] - plan_pcu.green["P1"])
    print(f"\ngreen-time difference on P1: {green_gap:.1f} s")

    summary = {
        "window_s": window,
        "naive": {
            "flow_p1": naive["P1"], "flow_p2": naive["P2"],
            "cycle": plan_naive.cycle, "green": plan_naive.green,
            "avg_delay": cmp_naive.optimized_avg_delay,
        },
        "pcu": {
            "flow_p1": pcu_flows["P1"], "flow_p2": pcu_flows["P2"],
            "cycle": plan_pcu.cycle, "green": plan_pcu.green,
            "avg_delay": cmp_pcu.optimized_avg_delay,
        },
        "green_p1_difference_s": green_gap,
    }
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "ablation_pcu.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
