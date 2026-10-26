"""Analyze an intersection clip (or synthetic scenario) and produce a signal plan.

Examples
--------
    # synthetic scenario (no video needed)
    .\\.venv\\Scripts\\python.exe scripts\\analyze_video.py --synthetic

    # real clip
    .\\.venv\\Scripts\\python.exe scripts\\analyze_video.py ^
        --video path\\to\\clip.mp4 --rois configs\\rois\\example.json --fps 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.analysis.roi import load_rois  # noqa: E402
from src.decision.intersection import load_intersection  # noqa: E402
from src.inference.analyze_video import (  # noqa: E402
    counts_from_frames,
    plan_from_counts,
    synthetic_frames,
)

DEFAULT_SYNTHETIC = {
    "A1": {"car": 40, "motorcycle": 25, "autorickshaw": 10},
    "A2": {"car": 30, "motorcycle": 20},
    "B1": {"car": 20, "bus": 4, "truck": 6},
    "B2": {"car": 15, "motorcycle": 10, "van": 3},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intersection analysis -> signal plan.")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--synthetic", action="store_true", help="Use a synthetic scenario.")
    p.add_argument("--rois", type=Path, default=ROOT / "configs" / "rois" / "example.json")
    p.add_argument("--intersection", type=Path,
                   default=ROOT / "configs" / "intersections" / "default.json")
    p.add_argument("--stride", type=int, default=config.FRAME_STRIDE)
    p.add_argument("--fps", type=float, default=4.0, help="Synthetic mode frame rate.")
    p.add_argument("--video-fps", type=float, default=None, help="Real clip frame rate.")
    p.add_argument("--window", type=float, default=None, help="Aggregation window (s).")
    p.add_argument("--weights", type=Path, default=config.DETECTOR_WEIGHTS)
    p.add_argument("--device", default="cpu",
                   help="'cpu' (thermal-safe) or a GPU index like 0.")
    p.add_argument("--tag", default="synthetic")
    return p.parse_args()


def _bar_counts(counts, intersection, out_path: Path) -> None:
    classes = config.TARGET_CLASS_NAMES
    approaches = list(counts.approach_counts.keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottoms = [0] * len(approaches)
    for cls in classes:
        vals = [counts.approach_counts.get(a, {}).get(cls, 0) for a in approaches]
        if not any(vals):
            continue
        ax.bar(approaches, vals, bottom=bottoms, label=cls)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_title("Vehicles per approach by class")
    ax.set_ylabel("vehicles")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _bar_delay(comparison, intersection, out_path: Path) -> None:
    phases = [p.id for p in intersection.phases]
    fixed = {r.phase_id: r.delay_s for r in comparison.fixed_rows}
    opt = {r.phase_id: r.delay_s for r in comparison.optimized_rows}
    x = range(len(phases))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - 0.2 for i in x], [fixed[p] for p in phases], width=0.4, label="fixed-time")
    ax.bar([i + 0.2 for i in x], [opt[p] for p in phases], width=0.4, label="Webster")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{p}\n({intersection.approaches[intersection.approach_for_phase(p).id].label})"
                        for p in phases])
    ax.set_ylabel("avg delay (s/vehicle)")
    ax.set_title(f"Average delay per approach - {comparison.reduction_pct:.1f}% overall reduction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rois = load_rois(args.rois)
    intersection = load_intersection(args.intersection)

    if args.synthetic or args.video is None:
        frames = synthetic_frames(rois, DEFAULT_SYNTHETIC, fps=args.fps)
        counts = counts_from_frames(frames, rois, fps=args.fps)
        print("mode      : synthetic")
    else:
        from src.models import load_detector
        detector = load_detector(weights=args.weights, device=args.device)
        from src.inference.analyze_video import count_clip
        counts = count_clip(detector, args.video, rois, stride=args.stride, fps=args.video_fps)
        print("mode      : video")

    print(f"rois      : {args.rois}")
    print(f"intersect : {intersection.name}  ({len(intersection.approaches)} approaches)")
    print(f"window    : {counts.duration_s:.1f} s  ({counts.processed_frames} frames)")

    plan, comparison, flows = plan_from_counts(counts, intersection, window_s=args.window)

    print("\n" + "=" * 62)
    print("COUNTS  (PCU-weighted)")
    print("=" * 62)
    for a in intersection.approaches:
        classes = counts.approach_counts.get(a, {})
        pcu = counts.pcu_by_approach.get(a, 0.0)
        naive = sum(classes.values())
        print(f"  {a} {intersection.approaches[a].label:<12} "
              f"vehicles={naive:<4} PCU={pcu:>6.1f}  flow={flows.get(_phase_of(intersection, a), 0):>7.0f} PCU/h")

    print("\n" + "=" * 62)
    print("SIGNAL PLAN")
    print("=" * 62)
    print(f"  critical flow ratio Y : {plan.critical_flow_ratio:.3f}"
          f"{'  (OVERSATURATED)' if plan.oversaturated else ''}")
    print(f"  lost time L           : {plan.lost_time:.1f} s")
    print(f"  optimized cycle       : {plan.cycle:.1f} s")
    for p in intersection.phases:
        print(f"    {p.id} green : {plan.green[p.id]:.1f} s")

    print("\n" + "=" * 62)
    print("DELAY COMPARISON  (fixed-time vs Webster)")
    print("=" * 62)
    print(f"  fixed-time avg delay : {comparison.fixed_avg_delay:6.2f} s/veh")
    print(f"  Webster avg delay    : {comparison.optimized_avg_delay:6.2f} s/veh")
    print(f"  reduction            : {comparison.reduction_pct:6.1f} %")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    counts_png = config.FIGURES_DIR / f"counts_{args.tag}.png"
    delay_png = config.FIGURES_DIR / f"delay_{args.tag}.png"
    _bar_counts(counts, intersection, counts_png)
    _bar_delay(comparison, intersection, delay_png)

    summary = {
        "tag": args.tag,
        "mode": "synthetic" if (args.synthetic or args.video is None) else "video",
        "approach_counts": counts.approach_counts,
        "pcu_by_approach": counts.pcu_by_approach,
        "flows_by_phase_pcu_h": flows,
        "plan": {
            "cycle": plan.cycle,
            "green": plan.green,
            "Y": plan.critical_flow_ratio,
            "oversaturated": plan.oversaturated,
        },
        "delay": {
            "fixed_avg_s": comparison.fixed_avg_delay,
            "optimized_avg_s": comparison.optimized_avg_delay,
            "reduction_pct": comparison.reduction_pct,
        },
    }
    out_json = config.FIGURES_DIR / f"analysis_{args.tag}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nfigures -> {counts_png}")
    print(f"          {delay_png}")
    print(f"summary -> {out_json}")
    return 0


def _phase_of(intersection, approach_id: str) -> str:
    for p in intersection.phases:
        if p.approach == approach_id:
            return p.id
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
