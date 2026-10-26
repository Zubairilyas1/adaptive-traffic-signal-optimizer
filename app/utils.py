"""Pure helper functions for the Streamlit app (no Streamlit imports -> testable)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Iterable

from src import config
from src.analysis.roi import LaneROI
from src.decision.delay import DelayComparison
from src.decision.intersection import Intersection, SignalPlan
from src.inference.analyze_video import ClipCounts


def temp_dir() -> Path:
    """Writable scratch directory (on D:, because C: is full on this machine)."""
    path = config.ROOT / ".build" / "app_tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_csv(records: list[dict], fieldnames: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue()


def lane_records(counts: ClipCounts) -> list[dict]:
    rows: list[dict] = []
    for lane, classes in counts.lane_counts.items():
        approach = ""
        for a, cl in counts.approach_counts.items():
            if cl == classes:
                approach = a
        for cls, n in sorted(classes.items()):
            rows.append({
                "lane": lane,
                "class": cls,
                "count": n,
                "pcu": round(n * config.pcu_for(cls), 2),
            })
    return rows


def counts_csv(counts: ClipCounts) -> str:
    rows: list[dict] = []
    for approach, classes in counts.approach_counts.items():
        for cls, n in sorted(classes.items()):
            rows.append({
                "approach": approach,
                "class": cls,
                "count": n,
                "pcu": round(n * config.pcu_for(cls), 2),
            })
    return _to_csv(rows, ["approach", "class", "count", "pcu"])


def plan_records(plan: SignalPlan, intersection: Intersection) -> list[dict]:
    rows: list[dict] = []
    for phase in intersection.phases:
        approach = intersection.approach_for_phase(phase.id)
        rows.append({
            "phase": phase.id,
            "approach": approach.id,
            "label": approach.label,
            "lanes": approach.lanes,
            "green_s": round(plan.green[phase.id], 1),
        })
    return rows


def plan_csv(plan: SignalPlan, intersection: Intersection) -> str:
    return _to_csv(plan_records(plan, intersection),
                   ["phase", "approach", "label", "lanes", "green_s"])


def delay_records(comparison: DelayComparison) -> list[dict]:
    fixed = {r.phase_id: r for r in comparison.fixed_rows}
    opt = {r.phase_id: r for r in comparison.optimized_rows}
    rows: list[dict] = []
    for phase_id in fixed:
        rows.append({
            "phase": phase_id,
            "approach": fixed[phase_id].approach_id,
            "fixed_delay_s": round(fixed[phase_id].delay_s, 2),
            "webster_delay_s": round(opt[phase_id].delay_s, 2),
            "degree_of_saturation": round(opt[phase_id].degree_of_saturation, 3),
        })
    rows.append({
        "phase": "AVERAGE",
        "approach": "-",
        "fixed_delay_s": round(comparison.fixed_avg_delay, 2),
        "webster_delay_s": round(comparison.optimized_avg_delay, 2),
        "degree_of_saturation": "",
    })
    return rows


def delay_csv(comparison: DelayComparison) -> str:
    return _to_csv(delay_records(comparison),
                   ["phase", "approach", "fixed_delay_s", "webster_delay_s",
                    "degree_of_saturation"])


def rois_to_json(rois: Iterable[LaneROI]) -> str:
    lanes = [
        {
            "lane_id": r.lane_id,
            "approach": r.approach,
            "polygon": [[float(x), float(y)] for x, y in r.polygon],
            "buffer_px": r.buffer_px,
            "hysteresis_frames": r.hysteresis_frames,
        }
        for r in rois
    ]
    return json.dumps({"lanes": lanes}, indent=2)


def rois_from_json(text: str) -> list[LaneROI]:
    data = json.loads(text)
    return [
        LaneROI(
            lane_id=lane["lane_id"],
            approach=lane["approach"],
            polygon=[(float(x), float(y)) for x, y in lane["polygon"]],
            buffer_px=float(lane.get("buffer_px", 3.0)),
            hysteresis_frames=int(lane.get("hysteresis_frames", 3)),
        )
        for lane in data["lanes"]
    ]


def list_roi_presets() -> list[Path]:
    d = config.ROOT / "configs" / "rois"
    return sorted(d.glob("*.json")) if d.exists() else []


def list_intersections() -> list[Path]:
    d = config.ROOT / "configs" / "intersections"
    return sorted(d.glob("*.json")) if d.exists() else []


def list_sample_clips() -> list[Path]:
    d = config.SAMPLES_DIR
    if not d.exists():
        return []
    clips = [p for p in d.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    return sorted(clips)


def demo_video_path() -> Path | None:
    """The standalone results-reel video, if it has been generated."""
    p = config.REPORTS_DIR / "demo_video.mp4"
    return p if p.exists() else None


def annotated_dir() -> Path:
    d = config.REPORTS_DIR / "annotated"
    d.mkdir(parents=True, exist_ok=True)
    return d


def annotated_for(clip: Path) -> Path:
    """Conventional annotated-video path for a given clip."""
    return annotated_dir() / f"{clip.stem}_annotated.mp4"


def list_annotated_videos() -> list[Path]:
    d = config.REPORTS_DIR / "annotated"
    return sorted(d.glob("*.mp4")) if d.exists() else []


def plan_pdf(plan, comparison, intersection, counts=None) -> bytes:
    """Render the recommended plan + delay comparison as a one/two-page PDF."""
    import io as _io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buf = _io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.suptitle("Traffic Signal Timing Optimizer - Recommended Plan",
                     fontsize=14, fontweight="bold", y=0.97)
        ax = fig.add_axes([0.07, 0.55, 0.86, 0.36])
        ax.axis("off")
        lines = [
            f"Intersection : {intersection.name}",
            f"Method       : Webster's optimal timing",
            f"Critical flow ratio Y : {plan.critical_flow_ratio:.3f}"
            + ("  (OVERSATURATED)" if plan.oversaturated else ""),
            f"Lost time L  : {plan.lost_time:.1f} s",
            f"Optimized cycle : {plan.cycle:.1f} s",
            "",
            "Phase green allocation:",
        ]
        for phase in intersection.phases:
            approach = intersection.approach_for_phase(phase.id)
            lines.append(
                f"   {phase.id} ({approach.label}): {plan.green[phase.id]:.1f} s"
            )
        lines += [
            "",
            f"Fixed-time avg delay : {comparison.fixed_avg_delay:.2f} s/veh",
            f"Webster avg delay    : {comparison.optimized_avg_delay:.2f} s/veh",
            f"Delay reduction      : {comparison.reduction_pct:.1f} %",
        ]
        ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11,
                family="monospace")

        ax2 = fig.add_axes([0.15, 0.10, 0.7, 0.32])
        phases = [p.id for p in intersection.phases]
        fixed = {r.phase_id: r.delay_s for r in comparison.fixed_rows}
        opt = {r.phase_id: r.delay_s for r in comparison.optimized_rows}
        x = range(len(phases))
        ax2.bar([i - 0.2 for i in x], [fixed[p] for p in phases], width=0.4, label="fixed-time")
        ax2.bar([i + 0.2 for i in x], [opt[p] for p in phases], width=0.4, label="Webster")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(phases)
        ax2.set_ylabel("avg delay (s/vehicle)")
        ax2.set_title("Average delay per phase")
        ax2.legend()
        fig.text(0.07, 0.045,
                 "Assumptions: PCU-weighted demand; constant saturation flow and "
                 "lost time; Webster valid under undersaturation. Assistive "
                 "prototype, not a certified signal controller.",
                 fontsize=8, color="gray")
        pdf.savefig(fig)
        plt.close(fig)

    return buf.getvalue()
