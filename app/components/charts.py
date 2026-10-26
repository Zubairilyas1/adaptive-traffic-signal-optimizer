"""Altair charts for counts, delay comparison, and real-time telemetry.

Synthesizes Image 1's clean visual clarity with Image 2's high-tech dark telemetry console:
- Carbon slate surfaces (#0d1424)
- High-contrast Cyan (#00f0ff) throughput bars & Coral (#ff3366) queue curves
- Monospace telemetry typography and subtle white gridlines
"""

from __future__ import annotations

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from app import _bootstrap  # noqa: F401

import altair as alt
import pandas as pd

from src import config
from src.analysis import pcu as pcu_mod
from src.analysis.queue import congestion_index
from src.decision.delay import DelayComparison
from src.decision.intersection import Intersection
from src.inference.analyze_video import ClipCounts


def _apply_telemetry_theme(chart: alt.Chart) -> alt.Chart:
    """Apply Image 2 high-tech dark telemetry console styling."""
    return (
        chart.configure(
            background="#0d1424",
        )
        .configure_view(
            stroke="transparent",
        )
        .configure_axis(
            domainColor="rgba(255, 255, 255, 0.14)",
            gridColor="rgba(255, 255, 255, 0.05)",
            tickColor="rgba(255, 255, 255, 0.14)",
            labelColor="#94a3b8",
            titleColor="#cbd5e1",
            labelFont="JetBrains Mono, Consolas, monospace",
            titleFont="JetBrains Mono, Consolas, monospace",
            labelFontSize=11,
            titleFontSize=12,
            titleFontWeight="bold",
        )
        .configure_title(
            color="#ffffff",
            font="JetBrains Mono, Consolas, monospace",
            fontSize=13,
            fontWeight="bold",
            anchor="start",
            offset=12,
        )
        .configure_legend(
            labelColor="#94a3b8",
            titleColor="#cbd5e1",
            labelFont="JetBrains Mono, Consolas, monospace",
            titleFont="JetBrains Mono, Consolas, monospace",
            labelFontSize=11,
            titleFontSize=12,
            titleFontWeight="bold",
        )
    )


def counts_chart(counts: ClipCounts) -> alt.Chart:
    rows = [
        {"approach": a, "class": cls, "count": n}
        for a, classes in counts.approach_counts.items()
        for cls, n in classes.items()
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame([{"approach": "none", "class": "none", "count": 0}])
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("approach:N", title="Approach", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count:Q", title="Vehicles", stack="zero"),
            color=alt.Color(
                "class:N",
                title="Class",
                scale=alt.Scale(
                    range=["#00f0ff", "#38bdf8", "#ff3366", "#00ff9d", "#ffaa00", "#a78bfa"]
                ),
            ),
            tooltip=["approach", "class", "count"],
        )
        .properties(height=300, title="⚡ Vehicles Per Approach by Class")
    )
    return _apply_telemetry_theme(chart)


def pcu_chart(counts: ClipCounts) -> alt.Chart:
    rows = [
        {"approach": a, "PCU": round(pcu, 1)}
        for a, pcu in counts.pcu_by_approach.items()
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame([{"approach": "none", "PCU": 0.0}])
    chart = (
        alt.Chart(df)
        .mark_bar(color="#00f0ff", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("approach:N", title="Approach", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("PCU:Q", title="PCU Demand"),
            tooltip=["approach", "PCU"],
        )
        .properties(height=280, title="⚡ PCU-Weighted Demand Per Approach")
    )
    return _apply_telemetry_theme(chart)


def lane_pcu_chart(counts: ClipCounts) -> alt.Chart:
    """Per-lane PCU bar chart."""
    rows = [{"lane": lane, "PCU": round(value, 1)}
            for lane, value in pcu_mod.lane_pcu(counts.lane_counts).items()]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame([{"lane": "none", "PCU": 0.0}])
    chart = (
        alt.Chart(df)
        .mark_bar(color="#38bdf8", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("lane:N", title="Lane", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("PCU:Q", title="PCU Demand"),
            tooltip=["lane", "PCU"],
        )
        .properties(height=300, title="⚡ Lane-Level PCU Demand Distribution")
    )
    return _apply_telemetry_theme(chart)


def lane_count_records(counts: ClipCounts) -> list[dict]:
    """Per-lane, per-class rows for tabbed operational ledgers."""
    rows = []
    for lane, classes in counts.lane_counts.items():
        for cls, n in sorted(classes.items()):
            rows.append({
                "lane": lane,
                "class": cls,
                "count": n,
                "pcu": round(n * config.pcu_for(cls), 2),
            })
    return rows


def delay_chart(comparison: DelayComparison, intersection: Intersection) -> alt.Chart:
    fixed = {r.phase_id: r.delay_s for r in comparison.fixed_rows}
    opt = {r.phase_id: r.delay_s for r in comparison.optimized_rows}
    rows = []
    for phase in intersection.phases:
        rows.append({"phase": phase.id, "plan": "fixed-time", "delay": round(fixed[phase.id], 2)})
        rows.append({"phase": phase.id, "plan": "Webster", "delay": round(opt[phase.id], 2)})
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("phase:N", title="Signal Phase", axis=alt.Axis(labelAngle=0)),
            xOffset="plan:N",
            y=alt.Y("delay:Q", title="Average Delay (s/vehicle)"),
            color=alt.Color(
                "plan:N",
                title="Control Regime",
                scale=alt.Scale(
                    domain=["fixed-time", "Webster"],
                    range=["#ff3366", "#00ff9d"],
                ),
            ),
            tooltip=["phase", "plan", "delay"],
        )
        .properties(
            height=320,
            title=f"⚡ Delay Reduction: Fixed-Time vs Adaptive Webster ({comparison.reduction_pct:.1f}% Improvement)",
        )
    )
    return _apply_telemetry_theme(chart)


def congestion_rows(flows_by_approach: dict[str, float], intersection: Intersection) -> list[dict]:
    rows = []
    for approach in intersection.approaches.values():
        flow = float(flows_by_approach.get(approach.id, 0.0))
        rows.append({
            "approach": approach.id,
            "label": approach.label,
            "lanes": approach.lanes,
            "flow_pcu_h": round(flow, 1),
            "congestion_index": round(congestion_index(flow, approach.saturation_flow), 3),
        })
    return rows


def compute_wasted_green_saved(counts: ClipCounts) -> float:
    """Tally total seconds of empty road green time saved by dynamic Gap-Out."""
    if not getattr(counts, "actuator_events", None):
        return 0.0
    saved = 0.0
    for e in counts.actuator_events:
        if getattr(e.reason, "name", "") == "GAP_OUT":
            saved += 8.5  # Average seconds saved per early green termination
    return round(saved, 1)


def actuation_timeline_chart(counts: ClipCounts) -> alt.Chart:
    """Synchronized dual-layer telemetry graph (Image 2 aesthetic):
    - Cyan histogram bars (#00f0ff) for vehicle arrival throughput
    - Overlaid Coral line curve (#ff3366) with point markers for waiting queue PCU
    - Vertical dashed markers for Gap-Out / Max-Out switching events
    """
    snaps = getattr(counts, "kinematic_snapshots", [])
    rows = []
    for snap in snaps:
        ts = round(snap.timestamp, 1)
        for app, pcu in snap.approach_halted_pcu.items():
            flow = 0.0
            if hasattr(snap, "approach_flowing_pcu") and snap.approach_flowing_pcu:
                flow = snap.approach_flowing_pcu.get(app, 0.0)
            rows.append({
                "Timestamp (s)": ts,
                "Approach": app,
                "Waiting Queue (PCU)": round(pcu, 1),
                "Throughput Flow (PCU)": round(flow, 1),
            })

    if not rows:
        df = pd.DataFrame([{
            "Timestamp (s)": 0.0,
            "Approach": "None",
            "Waiting Queue (PCU)": 0.0,
            "Throughput Flow (PCU)": 0.0,
        }])
    else:
        df = pd.DataFrame(rows)

    base = alt.Chart(df).encode(
        x=alt.X("Timestamp (s):Q", title="Video Timeline (seconds)"),
    )

    # 1. Cyan Bars: Throughput Flow (Image 2 cyan histogram bars)
    bars = base.mark_bar(
        color="#00f0ff",
        opacity=0.35,
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2,
    ).encode(
        y=alt.Y("Throughput Flow (PCU):Q", title="PCU Demand & Flow"),
        tooltip=["Timestamp (s)", "Approach", "Throughput Flow (PCU)", "Waiting Queue (PCU)"],
    )

    # 2. Coral Line: Waiting Queue Curve (Image 2 coral curve with markers)
    lines = base.mark_line(
        strokeWidth=2.5,
        point=alt.OverlayMarkDef(size=50, filled=True),
    ).encode(
        y=alt.Y("Waiting Queue (PCU):Q", title="Standing Waiting Queue (PCU)"),
        color=alt.Color(
            "Approach:N",
            title="Approach",
            scale=alt.Scale(range=["#ff3366", "#00f0ff", "#00ff9d", "#ffaa00"]),
        ),
        tooltip=["Timestamp (s)", "Approach", "Waiting Queue (PCU)", "Throughput Flow (PCU)"],
    )

    chart_base = (bars + lines).properties(
        height=320,
        title="⚡ Telemetry: Throughput (Cyan Bars) vs Waiting Queue (Coral Curve)",
    )

    events = getattr(counts, "actuator_events", [])
    if events:
        event_rows = [
            {
                "Timestamp (s)": round(e.timestamp, 1),
                "Event": e.reason.name,
                "Rationale": e.message,
            }
            for e in events
        ]
        edf = pd.DataFrame(event_rows)
        rule = alt.Chart(edf).mark_rule(
            color="#ffaa00",
            strokeDash=[4, 4],
            strokeWidth=2,
        ).encode(
            x="Timestamp (s):Q",
            tooltip=["Timestamp (s)", "Event", "Rationale"],
        )
        combined = (chart_base + rule).interactive()
    else:
        combined = chart_base.interactive()

    return _apply_telemetry_theme(combined)
