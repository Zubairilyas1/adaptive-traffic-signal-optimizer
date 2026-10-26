"""Page 1 - Intersection Analysis Console.

Real-time video perception, kinematic queue decomposition, and closed-loop actuation.
"""

from __future__ import annotations

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from app import _bootstrap  # noqa: F401

from pathlib import Path
import pandas as pd
import streamlit as st

from app.components import charts, video
from app.components.cards import (
    render_executive_kpi_card_row,
    render_signal_light_widget,
    show_disclaimer,
)
from app.components.roi_editor import render_roi_editor
from app.components.state import get_detector, get_intersection, init_state
from app.components.theme import apply_theme, render_breadcrumb, render_header
from app.utils import (
    annotated_for,
    counts_csv,
    list_intersections,
    list_sample_clips,
)
from src import config
from src.decision.dual_regime import OptimizationRegime
from src.decision.intersection import FixedPlan
from src.inference.analyze_video import plan_from_counts

init_state()
apply_theme()

# Top Breadcrumb & Console Header (Image 1 style)
render_breadcrumb("OPERATIONS > INTERSECTION COMMAND CONSOLE")
render_header(
    title="Intersection Analysis Console",
    subtitle="Live edge perception, kinematic queue decomposition, and second-by-second signal actuation",
    chips=[
        ("EDGE PERCEPTION: ACTIVE", "chip-green"),
        ("QUEUE METRIC: SPATIAL METERS (IPM)", "chip-cyan"),
        ("ACTUATION: CLOSED-LOOP (GAP-OUT / MAX-OUT)", "chip-green"),
    ],
)

# Unified Operator Deck (Two Columns)
deck_left, deck_right = st.columns([1, 1])

with deck_left:
    st.markdown("#### 1. Topology & Geometry")
    inter_paths = list_intersections()
    inter_names = [p.name for p in inter_paths]
    inter_choice = st.selectbox("Road Configuration Preset", inter_names, index=0, key="inter_select")
    inter_path = inter_paths[inter_names.index(inter_choice)]
    st.session_state["intersection_path"] = str(inter_path)
    intersection = get_intersection(str(inter_path))
    
    st.caption(
        f"Topology: `{len(intersection.approaches)} approaches` · "
        f"`{len(intersection.phases)} phases` · "
        f"Lost time: `{intersection.total_lost_time:.0f}s / cycle`"
    )

    rois, roi_path = render_roi_editor()

if not rois:
    st.stop()

st.session_state["rois"] = rois

with deck_right:
    st.markdown("#### 2. Video Feed Ingestion")
    mode = st.radio(
        "Ingest Mode",
        [
            "Bundled Sample Clip",
            "YouTube Video URL",
            "Upload Local Video",
            "Synthetic Simulation",
        ],
        horizontal=True,
    )

    stride = st.slider(
        "Sampling Stride (process every Nth frame)",
        min_value=1,
        max_value=8,
        value=config.FRAME_STRIDE,
        help="Higher stride reduces CPU latency.",
    )

    uploaded = None
    clip_path = None

    if mode == "Bundled Sample Clip":
        clips = list_sample_clips()
        if clips:
            names = [c.name for c in clips]
            chosen = st.selectbox("Sample Clip Archive", names, key="sample_clip")
            clip_path = config.SAMPLES_DIR / chosen
            st.caption(f"Source: `{clip_path.name}` ({clip_path.stat().st_size / 1e6:.1f} MB)")
        else:
            st.warning("No sample clips found in data/samples/.")

    elif mode == "YouTube Video URL":
        yt_url = st.text_input(
            "YouTube Stream URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any public traffic camera video link.",
        )
        max_dur = st.slider("Max Duration (seconds)", 15, 120, 45, step=15)
        if yt_url:
            from src.data.youtube import download_youtube_video, is_youtube_url
            if not is_youtube_url(yt_url):
                st.error("Invalid YouTube URL.")
            else:
                if st.button("Fetch YouTube Clip"):
                    with st.status("Fetching video stream from YouTube ...", expanded=True) as status:
                        try:
                            path_out, title_out, dur_out = download_youtube_video(
                                yt_url, max_duration_s=float(max_dur)
                            )
                            st.session_state["yt_clip_path"] = str(path_out)
                            st.session_state["yt_title"] = title_out
                            status.update(label=f"Downloaded '{title_out}' ({dur_out:.0f}s)", state="complete")
                        except Exception as exc:  # noqa: BLE001
                            status.update(label=f"Download failed: {exc}", state="error")
                            st.error(f"Error: {exc}")

                yt_path_str = st.session_state.get("yt_clip_path")
                if yt_path_str and Path(yt_path_str).exists():
                    clip_path = Path(yt_path_str)
                    st.success(f"Stream Loaded: {st.session_state.get('yt_title', clip_path.name)}")

    elif mode == "Upload Local Video":
        uploaded = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    run = st.button("⚡ Initiate AI Analysis & Actuation", type="primary", use_container_width=True)

# Processing Trigger
if run:
    try:
        if mode == "Synthetic Simulation":
            counts, keyframes = video.run_synthetic(rois)
            st.info("Synthetic scenario completed (deterministic benchmark).")
        else:
            if clip_path is None and uploaded is not None:
                clip_path = video.save_upload(uploaded.name, uploaded.getbuffer())
            if clip_path is None and mode == "YouTube Video URL":
                yt_path_str = st.session_state.get("yt_clip_path")
                if yt_path_str and Path(yt_path_str).exists():
                    clip_path = Path(yt_path_str)
            if clip_path is None:
                st.error("No video stream loaded. Please select or fetch a clip first.")
                st.stop()

            detector = get_detector(st.session_state["detector_weights"])
            fps = video.probe_fps(clip_path)
            nframes = video.frame_count(clip_path)
            annotated_target = annotated_for(clip_path)

            with st.status("Executing YOLOv8 Tracking + Dynamic Actuation ...", expanded=False) as status:
                counts, keyframes = video.run_on_video(
                    detector,
                    clip_path,
                    rois,
                    stride=stride,
                    fps=fps,
                    save_annotated_path=annotated_target,
                    intersection=intersection,
                )
                status.update(label="Inference & Actuation Complete", state="complete")

        st.session_state["counts"] = counts
        st.session_state["keyframes"] = keyframes
        st.session_state["plan"] = None
        st.session_state["comparison"] = None
        st.session_state["last_clip"] = (
            str(clip_path) if mode != "Synthetic Simulation" and clip_path else None
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Analysis failed: {exc}")

counts = st.session_state.get("counts")

# Results Deck
if counts is not None:
    st.divider()

    # Precalculate key values
    intersection = get_intersection(st.session_state["intersection_path"])
    window = float(round(counts.duration_s)) if counts.duration_s > 0 else 60.0
    fixed_cycle = 90.0
    fixed_green = {phase.id: fixed_cycle / len(intersection.phases) for phase in intersection.phases}
    fixed_plan = FixedPlan(cycle=fixed_cycle, green=fixed_green)

    last_snap = counts.kinematic_snapshots[-1] if counts.kinematic_snapshots else None
    max_queue_m = max(last_snap.approach_queue_m.values()) if last_snap and last_snap.approach_queue_m else 0.0
    wasted_saved = charts.compute_wasted_green_saved(counts)

    # 4 Executive KPI Stat Tiles (Image 1 style dark-header cards with circular icon badges)
    render_executive_kpi_card_row([
        {
            "header": "Total Vehicles",
            "value": f"{counts.total_vehicles}",
            "delta": "DE-DUPLICATED",
            "delta_positive": True,
            "icon": "🚗",
            "icon_bg": "rgba(0, 240, 255, 0.12)",
            "icon_color": "#00f0ff",
        },
        {
            "header": "PCU Demand Load",
            "value": f"{sum(counts.pcu_by_approach.values()):.1f}",
            "delta": "EQUIVALENT PCU",
            "delta_positive": True,
            "icon": "⚡",
            "icon_bg": "rgba(0, 255, 157, 0.12)",
            "icon_color": "#00ff9d",
        },
        {
            "header": "Max Queue Backlog",
            "value": f"{max_queue_m:.1f} m",
            "delta": "IPM HOMOGRAPHY",
            "delta_positive": max_queue_m < 50.0,
            "icon": "📏",
            "icon_bg": "rgba(255, 170, 0, 0.12)",
            "icon_color": "#ffaa00",
        },
        {
            "header": "Wasted Green Saved",
            "value": f"{wasted_saved:.1f} s",
            "delta": "DYNAMIC GAP-OUT",
            "delta_positive": True,
            "icon": "⏱️",
            "icon_bg": "rgba(255, 51, 102, 0.12)",
            "icon_color": "#ff3366",
        },
    ])

    # Side-by-Side Video & Dynamics Visualizer
    last_clip = st.session_state.get("last_clip")
    if last_clip:
        clip_path_obj = Path(last_clip)
        annotated = annotated_for(clip_path_obj)
        v_col, g_col = st.columns([1, 1])

        with v_col:
            st.markdown("##### Annotated Video with Live HUD")
            if annotated.exists():
                st.video(str(annotated))
                st.caption(f"Live HUD stamps 3-aspect signal lights, speed vectors, and AI decision ticker.")

        with g_col:
            st.markdown("##### Synchronized Real-Time Queue Dynamics")
            if getattr(counts, "actuator_events", None) or getattr(counts, "kinematic_snapshots", None):
                st.altair_chart(charts.actuation_timeline_chart(counts), use_container_width=True)
            else:
                st.altair_chart(charts.pcu_chart(counts), use_container_width=True)

    # Operational Ledger Tabs
    tab_spatial, tab_ledger, tab_plan, tab_export = st.tabs([
        "📐 Spatial Queue & Approach Backlog",
        "⚡ Second-by-Second Actuation Log",
        "🚦 Dual-Regime Signal Plan",
        "📥 Data Export",
    ])

    with tab_spatial:
        rows = []
        for a, classes in counts.approach_counts.items():
            h_pcu = counts.halted_queue_pcu.get(a, 0.0)
            f_pcu = counts.flowing_throughput_pcu.get(a, 0.0)
            q_m = last_snap.approach_queue_m.get(a, 0.0) if last_snap else 0.0
            rows.append({
                "Approach": a,
                "Vehicles": sum(classes.values()),
                "Total Demand (PCU)": round(counts.pcu_by_approach.get(a, 0.0), 1),
                "Halted Queue (PCU)": round(h_pcu, 1),
                "Flowing (PCU)": round(f_pcu, 1),
                "Physical Queue Backlog": f"{q_m:.1f} m",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        c_pcu1, c_pcu2 = st.columns(2)
        with c_pcu1:
            st.altair_chart(charts.counts_chart(counts), use_container_width=True)
        with c_pcu2:
            st.altair_chart(charts.lane_pcu_chart(counts), use_container_width=True)

    with tab_ledger:
        if getattr(counts, "actuator_events", None):
            event_rows = [
                {
                    "Timestamp": f"{e.timestamp:.1f} s",
                    "Active Phase": e.from_phase,
                    "Aspect": e.aspect.value.upper(),
                    "Decision Trigger": e.reason.name,
                    "AI Rationale": e.message,
                    "Waiting Queue (PCU)": round(e.waiting_queue_pcu, 1),
                }
                for e in counts.actuator_events
            ]
            st.dataframe(pd.DataFrame(event_rows), width="stretch", hide_index=True)
        else:
            st.info("No discrete phase switches triggered (traffic operated uninterrupted within minimum green).")

    with tab_plan:
        try:
            plan, comparison, flows = plan_from_counts(
                counts, intersection, window_s=window, fixed_plan=fixed_plan
            )
            st.session_state["plan"] = plan
            st.session_state["comparison"] = comparison
            st.session_state["flows"] = flows

            if hasattr(plan, "regime"):
                if plan.regime == OptimizationRegime.UNDERSATURATED_WEBSTER:
                    st.success(f"🟢 **{plan.regime.value}** — {plan.regime_rationale}")
                elif plan.regime == OptimizationRegime.NEAR_CAPACITY:
                    st.warning(f"🟡 **{plan.regime.value}** — {plan.regime_rationale}")
                else:
                    st.error(f"🔴 **{plan.regime.value}** — {plan.regime_rationale}")

            m_col = st.columns(4)
            m_col[0].metric("Critical Flow Ratio Y", f"{plan.critical_flow_ratio:.3f}")
            m_col[1].metric("Optimized Cycle", f"{plan.cycle:.1f} s")
            m_col[2].metric("Fixed Baseline Delay", f"{comparison.fixed_avg_delay:.2f} s/veh")
            m_col[3].metric("Adaptive Delay", f"{comparison.optimized_avg_delay:.2f} s/veh",
                            delta=f"-{comparison.improvement_s:.2f} s")

            render_signal_light_widget(plan, intersection)

            st.markdown("##### Delay Reduction Comparison")
            st.altair_chart(charts.delay_chart(comparison, intersection), use_container_width=True)

        except Exception as exc:  # noqa: BLE001
            st.warning(f"Optimization note: {exc}")

    with tab_export:
        st.download_button(
            "Download Vehicle Counts (CSV)",
            data=counts_csv(counts).encode("utf-8"),
            file_name="vehicle_counts.csv",
            mime="text/csv",
            key="dl_counts_tab",
        )

show_disclaimer()
