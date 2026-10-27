"""Page 1 - Intersection Analysis & Real-Time Enforcement Console.

Real-time video perception, kinematic queue decomposition, closed-loop actuation,
NEMA fail-safe supervision, red-light enforcement, and SQLite telemetry.
"""

from __future__ import annotations

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from app import _bootstrap  # noqa: F401

from pathlib import Path
import time
import cv2
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
from src.analysis.hud import HUDOverlayRenderer
from src.analysis.kinematics import KinematicLaneCounter
from src.analysis.violations import RedLightViolationDetector, stoplines_from_rois
from src.db.database import (
    get_summary_stats,
    log_actuation_event,
    log_telemetry,
    query_actuation_events,
    query_telemetry,
    query_violations,
)
from src.decision.actuator import AdaptiveActuator, SignalAspect
from src.decision.dual_regime import OptimizationRegime
from src.decision.fail_safe import FailureReason, HealthState, NEMAFailSafeWatchdog
from src.decision.intersection import FixedPlan
from src.hardware.controller_bridge import ControllerBridgeMode, SignalControllerBridge
from src.inference.analyze_video import plan_from_counts
from src.inference.stream_worker import VideoStreamWorker

init_state()
apply_theme()

# Top Breadcrumb & Console Header (Image 1 style)
render_breadcrumb("OPERATIONS > INTERSECTION COMMAND CONSOLE")
render_header(
    title="Intersection Analysis & Enforcement Console",
    subtitle="Live edge perception, kinematic queue decomposition, closed-loop actuation, and red-light enforcement",
    chips=[
        ("EDGE PERCEPTION: ACTIVE", "chip-green"),
        ("NEMA FAIL-SAFE: ACTIVE", "chip-green"),
        ("DATABASE: SQLITE WAL", "chip-cyan"),
        ("CONTROLLER BRIDGE: READY", "chip-amber"),
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
            "Live Laptop/USB Camera",
            "Phone WiFi IP Camera / RTSP",
            "Upload Local Video",
            "YouTube Video URL",
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
    live_source = None
    loop_video = False

    if mode == "Bundled Sample Clip":
        clips = list_sample_clips()
        if clips:
            names = [c.name for c in clips]
            chosen = st.selectbox("Sample Clip Archive", names, key="sample_clip")
            clip_path = config.SAMPLES_DIR / chosen
            loop_video = st.checkbox("Continuous Loop Mode", value=False, help="Repeat playback indefinitely for demonstrations.")
            st.caption(f"Source: `{clip_path.name}` ({clip_path.stat().st_size / 1e6:.1f} MB)")
        else:
            st.warning("No sample clips found in data/samples/.")

    elif mode == "Live Laptop/USB Camera":
        cam_id = st.number_input(
            "Camera Device Index",
            min_value=0,
            max_value=4,
            value=0,
            help="Device 0 is typically your built-in laptop webcam; 1 is an external USB camera.",
        )
        live_source = int(cam_id)
        st.caption(f"Selected hardware video capture: `Device {cam_id}`")

    elif mode == "Phone WiFi IP Camera / RTSP":
        rtsp_url = st.text_input(
            "Camera Stream URL",
            value="http://192.168.1.5:8080/video",
            help="Install a free app like 'IP Webcam' on Android/iOS to stream your phone camera over WiFi via RTSP or HTTP.",
        )
        live_source = rtsp_url.strip()
        st.caption("Broadcast phone camera video directly into edge perception.")

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

    is_live_mode = mode in ("Live Laptop/USB Camera", "Phone WiFi IP Camera / RTSP") or loop_video
    if is_live_mode:
        live_dur = st.slider(
            "Live Stream Duration (seconds)",
            min_value=10,
            max_value=120,
            value=30,
            step=10,
            help="Duration for active edge perception and HUD rendering before completing.",
        )
    else:
        live_dur = 30

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    c_act1, c_act2 = st.columns(2)
    with c_act1:
        run_batch = st.button("⚡ Run Batch Analysis & Plan", type="primary", use_container_width=True)
    with c_act2:
        run_live = st.button("🔴 Launch Live Stream Console", disabled=not is_live_mode, use_container_width=True)

# -------------------------------------------------------------
# LIVE STREAM OPERATOR CONSOLE (Real-Time Edge Mode)
# -------------------------------------------------------------
if run_live:
    src_input = live_source if live_source is not None else clip_path
    if src_input is None:
        st.error("Please configure a valid camera source or sample video.")
        st.stop()

    st.divider()
    st.markdown("### 🔴 Live Real-Time Perception & Enforcement Desk")
    
    detector = get_detector(st.session_state["detector_weights"])
    watchdog = NEMAFailSafeWatchdog(timeout_s=2.5, min_variance=4.0)
    bridge = SignalControllerBridge(mode=ControllerBridgeMode.VIRTUAL)
    bridge.open()
    
    stoplines = stoplines_from_rois(rois)
    viol_detector = RedLightViolationDetector(stoplines)
    kinematic_counter = KinematicLaneCounter(rois)
    actuator = AdaptiveActuator(intersection)
    hud = HUDOverlayRenderer(rois, intersection=intersection)

    col_view, col_stats = st.columns([3, 2])

    with col_stats:
        status_box = st.empty()
        kpi_box = st.empty()
        viol_alert = st.empty()

    with col_view:
        video_viewport = st.empty()

    try:
        with VideoStreamWorker(src_input, loop=True) as worker:
            frame_idx = 0
            start_t = time.monotonic()
            now_ts = 0.0

            while worker.is_running and now_ts < float(live_dur):
                ret, frame, ts = worker.read(timeout=0.2)
                if not ret or frame is None:
                    continue

                frame_idx += 1
                if frame_idx % stride != 0:
                    continue

                now_ts = time.monotonic() - start_t

                # 1. Feed Watchdog Heartbeat
                watchdog.feed_heartbeat(now_ts, frame=frame)
                h_state, fail_reason, health_msg = watchdog.evaluate(now_ts)

                # 2. YOLOv8 Tracking Inference
                dets = detector(frame)
                snap = kinematic_counter.update(dets, timestamp=now_ts)

                # 3. Dynamic Actuation vs Safety Failsafe
                if watchdog.is_failsafe_active:
                    ctrl = watchdog.step_fallback(now_ts, intersection)
                else:
                    ctrl = actuator.step(now_ts, snapshot=snap)

                # 4. Red-Light Violation Detector
                aspects_map = {
                    disp.approach: disp.aspect.value.upper()
                    for disp in ctrl.phase_displays.values()
                }
                new_viols = viol_detector.update(
                    dets,
                    aspects_map,
                    timestamp=now_ts,
                    frame=frame,
                    intersection_id=intersection.name,
                )

                # 5. Broadcast to Hardware Bridge
                bridge.send_state(ctrl, is_failsafe=watchdog.is_failsafe_active)

                # 6. Render Vehicle Boxes + Live HUD
                vis = frame.copy()
                for det in dets:
                    bbox = det.get("bbox")
                    if bbox:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        tid = det.get("track_id", "")
                        cls = det.get("class_name", "")
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 120), 2)
                        cv2.putText(vis, f"#{tid} {cls}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)

                vis = hud.render(vis, controller_state=ctrl, snapshot=snap, timestamp=now_ts)
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                video_viewport.image(vis_rgb, channels="RGB", use_container_width=True)

                # 7. Update Live Metrics
                with status_box:
                    if watchdog.is_failsafe_active:
                        st.error(health_msg)
                    else:
                        st.success(f"{health_msg} | Control: **{ctrl.active_phase_id} ({ctrl.active_aspect.value.upper()})** | {now_ts:.0f}s / {live_dur}s")

                with kpi_box:
                    render_executive_kpi_card_row([
                        {
                            "header": "Active Phase",
                            "value": f"{ctrl.active_phase_id} {ctrl.active_aspect.value.upper()}",
                            "delta": "REAL-TIME ASPECT",
                            "delta_positive": ctrl.active_aspect == SignalAspect.GREEN,
                            "icon": "🚦",
                            "icon_bg": "rgba(0, 240, 255, 0.12)",
                            "icon_color": "#00f0ff",
                        },
                        {
                            "header": "Current Queue",
                            "value": f"{snap.total_halted_pcu:.1f} PCU",
                            "delta": "HALTED DEMAND",
                            "delta_positive": snap.total_halted_pcu < 10.0,
                            "icon": "⚡",
                            "icon_bg": "rgba(0, 255, 157, 0.12)",
                            "icon_color": "#00ff9d",
                        },
                        {
                            "header": "Violations",
                            "value": f"{viol_detector.total_violations}",
                            "delta": "RED-LIGHT RUNNERS",
                            "delta_positive": viol_detector.total_violations == 0,
                            "icon": "🚨",
                            "icon_bg": "rgba(255, 51, 102, 0.12)",
                            "icon_color": "#ff3366",
                        },
                        {
                            "header": "Camera Link",
                            "value": f"{worker.get_metrics()['effective_fps']:.0f} FPS",
                            "delta": "EDGE STREAM",
                            "delta_positive": True,
                            "icon": "📹",
                            "icon_bg": "rgba(255, 170, 0, 0.12)",
                            "icon_color": "#ffaa00",
                        },
                    ])

                if new_viols:
                    with viol_alert:
                        v = new_viols[-1]
                        st.error(f"🚨 **RED-LIGHT VIOLATION:** Track #{v.track_id} ({v.vehicle_class.upper()}) at {v.speed_km_h:.1f} km/h!")

            st.success(f"✅ Live edge stream completed ({now_ts:.1f}s processed). Telemetry and violations logged to SQLite.")

    except Exception as exc:
        st.error(f"Live streaming interrupted: {exc}")
    finally:
        bridge.close()

# -------------------------------------------------------------
# BATCH VIDEO ANALYSIS & DELAY REPORT (Analytical Mode)
# -------------------------------------------------------------
if run_batch:
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
                st.caption("Live HUD stamps 3-aspect signal lights, speed vectors, and AI decision ticker.")

        with g_col:
            st.markdown("##### Synchronized Real-Time Queue Dynamics")
            if getattr(counts, "actuator_events", None) or getattr(counts, "kinematic_snapshots", None):
                st.altair_chart(charts.actuation_timeline_chart(counts), use_container_width=True)
            else:
                st.altair_chart(charts.pcu_chart(counts), use_container_width=True)

    # Operational Ledger Tabs
    tab_spatial, tab_ledger, tab_plan, tab_viols, tab_export = st.tabs([
        "📐 Spatial Queue & Approach Backlog",
        "⚡ Second-by-Second Actuation Log",
        "🚦 Dual-Regime Signal Plan",
        "🚨 Red-Light Enforcement & Evidence",
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

    with tab_viols:
        recorded_viols = query_violations(limit=50)
        if recorded_viols:
            st.markdown("##### 🚨 Monitored Stopline Infractions (SQLite Audit Store)")
            v_df = pd.DataFrame(recorded_viols)
            display_cols = ["id", "timestamp", "approach", "track_id", "vehicle_class", "speed_km_h", "image_path"]
            available_cols = [c for c in display_cols if c in v_df.columns]
            st.dataframe(v_df[available_cols], width="stretch", hide_index=True)

            st.markdown("##### Photographic Evidence Gallery")
            img_cols = st.columns(min(3, len(recorded_viols)))
            for idx, r_viol in enumerate(recorded_viols[:3]):
                p_img = Path(r_viol.get("image_path", ""))
                if p_img.exists():
                    with img_cols[idx]:
                        st.image(str(p_img), caption=f"ID #{r_viol.get('track_id')} ({r_viol.get('vehicle_class')}) @ {r_viol.get('speed_km_h')} km/h")
        else:
            st.info("🟢 No red-light violations detected. Monitored stoplines are currently clear of infractions.")

    with tab_export:
        st.download_button(
            "Download Vehicle Counts (CSV)",
            data=counts_csv(counts).encode("utf-8"),
            file_name="vehicle_counts.csv",
            mime="text/csv",
            key="dl_counts_tab",
        )

show_disclaimer()
