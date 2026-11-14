"""Page 6 - Live RTSP / IP Camera Ingestion & NEMA TS2 Actuator Studio."""

from __future__ import annotations

import time
from typing import Optional

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from app import _bootstrap  # noqa: F401

import pandas as pd
import streamlit as st

from app.components.state import init_state
from app.components.theme import apply_theme, render_breadcrumb, render_header
from src.decision.fail_safe import HealthState
from src.inference.live_cctv_worker import (
    LiveCCTVWorker,
    LiveCCTVWorkerConfig,
)
from src.inference.rtsp_stream import (
    StreamSourceConfig,
    StreamSourceType,
)
from src.sumo.nema_network import NEMA_8_MOVEMENTS

init_state()
apply_theme()

render_breadcrumb("OPERATIONS > LIVE CCTV & RTSP STREAM STUDIO")
render_header(
    title="Live CCTV / RTSP Stream Ingestion & Dual-Ring Actuator",
    subtitle="Zero-latency frame-dropping circular buffer connected to YOLOv8 tracking and NEMA TS2 Dual-Ring AI actuation.",
    badge_label="LIVE RTSP INGESTION",
    badge_color="teal",
)

# ---------------------------------------------------------------------------
# Sidebar: Stream Source Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📡 Camera Ingestion Settings")

    source_choice = st.radio(
        "Camera Source",
        options=[
            "Mock Intersection CCTV (Multi-Quadrant Simulation)",
            "Local USB / Webcam (Device 0)",
            "Custom Network RTSP / IP Camera",
        ],
        index=0,
    )

    rtsp_url = ""
    if source_choice == "Custom Network RTSP / IP Camera":
        rtsp_url = st.text_input(
            "RTSP Stream URL",
            value="rtsp://192.168.1.100:554/live",
            help="Enter a valid rtsp:// or http:// video stream URI.",
        )

    target_fps = st.slider("Target Ingestion FPS", min_value=10, max_value=30, value=20, step=5)
    policy_mode = st.selectbox("AI Actuation Policy", options=["max_pressure", "fixed", "ppo"], index=0)

    st.markdown("---")

    col_btn1, col_btn2 = st.columns(2)
    start_btn = col_btn1.button("▶ Start Stream", use_container_width=True, type="primary")
    stop_btn = col_btn2.button("⏹ Stop Stream", use_container_width=True)

# ---------------------------------------------------------------------------
# Session State Lifecycle Management
# ---------------------------------------------------------------------------
if "cctv_worker" not in st.session_state:
    st.session_state["cctv_worker"] = None

# Handle Stop
if stop_btn and st.session_state["cctv_worker"] is not None:
    st.session_state["cctv_worker"].stop()
    st.session_state["cctv_worker"] = None
    st.success("CCTV Ingestion Stream stopped.")
    st.rerun()

# Handle Start
if start_btn:
    if st.session_state["cctv_worker"] is not None:
        st.session_state["cctv_worker"].stop()
        st.session_state["cctv_worker"] = None

    if source_choice.startswith("Mock"):
        src_type = StreamSourceType.MOCK
        uri = "mock://intersection_quadrant_cctv"
        cam_idx = 0
    elif source_choice.startswith("Local"):
        src_type = StreamSourceType.WEBCAM
        uri = "0"
        cam_idx = 0
    else:
        src_type = StreamSourceType.RTSP
        uri = rtsp_url.strip()
        cam_idx = 0

    stream_cfg = StreamSourceConfig(
        source_type=src_type,
        uri=uri,
        webcam_index=cam_idx,
        target_fps=float(target_fps),
        frame_width=1280,
        frame_height=720,
    )

    worker_cfg = LiveCCTVWorkerConfig(
        stream_config=stream_cfg,
        use_detector=True,
        policy_mode=policy_mode,
        hud_overlay=True,
        target_fps=float(target_fps),
        fail_safe_timeout_s=3.0,
    )

    worker = LiveCCTVWorker(worker_cfg)
    worker.start()
    st.session_state["cctv_worker"] = worker
    st.success(f"Started ingestion from: {uri}")
    time.sleep(0.3)
    st.rerun()

worker: Optional[LiveCCTVWorker] = st.session_state["cctv_worker"]
is_streaming = worker is not None and worker._running

# ---------------------------------------------------------------------------
# Top Executive Stream Telemetry Bar
# ---------------------------------------------------------------------------
if is_streaming and worker is not None:
    telem = worker.get_telemetry()
    t1, t2, t3, t4, t5 = st.columns(5)

    t1.metric("Capture FPS", f"{telem.stream_telemetry.capture_fps:.1f} FPS", "Zero Socket Lag")
    t2.metric("Latency", f"{telem.stream_telemetry.frame_latency_ms:.1f} ms", "<30ms Freshness")
    t3.metric("Drop Rate", f"{telem.stream_telemetry.drop_rate_pct:.1f}%", f"{telem.stream_telemetry.total_dropped_frames} dropped")
    t4.metric("Inference FPS", f"{telem.fps_inference:.1f} FPS", f"{telem.active_vehicles_count} vehicles tracked")

    health_label = "HEALTHY (LIVE AI)" if telem.health_state == HealthState.NORMAL else "FAILSAFE (FIXED)"
    t5.metric("Watchdog Health", health_label, delta="Auto-Recovering" if telem.health_state == HealthState.NORMAL else "Feed Loss")
else:
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Capture FPS", "0.0 FPS", "Standby")
    t2.metric("Latency", "0.0 ms", "Offline")
    t3.metric("Drop Rate", "0.0%", "Queue Empty")
    t4.metric("Inference FPS", "0.0 FPS", "Idle")
    t5.metric("Watchdog Health", "STANDBY", delta="Awaiting Start")

st.markdown("---")

# ---------------------------------------------------------------------------
# Main Tabs: 1. Live Viewport | 2. Emergency Preemption | 3. Telemetry & Queues
# ---------------------------------------------------------------------------
tab_view, tab_preempt, tab_charts = st.tabs([
    "📹 Live Viewport & NEMA HUD",
    "🚨 Emergency Vehicle Preemption Dispatcher",
    "📊 8-Phase Telemetry & Metric Queues",
])

# ---------------------------------------------------------------------------
# TAB 1: Live Viewport & NEMA HUD
# ---------------------------------------------------------------------------
with tab_view:
    if not is_streaming or worker is None:
        st.info("👈 Select a camera source from the sidebar and click **▶ Start Stream** to begin live ingestion.")
        st.image(
            "https://placehold.co/1280x720/0a0f1d/00f0ff?text=Live+CCTV+Stream+Standby+Press+Start",
            caption="CCTV Viewport Standby",
            use_container_width=True,
        )
    else:
        has_frame, frame_bgr, ts = worker.get_latest_frame()
        telem = worker.get_telemetry()

        if has_frame and frame_bgr is not None:
            # Convert BGR to RGB for Streamlit display
            frame_rgb = frame_bgr[:, :, ::-1]
            st.image(frame_rgb, caption=f"Live Feed: {telem.source_uri} | Capture TS: {ts:.3f}", use_container_width=True)
        else:
            st.warning("Connecting to camera stream... waiting for frames.")

        # Dual-Ring Live Status Badges
        c_r1, c_r2, c_bar = st.columns([4, 4, 3])

        with c_r1:
            p1_name = NEMA_8_MOVEMENTS.get(telem.ring1_phase, f"Phase {telem.ring1_phase}")
            st.markdown(f"""
            <div style="background: #0d1424; border: 2px solid #00f0ff; border-radius: 8px; padding: 12px;">
                <div style="font-weight: 700; color: #00f0ff;">🔵 RING 1: {p1_name}</div>
                <div style="font-size: 1.1rem; margin-top: 4px; color: #fff;">
                    Status: <b>{telem.ring1_interval}</b> ({telem.ring1_elapsed_s:.1f}s elapsed)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_r2:
            p2_name = NEMA_8_MOVEMENTS.get(telem.ring2_phase, f"Phase {telem.ring2_phase}")
            st.markdown(f"""
            <div style="background: #0d1424; border: 2px solid #10b981; border-radius: 8px; padding: 12px;">
                <div style="font-weight: 700; color: #10b981;">🟢 RING 2: {p2_name}</div>
                <div style="font-size: 1.1rem; margin-top: 4px; color: #fff;">
                    Status: <b>{telem.ring2_interval}</b> ({telem.ring2_elapsed_s:.1f}s elapsed)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_bar:
            st.markdown(f"""
            <div style="background: #0d1424; border: 2px solid #8b5cf6; border-radius: 8px; padding: 12px;">
                <div style="font-weight: 700; color: #8b5cf6;">❚❚ BARRIER & TLS MATRIX</div>
                <div style="font-size: 0.95rem; margin-top: 4px; color: #fff;">
                    Stage: <b>{telem.current_stage}</b><br>
                    SUMO TLS: <code style="color: #00f0ff;">{telem.tls_state_12}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Refresh trigger button
        if st.button("🔄 Poll / Refresh Viewport Frame", use_container_width=True):
            st.rerun()

# ---------------------------------------------------------------------------
# TAB 2: Emergency Preemption Dispatcher
# ---------------------------------------------------------------------------
with tab_preempt:
    st.markdown("### 🚨 High-Priority Emergency Vehicle Preemption")
    st.caption("Immediately forces crossing traffic through yellow/all-red clearance intervals and opens an exclusive green wave corridor.")

    if is_streaming and worker is not None:
        telem = worker.get_telemetry()
        if telem.active_preemption:
            preempt_phase_name = NEMA_8_MOVEMENTS.get(telem.active_preemption, f"Phase {telem.active_preemption}")
            st.error(f"🚑 EMERGENCY PREEMPTION IN PROGRESS: Priority Green Wave granted for **{preempt_phase_name}**!")
            if st.button("🛑 Clear Emergency Preemption Wave", type="primary", use_container_width=True):
                worker.clear_preemption()
                st.success("Emergency preemption cleared.")
                time.sleep(0.2)
                st.rerun()
        else:
            st.success("✅ Preemption System Standby: All 8 approaches operating under normal adaptive dual-ring control.")

        st.markdown("#### Dispatch Emergency Wave by Movement Approach:")
        row1_cols = st.columns(4)
        row2_cols = st.columns(4)

        # Phases 1..4 (Row 1)
        for i, phase_id in enumerate([1, 2, 3, 4]):
            p_name = NEMA_8_MOVEMENTS.get(phase_id, f"Phase {phase_id}")
            if row1_cols[i].button(f"🚑 Dispatch\n{p_name}", key=f"preempt_{phase_id}", use_container_width=True):
                worker.request_preemption(phase_id)
                st.warning(f"Dispatched Emergency Preemption on Phase {phase_id}!")
                time.sleep(0.2)
                st.rerun()

        # Phases 5..8 (Row 2)
        for i, phase_id in enumerate([5, 6, 7, 8]):
            p_name = NEMA_8_MOVEMENTS.get(phase_id, f"Phase {phase_id}")
            if row2_cols[i].button(f"🚑 Dispatch\n{p_name}", key=f"preempt_{phase_id}", use_container_width=True):
                worker.request_preemption(phase_id)
                st.warning(f"Dispatched Emergency Preemption on Phase {phase_id}!")
                time.sleep(0.2)
                st.rerun()
    else:
        st.info("Start camera ingestion stream to enable the live emergency preemption dispatcher.")

# ---------------------------------------------------------------------------
# TAB 3: Real-Time Telemetry & Metric Queues
# ---------------------------------------------------------------------------
with tab_charts:
    st.markdown("### 📊 8-Phase Metric Queues & PCU Demand Pressure")

    if is_streaming and worker is not None:
        telem = worker.get_telemetry()

        # Build Dataframe for 8 movements
        chart_data = []
        for pid in range(1, 9):
            chart_data.append({
                "Phase": f"Φ{pid} {NEMA_8_MOVEMENTS.get(pid, '')}",
                "Queue (m)": telem.queues_m.get(pid, 0.0),
                "PCU Demand": telem.pcu_counts.get(pid, 0.0),
            })
        df_metrics = pd.DataFrame(chart_data)

        c_q, c_pcu = st.columns(2)
        with c_q:
            st.markdown("#### Metric Queue Length ($m$) per Movement")
            st.bar_chart(df_metrics.set_index("Phase")["Queue (m)"], color="#ef4444")

        with c_pcu:
            st.markdown("#### PCU Vehicular Demand per Movement")
            st.bar_chart(df_metrics.set_index("Phase")["PCU Demand"], color="#00f0ff")

        st.markdown("#### 🧠 Live RL Decision Rationale")
        st.markdown(f"""
        <div style="background: #0d1424; border-left: 4px solid #00f0ff; border-radius: 4px; padding: 12px;">
            <div style="font-weight: 700; color: #00f0ff; font-size: 1.05rem;">
                Action: {telem.latest_action_name} | Step Reward: {telem.latest_reward:+.3f}
            </div>
            <div style="color: #cbd5e1; margin-top: 6px; font-size: 0.95rem;">
                Rationale: <i>{telem.latest_rationale}</i>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Start camera stream to view live queue depth distributions and RL rationale.")
