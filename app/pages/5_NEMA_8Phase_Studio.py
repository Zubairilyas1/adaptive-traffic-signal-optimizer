"""Page 5 - NEMA TS2 Dual-Ring 8-Phase Studio & Actuation Visualizer."""

from __future__ import annotations

from typing import Any

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from app import _bootstrap  # noqa: F401

import pandas as pd
import streamlit as st

from app.components.state import init_state
from app.components.theme import apply_theme, render_breadcrumb, render_header, render_kpi_grid
from src.decision.nema_dual_ring import (
    NEMADecision,
    NEMADualRingController,
    NEMAPhase,
    PhaseInterval,
    RingBarrierStage,
    are_phases_compatible,
)
from src.reports.audit_report import MunicipalAuditData, generate_municipal_pdf_report
from src.rl.nema_agent import (
    NEMAAction,
    NEMADualRingAgent,
    NEMAPolicyDecision,
)
from src.sumo.nema_network import NEMA_8_MOVEMENTS, build_nema_tls_state_string

init_state()
apply_theme()

render_breadcrumb("OPERATIONS > NEMA TS2 DUAL-RING STUDIO")
render_header(
    title="NEMA TS2 Dual-Ring 8-Phase Controller Studio",
    subtitle="Commercial-grade municipal dual-ring barrier actuation with dedicated protected left-turn movements and emergency preemption.",
    badge_label="NEMA TS2 STANDARD",
    badge_color="teal",
)

# ---------------------------------------------------------------------------
# Session State Initialization for NEMA Controller
# ---------------------------------------------------------------------------
if "nema_agent" not in st.session_state:
    st.session_state["nema_agent"] = NEMADualRingAgent(policy_mode="max_pressure")
    st.session_state["nema_history"] = []
    st.session_state["nema_sim_time"] = 0.0

agent: NEMADualRingAgent = st.session_state["nema_agent"]

# ---------------------------------------------------------------------------
# Top Executive Metrics Row (Responsive Auto-Fit Grid)
# ---------------------------------------------------------------------------
curr_dec = agent.controller
stage_label = "Stage 1 (Major N-S)" if curr_dec.current_stage == RingBarrierStage.STAGE_1_MAJOR_NS else "Stage 2 (Minor E-W)"

render_kpi_grid([
    {
        "label": "Barrier Stage",
        "value": stage_label,
        "delta": "Coordinated" if curr_dec.ring1_phase in (2, 4) else "Protected Lefts",
        "delta_positive": True,
        "icon": "🛡️",
    },
    {
        "label": "Ring 1 Active Movement",
        "value": f"Phase {curr_dec.ring1_phase.value} · {curr_dec.ring1_interval.value}",
        "delta": f"{curr_dec.ring1_elapsed_s:.1f}s Elapsed",
        "delta_positive": True,
        "icon": "🔵",
    },
    {
        "label": "Ring 2 Active Movement",
        "value": f"Phase {curr_dec.ring2_phase.value} · {curr_dec.ring2_interval.value}",
        "delta": f"{curr_dec.ring2_elapsed_s:.1f}s Elapsed",
        "delta_positive": True,
        "icon": "🟣",
    },
    {
        "label": "Barrier Synchronization",
        "value": "Holding at Barrier" if curr_dec.ring1_interval == PhaseInterval.RED_HOLD or curr_dec.ring2_interval == PhaseInterval.RED_HOLD else "Concurrent Flow",
        "delta": "Zero-Conflict Lock",
        "delta_positive": True,
        "icon": "🔒",
    },
])

st.markdown("---")

# ---------------------------------------------------------------------------
# Main Tabs: 1. Dual-Ring Visualizer | 2. Demand Injector | 3. Signal Matrix & PDF
# ---------------------------------------------------------------------------
tab_viz, tab_demand, tab_matrix = st.tabs([
    "🚦 Dual-Ring Barrier Visualizer & Live Actuator",
    "🎛️ Demand Injector & Emergency Preemption",
    "📑 12-Link SUMO Matrix & Municipal PDF Audit",
])

# ---------------------------------------------------------------------------
# TAB 1: Dual-Ring Visualizer
# ---------------------------------------------------------------------------
with tab_viz:
    st.markdown("### 🚦 NEMA TS2 Dual-Ring Barrier State Machine")
    st.caption("Parallel rings execute independently within each stage, but must cross the barrier simultaneously to guarantee zero conflicting movements.")

    col_r1, col_bar, col_r2 = st.columns([5, 1, 5])

    def _interval_color(interval: PhaseInterval) -> str:
        if interval == PhaseInterval.GREEN:
            return "#00f0ff"
        elif interval == PhaseInterval.YELLOW_CHANGE:
            return "#facc15"
        elif interval == PhaseInterval.ALL_RED_CLEARANCE:
            return "#ef4444"
        else:
            return "#475569"

    with col_r1:
        st.markdown("#### 🔵 Ring 1 (Top)")
        p1 = curr_dec.ring1_phase
        int1 = curr_dec.ring1_interval
        cfg1 = curr_dec.configs[p1]
        c1 = _interval_color(int1)

        st.markdown(f"""
        <div style="background: #0d1424; border: 2px solid {c1}; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #ffffff;">Phase {p1.value}: {p1.name}</span>
                <span style="background: {c1}; color: #000; font-weight: 800; padding: 4px 8px; border-radius: 4px; font-size: 0.85rem;">{int1.value}</span>
            </div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 8px;">
                Elapsed: <b>{curr_dec.ring1_elapsed_s:.1f}s</b> / Min: {cfg1.min_green_s}s / Max: {cfg1.max_green_s}s
            </p>
            <div style="background: #1e293b; height: 10px; border-radius: 5px; overflow: hidden; margin-top: 8px;">
                <div style="background: {c1}; width: {min(100.0, (curr_dec.ring1_elapsed_s / cfg1.max_green_s) * 100):.1f}%; height: 100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Ring 1 Sequential Loop:** `Phase 1 (SB Left)` ➔ `Phase 2 (NB Thru)` ❚❚ **[BARRIER]** ❚❚ `Phase 3 (WB Left)` ➔ `Phase 4 (EB Thru)`")

    with col_bar:
        st.markdown("<div style='text-align: center; padding-top: 36px;'>", unsafe_allow_html=True)
        if curr_dec.ring1_interval == PhaseInterval.RED_HOLD or curr_dec.ring2_interval == PhaseInterval.RED_HOLD:
            st.markdown(
                """<div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 8px; padding: 12px 6px; text-align: center; color: #fca5a5; font-size: 0.8rem; font-weight: 700; line-height: 1.4;">
                ❚❚<br>BARRIER<br>LOCKED
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 8px; padding: 12px 6px; text-align: center; color: #6ee7b7; font-size: 0.8rem; font-weight: 700; line-height: 1.4;">
                ❚❚<br>STAGE<br>ACTIVE
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r2:
        st.markdown("#### 🟣 Ring 2 (Bottom)")
        p2 = curr_dec.ring2_phase
        int2 = curr_dec.ring2_interval
        cfg2 = curr_dec.configs[p2]
        c2 = _interval_color(int2)

        st.markdown(f"""
        <div style="background: #0d1424; border: 2px solid {c2}; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #ffffff;">Phase {p2.value}: {p2.name}</span>
                <span style="background: {c2}; color: #000; font-weight: 800; padding: 4px 8px; border-radius: 4px; font-size: 0.85rem;">{int2.value}</span>
            </div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 8px;">
                Elapsed: <b>{curr_dec.ring2_elapsed_s:.1f}s</b> / Min: {cfg2.min_green_s}s / Max: {cfg2.max_green_s}s
            </p>
            <div style="background: #1e293b; height: 10px; border-radius: 5px; overflow: hidden; margin-top: 8px;">
                <div style="background: {c2}; width: {min(100.0, (curr_dec.ring2_elapsed_s / cfg2.max_green_s) * 100):.1f}%; height: 100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Ring 2 Sequential Loop:** `Phase 5 (NB Left)` ➔ `Phase 6 (SB Thru)` ❚❚ **[BARRIER]** ❚❚ `Phase 7 (EB Left)` ➔ `Phase 8 (WB Thru)`")

    st.markdown("---")

    # Step Execution Controls
    c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns([2, 2, 2, 2])

    with c_ctrl1:
        if st.button("▶️ Step Controller (1.0s)", use_container_width=True):
            # Extract demand from session state or defaults
            q_m = st.session_state.get("nema_queues", {p: 15.0 for p in NEMAPhase})
            pcu = st.session_state.get("nema_pcu", {p: 10.0 for p in NEMAPhase})
            hway = st.session_state.get("nema_headway", {p.value: 2.0 for p in NEMAPhase})
            preempt = st.session_state.get("nema_preemption", None)

            dec = agent.step(dt=1.0, queues_m=q_m, pcu_counts=pcu, headways=hway, preemption_call=preempt)
            st.session_state["nema_sim_time"] += 1.0
            st.session_state["nema_history"].append(dec.to_dict())
            st.rerun()

    with c_ctrl2:
        if st.button("⏩ Run 10s Fast-Forward", use_container_width=True):
            q_m = st.session_state.get("nema_queues", {p: 15.0 for p in NEMAPhase})
            pcu = st.session_state.get("nema_pcu", {p: 10.0 for p in NEMAPhase})
            hway = st.session_state.get("nema_headway", {p.value: 2.0 for p in NEMAPhase})
            preempt = st.session_state.get("nema_preemption", None)

            for _ in range(10):
                dec = agent.step(dt=1.0, queues_m=q_m, pcu_counts=pcu, headways=hway, preemption_call=preempt)
                st.session_state["nema_sim_time"] += 1.0
                st.session_state["nema_history"].append(dec.to_dict())
            st.rerun()

    with c_ctrl3:
        if st.button("🔄 Reset Controller State", use_container_width=True):
            st.session_state["nema_agent"] = NEMADualRingAgent(policy_mode="max_pressure")
            st.session_state["nema_history"] = []
            st.session_state["nema_sim_time"] = 0.0
            st.session_state["nema_preemption"] = None
            st.rerun()

    with c_ctrl4:
        st.info(f"Sim Time: **{st.session_state['nema_sim_time']:.0f}s** | Steps: **{len(st.session_state['nema_history'])}**")

    # Recent Telemetry Log Table
    if st.session_state["nema_history"]:
        st.markdown("#### 📊 Live Closed-Loop Decision Log")
        df_hist = pd.DataFrame(st.session_state["nema_history"][-8:])
        st.dataframe(df_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: Demand Injector & Preemption
# ---------------------------------------------------------------------------
with tab_demand:
    st.markdown("### 🎛️ 8-Movement Traffic Demand & Queue Simulator")
    st.caption("Adjust real-time queues and vehicle arrivals across all 8 standard NEMA movements to test dynamic gap-out and barrier coordination.")

    c_d1, c_d2 = st.columns(2)

    if "nema_queues" not in st.session_state:
        st.session_state["nema_queues"] = {p: 15.0 for p in NEMAPhase}
    if "nema_pcu" not in st.session_state:
        st.session_state["nema_pcu"] = {p: 10.0 for p in NEMAPhase}
    if "nema_headway" not in st.session_state:
        st.session_state["nema_headway"] = {p.value: 1.8 for p in NEMAPhase}

    with c_d1:
        st.markdown("#### Stage 1 Movements: Major Street (North-South)")
        q1 = st.slider("Phase 1: SB Left Queue (meters)", 0.0, 80.0, 15.0, key="slider_q1")
        q2 = st.slider("Phase 2: NB Through Queue (meters)", 0.0, 100.0, 35.0, key="slider_q2")
        q5 = st.slider("Phase 5: NB Left Queue (meters)", 0.0, 80.0, 12.0, key="slider_q5")
        q6 = st.slider("Phase 6: SB Through Queue (meters)", 0.0, 100.0, 30.0, key="slider_q6")

        st.session_state["nema_queues"][NEMAPhase.P1_SB_LEFT] = q1
        st.session_state["nema_queues"][NEMAPhase.P2_NB_THROUGH] = q2
        st.session_state["nema_queues"][NEMAPhase.P5_NB_LEFT] = q5
        st.session_state["nema_queues"][NEMAPhase.P6_SB_THROUGH] = q6

    with c_d2:
        st.markdown("#### Stage 2 Movements: Minor Street (East-West)")
        q3 = st.slider("Phase 3: WB Left Queue (meters)", 0.0, 80.0, 10.0, key="slider_q3")
        q4 = st.slider("Phase 4: EB Through Queue (meters)", 0.0, 80.0, 20.0, key="slider_q4")
        q7 = st.slider("Phase 7: EB Left Queue (meters)", 0.0, 80.0, 8.0, key="slider_q7")
        q8 = st.slider("Phase 8: WB Through Queue (meters)", 0.0, 80.0, 18.0, key="slider_q8")

        st.session_state["nema_queues"][NEMAPhase.P3_WB_LEFT] = q3
        st.session_state["nema_queues"][NEMAPhase.P4_EB_THROUGH] = q4
        st.session_state["nema_queues"][NEMAPhase.P7_EB_LEFT] = q7
        st.session_state["nema_queues"][NEMAPhase.P8_WB_THROUGH] = q8

    st.markdown("---")
    st.markdown("### 🚑 Emergency Vehicle Preemption Priority Dispatcher")
    st.caption("Instantly seizes control from standard actuation, expediting clearance to open a safe green wave corridor.")

    c_pre1, c_pre2, c_pre3 = st.columns([3, 2, 2])

    with c_pre1:
        target_p = st.selectbox(
            "Select Emergency Approach Corridor:",
            [
                (NEMAPhase.P3_WB_LEFT.value, "Phase 3: Westbound Left-Turn Corridor"),
                (NEMAPhase.P4_EB_THROUGH.value, "Phase 4: Eastbound Arterial Through"),
                (NEMAPhase.P2_NB_THROUGH.value, "Phase 2: Northbound Arterial Through"),
                (NEMAPhase.P1_SB_LEFT.value, "Phase 1: Southbound Left-Turn Corridor"),
            ],
            format_func=lambda x: x[1],
        )

    with c_pre2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🚨 Dispatch Priority", type="primary", use_container_width=True):
            st.session_state["nema_preemption"] = target_p[0]
            st.warning(f"🚨 Preemption Priority Called for Phase {target_p[0]}!")
            st.rerun()

    with c_pre3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("✅ Clear Preemption", use_container_width=True):
            st.session_state["nema_preemption"] = None
            st.success("Preemption cleared. Resuming normal dual-ring actuation.")
            st.rerun()

# ---------------------------------------------------------------------------
# TAB 3: 12-Link SUMO Signal Matrix & Municipal Audit PDF
# ---------------------------------------------------------------------------
with tab_matrix:
    st.markdown("### 🚦 12-Link Microscopic SUMO Signal State Matrix")
    st.caption("Shows the instantaneous signal aspect applied across all 12 junction turning links in the microscopic SUMO simulator.")

    tls_str = build_nema_tls_state_string(
        ring1_phase=curr_dec.ring1_phase,
        ring1_interval=curr_dec.ring1_interval,
        ring2_phase=curr_dec.ring2_phase,
        ring2_interval=curr_dec.ring2_interval,
    )

    cols_grid = st.columns(6)
    link_names = [
        "Link 0: N2C Right (P2)", "Link 1: N2C Thru (P2)", "Link 2: N2C Left (P5)",
        "Link 3: S2C Right (P6)", "Link 4: S2C Thru (P6)", "Link 5: S2C Left (P1)",
        "Link 6: E2C Right (P4)", "Link 7: E2C Thru (P4)", "Link 8: E2C Left (P7)",
        "Link 9: W2C Right (P8)", "Link 10: W2C Thru (P8)", "Link 11: W2C Left (P3)",
    ]

    for idx, (ch, name) in enumerate(zip(tls_str, link_names)):
        col = cols_grid[idx % 6]
        bg_col = "#22c55e" if ch == "G" else ("#eab308" if ch == "y" else "#ef4444")
        col.markdown(f"""
        <div style="background: #0d1424; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; text-align: center; margin-bottom: 8px;">
            <div style="font-size: 0.75rem; color: #94a3b8; height: 32px; overflow: hidden;">{name}</div>
            <div style="background: {bg_col}; color: #000; font-weight: 800; font-size: 1.25rem; border-radius: 4px; margin-top: 4px;">{ch}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"**SUMO Signal String (12 Links):** `{tls_str}`")

    st.markdown("---")
    st.markdown("### 📑 Official Municipal Traffic Engineering Audit Report (PDF)")
    st.caption("Generate a certified vector PDF engineering audit document containing the NEMA TS2 dual-ring timing calculations.")

    if st.button("📥 Generate & Download Municipal NEMA Audit Report (PDF)", use_container_width=True):
        muni_data = MunicipalAuditData(
            intersection_name="Central Arterial Junction (NEMA TS2 8-Phase)",
            controller_mode=f"NEMA TS2 Dual-Ring Actuated ({stage_label})",
            cycle_length_s=curr_dec.configs[curr_dec.ring1_phase].max_green_s + curr_dec.configs[curr_dec.ring2_phase].max_green_s,
            phase_greens_s={
                f"Ring 1 P{curr_dec.ring1_phase.value}": curr_dec.ring1_elapsed_s,
                f"Ring 2 P{curr_dec.ring2_phase.value}": curr_dec.ring2_elapsed_s,
            },
            metric_queues_m={f"Phase {p.value} ({p.name})": float(st.session_state.get("nema_queues", {}).get(p, 15.0)) for p in NEMAPhase},
        )
        pdf_bytes = generate_municipal_pdf_report(muni_data)
        st.download_button(
            label="💾 Download Official PDF Timing Sheet",
            data=pdf_bytes,
            file_name="NEMA_TS2_Dual_Ring_Traffic_Audit.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
