"""OptiFlow ITS - Traffic Signal Timing Orchestrator.

Industrial-grade Streamlit entry point:
- Command Center Executive Deck
- Real-time AI component status chips
- Interactive Road Topology Explorer
- Live HUD video demonstrations
- Validated SUMO Controller Benchmarks
"""

from __future__ import annotations

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from app import _bootstrap  # noqa: F401

from pathlib import Path
import streamlit as st

from app.components.cards import show_disclaimer
from app.components.state import init_state
from app.components.theme import apply_theme, render_header, render_kpi_grid
from app.utils import demo_video_path, list_annotated_videos, list_intersections, list_sample_clips
from src import config
from src.decision.intersection import load_intersection

st.set_page_config(
    page_title="OptiFlow ITS · Traffic Signal Orchestrator",
    page_icon="🚦",
    layout="wide",
)

init_state()
apply_theme()

with st.sidebar:
    st.subheader("System Configuration")
    weights_options = {
        "Fine-tuned (7 classes)": config.BEST_WEIGHTS,
        "COCO baseline (4 classes)": config.COCO_WEIGHTS,
    }
    available = {k: v for k, v in weights_options.items() if v.exists()}
    if available:
        label = st.radio("Detector Weights", list(available.keys()), index=0)
        st.session_state["detector_weights"] = str(available[label])
    else:
        st.error("No detector weights found.")
    st.divider()
    st.caption("OptiFlow ITS Engine v2.4")
    st.caption(f"Hardware Mode: CPU Inference")

# Executive Operations Header
best_exists = config.BEST_WEIGHTS.exists()
status_chips = [
    ("DETECTOR: 7-CLASS FINE-TUNED" if best_exists else "DETECTOR: COCO BASELINE", "chip-green"),
    ("ACTUATOR: CLOSED-LOOP NEMA TS2", "chip-blue"),
    ("CALIBRATION: IPM 4-POINT HOMOGRAPHY", "chip-amber"),
    ("COORDINATION: 3-JUNCTION GREEN WAVE", "chip-green"),
]

render_header(
    title="OptiFlow ITS · Adaptive Signal Timing Orchestrator",
    subtitle="Edge-Native Computer Vision Traffic Perception, NEMA TS2 Closed-Loop Actuation, & Green-Wave Arterial Coordination",
    chips=status_chips,
)

# Core Telemetry KPI Strip (Responsive Auto-Fit Grid)
render_kpi_grid([
    {
        "label": "Tracked Classes",
        "value": f"{len(config.TARGET_CLASS_NAMES)} Types",
        "delta": "Car, Bike, Bus, Rickshaw, Truck",
        "delta_positive": True,
        "icon": "🚗",
        "help": "Full IRC:106 heterogeneous vehicle classes",
    },
    {
        "label": "Detection mAP@0.5",
        "value": "0.944" if best_exists else "0.727",
        "delta": "+0.217 vs COCO Baseline",
        "delta_positive": True,
        "icon": "🎯",
        "help": "Fine-tuned YOLOv8n validation accuracy",
    },
    {
        "label": "Actuation Precision",
        "value": "1.0 s",
        "delta": "Second-by-Second Closed-Loop",
        "delta_positive": True,
        "icon": "⏱️",
        "help": "NEMA TS2 dual-ring clearance interlock frequency",
    },
    {
        "label": "Control Regimes",
        "value": "3 Regimes",
        "delta": "Webster + Gating + Preemption",
        "delta_positive": True,
        "icon": "🛡️",
        "help": "Regime 1: Webster, Regime 2: Capacity Stretch, Regime 3: Spillback Gating",
    },
])

st.divider()

# Interactive Road Topology Explorer
st.subheader("Intersection Road Topologies")
st.caption("Select a real-world road layout to inspect geometric lane routing and phase configurations:")

topologies = {
    "↔️ Two-Way Parallel Corridor": ("twoway_corridor.json", "Parallel arterial with concurrent green phases. Unequal volumes never stop opposing lanes."),
    "🔀 3-Way T-Junction": ("t_junction.json", "High-priority main highway with demand-actuated side-street green intervals."),
    "🛣️ Single Road Corridor": ("single_road.json", "Continuous one-way corridor monitoring flow density with zero cross-street stops."),
    "➕ 4-Way Crossroad": ("fourway_crossroad.json", "Balanced 4-approach junction with cross-traffic conflict matrices and left-turn protection."),
}

topo_choice = st.selectbox("Active Geometry Preset", list(topologies.keys()), index=0)
cfg_file, desc = topologies[topo_choice]
cfg_path = config.ROOT / "configs" / "intersections" / cfg_file

if cfg_path.exists():
    inter_data = load_intersection(cfg_path)
    c_left, c_right = st.columns([1, 2])
    with c_left:
        st.markdown(f"**Operational Logic:**\n\n{desc}")
        st.metric("Total Lost Time", f"{inter_data.total_lost_time:.0f} s / cycle")
        st.metric("Controlled Phases", f"{len(inter_data.phases)} Phases")
    with c_right:
        lane_badges = []
        for p in inter_data.phases:
            app = inter_data.approaches.get(p.approach)
            label = app.label if app else p.approach
            lanes = app.lanes if app else 1
            lane_badges.append(f"- **Phase {p.id}** ({label}): `{lanes} lane(s)` · Saturation Flow: `{lanes * config.SATURATION_FLOW:.0f} PCU/h`")
        st.markdown("**Phase & Approach Allocation:**\n" + "\n".join(lane_badges))

st.divider()

# Tabbed Demonstration & Benchmark Deck
tab_demo, tab_arch, tab_benchmarks = st.tabs([
    "📹 Video Demonstration & Live HUD",
    "⚙️ System Architecture",
    "📊 Controller Benchmarks (SUMO Simulation)",
])

with tab_demo:
    annotated = list_annotated_videos()
    video_loaded = False
    if annotated:
        which = st.selectbox("Select Annotated Inspection Video", [p.name for p in annotated], key="home_ann_select")
        chosen = next((p for p in annotated if p.name == which), None)
        if chosen and chosen.exists():
            st.video(str(chosen))
            st.caption(f"Inspection video: `{chosen.name}` with live 3-aspect signal lights and AI decision ticker.")
            video_loaded = True

    if not video_loaded:
        demo = demo_video_path()
        if demo is not None and demo.exists():
            st.video(str(demo))
            st.caption("Benchmark demonstration clip with real-time HUD telemetry overlay.")
            video_loaded = True

    if not video_loaded:
        st.info("💡 **Ready to run live inspection.** Navigate to **1 · Intersection Analysis** to run real-time YOLOv8 detection or upload footage.")

with tab_arch:
    st.markdown(
        """
        ```mermaid
        flowchart LR
            V[Intersection Video] --> Y[YOLOv8 + ByteTrack]
            Y --> K[Velocity Decomposition: Stopped vs Flowing]
            K --> I[IPM 4-Point Homography: Distance in Meters]
            I --> D[Dynamic Saturation Calibrator: s = 3600 / h]
            D --> DR[Dual-Regime Engine: Webster vs Anti-Spillback Gating]
            DR --> ACT[Adaptive Actuator: Gap-Out & Dilemma Zone]
            ACT --> HUD[Live Video HUD & Signal Aspects]
        ```
        """
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            r"""
            **Analysis & Perception Layer:**
            - **Fine-tuned YOLOv8n:** 7 classes aligned to Asian mixed traffic.
            - **Velocity Decomposition:** Identifies true halted queues (< 5 km/h) vs flowing traffic.
            - **IPM Homography:** Computes physical distance backwards from stopline in **meters ($m$)**.
            """
        )
    with col_b:
        st.markdown(
            r"""
            **Decision & Control Layer:**
            - **Dual-Regime Engine:** Switches from Webster delay minimization to Gating at $Y \ge 0.95$.
            - **Closed-Loop Actuation:** Second-by-second Gap-Out, Max-Out, Dilemma Zone protection.
            - **Dynamic Headway Tracking:** Calibrates real site saturation flow from observed vehicle departures.
            """
        )

with tab_benchmarks:
    st.markdown("**Validated Controller Comparison (SUMO Micro-Simulation):**")
    bench_data = [
        {"Controller Type": "Fixed-Time Baseline", "Normal Traffic Delay": "658.5 s/veh", "Stress Traffic Delay": "472.8 s/veh", "Efficiency Gain": "Baseline (0%)"},
        {"Controller Type": "Rule-Based Dynamic Actuation", "Normal Traffic Delay": "335.6 s/veh", "Stress Traffic Delay": "107.8 s/veh", "Efficiency Gain": "+49.0% Delay Reduction"},
        {"Controller Type": "PPO Reinforcement Learning", "Normal Traffic Delay": "279.0 s/veh", "Stress Traffic Delay": "97.6 s/veh", "Efficiency Gain": "+57.6% (Peak: +79.4%)"},
    ]
    st.dataframe(bench_data, width="stretch", hide_index=True)
    st.caption("Benchmarks simulated under identical seed and volume conditions in SUMO micro-simulation environment.")

st.divider()

col_cta1, col_cta2 = st.columns([3, 1])
with col_cta1:
    st.markdown("**Ready to analyze an intersection video?** Navigate to the analysis console:")
with col_cta2:
    if st.button("Open Analysis Console →", type="primary"):
        st.switch_page("pages/1_Intersection_Analysis.py")

show_disclaimer()
