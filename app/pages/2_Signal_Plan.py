"""Page 2 - Signal Plan: baseline vs adaptive optimization with delay metrics."""

from __future__ import annotations

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from app import _bootstrap  # noqa: F401

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.cards import (
    render_executive_kpi_card_row,
    render_signal_light_widget,
    show_disclaimer,
)
from app.components.state import get_intersection, init_state
from app.components.theme import apply_theme, render_breadcrumb, render_header
from app.utils import delay_csv, delay_records, plan_csv, plan_pdf, plan_records
from src import config
from src.decision.dual_regime import OptimizationRegime
from src.decision.intersection import FixedPlan
from src.inference.analyze_video import plan_from_counts

init_state()
apply_theme()

# Top Breadcrumb & Console Header (Image 1 style)
render_breadcrumb("OPERATIONS > SIGNAL TIMING OPTIMIZER")
render_header(
    title="Signal Timing Plan & Delay Benchmarks",
    subtitle="Comparative delay analysis: Fixed-time baseline vs. Dual-Regime adaptive allocation",
    chips=[
        ("OPTIMIZER: DUAL-REGIME", "chip-green"),
        ("DELAY MODEL: WEBSTER THREE-TERM", "chip-cyan"),
    ],
)

counts = st.session_state.get("counts")
if counts is None:
    st.info("No video counts loaded yet. Run analysis on **1 · Intersection Analysis** first to produce counts.")
    st.stop()

intersection = get_intersection(st.session_state["intersection_path"])

st.subheader("Timing Parameters")
p1, p2, p3 = st.columns([1, 1, 1])
window = p1.number_input(
    "Aggregation Window (s)",
    min_value=10.0,
    value=float(round(counts.duration_s)) if counts.duration_s > 0 else 60.0,
    step=10.0,
)
fixed_cycle = p2.number_input("Fixed Baseline Cycle (s)", min_value=30.0, value=90.0, step=5.0)
p3.caption("Fixed plan divides cycle evenly across approaches as the reference baseline.")

fixed_green = {phase.id: fixed_cycle / len(intersection.phases) for phase in intersection.phases}
fixed_plan = FixedPlan(cycle=fixed_cycle, green=fixed_green)

try:
    plan, comparison, flows = plan_from_counts(
        counts, intersection, window_s=window, fixed_plan=fixed_plan
    )
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not compute plan: {exc}")
    st.stop()

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
elif plan.oversaturated:
    st.warning(
        "Intersection is oversaturated (Y >= 1): maximum practical cycle is enforced as safety bound.",
        icon="🚧",
    )

st.divider()

# Executive KPI Stat Tiles (Image 1 style)
render_executive_kpi_card_row([
    {
        "header": "Critical Flow Ratio Y",
        "value": f"{plan.critical_flow_ratio:.3f}",
        "delta": "SUM (v_i / s_i)",
        "delta_positive": plan.critical_flow_ratio < 0.85,
        "icon": "📈",
        "icon_bg": "rgba(0, 240, 255, 0.12)",
        "icon_color": "#00f0ff",
    },
    {
        "header": "Optimized Cycle",
        "value": f"{plan.cycle:.1f} s",
        "delta": "WEBSTER OPTIMUM",
        "delta_positive": True,
        "icon": "⏱️",
        "icon_bg": "rgba(0, 255, 157, 0.12)",
        "icon_color": "#00ff9d",
    },
    {
        "header": "Fixed Baseline Delay",
        "value": f"{comparison.fixed_avg_delay:.2f} s",
        "delta": "UNTIMED REFERENCE",
        "delta_positive": False,
        "icon": "⚠️",
        "icon_bg": "rgba(255, 170, 0, 0.12)",
        "icon_color": "#ffaa00",
    },
    {
        "header": "Adaptive Delay",
        "value": f"{comparison.optimized_avg_delay:.2f} s",
        "delta": f"-{comparison.improvement_s:.2f}s ({comparison.reduction_pct:.1f}%)",
        "delta_positive": True,
        "icon": "⚡",
        "icon_bg": "rgba(255, 51, 102, 0.12)",
        "icon_color": "#ff3366",
    },
])

render_signal_light_widget(plan, intersection)

st.divider()

# Allocation & Delay Reduction Tabs
tab_alloc, tab_delay, tab_congestion = st.tabs([
    "⏱️ Green Allocation Table",
    "📊 Delay Comparison Chart",
    "🚦 Approach Congestion Index",
])

with tab_alloc:
    plan_df = pd.DataFrame(plan_records(plan, intersection))
    plan_df["share_%"] = (plan_df["green_s"] / plan.cycle * 100).round(1)
    st.dataframe(plan_df, width="stretch", hide_index=True)

    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        st.download_button(
            "Download Signal Plan (CSV)",
            plan_csv(plan, intersection).encode("utf-8"),
            file_name="signal_plan.csv",
            mime="text/csv",
            key="dl_plan",
        )
    with c_dl2:
        st.download_button(
            "Download Plan & Delay Report (PDF)",
            plan_pdf(plan, comparison, intersection, counts=counts),
            file_name="signal_plan.pdf",
            mime="application/pdf",
            key="dl_plan_pdf",
        )

with tab_delay:
    st.altair_chart(charts.delay_chart(comparison, intersection), use_container_width=True)
    delay_df = pd.DataFrame(delay_records(comparison))
    st.dataframe(delay_df, width="stretch", hide_index=True)

with tab_congestion:
    approach_flow = {a: pcu * 3600.0 / window for a, pcu in counts.pcu_by_approach.items()}
    st.dataframe(
        pd.DataFrame(charts.congestion_rows(approach_flow, intersection)),
        width="stretch",
        hide_index=True,
    )

show_disclaimer()
