"""Small UI helpers: metric cards, engineering disclaimer, signal visualizer.

Synthesizes Image 1's dark-header KPI cards with Image 2's mission-critical palette.
"""

from __future__ import annotations

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from app import _bootstrap  # noqa: F401

import pandas as pd
import streamlit as st


DISCLAIMER_TEXT = (
    "Research prototype for autonomous signal timing estimation. "
    "All real-world deployments require validation and review by a licensed municipal traffic engineer."
)


def show_disclaimer() -> None:
    """Render a clean, professional engineering advisory notice in the footer."""
    html = f"""
    <div style="margin-top: 2.5rem; padding: 12px 16px; background-color: #0d1424; border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; font-size: 0.8rem; color: #94a3b8; display: flex; align-items: center; gap: 10px;">
        <span style="color: #ffaa00; font-size: 1.1rem;">⚠️</span>
        <span><b>Engineering Advisory Notice:</b> {DISCLAIMER_TEXT}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_executive_kpi_card_row(cards: list[dict]) -> None:
    """Render Image 1 style dark-header KPI cards with right-hand circular icon badges."""
    cols = st.columns(len(cards))
    for idx, c in enumerate(cards):
        header = c.get("header", "METRIC")
        val = c.get("value", "0")
        delta = c.get("delta", "")
        delta_color = "#00ff9d" if c.get("delta_positive", True) else "#ff3366"
        icon = c.get("icon", "📈")
        icon_bg = c.get("icon_bg", "rgba(0, 240, 255, 0.12)")
        icon_color = c.get("icon_color", "#00f0ff")

        card_html = f"""
        <div style="background: #0d1424; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; overflow: hidden; box-shadow: 0 4px 14px rgba(0,0,0,0.4); margin-bottom: 12px;">
            <div style="background: #141f36; padding: 7px 14px; font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid rgba(255,255,255,0.05);">
                {header}
            </div>
            <div style="padding: 14px 16px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-family: 'JetBrains Mono', Consolas, monospace; font-size: 1.65rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em;">
                        {val}
                    </div>
                    <div style="font-size: 0.74rem; font-weight: 600; color: {delta_color}; margin-top: 4px; display: flex; align-items: center; gap: 4px;">
                        {delta}
                    </div>
                </div>
                <div style="width: 40px; height: 40px; border-radius: 50%; background: {icon_bg}; border: 1px solid {icon_color}33; display: flex; align-items: center; justify-content: center; font-size: 1.15rem;">
                    {icon}
                </div>
            </div>
        </div>
        """
        with cols[idx]:
            st.markdown(card_html, unsafe_allow_html=True)


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def dataframe_csv_download(
    df: pd.DataFrame,
    csv_text: str,
    label: str,
    filename: str,
    key: str,
) -> None:
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        label,
        data=csv_text.encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def render_signal_light_widget(plan, intersection) -> None:
    """Render an authentic, high-fidelity physical traffic signal fixture."""
    st.markdown("### 🚦 Signal Timing Allocation")
    phases = list(plan.green.items())
    if not phases:
        return

    cols = st.columns(len(phases))

    for idx, (phase_id, green_s) in enumerate(phases):
        phase_obj = next((p for p in intersection.phases if p.id == phase_id), None)
        phase_name = getattr(phase_obj, "name", None) or f"Phase {phase_id}"
        app_val = getattr(phase_obj, "approach", None) or getattr(phase_obj, "approaches", phase_id)
        approaches = ", ".join(app_val) if isinstance(app_val, (list, tuple)) else str(app_val)
        share_pct = (green_s / plan.cycle * 100) if plan.cycle > 0 else 0.0

        is_active = (idx == 0)
        red_color = "#ff3366" if not is_active else "#2d151c"
        yellow_color = "#2d2315"
        green_color = "#00ff9d" if is_active else "#0a261a"
        green_glow = "0 0 16px rgba(0, 255, 157, 0.45)" if is_active else "none"
        red_glow = "0 0 16px rgba(255, 51, 102, 0.45)" if not is_active else "none"
        border_highlight = "1px solid rgba(0, 255, 157, 0.4)" if is_active else "1px solid rgba(255, 255, 255, 0.08)"

        html_code = f"""
        <div style="background-color: #0d1424; border-radius: 10px; padding: 18px 14px; border: {border_highlight}; text-align: center; box-shadow: 0 4px 14px rgba(0,0,0,0.4);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{phase_name}</span>
                <span style="font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.06); color: #cbd5e1; font-family: monospace;">{share_pct:.1f}%</span>
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 12px; text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                Approach: <span style="color: #e2e8f0; font-weight: 500;">{approaches}</span>
            </div>
            
            <!-- Signal Housing -->
            <div style="background: linear-gradient(180deg, #070a12 0%, #121a2d 100%); width: 56px; margin: 0 auto 14px auto; padding: 8px 6px; border-radius: 28px; border: 1px solid rgba(255,255,255,0.1); box-shadow: inset 0 2px 4px rgba(0,0,0,0.8);">
                <div style="width: 36px; height: 36px; border-radius: 50%; margin: 4px auto; background-color: {red_color}; box-shadow: {red_glow}; border: 1px solid rgba(0,0,0,0.5);"></div>
                <div style="width: 36px; height: 36px; border-radius: 50%; margin: 4px auto; background-color: {yellow_color}; border: 1px solid rgba(0,0,0,0.5);"></div>
                <div style="width: 36px; height: 36px; border-radius: 50%; margin: 4px auto; background-color: {green_color}; box-shadow: {green_glow}; border: 1px solid rgba(0,0,0,0.5);"></div>
            </div>
            
            <div style="font-family: 'JetBrains Mono', Consolas, monospace; font-size: 1.5rem; font-weight: 700; color: {'#00ff9d' if is_active else '#f1f5f9'};">
                {green_s:.1f} <span style="font-size: 0.85rem; font-weight: 400; color: #94a3b8;">s</span>
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">
                {'Active Green Phase' if is_active else 'Resting Red Phase'}
            </div>
        </div>
        """
        with cols[idx]:
            st.markdown(html_code, unsafe_allow_html=True)
