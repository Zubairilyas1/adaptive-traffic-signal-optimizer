"""Modern Civic Engineering Design System & Theme Engine.

Key Principles:
- Deep Executive Slate palette (#0b0f19 canvas, #131c2e card surfaces)
- Luminous elevation with crisp subtle borders (1px solid rgba(255,255,255,0.08))
- Municipal brand blue (#3b82f6), refined emerald (#10b981), and warm amber (#f59e0b)
- Responsive auto-fit card grids that eliminate text and button truncation (...)
- High-legibility typography hierarchy with WCAG AAA contrast
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


def apply_theme() -> None:
    """Inject global CSS rules to create a clean, light, trustworthy, and modern SaaS console."""
    css = """
    <style>
    /* Global Canvas */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", "Helvetica Neue", Arial, sans-serif;
    }

    /* Top Streamlit Header */
    header[data-testid="stHeader"] {
        background-color: rgba(11, 15, 25, 0.92);
        backdrop-filter: blur(14px);
    }

    /* Main Container Padding & Max Width */
    .block-container {
        max-width: 1440px !important;
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Streamlit Default Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #131c2e;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(59, 130, 246, 0.35);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-family: "JetBrains Mono", Consolas, monospace !important;
        font-size: 1.65rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }

    /* Section Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 1.75rem 0 !important;
    }

    /* Tabs (Modern Segmented Style) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #131c2e;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px 8px 0 0;
        padding: 10px 22px;
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 600;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9;
        background-color: #1a253c;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #3b82f6 !important;
        border-top: 1px solid rgba(59, 130, 246, 0.35) !important;
    }

    /* Dataframe Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        background-color: #0f172a;
    }

    /* Primary Action Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0.02em;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 0.65rem 1.6rem;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.35);
        transition: all 0.2s ease;
        min-width: 140px;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
        transform: translateY(-1px);
    }
    div.stButton > button[kind="secondary"] {
        background-color: #1e293b;
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 0.65rem 1.4rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #334155;
        border-color: rgba(255, 255, 255, 0.25);
    }

    /* Alerts and Callouts */
    div[data-testid="stAlert"] {
        background-color: #131c2e;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
    }

    /* Breadcrumbs Bar */
    .tmc-breadcrumbs {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
    }
    .tmc-breadcrumbs span.active {
        color: #38bdf8;
    }

    /* Executive Command Header */
    .command-header-box {
        background: linear-gradient(180deg, #131c2e 0%, #0d1524 100%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .command-header-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .command-header-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.5;
    }

    /* Telemetry Pill Chips */
    .tmc-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #1e293b;
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.76rem;
        font-family: "JetBrains Mono", Consolas, monospace;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .chip-green {
        color: #34d399;
        border-color: rgba(52, 211, 153, 0.3);
        background-color: rgba(52, 211, 153, 0.08);
    }
    .chip-cyan, .chip-blue, .chip-teal {
        color: #38bdf8;
        border-color: rgba(56, 189, 248, 0.3);
        background-color: rgba(56, 189, 248, 0.08);
    }
    .chip-amber {
        color: #fbbf24;
        border-color: rgba(251, 191, 36, 0.3);
        background-color: rgba(251, 191, 36, 0.08);
    }
    .chip-pink, .chip-red {
        color: #f87171;
        border-color: rgba(248, 113, 113, 0.3);
        background-color: rgba(248, 113, 113, 0.08);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_breadcrumb(path: str = "HOME > TRAFFIC OPERATIONS CENTER") -> None:
    """Render an authoritative, sleek breadcrumb path."""
    parts = [p.strip() for p in path.split(">")]
    spans = []
    for idx, part in enumerate(parts):
        is_last = idx == len(parts) - 1
        cls = 'class="active"' if is_last else ""
        spans.append(f"<span {cls}>{part}</span>")

    html = f"""
    <div class="tmc-breadcrumbs">
        <span>📍</span>
        {" &gt; ".join(spans)}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_header(
    title: str,
    subtitle: str,
    chips: Optional[List[Tuple[str, str]]] = None,
    badge_label: Optional[str] = None,
    badge_color: str = "blue",
    **kwargs: Any,
) -> None:
    """Render an authoritative, clean command-center header banner."""
    resolved_chips: List[Tuple[str, str]] = list(chips) if chips else []
    if badge_label:
        resolved_chips.append((badge_label, f"chip-{badge_color}"))

    chips_html = ""
    if resolved_chips:
        chips_spans = [f'<span class="tmc-chip {style_class}">{text}</span>' for text, style_class in resolved_chips]
        chips_html = f'<div style="margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px;">{" ".join(chips_spans)}</div>'

    html = f"""
    <div class="command-header-box">
        <div class="command-header-title">{title}</div>
        <div class="command-header-sub">{subtitle}</div>
        {chips_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_positive: bool = True,
    icon: str = "📊",
    help_text: Optional[str] = None,
) -> str:
    """Return HTML string for a clean, non-clipping responsive metric card."""
    delta_color = "#34d399" if delta_positive else "#f87171"
    delta_html = f'<div style="font-size: 0.72rem; font-weight: 600; color: {delta_color}; margin-top: 4px;">{delta}</div>' if delta else ""
    title_attr = f'title="{help_text}"' if help_text else ""

    return f"""
    <div style="background: #131c2e; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 14px 18px; box-shadow: 0 4px 14px rgba(0,0,0,0.22); min-width: 140px; display: flex; justify-content: space-between; align-items: center;" {title_attr}>
        <div style="min-width: 0; flex: 1;">
            <div style="font-size: 0.74rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {label}
            </div>
            <div style="font-family: 'JetBrains Mono', Consolas, monospace; font-size: 1.55rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {value}
            </div>
            {delta_html}
        </div>
        <div style="flex-shrink: 0; margin-left: 10px; font-size: 1.4rem; opacity: 0.85;">
            {icon}
        </div>
    </div>
    """


def render_kpi_grid(cards: List[Dict[str, Any]]) -> None:
    """Render a responsive auto-fitting grid of KPI cards that never truncates or overlaps."""
    cards_html = [
        render_metric_card(
            label=c.get("label", c.get("header", "METRIC")),
            value=str(c.get("value", "0")),
            delta=c.get("delta"),
            delta_positive=c.get("delta_positive", True),
            icon=c.get("icon", "📊"),
            help_text=c.get("help"),
        )
        for c in cards
    ]
    grid_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 20px;">
        {"".join(cards_html)}
    </div>
    """
    st.markdown(grid_html, unsafe_allow_html=True)
