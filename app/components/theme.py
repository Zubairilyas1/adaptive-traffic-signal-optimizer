"""Theme and styling engine: Synthesis of Enterprise Dashboard (Image 1) & High-Tech Dark Telemetry (Image 2).

Key Features:
- Carbon Slate command palette (#070a12 canvas, #0d1424 card surfaces)
- Image 1 Layout: Dark header pill KPI cards, breadcrumbs, clean circular icon badges
- Image 2 Telemetry: High-contrast cyan (#00f0ff) & coral pink (#ff3366) data curves
- Monospace precision typography ('JetBrains Mono', Consolas, monospace)
"""

from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """Inject global CSS rules to create an irresistible, authentic TMC console."""
    css = """
    <style>
    /* Global Canvas */
    .stApp {
        background-color: #070a12;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Top Streamlit Header */
    header[data-testid="stHeader"] {
        background-color: rgba(7, 10, 18, 0.90);
        backdrop-filter: blur(12px);
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Streamlit Default Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #0e1628;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetricValue"] {
        color: #00f0ff !important;
        font-family: "JetBrains Mono", Consolas, monospace !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
    }

    /* Section Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 1.5rem 0 !important;
    }

    /* Tabs (Image 1 / Image 2 style) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        padding-bottom: 2px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #0d1424;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #162238 !important;
        color: #00f0ff !important;
        border-bottom: 2px solid #00f0ff !important;
        border-top: 1px solid rgba(0, 240, 255, 0.3) !important;
    }

    /* Dataframe Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        background-color: #0a0f1c;
    }

    /* Primary Action Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0.02em;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 6px;
        padding: 0.6rem 1.4rem;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.5);
    }
    div.stButton > button[kind="secondary"] {
        background-color: #121a2d;
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        font-weight: 600;
    }

    /* Alerts and Callouts */
    div[data-testid="stAlert"] {
        background-color: #0d1527;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }

    /* Breadcrumbs Bar (Image 1 Style) */
    .tmc-breadcrumbs {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .tmc-breadcrumbs a, .tmc-breadcrumbs span.active {
        color: #38bdf8;
    }

    /* Executive Command Header */
    .command-header-box {
        background: linear-gradient(180deg, #10182b 0%, #0a0f1c 100%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 10px;
        padding: 22px 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .command-header-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .command-header-sub {
        font-size: 0.9rem;
        color: #94a3b8;
        margin: 0;
    }

    /* Telemetry Pill Chips */
    .tmc-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #162035;
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-family: "JetBrains Mono", Consolas, monospace;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .chip-green {
        color: #00ff9d;
        border-color: rgba(0, 255, 157, 0.3);
        background-color: rgba(0, 255, 157, 0.08);
    }
    .chip-cyan {
        color: #00f0ff;
        border-color: rgba(0, 240, 255, 0.3);
        background-color: rgba(0, 240, 255, 0.08);
    }
    .chip-amber {
        color: #ffaa00;
        border-color: rgba(255, 170, 0, 0.3);
        background-color: rgba(255, 170, 0, 0.08);
    }
    .chip-pink {
        color: #ff3366;
        border-color: rgba(255, 51, 102, 0.3);
        background-color: rgba(255, 51, 102, 0.08);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_breadcrumb(path: str = "HOME > TRAFFIC OPERATIONS CENTER") -> None:
    """Render an Image 1 style sleek breadcrumb path."""
    html = f"""
    <div class="tmc-breadcrumbs">
        <span>📍 {path}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, chips: list[tuple[str, str]] | None = None) -> None:
    """Render an authoritative, clean command-center header banner."""
    chips_html = ""
    if chips:
        chips_spans = [f'<span class="tmc-chip {style_class}">{text}</span>' for text, style_class in chips]
        chips_html = f'<div style="margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px;">{" ".join(chips_spans)}</div>'

    html = f"""
    <div class="command-header-box">
        <div class="command-header-title">{title}</div>
        <div class="command-header-sub">{subtitle}</div>
        {chips_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
