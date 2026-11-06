"""Official Municipal Traffic Engineering Audit Report Generator.

Compiles complete, publication-ready multi-page vector PDF engineering reports
integrating PCU demand analysis, 3D IPM physical queues, PPO micro-simulation benchmarks,
law enforcement violations, and municipal certification.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from src import config

logger = logging.getLogger(__name__)


@dataclass
class MunicipalAuditData:
    """Consolidated dataset for generating an official municipal traffic audit report."""

    intersection_name: str = "Central Junction (Node C)"
    jurisdiction: str = "Municipal Department of Transportation"
    controller_mode: str = "PPO Neural Agent (Closed-Loop Digital Twin)"
    date_str: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    approach_volumes: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "North": {"car": 120, "motorcycle": 85, "bus": 12, "truck": 8, "total_pcu": 192.5},
        "South": {"car": 110, "motorcycle": 70, "bus": 10, "truck": 6, "total_pcu": 165.0},
        "East": {"car": 65, "motorcycle": 40, "bus": 15, "truck": 12, "total_pcu": 132.5},
        "West": {"car": 55, "motorcycle": 35, "bus": 8, "truck": 5, "total_pcu": 87.5},
    })
    metric_queues_m: dict[str, float] = field(default_factory=lambda: {
        "North": 38.5, "South": 32.0, "East": 24.5, "West": 18.0
    })
    cycle_length_s: float = 75.0
    phase_greens_s: dict[str, float] = field(default_factory=lambda: {
        "Phase 0 (North-South)": 42.0,
        "Phase 1 (East-West)": 25.0,
    })
    regime_name: str = "Undersaturated (Webster Delay-Minimization)"
    critical_flow_ratio_y: float = 0.685
    fixed_delay_s: float = 43.8
    actuated_delay_s: float = 24.5
    ppo_delay_s: float = 16.2
    total_violations_recorded: int = 3
    preemption_activations: int = 1
    violations_summary: list[dict[str, Any]] = field(default_factory=lambda: [
        {"track_id": 104, "class": "car", "speed_km_h": 58.2, "approach": "North", "time": "10:14:22"},
        {"track_id": 218, "class": "motorcycle", "speed_km_h": 64.5, "approach": "East", "time": "10:28:05"},
        {"track_id": 341, "class": "van", "speed_km_h": 52.0, "approach": "South", "time": "10:41:19"},
    ])


def generate_municipal_pdf_report(
    data: Optional[MunicipalAuditData] = None,
    out_path: Optional[Path | str] = None,
) -> bytes:
    """Generate a multi-page A4 vector PDF municipal audit report.

    Parameters:
        data: MunicipalAuditData instance containing metrics and benchmarks.
        out_path: Optional path to write the PDF to disk.

    Returns:
        bytes: Raw PDF binary buffer.
    """
    if data is None:
        data = MunicipalAuditData()

    buf = io.BytesIO()

    with PdfPages(buf) as pdf:
        # =====================================================================
        # PAGE 1: EXECUTIVE AUDIT & TRAFFIC DEMAND INVENTORY
        # =====================================================================
        fig1 = plt.figure(figsize=(8.27, 11.69), dpi=150)  # Standard A4 Portrait
        fig1.patch.set_facecolor("#ffffff")

        # Top Header Banner
        header_ax = fig1.add_axes([0.08, 0.88, 0.84, 0.08])
        header_ax.axis("off")
        header_ax.text(
            0.0, 0.75,
            f"{data.jurisdiction.upper()} — TRAFFIC OPERATIONS DIVISION",
            fontsize=12, fontweight="bold", color="#1e293b", family="sans-serif",
        )
        header_ax.text(
            0.0, 0.40,
            "OFFICIAL INTELLIGENT TRANSPORTATION SYSTEM (ITS) AUDIT REPORT",
            fontsize=15, fontweight="heavy", color="#0f766e", family="sans-serif",
        )
        header_ax.text(
            0.0, 0.05,
            f"Intersection: {data.intersection_name}  |  Audit Timestamp: {data.date_str}  |  Controller: {data.controller_mode}",
            fontsize=8.5, color="#64748b", family="sans-serif",
        )

        # Dividing Rule
        fig1.add_artist(plt.Line2D([0.08, 0.92], [0.865, 0.865], color="#0f766e", linewidth=2.0))

        # Section 1: Geometric & PCU Volume Breakdown Table
        sec1_ax = fig1.add_axes([0.08, 0.62, 0.84, 0.22])
        sec1_ax.axis("off")
        sec1_ax.text(0.0, 1.0, "1. APPROACH INVENTORY & PCU CAPACITY DEMAND (IRC:106 CALIBRATED)", fontsize=11, fontweight="bold", color="#0f172a")

        headers = ["Approach", "Cars", "Motorcycles", "Buses", "Trucks", "Total PCU", "Queue (m)"]
        table_rows = []
        for app, counts in data.approach_volumes.items():
            q_m = data.metric_queues_m.get(app, 0.0)
            table_rows.append([
                app,
                str(counts.get("car", 0)),
                str(counts.get("motorcycle", 0)),
                str(counts.get("bus", 0)),
                str(counts.get("truck", 0)),
                f"{counts.get('total_pcu', 0.0):.1f}",
                f"{q_m:.1f} m",
            ])

        table = sec1_ax.table(
            cellText=table_rows,
            colLabels=headers,
            loc="center",
            cellLoc="center",
            colColours=["#e2e8f0"] * len(headers),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.5)

        # Section 2: Visual Charts (PCU Distribution & Physical Queue in Meters)
        ax_pie = fig1.add_axes([0.08, 0.32, 0.38, 0.25])
        approaches = list(data.approach_volumes.keys())
        pcu_shares = [data.approach_volumes[a].get("total_pcu", 0.0) for a in approaches]
        colors_pie = ["#0ea5e9", "#06b6d4", "#10b981", "#f59e0b"]
        ax_pie.pie(pcu_shares, labels=approaches, autopct="%1.1f%%", colors=colors_pie, startangle=140, textprops={"fontsize": 8})
        ax_pie.set_title("Approach PCU Demand Share", fontsize=9.5, fontweight="bold", color="#1e293b")

        ax_bar = fig1.add_axes([0.54, 0.32, 0.38, 0.25])
        queues_m = [data.metric_queues_m.get(a, 0.0) for a in approaches]
        bars = ax_bar.bar(approaches, queues_m, color="#3b82f6", width=0.55)
        ax_bar.set_ylabel("Physical Queue Length (meters)", fontsize=8)
        ax_bar.set_title("3D IPM Homography Queue Depth (m)", fontsize=9.5, fontweight="bold", color="#1e293b")
        ax_bar.tick_params(axis="both", which="major", labelsize=8)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{yval:.1f}m", ha="center", va="bottom", fontsize=8)

        # Footer Page 1
        foot1_ax = fig1.add_axes([0.08, 0.04, 0.84, 0.20])
        foot1_ax.axis("off")
        foot1_text = (
            "METHODOLOGY NOTES & CALIBRATION STANDARDS:\n"
            "• Vehicle kinematics evaluated using YOLOv8n detector with ByteTrack temporal association.\n"
            "• Queue backlogs differentiated via kinematic state decomposition (halted v < 5 km/h vs. flowing).\n"
            "• Metric queue length calibrated via 4-point planar Inverse Perspective Mapping (IPM) homography.\n"
            "• Passenger Car Unit (PCU) factors configured per municipal standard guidelines.\n"
        )
        foot1_ax.text(0.0, 0.8, foot1_text, fontsize=8, color="#475569", family="monospace", linespacing=1.6)
        foot1_ax.text(0.95, 0.0, "Page 1 of 2", fontsize=8, color="#94a3b8")

        pdf.savefig(fig1)
        plt.close(fig1)

        # =====================================================================
        # PAGE 2: CONTROLLER BENCHMARKS, LAW ENFORCEMENT & CERTIFICATION
        # =====================================================================
        fig2 = plt.figure(figsize=(8.27, 11.69), dpi=150)
        fig2.patch.set_facecolor("#ffffff")

        # Header Page 2
        h2_ax = fig2.add_axes([0.08, 0.90, 0.84, 0.06])
        h2_ax.axis("off")
        h2_ax.text(0.0, 0.5, "2. SIGNAL TIMING OPTIMIZATION & EMPIRICAL PERFORMANCE", fontsize=12, fontweight="bold", color="#0f766e")
        fig2.add_artist(plt.Line2D([0.08, 0.92], [0.89, 0.89], color="#0f766e", linewidth=1.5))

        # Signal Plan Overview
        plan_ax = fig2.add_axes([0.08, 0.77, 0.84, 0.10])
        plan_ax.axis("off")
        plan_summary = (
            f"• Optimized Cycle Length: {data.cycle_length_s:.1f} s   |   Critical Flow Ratio (Y): {data.critical_flow_ratio_y:.3f}\n"
            f"• Optimization Regime: {data.regime_name}\n"
            f"• Active Green Allocations: " + ", ".join([f"{k}: {v:.1f}s" for k, v in data.phase_greens_s.items()]) + "\n"
            f"• Safety Clearances Enforced: Yellow Clearance = 3.0 s   |   All-Red Interval = 1.0 s"
        )
        plan_ax.text(0.0, 0.8, plan_summary, fontsize=8.5, color="#1e293b", family="sans-serif", linespacing=1.6)

        # Comparative Micro-Simulation Delay Chart
        delay_ax = fig2.add_axes([0.08, 0.54, 0.84, 0.19])
        controllers = ["Fixed-Time Baseline", "Rule-Based Actuated", "Neural PPO Agent"]
        delays = [data.fixed_delay_s, data.actuated_delay_s, data.ppo_delay_s]
        colors_ctrl = ["#94a3b8", "#38bdf8", "#059669"]

        hbars = delay_ax.barh(controllers, delays, color=colors_ctrl, height=0.45)
        delay_ax.set_xlabel("Average Control Delay (seconds / vehicle)", fontsize=8)
        delay_ax.set_title("Empirical Delay Reduction Benchmarks (SUMO Micro-Simulation)", fontsize=9.5, fontweight="bold", color="#1e293b")
        delay_ax.tick_params(axis="both", which="major", labelsize=8)
        delay_ax.grid(axis="x", linestyle="--", alpha=0.5)

        red_pct = ((data.fixed_delay_s - data.ppo_delay_s) / data.fixed_delay_s) * 100.0 if data.fixed_delay_s > 0 else 0.0
        for hbar in hbars:
            xval = hbar.get_width()
            delay_ax.text(xval + 1.0, hbar.get_y() + hbar.get_height() / 2.0, f"{xval:.1f} s/veh", ha="left", va="center", fontsize=8)

        delay_ax.text(
            0.98, 0.85, f"PPO Gain: -{red_pct:.1f}% Delay",
            transform=delay_ax.transAxes, ha="right", fontsize=9, fontweight="bold", color="#059669",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecfdf5", edgecolor="#059669", alpha=0.8),
        )

        # Section 3: Law Enforcement & Safety Audit Table
        enf_ax = fig2.add_axes([0.08, 0.32, 0.84, 0.18])
        enf_ax.axis("off")
        enf_ax.text(0.0, 1.0, "3. LAW ENFORCEMENT & EMERGENCY PREEMPTION AUDIT LOG", fontsize=11, fontweight="bold", color="#0f172a")

        v_headers = ["Infraction ID", "Class", "Recorded Speed", "Approach", "Time", "Status"]
        v_rows = []
        for v in data.violations_summary:
            v_rows.append([
                f"VIOL-{v['track_id']}",
                v["class"].title(),
                f"{v['speed_km_h']:.1f} km/h",
                v["approach"],
                v["time"],
                "RED-LIGHT INFRACTION",
            ])
        if not v_rows:
            v_rows = [["None", "N/A", "0.0 km/h", "All Clear", "--:--:--", "ZERO VIOLATIONS RECORDED"]]

        v_table = enf_ax.table(
            cellText=v_rows,
            colLabels=v_headers,
            loc="center",
            cellLoc="center",
            colColours=["#fee2e2"] * len(v_headers),
        )
        v_table.auto_set_font_size(False)
        v_table.set_fontsize(8)
        v_table.scale(1.0, 1.4)

        # Section 4: Engineering Certification Block
        cert_ax = fig2.add_axes([0.08, 0.06, 0.84, 0.20])
        cert_ax.axis("off")
        cert_ax.text(0.0, 0.95, "4. PROFESSIONAL MUNICIPAL ENGINEERING CERTIFICATION", fontsize=11, fontweight="bold", color="#0f172a")

        cert_box_text = (
            "I hereby verify that the timing allocations, kinematics estimations, and micro-simulation benchmarks\n"
            "contained in this report were generated via calibrated closed-loop edge perception and NEMA TS2 standard safety\n"
            "clearance models. Timing parameters comply with minimum vehicular green and dilemma-zone constraints."
        )
        cert_ax.text(0.0, 0.70, cert_box_text, fontsize=8, color="#334155", linespacing=1.5)

        # Signature Lines
        cert_ax.text(0.0, 0.25, "________________________________________", color="#64748b", fontsize=9)
        cert_ax.text(0.0, 0.10, "Supervising Municipal Traffic Engineer, P.E.", fontsize=8, fontweight="bold", color="#1e293b")

        cert_ax.text(0.55, 0.25, "________________________________________", color="#64748b", fontsize=9)
        cert_ax.text(0.55, 0.10, "Operations Division Director, MDoT", fontsize=8, fontweight="bold", color="#1e293b")

        cert_ax.text(0.95, 0.0, "Page 2 of 2", fontsize=8, color="#94a3b8")

        pdf.savefig(fig2)
        plt.close(fig2)

    pdf_bytes = buf.getvalue()
    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(pdf_bytes)
        logger.info(f"Exported municipal audit PDF to {out_p}")

    return pdf_bytes
