"""Smoke and Component Tests for NEMA TS2 8-Phase Studio UI."""

import importlib
import py_compile
from pathlib import Path
import pytest

from src.decision.nema_dual_ring import NEMADualRingController, NEMAPhase
from src.reports.audit_report import MunicipalAuditData, generate_municipal_pdf_report
from src.rl.nema_agent import NEMADualRingAgent
from src.sumo.nema_network import build_nema_tls_state_string


def test_nema_studio_syntax_compilation():
    """Verifies that 5_NEMA_8Phase_Studio.py compiles cleanly with zero syntax errors."""
    page_path = Path("app/pages/5_NEMA_8Phase_Studio.py")
    assert page_path.exists()
    # py_compile checks for SyntaxError, IndentationError, etc.
    py_compile.compile(str(page_path), doraise=True)


def test_nema_studio_components_and_pdf_generation():
    """Verifies that all components invoked by the Studio generate valid outputs."""
    agent = NEMADualRingAgent()
    queues_m = {p: 12.0 for p in NEMAPhase}
    pcu = {p: 8.0 for p in NEMAPhase}
    headways = {p.value: 2.0 for p in NEMAPhase}

    dec = agent.step(dt=1.0, queues_m=queues_m, pcu_counts=pcu, headways=headways)
    assert len(dec.signal_string_12) == 12

    muni_data = MunicipalAuditData(
        intersection_name="Central Arterial Junction (NEMA TS2 8-Phase)",
        controller_mode="NEMA TS2 Dual-Ring Actuated",
        cycle_length_s=85.0,
        phase_greens_s={
            f"Ring 1 P{dec.ring1_phase.value}": dec.ring1_elapsed_s,
            f"Ring 2 P{dec.ring2_phase.value}": dec.ring2_elapsed_s,
        },
        metric_queues_m={f"Phase {p.value}": 12.0 for p in NEMAPhase},
    )
    pdf_bytes = generate_municipal_pdf_report(muni_data)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
