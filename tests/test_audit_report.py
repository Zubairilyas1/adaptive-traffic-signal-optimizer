"""Unit tests for Municipal Traffic Engineering Audit Report Generator."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.reports.audit_report import (
    MunicipalAuditData,
    generate_municipal_pdf_report,
)


def test_generate_default_municipal_pdf():
    pdf_bytes = generate_municipal_pdf_report()

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 15_000  # Should be substantial multi-page vector PDF
    assert pdf_bytes[:4] == b"%PDF"


def test_generate_custom_municipal_pdf(tmp_path: Path):
    custom_data = MunicipalAuditData(
        intersection_name="Downtown Corridor & 5th Ave",
        jurisdiction="City Transportation Bureau",
        controller_mode="Dual-Regime Adaptive Optimizer",
        cycle_length_s=90.0,
        fixed_delay_s=55.0,
        ppo_delay_s=22.0,
        metric_queues_m={"North": 45.0, "South": 40.0, "East": 15.0, "West": 10.0},
    )

    out_file = tmp_path / "test_audit.pdf"
    res_bytes = generate_municipal_pdf_report(custom_data, out_path=out_file)

    assert out_file.exists()
    assert out_file.stat().st_size == len(res_bytes)
    assert out_file.stat().st_size > 15_000
