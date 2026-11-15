"""Unit and Integration Tests for Master Thesis & Defense PDF Engine."""

from pathlib import Path
import re

from src.reports.master_thesis_pdf import (
    MasterThesisReportGenerator,
    generate_master_thesis_pdf,
)


def test_master_thesis_pdf_generation(tmp_path: Path):
    """Verifies that the master thesis generator compiles an authoritative multi-page PDF."""
    target_pdf = tmp_path / "Test_Engineering_Thesis.pdf"
    assert not target_pdf.exists()

    result_path = generate_master_thesis_pdf(output_path=target_pdf)
    assert result_path == target_pdf
    assert target_pdf.exists()

    # Verify PDF file size and header
    pdf_bytes = target_pdf.read_bytes()
    assert len(pdf_bytes) > 20000  # At least 20KB for a comprehensive report
    assert pdf_bytes.startswith(b"%PDF-")

    # Verify that multi-page structure is present (>= 10 pages)
    page_matches = re.findall(rb"/Type\s*/Page[^s]", pdf_bytes)
    assert len(page_matches) >= 10


def test_master_thesis_generator_class_instantiation(tmp_path: Path):
    """Verifies that the generator class initializes custom typography styles and builds cleanly."""
    out = tmp_path / "Custom_Thesis.pdf"
    gen = MasterThesisReportGenerator(output_path=out)
    assert "CoverTitle" in gen.styles
    assert "ChapterHeading" in gen.styles
    assert "EquationBox" in gen.styles

    built_path = gen.generate()
    assert built_path.exists()
    assert built_path.stat().st_size > 10000
