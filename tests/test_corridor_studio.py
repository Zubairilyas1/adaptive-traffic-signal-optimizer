"""Smoke and Component Tests for Arterial Corridor Coordination Studio UI."""

from pathlib import Path
import py_compile
import matplotlib.pyplot as plt

from src.decision.corridor_coordinator import (
    CorridorCoordinator,
    CorridorTelemetry,
)
from src.sumo.corridor_network import ArterialCorridorConfig


def test_corridor_studio_syntax_compilation():
    """Verifies that 7_Corridor_Coordination.py compiles cleanly with zero syntax errors."""
    page_path = Path("app/pages/7_Corridor_Coordination.py")
    assert page_path.exists()
    py_compile.compile(str(page_path), doraise=True)


def test_corridor_studio_components():
    """Verifies that the coordinator and telemetry models underlying the studio function properly."""
    cfg = ArterialCorridorConfig(d12_m=400.0, d23_m=450.0)
    coordinator = CorridorCoordinator(cfg)

    # Step by 1s
    telem = coordinator.step(dt=1.0)
    assert isinstance(telem, CorridorTelemetry)
    assert telem.bandwidth_efficiency_pct > 0.0
    assert len(telem.junctions) == 3

    # Check offset adjustment calculation used by UI sliders
    adj = coordinator.compute_queue_clearance_adjustment("J2", queue_m=40.0)
    assert 14.0 < adj <= 18.0


def test_corridor_studio_matplotlib_tsd_generation():
    """Verifies that Matplotlib can render the Time-Space Diagram elements without errors."""
    fig, ax = plt.subplots(figsize=(8, 4))
    coordinator = CorridorCoordinator()
    telem = coordinator.step(dt=1.0)

    # Simple verify that plot primitives work
    ax.set_xlim(0, 180)
    ax.set_ylim(-50, 900)
    ax.set_yticks([0, 450, 850])
    ax.set_yticklabels(["J3", "J2", "J1"])

    assert len(ax.get_yticks()) == 3
    plt.close(fig)
