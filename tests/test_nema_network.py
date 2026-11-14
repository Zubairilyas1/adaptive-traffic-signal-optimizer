"""Unit and Integration Tests for NEMA TS2 8-Phase SUMO Network Generator."""

from pathlib import Path
import pytest

from src.decision.nema_dual_ring import NEMAPhase, PhaseInterval
from src.sumo.nema_network import (
    DEFAULT_8PHASE_DEMAND,
    NEMA_8_MOVEMENTS,
    _nema_connections_xml,
    _nema_edges_xml,
    _nema_nodes_xml,
    _nema_routes_xml,
    _nema_sumocfg,
    build_nema_8phase_intersection,
    build_nema_tls_state_string,
    find_sumo_binary,
)


def test_find_sumo_binary():
    """Verifies netconvert binary is resolved properly."""
    binary = find_sumo_binary("netconvert")
    assert isinstance(binary, str)
    assert len(binary) > 0


def test_nema_xml_components():
    """Verifies XML component generation produces valid structural elements."""
    nodes = _nema_nodes_xml()
    assert "<node id=\"C\"" in nodes
    assert "type=\"traffic_light\"" in nodes

    edges = _nema_edges_xml()
    assert 'id="N2C"' in edges
    assert 'numLanes="3"' in edges

    connections = _nema_connections_xml()
    assert 'from="N2C" to="C2E" fromLane="2"' in connections  # Protected Left
    assert 'from="S2C" to="C2W" fromLane="2"' in connections  # Protected Left

    routes = _nema_routes_xml(DEFAULT_8PHASE_DEMAND, duration=300.0)
    assert '<vType id="car"' in routes
    assert 'id="flow_P1_0"' in routes
    assert 'id="flow_P2_0"' in routes

    cfg = _nema_sumocfg("test.net.xml", "test.rou.xml", end=300.0)
    assert '<net-file value="test.net.xml"/>' in cfg


def test_nema_8_movements_structure():
    """Verifies that all 8 NEMA phases are mapped with valid edge and lane definitions."""
    assert len(NEMA_8_MOVEMENTS) == 8
    for phase, moves in NEMA_8_MOVEMENTS.items():
        assert len(moves) >= 1
        for inc, lane, out in moves:
            assert inc.endswith("2C")
            assert out.startswith("C2")
            assert lane in (0, 1, 2)


def test_build_nema_tls_state_string_dual_throughs():
    """Verifies signal string generation for Phase 2 & 6 Coordinated Throughs."""
    state = build_nema_tls_state_string(
        ring1_phase=NEMAPhase.P2_NB_THROUGH,
        ring1_interval=PhaseInterval.GREEN,
        ring2_phase=NEMAPhase.P6_SB_THROUGH,
        ring2_interval=PhaseInterval.GREEN,
    )
    assert len(state) == 12
    # P2 (Links 0, 1) and P6 (Links 3, 4) should be 'G'
    assert state[0] == "G" and state[1] == "G"
    assert state[3] == "G" and state[4] == "G"
    # Protected lefts (P5 Link 2, P1 Link 5) must be 'r'
    assert state[2] == "r" and state[5] == "r"


def test_build_nema_tls_state_string_dual_lefts():
    """Verifies signal string generation for Phase 1 & 5 Dual Leading Lefts."""
    state = build_nema_tls_state_string(
        ring1_phase=NEMAPhase.P1_SB_LEFT,
        ring1_interval=PhaseInterval.GREEN,
        ring2_phase=NEMAPhase.P5_NB_LEFT,
        ring2_interval=PhaseInterval.GREEN,
    )
    assert len(state) == 12
    # P1 (Link 5) and P5 (Link 2) should be 'G'
    assert state[2] == "G"
    assert state[5] == "G"
    # Opposing throughs must be 'r'
    assert state[0] == "r" and state[1] == "r"
    assert state[3] == "r" and state[4] == "r"


def test_build_nema_tls_state_string_yellow_and_red():
    """Verifies yellow and red clearances translate properly to 'y' and 'r'."""
    yellow_state = build_nema_tls_state_string(
        ring1_phase=NEMAPhase.P2_NB_THROUGH,
        ring1_interval=PhaseInterval.YELLOW_CHANGE,
        ring2_phase=NEMAPhase.P6_SB_THROUGH,
        ring2_interval=PhaseInterval.GREEN,
    )
    assert "y" in yellow_state
    assert yellow_state[0] == "y" and yellow_state[1] == "y"
    assert yellow_state[3] == "G" and yellow_state[4] == "G"

    all_red_state = build_nema_tls_state_string(
        ring1_phase=NEMAPhase.P2_NB_THROUGH,
        ring1_interval=PhaseInterval.ALL_RED_CLEARANCE,
        ring2_phase=NEMAPhase.P6_SB_THROUGH,
        ring2_interval=PhaseInterval.ALL_RED_CLEARANCE,
    )
    assert all_red_state == "r" * 12


def test_build_nema_8phase_network_compilation(tmp_path: Path):
    """Verifies complete netconvert compilation of the NEMA 8-phase network."""
    cfg_file = build_nema_8phase_intersection(
        demand=DEFAULT_8PHASE_DEMAND,
        duration=60.0,
        out_dir=tmp_path,
        rebuild=True,
    )
    assert cfg_file.exists()
    assert (tmp_path / "intersection_nema.net.xml").exists()
    assert (tmp_path / "intersection_nema.rou.xml").exists()
    assert (tmp_path / "intersection_nema.nod.xml").exists()
    assert (tmp_path / "intersection_nema.edg.xml").exists()
    assert (tmp_path / "intersection_nema.con.xml").exists()
    assert (tmp_path / "intersection_nema.net.xml").stat().st_size > 1000
