"""Unit and Integration Tests for Corridor Green-Wave Coordination and Offset Optimization."""

from src.decision.corridor_coordinator import (
    CorridorCoordinator,
    CorridorTelemetry,
    RobertsonPlatoonParams,
)
from src.decision.nema_dual_ring import NEMAPhase
from src.sumo.corridor_network import ArterialCorridorConfig


def test_robertson_platoon_dispersion():
    """Verifies that Robertson's platoon dispersion model calculates physically sound values."""
    params = RobertsonPlatoonParams(alpha=0.35, beta=0.80)

    # At travel_time = 0, no dispersion occurs (F = 1.0)
    assert params.smoothing_factor(0.0) == 1.0

    # At travel_time = 32.0s (400m at 45 km/h):
    # F = 1 / (1 + 0.35 * 0.80 * 32.0) = 1 / (1 + 8.96) = 1 / 9.96 ~= 0.1004
    f_32 = params.smoothing_factor(32.0)
    assert 0.09 < f_32 < 0.12

    # Verify flow smoothing
    upstream_peak = 1800.0  # Dense departing platoon
    prev_downstream = 400.0  # Ambient downstream flow
    downstream_arr = params.disperse_flow(upstream_peak, prev_downstream, travel_time_s=32.0)

    assert prev_downstream < downstream_arr < upstream_peak
    assert round(downstream_arr, 1) == round(f_32 * upstream_peak + (1.0 - f_32) * prev_downstream, 1)


def test_corridor_theoretical_offsets():
    """Verifies that ideal progression offsets for J1, J2, and J3 are computed accurately."""
    # d12 = 400m, d23 = 450m, v_prog = 45 km/h (12.5 m/s), C = 90s
    cfg = ArterialCorridorConfig(d12_m=400.0, d23_m=450.0, progression_speed_kmh=45.0, common_cycle_s=90.0)
    coordinator = CorridorCoordinator(cfg)

    offsets = coordinator.theoretical_offsets
    assert len(offsets) == 3

    # Junction 1 (North Gate) is origin: offset = 0.0s
    assert offsets["J1"] == 0.0

    # Junction 2 (Central Plaza): 400m / 12.5 m/s = 32.0s
    assert round(offsets["J2"], 1) == 32.0

    # Junction 3 (South Boulevard): (400m + 450m) / 12.5 m/s = 850 / 12.5 = 68.0s
    assert round(offsets["J3"], 1) == 68.0


def test_queue_clearance_offset_adjustment():
    """Verifies that residual queues advance the offset to clear waiting vehicles prior to platoon arrival."""
    coordinator = CorridorCoordinator()

    # Zero queue = no offset adjustment
    assert coordinator.compute_queue_clearance_adjustment("J2", queue_m=0.0) == 0.0

    # Queue of 30 meters = 6 cars * 1.9s = 11.4s clearance required
    adj_30 = coordinator.compute_queue_clearance_adjustment("J2", queue_m=30.0)
    assert round(adj_30, 1) == 11.4

    # Large queue clamps to maximum 18.0s to preserve stage boundaries
    adj_large = coordinator.compute_queue_clearance_adjustment("J2", queue_m=100.0)
    assert adj_large == 18.0


def test_corridor_step_and_telemetry():
    """Verifies that stepping the coordinator advances all 3 intersections and produces valid telemetry."""
    coordinator = CorridorCoordinator()

    # Step by 1.0 second
    telem = coordinator.step(dt=1.0)

    assert isinstance(telem, CorridorTelemetry)
    assert telem.corridor_time_s == 1.0
    assert telem.common_cycle_s == 90.0
    assert telem.bandwidth_efficiency_pct > 30.0
    assert len(telem.junctions) == 3

    # Check junction states
    j1_status = telem.junctions["J1"]
    assert j1_status.junction_id == "J1"
    assert j1_status.ring1_phase in range(1, 9)
    assert j1_status.ring2_phase in range(1, 9)

    t_dict = telem.to_dict()
    assert "junctions" in t_dict
    assert "J1" in t_dict["junctions"]
    assert "J2" in t_dict["junctions"]
    assert "J3" in t_dict["junctions"]


def test_corridor_emergency_preemption_wave():
    """Verifies that triggering corridor preemption broadcasts emergency wave across all 3 junctions."""
    coordinator = CorridorCoordinator()

    # Trigger Southbound preemption wave
    coordinator.trigger_corridor_preemption(target_phase=NEMAPhase.P6_SB_THROUGH)
    assert coordinator._corridor_preemption is True

    # Immediate step: J1 receives preemption immediately
    telem = coordinator.step(dt=0.5)
    assert telem.corridor_preemption_active is True
    assert telem.junctions["J1"].preemption_active is True

    # Clear preemption
    coordinator.clear_corridor_preemption()
    assert coordinator._corridor_preemption is False

    telem_cleared = coordinator.step(dt=0.5)
    assert telem_cleared.corridor_preemption_active is False
