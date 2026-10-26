"""Unit tests for the closed-loop real-time adaptive signal actuator."""

from __future__ import annotations

import pytest

from src import config
from src.analysis.kinematics import IntersectionSnapshot, LaneQueueSnapshot
from src.decision.actuator import (
    AdaptiveActuator,
    DecisionReason,
    SignalAspect,
)
from src.decision.intersection import load_intersection


@pytest.fixture
def test_intersection():
    path = config.ROOT / "configs" / "intersections" / "default.json"
    return load_intersection(path)


def test_actuator_enforces_min_green(test_intersection):
    actuator = AdaptiveActuator(test_intersection, min_green=7.0, max_green=30.0)

    # Initial state at t=0
    state = actuator.step(timestamp=0.0)
    assert state.active_aspect == SignalAspect.GREEN
    assert state.active_reason == DecisionReason.MIN_GREEN

    # At t=4.0s (less than 7.0s min_green), even if conflicting queue is high
    dummy_snap = IntersectionSnapshot(
        timestamp=4.0,
        approach_halted_pcu={"A": 0.0, "B": 15.0},
    )
    state = actuator.step(timestamp=4.0, snapshot=dummy_snap)
    assert state.active_aspect == SignalAspect.GREEN
    assert state.active_reason == DecisionReason.MIN_GREEN


def test_actuator_rest_in_green_when_no_conflicting_demand(test_intersection):
    actuator = AdaptiveActuator(test_intersection, min_green=7.0)

    # Advance past min_green to t=10.0s with zero conflicting queue
    dummy_snap = IntersectionSnapshot(
        timestamp=10.0,
        approach_halted_pcu={"A": 1.0, "B": 0.0},  # B has zero waiting cars!
    )
    state = actuator.step(timestamp=10.0, snapshot=dummy_snap)

    assert state.active_aspect == SignalAspect.GREEN
    assert state.active_reason == DecisionReason.REST_IN_GREEN


def test_actuator_triggers_gap_out_and_clearance(test_intersection):
    actuator = AdaptiveActuator(
        test_intersection,
        min_green=7.0,
        yellow_time=3.0,
        all_red_time=1.0,
        gap_threshold_s=3.0,
    )

    # 1. Step up to t=7.0s (min_green reached)
    actuator.step(0.0)
    actuator.step(7.0)

    # 2. At t=11.0s (gap is 11.0 - 0.0 = 11.0s > 3.0s), and approach B has 8.0 PCU waiting!
    dummy_snap = IntersectionSnapshot(
        timestamp=11.0,
        approach_halted_pcu={"A": 0.0, "B": 8.0},
    )
    state = actuator.step(timestamp=11.0, snapshot=dummy_snap)

    # Should trigger GAP-OUT and transition to YELLOW!
    assert state.active_aspect == SignalAspect.YELLOW
    assert len(actuator.events_log) >= 1
    assert actuator.events_log[-1].reason == DecisionReason.GAP_OUT

    # 3. Advance through yellow clearance (3.0s elapsed -> t=14.1s)
    state_yellow = actuator.step(timestamp=14.1, snapshot=dummy_snap)
    assert state_yellow.active_aspect == SignalAspect.RED  # Enters ALL-RED

    # 4. Advance through all-red clearance (1.0s elapsed -> t=15.2s)
    state_next = actuator.step(timestamp=15.2, snapshot=dummy_snap)
    assert state_next.active_aspect == SignalAspect.GREEN
    assert state_next.active_phase_id == "P2"  # Phase P2 (Approach B) is now GREEN!


def test_actuator_triggers_max_out(test_intersection):
    actuator = AdaptiveActuator(
        test_intersection,
        min_green=7.0,
        max_green=20.0,
        gap_threshold_s=100.0,  # Prevent gap-out
    )

    # Traffic flowing continuously on Phase 1 up to t=20.0s (max_green)
    # But Phase 2 has 10 PCU waiting!
    for t in range(0, 20):
        dummy_snap = IntersectionSnapshot(
            timestamp=float(t),
            approach_halted_pcu={"A": 0.0, "B": 10.0},
        )
        actuator.step(float(t), snapshot=dummy_snap)

    # At t=20.0s, max-out MUST trigger
    dummy_snap_20 = IntersectionSnapshot(
        timestamp=20.0,
        approach_halted_pcu={"A": 0.0, "B": 10.0},
    )
    state_20 = actuator.step(20.0, snapshot=dummy_snap_20)
    assert state_20.active_aspect == SignalAspect.YELLOW
    assert actuator.events_log[-1].reason == DecisionReason.MAX_OUT


def test_actuator_dilemma_zone_protection(test_intersection):
    actuator = AdaptiveActuator(
        test_intersection,
        min_green=7.0,
        max_green=30.0,
        gap_threshold_s=3.0,
    )

    # Advance past min_green to t=8.0s
    actuator.step(0.0)
    actuator.step(7.0)

    # A car is approaching at 45 km/h on Approach A within 1.0s headway!
    lane_snap = LaneQueueSnapshot(
        lane_id="A1",
        approach="A",
        flowing_count=1,
        avg_speed_km_h=45.0,
    )
    dummy_snap = IntersectionSnapshot(
        timestamp=8.0,
        lane_snapshots={"A1": lane_snap},
        approach_halted_pcu={"A": 0.0, "B": 10.0},
    )

    state = actuator.step(8.0, snapshot=dummy_snap)
    # Dilemma zone should HOLD green, not trigger gap-out!
    assert state.active_aspect == SignalAspect.GREEN
    assert state.active_reason == DecisionReason.DILEMMA_ZONE
