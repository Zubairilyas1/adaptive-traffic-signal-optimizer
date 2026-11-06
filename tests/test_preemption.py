"""Unit tests for Emergency Vehicle Preemption (EVP) detector and coordinator."""

from __future__ import annotations

import pytest

from src.analysis.preemption import (
    EmergencyPreemptionDetector,
    PreemptionCoordinator,
    PreemptionRequest,
    PreemptionState,
)


def test_emergency_preemption_detector_classes():
    detector = EmergencyPreemptionDetector()

    assert detector.is_emergency_class("ambulance") is True
    assert detector.is_emergency_class("fire_truck") is True
    assert detector.is_emergency_class("police") is True
    assert detector.is_emergency_class("police car") is True
    assert detector.is_emergency_class("car") is False
    assert detector.is_emergency_class("bus") is False


def test_emergency_preemption_temporal_confirmation():
    detector = EmergencyPreemptionDetector(min_consecutive_frames=3)

    det_ambulance = {
        "track_id": 999,
        "class_name": "ambulance",
        "bbox": [200, 100, 260, 220],
        "speed_km_h": 50.0,
        "approach": "North",
    }

    # Frame 1: candidate recorded, not yet confirmed
    reqs1 = detector.update([det_ambulance], timestamp=1.0)
    assert len(reqs1) == 0

    # Frame 2: candidate updated, not yet confirmed
    reqs2 = detector.update([det_ambulance], timestamp=1.1)
    assert len(reqs2) == 0

    # Frame 3: threshold reached -> confirmed preemption request!
    reqs3 = detector.update([det_ambulance], timestamp=1.2)
    assert len(reqs3) == 1
    req = reqs3[0]
    assert req.track_id == 999
    assert req.approach == "North"
    assert req.confirmed is True


def test_preemption_coordinator_lifecycle():
    coord = PreemptionCoordinator(
        yellow_clearance_s=2.0,
        all_red_clearance_s=1.0,
        post_passage_hold_s=2.0,
        max_preempt_duration_s=10.0,
    )

    req = PreemptionRequest(
        track_id=888,
        vehicle_class="fire_truck",
        approach="East",
        speed_km_h=45.0,
        distance_to_stopline_m=40.0,
        eta_seconds=3.2,
        timestamp=5.0,
    )

    # Currently Phase 0 (North-South) is active; preemption target is Phase 1 (East-West)
    coord.trigger_preemption(req, current_phase=0)
    assert coord.state == PreemptionState.CLEARING
    assert coord.target_phase == 1

    # Step 1: in yellow clearance
    st1, phase1, msg1 = coord.step(dt=1.0)
    assert st1 == PreemptionState.CLEARING
    assert "Fast Yellow" in msg1

    # Step 2: yellow expires -> all red
    st2, phase2, msg2 = coord.step(dt=1.0)
    assert st2 == PreemptionState.CLEARING
    assert "All-Red" in msg2

    # Step 3: clearance complete -> PREEMPTING with override_phase = 1
    st3, phase3, msg3 = coord.step(dt=1.0)
    assert st3 == PreemptionState.PREEMPTING
    assert phase3 == 1
    assert "EMERGENCY GREEN OPENED" in msg3

    # Step through max duration to recovery
    for _ in range(10):
        coord.step(dt=1.0)

    assert coord.state in (PreemptionState.RECOVERING, PreemptionState.IDLE)


def test_preemption_coordinator_cancel():
    coord = PreemptionCoordinator()
    req = PreemptionRequest(
        track_id=1, vehicle_class="police", approach="N", speed_km_h=40,
        distance_to_stopline_m=20, eta_seconds=2, timestamp=0,
    )
    coord.trigger_preemption(req, current_phase=1)
    assert coord.state != PreemptionState.IDLE

    coord.cancel()
    assert coord.state == PreemptionState.IDLE
    assert coord.active_request is None
