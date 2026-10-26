"""Unit tests for velocity-aware queue decomposition and kinematics."""

from __future__ import annotations

import pytest

from src.analysis.kinematics import (
    KinematicLaneCounter,
    MotionState,
    VehicleKinematics,
)
from src.analysis.roi import LaneROI


def test_vehicle_kinematics_stationary_becomes_halted():
    veh = VehicleKinematics(track_id=1, class_name="car")

    # Initial position at t=0
    veh.update((100.0, 100.0), timestamp=0.0)
    assert len(veh.history) == 1

    # Keep stationary for 1.0 second (t=0.2, 0.4, 0.6, 0.8, 1.0)
    for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
        state = veh.update((100.0, 100.0), timestamp=t)

    assert veh.speed_px_s < 1.0
    assert veh.speed_km_h < 1.0
    assert state == MotionState.HALTED
    assert veh.state == MotionState.HALTED


def test_vehicle_kinematics_moving_fast_becomes_flowing():
    veh = VehicleKinematics(track_id=2, class_name="bus")

    # Vehicle traveling 100 pixels per 0.2s (500 px/s ~ 90 km/h)
    x = 50.0
    for t in [0.0, 0.2, 0.4, 0.6]:
        state = veh.update((x, 100.0), timestamp=t)
        x += 100.0

    assert veh.speed_px_s > 100.0
    assert veh.speed_km_h > 15.0
    assert state == MotionState.FLOWING
    assert veh.pcu == 3.5  # Bus PCU factor


def test_kinematic_lane_counter_separates_halted_from_flowing():
    # Simple rectangular lane ROI: (0,0) to (200, 200)
    roi = LaneROI(
        lane_id="North_L1",
        approach="North",
        polygon=[(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)],
    )
    counter = KinematicLaneCounter([roi])

    # Simulation setup:
    # Car 101: Stopped at stopline (100, 180)
    # Car 102: Speeding through green from (100, 20) to (100, 160)
    timestamps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for t in timestamps:
        dets = [
            # Car 101: stationary bottom-center anchor at (100, 180)
            {"track_id": 101, "bbox": (80.0, 140.0, 120.0, 180.0), "class_name": "car"},
            # Car 102: moving fast downwards
            {"track_id": 102, "bbox": (80.0, t * 150.0, 120.0, t * 150.0 + 40.0), "class_name": "car"},
        ]
        snap = counter.update(dets, timestamp=t)

    # Inspect snapshot at t=1.0s
    lane_snap = snap.lane_snapshots["North_L1"]

    # Car 101 should be HALTED (queue backlog)
    assert lane_snap.halted_count == 1
    assert lane_snap.halted_pcu == 1.0

    # Car 102 should be FLOWING (throughput)
    assert lane_snap.flowing_count == 1
    assert lane_snap.flowing_pcu == 1.0

    # Approach level queries
    halted_dict = counter.current_halted_queue_pcu()
    assert halted_dict["North"] == 1.0

    flowing_dict = counter.current_flowing_throughput_pcu()
    assert flowing_dict["North"] == 1.0

    # Ensure timeline history is logged
    assert len(counter.timeline_history) == len(timestamps)
