"""Unit tests for Live Video-to-SUMO Co-Simulation Synchronizer."""

from __future__ import annotations

import time
import pytest

from src.sumo.co_simulation import (
    DigitalTwinSynchronizer,
    DigitalTwinTelemetry,
    SumoSyncWorker,
)


def test_telemetry_dataclass():
    telem = DigitalTwinTelemetry(
        timestamp=10.0,
        sim_time=10.0,
        queues_sumo={"N": 3, "S": 1, "E": 0, "W": 0},
        queues_cam={"N": 2.5, "S": 1.0, "E": 0.0, "W": 0.0},
        mean_speeds={"N": 25.0, "S": 30.0, "E": 40.0, "W": 40.0},
        waiting_time_s=12.5,
        throughput=5,
        active_phase=0,
        active_aspect="GREEN",
        tls_state="GGGGnnnn",
        divergence_mae=0.125,
        active_vehicles=4,
        injected_count=2,
    )

    d = telem.to_dict()
    assert d["sim_time"] == 10.0
    assert d["active_aspect"] == "GREEN"
    assert d["divergence_mae"] == 0.125
    assert d["queues_sumo"]["N"] == 3


def test_synchronizer_lifecycle_and_step():
    zero_demand = {"N": 0, "S": 0, "E": 0, "W": 0}
    sync = DigitalTwinSynchronizer(demand=zero_demand, duration=20.0, step_size=1.0)

    try:
        init_telem = sync.start(seed=42)
        assert sync.is_running is True
        assert init_telem is not None
        assert init_telem.sim_time == 0.0

        # Inject a vehicle
        ok, veh_id = sync.inject(track_id=10, class_name="truck", approach="South")
        assert ok is True
        assert "cam_S_10" in veh_id

        # Step synchronizer with camera queues
        cam_q = {"North": 3.0, "South": 2.5, "East": 0.0, "West": 1.0}
        telem = sync.step(video_time=1.0, camera_queues=cam_q)

        assert telem.sim_time == 1.0
        assert telem.injected_count == 1
        assert telem.queues_cam["S"] == 2.5
        assert "S" in telem.queues_sumo
        assert telem.divergence_mae >= 0.0
        assert telem.active_aspect in ("GREEN", "YELLOW", "ALL_RED")

        # Step again
        telem2 = sync.step(video_time=2.0)
        assert telem2.sim_time == 2.0
    finally:
        sync.close()
        assert sync.is_running is False


def test_synchronizer_phase_switch():
    zero_demand = {"N": 0, "S": 0, "E": 0, "W": 0}
    sync = DigitalTwinSynchronizer(
        demand=zero_demand, duration=30.0, step_size=1.0, min_green=2.0, yellow=1.0
    )

    try:
        sync.start(seed=42)
        initial_phase = sync.env.phase
        assert initial_phase == 0

        # Step beyond min_green
        sync.step(video_time=1.0)
        sync.step(video_time=2.0)

        # Trigger phase switch action=1
        telem = sync.step(action=1)
        # Should have switched phase to 1
        assert telem.active_phase == 1
    finally:
        sync.close()


def test_sumo_sync_worker_thread():
    zero_demand = {"N": 0, "S": 0, "E": 0, "W": 0}
    worker = SumoSyncWorker(
        demand=zero_demand,
        duration=20.0,
        step_rate_hz=10.0,  # Fast 10Hz stepping for quick test
        min_green=2.0,
        yellow=1.0,
        seed=42,
    )

    try:
        worker.start()
        # Give thread up to 1.5s to initialize
        for _ in range(15):
            if worker.is_running:
                break
            time.sleep(0.1)

        assert worker.is_running is True

        # Post asynchronous requests
        worker.post_injection(track_id=99, class_name="bus", approach="West")
        worker.post_camera_queues({"West": 4.0, "North": 1.0})
        worker.post_action(0)

        # Allow worker loop to process
        time.sleep(0.4)

        telem = worker.get_telemetry()
        assert telem is not None
        assert telem.sim_time > 0.0
        assert telem.queues_cam["W"] == 4.0
    finally:
        worker.stop(timeout=2.0)
        assert worker.is_running is False
