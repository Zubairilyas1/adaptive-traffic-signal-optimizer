"""Unit and Integration Tests for Live CCTV Worker and NEMA TS2 Actuator Pipeline."""

import time

from src.decision.fail_safe import FailureReason, HealthState
from src.inference.live_cctv_worker import (
    LiveCCTVTelemetry,
    LiveCCTVWorker,
    LiveCCTVWorkerConfig,
)
from src.inference.rtsp_stream import (
    StreamSourceConfig,
    StreamSourceType,
)


def test_live_cctv_worker_lifecycle():
    """Verifies complete start, frame acquisition, telemetry, and stop lifecycle."""
    stream_cfg = StreamSourceConfig(
        source_type=StreamSourceType.MOCK,
        uri="mock://intersection_north",
        target_fps=20.0,
        frame_width=640,
        frame_height=480,
    )
    worker_cfg = LiveCCTVWorkerConfig(
        stream_config=stream_cfg,
        use_detector=False,  # Use synthetic detector for deterministic unit tests
        target_fps=20.0,
        fail_safe_timeout_s=2.0,
    )

    worker = LiveCCTVWorker(worker_cfg)
    assert not worker._running

    worker.start()
    assert worker._running

    try:
        # Wait for worker to ingest and process at least 2 frames
        time.sleep(0.4)

        has_frame, frame, ts = worker.get_latest_frame()
        assert has_frame is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert ts > 0.0

        telemetry = worker.get_telemetry()
        assert isinstance(telemetry, LiveCCTVTelemetry)
        assert telemetry.is_running is True
        assert telemetry.total_frames_processed >= 1
        assert telemetry.health_state == HealthState.NORMAL
        assert len(telemetry.tls_state_12) == 12
        assert telemetry.ring1_phase in range(1, 9)
        assert telemetry.ring2_phase in range(1, 9)

        # Verify telemetry dictionary serializability
        t_dict = telemetry.to_dict()
        assert "ring1_phase" in t_dict
        assert "tls_state_12" in t_dict
        assert "queues_m" in t_dict
        assert "pcu_counts" in t_dict
    finally:
        worker.stop()

    assert not worker._running


def test_live_cctv_worker_preemption():
    """Verifies emergency vehicle preemption dispatch and clearing in live worker."""
    stream_cfg = StreamSourceConfig(
        source_type=StreamSourceType.MOCK,
        uri="mock://emergency_test",
        target_fps=20.0,
    )
    worker_cfg = LiveCCTVWorkerConfig(
        stream_config=stream_cfg,
        use_detector=False,
        target_fps=20.0,
    )
    worker = LiveCCTVWorker(worker_cfg)
    worker.start()

    try:
        time.sleep(0.3)

        # Trigger preemption for Phase 2 (Northbound Through)
        worker.request_preemption(phase_id=2)
        time.sleep(0.3)

        telem = worker.get_telemetry()
        assert telem.active_preemption == 2

        # Clear preemption
        worker.clear_preemption()
        time.sleep(0.2)

        telem_cleared = worker.get_telemetry()
        assert telem_cleared.active_preemption is None
    finally:
        worker.stop()


def test_live_cctv_worker_pause_resume():
    """Verifies pausing and resuming the inference loop."""
    stream_cfg = StreamSourceConfig(
        source_type=StreamSourceType.MOCK,
        uri="mock://pause_test",
        target_fps=20.0,
    )
    worker_cfg = LiveCCTVWorkerConfig(
        stream_config=stream_cfg,
        use_detector=False,
        target_fps=20.0,
    )
    worker = LiveCCTVWorker(worker_cfg)
    worker.start()

    try:
        time.sleep(0.3)
        initial_frames = worker.get_telemetry().total_frames_processed
        assert initial_frames >= 1

        worker.pause()
        time.sleep(0.3)
        paused_frames = worker.get_telemetry().total_frames_processed
        # Processing should have halted or increased by at most 1 in-flight frame
        assert paused_frames <= initial_frames + 1

        worker.resume()
        time.sleep(0.3)
        resumed_frames = worker.get_telemetry().total_frames_processed
        assert resumed_frames > paused_frames
    finally:
        worker.stop()


def test_live_cctv_worker_failsafe_timeout():
    """Verifies fail-safe fallback trigger on camera feed loss."""
    stream_cfg = StreamSourceConfig(
        source_type=StreamSourceType.MOCK,
        uri="mock://failsafe_test",
        target_fps=10.0,
    )
    worker_cfg = LiveCCTVWorkerConfig(
        stream_config=stream_cfg,
        use_detector=False,
        fail_safe_timeout_s=0.2,
    )
    worker = LiveCCTVWorker(worker_cfg)

    # Directly test the fail safe handler
    worker._handle_fail_safe_timeout()

    assert worker._health_state == HealthState.FAILSAFE_ACTIVE
    assert worker._failure_reason == FailureReason.FEED_TIMEOUT

    telem = worker.get_telemetry()
    assert telem.health_state == HealthState.FAILSAFE_ACTIVE
    assert telem.failure_reason == FailureReason.FEED_TIMEOUT
