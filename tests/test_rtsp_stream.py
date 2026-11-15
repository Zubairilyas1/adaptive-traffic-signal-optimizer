"""Unit and Integration Tests for Low-Latency Threaded RTSP Stream Engine."""

import time
import numpy as np

from src.inference.rtsp_stream import (
    MockTrafficStreamGenerator,
    StreamConnectionState,
    StreamHealthTelemetry,
    StreamSourceConfig,
    StreamSourceType,
    ThreadedFrameReader,
)


def test_stream_config_and_enums():
    """Verifies stream configuration defaults and lifecycle states."""
    cfg = StreamSourceConfig(source_type=StreamSourceType.MOCK, target_fps=30.0)
    assert cfg.source_type == StreamSourceType.MOCK
    assert cfg.target_fps == 30.0
    assert cfg.frame_width == 1280
    assert cfg.frame_height == 720
    assert cfg.reconnect_timeout_s == 3.0


def test_mock_traffic_stream_generator():
    """Verifies headless synthetic frame generator renders valid roadway frames."""
    gen = MockTrafficStreamGenerator(width=640, height=480, fps=30.0)
    frame1 = gen.read_frame()

    assert isinstance(frame1, np.ndarray)
    assert frame1.shape == (480, 640, 3)
    assert frame1.dtype == np.uint8
    assert gen.frame_idx == 1

    frame2 = gen.read_frame()
    assert gen.frame_idx == 2
    # Moving vehicles must create difference between frames
    assert not np.array_equal(frame1, frame2)


def test_threaded_frame_reader_lifecycle():
    """Verifies thread lifecycle: start, frame consumption, and clean stop."""
    cfg = StreamSourceConfig(source_type=StreamSourceType.MOCK, target_fps=60.0)
    reader = ThreadedFrameReader(cfg)

    assert not reader.is_alive()
    assert reader.state == StreamConnectionState.DISCONNECTED

    reader.start()
    assert reader.is_alive()

    # Allow worker thread to produce initial frames
    time.sleep(0.1)
    frame, ts, f_idx = reader.get_latest_frame(timeout=1.0)

    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert ts > 0.0
    assert f_idx >= 1
    assert reader.state == StreamConnectionState.CONNECTED

    reader.stop()
    assert not reader.is_alive()
    assert reader.state == StreamConnectionState.DISCONNECTED


def test_zero_lag_frame_dropping():
    """Simulates slow consumer and fast producer; verifies stale frames are dropped to guarantee fresh latency."""
    cfg = StreamSourceConfig(
        source_type=StreamSourceType.MOCK,
        target_fps=100.0,  # Fast producer: 100 FPS
    )
    reader = ThreadedFrameReader(cfg)
    reader.start()

    # Slow consumer sleeps while producer generates ~15 frames into 1-slot buffer
    time.sleep(0.15)

    frame, ts, f_idx = reader.get_latest_frame(timeout=1.0)
    assert frame is not None

    telemetry = reader.get_telemetry()
    # At least some frames must have been dropped to preserve real-time zero lag
    assert telemetry.total_dropped_frames > 0
    assert telemetry.total_captured_frames >= telemetry.total_dropped_frames
    assert telemetry.drop_rate_pct > 0.0

    reader.stop()


def test_telemetry_metrics_and_serialization():
    """Verifies accurate calculation and serialization of stream telemetry."""
    cfg = StreamSourceConfig(source_type=StreamSourceType.MOCK, target_fps=30.0)
    reader = ThreadedFrameReader(cfg)
    reader.start()

    time.sleep(0.1)
    reader.get_latest_frame(timeout=0.5)

    telem = reader.get_telemetry()
    assert isinstance(telem, StreamHealthTelemetry)
    assert telem.connection_state == StreamConnectionState.CONNECTED
    assert telem.total_captured_frames >= 1
    assert telem.uptime_s > 0.0

    d = telem.to_dict()
    assert "capture_fps" in d
    assert "drop_rate_pct" in d
    assert "connection_state" in d
    assert d["connection_state"] == "CONNECTED"

    reader.stop()


def test_failed_connection_state():
    """Verifies handling when an invalid RTSP URL or device is configured."""
    cfg = StreamSourceConfig(
        source_type=StreamSourceType.RTSP,
        uri="rtsp://nonexistent_host_12345:554/invalid_stream",
        max_backoff_s=0.5,
    )
    reader = ThreadedFrameReader(cfg)
    reader.start()

    time.sleep(0.2)
    # Cannot connect to invalid host -> transitions to RECONNECTING or FAILED
    assert reader.state in (StreamConnectionState.FAILED, StreamConnectionState.RECONNECTING, StreamConnectionState.CONNECTING)

    reader.stop()
    assert reader.state == StreamConnectionState.DISCONNECTED
