"""Unit tests for threaded VideoStreamWorker."""

from __future__ import annotations

from pathlib import Path
import time
import pytest

from src import config
from src.inference.stream_worker import VideoStreamWorker


@pytest.fixture
def sample_video() -> Path:
    p = config.SAMPLES_DIR / "traffic_clip.mp4"
    if not p.exists():
        pytest.skip("Sample video traffic_clip.mp4 not found")
    return p


def test_parse_source():
    assert VideoStreamWorker._parse_source(0) == 0
    assert VideoStreamWorker._parse_source("1") == 1
    assert VideoStreamWorker._parse_source("rtsp://192.168.1.1:554/live") == "rtsp://192.168.1.1:554/live"
    p = Path("test.mp4")
    assert VideoStreamWorker._parse_source(p) == "test.mp4"


def test_stream_worker_read_file(sample_video: Path):
    worker = VideoStreamWorker(sample_video, loop=False)
    worker.start()
    assert worker.is_running
    assert worker.fps > 0
    assert worker.frame_size[0] > 0
    assert worker.frame_size[1] > 0

    frames_received = 0
    for _ in range(5):
        ret, frame, ts = worker.read(timeout=1.0)
        if ret and frame is not None:
            frames_received += 1
            assert frame.shape[0] == worker.frame_size[1]
            assert frame.shape[1] == worker.frame_size[0]
            assert ts >= 0.0

    assert frames_received >= 3

    metrics = worker.get_metrics()
    assert metrics["is_running"] is True
    assert metrics["total_frames_decoded"] >= frames_received

    worker.stop()
    assert worker.is_running is False


def test_stream_worker_context_manager(sample_video: Path):
    with VideoStreamWorker(sample_video, loop=True) as worker:
        assert worker.is_running
        ret, frame, ts = worker.read(timeout=1.0)
        assert ret is True
        assert frame is not None
    assert worker.is_running is False


def test_stream_worker_loop_mode(sample_video: Path):
    worker = VideoStreamWorker(sample_video, loop=True)
    worker.start()
    try:
        # Should be able to continuously read frames without stopping
        ret, frame, _ = worker.read(timeout=1.0)
        assert ret is True
        assert frame is not None
    finally:
        worker.stop()
