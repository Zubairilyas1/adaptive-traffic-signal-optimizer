"""Smoke and Component Tests for Live CCTV / RTSP Studio UI."""

import py_compile
from pathlib import Path
import time

from src.decision.fail_safe import HealthState
from src.inference.live_cctv_worker import (
    LiveCCTVTelemetry,
    LiveCCTVWorker,
    LiveCCTVWorkerConfig,
)
from src.inference.rtsp_stream import (
    StreamSourceConfig,
    StreamSourceType,
)
from src.sumo.nema_network import NEMA_8_MOVEMENTS


def test_cctv_studio_syntax_compilation():
    """Verifies that 6_Live_CCTV_Studio.py compiles cleanly with zero syntax errors."""
    page_path = Path("app/pages/6_Live_CCTV_Studio.py")
    assert page_path.exists()
    py_compile.compile(str(page_path), doraise=True)


def test_cctv_studio_components_and_movements():
    """Verifies that all 8 movements and telemetry fields rendered in UI are valid."""
    assert len(NEMA_8_MOVEMENTS) == 8
    for pid in range(1, 9):
        assert pid in NEMA_8_MOVEMENTS
        assert len(NEMA_8_MOVEMENTS[pid]) > 0

    telem = LiveCCTVTelemetry()
    assert telem.health_state == HealthState.NORMAL
    assert len(telem.queues_m) == 8
    assert len(telem.pcu_counts) == 8
    assert telem.ring1_phase in range(1, 9)
    assert telem.ring2_phase in range(1, 9)


def test_cctv_studio_mock_worker_pipeline():
    """Verifies that the worker pipeline underlying the studio starts, grabs frames, and stops."""
    stream_cfg = StreamSourceConfig(
        source_type=StreamSourceType.MOCK,
        uri="mock://studio_test",
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
        has_frame, frame, ts = worker.get_latest_frame()
        assert has_frame is True
        assert frame is not None
        assert ts > 0.0

        telem = worker.get_telemetry()
        assert telem.is_running is True
        assert len(telem.tls_state_12) == 12
    finally:
        worker.stop()
