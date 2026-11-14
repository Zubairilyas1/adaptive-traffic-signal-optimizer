"""Video inference: detections -> counts -> signal plan and live streaming."""

from src.inference.analyze_video import count_clip, render_annotated_video
from src.inference.rtsp_stream import (
    MockTrafficStreamGenerator,
    StreamConnectionState,
    StreamHealthTelemetry,
    StreamSourceConfig,
    StreamSourceType,
    ThreadedFrameReader,
)
from src.inference.stream_worker import VideoStreamWorker

__all__ = [
    "VideoStreamWorker",
    "count_clip",
    "render_annotated_video",
    "MockTrafficStreamGenerator",
    "StreamConnectionState",
    "StreamHealthTelemetry",
    "StreamSourceConfig",
    "StreamSourceType",
    "ThreadedFrameReader",
]
