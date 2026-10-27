"""Threaded non-blocking video stream ingestion worker.

Solves the real-time buffer bloat problem in OpenCV:
- Runs frame decoding in a dedicated background worker thread.
- Uses a single-slot ring buffer so perception always reads the freshest frame.
- Automatically discards intermediate stale frames when inference latency > frame time.
- Supports Webcams (device 0, 1), RTSP/HTTP streams (IP cameras / phones), and looping MP4s.
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
import time
from typing import Any, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

StreamSource = Union[int, str, Path]


class VideoStreamWorker:
    """Thread-safe, non-blocking video capture daemon with zero-latency frame dropping."""

    def __init__(
        self,
        source: StreamSource = 0,
        target_fps: float | None = None,
        loop: bool = False,
        reconnect_delay_s: float = 2.0,
    ) -> None:
        self.raw_source = source
        self.source = self._parse_source(source)
        self.target_fps = target_fps
        self.loop = loop
        self.reconnect_delay_s = reconnect_delay_s

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()

        # Thread-shared frame buffer
        self._latest_frame: np.ndarray | None = None
        self._latest_timestamp: float = 0.0
        self._frame_count: int = 0
        self._dropped_count: int = 0
        self._start_time: float = 0.0
        self._native_fps: float = 25.0
        self._frame_size: tuple[int, int] = (0, 0)
        self._is_live_stream: bool = False

    @staticmethod
    def _parse_source(src: StreamSource) -> int | str:
        if isinstance(src, int):
            return src
        src_str = str(src).strip()
        if src_str.isdigit():
            return int(src_str)
        return src_str

    def _open_capture(self) -> bool:
        if self._cap is not None:
            self._cap.release()

        # Configure OpenCV VideoCapture
        if isinstance(self.source, int):
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.source)
            self._is_live_stream = True
        else:
            self._cap = cv2.VideoCapture(self.source)
            src_str = str(self.source).lower()
            self._is_live_stream = src_str.startswith(("rtsp://", "http://", "https://"))

        if not self._cap.isOpened():
            logger.warning(f"Failed to open video source: {self.source}")
            return False

        # Read native properties
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._native_fps = fps if fps > 0.0 else (self.target_fps or 25.0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._frame_size = (w, h)

        # Attempt to set lower buffer size on live streams to prevent driver latency
        if self._is_live_stream:
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        return True

    def start(self) -> VideoStreamWorker:
        """Start the background ingestion thread."""
        with self._lock:
            if self._running:
                return self
            if not self._open_capture():
                raise ConnectionError(f"Cannot initialize video source: {self.source}")

            self._running = True
            self._start_time = time.monotonic()
            self._frame_count = 0
            self._dropped_count = 0
            self._new_frame_event.clear()

            self._thread = threading.Thread(
                target=self._worker_loop,
                name=f"VideoStreamWorker-{self.source}",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop background ingestion and release camera resources."""
        with self._lock:
            self._running = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)

        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._latest_frame = None

    def _worker_loop(self) -> None:
        """Dedicated background thread continuously polling the camera hardware."""
        frame_interval = 1.0 / self._native_fps if self._native_fps > 0 else 0.033

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if not self._open_capture():
                    time.sleep(self.reconnect_delay_s)
                    continue

            ret, frame = self._cap.read()
            now = time.monotonic()

            if not ret or frame is None:
                if self.loop and not self._is_live_stream:
                    # Rewind file to beginning for infinite loop playback
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                elif self._is_live_stream:
                    # Connection dropped; pause and attempt reconnect
                    time.sleep(0.1)
                    continue
                else:
                    # File ended and loop is False
                    break

            # Atomic single-frame swap in the ring buffer
            with self._lock:
                if self._latest_frame is not None:
                    self._dropped_count += 1
                self._latest_frame = frame
                self._latest_timestamp = now - self._start_time
                self._frame_count += 1
                self._new_frame_event.set()

            # Pacing sleep for offline files to mimic real-time camera FPS
            if not self._is_live_stream:
                time.sleep(frame_interval * 0.85)

        with self._lock:
            self._running = False
            self._new_frame_event.set()

    def read(self, timeout: float = 0.5) -> tuple[bool, np.ndarray | None, float]:
        """Read the freshest available frame (non-blocking with timeout).

        Returns:
            (ret, frame, timestamp_seconds)
        """
        if not self._running and self._latest_frame is None:
            return False, None, 0.0

        signaled = self._new_frame_event.wait(timeout=timeout)
        if not signaled:
            with self._lock:
                if self._latest_frame is not None:
                    return True, self._latest_frame.copy(), self._latest_timestamp
            return False, None, 0.0

        with self._lock:
            self._new_frame_event.clear()
            if self._latest_frame is None:
                return False, None, 0.0
            frame_copy = self._latest_frame.copy()
            ts = self._latest_timestamp
            # Clear stored frame to indicate it has been consumed
            self._latest_frame = None

        return True, frame_copy, ts

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def fps(self) -> float:
        return self._native_fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    def get_metrics(self) -> dict[str, Any]:
        """Return diagnostic throughput and latency metrics."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time if self._start_time > 0 else 1e-5
            fps_effective = self._frame_count / max(0.1, elapsed)
            return {
                "source": str(self.source),
                "is_running": self._running,
                "is_live": self._is_live_stream,
                "native_fps": self._native_fps,
                "effective_fps": round(fps_effective, 1),
                "total_frames_decoded": self._frame_count,
                "dropped_frames": self._dropped_count,
                "width": self._frame_size[0],
                "height": self._frame_size[1],
            }

    def __enter__(self) -> VideoStreamWorker:
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
