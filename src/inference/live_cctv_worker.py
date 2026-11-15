"""Decoupled Live CCTV Worker and Closed-Loop NEMA TS2 Actuator Pipeline.

Connects the non-blocking ThreadedFrameReader to real-time object detection/tracking,
kinematic velocity estimation, 8-phase queue estimation, and the safety-certified
NEMA TS2 Dual-Ring Reinforcement Learning agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src import config
from src.analysis.roi import LaneROI
from src.decision.fail_safe import FailureReason, HealthState
from src.decision.nema_dual_ring import (
    PhaseInterval,
    RingBarrierStage,
)
from src.inference.rtsp_stream import (
    StreamHealthTelemetry,
    StreamSourceConfig,
    ThreadedFrameReader,
)
from src.rl.nema_agent import (
    NEMADualRingAgent,
    NEMAPolicyDecision,
)
from src.sumo.nema_network import build_nema_tls_state_string

logger = logging.getLogger(__name__)


@dataclass
class TrackedCCTVVehicle:
    """Vehicle tracklet state on the live CCTV viewport."""

    track_id: int
    class_name: str
    pcu: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]
    speed_kmh: float
    phase_id: int  # 1 to 8
    is_stopped: bool
    first_seen: float
    last_seen: float


@dataclass
class LiveCCTVTelemetry:
    """Comprehensive real-time telemetry snapshot from the CCTV pipeline."""

    is_running: bool = False
    is_paused: bool = False
    source_uri: str = ""
    health_state: HealthState = HealthState.NORMAL
    failure_reason: FailureReason = FailureReason.NONE
    stream_telemetry: StreamHealthTelemetry = field(default_factory=StreamHealthTelemetry)
    fps_inference: float = 0.0
    total_frames_processed: int = 0
    active_vehicles_count: int = 0
    stopped_vehicles_count: int = 0
    ring1_phase: int = 2
    ring1_interval: str = PhaseInterval.GREEN.value
    ring1_elapsed_s: float = 0.0
    ring2_phase: int = 6
    ring2_interval: str = PhaseInterval.GREEN.value
    ring2_elapsed_s: float = 0.0
    current_stage: str = RingBarrierStage.STAGE_1_MAJOR_NS.value
    is_at_barrier: bool = False
    active_preemption: Optional[int] = None
    queues_m: Dict[int, float] = field(default_factory=lambda: {p: 0.0 for p in range(1, 9)})
    pcu_counts: Dict[int, float] = field(default_factory=lambda: {p: 0.0 for p in range(1, 9)})
    tls_state_12: str = "rrrrrrrrrrrr"
    latest_action_name: str = "HOLD_BOTH"
    latest_reward: float = 0.0
    latest_rationale: str = "Initializing"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "source_uri": self.source_uri,
            "health_state": self.health_state.value,
            "failure_reason": self.failure_reason.value,
            "fps_inference": round(self.fps_inference, 1),
            "total_frames_processed": self.total_frames_processed,
            "active_vehicles_count": self.active_vehicles_count,
            "stopped_vehicles_count": self.stopped_vehicles_count,
            "ring1_phase": self.ring1_phase,
            "ring1_interval": self.ring1_interval,
            "ring1_elapsed_s": round(self.ring1_elapsed_s, 1),
            "ring2_phase": self.ring2_phase,
            "ring2_interval": self.ring2_interval,
            "ring2_elapsed_s": round(self.ring2_elapsed_s, 1),
            "current_stage": self.current_stage,
            "is_at_barrier": self.is_at_barrier,
            "active_preemption": self.active_preemption,
            "queues_m": {k: round(v, 1) for k, v in self.queues_m.items()},
            "pcu_counts": {k: round(v, 1) for k, v in self.pcu_counts.items()},
            "tls_state_12": self.tls_state_12,
            "latest_action_name": self.latest_action_name,
            "latest_reward": round(self.latest_reward, 3),
            "latest_rationale": self.latest_rationale,
            "capture_fps": round(self.stream_telemetry.capture_fps, 1),
            "drop_rate_pct": round(self.stream_telemetry.drop_rate_pct, 1),
            "latency_ms": round(self.stream_telemetry.frame_latency_ms, 1),
        }


@dataclass
class LiveCCTVWorkerConfig:
    """Configuration options for the Live CCTV Worker."""

    stream_config: StreamSourceConfig
    detector_weights: Optional[Union[Path, str]] = None
    use_detector: bool = True
    policy_mode: str = "max_pressure"
    hud_overlay: bool = True
    target_fps: float = 15.0
    fail_safe_timeout_s: float = 3.0
    min_speed_stopped_kmh: float = 5.0
    car_length_m: float = 5.0  # Approx road distance occupied per stopped vehicle in queue


class LiveCCTVWorker:
    """Threaded worker bridging live RTSP frames to NEMA TS2 adaptive control."""

    def __init__(
        self,
        config: LiveCCTVWorkerConfig,
        agent: Optional[NEMADualRingAgent] = None,
        rois: Optional[List[LaneROI]] = None,
    ) -> None:
        self.config = config
        self.reader = ThreadedFrameReader(config.stream_config)
        self.agent = agent or NEMADualRingAgent(policy_mode=config.policy_mode)
        self.rois = rois or []

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False
        self._lock = threading.Lock()

        # Fail-safe watchdog state
        self._last_successful_frame_ts = time.time()
        self._health_state = HealthState.NORMAL
        self._failure_reason = FailureReason.NONE

        # Vehicle tracking & kinematics
        self._detector: Optional[Any] = None
        self._tracked_vehicles: Dict[int, TrackedCCTVVehicle] = {}
        self._preemption_request: Optional[int] = None

        # Output buffers
        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._latest_frame_ts: float = 0.0
        self._frames_processed: int = 0
        self._fps_inference: float = 0.0
        self._latest_decision: Optional[NEMAPolicyDecision] = None
        self._telemetry = LiveCCTVTelemetry(source_uri=config.stream_config.uri)

        # Initialize detector if requested and feasible
        if self.config.use_detector:
            self._init_detector()

    def _init_detector(self) -> None:
        """Lazily initialize YOLOv8 detector if available."""
        try:
            from src.models.detector import VehicleDetector

            weights = self.config.detector_weights
            self._detector = VehicleDetector(weights=weights)
            logger.info("LiveCCTVWorker: Loaded YOLOv8 detector successfully.")
        except Exception as e:
            logger.warning(
                f"LiveCCTVWorker: YOLO detector unavailable ({e}). Fallback to synthetic tracker."
            )
            self._detector = None

    def start(self) -> None:
        """Starts frame ingestion and inference daemon thread."""
        if self._running:
            return

        self._running = True
        self._paused = False
        self.reader.start()

        self._thread = threading.Thread(
            target=self._worker_loop,
            name="LiveCCTVWorkerThread",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"LiveCCTVWorker started on source: {self.config.stream_config.uri}")

    def stop(self) -> None:
        """Stops inference and cleans up frame ingestion resources."""
        self._running = False
        self.reader.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("LiveCCTVWorker stopped.")

    def pause(self) -> None:
        """Pauses frame processing without closing camera connection."""
        self._paused = True

    def resume(self) -> None:
        """Resumes active frame processing."""
        self._paused = False

    def request_preemption(self, phase_id: int) -> None:
        """Invocates high-priority emergency preemption on specified NEMA phase (1..8)."""
        if 1 <= phase_id <= 8:
            with self._lock:
                self._preemption_request = phase_id
            logger.info(f"LiveCCTVWorker: Preemption requested for Phase {phase_id}")

    def clear_preemption(self) -> None:
        """Clears active emergency preemption request."""
        with self._lock:
            self._preemption_request = None
            self.agent.controller.preemption_target = None
        logger.info("LiveCCTVWorker: Preemption cleared.")

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Returns the most recent annotated frame in thread-safe fashion."""
        with self._lock:
            if self._latest_annotated_frame is None:
                return False, None, 0.0
            return True, self._latest_annotated_frame.copy(), self._latest_frame_ts

    def get_telemetry(self) -> LiveCCTVTelemetry:
        """Returns current operational telemetry snapshot."""
        with self._lock:
            # Sync stream telemetry
            self._telemetry.stream_telemetry = self.reader.get_telemetry()
            self._telemetry.is_running = self._running
            self._telemetry.is_paused = self._paused
            self._telemetry.fps_inference = self._fps_inference
            self._telemetry.total_frames_processed = self._frames_processed
            self._telemetry.health_state = self._health_state
            self._telemetry.failure_reason = self._failure_reason

            # Sync controller state
            ctrl = self.agent.controller
            self._telemetry.ring1_phase = ctrl.ring1_phase.value
            self._telemetry.ring1_interval = ctrl.ring1_interval.value
            self._telemetry.ring1_elapsed_s = ctrl.ring1_elapsed_s
            self._telemetry.ring2_phase = ctrl.ring2_phase.value
            self._telemetry.ring2_interval = ctrl.ring2_interval.value
            self._telemetry.ring2_elapsed_s = ctrl.ring2_elapsed_s
            self._telemetry.current_stage = ctrl.current_stage.value
            self._telemetry.is_at_barrier = (
                ctrl.ring1_interval == PhaseInterval.RED_HOLD
                or ctrl.ring2_interval == PhaseInterval.RED_HOLD
            )
            self._telemetry.active_preemption = self._preemption_request

            # 12-link TLS signal string
            self._telemetry.tls_state_12 = build_nema_tls_state_string(
                ctrl.ring1_phase,
                ctrl.ring1_interval,
                ctrl.ring2_phase,
                ctrl.ring2_interval,
            )

            return self._telemetry

    def _worker_loop(self) -> None:
        """Continuous inference and actuation cycle."""
        last_step_time = time.time()
        fps_calc_time = time.time()
        frames_since_calc = 0

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            loop_start = time.time()

            # 1. Read latest frame from non-blocking circular queue
            has_frame, raw_frame, frame_ts = self.reader.read(timeout=0.08)

            now = time.time()
            if not has_frame or raw_frame is None:
                # Check fail-safe watchdog timer
                if (now - self._last_successful_frame_ts) > self.config.fail_safe_timeout_s:
                    self._handle_fail_safe_timeout()
                time.sleep(0.01)
                continue

            self._last_successful_frame_ts = now
            self._health_state = HealthState.NORMAL
            self._failure_reason = FailureReason.NONE

            # 2. Vehicle Detection & Tracking
            detections = self._detect_vehicles(raw_frame)

            # 3. Approach / Movement Phase Classification ($\phi_1..\phi_8$)
            queues_m, pcu_counts, headways = self._classify_phases_and_queues(
                detections, raw_frame.shape
            )

            # 4. Actuate NEMA TS2 Agent
            dt = max(0.02, min(0.5, now - last_step_time))
            last_step_time = now

            with self._lock:
                preempt_call = self._preemption_request

            decision = self.agent.step(
                dt=dt,
                queues_m=queues_m,
                pcu_counts=pcu_counts,
                headways=headways,
                preemption_call=preempt_call,
            )

            # 5. Render HUD Overlay
            if self.config.hud_overlay:
                annotated_frame = self._render_hud(
                    raw_frame, detections, decision, queues_m, pcu_counts
                )
            else:
                annotated_frame = raw_frame

            # 6. Update thread-safe output variables
            with self._lock:
                self._latest_annotated_frame = annotated_frame
                self._latest_frame_ts = frame_ts
                self._frames_processed += 1
                self._latest_decision = decision
                self._telemetry.queues_m = queues_m
                self._telemetry.pcu_counts = pcu_counts
                self._telemetry.latest_action_name = decision.action_name
                self._telemetry.latest_reward = decision.reward
                self._telemetry.latest_rationale = decision.rationale
                self._telemetry.active_vehicles_count = len(detections)
                self._telemetry.stopped_vehicles_count = sum(
                    1 for v in self._tracked_vehicles.values() if v.is_stopped
                )

            # Calculate inference FPS
            frames_since_calc += 1
            if now - fps_calc_time >= 1.0:
                self._fps_inference = frames_since_calc / (now - fps_calc_time)
                frames_since_calc = 0
                fps_calc_time = now

            # Sleep briefly if running faster than target FPS to conserve CPU
            elapsed = time.time() - loop_start
            target_period = 1.0 / self.config.target_fps
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

    def _detect_vehicles(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Runs detector on frame, or generates synthetic detections if no model loaded."""
        if self._detector is not None:
            try:
                dets = self._detector(frame, track=True)
                return dets
            except Exception as e:
                logger.debug(f"Detector error: {e}, falling back to synthetic generator")

        # Fallback synthetic detection generator for test/offline/mock mode
        h, w = frame.shape[:2]
        t = time.time()
        synthetic_dets: List[Dict[str, Any]] = []

        # Generate 4-8 synthetic vehicles moving in realistic corridors
        classes = ["car", "car", "bus", "motorcycle", "truck"]
        for i in range(6):
            base_x = (int((t * 40 + i * 160)) % (w - 80)) + 40
            base_y = (int((t * 30 + i * 120)) % (h - 80)) + 40
            c_name = classes[i % len(classes)]
            synthetic_dets.append(
                {
                    "track_id": 100 + i,
                    "class_name": c_name,
                    "confidence": 0.88,
                    "bbox": (
                        float(base_x - 20),
                        float(base_y - 15),
                        float(base_x + 20),
                        float(base_y + 15),
                    ),
                    "pcu": config.pcu_for(c_name),
                }
            )
        return synthetic_dets

    def _classify_phases_and_queues(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, ...],
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        """Maps detected vehicles to NEMA phases (1..8) and computes queues and PCU."""
        h, w = frame_shape[0], frame_shape[1]
        now = time.time()

        queues_m: Dict[int, float] = {p: 0.0 for p in range(1, 9)}
        pcu_counts: Dict[int, float] = {p: 0.0 for p in range(1, 9)}
        headways: Dict[int, float] = {p: 2.5 for p in range(1, 9)}

        current_track_ids = set()

        for det in detections:
            track_id = det.get("track_id", 0)
            bbox = det.get("bbox", (0.0, 0.0, 0.0, 0.0))
            class_name = det.get("class_name", "car")
            pcu_val = det.get("pcu", 1.0)

            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            norm_x = cx / max(1.0, float(w))
            norm_y = cy / max(1.0, float(h))

            # Classify movement phase based on spatial geometry
            phase_id = self._map_position_to_nema_phase(norm_x, norm_y)

            # Kinematic velocity estimation
            if track_id in self._tracked_vehicles:
                prev = self._tracked_vehicles[track_id]
                dt = max(0.01, now - prev.last_seen)
                dx_px = cx - prev.center[0]
                dy_px = cy - prev.center[1]
                dist_px = np.hypot(dx_px, dy_px)
                # Calibrate roughly 1 px = 0.05m
                speed_ms = (dist_px * 0.05) / dt
                speed_kmh = speed_ms * 3.6
                is_stopped = speed_kmh < self.config.min_speed_stopped_kmh
            else:
                speed_kmh = 30.0  # initial assumption
                is_stopped = False

            first_seen = (
                self._tracked_vehicles[track_id].first_seen
                if track_id in self._tracked_vehicles
                else now
            )
            self._tracked_vehicles[track_id] = TrackedCCTVVehicle(
                track_id=track_id,
                class_name=class_name,
                pcu=pcu_val,
                bbox=bbox,
                center=(cx, cy),
                speed_kmh=speed_kmh,
                phase_id=phase_id,
                is_stopped=is_stopped,
                first_seen=first_seen,
                last_seen=now,
            )
            current_track_ids.add(track_id)

            # Accumulate PCU
            pcu_counts[phase_id] = pcu_counts.get(phase_id, 0.0) + pcu_val

            # Accumulate queue depth if stopped
            if is_stopped:
                queues_m[phase_id] = (
                    queues_m.get(phase_id, 0.0) + self.config.car_length_m * pcu_val
                )

        # Evict stale tracklets older than 3.0s
        for tid in list(self._tracked_vehicles.keys()):
            if tid not in current_track_ids and (now - self._tracked_vehicles[tid].last_seen) > 3.0:
                del self._tracked_vehicles[tid]

        return queues_m, pcu_counts, headways

    @staticmethod
    def _map_position_to_nema_phase(norm_x: float, norm_y: float) -> int:
        """Determines NEMA phase (1..8) from normalized coordinates (0..1).

        Standard Dual-Ring Movement Mapping:
        - Major Arterial (N-S):
            - Top-Left (Northbound): Φ2 (Through/Right) or Φ5 (Protected Left)
            - Top-Right (Southbound): Φ6 (Through/Right) or Φ1 (Protected Left)
        - Minor Cross-Street (E-W):
            - Bottom-Left (Eastbound): Φ4 (Through/Right) or Φ7 (Protected Left)
            - Bottom-Right (Westbound): Φ8 (Through/Right) or Φ3 (Protected Left)
        """
        if norm_y < 0.5:
            # North-South approaches
            if norm_x < 0.5:
                # Northbound corridor
                return 5 if norm_x > 0.35 else 2  # Inner pocket = left turn (Φ5)
            else:
                # Southbound corridor
                return 1 if norm_x < 0.65 else 6  # Inner pocket = left turn (Φ1)
        else:
            # East-West approaches
            if norm_x < 0.5:
                # Eastbound corridor
                return 7 if norm_y > 0.75 else 4  # Inner lane = left turn (Φ7)
            else:
                # Westbound corridor
                return 3 if norm_y > 0.75 else 8  # Inner lane = left turn (Φ3)

    def _render_hud(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        decision: NEMAPolicyDecision,
        queues_m: Dict[int, float],
        pcu_counts: Dict[int, float],
    ) -> np.ndarray:
        """Draws visual HUD banner, bounding boxes, and phase telemetry onto frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. Draw Bounding Boxes with Class and Speed
        for det in detections:
            tid = det.get("track_id", 0)
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            veh = self._tracked_vehicles.get(tid)

            # Color code: Red if stopped, Green if moving
            box_color = (0, 0, 240) if (veh and veh.is_stopped) else (0, 220, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

            speed_str = f"{veh.speed_kmh:.0f}km/h" if veh else ""
            phase_str = f"P{veh.phase_id}" if veh else ""
            label = f"#{tid} {det.get('class_name', 'veh')} {phase_str} {speed_str}".strip()

            cv2.putText(
                vis,
                label,
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 2. Top HUD Telemetry Bar
        banner_h = 60
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)

        # Ring 1 Badge
        r1_color = (
            (0, 220, 0)
            if decision.ring1_interval == PhaseInterval.GREEN
            else (0, 220, 220)
            if decision.ring1_interval == PhaseInterval.YELLOW
            else (0, 0, 220)
        )
        r1_text = f"RING 1: {decision.ring1_phase.name} [{decision.ring1_interval.value}] ({decision.ring1_elapsed_s:.1f}s)"
        cv2.putText(
            vis,
            r1_text,
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            r1_color,
            2,
            cv2.LINE_AA,
        )

        # Ring 2 Badge
        r2_color = (
            (0, 220, 0)
            if decision.ring2_interval == PhaseInterval.GREEN
            else (0, 220, 220)
            if decision.ring2_interval == PhaseInterval.YELLOW
            else (0, 0, 220)
        )
        r2_text = f"RING 2: {decision.ring2_phase.name} [{decision.ring2_interval.value}] ({decision.ring2_elapsed_s:.1f}s)"
        cv2.putText(
            vis,
            r2_text,
            (15, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            r2_color,
            2,
            cv2.LINE_AA,
        )

        # Barrier & Stage Info (Right-aligned)
        stage_name = "STAGE 1: MAJOR N-S" if decision.current_stage == RingBarrierStage.STAGE_1_MAJOR_NS else "STAGE 2: MINOR E-W"
        barrier_text = f"BARRIER: {stage_name} | ACTION: {decision.action_name}"
        cv2.putText(
            vis,
            barrier_text,
            (max(200, w - 460), 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Preemption & Rate Info
        preempt_str = (
            f"PREEMPTION: ACTIVE ({decision.active_preemption.name})"
            if decision.active_preemption
            else "PREEMPTION: NONE"
        )
        rate_str = f"INF FPS: {self._fps_inference:.1f} | {preempt_str}"
        cv2.putText(
            vis,
            rate_str,
            (max(200, w - 460), 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255) if decision.active_preemption else (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        return vis

    def _handle_fail_safe_timeout(self) -> None:
        """Invokes NEMA TS2 fail-safe fixed-split fallback upon feed loss."""
        self._health_state = HealthState.FAILSAFE_ACTIVE
        self._failure_reason = FailureReason.FEED_TIMEOUT
        logger.warning(
            "LiveCCTVWorker: Video stream heartbeat lost (>3s)! Fallback to NEMA fixed-time."
        )

        # Step controller forward safely in fixed-time mode
        dt = 0.5
        self.agent.controller.step(
            dt=dt,
            vehicle_calls={p: True for p in range(1, 9)},
            headways={p: 2.5 for p in range(1, 9)},
            preemption_call=None,
        )
