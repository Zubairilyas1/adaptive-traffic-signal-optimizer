"""Vehicle kinematics and velocity-aware queue decomposition.

Differentiates between halted/waiting vehicles (queue backlog) and
flowing/discharging vehicles (throughput) using trajectory tracking.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Iterable, List, Optional, Tuple

from src import config
from src.config import pcu_for
from src.analysis.roi import LaneROI, anchor_point, point_in_polygon

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


class MotionState(Enum):
    """Kinematic motion state of a tracked vehicle."""

    HALTED = "halted"  # Stopped or crawling (< 5 km/h) -> Waiting Queue Backlog
    CREEPING = "creeping"  # Slow / Decelerating (5 to 15 km/h) -> Approaching Queue
    FLOWING = "flowing"  # Cruising / Discharging (>= 15 km/h) -> Active Throughput


@dataclass
class TrajectoryPoint:
    x: float
    y: float
    timestamp: float


class VehicleKinematics:
    """Tracks position history and estimates smoothed velocity for a single vehicle."""

    def __init__(
        self,
        track_id: int,
        class_name: str,
        max_history: int = 15,
        px_to_meter: float = 0.05,  # Nominal camera calibration (1 px ~ 5 cm)
    ):
        self.track_id = track_id
        self.class_name = config.canonical_class(class_name)
        self.pcu = pcu_for(self.class_name)
        self.max_history = max_history
        self.px_to_meter = px_to_meter
        self.history: Deque[TrajectoryPoint] = deque(maxlen=max_history)

        self.speed_px_s: float = 0.0
        self.speed_km_h: float = 0.0
        self.state: MotionState = MotionState.FLOWING
        self.halted_duration: float = 0.0
        self._last_state_change: float = 0.0
        self.last_bbox: Optional[tuple[float, float, float, float]] = None

    def update(self, pt: Point, timestamp: float) -> MotionState:
        """Update trajectory and compute velocity."""
        curr = TrajectoryPoint(pt[0], pt[1], timestamp)

        if self.history:
            prev = self.history[-1]
            dt = curr.timestamp - prev.timestamp
            if dt > 1e-4:
                dx = curr.x - prev.x
                dy = curr.y - prev.y
                dist_px = (dx * dx + dy * dy) ** 0.5
                inst_speed_px_s = dist_px / dt
                inst_speed_km_h = (dist_px * self.px_to_meter / dt) * 3.6

                # Exponential Moving Average (EMA) smoothing to reduce detection box jitter
                alpha = 0.4
                self.speed_px_s = alpha * inst_speed_px_s + (1.0 - alpha) * self.speed_px_s
                self.speed_km_h = alpha * inst_speed_km_h + (1.0 - alpha) * self.speed_km_h

                # Classify motion state (Industry standard thresholds)
                # < 5 km/h (or ~25 px/s) = Halted queue
                if self.speed_km_h < 5.0 or self.speed_px_s < 25.0:
                    self.halted_duration += dt
                    if self.halted_duration >= 0.5:
                        self.state = MotionState.HALTED
                elif self.speed_km_h >= 15.0 or self.speed_px_s >= 80.0:
                    self.halted_duration = 0.0
                    self.state = MotionState.FLOWING
                else:
                    self.halted_duration = 0.0
                    self.state = MotionState.CREEPING

        self.history.append(curr)
        return self.state


@dataclass
class LaneQueueSnapshot:
    """Instantaneous queue and flow metrics for a lane at a specific second."""

    lane_id: str
    approach: str
    halted_count: int = 0
    halted_pcu: float = 0.0
    flowing_count: int = 0
    flowing_pcu: float = 0.0
    avg_speed_km_h: float = 0.0
    queue_length_m: float = 0.0


@dataclass
class IntersectionSnapshot:
    """Intersection-wide queue state at timestamp t."""

    timestamp: float
    lane_snapshots: dict[str, LaneQueueSnapshot] = field(default_factory=dict)
    approach_halted_pcu: dict[str, float] = field(default_factory=dict)
    approach_flowing_pcu: dict[str, float] = field(default_factory=dict)
    approach_queue_m: dict[str, float] = field(default_factory=dict)
    total_halted_pcu: float = 0.0
    total_queue_m: float = 0.0


class KinematicLaneCounter:
    """Velocity-aware lane counter separating true waiting queues from flowing vehicles."""

    def __init__(self, rois: Iterable[LaneROI], px_to_meter: float = 0.05, ipm_transformer: Optional[Any] = None):
        self.rois = list(rois)
        self.lane_approach = {r.lane_id: r.approach for r in self.rois}
        self.px_to_meter = px_to_meter

        if ipm_transformer is None:
            from src.analysis.ipm import IPMTransformer
            self.ipm = IPMTransformer()
        else:
            self.ipm = ipm_transformer

        self.vehicles: dict[int, VehicleKinematics] = {}
        self._visited: set[tuple[int, str]] = set()
        self.timeline_history: list[IntersectionSnapshot] = []

    def update(self, detections: Iterable[Any], timestamp: float = 0.0) -> IntersectionSnapshot:
        """Process frame detections, update kinematic states, and return instantaneous queue."""
        active_in_lane: dict[str, list[VehicleKinematics]] = defaultdict(list)

        for det in detections:
            track_id = getattr(det, "track_id", None)
            if isinstance(det, dict):
                track_id = det.get("track_id")
                bbox = det.get("bbox")
                class_name = det.get("class_name", "car")
            else:
                bbox = getattr(det, "bbox", None)
                class_name = getattr(det, "class_name", "car")

            if track_id is None or bbox is None:
                continue

            tid = int(track_id)
            if tid not in self.vehicles:
                self.vehicles[tid] = VehicleKinematics(
                    track_id=tid,
                    class_name=class_name,
                    px_to_meter=self.px_to_meter,
                )

            veh = self.vehicles[tid]
            veh.last_bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            pt = anchor_point(bbox)
            veh.update(pt, timestamp)

            # Check which lane ROI contains this vehicle
            for roi in self.rois:
                if point_in_polygon(pt, roi.polygon):
                    active_in_lane[roi.lane_id].append(veh)
                    self._visited.add((tid, roi.lane_id))

        # Build snapshot for this frame
        snapshot = self._build_snapshot(timestamp, active_in_lane)
        self.timeline_history.append(snapshot)
        return snapshot

    def _build_snapshot(
        self,
        timestamp: float,
        active_in_lane: dict[str, list[VehicleKinematics]],
    ) -> IntersectionSnapshot:
        lane_snaps: dict[str, LaneQueueSnapshot] = {}
        app_halted: dict[str, float] = defaultdict(float)
        app_flowing: dict[str, float] = defaultdict(float)
        app_queue_m: dict[str, float] = defaultdict(float)

        for roi in self.rois:
            lane_id = roi.lane_id
            approach = roi.approach
            vehs = active_in_lane.get(lane_id, [])

            halted = [v for v in vehs if v.state == MotionState.HALTED]
            flowing = [v for v in vehs if v.state == MotionState.FLOWING]

            h_pcu = sum(v.pcu for v in halted)
            f_pcu = sum(v.pcu for v in flowing)
            avg_spd = (sum(v.speed_km_h for v in vehs) / len(vehs)) if vehs else 0.0

            halted_boxes = [v.last_bbox for v in halted if v.last_bbox is not None]
            queue_m = self.ipm.compute_queue_length_meters(halted_boxes)

            lane_snaps[lane_id] = LaneQueueSnapshot(
                lane_id=lane_id,
                approach=approach,
                halted_count=len(halted),
                halted_pcu=h_pcu,
                flowing_count=len(flowing),
                flowing_pcu=f_pcu,
                avg_speed_km_h=avg_spd,
                queue_length_m=round(queue_m, 1),
            )

            app_halted[approach] += h_pcu
            app_flowing[approach] += f_pcu
            app_queue_m[approach] = max(app_queue_m[approach], round(queue_m, 1))

        return IntersectionSnapshot(
            timestamp=timestamp,
            lane_snapshots=lane_snaps,
            approach_halted_pcu=dict(app_halted),
            approach_flowing_pcu=dict(app_flowing),
            approach_queue_m=dict(app_queue_m),
            total_halted_pcu=sum(app_halted.values()),
            total_queue_m=sum(app_queue_m.values()),
        )

    def current_halted_queue_pcu(self) -> dict[str, float]:
        """Return the latest instantaneous halted queue PCU per approach."""
        if not self.timeline_history:
            return {a: 0.0 for a in set(self.lane_approach.values())}
        return self.timeline_history[-1].approach_halted_pcu

    def current_flowing_throughput_pcu(self) -> dict[str, float]:
        """Return the latest instantaneous flowing throughput PCU per approach."""
        if not self.timeline_history:
            return {a: 0.0 for a in set(self.lane_approach.values())}
        return self.timeline_history[-1].approach_flowing_pcu
