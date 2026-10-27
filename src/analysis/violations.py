"""Red-light crossing and speed enforcement detection engine.

Detects vehicles crossing the stopline while the approach signal aspect is RED,
captures high-resolution photographic evidence, and logs infractions to SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src import config
from src.analysis.roi import LaneROI, anchor_point
from src.db.database import log_violation

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.decision.actuator import SignalAspect

Point = Tuple[float, float]
BBox = Sequence[float]


def _ccw(A: Point, B: Point, C: Point) -> bool:
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def line_segments_intersect(A: Point, B: Point, C: Point, D: Point) -> bool:
    """Return True if line segment AB intersects line segment CD."""
    return _ccw(A, C, D) != _ccw(B, C, D) and _ccw(A, B, C) != _ccw(A, B, D)


@dataclass
class Stopline:
    """A physical stopline threshold demarcating the intersection entry boundary."""

    line_id: str
    approach: str
    pt1: Point
    pt2: Point
    lane_id: Optional[str] = None


@dataclass
class ViolationEvent:
    """Record of a vehicle running a red signal light."""

    track_id: int
    vehicle_class: str
    speed_km_h: float
    timestamp: float
    approach: str
    image_path: str
    bbox: tuple[float, float, float, float]
    aspect: str = "RED"
    line_id: str = ""


def stoplines_from_rois(rois: Iterable[LaneROI]) -> list[Stopline]:
    """Auto-generate stoplines from the downstream exit edge of each lane ROI polygon."""
    stoplines = []
    for r in rois:
        poly = r.polygon
        if len(poly) < 4:
            continue
        # In a 4-point ROI polygon ordered clockwise/counterclockwise,
        # the downstream segment entering the intersection is typically between pt[1] and pt[2]
        # or the edge with highest y-value (or lowest, depending on camera perspective).
        # We take the edge that spans the width of the lane closest to the intersection.
        pts_sorted_by_y = sorted(poly, key=lambda p: p[1], reverse=True)
        pt1, pt2 = pts_sorted_by_y[0], pts_sorted_by_y[1]
        stoplines.append(
            Stopline(
                line_id=f"stopline_{r.lane_id}",
                approach=r.approach,
                pt1=(float(pt1[0]), float(pt1[1])),
                pt2=(float(pt2[0]), float(pt2[1])),
                lane_id=r.lane_id,
            )
        )
    return stoplines


class RedLightViolationDetector:
    """Monitors vehicle trajectories across stoplines during RED signal phases."""

    def __init__(
        self,
        stoplines: Iterable[Stopline],
        evidence_dir: Path | str | None = None,
        speed_tolerance_km_h: float = 5.0,
    ) -> None:
        self.stoplines = list(stoplines)
        self.evidence_dir = Path(evidence_dir or config.VIOLATIONS_DIR)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.speed_tolerance_km_h = speed_tolerance_km_h

        # State tracking
        self._last_positions: dict[int, Point] = {}
        self._violated_tracks: set[int] = set()
        self._violations: list[ViolationEvent] = []

    def reset(self) -> None:
        """Reset internal trajectory history and violation flags."""
        self._last_positions.clear()
        self._violated_tracks.clear()
        self._violations.clear()

    def update(
        self,
        detections: list[dict[str, Any]],
        aspect_by_approach: dict[str, SignalAspect | str],
        timestamp: float,
        frame: np.ndarray | None = None,
        intersection_id: str = "default_intersection",
    ) -> list[ViolationEvent]:
        """Evaluate vehicle positions against stoplines and active red aspects.

        Args:
            detections: List of detection dicts with 'track_id', 'bbox', 'class_name', 'speed_km_h'.
            aspect_by_approach: Map from approach name to current SignalAspect ('red', 'green', 'yellow').
            timestamp: Video timestamp in seconds.
            frame: Optional BGR frame to crop and save photographic violation proof.
            intersection_id: ID of intersection for database audit logging.

        Returns:
            List of newly flagged ViolationEvent records.
        """
        new_violations: list[ViolationEvent] = []

        # Standardize aspect lookup strings to uppercase
        norm_aspects: dict[str, str] = {}
        for app, asp in aspect_by_approach.items():
            val = asp.value.upper() if hasattr(asp, "value") else str(asp).upper()
            norm_aspects[str(app)] = val

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            curr_pos = anchor_point(bbox)
            prev_pos = self._last_positions.get(track_id)
            self._last_positions[track_id] = curr_pos

            # If no previous position or already cited, continue
            if prev_pos is None or track_id in self._violated_tracks:
                continue

            speed_km_h = float(det.get("speed_km_h", 0.0))
            class_name = str(det.get("class_name", "vehicle"))

            # Check each stopline
            for line in self.stoplines:
                approach_aspect = norm_aspects.get(line.approach, "GREEN")
                # A violation occurs if aspect is RED and vehicle crosses the stopline
                if approach_aspect != "RED":
                    continue

                if line_segments_intersect(prev_pos, curr_pos, line.pt1, line.pt2):
                    self._violated_tracks.add(track_id)

                    # Save photo evidence snapshot if frame is provided
                    image_path_str = ""
                    if frame is not None and frame.size > 0:
                        image_path_str = self._capture_evidence(
                            frame, bbox, track_id, class_name, speed_km_h, timestamp
                        )
                    else:
                        image_path_str = f"evidence_sim_{timestamp:.1f}_{track_id}.jpg"

                    v = ViolationEvent(
                        track_id=track_id,
                        vehicle_class=class_name,
                        speed_km_h=speed_km_h,
                        timestamp=timestamp,
                        approach=line.approach,
                        image_path=image_path_str,
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        aspect=approach_aspect,
                        line_id=line.line_id,
                    )

                    new_violations.append(v)
                    self._violations.append(v)

                    # Automatically persist infraction into SQLite database
                    try:
                        log_violation(
                            intersection_id=intersection_id,
                            track_id=track_id,
                            vehicle_class=class_name,
                            speed_km_h=speed_km_h,
                            timestamp=timestamp,
                            approach=line.approach,
                            image_path=image_path_str,
                        )
                    except Exception:
                        pass

                    break  # Avoid double-triggering across multiple lines for the same detection

        return new_violations

    def _capture_evidence(
        self,
        frame: np.ndarray,
        bbox: BBox,
        track_id: int,
        vehicle_class: str,
        speed_km_h: float,
        timestamp: float,
    ) -> str:
        """Stamp and save a high-resolution photographic violation evidence card."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        evidence = frame.copy()

        # Draw red violation bounding box on full context frame
        cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Draw evidence banner
        banner_text = f"RED LIGHT VIOLATION | ID:{track_id} {vehicle_class.upper()} | {speed_km_h:.1f} km/h | T:{timestamp:.1f}s"
        cv2.rectangle(evidence, (x1, max(0, y1 - 28)), (x1 + 420, y1), (0, 0, 200), -1)
        cv2.putText(
            evidence,
            banner_text,
            (x1 + 6, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        filename = f"viol_t{timestamp:.1f}_id{track_id}_{vehicle_class}.jpg"
        out_file = self.evidence_dir / filename
        cv2.imwrite(str(out_file), evidence)
        return str(out_file)

    @property
    def total_violations(self) -> int:
        return len(self._violations)

    def get_violations(self) -> list[ViolationEvent]:
        return list(self._violations)
