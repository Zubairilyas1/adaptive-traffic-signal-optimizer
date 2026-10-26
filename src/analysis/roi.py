"""Lane regions of interest and per-track zone tracking.

Lane assignment is a geometric operation (which polygon contains the vehicle's
ground-contact point), not a learned task - the learning is in the detector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Sequence

Point = tuple[float, float]
Polygon = list[Point]
BBox = Sequence[float]  # x1, y1, x2, y2


class ZoneEventType(Enum):
    ENTER = "enter"
    EXIT = "exit"


@dataclass
class ZoneEvent:
    track_id: int
    lane_id: str
    event_type: ZoneEventType
    timestamp: float


@dataclass
class LaneROI:
    lane_id: str
    approach: str
    polygon: Polygon
    buffer_px: float = 3.0
    hysteresis_frames: int = 3


def anchor_point(bbox: BBox) -> Point:
    """Bottom-centre of a box - the vehicle's contact point with the road."""
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return ((x1 + x2) / 2.0, y2)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Even-odd ray-casting test."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            else:
                x_intersect = p1x
            if p1x == p2x or x <= x_intersect:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def buffer_polygon(polygon: Polygon, buffer_px: float) -> Polygon:
    """Shrink a polygon inward toward its centroid (reduces edge jitter)."""
    if buffer_px <= 0:
        return list(polygon)
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    out: Polygon = []
    for x, y in polygon:
        dx, dy = x - cx, y - cy
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        out.append((x - dx / norm * buffer_px, y - dy / norm * buffer_px))
    return out


class ZoneTracker:
    """Per-track hysteresis state machine over a set of lane ROIs."""

    def __init__(self, rois: Iterable[LaneROI]):
        self.rois = list(rois)
        self._buffered = {
            r.lane_id: buffer_polygon(r.polygon, r.buffer_px) for r in self.rois
        }
        # per track: confirmed lane, candidate lane, consecutive frames
        self._state: dict[int, dict] = {}

    def _lane_at(self, bbox: BBox) -> Optional[str]:
        pt = anchor_point(bbox)
        for roi in self.rois:
            if point_in_polygon(pt, self._buffered[roi.lane_id]):
                return roi.lane_id
        return None

    def update(self, track_id: int, bbox: BBox, timestamp: float = 0.0) -> list[ZoneEvent]:
        raw = self._lane_at(bbox)
        st = self._state.setdefault(
            track_id, {"confirmed": None, "candidate": None, "count": 0, "hyst": 3}
        )

        if raw is not None:
            st["hyst"] = next(
                (r.hysteresis_frames for r in self.rois if r.lane_id == raw), 3
            )

        if raw == st["candidate"]:
            st["count"] += 1
        else:
            st["candidate"] = raw
            st["count"] = 1

        events: list[ZoneEvent] = []
        if st["count"] >= st["hyst"] and raw != st["confirmed"]:
            old = st["confirmed"]
            if old is not None:
                events.append(ZoneEvent(track_id, old, ZoneEventType.EXIT, timestamp))
            if raw is not None:
                events.append(ZoneEvent(track_id, raw, ZoneEventType.ENTER, timestamp))
            st["confirmed"] = raw

        return events

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)


def scale_rois_to_resolution(
    rois: list[LaneROI],
    target_width: int,
    target_height: int,
    source_width: int = 1280,
    source_height: int = 720,
) -> list[LaneROI]:
    """Proportionally scale ROI polygons to match a target video resolution."""
    if (
        target_width <= 0
        or target_height <= 0
        or (target_width == source_width and target_height == source_height)
    ):
        return rois

    sx = float(target_width) / float(source_width)
    sy = float(target_height) / float(source_height)

    scaled: list[LaneROI] = []
    for r in rois:
        new_poly = [(p[0] * sx, p[1] * sy) for p in r.polygon]
        scaled.append(
            LaneROI(
                lane_id=r.lane_id,
                approach=r.approach,
                polygon=new_poly,
                buffer_px=max(1.0, r.buffer_px * min(sx, sy)),
                hysteresis_frames=r.hysteresis_frames,
            )
        )
    return scaled


def load_rois(
    path: Path | str,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> list[LaneROI]:
    """Load lane ROIs from JSON and optionally auto-scale to target resolution:
    {"frame_width": 1280, "frame_height": 720, "lanes": [{"lane_id": "North_L1", "approach": "North", "polygon": [[x,y], ...]}]}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    src_w = int(data.get("frame_width", 1280))
    src_h = int(data.get("frame_height", 720))

    rois: list[LaneROI] = []
    for lane in data["lanes"]:
        rois.append(
            LaneROI(
                lane_id=lane["lane_id"],
                approach=lane["approach"],
                polygon=[(float(x), float(y)) for x, y in lane["polygon"]],
                buffer_px=float(lane.get("buffer_px", 3.0)),
                hysteresis_frames=int(lane.get("hysteresis_frames", 3)),
            )
        )

    if target_width and target_height:
        rois = scale_rois_to_resolution(
            rois, target_width, target_height, source_width=src_w, source_height=src_h
        )

    return rois
