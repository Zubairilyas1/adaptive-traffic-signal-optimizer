"""Per-lane vehicle counting with track-id de-duplication."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Optional

from src import config
from src.analysis.roi import LaneROI, ZoneEventType, ZoneTracker


def _field(det: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dict or an object (e.g. Detection)."""
    if isinstance(det, dict):
        return det.get(name, default)
    return getattr(det, name, default)


class LaneCounter:
    """Counts each vehicle once per lane it enters, keyed by tracking id."""

    def __init__(self, rois: Iterable[LaneROI]):
        self.rois = list(rois)
        self.lane_approach = {r.lane_id: r.approach for r in self.rois}
        self.tracker = ZoneTracker(self.rois)
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._visited: set[tuple[int, str]] = set()
        self._class_of: dict[int, str] = {}

    def update(self, detections: Iterable[Any], timestamp: float = 0.0) -> None:
        for det in detections:
            track_id = _field(det, "track_id")
            bbox = _field(det, "bbox")
            class_name = _field(det, "class_name", "unknown")
            if track_id is None or bbox is None:
                continue  # untracked detections cannot be de-duplicated
            canonical = config.canonical_class(class_name)
            self._class_of[track_id] = canonical

            for event in self.tracker.update(int(track_id), bbox, timestamp):
                if event.event_type is not ZoneEventType.ENTER:
                    continue
                key = (event.track_id, event.lane_id)
                if key in self._visited:
                    continue
                self._visited.add(key)
                self.counts[event.lane_id][canonical] += 1

    def counts_by_lane(self) -> dict[str, dict[str, int]]:
        return {lane: dict(classes) for lane, classes in self.counts.items()}

    def counts_by_approach(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for lane, classes in self.counts.items():
            approach = self.lane_approach.get(lane, lane)
            for cls, n in classes.items():
                out[approach][cls] += n
        return {a: dict(c) for a, c in out.items()}

    def total_vehicles(self) -> int:
        return sum(sum(classes.values()) for classes in self.counts.values())

    def unique_tracks(self) -> int:
        return len({tid for tid, _ in self._visited})
