"""Tests for track-de-duplicated lane counting."""

from src.analysis.counting import LaneCounter
from src.analysis.roi import LaneROI

BOX_SQ = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def _counter():
    rois = [
        LaneROI("A1", "A", BOX_SQ, buffer_px=0, hysteresis_frames=3),
        LaneROI("B1", "B", [(20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)],
                buffer_px=0, hysteresis_frames=3),
    ]
    return LaneCounter(rois)


def test_same_track_counted_once():
    counter = _counter()
    det = {"track_id": 1, "bbox": (4.0, 4.0, 6.0, 6.0), "class_name": "car"}
    for _ in range(10):
        counter.update([det], timestamp=0.0)
    assert counter.counts_by_lane()["A1"]["car"] == 1
    assert counter.total_vehicles() == 1


def test_distinct_tracks_counted_separately():
    counter = _counter()
    for tid in (1, 2, 3):
        det = {"track_id": tid, "bbox": (4.0, 4.0, 6.0, 6.0), "class_name": "motorcycle"}
        for _ in range(3):
            counter.update([det], timestamp=0.0)
    assert counter.counts_by_lane()["A1"]["motorcycle"] == 3


def test_approach_aggregation():
    counter = _counter()
    a = {"track_id": 1, "bbox": (4.0, 4.0, 6.0, 6.0), "class_name": "car"}
    b = {"track_id": 2, "bbox": (24.0, 4.0, 26.0, 6.0), "class_name": "bus"}
    for _ in range(3):
        counter.update([a, b], timestamp=0.0)
    by_app = counter.counts_by_approach()
    assert by_app["A"]["car"] == 1
    assert by_app["B"]["bus"] == 1


def test_untracked_detections_ignored():
    counter = _counter()
    det = {"track_id": None, "bbox": (4.0, 4.0, 6.0, 6.0), "class_name": "car"}
    for _ in range(5):
        counter.update([det], timestamp=0.0)
    assert counter.total_vehicles() == 0
