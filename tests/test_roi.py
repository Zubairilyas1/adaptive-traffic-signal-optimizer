"""Tests for lane ROIs and per-track zone tracking."""

from src.analysis.roi import (
    LaneROI,
    ZoneEventType,
    ZoneTracker,
    anchor_point,
    buffer_polygon,
    point_in_polygon,
)

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_point_in_polygon_inside_and_outside():
    assert point_in_polygon((5.0, 5.0), SQUARE)
    assert not point_in_polygon((15.0, 5.0), SQUARE)
    assert not point_in_polygon((-1.0, 5.0), SQUARE)


def test_top_level_bottom_center_anchor():
    assert anchor_point((0.0, 0.0, 10.0, 20.0)) == (5.0, 20.0)


def test_buffer_shrinks_toward_centroid():
    shrunk = buffer_polygon(SQUARE, 1.0)
    assert shrunk[0][0] > 0.0 and shrunk[0][1] > 0.0
    assert shrunk[2][0] < 10.0 and shrunk[2][1] < 10.0


def _roi():
    return LaneROI(lane_id="A1", approach="A", polygon=SQUARE, buffer_px=0, hysteresis_frames=3)


def test_zone_tracker_enter_after_hysteresis():
    tracker = ZoneTracker([_roi()])
    # inside box (anchor bottom-centre lands in the square)
    box_in = (4.0, 4.0, 6.0, 6.0)
    events = []
    for _ in range(3):
        events += tracker.update(1, box_in, 0.0)
    enters = [e for e in events if e.event_type is ZoneEventType.ENTER]
    assert len(enters) == 1
    assert enters[0].lane_id == "A1"


def test_zone_tracker_exit_on_leaving():
    tracker = ZoneTracker([_roi()])
    box_in = (4.0, 4.0, 6.0, 6.0)
    box_out = (40.0, 4.0, 46.0, 6.0)
    for _ in range(3):
        tracker.update(1, box_in, 0.0)
    events = []
    for _ in range(3):
        events += tracker.update(1, box_out, 1.0)
    assert any(e.event_type is ZoneEventType.EXIT for e in events)
