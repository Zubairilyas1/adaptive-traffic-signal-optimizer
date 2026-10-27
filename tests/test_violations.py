"""Unit tests for Red-Light Violation Detector and Stopline crossing math."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.analysis.roi import LaneROI
from src.analysis.violations import (
    RedLightViolationDetector,
    Stopline,
    line_segments_intersect,
    stoplines_from_rois,
)
from src.db.database import init_db, query_violations
from src.decision.actuator import SignalAspect


def test_line_segments_intersect():
    # Crossing lines
    assert line_segments_intersect((0, 0), (10, 10), (0, 10), (10, 0)) is True

    # Parallel non-intersecting lines
    assert line_segments_intersect((0, 0), (10, 0), (0, 5), (10, 5)) is False

    # Collinear or separated lines
    assert line_segments_intersect((0, 0), (2, 2), (5, 5), (10, 10)) is False


def test_stoplines_from_rois():
    rois = [
        LaneROI(
            lane_id="North_L1",
            approach="North",
            polygon=[(100.0, 50.0), (200.0, 50.0), (200.0, 300.0), (100.0, 300.0)],
        )
    ]
    stoplines = stoplines_from_rois(rois)
    assert len(stoplines) == 1
    s = stoplines[0]
    assert s.lane_id == "North_L1"
    assert s.approach == "North"
    # Should select the downstream edge with highest y
    assert s.pt1[1] == 300.0
    assert s.pt2[1] == 300.0


def test_violation_not_flagged_on_green(tmp_path: Path):
    stopline = Stopline("stop_N", "North", (100.0, 200.0), (300.0, 200.0))
    detector = RedLightViolationDetector([stopline], evidence_dir=tmp_path)

    # Frame 1: Vehicle before line
    dets_f1 = [{"track_id": 1, "bbox": [150.0, 100.0, 250.0, 180.0], "class_name": "car"}]
    detector.update(dets_f1, {"North": SignalAspect.GREEN}, timestamp=1.0)

    # Frame 2: Vehicle crosses line, but aspect is GREEN
    dets_f2 = [{"track_id": 1, "bbox": [150.0, 150.0, 250.0, 220.0], "class_name": "car"}]
    viols = detector.update(dets_f2, {"North": SignalAspect.GREEN}, timestamp=2.0)

    assert len(viols) == 0
    assert detector.total_violations == 0


def test_violation_flagged_on_red_with_evidence(tmp_path: Path):
    db_file = tmp_path / "test_viols.db"
    init_db(db_file)

    stopline = Stopline("stop_N", "North", (100.0, 200.0), (300.0, 200.0))
    detector = RedLightViolationDetector([stopline], evidence_dir=tmp_path)

    dummy_frame = np.full((400, 400, 3), 120, dtype=np.uint8)

    # Frame 1: Vehicle before line
    dets_f1 = [{"track_id": 5, "bbox": [150.0, 100.0, 250.0, 180.0], "class_name": "motorcycle", "speed_km_h": 42.0}]
    detector.update(dets_f1, {"North": SignalAspect.RED}, timestamp=10.0, frame=dummy_frame)

    # Frame 2: Vehicle crosses stopline on RED aspect
    dets_f2 = [{"track_id": 5, "bbox": [150.0, 150.0, 250.0, 220.0], "class_name": "motorcycle", "speed_km_h": 45.5}]
    viols = detector.update(
        dets_f2,
        {"North": SignalAspect.RED},
        timestamp=11.0,
        frame=dummy_frame,
        intersection_id="inter_test",
    )

    assert len(viols) == 1
    v = viols[0]
    assert v.track_id == 5
    assert v.vehicle_class == "motorcycle"
    assert v.speed_km_h == 45.5
    assert v.approach == "North"
    assert v.aspect == "RED"
    assert Path(v.image_path).exists()  # Evidence photo was stamped and saved to disk

    # Frame 3: Vehicle continues moving, should NOT be re-flagged
    dets_f3 = [{"track_id": 5, "bbox": [150.0, 220.0, 250.0, 290.0], "class_name": "motorcycle", "speed_km_h": 46.0}]
    viols_f3 = detector.update(dets_f3, {"North": SignalAspect.RED}, timestamp=12.0)
    assert len(viols_f3) == 0
    assert detector.total_violations == 1
