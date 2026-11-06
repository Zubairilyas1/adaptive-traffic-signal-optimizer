"""Unit tests for Inverse Perspective Mapping (IPM) and physical meter queues."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.analysis.ipm import IPMCalibration, IPMTransformer
from src.analysis.ipm_calibration import (
    ApproachIPMConfig,
    MultiApproachIPM,
)
from src.analysis.kinematics import KinematicLaneCounter
from src.analysis.roi import LaneROI


def test_ipm_default_transformer():
    transformer = IPMTransformer(img_shape=(768, 432))
    assert transformer._homography is not None

    # Test mapping a point at the bottom center of the frame (near stopline)
    x_m, y_m = transformer.pixel_to_ground(384.0, 410.0)
    assert 0.0 <= x_m <= 7.0
    assert -5.0 <= y_m <= 15.0

    # Invert back to pixel
    u, v = transformer.ground_to_pixel(x_m, y_m)
    assert pytest.approx(u, abs=1.0) == 384.0
    assert pytest.approx(v, abs=1.0) == 410.0


def test_ipm_queue_length_meters():
    transformer = IPMTransformer(img_shape=(768, 432))

    # Empty queue returns 0 meters
    assert transformer.compute_queue_length_meters([]) == 0.0

    # Two bounding boxes in queue: one near stopline (y=410), one further back (y=250)
    box_near = (350.0, 360.0, 420.0, 410.0)
    box_far = (360.0, 220.0, 400.0, 250.0)

    dist_near = transformer.distance_to_stopline_meters(box_near)
    dist_far = transformer.distance_to_stopline_meters(box_far)

    assert dist_far > dist_near

    total_queue_m = transformer.compute_queue_length_meters([box_near, box_far])
    assert total_queue_m == dist_far
    assert total_queue_m > 10.0


def test_ipm_warp_perspective_and_draw():
    transformer = IPMTransformer(img_shape=(640, 480))
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test BEV warp
    bev = transformer.warp_perspective(dummy_frame, out_size=(200, 400))
    assert bev.shape == (400, 200, 3)

    # Test polygon overlay
    drawn = transformer.draw_calibration_polygon(dummy_frame)
    assert drawn.shape == dummy_frame.shape


def test_approach_ipm_config_create_default():
    cfg = ApproachIPMConfig.create_default(
        approach="North", img_width=1920, img_height=1080, road_width_m=7.0, corridor_length_m=60.0
    )
    assert cfg.approach == "North"
    assert len(cfg.image_points) == 4
    assert len(cfg.ground_points) == 4
    assert cfg.corridor_length_m == 60.0

    calib = cfg.to_calibration()
    assert isinstance(calib, IPMCalibration)


def test_multi_approach_ipm_lifecycle():
    multi = MultiApproachIPM.default_for_approaches(
        intersection_id="node_test",
        approaches=["North", "South", "East", "West"],
        img_shape=(1280, 720),
    )

    assert len(multi.configs) == 4
    t_north = multi.get_transformer("North")
    assert isinstance(t_north, IPMTransformer)

    # Fallback to default for unconfigured approach
    t_unknown = multi.get_transformer("Unknown")
    assert isinstance(t_unknown, IPMTransformer)

    # Test BEV warp
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    bev = multi.warp_to_bev(dummy_img, approach="North", out_size=(150, 300))
    assert bev.shape == (300, 150, 3)

    # Test drawing overlay
    overlay = multi.draw_calibration_overlay(dummy_img, approach="North")
    assert overlay.shape == dummy_img.shape

    # Test queue calculation
    box_far = (600.0, 250.0, 680.0, 280.0)
    q_m = multi.compute_queue_length_meters("North", [box_far])
    assert q_m > 0.0


def test_multi_approach_ipm_save_and_load(tmp_path: Path):
    multi = MultiApproachIPM.default_for_approaches(
        intersection_id="save_test",
        approaches=["North", "South"],
        img_shape=(1280, 720),
    )

    save_file = tmp_path / "calib.json"
    multi.save(save_file)
    assert save_file.exists()

    loaded = MultiApproachIPM.load(save_file)
    assert loaded.intersection_id == "save_test"
    assert "North" in loaded.configs
    assert "South" in loaded.configs
    assert loaded.img_width == 1280


def test_kinematic_counter_with_multi_approach_ipm():
    rois = [
        LaneROI(lane_id="Lane1", approach="North", polygon=[(100, 100), (300, 100), (300, 600), (100, 600)]),
    ]
    multi = MultiApproachIPM.default_for_approaches(
        intersection_id="kin_test",
        approaches=["North"],
        img_shape=(1280, 720),
    )
    counter = KinematicLaneCounter(rois, ipm_transformer=multi)

    # Add detection inside lane
    det = {"track_id": 1, "bbox": [150, 300, 250, 450], "class_name": "car"}
    snap = counter.update([det], timestamp=0.0)

    assert "Lane1" in snap.lane_snapshots
    assert "North" in snap.approach_queue_m
    assert snap.approach_queue_m["North"] >= 0.0
