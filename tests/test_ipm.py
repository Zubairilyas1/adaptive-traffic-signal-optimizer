"""Unit tests for Inverse Perspective Mapping (IPM) and physical meter queues."""

from __future__ import annotations

import pytest

from src.analysis.ipm import IPMCalibration, IPMTransformer


def test_ipm_default_transformer():
    transformer = IPMTransformer(img_shape=(768, 432))
    assert transformer._homography is not None

    # Test mapping a point at the bottom center of the frame (near stopline)
    # Bottom center is approx (384, 410)
    x_m, y_m = transformer.pixel_to_ground(384.0, 410.0)
    # Should map to realistic ground coordinate (0 to 7m width, small Y distance)
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

    # Box further up in the frame is farther from the stopline in meters
    assert dist_far > dist_near

    total_queue_m = transformer.compute_queue_length_meters([box_near, box_far])
    assert total_queue_m == dist_far
    assert total_queue_m > 10.0  # Physical queue should be substantial distance
