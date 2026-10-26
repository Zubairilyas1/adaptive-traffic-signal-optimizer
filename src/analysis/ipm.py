"""Inverse Perspective Mapping (IPM) & Real-World Metric Calibration.

Transforms camera perspective pixel coordinates (u, v) into planar ground-plane
world coordinates (X, Y) in meters via 4-point homography.

This solves Flaw #6: perspective distortion where distant queues appear
microscopic in pixels, and enables calculating physical queue backlogs in meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
import cv2
import numpy as np

Point2D = Tuple[float, float]


@dataclass
class IPMCalibration:
    """4-point calibration mapping camera image pixels to ground meters.

    Points order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    """
    image_points: list[Point2D]
    ground_points: list[Point2D]  # (X_meters, Y_meters)
    stopline_ground_y: float = 0.0  # Y coordinate of stopline in meters


class IPMTransformer:
    """Calculates planar homography matrix and converts between pixels and meters."""

    def __init__(self, calibration: Optional[IPMCalibration] = None, img_shape: Tuple[int, int] = (768, 432)):
        self.img_width, self.img_height = img_shape
        if calibration is None:
            # Default camera perspective trapezoid for standard intersection viewing angle
            self.calibration = self._default_calibration(self.img_width, self.img_height)
        else:
            self.calibration = calibration

        self._homography: Optional[np.ndarray] = None
        self._inv_homography: Optional[np.ndarray] = None
        self._compute_matrices()

    def _default_calibration(self, w: int, h: int) -> IPMCalibration:
        """Heuristic camera calibration representing a 30m x 7m roadway corridor."""
        src = [
            (float(w * 0.35), float(h * 0.25)),  # Top-left (far)
            (float(w * 0.65), float(h * 0.25)),  # Top-right (far)
            (float(w * 0.85), float(h * 0.95)),  # Bottom-right (near stopline)
            (float(w * 0.15), float(h * 0.95)),  # Bottom-left (near stopline)
        ]
        # Real-world metric frame: (X width in meters: 0 to 7.0m, Y distance in meters: 0 to 50.0m)
        dst = [
            (0.0, 50.0),    # Far left (50m away)
            (7.0, 50.0),    # Far right (50m away)
            (7.0, 0.0),     # Near right (at stopline, Y=0m)
            (0.0, 0.0),     # Near left (at stopline, Y=0m)
        ]
        return IPMCalibration(image_points=src, ground_points=dst, stopline_ground_y=0.0)

    def _compute_matrices(self) -> None:
        src = np.array(self.calibration.image_points, dtype=np.float32)
        dst = np.array(self.calibration.ground_points, dtype=np.float32)
        if len(src) != 4 or len(dst) != 4:
            raise ValueError("IPM calibration requires exactly 4 correspondence points.")
        self._homography = cv2.getPerspectiveTransform(src, dst)
        self._inv_homography = np.linalg.inv(self._homography)

    def pixel_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """Convert pixel coordinate (u, v) to ground coordinate (X_meters, Y_meters)."""
        pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self._homography)
        x_m, y_m = warped[0][0]
        return float(x_m), float(y_m)

    def ground_to_pixel(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """Convert ground metric coordinate (X_meters, Y_meters) to image pixel (u, v)."""
        pt = np.array([[[float(x_m), float(y_m)]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self._inv_homography)
        u, v = warped[0][0]
        return float(u), float(v)

    def distance_to_stopline_meters(self, bbox: Sequence[float]) -> float:
        """Calculate the distance in physical meters from the vehicle contact point to stopline."""
        # Contact point is bottom-center of bounding box
        u = (bbox[0] + bbox[2]) / 2.0
        v = bbox[3]
        _, y_m = self.pixel_to_ground(u, v)
        dist = abs(y_m - self.calibration.stopline_ground_y)
        return float(max(0.0, dist))

    def compute_queue_length_meters(self, halted_bboxes: Iterable[Sequence[float]]) -> float:
        """Calculate spatial physical queue length in meters back from the stopline.

        Measures the span to the furthest stopped vehicle in the queue.
        """
        boxes = list(halted_bboxes)
        if not boxes:
            return 0.0
        distances = [self.distance_to_stopline_meters(box) for box in boxes]
        return float(max(distances))
