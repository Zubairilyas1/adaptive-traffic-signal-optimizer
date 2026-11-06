"""Multi-Approach Inverse Perspective Mapping (IPM) & Calibration Store.

Manages 4-point metric homography mappings for multiple intersection approaches,
enabling true ground-plane queue measurement in meters and Bird's-Eye View (BEV) synthesis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src import config
from src.analysis.ipm import IPMCalibration, IPMTransformer, Point2D

logger = logging.getLogger(__name__)

CALIBRATIONS_DIR = config.DATA_DIR / "calibrations"


@dataclass
class ApproachIPMConfig:
    """Homography configuration for an individual approach."""

    approach: str
    image_points: list[Point2D]  # [Top-Left, Top-Right, Bottom-Right, Bottom-Left] in pixels
    ground_points: list[Point2D]  # [(0, L), (W, L), (W, 0), (0, 0)] in meters
    road_width_m: float = 7.0  # Nominal roadway width (e.g. 2 lanes ~ 7.0m)
    corridor_length_m: float = 50.0  # Metric depth of the approach corridor
    stopline_ground_y: float = 0.0  # Y position of stopline in meters

    def to_calibration(self) -> IPMCalibration:
        return IPMCalibration(
            image_points=self.image_points,
            ground_points=self.ground_points,
            stopline_ground_y=self.stopline_ground_y,
        )

    @classmethod
    def create_default(
        cls,
        approach: str,
        img_width: int = 1280,
        img_height: int = 720,
        road_width_m: float = 7.0,
        corridor_length_m: float = 50.0,
    ) -> ApproachIPMConfig:
        """Create a default geometric trapezoid scaled to frame resolution."""
        w, h = float(img_width), float(img_height)
        src = [
            (w * 0.35, h * 0.30),  # Top-left (far)
            (w * 0.65, h * 0.30),  # Top-right (far)
            (w * 0.88, h * 0.95),  # Bottom-right (near stopline)
            (w * 0.12, h * 0.95),  # Bottom-left (near stopline)
        ]
        dst = [
            (0.0, corridor_length_m),
            (road_width_m, corridor_length_m),
            (road_width_m, 0.0),
            (0.0, 0.0),
        ]
        return cls(
            approach=approach,
            image_points=src,
            ground_points=dst,
            road_width_m=road_width_m,
            corridor_length_m=corridor_length_m,
            stopline_ground_y=0.0,
        )


class MultiApproachIPM:
    """Orchestrates approach-specific IPM transformers for an intersection."""

    def __init__(
        self,
        intersection_id: str = "default",
        configs: Optional[dict[str, ApproachIPMConfig]] = None,
        img_shape: tuple[int, int] = (1280, 720),
    ) -> None:
        self.intersection_id = intersection_id
        self.img_width, self.img_height = img_shape
        self.configs: dict[str, ApproachIPMConfig] = configs or {}
        self._transformers: dict[str, IPMTransformer] = {}
        self._default_transformer = IPMTransformer(img_shape=img_shape)

        for app, cfg in self.configs.items():
            self._transformers[app] = IPMTransformer(
                calibration=cfg.to_calibration(),
                img_shape=img_shape,
            )

    def set_approach_config(self, cfg: ApproachIPMConfig) -> None:
        """Register or update calibration for an approach."""
        self.configs[cfg.approach] = cfg
        self._transformers[cfg.approach] = IPMTransformer(
            calibration=cfg.to_calibration(),
            img_shape=(self.img_width, self.img_height),
        )

    def get_transformer(self, approach: Optional[str] = None) -> IPMTransformer:
        """Retrieve approach-specific IPM transformer, or default."""
        if approach and approach in self._transformers:
            return self._transformers[approach]
        return self._default_transformer

    def compute_queue_length_meters(
        self,
        approach: str,
        halted_bboxes: Iterable[Sequence[float]],
    ) -> float:
        """Calculate spatial physical queue length in meters for an approach."""
        transformer = self.get_transformer(approach)
        return transformer.compute_queue_length_meters(halted_bboxes)

    def warp_to_bev(
        self,
        frame: np.ndarray,
        approach: Optional[str] = None,
        out_size: tuple[int, int] = (300, 600),
    ) -> np.ndarray:
        """Generate an orthographic Bird's-Eye View (BEV) top-down perspective of the approach corridor."""
        transformer = self.get_transformer(approach)
        out_w, out_h = out_size

        src = np.array(transformer.calibration.image_points, dtype=np.float32)
        dst = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ], dtype=np.float32)

        m_bev = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, m_bev, (out_w, out_h), flags=cv2.INTER_LINEAR)

    def draw_calibration_overlay(
        self,
        frame: np.ndarray,
        approach: Optional[str] = None,
        color: tuple[int, int, int] = (0, 240, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw the 4-point calibration corridor polygon and metric grid onto the image."""
        transformer = self.get_transformer(approach)
        overlay = frame.copy()
        pts = np.array(transformer.calibration.image_points, dtype=np.int32).reshape((-1, 1, 2))

        # Semi-transparent corridor highlight
        poly_layer = frame.copy()
        cv2.fillPoly(poly_layer, [pts], (color[0], color[1], color[2]))
        cv2.addWeighted(poly_layer, 0.15, overlay, 0.85, 0, overlay)

        # Draw boundary lines
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=thickness)

        # Draw corner handles
        labels = ["TL (Far)", "TR (Far)", "BR (Stopline)", "BL (Stopline)"]
        for pt, label in zip(transformer.calibration.image_points, labels):
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(overlay, (px, py), 5, (0, 255, 157), -1)
            cv2.circle(overlay, (px, py), 7, (0, 0, 0), 1)

        return overlay

    def to_dict(self) -> dict[str, Any]:
        return {
            "intersection_id": self.intersection_id,
            "img_width": self.img_width,
            "img_height": self.img_height,
            "approaches": {
                app: {
                    "approach": cfg.approach,
                    "image_points": cfg.image_points,
                    "ground_points": cfg.ground_points,
                    "road_width_m": cfg.road_width_m,
                    "corridor_length_m": cfg.corridor_length_m,
                    "stopline_ground_y": cfg.stopline_ground_y,
                }
                for app, cfg in self.configs.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiApproachIPM:
        iid = data.get("intersection_id", "default")
        w = int(data.get("img_width", 1280))
        h = int(data.get("img_height", 720))
        configs = {}
        for app, app_data in data.get("approaches", {}).items():
            configs[app] = ApproachIPMConfig(
                approach=app,
                image_points=[(float(p[0]), float(p[1])) for p in app_data["image_points"]],
                ground_points=[(float(p[0]), float(p[1])) for p in app_data["ground_points"]],
                road_width_m=float(app_data.get("road_width_m", 7.0)),
                corridor_length_m=float(app_data.get("corridor_length_m", 50.0)),
                stopline_ground_y=float(app_data.get("stopline_ground_y", 0.0)),
            )
        return cls(intersection_id=iid, configs=configs, img_shape=(w, h))

    def save(self, path: Optional[Path | str] = None) -> Path:
        """Persist calibration to JSON file."""
        if path is None:
            CALIBRATIONS_DIR.mkdir(parents=True, exist_ok=True)
            path = CALIBRATIONS_DIR / f"{self.intersection_id}_ipm.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"Saved IPM calibration to {path}")
        return path

    @classmethod
    def load(cls, path: Path | str) -> MultiApproachIPM:
        """Load calibration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"IPM calibration file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def default_for_approaches(
        cls,
        intersection_id: str = "default",
        approaches: Sequence[str] = ("North", "South", "East", "West"),
        img_shape: tuple[int, int] = (1280, 720),
    ) -> MultiApproachIPM:
        """Create a MultiApproachIPM with default geometric calibrations for specified approaches."""
        configs = {
            app: ApproachIPMConfig.create_default(
                approach=app,
                img_width=img_shape[0],
                img_height=img_shape[1],
            )
            for app in approaches
        }
        return cls(intersection_id=intersection_id, configs=configs, img_shape=img_shape)
