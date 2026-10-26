"""Vehicle detection wrapper built on Ultralytics YOLO.

Layer 1 of the pipeline: turns frames into per-object detections. Lane
assignment, PCU weighting and signal optimisation are deliberately NOT done
here - they consume this module's output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src import config

ImageInput = Union[str, Path, np.ndarray]


@dataclass
class Detection:
    """A single detected vehicle."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (original coords)
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None

    @property
    def canonical_name(self) -> str:
        return config.canonical_class(self.class_name)

    @property
    def pcu(self) -> float:
        return config.pcu_for(self.class_name)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "canonical_name": self.canonical_name,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "pcu": self.pcu,
        }


class VehicleDetector:
    """Thin, explicit wrapper around an Ultralytics YOLO model."""

    def __init__(
        self,
        weights: Union[str, Path, None] = None,
        device: Optional[Union[int, str]] = None,
        conf: float = config.CONF_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        imgsz: int = config.IMGSZ,
        target_classes: Sequence[str] = config.TARGET_CLASS_NAMES,
        max_det: int = config.MAX_DET,
    ) -> None:
        self.weights = Path(weights) if weights else config.DETECTOR_WEIGHTS
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Detector weights not found: {self.weights}. "
                "See README for how to obtain yolov8n.pt."
            )

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = self._resolve_device(device)

        self.model = YOLO(str(self.weights))
        self.class_names: dict[int, str] = dict(self.model.names)
        self.target_classes = {config.canonical_class(c) for c in target_classes}
        self.class_ids = self._resolve_class_ids()

    def _resolve_device(self, device: Optional[Union[int, str]]) -> Union[int, str]:
        if device is not None:
            return device
        return config.DEVICE if torch.cuda.is_available() else "cpu"

    def _resolve_class_ids(self) -> Optional[List[int]]:
        """Map target class names to the model's class indices.

        Returns None when nothing matches, so callers do not accidentally
        filter out every detection (e.g. a fine-tuned model with new names).
        """
        ids = [
            idx
            for idx, name in self.class_names.items()
            if config.canonical_class(name) in self.target_classes
        ]
        return ids or None

    @property
    def is_cuda(self) -> bool:
        return self.device != "cpu" and torch.cuda.is_available()

    def predict(self, image: ImageInput) -> List[Detection]:
        """Run detection on a single image; returns filtered detections."""
        result = self.infer(image)
        return [self._to_detection(box) for box in result.boxes]

    def infer(self, image: ImageInput):
        """Run detection and return the raw Ultralytics result object."""
        results = self.model.predict(
            source=image,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.class_ids,
            max_det=self.max_det,
            verbose=False,
        )
        return results[0]

    def iter_video(self, source: ImageInput, stride: int = config.FRAME_STRIDE):
        """Yield (frame_index, result) for a video, sampling every ``stride``."""
        results = self.model.predict(
            source=source,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.class_ids,
            max_det=self.max_det,
            stream=True,
            verbose=False,
            vid_stride=stride,
        )
        for i, result in enumerate(results):
            yield i * stride, result

    def track_video(
        self,
        source: ImageInput,
        stride: int = config.FRAME_STRIDE,
        tracker: str = "bytetrack.yaml",
        persist: bool = True,
    ):
        """Yield (frame_index, result) with tracking ids (ByteTrack)."""
        results = self.model.track(
            source=source,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.class_ids,
            max_det=self.max_det,
            stream=True,
            verbose=False,
            tracker=tracker,
            persist=persist,
            vid_stride=stride,
        )
        for i, result in enumerate(results):
            yield i * stride, result

    def result_to_dicts(self, result) -> list[dict]:
        """Convert a tracked result to plain dicts with track_id when present."""
        boxes = result.boxes
        ids = boxes.id
        out: list[dict] = []
        for j, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            coords = box.xyxy[0].tolist()
            out.append({
                "track_id": int(ids[j]) if ids is not None else None,
                "bbox": (coords[0], coords[1], coords[2], coords[3]),
                "class_id": cls_id,
                "class_name": self.class_names.get(cls_id, f"class_{cls_id}"),
                "confidence": float(box.conf[0]),
            })
        return out

    def _to_detection(self, box) -> Detection:
        cls_id = int(box.cls[0])
        coords = box.xyxy[0].tolist()
        return Detection(
            bbox=(coords[0], coords[1], coords[2], coords[3]),
            class_id=cls_id,
            class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
            confidence=float(box.conf[0]),
        )

    @staticmethod
    def draw(image: ImageInput, detections: Sequence[Detection]) -> np.ndarray:
        """Return a copy of the image with bounding boxes drawn."""
        if isinstance(image, (str, Path)):
            vis = cv2.imread(str(image))
            if vis is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        else:
            vis = image.copy()

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            color = (0, 200, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f} ({det.pcu:.1f}u)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                vis,
                label,
                (x1 + 2, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return vis


def load_detector(
    weights: Union[str, Path, None] = None,
    device: Optional[Union[int, str]] = None,
    **kwargs,
) -> VehicleDetector:
    """Convenience factory for a configured :class:`VehicleDetector`."""
    return VehicleDetector(weights=weights, device=device, **kwargs)
