"""Model definitions and inference wrappers."""

from .detector import Detection, VehicleDetector, load_detector

__all__ = ["Detection", "VehicleDetector", "load_detector"]
