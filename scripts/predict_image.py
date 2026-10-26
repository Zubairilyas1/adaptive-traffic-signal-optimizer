"""CLI: run the vehicle detector on a single image.

Examples
--------
    .\\.venv\\Scripts\\python.exe scripts\\predict_image.py data\\samples\\bus.jpg
    .\\.venv\\Scripts\\python.exe scripts\\predict_image.py img.jpg --conf 0.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.models import load_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vehicle detection on an image.")
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument("--weights", type=Path, default=config.DETECTOR_WEIGHTS)
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=config.IMGSZ)
    parser.add_argument("--device", default=None, help="'cpu', 0, 1, ...")
    parser.add_argument("--out", type=Path, default=None, help="Output image path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}")
        return 1

    detector = load_detector(
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    print(f"weights : {detector.weights}")
    print(f"device  : {detector.device} (cuda={detector.is_cuda})")
    print(f"classes : {detector.class_ids}")

    detections = detector.predict(args.image)

    print(f"\n{len(detections)} detection(s):")
    totals: dict[str, int] = {}
    pcu_total = 0.0
    for det in detections:
        totals[det.class_name] = totals.get(det.class_name, 0) + 1
        pcu_total += det.pcu
        x1, y1, x2, y2 = (round(v, 1) for v in det.bbox)
        print(f"  {det.class_name:<12} {det.confidence:.2f}  box=({x1},{y1},{x2},{y2})")

    print(f"\nby class : {totals}")
    print(f"total PCU: {pcu_total:.1f}")

    annotated = detector.draw(args.image, detections)
    out_path = args.out or (config.FIGURES_DIR / f"{args.image.stem}_detected.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"\nannotated image -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
