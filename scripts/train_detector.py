"""Fine-tune YOLOv8n on the merged traffic dataset.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\train_detector.py --epochs 50
    # quick timing/smoke run:
    .\\.venv\\Scripts\\python.exe scripts\\train_detector.py --epochs 1 --fraction 0.1
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n.")
    parser.add_argument("--data", type=Path, default=config.MERGED_YAML)
    parser.add_argument("--weights", type=Path, default=config.DETECTOR_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of training set to use (for quick runs).")
    parser.add_argument("--name", default="traffic_yolov8n")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        print(f"ERROR: data yaml not found: {args.data}")
        return 1

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("TRAINING START")
    print(f"data     : {args.data}")
    print(f"weights  : {args.weights}")
    print(f"epochs   : {args.epochs}  imgsz={args.imgsz}  batch={args.batch}")
    print(f"device   : {device}  workers={args.workers}  fraction={args.fraction}")
    print("=" * 60, flush=True)

    model = YOLO(str(args.weights))
    t0 = time.time()
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=device,
        fraction=args.fraction,
        seed=42,
        pretrained=True,
        cos_lr=True,
        amp=True,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )
    elapsed = time.time() - t0

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    print(f"\nTraining finished in {elapsed / 60:.1f} min")
    print(f"save_dir : {save_dir}")
    print(f"best.pt  : {best}  (exists={best.exists()})")

    if best.exists():
        config.BEST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best, config.BEST_WEIGHTS)
        print(f"copied   -> {config.BEST_WEIGHTS}")
    print("TRAINING DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
