"""Evaluate a YOLO detector on a dataset and report mAP.

Works for the zero-shot COCO baseline (on the COCO-aligned set) and for the
fine-tuned 7-class model (on the merged set).

Usage
-----
    # zero-shot COCO baseline
    .\\.venv\\Scripts\\python.exe scripts\\evaluate_detector.py \
        --weights models\\yolov8n.pt --data data\\processed\\coco_eval\\data_coco.yaml \
        --split test --focus 2 3 5 7 --tag coco_baseline

    # fine-tuned model
    .\\.venv\\Scripts\\python.exe scripts\\evaluate_detector.py \
        --weights models\\yolov8n_traffic_best.pt \
        --data data\\processed\\traffic\\data.yaml --split test --tag finetuned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO detector.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=config.IMGSZ)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument(
        "--focus", type=int, nargs="*", default=None,
        help="Class ids to average mAP over (e.g. 2 3 5 7 for COCO vehicles).",
    )
    parser.add_argument("--tag", default="eval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}")
        return 1
    if not args.data.exists():
        print(f"ERROR: data yaml not found: {args.data}")
        return 1

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"weights : {args.weights}")
    print(f"data    : {args.data}")
    print(f"split   : {args.split}  device={device}  imgsz={args.imgsz}")

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        device=device,
        plots=True,
        project=str(config.REPORTS_DIR),
        name=f"eval_{args.tag}",
        verbose=False,
    )

    box = metrics.box
    present = list(box.ap_class_index)
    names = metrics.names

    per_class = []
    for i, cid in enumerate(present):
        per_class.append({
            "class_id": int(cid),
            "name": names[int(cid)],
            "ap50": float(box.ap50[i]),
            "ap50_95": float(box.ap[i]),
        })

    print("\n" + "=" * 60)
    print(f"RESULTS  ({args.tag})")
    print("=" * 60)
    print(f"mAP@0.5      : {box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {box.map:.4f}")
    print("\nPer-class:")
    for row in per_class:
        print(f"  {row['class_id']:>2} {row['name']:<14} "
              f"AP@0.5={row['ap50']:.4f}  AP@0.5:0.95={row['ap50_95']:.4f}")

    summary = {
        "tag": args.tag,
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
    }

    if args.focus:
        focus_set = set(args.focus)
        focused = [r for r in per_class if r["class_id"] in focus_set]
        if focused:
            f50 = sum(r["ap50"] for r in focused) / len(focused)
            f95 = sum(r["ap50_95"] for r in focused) / len(focused)
            print(f"\nFocused mAP over classes {sorted(focus_set)}:")
            print(f"  mAP@0.5      : {f50:.4f}")
            print(f"  mAP@0.5:0.95 : {f95:.4f}")
            summary["focus_classes"] = sorted(focus_set)
            summary["focus_map50"] = f50
            summary["focus_map50_95"] = f95

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_json = config.FIGURES_DIR / f"metrics_{args.tag}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nmetrics -> {out_json}")
    print(f"plots   -> {config.REPORTS_DIR / f'eval_{args.tag}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
