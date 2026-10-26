"""Count accuracy: predicted vs ground-truth box counts on the test split.

Counts are what the analysis layer actually consumes, so detection mAP alone
does not prove the counting works. This measures the counting directly.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\count_accuracy.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.models import load_detector  # noqa: E402


def gt_count(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predicted vs ground-truth counts.")
    p.add_argument("--weights", type=Path, default=config.BEST_WEIGHTS)
    p.add_argument("--split", default="test")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="cpu",
                   help="'cpu' (default, thermal-safe) or a GPU index like 0.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    images_dir = config.MERGED_DIR / args.split / "images"
    labels_dir = config.MERGED_DIR / args.split / "labels"
    images = sorted(p for p in images_dir.glob("*.jpg"))
    if not images:
        print(f"ERROR: no images under {images_dir}")
        return 1

    detector = load_detector(weights=args.weights, conf=args.conf, device=args.device)
    print(f"weights : {detector.weights}")
    print(f"device  : {args.device}")
    print(f"images  : {len(images)}")

    pred_counts: list[int] = []
    gt_counts: list[int] = []

    results = detector.model.predict(
        source=[str(p) for p in images],
        device=args.device,
        conf=detector.conf,
        iou=detector.iou,
        imgsz=detector.imgsz,
        classes=detector.class_ids,
        max_det=detector.max_det,
        batch=1,
        stream=True,
        verbose=False,
    )
    for path, result in zip(images, results):
        pred_counts.append(int(len(result.boxes)))
        gt_counts.append(gt_count(labels_dir / f"{path.stem}.txt"))

    pred = np.array(pred_counts)
    gt = np.array(gt_counts)
    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    exact = float(np.mean(pred == gt) * 100)

    print("\n" + "=" * 50)
    print("COUNT ACCURACY (test split)")
    print("=" * 50)
    print(f"  images        : {len(images)}")
    print(f"  GT boxes      : {int(gt.sum())}")
    print(f"  predicted     : {int(pred.sum())}")
    print(f"  MAE           : {mae:.3f} vehicles/image")
    print(f"  RMSE          : {rmse:.3f}")
    print(f"  bias          : {bias:+.3f} (negative = under-counting)")
    print(f"  exact matches : {exact:.1f}%")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(gt, pred, s=10, alpha=0.4, color="#4C78A8")
    lim = max(int(gt.max()), int(pred.max()), 1)
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="perfect")
    ax.set_xlabel("ground-truth vehicles/image")
    ax.set_ylabel("predicted vehicles/image")
    ax.set_title(f"Count accuracy (MAE = {mae:.2f}, bias = {bias:+.2f})")
    ax.legend()
    fig.tight_layout()
    out_png = config.FIGURES_DIR / "count_accuracy.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    summary = {
        "images": len(images),
        "gt_boxes": int(gt.sum()),
        "pred_boxes": int(pred.sum()),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "exact_match_pct": exact,
    }
    out_json = config.FIGURES_DIR / "metrics_count.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nfigure -> {out_png}")
    print(f"metrics -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
