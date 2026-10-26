"""Domain-gap test on external (Pakistani) traffic footage.

The external dataset's labels are not usable (garbled class names), so this is
a QUALITATIVE generalisation check plus detection statistics - not a mAP. It
shows whether the model, trained on non-local data, still fires on local roads.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\domain_gap.py --limit 12
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

EXT_DIR = config.RAW_DIR / "pakistan" / "test" / "images"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qualitative domain-gap test.")
    p.add_argument("--weights", type=Path, default=config.BEST_WEIGHTS)
    p.add_argument("--limit", type=int, default=12, help="Images to include in the sheet.")
    p.add_argument("--sample", type=int, default=60, help="Images to run for statistics.")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not EXT_DIR.exists():
        print(f"ERROR: external images not found: {EXT_DIR}")
        return 1

    images = sorted(EXT_DIR.glob("*.jpg"))
    if not images:
        return 1
    rng = np.random.default_rng(42)
    sample = list(rng.choice(images, size=min(args.sample, len(images)), replace=False))

    detector = load_detector(weights=args.weights, device=args.device)
    print(f"weights : {detector.weights}")
    print(f"device  : {args.device}")
    print(f"images  : {len(sample)} sampled from {EXT_DIR}")

    class_counter: Counter = Counter()
    per_image: list[int] = []
    sheet: list[np.ndarray] = []

    results = detector.model.predict(
        source=[str(p) for p in sample],
        device=args.device, conf=detector.conf, iou=detector.iou, imgsz=detector.imgsz,
        classes=detector.class_ids, max_det=detector.max_det, batch=1, stream=True, verbose=False,
    )
    for i, result in enumerate(results):
        n = int(len(result.boxes))
        per_image.append(n)
        for cid in result.boxes.cls.tolist():
            class_counter[detector.class_names.get(int(cid), str(int(cid)))] += 1
        if i < args.limit:
            sheet.append(result.plot())

    mean_det = float(np.mean(per_image)) if per_image else 0.0
    print("\n" + "=" * 50)
    print("DOMAIN GAP (Pakistani footage, qualitative)")
    print("=" * 50)
    print(f"  images sampled     : {len(per_image)}")
    print(f"  mean detections/img: {mean_det:.2f}")
    print(f"  class distribution : {dict(class_counter)}")

    n = min(len(sheet), args.limit)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img in zip(axes, sheet):
        ax.imshow(img[:, :, ::-1])
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Fine-tuned detector on external Pakistani traffic footage", fontsize=13)
    fig.tight_layout()
    out_png = config.FIGURES_DIR / "domain_gap.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    summary = {
        "clip": "external: subek-yolo/pakistan traffic (test split)",
        "images_sampled": len(per_image),
        "mean_detections_per_image": mean_det,
        "class_distribution": dict(class_counter),
        "note": ("Qualitative only: external labels have garbled class names, so no mAP. "
                 "Confirms the model still detects vehicles on local footage; magnitude of "
                 "the domain gap must be quantified with hand-labelled local data."),
    }
    out_json = config.FIGURES_DIR / "metrics_domain_gap.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  figure  -> {out_png}")
    print(f"  summary -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
