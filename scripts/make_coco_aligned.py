"""Create a COCO-aligned eval set for a zero-shot baseline.

A COCO-pretrained YOLOv8 only knows car(2), motorcycle(3), bus(5), truck(7).
This script filters the merged dataset to those four classes and remaps label
ids to the COCO indices, so ``YOLO('yolov8n.pt').val()`` produces a meaningful
zero-shot mAP. Classes absent from COCO (autorickshaw, van, tractor) are
excluded by construction - the fine-tuned model is evaluated on all 7 classes.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\make_coco_aligned.py --clean
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data import dataset_io as dio  # noqa: E402

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# merged-class-name -> COCO index
MERGED_TO_COCO = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build COCO-aligned eval set.")
    parser.add_argument("--source", type=Path, default=config.MERGED_DIR)
    parser.add_argument("--output", type=Path, default=config.PROCESSED_DIR / "coco_eval")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = args.source
    out = args.output

    names = dio.parse_names(src / "data.yaml")
    name_to_id = {v: k for k, v in names.items()}
    keep_ids = {name_to_id[n]: c for n, c in MERGED_TO_COCO.items() if n in name_to_id}
    print(f"source     : {src}")
    print(f"merged cls : {names}")
    print(f"keeping    : " + ", ".join(f"{n}(->{c})" for n, c in MERGED_TO_COCO.items()))

    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    for split in ("train", "val", "test"):
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if not img_dir.exists():
            continue
        out_img = out / split / "images"
        out_lbl = out / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)

        kept = 0
        for img in dio.image_files(img_dir):
            label = lbl_dir / f"{img.stem}.txt"
            if not label.exists():
                continue
            rows = []
            for r in dio.read_labels(label):
                if int(r[0]) in keep_ids:
                    rows.append([keep_ids[int(r[0])], r[1], r[2], r[3], r[4]])
            if not rows:
                continue
            shutil.copyfile(img, out_img / img.name)
            dio.write_labels(out_lbl / f"{img.stem}.txt", rows)
            kept += 1
            for r in rows:
                counts[int(r[0])] += 1
        print(f"  {split:<5} kept {kept} images")

    yaml_names = ", ".join(f"'{n}'" for n in COCO_NAMES)
    (out / "data_coco.yaml").write_text(
        "train: val/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        f"nc: {len(COCO_NAMES)}\n"
        f"names: [{yaml_names}]\n",
        encoding="utf-8",
    )

    print("\nCOCO-aligned class counts (all splits):")
    for cid in sorted(counts):
        print(f"  {cid:>2} {COCO_NAMES[cid]:<12} {counts[cid]}")
    print(f"\ndata_coco.yaml -> {out / 'data_coco.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
