"""Merge multiple Roboflow YOLO datasets into one unified 7-class dataset.

Sources (downloaded under data/raw/):
    base   -> roboflow/     (bus, car, motorbike, truck)
    amit   -> amit/         (Indian traffic objects)
    rnklm  -> rnklm/        (adds van, threewheel)

Target classes (unified index order):
    0 car, 1 motorcycle, 2 autorickshaw, 3 van, 4 tractor, 5 bus, 6 truck

Non-target classes (Fixed Obstacle, Modified, ...) are dropped. Images that
end up with zero target boxes are skipped.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\merge_datasets.py
    .\\.venv\\Scripts\\python.exe scripts\\merge_datasets.py --clean
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data import dataset_io as dio  # noqa: E402

TARGET_NAMES = ["car", "motorcycle", "autorickshaw", "van", "tractor", "bus", "truck"]
TARGET_INDEX = {name: i for i, name in enumerate(TARGET_NAMES)}

# normalized source class name -> unified target class name (None = drop)
CLASS_MAP: dict[str, str | None] = {
    "car": "car",
    "cars": "car",
    "motorbike": "motorcycle",
    "motorcycle": "motorcycle",
    "two wheeler": "motorcycle",
    "two-wheeler": "motorcycle",
    "three wheeler": "autorickshaw",
    "three-wheeler": "autorickshaw",
    "threewheel": "autorickshaw",
    "three-wheel": "autorickshaw",
    "rickshaw": "autorickshaw",
    "auto": "autorickshaw",
    "auto-rickshaw": "autorickshaw",
    "auto rickshaw": "autorickshaw",
    "autorickshaw": "autorickshaw",
    "vikram": "autorickshaw",
    "van": "van",
    "pickup-van": "van",
    "pickup van": "van",
    "microbus": "van",
    "minibus": "van",
    "tractor": "tractor",
    "bus": "bus",
    "truck": "truck",
    "lorry": "truck",
}

SOURCES = {
    "base": config.RAW_DIR / "roboflow",
    "amit": config.RAW_DIR / "amit",
    "rnklm": config.RAW_DIR / "rnklm",
}


def remap_class(name: str) -> int | None:
    target = CLASS_MAP.get(dio.normalize_class(name))
    return TARGET_INDEX[target] if target else None


def merge_source(
    name: str,
    src_root: Path,
    out_root: Path,
    stats: dict,
    class_counts: Counter,
) -> None:
    yaml_path = src_root / "data.yaml"
    names = dio.parse_names(yaml_path)
    print(f"\n[{name}] {src_root}")
    print(f"      classes: {names}")

    for split_dir in ("train", "valid", "test"):
        images_dir = src_root / split_dir / "images"
        labels_dir = src_root / split_dir / "labels"
        if not images_dir.exists():
            continue

        out_split = dio.SPLIT_ALIASES.get(split_dir, split_dir)
        out_images = out_root / out_split / "images"
        out_labels = out_root / out_split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)

        n_images = n_kept = n_boxes = 0
        for img in dio.image_files(images_dir):
            n_images += 1
            label = labels_dir / f"{img.stem}.txt"
            if not label.exists():
                continue

            new_rows: list[list[float]] = []
            for row in dio.read_labels(label):
                src_id = int(row[0])
                target = remap_class(names.get(src_id, ""))
                if target is None:
                    continue
                new_rows.append([target, row[1], row[2], row[3], row[4]])

            if not new_rows:
                continue

            new_stem = f"{name}_{img.stem}"
            shutil.copyfile(img, out_images / f"{new_stem}{img.suffix.lower()}")
            dio.write_labels(out_labels / f"{new_stem}.txt", new_rows)
            n_kept += 1
            n_boxes += len(new_rows)
            for r in new_rows:
                class_counts[int(r[0])] += 1

        stats[name][out_split] = {"images": n_kept, "boxes": n_boxes, "scanned": n_images}
        print(f"      {split_dir:>5} -> {out_split:<5} kept {n_kept}/{n_images} imgs, {n_boxes} boxes")


def write_data_yaml(out_root: Path) -> None:
    names = ", ".join(f"'{n}'" for n in TARGET_NAMES)
    content = (
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n\n"
        f"nc: {len(TARGET_NAMES)}\n"
        f"names: [{names}]\n\n"
        f"# Merged dataset - see data/README.md for sources and limitations\n"
        f"roboflow:\n"
        f"  merged_from: {list(SOURCES)}\n"
    )
    (out_root / "data.yaml").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Roboflow datasets.")
    parser.add_argument("--output", type=Path, default=config.PROCESSED_DIR / "traffic")
    parser.add_argument("--clean", action="store_true", help="Delete output first.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = args.output

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"output: {out_root}")
    stats: dict = defaultdict(dict)
    class_counts: Counter = Counter()

    for name, src_root in SOURCES.items():
        if not (src_root / "data.yaml").exists():
            print(f"\n[{name}] SKIPPED - not found: {src_root / 'data.yaml'}")
            continue
        merge_source(name, src_root, out_root, stats, class_counts)

    write_data_yaml(out_root)

    print("\n" + "=" * 60)
    print("MERGED CLASS DISTRIBUTION")
    print("=" * 60)
    total = sum(class_counts.values())
    for i, name in enumerate(TARGET_NAMES):
        count = class_counts.get(i, 0)
        pct = count / total * 100 if total else 0
        print(f"  {i} {name:<14} {count:>7}  ({pct:4.1f}%)")
    print(f"  {'TOTAL':<16} {total:>7}")

    grand_images = sum(s["images"] for per in stats.values() for s in per.values())
    print(f"\nTotal images written: {grand_images}")
    print(f"data.yaml written    : {out_root / 'data.yaml'}")
    print("\nNext: .\\.venv\\Scripts\\python.exe scripts\\verify_dataset.py "
          f"--root \"{out_root}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
