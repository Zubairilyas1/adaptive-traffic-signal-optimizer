"""Verify a downloaded YOLOv8 dataset.

Counts images and instances per class for every split, flags empty-label
images, and reports the class distribution. This is the Phase 1 dataset gate.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\verify_dataset.py
    .\\.venv\\Scripts\\python.exe scripts\\verify_dataset.py --root data\\raw\\roboflow
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_data_yaml(root: Path) -> Path | None:
    candidates = list(root.rglob("data.yaml")) + list(root.rglob("*.yaml"))
    return candidates[0] if candidates else None


def parse_names(yaml_path: Path) -> dict[int, str]:
    """Parse the 'names' field of a Roboflow data.yaml.

    Handles both inline lists (``names: ['bus', 'car']``) and block lists
    (``names:`` followed by ``- bus`` lines).
    """
    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^names:\s*(.*)$", text, flags=re.MULTILINE)
    if not match:
        return {}

    inline = match.group(1).strip()
    if inline.startswith("[") and inline.endswith("]"):
        items = inline[1:-1].split(",")
        return {i: item.strip().strip("'\"") for i, item in enumerate(items) if item.strip()}

    names: dict[int, str] = {}
    for line in text[match.end():].splitlines():
        stripped = line.strip()
        if stripped.startswith("-"):
            names[len(names)] = stripped[1:].strip().strip("'\"")
        elif stripped and not stripped.startswith(("#", " ")):
            break
    return names


def count_split(split_dir: Path, names: dict[int, str]) -> dict:
    images = [p for p in (split_dir / "images").rglob("*")
              if p.suffix.lower() in IMG_EXTS]
    if not images:  # fallback: images directly under split_dir
        images = [p for p in split_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

    label_dir = split_dir / "labels"
    if not label_dir.exists():
        label_dir = split_dir

    class_counts: Counter[int] = Counter()
    empty_labels = 0
    missing_labels = 0

    for img in images:
        label = label_dir / f"{img.stem}.txt"
        if not label.exists():
            missing_labels += 1
            continue
        text = label.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            empty_labels += 1
            continue
        for line in text.splitlines():
            parts = line.split()
            if parts:
                class_counts[int(float(parts[0]))] += 1

    return {
        "images": len(images),
        "class_counts": class_counts,
        "empty_labels": empty_labels,
        "missing_labels": missing_labels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a YOLOv8 dataset.")
    parser.add_argument("--root", type=Path, default=config.RAW_DIR / "roboflow")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        print(f"ERROR: dataset root not found: {root}")
        print("Run scripts\\download_dataset.py first.")
        return 1

    yaml_path = find_data_yaml(root)
    names = parse_names(yaml_path) if yaml_path else {}
    print(f"root      : {root}")
    print(f"data.yaml : {yaml_path}")
    print(f"classes   : {names or '(not parsed)'}\n")

    grand_total = 0
    overall: Counter[int] = Counter()

    for split in ("train", "valid", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        stats = count_split(split_dir, names)
        grand_total += stats["images"]
        overall.update(stats["class_counts"])
        print(f"[{split}] images={stats['images']}  "
              f"empty_labels={stats['empty_labels']}  "
              f"missing_labels={stats['missing_labels']}")
        for cid in sorted(stats["class_counts"]):
            label = names.get(cid, f"class_{cid}")
            print(f"        {cid:>2} {label:<15} {stats['class_counts'][cid]}")
        print()

    if grand_total == 0:
        print("WARNING: no images found. Check the download path/format.")
        return 1

    print("=" * 50)
    print(f"TOTAL images: {grand_total}")
    print("Instances per class (all splits):")
    for cid in sorted(overall):
        label = names.get(cid, f"class_{cid}")
        print(f"  {cid:>2} {label:<15} {overall[cid]}")

    thin = [names.get(c, f"class_{c}") for c in overall if overall[c] < 100]
    if thin:
        print(f"\nNOTE: thin classes (<100 instances): {thin}")
        print("      Mention as a data limitation; consider oversampling.")
    print("\nDATASET GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
