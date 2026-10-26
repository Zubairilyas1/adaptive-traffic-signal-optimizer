"""Minimal readers/writers for YOLO-format datasets and Roboflow data.yaml files."""

from __future__ import annotations

import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"valid": "val", "validation": "val", "train": "train", "test": "test", "val": "val"}


def parse_names(yaml_path: Path) -> dict[int, str]:
    """Parse the 'names' field of a YOLO data.yaml (inline or block list)."""
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


def image_files(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def read_labels(label_path: Path) -> list[list[float]]:
    """Return YOLO label rows as lists of floats: [cls, x, y, w, h]."""
    rows: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append([float(v) for v in parts[:5]])
    return rows


def write_labels(label_path: Path, rows: list[list[float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}" for r in rows
    )
    label_path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def normalize_class(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())
