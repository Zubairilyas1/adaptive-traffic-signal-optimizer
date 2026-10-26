"""Package the project into a single submission zip (one per group).

Walks the tree with directory pruning so it never descends into .venv / .build /
data (fast). Keeps source, tests, reports, slides, sim config and trained weights.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\package_submission.py
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "submission_traffic_signal.zip"

EXCLUDE_DIRS = {
    ".venv", ".build", ".git", ".pytest_cache", "__pycache__",
    "node_modules", "data", "runs", "logs",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_NAMES = {".env", "traffic_yolo.zip"}
EXCLUDE_PREFIXES = {"reports/eval_coco_baseline-2"}


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if path.suffix in EXCLUDE_SUFFIXES or path.name in EXCLUDE_NAMES:
        return False
    if any(rel.startswith(p) for p in EXCLUDE_PREFIXES):
        return False
    return True


def main() -> int:
    if OUT.exists():
        OUT.unlink()

    count = 0
    total = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            # prune excluded directories in place (prevents descending)
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for name in filenames:
                full = Path(dirpath) / name
                if not included(full):
                    continue
                zf.write(full, full.relative_to(ROOT).as_posix())
                count += 1
                total += full.stat().st_size

    print(f"files : {count}")
    print(f"raw   : {total / 1e6:.1f} MB")
    print(f"zip   : {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
