"""Rebuild a valid Ultralytics .pt from the downloaded Kaggle zip.

The download contains `best/data.pkl`, `best/data/...` (an unzipped PyTorch
checkpoint wrapped in a folder). torch needs those entries at the archive root,
in the original order and compression. This rewrites them preserving both.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ZIP = ROOT / "models" / "yolov8n_traffic_best.zip"
OUT = ROOT / "models" / "yolov8n_traffic_best.pt"
PREFIX = "best/"


def main() -> int:
    if not SRC_ZIP.exists():
        raise FileNotFoundError(f"Missing source zip: {SRC_ZIP}")

    if OUT.exists():
        OUT.unlink()

    count = 0
    with zipfile.ZipFile(SRC_ZIP, "r") as zin, \
            zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as zout:
        for info in zin.infolist():
            name = info.filename
            if name.endswith("/"):
                continue
            new_name = name[len(PREFIX):] if name.startswith(PREFIX) else name
            data = zin.read(name)
            zi = zipfile.ZipInfo(new_name, date_time=info.date_time)
            zi.compress_type = info.compress_type
            zi.external_attr = info.external_attr
            zout.writestr(zi, data)
            count += 1

    print(f"rewrote {count} entries -> {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)")
    with zipfile.ZipFile(OUT) as z:
        print("first 6 entries:", z.namelist()[:6])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
