"""Download a Roboflow Universe dataset in YOLOv8 format.

Uses the Roboflow export REST API directly (the high-level client proved
unreliable on this machine, silently writing no files).

Flow:
    1. GET /{workspace}/{project}/{version}/{format} -> signed export link
    2. download the zip
    3. extract into the destination folder

Credentials come from .env (see .env.example).

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\download_dataset.py \
        --workspace thesis-swpxf --project traffic-surveillance-system --version 8

If --workspace/--project/--version are omitted, the values in .env are used.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src import config  # noqa: E402

load_dotenv(ROOT / ".env")

API_ROOT = "https://api.roboflow.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Roboflow YOLOv8 dataset.")
    parser.add_argument("--workspace", default=os.getenv("ROBOFLOW_WORKSPACE"))
    parser.add_argument("--project", default=os.getenv("ROBOFLOW_PROJECT"))
    parser.add_argument("--version", default=os.getenv("ROBOFLOW_VERSION"))
    parser.add_argument("--format", default=os.getenv("ROBOFLOW_FORMAT", "yolov8"))
    parser.add_argument("--location", type=Path, default=config.DATASET_DIR)
    return parser.parse_args()


def get_export_link(workspace: str, project: str, version: str, fmt: str, api_key: str) -> str:
    url = f"{API_ROOT}/{workspace}/{project}/{version}/{fmt}"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, params={"api_key": api_key}, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            if "export" not in payload:
                raise RuntimeError(f"No export in API response: {payload}")
            size_mb = payload["export"].get("size")
            if size_mb is not None:
                print(f"export size : {size_mb:.1f} MB")
            return payload["export"]["link"]
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            print(f"  export link attempt {attempt}/3 failed: {exc}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Could not obtain export link: {last_error}")


def download_zip(link: str, dest: Path, attempts: int = 5) -> Path:
    """Download (or resume) the export zip, retrying on broken connections."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, attempts + 1):
        pos = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        mode = "ab" if pos else "wb"
        try:
            with requests.get(link, stream=True, timeout=600,
                              headers=headers, allow_redirects=True) as r:
                if pos and r.status_code == 200:
                    pos, mode = 0, "wb"  # server ignored Range -> restart
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) + pos
                done = pos
                with dest.open(mode) as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done / 1e6:6.1f} / {total / 1e6:6.1f} MB"
                                  f" ({done / total * 100:5.1f}%)", end="", flush=True)
                print()
            if total and dest.stat().st_size < total:
                raise IOError(f"incomplete: {dest.stat().st_size}/{total}")
            return dest
        except (requests.RequestException, IOError) as exc:
            print(f"\n  download attempt {attempt}/{attempts} interrupted: {exc}")
            if attempt < attempts:
                time.sleep(5)
    raise RuntimeError(f"Could not download {link} after {attempts} attempts")


def main() -> int:
    args = parse_args()
    api_key = os.getenv("ROBOFLOW_API_KEY")

    missing = [
        name
        for name, value in (
            ("ROBOFLOW_API_KEY", api_key),
            ("workspace", args.workspace),
            ("project", args.project),
            ("version", args.version),
        )
        if not value
    ]
    if missing:
        print("ERROR: missing required value(s): " + ", ".join(missing))
        print("Fill them in .env or pass as flags. Get slugs from the dataset page:")
        print("  <dataset> -> Download Dataset -> YOLOv8 -> Show download code")
        return 1

    print(f"workspace : {args.workspace}")
    print(f"project   : {args.project}")
    print(f"version   : {args.version}")
    print(f"format    : {args.format}")
    print(f"location  : {args.location}")

    try:
        link = get_export_link(args.workspace, args.project, args.version,
                               args.format, api_key)
    except requests.HTTPError as exc:
        print(f"ERROR: export request failed: {exc}")
        return 1

    zip_path = config.RAW_DIR / f"{args.project}-{args.version}.zip"
    download_zip(link, zip_path)

    print(f"extracting -> {args.location}")
    args.location.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.location)

    zip_path.unlink(missing_ok=True)
    print("\nDownload complete.")
    print(f"dataset root : {args.location}")
    print("Next: .\\.venv\\Scripts\\python.exe scripts\\verify_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
