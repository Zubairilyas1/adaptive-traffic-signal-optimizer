"""Render an annotated traffic video (YOLO boxes + labels + IDs) in H.264.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\render_annotated_video.py ^
        --video data\\samples\\street_clip.mp4 ^
        --rois configs\\rois\\clip_768x432.json ^
        --out reports\\annotated\\street_clip_annotated.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.analysis.roi import load_rois  # noqa: E402
from src.inference.analyze_video import render_annotated_video  # noqa: E402
from src.models import load_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render an annotated video.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--rois", type=Path, default=None)
    p.add_argument("--weights", type=Path, default=config.BEST_WEIGHTS)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out or (config.REPORTS_DIR / "annotated" / f"{args.video.stem}_annotated.mp4")
    rois = load_rois(args.rois) if args.rois else None

    detector = load_detector(weights=args.weights, device=args.device)
    print(f"video  : {args.video}")
    print(f"device : {args.device}")
    print(f"out    : {out}")

    def tick(n: int) -> None:
        if n % 25 == 0:
            print(f"\r  frames processed: {n}", end="", flush=True)

    render_annotated_video(detector, args.video, out, rois=rois, stride=args.stride, progress=tick)
    print(f"\nannotated video -> {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
