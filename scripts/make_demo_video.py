"""Render a standalone demo video (results walkthrough) -> reports/demo_video.mp4.

This is a narrated-by-text reel built from the project's own figures, so you can
present the results even without a live run.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\make_demo_video.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

F = config.FIGURES_DIR
KT = config.REPORTS_DIR / "kaggle_training"
W, H, FPS = 1280, 720, 24


def _fig(title: str):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("#0f1116")
    return fig


def _to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr


def text_slide(title: str, bullets: list[str]) -> np.ndarray:
    fig = _fig(title)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.86, title, color="white", fontsize=30, fontweight="bold", va="top")
    y = 0.68
    for b in bullets:
        ax.text(0.09, y, "-", color="#54A24B", fontsize=20, va="top")
        ax.text(0.12, y, b, color="#e6e6e6", fontsize=20, va="top", wrap=True)
        y -= 0.13
    return _to_array(fig)


def image_slide(title: str, image: Path, caption: str = "") -> np.ndarray:
    fig = _fig(title)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.965, title, color="white", fontsize=24, fontweight="bold",
            ha="center", va="top")
    if image.exists():
        img = plt.imread(str(image))
        ax.imshow(img, aspect="auto", extent=[0.08, 0.92, 0.10, 0.90])
    else:
        ax.text(0.5, 0.5, f"missing figure: {image.name}", color="red",
                ha="center", va="center", fontsize=18)
    if caption:
        ax.text(0.5, 0.03, caption, color="#9fd3a0", fontsize=17, ha="center")
    return _to_array(fig)


def main() -> int:
    slides: list[np.ndarray] = []

    slides.append(text_slide(
        "Traffic Signal Timing Optimizer",
        ["AI2002 Artificial Intelligence - Semester Project",
         "Path A (Software-Based)",
         "Detection + PCU analysis + Webster optimization",
         "Group: [MEMBER 1] | [MEMBER 2] | [MEMBER 3]"],
    ))
    slides.append(text_slide(
        "Problem & Motivation",
        ["Fixed-time signals waste green on empty approaches",
         "Adaptive systems need buried loop sensors - costly",
         "Camera-only solution: no road works required",
         "Goal: recommend a better green split, measure delay saved"],
    ))
    slides.append(image_slide("System Pipeline", F / "pipeline.png",
                              "Model -> Analysis -> SUMO -> PPO control"))
    slides.append(text_slide(
        "Simulation & Reinforcement Learning",
        ["Measured demand drives a SUMO intersection (TraCI)",
         "Agent observes per-approach queues; reward = -queue",
         "PPO learns when to switch the signal phase",
         "Compared to Fixed-Time and Rule-Based (actuated) control",
         "Trained on CPU - no GPU thermal risk"],
    ))
    slides.append(image_slide("Results - Adaptive Control", F / "rl_comparison.png",
                              "PPO delay reduction vs Fixed-Time: 57.6% normal / 79.4% stress"))
    slides.append(image_slide("Detection Training", KT / "results.png",
                              "Fine-tuned YOLOv8n (7 vehicle classes)"))
    slides.append(text_slide(
        "Detection Accuracy (test split)",
        ["COCO zero-shot baseline: mAP@0.5 = 0.727 (4 classes)",
         "Fine-tuned (ours): mAP@0.5 = 0.944 (7 classes)",
         "+0.217 mAP, plus autorickshaw / van / tractor COCO cannot detect",
         "Per-class AP@0.5: autorickshaw 0.976, bus 0.974, van 0.957"],
    ))
    slides.append(image_slide("Count Accuracy", F / "count_accuracy.png",
                              "MAE 0.395 vehicles/image - 74.8% exact counts"))
    slides.append(image_slide("Delay Reduction", F / "delay_synthetic.png",
                              "Fixed 20.64 -> Webster 14.01 s/veh  =  32.1% reduction"))
    slides.append(image_slide("Domain Gap (local footage)", F / "domain_gap.png",
                              "Detects all 7 classes on Pakistani traffic (qualitative)"))
    slides.append(text_slide(
        "Conclusions",
        ["Small fine-tuned detector + PCU analysis recommends better plans",
         "32.1% average delay reduction on the demonstration",
         "Honest limits: video-frame leakage, domain gap, manual ROIs",
         "Future: local labelled data, multi-intersection coordination, edge deploy"],
    ))

    out = config.REPORTS_DIR / "demo_video.mp4"
    hold = FPS * 4  # 4 seconds per slide

    try:
        import imageio_ffmpeg
    except ImportError:
        print("ERROR: imageio-ffmpeg not installed. Run: pip install imageio-ffmpeg")
        return 1

    # H.264 (libx264) + yuv420p is required for browser playback (OpenCV's mp4v is not).
    writer = imageio_ffmpeg.write_frames(
        str(out),
        size=(W, H),
        fps=FPS,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)  # seed the generator
    for arr in slides:
        rgb = np.ascontiguousarray(arr, dtype=np.uint8)
        for _ in range(hold):
            writer.send(rgb.tobytes())
    writer.close()

    print(f"demo video -> {out}  ({len(slides)} slides, ~{len(slides) * 4}s, H.264)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
