"""Build reports/slides.pptx (presentation deck) with python-pptx.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\build_slides_pptx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pptx import Presentation  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from src import config  # noqa: E402

F = config.FIGURES_DIR
KT = config.REPORTS_DIR / "kaggle_training"
EF = config.REPORTS_DIR / "eval_finetuned"


def add_bullets(prs, title, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    body = slide.placeholders[1].text_frame
    body.clear()
    for i, b in enumerate(bullets):
        p = body.paragraphs[0] if i == 0 else body.add_paragraph()
        p.text = b
        p.font.size = Pt(18)
    return slide


def add_image(prs, title, image, caption=""):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    if Path(image).exists():
        slide.shapes.add_picture(str(image), Inches(0.7), Inches(1.5), height=Inches(5.3))
    if caption:
        tb = slide.shapes.add_textbox(Inches(0.7), Inches(6.9), Inches(12), Inches(0.5))
        tb.text_frame.text = caption
    return slide


def build() -> int:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # 1 title
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "Traffic Signal Timing Optimizer from Intersection Video"
    s.placeholders[1].text = (
        "AI2002 Artificial Intelligence - Semester Project (Fall 2026)\n"
        "Path A (Software-Based)\n"
        "[MEMBER 1]  |  [MEMBER 2]  |  [MEMBER 3]\n"
        "06 December 2026"
    )

    add_bullets(prs, "Problem", [
        "Many intersections run fixed-time cycles set once and never adapted",
        "Green time wasted on empty approaches while others queue",
        "Adaptive systems need buried loop sensors - costly to install",
        "Root cause: the signal does not know actual demand per approach",
    ])

    add_bullets(prs, "Motivation & Real-World Justification", [
        "Stakeholder: traffic engineer / municipal operator",
        "Decision: what green split minimises total delay this cycle?",
        "Camera-only solution needs no road works - uses existing CCTV",
        "Affects daily commuters, fuel use and emissions",
    ])

    add_bullets(prs, "Objectives", [
        "Detect & classify vehicles with a trained model",
        "Convert counts to capacity-aware demand (PCU)",
        "Compute an optimal signal plan (Webster's method)",
        "Quantify delay reduction vs fixed-time baseline",
        "Demonstrate the complete flow in a Streamlit app",
    ])

    add_image(prs, "System Architecture", F / "pipeline.png",
              "Model -> Analysis -> Decision, wrapped in a Streamlit app")

    add_bullets(prs, "Layer 1 - Detection Model", [
        "YOLOv8n single-stage detector, fine-tuned from COCO weights",
        "7 classes: car, motorcycle, autorickshaw, van, tractor, bus, truck",
        "Fine-tuning needed: 3 classes do not exist in COCO at all",
        "Trained on Kaggle GPU (local GPU thermally throttles)",
        "~34 FPS inference at 640 px on the local T1200",
    ])

    add_bullets(prs, "Dataset", [
        "Three public Roboflow datasets merged into one 7-class set",
        "8,513 images (train 6,746 / val 962 / test 805), 21,275 boxes",
        "Unified class index via explicit remapping",
        "Documented limitations: class imbalance, video-derived frames",
    ])

    add_bullets(prs, "Layer 2 - Tracking & Lane Counting", [
        "ByteTrack assigns persistent IDs -> count once, not per frame",
        "Geometric lane ROIs (polygon + ray-casting + hysteresis)",
        "Lane assignment is geometry, not a learned task - honest scoping",
        "Output: de-duplicated per-lane class counts",
    ])

    add_bullets(prs, "Key Insight - PCU Weighting", [
        "Naive counting: one vehicle = one vehicle",
        "But a bus occupies far more capacity than a motorcycle",
        "Passenger Car Units: motorcycle 0.5 ... bus/truck 3.5",
        "This is why the detector must classify, not just detect",
    ])

    add_bullets(prs, "Layer 3 - Webster's Method", [
        "Flow ratio y = q / s  (arrival flow / saturation flow)",
        "Optimal cycle  Co = (1.5 L + 5) / (1 - Y)",
        "Green split    g_i = (y_i / Y) (Co - L)",
        "Clamped to practical cycle/green limits",
        "Delay compared against the fixed-time plan",
        "Oversaturation (Y >= 1) is flagged, not silently mishandled",
    ])

    add_bullets(prs, "Simulation & Reinforcement Learning", [
        "Measured demand drives a SUMO four-approach intersection (TraCI)",
        "PPO agent: state = per-approach queues + phase + elapsed green",
        "Action = keep/switch phase; reward = negative total queue",
        "Compared against Fixed-Time and Rule-Based (actuated) controllers",
        "Trained on CPU (~120k steps, no GPU thermal risk)",
    ])

    add_image(prs, "Results - Adaptive Control", F / "rl_comparison.png",
              "PPO cuts delay vs Fixed-Time by 57.6% (normal) / 79.4% (stress)")

    add_image(prs, "Results - Detection", KT / "results.png",
              "Training curves: loss and mAP over epochs")

    add_bullets(prs, "Results - Detection Accuracy", [
        "COCO zero-shot baseline: mAP@0.5 = 0.727 (only 4 classes)",
        "Fine-tuned (ours): mAP@0.5 = 0.944 (7 classes)",
        "+0.217 mAP@0.5, and 3 classes COCO cannot detect",
        "Per-class AP@0.5: autorickshaw 0.976, bus 0.974, van 0.957",
    ])

    add_image(prs, "Results - Count Accuracy", F / "count_accuracy.png",
              "MAE 0.395 vehicles/image, 74.8% exact counts (805 images)")

    add_bullets(prs, "Results - PCU Ablation", [
        "Same scenario, two plans: naive count vs PCU-weighted",
        "Green P1: 26.1 s (naive) vs 23.0 s (PCU) - a 3.1 s shift",
        "Naive counting under-serves the heavy-vehicle approach",
        "Proves PCU weighting changes the recommendation, not cosmetic",
    ])

    add_image(prs, "Results - Delay Reduction", F / "delay_synthetic.png",
              "Fixed-time 20.64 s/veh -> Webster 14.01 s/veh  =  32.1% reduction")

    add_image(prs, "Domain Gap - External Footage", F / "domain_gap.png",
              "On Pakistani footage: 2.53 detections/img, all 7 classes found (qualitative)")

    add_bullets(prs, "Limitations", [
        "Class imbalance (bus/van thin); per-class metrics reported",
        "Video-derived frames -> likely near-duplicate leakage; 0.944 optimistic",
        "Domain gap: not Pakistan data; external labels unusable for mAP",
        "PCU & saturation flow are documented constants, not measured",
        "Lane ROIs manual; Webster valid only under undersaturation",
        "Assistive prototype, not a certified signal controller",
    ])

    add_bullets(prs, "Conclusion & Future Work", [
        "A small fine-tuned detector + PCU analysis recommends better signal plans",
        "32.1% average delay reduction vs fixed-time on the demonstration",
        "Future: hand-labelled local data, multi-intersection coordination,",
        "learned lane assignment, RL adaptive control, edge ONNX deployment",
    ])

    add_bullets(prs, "Thank You", [
        "Questions?",
        "Code, report and figures are organised under the project repository",
    ])

    out = config.REPORTS_DIR / "slides.pptx"
    prs.save(out)
    print(f"slides -> {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
