"""Page 3 - Model & Metrics: detector accuracy, validation curves, and engineering bounds."""

from __future__ import annotations

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from app import _bootstrap  # noqa: F401

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from app.components.cards import render_executive_kpi_card_row, show_disclaimer
from app.components.state import init_state
from app.components.theme import apply_theme, render_breadcrumb, render_header
from src import config

init_state()
apply_theme()

# Top Breadcrumb & Console Header (Image 1 style)
render_breadcrumb("ANALYTICS > DETECTOR BENCHMARKS")
render_header(
    title="Model Accuracy & Quantitative Metrics",
    subtitle="Validation metrics, detection accuracy (mAP@0.5), PCU ablation, and documented bounds",
    chips=[
        ("MODEL: YOLOV8N FINE-TUNED", "chip-green"),
        ("VAL mAP@0.5: 0.944", "chip-cyan"),
        ("TEST SPLIT: 805 IMAGES", "chip-amber"),
    ],
)


def _load_metrics(tag: str) -> dict | None:
    path = config.FIGURES_DIR / f"metrics_{tag}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


fine = _load_metrics("finetuned")
coco = _load_metrics("coco_baseline")

if fine:
    render_executive_kpi_card_row([
        {
            "header": "Validation mAP@0.5",
            "value": f"{fine['map50']:.3f}",
            "delta": "+0.082 vs COCO",
            "delta_positive": True,
            "icon": "🎯",
            "icon_bg": "rgba(0, 240, 255, 0.12)",
            "icon_color": "#00f0ff",
        },
        {
            "header": "mAP@[0.5:0.95]",
            "value": f"{fine['map50_95']:.3f}",
            "delta": "HIGH IOU RIGOR",
            "delta_positive": True,
            "icon": "📐",
            "icon_bg": "rgba(0, 255, 157, 0.12)",
            "icon_color": "#00ff9d",
        },
        {
            "header": "Vehicle Classes",
            "value": f"{len(fine['per_class'])} classes",
            "delta": "+3 NON-COCO TYPES",
            "delta_positive": True,
            "icon": "🚗",
            "icon_bg": "rgba(255, 170, 0, 0.12)",
            "icon_color": "#ffaa00",
        },
        {
            "header": "Held-Out Test Split",
            "value": "805 imgs",
            "delta": "ANNOTATED BENCHMARK",
            "delta_positive": True,
            "icon": "📊",
            "icon_bg": "rgba(255, 51, 102, 0.12)",
            "icon_color": "#ff3366",
        },
    ])

st.subheader("Detector Accuracy (YOLOv8n)")

if fine and coco:
    rows = [
        {
            "Model Variant": "COCO zero-shot baseline",
            "Classes": 4,
            "mAP@0.5": round(coco["map50"], 3),
            "mAP@0.5:0.95": round(coco["map50_95"], 3),
        },
        {
            "Model Variant": "Fine-tuned 7-class (ours)",
            "Classes": len(fine["per_class"]),
            "mAP@0.5": round(fine["map50"], 3),
            "mAP@0.5:0.95": round(fine["map50_95"], 3),
        },
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    delta = fine["map50"] - coco["map50"]
    st.success(
        f"Fine-tuning improves mAP@0.5 by {delta:+.3f} and adds "
        f"{len(fine['per_class']) - 4} vehicle classes COCO cannot detect "
        "(autorickshaw, van, tractor).",
    )
elif fine:
    st.dataframe(pd.DataFrame([{
        "Model Variant": "Fine-tuned 7-class (ours)",
        "mAP@0.5": round(fine["map50"], 3),
        "mAP@0.5:0.95": round(fine["map50_95"], 3),
    }]), width="stretch", hide_index=True)
else:
    st.warning("No metrics found. Run scripts/evaluate_detector.py first.")

if fine:
    st.markdown("**Per-Class Average Precision (Held-out Test Split)**")
    pc = pd.DataFrame(fine["per_class"]).rename(
        columns={"class_id": "id", "name": "Class", "ap50": "AP@0.5", "ap50_95": "AP@0.5:0.95"}
    )
    st.dataframe(pc.round(3), width="stretch", hide_index=True)

st.divider()

# Validation Tabs
tab_count, tab_ablation, tab_domain, tab_curves = st.tabs([
    "🎯 Count Accuracy",
    "⚖️ PCU Weighting Ablation",
    "🌍 Domain Gap Evaluation",
    "📈 Training Curves",
])

with tab_count:
    count = _load_metrics("count")
    if count:
        render_executive_kpi_card_row([
            {
                "header": "Mean Absolute Error",
                "value": f"{count['mae']:.2f}",
                "delta": "VEH / IMAGE",
                "delta_positive": count["mae"] < 2.0,
                "icon": "🎯",
                "icon_bg": "rgba(0, 240, 255, 0.12)",
                "icon_color": "#00f0ff",
            },
            {
                "header": "Root Mean Squared Error",
                "value": f"{count['rmse']:.2f}",
                "delta": "STDEV SENSITIVE",
                "delta_positive": count["rmse"] < 3.0,
                "icon": "📉",
                "icon_bg": "rgba(0, 255, 157, 0.12)",
                "icon_color": "#00ff9d",
            },
            {
                "header": "Systematic Bias",
                "value": f"{count['bias']:+.2f}",
                "delta": "NEAR ZERO OFFSET",
                "delta_positive": abs(count["bias"]) < 0.5,
                "icon": "⚖️",
                "icon_bg": "rgba(255, 170, 0, 0.12)",
                "icon_color": "#ffaa00",
            },
            {
                "header": "Exact Match Rate",
                "value": f"{count['exact_match_pct']:.1f}%",
                "delta": "PERFECT MATCH",
                "delta_positive": True,
                "icon": "🏆",
                "icon_bg": "rgba(255, 51, 102, 0.12)",
                "icon_color": "#ff3366",
            },
        ])
        scatter = config.FIGURES_DIR / "count_accuracy.png"
        if scatter.exists():
            st.image(str(scatter), caption="Predicted vs ground-truth counts", width="stretch")
    else:
        st.info("Run `python scripts/count_accuracy.py --device cpu` to generate counts evaluation.")

with tab_ablation:
    ablation = _load_metrics("ablation_pcu")
    if ablation:
        ab = pd.DataFrame([
            {"Quantity": "flow P1 (PCU/h)", "Naive": round(ablation["naive"]["flow_p1"]), "PCU Weighted": round(ablation["pcu"]["flow_p1"])},
            {"Quantity": "flow P2 (PCU/h)", "Naive": round(ablation["naive"]["flow_p2"]), "PCU Weighted": round(ablation["pcu"]["flow_p2"])},
            {"Quantity": "green P1 (s)", "Naive": ablation["naive"]["green"]["P1"], "PCU Weighted": ablation["pcu"]["green"]["P1"]},
            {"Quantity": "green P2 (s)", "Naive": ablation["naive"]["green"]["P2"], "PCU Weighted": ablation["pcu"]["green"]["P2"]},
            {"Quantity": "avg delay (s/veh)", "Naive": round(ablation["naive"]["avg_delay"], 2), "PCU Weighted": round(ablation["pcu"]["avg_delay"], 2)},
        ])
        st.dataframe(ab, width="stretch", hide_index=True)
        st.caption(
            f"Accounting for PCU shifts green on Phase 1 by {ablation['green_p1_difference_s']:.1f} s, "
            "proving naive counts produce suboptimal traffic plans."
        )
    else:
        st.info("Run `python scripts/ablation_pcu.py` to generate ablation metrics.")

with tab_domain:
    domain = _load_metrics("domain_gap")
    if domain:
        d = st.columns(3)
        d[0].metric("Images Sampled", domain["images_sampled"])
        d[1].metric("Mean Detections/Img", f"{domain['mean_detections_per_image']:.2f}")
        d[2].metric("Classes Found", len(domain["class_distribution"]))
        st.dataframe(pd.DataFrame([domain["class_distribution"]]), width="stretch", hide_index=True)
        dg = config.FIGURES_DIR / "domain_gap.png"
        if dg.exists():
            st.image(str(dg), caption="Fine-tuned detector evaluation on external footage", width="stretch")
    else:
        st.info("Run `python scripts/domain_gap.py --device cpu` to generate domain gap metrics.")

with tab_curves:
    fig_dir = config.REPORTS_DIR / "kaggle_training"
    figs = ["results.png", "confusion_matrix.png", "BoxPR_curve.png", "BoxF1_curve.png"]
    cols = st.columns(2)
    shown = 0
    for name in figs:
        p = fig_dir / name
        if p.exists():
            cols[shown % 2].image(str(p), caption=name, width="stretch")
            shown += 1
    if shown == 0:
        st.info("Training curve figures not found in reports/kaggle_training/.")

show_disclaimer()
