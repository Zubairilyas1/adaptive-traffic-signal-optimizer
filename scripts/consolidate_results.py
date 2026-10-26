"""Consolidate all evaluation metrics into one summary (MD + JSON).

Reads whatever metrics exist under reports/figures/ and writes
reports/results_summary.md and .json for the report and slides.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\consolidate_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


def load(tag: str) -> dict | None:
    path = config.FIGURES_DIR / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> int:
    figs = config.FIGURES_DIR
    fine = load("metrics_finetuned")
    coco = load("metrics_coco_baseline")
    count = load("metrics_count")
    ablation = load("ablation_pcu")
    synth = load("analysis_synthetic")

    lines: list[str] = ["# Results Summary", ""]
    summary: dict = {}

    lines.append("## Detection accuracy (test split)")
    lines.append("")
    lines.append("| Model | Classes | mAP@0.5 | mAP@0.5:0.95 |")
    lines.append("|---|---|---|---|")
    if coco:
        lines.append(f"| COCO zero-shot baseline | 4 | {coco['map50']:.3f} | {coco['map50_95']:.3f} |")
        summary["coco"] = coco
    if fine:
        lines.append(f"| Fine-tuned (ours) | {len(fine['per_class'])} | {fine['map50']:.3f} | {fine['map50_95']:.3f} |")
        summary["finetuned"] = fine
    lines.append("")
    if fine:
        lines.append("### Per-class AP (fine-tuned)")
        lines.append("")
        lines.append("| id | class | AP@0.5 | AP@0.5:0.95 |")
        lines.append("|---|---|---|---|")
        for row in fine["per_class"]:
            lines.append(f"| {row['class_id']} | {row['name']} | {row['ap50']:.3f} | {row['ap50_95']:.3f} |")
        lines.append("")

    if count:
        lines.append("## Count accuracy (counting is what the analysis layer uses)")
        lines.append("")
        lines.append(f"- Images: {count['images']}, ground-truth boxes: {count['gt_boxes']}, predicted: {count['pred_boxes']}")
        lines.append(f"- **MAE**: {count['mae']:.3f} vehicles/image")
        lines.append(f"- RMSE: {count['rmse']:.3f}")
        lines.append(f"- Bias: {count['bias']:+.3f}")
        lines.append(f"- Exact-count matches: {count['exact_match_pct']:.1f}%")
        lines.append("")
        summary["count"] = count

    if ablation:
        lines.append("## Ablation - does PCU weighting matter?")
        lines.append("")
        lines.append(f"Window: {ablation['window_s']:.0f}s")
        lines.append("")
        lines.append("| Quantity | Naive count | PCU-weighted |")
        lines.append("|---|---|---|")
        lines.append(f"| flow P1 (PCU/h) | {ablation['naive']['flow_p1']:.0f} | {ablation['pcu']['flow_p1']:.0f} |")
        lines.append(f"| flow P2 (PCU/h) | {ablation['naive']['flow_p2']:.0f} | {ablation['pcu']['flow_p2']:.0f} |")
        lines.append(f"| cycle (s) | {ablation['naive']['cycle']:.1f} | {ablation['pcu']['cycle']:.1f} |")
        lines.append(f"| green P1 (s) | {ablation['naive']['green']['P1']:.1f} | {ablation['pcu']['green']['P1']:.1f} |")
        lines.append(f"| green P2 (s) | {ablation['naive']['green']['P2']:.1f} | {ablation['pcu']['green']['P2']:.1f} |")
        lines.append(f"| avg delay (s/veh) | {ablation['naive']['avg_delay']:.2f} | {ablation['pcu']['avg_delay']:.2f} |")
        lines.append("")
        lines.append(f"**Green-time difference on P1: {ablation['green_p1_difference_s']:.1f} s** - PCU weighting changes the recommended plan.")
        lines.append("")
        summary["ablation"] = ablation

    if synth:
        lines.append("## Signal optimization (synthetic 2-phase demo)")
        lines.append("")
        lines.append(f"- Critical flow ratio Y: {synth['plan']['Y']:.3f}")
        lines.append(f"- Optimized cycle: {synth['plan']['cycle']:.1f} s")
        lines.append(f"- Fixed-time avg delay: {synth['delay']['fixed_avg_s']:.2f} s/veh")
        lines.append(f"- Webster avg delay: {synth['delay']['optimized_avg_s']:.2f} s/veh")
        lines.append(f"- **Delay reduction: {synth['delay']['reduction_pct']:.1f}%**")
        lines.append("")
        summary["synthetic"] = synth

    domain = load("metrics_domain_gap")
    if domain:
        lines.append("## Domain gap (external clip)")
        lines.append("")
        lines.append(f"- External clip: {domain.get('clip', 'n/a')}")
        lines.append(f"- Detection quality vs benchmark: {domain.get('note', '')}")
        lines.append("")
        summary["domain_gap"] = domain

    rl = load("rl_comparison")
    if rl:
        lines.append("## Adaptive control: Fixed-Time vs Rule-Based vs PPO (SUMO)")
        lines.append("")
        for scenario in ("normal", "stress"):
            rows = rl.get(scenario, [])
            if not rows:
                continue
            lines.append(f"**{scenario.capitalize()} demand**")
            lines.append("")
            lines.append("| Controller | Avg queue | Avg delay (s/veh) | Throughput |")
            lines.append("|---|---|---|---|")
            for r in rows:
                lines.append(f"| {r['controller']} | {r['avg_queue']:.2f} | {r['avg_delay']:.2f} | {r['throughput']} |")
            lines.append("")
        if rl.get("ppo_delay_reduction_normal_pct") is not None:
            lines.append(f"- PPO delay reduction vs Fixed-Time: normal "
                         f"{rl['ppo_delay_reduction_normal_pct']:.1f}%, stress "
                         f"{rl['ppo_delay_reduction_stress_pct']:.1f}%")
            lines.append("")
        summary["rl"] = rl

    report = config.REPORTS_DIR / "results_summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    json_out = config.REPORTS_DIR / "results_summary.json"
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nwritten -> {report}")
    print(f"written -> {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
