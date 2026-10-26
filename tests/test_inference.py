"""End-to-end tests for the inference glue (synthetic, no video)."""

from src.analysis.roi import LaneROI, load_rois
from src.decision.intersection import Approach, FixedPlan, Intersection, Phase, load_intersection
from src.inference.analyze_video import counts_from_frames, plan_from_counts, synthetic_frames

A1 = LaneROI("A1", "A", [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
             buffer_px=0, hysteresis_frames=3)
B1 = LaneROI("B1", "B", [(20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)],
             buffer_px=0, hysteresis_frames=3)


def _intersection() -> Intersection:
    return Intersection(
        name="t",
        approaches={"A": Approach("A", "NS", 2), "B": Approach("B", "EW", 2)},
        phases=[Phase("P1", "A"), Phase("P2", "B")],
        lost_time_per_phase=4.0,
        fixed_plan=FixedPlan(cycle=90.0, green={"P1": 42.0, "P2": 42.0}),
    )


def test_synthetic_counts_are_exact():
    rois = [A1, B1]
    spec = {"A1": {"car": 5, "motorcycle": 4}, "B1": {"bus": 2}}
    frames = synthetic_frames(rois, spec, fps=10.0)
    counts = counts_from_frames(frames, rois, fps=10.0)
    assert counts.approach_counts["A"] == {"car": 5, "motorcycle": 4}
    assert counts.approach_counts["B"] == {"bus": 2}
    assert counts.total_vehicles == 11


def test_pcu_weighting_applied():
    rois = [A1]
    frames = synthetic_frames(rois, {"A1": {"motorcycle": 10}}, fps=10.0)
    counts = counts_from_frames(frames, rois, fps=10.0)
    # 10 motorcycles at 0.5 PCU each = 5.0, not 10
    assert counts.pcu_by_approach["A"] == 5.0


def test_plan_from_counts_reduces_delay_when_imbalanced():
    rois = [A1, B1]
    spec = {"A1": {"car": 60, "bus": 6}, "B1": {"car": 10}}
    frames = synthetic_frames(rois, spec, fps=10.0)
    counts = counts_from_frames(frames, rois, fps=10.0)
    plan, comparison, flows = plan_from_counts(counts, _intersection(), window_s=300.0)
    assert set(flows) == {"P1", "P2"}
    assert flows["P1"] > flows["P2"]
    assert comparison.optimized_avg_delay <= comparison.fixed_avg_delay
    assert plan.green["P1"] > plan.green["P2"]


def test_bundled_config_files_load():
    rois = load_rois("configs/rois/example.json")
    inter = load_intersection("configs/intersections/default.json")
    assert {r.approach for r in rois} == {"A", "B"}
    assert len(inter.phases) == 2
    assert inter.fixed_plan is not None


def test_run_on_video_encodes_playable_output_video(tmp_path):
    import cv2, sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "app"))
    from app.components import video
    from src.models import load_detector
    from src import config

    clip_path = config.SAMPLES_DIR / "traffic_clip.mp4"
    if not clip_path.exists():
        return  # skip if clip not found

    rois = load_rois(config.ROOT / "configs" / "rois" / "clip_768x432.json")
    detector = load_detector(weights=config.BEST_WEIGHTS, device="cpu")
    out_video = tmp_path / "test_annotated.mp4"

    counts, keyframes = video.run_on_video(
        detector, clip_path, rois, stride=2, fps=25.0, save_annotated_path=out_video
    )

    assert out_video.exists()
    assert out_video.stat().st_size > 0
    cap = cv2.VideoCapture(str(out_video))
    assert cap.isOpened()
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    assert n_frames > 0

