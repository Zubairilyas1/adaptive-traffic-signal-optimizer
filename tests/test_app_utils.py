"""Tests for the app's pure helper functions (no Streamlit required)."""

from app import utils
from src.analysis.roi import LaneROI
from src.decision.baseline import evaluate
from src.decision.intersection import Approach, FixedPlan, Intersection, Phase
from src.inference.analyze_video import ClipCounts


def _counts() -> ClipCounts:
    return ClipCounts(
        lane_counts={"A1": {"car": 3, "motorcycle": 2}},
        approach_counts={"A": {"car": 3, "motorcycle": 2}},
        processed_frames=100,
        duration_s=60.0,
    )


def _intersection() -> Intersection:
    return Intersection(
        name="t",
        approaches={"A": Approach("A", "NS", 2), "B": Approach("B", "EW", 2)},
        phases=[Phase("P1", "A"), Phase("P2", "B")],
        lost_time_per_phase=4.0,
        fixed_plan=FixedPlan(cycle=90.0, green={"P1": 42.0, "P2": 42.0}),
    )


def test_counts_csv_has_header_and_pcu():
    csv_text = utils.counts_csv(_counts())
    lines = csv_text.strip().splitlines()
    assert lines[0] == "approach,class,count,pcu"
    body = "\n".join(lines[1:])
    assert "A,car,3,3.0" in body
    assert "A,motorcycle,2,1.0" in body  # 2 * 0.5 PCU


def test_plan_and_delay_csv():
    inter = _intersection()
    plan, comparison = evaluate(inter, {"P1": 1200.0, "P2": 600.0})
    plan_text = utils.plan_csv(plan, inter)
    assert "phase,approach,label,lanes,green_s" in plan_text
    assert "P1" in plan_text and "P2" in plan_text

    delay_text = utils.delay_csv(comparison)
    assert "fixed_delay_s" in delay_text
    assert "AVERAGE" in delay_text


def test_roi_json_roundtrip():
    rois = [LaneROI("A1", "A", [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])]
    text = utils.rois_to_json(rois)
    back = utils.rois_from_json(text)
    assert len(back) == 1
    assert back[0].lane_id == "A1"
    assert back[0].polygon == rois[0].polygon


def test_presets_and_intersections_exist():
    assert utils.list_roi_presets(), "no ROI presets found"
    assert utils.list_intersections(), "no intersection configs found"


def test_sample_clips_and_demo_video_helpers():
    clips = utils.list_sample_clips()
    assert isinstance(clips, list)
    for clip in clips:
        assert clip.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    # demo_video_path returns a Path or None, never raises
    demo = utils.demo_video_path()
    assert demo is None or demo.exists()
