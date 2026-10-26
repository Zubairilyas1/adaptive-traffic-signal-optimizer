"""Turn detections (real video or synthetic) into counts, flows and a plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import cv2
import numpy as np

from src.analysis import flow as flow_mod
from src.analysis import pcu as pcu_mod
from src.analysis.counting import LaneCounter
from src.analysis.roi import LaneROI
from src.decision.baseline import evaluate
from src.decision.delay import DelayComparison
from src.decision.intersection import Intersection, SignalPlan


@dataclass
class ClipCounts:
    lane_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    approach_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    processed_frames: int = 0
    duration_s: float = 0.0
    kinematic_snapshots: list[Any] = field(default_factory=list)
    actuator_history: list[Any] = field(default_factory=list)
    actuator_events: list[Any] = field(default_factory=list)

    @property
    def pcu_by_approach(self) -> dict[str, float]:
        return pcu_mod.approach_pcu(self.approach_counts)

    @property
    def halted_queue_pcu(self) -> dict[str, float]:
        """Return the latest instantaneous halted waiting queue PCU per approach."""
        if not self.kinematic_snapshots:
            return self.pcu_by_approach
        return self.kinematic_snapshots[-1].approach_halted_pcu

    @property
    def flowing_throughput_pcu(self) -> dict[str, float]:
        """Return the latest instantaneous flowing throughput PCU per approach."""
        if not self.kinematic_snapshots:
            return {}
        return self.kinematic_snapshots[-1].approach_flowing_pcu

    @property
    def total_vehicles(self) -> int:
        return sum(sum(c.values()) for c in self.approach_counts.values())


def _finalise(
    counter: LaneCounter,
    processed_frames: int,
    duration_s: float,
    kinematic_counter: Optional[Any] = None,
    actuator: Optional[Any] = None,
) -> ClipCounts:
    snapshots = kinematic_counter.timeline_history if kinematic_counter is not None else []
    act_history = actuator.state_history if actuator is not None else []
    act_events = actuator.events_log if actuator is not None else []
    return ClipCounts(
        lane_counts=counter.counts_by_lane(),
        approach_counts=counter.counts_by_approach(),
        processed_frames=processed_frames,
        duration_s=duration_s,
        kinematic_snapshots=snapshots,
        actuator_history=act_history,
        actuator_events=act_events,
    )


def count_clip(
    detector,
    source,
    rois: Iterable[LaneROI],
    stride: int = 1,
    fps: Optional[float] = None,
    intersection: Optional[Intersection] = None,
) -> ClipCounts:
    """Run detection + ByteTrack over a video and count vehicles per lane."""
    if fps is None:
        cap = cv2.VideoCapture(str(source))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.release()
        if fps <= 0:
            fps = 25.0

    from src.analysis.kinematics import KinematicLaneCounter
    from src.decision.actuator import AdaptiveActuator

    counter = LaneCounter(rois)
    kinematic_counter = KinematicLaneCounter(rois)
    actuator = AdaptiveActuator(intersection) if intersection is not None else None

    n = 0
    last_idx = 0
    for idx, result in detector.track_video(source, stride=stride):
        dets = detector.result_to_dicts(result)
        ts = (idx / fps) if fps > 0 else float(idx)
        counter.update(dets, timestamp=ts)
        snap = kinematic_counter.update(dets, timestamp=ts)
        if actuator is not None:
            actuator.step(ts, snapshot=snap)
        n += 1
        last_idx = idx

    duration_s = (last_idx / fps) if (fps and fps > 0) else float(n * stride)
    return _finalise(
        counter,
        n,
        duration_s,
        kinematic_counter=kinematic_counter,
        actuator=actuator,
    )


def counts_from_frames(frames: Iterable[list[dict]], rois: Iterable[LaneROI],
                       fps: float = 1.0) -> ClipCounts:
    """Build counts from pre-computed detection frames (synthetic or cached)."""
    counter = LaneCounter(rois)
    n = 0
    for t, dets in enumerate(frames):
        counter.update(dets, timestamp=t / fps)
        n += 1
    return _finalise(counter, n, n / fps if fps else float(n))


def plan_from_counts(
    counts: ClipCounts,
    intersection: Intersection,
    window_s: Optional[float] = None,
    fixed_plan=None,
) -> tuple[SignalPlan, DelayComparison, dict[str, float]]:
    """Convert counts to PCU flows and produce an optimized plan + comparison."""
    duration = window_s if window_s is not None else counts.duration_s
    if duration <= 0:
        raise ValueError("Clip duration is zero; cannot compute flow rates.")

    approach_flow = {
        a: flow_mod.pcu_per_hour(pcu, duration)
        for a, pcu in counts.pcu_by_approach.items()
    }
    for approach in intersection.approaches.values():
        approach_flow.setdefault(approach.id, 0.0)

    flows_by_phase = flow_mod.flows_by_phase(intersection, approach_flow)
    plan, comparison = evaluate(intersection, flows_by_phase, fixed_plan=fixed_plan)
    return plan, comparison, flows_by_phase


def _polygon_centroid(polygon: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def synthetic_frames(
    rois: Iterable[LaneROI],
    spec: dict[str, dict[str, int]],
    fps: float = 10.0,
    gap_frames: int = 2,
) -> list[list[dict]]:
    """Generate detection frames that place vehicles inside each lane ROI.

    ``spec`` maps lane_id -> {class_name: count}. Each vehicle stays inside its
    lane for enough frames to pass the hysteresis threshold, so it is counted
    exactly once. Used to exercise the pipeline with no video.
    """
    rois = list(rois)
    by_lane = {r.lane_id: r for r in rois}
    frames: list[list[dict]] = []
    track_id = 1

    # order vehicles lane by lane; each occupies a run of frames
    for lane_id, classes in spec.items():
        roi = by_lane[lane_id]
        cx, cy = _polygon_centroid(roi.polygon)
        box = (cx - 20, cy - 30, cx + 20, cy)
        for class_name, n in classes.items():
            for _ in range(n):
                for _ in range(roi.hysteresis_frames + 1):
                    frames.append([{
                        "track_id": track_id,
                        "bbox": box,
                        "class_name": class_name,
                        "confidence": 0.9,
                    }])
                for _ in range(gap_frames):
                    frames.append([])
                track_id += 1
    return frames


_PALETTE = {"A": (60, 200, 60), "B": (200, 140, 60)}


def _overlay_rois(image: np.ndarray, rois: Iterable[LaneROI]) -> np.ndarray:
    vis = image.copy()
    for roi in rois:
        pts = np.array(roi.polygon, dtype=np.int32).reshape(-1, 1, 2)
        color = _PALETTE.get(roi.approach, (240, 240, 240))
        cv2.polylines(vis, [pts], True, color, 2)
        cx = int(sum(p[0] for p in roi.polygon) / len(roi.polygon))
        cy = int(sum(p[1] for p in roi.polygon) / len(roi.polygon))
        cv2.putText(vis, roi.lane_id, (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
    return vis


def render_annotated_video(
    detector,
    source,
    out_path,
    rois: Optional[Iterable[LaneROI]] = None,
    stride: int = 1,
    fps: Optional[float] = None,
    progress: Optional[Callable[[int], None]] = None,
    intersection: Optional[Intersection] = None,
) -> Path:
    """Write an annotated MP4 (boxes + class labels + track IDs + Live HUD) in H.264.

    Uses ByteTrack for stable IDs and result.plot() for the overlay, combined
    with HUDOverlayRenderer for live traffic signal badges and dynamic actuation ticker.
    """
    import imageio_ffmpeg
    from src.analysis.hud import HUDOverlayRenderer
    from src.analysis.kinematics import KinematicLaneCounter
    from src.decision.actuator import AdaptiveActuator

    src = Path(source)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {src}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    out_fps = max(1.0, in_fps / max(1, stride))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio_ffmpeg.write_frames(
        str(out_path), size=(width, height), fps=out_fps,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        macro_block_size=1, output_params=["-movflags", "+faststart"],
    )
    writer.send(None)

    n = 0
    rois_list = list(rois) if rois else None
    hud = HUDOverlayRenderer(rois_list, intersection=intersection) if rois_list else None
    actuator = AdaptiveActuator(intersection) if intersection is not None else None
    kinematic_counter = KinematicLaneCounter(rois_list) if rois_list else None

    for result in detector.model.track(
        source=str(src),
        stream=True,
        persist=True,
        vid_stride=max(1, stride),
        device=detector.device,
        conf=detector.conf,
        iou=detector.iou,
        imgsz=detector.imgsz,
        classes=detector.class_ids,
        max_det=detector.max_det,
        tracker="bytetrack.yaml",
        verbose=False,
    ):
        vis = result.plot()
        if rois_list:
            vis = _overlay_rois(vis, rois_list)
            dets = detector.result_to_dicts(result)
            ts = (n * stride / in_fps) if in_fps else float(n)
            snap = kinematic_counter.update(dets, timestamp=ts) if kinematic_counter else None
            ctrl = actuator.step(ts, snapshot=snap) if actuator else None
            if hud:
                vis = hud.render(vis, controller_state=ctrl, snapshot=snap, timestamp=ts)
        writer.send(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB).tobytes())
        n += 1
        if progress:
            progress(n)

    writer.close()
    return out_path
