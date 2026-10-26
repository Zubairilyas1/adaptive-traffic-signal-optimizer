"""Video upload handling, ROI drawing and inference drivers for the app."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from pathlib import Path

import cv2
import numpy as np

from app.utils import temp_dir
from src.analysis.counting import LaneCounter
from src.analysis.roi import LaneROI
from src.inference.analyze_video import (
    ClipCounts,
    counts_from_frames,
    render_annotated_video,
    synthetic_frames,
)

DEFAULT_SYNTHETIC = {
    "A1": {"car": 40, "motorcycle": 25, "autorickshaw": 10},
    "A2": {"car": 30, "motorcycle": 20},
    "B1": {"car": 20, "bus": 4, "truck": 6},
    "B2": {"car": 15, "motorcycle": 10, "van": 3},
}

_PALETTE = {
    "A": (60, 200, 60),
    "B": (200, 140, 60),
}


def save_upload(name: str, data: bytes) -> Path:
    path = temp_dir() / name
    path.write_bytes(data)
    return path


def probe_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return float(fps) if fps and fps > 0 else 25.0


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


def first_frame(path: Path) -> np.ndarray | None:
    """First frame of a video (or an image read directly), for ROI editing."""
    if Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return cv2.imread(str(path))
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw_rois(image: np.ndarray, rois: list[LaneROI]) -> np.ndarray:
    vis = image.copy()
    for roi in rois:
        pts = np.array(roi.polygon, dtype=np.int32).reshape(-1, 1, 2)
        color = _PALETTE.get(roi.approach, (240, 240, 240))
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        x = int(sum(p[0] for p in roi.polygon) / len(roi.polygon))
        y = int(sum(p[1] for p in roi.polygon) / len(roi.polygon))
        cv2.putText(vis, roi.lane_id, (x - 15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
    return vis


def _canvas_size(rois: list[LaneROI]) -> tuple[int, int]:
    xs = [p[0] for r in rois for p in r.polygon]
    ys = [p[1] for r in rois for p in r.polygon]
    w = int(max(xs) + 40) if xs else 1280
    h = int(max(ys) + 40) if ys else 720
    return max(w, 320), max(h, 240)


def run_on_video(
    detector,
    path: Path,
    rois: list[LaneROI],
    stride: int = 1,
    fps: float | None = None,
    max_keyframes: int = 4,
    save_annotated_path: Path | None = None,
    intersection: Any | None = None,
) -> tuple[ClipCounts, list[np.ndarray]]:
    from src.analysis.hud import HUDOverlayRenderer
    from src.analysis.kinematics import KinematicLaneCounter
    from src.decision.actuator import AdaptiveActuator

    counter = LaneCounter(rois)
    kinematic_counter = KinematicLaneCounter(rois)
    actuator = AdaptiveActuator(intersection) if intersection is not None else None
    hud = HUDOverlayRenderer(rois, intersection=intersection)
    keyframes: list[np.ndarray] = []
    n = 0
    last_idx = 0
    capture_every = max(4, stride * 4)

    writer = None
    if save_annotated_path is not None:
        import imageio_ffmpeg
        cap = cv2.VideoCapture(str(path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 768)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 432)
        in_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        out_fps = max(1.0, in_fps / max(1, stride))
        save_annotated_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio_ffmpeg.write_frames(
            str(save_annotated_path), size=(width, height), fps=out_fps,
            codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            macro_block_size=1, output_params=["-movflags", "+faststart"],
        )
        writer.send(None)

    try:
        for idx, result in detector.track_video(str(path), stride=stride):
            dets = detector.result_to_dicts(result)
            ts = (idx / fps) if fps else float(idx)
            counter.update(dets, timestamp=ts)
            snap = kinematic_counter.update(dets, timestamp=ts)
            ctrl_state = None
            if actuator is not None:
                ctrl_state = actuator.step(ts, snapshot=snap)
            n += 1
            last_idx = idx
            vis = draw_rois(result.plot(), rois)
            vis = hud.render(vis, controller_state=ctrl_state, snapshot=snap, timestamp=ts)
            if len(keyframes) < max_keyframes and (n == 1 or n % capture_every == 0):
                keyframes.append(vis)
            if writer is not None:
                writer.send(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB).tobytes())
    finally:
        if writer is not None:
            writer.close()

    duration = (last_idx / fps) if (fps and fps > 0) else float(n * stride)
    
    # Automatically persist session telemetry & actuation events to SQLite
    try:
        from src.db.database import log_actuation_event, log_telemetry_batch
        inter_id = intersection.id if intersection else "default_intersection"
        telemetry_rows = []
        for snap in kinematic_counter.timeline_history:
            for app, h_pcu in snap.approach_halted_pcu.items():
                f_pcu = snap.approach_flowing_pcu.get(app, 0.0) if hasattr(snap, "approach_flowing_pcu") else 0.0
                q_m = snap.approach_queue_m.get(app, 0.0) if hasattr(snap, "approach_queue_m") else 0.0
                telemetry_rows.append({
                    "intersection_id": inter_id,
                    "timestamp": snap.timestamp,
                    "approach": app,
                    "halted_pcu": h_pcu,
                    "flowing_pcu": f_pcu,
                    "queue_length_m": q_m,
                })
        log_telemetry_batch(telemetry_rows)
        if actuator is not None:
            for e in actuator.events_log:
                log_actuation_event(
                    intersection_id=inter_id,
                    timestamp=e.timestamp,
                    from_phase=e.from_phase,
                    to_phase=e.to_phase,
                    aspect=e.aspect.value.upper() if hasattr(e.aspect, "value") else str(e.aspect),
                    reason=e.reason.name if hasattr(e.reason, "name") else str(e.reason),
                    message=e.message,
                    waiting_queue_pcu=e.waiting_queue_pcu,
                )
    except Exception:
        pass

    counts = ClipCounts(
        lane_counts=counter.counts_by_lane(),
        approach_counts=counter.counts_by_approach(),
        processed_frames=n,
        duration_s=duration,
        kinematic_snapshots=kinematic_counter.timeline_history,
        actuator_history=actuator.state_history if actuator is not None else [],
        actuator_events=actuator.events_log if actuator is not None else [],
    )
    return counts, keyframes


def run_synthetic(
    rois: list[LaneROI],
    spec: dict[str, dict[str, int]] | None = None,
    fps: float = 4.0,
) -> tuple[ClipCounts, list[np.ndarray]]:
    spec = spec or DEFAULT_SYNTHETIC
    frames = synthetic_frames(rois, spec, fps=fps)
    counts = counts_from_frames(frames, rois, fps=fps)
    w, h = _canvas_size(rois)
    canvas = np.full((h, w, 3), 45, dtype=np.uint8)
    return counts, [draw_rois(canvas, rois)]


def render_annotated(detector, src: Path, out: Path, rois=None, stride: int = 1, progress=None):
    """Render a video with YOLO boxes drawn on every frame (H.264, playable)."""
    return render_annotated_video(detector, src, out, rois=rois, stride=stride, progress=progress)
