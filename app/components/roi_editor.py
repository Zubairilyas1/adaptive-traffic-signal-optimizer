"""ROI editor: preset selection + editable polygons with a visual overlay."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app.components.state import get_rois
from app.components.video import draw_rois, first_frame
from app.utils import list_roi_presets, list_sample_clips, rois_from_json
from src.analysis.roi import LaneROI
from src import config


def _to_df(rois: list[LaneROI]) -> pd.DataFrame:
    rows = []
    for r in rois:
        poly = r.polygon[:4]
        row = {"lane_id": r.lane_id, "approach": r.approach}
        for i in range(4):
            x, y = poly[i] if i < len(poly) else (0.0, 0.0)
            row[f"x{i + 1}"] = int(x)
            row[f"y{i + 1}"] = int(y)
        rows.append(row)
    return pd.DataFrame(rows)


def _from_df(df: pd.DataFrame, base: dict[str, LaneROI]) -> list[LaneROI]:
    rois: list[LaneROI] = []
    for _, row in df.iterrows():
        lane_id = str(row["lane_id"]).strip()
        if not lane_id:
            continue
        polygon = [(float(row[f"x{i + 1}"]), float(row[f"y{i + 1}"])) for i in range(4)]
        b = base.get(lane_id)
        rois.append(LaneROI(
            lane_id=lane_id,
            approach=str(row["approach"]).strip(),
            polygon=polygon,
            buffer_px=b.buffer_px if b else 3.0,
            hysteresis_frames=b.hysteresis_frames if b else 3,
        ))
    return rois


def _background() -> np.ndarray:
    """A frame to draw ROIs on: uploaded image, sample clip frame, or blank."""
    up = st.file_uploader("Background image (optional) — a frame of your footage",
                          type=["jpg", "jpeg", "png"], key="roi_bg")
    if up is not None:
        arr = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            st.session_state["roi_bg_size"] = (arr.shape[1], arr.shape[0])
            return arr

    clips = list_sample_clips()
    if clips:
        names = ["(blank canvas)"] + [c.name for c in clips]
        choice = st.selectbox("Background frame from clip", names, key="roi_bg_clip")
        if choice != "(blank canvas)":
            clip = config.SAMPLES_DIR / choice
            frame = first_frame(clip)
            if frame is not None:
                st.session_state["roi_bg_size"] = (frame.shape[1], frame.shape[0])
                return frame

    w, h = st.session_state.get("roi_bg_size", (1280, 720))
    return np.full((h, w, 3), 45, dtype=np.uint8)


TOPOLOGY_INTERSECTION_MAP = {
    "preset_single_road.json": "single_road.json",
    "preset_twoway_corridor.json": "twoway_corridor.json",
    "preset_t_junction.json": "t_junction.json",
    "preset_fourway_crossroad.json": "fourway_crossroad.json",
}


def render_roi_editor() -> tuple[list[LaneROI], str]:
    presets = list_roi_presets()
    if not presets:
        st.error("No ROI presets found under configs/rois/.")
        return [], ""

    names = [p.name for p in presets]
    choice = st.selectbox(
        "Road Topology / Lane Preset",
        names,
        index=0,
        key="roi_preset",
        help="Select the road geometry matching your video (Single Road, Two-Way, T-Junction, or 4-Way).",
    )
    path = presets[names.index(choice)]
    base_rois = get_rois(str(path))

    # Auto-link matching intersection config if available
    matched_inter = TOPOLOGY_INTERSECTION_MAP.get(choice)
    if matched_inter:
        target_inter = config.ROOT / "configs" / "intersections" / matched_inter
        if target_inter.exists():
            st.session_state["intersection_path"] = str(target_inter)

    st.caption("Each lane is a 4-vertex polygon in **frame pixel** coordinates.")
    df = _to_df(base_rois)
    edited = st.data_editor(
        df, width="stretch", num_rows="dynamic", key="roi_editor",
    )

    with st.expander("Load a custom ROI JSON"):
        upload = st.file_uploader("ROI .json", type="json", key="roi_upload")
        if upload is not None:
            try:
                edited = _to_df(rois_from_json(upload.read().decode("utf-8")))
                st.success("Custom ROI loaded.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not parse ROI JSON: {exc}")

    rois = _from_df(edited, {r.lane_id: r for r in base_rois})

    st.markdown("**ROI preview** (polygons drawn on the background frame)")
    bg = _background()
    overlay = draw_rois(bg, rois)
    st.image(overlay, channels="BGR", width="stretch")

    return rois, str(path)
