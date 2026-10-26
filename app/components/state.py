"""Cached loaders and session state shared across pages."""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src + app to sys.path)

import streamlit as st

from src import config
from src.analysis.roi import LaneROI, load_rois
from src.decision.intersection import Intersection, load_intersection
from src.models import VehicleDetector, load_detector


import logging

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading detector ...")
def get_detector(weights: str) -> VehicleDetector:
    det = load_detector(weights=weights)
    # Warm-up pass to initialize PyTorch/CUDA contexts
    try:
        import numpy as np
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        det.predict(dummy)
    except Exception as exc:
        logger.warning(f"Detector warm-up pass failed (non-fatal): {exc}")
    return det


@st.cache_data(show_spinner=False)
def get_intersection(path: str) -> Intersection:
    return load_intersection(path)


@st.cache_data(show_spinner=False)
def get_rois(path: str) -> list[LaneROI]:
    return load_rois(path)


def init_state() -> None:
    st.session_state.setdefault("counts", None)
    st.session_state.setdefault("plan", None)
    st.session_state.setdefault("comparison", None)
    st.session_state.setdefault("flows", None)
    st.session_state.setdefault("keyframes", [])
    st.session_state.setdefault("last_clip", None)
    st.session_state.setdefault("rois", None)
    st.session_state.setdefault("intersection_path", str(config.ROOT / "configs" / "intersections" / "default.json"))
    st.session_state.setdefault("roi_path", str(config.ROOT / "configs" / "rois" / "example.json"))
    st.session_state.setdefault("detector_weights", str(config.DETECTOR_WEIGHTS))
