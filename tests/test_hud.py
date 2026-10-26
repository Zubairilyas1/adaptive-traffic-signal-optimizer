"""Unit tests for Heads-Up Display (HUD) overlay engine and actuation timeline charts."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.analysis.hud import HUDOverlayRenderer
from src.analysis.kinematics import IntersectionSnapshot, LaneQueueSnapshot
from src.analysis.roi import LaneROI
from app.components.charts import actuation_timeline_chart, compute_wasted_green_saved
from src.decision.actuator import (
    ActuationEvent,
    AdaptiveActuator,
    ControllerState,
    DecisionReason,
    SignalAspect,
    SignalPhaseDisplay,
)
from src.decision.intersection import load_intersection
from src.inference.analyze_video import ClipCounts


@pytest.fixture
def sample_rois():
    return [
        LaneROI(lane_id="North_L1", approach="North", polygon=[(100, 100), (200, 100), (200, 300), (100, 300)]),
        LaneROI(lane_id="East_L1", approach="East", polygon=[(300, 100), (400, 100), (400, 300), (300, 300)]),
    ]


@pytest.fixture
def dummy_frame():
    return np.zeros((432, 768, 3), dtype=np.uint8)


def test_hud_renderer_blank_state(sample_rois, dummy_frame):
    renderer = HUDOverlayRenderer(sample_rois)
    out = renderer.render(dummy_frame, timestamp=0.0)

    assert out.shape == dummy_frame.shape
    # Check that overlay banner was drawn (top rows should no longer be purely black)
    assert np.any(out[10:40, :, :] > 0)


def test_hud_renderer_with_controller_state(sample_rois, dummy_frame):
    renderer = HUDOverlayRenderer(sample_rois)

    displays = {
        "P1": SignalPhaseDisplay(
            phase_id="P1",
            approach="North",
            aspect=SignalAspect.GREEN,
            elapsed_in_phase_s=12.5,
            time_remaining_s=17.5,
        ),
        "P2": SignalPhaseDisplay(
            phase_id="P2",
            approach="East",
            aspect=SignalAspect.RED,
            elapsed_in_phase_s=0.0,
            time_remaining_s=0.0,
        ),
    }

    state = ControllerState(
        timestamp=12.5,
        active_phase_id="P1",
        active_aspect=SignalAspect.GREEN,
        active_reason=DecisionReason.EXTEND_HEADWAY,
        reason_text="Phase P1 green active",
        phase_displays=displays,
        active_queue_pcu=3.5,
        conflicting_queue_pcu=6.0,
    )

    out = renderer.render(dummy_frame, controller_state=state, timestamp=12.5)
    assert out.shape == dummy_frame.shape
    # Signal box area on the top-right should have active green and red pixels
    assert np.any(out[:, :, 1] > 200)  # Green channel active


def test_actuation_timeline_chart_and_wasted_green():
    snaps = [
        IntersectionSnapshot(timestamp=0.0, approach_halted_pcu={"North": 0.0, "East": 5.0}),
        IntersectionSnapshot(timestamp=5.0, approach_halted_pcu={"North": 1.0, "East": 6.5}),
        IntersectionSnapshot(timestamp=10.0, approach_halted_pcu={"North": 0.0, "East": 8.0}),
    ]

    events = [
        ActuationEvent(
            timestamp=10.0,
            from_phase="P1",
            to_phase="P2",
            aspect=SignalAspect.YELLOW,
            reason=DecisionReason.GAP_OUT,
            message="Gap-out triggered",
            waiting_queue_pcu=8.0,
        )
    ]

    counts = ClipCounts(
        kinematic_snapshots=snaps,
        actuator_events=events,
    )

    chart = actuation_timeline_chart(counts)
    assert chart is not None

    saved = compute_wasted_green_saved(counts)
    assert saved == 8.5
