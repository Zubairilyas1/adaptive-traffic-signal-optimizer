"""Heads-Up Display (HUD) Video Overlay Engine.

Stamps real-time commercial ITS graphics onto video frames:
1. Live 3-Aspect Traffic Signal Badges (🟢 Green, 🟡 Yellow, 🔴 Red) per approach.
2. Real-Time AI Autonomous Decision Ticker Banner with dynamic triggers
   (Gap-Out, Max-Out, Dilemma Zone, Rest-in-Green, Yellow/All-Red Clearance).
3. Kinematic motion badges (Halted vs Flowing speed tags) on detected vehicles.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.analysis.kinematics import IntersectionSnapshot, MotionState
from src.analysis.roi import LaneROI
from src.decision.actuator import ControllerState, DecisionReason, SignalAspect


class HUDOverlayRenderer:
    """Renders professional ITS traffic signal overlays directly onto video frames."""

    def __init__(
        self,
        rois: Sequence[LaneROI],
        intersection: Optional[Any] = None,
    ):
        self.rois = list(rois)
        self.intersection = intersection

        # Map approach -> list of rois
        self.approach_rois: dict[str, list[LaneROI]] = {}
        for r in self.rois:
            self.approach_rois.setdefault(r.approach, []).append(r)

    def render(
        self,
        frame: np.ndarray,
        controller_state: Optional[ControllerState] = None,
        snapshot: Optional[IntersectionSnapshot] = None,
        timestamp: float = 0.0,
    ) -> np.ndarray:
        """Render signal badges, AI decision ticker, and HUD metrics onto the frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. Draw Decision Ticker Banner at the top
        vis = self._draw_decision_ticker(vis, controller_state, snapshot, timestamp, w, h)

        # 2. Draw 3-Aspect Traffic Signal Light Badges for each approach
        vis = self._draw_traffic_signals(vis, controller_state, w, h)

        return vis

    def _draw_decision_ticker(
        self,
        vis: np.ndarray,
        controller_state: Optional[ControllerState],
        snapshot: Optional[IntersectionSnapshot],
        timestamp: float,
        w: int,
        h: int,
    ) -> np.ndarray:
        """Render a translucent dark HUD banner at the top showing second-by-second AI actions."""
        banner_h = max(55, int(h * 0.10))
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 18, 22), -1)
        cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)
        cv2.line(vis, (0, banner_h), (w, banner_h), (50, 180, 255), 2)

        # Time & Active Phase Badge
        ts_str = f"T={timestamp:05.1f}s"
        cv2.putText(vis, ts_str, (12, int(banner_h * 0.40)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if controller_state is not None:
            active_p = controller_state.active_phase_id
            aspect = controller_state.active_aspect.value.upper()
            aspect_color = (0, 255, 0) if aspect == "GREEN" else ((0, 215, 255) if aspect == "YELLOW" else (0, 0, 255))

            # Phase pill
            pill_text = f"PHASE {active_p}: {aspect}"
            cv2.putText(vis, pill_text, (120, int(banner_h * 0.40)), cv2.FONT_HERSHEY_DUPLEX, 0.55, aspect_color, 1, cv2.LINE_AA)

            # Decision Reason Ticker
            reason_text = f">> AI DECISION: {controller_state.reason_text}"
            reason_color = (100, 255, 120) if "green" in reason_text.lower() else (
                (0, 220, 255) if "gap-out" in reason_text.lower() or "yellow" in reason_text.lower() else (220, 220, 255)
            )
            cv2.putText(vis, reason_text, (12, int(banner_h * 0.82)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, reason_color, 1, cv2.LINE_AA)

            # Queue metric on right
            q_text = f"Active Q: {controller_state.active_queue_pcu:.1f}u | Cross Q: {controller_state.conflicting_queue_pcu:.1f}u"
            (tw, _), _ = cv2.getTextSize(q_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(vis, q_text, (max(12, w - tw - 15), int(banner_h * 0.40)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, "Adaptive Actuator: Standby", (120, int(banner_h * 0.60)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        return vis

    def _draw_traffic_signals(
        self,
        vis: np.ndarray,
        controller_state: Optional[ControllerState],
        w: int,
        h: int,
    ) -> np.ndarray:
        """Render 3-aspect traffic light indicators (Red/Yellow/Green) for each approach."""
        if controller_state is None or not controller_state.phase_displays:
            return vis

        # Render signal boxes on top right or above approach ROIs
        start_x = w - 110
        start_y = max(65, int(h * 0.12))

        for idx, (p_id, display) in enumerate(controller_state.phase_displays.items()):
            box_x = start_x - (idx * 105)
            box_y = start_y
            box_w = 95
            box_h = 75

            # Dark housing
            cv2.rectangle(vis, (box_x, box_y), (box_x + box_w, box_y + box_h), (25, 25, 25), -1)
            cv2.rectangle(vis, (box_x, box_y), (box_x + box_w, box_y + box_h), (120, 120, 120), 1)

            # Title
            label = f"{display.approach} ({p_id})"
            cv2.putText(vis, label, (box_x + 5, box_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

            # Three lamps: Red (left), Yellow (center), Green (right)
            lamp_y = box_y + 35
            radius = 10

            is_red = display.aspect == SignalAspect.RED
            is_yellow = display.aspect == SignalAspect.YELLOW
            is_green = display.aspect == SignalAspect.GREEN

            # Red Lamp
            red_col = (0, 0, 255) if is_red else (0, 0, 60)
            cv2.circle(vis, (box_x + 20, lamp_y), radius, red_col, -1)
            cv2.circle(vis, (box_x + 20, lamp_y), radius, (80, 80, 80), 1)

            # Yellow Lamp
            yel_col = (0, 215, 255) if is_yellow else (0, 50, 60)
            cv2.circle(vis, (box_x + 48, lamp_y), radius, yel_col, -1)
            cv2.circle(vis, (box_x + 48, lamp_y), radius, (80, 80, 80), 1)

            # Green Lamp
            grn_col = (0, 255, 0) if is_green else (0, 60, 0)
            cv2.circle(vis, (box_x + 75, lamp_y), radius, grn_col, -1)
            cv2.circle(vis, (box_x + 75, lamp_y), radius, (80, 80, 80), 1)

            # Elapsed / Remaining text
            if is_green:
                dur_text = f"Green: {display.elapsed_in_phase_s:.1f}s"
                text_col = (0, 255, 0)
            elif is_yellow:
                dur_text = f"Yellow: {display.elapsed_in_phase_s:.1f}s"
                text_col = (0, 215, 255)
            else:
                dur_text = "RED (Stopped)"
                text_col = (100, 100, 255)

            cv2.putText(vis, dur_text, (box_x + 5, box_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_col, 1, cv2.LINE_AA)

        return vis
