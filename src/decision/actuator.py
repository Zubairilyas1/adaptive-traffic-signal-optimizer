"""Closed-loop real-time adaptive traffic signal actuator.

Implements NEMA TS2 style dynamic actuation:
- Second-by-second phase state machine
- Dilemma-zone vehicle extension
- Gap-out early termination (no wasted green)
- Rest-in-green for uninterrupted flow
- Max-out safety clamps and clearance intervals
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from src import config
from src.analysis.kinematics import IntersectionSnapshot
from src.decision.intersection import Intersection


class SignalAspect(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class DecisionReason(Enum):
    MIN_GREEN = "Holding minimum safe green (7.0s)"
    REST_IN_GREEN = "Resting in green (zero conflicting demand)"
    DILEMMA_ZONE = "Extending green: approaching vehicle clearing dilemma zone"
    EXTEND_HEADWAY = "Extending green: vehicles actively discharging"
    GAP_OUT = "GAP-OUT: Green lane cleared (Headway > 3.0s) -> Yielding to waiting queue"
    MAX_OUT = "MAX-OUT: Maximum green reached -> Forcing phase transition"
    YELLOW_CLEARANCE = "Yellow clearance interval (3.0s)"
    ALL_RED_CLEARANCE = "All-red clearance interval (1.0s)"
    PHASE_START = "Green phase activated"


@dataclass
class ActuationEvent:
    timestamp: float
    from_phase: str
    to_phase: str
    aspect: SignalAspect
    reason: DecisionReason
    message: str
    waiting_queue_pcu: float


@dataclass
class SignalPhaseDisplay:
    phase_id: str
    approach: str
    aspect: SignalAspect
    elapsed_in_phase_s: float
    time_remaining_s: float


@dataclass
class ControllerState:
    timestamp: float
    active_phase_id: str
    active_aspect: SignalAspect
    active_reason: DecisionReason
    reason_text: str
    phase_displays: dict[str, SignalPhaseDisplay]
    active_queue_pcu: float
    conflicting_queue_pcu: float


class AdaptiveActuator:
    """Real-time second-by-second closed-loop signal actuator."""

    def __init__(
        self,
        intersection: Intersection,
        min_green: float = config.MIN_GREEN,
        max_green: float = config.MAX_GREEN,
        yellow_time: float = 3.0,
        all_red_time: float = 1.0,
        gap_threshold_s: float = 3.0,
    ):
        self.intersection = intersection
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time
        self.gap_threshold_s = gap_threshold_s

        self.phases = [p.id for p in self.intersection.phases]
        if not self.phases:
            self.phases = ["P1"]

        self.current_phase_idx: int = 0
        self.next_phase_idx: int = 0
        self.current_aspect: SignalAspect = SignalAspect.GREEN
        self.phase_start_time: float = 0.0
        self.last_vehicle_seen_time: float = 0.0

        self.events_log: list[ActuationEvent] = []
        self.state_history: list[ControllerState] = []

    @property
    def active_phase_id(self) -> str:
        return self.phases[self.current_phase_idx]

    def _get_approach_for_phase(self, phase_id: str) -> str:
        for p in self.intersection.phases:
            if p.id == phase_id:
                return getattr(p, "approach", phase_id)
        return phase_id

    def step(self, timestamp: float, snapshot: Optional[IntersectionSnapshot] = None) -> ControllerState:
        """Evaluate real-time intersection traffic and advance actuation state machine."""
        elapsed = max(0.0, timestamp - self.phase_start_time)
        active_phase = self.active_phase_id
        active_app = self._get_approach_for_phase(active_phase)

        # Retrieve waiting queue on conflicting red approaches
        conflicting_queue = 0.0
        active_lane_speed = 0.0
        active_halted_pcu = 0.0

        if snapshot is not None:
            active_halted_pcu = snapshot.approach_halted_pcu.get(active_app, 0.0)
            for app, pcu in snapshot.approach_halted_pcu.items():
                if app != active_app:
                    conflicting_queue += pcu

            # Check if vehicles are moving through active green approach
            for lane_id, l_snap in snapshot.lane_snapshots.items():
                if l_snap.approach == active_app:
                    if l_snap.flowing_count > 0:
                        self.last_vehicle_seen_time = timestamp
                        active_lane_speed = max(active_lane_speed, l_snap.avg_speed_km_h)

        headway_gap = timestamp - self.last_vehicle_seen_time
        reason = DecisionReason.EXTEND_HEADWAY
        reason_msg = f"Phase {active_phase} green active"

        # --- STATE 1: GREEN PHASE ---
        if self.current_aspect == SignalAspect.GREEN:
            # Rule 1: Minimum Safe Green guarantee
            if elapsed < self.min_green:
                reason = DecisionReason.MIN_GREEN
                reason_msg = f"Holding minimum safe green ({self.min_green - elapsed:.1f}s remaining)"

            # Rule 2: Rest-in-Green (No waiting traffic on red phases)
            elif conflicting_queue <= 0.0:
                reason = DecisionReason.REST_IN_GREEN
                reason_msg = f"Resting in green on {active_app}: zero conflicting demand"

            # Rule 3: Dilemma Zone Protection (Fast car approaching stopline)
            elif active_lane_speed > 30.0 and headway_gap < 2.5:
                reason = DecisionReason.DILEMMA_ZONE
                reason_msg = f"Extending green: vehicle clearing dilemma zone at {active_lane_speed:.0f} km/h"

            # Rule 4: Gap-Out (Lane cleared -> Cut green early!)
            elif headway_gap >= self.gap_threshold_s:
                reason = DecisionReason.GAP_OUT
                reason_msg = f"GAP-OUT: Lane cleared (gap {headway_gap:.1f}s) -> Yielding to {conflicting_queue:.1f} PCU waiting queue!"
                self._transition_to(SignalAspect.YELLOW, timestamp, reason, reason_msg, conflicting_queue)

            # Rule 5: Max-Out (Maximum allowable green elapsed)
            elif elapsed >= self.max_green:
                reason = DecisionReason.MAX_OUT
                reason_msg = f"MAX-OUT: Max green ({self.max_green:.0f}s) reached -> Forcing switch to waiting queue"
                self._transition_to(SignalAspect.YELLOW, timestamp, reason, reason_msg, conflicting_queue)

        # --- STATE 2: YELLOW CLEARANCE ---
        elif self.current_aspect == SignalAspect.YELLOW:
            reason = DecisionReason.YELLOW_CLEARANCE
            rem_yellow = max(0.0, self.yellow_time - elapsed)
            reason_msg = f"Yellow clearance: {rem_yellow:.1f}s remaining"

            if elapsed >= self.yellow_time:
                self._transition_to(SignalAspect.RED, timestamp, DecisionReason.ALL_RED_CLEARANCE, "All-red junction clearance", conflicting_queue)

        # --- STATE 3: ALL-RED CLEARANCE ---
        elif self.current_aspect == SignalAspect.RED:
            reason = DecisionReason.ALL_RED_CLEARANCE
            rem_red = max(0.0, self.all_red_time - elapsed)
            reason_msg = f"All-red junction clearance: {rem_red:.1f}s remaining"

            if elapsed >= self.all_red_time:
                # Advance to next phase
                self.current_phase_idx = (self.current_phase_idx + 1) % len(self.phases)
                next_p = self.active_phase_id
                self._transition_to(SignalAspect.GREEN, timestamp, DecisionReason.PHASE_START, f"Green phase activated for {next_p}", conflicting_queue)

        # Build full controller state display for UI
        displays: dict[str, SignalPhaseDisplay] = {}
        for idx, p_id in enumerate(self.phases):
            app = self._get_approach_for_phase(p_id)
            if idx == self.current_phase_idx:
                aspect = self.current_aspect
                rem = max(0.0, self.max_green - elapsed) if aspect == SignalAspect.GREEN else 0.0
            else:
                aspect = SignalAspect.RED
                rem = 0.0

            displays[p_id] = SignalPhaseDisplay(
                phase_id=p_id,
                approach=app,
                aspect=aspect,
                elapsed_in_phase_s=elapsed if idx == self.current_phase_idx else 0.0,
                time_remaining_s=rem,
            )

        state = ControllerState(
            timestamp=timestamp,
            active_phase_id=self.active_phase_id,
            active_aspect=self.current_aspect,
            active_reason=reason,
            reason_text=reason_msg,
            phase_displays=displays,
            active_queue_pcu=active_halted_pcu,
            conflicting_queue_pcu=conflicting_queue,
        )
        self.state_history.append(state)
        return state

    def _transition_to(
        self,
        new_aspect: SignalAspect,
        timestamp: float,
        reason: DecisionReason,
        message: str,
        conflicting_queue: float,
    ) -> None:
        event = ActuationEvent(
            timestamp=timestamp,
            from_phase=self.active_phase_id,
            to_phase=self.active_phase_id,
            aspect=new_aspect,
            reason=reason,
            message=message,
            waiting_queue_pcu=conflicting_queue,
        )
        self.events_log.append(event)
        self.current_aspect = new_aspect
        self.phase_start_time = timestamp
        self.last_vehicle_seen_time = timestamp
