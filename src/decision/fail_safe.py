"""NEMA TS2 compliant safety fail-safe watchdog.

Ensures real-world intersection safety by monitoring camera health,
frame delivery heartbeats, and image quality:
- Camera Feed Drop Watchdog (> 2.0s without frames triggers failsafe)
- Image Occlusion / Blackout Guard (detects covered lens or glitch)
- Automatic fallback to safe Pre-Timed Fixed-Cycle operation
- Safe clearance intervals (no instant snaps / yellow-trap prevention)
- Auto-recovery when healthy perception stream resumes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Optional

import numpy as np

from src import config
from src.decision.actuator import (
    AdaptiveActuator,
    ControllerState,
    DecisionReason,
    SignalAspect,
    SignalPhaseDisplay,
)
from src.decision.intersection import Intersection

logger = logging.getLogger(__name__)


class HealthState(Enum):
    """Operational health status of the traffic perception pipeline."""

    NORMAL = "normal"  # Healthy, live adaptive AI actuation
    DEGRADED = "degraded"  # Intermittent frame latency, monitoring
    FAILSAFE_ACTIVE = "failsafe_active"  # Camera lost/occluded: pre-timed backup active


class FailureReason(Enum):
    """Specific cause of a safety failsafe transition."""

    NONE = "Normal operation"
    FEED_TIMEOUT = "Video stream heartbeat timeout (>2.0s)"
    LENS_OCCLUDED = "Camera lens occluded or black screen detected"
    FRAME_CORRUPT = "Corrupted or zero-dimension frame payload"


@dataclass
class FailSafeEvent:
    """Audit record of a failsafe transition."""

    timestamp: float
    state: HealthState
    reason: FailureReason
    message: str


class NEMAFailSafeWatchdog:
    """Supervisory safety monitor ensuring fail-operational signal reliability."""

    def __init__(
        self,
        timeout_s: float = 2.0,
        min_variance: float = 5.0,
        recovery_window_s: float = 3.0,
        fallback_green_s: float = 25.0,
        yellow_s: float = 3.0,
        all_red_s: float = 1.0,
    ) -> None:
        self.timeout_s = timeout_s
        self.min_variance = min_variance
        self.recovery_window_s = recovery_window_s
        self.fallback_green_s = fallback_green_s
        self.yellow_s = yellow_s
        self.all_red_s = all_red_s

        self.state: HealthState = HealthState.NORMAL
        self.last_failure_reason: FailureReason = FailureReason.NONE
        self.last_heartbeat_time: float | None = None
        self.healthy_since_time: float | None = None
        self.failsafe_entered_time: float | None = None
        self.event_log: list[FailSafeEvent] = []

        # Internal pre-timed fallback controller timing state
        self._fallback_phase_idx: int = 0
        self._fallback_aspect: SignalAspect = SignalAspect.GREEN
        self._fallback_phase_start: float | None = None

    def feed_heartbeat(
        self,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> bool:
        """Register arrival of a new video frame and test image variance."""
        # Check frame integrity if provided
        if frame is not None:
            if frame.size == 0 or len(frame.shape) < 2:
                self._record_failure(timestamp, FailureReason.FRAME_CORRUPT, "Frame is empty or zero-sized")
                return False

            # Compute variance to catch black screens, total lens blur, or static disconnect frames
            var = float(np.var(frame))
            if var < self.min_variance:
                self._record_failure(
                    timestamp,
                    FailureReason.LENS_OCCLUDED,
                    f"Frame variance too low ({var:.1f} < {self.min_variance}): camera occluded",
                )
                return False

        # Healthy frame received
        self.last_heartbeat_time = timestamp
        if self.healthy_since_time is None:
            self.healthy_since_time = timestamp
        return True

    def _record_failure(
        self,
        timestamp: float,
        reason: FailureReason,
        message: str,
    ) -> None:
        self.last_failure_reason = reason
        self.healthy_since_time = None
        if self.state != HealthState.FAILSAFE_ACTIVE:
            self.state = HealthState.FAILSAFE_ACTIVE
            self.failsafe_entered_time = timestamp
            ev = FailSafeEvent(timestamp, self.state, reason, message)
            self.event_log.append(ev)
            logger.warning(f"NEMA Safety Watchdog triggered: {message}")

    def evaluate(self, timestamp: float) -> tuple[HealthState, FailureReason, str]:
        """Evaluate heartbeat freshness and return current health state."""
        # In failsafe mode: check if enough consecutive healthy seconds have elapsed to recover
        if self.state == HealthState.FAILSAFE_ACTIVE:
            healthy_dur = (
                timestamp - self.healthy_since_time
                if self.healthy_since_time is not None
                else 0.0
            )
            if healthy_dur >= self.recovery_window_s:
                self.state = HealthState.NORMAL
                self.last_failure_reason = FailureReason.NONE
                ev = FailSafeEvent(
                    timestamp,
                    self.state,
                    FailureReason.NONE,
                    f"Perception stream recovered: healthy for {healthy_dur:.1f}s",
                )
                self.event_log.append(ev)
                return HealthState.NORMAL, FailureReason.NONE, "🟢 Feed recovered: Adaptive AI restored."
            else:
                return (
                    HealthState.FAILSAFE_ACTIVE,
                    self.last_failure_reason,
                    f"⚠️ FAIL-SAFE ENGAGED ({self.last_failure_reason.value}): Enforcing pre-timed backup.",
                )

        # In NORMAL mode: check for feed heartbeat timeout
        if self.last_heartbeat_time is None:
            time_since_hb = float("inf")
        else:
            time_since_hb = timestamp - self.last_heartbeat_time

        if time_since_hb > self.timeout_s:
            self._record_failure(
                timestamp,
                FailureReason.FEED_TIMEOUT,
                f"Video heartbeat delayed by {time_since_hb:.1f}s (> {self.timeout_s}s limit)",
            )
            msg = f"⚠️ FAIL-SAFE ENGAGED: Camera feed lost for {time_since_hb:.1f}s. Enforcing pre-timed backup."
            return HealthState.FAILSAFE_ACTIVE, FailureReason.FEED_TIMEOUT, msg

        return HealthState.NORMAL, FailureReason.NONE, "🟢 Perception pipeline healthy."

    def step_fallback(
        self,
        timestamp: float,
        intersection: Intersection,
    ) -> ControllerState:
        """Advance a safe, pre-timed fixed-cycle state machine when failsafe is engaged."""
        phases = [p.id for p in intersection.phases] or ["P1", "P2"]
        num_phases = len(phases)

        if self._fallback_phase_start is None:
            self._fallback_phase_start = timestamp

        elapsed = max(0.0, timestamp - self._fallback_phase_start)
        active_phase_id = phases[self._fallback_phase_idx]

        # Phase timing: Green -> Yellow -> All-Red
        if self._fallback_aspect == SignalAspect.GREEN:
            if elapsed >= self.fallback_green_s:
                self._fallback_aspect = SignalAspect.YELLOW
                self._fallback_phase_start = timestamp
                elapsed = 0.0
        elif self._fallback_aspect == SignalAspect.YELLOW:
            if elapsed >= self.yellow_s:
                self._fallback_aspect = SignalAspect.RED
                self._fallback_phase_start = timestamp
                elapsed = 0.0
        elif self._fallback_aspect == SignalAspect.RED:
            if elapsed >= self.all_red_s:
                # Transition to next phase
                self._fallback_phase_idx = (self._fallback_phase_idx + 1) % num_phases
                self._fallback_aspect = SignalAspect.GREEN
                self._fallback_phase_start = timestamp
                elapsed = 0.0
                active_phase_id = phases[self._fallback_phase_idx]

        active_app = active_phase_id
        for p in intersection.phases:
            if p.id == active_phase_id:
                active_app = getattr(p, "approach", active_phase_id)

        # Build phase displays
        displays = {}
        for p in intersection.phases:
            p_app = getattr(p, "approach", p.id)
            if p.id == active_phase_id:
                asp = self._fallback_aspect
                rem = max(0.0, self.fallback_green_s - elapsed) if asp == SignalAspect.GREEN else 0.0
            else:
                asp = SignalAspect.RED
                rem = 0.0
            displays[p.id] = SignalPhaseDisplay(
                phase_id=p.id,
                approach=p_app,
                aspect=asp,
                elapsed_in_phase_s=elapsed,
                time_remaining_s=rem,
            )

        return ControllerState(
            timestamp=timestamp,
            active_phase_id=active_phase_id,
            active_aspect=self._fallback_aspect,
            active_reason=DecisionReason.MIN_GREEN,
            reason_text=f"NEMA SAFETY FAIL-SAFE: Enforcing pre-timed plan ({active_phase_id} {self._fallback_aspect.value.upper()})",
            phase_displays=displays,
            active_queue_pcu=0.0,
            conflicting_queue_pcu=0.0,
        )

    @property
    def is_failsafe_active(self) -> bool:
        return self.state == HealthState.FAILSAFE_ACTIVE

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "health_state": self.state.value,
            "failure_reason": self.last_failure_reason.value,
            "timeout_limit_s": self.timeout_s,
            "total_failsafe_events": len(self.event_log),
        }
