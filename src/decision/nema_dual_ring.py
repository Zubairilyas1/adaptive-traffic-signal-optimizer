"""NEMA TS2 Dual-Ring 8-Phase Barrier Controller.

Implements the standard North American and international NEMA TS2 / Caltrans 2070
dual-ring barrier state machine for 4-leg arterial intersections with dedicated
protected left-turn phases.

Standard NEMA Phase Mapping:
  Ring 1:
    Phase 1: Southbound Left-Turn (Protected)
    Phase 2: Northbound Through + Right
    --- BARRIER 1 ---
    Phase 3: Westbound Left-Turn (Protected)
    Phase 4: Eastbound Through + Right
    --- BARRIER 2 ---
  Ring 2:
    Phase 5: Northbound Left-Turn (Protected)
    Phase 6: Southbound Through + Right
    --- BARRIER 1 ---
    Phase 7: Eastbound Left-Turn (Protected)
    Phase 8: Westbound Through + Right
    --- BARRIER 2 ---

Safety Invariants:
1. Concurrency: Phases in Ring 1 and Ring 2 within the same stage can run concurrently.
2. Barrier Crossing: Neither ring may cross a barrier until BOTH rings have completed
   their clearance intervals (Yellow Change + Red Clearance).
3. Non-conflicting pairs:
   Stage 1 (Major N-S): (1,5), (1,6), (2,5), (2,6)
   Stage 2 (Minor E-W): (3,7), (3,8), (4,7), (4,8)
4. Cross-stage pairs are strictly forbidden (e.g., (1,3), (2,4) cannot occur).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class NEMAPhase(IntEnum):
    """The standard 8 vehicular phases defined in NEMA TS2."""
    P1_SB_LEFT = 1
    P2_NB_THROUGH = 2
    P3_WB_LEFT = 3
    P4_EB_THROUGH = 4
    P5_NB_LEFT = 5
    P6_SB_THROUGH = 6
    P7_EB_LEFT = 7
    P8_WB_THROUGH = 8


class PhaseInterval(Enum):
    """Interval state of an individual phase."""
    GREEN = "GREEN"
    YELLOW_CHANGE = "YELLOW_CHANGE"
    ALL_RED_CLEARANCE = "ALL_RED_CLEARANCE"
    RED_HOLD = "RED_HOLD"  # Waiting for partner ring at barrier or red rest


class RingBarrierStage(Enum):
    """Barrier stages separating major street from minor street."""
    STAGE_1_MAJOR_NS = "STAGE_1_MAJOR_NS"
    STAGE_2_MINOR_EW = "STAGE_2_MINOR_EW"


# Ring definitions
RING_1_PHASES: frozenset[NEMAPhase] = frozenset({
    NEMAPhase.P1_SB_LEFT,
    NEMAPhase.P2_NB_THROUGH,
    NEMAPhase.P3_WB_LEFT,
    NEMAPhase.P4_EB_THROUGH,
})

RING_2_PHASES: frozenset[NEMAPhase] = frozenset({
    NEMAPhase.P5_NB_LEFT,
    NEMAPhase.P6_SB_THROUGH,
    NEMAPhase.P7_EB_LEFT,
    NEMAPhase.P8_WB_THROUGH,
})

# Stage partition per ring
RING_1_STAGE_1 = (NEMAPhase.P1_SB_LEFT, NEMAPhase.P2_NB_THROUGH)
RING_1_STAGE_2 = (NEMAPhase.P3_WB_LEFT, NEMAPhase.P4_EB_THROUGH)
RING_2_STAGE_1 = (NEMAPhase.P5_NB_LEFT, NEMAPhase.P6_SB_THROUGH)
RING_2_STAGE_2 = (NEMAPhase.P7_EB_LEFT, NEMAPhase.P8_WB_THROUGH)

# Complete set of mathematically compatible concurrent phase pairs
LEGAL_CONCURRENT_PAIRS: frozenset[Tuple[NEMAPhase, NEMAPhase]] = frozenset({
    # Stage 1: Major Street (North-South)
    (NEMAPhase.P1_SB_LEFT, NEMAPhase.P5_NB_LEFT),        # Dual leading lefts
    (NEMAPhase.P1_SB_LEFT, NEMAPhase.P6_SB_THROUGH),     # Lead-lag: SB left + SB through
    (NEMAPhase.P2_NB_THROUGH, NEMAPhase.P5_NB_LEFT),     # Lead-lag: NB through + NB left
    (NEMAPhase.P2_NB_THROUGH, NEMAPhase.P6_SB_THROUGH),  # Coordinated throughs
    # Stage 2: Minor Street (East-West)
    (NEMAPhase.P3_WB_LEFT, NEMAPhase.P7_EB_LEFT),        # Dual leading lefts
    (NEMAPhase.P3_WB_LEFT, NEMAPhase.P8_WB_THROUGH),     # Lead-lag: WB left + WB through
    (NEMAPhase.P4_EB_THROUGH, NEMAPhase.P7_EB_LEFT),     # Lead-lag: EB through + EB left
    (NEMAPhase.P4_EB_THROUGH, NEMAPhase.P8_WB_THROUGH),  # Coordinated throughs
})


def are_phases_compatible(p1: NEMAPhase | int, p2: NEMAPhase | int) -> bool:
    """Returns True if the two phases can safely receive green concurrently."""
    phase1 = NEMAPhase(p1)
    phase2 = NEMAPhase(p2)
    return (phase1, phase2) in LEGAL_CONCURRENT_PAIRS or (phase2, phase1) in LEGAL_CONCURRENT_PAIRS


@dataclass
class PhaseTimingConfig:
    """Timing parameters for an individual NEMA phase."""

    phase: NEMAPhase
    min_green_s: float = 7.0
    max_green_s: float = 60.0
    yellow_change_s: float = 3.5
    red_clearance_s: float = 1.5
    vehicle_extension_s: float = 2.5  # Max allowable headway before gap-out


def default_timing_configs() -> dict[NEMAPhase, PhaseTimingConfig]:
    """Provides standard NEMA timing defaults (protected lefts have lower min/max)."""
    return {
        # Ring 1
        NEMAPhase.P1_SB_LEFT: PhaseTimingConfig(
            phase=NEMAPhase.P1_SB_LEFT, min_green_s=5.0, max_green_s=25.0, yellow_change_s=3.0, red_clearance_s=1.5
        ),
        NEMAPhase.P2_NB_THROUGH: PhaseTimingConfig(
            phase=NEMAPhase.P2_NB_THROUGH, min_green_s=7.0, max_green_s=60.0, yellow_change_s=3.5, red_clearance_s=1.5
        ),
        NEMAPhase.P3_WB_LEFT: PhaseTimingConfig(
            phase=NEMAPhase.P3_WB_LEFT, min_green_s=5.0, max_green_s=20.0, yellow_change_s=3.0, red_clearance_s=1.5
        ),
        NEMAPhase.P4_EB_THROUGH: PhaseTimingConfig(
            phase=NEMAPhase.P4_EB_THROUGH, min_green_s=7.0, max_green_s=45.0, yellow_change_s=3.5, red_clearance_s=1.5
        ),
        # Ring 2
        NEMAPhase.P5_NB_LEFT: PhaseTimingConfig(
            phase=NEMAPhase.P5_NB_LEFT, min_green_s=5.0, max_green_s=25.0, yellow_change_s=3.0, red_clearance_s=1.5
        ),
        NEMAPhase.P6_SB_THROUGH: PhaseTimingConfig(
            phase=NEMAPhase.P6_SB_THROUGH, min_green_s=7.0, max_green_s=60.0, yellow_change_s=3.5, red_clearance_s=1.5
        ),
        NEMAPhase.P7_EB_LEFT: PhaseTimingConfig(
            phase=NEMAPhase.P7_EB_LEFT, min_green_s=5.0, max_green_s=20.0, yellow_change_s=3.0, red_clearance_s=1.5
        ),
        NEMAPhase.P8_WB_THROUGH: PhaseTimingConfig(
            phase=NEMAPhase.P8_WB_THROUGH, min_green_s=7.0, max_green_s=45.0, yellow_change_s=3.5, red_clearance_s=1.5
        ),
    }


@dataclass
class NEMADecision:
    """Consolidated actuation output of the NEMA Dual-Ring controller."""

    ring1_phase: NEMAPhase
    ring1_interval: PhaseInterval
    ring1_elapsed_s: float
    ring2_phase: NEMAPhase
    ring2_interval: PhaseInterval
    ring2_elapsed_s: float
    current_stage: RingBarrierStage
    is_at_barrier: bool
    signal_string_8: str  # e.g., 'GGrrGGrr' for 8 movements
    rationale: str
    active_preemption: Optional[NEMAPhase] = None


class NEMADualRingController:
    """Full-featured NEMA TS2 Dual-Ring 8-Phase Actuated Controller."""

    def __init__(
        self,
        configs: Optional[dict[NEMAPhase, PhaseTimingConfig]] = None,
        initial_stage: RingBarrierStage = RingBarrierStage.STAGE_1_MAJOR_NS,
    ) -> None:
        self.configs = configs or default_timing_configs()
        self.current_stage = initial_stage

        # Initial active phases: Dual Throughs (Phase 2 & Phase 6)
        self.ring1_phase: NEMAPhase = NEMAPhase.P2_NB_THROUGH
        self.ring1_interval: PhaseInterval = PhaseInterval.GREEN
        self.ring1_elapsed_s: float = 0.0

        self.ring2_phase: NEMAPhase = NEMAPhase.P6_SB_THROUGH
        self.ring2_interval: PhaseInterval = PhaseInterval.GREEN
        self.ring2_elapsed_s: float = 0.0

        # Preemption target
        self.preemption_target: Optional[NEMAPhase] = None

    def _next_phase_in_ring(self, ring_id: int, current_phase: NEMAPhase) -> NEMAPhase:
        """Determines the standard sequential phase within a ring."""
        sequence_map = {
            # Ring 1
            NEMAPhase.P1_SB_LEFT: NEMAPhase.P2_NB_THROUGH,
            NEMAPhase.P2_NB_THROUGH: NEMAPhase.P3_WB_LEFT,
            NEMAPhase.P3_WB_LEFT: NEMAPhase.P4_EB_THROUGH,
            NEMAPhase.P4_EB_THROUGH: NEMAPhase.P1_SB_LEFT,
            # Ring 2
            NEMAPhase.P5_NB_LEFT: NEMAPhase.P6_SB_THROUGH,
            NEMAPhase.P6_SB_THROUGH: NEMAPhase.P7_EB_LEFT,
            NEMAPhase.P7_EB_LEFT: NEMAPhase.P8_WB_THROUGH,
            NEMAPhase.P8_WB_THROUGH: NEMAPhase.P5_NB_LEFT,
        }
        return sequence_map[current_phase]

    def _is_barrier_boundary(self, phase: NEMAPhase) -> bool:
        """Returns True if the phase is the last phase before a barrier."""
        return phase in (
            NEMAPhase.P2_NB_THROUGH,
            NEMAPhase.P4_EB_THROUGH,
            NEMAPhase.P6_SB_THROUGH,
            NEMAPhase.P8_WB_THROUGH,
        )

    def step(
        self,
        dt: float,
        vehicle_calls: dict[int, bool],
        headways: dict[int, float],
        preemption_call: Optional[int] = None,
    ) -> NEMADecision:
        """Advances the dual-ring controller by dt seconds.

        Parameters:
            dt: Time step in seconds (typically 0.1 to 1.0).
            vehicle_calls: Dict mapping phase int (1..8) to boolean demand call.
            headways: Dict mapping phase int (1..8) to current approach headway (seconds).
            preemption_call: Optional emergency phase ID (1..8) requiring immediate green wave.

        Returns:
            NEMADecision with the state of both rings and the signal string.
        """
        self.ring1_elapsed_s += dt
        self.ring2_elapsed_s += dt

        rationale_items: list[str] = []

        # 1. Emergency Preemption Handling
        if preemption_call is not None:
            self.preemption_target = NEMAPhase(preemption_call)
            rationale_items.append(f"PREEMPTION ACTIVE for Phase {self.preemption_target.value}")
            self._handle_preemption_step(self.preemption_target)
        else:
            self.preemption_target = None
            # Standard Actuated Dual-Ring Progression
            self._update_ring1(dt, vehicle_calls, headways, rationale_items)
            self._update_ring2(dt, vehicle_calls, headways, rationale_items)

        # 2. Barrier Synchronization Check
        r1_at_barrier = self._is_barrier_boundary(self.ring1_phase) and self.ring1_interval == PhaseInterval.RED_HOLD
        r2_at_barrier = self._is_barrier_boundary(self.ring2_phase) and self.ring2_interval == PhaseInterval.RED_HOLD

        if r1_at_barrier and r2_at_barrier:
            # Both rings have cleared to the barrier! Cross the barrier simultaneously.
            self._cross_barrier()
            rationale_items.append(f"BARRIER CROSSED -> Entering {self.current_stage.value}")

        # 3. Build 8-Character Signal String (Phase 1 to Phase 8)
        signal_chars = ["r"] * 8
        for p_idx, p in enumerate(NEMAPhase):
            if p == self.ring1_phase:
                signal_chars[p_idx] = self._interval_to_char(self.ring1_interval)
            elif p == self.ring2_phase:
                signal_chars[p_idx] = self._interval_to_char(self.ring2_interval)

        # 4. Strict Safety Verification (Fail-Safe Assertion)
        if self.ring1_interval == PhaseInterval.GREEN and self.ring2_interval == PhaseInterval.GREEN:
            assert are_phases_compatible(self.ring1_phase, self.ring2_phase), (
                f"FATAL NEMA VIOLATION: Incompatible greens active simultaneously: "
                f"Ring 1 Phase {self.ring1_phase.value} vs Ring 2 Phase {self.ring2_phase.value}"
            )

        rationale = " | ".join(rationale_items) if rationale_items else "Dual-Ring Normal Actuation"

        return NEMADecision(
            ring1_phase=self.ring1_phase,
            ring1_interval=self.ring1_interval,
            ring1_elapsed_s=self.ring1_elapsed_s,
            ring2_phase=self.ring2_phase,
            ring2_interval=self.ring2_interval,
            ring2_elapsed_s=self.ring2_elapsed_s,
            current_stage=self.current_stage,
            is_at_barrier=r1_at_barrier or r2_at_barrier,
            signal_string_8="".join(signal_chars),
            rationale=rationale,
            active_preemption=self.preemption_target,
        )

    def _interval_to_char(self, interval: PhaseInterval) -> str:
        """Converts phase interval to SUMO/hardware character (G, y, r)."""
        if interval == PhaseInterval.GREEN:
            return "G"
        elif interval == PhaseInterval.YELLOW_CHANGE:
            return "y"
        else:
            return "r"

    def _update_ring1(
        self,
        dt: float,
        vehicle_calls: dict[int, bool],
        headways: dict[int, float],
        rationale: list[str],
    ) -> None:
        """Advances Ring 1 state machine through Green, Yellow, Red Clearance, and Next Phase."""
        cfg = self.configs[self.ring1_phase]

        if self.ring1_interval == PhaseInterval.GREEN:
            # Check gap-out or max-out
            headway = headways.get(self.ring1_phase.value, 0.0)
            is_gap_out = headway > cfg.vehicle_extension_s and self.ring1_elapsed_s >= cfg.min_green_s
            is_max_out = self.ring1_elapsed_s >= cfg.max_green_s
            next_phase = self._next_phase_in_ring(1, self.ring1_phase)
            has_conflicting_call = vehicle_calls.get(next_phase.value, False)

            if (is_gap_out and has_conflicting_call) or is_max_out:
                self.ring1_interval = PhaseInterval.YELLOW_CHANGE
                self.ring1_elapsed_s = 0.0
                reason = "MAX-OUT" if is_max_out else f"GAP-OUT (h={headway:.1f}s)"
                rationale.append(f"Ring 1 P{self.ring1_phase.value} {reason} -> Yellow")

        elif self.ring1_interval == PhaseInterval.YELLOW_CHANGE:
            if self.ring1_elapsed_s >= cfg.yellow_change_s:
                self.ring1_interval = PhaseInterval.ALL_RED_CLEARANCE
                self.ring1_elapsed_s = 0.0
                rationale.append(f"Ring 1 P{self.ring1_phase.value} -> All-Red Clearance")

        elif self.ring1_interval == PhaseInterval.ALL_RED_CLEARANCE:
            if self.ring1_elapsed_s >= cfg.red_clearance_s:
                # If this phase touches the barrier, hold in RED_HOLD until Ring 2 arrives
                if self._is_barrier_boundary(self.ring1_phase):
                    self.ring1_interval = PhaseInterval.RED_HOLD
                    self.ring1_elapsed_s = 0.0
                    rationale.append(f"Ring 1 P{self.ring1_phase.value} waiting at BARRIER")
                else:
                    # Advance to next phase within the same stage
                    self.ring1_phase = self._next_phase_in_ring(1, self.ring1_phase)
                    self.ring1_interval = PhaseInterval.GREEN
                    self.ring1_elapsed_s = 0.0
                    rationale.append(f"Ring 1 -> Phase {self.ring1_phase.value} GREEN")

    def _update_ring2(
        self,
        dt: float,
        vehicle_calls: dict[int, bool],
        headways: dict[int, float],
        rationale: list[str],
    ) -> None:
        """Advances Ring 2 state machine through Green, Yellow, Red Clearance, and Next Phase."""
        cfg = self.configs[self.ring2_phase]

        if self.ring2_interval == PhaseInterval.GREEN:
            headway = headways.get(self.ring2_phase.value, 0.0)
            is_gap_out = headway > cfg.vehicle_extension_s and self.ring2_elapsed_s >= cfg.min_green_s
            is_max_out = self.ring2_elapsed_s >= cfg.max_green_s
            next_phase = self._next_phase_in_ring(2, self.ring2_phase)
            has_conflicting_call = vehicle_calls.get(next_phase.value, False)

            if (is_gap_out and has_conflicting_call) or is_max_out:
                self.ring2_interval = PhaseInterval.YELLOW_CHANGE
                self.ring2_elapsed_s = 0.0
                reason = "MAX-OUT" if is_max_out else f"GAP-OUT (h={headway:.1f}s)"
                rationale.append(f"Ring 2 P{self.ring2_phase.value} {reason} -> Yellow")

        elif self.ring2_interval == PhaseInterval.YELLOW_CHANGE:
            if self.ring2_elapsed_s >= cfg.yellow_change_s:
                self.ring2_interval = PhaseInterval.ALL_RED_CLEARANCE
                self.ring2_elapsed_s = 0.0
                rationale.append(f"Ring 2 P{self.ring2_phase.value} -> All-Red Clearance")

        elif self.ring2_interval == PhaseInterval.ALL_RED_CLEARANCE:
            if self.ring2_elapsed_s >= cfg.red_clearance_s:
                if self._is_barrier_boundary(self.ring2_phase):
                    self.ring2_interval = PhaseInterval.RED_HOLD
                    self.ring2_elapsed_s = 0.0
                    rationale.append(f"Ring 2 P{self.ring2_phase.value} waiting at BARRIER")
                else:
                    self.ring2_phase = self._next_phase_in_ring(2, self.ring2_phase)
                    self.ring2_interval = PhaseInterval.GREEN
                    self.ring2_elapsed_s = 0.0
                    rationale.append(f"Ring 2 -> Phase {self.ring2_phase.value} GREEN")

    def _cross_barrier(self) -> None:
        """Synchronously advances both Ring 1 and Ring 2 across the barrier."""
        if self.current_stage == RingBarrierStage.STAGE_1_MAJOR_NS:
            self.current_stage = RingBarrierStage.STAGE_2_MINOR_EW
            self.ring1_phase = NEMAPhase.P3_WB_LEFT
            self.ring2_phase = NEMAPhase.P7_EB_LEFT
        else:
            self.current_stage = RingBarrierStage.STAGE_1_MAJOR_NS
            self.ring1_phase = NEMAPhase.P1_SB_LEFT
            self.ring2_phase = NEMAPhase.P5_NB_LEFT

        self.ring1_interval = PhaseInterval.GREEN
        self.ring1_elapsed_s = 0.0
        self.ring2_interval = PhaseInterval.GREEN
        self.ring2_elapsed_s = 0.0

    def _handle_preemption_step(self, target_phase: NEMAPhase) -> None:
        """Expedites clearance to open green for emergency vehicle corridor safely."""
        partner_map = {
            NEMAPhase.P1_SB_LEFT: NEMAPhase.P5_NB_LEFT,
            NEMAPhase.P2_NB_THROUGH: NEMAPhase.P6_SB_THROUGH,
            NEMAPhase.P3_WB_LEFT: NEMAPhase.P7_EB_LEFT,
            NEMAPhase.P4_EB_THROUGH: NEMAPhase.P8_WB_THROUGH,
            NEMAPhase.P5_NB_LEFT: NEMAPhase.P1_SB_LEFT,
            NEMAPhase.P6_SB_THROUGH: NEMAPhase.P2_NB_THROUGH,
            NEMAPhase.P7_EB_LEFT: NEMAPhase.P3_WB_LEFT,
            NEMAPhase.P8_WB_THROUGH: NEMAPhase.P4_EB_THROUGH,
        }

        if target_phase in RING_1_PHASES:
            target_r1 = target_phase
            target_r2 = partner_map[target_phase]
        else:
            target_r2 = target_phase
            target_r1 = partner_map[target_phase]

        # Stage determination
        target_stage = (
            RingBarrierStage.STAGE_2_MINOR_EW
            if target_r1 in RING_1_STAGE_2
            else RingBarrierStage.STAGE_1_MAJOR_NS
        )

        # Clear Ring 1
        if self.ring1_phase != target_r1 or self.ring1_interval != PhaseInterval.GREEN:
            if self.ring1_interval == PhaseInterval.GREEN:
                self.ring1_interval = PhaseInterval.YELLOW_CHANGE
                self.ring1_elapsed_s = 0.0
            elif self.ring1_interval == PhaseInterval.YELLOW_CHANGE and self.ring1_elapsed_s >= 3.0:
                self.ring1_interval = PhaseInterval.ALL_RED_CLEARANCE
                self.ring1_elapsed_s = 0.0
            elif self.ring1_interval == PhaseInterval.ALL_RED_CLEARANCE and self.ring1_elapsed_s >= 1.5:
                self.ring1_phase = target_r1
                self.ring1_interval = PhaseInterval.RED_HOLD
                self.ring1_elapsed_s = 0.0

        # Clear Ring 2
        if self.ring2_phase != target_r2 or self.ring2_interval != PhaseInterval.GREEN:
            if self.ring2_interval == PhaseInterval.GREEN:
                self.ring2_interval = PhaseInterval.YELLOW_CHANGE
                self.ring2_elapsed_s = 0.0
            elif self.ring2_interval == PhaseInterval.YELLOW_CHANGE and self.ring2_elapsed_s >= 3.0:
                self.ring2_interval = PhaseInterval.ALL_RED_CLEARANCE
                self.ring2_elapsed_s = 0.0
            elif self.ring2_interval == PhaseInterval.ALL_RED_CLEARANCE and self.ring2_elapsed_s >= 1.5:
                self.ring2_phase = target_r2
                self.ring2_interval = PhaseInterval.RED_HOLD
                self.ring2_elapsed_s = 0.0

        # When both rings have cleared, simultaneously grant emergency green
        if (
            self.ring1_phase == target_r1
            and self.ring2_phase == target_r2
            and self.ring1_interval in (PhaseInterval.RED_HOLD, PhaseInterval.GREEN)
            and self.ring2_interval in (PhaseInterval.RED_HOLD, PhaseInterval.GREEN)
        ):
            self.ring1_interval = PhaseInterval.GREEN
            self.ring2_interval = PhaseInterval.GREEN
            self.current_stage = target_stage
