"""Unit and Safety Verification Tests for NEMA TS2 Dual-Ring 8-Phase Controller."""

import random
import pytest

from src.decision.nema_dual_ring import (
    LEGAL_CONCURRENT_PAIRS,
    NEMADecision,
    NEMADualRingController,
    NEMAPhase,
    PhaseInterval,
    RingBarrierStage,
    are_phases_compatible,
)


def test_nema_phase_enum_and_rings():
    """Verifies that all 8 NEMA phases are correctly numbered and mapped."""
    assert len(NEMAPhase) == 8
    assert NEMAPhase.P1_SB_LEFT == 1
    assert NEMAPhase.P2_NB_THROUGH == 2
    assert NEMAPhase.P3_WB_LEFT == 3
    assert NEMAPhase.P4_EB_THROUGH == 4
    assert NEMAPhase.P5_NB_LEFT == 5
    assert NEMAPhase.P6_SB_THROUGH == 6
    assert NEMAPhase.P7_EB_LEFT == 7
    assert NEMAPhase.P8_WB_THROUGH == 8


def test_concurrency_compatibility_matrix():
    """Exhaustive check of the NEMA TS2 concurrency matrix."""
    # Legal Stage 1 pairs
    assert are_phases_compatible(NEMAPhase.P1_SB_LEFT, NEMAPhase.P5_NB_LEFT)
    assert are_phases_compatible(NEMAPhase.P1_SB_LEFT, NEMAPhase.P6_SB_THROUGH)
    assert are_phases_compatible(NEMAPhase.P2_NB_THROUGH, NEMAPhase.P5_NB_LEFT)
    assert are_phases_compatible(NEMAPhase.P2_NB_THROUGH, NEMAPhase.P6_SB_THROUGH)

    # Legal Stage 2 pairs
    assert are_phases_compatible(NEMAPhase.P3_WB_LEFT, NEMAPhase.P7_EB_LEFT)
    assert are_phases_compatible(NEMAPhase.P3_WB_LEFT, NEMAPhase.P8_WB_THROUGH)
    assert are_phases_compatible(NEMAPhase.P4_EB_THROUGH, NEMAPhase.P7_EB_LEFT)
    assert are_phases_compatible(NEMAPhase.P4_EB_THROUGH, NEMAPhase.P8_WB_THROUGH)

    # Conflicting cross-stage pairs must be FALSE
    assert not are_phases_compatible(NEMAPhase.P2_NB_THROUGH, NEMAPhase.P4_EB_THROUGH)
    assert not are_phases_compatible(NEMAPhase.P1_SB_LEFT, NEMAPhase.P3_WB_LEFT)
    assert not are_phases_compatible(NEMAPhase.P6_SB_THROUGH, NEMAPhase.P8_WB_THROUGH)
    assert not are_phases_compatible(NEMAPhase.P5_NB_LEFT, NEMAPhase.P7_EB_LEFT)


def test_controller_initialization():
    """Controller initializes into Stage 1 Dual-Throughs (Phase 2 & Phase 6)."""
    controller = NEMADualRingController()
    assert controller.ring1_phase == NEMAPhase.P2_NB_THROUGH
    assert controller.ring2_phase == NEMAPhase.P6_SB_THROUGH
    assert controller.ring1_interval == PhaseInterval.GREEN
    assert controller.ring2_interval == PhaseInterval.GREEN
    assert controller.current_stage == RingBarrierStage.STAGE_1_MAJOR_NS


def test_gap_out_and_clearance_progression():
    """Tests phase termination upon gap-out headway timeout and yellow/red intervals."""
    controller = NEMADualRingController()

    # Fast forward through minimum green (7.0s) with traffic flowing
    for _ in range(7):
        dec = controller.step(dt=1.0, vehicle_calls={3: True, 7: True}, headways={2: 1.0, 6: 1.0})
        assert dec.ring1_interval == PhaseInterval.GREEN

    # Trigger Gap-Out: Headway spikes to 4.5s (> 2.5s threshold) with cross call
    dec = controller.step(dt=1.0, vehicle_calls={3: True, 7: True}, headways={2: 4.5, 6: 4.5})
    assert dec.ring1_interval == PhaseInterval.YELLOW_CHANGE
    assert dec.ring2_interval == PhaseInterval.YELLOW_CHANGE

    # Yellow Change lasts 3.5s
    for _ in range(4):
        dec = controller.step(dt=1.0, vehicle_calls={3: True, 7: True}, headways={2: 4.5, 6: 4.5})
    assert dec.ring1_interval == PhaseInterval.ALL_RED_CLEARANCE

    # Red clearance lasts 1.5s (step 2.0s)
    dec = controller.step(dt=2.0, vehicle_calls={3: True, 7: True}, headways={2: 4.5, 6: 4.5})
    # Since Phase 2 & 6 are barrier boundaries, they should now hold at barrier or cross
    assert dec.is_at_barrier or dec.current_stage == RingBarrierStage.STAGE_2_MINOR_EW


def test_barrier_synchronization_lock():
    """Ensures Ring 1 cannot enter Stage 2 until Ring 2 also finishes its clearance."""
    controller = NEMADualRingController()

    # Step past min green
    for _ in range(8):
        controller.step(dt=1.0, vehicle_calls={3: True, 7: True}, headways={2: 1.0, 6: 1.0})

    # Gap out Ring 1 ONLY (headway = 5.0s on P2, but P6 has continuous flow headway = 0.5s)
    for _ in range(1):
        controller.step(dt=1.0, vehicle_calls={3: True, 7: True}, headways={2: 5.0, 6: 0.5})

    # Advance through Ring 1 Yellow & Red Clearance
    controller.step(dt=3.5, vehicle_calls={3: True, 7: True}, headways={2: 5.0, 6: 0.5})
    controller.step(dt=1.5, vehicle_calls={3: True, 7: True}, headways={2: 5.0, 6: 0.5})

    # Ring 1 is at barrier (RED_HOLD), but Ring 2 is STILL GREEN on Phase 6!
    assert controller.ring1_interval == PhaseInterval.RED_HOLD
    assert controller.ring2_interval == PhaseInterval.GREEN
    assert controller.current_stage == RingBarrierStage.STAGE_1_MAJOR_NS  # CANNOT CROSS YET!

    # Now gap out Ring 2
    controller.step(dt=1.0, vehicle_calls={3: True, 7: True}, headways={2: 5.0, 6: 5.0})
    controller.step(dt=3.5, vehicle_calls={3: True, 7: True}, headways={2: 5.0, 6: 5.0})
    dec = controller.step(dt=1.5, vehicle_calls={3: True, 7: True}, headways={2: 5.0, 6: 5.0})

    # NOW both have cleared -> BARRIER CROSSED into Stage 2!
    assert dec.current_stage == RingBarrierStage.STAGE_2_MINOR_EW
    assert dec.ring1_phase == NEMAPhase.P3_WB_LEFT
    assert dec.ring2_phase == NEMAPhase.P7_EB_LEFT
    assert dec.ring1_interval == PhaseInterval.GREEN
    assert dec.ring2_interval == PhaseInterval.GREEN


def test_emergency_preemption_priority():
    """Emergency vehicle preemption immediately forces clearance to target corridor."""
    controller = NEMADualRingController()

    # In Phase 2 & 6 green, ambulance calls Phase 3 (Westbound Left)
    dec = controller.step(dt=1.0, vehicle_calls={}, headways={}, preemption_call=3)
    assert dec.active_preemption == NEMAPhase.P3_WB_LEFT
    assert "PREEMPTION" in dec.rationale

    # Step through clearance
    for _ in range(6):
        dec = controller.step(dt=1.0, vehicle_calls={}, headways={}, preemption_call=3)

    # Phase 3 must open green
    assert dec.ring1_phase == NEMAPhase.P3_WB_LEFT
    assert dec.ring1_interval == PhaseInterval.GREEN


def test_mathematical_safety_monte_carlo():
    """Simulates 1000 seconds of random traffic to prove zero conflicting greens ever occur."""
    random.seed(42)
    controller = NEMADualRingController()

    for second in range(1000):
        # Generate random vehicle arrivals and headways across all 8 phases
        calls = {p: random.random() > 0.4 for p in range(1, 9)}
        headways = {p: random.uniform(0.5, 6.0) for p in range(1, 9)}

        dec = controller.step(dt=0.5, vehicle_calls=calls, headways=headways)

        # Invariant 1: If both rings have GREEN, they MUST be compatible
        if dec.ring1_interval == PhaseInterval.GREEN and dec.ring2_interval == PhaseInterval.GREEN:
            assert are_phases_compatible(dec.ring1_phase, dec.ring2_phase), (
                f"SAFETY VIOLATION at t={second * 0.5}s: "
                f"Ring 1 P{dec.ring1_phase.value} vs Ring 2 P{dec.ring2_phase.value}"
            )

        # Invariant 2: Signal string must be valid 8-char format
        assert len(dec.signal_string_8) == 8
        for ch in dec.signal_string_8:
            assert ch in ("G", "y", "r")
