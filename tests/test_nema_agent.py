"""Unit and Safety Verification Tests for NEMA TS2 Dual-Ring RL Agent & Action Masking."""

import numpy as np
import pytest

from src.decision.nema_dual_ring import (
    NEMADualRingController,
    NEMAPhase,
    PhaseInterval,
    RingBarrierStage,
    are_phases_compatible,
)
from src.rl.nema_agent import (
    NEMAAction,
    NEMADualRingAgent,
    NEMAObservationAssembler,
    NEMAPolicyDecision,
    NEMASafetyActionMask,
)


def test_observation_assembler_dimensions_and_bounds():
    """Verifies observation vector is strictly 16-dimensional with values in [0.0, 1.0]."""
    assembler = NEMAObservationAssembler(max_dist_m=100.0, max_pcu=50.0)
    controller = NEMADualRingController()

    queues_m = {p: float(p.value * 10) for p in NEMAPhase}
    pcu_counts = {p: float(p.value * 5) for p in NEMAPhase}

    obs = assembler.assemble(queues_m, pcu_counts, controller)

    assert obs.shape == (16,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)

    # Verify first 8 elements match normalized metric queues
    for p in NEMAPhase:
        idx = p.value - 1
        expected = min((p.value * 10) / 100.0, 1.0)
        assert np.isclose(obs[idx], expected)


def test_safety_action_masking_min_green_lock():
    """Ensures yield actions are strictly masked out when elapsed green < min_green."""
    controller = NEMADualRingController()
    controller.ring1_elapsed_s = 2.0  # < min_green (7.0s)
    controller.ring2_elapsed_s = 2.0

    mask = NEMASafetyActionMask.compute_mask(controller)

    assert mask.shape == (5,)
    assert mask[NEMAAction.HOLD_BOTH] is np.True_
    # All yield actions must be FALSE to protect driver safety
    assert mask[NEMAAction.YIELD_RING1] is np.False_
    assert mask[NEMAAction.YIELD_RING2] is np.False_
    assert mask[NEMAAction.YIELD_BOTH] is np.False_
    assert mask[NEMAAction.EXPEDITE_BARRIER] is np.False_


def test_safety_action_masking_max_green_clamp():
    """Ensures hold action is strictly masked out when elapsed green >= max_green."""
    controller = NEMADualRingController()
    cfg1 = controller.configs[controller.ring1_phase]
    controller.ring1_elapsed_s = cfg1.max_green_s + 1.0

    mask = NEMASafetyActionMask.compute_mask(controller)

    # Holding is illegal once maximum green is exceeded
    assert mask[NEMAAction.HOLD_BOTH] is np.False_
    assert mask[NEMAAction.YIELD_RING1] is np.True_


def test_agent_step_and_decision_telemetry():
    """Tests that the agent executes a closed-loop step and produces valid telemetry."""
    agent = NEMADualRingAgent(policy_mode="max_pressure")

    queues_m = {p: 15.0 for p in NEMAPhase}
    pcu_counts = {p: 10.0 for p in NEMAPhase}
    headways = {p.value: 1.2 for p in NEMAPhase}

    dec = agent.step(
        dt=1.0,
        queues_m=queues_m,
        pcu_counts=pcu_counts,
        headways=headways,
    )

    assert isinstance(dec, NEMAPolicyDecision)
    assert dec.action_name in [a.name for a in NEMAAction]
    assert len(dec.signal_string_12) == 12
    assert dec.observation.shape == (16,)
    assert dec.reward < 0.0  # Total queue penalty

    telemetry = dec.to_dict()
    assert "safe_action" in telemetry
    assert "signal_string_12" in telemetry


def test_agent_preemption_override():
    """Emergency vehicle preemption overrides agent actions to guarantee green wave."""
    agent = NEMADualRingAgent(policy_mode="max_pressure")

    queues_m = {p: 20.0 for p in NEMAPhase}
    pcu_counts = {p: 15.0 for p in NEMAPhase}
    headways = {p.value: 1.0 for p in NEMAPhase}

    # Ambulance approaching on Phase 3
    dec = agent.step(
        dt=1.0,
        queues_m=queues_m,
        pcu_counts=pcu_counts,
        headways=headways,
        preemption_call=3,
    )

    assert dec.active_preemption == NEMAPhase.P3_WB_LEFT
    assert dec.reward > 0.0  # Preemption compliance bonus


def test_closed_loop_100_step_simulation():
    """Runs 100 continuous seconds of closed-loop simulation verifying mathematical invariants."""
    agent = NEMADualRingAgent(policy_mode="max_pressure")

    for step_idx in range(100):
        # Semi-random dynamic arrivals
        queues_m = {p: float((step_idx * p.value) % 40) for p in NEMAPhase}
        pcu_counts = {p: float((step_idx + p.value) % 25) for p in NEMAPhase}
        headways = {p.value: 1.0 + float(step_idx % 4) for p in NEMAPhase}

        dec = agent.step(
            dt=1.0,
            queues_m=queues_m,
            pcu_counts=pcu_counts,
            headways=headways,
        )

        # Invariant 1: Signal string is valid 12 chars
        assert len(dec.signal_string_12) == 12
        for ch in dec.signal_string_12:
            assert ch in ("G", "y", "r")

        # Invariant 2: Active green phases are ALWAYS compatible
        if dec.ring1_interval == PhaseInterval.GREEN and dec.ring2_interval == PhaseInterval.GREEN:
            assert are_phases_compatible(dec.ring1_phase, dec.ring2_phase), (
                f"SAFETY VIOLATION at step {step_idx}: "
                f"P{dec.ring1_phase.value} vs P{dec.ring2_phase.value}"
            )
