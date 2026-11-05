"""Unit tests for Live PPO Policy Engine, Observation Assembler, and NEMA Actuator."""

from __future__ import annotations

import numpy as np
import pytest

from src.rl.live_policy import (
    LivePolicyActuator,
    LivePPOController,
    LiveSignalAspect,
    PolicyDecision,
    assemble_observation,
)


def test_assemble_observation():
    # Test dictionary input
    queues_dict = {"North": 10.0, "South": 5.0, "East": 0.0, "West": 20.0}
    obs = assemble_observation(queues_dict, active_phase=1, elapsed_green=30.0)

    assert obs.shape == (6,)
    assert obs.dtype == np.float32
    assert np.isclose(obs[0], 0.5)  # 10 / 20
    assert np.isclose(obs[1], 0.25)  # 5 / 20
    assert np.isclose(obs[2], 0.0)  # 0 / 20
    assert np.isclose(obs[3], 1.0)  # 20 / 20
    assert np.isclose(obs[4], 1.0)  # phase 1
    assert np.isclose(obs[5], 0.5)  # 30 / 60

    # Test queue clipping
    obs_clipped = assemble_observation([40.0, 0.0, 0.0, 0.0], active_phase=0, elapsed_green=90.0)
    assert np.isclose(obs_clipped[0], 1.0)  # clamped at 1.0
    assert np.isclose(obs_clipped[5], 1.0)  # clamped at 1.0


def test_live_ppo_controller():
    controller = LivePPOController()
    assert controller.is_available is True

    obs = np.array([0.5, 0.4, 0.1, 0.0, 0.0, 0.3], dtype=np.float32)
    action, conf, probs = controller.predict(obs)

    assert action in (0, 1)
    assert 0.0 <= conf <= 1.0
    assert len(probs) == 2
    assert np.isclose(probs.sum(), 1.0, atol=1e-4)


def test_live_policy_actuator_fixed_time():
    actuator = LivePolicyActuator(
        controller_type="fixed",
        min_green=5.0,
        fixed_green=10.0,
        yellow_time=2.0,
        all_red_time=1.0,
    )
    actuator.reset(phase=0)

    # Step during green (0 to 9s)
    for _ in range(9):
        dec = actuator.step([0, 0, 0, 0], dt=1.0)
        assert dec.active_aspect == LiveSignalAspect.GREEN
        assert dec.action == 0
        assert dec.active_phase == 0

    # Step at 10s: fixed green expires -> triggers yellow
    dec10 = actuator.step([0, 0, 0, 0], dt=1.0)
    assert dec10.action == 1
    assert dec10.active_aspect == LiveSignalAspect.YELLOW

    # Step 11s: in yellow (1s elapsed)
    dec11 = actuator.step([0, 0, 0, 0], dt=1.0)
    assert dec11.active_aspect == LiveSignalAspect.YELLOW

    # Step 12s: yellow expires (2s elapsed) -> all-red
    dec12 = actuator.step([0, 0, 0, 0], dt=1.0)
    assert dec12.active_aspect == LiveSignalAspect.ALL_RED

    # Step 13s: all-red expires (1s elapsed) -> new green opened for Phase 1!
    dec13 = actuator.step([0, 0, 0, 0], dt=1.0)
    assert dec13.active_aspect == LiveSignalAspect.GREEN
    assert dec13.active_phase == 1
    assert dec13.switched is True
    assert dec13.green_elapsed_s == 0.0


def test_live_policy_actuator_min_green_clamp():
    actuator = LivePolicyActuator(
        controller_type="ppo",
        min_green=10.0,
        yellow_time=3.0,
        all_red_time=1.0,
    )
    actuator.reset(phase=0)

    # Force a queue state where PPO wants to switch immediately
    heavy_cross = [0.0, 0.0, 15.0, 15.0]

    # At t=2.0s: PPO raw_action may be 1, but must be clamped by min_green
    for _ in range(2):
        dec = actuator.step(heavy_cross, dt=1.0)

    if dec.raw_action == 1:
        assert dec.action == 0
        assert "CLAMPED" in dec.rationale

    # Advance until min_green (10.0s) is satisfied
    for _ in range(8):
        dec = actuator.step(heavy_cross, dt=1.0)

    # Once min_green satisfied, if PPO requests switch, it should initiate yellow
    if dec.raw_action == 1:
        assert dec.active_aspect == LiveSignalAspect.YELLOW
        assert dec.action == 1


def test_live_policy_actuator_hot_swap():
    actuator = LivePolicyActuator(controller_type="fixed")
    assert actuator.controller_type == "fixed"

    actuator.set_controller("actuated")
    assert actuator.controller_type == "actuated"

    actuator.set_controller("ppo")
    assert actuator.controller_type == "ppo"
