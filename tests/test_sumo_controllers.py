"""Tests for the signal controllers (no SUMO required - uses a fake env)."""

import numpy as np

from src.rl.controllers import ActuatedController, FixedTimeController


class FakeEnv:
    """Minimal stand-in for SumoIntersectionEnv."""

    def __init__(self, queues, phase=0, green_elapsed=0.0):
        self._q = np.asarray(queues, dtype=float)
        self.phase = phase
        self.green_elapsed = green_elapsed

    def raw_queues(self):
        return self._q

    def _observation(self):
        return np.concatenate([np.clip(self._q / 20.0, 0, 1), [self.phase, 0.0]])


def test_fixed_time_switches_after_green():
    ctrl = FixedTimeController(green=30.0)
    assert ctrl.act(FakeEnv([0, 0, 0, 0], green_elapsed=29.0)) == 0
    assert ctrl.act(FakeEnv([0, 0, 0, 0], green_elapsed=30.0)) == 1


def test_actuated_respects_min_green():
    ctrl = ActuatedController(min_green=10.0)
    # other approach congested but min green not met -> keep
    env = FakeEnv([1, 1, 9, 9], phase=0, green_elapsed=5.0)
    assert ctrl.act(env) == 0


def test_actuated_switches_to_busier_approach():
    ctrl = ActuatedController(min_green=10.0, max_green=50.0, margin=2.0)
    # currently NS green, but EW has far bigger queue -> switch
    env = FakeEnv([1, 1, 12, 12], phase=0, green_elapsed=15.0)
    assert ctrl.act(env) == 1


def test_actuated_keeps_when_current_is_busier():
    ctrl = ActuatedController(min_green=10.0, max_green=50.0, margin=2.0)
    env = FakeEnv([15, 15, 1, 1], phase=0, green_elapsed=15.0)
    assert ctrl.act(env) == 0


def test_actuated_forced_switch_at_max_green():
    ctrl = ActuatedController(min_green=10.0, max_green=50.0)
    env = FakeEnv([15, 15, 1, 1], phase=0, green_elapsed=50.0)
    assert ctrl.act(env) == 1
