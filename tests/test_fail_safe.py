"""Unit tests for NEMAFailSafeWatchdog and safety fallback mechanisms."""

from __future__ import annotations

import numpy as np
import pytest

from src.decision.actuator import SignalAspect
from src.decision.fail_safe import FailureReason, HealthState, NEMAFailSafeWatchdog
from src.decision.intersection import Approach, Intersection, Phase


@pytest.fixture
def sample_intersection() -> Intersection:
    return Intersection(
        name="Test 2-Phase",
        approaches={
            "North": Approach(id="North", label="Northbound", lanes=2),
            "East": Approach(id="East", label="Eastbound", lanes=2),
        },
        phases=[
            Phase(id="P1", approach="North"),
            Phase(id="P2", approach="East"),
        ],
    )


def test_watchdog_healthy_heartbeat():
    dog = NEMAFailSafeWatchdog(timeout_s=2.0)
    normal_frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    ok = dog.feed_heartbeat(timestamp=1.0, frame=normal_frame)
    assert ok is True

    state, reason, msg = dog.evaluate(timestamp=1.5)
    assert state == HealthState.NORMAL
    assert reason == FailureReason.NONE
    assert dog.is_failsafe_active is False


def test_watchdog_black_frame_occlusion():
    dog = NEMAFailSafeWatchdog(timeout_s=2.0, min_variance=5.0)
    black_frame = np.zeros((200, 200, 3), dtype=np.uint8)

    ok = dog.feed_heartbeat(timestamp=1.0, frame=black_frame)
    assert ok is False

    state, reason, msg = dog.evaluate(timestamp=1.1)
    assert state == HealthState.FAILSAFE_ACTIVE
    assert reason == FailureReason.LENS_OCCLUDED
    assert dog.is_failsafe_active is True


def test_watchdog_feed_timeout():
    dog = NEMAFailSafeWatchdog(timeout_s=2.0)

    # Initial frame at t=0
    dog.feed_heartbeat(timestamp=0.0)
    assert dog.evaluate(timestamp=1.0)[0] == HealthState.NORMAL

    # Fast forward past timeout threshold (t=3.0, > 2.0s gap)
    state, reason, msg = dog.evaluate(timestamp=3.1)
    assert state == HealthState.FAILSAFE_ACTIVE
    assert reason == FailureReason.FEED_TIMEOUT
    assert "lost" in msg.lower() or "timeout" in msg.lower()


def test_watchdog_fallback_pretimed_cycle(sample_intersection: Intersection):
    dog = NEMAFailSafeWatchdog(fallback_green_s=10.0, yellow_s=3.0, all_red_s=1.0)
    dog.state = HealthState.FAILSAFE_ACTIVE

    # Step at start of P1 green
    c1 = dog.step_fallback(timestamp=0.0, intersection=sample_intersection)
    assert c1.active_phase_id == "P1"
    assert c1.active_aspect == SignalAspect.GREEN

    # Step at t=10.1 (should transition to YELLOW)
    c2 = dog.step_fallback(timestamp=10.1, intersection=sample_intersection)
    assert c2.active_phase_id == "P1"
    assert c2.active_aspect == SignalAspect.YELLOW

    # Step at t=13.2 (should transition to RED)
    c3 = dog.step_fallback(timestamp=13.2, intersection=sample_intersection)
    assert c3.active_aspect == SignalAspect.RED

    # Step at t=14.3 (all red finished -> P2 GREEN)
    c4 = dog.step_fallback(timestamp=14.3, intersection=sample_intersection)
    assert c4.active_phase_id == "P2"
    assert c4.active_aspect == SignalAspect.GREEN


def test_watchdog_recovery_window():
    dog = NEMAFailSafeWatchdog(timeout_s=2.0, recovery_window_s=3.0)
    dog.state = HealthState.FAILSAFE_ACTIVE
    dog.last_failure_reason = FailureReason.FEED_TIMEOUT

    # Frame resumes at t=10.0
    dog.feed_heartbeat(timestamp=10.0)

    # At t=11.5, only 1.5s healthy (less than 3.0s recovery window)
    state, _, _ = dog.evaluate(timestamp=11.5)
    assert state == HealthState.FAILSAFE_ACTIVE

    # Continue feeding heartbeats to t=13.2 (3.2s healthy)
    dog.feed_heartbeat(timestamp=13.2)
    state, reason, msg = dog.evaluate(timestamp=13.2)
    assert state == HealthState.NORMAL
    assert reason == FailureReason.NONE
    assert dog.is_failsafe_active is False
