"""Webster method - validated against hand-worked values.

Hand example:
    L = 8 s (2 phases x 4 s), y1 = 0.2, y2 = 0.3  ->  Y = 0.5
    Co = (1.5*8 + 5) / (1 - 0.5) = 17 / 0.5 = 34 s
    g1 = (0.2/0.5)*(34-8) = 10.4 s ; g2 = (0.3/0.5)*26 = 15.6 s
"""

import math

from src.decision.intersection import Approach, FixedPlan, Intersection, Phase
from src.decision.webster import optimize, webster_cycle, webster_greens


def test_webster_cycle_hand_worked_example():
    cycle, over = webster_cycle(0.5, lost_time=8.0, min_cycle=10.0, max_cycle=120.0)
    assert math.isclose(cycle, 34.0, abs_tol=1e-6)
    assert over is False


def test_webster_greens_hand_worked_example():
    greens = webster_greens({"P1": 0.2, "P2": 0.3}, cycle=34.0, lost_time=8.0,
                            min_green=7.0, max_green=60.0)
    assert math.isclose(greens["P1"], 10.4, abs_tol=1e-6)
    assert math.isclose(greens["P2"], 15.6, abs_tol=1e-6)
    assert math.isclose(sum(greens.values()), 26.0, abs_tol=1e-6)


def test_oversaturated_flagged():
    cycle, over = webster_cycle(1.0, lost_time=8.0, min_cycle=40.0, max_cycle=120.0)
    assert over is True
    assert cycle == 120.0


def test_greens_clamped_to_minimum():
    # extreme imbalance would starve P2 below min green; clamping must apply
    greens = webster_greens({"P1": 0.9, "P2": 0.02}, cycle=60.0, lost_time=8.0,
                            min_green=7.0, max_green=60.0)
    assert greens["P2"] >= 7.0
    assert math.isclose(sum(greens.values()), 52.0, abs_tol=1e-6)


def _intersection() -> Intersection:
    return Intersection(
        name="t",
        approaches={"A": Approach("A", "NS", 2), "B": Approach("B", "EW", 2)},
        phases=[Phase("P1", "A"), Phase("P2", "B")],
        lost_time_per_phase=4.0,
        fixed_plan=FixedPlan(cycle=90.0, green={"P1": 42.0, "P2": 42.0}),
    )


def test_optimize_respects_min_cycle():
    plan = optimize(_intersection(), {"P1": 0.2, "P2": 0.3})
    # raw Webster cycle = 34 s, clamped up to config MIN_CYCLE (40 s)
    assert plan.cycle == 40.0
    assert math.isclose(sum(plan.green.values()), 32.0, abs_tol=1e-6)
