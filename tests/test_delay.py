"""Webster delay formula - validated against a hand-worked example.

Hand-worked example (uniform + overflow, correction off):
    C=60 s, g=30 s, q=600 veh/h, s=1800 veh/h
    lambda = 30/60 = 0.5
    q = 0.16667 veh/s, s = 0.5 veh/s, x = q/(lambda*s) = 0.66667
    d1 = 60*(0.5)^2 / (2*(1 - 0.66667*0.5)) = 11.25 s
    d2 = x^2 / (2*q*(1-x)) = 4.0 s
    d  = 15.25 s/veh

Hand-worked example (with correction term d3):
    d3 = -0.65 * (60 / (1/6)^2)^(1/3) * (2/3)^(2 + 5*0.5) = -1.3615 s
    d  = 11.25 + 4.0 - 1.3615 = 13.8885 s/veh
"""

import math

from src.decision.baseline import evaluate
from src.decision.delay import webster_delay
from src.decision.intersection import Approach, FixedPlan, Intersection, Phase


def test_webster_delay_hand_worked_example():
    delay, x, over = webster_delay(60.0, 30.0, 600.0, 1800.0, include_correction=False)
    assert math.isclose(x, 2.0 / 3.0, abs_tol=1e-6)
    assert math.isclose(delay, 15.25, abs_tol=0.05)
    assert over is False


def test_webster_delay_with_correction_term():
    delay, x, over = webster_delay(60.0, 30.0, 600.0, 1800.0, include_correction=True)
    assert math.isclose(x, 2.0 / 3.0, abs_tol=1e-6)
    # Expected: ~13.89 s/veh
    assert math.isclose(delay, 13.8885, abs_tol=0.05)
    assert over is False


def test_zero_flow_gives_zero_delay():
    delay, x, over = webster_delay(60.0, 30.0, 0.0, 1800.0)
    assert delay == 0.0
    assert x == 0.0


def test_more_green_reduces_delay():
    low = webster_delay(90.0, 30.0, 1200.0, 3800.0)[0]
    high = webster_delay(90.0, 55.0, 1200.0, 3800.0)[0]
    assert high < low


def test_oversaturated_flagged():
    _, _, over = webster_delay(60.0, 10.0, 3600.0, 1800.0)
    assert over is True


def _intersection() -> Intersection:
    return Intersection(
        name="t",
        approaches={"A": Approach("A", "NS", 2), "B": Approach("B", "EW", 2)},
        phases=[Phase("P1", "A"), Phase("P2", "B")],
        lost_time_per_phase=4.0,
        fixed_plan=FixedPlan(cycle=90.0, green={"P1": 42.0, "P2": 42.0}),
    )


def test_optimized_plan_reduces_delay_on_imbalanced_flow():
    # P1 is much busier than P2; the fixed equal split is wasteful
    flows = {"P1": 1500.0, "P2": 600.0}
    plan, comparison = evaluate(_intersection(), flows)
    assert comparison.fixed_avg_delay > 0
    assert comparison.optimized_avg_delay < comparison.fixed_avg_delay
    assert comparison.reduction_pct > 0
    # the busier phase should receive more green
    assert plan.green["P1"] > plan.green["P2"]
