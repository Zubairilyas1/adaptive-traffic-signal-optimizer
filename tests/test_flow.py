"""Tests for PCU weighting and flow conversion."""

import math

from src.analysis import flow, pcu
from src.decision.intersection import Approach, Intersection, Phase


def test_counts_to_pcu_matches_hand_calculation():
    counts = {"car": 10, "motorcycle": 20, "bus": 2}
    # 10*1.0 + 20*0.5 + 2*3.5 = 10 + 10 + 7 = 27
    assert pcu.counts_to_pcu(counts) == 27.0
    assert pcu.vehicle_count(counts) == 32  # naive count is higher


def test_pcu_weighting_reduces_motorcycle_heavy_flow():
    counts = {"motorcycle": 100}
    assert pcu.counts_to_pcu(counts) == 50.0
    assert pcu.counts_to_pcu(counts) < pcu.vehicle_count(counts)


def test_pcu_per_hour_extrapolation():
    assert flow.pcu_per_hour(60, 60) == 3600.0
    assert flow.pcu_per_hour(27, 60) == 1620.0
    assert math.isclose(flow.pcu_per_hour(1, 30), 120.0)


def test_flow_ratio():
    assert math.isclose(flow.flow_ratio(1900, 3800), 0.5)


def test_flows_by_phase_maps_approach_to_phase():
    inter = Intersection(
        name="t",
        approaches={"A": Approach("A", "NS", 2), "B": Approach("B", "EW", 2)},
        phases=[Phase("P1", "A"), Phase("P2", "B")],
    )
    flows = flow.flows_by_phase(inter, {"A": 1200.0, "B": 800.0})
    assert flows == {"P1": 1200.0, "P2": 800.0}
