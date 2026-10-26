"""Unit tests for the Dual-Regime Optimization Engine."""

from __future__ import annotations

import pytest

from src import config
from src.decision.dual_regime import (
    DualRegimeOptimizer,
    OptimizationRegime,
)
from src.decision.intersection import load_intersection


@pytest.fixture
def test_intersection():
    path = config.ROOT / "configs" / "intersections" / "default.json"
    return load_intersection(path)


def test_dual_regime_undersaturated_webster(test_intersection):
    optimizer = DualRegimeOptimizer(min_cycle=40.0, max_cycle=120.0)

    # Low flow ratios: Y = 0.35 + 0.25 = 0.60 (< 0.85)
    flow_ratios = {"P1": 0.35, "P2": 0.25}
    plan = optimizer.optimize(test_intersection, flow_ratios)

    assert plan.regime == OptimizationRegime.UNDERSATURATED_WEBSTER
    assert plan.oversaturated is False
    assert plan.bottleneck_phase == "P1"
    # Webster optimal cycle: (1.5 * 8 + 5) / (1 - 0.60) = 17 / 0.40 = 42.5s
    assert pytest.approx(plan.cycle, abs=1.0) == 42.5
    assert plan.green["P1"] > plan.green["P2"]


def test_dual_regime_near_capacity(test_intersection):
    optimizer = DualRegimeOptimizer(min_cycle=40.0, max_cycle=120.0)

    # Moderate-high flow ratios: Y = 0.50 + 0.40 = 0.90 (0.85 <= Y < 0.95)
    flow_ratios = {"P1": 0.50, "P2": 0.40}
    plan = optimizer.optimize(test_intersection, flow_ratios)

    assert plan.regime == OptimizationRegime.NEAR_CAPACITY
    assert plan.oversaturated is False
    assert plan.cycle > 90.0  # Cycle stretched to maximize capacity
    assert plan.green["P1"] > plan.green["P2"]


def test_dual_regime_oversaturated_gating(test_intersection):
    optimizer = DualRegimeOptimizer(min_cycle=40.0, max_cycle=120.0)

    # Severe congestion / rush hour: Y = 0.65 + 0.55 = 1.20 (>= 0.95)
    # Classical Webster would calculate negative or infinite cycle!
    flow_ratios = {"P1": 0.65, "P2": 0.55}
    plan = optimizer.optimize(test_intersection, flow_ratios)

    assert plan.regime == OptimizationRegime.OVERSATURATED_GATING
    assert plan.oversaturated is True
    assert plan.cycle == 120.0  # Cycle locked at C_max to minimize lost time percentage
    assert plan.bottleneck_phase == "P1"
    assert "P2" in plan.metered_phases
    # Bottleneck phase P1 gets maximum throughput flush green
    assert plan.green["P1"] > plan.green["P2"]
    # Metered phase gets controlled green to prevent downstream gridlock
    assert plan.green["P2"] <= 20.0
