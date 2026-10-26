"""High-level baseline evaluation: fixed-time plan vs Webster-optimized plan."""

from __future__ import annotations

from src.decision.delay import DelayComparison, compare
from src.decision.dual_regime import DualRegimeOptimizer
from src.decision.intersection import FixedPlan, Intersection, SignalPlan


def flow_ratios(
    intersection: Intersection,
    flows_by_phase: dict[str, float],
) -> dict[str, float]:
    """Convert arrival flows (PCU/h) to flow ratios y_i = q_i / s_i per phase."""
    ratios: dict[str, float] = {}
    for phase in intersection.phases:
        approach = intersection.approach_for_phase(phase.id)
        q = float(flows_by_phase.get(phase.id, 0.0))
        ratios[phase.id] = q / approach.saturation_flow if approach.saturation_flow > 0 else 0.0
    return ratios


def evaluate(
    intersection: Intersection,
    flows_by_phase: dict[str, float],
    fixed_plan: FixedPlan | None = None,
    include_correction: bool = True,
) -> tuple[SignalPlan, DelayComparison]:
    """Optimize the signal plan using Dual-Regime logic and compare against fixed baseline."""
    ratios = flow_ratios(intersection, flows_by_phase)
    optimizer = DualRegimeOptimizer()
    plan = optimizer.optimize(intersection, ratios)
    comparison = compare(
        intersection,
        flows_by_phase,
        plan,
        fixed_plan=fixed_plan,
        include_correction=include_correction,
    )
    return plan, comparison
