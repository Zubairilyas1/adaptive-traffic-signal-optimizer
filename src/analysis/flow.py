"""Convert counts to arrival flow rates and flow ratios."""

from __future__ import annotations

from src import config


def pcu_per_hour(pcu: float, window_seconds: float) -> float:
    """Extrapolate a PCU count observed over a window to a PCU/hour rate."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    return pcu * 3600.0 / window_seconds


def flow_ratio(flow_pcu_h: float, saturation_pcu_h: float) -> float:
    """y = q / s."""
    if saturation_pcu_h <= 0:
        raise ValueError("saturation flow must be positive")
    return flow_pcu_h / saturation_pcu_h


def flows_by_phase(
    intersection,
    approach_flow_pcu_h: dict[str, float],
) -> dict[str, float]:
    """Map per-approach arrival flows to per-phase arrival flows.

    If several approaches share a phase (not used in the current 2-phase
    model) their flows are summed.
    """
    flows: dict[str, float] = {phase.id: 0.0 for phase in intersection.phases}
    for phase in intersection.phases:
        flows[phase.id] += float(approach_flow_pcu_h.get(phase.approach, 0.0))
    return flows
