"""Average control delay (Webster's delay formula) and plan comparison.

Webster per-approach average delay (seconds/vehicle):

    d = C(1 - lambda)^2 / (2(1 - min(1, x) * lambda))          [uniform delay]
      + x^2 / (2 * q * (1 - x))                                 [overflow delay]
      - 0.65 * (C / q^2)^(1/3) * x^(2 + 5*lambda)               [correction, optional]

where lambda = g / C, q = arrival flow (PCU/s), s = saturation flow (PCU/s),
x = q / (lambda * s) is the degree of saturation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.decision.intersection import FixedPlan, Intersection, SignalPlan


@dataclass
class ApproachDelay:
    phase_id: str
    approach_id: str
    flow_pcu_h: float
    delay_s: float  # average delay per vehicle, seconds
    degree_of_saturation: float
    oversaturated: bool

    @property
    def weighted_delay(self) -> float:
        return self.delay_s * self.flow_pcu_h


@dataclass
class DelayComparison:
    fixed_avg_delay: float
    optimized_avg_delay: float
    reduction_pct: float
    fixed_rows: list[ApproachDelay] = field(default_factory=list)
    optimized_rows: list[ApproachDelay] = field(default_factory=list)
    oversaturated: bool = False

    @property
    def improvement_s(self) -> float:
        return self.fixed_avg_delay - self.optimized_avg_delay


def webster_delay(
    cycle: float,
    green: float,
    flow_pcu_h: float,
    saturation_pcu_h: float,
    include_correction: bool = True,
) -> tuple[float, float, bool]:
    """Return (avg_delay_seconds, degree_of_saturation, oversaturated)."""
    if flow_pcu_h <= 0:
        return 0.0, 0.0, False
    if cycle <= 0 or green <= 0 or saturation_pcu_h <= 0:
        raise ValueError("cycle, green and saturation flow must be positive")

    lam = green / cycle
    q = flow_pcu_h / 3600.0
    s = saturation_pcu_h / 3600.0
    x = q / (lam * s)

    oversaturated = x >= 1.0
    xc = min(x, 0.98)

    d1 = cycle * (1.0 - lam) ** 2 / (2.0 * (1.0 - min(1.0, xc) * lam))
    d2 = (xc ** 2) / (2.0 * q * (1.0 - xc))

    d3 = 0.0
    if include_correction and not oversaturated:
        d3 = -0.65 * (cycle / q ** 2) ** (1.0 / 3.0) * (x ** (2.0 + 5.0 * lam))

    delay = max(0.0, d1 + d2 + d3)
    return delay, x, oversaturated


def _average_delay(
    cycle: float,
    greens: dict[str, float],
    flows_by_phase: dict[str, float],
    intersection: Intersection,
    include_correction: bool,
) -> tuple[float, list[ApproachDelay], bool]:
    total_q = 0.0
    total_weighted = 0.0
    rows: list[ApproachDelay] = []
    any_over = False

    for phase in intersection.phases:
        approach = intersection.approach_for_phase(phase.id)
        q = float(flows_by_phase.get(phase.id, 0.0))
        delay, x, over = webster_delay(
            cycle, greens[phase.id], q, approach.saturation_flow, include_correction
        )
        any_over = any_over or over
        rows.append(ApproachDelay(
            phase_id=phase.id,
            approach_id=approach.id,
            flow_pcu_h=q,
            delay_s=delay,
            degree_of_saturation=x,
            oversaturated=over,
        ))
        total_q += q
        total_weighted += delay * q

    avg = total_weighted / total_q if total_q > 0 else 0.0
    return avg, rows, any_over


def compare(
    intersection: Intersection,
    flows_by_phase: dict[str, float],
    optimized_plan: SignalPlan,
    fixed_plan: FixedPlan | None = None,
    include_correction: bool = True,
) -> DelayComparison:
    """Compare the optimized Webster plan against the fixed-time baseline."""
    fixed = fixed_plan or intersection.fixed_plan
    if fixed is None:
        raise ValueError("No fixed-time plan provided or configured.")

    fixed_avg, fixed_rows, over_f = _average_delay(
        fixed.cycle, fixed.green, flows_by_phase, intersection, include_correction
    )
    opt_avg, opt_rows, over_o = _average_delay(
        optimized_plan.cycle, optimized_plan.green, flows_by_phase, intersection, include_correction
    )

    reduction = (fixed_avg - opt_avg) / fixed_avg * 100.0 if fixed_avg > 0 else 0.0

    return DelayComparison(
        fixed_avg_delay=fixed_avg,
        optimized_avg_delay=opt_avg,
        reduction_pct=reduction,
        fixed_rows=fixed_rows,
        optimized_rows=opt_rows,
        oversaturated=over_f or over_o,
    )
