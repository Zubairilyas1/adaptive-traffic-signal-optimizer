"""Webster's optimal signal-timing method.

References (cite in the report):
    Webster, F.V. (1958) "Traffic Signal Settings", Road Research Technical
    Paper No. 39, HMSO, London.

Cycle length:      Co = (1.5 * L + 5) / (1 - Y)
Green allocation:  g_i = (y_i / Y) * (Co - L)
where Y = sum of critical flow ratios y_i = q_i / s_i, and L = total lost time.

This is an established traffic-engineering method, not an original contribution.
The project's contribution is the camera-based PCU queue estimation that
produces q_i, and the end-to-end integration.
"""

from __future__ import annotations

from src import config
from src.decision.intersection import Intersection, SignalPlan


def webster_cycle(
    critical_flow_ratio: float,
    lost_time: float,
    min_cycle: float = config.MIN_CYCLE,
    max_cycle: float = config.MAX_CYCLE,
) -> tuple[float, bool]:
    """Return (cycle_seconds, oversaturated).

    When Y >= 1 the intersection is oversaturated and no finite optimal cycle
    exists; the practical maximum is returned and the flag is set.
    """
    if lost_time <= 0:
        raise ValueError("lost_time must be positive")
    if critical_flow_ratio >= 1.0:
        return max_cycle, True
    if critical_flow_ratio < 0:
        raise ValueError("critical_flow_ratio must be non-negative")

    cycle = (1.5 * lost_time + 5.0) / (1.0 - critical_flow_ratio)
    cycle = max(min_cycle, min(max_cycle, cycle))
    return cycle, False


def webster_greens(
    flow_ratios: dict[str, float],
    cycle: float,
    lost_time: float,
    min_green: float = config.MIN_GREEN,
    max_green: float = config.MAX_GREEN,
) -> dict[str, float]:
    """Allocate effective green per phase, clamped to practical limits.

    The available green (cycle - lost time) is split in proportion to each
    phase's flow ratio. Clamping can leave a small residual, which is
    redistributed proportionally among the phases that are not at a limit.
    """
    total_y = sum(flow_ratios.values())
    available = max(0.0, cycle - lost_time)
    if total_y <= 0:
        share = available / len(flow_ratios)
        return {pid: share for pid in flow_ratios}

    greens = {
        pid: (y / total_y) * available
        for pid, y in flow_ratios.items()
    }

    greens = {pid: max(min_green, min(max_green, g)) for pid, g in greens.items()}

    residual = available - sum(greens.values())
    if abs(residual) > 1e-6:
        free = [pid for pid, g in greens.items() if min_green < g < max_green]
        if free:
            for pid in free:
                greens[pid] += residual / len(free)
        else:
            for pid in greens:
                greens[pid] += residual / len(greens)

    return greens


def optimize(
    intersection: Intersection,
    flow_ratios: dict[str, float],
    min_cycle: float = config.MIN_CYCLE,
    max_cycle: float = config.MAX_CYCLE,
    min_green: float = config.MIN_GREEN,
    max_green: float = config.MAX_GREEN,
) -> SignalPlan:
    """Produce an optimal Webster plan from per-phase flow ratios."""
    lost_time = intersection.total_lost_time
    Y = sum(flow_ratios.values())

    cycle, oversaturated = webster_cycle(Y, lost_time, min_cycle, max_cycle)
    greens = webster_greens(flow_ratios, cycle, lost_time, min_green, max_green)

    return SignalPlan(
        cycle=cycle,
        green=greens,
        lost_time=lost_time,
        critical_flow_ratio=Y,
        method="webster",
        oversaturated=oversaturated,
    )
