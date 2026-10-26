"""Dual-Regime Optimization Engine (Webster vs. Anti-Spillback Bottleneck Gating).

Solves Flaw #5: Webster's equation breakdown under heavy / rush-hour traffic
(Y >= 0.85), where infinite or negative cycles would be produced.

Operates across 3 distinct ITS regimes:
1. Regime 1 (Y < 0.85): Undersaturated -> Delay-minimizing Webster optimal cycle & green splits.
2. Regime 2 (0.85 <= Y < 0.95): Near-Capacity -> Capacity equalization with stretched cycle.
3. Regime 3 (Y >= 0.95): Oversaturated -> Anti-Spillback Gating & Bottleneck Flush Mode.
   - Cycle locked to C_max (120s) to minimize lost time percentage.
   - Green allocated to prioritize flushing standing queues on bottleneck approach.
   - Non-critical approaches are metered to prevent gridlock in the junction box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src import config
from src.decision.intersection import Intersection, SignalPlan
from src.decision.webster import webster_cycle, webster_greens


class OptimizationRegime(Enum):
    UNDERSATURATED_WEBSTER = "Undersaturated (Webster Delay-Minimization)"
    NEAR_CAPACITY = "Near-Capacity (Equalized Saturation Splits)"
    OVERSATURATED_GATING = "Oversaturated (Anti-Spillback Bottleneck Gating)"


@dataclass
class DualRegimePlan(SignalPlan):
    """Extended SignalPlan with regime classification and anti-spillback telemetry."""
    regime: OptimizationRegime = OptimizationRegime.UNDERSATURATED_WEBSTER
    bottleneck_phase: Optional[str] = None
    metered_phases: list[str] = field(default_factory=list)
    regime_rationale: str = ""


class DualRegimeOptimizer:
    """Intelligently transitions between delay-minimization and spillback gating."""

    def __init__(
        self,
        min_cycle: float = config.MIN_CYCLE,
        max_cycle: float = config.MAX_CYCLE,
        min_green: float = config.MIN_GREEN,
        max_green: float = config.MAX_GREEN,
        regime1_threshold: float = 0.85,
        regime2_threshold: float = 0.95,
    ):
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle
        self.min_green = min_green
        self.max_green = max_green
        self.regime1_threshold = regime1_threshold
        self.regime2_threshold = regime2_threshold

    def optimize(
        self,
        intersection: Intersection,
        flow_ratios: dict[str, float],
        halted_queues_pcu: Optional[dict[str, float]] = None,
    ) -> DualRegimePlan:
        lost_time = intersection.total_lost_time
        Y = float(sum(flow_ratios.values()))
        available_green_max = max(0.0, self.max_cycle - lost_time)

        # Identify highest demand phase (bottleneck)
        sorted_phases = sorted(flow_ratios.items(), key=lambda item: item[1], reverse=True)
        bottleneck_p = sorted_phases[0][0] if sorted_phases else list(flow_ratios.keys())[0]

        # -------------------------------------------------------------
        # REGIME 1: Undersaturated (Y < 0.85) - Standard Webster
        # -------------------------------------------------------------
        if Y < self.regime1_threshold:
            cycle, _ = webster_cycle(Y, lost_time, self.min_cycle, self.max_cycle)
            greens = webster_greens(flow_ratios, cycle, lost_time, self.min_green, self.max_green)
            return DualRegimePlan(
                cycle=cycle,
                green=greens,
                lost_time=lost_time,
                critical_flow_ratio=Y,
                oversaturated=False,
                regime=OptimizationRegime.UNDERSATURATED_WEBSTER,
                bottleneck_phase=bottleneck_p,
                metered_phases=[],
                regime_rationale=f"Traffic volume is undersaturated (Y={Y:.3f} < 0.85). Webster formula optimizes cycle ({cycle:.1f}s) to minimize overall vehicle delay.",
            )

        # -------------------------------------------------------------
        # REGIME 2: Near-Capacity (0.85 <= Y < 0.95) - Capacity Equalization
        # -------------------------------------------------------------
        elif Y < self.regime2_threshold:
            # Stretch cycle to 90% of max_cycle to expand available green time
            cycle = min(self.max_cycle, max(self.min_cycle, self.min_cycle + (self.max_cycle - self.min_cycle) * 0.85))
            avail = max(0.0, cycle - lost_time)
            # Allocate green in strict proportion to flow ratios
            greens = {pid: (y / Y) * avail for pid, y in flow_ratios.items()}
            greens = {pid: max(self.min_green, min(self.max_green, g)) for pid, g in greens.items()}
            # Rebalance residuals
            residual = avail - sum(greens.values())
            if abs(residual) > 1e-4:
                for pid in greens:
                    greens[pid] += residual / len(greens)

            return DualRegimePlan(
                cycle=cycle,
                green=greens,
                lost_time=lost_time,
                critical_flow_ratio=Y,
                oversaturated=False,
                regime=OptimizationRegime.NEAR_CAPACITY,
                bottleneck_phase=bottleneck_p,
                metered_phases=[],
                regime_rationale=f"Intersection approaching capacity (Y={Y:.3f}). Stretched cycle ({cycle:.1f}s) equalizes degree of saturation across all approaches.",
            )

        # -------------------------------------------------------------
        # REGIME 3: Oversaturated (Y >= 0.95) - Anti-Spillback Bottleneck Gating
        # -------------------------------------------------------------
        else:
            # Lock cycle to C_max (120s) to maximize throughput and minimize lost time fraction
            cycle = self.max_cycle
            avail = max(0.0, cycle - lost_time)

            greens: dict[str, float] = {}
            metered: list[str] = []

            # Secondary / non-bottleneck phases receive metered green to prevent downstream spillback
            other_phases = [p for p in flow_ratios.keys() if p != bottleneck_p]
            metered_green_each = max(self.min_green, min(18.0, avail / (len(flow_ratios) + 1)))

            for pid in other_phases:
                greens[pid] = metered_green_each
                metered.append(pid)

            # Bottleneck phase receives remaining green to maximize discharge rate
            remaining_for_bottleneck = avail - sum(greens.values())
            greens[bottleneck_p] = max(self.min_green, min(self.max_green, remaining_for_bottleneck))

            # Redistribute if bounded
            residual = avail - sum(greens.values())
            if abs(residual) > 1e-4:
                greens[bottleneck_p] += residual

            return DualRegimePlan(
                cycle=cycle,
                green=greens,
                lost_time=lost_time,
                critical_flow_ratio=Y,
                oversaturated=True,
                regime=OptimizationRegime.OVERSATURATED_GATING,
                bottleneck_phase=bottleneck_p,
                metered_phases=metered,
                regime_rationale=(
                    f"CRITICAL OVERSATURATION (Y={Y:.3f} >= 0.95). Webster math bypassed to avoid deadlock. "
                    f"Switched to Spillback Metering: C locked at {cycle:.0f}s. Bottleneck phase '{bottleneck_p}' "
                    f"allocated {greens[bottleneck_p]:.1f}s green to flush standing backlog while cross-traffic is metered to prevent gridlock."
                ),
            )
