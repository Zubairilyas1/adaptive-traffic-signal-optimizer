"""Intersection, approach, phase and signal-plan data models.

Kept separate from the analysis layer so the decision logic can be unit-tested
with synthetic flows and no video.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src import config


@dataclass(frozen=True)
class Approach:
    id: str
    label: str
    lanes: int = 2

    @property
    def saturation_flow(self) -> float:
        """Approach saturation flow in PCU/hour."""
        return self.lanes * config.SATURATION_FLOW


@dataclass(frozen=True)
class Phase:
    id: str
    approach: str


@dataclass
class FixedPlan:
    """The intersection's existing (non-adaptive) timing plan."""

    cycle: float
    green: dict[str, float]

    def green_for(self, phase_id: str) -> float:
        return self.green[phase_id]


@dataclass
class Intersection:
    name: str
    approaches: dict[str, Approach]
    phases: list[Phase]
    lost_time_per_phase: float = config.LOST_TIME_PER_PHASE
    fixed_plan: FixedPlan | None = None

    @property
    def total_lost_time(self) -> float:
        return self.lost_time_per_phase * len(self.phases)

    def approach_for_phase(self, phase_id: str) -> Approach:
        for phase in self.phases:
            if phase.id == phase_id:
                return self.approaches[phase.approach]
        raise KeyError(f"Unknown phase: {phase_id}")


@dataclass
class SignalPlan:
    """The recommended adaptive plan produced by Webster's method."""

    cycle: float
    green: dict[str, float]
    lost_time: float
    critical_flow_ratio: float
    method: str = "webster"
    oversaturated: bool = False


def load_intersection(path: Path | str) -> Intersection:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    approaches = {
        a["id"]: Approach(id=a["id"], label=a.get("label", a["id"]), lanes=int(a.get("lanes", 2)))
        for a in data["approaches"]
    }
    phases = [Phase(id=p["id"], approach=p["approach"]) for p in data["phases"]]

    fixed = None
    if "fixed_plan" in data:
        fp = data["fixed_plan"]
        fixed = FixedPlan(cycle=float(fp["cycle"]), green={k: float(v) for k, v in fp["green"].items()})

    return Intersection(
        name=data.get("name", "intersection"),
        approaches=approaches,
        phases=phases,
        lost_time_per_phase=float(data.get("lost_time_per_phase", config.LOST_TIME_PER_PHASE)),
        fixed_plan=fixed,
    )
