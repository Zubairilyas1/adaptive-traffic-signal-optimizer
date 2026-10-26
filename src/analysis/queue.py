"""Simple congestion indicators (degree of saturation based)."""

from __future__ import annotations

from dataclasses import dataclass


def congestion_index(flow_pcu_h: float, saturation_pcu_h: float) -> float:
    """Degree of saturation proxy: flow / capacity (>1 means oversaturated)."""
    if saturation_pcu_h <= 0:
        raise ValueError("saturation flow must be positive")
    return flow_pcu_h / saturation_pcu_h


def congestion_label(index: float) -> str:
    if index < 0.6:
        return "Low"
    if index < 0.85:
        return "Moderate"
    if index < 1.0:
        return "High"
    return "Oversaturated"


@dataclass
class Congestion:
    approach_id: str
    flow_pcu_h: float
    saturation_pcu_h: float

    @property
    def index(self) -> float:
        return congestion_index(self.flow_pcu_h, self.saturation_pcu_h)

    @property
    def label(self) -> str:
        return congestion_label(self.index)
