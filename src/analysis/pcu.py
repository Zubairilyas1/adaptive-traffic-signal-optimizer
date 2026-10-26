"""Passenger Car Unit (PCU) conversion.

Naive vehicle counts over-weight motorcycles and under-weight buses; PCU
weighting expresses each vehicle as its road-space equivalent of a car.
"""

from __future__ import annotations

from src import config


def class_pcu(class_name: str) -> float:
    return config.pcu_for(class_name)


def counts_to_pcu(counts_by_class: dict[str, int]) -> float:
    """Total PCU for a mapping of class name -> count."""
    return sum(count * config.pcu_for(cls) for cls, count in counts_by_class.items())


def lane_pcu(lane_counts: dict[str, dict[str, int]]) -> dict[str, float]:
    """PCU total per lane. Also reports the equivalent vehicle count."""
    return {lane: counts_to_pcu(classes) for lane, classes in lane_counts.items()}


def approach_pcu(
    approach_counts: dict[str, dict[str, int]],
) -> dict[str, float]:
    """PCU total per approach."""
    return {a: counts_to_pcu(classes) for a, classes in approach_counts.items()}


def vehicle_count(counts_by_class: dict[str, int]) -> int:
    return sum(counts_by_class.values())
