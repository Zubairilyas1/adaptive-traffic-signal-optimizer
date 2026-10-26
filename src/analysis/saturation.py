"""Dynamic Saturation Flow Headway Calibration.

Measures real-world saturation flow from observed departure headways of vehicles
discharging across the stopline during green intervals, eliminating reliance on
an uncalibrated constant textbook value (1900 PCU/h).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src import config


@dataclass
class SaturationEstimate:
    lane_id: str
    calibrated_saturation_pcu_h: float
    sample_count: int
    mean_headway_s: float
    is_calibrated: bool


class SaturationFlowCalibrator:
    """Calibrates saturation flow rate based on discharge headway measurements."""

    def __init__(
        self,
        base_saturation_flow: float = config.SATURATION_FLOW,
        min_saturation: float = 1200.0,
        max_saturation: float = 2400.0,
        ema_alpha: float = 0.25,
    ):
        self.base_saturation = base_saturation_flow
        self.min_saturation = min_saturation
        self.max_saturation = max_saturation
        self.ema_alpha = ema_alpha

        # Per-lane state: departure timestamps and smoothed headway
        self._departures: dict[str, list[float]] = {}
        self._smoothed_headway: dict[str, float] = {}
        self._sample_counts: dict[str, int] = {}

    def record_departure(self, lane_id: str, timestamp: float, vehicle_in_queue_index: int) -> None:
        """Record the timestamp of a vehicle crossing the stopline.

        Only vehicles departing 4th or later in the green interval queue
        are used for saturation headway calculation (startup lost time applies to 1-3).
        """
        if lane_id not in self._departures:
            self._departures[lane_id] = []

        lane_deps = self._departures[lane_id]
        if vehicle_in_queue_index >= 4 and lane_deps:
            prev_ts = lane_deps[-1]
            headway = timestamp - prev_ts
            # Filter valid departure headways (typically between 0.8s and 4.0s)
            if 0.8 <= headway <= 4.0:
                current_smoothed = self._smoothed_headway.get(lane_id, 3600.0 / self.base_saturation)
                updated = self.ema_alpha * headway + (1.0 - self.ema_alpha) * current_smoothed
                self._smoothed_headway[lane_id] = updated
                self._sample_counts[lane_id] = self._sample_counts.get(lane_id, 0) + 1

        lane_deps.append(timestamp)

    def reset_phase_departures(self, lane_id: Optional[str] = None) -> None:
        """Reset departure timestamps at the start/end of a green phase."""
        if lane_id is not None:
            self._departures.pop(lane_id, None)
        else:
            self._departures.clear()

    def get_saturation_flow(self, lane_id: str) -> SaturationEstimate:
        """Return the calibrated saturation flow for a given lane."""
        samples = self._sample_counts.get(lane_id, 0)
        if samples >= 3 and lane_id in self._smoothed_headway:
            mean_h = self._smoothed_headway[lane_id]
            raw_sat = 3600.0 / mean_h if mean_h > 0 else self.base_saturation
            clamped_sat = max(self.min_saturation, min(self.max_saturation, raw_sat))
            return SaturationEstimate(
                lane_id=lane_id,
                calibrated_saturation_pcu_h=round(clamped_sat, 1),
                sample_count=samples,
                mean_headway_s=round(mean_h, 2),
                is_calibrated=True,
            )

        return SaturationEstimate(
            lane_id=lane_id,
            calibrated_saturation_pcu_h=self.base_saturation,
            sample_count=samples,
            mean_headway_s=round(3600.0 / self.base_saturation, 2),
            is_calibrated=False,
        )
