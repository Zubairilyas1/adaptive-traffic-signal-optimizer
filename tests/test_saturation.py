"""Unit tests for dynamic saturation flow headway calibration."""

from __future__ import annotations

import pytest

from src.analysis.saturation import SaturationFlowCalibrator


def test_saturation_calibrator_uncalibrated_fallback():
    calibrator = SaturationFlowCalibrator(base_saturation_flow=1900.0)
    est = calibrator.get_saturation_flow("lane_1")

    assert est.is_calibrated is False
    assert est.calibrated_saturation_pcu_h == 1900.0
    assert est.sample_count == 0


def test_saturation_calibrator_records_headways():
    calibrator = SaturationFlowCalibrator(base_saturation_flow=1900.0)

    # Vehicles 1, 2, 3 depart during startup lost time
    calibrator.record_departure("lane_1", timestamp=2.0, vehicle_in_queue_index=1)
    calibrator.record_departure("lane_1", timestamp=4.5, vehicle_in_queue_index=2)
    calibrator.record_departure("lane_1", timestamp=6.5, vehicle_in_queue_index=3)

    # Vehicles 4, 5, 6, 7 depart with steady 1.8s headway (3600 / 1.8 = 2000 PCU/h)
    calibrator.record_departure("lane_1", timestamp=8.3, vehicle_in_queue_index=4)  # headway 1.8s
    calibrator.record_departure("lane_1", timestamp=10.1, vehicle_in_queue_index=5) # headway 1.8s
    calibrator.record_departure("lane_1", timestamp=11.9, vehicle_in_queue_index=6) # headway 1.8s
    calibrator.record_departure("lane_1", timestamp=13.7, vehicle_in_queue_index=7) # headway 1.8s

    est = calibrator.get_saturation_flow("lane_1")
    assert est.is_calibrated is True
    assert est.sample_count >= 3
    # Calibrated saturation should adapt towards 2000 PCU/h
    assert 1900.0 < est.calibrated_saturation_pcu_h <= 2050.0
    assert pytest.approx(est.mean_headway_s, abs=0.15) == 1.8
