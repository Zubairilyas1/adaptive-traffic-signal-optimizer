"""Unit tests for road topologies, semantic presets, and resolution scaling."""

from __future__ import annotations

import pytest

from src import config
from src.analysis.roi import LaneROI, load_rois, scale_rois_to_resolution
from src.decision.intersection import load_intersection


def test_scale_rois_to_resolution():
    base_rois = [
        LaneROI(
            lane_id="Lane1",
            approach="North",
            polygon=[(100.0, 100.0), (300.0, 100.0), (300.0, 500.0), (100.0, 500.0)],
        )
    ]

    # Scale from 1280x720 to 1920x1080 (1.5x)
    scaled = scale_rois_to_resolution(
        base_rois,
        target_width=1920,
        target_height=1080,
        source_width=1280,
        source_height=720,
    )

    assert len(scaled) == 1
    p0 = scaled[0].polygon[0]
    p2 = scaled[0].polygon[2]

    assert p0 == (150.0, 150.0)
    assert p2 == (450.0, 750.0)


def test_all_intersection_topology_configs_valid():
    topologies = [
        "single_road.json",
        "twoway_corridor.json",
        "t_junction.json",
        "fourway_crossroad.json",
    ]

    for topo_file in topologies:
        path = config.ROOT / "configs" / "intersections" / topo_file
        assert path.exists(), f"Missing config: {topo_file}"
        inter = load_intersection(path)
        assert len(inter.approaches) >= 1
        assert len(inter.phases) >= 1
        assert inter.total_lost_time >= 0.0


def test_all_roi_presets_valid():
    presets = [
        "preset_single_road.json",
        "preset_twoway_corridor.json",
        "preset_t_junction.json",
        "preset_fourway_crossroad.json",
    ]

    for preset_file in presets:
        path = config.ROOT / "configs" / "rois" / preset_file
        assert path.exists(), f"Missing preset: {preset_file}"
        rois = load_rois(path)
        assert len(rois) >= 1
        for roi in rois:
            assert len(roi.polygon) == 4
            assert roi.lane_id
            assert roi.approach
