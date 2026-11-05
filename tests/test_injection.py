"""Unit tests for SUMO vehicle injection, type catalog, and trace route export."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import config
from src.sumo.injection import (
    APPROACH_EDGES,
    CATALOG_VEHICLE_TYPES,
    VehicleInjector,
    VehicleTypeSpec,
    normalize_approach,
    register_vehicle_types,
    setup_approach_routes,
)
from src.sumo.trace_route import (
    create_trace_sumocfg,
    export_trace_routes,
    generate_trace_routes_xml,
)


def test_normalize_approach():
    # Direct matches
    assert normalize_approach("N") == "N"
    assert normalize_approach("S") == "S"
    assert normalize_approach("E") == "E"
    assert normalize_approach("W") == "W"

    # Edge matches
    assert normalize_approach("N2C") == "N"
    assert normalize_approach("s2c") == "S"
    assert normalize_approach("E2C") == "E"
    assert normalize_approach("W2C") == "W"

    # Cardinal names
    assert normalize_approach("North") == "N"
    assert normalize_approach("southbound") == "S"
    assert normalize_approach("East_Lane_0") == "E"
    assert normalize_approach("West_Approach") == "W"

    # Letter aliases
    assert normalize_approach("Approach A") == "N"
    assert normalize_approach("Approach B") == "S"
    assert normalize_approach("Approach C") == "E"
    assert normalize_approach("Approach D") == "W"

    # Fallback
    assert normalize_approach("") == "N"
    assert normalize_approach(None) == "N"


def test_catalog_vehicle_types():
    for target in config.TARGET_CLASS_NAMES:
        assert target in CATALOG_VEHICLE_TYPES
        spec = CATALOG_VEHICLE_TYPES[target]
        assert isinstance(spec, VehicleTypeSpec)
        assert spec.length > 0.0
        assert spec.max_speed > 0.0
        assert spec.pcu > 0.0
        assert len(spec.color) == 4

    # Bus should be longer than car
    assert CATALOG_VEHICLE_TYPES["bus"].length > CATALOG_VEHICLE_TYPES["car"].length
    # Motorcycle PCU should be 0.5
    assert CATALOG_VEHICLE_TYPES["motorcycle"].pcu == 0.5


def test_mock_vehicle_injector():
    mock_conn = MagicMock()
    mock_conn.vehicletype.getIDList.return_value = ["DEFAULT_VEHTYPE"]
    mock_conn.route.getIDList.return_value = []

    injector = VehicleInjector(conn=mock_conn, auto_setup=True)

    # Check setup calls
    assert mock_conn.vehicletype.copy.called
    assert mock_conn.route.add.called

    # Test vehicle injection
    success, veh_id = injector.inject_vehicle(
        track_id=42, class_name="bus", approach="East", turn="through"
    )
    assert success is True
    assert "cam_E_42" in veh_id

    mock_conn.vehicle.add.assert_called_with(
        vehID=veh_id,
        routeID="route_E_through",
        typeID="vtype_bus",
        depart="now",
        departLane="best",
        departPos="base",
        departSpeed="max",
    )

    metrics = injector.get_metrics()
    assert metrics["injected_count"] == 1
    assert metrics["injected_by_approach"]["E"] == 1
    assert metrics["injected_by_class"]["bus"] == 1

    # Inject from event
    class FakeEvent:
        track_id = 99
        approach = "North"
        class_name = "motorcycle"

    success_evt, veh_evt = injector.inject_from_event(FakeEvent())
    assert success_evt is True
    assert "cam_N_99" in veh_evt

    # Reset
    injector.reset()
    assert injector.injected_count == 0


def test_trace_route_xml_generation():
    departures = [
        {"depart_time": 1.5, "approach": "North", "class_name": "car", "track_id": 1},
        {"depart_time": 4.0, "approach": "South", "class_name": "bus", "track_id": 2, "turn": "left"},
    ]

    xml = generate_trace_routes_xml(departures, duration=30.0)
    root = ET.fromstring(xml)

    vtypes = root.findall("vType")
    assert len(vtypes) >= len(CATALOG_VEHICLE_TYPES)

    routes = root.findall("route")
    assert len(routes) == 8  # 4 approaches * 2 turns

    vehicles = root.findall("vehicle")
    assert len(vehicles) == 2
    assert vehicles[0].attrib["type"] == "vtype_car"
    assert vehicles[0].attrib["depart"] == "1.50"
    assert vehicles[1].attrib["type"] == "vtype_bus"
    assert vehicles[1].attrib["route"] == "route_S_left"


def test_trace_route_export_and_sumocfg():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        rou_file = tmp_path / "test.rou.xml"
        cfg_file = tmp_path / "test.sumocfg"
        net_file = tmp_path / "test.net.xml"
        net_file.write_text("<net/>", encoding="utf-8")

        departures = [
            {"depart_time": 0.5, "approach": "W", "class_name": "truck", "track_id": 10},
        ]

        out_rou = export_trace_routes(departures, rou_file)
        assert out_rou.exists()
        assert "<vehicle" in out_rou.read_text(encoding="utf-8")

        out_cfg = create_trace_sumocfg(net_file, rou_file, cfg_file, duration=100.0)
        assert out_cfg.exists()
        cfg_text = out_cfg.read_text(encoding="utf-8")
        assert "test.net.xml" in cfg_text
        assert "test.rou.xml" in cfg_text


def test_live_sumo_injection():
    """Live headless TraCI integration test."""
    from src.sumo.env import SumoIntersectionEnv

    env = SumoIntersectionEnv(duration=15.0)
    obs, _ = env.reset(seed=42)

    injector = VehicleInjector(conn=env.conn, auto_setup=True)

    # Inject a car, a bus, and a motorcycle
    s1, id1 = injector.inject_vehicle(track_id=101, class_name="car", approach="North", turn="through")
    s2, id2 = injector.inject_vehicle(track_id=102, class_name="bus", approach="East", turn="left")
    s3, id3 = injector.inject_vehicle(track_id=103, class_name="motorcycle", approach="South", turn="through")

    assert s1 is True and s2 is True and s3 is True

    # Advance simulation steps
    for _ in range(5):
        env.conn.simulationStep()

    active_vehs = env.conn.vehicle.getIDList()
    assert id1 in active_vehs or id2 in active_vehs or id3 in active_vehs

    metrics = injector.get_metrics()
    assert metrics["injected_count"] == 3
    assert metrics["injected_by_approach"]["N"] == 1
    assert metrics["injected_by_approach"]["E"] == 1
    assert metrics["injected_by_approach"]["S"] == 1

    env.close()
