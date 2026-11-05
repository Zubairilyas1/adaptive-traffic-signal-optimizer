"""SUMO simulation: network building, environment, injection, co-simulation, and control."""

from src.sumo.network import build_intersection, SIM_DIR, APPROACHES
from src.sumo.env import SumoIntersectionEnv
from src.sumo.injection import (
    VehicleInjector,
    VehicleTypeSpec,
    CATALOG_VEHICLE_TYPES,
    normalize_approach,
    register_vehicle_types,
    setup_approach_routes,
)
from src.sumo.trace_route import export_trace_routes, create_trace_sumocfg
from src.sumo.co_simulation import (
    DigitalTwinTelemetry,
    DigitalTwinSynchronizer,
    SumoSyncWorker,
)

__all__ = [
    "build_intersection",
    "SIM_DIR",
    "APPROACHES",
    "SumoIntersectionEnv",
    "VehicleInjector",
    "VehicleTypeSpec",
    "CATALOG_VEHICLE_TYPES",
    "normalize_approach",
    "register_vehicle_types",
    "setup_approach_routes",
    "export_trace_routes",
    "create_trace_sumocfg",
    "DigitalTwinTelemetry",
    "DigitalTwinSynchronizer",
    "SumoSyncWorker",
]
