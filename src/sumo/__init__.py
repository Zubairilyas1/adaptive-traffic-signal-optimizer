"""SUMO simulation: network building, environment, injection, co-simulation, and control."""

from src.sumo.corridor_network import (
    ArterialCorridorConfig,
    CorridorJunctionSpec,
    default_corridor_junctions,
    generate_corridor_network,
)
from src.sumo.nema_network import (
    DEFAULT_8PHASE_DEMAND,
    NEMA_8_MOVEMENTS,
    SIM_NEMA_DIR,
    build_nema_8phase_intersection,
    build_nema_tls_state_string,
    find_sumo_binary,
)
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
    "SIM_NEMA_DIR",
    "NEMA_8_MOVEMENTS",
    "DEFAULT_8PHASE_DEMAND",
    "find_sumo_binary",
    "build_nema_tls_state_string",
    "build_nema_8phase_intersection",
    "ArterialCorridorConfig",
    "CorridorJunctionSpec",
    "default_corridor_junctions",
    "generate_corridor_network",
]
