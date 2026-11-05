"""Dynamic TraCI vehicle injection and vehicle type catalog for SUMO.

Maps edge perception (YOLOv8 + ByteTrack detections and ZoneTracker arrival events)
into real-time SUMO vehicle insertions on incoming intersection edges.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src import config

logger = logging.getLogger(__name__)

# Standard edge names from src/sumo/network.py
APPROACH_EDGES: dict[str, str] = {
    "N": "N2C",
    "S": "S2C",
    "E": "E2C",
    "W": "W2C",
}

# Outgoing target edges from src/sumo/network.py: (through, left)
APPROACH_TURNS: dict[str, tuple[str, str]] = {
    "N": ("C2S", "C2E"),
    "S": ("C2N", "C2W"),
    "E": ("C2W", "C2N"),
    "W": ("C2E", "C2S"),
}


@dataclass(frozen=True)
class VehicleTypeSpec:
    """Specification for a SUMO vehicle type (vType)."""

    type_id: str
    vclass: str  # SUMO vehicle class (e.g. "passenger", "bus", "truck", "motorcycle", "bicycle")
    length: float  # Length in meters
    max_speed: float  # Max speed in m/s
    accel: float  # Acceleration capability in m/s^2
    decel: float  # Deceleration capability in m/s^2
    color: tuple[int, int, int, int]  # RGBA color (0-255)
    pcu: float  # Passenger Car Unit factor


# Canonical vehicle type specifications matching src/config.py
CATALOG_VEHICLE_TYPES: dict[str, VehicleTypeSpec] = {
    "car": VehicleTypeSpec(
        type_id="vtype_car",
        vclass="passenger",
        length=4.5,
        max_speed=13.89,  # ~50 km/h
        accel=2.6,
        decel=4.5,
        color=(0, 240, 255, 255),  # Cyan
        pcu=1.0,
    ),
    "motorcycle": VehicleTypeSpec(
        type_id="vtype_motorcycle",
        vclass="motorcycle",
        length=2.2,
        max_speed=14.5,  # ~52 km/h
        accel=3.0,
        decel=5.0,
        color=(255, 255, 0, 255),  # Yellow
        pcu=0.5,
    ),
    "autorickshaw": VehicleTypeSpec(
        type_id="vtype_autorickshaw",
        vclass="passenger",
        length=3.0,
        max_speed=10.0,  # ~36 km/h
        accel=2.0,
        decel=3.5,
        color=(0, 255, 157, 255),  # Emerald Green
        pcu=1.5,
    ),
    "van": VehicleTypeSpec(
        type_id="vtype_van",
        vclass="passenger",
        length=5.5,
        max_speed=13.0,
        accel=2.2,
        decel=4.0,
        color=(180, 100, 255, 255),  # Purple
        pcu=1.5,
    ),
    "bus": VehicleTypeSpec(
        type_id="vtype_bus",
        vclass="bus",
        length=12.0,
        max_speed=10.0,
        accel=1.5,
        decel=3.5,
        color=(255, 170, 0, 255),  # Amber
        pcu=2.5,
    ),
    "truck": VehicleTypeSpec(
        type_id="vtype_truck",
        vclass="truck",
        length=10.0,
        max_speed=11.1,  # ~40 km/h
        accel=1.5,
        decel=3.5,
        color=(255, 128, 0, 255),  # Orange
        pcu=2.5,
    ),
    "tractor": VehicleTypeSpec(
        type_id="vtype_tractor",
        vclass="truck",
        length=4.2,
        max_speed=7.0,
        accel=1.0,
        decel=2.5,
        color=(160, 160, 160, 255),  # Grey
        pcu=3.0,
    ),
    "bicycle": VehicleTypeSpec(
        type_id="vtype_bicycle",
        vclass="bicycle",
        length=1.8,
        max_speed=5.0,
        accel=1.2,
        decel=2.0,
        color=(50, 205, 50, 255),  # Lime Green
        pcu=0.3,
    ),
}


def normalize_approach(approach: str) -> str:
    """Normalize arbitrary approach labels into canonical SUMO approaches ('N', 'S', 'E', 'W').

    Handles inputs like 'North', 'Northbound', 'N2C', 'Approach A', 'Lane_N_1', etc.
    """
    if not approach:
        return "N"
    clean = str(approach).strip().upper()

    # Direct single-letter match
    if clean in ("N", "S", "E", "W"):
        return clean

    # Edge match e.g. "N2C" -> "N"
    if len(clean) >= 3 and clean[1:3] == "2C" and clean[0] in ("N", "S", "E", "W"):
        return clean[0]

    # Explicit multi-word patterns: Approach A/B/C/D
    if "APPROACH A" in clean:
        return "N"
    if "APPROACH B" in clean:
        return "S"
    if "APPROACH C" in clean:
        return "E"
    if "APPROACH D" in clean:
        return "W"

    # Cardinal words
    if clean.startswith("NORTH") or clean.startswith("NB") or clean.startswith("N_") or clean.startswith("N-"):
        return "N"
    if clean.startswith("SOUTH") or clean.startswith("SB") or clean.startswith("S_") or clean.startswith("S-"):
        return "S"
    if clean.startswith("EAST") or clean.startswith("EB") or clean.startswith("E_") or clean.startswith("E-"):
        return "E"
    if clean.startswith("WEST") or clean.startswith("WB") or clean.startswith("W_") or clean.startswith("W-"):
        return "W"

    # Tokenized search
    tokens = clean.replace("_", " ").replace("-", " ").split()
    for tok in tokens:
        if tok in ("N", "NORTH", "NORTHBOUND", "A"):
            return "N"
        if tok in ("S", "SOUTH", "SOUTHBOUND", "B"):
            return "S"
        if tok in ("E", "EAST", "EASTBOUND", "C"):
            return "E"
        if tok in ("W", "WEST", "WESTBOUND", "D"):
            return "W"

    if clean.startswith("N"):
        return "N"
    if clean.startswith("S"):
        return "S"
    if clean.startswith("E"):
        return "E"
    if clean.startswith("W"):
        return "W"

    return "N"


def register_vehicle_types(conn: Any, types: dict[str, VehicleTypeSpec] | None = None) -> list[str]:
    """Register custom vehicle types into an active TraCI connection.

    Uses `DEFAULT_VEHTYPE` as template to inherit standard car-following models.
    """
    types = types or CATALOG_VEHICLE_TYPES
    registered: list[str] = []

    try:
        existing_types = set(conn.vehicletype.getIDList())
    except Exception as exc:
        logger.warning(f"Could not retrieve existing vehicle types: {exc}")
        existing_types = set()

    for key, spec in types.items():
        vtype_id = spec.type_id
        try:
            if vtype_id not in existing_types:
                conn.vehicletype.copy("DEFAULT_VEHTYPE", vtype_id)
                existing_types.add(vtype_id)

            conn.vehicletype.setLength(vtype_id, float(spec.length))
            conn.vehicletype.setMaxSpeed(vtype_id, float(spec.max_speed))
            conn.vehicletype.setAccel(vtype_id, float(spec.accel))
            conn.vehicletype.setDecel(vtype_id, float(spec.decel))
            conn.vehicletype.setColor(vtype_id, spec.color)
            try:
                conn.vehicletype.setVehicleClass(vtype_id, spec.vclass)
            except Exception:
                pass  # Ignore if specific vclass is unsupported by current net
            registered.append(vtype_id)
        except Exception as exc:
            logger.warning(f"Failed to configure vType {vtype_id}: {exc}")

    return registered


def setup_approach_routes(conn: Any) -> list[str]:
    """Ensure standard through and left-turn routes exist for all 4 approaches."""
    created_routes: list[str] = []
    try:
        existing_routes = set(conn.route.getIDList())
    except Exception as exc:
        logger.warning(f"Could not retrieve existing routes: {exc}")
        existing_routes = set()

    for app, inc in APPROACH_EDGES.items():
        through_out, left_out = APPROACH_TURNS[app]
        route_through = f"route_{app}_through"
        route_left = f"route_{app}_left"

        for rid, edges in [(route_through, [inc, through_out]), (route_left, [inc, left_out])]:
            if rid not in existing_routes:
                try:
                    conn.route.add(rid, edges)
                    existing_routes.add(rid)
                    created_routes.append(rid)
                except Exception as exc:
                    logger.warning(f"Could not add route {rid}: {exc}")

    return created_routes


class VehicleInjector:
    """Manages real-time dynamic vehicle injection into SUMO via TraCI."""

    def __init__(self, conn: Any = None, auto_setup: bool = True) -> None:
        self.conn = conn
        self.auto_setup = auto_setup
        self._spawn_counter: int = 0
        self._initialized: bool = False

        # Operational metrics
        self.injected_count: int = 0
        self.failures_count: int = 0
        self.injected_by_approach: dict[str, int] = {a: 0 for a in APPROACH_EDGES}
        self.injected_by_class: dict[str, int] = {k: 0 for k in CATALOG_VEHICLE_TYPES}

        if self.conn is not None and self.auto_setup:
            self.setup()

    def bind_connection(self, conn: Any) -> None:
        """Bind or rebind a TraCI connection (e.g. after simulation reset)."""
        self.conn = conn
        self._initialized = False
        if self.auto_setup and self.conn is not None:
            self.setup()

    def setup(self) -> None:
        """Initialize vehicle types and routes in SUMO."""
        if self.conn is None:
            return
        register_vehicle_types(self.conn)
        setup_approach_routes(self.conn)
        self._initialized = True

    def resolve_vtype(self, class_name: str) -> str:
        """Resolve detection class string to a registered SUMO vType ID."""
        canonical = config.canonical_class(class_name)
        if canonical in CATALOG_VEHICLE_TYPES:
            return CATALOG_VEHICLE_TYPES[canonical].type_id
        return CATALOG_VEHICLE_TYPES["car"].type_id

    def inject_vehicle(
        self,
        track_id: int | str,
        class_name: str = "car",
        approach: str = "N",
        turn: str = "random",
        depart_lane: str = "best",
        depart_speed: str | float = "max",
        depart_pos: str = "base",
    ) -> tuple[bool, str]:
        """Dynamically inject a single detected vehicle into the SUMO network.

        Parameters
        ----------
        track_id : int | str
            Unique vehicle tracking ID from ByteTrack.
        class_name : str
            Vehicle class (car, bus, motorcycle, truck, etc.).
        approach : str
            Incoming approach (North, South, East, West, or raw ROI label).
        turn : str
            Movement direction: 'through', 'left', or 'random' (70% through / 30% left).
        depart_lane : str
            SUMO departure lane ('first', 'best', 'random', or specific index '0', '1').
        depart_speed : str | float
            Departure speed in m/s, or 'max', 'random', 'desired'.
        depart_pos : str
            Departure position on incoming edge ('base', 'random', 'free').

        Returns
        -------
        tuple[bool, str]
            (success, vehicle_id_or_error_message)
        """
        if self.conn is None:
            self.failures_count += 1
            return False, "TraCI connection not bound"

        if not self._initialized:
            self.setup()

        app = normalize_approach(approach)
        vtype = self.resolve_vtype(class_name)

        # Decide route
        if turn == "through":
            route_id = f"route_{app}_through"
        elif turn == "left":
            route_id = f"route_{app}_left"
        else:
            # 70% straight through, 30% turn
            route_id = f"route_{app}_through" if random.random() < 0.70 else f"route_{app}_left"

        self._spawn_counter += 1
        veh_id = f"cam_{app}_{track_id}_{self._spawn_counter}"

        try:
            self.conn.vehicle.add(
                vehID=veh_id,
                routeID=route_id,
                typeID=vtype,
                depart="now",
                departLane=depart_lane,
                departPos=depart_pos,
                departSpeed=str(depart_speed),
            )
            self.injected_count += 1
            self.injected_by_approach[app] += 1
            canonical_cls = config.canonical_class(class_name)
            self.injected_by_class[canonical_cls] = self.injected_by_class.get(canonical_cls, 0) + 1
            return True, veh_id
        except Exception as exc:
            self.failures_count += 1
            logger.debug(f"Vehicle injection rejected by TraCI for {veh_id}: {exc}")
            return False, str(exc)

    def inject_from_event(
        self,
        event: Any,
        class_name: str = "car",
        approach: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Convenience method to inject a vehicle directly from a ZoneEvent or detection dict."""
        track_id = getattr(event, "track_id", None)
        if track_id is None and isinstance(event, dict):
            track_id = event.get("track_id", self._spawn_counter)
        if track_id is None:
            track_id = self._spawn_counter

        app = approach or getattr(event, "approach", None) or getattr(event, "lane_id", "N")
        cls_name = getattr(event, "class_name", class_name)
        return self.inject_vehicle(track_id=track_id, class_name=cls_name, approach=app)

    def get_metrics(self) -> dict[str, Any]:
        """Return cumulative injection telemetry."""
        return {
            "injected_count": self.injected_count,
            "failures_count": self.failures_count,
            "injected_by_approach": dict(self.injected_by_approach),
            "injected_by_class": dict(self.injected_by_class),
        }

    def reset(self) -> None:
        """Reset internal metrics and counters."""
        self._spawn_counter = 0
        self.injected_count = 0
        self.failures_count = 0
        self.injected_by_approach = {a: 0 for a in APPROACH_EDGES}
        self.injected_by_class = {k: 0 for k in CATALOG_VEHICLE_TYPES}
