"""Live Closed-Loop Digital Twin: Video-to-SUMO Co-Simulation Synchronizer.

Coordinates real-world perception with SUMO micro-simulation:
- Advances SUMO simulation in lockstep with video playback time
- Extracts real-time ground-truth queues and operational metrics via TraCI
- Calculates perception-simulation divergence index (MAE)
- Supports both synchronous step-by-step and asynchronous threaded execution
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.sumo.env import APPROACHES, SumoIntersectionEnv
from src.sumo.injection import VehicleInjector, normalize_approach

logger = logging.getLogger(__name__)


@dataclass
class DigitalTwinTelemetry:
    """Instantaneous snapshot of perception-simulation co-existence."""

    timestamp: float  # Video or wall-clock timestamp (s)
    sim_time: float  # SUMO simulation time (s)
    queues_sumo: dict[str, int]  # Approach -> halting vehicle count in SUMO
    queues_cam: dict[str, float]  # Approach -> measured camera PCU queue
    mean_speeds: dict[str, float]  # Approach -> average approach speed (km/h)
    waiting_time_s: float  # Total edge waiting time (veh-seconds)
    throughput: int  # Total departed/arrived vehicles
    active_phase: int  # 0: North-South Green, 1: East-West Green
    active_aspect: str  # "GREEN", "YELLOW", "ALL_RED"
    tls_state: str  # Raw TraCI TLS link string (e.g. "GGGGnnnn")
    divergence_mae: float  # Mean Absolute Error between camera and SUMO queues
    active_vehicles: int  # Vehicles currently active in SUMO
    injected_count: int  # Cumulative camera vehicles injected

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DigitalTwinSynchronizer:
    """Synchronous coordinator binding camera perception to SUMO simulation."""

    def __init__(
        self,
        demand: Optional[dict[str, float]] = None,
        duration: float = 3600.0,
        step_size: float = 1.0,
        min_green: float = 10.0,
        yellow: float = 3.0,
        gui: bool = False,
        seed: int = 42,
    ) -> None:
        self.demand = demand
        self.duration = duration
        self.step_size = step_size
        self.min_green = min_green
        self.yellow = yellow
        self.gui = gui
        self.seed = seed

        self.env: Optional[SumoIntersectionEnv] = None
        self.injector: Optional[VehicleInjector] = None
        self.latest_telemetry: Optional[DigitalTwinTelemetry] = None

        self._sim_time: float = 0.0
        self._video_time: float = 0.0
        self._last_cam_queues: dict[str, float] = {a: 0.0 for a in APPROACHES}
        self._running: bool = False

    @property
    def is_running(self) -> bool:
        return self._running and self.env is not None and self.env.conn is not None

    def start(self, seed: Optional[int] = None) -> DigitalTwinTelemetry:
        """Start or restart the SUMO simulation and initialize the vehicle injector."""
        if seed is not None:
            self.seed = seed

        if self.env is not None:
            self.close()

        # Build SumoIntersectionEnv with delta_time=step_size
        self.env = SumoIntersectionEnv(
            demand=self.demand,
            duration=self.duration,
            delta_time=self.step_size,
            min_green=self.min_green,
            yellow=self.yellow,
            seed=self.seed,
            gui=self.gui,
            label="digital_twin",
        )
        self.env.reset(seed=self.seed)

        # Bind vehicle injector to active TraCI connection
        self.injector = VehicleInjector(conn=self.env.conn, auto_setup=True)
        self._sim_time = self.env._sim_time
        self._video_time = 0.0
        self._running = True

        self.latest_telemetry = self._extract_telemetry(timestamp=0.0)
        return self.latest_telemetry

    def close(self) -> None:
        """Safely terminate the SUMO TraCI connection."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception as exc:
                logger.warning(f"Error while closing SUMO environment: {exc}")
            self.env = None
        self.injector = None
        self._running = False

    def inject(
        self,
        track_id: int | str,
        class_name: str = "car",
        approach: str = "N",
        turn: str = "random",
    ) -> tuple[bool, str]:
        """Inject a detected camera vehicle into the SUMO network."""
        if not self.is_running or self.injector is None:
            return False, "Simulation not running"
        return self.injector.inject_vehicle(
            track_id=track_id,
            class_name=class_name,
            approach=approach,
            turn=turn,
        )

    def step(
        self,
        video_time: Optional[float] = None,
        camera_queues: Optional[dict[str, float]] = None,
        action: Optional[int] = None,
    ) -> DigitalTwinTelemetry:
        """Advance simulation in synchronization with video time.

        Parameters
        ----------
        video_time : Optional[float]
            Current video playback timestamp in seconds.
        camera_queues : Optional[dict[str, float]]
            Current camera-estimated PCU queue per approach.
        action : Optional[int]
            0 = hold green phase, 1 = switch phase (with yellow clearance).
        """
        if not self.is_running:
            raise RuntimeError("Simulation not running. Call start() first.")

        if camera_queues is not None:
            for k, v in camera_queues.items():
                norm_k = normalize_approach(k)
                self._last_cam_queues[norm_k] = float(v)

        target_time = video_time if video_time is not None else (self._sim_time + self.step_size)
        self._video_time = target_time

        # If action is specified, apply through env step logic
        if action is not None:
            self.env.step(action)
            self._sim_time = self.env._sim_time
        else:
            # Advance simulation step by step until aligned with target video time
            while self._sim_time < target_time:
                self.env.conn.simulationStep()
                self._sim_time += 1.0
                self.env._sim_time = self._sim_time
                self.env._green_elapsed += 1.0

        self.latest_telemetry = self._extract_telemetry(timestamp=self._video_time)
        return self.latest_telemetry

    def _extract_telemetry(self, timestamp: float) -> DigitalTwinTelemetry:
        """Query real-time ground-truth state from SUMO via TraCI."""
        conn = self.env.conn
        approach_ids = list(APPROACHES)

        sumo_queues: dict[str, int] = {}
        mean_speeds: dict[str, float] = {}
        total_waiting_time = 0.0

        for a in approach_ids:
            inc_edge = APPROACHES[a][0]
            try:
                q_val = int(conn.edge.getLastStepHaltingNumber(inc_edge))
                spd_mps = float(conn.edge.getLastStepMeanSpeed(inc_edge))
                wt = float(conn.edge.getWaitingTime(inc_edge))
            except Exception:
                q_val = 0
                spd_mps = 0.0
                wt = 0.0

            sumo_queues[a] = q_val
            mean_speeds[a] = round(spd_mps * 3.6, 1)  # m/s to km/h
            total_waiting_time += wt

        try:
            throughput = int(conn.simulation.getArrivedNumber())
            active_vehs = len(conn.vehicle.getIDList())
            tls_state = str(conn.trafficlight.getRedYellowGreenState(self.env.tls_id))
        except Exception:
            throughput = 0
            active_vehs = 0
            tls_state = ""

        # Determine active aspect
        if "y" in tls_state.lower():
            active_aspect = "YELLOW"
        elif "g" in tls_state.lower():
            active_aspect = "GREEN"
        else:
            active_aspect = "ALL_RED"

        # Calculate Divergence MAE between Camera PCU and SUMO Halting Count
        diffs = [
            abs(float(self._last_cam_queues.get(a, 0.0)) - float(sumo_queues.get(a, 0)))
            for a in approach_ids
        ]
        divergence_mae = float(np.mean(diffs)) if diffs else 0.0

        injected = self.injector.injected_count if self.injector else 0

        return DigitalTwinTelemetry(
            timestamp=round(timestamp, 2),
            sim_time=round(self._sim_time, 2),
            queues_sumo=sumo_queues,
            queues_cam={a: round(float(self._last_cam_queues.get(a, 0.0)), 2) for a in approach_ids},
            mean_speeds=mean_speeds,
            waiting_time_s=round(total_waiting_time, 1),
            throughput=throughput,
            active_phase=self.env.phase,
            active_aspect=active_aspect,
            tls_state=tls_state,
            divergence_mae=round(divergence_mae, 2),
            active_vehicles=active_vehs,
            injected_count=injected,
        )


class SumoSyncWorker(threading.Thread):
    """Threaded background worker for non-blocking asynchronous Digital Twin execution."""

    def __init__(
        self,
        demand: Optional[dict[str, float]] = None,
        duration: float = 3600.0,
        step_rate_hz: float = 1.0,
        min_green: float = 10.0,
        yellow: float = 3.0,
        seed: int = 42,
    ) -> None:
        super().__init__(daemon=True, name="SumoSyncWorker")
        self.demand = demand
        self.duration = duration
        self.step_rate_hz = max(0.2, float(step_rate_hz))
        self.step_interval = 1.0 / self.step_rate_hz
        self.min_green = min_green
        self.yellow = yellow
        self.seed = seed

        self.synchronizer = DigitalTwinSynchronizer(
            demand=self.demand,
            duration=self.duration,
            step_size=1.0,
            min_green=self.min_green,
            yellow=self.yellow,
            seed=self.seed,
        )

        self._inbox: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest_telemetry: Optional[DigitalTwinTelemetry] = None
        self._running: bool = False

    @property
    def is_running(self) -> bool:
        return self._running and not self._stop_event.is_set()

    def post_injection(
        self,
        track_id: int | str,
        class_name: str = "car",
        approach: str = "N",
        turn: str = "random",
    ) -> None:
        """Schedule a dynamic vehicle injection asynchronously."""
        self._inbox.put(("inject", {
            "track_id": track_id,
            "class_name": class_name,
            "approach": approach,
            "turn": turn,
        }))

    def post_camera_queues(self, camera_queues: dict[str, float]) -> None:
        """Update the perceived camera PCU queues."""
        self._inbox.put(("cam_queues", dict(camera_queues)))

    def post_action(self, action: int) -> None:
        """Schedule a traffic light action (0 keep, 1 switch)."""
        self._inbox.put(("action", int(action)))

    def get_telemetry(self) -> Optional[DigitalTwinTelemetry]:
        """Retrieve the freshest telemetry snapshot in a thread-safe manner."""
        with self._lock:
            return self._latest_telemetry

    def run(self) -> None:
        """Main worker execution loop running inside dedicated thread."""
        logger.info("SumoSyncWorker thread starting...")
        try:
            init_telem = self.synchronizer.start(seed=self.seed)
            with self._lock:
                self._latest_telemetry = init_telem
            self._running = True
        except Exception as exc:
            logger.error(f"Failed to initialize SUMO in worker thread: {exc}")
            self._running = False
            return

        sim_clock = 0.0
        while not self._stop_event.is_set():
            t_start = time.perf_counter()

            # Process all pending messages in inbox
            pending_action: Optional[int] = None
            pending_queues: Optional[dict[str, float]] = None

            while not self._inbox.empty():
                try:
                    msg_type, payload = self._inbox.get_nowait()
                    if msg_type == "inject":
                        self.synchronizer.inject(**payload)
                    elif msg_type == "cam_queues":
                        pending_queues = payload
                    elif msg_type == "action":
                        pending_action = payload
                except queue.Empty:
                    break

            sim_clock += 1.0
            try:
                telem = self.synchronizer.step(
                    video_time=sim_clock,
                    camera_queues=pending_queues,
                    action=pending_action,
                )
                with self._lock:
                    self._latest_telemetry = telem
            except Exception as exc:
                logger.warning(f"Error during worker step: {exc}")
                break

            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.01, self.step_interval - elapsed)
            time.sleep(sleep_time)

        logger.info("SumoSyncWorker shutting down...")
        self.synchronizer.close()
        self._running = False

    def stop(self, timeout: float = 3.0) -> None:
        """Request graceful shutdown of the worker thread and join."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)
        self.synchronizer.close()
        self._running = False
