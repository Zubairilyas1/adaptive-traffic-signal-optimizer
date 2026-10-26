"""Gymnasium environment wrapping a SUMO 2-phase intersection (TraCI).

State  : per-approach halting queue (4) + current phase + elapsed green ratio.
Action : 0 = keep phase, 1 = switch (min-green and yellow enforced).
Reward : negative total halting queue (queue length -> delay proxy).

The environment is the substrate for all three controllers: fixed-time,
rule-based (actuated) and PPO.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from src import config
from src.sumo.network import APPROACHES, build_intersection

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None


def sumo_home() -> str:
    return str(Path(sys.prefix) / "Lib" / "site-packages" / "sumo")


def sumo_binary() -> str:
    return str(Path(sys.prefix) / "Lib" / "site-packages" / "sumo" / "bin" / "sumo.exe")


def _ensure_sumo_env() -> None:
    os.environ.setdefault("SUMO_HOME", sumo_home())
    binp = str(Path(sumo_home()) / "bin")
    if binp not in os.environ.get("PATH", ""):
        os.environ["PATH"] = binp + os.pathsep + os.environ.get("PATH", "")


class SumoIntersectionEnv(gym.Env if gym else object):
    """Two-phase adaptive signal control environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        demand: dict[str, float] | None = None,
        duration: float = 1200.0,
        delta_time: float = 5.0,
        min_green: float = 10.0,
        yellow: float = 3.0,
        seed: int = 42,
        gui: bool = False,
        label: str = "rl",
    ) -> None:
        _ensure_sumo_env()
        self.demand = demand
        self.duration = duration
        self.delta_time = delta_time
        self.min_green = min_green
        self.yellow = yellow
        self.seed = seed
        self.gui = gui
        self.label = label

        self.cfg = build_intersection(demand=demand, duration=duration, seed=seed)
        self.approach_ids = list(APPROACHES)
        self.incoming_edges = [APPROACHES[a][0] for a in self.approach_ids]

        if spaces is not None:
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(len(self.approach_ids) + 2,), dtype=np.float32
            )

        self.conn = None
        self.tls_id: Optional[str] = None
        self.link_is_ns: list[bool] = []
        self._states: dict[str, str] = {}
        self._phase = 0
        self._green_elapsed = 0.0
        self._sim_time = 0.0

    # ---------- state string construction ----------
    def _build_phase_states(self) -> None:
        tls = self.tls_id
        controlled = self.conn.trafficlight.getControlledLinks(tls)
        self.link_is_ns = []
        for link in controlled:
            from_lane = link[0][0] if link else ""
            edge = from_lane.rsplit("_", 1)[0]
            self.link_is_ns.append(edge[:1] in ("N", "S"))
        self._states["ns_green"] = "".join("G" if x else "r" for x in self.link_is_ns)
        self._states["ew_green"] = "".join("r" if x else "G" for x in self.link_is_ns)
        self._states["ns_yellow"] = "".join("y" if x else "r" for x in self.link_is_ns)
        self._states["ew_yellow"] = "".join("r" if x else "y" for x in self.link_is_ns)

    def _apply(self, key: str) -> None:
        self.conn.trafficlight.setRedYellowGreenState(self.tls_id, self._states[key])

    # ---------- observation ----------
    def _queues(self) -> np.ndarray:
        q = [self.conn.edge.getLastStepHaltingNumber(edge) for edge in self.incoming_edges]
        return np.asarray(q, dtype=np.float32)

    def _observation(self) -> np.ndarray:
        q = self._queues()
        q_norm = np.clip(q / 20.0, 0.0, 1.0)
        phase = np.float32(self._phase)
        elapsed = np.float32(min(1.0, self._green_elapsed / 60.0))
        return np.concatenate([q_norm, [phase, elapsed]]).astype(np.float32)

    def _reward(self) -> float:
        return -float(self._queues().sum())

    # ---------- gym API ----------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.seed = seed
        if self.conn is not None and self.conn.simulation.getTime() > 0:
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass

        cmd = [sumo_binary(), "-c", str(self.cfg), "--no-step-log", "true",
               "--no-warnings", "true", "--time-to-teleport", "-1",
               "--seed", str(self.seed)]
        if self.gui:
            cmd[0] = sumo_binary().replace("sumo.exe", "sumo-gui.exe")

        import traci
        traci.start(cmd)
        self.conn = traci
        self.tls_id = self.conn.trafficlight.getIDList()[0]
        self._build_phase_states()

        self._phase = 0
        self._green_elapsed = 0.0
        self._sim_time = 0.0
        self._apply("ns_green")
        self.conn.simulationStep()
        return self._observation(), {}

    def step(self, action: int):
        switched = False
        if action == 1 and self._green_elapsed >= self.min_green:
            self._apply("ns_yellow" if self._phase == 0 else "ew_yellow")
            for _ in range(int(self.yellow)):
                self.conn.simulationStep()
                self._sim_time += 1.0
            self._phase = 1 - self._phase
            self._apply("ns_green" if self._phase == 0 else "ew_green")
            self._green_elapsed = 0.0
            switched = True

        for _ in range(int(self.delta_time)):
            self.conn.simulationStep()
            self._sim_time += 1.0
        self._green_elapsed += self.delta_time

        reward = self._reward()
        terminated = self._sim_time >= self.duration
        return self._observation(), reward, terminated, False, {
            "phase": self._phase, "switched": switched, "time": self._sim_time,
        }

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass
            self.conn = None

    # ---------- non-RL control helpers ----------
    def raw_queues(self) -> np.ndarray:
        return self._queues()

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def green_elapsed(self) -> float:
        return self._green_elapsed
