"""NEMA TS2 Dual-Ring 8-Phase Geometric SUMO Network Generator.

Builds an arterial 4-leg intersection with dedicated protected left-turn bays
and generates the complete microscopic SUMO network (.net.xml), routes (.rou.xml),
and configuration (.sumocfg) calibrated for 8 distinct vehicular turning movements.

Movement to NEMA Phase Mapping:
  - Phase 1 (SB Left): S2C (Lane 2) -> C2W
  - Phase 2 (NB Through + Right): N2C (Lane 1) -> C2S, N2C (Lane 0) -> C2W
  - Phase 3 (WB Left): W2C (Lane 2) -> C2N
  - Phase 4 (EB Through + Right): E2C (Lane 1) -> C2W, E2C (Lane 0) -> C2N
  - Phase 5 (NB Left): N2C (Lane 2) -> C2E
  - Phase 6 (SB Through + Right): S2C (Lane 1) -> C2N, S2C (Lane 0) -> C2E
  - Phase 7 (EB Left): E2C (Lane 2) -> C2S
  - Phase 8 (WB Through + Right): W2C (Lane 1) -> C2E, W2C (Lane 0) -> C2S
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src import config
from src.decision.nema_dual_ring import NEMAPhase, PhaseInterval

logger = logging.getLogger(__name__)

SIM_NEMA_DIR = config.ROOT / "sim" / "nema"

# 8 distinct vehicular movements: Phase -> (Incoming Edge, Incoming Lane, Outgoing Edge)
NEMA_8_MOVEMENTS: dict[NEMAPhase, list[tuple[str, int, str]]] = {
    NEMAPhase.P1_SB_LEFT: [("S2C", 2, "C2W")],
    NEMAPhase.P2_NB_THROUGH: [("N2C", 0, "C2W"), ("N2C", 1, "C2S")],
    NEMAPhase.P3_WB_LEFT: [("W2C", 2, "C2N")],
    NEMAPhase.P4_EB_THROUGH: [("E2C", 0, "C2N"), ("E2C", 1, "C2W")],
    NEMAPhase.P5_NB_LEFT: [("N2C", 2, "C2E")],
    NEMAPhase.P6_SB_THROUGH: [("S2C", 0, "C2E"), ("S2C", 1, "C2N")],
    NEMAPhase.P7_EB_LEFT: [("E2C", 2, "C2S")],
    NEMAPhase.P8_WB_THROUGH: [("W2C", 0, "C2S"), ("W2C", 1, "C2E")],
}

DEFAULT_8PHASE_DEMAND: dict[NEMAPhase, float] = {
    NEMAPhase.P1_SB_LEFT: 250.0,
    NEMAPhase.P2_NB_THROUGH: 950.0,
    NEMAPhase.P3_WB_LEFT: 180.0,
    NEMAPhase.P4_EB_THROUGH: 600.0,
    NEMAPhase.P5_NB_LEFT: 220.0,
    NEMAPhase.P6_SB_THROUGH: 850.0,
    NEMAPhase.P7_EB_LEFT: 150.0,
    NEMAPhase.P8_WB_THROUGH: 550.0,
}


def find_sumo_binary(name: str) -> str:
    """Locates the SUMO executable across Windows venv, Linux PATH, and SUMO_HOME."""
    ext = ".exe" if sys.platform == "win32" else ""
    binary_name = f"{name}{ext}"

    # 1. Check system PATH
    found = shutil.which(name) or shutil.which(binary_name)
    if found:
        return found

    # 2. Check local Python virtual environment
    venv_bin = Path(sys.prefix) / "Lib" / "site-packages" / "sumo" / "bin" / binary_name
    if venv_bin.exists():
        return str(venv_bin)

    # 3. Check SUMO_HOME environment variable
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        home_bin = Path(sumo_home) / "bin" / binary_name
        if home_bin.exists():
            return str(home_bin)

    return binary_name


def _nema_nodes_xml() -> str:
    """Node definitions with central traffic light intersection."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="N" x="0" y="250" type="priority"/>
    <node id="S" x="0" y="-250" type="priority"/>
    <node id="E" x="250" y="0" type="priority"/>
    <node id="W" x="-250" y="0" type="priority"/>
    <node id="C" x="0" y="0" type="traffic_light"/>
</nodes>
"""


def _nema_edges_xml() -> str:
    """Multi-lane edge definitions with 3 incoming lanes (including dedicated left-turn bay)."""
    in_lanes = 'numLanes="3" speed="13.89"'   # Lane 0: Right, Lane 1: Thru, Lane 2: Left
    out_lanes = 'numLanes="2" speed="13.89"'  # 2 receiving lanes per approach

    in_edges = "\n".join(
        f'    <edge id="{src}2C" from="{src}" to="C" {in_lanes}/>'
        for src in ("N", "S", "E", "W")
    )
    out_edges = "\n".join(
        f'    <edge id="C2{dst}" from="C" to="{dst}" {out_lanes}/>'
        for dst in ("N", "S", "E", "W")
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<edges>
{in_edges}
{out_edges}
</edges>
"""


def _nema_connections_xml() -> str:
    """Explicit lane-to-lane turning connections enforcing dedicated left-turn bays."""
    rows: list[str] = [
        # North approach (N2C)
        '    <connection from="N2C" to="C2W" fromLane="0" toLane="0"/>',  # Right
        '    <connection from="N2C" to="C2S" fromLane="1" toLane="0"/>',  # Thru
        '    <connection from="N2C" to="C2E" fromLane="2" toLane="1"/>',  # Dedicated Left (P5)
        # South approach (S2C)
        '    <connection from="S2C" to="C2E" fromLane="0" toLane="0"/>',  # Right
        '    <connection from="S2C" to="C2N" fromLane="1" toLane="0"/>',  # Thru
        '    <connection from="S2C" to="C2W" fromLane="2" toLane="1"/>',  # Dedicated Left (P1)
        # East approach (E2C)
        '    <connection from="E2C" to="C2N" fromLane="0" toLane="0"/>',  # Right
        '    <connection from="E2C" to="C2W" fromLane="1" toLane="0"/>',  # Thru
        '    <connection from="E2C" to="C2S" fromLane="2" toLane="1"/>',  # Dedicated Left (P7)
        # West approach (W2C)
        '    <connection from="W2C" to="C2S" fromLane="0" toLane="0"/>',  # Right
        '    <connection from="W2C" to="C2E" fromLane="1" toLane="0"/>',  # Thru
        '    <connection from="W2C" to="C2N" fromLane="2" toLane="1"/>',  # Dedicated Left (P3)
    ]
    return '<?xml version="1.0" encoding="UTF-8"?>\n<connections>\n' + "\n".join(rows) + "\n</connections>\n"


def _nema_routes_xml(
    demand: dict[NEMAPhase, float],
    duration: float,
    seed: int = 42,
) -> str:
    """Generates calibrated multi-class flow streams across all 8 NEMA turning movements."""
    flows: list[str] = [
        '    <vType id="car" length="5.0" minGap="2.5" maxSpeed="13.89" accel="2.6" decel="4.5" vClass="passenger"/>',
        '    <vType id="bus" length="12.0" minGap="3.0" maxSpeed="11.11" accel="1.2" decel="3.0" vClass="bus"/>',
        '    <vType id="motorcycle" length="2.2" minGap="1.5" maxSpeed="16.67" accel="3.5" decel="5.0" vClass="motorcycle"/>',
    ]

    for phase, move_list in NEMA_8_MOVEMENTS.items():
        vph = float(demand.get(phase, 0.0))
        if vph <= 0.0:
            continue

        per_move_vph = vph / len(move_list)
        for idx, (inc, lane, out) in enumerate(move_list):
            flow_id = f"flow_P{phase.value}_{idx}"
            vtype = "car" if phase in (NEMAPhase.P1_SB_LEFT, NEMAPhase.P3_WB_LEFT, NEMAPhase.P5_NB_LEFT, NEMAPhase.P7_EB_LEFT) else "car"
            flows.append(
                f'    <flow id="{flow_id}" begin="0" end="{duration:.0f}" '
                f'vehsPerHour="{per_move_vph:.1f}" from="{inc}" to="{out}" '
                f'departLane="{lane}" departSpeed="max" type="{vtype}"/>'
            )

    return '<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n' + "\n".join(flows) + "\n</routes>\n"


def _nema_sumocfg(net: str, rou: str, end: float, seed: int = 42) -> str:
    """Generates SUMO configuration wrapper."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net}"/>
        <route-files value="{rou}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{end:.0f}"/>
        <step-length value="1"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <random_number>
        <seed value="{seed}"/>
    </random_number>
</configuration>
"""


def build_nema_tls_state_string(
    ring1_phase: NEMAPhase,
    ring1_interval: PhaseInterval,
    ring2_phase: NEMAPhase,
    ring2_interval: PhaseInterval,
) -> str:
    """Constructs the 12-link signal state string for SUMO from active NEMA phase states.

    The 12 links ordered by connection definitions:
      Link 0: N2C -> C2W (Right - P2)
      Link 1: N2C -> C2S (Thru - P2)
      Link 2: N2C -> C2E (Left - P5)
      Link 3: S2C -> C2E (Right - P6)
      Link 4: S2C -> C2N (Thru - P6)
      Link 5: S2C -> C2W (Left - P1)
      Link 6: E2C -> C2N (Right - P4)
      Link 7: E2C -> C2W (Thru - P4)
      Link 8: E2C -> C2S (Left - P7)
      Link 9: W2C -> C2S (Right - P8)
      Link 10: W2C -> C2E (Thru - P8)
      Link 11: W2C -> C2N (Left - P3)
    """
    # Link index to responsible phase mapping
    link_to_phase: dict[int, NEMAPhase] = {
        0: NEMAPhase.P2_NB_THROUGH,
        1: NEMAPhase.P2_NB_THROUGH,
        2: NEMAPhase.P5_NB_LEFT,
        3: NEMAPhase.P6_SB_THROUGH,
        4: NEMAPhase.P6_SB_THROUGH,
        5: NEMAPhase.P1_SB_LEFT,
        6: NEMAPhase.P4_EB_THROUGH,
        7: NEMAPhase.P4_EB_THROUGH,
        8: NEMAPhase.P7_EB_LEFT,
        9: NEMAPhase.P8_WB_THROUGH,
        10: NEMAPhase.P8_WB_THROUGH,
        11: NEMAPhase.P3_WB_LEFT,
    }

    chars: list[str] = ["r"] * 12

    for link_idx, phase in link_to_phase.items():
        if phase == ring1_phase:
            if ring1_interval == PhaseInterval.GREEN:
                chars[link_idx] = "G"
            elif ring1_interval == PhaseInterval.YELLOW_CHANGE:
                chars[link_idx] = "y"
            else:
                chars[link_idx] = "r"
        elif phase == ring2_phase:
            if ring2_interval == PhaseInterval.GREEN:
                chars[link_idx] = "G"
            elif ring2_interval == PhaseInterval.YELLOW_CHANGE:
                chars[link_idx] = "y"
            else:
                chars[link_idx] = "r"

    return "".join(chars)


def build_nema_8phase_intersection(
    demand: dict[NEMAPhase, float] | None = None,
    duration: float = 1200.0,
    seed: int = 42,
    out_dir: Path | None = None,
    rebuild: bool = True,
) -> Path:
    """Builds the 8-phase geometric SUMO network files and compiles via netconvert."""
    out = out_dir or SIM_NEMA_DIR
    out.mkdir(parents=True, exist_ok=True)
    demand = demand or DEFAULT_8PHASE_DEMAND

    nod_file = out / "intersection_nema.nod.xml"
    edg_file = out / "intersection_nema.edg.xml"
    con_file = out / "intersection_nema.con.xml"
    rou_file = out / "intersection_nema.rou.xml"
    cfg_file = out / "intersection_nema.sumocfg"
    net_file = out / "intersection_nema.net.xml"

    nod_file.write_text(_nema_nodes_xml(), encoding="utf-8")
    edg_file.write_text(_nema_edges_xml(), encoding="utf-8")
    con_file.write_text(_nema_connections_xml(), encoding="utf-8")
    rou_file.write_text(_nema_routes_xml(demand, duration, seed), encoding="utf-8")
    cfg_file.write_text(
        _nema_sumocfg("intersection_nema.net.xml", "intersection_nema.rou.xml", duration, seed),
        encoding="utf-8",
    )

    if rebuild or not net_file.exists():
        netconvert_bin = find_sumo_binary("netconvert")
        cmd = [
            netconvert_bin,
            "--node-files", str(nod_file),
            "--edge-files", str(edg_file),
            "--connection-files", str(con_file),
            "--output-file", str(net_file),
            "--no-turnarounds", "true",
            "--tls.default-type", "static",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"netconvert failed for NEMA 8-phase network:\n{proc.stderr}\n{proc.stdout}")

    return cfg_file
