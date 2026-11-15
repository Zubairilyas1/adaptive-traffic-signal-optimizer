"""Multi-Intersection Arterial Corridor Geometric SUMO Network Generator.

Procedurally builds an arterial corridor consisting of 3 consecutive 4-leg
intersections with dedicated protected left-turn bays and coordinated links:
- Junction 1: North Gate (Coord: y = 850m)
- Junction 2: Central Plaza (Coord: y = 450m, d12 = 400m)
- Junction 3: South Boulevard (Coord: y = 0m, d23 = 450m)

Generates nodes, edges, connections, calibrated multi-class route streams,
and compiles the network using netconvert for green-wave micro-simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import subprocess
from typing import List, Optional, Tuple

from src import config
from src.sumo.nema_network import find_sumo_binary

logger = logging.getLogger(__name__)

SIM_CORRIDOR_DIR = config.ROOT / "sim" / "corridor"


@dataclass
class CorridorJunctionSpec:
    """Specification of a single coordinated intersection along the corridor."""

    junction_id: str
    name: str
    y_pos: float
    cross_street_name: str
    distance_to_next_m: float = 0.0


@dataclass
class ArterialCorridorConfig:
    """Geometric and operational configuration for the 3-junction corridor."""

    d12_m: float = 400.0  # Spacing between Junction 1 (North Gate) and Junction 2 (Central Plaza)
    d23_m: float = 450.0  # Spacing between Junction 2 (Central Plaza) and Junction 3 (South Boulevard)
    cross_length_m: float = 250.0  # Distance of minor street entry/exit nodes
    progression_speed_kmh: float = 45.0  # Design arterial green-wave speed (12.5 m/s)
    common_cycle_s: float = 90.0  # Coordinated common cycle length
    speed_limit_mps: float = 13.89  # 50 km/h speed limit on arterial links


def default_corridor_junctions(cfg: Optional[ArterialCorridorConfig] = None) -> List[CorridorJunctionSpec]:
    """Returns the canonical 3-junction specification along the North-South arterial."""
    c = cfg or ArterialCorridorConfig()
    y3 = 0.0
    y2 = y3 + c.d23_m
    y1 = y2 + c.d12_m

    return [
        CorridorJunctionSpec("J1", "North Gate", y1, "Cross St 1", distance_to_next_m=c.d12_m),
        CorridorJunctionSpec("J2", "Central Plaza", y2, "Cross St 2", distance_to_next_m=c.d23_m),
        CorridorJunctionSpec("J3", "South Boulevard", y3, "Cross St 3", distance_to_next_m=0.0),
    ]


def _corridor_nodes_xml(junctions: List[CorridorJunctionSpec], cross_len: float) -> str:
    """Generates XML nodes for the 3 arterial traffic lights and all peripheral endpoints."""
    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<nodes>',
    ]

    # Northernmost arterial entry/exit node
    top_y = junctions[0].y_pos + cross_len
    lines.append(f'    <node id="N_ENTRY" x="0" y="{top_y:.1f}" type="priority"/>')

    # Southernmost arterial entry/exit node
    bot_y = junctions[-1].y_pos - cross_len
    lines.append(f'    <node id="S_ENTRY" x="0" y="{bot_y:.1f}" type="priority"/>')

    # 3 Coordinated Traffic Light Intersections and their East/West minor street endpoints
    for j in junctions:
        lines.append(f'    <node id="{j.junction_id}" x="0" y="{j.y_pos:.1f}" type="traffic_light"/>')
        lines.append(f'    <node id="W_{j.junction_id}" x="{-cross_len:.1f}" y="{j.y_pos:.1f}" type="priority"/>')
        lines.append(f'    <node id="E_{j.junction_id}" x="{cross_len:.1f}" y="{j.y_pos:.1f}" type="priority"/>')

    lines.append('</nodes>\n')
    return "\n".join(lines)


def _corridor_edges_xml(junctions: List[CorridorJunctionSpec], speed_limit: float) -> str:
    """Generates multi-lane edges for the arterial backbone and minor cross-streets."""
    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<edges>',
    ]

    in_lanes = f'numLanes="3" speed="{speed_limit:.2f}"'   # 3 incoming lanes (Right, Thru, Dedicated Left)
    out_lanes = f'numLanes="2" speed="{speed_limit:.2f}"'  # 2 receiving lanes
    arterial_lanes = f'numLanes="3" speed="{speed_limit:.2f}"'  # 3 continuous arterial lanes

    # 1. Northern Entry / Exit to Junction 1
    j1 = junctions[0].junction_id
    lines.append(f'    <edge id="N_ENTRY2{j1}" from="N_ENTRY" to="{j1}" {in_lanes}/>')
    lines.append(f'    <edge id="{j1}2N_ENTRY" from="{j1}" to="N_ENTRY" {out_lanes}/>')

    # 2. Inter-Junction Arterial Segments (J1 <-> J2, J2 <-> J3)
    for i in range(len(junctions) - 1):
        curr_id = junctions[i].junction_id
        next_id = junctions[i + 1].junction_id

        # Southbound arterial edge (curr -> next)
        lines.append(f'    <edge id="{curr_id}2{next_id}" from="{curr_id}" to="{next_id}" {arterial_lanes}/>')
        # Northbound arterial edge (next -> curr)
        lines.append(f'    <edge id="{next_id}2{curr_id}" from="{next_id}" to="{curr_id}" {arterial_lanes}/>')

    # 3. Southern Entry / Exit from Junction 3
    j_last = junctions[-1].junction_id
    lines.append(f'    <edge id="{j_last}2S_ENTRY" from="{j_last}" to="S_ENTRY" {out_lanes}/>')
    lines.append(f'    <edge id="S_ENTRY2{j_last}" from="S_ENTRY" to="{j_last}" {in_lanes}/>')

    # 4. East and West Cross-Street Edges for all 3 Junctions
    for j in junctions:
        jid = j.junction_id
        # West minor approaches
        lines.append(f'    <edge id="W_{jid}2{jid}" from="W_{jid}" to="{jid}" {in_lanes}/>')
        lines.append(f'    <edge id="{jid}2W_{jid}" from="{jid}" to="W_{jid}" {out_lanes}/>')
        # East minor approaches
        lines.append(f'    <edge id="E_{jid}2{jid}" from="E_{jid}" to="{jid}" {in_lanes}/>')
        lines.append(f'    <edge id="{jid}2E_{jid}" from="{jid}" to="E_{jid}" {out_lanes}/>')

    lines.append('</edges>\n')
    return "\n".join(lines)


def _corridor_connections_xml(junctions: List[CorridorJunctionSpec]) -> str:
    """Explicit lane turning connections enforcing dedicated left-turn pockets on all 3 junctions."""
    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<connections>',
    ]

    for i, j in enumerate(junctions):
        jid = j.junction_id

        # Identify incoming & outgoing edge names for North & South approaches
        north_in = "N_ENTRY2" + jid if i == 0 else f"{junctions[i - 1].junction_id}2{jid}"
        north_out = jid + "2N_ENTRY" if i == 0 else f"{jid}2{junctions[i - 1].junction_id}"

        south_in = "S_ENTRY2" + jid if i == len(junctions) - 1 else f"{junctions[i + 1].junction_id}2{jid}"
        south_out = jid + "2S_ENTRY" if i == len(junctions) - 1 else f"{jid}2{junctions[i + 1].junction_id}"

        west_in = f"W_{jid}2{jid}"
        west_out = f"{jid}2W_{jid}"
        east_in = f"E_{jid}2{jid}"
        east_out = f"{jid}2E_{jid}"

        # --- North Approach Connections (Southbound Traffic) ---
        # Lane 0: Right Turn to West
        lines.append(f'    <connection from="{north_in}" to="{west_out}" fromLane="0" toLane="0"/>')
        # Lane 1: Through to South
        lines.append(f'    <connection from="{north_in}" to="{south_out}" fromLane="1" toLane="0"/>')
        # Lane 2: Protected Left Turn to East (Phase 5)
        lines.append(f'    <connection from="{north_in}" to="{east_out}" fromLane="2" toLane="1"/>')

        # --- South Approach Connections (Northbound Traffic) ---
        # Lane 0: Right Turn to East
        lines.append(f'    <connection from="{south_in}" to="{east_out}" fromLane="0" toLane="0"/>')
        # Lane 1: Through to North
        lines.append(f'    <connection from="{south_in}" to="{north_out}" fromLane="1" toLane="0"/>')
        # Lane 2: Protected Left Turn to West (Phase 1)
        lines.append(f'    <connection from="{south_in}" to="{west_out}" fromLane="2" toLane="1"/>')

        # --- East Minor Approach Connections ---
        # Lane 0: Right Turn to North
        lines.append(f'    <connection from="{east_in}" to="{north_out}" fromLane="0" toLane="0"/>')
        # Lane 1: Through to West
        lines.append(f'    <connection from="{east_in}" to="{west_out}" fromLane="1" toLane="0"/>')
        # Lane 2: Dedicated Left to South (Phase 7)
        lines.append(f'    <connection from="{east_in}" to="{south_out}" fromLane="2" toLane="1"/>')

        # --- West Minor Approach Connections ---
        # Lane 0: Right Turn to South
        lines.append(f'    <connection from="{west_in}" to="{south_out}" fromLane="0" toLane="0"/>')
        # Lane 1: Through to East
        lines.append(f'    <connection from="{west_in}" to="{east_out}" fromLane="1" toLane="0"/>')
        # Lane 2: Dedicated Left to North (Phase 3)
        lines.append(f'    <connection from="{west_in}" to="{north_out}" fromLane="2" toLane="1"/>')

    lines.append('</connections>\n')
    return "\n".join(lines)


def _corridor_routes_xml(
    junctions: List[CorridorJunctionSpec],
    arterial_vph: float = 1200.0,
    cross_street_vph: float = 350.0,
    duration_s: float = 3600.0,
) -> str:
    """Generates continuous arterial platoon flows and cross-street injection streams."""
    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes>',
        '    <vType id="car" length="5.0" minGap="2.5" maxSpeed="13.89" accel="2.6" decel="4.5" vClass="passenger"/>',
        '    <vType id="bus" length="12.0" minGap="3.0" maxSpeed="11.11" accel="1.2" decel="3.0" vClass="bus"/>',
        '    <vType id="ambulance" length="6.5" minGap="2.0" maxSpeed="20.0" accel="4.0" decel="6.0" vClass="emergency"/>',
    ]

    j1 = junctions[0].junction_id
    j3 = junctions[-1].junction_id

    # 1. Main Arterial Corridors (Through Platoons)
    # Southbound Platoon (N_ENTRY -> S_ENTRY through J1, J2, J3)
    sb_edges = f"N_ENTRY2{j1} {j1}2J2 J22{j3} {j3}2S_ENTRY"
    lines.append(f'    <route id="route_arterial_SB" edges="{sb_edges}"/>')
    lines.append(
        f'    <flow id="flow_arterial_SB" begin="0" end="{duration_s:.0f}" '
        f'vehsPerHour="{arterial_vph:.1f}" route="route_arterial_SB" departLane="1" departSpeed="max" type="car"/>'
    )

    # Northbound Platoon (S_ENTRY -> N_ENTRY through J3, J2, J1)
    nb_edges = f"S_ENTRY2{j3} {j3}2J2 J22{j1} {j1}2N_ENTRY"
    lines.append(f'    <route id="route_arterial_NB" edges="{nb_edges}"/>')
    lines.append(
        f'    <flow id="flow_arterial_NB" begin="0" end="{duration_s:.0f}" '
        f'vehsPerHour="{arterial_vph:.1f}" route="route_arterial_NB" departLane="1" departSpeed="max" type="car"/>'
    )

    # 2. Cross-Street Flows for each intersection
    for j in junctions:
        jid = j.junction_id
        # Eastbound cross-traffic (W_J -> E_J)
        lines.append(
            f'    <flow id="flow_cross_EB_{jid}" begin="0" end="{duration_s:.0f}" '
            f'vehsPerHour="{cross_street_vph:.1f}" from="W_{jid}2{jid}" to="{jid}2E_{jid}" '
            f'departLane="1" departSpeed="max" type="car"/>'
        )
        # Westbound cross-traffic (E_J -> W_J)
        lines.append(
            f'    <flow id="flow_cross_WB_{jid}" begin="0" end="{duration_s:.0f}" '
            f'vehsPerHour="{cross_street_vph:.1f}" from="E_{jid}2{jid}" to="{jid}2W_{jid}" '
            f'departLane="1" departSpeed="max" type="car"/>'
        )

    lines.append('</routes>\n')
    return "\n".join(lines)


def _corridor_sumocfg(net: str, rou: str, duration_s: float, seed: int = 42) -> str:
    """Generates the SUMO configuration file for corridor co-simulation."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net}"/>
        <route-files value="{rou}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{duration_s:.0f}"/>
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


def generate_corridor_network(
    cfg: Optional[ArterialCorridorConfig] = None,
    output_dir: Optional[Path] = None,
    compile_net: bool = True,
    duration_s: float = 3600.0,
    seed: int = 42,
) -> Tuple[Path, Path, Path]:
    """Compiles the 3-junction arterial corridor network and route configuration.

    Returns:
        Tuple of (net_path, rou_path, cfg_path).
    """
    config_obj = cfg or ArterialCorridorConfig()
    out = output_dir or SIM_CORRIDOR_DIR
    out.mkdir(parents=True, exist_ok=True)

    junctions = default_corridor_junctions(config_obj)

    nod_file = out / "corridor.nod.xml"
    edg_file = out / "corridor.edg.xml"
    con_file = out / "corridor.con.xml"
    net_file = out / "corridor.net.xml"
    rou_file = out / "corridor.rou.xml"
    cfg_file = out / "corridor.sumocfg"

    # Write node, edge, and connection definitions
    nod_file.write_text(_corridor_nodes_xml(junctions, config_obj.cross_length_m), encoding="utf-8")
    edg_file.write_text(_corridor_edges_xml(junctions, config_obj.speed_limit_mps), encoding="utf-8")
    con_file.write_text(_corridor_connections_xml(junctions), encoding="utf-8")

    # Write routes and SUMO config
    rou_file.write_text(_corridor_routes_xml(junctions, duration_s=duration_s), encoding="utf-8")
    cfg_file.write_text(_corridor_sumocfg(net_file.name, rou_file.name, duration_s, seed=seed), encoding="utf-8")

    if compile_net:
        netconvert_bin = find_sumo_binary("netconvert")
        cmd = [
            netconvert_bin,
            f"--node-files={nod_file}",
            f"--edge-files={edg_file}",
            f"--connection-files={con_file}",
            f"--output-file={net_file}",
            "--no-turnarounds=true",
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Compiled corridor network successfully with netconvert.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"netconvert compilation skipped or failed: {e}. Writing placeholder network.")
            if not net_file.exists():
                net_file.write_text('<?xml version="1.0"?><net/>', encoding="utf-8")
    else:
        if not net_file.exists():
            net_file.write_text('<?xml version="1.0"?><net/>', encoding="utf-8")

    return net_file, rou_file, cfg_file
