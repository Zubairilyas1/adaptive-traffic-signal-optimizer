"""Build a 2-phase (N-S / E-W) four-approach SUMO intersection.

Generates nodes/edges/connections/route/sumocfg files and runs netconvert to
produce a .net.xml with a single traffic light. Demand can be supplied per
approach so that the CV pipeline's measured arrivals can drive the simulation.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\sumo_build.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src import config

SIM_DIR = config.ROOT / "sim"

# approach -> (incoming edge, outgoing edges)
APPROACHES = {
    "N": ("N2C", ["C2S", "C2E"]),
    "S": ("S2C", ["C2N", "C2W"]),
    "E": ("E2C", ["C2W", "C2N"]),
    "W": ("W2C", ["C2E", "C2S"]),
}

DEFAULT_DEMAND = {"N": 1600, "S": 1400, "E": 900, "W": 800}  # vehicles/hour (imbalanced)


def _sumo_bin(name: str) -> str:
    home = SIM_DIR
    exe = Path(sys.prefix) / "Lib" / "site-packages" / "sumo" / "bin" / name
    return str(exe)


def sumo_home() -> str:
    return str(Path(sys.prefix) / "Lib" / "site-packages" / "sumo")


def _nodes_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="N" x="0" y="200" type="priority"/>
    <node id="S" x="0" y="-200" type="priority"/>
    <node id="E" x="200" y="0" type="priority"/>
    <node id="W" x="-200" y="0" type="priority"/>
    <node id="C" x="0" y="0" type="traffic_light"/>
</nodes>
"""


def _edges_xml() -> str:
    lanes = 'numLanes="2" speed="13.89"'
    in_edges = "\n".join(
        f'    <edge id="{src}2C" from="{src}" to="C" {lanes}/>'
        for src in ("N", "S", "E", "W")
    )
    out_edges = "\n".join(
        f'    <edge id="C2{dst}" from="C" to="{dst}" {lanes}/>'
        for dst in ("N", "S", "E", "W")
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<edges>
{in_edges}
{out_edges}
</edges>
"""


def _connections_xml() -> str:
    rows = []
    for by, (inc, outs) in APPROACHES.items():
        rows.append(f'    <connection from="{inc}" to="{outs[0]}" fromLane="0" toLane="0"/>')
        rows.append(f'    <connection from="{inc}" to="{outs[1]}" fromLane="1" toLane="0"/>')
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<connections>\n" + "\n".join(rows) + "\n</connections>\n"


def _routes_xml(demand: dict[str, float], duration: float, seed: int = 42) -> str:
    flows = []
    for by, (inc, outs) in APPROACHES.items():
        veh = float(demand.get(by, 0.0))
        half = veh / 2.0
        for i, out in enumerate(outs):
            flows.append(
                f'    <flow id="{by}_{i}" begin="0" end="{duration:.0f}" '
                f'vehsPerHour="{half:.1f}" from="{inc}" to="{out}" '
                f'departLane="best" departSpeed="max"/>'
            )
    return ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<routes>\n" + "\n".join(flows) + "\n</routes>\n")


def _sumocfg(net: str, rou: str, end: float, seed: int = 42, gui: bool = False) -> str:
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


def build_intersection(
    demand: dict[str, float] | None = None,
    duration: float = 1200.0,
    seed: int = 42,
    out_dir: Path | None = None,
    rebuild: bool = True,
) -> Path:
    """Write SUMO config files and return the path to the .sumocfg."""
    out = out_dir or SIM_DIR
    out.mkdir(parents=True, exist_ok=True)
    demand = demand or DEFAULT_DEMAND

    (out / "intersection.nod.xml").write_text(_nodes_xml(), encoding="utf-8")
    (out / "intersection.edg.xml").write_text(_edges_xml(), encoding="utf-8")
    (out / "intersection.con.xml").write_text(_connections_xml(), encoding="utf-8")
    (out / "intersection.rou.xml").write_text(_routes_xml(demand, duration, seed), encoding="utf-8")
    (out / "intersection.sumocfg").write_text(
        _sumocfg("intersection.net.xml", "intersection.rou.xml", duration, seed),
        encoding="utf-8",
    )

    net = out / "intersection.net.xml"
    if rebuild or not net.exists():
        cmd = [
            _sumo_bin("netconvert.exe"),
            "--node-files", str(out / "intersection.nod.xml"),
            "--edge-files", str(out / "intersection.edg.xml"),
            "--connection-files", str(out / "intersection.con.xml"),
            "--output-file", str(net),
            "--no-turnarounds", "true",
            "--tls.default-type", "static",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"netconvert failed:\n{proc.stderr}\n{proc.stdout}")

    return out / "intersection.sumocfg"


if __name__ == "__main__":
    cfg = build_intersection()
    print(f"built -> {cfg}")
