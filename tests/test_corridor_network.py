"""Unit and Integration Tests for Multi-Intersection Arterial Corridor SUMO Network Generator."""

from pathlib import Path
import xml.etree.ElementTree as ET

from src.sumo.corridor_network import (
    ArterialCorridorConfig,
    default_corridor_junctions,
    generate_corridor_network,
    _corridor_connections_xml,
    _corridor_edges_xml,
    _corridor_nodes_xml,
    _corridor_routes_xml,
)


def test_default_corridor_junctions():
    """Verifies that the canonical 3-junction corridor specifications are accurate."""
    cfg = ArterialCorridorConfig(d12_m=400.0, d23_m=450.0)
    junctions = default_corridor_junctions(cfg)

    assert len(junctions) == 3
    j1, j2, j3 = junctions

    assert j1.junction_id == "J1"
    assert j2.junction_id == "J2"
    assert j3.junction_id == "J3"

    # Coordinates
    assert j3.y_pos == 0.0
    assert j2.y_pos == 450.0
    assert j1.y_pos == 850.0

    # Spacing
    assert j1.distance_to_next_m == 400.0
    assert j2.distance_to_next_m == 450.0
    assert j3.distance_to_next_m == 0.0


def test_corridor_xml_structures():
    """Verifies that generated node, edge, connection, and route XMLs are syntactically valid XML."""
    junctions = default_corridor_junctions()

    # 1. Nodes
    nodes_xml = _corridor_nodes_xml(junctions, cross_len=250.0)
    root_nodes = ET.fromstring(nodes_xml)
    node_ids = {n.attrib["id"] for n in root_nodes.findall("node")}
    assert "J1" in node_ids
    assert "J2" in node_ids
    assert "J3" in node_ids
    assert "N_ENTRY" in node_ids
    assert "S_ENTRY" in node_ids
    assert "W_J1" in node_ids
    assert "E_J3" in node_ids

    # 2. Edges
    edges_xml = _corridor_edges_xml(junctions, speed_limit=13.89)
    root_edges = ET.fromstring(edges_xml)
    edge_ids = {e.attrib["id"] for e in root_edges.findall("edge")}
    assert "J12J2" in edge_ids
    assert "J22J1" in edge_ids
    assert "J22J3" in edge_ids
    assert "J32J2" in edge_ids
    assert "N_ENTRY2J1" in edge_ids
    assert "S_ENTRY2J3" in edge_ids

    # 3. Connections
    conns_xml = _corridor_connections_xml(junctions)
    root_conns = ET.fromstring(conns_xml)
    assert len(root_conns.findall("connection")) >= 36  # 12 connections per junction * 3 junctions

    # 4. Routes
    routes_xml = _corridor_routes_xml(junctions, duration_s=1800.0)
    root_routes = ET.fromstring(routes_xml)
    flow_ids = {f.attrib["id"] for f in root_routes.findall("flow")}
    assert "flow_arterial_SB" in flow_ids
    assert "flow_arterial_NB" in flow_ids
    assert "flow_cross_EB_J1" in flow_ids
    assert "flow_cross_WB_J2" in flow_ids


def test_generate_corridor_network_files(tmp_path: Path):
    """Verifies that generate_corridor_network produces all files and configs."""
    cfg = ArterialCorridorConfig(d12_m=350.0, d23_m=400.0)
    net_path, rou_path, cfg_path = generate_corridor_network(
        cfg=cfg,
        output_dir=tmp_path,
        compile_net=False,  # Test file emission independently from host netconvert binary
        duration_s=1200.0,
    )

    assert net_path.exists()
    assert rou_path.exists()
    assert cfg_path.exists()

    # Check that nod, edg, and con files were generated
    assert (tmp_path / "corridor.nod.xml").exists()
    assert (tmp_path / "corridor.edg.xml").exists()
    assert (tmp_path / "corridor.con.xml").exists()

    # Check sumocfg content
    cfg_content = cfg_path.read_text(encoding="utf-8")
    assert "<net-file" in cfg_content
    assert "<route-files" in cfg_content
    assert 'step-length value="1"' in cfg_content


def test_compile_corridor_network_with_netconvert(tmp_path: Path):
    """Verifies that netconvert successfully compiles the 3-junction corridor network."""
    cfg = ArterialCorridorConfig(d12_m=400.0, d23_m=450.0)
    net_path, rou_path, cfg_path = generate_corridor_network(
        cfg=cfg,
        output_dir=tmp_path,
        compile_net=True,
        duration_s=600.0,
    )
    assert net_path.exists()
    assert net_path.stat().st_size > 500  # Non-trivial compiled network
    net_content = net_path.read_text(encoding="utf-8")
    assert 'id="J1"' in net_content
    assert 'id="J2"' in net_content
    assert 'id="J3"' in net_content
