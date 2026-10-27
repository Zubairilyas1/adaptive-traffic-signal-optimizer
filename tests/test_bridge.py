"""Unit tests for SignalControllerBridge and hardware communication."""

from __future__ import annotations

import json
import socket
import pytest

from src.decision.actuator import (
    ControllerState,
    DecisionReason,
    SignalAspect,
    SignalPhaseDisplay,
)
from src.hardware.controller_bridge import (
    ControllerBridgeMode,
    SignalControllerBridge,
    SignalStatePacket,
    get_arduino_sketch,
)


@pytest.fixture
def sample_controller_state() -> ControllerState:
    displays = {
        "P1": SignalPhaseDisplay(
            phase_id="P1",
            approach="North",
            aspect=SignalAspect.GREEN,
            elapsed_in_phase_s=14.0,
            time_remaining_s=16.0,
        ),
        "P2": SignalPhaseDisplay(
            phase_id="P2",
            approach="East",
            aspect=SignalAspect.RED,
            elapsed_in_phase_s=0.0,
            time_remaining_s=0.0,
        ),
    }
    return ControllerState(
        timestamp=14.0,
        active_phase_id="P1",
        active_aspect=SignalAspect.GREEN,
        active_reason=DecisionReason.EXTEND_HEADWAY,
        reason_text="Holding active green",
        phase_displays=displays,
        active_queue_pcu=2.5,
        conflicting_queue_pcu=5.0,
    )


def test_signal_packet_serialization(sample_controller_state: ControllerState):
    bridge = SignalControllerBridge(mode=ControllerBridgeMode.VIRTUAL)
    packet = bridge.send_state(sample_controller_state)

    assert packet.active_phase == "P1"
    assert packet.active_aspect == "GREEN"
    assert packet.time_remaining_s == 16.0
    assert packet.displays["North"] == "GREEN"
    assert packet.displays["East"] == "RED"

    # Verify JSON
    json_str = packet.to_json()
    data = json.loads(json_str)
    assert data["active_phase"] == "P1"
    assert data["active_aspect"] == "GREEN"

    # Verify NTCIP simulation string
    ntcip = packet.to_ntcip_sim()
    assert "PH1:GREEN" in ntcip


def test_virtual_bridge_history(sample_controller_state: ControllerState):
    with SignalControllerBridge(mode=ControllerBridgeMode.VIRTUAL, max_history=10) as bridge:
        assert bridge.is_connected is True
        for i in range(5):
            sample_controller_state.timestamp = float(i)
            bridge.send_state(sample_controller_state)

        latest = bridge.get_latest_state()
        assert latest is not None
        assert latest.timestamp == 4.0
        assert bridge.packet_count == 5

        history = bridge.get_history(limit=3)
        assert len(history) == 3
        assert history[-1].timestamp == 4.0


def test_udp_broadcast_bridge(sample_controller_state: ControllerState):
    # Setup a local listener socket
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 9998))
    listener.settimeout(1.0)

    bridge = SignalControllerBridge(
        mode=ControllerBridgeMode.UDP,
        udp_host="127.0.0.1",
        udp_port=9998,
    )
    bridge.open()
    assert bridge.is_connected is True

    try:
        bridge.send_state(sample_controller_state)
        data, _ = listener.recvfrom(2048)
        payload = json.loads(data.decode("utf-8"))
        assert payload["active_phase"] == "P1"
        assert payload["active_aspect"] == "GREEN"
    finally:
        bridge.close()
        listener.close()


def test_arduino_sketch_template():
    code = get_arduino_sketch()
    assert "void setup()" in code
    assert "void loop()" in code
    assert "Serial.begin(115200)" in code
    assert "P1_GRN" in code
