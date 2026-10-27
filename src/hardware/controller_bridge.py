"""Physical and virtual traffic signal controller communication bridge.

Connects the AI decision engine to external hardware and simulators:
- Real USB Serial COM port (Arduino, ESP32, Raspberry Pi LED fixtures)
- High-speed UDP telemetry broadcast (Digital twins, SUMO, network dashboards)
- Zero-dependency thread-safe Virtual Emulation for local testing
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import socket
from typing import Any, Callable, Optional

from src.decision.actuator import ControllerState, SignalAspect

logger = logging.getLogger(__name__)

# Optional hardware serial dependency
try:
    import serial
except ImportError:
    serial = None


class ControllerBridgeMode(Enum):
    """Communication mode for signal controller bridge."""

    VIRTUAL = "virtual"  # In-memory thread-safe state queue (default)
    SERIAL = "serial"  # Physical USB COM port (115200 baud)
    UDP = "udp"  # Local UDP socket broadcast (JSON datagrams)


@dataclass
class SignalStatePacket:
    """Standardized telemetry payload transmitted to physical/virtual controllers."""

    timestamp: float
    active_phase: str
    active_aspect: str  # "GREEN", "YELLOW", "RED"
    time_remaining_s: float
    elapsed_s: float
    is_failsafe: bool
    reason_text: str
    displays: dict[str, str]  # approach -> aspect

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_ntcip_sim(self) -> str:
        """Simulate an NTCIP 1202 telemetry string."""
        phase_num = "".join(filter(str.isdigit, self.active_phase)) or "1"
        return f"NTCIP:PH{phase_num}:{self.active_aspect}:{self.time_remaining_s:.1f}s"


class SignalControllerBridge:
    """Universal bridge connecting AI controller states to physical relays or virtual monitors."""

    def __init__(
        self,
        mode: ControllerBridgeMode = ControllerBridgeMode.VIRTUAL,
        port: str = "COM3",
        baud_rate: int = 115200,
        udp_host: str = "127.0.0.1",
        udp_port: int = 9999,
        max_history: int = 100,
    ) -> None:
        self.mode = mode
        self.port = port
        self.baud_rate = baud_rate
        self.udp_host = udp_host
        self.udp_port = udp_port

        self._serial_conn: Any = None
        self._udp_sock: Optional[socket.socket] = None
        self._history: deque[SignalStatePacket] = deque(maxlen=max_history)
        self.latest_packet: Optional[SignalStatePacket] = None
        self.packet_count: int = 0
        self.is_connected: bool = False

    def open(self) -> bool:
        """Initialize communication channel."""
        if self.mode == ControllerBridgeMode.SERIAL:
            if serial is None:
                logger.warning("pyserial is not installed. Falling back to VIRTUAL mode.")
                self.mode = ControllerBridgeMode.VIRTUAL
                self.is_connected = True
                return True
            try:
                self._serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baud_rate,
                    timeout=0.05,
                    write_timeout=0.1,
                )
                self.is_connected = True
                logger.info(f"Connected to hardware signal controller on {self.port}")
                return True
            except Exception as exc:
                logger.warning(f"Could not open serial port {self.port} ({exc}). Using VIRTUAL fallback.")
                self.is_connected = False
                return False

        elif self.mode == ControllerBridgeMode.UDP:
            try:
                self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.is_connected = True
                return True
            except Exception as exc:
                logger.warning(f"Could not open UDP socket: {exc}")
                self.is_connected = False
                return False

        # Virtual mode always succeeds
        self.is_connected = True
        return True

    def send_state(
        self,
        state: ControllerState,
        is_failsafe: bool = False,
    ) -> SignalStatePacket:
        """Convert ControllerState into a hardware packet and transmit."""
        active_disp = state.phase_displays.get(state.active_phase_id)
        rem = active_disp.time_remaining_s if active_disp else 0.0
        elapsed = active_disp.elapsed_in_phase_s if active_disp else 0.0

        displays = {
            disp.approach: disp.aspect.value.upper()
            for disp in state.phase_displays.values()
        }

        packet = SignalStatePacket(
            timestamp=state.timestamp,
            active_phase=state.active_phase_id,
            active_aspect=state.active_aspect.value.upper(),
            time_remaining_s=round(rem, 1),
            elapsed_s=round(elapsed, 1),
            is_failsafe=is_failsafe,
            reason_text=state.reason_text,
            displays=displays,
        )

        self.latest_packet = packet
        self._history.append(packet)
        self.packet_count += 1

        payload_bytes = (packet.to_json() + "\n").encode("utf-8")

        # 1. Transmit over Serial if active
        if self.mode == ControllerBridgeMode.SERIAL and self._serial_conn and self._serial_conn.is_open:
            try:
                self._serial_conn.write(payload_bytes)
                self._serial_conn.flush()
            except Exception as exc:
                logger.warning(f"Serial write error: {exc}")

        # 2. Transmit over UDP if active
        elif self.mode == ControllerBridgeMode.UDP and self._udp_sock:
            try:
                self._udp_sock.sendto(payload_bytes, (self.udp_host, self.udp_port))
            except Exception as exc:
                logger.warning(f"UDP send error: {exc}")

        return packet

    def get_latest_state(self) -> Optional[SignalStatePacket]:
        return self.latest_packet

    def get_history(self, limit: int = 50) -> list[SignalStatePacket]:
        items = list(self._history)
        return items[-limit:]

    def close(self) -> None:
        """Close physical ports and sockets."""
        if self._serial_conn is not None:
            try:
                self._serial_conn.close()
            except Exception:
                pass
            self._serial_conn = None

        if self._udp_sock is not None:
            try:
                self._udp_sock.close()
            except Exception:
                pass
            self._udp_sock = None

        self.is_connected = False

    def __enter__(self) -> SignalControllerBridge:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def get_arduino_sketch() -> str:
    """Return ready-to-flash Arduino C++ source code for physical LED traffic lights."""
    return """/*
 * OptiFlow ITS - Arduino 2-Phase Signal Controller Receiver
 * Pins:
 *   Phase 1 (North-South): Red=2, Yellow=3, Green=4
 *   Phase 2 (East-West):   Red=5, Yellow=6, Green=7
 */

const int P1_RED = 2, P1_YEL = 3, P1_GRN = 4;
const int P2_RED = 5, P2_YEL = 6, P2_GRN = 7;

void setup() {
  Serial.begin(115200);
  pinMode(P1_RED, OUTPUT); pinMode(P1_YEL, OUTPUT); pinMode(P1_GRN, OUTPUT);
  pinMode(P2_RED, OUTPUT); pinMode(P2_YEL, OUTPUT); pinMode(P2_GRN, OUTPUT);
  // Default all-red safety startup
  digitalWrite(P1_RED, HIGH);
  digitalWrite(P2_RED, HIGH);
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\\n');
    line.trim();
    if (line.length() == 0) return;

    // Fast parse for active aspect and phase
    if (line.indexOf("\\"active_phase\\": \\"P1\\"") > 0 || line.indexOf("\\"P1\\"") > 0) {
      if (line.indexOf("\\"GREEN\\"") > 0) {
        digitalWrite(P1_GRN, HIGH); digitalWrite(P1_YEL, LOW); digitalWrite(P1_RED, LOW);
        digitalWrite(P2_GRN, LOW);  digitalWrite(P2_YEL, LOW); digitalWrite(P2_RED, HIGH);
      } else if (line.indexOf("\\"YELLOW\\"") > 0) {
        digitalWrite(P1_GRN, LOW);  digitalWrite(P1_YEL, HIGH); digitalWrite(P1_RED, LOW);
        digitalWrite(P2_GRN, LOW);  digitalWrite(P2_YEL, LOW);  digitalWrite(P2_RED, HIGH);
      } else {
        digitalWrite(P1_GRN, LOW);  digitalWrite(P1_YEL, LOW);  digitalWrite(P1_RED, HIGH);
      }
    } else if (line.indexOf("\\"active_phase\\": \\"P2\\"") > 0 || line.indexOf("\\"P2\\"") > 0) {
      if (line.indexOf("\\"GREEN\\"") > 0) {
        digitalWrite(P1_GRN, LOW);  digitalWrite(P1_YEL, LOW);  digitalWrite(P1_RED, HIGH);
        digitalWrite(P2_GRN, HIGH); digitalWrite(P2_YEL, LOW); digitalWrite(P2_RED, LOW);
      } else if (line.indexOf("\\"YELLOW\\"") > 0) {
        digitalWrite(P1_GRN, LOW);  digitalWrite(P1_YEL, LOW);  digitalWrite(P1_RED, HIGH);
        digitalWrite(P2_GRN, LOW);  digitalWrite(P2_YEL, HIGH); digitalWrite(P2_RED, LOW);
      } else {
        digitalWrite(P2_GRN, LOW);  digitalWrite(P2_YEL, LOW);  digitalWrite(P2_RED, HIGH);
      }
    }
  }
}
"""
