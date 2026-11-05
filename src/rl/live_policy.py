"""Live Reinforcement Learning Signal Policy and NEMA Clearance Actuator.

Executes learned PPO policy weights or classical baselines in real time:
- Constructs canonical 6D observation vectors from CV PCU queues or TraCI simulation
- Computes policy probabilities, confidence, and human-readable decision rationales
- Enforces NEMA clearance intervals (min-green, yellow clearance, all-red)
- Integrates bi-directionally with TraCI and hardware controller bridges
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from src import config

logger = logging.getLogger(__name__)

# Canonically ordered approaches for 6D observation vector: [Q_N, Q_S, Q_E, Q_W, phase, green_ratio]
OBS_APPROACHES = ("N", "S", "E", "W")


class LiveSignalAspect(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ALL_RED = "all_red"


@dataclass
class PolicyDecision:
    """Detailed output from a policy evaluation step."""

    action: int  # Actuated action: 0 = hold green, 1 = initiate phase switch
    raw_action: int  # Unclamped policy prediction before safety rules
    policy_name: str  # "PPO", "Rule-Based", or "Fixed-Time"
    active_phase: int  # Current active phase index (0: N-S, 1: E-W)
    active_aspect: LiveSignalAspect  # Current signal aspect (GREEN, YELLOW, ALL_RED)
    green_elapsed_s: float  # Elapsed green time for current phase
    time_in_state_s: float  # Time spent in current aspect (green, yellow, or all-red)
    observation: np.ndarray  # The 6D state vector evaluated
    probabilities: np.ndarray  # Action probabilities [P(hold), P(switch)]
    confidence: float  # Confidence of chosen action (0.0 to 1.0)
    rationale: str  # Human-interpretable reasoning for this decision
    switched: bool = False  # True on the exact step where green phase switched

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "raw_action": self.raw_action,
            "policy_name": self.policy_name,
            "active_phase": self.active_phase,
            "active_aspect": self.active_aspect.value,
            "green_elapsed_s": round(self.green_elapsed_s, 2),
            "time_in_state_s": round(self.time_in_state_s, 2),
            "observation": [round(float(x), 4) for x in self.observation],
            "probabilities": [round(float(p), 4) for p in self.probabilities],
            "confidence": round(float(self.confidence), 4),
            "rationale": self.rationale,
            "switched": self.switched,
        }


def assemble_observation(
    queues: Union[dict[str, float], Sequence[float]],
    active_phase: int = 0,
    elapsed_green: float = 0.0,
    queue_norm_scale: float = 20.0,
    green_norm_scale: float = 60.0,
) -> np.ndarray:
    """Build the canonical 6-element observation vector.

    [q_N/20, q_S/20, q_E/20, q_W/20, phase, green_elapsed/60]
    """
    if isinstance(queues, dict):
        raw_q = [
            float(queues.get("N", queues.get("North", 0.0))),
            float(queues.get("S", queues.get("South", 0.0))),
            float(queues.get("E", queues.get("East", 0.0))),
            float(queues.get("W", queues.get("West", 0.0))),
        ]
    else:
        raw_q = [float(q) for q in list(queues)[:4]]
        while len(raw_q) < 4:
            raw_q.append(0.0)

    q_arr = np.asarray(raw_q, dtype=np.float32)
    q_norm = np.clip(q_arr / float(queue_norm_scale), 0.0, 1.0)
    phase_val = np.float32(active_phase)
    elapsed_val = np.float32(min(1.0, max(0.0, elapsed_green / float(green_norm_scale))))

    return np.concatenate([q_norm, [phase_val, elapsed_val]]).astype(np.float32)


class LivePPOController:
    """Real-time inference wrapper around trained Stable-Baselines3 PPO model."""

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        deterministic: bool = True,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path) if model_path else config.PPO_MODEL.with_suffix(".zip")
        self.deterministic = deterministic
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path.exists():
            try:
                from stable_baselines3 import PPO

                self.model = PPO.load(str(self.model_path), device=self.device)
                logger.info(f"Loaded PPO model from {self.model_path}")
            except Exception as exc:
                logger.warning(f"Failed to load PPO model from {self.model_path}: {exc}")
                self.model = None
        else:
            logger.warning(f"PPO weights not found at {self.model_path}")

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def predict(self, obs: np.ndarray) -> tuple[int, float, np.ndarray]:
        """Predict action, confidence, and class probabilities from 6D observation vector.

        Returns (action, confidence, probabilities)
        """
        obs = np.asarray(obs, dtype=np.float32)

        if self.model is None:
            # Fallback heuristic if weights missing
            # If cross-queue > active-queue, switch
            ns = obs[0] + obs[1]
            ew = obs[2] + obs[3]
            phase = int(obs[4])
            curr, other = (ns, ew) if phase == 0 else (ew, ns)
            act = 1 if other > (curr + 0.1) else 0
            probs = np.array([0.2, 0.8] if act == 1 else [0.8, 0.2], dtype=np.float32)
            return act, float(probs[act]), probs

        try:
            import torch

            obs_tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.detach().cpu().numpy()[0]

            if self.deterministic:
                action = int(np.argmax(probs))
            else:
                action = int(self.model.predict(obs, deterministic=False)[0])

            confidence = float(probs[action])
            return action, confidence, probs
        except Exception as exc:
            logger.warning(f"Error during PPO predict: {exc}")
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            act = int(action)
            probs = np.array([0.5, 0.5], dtype=np.float32)
            probs[act] = 0.8
            probs[1 - act] = 0.2
            return act, float(probs[act]), probs


class LivePolicyActuator:
    """NEMA-compliant closed-loop signal actuator supporting multiple controllers."""

    def __init__(
        self,
        controller_type: str = "ppo",
        min_green: float = 10.0,
        max_green: float = 60.0,
        yellow_time: float = 3.0,
        all_red_time: float = 1.0,
        fixed_green: float = 30.0,
        actuated_margin: float = 2.0,
        ppo_model_path: Optional[Union[Path, str]] = None,
    ) -> None:
        self.controller_type = controller_type.lower()
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time
        self.fixed_green = fixed_green
        self.actuated_margin = actuated_margin

        # Controllers
        self.ppo = LivePPOController(model_path=ppo_model_path)

        # State machine
        self.active_phase: int = 0  # 0: North-South, 1: East-West
        self.active_aspect: LiveSignalAspect = LiveSignalAspect.GREEN
        self.green_elapsed_s: float = 0.0
        self.time_in_state_s: float = 0.0

        # Operational metrics
        self.total_switches: int = 0
        self.last_decision: Optional[PolicyDecision] = None

    def set_controller(self, controller_type: str) -> None:
        """Hot-swap controller mode ('ppo', 'actuated', 'fixed') without resetting signal state."""
        self.controller_type = controller_type.lower()

    def reset(self, phase: int = 0) -> None:
        """Reset signal actuator to initial green state."""
        self.active_phase = phase
        self.active_aspect = LiveSignalAspect.GREEN
        self.green_elapsed_s = 0.0
        self.time_in_state_s = 0.0
        self.total_switches = 0
        self.last_decision = None

    def _evaluate_policy_raw(self, obs: np.ndarray, raw_queues: np.ndarray) -> tuple[int, float, np.ndarray, str]:
        """Compute the unconstrained decision from the selected controller."""
        c_type = self.controller_type

        if c_type in ("fixed", "fixed-time"):
            raw_act = 1 if self.green_elapsed_s >= self.fixed_green else 0
            conf = 1.0
            probs = np.array([0.0, 1.0] if raw_act == 1 else [1.0, 0.0], dtype=np.float32)
            rat = (
                f"Fixed green expired ({self.green_elapsed_s:.1f}s >= {self.fixed_green:.1f}s) -> Requesting switch"
                if raw_act == 1
                else f"Fixed green holding ({self.green_elapsed_s:.1f}s < {self.fixed_green:.1f}s)"
            )
            return raw_act, conf, probs, rat

        if c_type in ("actuated", "rule-based"):
            ns = float(raw_queues[0] + raw_queues[1])
            ew = float(raw_queues[2] + raw_queues[3])
            curr, other = (ns, ew) if self.active_phase == 0 else (ew, ns)

            if (other - curr) >= self.actuated_margin:
                raw_act = 1
                rat = f"Cross queue exceeds active queue by {other - curr:.1f} PCU (margin={self.actuated_margin}) -> Requesting switch"
            elif self.green_elapsed_s >= self.max_green:
                raw_act = 1
                rat = f"Max green reached ({self.green_elapsed_s:.1f}s >= {self.max_green:.1f}s) -> Forcing switch"
            else:
                raw_act = 0
                rat = f"Active corridor dominant (Curr: {curr:.1f} vs Cross: {other:.1f} PCU) -> Holding green"

            probs = np.array([0.1, 0.9] if raw_act == 1 else [0.9, 0.1], dtype=np.float32)
            return raw_act, 1.0, probs, rat

        # Default: PPO Neural Agent
        raw_act, conf, probs = self.ppo.predict(obs)
        if raw_act == 1:
            rat = f"PPO Neural Policy predicts switch (confidence {conf*100:.1f}%) based on queue balance"
        else:
            rat = f"PPO Neural Policy predicts hold green (confidence {conf*100:.1f}%)"
        return raw_act, conf, probs, rat

    def step(
        self,
        queues: Union[dict[str, float], Sequence[float]],
        dt: float = 1.0,
    ) -> PolicyDecision:
        """Advance the signal state machine by dt seconds and evaluate control policy.

        Enforces NEMA yellow clearance (3s), all-red interval (1s), and min-green clamping.
        """
        obs = assemble_observation(queues, self.active_phase, self.green_elapsed_s)
        raw_q = np.array([
            obs[0] * 20.0,
            obs[1] * 20.0,
            obs[2] * 20.0,
            obs[3] * 20.0,
        ], dtype=np.float32)

        self.time_in_state_s += dt
        switched = False

        # State 1: In Yellow Clearance
        if self.active_aspect == LiveSignalAspect.YELLOW:
            if self.time_in_state_s >= self.yellow_time:
                # Yellow expired -> Transition to All-Red
                self.active_aspect = LiveSignalAspect.ALL_RED
                self.time_in_state_s = 0.0
                decision = PolicyDecision(
                    action=0,
                    raw_action=1,
                    policy_name=self.controller_type.upper(),
                    active_phase=self.active_phase,
                    active_aspect=self.active_aspect,
                    green_elapsed_s=self.green_elapsed_s,
                    time_in_state_s=self.time_in_state_s,
                    observation=obs,
                    probabilities=np.array([1.0, 0.0], dtype=np.float32),
                    confidence=1.0,
                    rationale=f"Yellow expired -> All-red clearance active ({self.all_red_time:.1f}s)",
                    switched=False,
                )
            else:
                decision = PolicyDecision(
                    action=0,
                    raw_action=1,
                    policy_name=self.controller_type.upper(),
                    active_phase=self.active_phase,
                    active_aspect=self.active_aspect,
                    green_elapsed_s=self.green_elapsed_s,
                    time_in_state_s=self.time_in_state_s,
                    observation=obs,
                    probabilities=np.array([1.0, 0.0], dtype=np.float32),
                    confidence=1.0,
                    rationale=f"Yellow clearance active ({self.time_in_state_s:.1f}s / {self.yellow_time:.1f}s)",
                    switched=False,
                )
            self.last_decision = decision
            return decision

        # State 2: In All-Red Clearance
        if self.active_aspect == LiveSignalAspect.ALL_RED:
            if self.time_in_state_s >= self.all_red_time:
                # All-red expired -> Open new Green phase
                self.active_phase = 1 - self.active_phase
                self.active_aspect = LiveSignalAspect.GREEN
                self.green_elapsed_s = 0.0
                self.time_in_state_s = 0.0
                self.total_switches += 1
                switched = True
                new_phase_name = "North-South" if self.active_phase == 0 else "East-West"
                decision = PolicyDecision(
                    action=0,
                    raw_action=1,
                    policy_name=self.controller_type.upper(),
                    active_phase=self.active_phase,
                    active_aspect=self.active_aspect,
                    green_elapsed_s=self.green_elapsed_s,
                    time_in_state_s=self.time_in_state_s,
                    observation=obs,
                    probabilities=np.array([1.0, 0.0], dtype=np.float32),
                    confidence=1.0,
                    rationale=f"Clearance complete -> {new_phase_name} GREEN OPENED",
                    switched=switched,
                )
            else:
                decision = PolicyDecision(
                    action=0,
                    raw_action=1,
                    policy_name=self.controller_type.upper(),
                    active_phase=self.active_phase,
                    active_aspect=self.active_aspect,
                    green_elapsed_s=self.green_elapsed_s,
                    time_in_state_s=self.time_in_state_s,
                    observation=obs,
                    probabilities=np.array([1.0, 0.0], dtype=np.float32),
                    confidence=1.0,
                    rationale=f"All-red intersection clearance ({self.time_in_state_s:.1f}s / {self.all_red_time:.1f}s)",
                    switched=False,
                )
            self.last_decision = decision
            return decision

        # State 3: In Active Green Phase
        self.green_elapsed_s += dt
        raw_act, conf, probs, rationale = self._evaluate_policy_raw(obs, raw_q)

        # Enforce safety guardrails: Minimum Green & Maximum Green
        if raw_act == 1:
            if self.green_elapsed_s < self.min_green:
                # Suppress switch: minimum green not met
                action = 0
                rationale = (
                    f"Switch requested by {self.controller_type.upper()} but CLAMPED by Min-Green "
                    f"({self.green_elapsed_s:.1f}s < {self.min_green:.1f}s required)"
                )
            else:
                # Trigger Yellow transition!
                action = 1
                self.active_aspect = LiveSignalAspect.YELLOW
                self.time_in_state_s = 0.0
                rationale = f"{rationale} -> Initiating {self.yellow_time:.1f}s Yellow Clearance"
        else:
            # Policy says hold green, but check Max-Green clamp
            if self.green_elapsed_s >= self.max_green:
                action = 1
                self.active_aspect = LiveSignalAspect.YELLOW
                self.time_in_state_s = 0.0
                rationale = f"MAX-GREEN safety limit reached ({self.green_elapsed_s:.1f}s >= {self.max_green:.1f}s) -> Forcing Yellow Clearance"
            else:
                action = 0

        decision = PolicyDecision(
            action=action,
            raw_action=raw_act,
            policy_name=self.controller_type.upper(),
            active_phase=self.active_phase,
            active_aspect=self.active_aspect,
            green_elapsed_s=self.green_elapsed_s,
            time_in_state_s=self.time_in_state_s,
            observation=obs,
            probabilities=probs,
            confidence=conf,
            rationale=rationale,
            switched=switched,
        )
        self.last_decision = decision
        return decision

    def sync_to_traci(self, conn: Any, tls_id: str, states: dict[str, str]) -> None:
        """Transmit current phase and aspect state to SUMO traffic light."""
        if conn is None or not tls_id or not states:
            return

        aspect = self.active_aspect
        phase = self.active_phase

        if aspect == LiveSignalAspect.YELLOW:
            key = "ns_yellow" if phase == 0 else "ew_yellow"
        elif aspect == LiveSignalAspect.ALL_RED:
            # All red state is all 'r'
            key = None
            all_red = "r" * len(states.get("ns_green", "rrrrrrrr"))
            try:
                conn.trafficlight.setRedYellowGreenState(tls_id, all_red)
                return
            except Exception:
                pass
        else:
            key = "ns_green" if phase == 0 else "ew_green"

        if key and key in states:
            try:
                conn.trafficlight.setRedYellowGreenState(tls_id, states[key])
            except Exception as exc:
                logger.debug(f"Failed to apply TLS state to TraCI: {exc}")
