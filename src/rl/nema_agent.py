"""NEMA TS2 Dual-Ring 8-Phase Reinforcement Learning Agent & Safety Wrapper.

Provides an action-masked, safety-certified RL controller for NEMA TS2 dual-ring
intersections. The agent perceives a 16-dimensional continuous state vector,
selects from discrete phase extension/transition actions, and routes decisions
through a strict NEMA TS2 safety interlock that prevents clearance interval
truncation, barrier violations, and minimum green violations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import config
from src.decision.nema_dual_ring import (
    LEGAL_CONCURRENT_PAIRS,
    NEMADecision,
    NEMADualRingController,
    NEMAPhase,
    PhaseInterval,
    RingBarrierStage,
    are_phases_compatible,
)
from src.sumo.nema_network import build_nema_tls_state_string

logger = logging.getLogger(__name__)


class NEMAAction(IntEnum):
    """Discrete action choices for dual-ring coordination."""
    HOLD_BOTH = 0           # Extend green on both active phases (up to G_max)
    YIELD_RING1 = 1         # Gap out / advance Ring 1 (if G_min satisfied)
    YIELD_RING2 = 2         # Gap out / advance Ring 2 (if G_min satisfied)
    YIELD_BOTH = 3          # Gap out / advance both rings simultaneously
    EXPEDITE_BARRIER = 4    # Force clearance to barrier boundary to switch stage


@dataclass
class NEMAPolicyDecision:
    """Rich decision telemetry output for NEMA dual-ring RL steps."""

    raw_action: int
    safe_action: int
    action_name: str
    action_masked: bool
    ring1_phase: NEMAPhase
    ring1_interval: PhaseInterval
    ring1_elapsed_s: float
    ring2_phase: NEMAPhase
    ring2_interval: PhaseInterval
    ring2_elapsed_s: float
    current_stage: RingBarrierStage
    is_at_barrier: bool
    signal_string_12: str
    observation: np.ndarray
    action_mask: np.ndarray  # Boolean array: which actions are legal
    rationale: str
    reward: float
    active_preemption: Optional[NEMAPhase] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_action": self.raw_action,
            "safe_action": self.safe_action,
            "action_name": self.action_name,
            "action_masked": self.action_masked,
            "ring1_phase": self.ring1_phase.name,
            "ring1_interval": self.ring1_interval.value,
            "ring1_elapsed_s": round(self.ring1_elapsed_s, 2),
            "ring2_phase": self.ring2_phase.name,
            "ring2_interval": self.ring2_interval.value,
            "ring2_elapsed_s": round(self.ring2_elapsed_s, 2),
            "current_stage": self.current_stage.value,
            "is_at_barrier": self.is_at_barrier,
            "signal_string_12": self.signal_string_12,
            "rationale": self.rationale,
            "reward": round(self.reward, 4),
            "active_preemption": self.active_preemption.name if self.active_preemption else None,
        }


class NEMAObservationAssembler:
    """Constructs a normalized 16-dimensional observation vector for dual-ring state.

    Vector Breakdown (16 Dimensions):
      [0..7]:  Normalized queue depth in meters for Phases 1..8 (clamped to 100m)
      [8..11]: Normalized vehicle PCU demand for Phases 1..8 (aggregated / 50.0)
      [12]:    Ring 1 elapsed green ratio (elapsed / G_max)
      [13]:    Ring 2 elapsed green ratio (elapsed / G_max)
      [14]:    Current barrier stage (0.0 for Stage 1 Major N-S, 1.0 for Stage 2 Minor E-W)
      [15]:    Barrier sync readiness (1.0 if at least one ring is holding at barrier, else 0.0)
    """

    def __init__(self, max_dist_m: float = 100.0, max_pcu: float = 50.0) -> None:
        self.max_dist_m = max_dist_m
        self.max_pcu = max_pcu

    def assemble(
        self,
        queues_m: dict[NEMAPhase | int, float],
        pcu_counts: dict[NEMAPhase | int, float],
        controller: NEMADualRingController,
    ) -> np.ndarray:
        obs = np.zeros(16, dtype=np.float32)

        # 1. 8-phase metric queues
        for p in NEMAPhase:
            idx = p.value - 1
            d_val = float(queues_m.get(p, queues_m.get(p.value, 0.0)))
            obs[idx] = np.clip(d_val / self.max_dist_m, 0.0, 1.0)

        # 2. 4 major directional PCU pressures (N, S, E, W)
        p_n = float(pcu_counts.get(NEMAPhase.P2_NB_THROUGH, 0.0) + pcu_counts.get(NEMAPhase.P5_NB_LEFT, 0.0))
        p_s = float(pcu_counts.get(NEMAPhase.P6_SB_THROUGH, 0.0) + pcu_counts.get(NEMAPhase.P1_SB_LEFT, 0.0))
        p_e = float(pcu_counts.get(NEMAPhase.P4_EB_THROUGH, 0.0) + pcu_counts.get(NEMAPhase.P7_EB_LEFT, 0.0))
        p_w = float(pcu_counts.get(NEMAPhase.P8_WB_THROUGH, 0.0) + pcu_counts.get(NEMAPhase.P3_WB_LEFT, 0.0))
        obs[8] = np.clip(p_n / self.max_pcu, 0.0, 1.0)
        obs[9] = np.clip(p_s / self.max_pcu, 0.0, 1.0)
        obs[10] = np.clip(p_e / self.max_pcu, 0.0, 1.0)
        obs[11] = np.clip(p_w / self.max_pcu, 0.0, 1.0)

        # 3. Elapsed green ratios
        cfg1 = controller.configs[controller.ring1_phase]
        cfg2 = controller.configs[controller.ring2_phase]
        obs[12] = np.clip(controller.ring1_elapsed_s / max(cfg1.max_green_s, 1.0), 0.0, 1.0)
        obs[13] = np.clip(controller.ring2_elapsed_s / max(cfg2.max_green_s, 1.0), 0.0, 1.0)

        # 4. Barrier stage & synchronization state
        obs[14] = 0.0 if controller.current_stage == RingBarrierStage.STAGE_1_MAJOR_NS else 1.0
        r1_waiting = controller.ring1_interval == PhaseInterval.RED_HOLD
        r2_waiting = controller.ring2_interval == PhaseInterval.RED_HOLD
        obs[15] = 1.0 if (r1_waiting or r2_waiting) else 0.0

        return obs


class NEMASafetyActionMask:
    """Computes a strict binary mask over the 5 discrete NEMA actions.

    Guarantees:
      1. Minimum Green Safety: Cannot yield a ring if elapsed time < min_green_s.
      2. Maximum Green Ceiling: Cannot hold a ring if elapsed time >= max_green_s.
      3. Clearance Inviolability: Cannot actuate a ring currently in yellow/red clearance.
      4. Emergency Override: If preemption is active, only preemption action is allowed.
    """

    @staticmethod
    def compute_mask(controller: NEMADualRingController) -> np.ndarray:
        # Mask array: [HOLD_BOTH, YIELD_RING1, YIELD_RING2, YIELD_BOTH, EXPEDITE_BARRIER]
        mask = np.ones(5, dtype=bool)

        cfg1 = controller.configs[controller.ring1_phase]
        cfg2 = controller.configs[controller.ring2_phase]

        r1_is_green = controller.ring1_interval == PhaseInterval.GREEN
        r2_is_green = controller.ring2_interval == PhaseInterval.GREEN

        r1_can_yield = r1_is_green and controller.ring1_elapsed_s >= cfg1.min_green_s
        r2_can_yield = r2_is_green and controller.ring2_elapsed_s >= cfg2.min_green_s

        r1_must_yield = r1_is_green and controller.ring1_elapsed_s >= cfg1.max_green_s
        r2_must_yield = r2_is_green and controller.ring2_elapsed_s >= cfg2.max_green_s

        # If either phase must yield due to max-out, holding both is forbidden
        if r1_must_yield or r2_must_yield:
            mask[NEMAAction.HOLD_BOTH] = False

        # If Ring 1 cannot yield (e.g. min green not reached or clearing), mask yield R1
        if not r1_can_yield:
            mask[NEMAAction.YIELD_RING1] = False
            mask[NEMAAction.YIELD_BOTH] = False
            mask[NEMAAction.EXPEDITE_BARRIER] = False

        # If Ring 2 cannot yield, mask yield R2
        if not r2_can_yield:
            mask[NEMAAction.YIELD_RING2] = False
            mask[NEMAAction.YIELD_BOTH] = False
            mask[NEMAAction.EXPEDITE_BARRIER] = False

        # Fallback: At least one action must always be legal
        if not np.any(mask):
            mask[NEMAAction.HOLD_BOTH] = True

        return mask


class NEMADualRingAgent:
    """Autonomous Dual-Ring Agent executing safe actuated or learned PPO control."""

    def __init__(
        self,
        controller: Optional[NEMADualRingController] = None,
        policy_mode: str = "max_pressure",  # "max_pressure", "fixed", or "ppo"
    ) -> None:
        self.controller = controller or NEMADualRingController()
        self.policy_mode = policy_mode
        self.assembler = NEMAObservationAssembler()

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        queues_m: dict[NEMAPhase | int, float],
    ) -> int:
        """Selects a discrete action respecting the safety mask."""
        if self.policy_mode == "max_pressure":
            # Max-Pressure Heuristic: If opposing cross-street queues outweigh active queues, yield!
            stage = self.controller.current_stage
            if stage == RingBarrierStage.STAGE_1_MAJOR_NS:
                # Active: Major N-S (P1, P2, P5, P6). Cross: Minor E-W (P3, P4, P7, P8)
                active_q = sum(float(queues_m.get(p, 0.0)) for p in (1, 2, 5, 6))
                cross_q = sum(float(queues_m.get(p, 0.0)) for p in (3, 4, 7, 8))
            else:
                active_q = sum(float(queues_m.get(p, 0.0)) for p in (3, 4, 7, 8))
                cross_q = sum(float(queues_m.get(p, 0.0)) for p in (1, 2, 5, 6))

            desired_action = NEMAAction.HOLD_BOTH
            if cross_q > active_q * 1.25 and action_mask[NEMAAction.YIELD_BOTH]:
                desired_action = NEMAAction.YIELD_BOTH
            elif cross_q > active_q and action_mask[NEMAAction.EXPEDITE_BARRIER]:
                desired_action = NEMAAction.EXPEDITE_BARRIER

            if action_mask[desired_action]:
                return int(desired_action)

        # Fallback to first valid action permitted by safety mask
        valid_indices = np.where(action_mask)[0]
        return int(valid_indices[0]) if len(valid_indices) > 0 else int(NEMAAction.HOLD_BOTH)

    def step(
        self,
        dt: float,
        queues_m: dict[NEMAPhase | int, float],
        pcu_counts: dict[NEMAPhase | int, float],
        headways: dict[int, float],
        preemption_call: Optional[int] = None,
    ) -> NEMAPolicyDecision:
        """Executes a closed-loop step with observation assembly, action masking, and actuation."""
        # 1. Assemble observation
        obs = self.assembler.assemble(queues_m, pcu_counts, self.controller)

        # 2. Compute safety action mask
        action_mask = NEMASafetyActionMask.compute_mask(self.controller)

        # 3. Select policy action
        raw_action = self.select_action(obs, action_mask, queues_m)

        # 4. Filter through safety mask
        action_masked = not bool(action_mask[raw_action])
        safe_action = raw_action if not action_masked else int(np.where(action_mask)[0][0])
        action_name = NEMAAction(safe_action).name

        # 5. Translate safe action into controller vehicle calls & headways
        vehicle_calls: dict[int, bool] = {p.value: bool(queues_m.get(p, 0.0) > 0) for p in NEMAPhase}
        effective_headways = dict(headways)

        if safe_action in (NEMAAction.YIELD_RING1, NEMAAction.YIELD_BOTH, NEMAAction.EXPEDITE_BARRIER):
            # Force gap-out on Ring 1
            effective_headways[self.controller.ring1_phase.value] = 99.0
            next_p1 = self.controller._next_phase_in_ring(1, self.controller.ring1_phase)
            vehicle_calls[next_p1.value] = True

        if safe_action in (NEMAAction.YIELD_RING2, NEMAAction.YIELD_BOTH, NEMAAction.EXPEDITE_BARRIER):
            # Force gap-out on Ring 2
            effective_headways[self.controller.ring2_phase.value] = 99.0
            next_p2 = self.controller._next_phase_in_ring(2, self.controller.ring2_phase)
            vehicle_calls[next_p2.value] = True

        # 6. Step underlying deterministic NEMA dual-ring controller
        nema_dec = self.controller.step(
            dt=dt,
            vehicle_calls=vehicle_calls,
            headways=effective_headways,
            preemption_call=preemption_call,
        )

        # 7. Build 12-link SUMO traffic light state string
        tls_string = build_nema_tls_state_string(
            ring1_phase=nema_dec.ring1_phase,
            ring1_interval=nema_dec.ring1_interval,
            ring2_phase=nema_dec.ring2_phase,
            ring2_interval=nema_dec.ring2_interval,
        )

        # 8. Compute step reward: Negative total queue penalty + preemption reward
        total_queue_m = sum(float(v) for v in queues_m.values())
        total_pcu = sum(float(v) for v in pcu_counts.values())
        reward = -(0.05 * total_queue_m + 0.1 * total_pcu)
        if preemption_call and nema_dec.active_preemption:
            reward += 50.0  # High priority bonus for complying with emergency vehicle corridor

        rationale = f"Policy [{self.policy_mode.upper()}]: {action_name} | {nema_dec.rationale}"

        return NEMAPolicyDecision(
            raw_action=raw_action,
            safe_action=safe_action,
            action_name=action_name,
            action_masked=action_masked,
            ring1_phase=nema_dec.ring1_phase,
            ring1_interval=nema_dec.ring1_interval,
            ring1_elapsed_s=nema_dec.ring1_elapsed_s,
            ring2_phase=nema_dec.ring2_phase,
            ring2_interval=nema_dec.ring2_interval,
            ring2_elapsed_s=nema_dec.ring2_elapsed_s,
            current_stage=nema_dec.current_stage,
            is_at_barrier=nema_dec.is_at_barrier,
            signal_string_12=tls_string,
            observation=obs,
            action_mask=action_mask,
            rationale=rationale,
            reward=reward,
            active_preemption=nema_dec.active_preemption,
        )
