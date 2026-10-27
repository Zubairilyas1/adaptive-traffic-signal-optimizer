"""Signal control decisions and NEMA safety watchdogs."""

from src.decision.actuator import AdaptiveActuator, ControllerState, DecisionReason, SignalAspect
from src.decision.dual_regime import DualRegimeOptimizer, DualRegimePlan, OptimizationRegime
from src.decision.fail_safe import FailureReason, HealthState, NEMAFailSafeWatchdog
from src.decision.intersection import Intersection, SignalPlan, load_intersection
from src.decision.webster import optimize, webster_cycle, webster_greens

__all__ = [
    "AdaptiveActuator",
    "ControllerState",
    "DecisionReason",
    "SignalAspect",
    "DualRegimeOptimizer",
    "DualRegimePlan",
    "OptimizationRegime",
    "FailureReason",
    "HealthState",
    "NEMAFailSafeWatchdog",
    "Intersection",
    "SignalPlan",
    "load_intersection",
    "optimize",
    "webster_cycle",
    "webster_greens",
]
