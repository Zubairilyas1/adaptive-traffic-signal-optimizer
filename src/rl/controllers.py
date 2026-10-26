"""The three signal controllers compared in the study.

Fixed-time      : alternate phases on a fixed green (no adaptation).
Rule-based      : actuated - extend green while the current phase is busier.
PPO             : learned policy (stable-baselines3) on the SUMO environment.

All expose ``reset()`` and ``act(env, obs) -> int`` (0 keep, 1 switch).
"""

from __future__ import annotations

import numpy as np


class FixedTimeController:
    """Classic pre-timed signal: fixed green per phase."""

    name = "Fixed-Time"

    def __init__(self, green: float = 30.0) -> None:
        self.green = green

    def reset(self) -> None:
        pass

    def act(self, env, obs=None) -> int:
        return 1 if env.green_elapsed >= self.green else 0


class ActuatedController:
    """Rule-based adaptive: switch when the other phase has built up enough queue."""

    name = "Rule-Based"

    def __init__(self, min_green: float = 10.0, max_green: float = 50.0, margin: float = 2.0) -> None:
        self.min_green = min_green
        self.max_green = max_green
        self.margin = margin

    def reset(self) -> None:
        pass

    def act(self, env, obs=None) -> int:
        q = env.raw_queues()          # [N, S, E, W]
        ns = q[0] + q[1]
        ew = q[2] + q[3]
        current, other = (ns, ew) if env.phase == 0 else (ew, ns)

        if env.green_elapsed < self.min_green:
            return 0
        if env.green_elapsed >= self.max_green:
            return 1
        # give way once the other approach is clearly more congested
        return 1 if (other - current) >= self.margin else 0


class PPOController:
    """Learned policy from a trained stable-baselines3 PPO model."""

    name = "PPO"

    def __init__(self, model, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    def reset(self) -> None:
        pass

    def act(self, env, obs=None) -> int:
        if obs is None:
            obs = env._observation()
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)
