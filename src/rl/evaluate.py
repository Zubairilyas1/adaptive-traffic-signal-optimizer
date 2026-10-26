"""Evaluate controllers on the SUMO intersection and collect traffic metrics.

Metrics
-------
avg_queue    : mean total halting vehicles across the episode
total_wait   : sum of edge waiting time per step (veh-seconds) -> delay proxy
throughput   : vehicles that completed their trip
avg_delay    : total_wait / throughput  (seconds per vehicle)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.sumo.env import SumoIntersectionEnv


@dataclass
class EpisodeMetrics:
    controller: str
    avg_queue: float
    total_wait: float
    throughput: int
    avg_delay: float
    steps: int
    sim_time: float

    def as_dict(self) -> dict:
        return asdict(self)


def run_episode(env: SumoIntersectionEnv, controller, seed: int = 42) -> EpisodeMetrics:
    obs, _ = env.reset(seed=seed)
    controller.reset()

    q_sum = 0.0
    wait_sum = 0.0
    arrived = 0
    steps = 0

    while True:
        action = controller.act(env, obs)
        obs, _, terminated, truncated, _ = env.step(action)

        q_sum += float(env.raw_queues().sum())
        wait_sum += float(sum(env.conn.edge.getWaitingTime(e) for e in env.incoming_edges))
        arrived += int(env.conn.simulation.getArrivedNumber())
        steps += 1

        if terminated or truncated:
            break

    avg_queue = q_sum / max(steps, 1)
    avg_delay = wait_sum / max(arrived, 1)
    return EpisodeMetrics(
        controller=getattr(controller, "name", controller.__class__.__name__),
        avg_queue=round(avg_queue, 3),
        total_wait=round(wait_sum, 1),
        throughput=arrived,
        avg_delay=round(avg_delay, 3),
        steps=steps,
        sim_time=env._sim_time,
    )


def evaluate_controllers(
    controllers: list,
    demand: dict[str, float] | None = None,
    duration: float = 1200.0,
    seeds: list[int] | None = None,
    delta_time: float = 5.0,
) -> list[dict]:
    """Run every controller across seeds; return mean metrics per controller."""
    seeds = seeds or [42]
    results: list[dict] = []

    for controller in controllers:
        runs = []
        for seed in seeds:
            env = SumoIntersectionEnv(
                demand=demand, duration=duration, seed=seed, delta_time=delta_time,
                label=getattr(controller, "name", "controller"),
            )
            try:
                runs.append(run_episode(env, controller, seed=seed).as_dict())
            finally:
                env.close()

        agg = {"controller": runs[0]["controller"], "runs": len(runs)}
        for key in ("avg_queue", "total_wait", "throughput", "avg_delay"):
            agg[key] = round(float(np.mean([r[key] for r in runs])), 3)
        results.append(agg)

    return results
