"""Train a PPO agent for adaptive signal control on the SUMO intersection.

Runs entirely on CPU (RL on a small state space does not need the GPU, and the
laptop's GPU thermally throttles), which also keeps the machine cool.
"""

from __future__ import annotations

from pathlib import Path

from src import config
from src.sumo.env import SumoIntersectionEnv


def train_ppo(
    timesteps: int = 150_000,
    duration: float = 900.0,
    seed: int = 42,
    demand: dict[str, float] | None = None,
    out: Path | None = None,
    device: str = "cpu",
    verbose: int = 1,
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    out = out or config.PPO_MODEL
    out.parent.mkdir(parents=True, exist_ok=True)

    env = SumoIntersectionEnv(demand=demand, duration=duration, seed=seed, label="train")
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        seed=seed,
        device=device,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(str(out))
    env.close()
    return out


if __name__ == "__main__":
    path = train_ppo(timesteps=1000)
    print(f"saved -> {path}")
