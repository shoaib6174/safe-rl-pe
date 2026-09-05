"""How long will the v2 gate take? 50 episodes per arm on the pre-registered
streams' first seeds is NOT a gate run; it only extrapolates wall-clock.

Usage: ./venv/bin/python scripts/pe2_step06_timing_probe.py
"""

from __future__ import annotations

import time

from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.scripted import (
    GreedyEvader,
    GreedyPursuer,
    ObstacleHuggingEvader,
    run_shielded_duel,
)
from pe2.shield import CBFShield

N = 50


def main() -> None:
    cfg = EnvConfig()
    for label, evader_cls, seed0 in (
        ("main", GreedyEvader, 800000),
        ("probe", ObstacleHuggingEvader, 810000),
    ):
        env = PursuitEvasionEnv(cfg)
        shield = CBFShield(cfg)
        t0 = time.perf_counter()
        steps = sum(
            run_shielded_duel(env, GreedyPursuer(), evader_cls(), shield, seed0 + i)[
                "steps"
            ]
            for i in range(N)
        )
        dt = time.perf_counter() - t0
        print(
            f"{label}: {N} eps, {steps} steps, {dt:.1f} s, {1e6 * dt / steps:.0f} us/step, "
            f"-> ~{dt / N * 1000 / 60:.1f} min per 1000-episode arm"
        )


if __name__ == "__main__":
    main()
