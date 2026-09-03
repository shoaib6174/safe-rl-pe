"""Step 3 visual gate: results/visual_gates/step_03/step_03_scripted_duels.gif

4 duels — one per scripted matchup (greedy x greedy, and each role vs its
FOV-limited counterpart, plus FL x FL) — rendered with the pursuer's FOV
overlay. Episodes use GATE_SEEDS below, chosen from the N=500 battery for
illustrative value (a capture with LOS breaks where the battery offers one;
a multi-break timeout for FLp x Ge to show searching). Every rendered
episode's outcome/steps/LOS-breaks are asserted equal to that seed's row in
step_03_battery.json (checklist: outcomes match the log line-by-line). Run
pe2_step03_battery.py first.

Checklist (inspect extracted frames):
- pursuer visibly pursues; evader flees and uses obstacles to break LOS
- FOV-limited pursuer visibly loses track after occlusion (searching
  behavior, not teleporting to target)
- outcomes match the log line-by-line for the rendered episodes (asserted)

Usage: ./venv/bin/python scripts/pe2_step03_visual_gate.py
"""

import json
import pathlib

import numpy as np

from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.render import FovOverlay, Renderer, Trails, save_gif
from pe2.scripted import (
    FovLimitedEvader,
    FovLimitedPursuer,
    GreedyEvader,
    GreedyPursuer,
)

OUT_DIR = pathlib.Path("results/visual_gates/step_03")
GIF_PATH = OUT_DIR / "step_03_scripted_duels.gif"
FPS = 20
STRIDE = 4
HOLD_FRAMES = 15

MATCHUPS = [
    ("Gp_x_Ge", GreedyPursuer, GreedyEvader),
    ("Gp_x_FLe", GreedyPursuer, FovLimitedEvader),
    ("FLp_x_Ge", FovLimitedPursuer, GreedyEvader),
    ("FLp_x_FLe", FovLimitedPursuer, FovLimitedEvader),
]
# Seeds picked from the N=500 battery (see module docstring).
GATE_SEEDS = {
    "Gp_x_Ge": 700427,  # capture @ 98 steps, 1 LOS break
    "Gp_x_FLe": 700352,  # capture @ 221 steps, 1 LOS break
    "FLp_x_Ge": 700220,  # timeout, 2 LOS breaks (sustained searching)
    "FLp_x_FLe": 700209,  # capture @ 561 steps, 2 LOS breaks
}


def main() -> None:
    with open(OUT_DIR / "step_03_battery.json") as f:
        battery = json.load(f)

    cfg = EnvConfig()
    env = PursuitEvasionEnv(cfg)
    renderer = Renderer(cfg, dpi=80, figsize=(5, 5))
    overlay = FovOverlay("pursuer")
    frames: list[np.ndarray] = []
    rendered_logs = []

    for name, p_cls, e_cls in MATCHUPS:
        seed = GATE_SEEDS[name]
        pursuer, evader = p_cls(), e_cls()
        pursuer.reset()
        evader.reset()
        trails = Trails()
        obs, _ = env.reset(seed=seed)
        los_prev = bool(obs["pursuer"][6] > 0.5)
        los_breaks = 0

        while True:
            v_p, w_p = pursuer.act(obs["pursuer"], env)
            v_e, w_e = evader.act(obs["evader"], env)
            obs, _, terminated, truncated, info = env.step([v_p, w_p, v_e, w_e])
            los = bool(obs["pursuer"][6] > 0.5)
            if los_prev and not los:
                los_breaks += 1
            los_prev = los
            trails.append("pursuer", *env.pursuer_pose[:2])
            trails.append("evader", *env.evader_pose[:2])
            if env.current_step % STRIDE == 0 or terminated or truncated:
                frames.append(
                    renderer.draw(env, trails, overlays=[overlay], title_prefix=name)
                )
            if terminated or truncated:
                break

        frames.extend([frames[-1]] * HOLD_FRAMES)
        rendered = {
            "matchup": name,
            "seed": seed,
            "outcome": info["outcome"],
            "steps": env.current_step,
            "los_breaks": los_breaks,
        }
        rendered_logs.append(rendered)

        # checklist: outcome matches the battery log line-by-line
        (battery_row,) = [row for row in battery["logs"][name] if row["seed"] == seed]
        for key in ("outcome", "steps", "los_breaks"):
            assert rendered[key] == battery_row[key], (
                f"{name} seed {seed}: rendered {key}={rendered[key]} "
                f"!= battery {key}={battery_row[key]}"
            )
        print(
            f"{name} (seed {seed}): {rendered['outcome']} at step "
            f"{rendered['steps']}, los_breaks={rendered['los_breaks']} "
            f"— matches battery row OK"
        )

    with open(OUT_DIR / "step_03_rendered_log.json", "w") as f:
        json.dump(rendered_logs, f, indent=2)

    save_gif(frames, str(GIF_PATH), fps=FPS)
    renderer.close()
    print(f"wrote {GIF_PATH} ({len(frames)} frames) + step_03_rendered_log.json")


if __name__ == "__main__":
    main()
