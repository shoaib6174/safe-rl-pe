"""Step 2 visual gate: results/visual_gates/step_02/step_02_occlusion.gif

Scripted straight-line evader passes behind an obstacle while the pursuer
rotates to track it (FOV always contains the evader, so LOS occlusion is the
ONLY mask in play — this isolates the ray-cast).

Two passes (deterministic geometry, hand-computed expectations):
- pass A (main): evader (-3, 2.5) -> (7, 2.5) at y=2.5, obstacle (0,0) r=1,
  pursuer (-3,-3). Occluded interval x in (0.35, 6.02) [solved analytically]
  -> exactly 2 LOS transitions (break, re-acquire).

Note: the re-acquire point (6.02, 2.5) is 10.6m from the pursuer — beyond the
locked 8m FOV range — so this gate uses fov_range=12.0 to keep the LOS break
the ONLY mask in play (occlusion geometry is independent of range). The
locked production value stays 8.0 (D18).
- pass B (grazing): evader (5, 2.5) -> (7, 2.5): starts occluded, crosses
  the shadow edge once near tangency -> exactly 1 transition, no flicker.
  (Ends at x=7: beyond that the evader exits the 12m FOV range, which would
  add a second, range-induced transition that isn't what this pass tests.)

Every step's visibility is logged with its rendered frame index to
step_02_los_events.json so the GIF's blip changes can be cross-checked
frame-by-frame (plan Step 2 checklist).

Usage: ./venv/bin/python scripts/pe2_step02_visual_gate.py
"""

import json
import pathlib

import numpy as np

from pe2.config import EnvConfig, SensingConfig
from pe2.env import PursuitEvasionEnv, wrap_angle
from pe2.render import FovOverlay, Renderer, Trails, save_gif
from pe2.sensing import is_visible

OUT_DIR = pathlib.Path("results/visual_gates/step_02")
FPS = 20
HOLD_FRAMES = 10

OBSTACLE = {"x": 0.0, "y": 0.0, "radius": 1.0}
PURSUER_START = (-3.0, -3.0, np.pi / 4)
PASSES = {
    "A_main": {
        "evader_start": (-3.0, 2.5, 0.0),
        "until_x": 7.0,
        "expect_transitions": 2,
    },
    "B_grazing": {
        "evader_start": (5.0, 2.5, 0.0),
        "until_x": 7.0,
        "expect_transitions": 1,
    },
}


def track_action(env: PursuitEvasionEnv) -> np.ndarray:
    """Pursuer rotates to face the evader (v=0); evader drives straight."""
    p = env.pursuer_pose
    e = env.evader_pose
    bearing = np.arctan2(e[1] - p[1], e[0] - p[0])
    omega = float(
        np.clip(5.0 * wrap_angle(bearing - p[2]), -env.cfg.omega_max, env.cfg.omega_max)
    )
    return np.array([0.0, omega, env.cfg.evader_v_max, 0.0])


def main() -> None:
    # Larger arena so the scripted passes fit inside the walls (default 10x10
    # clips x at +/-4.85; both passes end at x=7.0). The
    # transition geometry depends only on pursuer/obstacle/evader-line
    # geometry, so expected transition counts are unaffected.
    cfg = EnvConfig(arena_size=20.0, sensing=SensingConfig(fov_range=12.0))
    env = PursuitEvasionEnv(cfg)
    renderer = Renderer(cfg, dpi=80, figsize=(5, 5))
    overlay = FovOverlay("pursuer")
    trails = Trails()
    frames: list[np.ndarray] = []
    log: dict[str, list] = {}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for pass_name, spec in PASSES.items():
        env.reset(seed=0)
        env.obstacles = [dict(OBSTACLE)]
        env.pursuer_pose = np.array(PURSUER_START, dtype=float)
        env.evader_pose = np.array(spec["evader_start"], dtype=float)
        env.current_step = 0
        trails.reset()

        steps = []
        while env.evader_pose[0] < spec["until_x"]:
            joint = track_action(env)
            env.step(joint)
            trails.append("pursuer", *env.pursuer_pose[:2])
            trails.append("evader", *env.evader_pose[:2])
            visible = bool(
                is_visible(
                    env.pursuer_pose,
                    (env.evader_pose[0], env.evader_pose[1]),
                    env.obstacles,
                    cfg.sensing,
                )
            )
            frames.append(
                renderer.draw(env, trails, overlays=[overlay], title_prefix=pass_name)
            )
            steps.append(
                {
                    "step": env.current_step,
                    "frame_idx": len(frames) - 1,
                    "evader_x": round(float(env.evader_pose[0]), 4),
                    "visible": visible,
                }
            )
        # hold terminal frame of the pass
        frames.extend([frames[-1]] * HOLD_FRAMES)

        transitions = [
            (steps[i]["step"], steps[i]["frame_idx"], steps[i]["visible"])
            for i in range(1, len(steps))
            if steps[i]["visible"] != steps[i - 1]["visible"]
        ]
        log[pass_name] = {
            "steps": steps,
            "transitions": transitions,
            "expected_transitions": spec["expect_transitions"],
        }
        print(
            f"{pass_name}: {len(steps)} steps, transitions={transitions} "
            f"(expected {spec['expect_transitions']})"
        )
        assert len(transitions) == spec["expect_transitions"], (
            f"{pass_name}: expected {spec['expect_transitions']} LOS transitions, "
            f"got {len(transitions)}"
        )

    with open(OUT_DIR / "step_02_los_events.json", "w") as f:
        json.dump(log, f, indent=2)

    out_path = OUT_DIR / "step_02_occlusion.gif"
    save_gif(frames, str(out_path), fps=FPS)
    renderer.close()
    print(f"wrote {out_path} ({len(frames)} frames) + step_02_los_events.json")


if __name__ == "__main__":
    main()
