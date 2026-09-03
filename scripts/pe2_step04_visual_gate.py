"""Step 4 visual gate: results/visual_gates/step_04/step_04_belief.gif

Scripted scenario: the pursuer P-tracks the evader (FOV always contains it,
so only LOS occlusion masks) while a LastSeenBelief tracker rides the
pursuer's obs stream. The evader drives straight into the obstacle shadow,
then TURNS while occluded — dead reckoning cannot follow a turn it cannot
see, so the ghost must visibly drift away from the truth (no snap-to-truth),
the sigma disk must grow, and on re-acquisition the belief must snap back to
the observation within one step (noiseless sim -> well under the ~1s gate).

Same gate-only scenario config as Step 2 (arena 20, fov_range 12) so the
occlusion geometry is legible; locked production defaults unchanged.

Automated assertions (also logged to step_04_belief_events.json):
- an occluded run of >= 10 steps occurs, then re-acquisition occurs
- mid-occlusion (20 steps after the turn starts) belief error > 0.3 m
- sigma strictly increasing across the occluded run
- at the re-acquisition step belief error ~ 0 (snap to observation)

Usage: ./venv/bin/python scripts/pe2_step04_visual_gate.py
"""

import json
import pathlib
from itertools import pairwise

import numpy as np

from pe2.belief import LastSeenBelief
from pe2.config import EnvConfig, SensingConfig
from pe2.env import PursuitEvasionEnv, wrap_angle
from pe2.render import BeliefOverlay, FovOverlay, Renderer, Trails, save_gif

OUT_DIR = pathlib.Path("results/visual_gates/step_04")
FPS = 20
HOLD_FRAMES = 15

OBSTACLE = {"x": 0.0, "y": 0.0, "radius": 1.0}
PURSUER_START = (-3.0, -3.0, np.pi / 4)
EVADER_START = (-3.0, 2.5, 0.0)
TURN_START_STEP = 80  # ~24 steps after the LOS break (step ~56): mid-occlusion
TURN_OMEGA = 1.0  # rad/s left turn while occluded
MAX_STEPS = 400
K_OMEGA = 5.0


def track_action(env: PursuitEvasionEnv) -> np.ndarray:
    """Pursuer rotates to face the evader (v=0); evader straight, then turns."""
    p = env.pursuer_pose
    e = env.evader_pose
    bearing = np.arctan2(e[1] - p[1], e[0] - p[0])
    omega = float(
        np.clip(K_OMEGA * wrap_angle(bearing - p[2]), -env.cfg.omega_max, env.cfg.omega_max)
    )
    omega_e = TURN_OMEGA if env.current_step >= TURN_START_STEP else 0.0
    return np.array([0.0, omega, env.cfg.evader_v_max, omega_e])


def main() -> None:
    cfg = EnvConfig(arena_size=20.0, sensing=SensingConfig(fov_range=12.0))
    env = PursuitEvasionEnv(cfg)
    renderer = Renderer(cfg, dpi=80, figsize=(5, 5))
    belief = LastSeenBelief(cfg)
    overlays = [FovOverlay("pursuer"), BeliefOverlay(belief)]
    trails = Trails()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env.reset(seed=0)
    env.obstacles = [dict(OBSTACLE)]
    env.pursuer_pose = np.array(PURSUER_START, dtype=float)
    env.evader_pose = np.array(EVADER_START, dtype=float)
    env.current_step = 0

    frames: list[np.ndarray] = []
    events = []  # per-step: visibility, belief error, sigma
    while env.current_step < MAX_STEPS:
        joint = track_action(env)
        obs, _, terminated, truncated, _ = env.step(joint)
        trails.append("pursuer", *env.pursuer_pose[:2])
        trails.append("evader", *env.evader_pose[:2])
        belief.update(obs["pursuer"])
        events.append(
            {
                "step": env.current_step,
                "visible": bool(obs["pursuer"][6] > 0.5),
                "belief_error": float(np.hypot(*(belief.mean - env.evader_pose[:2])))
                if belief.seen
                else None,
                "sigma": belief.sigma if belief.seen else None,
            }
        )
        frames.append(renderer.draw(env, trails, overlays=overlays, title_prefix="belief"))
        if terminated or truncated:
            break
    frames.extend([frames[-1]] * HOLD_FRAMES)

    with open(OUT_DIR / "step_04_belief_events.json", "w") as f:
        json.dump(events, f, indent=2)

    # --- assertions (checklist, automated part) ---
    vis = [e["visible"] for e in events]
    assert any(vis), "evader never visible — scenario broken"

    # first occluded run (visible -> >=10 consecutive invisible -> visible)
    first_break = next(i for i in range(1, len(vis)) if vis[i - 1] and not vis[i])
    reacquire = next(
        (i for i in range(first_break + 10, len(vis)) if not vis[i - 1] and vis[i]),
        None,
    )
    assert reacquire is not None, "no re-acquisition after occlusion"
    assert reacquire - first_break >= 10, "occluded run too short to be meaningful"
    print(f"occluded run: steps {events[first_break]['step']}..{events[reacquire - 1]['step']} "
          f"({reacquire - first_break} steps), re-acquired at step {events[reacquire]['step']}")

    # sigma strictly increasing across the occluded run
    sigmas = [events[i]["sigma"] for i in range(first_break, reacquire)]
    assert all(b > a for a, b in pairwise(sigmas)), "sigma not growing"

    # mid-occlusion error (10 steps after the turn starts): no snap-to-truth
    probe = TURN_START_STEP + 10
    assert not vis[probe - 1], f"probe step {probe} should be occluded"
    err_probe = events[probe - 1]["belief_error"]
    assert err_probe > 0.3, f"belief followed truth while occluded (err={err_probe:.3f})"
    print(f"mid-occlusion (step {probe}): belief error = {err_probe:.3f} m > 0.3 OK (no leak)")

    # re-acquisition: error ~ 0 within one step (noiseless sim << 1s gate)
    err_re = events[reacquire]["belief_error"]
    assert err_re < 1e-6, f"belief did not snap to observation on re-acquisition ({err_re})"
    print(f"re-acquisition: belief error = {err_re:.2e} m within 1 step (<< ~1s gate) OK")

    save_gif(frames, str(OUT_DIR / "step_04_belief.gif"), fps=FPS)
    renderer.close()
    print(f"wrote {OUT_DIR / 'step_04_belief.gif'} ({len(frames)} frames) + events JSON")


if __name__ == "__main__":
    main()
