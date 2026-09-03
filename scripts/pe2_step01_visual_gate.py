"""Step 1 visual gate: results/visual_gates/step_01/step_01_random_rollouts.gif

Seed 0, 3 episodes, random actions. While generating, asserts the same
invariants as the test suite (in-arena, no obstacle penetration) so the
GIF's checklist item "zero tunneling frames" is backed by an assertion on
every rendered episode, not just eyeballing.

Deterministic: all randomness derives from seed 0 via pe2.rng (plan,
assumption 5). Re-running reproduces the identical GIF frame-for-frame.

Usage: ./venv/bin/python scripts/pe2_step01_visual_gate.py
"""

import pathlib

import numpy as np

from pe2.config import EnvConfig
from pe2.env import OUTCOME_CAPTURE, PursuitEvasionEnv
from pe2.render import Renderer, Trails, save_gif
from pe2.rng import derive_seed, make_rng

GATE_SEED = 0
N_EPISODES = 3
STRIDE = 4  # render every 4th sim step (dt=0.05 -> 5 frames/sim-second)
FPS = 20  # 4x real-time playback
HOLD_FRAMES = 15  # terminal-frame hold between episodes (0.75s at 20 fps)
OUT_DIR = pathlib.Path("results/visual_gates/step_01")


def random_action(rng: np.random.Generator, cfg: EnvConfig) -> np.ndarray:
    return np.array(
        [
            rng.uniform(0, cfg.v_max),
            rng.uniform(-cfg.omega_max, cfg.omega_max),
            rng.uniform(0, cfg.evader_v_max),
            rng.uniform(-cfg.omega_max, cfg.omega_max),
        ]
    )


def check_invariants(env: PursuitEvasionEnv, step: int) -> None:
    cfg = env.cfg
    bound = cfg.arena_size / 2.0 - cfg.agent_radius
    for pose in (env.pursuer_pose, env.evader_pose):
        assert np.all(np.isfinite(pose)), f"NaN pose at step {step}"
        assert abs(pose[0]) <= bound + 1e-9 and abs(pose[1]) <= bound + 1e-9
        for obs in env.obstacles:
            d = np.hypot(pose[0] - obs["x"], pose[1] - obs["y"])
            assert d >= obs["radius"] + cfg.agent_radius - 1e-9


def main() -> None:
    cfg = EnvConfig()
    env = PursuitEvasionEnv(cfg)
    renderer = Renderer(cfg, dpi=80, figsize=(5, 5))
    action_rng = make_rng(derive_seed(GATE_SEED, "step_01_gate_actions"))
    trails = Trails()
    frames: list[np.ndarray] = []

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for ep in range(N_EPISODES):
        title = f"episode {ep + 1}/{N_EPISODES}"
        env.reset(seed=derive_seed(GATE_SEED, f"step_01_gate_ep{ep}"))
        trails.reset()
        outcome, capture_step, end_step = None, None, None
        for step in range(cfg.max_steps):
            joint = random_action(action_rng, cfg)
            _, _, terminated, truncated, _ = env.step(joint)
            check_invariants(env, step)
            trails.append("pursuer", *env.pursuer_pose[:2])
            trails.append("evader", *env.evader_pose[:2])
            if step % STRIDE == 0:
                frames.append(renderer.draw(env, trails, title_prefix=title))
            if terminated or truncated:
                outcome = env.outcome
                end_step = step
                if env.outcome == OUTCOME_CAPTURE:
                    capture_step = step
                # hold on the terminal frame so the episode end is visible
                terminal_frame = renderer.draw(env, trails, title_prefix=title)
                frames.extend([terminal_frame] * HOLD_FRAMES)
                break
        print(
            f"episode {ep}: outcome={outcome} end_step={end_step} "
            f"capture_step={capture_step} "
            f"(capture==end: {capture_step == end_step if capture_step is not None else 'n/a'})"
        )

    out_path = OUT_DIR / "step_01_random_rollouts.gif"
    save_gif(frames, str(out_path), fps=FPS)
    renderer.close()
    print(f"wrote {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
