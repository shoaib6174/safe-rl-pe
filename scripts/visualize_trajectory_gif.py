"""Generate animated GIF of evader vs greedy pursuer trajectory."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from stable_baselines3 import PPO

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.rewards import RewardComputer
from envs.wrappers import SingleAgentPEWrapper, FixedSpeedWrapper
from training.baselines import GreedyPursuerPolicy


def run_episode(model, greedy_pursuer, env_kwargs, seed=None):
    """Run one episode and record trajectories."""
    arena_w = env_kwargs["arena_width"]
    arena_h = env_kwargs["arena_height"]
    diagonal = np.sqrt(arena_w**2 + arena_h**2)

    reward_computer = RewardComputer(
        distance_scale=env_kwargs.get("distance_scale", 1.0),
        d_max=diagonal,
        use_visibility_reward=env_kwargs.get("use_visibility_reward", False),
        visibility_weight=env_kwargs.get("visibility_weight", 0.1),
        survival_bonus=env_kwargs.get("survival_bonus", 0.03),
        timeout_penalty=env_kwargs.get("timeout_penalty", -50.0),
        capture_bonus=env_kwargs.get("capture_bonus", 50.0),
    )

    base_env = PursuitEvasionEnv(
        arena_width=arena_w,
        arena_height=arena_h,
        max_steps=env_kwargs.get("max_steps", 600),
        capture_radius=env_kwargs.get("capture_radius", 0.5),
        n_obstacles=env_kwargs.get("n_obstacles", 2),
        pursuer_v_max=env_kwargs.get("pursuer_v_max", 1.0),
        evader_v_max=env_kwargs.get("evader_v_max", 1.0),
        n_obstacle_obs=env_kwargs.get("n_obstacle_obs", 2),
        reward_computer=reward_computer,
    )
    if seed is not None:
        base_env.np_random = np.random.default_rng(seed)

    single_env = SingleAgentPEWrapper(
        base_env, role="evader", opponent_policy=greedy_pursuer,
    )
    env = FixedSpeedWrapper(single_env, v_max=base_env.evader_v_max)

    obs, _ = env.reset()
    done = False

    pursuer_traj = []
    evader_traj = []
    obstacles = []

    # Record initial positions
    pursuer_traj.append((base_env.pursuer_state[0], base_env.pursuer_state[1]))
    evader_traj.append((base_env.evader_state[0], base_env.evader_state[1]))

    # Record obstacles
    for obs_obj in base_env.obstacles:
        obstacles.append((obs_obj["x"], obs_obj["y"], obs_obj["radius"]))

    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        pursuer_traj.append((base_env.pursuer_state[0], base_env.pursuer_state[1]))
        evader_traj.append((base_env.evader_state[0], base_env.evader_state[1]))

    escaped = truncated
    captured = terminated
    env.close()

    return {
        "pursuer": np.array(pursuer_traj),
        "evader": np.array(evader_traj),
        "obstacles": obstacles,
        "steps": steps,
        "escaped": escaped,
        "captured": captured,
        "arena_w": arena_w,
        "arena_h": arena_h,
    }


def make_gif(ep, output_path, fps=30, skip=2):
    """Create animated GIF from episode data."""
    arena_w = ep["arena_w"]
    arena_h = ep["arena_h"]
    half_w = arena_w / 2
    half_h = arena_h / 2
    p = ep["pursuer"]
    e = ep["evader"]
    n_frames = len(p)

    # Subsample frames for smaller GIF
    frame_indices = list(range(0, n_frames, skip))
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)

    fig, ax = plt.subplots(figsize=(6, 6))

    def draw_frame(idx):
        ax.clear()
        i = frame_indices[idx]

        # Arena
        ax.set_xlim(-half_w - 0.5, half_w + 0.5)
        ax.set_ylim(-half_h - 0.5, half_h + 0.5)
        ax.add_patch(patches.Rectangle(
            (-half_w, -half_h), arena_w, arena_h,
            linewidth=2, edgecolor="black", facecolor="lightyellow",
        ))

        # Obstacles
        for ox, oy, r in ep["obstacles"]:
            ax.add_patch(patches.Circle(
                (ox, oy), r, facecolor="gray", edgecolor="black",
                alpha=0.6, zorder=3,
            ))

        # Trail up to current frame
        trail_start = max(0, i - 150)
        if i > 0:
            ax.plot(p[trail_start:i+1, 0], p[trail_start:i+1, 1],
                    color="red", linewidth=1.0, alpha=0.4)
            ax.plot(e[trail_start:i+1, 0], e[trail_start:i+1, 1],
                    color="blue", linewidth=1.0, alpha=0.4)

        # Current positions
        ax.plot(p[i, 0], p[i, 1], "o", color="red", markersize=10,
                zorder=5, label="Pursuer (greedy)")
        ax.plot(e[i, 0], e[i, 1], "s", color="blue", markersize=10,
                zorder=5, label="Evader (learned)")

        # Start markers (faded)
        ax.plot(p[0, 0], p[0, 1], "^", color="red", markersize=7,
                alpha=0.3, zorder=4)
        ax.plot(e[0, 0], e[0, 1], "s", color="blue", markersize=7,
                alpha=0.3, zorder=4)

        # Capture X on final frame
        if i == n_frames - 1 and ep["captured"]:
            ax.plot(e[i, 0], e[i, 1], "x", color="black",
                    markersize=15, markeredgewidth=3, zorder=6)

        # Distance line
        ax.plot([p[i, 0], e[i, 0]], [p[i, 1], e[i, 1]],
                "--", color="gray", linewidth=0.5, alpha=0.5)

        outcome = "ESCAPED" if ep["escaped"] else "CAPTURED"
        color = "green" if ep["escaped"] else "red"
        step_num = min(i, ep["steps"])
        ax.set_title(
            f"Evader (1.0x speed) vs Greedy Pursuer  |  "
            f"Step {step_num}/{ep['steps']}  |  {outcome}",
            fontsize=10, color=color if i == n_frames - 1 else "black",
            fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")

    print(f"Generating {len(frame_indices)} frames...")
    anim = animation.FuncAnimation(
        fig, draw_frame, frames=len(frame_indices),
        interval=1000 // fps, repeat=False,
    )

    anim.save(output_path, writer="pillow", fps=fps)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate animated trajectory GIF")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="trajectory.gif")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--arena_width", type=float, default=10.0)
    parser.add_argument("--arena_height", type=float, default=10.0)
    parser.add_argument("--evader_v_max", type=float, default=1.0)
    parser.add_argument("--n_obstacles", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--visibility_weight", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip", type=int, default=2,
                        help="Only render every Nth step (default: 2)")
    args = parser.parse_args()

    env_kwargs = {
        "arena_width": args.arena_width,
        "arena_height": args.arena_height,
        "max_steps": args.max_steps,
        "capture_radius": 0.5,
        "n_obstacles": args.n_obstacles,
        "pursuer_v_max": 1.0,
        "evader_v_max": args.evader_v_max,
        "distance_scale": 1.0,
        "use_visibility_reward": True,
        "visibility_weight": args.visibility_weight,
        "survival_bonus": 0.03,
        "timeout_penalty": -50.0,
        "capture_bonus": 50.0,
        "n_obstacle_obs": args.n_obstacles,
    }

    greedy_pursuer = GreedyPursuerPolicy(
        v_max=1.0, omega_max=2.84, K_p=3.0,
        arena_half_w=args.arena_width / 2,
        arena_half_h=args.arena_height / 2,
    )

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    # Try seeds until we find an escape episode for best visualization
    print("Searching for an escape episode...")
    best_ep = None
    for s in range(args.seed, args.seed + 20):
        ep = run_episode(model, greedy_pursuer, env_kwargs, seed=s)
        status = "ESCAPED" if ep["escaped"] else f"CAPTURED at {ep['steps']}"
        print(f"  seed={s}: {status}")
        if ep["escaped"]:
            best_ep = ep
            print(f"  -> Using seed {s} (escaped!)")
            break
        if best_ep is None or ep["steps"] > best_ep["steps"]:
            best_ep = ep

    if not best_ep["escaped"]:
        print(f"  No escape found; using longest episode ({best_ep['steps']} steps)")

    make_gif(best_ep, args.output, fps=args.fps, skip=args.skip)


if __name__ == "__main__":
    main()
