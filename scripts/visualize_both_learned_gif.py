"""Generate animated GIF with 3x3 grid: learned pursuer vs learned evader."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from stable_baselines3 import PPO

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.rewards import RewardComputer
from envs.wrappers import FixedSpeedWrapper


def run_episode(pursuer_model, evader_model, env_kwargs, seed=None):
    """Run one episode with both learned agents and record trajectories."""
    arena_w = env_kwargs["arena_width"]
    arena_h = env_kwargs["arena_height"]
    diagonal = np.sqrt(arena_w**2 + arena_h**2)

    reward_computer = RewardComputer(
        distance_scale=env_kwargs.get("distance_scale", 1.0),
        d_max=diagonal,
        use_visibility_reward=env_kwargs.get("use_visibility_reward", False),
        visibility_weight=env_kwargs.get("visibility_weight", 0.1),
        survival_bonus=env_kwargs.get("survival_bonus", 0.1),
        timeout_penalty=env_kwargs.get("timeout_penalty", 0.0),
        capture_bonus=env_kwargs.get("capture_bonus", 5.0),
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

    obs, _ = base_env.reset()

    pursuer_traj = []
    evader_traj = []
    obstacles = []

    pursuer_traj.append((base_env.pursuer_state[0], base_env.pursuer_state[1]))
    evader_traj.append((base_env.evader_state[0], base_env.evader_state[1]))

    for obs_obj in base_env.obstacles:
        obstacles.append((obs_obj["x"], obs_obj["y"], obs_obj["radius"]))

    p_action_dim = pursuer_model.action_space.shape[0]
    e_action_dim = evader_model.action_space.shape[0]

    steps = 0
    done = False
    while not done:
        p_raw, _ = pursuer_model.predict(obs["pursuer"], deterministic=True)
        e_raw, _ = evader_model.predict(obs["evader"], deterministic=True)

        # Auto-detect: 1D [omega] -> expand to [v_max, omega]; 2D -> use directly
        if p_action_dim == 1:
            p_action = np.array(
                [base_env.pursuer_v_max, p_raw[0]], dtype=np.float32)
        else:
            p_action = p_raw
        if e_action_dim == 1:
            e_action = np.array(
                [base_env.evader_v_max, e_raw[0]], dtype=np.float32)
        else:
            e_action = e_raw

        obs, rewards, terminated, truncated, info = base_env.step(
            p_action, e_action)
        done = terminated or truncated
        steps += 1

        pursuer_traj.append(
            (base_env.pursuer_state[0], base_env.pursuer_state[1]))
        evader_traj.append(
            (base_env.evader_state[0], base_env.evader_state[1]))

    base_env.close()

    return {
        "pursuer": np.array(pursuer_traj),
        "evader": np.array(evader_traj),
        "obstacles": obstacles,
        "steps": steps,
        "escaped": truncated,
        "captured": terminated,
        "arena_w": arena_w,
        "arena_h": arena_h,
    }


def make_grid_gif(episodes, output_path, title, fps=30, skip=3):
    """Create animated GIF with 3x3 grid of episodes."""
    n = len(episodes)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    max_frames = max(len(ep["pursuer"]) for ep in episodes)
    frame_indices = list(range(0, max_frames, skip))
    if frame_indices[-1] != max_frames - 1:
        frame_indices.append(max_frames - 1)

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    def draw_frame(frame_idx):
        fi = frame_indices[frame_idx]
        for ep_i, ep in enumerate(episodes):
            if ep_i >= len(axes):
                break
            ax = axes[ep_i]
            ax.clear()

            arena_w = ep["arena_w"]
            arena_h = ep["arena_h"]
            half_w = arena_w / 2
            half_h = arena_h / 2
            p = ep["pursuer"]
            e = ep["evader"]
            n_pts = len(p)

            i = min(fi, n_pts - 1)
            ep_done = (i == n_pts - 1) and (fi >= n_pts - 1)

            # Arena
            ax.set_xlim(-half_w - 0.5, half_w + 0.5)
            ax.set_ylim(-half_h - 0.5, half_h + 0.5)
            ax.add_patch(patches.Rectangle(
                (-half_w, -half_h), arena_w, arena_h,
                linewidth=1.5, edgecolor="black", facecolor="lightyellow",
            ))

            # Obstacles
            for ox, oy, r in ep["obstacles"]:
                ax.add_patch(patches.Circle(
                    (ox, oy), r, facecolor="gray", edgecolor="black",
                    alpha=0.6, zorder=3,
                ))

            # Trail
            trail_start = max(0, i - 200)
            if i > 0:
                ax.plot(p[trail_start:i+1, 0], p[trail_start:i+1, 1],
                        color="red", linewidth=0.8, alpha=0.4)
                ax.plot(e[trail_start:i+1, 0], e[trail_start:i+1, 1],
                        color="blue", linewidth=0.8, alpha=0.4)

            # Current positions
            ax.plot(p[i, 0], p[i, 1], "o", color="red", markersize=8,
                    zorder=5)
            ax.plot(e[i, 0], e[i, 1], "s", color="blue", markersize=8,
                    zorder=5)

            # Distance line
            ax.plot([p[i, 0], e[i, 0]], [p[i, 1], e[i, 1]],
                    "--", color="gray", linewidth=0.5, alpha=0.4)

            # Capture X
            if ep_done and ep["captured"]:
                ax.plot(e[-1, 0], e[-1, 1], "x", color="black",
                        markersize=12, markeredgewidth=3, zorder=6)

            # Title
            if ep_done:
                outcome = "ESCAPED" if ep["escaped"] else "CAPTURED"
                color = "green" if ep["escaped"] else "red"
            else:
                outcome = "..."
                color = "black"
            step_num = min(i, ep["steps"])
            ax.set_title(
                f"Ep {ep_i+1}: {outcome} ({step_num}/{ep['steps']})",
                fontsize=9, color=color, fontweight="bold",
            )
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.15)
            ax.tick_params(labelsize=7)

    print(f"Generating {len(frame_indices)} frames for {n} episodes...")
    anim = animation.FuncAnimation(
        fig, draw_frame, frames=len(frame_indices),
        interval=1000 // fps, repeat=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3x3 grid GIF: learned pursuer vs learned evader")
    parser.add_argument("--pursuer_model", type=str, required=True,
                        help="Path to trained pursuer model")
    parser.add_argument("--evader_model", type=str, required=True,
                        help="Path to trained evader model")
    parser.add_argument("--output", type=str, default="grid_both_learned.gif")
    parser.add_argument("--n_episodes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--arena_width", type=float, default=10.0)
    parser.add_argument("--arena_height", type=float, default=10.0)
    parser.add_argument("--pursuer_v_max", type=float, default=1.0)
    parser.add_argument("--evader_v_max", type=float, default=1.0)
    parser.add_argument("--n_obstacles", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--visibility_weight", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip", type=int, default=4,
                        help="Only render every Nth step (default: 4)")
    args = parser.parse_args()

    env_kwargs = {
        "arena_width": args.arena_width,
        "arena_height": args.arena_height,
        "max_steps": args.max_steps,
        "capture_radius": 0.5,
        "n_obstacles": args.n_obstacles,
        "pursuer_v_max": args.pursuer_v_max,
        "evader_v_max": args.evader_v_max,
        "distance_scale": 1.0,
        "use_visibility_reward": True,
        "visibility_weight": args.visibility_weight,
        "survival_bonus": 0.1,
        "timeout_penalty": 0.0,
        "capture_bonus": 5.0,
        "n_obstacle_obs": args.n_obstacles,
    }

    print(f"Loading pursuer model: {args.pursuer_model}")
    pursuer_model = PPO.load(args.pursuer_model, device="cpu")

    print(f"Loading evader model: {args.evader_model}")
    evader_model = PPO.load(args.evader_model, device="cpu")

    episodes = []
    captures = 0
    for i in range(args.n_episodes):
        ep = run_episode(pursuer_model, evader_model, env_kwargs,
                         seed=args.seed + i)
        status = "ESCAPED" if ep["escaped"] else f"CAPTURED at {ep['steps']}"
        print(f"  Ep {i+1}: {status}")
        if ep["captured"]:
            captures += 1
        episodes.append(ep)

    print(f"\nCapture rate: {captures}/{args.n_episodes} "
          f"({captures/args.n_episodes:.0%})")

    title = (
        f"Learned Pursuer vs Learned Evader (both 1.0x)  |  "
        f"{args.arena_width:.0f}x{args.arena_height:.0f} arena, "
        f"{args.n_obstacles} obstacles  |  "
        f"Captured: {captures}/{args.n_episodes}"
    )

    make_grid_gif(episodes, args.output, title=title,
                  fps=args.fps, skip=args.skip)


if __name__ == "__main__":
    main()
