"""Generate animated GIF with 3x3 grid of evader vs greedy pursuer episodes."""
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
        timeout_penalty=env_kwargs.get("timeout_penalty", 0.0),
        capture_bonus=env_kwargs.get("capture_bonus", 5.0),
    )

    base_env = PursuitEvasionEnv(
        arena_width=arena_w,
        arena_height=arena_h,
        max_steps=env_kwargs.get("max_steps", 1200),
        capture_radius=env_kwargs.get("capture_radius", 0.5),
        n_obstacles=env_kwargs.get("n_obstacles", 2),
        pursuer_v_max=env_kwargs.get("pursuer_v_max", 1.0),
        evader_v_max=env_kwargs.get("evader_v_max", 1.0),
        n_obstacle_obs=env_kwargs.get("n_obstacle_obs", 2),
        reward_computer=reward_computer,
        partial_obs=env_kwargs.get("partial_obs", False),
    )
    if seed is not None:
        base_env.np_random = np.random.default_rng(seed)

    single_env = SingleAgentPEWrapper(
        base_env, role="evader", opponent_policy=greedy_pursuer,
    )
    # Auto-detect: if model has 1D action space, use FixedSpeedWrapper
    action_dim = model.action_space.shape[0]
    if action_dim == 1:
        env = FixedSpeedWrapper(single_env, v_max=base_env.evader_v_max)
    else:
        env = single_env

    obs, _ = env.reset()
    done = False

    pursuer_traj = []
    evader_traj = []
    obstacles = []

    pursuer_traj.append((base_env.pursuer_state[0], base_env.pursuer_state[1]))
    evader_traj.append((base_env.evader_state[0], base_env.evader_state[1]))

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

    env.close()

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


def make_grid_gif(episodes, output_path, fps=30, skip=3):
    """Create animated GIF with 3x3 grid of episodes."""
    n = len(episodes)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    # Find max frames across all episodes
    max_frames = max(len(ep["pursuer"]) for ep in episodes)
    frame_indices = list(range(0, max_frames, skip))
    if frame_indices[-1] != max_frames - 1:
        frame_indices.append(max_frames - 1)

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Evader (learned, 1.0x) vs Greedy Pursuer  |  "
        f"{episodes[0]['arena_w']:.0f}x{episodes[0]['arena_h']:.0f} arena, "
        f"{len(episodes[0]['obstacles'])} obstacles",
        fontsize=12, fontweight="bold", y=0.98,
    )

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

            # Clamp frame index to this episode's length
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
            ax.plot(p[i, 0], p[i, 1], "o", color="red", markersize=8, zorder=5)
            ax.plot(e[i, 0], e[i, 1], "s", color="blue", markersize=8, zorder=5)

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
        description="Generate 3x3 grid animated trajectory GIF")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="grid_trajectory.gif")
    parser.add_argument("--n_episodes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--arena_width", type=float, default=10.0)
    parser.add_argument("--arena_height", type=float, default=10.0)
    parser.add_argument("--evader_v_max", type=float, default=1.0)
    parser.add_argument("--n_obstacles", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--visibility_weight", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip", type=int, default=4,
                        help="Only render every Nth step (default: 4)")
    parser.add_argument("--partial_obs", action="store_true",
                        help="Enable LOS-based partial observability")
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
        "survival_bonus": 0.1,
        "timeout_penalty": 0.0,
        "capture_bonus": 5.0,
        "n_obstacle_obs": args.n_obstacles,
        "partial_obs": args.partial_obs,
    }

    greedy_pursuer = GreedyPursuerPolicy(
        v_max=1.0, omega_max=2.84, K_p=3.0,
        arena_half_w=args.arena_width / 2,
        arena_half_h=args.arena_height / 2,
    )

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    episodes = []
    for i in range(args.n_episodes):
        ep = run_episode(model, greedy_pursuer, env_kwargs,
                         seed=args.seed + i)
        status = "ESCAPED" if ep["escaped"] else f"CAPTURED at {ep['steps']}"
        print(f"  Ep {i+1}: {status}")
        episodes.append(ep)

    make_grid_gif(episodes, args.output, fps=args.fps, skip=args.skip)


if __name__ == "__main__":
    main()
