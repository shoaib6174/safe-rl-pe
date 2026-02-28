"""Visualize evader trajectories against greedy pursuer.

Generates trajectory plots showing how the trained evader moves
against the greedy proportional-navigation pursuer.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        evader_v_max=env_kwargs.get("evader_v_max", 1.05),
        n_obstacle_obs=env_kwargs.get("n_obstacle_obs", 2),
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
    state = base_env.state
    pursuer_traj.append((state["pursuer_x"], state["pursuer_y"]))
    evader_traj.append((state["evader_x"], state["evader_y"]))

    # Record obstacles
    for obs_obj in base_env.obstacles:
        obstacles.append((obs_obj.x, obs_obj.y, obs_obj.radius))

    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        state = base_env.state
        pursuer_traj.append((state["pursuer_x"], state["pursuer_y"]))
        evader_traj.append((state["evader_x"], state["evader_y"]))

    escaped = info.get("timeout", False)
    captured = info.get("captured", False)
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


def plot_trajectories(episodes, title, output_path):
    """Plot trajectory grid."""
    n = len(episodes)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for i, ep in enumerate(episodes):
        ax = axes[i]
        arena_w = ep["arena_w"]
        arena_h = ep["arena_h"]

        # Arena boundary
        ax.set_xlim(-0.5, arena_w + 0.5)
        ax.set_ylim(-0.5, arena_h + 0.5)
        ax.add_patch(patches.Rectangle(
            (0, 0), arena_w, arena_h,
            linewidth=2, edgecolor="black", facecolor="lightyellow",
        ))

        # Obstacles
        for ox, oy, r in ep["obstacles"]:
            ax.add_patch(patches.Circle(
                (ox, oy), r, facecolor="gray", edgecolor="black",
                alpha=0.6, zorder=3,
            ))

        # Trajectories
        p = ep["pursuer"]
        e = ep["evader"]

        ax.plot(p[:, 0], p[:, 1], color="red", linewidth=1.0, alpha=0.7,
                label="Pursuer (greedy)")
        ax.plot(e[:, 0], e[:, 1], color="blue", linewidth=1.0, alpha=0.7,
                label="Evader (learned)")

        # Start markers
        ax.plot(p[0, 0], p[0, 1], "r^", markersize=10, zorder=5)
        ax.plot(e[0, 0], e[0, 1], "bs", markersize=10, zorder=5)

        # End markers
        ax.plot(p[-1, 0], p[-1, 1], "rv", markersize=10, zorder=5)
        ax.plot(e[-1, 0], e[-1, 1], "bD", markersize=10, zorder=5)

        # Capture point
        if ep["captured"]:
            ax.plot(e[-1, 0], e[-1, 1], "x", color="black",
                    markersize=15, markeredgewidth=3, zorder=6)

        outcome = "ESCAPED" if ep["escaped"] else "CAPTURED"
        color = "green" if ep["escaped"] else "red"
        ax.set_title(f"Ep {i+1}: {outcome} ({ep['steps']} steps)",
                     fontsize=11, color=color, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    # Hide empty subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize evader vs greedy pursuer trajectories")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to evader model (evader_final.zip)")
    parser.add_argument("--output", type=str, default="trajectories.png",
                        help="Output image path")
    parser.add_argument("--n_episodes", type=int, default=9,
                        help="Number of episodes to visualize")
    parser.add_argument("--arena_width", type=float, default=10.0)
    parser.add_argument("--arena_height", type=float, default=10.0)
    parser.add_argument("--evader_v_max", type=float, default=1.05)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--n_obstacles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=100)
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
        "visibility_weight": 0.1,
        "survival_bonus": 0.03,
        "timeout_penalty": -50.0,
        "capture_bonus": 50.0,
        "n_obstacle_obs": 2,
    }

    greedy_pursuer = GreedyPursuerPolicy(
        v_max=1.0, omega_max=2.84, K_p=3.0,
        arena_half_w=args.arena_width / 2,
        arena_half_h=args.arena_height / 2,
    )

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    title = (f"Evader (learned) vs Greedy Pursuer\n"
             f"{args.arena_width:.0f}x{args.arena_height:.0f} arena, "
             f"{args.evader_v_max}x speed, "
             f"{args.max_steps} max steps, "
             f"{args.n_obstacles} obstacles")

    episodes = []
    for i in range(args.n_episodes):
        ep = run_episode(model, greedy_pursuer, env_kwargs,
                         seed=args.seed + i)
        episodes.append(ep)
        status = "ESCAPED" if ep["escaped"] else "CAPTURED"
        print(f"  Ep {i+1}: {status} at step {ep['steps']}")

    plot_trajectories(episodes, title, args.output)

    n_escaped = sum(1 for ep in episodes if ep["escaped"])
    n_captured = sum(1 for ep in episodes if ep["captured"])
    avg_steps = np.mean([ep["steps"] for ep in episodes])
    print(f"\nSummary: {n_escaped}/{args.n_episodes} escaped, "
          f"{n_captured} captured, avg {avg_steps:.0f} steps")


if __name__ == "__main__":
    main()
