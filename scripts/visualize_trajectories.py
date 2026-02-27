#!/usr/bin/env python3
"""Visualize game trajectories from trained models.

Loads pursuer/evader models and runs episodes, recording trajectories.
Renders them as matplotlib plots (no display needed).

Usage:
    ./venv/bin/python scripts/visualize_trajectories.py \
        --pursuer_model results/stage3/run_h_curriculum/final/pursuer.zip \
        --evader_model results/stage3/run_h_curriculum/final/evader.zip \
        --config results/stage3/run_h_curriculum/config.json \
        --n_episodes 5 \
        --output results/stage3/run_h_trajectories/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import FixedSpeedWrapper, FixedSpeedModelAdapter, SingleAgentPEWrapper
from envs.partial_obs_wrapper import PartialObsWrapper
from envs.opponent_adapter import PartialObsOpponentAdapter


def run_episode(base_env, p_adapter, e_adapter, config, seed=0):
    """Run one episode using PartialObsOpponentAdapters and record trajectories."""
    obs, _ = base_env.reset(seed=seed)
    p_adapter.reset()
    e_adapter.reset()

    fixed_speed = config.get("fixed_speed", False)
    pursuer_v_max = config.get("pursuer_v_max", 1.0)

    pursuer_traj = []
    evader_traj = []
    pursuer_headings = []
    evader_headings = []
    steps = 0
    captured = False

    done = False
    while not done:
        px, py, ptheta = base_env.pursuer_state
        ex, ey, etheta = base_env.evader_state
        pursuer_traj.append((px, py))
        evader_traj.append((ex, ey))
        pursuer_headings.append(ptheta)
        evader_headings.append(etheta)

        # Both agents predict through their partial obs adapters
        p_action, _ = p_adapter.predict(obs["pursuer"], deterministic=True)
        e_action, _ = e_adapter.predict(obs["evader"], deterministic=True)

        # Expand fixed-speed 1D [omega] -> 2D [v_max, omega]
        if fixed_speed:
            if p_action.shape[-1] == 1:
                p_action = np.array([pursuer_v_max, p_action[0]], dtype=np.float32)
            if e_action.shape[-1] == 1:
                e_action = np.array([base_env.evader_v_max, e_action[0]], dtype=np.float32)

        obs, rewards, terminated, truncated, info = base_env.step(p_action, e_action)
        steps += 1
        done = terminated or truncated

    # Record final position
    px, py, ptheta = base_env.pursuer_state
    ex, ey, etheta = base_env.evader_state
    pursuer_traj.append((px, py))
    evader_traj.append((ex, ey))
    pursuer_headings.append(ptheta)
    evader_headings.append(etheta)

    if "episode_metrics" in info:
        captured = info["episode_metrics"]["captured"]

    # Record obstacle positions
    obstacles = []
    if hasattr(base_env, "obstacles"):
        for obs in base_env.obstacles:
            obstacles.append({"x": obs["x"], "y": obs["y"], "r": obs["radius"]})

    return {
        "pursuer": np.array(pursuer_traj),
        "evader": np.array(evader_traj),
        "pursuer_headings": np.array(pursuer_headings),
        "evader_headings": np.array(evader_headings),
        "steps": steps,
        "captured": captured,
        "obstacles": obstacles,
    }


def plot_trajectory(traj_data, config, ep_idx, output_path):
    """Plot a single episode trajectory."""
    arena_w = config["arena_width"]
    arena_h = config["arena_height"]
    capture_radius = config["capture_radius"]

    p_traj = traj_data["pursuer"]
    e_traj = traj_data["evader"]
    steps = traj_data["steps"]
    captured = traj_data["captured"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Arena boundary
    ax.add_patch(plt.Rectangle(
        (-arena_w / 2, -arena_h / 2), arena_w, arena_h,
        fill=False, edgecolor="black", linewidth=2
    ))

    # Obstacles
    for obs in traj_data.get("obstacles", []):
        ax.add_patch(Circle(
            (obs["x"], obs["y"]), obs["r"],
            fill=True, facecolor="#8B4513", edgecolor="black",
            linewidth=1.5, alpha=0.7, zorder=3
        ))

    # Trajectory lines with fading
    n_p = len(p_traj)
    n_e = len(e_traj)

    # Draw trajectories as colored line segments with alpha gradient
    for i in range(n_p - 1):
        alpha = 0.2 + 0.8 * (i / max(n_p - 1, 1))
        ax.plot(p_traj[i:i+2, 0], p_traj[i:i+2, 1],
                color="#d62728", alpha=alpha, linewidth=1.5)
    for i in range(n_e - 1):
        alpha = 0.2 + 0.8 * (i / max(n_e - 1, 1))
        ax.plot(e_traj[i:i+2, 0], e_traj[i:i+2, 1],
                color="#1f77b4", alpha=alpha, linewidth=1.5)

    # Start positions (large circles)
    ax.plot(p_traj[0, 0], p_traj[0, 1], "o", color="#d62728", markersize=12,
            markeredgecolor="black", markeredgewidth=1.5, zorder=10)
    ax.plot(e_traj[0, 0], e_traj[0, 1], "o", color="#1f77b4", markersize=12,
            markeredgecolor="black", markeredgewidth=1.5, zorder=10)

    # Start labels
    ax.annotate("P start", (p_traj[0, 0], p_traj[0, 1]),
                textcoords="offset points", xytext=(10, 10),
                fontsize=8, color="#d62728", fontweight="bold")
    ax.annotate("E start", (e_traj[0, 0], e_traj[0, 1]),
                textcoords="offset points", xytext=(10, 10),
                fontsize=8, color="#1f77b4", fontweight="bold")

    # End positions (triangles showing heading)
    p_head = traj_data["pursuer_headings"][-1]
    e_head = traj_data["evader_headings"][-1]

    # Pursuer end marker with heading arrow
    ax.annotate("", xy=(p_traj[-1, 0] + 0.3 * np.cos(p_head),
                        p_traj[-1, 1] + 0.3 * np.sin(p_head)),
                xytext=(p_traj[-1, 0], p_traj[-1, 1]),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=2), zorder=11)
    ax.plot(p_traj[-1, 0], p_traj[-1, 1], "^", color="#d62728", markersize=10,
            markeredgecolor="black", markeredgewidth=1.5, zorder=11)

    # Evader end marker with heading arrow
    ax.annotate("", xy=(e_traj[-1, 0] + 0.3 * np.cos(e_head),
                        e_traj[-1, 1] + 0.3 * np.sin(e_head)),
                xytext=(e_traj[-1, 0], e_traj[-1, 1]),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2), zorder=11)
    ax.plot(e_traj[-1, 0], e_traj[-1, 1], "^", color="#1f77b4", markersize=10,
            markeredgecolor="black", markeredgewidth=1.5, zorder=11)

    # Capture radius at capture point
    if captured:
        capture_circle = Circle(
            (p_traj[-1, 0], p_traj[-1, 1]), capture_radius,
            fill=True, facecolor="#ff000020", edgecolor="red",
            linewidth=1.5, linestyle="--", zorder=5
        )
        ax.add_patch(capture_circle)
        ax.annotate("CAPTURED", (p_traj[-1, 0], p_traj[-1, 1]),
                    textcoords="offset points", xytext=(15, -15),
                    fontsize=10, color="red", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

    # Time markers every 50 steps
    marker_interval = max(steps // 6, 1)
    for i in range(0, n_p, marker_interval):
        if i == 0 or i == n_p - 1:
            continue
        t_sec = i * 0.05  # dt=0.05
        ax.plot(p_traj[i, 0], p_traj[i, 1], ".", color="#d62728", markersize=6, zorder=8)
        ax.annotate(f"{t_sec:.0f}s", (p_traj[i, 0], p_traj[i, 1]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6, color="#d62728", alpha=0.7)
    for i in range(0, n_e, marker_interval):
        if i == 0 or i == n_e - 1:
            continue
        t_sec = i * 0.05
        ax.plot(e_traj[i, 0], e_traj[i, 1], ".", color="#1f77b4", markersize=6, zorder=8)
        ax.annotate(f"{t_sec:.0f}s", (e_traj[i, 0], e_traj[i, 1]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6, color="#1f77b4", alpha=0.7)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color="#d62728", linewidth=2, label="Pursuer"),
        plt.Line2D([0], [0], color="#1f77b4", linewidth=2, label="Evader"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=8, label="Start"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                   markersize=8, label="End"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    # Labels
    outcome = "Captured" if captured else "Timeout"
    t_total = steps * 0.05
    ax.set_title(f"Episode {ep_idx + 1} — {outcome} after {steps} steps ({t_total:.1f}s)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-arena_w / 2 - 0.5, arena_w / 2 + 0.5)
    ax.set_ylim(-arena_h / 2 - 0.5, arena_h / 2 + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_overview(all_trajs, config, output_path):
    """Plot all trajectories on one overview figure."""
    n = len(all_trajs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    arena_w = config["arena_width"]
    arena_h = config["arena_height"]

    n_captured = sum(1 for t in all_trajs if t["captured"])

    for idx, (traj, ax) in enumerate(zip(all_trajs, axes)):
        p = traj["pursuer"]
        e = traj["evader"]

        ax.add_patch(plt.Rectangle(
            (-arena_w / 2, -arena_h / 2), arena_w, arena_h,
            fill=False, edgecolor="black", linewidth=1.5
        ))

        for obs in traj.get("obstacles", []):
            ax.add_patch(Circle(
                (obs["x"], obs["y"]), obs["r"],
                fill=True, facecolor="#8B4513", edgecolor="black",
                linewidth=1, alpha=0.6, zorder=3
            ))
        ax.plot(p[:, 0], p[:, 1], "-", color="#d62728", alpha=0.7, linewidth=1.2)
        ax.plot(e[:, 0], e[:, 1], "-", color="#1f77b4", alpha=0.7, linewidth=1.2)
        ax.plot(p[0, 0], p[0, 1], "o", color="#d62728", markersize=8, markeredgecolor="black")
        ax.plot(e[0, 0], e[0, 1], "o", color="#1f77b4", markersize=8, markeredgecolor="black")
        ax.plot(p[-1, 0], p[-1, 1], "^", color="#d62728", markersize=8, markeredgecolor="black")
        ax.plot(e[-1, 0], e[-1, 1], "^", color="#1f77b4", markersize=8, markeredgecolor="black")

        if traj["captured"]:
            ax.add_patch(Circle((p[-1, 0], p[-1, 1]), config["capture_radius"],
                               fill=False, edgecolor="red", linestyle="--", linewidth=1))

        outcome = "Captured" if traj["captured"] else "Timeout"
        t_sec = traj["steps"] * 0.05
        ax.set_title(f"Ep {idx + 1}: {outcome} ({t_sec:.1f}s)", fontsize=10)
        ax.set_xlim(-arena_w / 2 - 0.5, arena_w / 2 + 0.5)
        ax.set_ylim(-arena_h / 2 - 0.5, arena_h / 2 + 0.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Run H Final Models — {n_captured}/{n} captured "
        f"({100 * n_captured / n:.0f}%)",
        fontsize=14, fontweight="bold"
    )

    legend_elements = [
        plt.Line2D([0], [0], color="#d62728", linewidth=2, label="Pursuer"),
        plt.Line2D([0], [0], color="#1f77b4", linewidth=2, label="Evader"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=11)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_episode_full_obs(base_env, pursuer_model, evader_model, config, seed=0):
    """Run one episode with full-obs models (flat obs vector, not dict)."""
    obs, _ = base_env.reset(seed=seed)
    fixed_speed = config.get("fixed_speed", False)
    pursuer_v_max = config.get("pursuer_v_max", 1.0)
    evader_v_max = config.get("evader_v_max", 1.0)

    pursuer_traj = []
    evader_traj = []
    pursuer_headings = []
    evader_headings = []
    steps = 0
    captured = False

    done = False
    while not done:
        px, py, ptheta = base_env.pursuer_state
        ex, ey, etheta = base_env.evader_state
        pursuer_traj.append((px, py))
        evader_traj.append((ex, ey))
        pursuer_headings.append(ptheta)
        evader_headings.append(etheta)

        # Full-obs models expect flat obs vectors (from SingleAgentPEWrapper)
        p_obs = obs["pursuer"]
        e_obs = obs["evader"]
        p_action, _ = pursuer_model.predict(p_obs, deterministic=True)
        e_action, _ = evader_model.predict(e_obs, deterministic=True)

        # Expand fixed-speed 1D [omega] -> 2D [v_max, omega]
        if fixed_speed:
            if p_action.shape[-1] == 1:
                p_action = np.array([pursuer_v_max, p_action[0]], dtype=np.float32)
            if e_action.shape[-1] == 1:
                e_action = np.array([evader_v_max, e_action[0]], dtype=np.float32)

        obs, rewards, terminated, truncated, info = base_env.step(p_action, e_action)
        steps += 1
        done = terminated or truncated

    # Record final position
    px, py, ptheta = base_env.pursuer_state
    ex, ey, etheta = base_env.evader_state
    pursuer_traj.append((px, py))
    evader_traj.append((ex, ey))
    pursuer_headings.append(ptheta)
    evader_headings.append(etheta)

    if "episode_metrics" in info:
        captured = info["episode_metrics"]["captured"]

    obstacles = []
    if hasattr(base_env, "obstacles"):
        for ob in base_env.obstacles:
            obstacles.append({"x": ob["x"], "y": ob["y"], "r": ob["radius"]})

    return {
        "pursuer": np.array(pursuer_traj),
        "evader": np.array(evader_traj),
        "pursuer_headings": pursuer_headings,
        "evader_headings": evader_headings,
        "captured": captured,
        "steps": steps,
        "obstacles": obstacles,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize game trajectories")
    parser.add_argument("--pursuer_model", type=str, required=True)
    parser.add_argument("--evader_model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=6)
    parser.add_argument("--output", type=str, default="results/stage3/trajectories/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n_obstacles", type=int, default=None,
                        help="Override n_obstacles from config (e.g. 3 for Level 3-4)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for model inference (cpu or cuda)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply overrides
    if args.n_obstacles is not None:
        config["n_obstacles"] = args.n_obstacles
    device = args.device

    full_obs = config.get("full_obs", False)

    # Load models
    print(f"Loading pursuer model: {args.pursuer_model}")
    pursuer = PPO.load(args.pursuer_model, device=device)
    print(f"Loading evader model: {args.evader_model}")
    evader = PPO.load(args.evader_model, device=device)

    # Create base env
    env_kwargs = {
        "arena_width": config["arena_width"],
        "arena_height": config["arena_height"],
        "max_steps": config["max_steps"],
        "capture_radius": config["capture_radius"],
        "distance_scale": config.get("distance_scale", 1.0),
        "pursuer_v_max": config.get("pursuer_v_max", 1.0),
        "evader_v_max": config.get("evader_v_max", 1.0),
        "n_obstacles": config.get("n_obstacles", 0),
        "n_obstacle_obs": config.get("n_obstacle_obs", 0),
    }
    base_env = PursuitEvasionEnv(**env_kwargs)
    print(f"Env: {config['arena_width']}x{config['arena_height']}m, "
          f"{config.get('n_obstacles', 0)} obstacles, "
          f"full_obs={full_obs}, device={device}")

    all_trajs = []
    for ep in range(args.n_episodes):
        print(f"Running episode {ep + 1}/{args.n_episodes}...")

        if full_obs:
            traj = run_episode_full_obs(
                base_env, pursuer, evader, config, seed=args.seed + ep)
        else:
            # Create partial-obs adapters for both agents
            p_adapter = PartialObsOpponentAdapter(
                model=pursuer, role="pursuer", base_env=base_env,
                deterministic=True,
            )
            e_adapter = PartialObsOpponentAdapter(
                model=evader, role="evader", base_env=base_env,
                deterministic=True,
            )
            traj = run_episode(
                base_env, p_adapter, e_adapter, config, seed=args.seed + ep)

        all_trajs.append(traj)

        outcome = "Captured" if traj["captured"] else "Timeout"
        print(f"  {outcome} after {traj['steps']} steps ({traj['steps'] * 0.05:.1f}s)")

        # Individual plot
        plot_trajectory(traj, config, ep, out_dir / f"trajectory_ep{ep:02d}.png")

    # Overview plot
    plot_overview(all_trajs, config, out_dir / "trajectory_overview.png")

    base_env.close()
    n_cap = sum(1 for t in all_trajs if t["captured"])
    print(f"\nDone: {n_cap}/{args.n_episodes} captured ({100 * n_cap / args.n_episodes:.0f}%)")
    print(f"Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
