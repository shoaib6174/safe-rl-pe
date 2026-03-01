"""Analyze SP4 trajectories: static PNG plots + behavior metrics."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from stable_baselines3 import PPO

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.rewards import RewardComputer, line_of_sight_blocked


def make_env(seed=None):
    """Create SP4-config env."""
    arena_w, arena_h = 10.0, 10.0
    diagonal = np.sqrt(arena_w**2 + arena_h**2)
    rc = RewardComputer(
        distance_scale=1.0, d_max=diagonal,
        use_visibility_reward=True, visibility_weight=0.5,
        survival_bonus=0.03, timeout_penalty=-50.0, capture_bonus=50.0,
    )
    env = PursuitEvasionEnv(
        arena_width=arena_w, arena_height=arena_h,
        max_steps=600, capture_radius=0.5,
        n_obstacles=3, pursuer_v_max=1.0, evader_v_max=1.0,
        n_obstacle_obs=3, reward_computer=rc,
        partial_obs=True, n_obstacles_min=0, n_obstacles_max=3,
        asymmetric_obs=False, sensing_radius=3.0, combined_masking=True,
    )
    if seed is not None:
        env.np_random = np.random.default_rng(seed)
    return env


def run_episode(pursuer_model, evader_model, seed=None):
    """Run one episode and record everything."""
    env = make_env(seed)
    obs, _ = env.reset()

    p_traj, e_traj = [], []
    distances = []
    p_visible_to_e = []  # can evader see pursuer?
    e_visible_to_p = []  # can pursuer see evader?
    obstacles = [(o["x"], o["y"], o["radius"]) for o in env.obstacles]

    p_traj.append(env.pursuer_state[:2].copy())
    e_traj.append(env.evader_state[:2].copy())

    done = False
    steps = 0
    while not done:
        p_action, _ = pursuer_model.predict(obs["pursuer"], deterministic=True)
        e_action, _ = evader_model.predict(obs["evader"], deterministic=True)
        obs, _, terminated, truncated, info = env.step(p_action, e_action)
        done = terminated or truncated
        steps += 1

        p_traj.append(env.pursuer_state[:2].copy())
        e_traj.append(env.evader_state[:2].copy())

        dist = np.linalg.norm(env.pursuer_state[:2] - env.evader_state[:2])
        distances.append(dist)

        los_clear = not line_of_sight_blocked(
            env.pursuer_state[:2], env.evader_state[:2], env.obstacles)
        in_range = dist <= 3.0
        visible = in_range and los_clear
        e_visible_to_p.append(visible)
        p_visible_to_e.append(visible)  # symmetric

    env.close()
    return {
        "p_traj": np.array(p_traj), "e_traj": np.array(e_traj),
        "obstacles": obstacles, "distances": distances,
        "e_visible_to_p": e_visible_to_p,
        "steps": steps, "escaped": truncated, "captured": terminated,
    }


def plot_trajectory(ep, ax, ep_num):
    """Plot a single episode trajectory on an axis."""
    p, e = ep["p_traj"], ep["e_traj"]
    half_w, half_h = 5.0, 5.0

    ax.set_xlim(-half_w - 0.3, half_w + 0.3)
    ax.set_ylim(-half_h - 0.3, half_h + 0.3)
    ax.add_patch(patches.Rectangle(
        (-half_w, -half_h), 10.0, 10.0,
        linewidth=1.5, edgecolor="black", facecolor="lightyellow"))

    for ox, oy, r in ep["obstacles"]:
        ax.add_patch(patches.Circle(
            (ox, oy), r, facecolor="gray", edgecolor="black", alpha=0.6, zorder=3))

    # Full trajectories
    ax.plot(p[:, 0], p[:, 1], color="red", linewidth=1.0, alpha=0.6, label="Pursuer")
    ax.plot(e[:, 0], e[:, 1], color="blue", linewidth=1.0, alpha=0.6, label="Evader")

    # Start markers
    ax.plot(p[0, 0], p[0, 1], "^", color="red", markersize=10, zorder=5)
    ax.plot(e[0, 0], e[0, 1], "^", color="blue", markersize=10, zorder=5)

    # End markers
    ax.plot(p[-1, 0], p[-1, 1], "o", color="red", markersize=8, zorder=5)
    ax.plot(e[-1, 0], e[-1, 1], "s", color="blue", markersize=8, zorder=5)

    if ep["captured"]:
        ax.plot(e[-1, 0], e[-1, 1], "x", color="black",
                markersize=12, markeredgewidth=3, zorder=6)

    outcome = "ESCAPED" if ep["escaped"] else "CAPTURED"
    color = "green" if ep["escaped"] else "red"
    ax.set_title(f"Ep {ep_num}: {outcome} ({ep['steps']} steps)", fontsize=9,
                 color=color, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=6)


def analyze_episodes(episodes):
    """Print behavior metrics for episodes."""
    captures = sum(1 for ep in episodes if ep["captured"])
    escapes = sum(1 for ep in episodes if ep["escaped"])
    n = len(episodes)

    print(f"\n{'='*60}")
    print(f"BEHAVIOR ANALYSIS ({n} episodes)")
    print(f"{'='*60}")
    print(f"  Captures: {captures}/{n} ({captures/n:.0%})")
    print(f"  Escapes:  {escapes}/{n} ({escapes/n:.0%})")

    all_dists = []
    all_vis = []
    ep_lengths = []
    for ep in episodes:
        all_dists.extend(ep["distances"])
        all_vis.extend(ep["e_visible_to_p"])
        ep_lengths.append(ep["steps"])

    print(f"\n  Avg episode length: {np.mean(ep_lengths):.0f} steps")
    print(f"  Avg distance: {np.mean(all_dists):.2f}m")
    print(f"  Min distance: {np.min(all_dists):.2f}m")
    print(f"  Max distance: {np.max(all_dists):.2f}m")
    print(f"  % time visible: {np.mean(all_vis):.1%}")

    # Movement analysis
    for label, key in [("Pursuer", "p_traj"), ("Evader", "e_traj")]:
        all_speeds = []
        for ep in episodes:
            traj = ep[key]
            diffs = np.diff(traj, axis=0)
            speeds = np.linalg.norm(diffs, axis=1) / 0.05  # dt=0.05
            all_speeds.extend(speeds)
        print(f"\n  {label}:")
        print(f"    Avg speed: {np.mean(all_speeds):.3f} m/s (max {1.0})")
        print(f"    Speed std: {np.std(all_speeds):.3f}")

    # Displacement analysis (does evader actually move?)
    print(f"\n  Displacement from start (final pos):")
    for ep_i, ep in enumerate(episodes[:6]):
        p_disp = np.linalg.norm(ep["p_traj"][-1] - ep["p_traj"][0])
        e_disp = np.linalg.norm(ep["e_traj"][-1] - ep["e_traj"][0])
        p_total = np.sum(np.linalg.norm(np.diff(ep["p_traj"], axis=0), axis=1))
        e_total = np.sum(np.linalg.norm(np.diff(ep["e_traj"], axis=0), axis=1))
        print(f"    Ep {ep_i+1}: P disp={p_disp:.1f}m (path={p_total:.1f}m) "
              f"| E disp={e_disp:.1f}m (path={e_total:.1f}m) "
              f"| {'ESC' if ep['escaped'] else 'CAP'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pursuer", type=str, required=True)
    parser.add_argument("--evader", type=str, required=True)
    parser.add_argument("--output", type=str, default="sp4_trajectories.png")
    parser.add_argument("--n_episodes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--label", type=str, default="SP4")
    args = parser.parse_args()

    print(f"Loading pursuer: {args.pursuer}")
    p_model = PPO.load(args.pursuer, device="cpu")
    print(f"Loading evader: {args.evader}")
    e_model = PPO.load(args.evader, device="cpu")

    episodes = []
    for i in range(args.n_episodes):
        ep = run_episode(p_model, e_model, seed=args.seed + i)
        status = "ESCAPED" if ep["escaped"] else f"CAPTURED at {ep['steps']}"
        print(f"  Ep {i+1}: {status}")
        episodes.append(ep)

    analyze_episodes(episodes)

    # Plot 3x3 grid
    n = min(args.n_episodes, 9)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n):
        plot_trajectory(episodes[i], axes[i], i + 1)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    captures = sum(1 for ep in episodes if ep["captured"])
    fig.suptitle(f"{args.label}: Captures {captures}/{n}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
