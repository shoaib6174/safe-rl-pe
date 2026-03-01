"""Analyze S1v6b trajectories: evader (1.2x speed) vs greedy pursuer (full obs).
Static PNG plots + behavior metrics."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from stable_baselines3 import PPO

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.observations import ObservationBuilder
from envs.rewards import RewardComputer, line_of_sight_blocked
from training.baselines import GreedyPursuerPolicy


def make_env(evader_speed=1.2, seed=None):
    """Create S1v6b-config env."""
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
        n_obstacles=3, pursuer_v_max=1.0, evader_v_max=evader_speed,
        n_obstacle_obs=3, reward_computer=rc,
        partial_obs=True, n_obstacles_min=0, n_obstacles_max=3,
        asymmetric_obs=False, sensing_radius=3.0, combined_masking=True,
    )
    if seed is not None:
        env.np_random = np.random.default_rng(seed)
    return env


def run_episode(greedy_pursuer, evader_model, evader_speed=1.2, seed=None):
    """Run one episode: greedy pursuer (full obs) vs trained evader (partial obs)."""
    env = make_env(evader_speed=evader_speed, seed=seed)
    obs, _ = env.reset()

    p_traj, e_traj = [], []
    distances = []
    e_visible_to_p = []
    p_speeds, e_speeds = [], []
    obstacles = [(o["x"], o["y"], o["radius"]) for o in env.obstacles]

    p_traj.append(env.pursuer_state[:2].copy())
    e_traj.append(env.evader_state[:2].copy())

    done = False
    steps = 0
    while not done:
        # Greedy pursuer gets FULL (unmasked) observation
        p_obs_full = env.obs_builder.build(
            self_state=env.pursuer_state,
            self_action=env.pursuer_action,
            opp_state=env.evader_state,
            opp_action=env.evader_action,
            obstacles=env.obstacles,
            los_blocked=False,  # unmasked
        )
        p_action, _ = greedy_pursuer.predict(p_obs_full, deterministic=True)

        # Evader gets partial obs (combined masking)
        e_action, _ = evader_model.predict(obs["evader"], deterministic=True)

        obs, _, terminated, truncated, info = env.step(p_action, e_action)
        done = terminated or truncated
        steps += 1

        p_traj.append(env.pursuer_state[:2].copy())
        e_traj.append(env.evader_state[:2].copy())

        dist = np.linalg.norm(env.pursuer_state[:2] - env.evader_state[:2])
        distances.append(dist)

        # Visibility check (combined: in-range AND clear LOS)
        los_clear = not line_of_sight_blocked(
            env.pursuer_state[:2], env.evader_state[:2], env.obstacles)
        in_range = dist <= 3.0
        visible = in_range and los_clear
        e_visible_to_p.append(visible)

        # Track speeds from position deltas
        if len(p_traj) >= 2:
            p_spd = np.linalg.norm(p_traj[-1] - p_traj[-2]) / 0.05
            e_spd = np.linalg.norm(e_traj[-1] - e_traj[-2]) / 0.05
            p_speeds.append(p_spd)
            e_speeds.append(e_spd)

    env.close()
    return {
        "p_traj": np.array(p_traj), "e_traj": np.array(e_traj),
        "obstacles": obstacles, "distances": distances,
        "e_visible_to_p": e_visible_to_p,
        "p_speeds": p_speeds, "e_speeds": e_speeds,
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

    # Start markers (triangle)
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
    vis_pct = np.mean(ep["e_visible_to_p"]) * 100 if ep["e_visible_to_p"] else 0
    ax.set_title(f"Ep {ep_num}: {outcome} ({ep['steps']}st, vis={vis_pct:.0f}%)",
                 fontsize=9, color=color, fontweight="bold")
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
    all_p_speeds = []
    all_e_speeds = []
    for ep in episodes:
        all_dists.extend(ep["distances"])
        all_vis.extend(ep["e_visible_to_p"])
        ep_lengths.append(ep["steps"])
        all_p_speeds.extend(ep["p_speeds"])
        all_e_speeds.extend(ep["e_speeds"])

    print(f"\n  Avg episode length: {np.mean(ep_lengths):.0f} steps")
    print(f"  Avg distance: {np.mean(all_dists):.2f}m")
    print(f"  Min distance: {np.min(all_dists):.2f}m")
    print(f"  Max distance: {np.max(all_dists):.2f}m")
    print(f"  % time visible: {np.mean(all_vis):.1%}")

    print(f"\n  Pursuer (greedy, v_max=1.0):")
    print(f"    Avg speed: {np.mean(all_p_speeds):.3f} m/s")
    print(f"    Speed std: {np.std(all_p_speeds):.3f}")

    print(f"\n  Evader (trained, v_max=1.2):")
    print(f"    Avg speed: {np.mean(all_e_speeds):.3f} m/s")
    print(f"    Speed std: {np.std(all_e_speeds):.3f}")

    # Per-episode breakdown
    print(f"\n  Per-episode breakdown:")
    for i, ep in enumerate(episodes):
        p_avg_spd = np.mean(ep["p_speeds"]) if ep["p_speeds"] else 0
        e_avg_spd = np.mean(ep["e_speeds"]) if ep["e_speeds"] else 0
        vis_pct = np.mean(ep["e_visible_to_p"]) * 100 if ep["e_visible_to_p"] else 0
        avg_dist = np.mean(ep["distances"]) if ep["distances"] else 0
        p_disp = np.linalg.norm(ep["p_traj"][-1] - ep["p_traj"][0])
        e_disp = np.linalg.norm(ep["e_traj"][-1] - ep["e_traj"][0])
        p_total = np.sum(np.linalg.norm(np.diff(ep["p_traj"], axis=0), axis=1))
        e_total = np.sum(np.linalg.norm(np.diff(ep["e_traj"], axis=0), axis=1))
        status = "ESC" if ep["escaped"] else "CAP"
        print(f"    Ep {i+1}: {status} {ep['steps']:3d}st | "
              f"P: spd={p_avg_spd:.2f} path={p_total:.1f}m | "
              f"E: spd={e_avg_spd:.2f} path={e_total:.1f}m | "
              f"vis={vis_pct:.0f}% dist={avg_dist:.1f}m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evader", type=str, required=True,
                        help="Path to evader model (e.g. evader_best.zip)")
    parser.add_argument("--output", type=str, default="s1v6b_trajectories.png")
    parser.add_argument("--n_episodes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--evader_speed", type=float, default=1.2)
    parser.add_argument("--label", type=str, default="S1v6b (1.2x speed vs greedy full-obs)")
    args = parser.parse_args()

    print(f"Config: evader_speed={args.evader_speed}x, combined masking 3.0m+LOS")
    print(f"Loading evader: {args.evader}")
    e_model = PPO.load(args.evader, device="cpu")

    greedy = GreedyPursuerPolicy(v_max=1.0, arena_half_w=5.0, arena_half_h=5.0)
    print(f"Pursuer: GreedyPursuerPolicy (full obs, v_max=1.0)")

    episodes = []
    for i in range(args.n_episodes):
        ep = run_episode(greedy, e_model, evader_speed=args.evader_speed,
                         seed=args.seed + i)
        status = "ESCAPED" if ep["escaped"] else f"CAPTURED at step {ep['steps']}"
        print(f"  Ep {i+1}: {status}")
        episodes.append(ep)

    analyze_episodes(episodes)

    # Plot grid
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
    fig.suptitle(f"{args.label}\nCaptures {captures}/{n}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
