"""Deep analysis of S1v6b: large-sample stats, distance/speed profiles,
escape vs capture comparison, visibility timelines, obstacle utilization."""
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
from training.baselines import GreedyPursuerPolicy


def make_env(evader_speed=1.2, seed=None):
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
    env = make_env(evader_speed=evader_speed, seed=seed)
    obs, _ = env.reset()

    p_traj, e_traj = [], []
    distances, bearings = [], []
    e_visible_to_p = []
    p_speeds, e_speeds = [], []
    obstacle_dists_e = []  # evader distance to nearest obstacle
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
            los_blocked=False,
        )
        p_action, _ = greedy_pursuer.predict(p_obs_full, deterministic=True)
        e_action, _ = evader_model.predict(obs["evader"], deterministic=True)

        obs, _, terminated, truncated, info = env.step(p_action, e_action)
        done = terminated or truncated
        steps += 1

        p_pos = env.pursuer_state[:2].copy()
        e_pos = env.evader_state[:2].copy()
        p_traj.append(p_pos)
        e_traj.append(e_pos)

        dist = np.linalg.norm(p_pos - e_pos)
        distances.append(dist)

        # Bearing from pursuer to evader
        dp = e_pos - p_pos
        bearing = np.arctan2(dp[1], dp[0])
        bearings.append(bearing)

        # Visibility
        los_clear = not line_of_sight_blocked(p_pos, e_pos, env.obstacles)
        in_range = dist <= 3.0
        visible = in_range and los_clear
        e_visible_to_p.append(visible)

        # Speeds
        if len(p_traj) >= 2:
            p_speeds.append(np.linalg.norm(p_traj[-1] - p_traj[-2]) / 0.05)
            e_speeds.append(np.linalg.norm(e_traj[-1] - e_traj[-2]) / 0.05)

        # Evader distance to nearest obstacle
        if obstacles:
            obs_dists = [np.linalg.norm(e_pos - np.array([ox, oy])) - r
                         for ox, oy, r in obstacles]
            obstacle_dists_e.append(min(obs_dists))
        else:
            obstacle_dists_e.append(float('inf'))

    env.close()
    return {
        "p_traj": np.array(p_traj), "e_traj": np.array(e_traj),
        "obstacles": obstacles, "distances": distances, "bearings": bearings,
        "e_visible_to_p": e_visible_to_p,
        "p_speeds": p_speeds, "e_speeds": e_speeds,
        "obstacle_dists_e": obstacle_dists_e,
        "steps": steps, "escaped": truncated, "captured": terminated,
        "n_obstacles": len(obstacles),
    }


def run_batch(greedy, evader_model, n_episodes, evader_speed, seed_base):
    episodes = []
    for i in range(n_episodes):
        ep = run_episode(greedy, evader_model, evader_speed=evader_speed,
                         seed=seed_base + i)
        episodes.append(ep)
    return episodes


def print_stats(episodes, label):
    n = len(episodes)
    captures = sum(1 for ep in episodes if ep["captured"])
    escapes = sum(1 for ep in episodes if ep["escaped"])
    esc_eps = [ep for ep in episodes if ep["escaped"]]
    cap_eps = [ep for ep in episodes if ep["captured"]]

    print(f"\n{'='*70}")
    print(f"  {label} — {n} episodes")
    print(f"{'='*70}")
    print(f"  Escape rate: {escapes}/{n} = {escapes/n:.1%}")
    print(f"  Capture rate: {captures}/{n} = {captures/n:.1%}")

    # Episode lengths
    all_lens = [ep["steps"] for ep in episodes]
    cap_lens = [ep["steps"] for ep in cap_eps]
    print(f"\n  Episode length:")
    print(f"    All:      {np.mean(all_lens):.0f} +/- {np.std(all_lens):.0f} steps")
    if cap_lens:
        print(f"    Captures: {np.mean(cap_lens):.0f} +/- {np.std(cap_lens):.0f} steps "
              f"(min={np.min(cap_lens)}, max={np.max(cap_lens)})")

    # Distances
    all_dists = [d for ep in episodes for d in ep["distances"]]
    esc_dists = [d for ep in esc_eps for d in ep["distances"]]
    cap_dists = [d for ep in cap_eps for d in ep["distances"]]
    print(f"\n  Inter-agent distance:")
    print(f"    All:     avg={np.mean(all_dists):.2f}m, "
          f"median={np.median(all_dists):.2f}m, std={np.std(all_dists):.2f}")
    if esc_dists:
        print(f"    Escapes: avg={np.mean(esc_dists):.2f}m, "
              f"median={np.median(esc_dists):.2f}m")
    if cap_dists:
        print(f"    Captures: avg={np.mean(cap_dists):.2f}m, "
              f"median={np.median(cap_dists):.2f}m")

    # Speeds
    all_p_spd = [s for ep in episodes for s in ep["p_speeds"]]
    all_e_spd = [s for ep in episodes for s in ep["e_speeds"]]
    esc_e_spd = [s for ep in esc_eps for s in ep["e_speeds"]]
    cap_e_spd = [s for ep in cap_eps for s in ep["e_speeds"]]
    print(f"\n  Speeds:")
    print(f"    Pursuer:  avg={np.mean(all_p_spd):.3f} m/s (max=1.0)")
    print(f"    Evader:   avg={np.mean(all_e_spd):.3f} m/s (max=1.2)")
    if esc_e_spd:
        print(f"    Evader (esc): avg={np.mean(esc_e_spd):.3f}")
    if cap_e_spd:
        print(f"    Evader (cap): avg={np.mean(cap_e_spd):.3f}")

    # Visibility
    all_vis = [v for ep in episodes for v in ep["e_visible_to_p"]]
    esc_vis = [v for ep in esc_eps for v in ep["e_visible_to_p"]]
    cap_vis = [v for ep in cap_eps for v in ep["e_visible_to_p"]]
    print(f"\n  Visibility (evader seen by pursuer):")
    print(f"    All:      {np.mean(all_vis):.1%}")
    if esc_vis:
        print(f"    Escapes:  {np.mean(esc_vis):.1%}")
    if cap_vis:
        print(f"    Captures: {np.mean(cap_vis):.1%}")

    # Obstacle proximity
    all_obs_dist = [d for ep in episodes for d in ep["obstacle_dists_e"]
                    if d != float('inf')]
    esc_obs_dist = [d for ep in esc_eps for d in ep["obstacle_dists_e"]
                    if d != float('inf')]
    if all_obs_dist:
        pct_near = sum(1 for d in all_obs_dist if d < 0.5) / len(all_obs_dist)
        pct_touching = sum(1 for d in all_obs_dist if d < 0.1) / len(all_obs_dist)
        print(f"\n  Obstacle proximity (evader):")
        print(f"    Avg dist to nearest: {np.mean(all_obs_dist):.2f}m")
        print(f"    % within 0.5m: {pct_near:.1%}")
        print(f"    % touching (<0.1m): {pct_touching:.1%}")
        if esc_obs_dist:
            esc_near = sum(1 for d in esc_obs_dist if d < 0.5) / len(esc_obs_dist)
            print(f"    Escapes % within 0.5m: {esc_near:.1%}")

    # Obstacle count vs escape rate
    obs_counts = {}
    for ep in episodes:
        c = ep["n_obstacles"]
        if c not in obs_counts:
            obs_counts[c] = {"total": 0, "escaped": 0}
        obs_counts[c]["total"] += 1
        if ep["escaped"]:
            obs_counts[c]["escaped"] += 1
    print(f"\n  Escape rate by obstacle count:")
    for c in sorted(obs_counts.keys()):
        d = obs_counts[c]
        print(f"    {c} obstacles: {d['escaped']}/{d['total']} = "
              f"{d['escaped']/d['total']:.0%}")

    # Spawn distance vs outcome
    spawn_dists_esc = [ep["distances"][0] for ep in esc_eps if ep["distances"]]
    spawn_dists_cap = [ep["distances"][0] for ep in cap_eps if ep["distances"]]
    print(f"\n  Initial spawn distance:")
    if spawn_dists_esc:
        print(f"    Escapes:  avg={np.mean(spawn_dists_esc):.2f}m, "
              f"min={np.min(spawn_dists_esc):.2f}m")
    if spawn_dists_cap:
        print(f"    Captures: avg={np.mean(spawn_dists_cap):.2f}m, "
              f"min={np.min(spawn_dists_cap):.2f}m")

    return {
        "escapes": escapes, "captures": captures, "n": n,
        "esc_eps": esc_eps, "cap_eps": cap_eps,
    }


def plot_distance_profiles(esc_eps, cap_eps, output_path, label):
    """Plot distance-over-time for escaped vs captured episodes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Escapes
    ax = axes[0]
    for ep in esc_eps[:20]:
        ax.plot(ep["distances"], alpha=0.3, color="blue", linewidth=0.8)
    if esc_eps:
        # Compute mean by padding to max length
        max_len = max(len(ep["distances"]) for ep in esc_eps[:20])
        padded = np.full((min(20, len(esc_eps)), max_len), np.nan)
        for i, ep in enumerate(esc_eps[:20]):
            padded[i, :len(ep["distances"])] = ep["distances"]
        mean_dist = np.nanmean(padded, axis=0)
        ax.plot(mean_dist, color="blue", linewidth=2, label="Mean")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Capture radius")
    ax.axhline(y=3.0, color="orange", linestyle="--", alpha=0.5, label="Sensing radius")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance (m)")
    ax.set_title(f"Escaped episodes ({len(esc_eps)})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.2)

    # Captures
    ax = axes[1]
    for ep in cap_eps[:20]:
        ax.plot(ep["distances"], alpha=0.3, color="red", linewidth=0.8)
    if cap_eps:
        max_len = max(len(ep["distances"]) for ep in cap_eps[:20])
        padded = np.full((min(20, len(cap_eps)), max_len), np.nan)
        for i, ep in enumerate(cap_eps[:20]):
            padded[i, :len(ep["distances"])] = ep["distances"]
        mean_dist = np.nanmean(padded, axis=0)
        ax.plot(mean_dist, color="red", linewidth=2, label="Mean")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Capture radius")
    ax.axhline(y=3.0, color="orange", linestyle="--", alpha=0.5, label="Sensing radius")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance (m)")
    ax.set_title(f"Captured episodes ({len(cap_eps)})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"{label} — Distance Profiles", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_speed_profiles(esc_eps, cap_eps, output_path, label):
    """Plot evader speed over time for escaped vs captured."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, eps, title, color in [
        (axes[0], esc_eps, "Escaped", "blue"),
        (axes[1], cap_eps, "Captured", "red"),
    ]:
        for ep in eps[:20]:
            ax.plot(ep["e_speeds"], alpha=0.2, color=color, linewidth=0.8)
        if eps:
            max_len = max(len(ep["e_speeds"]) for ep in eps[:20])
            padded = np.full((min(20, len(eps)), max_len), np.nan)
            for i, ep in enumerate(eps[:20]):
                padded[i, :len(ep["e_speeds"])] = ep["e_speeds"]
            mean_spd = np.nanmean(padded, axis=0)
            ax.plot(mean_spd, color=color, linewidth=2, label="Mean evader")

            # Also plot pursuer
            max_len_p = max(len(ep["p_speeds"]) for ep in eps[:20])
            padded_p = np.full((min(20, len(eps)), max_len_p), np.nan)
            for i, ep in enumerate(eps[:20]):
                padded_p[i, :len(ep["p_speeds"])] = ep["p_speeds"]
            mean_spd_p = np.nanmean(padded_p, axis=0)
            ax.plot(mean_spd_p, color="gray", linewidth=2, alpha=0.7, label="Mean pursuer")

        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Pursuer v_max")
        ax.axhline(y=1.2, color=color, linestyle=":", alpha=0.5, label="Evader v_max")
        ax.set_xlabel("Step")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title(f"{title} ({len(eps)} eps)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.5)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{label} — Speed Profiles", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_visibility_timeline(esc_eps, cap_eps, output_path, label):
    """Plot visibility over time as heatmap-like strips."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    for ax, eps, title, cmap in [
        (axes[0], esc_eps, "Escaped", "Blues"),
        (axes[1], cap_eps, "Captured", "Reds"),
    ]:
        if not eps:
            ax.set_title(f"{title} (0 episodes)")
            continue
        n_show = min(30, len(eps))
        max_len = max(len(ep["e_visible_to_p"]) for ep in eps[:n_show])
        vis_matrix = np.full((n_show, max_len), np.nan)
        for i, ep in enumerate(eps[:n_show]):
            vis = ep["e_visible_to_p"]
            vis_matrix[i, :len(vis)] = [float(v) for v in vis]

        ax.imshow(vis_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xlabel("Step")
        ax.set_ylabel("Episode")
        pct = np.nanmean(vis_matrix) * 100
        ax.set_title(f"{title} ({n_show} eps, avg vis={pct:.0f}%)")

    fig.suptitle(f"{label} — Visibility Timeline", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_obstacle_proximity(esc_eps, cap_eps, output_path, label):
    """Plot evader distance to nearest obstacle over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, eps, title, color in [
        (axes[0], esc_eps, "Escaped", "blue"),
        (axes[1], cap_eps, "Captured", "red"),
    ]:
        valid = [ep for ep in eps if any(d != float('inf') for d in ep["obstacle_dists_e"])]
        for ep in valid[:20]:
            dists = [d if d != float('inf') else np.nan for d in ep["obstacle_dists_e"]]
            ax.plot(dists, alpha=0.2, color=color, linewidth=0.8)
        if valid:
            max_len = max(len(ep["obstacle_dists_e"]) for ep in valid[:20])
            padded = np.full((min(20, len(valid)), max_len), np.nan)
            for i, ep in enumerate(valid[:20]):
                dists = [d if d != float('inf') else np.nan for d in ep["obstacle_dists_e"]]
                padded[i, :len(dists)] = dists
            mean_d = np.nanmean(padded, axis=0)
            ax.plot(mean_d, color=color, linewidth=2, label="Mean")

        ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.3)
        ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="0.5m threshold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Dist to nearest obstacle (m)")
        ax.set_title(f"{title} ({len(eps)} eps)")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.5, 8)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{label} — Evader Obstacle Proximity", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_distributions(episodes, output_path, label):
    """Plot histograms: capture time, spawn distance, per-ep escape rate."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    esc_eps = [ep for ep in episodes if ep["escaped"]]
    cap_eps = [ep for ep in episodes if ep["captured"]]

    # 1) Capture time distribution
    ax = axes[0, 0]
    if cap_eps:
        cap_times = [ep["steps"] for ep in cap_eps]
        ax.hist(cap_times, bins=20, color="red", alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(cap_times), color="darkred", linestyle="--",
                   label=f"Mean={np.mean(cap_times):.0f}")
        ax.legend()
    ax.set_xlabel("Steps to capture")
    ax.set_ylabel("Count")
    ax.set_title("Capture time distribution")
    ax.grid(True, alpha=0.2)

    # 2) Spawn distance: escaped vs captured
    ax = axes[0, 1]
    if esc_eps:
        spawn_esc = [ep["distances"][0] for ep in esc_eps if ep["distances"]]
        ax.hist(spawn_esc, bins=15, color="blue", alpha=0.5, label="Escaped",
                edgecolor="black")
    if cap_eps:
        spawn_cap = [ep["distances"][0] for ep in cap_eps if ep["distances"]]
        ax.hist(spawn_cap, bins=15, color="red", alpha=0.5, label="Captured",
                edgecolor="black")
    ax.set_xlabel("Initial spawn distance (m)")
    ax.set_ylabel("Count")
    ax.set_title("Spawn distance by outcome")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 3) Per-episode evader avg speed: escaped vs captured
    ax = axes[1, 0]
    if esc_eps:
        esc_spd = [np.mean(ep["e_speeds"]) for ep in esc_eps if ep["e_speeds"]]
        ax.hist(esc_spd, bins=15, color="blue", alpha=0.5, label="Escaped",
                edgecolor="black")
    if cap_eps:
        cap_spd = [np.mean(ep["e_speeds"]) for ep in cap_eps if ep["e_speeds"]]
        ax.hist(cap_spd, bins=15, color="red", alpha=0.5, label="Captured",
                edgecolor="black")
    ax.set_xlabel("Avg evader speed (m/s)")
    ax.set_ylabel("Count")
    ax.set_title("Evader speed by outcome")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 4) Per-episode visibility: escaped vs captured
    ax = axes[1, 1]
    if esc_eps:
        esc_vis = [np.mean(ep["e_visible_to_p"]) for ep in esc_eps]
        ax.hist(esc_vis, bins=15, color="blue", alpha=0.5, label="Escaped",
                edgecolor="black")
    if cap_eps:
        cap_vis = [np.mean(ep["e_visible_to_p"]) for ep in cap_eps]
        ax.hist(cap_vis, bins=15, color="red", alpha=0.5, label="Captured",
                edgecolor="black")
    ax.set_xlabel("% time visible")
    ax.set_ylabel("Count")
    ax.set_title("Visibility by outcome")
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"{label} — Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evader", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/s1v6b_analysis")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--evader_speed", type=float, default=1.2)
    parser.add_argument("--label", type=str, default="S1v6b")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Config: evader_speed={args.evader_speed}x, combined masking 3.0m+LOS")
    print(f"Running {args.n_episodes} episodes...")

    e_model = PPO.load(args.evader, device="cpu")
    greedy = GreedyPursuerPolicy(v_max=1.0, arena_half_w=5.0, arena_half_h=5.0)

    episodes = run_batch(greedy, e_model, args.n_episodes, args.evader_speed, args.seed)

    stats = print_stats(episodes, args.label)
    esc_eps = stats["esc_eps"]
    cap_eps = stats["cap_eps"]

    # Generate all plots
    plot_distance_profiles(esc_eps, cap_eps,
                           str(out / "distance_profiles.png"), args.label)
    plot_speed_profiles(esc_eps, cap_eps,
                        str(out / "speed_profiles.png"), args.label)
    plot_visibility_timeline(esc_eps, cap_eps,
                             str(out / "visibility_timeline.png"), args.label)
    plot_obstacle_proximity(esc_eps, cap_eps,
                            str(out / "obstacle_proximity.png"), args.label)
    plot_distributions(episodes,
                       str(out / "distributions.png"), args.label)

    print(f"\nAll plots saved to {out}/")


if __name__ == "__main__":
    main()
