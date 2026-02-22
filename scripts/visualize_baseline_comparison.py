"""Visualize Baseline PPO vs Baseline PPO + VCP-CBF Filter.

Generates:
  1. Trajectory plots (bird's-eye view with obstacles and arena)
  2. Distance-over-time plots for sample episodes
  3. CBF margin time series showing safety constraint status
  4. Action comparison (nominal vs filtered) showing filter interventions
  5. Statistical bar charts across many episodes

Usage:
    python scripts/visualize_baseline_comparison.py \
        --model-path models/obstacle_ppo_42/final_model.zip \
        --n-traj-episodes 5 --n-stat-episodes 100
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper
from evaluation.comparison_framework import compute_cbf_margins
from safety.vcp_cbf import VCPCBFFilter
from stable_baselines3 import PPO


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLORS = {
    "pursuer": "#1f77b4",       # blue
    "evader": "#d62728",        # red
    "pursuer_light": "#aec7e8",
    "evader_light": "#ff9896",
    "obstacle": "#7f7f7f",      # gray
    "danger_zone": "#ffcccc",
    "arena_border": "#333333",
    "safe": "#2ca02c",          # green
    "warning": "#ff7f0e",       # orange
    "danger": "#d62728",        # red
    "intervention": "#9467bd",  # purple
    "baseline": "#1f77b4",
    "filtered": "#ff7f0e",
}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_eval_env(n_obstacles=2, arena_size=20.0, dt=0.05, max_steps=600):
    env = PursuitEvasionEnv(
        arena_width=arena_size, arena_height=arena_size,
        dt=dt, max_steps=max_steps,
        capture_radius=0.5, collision_radius=0.3, robot_radius=0.15,
        pursuer_v_max=1.0, pursuer_omega_max=2.84,
        evader_v_max=1.0, evader_omega_max=2.84,
        n_obstacles=n_obstacles,
        obstacle_radius_range=(0.3, 1.0), obstacle_margin=0.5,
        n_obstacle_obs=min(n_obstacles, 3),
    )
    return SingleAgentPEWrapper(env, role="pursuer", opponent_policy=None)


# ---------------------------------------------------------------------------
# Data collection: run one episode and record per-step data
# ---------------------------------------------------------------------------
def collect_episode_data(model, env, pe_env, safety_filter, seed,
                         arena_half_w=10.0, arena_half_h=10.0):
    """Run one episode and return per-step trajectory data."""
    obs, info = env.reset(seed=seed)

    data = {
        "pursuer_xy": [], "pursuer_theta": [],
        "evader_xy": [], "evader_theta": [],
        "distance": [], "cbf_margins": [], "min_cbf_margin": [],
        "action_nominal": [], "action_executed": [],
        "qp_correction": [], "intervened": [],
        "rewards": [],
        "obstacles": pe_env.obstacles.copy(),
    }

    expected_obs_dim = model.observation_space.shape[0]
    done = False
    ep_reward = 0.0
    captured = False

    while not done:
        p_state = pe_env.pursuer_state.copy()
        e_state = pe_env.evader_state.copy()
        obstacles = pe_env.obstacles

        data["pursuer_xy"].append(p_state[:2].copy())
        data["pursuer_theta"].append(p_state[2])
        data["evader_xy"].append(e_state[:2].copy())
        data["evader_theta"].append(e_state[2])
        data["distance"].append(np.linalg.norm(p_state[:2] - e_state[:2]))

        # CBF margins
        margins = compute_cbf_margins(
            p_state, obstacles, e_state,
            arena_half_w, arena_half_h,
            d=0.1, alpha=1.0, robot_radius=0.15, r_min=0.35,
        )
        data["cbf_margins"].append(margins)
        data["min_cbf_margin"].append(min(margins) if margins else 0.0)

        # Get nominal action
        model_obs = obs[:expected_obs_dim] if len(obs) > expected_obs_dim else obs
        action_nom, _ = model.predict(model_obs, deterministic=True)
        data["action_nominal"].append(action_nom.copy())

        # Apply filter if provided
        if safety_filter is not None:
            action_exec, finfo = safety_filter.filter_action(
                action_nom.copy(), p_state,
                obstacles=obstacles, opponent_state=e_state,
            )
            correction = float(np.linalg.norm(action_exec - action_nom))
            data["action_executed"].append(action_exec.copy())
            data["qp_correction"].append(correction)
            data["intervened"].append(correction > 0.01)
        else:
            data["action_executed"].append(action_nom.copy())
            data["qp_correction"].append(0.0)
            data["intervened"].append(False)

        obs, reward, terminated, truncated, step_info = env.step(
            data["action_executed"][-1]
        )
        done = terminated or truncated
        ep_reward += reward
        data["rewards"].append(reward)

    ep_metrics = step_info.get("episode_metrics", {})
    data["captured"] = ep_metrics.get("captured", False)
    data["episode_reward"] = ep_reward
    data["episode_length"] = len(data["pursuer_xy"])

    # Convert lists to arrays
    for key in ["pursuer_xy", "evader_xy", "action_nominal", "action_executed"]:
        data[key] = np.array(data[key])
    for key in ["pursuer_theta", "evader_theta", "distance", "min_cbf_margin",
                 "qp_correction", "rewards"]:
        data[key] = np.array(data[key])
    data["intervened"] = np.array(data["intervened"])

    return data


# ---------------------------------------------------------------------------
# Plot 1: Trajectory bird's-eye view
# ---------------------------------------------------------------------------
def plot_trajectories(baseline_data, filtered_data, output_dir, episode_idx):
    """Side-by-side trajectory plots for one episode."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    for ax, data, title in zip(
        axes,
        [baseline_data, filtered_data],
        ["Baseline PPO", "Baseline PPO + VCP-CBF Filter"],
    ):
        arena_half = 10.0

        # Arena boundary
        rect = mpatches.Rectangle(
            (-arena_half, -arena_half), 2 * arena_half, 2 * arena_half,
            linewidth=2, edgecolor=COLORS["arena_border"],
            facecolor="#f7f7f7", zorder=0,
        )
        ax.add_patch(rect)

        # Obstacles with danger zones
        for obs in data["obstacles"]:
            # Danger zone (obstacle + robot_radius)
            danger = plt.Circle(
                (obs["x"], obs["y"]), obs["radius"] + 0.15,
                color=COLORS["danger_zone"], alpha=0.4, zorder=1,
            )
            ax.add_patch(danger)
            # Obstacle body
            circle = plt.Circle(
                (obs["x"], obs["y"]), obs["radius"],
                color=COLORS["obstacle"], alpha=0.7, zorder=2,
            )
            ax.add_patch(circle)

        # Pursuer trajectory (color-coded by time)
        p_xy = data["pursuer_xy"]
        e_xy = data["evader_xy"]
        n_steps = len(p_xy)
        t_norm = np.linspace(0, 1, n_steps)

        # Use line collection for color gradient
        for xy, base_color, label in [
            (p_xy, COLORS["pursuer"], "Pursuer"),
            (e_xy, COLORS["evader"], "Evader"),
        ]:
            points = xy.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            cmap = plt.cm.Blues if "Pursuer" in label else plt.cm.Reds
            lc = LineCollection(segments, cmap=cmap, linewidths=1.5, zorder=3)
            lc.set_array(t_norm[:-1])
            lc.set_clim(0, 1)
            ax.add_collection(lc)

            # Start and end markers
            ax.plot(xy[0, 0], xy[0, 1], "o", color=base_color, markersize=8,
                    zorder=5, label=f"{label} start")
            ax.plot(xy[-1, 0], xy[-1, 1], "s", color=base_color, markersize=8,
                    zorder=5, markeredgecolor="black", markeredgewidth=1)

            # Heading arrow at end
            theta = data["pursuer_theta"][-1] if "Pursuer" in label else data["evader_theta"][-1]
            dx, dy = 0.8 * np.cos(theta), 0.8 * np.sin(theta)
            ax.annotate("", xy=(xy[-1, 0] + dx, xy[-1, 1] + dy),
                        xytext=(xy[-1, 0], xy[-1, 1]),
                        arrowprops=dict(arrowstyle="->", color=base_color, lw=2),
                        zorder=6)

        # Highlight interventions for filtered approach
        if data["intervened"].any():
            interv_xy = data["pursuer_xy"][data["intervened"]]
            ax.scatter(interv_xy[:, 0], interv_xy[:, 1], c=COLORS["intervention"],
                       s=6, alpha=0.5, zorder=4, label="Filter intervention")

        # Capture marker
        if data["captured"]:
            ax.plot(p_xy[-1, 0], p_xy[-1, 1], "*", color="gold", markersize=15,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=7,
                    label="Capture")

        outcome = "CAPTURED" if data["captured"] else "TIMEOUT"
        reward = data["episode_reward"]
        ax.set_title(f"{title}\n{outcome} | reward={reward:.1f} | "
                     f"steps={data['episode_length']}", fontsize=11)
        ax.set_xlim(-arena_half - 0.5, arena_half + 0.5)
        ax.set_ylim(-arena_half - 0.5, arena_half + 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Episode {episode_idx}: Trajectory Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, f"trajectory_ep{episode_idx:02d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 2: Distance over time
# ---------------------------------------------------------------------------
def plot_distance(baseline_data, filtered_data, output_dir, episode_idx):
    """Distance-over-time comparison for one episode."""
    fig, ax = plt.subplots(figsize=(10, 4))

    dt = 0.05
    t_base = np.arange(len(baseline_data["distance"])) * dt
    t_filt = np.arange(len(filtered_data["distance"])) * dt

    ax.plot(t_base, baseline_data["distance"], color=COLORS["baseline"],
            label="Baseline PPO", linewidth=1.5)
    ax.plot(t_filt, filtered_data["distance"], color=COLORS["filtered"],
            label="Baseline + Filter", linewidth=1.5, linestyle="--")
    ax.axhline(y=0.5, color="gold", linestyle=":", alpha=0.7, label="Capture radius")

    # Mark capture points
    for data, color, t_arr in [
        (baseline_data, COLORS["baseline"], t_base),
        (filtered_data, COLORS["filtered"], t_filt),
    ]:
        if data["captured"]:
            ax.axvline(x=t_arr[-1], color=color, linestyle=":", alpha=0.5)
            ax.scatter([t_arr[-1]], [data["distance"][-1]], color=color,
                       marker="*", s=100, zorder=5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pursuer-Evader Distance (m)")
    ax.set_title(f"Episode {episode_idx}: Distance Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(output_dir, f"distance_ep{episode_idx:02d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 3: CBF margin time series
# ---------------------------------------------------------------------------
def plot_cbf_margins(baseline_data, filtered_data, output_dir, episode_idx):
    """CBF margin over time showing safety constraint status."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    dt = 0.05
    for ax, data, title, color in zip(
        axes,
        [baseline_data, filtered_data],
        ["Baseline PPO", "Baseline + Filter"],
        [COLORS["baseline"], COLORS["filtered"]],
    ):
        t = np.arange(len(data["min_cbf_margin"])) * dt
        margins = data["min_cbf_margin"]

        # Color by safety status
        safe = margins >= 0
        ax.fill_between(t, margins, 0, where=safe, alpha=0.15,
                        color=COLORS["safe"], label="Safe (h >= 0)")
        ax.fill_between(t, margins, 0, where=~safe, alpha=0.3,
                        color=COLORS["danger"], label="Violated (h < 0)")
        ax.plot(t, margins, color=color, linewidth=1.0)
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")

        # Count violations
        n_violations = np.sum(margins < -1e-4)
        pct = n_violations / len(margins) * 100
        ax.set_title(f"{title} | violations: {n_violations}/{len(margins)} ({pct:.1f}%)",
                     fontsize=10)
        ax.set_ylabel("Min CBF Margin (h)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Episode {episode_idx}: CBF Safety Margins", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, f"cbf_margins_ep{episode_idx:02d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 4: Action comparison (nominal vs filtered)
# ---------------------------------------------------------------------------
def plot_actions(baseline_data, filtered_data, output_dir, episode_idx):
    """Compare nominal vs executed actions, highlighting filter interventions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True)

    dt = 0.05
    t_filt = np.arange(len(filtered_data["action_nominal"])) * dt
    t_base = np.arange(len(baseline_data["action_nominal"])) * dt

    labels = ["v (m/s)", "omega (rad/s)"]
    for i, label in enumerate(labels):
        # Baseline (left column) - no filter, so nominal = executed
        ax = axes[i][0]
        ax.plot(t_base, baseline_data["action_nominal"][:, i],
                color=COLORS["baseline"], linewidth=1.0, label="Action")
        ax.set_ylabel(label)
        if i == 0:
            ax.set_title("Baseline PPO", fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)

        # Filtered (right column) - show nominal vs executed
        ax = axes[i][1]
        ax.plot(t_filt, filtered_data["action_nominal"][:, i],
                color=COLORS["baseline"], linewidth=1.0, alpha=0.5,
                label="Nominal")
        ax.plot(t_filt, filtered_data["action_executed"][:, i],
                color=COLORS["filtered"], linewidth=1.0, label="Executed (filtered)")

        # Highlight intervention regions
        interv = filtered_data["intervened"]
        if interv.any():
            interv_t = t_filt[interv]
            for t_i in interv_t:
                ax.axvline(x=t_i, color=COLORS["intervention"], alpha=0.03,
                           linewidth=1)

        ax.set_ylabel(label)
        if i == 0:
            n_interv = interv.sum()
            pct = n_interv / len(interv) * 100
            ax.set_title(f"Baseline + Filter | interventions: {n_interv} ({pct:.1f}%)",
                         fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")

    fig.suptitle(f"Episode {episode_idx}: Actions", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, f"actions_ep{episode_idx:02d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 5: Statistical summary bar charts
# ---------------------------------------------------------------------------
def plot_statistics(all_baseline, all_filtered, output_dir):
    """Bar charts comparing aggregate metrics across all episodes."""
    # Compute stats
    def stats(data_list, key):
        vals = [d[key] for d in data_list]
        if isinstance(vals[0], (bool, np.bool_)):
            return np.mean(vals), 0.0
        return np.mean(vals), np.std(vals)

    n_base = len(all_baseline)
    n_filt = len(all_filtered)

    base_capture = np.mean([d["captured"] for d in all_baseline]) * 100
    filt_capture = np.mean([d["captured"] for d in all_filtered]) * 100

    base_reward_m, base_reward_s = stats(all_baseline, "episode_reward")
    filt_reward_m, filt_reward_s = stats(all_filtered, "episode_reward")

    # Safety violations per episode
    def violation_rate(data_list):
        total_violations = 0
        total_steps = 0
        for d in data_list:
            total_violations += np.sum(d["min_cbf_margin"] < -1e-4)
            total_steps += len(d["min_cbf_margin"])
        return total_violations / max(total_steps, 1) * 100
    base_viol = violation_rate(all_baseline)
    filt_viol = violation_rate(all_filtered)

    # Intervention rate
    def interv_rate(data_list):
        total_interv = sum(d["intervened"].sum() for d in data_list)
        total_steps = sum(len(d["intervened"]) for d in data_list)
        return total_interv / max(total_steps, 1) * 100
    filt_interv = interv_rate(all_filtered)

    # Mean min distance
    base_mindist = np.mean([d["distance"].min() for d in all_baseline])
    filt_mindist = np.mean([d["distance"].min() for d in all_filtered])

    # Mean episode length
    base_len_m = np.mean([d["episode_length"] for d in all_baseline])
    filt_len_m = np.mean([d["episode_length"] for d in all_filtered])

    # --- Figure ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    bar_width = 0.35
    x = np.array([0, 1])
    labels = ["Baseline", "Base+Filter"]
    colors = [COLORS["baseline"], COLORS["filtered"]]

    # 1. Capture Rate
    ax = axes[0, 0]
    bars = ax.bar(x, [base_capture, filt_capture], color=colors, width=bar_width * 2)
    ax.set_ylabel("Capture Rate (%)")
    ax.set_title("Capture Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, [base_capture, filt_capture]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Mean Reward
    ax = axes[0, 1]
    bars = ax.bar(x, [base_reward_m, filt_reward_m], color=colors, width=bar_width * 2,
                  yerr=[base_reward_s, filt_reward_s], capsize=5)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Mean Episode Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for bar, val in zip(bars, [base_reward_m, filt_reward_m]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    # 3. Safety Violations
    ax = axes[0, 2]
    bars = ax.bar(x, [base_viol, filt_viol], color=colors, width=bar_width * 2)
    ax.set_ylabel("CBF Violation Rate (%)")
    ax.set_title("Safety Violations")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for bar, val in zip(bars, [base_viol, filt_viol]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10)

    # 4. Intervention Rate (only filtered has interventions)
    ax = axes[1, 0]
    bars = ax.bar(x, [0, filt_interv], color=colors, width=bar_width * 2)
    ax.set_ylabel("Intervention Rate (%)")
    ax.set_title("Filter Intervention Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.text(bars[0].get_x() + bars[0].get_width() / 2, 1, "N/A",
            ha="center", va="bottom", fontsize=10, color="gray")
    ax.text(bars[1].get_x() + bars[1].get_width() / 2, bars[1].get_height() + 0.5,
            f"{filt_interv:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 5. Mean Min Distance
    ax = axes[1, 1]
    bars = ax.bar(x, [base_mindist, filt_mindist], color=colors, width=bar_width * 2)
    ax.set_ylabel("Mean Min Distance (m)")
    ax.set_title("Mean Minimum Distance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(y=0.5, color="gold", linestyle=":", alpha=0.7, label="Capture radius")
    for bar, val in zip(bars, [base_mindist, filt_mindist]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}m", ha="center", va="bottom", fontsize=10)
    ax.legend(fontsize=8)

    # 6. Mean Episode Length
    ax = axes[1, 2]
    bars = ax.bar(x, [base_len_m, filt_len_m], color=colors, width=bar_width * 2)
    ax.set_ylabel("Mean Episode Length (steps)")
    ax.set_title("Mean Episode Length")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for bar, val in zip(bars, [base_len_m, filt_len_m]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle(f"Statistical Comparison ({n_base} episodes)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "statistics_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 6: Reward distribution
# ---------------------------------------------------------------------------
def plot_reward_distribution(all_baseline, all_filtered, output_dir):
    """Histogram of episode rewards for both approaches."""
    fig, ax = plt.subplots(figsize=(10, 5))

    base_rewards = [d["episode_reward"] for d in all_baseline]
    filt_rewards = [d["episode_reward"] for d in all_filtered]

    bins = np.linspace(
        min(min(base_rewards), min(filt_rewards)) - 5,
        max(max(base_rewards), max(filt_rewards)) + 5,
        40,
    )
    ax.hist(base_rewards, bins=bins, alpha=0.6, color=COLORS["baseline"],
            label=f"Baseline (mean={np.mean(base_rewards):.1f})", edgecolor="white")
    ax.hist(filt_rewards, bins=bins, alpha=0.6, color=COLORS["filtered"],
            label=f"Base+Filter (mean={np.mean(filt_rewards):.1f})", edgecolor="white")

    ax.axvline(np.mean(base_rewards), color=COLORS["baseline"], linestyle="--", linewidth=2)
    ax.axvline(np.mean(filt_rewards), color=COLORS["filtered"], linestyle="--", linewidth=2)

    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution")
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(output_dir, "reward_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 7: Multi-episode trajectory overview
# ---------------------------------------------------------------------------
def plot_multi_trajectory_overview(all_baseline, all_filtered, output_dir, n_show=5):
    """Grid of trajectory thumbnails for multiple episodes."""
    n_show = min(n_show, len(all_baseline))
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    arena_half = 10.0

    for col in range(n_show):
        for row, (data, row_label) in enumerate([
            (all_baseline[col], "Baseline"),
            (all_filtered[col], "Base+Filter"),
        ]):
            ax = axes[row, col]

            # Arena
            rect = mpatches.Rectangle(
                (-arena_half, -arena_half), 2 * arena_half, 2 * arena_half,
                linewidth=1, edgecolor=COLORS["arena_border"],
                facecolor="#f7f7f7", zorder=0,
            )
            ax.add_patch(rect)

            # Obstacles
            for obs in data["obstacles"]:
                circle = plt.Circle(
                    (obs["x"], obs["y"]), obs["radius"],
                    color=COLORS["obstacle"], alpha=0.5, zorder=1,
                )
                ax.add_patch(circle)

            # Trajectories
            p_xy = data["pursuer_xy"]
            e_xy = data["evader_xy"]
            ax.plot(p_xy[:, 0], p_xy[:, 1], color=COLORS["pursuer"],
                    linewidth=0.8, alpha=0.7)
            ax.plot(e_xy[:, 0], e_xy[:, 1], color=COLORS["evader"],
                    linewidth=0.8, alpha=0.7)

            # Start/end
            ax.plot(p_xy[0, 0], p_xy[0, 1], "o", color=COLORS["pursuer"], markersize=5)
            ax.plot(e_xy[0, 0], e_xy[0, 1], "o", color=COLORS["evader"], markersize=5)

            if data["captured"]:
                ax.plot(p_xy[-1, 0], p_xy[-1, 1], "*", color="gold", markersize=10,
                        markeredgecolor="black", markeredgewidth=0.3, zorder=5)

            outcome = "CAP" if data["captured"] else "TO"
            ax.set_title(f"Ep{col} {outcome} r={data['episode_reward']:.0f}", fontsize=8)
            ax.set_xlim(-arena_half - 0.3, arena_half + 0.3)
            ax.set_ylim(-arena_half - 0.3, arena_half + 0.3)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)

            if col == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight="bold")

    fig.suptitle("Multi-Episode Trajectory Overview", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "trajectory_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize Baseline PPO vs Baseline + VCP-CBF Filter"
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to obstacle-trained SB3 PPO model")
    parser.add_argument("--n-traj-episodes", type=int, default=5,
                        help="Episodes for detailed trajectory plots")
    parser.add_argument("--n-stat-episodes", type=int, default=100,
                        help="Episodes for statistical comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-obstacles", type=int, default=2)
    parser.add_argument("--arena-size", type=float, default=20.0)
    parser.add_argument("--output-dir", type=str,
                        default="results/visualizations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    arena_half = args.arena_size / 2

    print(f"Loading model: {args.model_path}")
    model = PPO.load(args.model_path)

    # Safety filter
    safety_filter = VCPCBFFilter(
        d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
        robot_radius=0.15, arena_half_w=arena_half, arena_half_h=arena_half,
    )

    # Env + unwrap to PE env
    env = make_eval_env(n_obstacles=args.n_obstacles, arena_size=args.arena_size)
    pe_env = env
    while hasattr(pe_env, "env"):
        if hasattr(pe_env, "pursuer_state"):
            break
        pe_env = pe_env.env

    # Total episodes needed
    n_total = max(args.n_traj_episodes, args.n_stat_episodes)
    print(f"Collecting data for {n_total} episodes (same seeds for both approaches)...")

    all_baseline = []
    all_filtered = []

    for ep in range(n_total):
        seed_ep = args.seed + ep

        # Baseline (no filter)
        base_data = collect_episode_data(
            model, env, pe_env, safety_filter=None,
            seed=seed_ep, arena_half_w=arena_half, arena_half_h=arena_half,
        )
        all_baseline.append(base_data)

        # Baseline + Filter (same seed for same obstacles/positions)
        filt_data = collect_episode_data(
            model, env, pe_env, safety_filter=safety_filter,
            seed=seed_ep, arena_half_w=arena_half, arena_half_h=arena_half,
        )
        all_filtered.append(filt_data)

        if (ep + 1) % 20 == 0 or ep + 1 == n_total:
            b_cap = np.mean([d["captured"] for d in all_baseline]) * 100
            f_cap = np.mean([d["captured"] for d in all_filtered]) * 100
            print(f"  [{ep + 1}/{n_total}] baseline={b_cap:.0f}% capture, "
                  f"filtered={f_cap:.0f}% capture")

    # --- Generate plots ---
    print("\nGenerating plots...")
    generated = []

    # Detailed per-episode plots
    for i in range(args.n_traj_episodes):
        generated.append(plot_trajectories(all_baseline[i], all_filtered[i], args.output_dir, i))
        generated.append(plot_distance(all_baseline[i], all_filtered[i], args.output_dir, i))
        generated.append(plot_cbf_margins(all_baseline[i], all_filtered[i], args.output_dir, i))
        generated.append(plot_actions(all_baseline[i], all_filtered[i], args.output_dir, i))

    # Multi-episode overview
    generated.append(plot_multi_trajectory_overview(
        all_baseline, all_filtered, args.output_dir, n_show=min(args.n_traj_episodes, 8)
    ))

    # Statistical plots (using all episodes)
    generated.append(plot_statistics(all_baseline, all_filtered, args.output_dir))
    generated.append(plot_reward_distribution(all_baseline, all_filtered, args.output_dir))

    print(f"\nGenerated {len(generated)} plots:")
    for p in generated:
        print(f"  {p}")

    # Print summary
    n = len(all_baseline)
    base_cap = np.mean([d["captured"] for d in all_baseline]) * 100
    filt_cap = np.mean([d["captured"] for d in all_filtered]) * 100
    base_rew = np.mean([d["episode_reward"] for d in all_baseline])
    filt_rew = np.mean([d["episode_reward"] for d in all_filtered])

    print(f"\n{'='*50}")
    print(f"SUMMARY ({n} episodes)")
    print(f"{'='*50}")
    print(f"{'Metric':<25} {'Baseline':>12} {'Base+Filter':>12}")
    print(f"{'-'*50}")
    print(f"{'Capture Rate':<25} {base_cap:>11.1f}% {filt_cap:>11.1f}%")
    print(f"{'Mean Reward':<25} {base_rew:>12.1f} {filt_rew:>12.1f}")

    env.close()


if __name__ == "__main__":
    main()
