"""Sweep VCP-CBF filter parameters to find optimal settings.

Tests alpha, epsilon, and DCBF mode across multiple episodes.
Generates comparison table and plots.

Usage:
    python scripts/sweep_filter_params.py \
        --model-path models/obstacle_ppo_42/final_model.zip \
        --n-episodes 100
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper
from evaluation.comparison_framework import compute_cbf_margins
from safety.vcp_cbf import VCPCBFFilter
from stable_baselines3 import PPO


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


def run_evaluation(model, env, pe_env, safety_filter, n_episodes, seed,
                   arena_half_w=10.0, arena_half_h=10.0, use_dcbf=False,
                   dcbf_gamma=0.1, dcbf_dt=0.05):
    """Run episodes and collect aggregate metrics."""
    expected_obs_dim = model.observation_space.shape[0]

    n_captures = 0
    n_violations = 0
    n_interventions = 0
    total_steps = 0
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False

        while not done:
            p_state = pe_env.pursuer_state.copy()
            e_state = pe_env.evader_state.copy()
            obstacles = pe_env.obstacles

            model_obs = obs[:expected_obs_dim] if len(obs) > expected_obs_dim else obs
            action, _ = model.predict(model_obs, deterministic=True)

            if safety_filter is not None:
                action_pre = action.copy()
                if use_dcbf:
                    action, finfo = safety_filter.dcbf_filter_action(
                        action, p_state, dt=dcbf_dt, gamma=dcbf_gamma,
                        obstacles=obstacles, opponent_state=e_state,
                    )
                else:
                    action, finfo = safety_filter.filter_action(
                        action, p_state,
                        obstacles=obstacles, opponent_state=e_state,
                    )
                correction = float(np.linalg.norm(action - action_pre))
                if correction > 0.01:
                    n_interventions += 1

            # Check CBF margins
            margins = compute_cbf_margins(
                p_state, obstacles, e_state,
                arena_half_w, arena_half_h,
                d=0.1, alpha=1.0, robot_radius=0.15, r_min=0.35,
            )
            if margins and min(margins) < -1e-4:
                n_violations += 1

            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            total_steps += 1

        episode_rewards.append(ep_reward)
        ep_metrics = step_info.get("episode_metrics", {})
        episode_lengths.append(ep_metrics.get("episode_length", 0))
        if ep_metrics.get("captured", False):
            n_captures += 1

    return {
        "capture_rate": n_captures / n_episodes * 100,
        "violation_rate": n_violations / max(total_steps, 1) * 100,
        "intervention_rate": n_interventions / max(total_steps, 1) * 100,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "total_steps": total_steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep CBF filter parameters")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/filter_sweep")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    arena_half = 10.0

    print(f"Loading model: {args.model_path}")
    model = PPO.load(args.model_path)

    env = make_eval_env()
    pe_env = env
    while hasattr(pe_env, "env"):
        if hasattr(pe_env, "pursuer_state"):
            break
        pe_env = pe_env.env

    # Define configurations to sweep
    configs = []

    # 1. No filter (baseline)
    configs.append({
        "name": "No Filter",
        "alpha": None, "epsilon": 0.0, "dcbf": False,
    })

    # 2. Alpha sweep with continuous-time CBF
    for alpha in [0.1, 0.3, 0.5, 0.7, 1.0]:
        configs.append({
            "name": f"CT alpha={alpha}",
            "alpha": alpha, "epsilon": 0.0, "dcbf": False,
        })

    # 3. Best alpha candidates + epsilon margin
    for alpha in [0.3, 0.5]:
        for epsilon in [0.02, 0.05]:
            configs.append({
                "name": f"CT a={alpha} e={epsilon}",
                "alpha": alpha, "epsilon": epsilon, "dcbf": False,
            })

    # 4. DCBF with different gamma values
    for gamma in [0.05, 0.1, 0.2, 0.5]:
        configs.append({
            "name": f"DCBF gamma={gamma}",
            "alpha": 1.0, "epsilon": 0.0, "dcbf": True, "gamma": gamma,
        })

    # Run all configurations
    results = []
    for i, cfg in enumerate(configs):
        name = cfg["name"]
        print(f"\n[{i+1}/{len(configs)}] {name}")

        if cfg["alpha"] is None:
            # No filter
            sf = None
            use_dcbf = False
        else:
            sf = VCPCBFFilter(
                d=0.1, alpha=cfg["alpha"], v_max=1.0, omega_max=2.84,
                robot_radius=0.15, arena_half_w=arena_half, arena_half_h=arena_half,
                epsilon=cfg["epsilon"],
            )
            use_dcbf = cfg.get("dcbf", False)

        t0 = time.time()
        r = run_evaluation(
            model, env, pe_env, sf,
            n_episodes=args.n_episodes, seed=args.seed,
            arena_half_w=arena_half, arena_half_h=arena_half,
            use_dcbf=use_dcbf,
            dcbf_gamma=cfg.get("gamma", 0.1),
        )
        elapsed = time.time() - t0
        r["name"] = name
        r["elapsed"] = elapsed
        r["config"] = cfg
        results.append(r)

        print(f"  capture={r['capture_rate']:.1f}%, viol={r['violation_rate']:.2f}%, "
              f"interv={r['intervention_rate']:.1f}%, reward={r['mean_reward']:.1f}, "
              f"time={elapsed:.1f}s")

    # Print comparison table
    print(f"\n{'='*100}")
    print(f"PARAMETER SWEEP RESULTS ({args.n_episodes} episodes each)")
    print(f"{'='*100}")
    header = f"{'Configuration':<25} {'Capture%':>9} {'Viol%':>8} {'Interv%':>9} {'Reward':>8} {'Steps':>7} {'Time':>6}"
    print(header)
    print("-" * 100)

    for r in results:
        print(f"{r['name']:<25} {r['capture_rate']:>8.1f}% {r['violation_rate']:>7.2f}% "
              f"{r['intervention_rate']:>8.1f}% {r['mean_reward']:>8.1f} {r['mean_length']:>7.0f} "
              f"{r['elapsed']:>5.1f}s")

    # Save results to text file
    results_path = os.path.join(args.output_dir, "sweep_results.txt")
    with open(results_path, "w") as f:
        f.write(f"VCP-CBF Filter Parameter Sweep Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Episodes per config: {args.n_episodes}\n")
        f.write(f"Seed: {args.seed}\n\n")

        f.write(f"{'Configuration':<25} {'Capture%':>9} {'Viol%':>8} {'Interv%':>9} "
                f"{'Reward':>8} {'MeanLen':>8}\n")
        f.write("-" * 80 + "\n")

        for r in results:
            f.write(f"{r['name']:<25} {r['capture_rate']:>8.1f}% {r['violation_rate']:>7.2f}% "
                    f"{r['intervention_rate']:>8.1f}% {r['mean_reward']:>8.1f} "
                    f"{r['mean_length']:>8.0f}\n")

    print(f"\nResults saved to: {results_path}")

    # --- Generate plots ---
    names = [r["name"] for r in results]
    capture_rates = [r["capture_rate"] for r in results]
    violation_rates = [r["violation_rate"] for r in results]
    intervention_rates = [r["intervention_rate"] for r in results]
    mean_rewards = [r["mean_reward"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Bar colors: baseline=gray, CT=blue shades, CT+eps=green, DCBF=orange
    colors = []
    for r in results:
        cfg = r["config"]
        if cfg["alpha"] is None:
            colors.append("#888888")
        elif cfg.get("dcbf"):
            colors.append("#ff7f0e")
        elif cfg["epsilon"] > 0:
            colors.append("#2ca02c")
        else:
            colors.append("#1f77b4")

    x = np.arange(len(names))

    # 1. Capture Rate
    ax = axes[0, 0]
    bars = ax.bar(x, capture_rates, color=colors)
    ax.set_ylabel("Capture Rate (%)")
    ax.set_title("Capture Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(80, 105)
    for bar, val in zip(bars, capture_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.0f}", ha="center", fontsize=7)

    # 2. Violation Rate
    ax = axes[0, 1]
    bars = ax.bar(x, violation_rates, color=colors)
    ax.set_ylabel("CBF Violation Rate (%)")
    ax.set_title("Safety Violations")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    for bar, val in zip(bars, violation_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", fontsize=7)

    # 3. Intervention Rate
    ax = axes[1, 0]
    bars = ax.bar(x, intervention_rates, color=colors)
    ax.set_ylabel("Intervention Rate (%)")
    ax.set_title("Filter Intervention Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    for bar, val in zip(bars, intervention_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontsize=7)

    # 4. Mean Reward
    ax = axes[1, 1]
    bars = ax.bar(x, mean_rewards, color=colors)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Mean Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    for bar, val in zip(bars, mean_rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}", ha="center", fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#888888", label="No Filter"),
        Patch(facecolor="#1f77b4", label="Continuous-Time CBF"),
        Patch(facecolor="#2ca02c", label="CT + Epsilon Margin"),
        Patch(facecolor="#ff7f0e", label="Discrete-Time CBF"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"VCP-CBF Filter Parameter Sweep ({args.n_episodes} episodes)",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "sweep_comparison.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")

    # --- Pareto frontier plot: Capture Rate vs Safety ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for r, c in zip(results, colors):
        ax.scatter(r["intervention_rate"], r["capture_rate"],
                   c=c, s=100, zorder=3, edgecolors="black", linewidth=0.5)
        ax.annotate(r["name"], (r["intervention_rate"], r["capture_rate"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Intervention Rate (%)")
    ax.set_ylabel("Capture Rate (%)")
    ax.set_title("Capture Rate vs Filter Intervention (Pareto Frontier)")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements, fontsize=8)

    pareto_path = os.path.join(args.output_dir, "pareto_frontier.png")
    fig.savefig(pareto_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pareto plot saved to: {pareto_path}")

    env.close()


if __name__ == "__main__":
    main()
