"""Evaluate trained BarrierNet PPO and compare with baseline PPO.

Usage:
    python scripts/evaluate_barriernet.py --barriernet-path checkpoints/barriernet_ds10/barriernet_final.pt
    python scripts/evaluate_barriernet.py --barriernet-path checkpoints/barriernet_fixed/barriernet_final.pt --baseline-path models/local_42/final_model.zip
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from agents.barriernet_ppo import BarrierNetPPO
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper
from evaluation.comparison_framework import (
    BarrierNetEvalAgent,
    SB3EvalAgent,
    evaluate_approach,
)


def make_eval_env(
    n_obstacles: int = 2,
    arena_size: float = 20.0,
    dt: float = 0.05,
    max_steps: int = 600,
) -> SingleAgentPEWrapper:
    """Create evaluation environment (default reward scale)."""
    env = PursuitEvasionEnv(
        arena_width=arena_size,
        arena_height=arena_size,
        dt=dt,
        max_steps=max_steps,
        capture_radius=0.5,
        collision_radius=0.3,
        robot_radius=0.15,
        pursuer_v_max=1.0,
        pursuer_omega_max=2.84,
        evader_v_max=1.0,
        evader_omega_max=2.84,
        n_obstacles=n_obstacles,
        obstacle_radius_range=(0.3, 1.0),
        obstacle_margin=0.5,
        n_obstacle_obs=min(n_obstacles, 3),
    )
    return SingleAgentPEWrapper(env, role="pursuer", opponent_policy=None)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BarrierNet vs Baseline")
    parser.add_argument("--barriernet-path", type=str, required=True, help="BarrierNet checkpoint path")
    parser.add_argument("--baseline-path", type=str, default=None, help="SB3 baseline model path")
    parser.add_argument("--n-episodes", type=int, default=200, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed")
    parser.add_argument("--obstacles", type=int, default=2, help="Number of obstacles")
    parser.add_argument("--arena-size", type=float, default=20.0, help="Arena size")
    parser.add_argument("--output-dir", type=str, default="results/phase2_5", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create evaluation environment
    env = make_eval_env(n_obstacles=args.obstacles, arena_size=args.arena_size)
    arena_half = args.arena_size / 2

    # --- Evaluate BarrierNet ---
    print(f"\n{'='*60}")
    print(f"Evaluating BarrierNet PPO ({args.barriernet_path})")
    print(f"{'='*60}")

    bn_agent = BarrierNetPPO.load(args.barriernet_path)
    bn_eval = BarrierNetEvalAgent(bn_agent)
    bn_result = evaluate_approach(
        bn_eval, env,
        n_episodes=args.n_episodes,
        approach_name="BarrierNet PPO",
        arena_half_w=arena_half,
        arena_half_h=arena_half,
        seed=args.seed,
    )

    print(f"\nBarrierNet Results:")
    print(f"  Capture rate: {bn_result.capture_rate:.1%}")
    print(f"  Mean reward: {bn_result.mean_episode_reward:.2f}")
    print(f"  Safety violation rate: {bn_result.safety_violation_rate:.4%}")
    print(f"  Mean QP correction: {bn_result.mean_qp_correction:.4f}")
    print(f"  Intervention rate: {bn_result.intervention_rate:.1%}")
    print(f"  Mean inference time: {bn_result.mean_inference_time_ms:.2f} ms")

    # --- Evaluate Baseline PPO (if provided) ---
    baseline_result = None
    if args.baseline_path and os.path.exists(args.baseline_path):
        print(f"\n{'='*60}")
        print(f"Evaluating Baseline PPO ({args.baseline_path})")
        print(f"{'='*60}")

        try:
            from stable_baselines3 import PPO
            baseline_model = PPO.load(args.baseline_path)
            baseline_eval = SB3EvalAgent(baseline_model)
            baseline_result = evaluate_approach(
                baseline_eval, env,
                n_episodes=args.n_episodes,
                approach_name="Baseline PPO",
                arena_half_w=arena_half,
                arena_half_h=arena_half,
                seed=args.seed,
            )

            print(f"\nBaseline PPO Results:")
            print(f"  Capture rate: {baseline_result.capture_rate:.1%}")
            print(f"  Mean reward: {baseline_result.mean_episode_reward:.2f}")
            print(f"  Safety violation rate: {baseline_result.safety_violation_rate:.4%}")
            print(f"  Mean inference time: {baseline_result.mean_inference_time_ms:.2f} ms")
        except ImportError:
            print("  SB3 not available, skipping baseline evaluation")

    # --- Comparison Summary ---
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'BarrierNet':>15}", end="")
    if baseline_result:
        print(f" {'Baseline PPO':>15}", end="")
    print()
    print("-" * 60)

    metrics = [
        ("Capture Rate", f"{bn_result.capture_rate:.1%}",
         f"{baseline_result.capture_rate:.1%}" if baseline_result else None),
        ("Mean Reward", f"{bn_result.mean_episode_reward:.2f}",
         f"{baseline_result.mean_episode_reward:.2f}" if baseline_result else None),
        ("Safety Violations", f"{bn_result.safety_violation_rate:.4%}",
         f"{baseline_result.safety_violation_rate:.4%}" if baseline_result else None),
        ("QP Correction", f"{bn_result.mean_qp_correction:.4f}",
         f"{baseline_result.mean_qp_correction:.4f}" if baseline_result else None),
        ("Inference Time (ms)", f"{bn_result.mean_inference_time_ms:.2f}",
         f"{baseline_result.mean_inference_time_ms:.2f}" if baseline_result else None),
    ]

    for name, bn_val, base_val in metrics:
        print(f"{name:<30} {bn_val:>15}", end="")
        if base_val:
            print(f" {base_val:>15}", end="")
        print()

    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_path, "w") as f:
        f.write("Phase 2.5 BarrierNet Evaluation Results\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"BarrierNet checkpoint: {args.barriernet_path}\n")
        f.write(f"Episodes: {args.n_episodes}\n\n")

        f.write(f"BarrierNet PPO:\n")
        f.write(f"  Capture rate: {bn_result.capture_rate:.1%}\n")
        f.write(f"  Mean reward: {bn_result.mean_episode_reward:.2f}\n")
        f.write(f"  Safety violation rate: {bn_result.safety_violation_rate:.4%}\n")
        f.write(f"  Mean QP correction: {bn_result.mean_qp_correction:.4f}\n")
        f.write(f"  Intervention rate: {bn_result.intervention_rate:.1%}\n")
        f.write(f"  Mean inference time: {bn_result.mean_inference_time_ms:.2f} ms\n")

        if baseline_result:
            f.write(f"\nBaseline PPO:\n")
            f.write(f"  Capture rate: {baseline_result.capture_rate:.1%}\n")
            f.write(f"  Mean reward: {baseline_result.mean_episode_reward:.2f}\n")
            f.write(f"  Safety violation rate: {baseline_result.safety_violation_rate:.4%}\n")
            f.write(f"  Mean inference time: {baseline_result.mean_inference_time_ms:.2f} ms\n")

    print(f"\nResults saved to: {results_path}")
    env.close()


if __name__ == "__main__":
    main()
