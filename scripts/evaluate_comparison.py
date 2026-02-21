"""Comparative evaluation: BarrierNet vs CBF-Beta for pursuit-evasion.

Runs 3 evaluation configurations:
1. CBF-Beta under training conditions (no external filter)
2. CBF-Beta with RCBF-QP deployment filter
3. BarrierNet end-to-end (same in training and deployment)

Produces comparison table, statistical tests, and saves results.

Usage:
    python scripts/evaluate_comparison.py \
        --barriernet checkpoints/barriernet/barriernet_final.pt \
        --cbf-beta models/selfplay_42/final/pursuer.zip \
        --n-episodes 200

    # BarrierNet only (if CBF-Beta not yet trained):
    python scripts/evaluate_comparison.py \
        --barriernet checkpoints/barriernet/barriernet_final.pt \
        --barriernet-only --n-episodes 100
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper
from evaluation.comparison_framework import (
    BarrierNetEvalAgent,
    SB3EvalAgent,
    EvaluationResult,
    evaluate_approach,
    compute_comparison,
    format_comparison_table,
)


def make_eval_env(
    n_obstacles: int = 2,
    arena_size: float = 20.0,
    max_steps: int = 600,
) -> SingleAgentPEWrapper:
    """Create evaluation environment."""
    env = PursuitEvasionEnv(
        arena_width=arena_size,
        arena_height=arena_size,
        dt=0.05,
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


def evaluate_barriernet(
    model_path: str,
    n_episodes: int,
    n_obstacles: int,
    arena_size: float,
    seed: int,
    training_hours: float = 0.0,
) -> EvaluationResult:
    """Evaluate BarrierNet agent."""
    from agents.barriernet_ppo import BarrierNetPPO

    agent = BarrierNetPPO.load(model_path)
    eval_agent = BarrierNetEvalAgent(agent)
    env = make_eval_env(n_obstacles=n_obstacles, arena_size=arena_size)

    print(f"\nEvaluating BarrierNet ({model_path})")
    result = evaluate_approach(
        eval_agent, env,
        n_episodes=n_episodes,
        approach_name="BarrierNet E2E",
        arena_half_w=arena_size / 2,
        arena_half_h=arena_size / 2,
        training_wall_clock_hours=training_hours,
        seed=seed,
    )
    env.close()
    return result


def evaluate_cbf_beta(
    model_path: str,
    n_episodes: int,
    n_obstacles: int,
    arena_size: float,
    seed: int,
    with_filter: bool = False,
    training_hours: float = 0.0,
) -> EvaluationResult:
    """Evaluate CBF-Beta agent, optionally with RCBF-QP filter."""
    from stable_baselines3 import PPO
    from safety.vcp_cbf import VCPCBFFilter

    model = PPO.load(model_path)
    safety_filter = None
    if with_filter:
        safety_filter = VCPCBFFilter(
            v_max=1.0,
            omega_max=2.84,
            alpha=1.0,
            d=0.1,
            arena_half_w=arena_size / 2,
            arena_half_h=arena_size / 2,
            robot_radius=0.15,
            r_min_separation=0.35,
        )

    eval_agent = SB3EvalAgent(model, safety_filter=safety_filter)
    env = make_eval_env(n_obstacles=n_obstacles, arena_size=arena_size)

    name = "CBF-Beta+RCBF-QP (Deploy)" if with_filter else "CBF-Beta (Train)"
    print(f"\nEvaluating {name} ({model_path})")
    result = evaluate_approach(
        eval_agent, env,
        n_episodes=n_episodes,
        approach_name=name,
        arena_half_w=arena_size / 2,
        arena_half_h=arena_size / 2,
        training_wall_clock_hours=training_hours,
        seed=seed,
    )
    env.close()
    return result


def save_result(result: EvaluationResult, path: str):
    """Save evaluation result to JSON."""
    data = {
        "approach_name": result.approach_name,
        "n_episodes": result.n_episodes,
        "safety_violation_rate": result.safety_violation_rate,
        "mean_min_cbf_margin": result.mean_min_cbf_margin,
        "capture_rate": result.capture_rate,
        "mean_capture_time": result.mean_capture_time,
        "mean_episode_reward": result.mean_episode_reward,
        "mean_inference_time_ms": result.mean_inference_time_ms,
        "training_wall_clock_hours": result.training_wall_clock_hours,
        "qp_infeasibility_rate": result.qp_infeasibility_rate,
        "mean_qp_correction": result.mean_qp_correction,
        "intervention_rate": result.intervention_rate,
        "episode_rewards": result.episode_rewards.tolist(),
        "cbf_margin_values": result.cbf_margin_values.tolist(),
        "episode_lengths": result.episode_lengths.tolist(),
        "min_distances": result.min_distances.tolist(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="BarrierNet vs CBF-Beta Comparison")
    parser.add_argument("--barriernet", type=str, help="BarrierNet checkpoint path")
    parser.add_argument("--cbf-beta", type=str, help="CBF-Beta SB3 model path")
    parser.add_argument("--n-episodes", type=int, default=200, help="Evaluation episodes")
    parser.add_argument("--obstacles", type=int, default=2, help="Number of obstacles")
    parser.add_argument("--arena-size", type=float, default=20.0, help="Arena size")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed")
    parser.add_argument("--output-dir", type=str, default="results/phase2_5/comparison")
    parser.add_argument("--barriernet-only", action="store_true",
                        help="Only evaluate BarrierNet (skip CBF-Beta)")
    parser.add_argument("--barriernet-train-hours", type=float, default=0.0,
                        help="BarrierNet training wall-clock hours")
    parser.add_argument("--cbf-beta-train-hours", type=float, default=0.0,
                        help="CBF-Beta training wall-clock hours")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    # Evaluate BarrierNet
    bn_result = None
    if args.barriernet:
        bn_result = evaluate_barriernet(
            args.barriernet, args.n_episodes, args.obstacles,
            args.arena_size, args.seed,
            training_hours=args.barriernet_train_hours,
        )
        save_result(bn_result, os.path.join(args.output_dir, "barriernet_result.json"))

        print(f"\n{'='*60}")
        print(f"BarrierNet Results ({args.n_episodes} episodes):")
        print(f"  Capture rate:     {bn_result.capture_rate:.2%}")
        print(f"  Safety violations: {bn_result.safety_violation_rate:.4%}")
        print(f"  Mean CBF margin:  {bn_result.mean_min_cbf_margin:.4f}")
        print(f"  Mean reward:      {bn_result.mean_episode_reward:.2f}")
        print(f"  Inference time:   {bn_result.mean_inference_time_ms:.2f}ms")
        print(f"  QP correction:    {bn_result.mean_qp_correction:.4f}")
        print(f"  Intervention rate: {bn_result.intervention_rate:.2%}")
        print(f"  Infeasibility:    {bn_result.qp_infeasibility_rate:.4%}")

    if args.barriernet_only:
        print(f"\nBarrierNet-only evaluation complete in {time.time() - t_start:.1f}s")
        return

    # Evaluate CBF-Beta (training conditions)
    cb_train_result = None
    cb_deploy_result = None
    if args.cbf_beta:
        cb_train_result = evaluate_cbf_beta(
            args.cbf_beta, args.n_episodes, args.obstacles,
            args.arena_size, args.seed,
            with_filter=False,
            training_hours=args.cbf_beta_train_hours,
        )
        save_result(cb_train_result, os.path.join(args.output_dir, "cbf_beta_train_result.json"))

        # Evaluate CBF-Beta (deployment with RCBF-QP)
        cb_deploy_result = evaluate_cbf_beta(
            args.cbf_beta, args.n_episodes, args.obstacles,
            args.arena_size, args.seed,
            with_filter=True,
            training_hours=args.cbf_beta_train_hours,
        )
        save_result(cb_deploy_result, os.path.join(args.output_dir, "cbf_beta_deploy_result.json"))

    # Full comparison
    if bn_result and cb_train_result and cb_deploy_result:
        report = compute_comparison(bn_result, cb_train_result, cb_deploy_result)
        table = format_comparison_table(report)

        print(f"\n{'='*80}")
        print("COMPARISON RESULTS")
        print(f"{'='*80}")
        print(table)

        # Save report
        report_path = os.path.join(args.output_dir, "comparison_report.md")
        with open(report_path, "w") as f:
            f.write("# BarrierNet vs CBF-Beta Comparison Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Episodes: {args.n_episodes}, Obstacles: {args.obstacles}, ")
            f.write(f"Arena: {args.arena_size}x{args.arena_size}\n\n")
            f.write(table)
        print(f"\nReport saved to {report_path}")

    print(f"\nTotal evaluation time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
