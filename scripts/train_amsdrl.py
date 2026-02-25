#!/usr/bin/env python3
"""AMS-DRL self-play training script for Phase 3.

Usage:
    # Stage 3 smoke test (3 phases, ~3-4h)
    ./venv/bin/python scripts/train_amsdrl.py \
        --max_phases 3 --timesteps_per_phase 200000 \
        --seed 42 --output results/stage3/smoke_test

    # Stage 4 full run (12 phases, ~8-15h)
    ./venv/bin/python scripts/train_amsdrl.py \
        --max_phases 12 --timesteps_per_phase 500000 \
        --eta 0.10 --seed 42 --output results/stage4/full_run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.amsdrl import AMSDRLSelfPlay
from training.ne_tools import analyze_convergence, plot_ne_convergence


def parse_args():
    parser = argparse.ArgumentParser(
        description="AMS-DRL self-play training for Phase 3"
    )

    # Self-play protocol
    parser.add_argument("--max_phases", type=int, default=12,
                        help="Maximum alternating phases (default: 12)")
    parser.add_argument("--timesteps_per_phase", type=int, default=500_000,
                        help="Training steps per phase (default: 500000)")
    parser.add_argument("--cold_start_timesteps", type=int, default=200_000,
                        help="Cold-start training steps for evader (default: 200000)")
    parser.add_argument("--eta", type=float, default=0.10,
                        help="NE convergence threshold (default: 0.10)")
    parser.add_argument("--eval_episodes", type=int, default=100,
                        help="Evaluation episodes per phase (default: 100)")

    # Environment
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--n_obstacles", type=int, default=0,
                        help="Number of obstacles (default: 0)")
    parser.add_argument("--arena_width", type=float, default=20.0,
                        help="Arena width in meters (default: 20.0)")
    parser.add_argument("--arena_height", type=float, default=20.0,
                        help="Arena height in meters (default: 20.0)")
    parser.add_argument("--max_steps", type=int, default=1200,
                        help="Max episode steps (default: 1200)")
    parser.add_argument("--capture_radius", type=float, default=0.5,
                        help="Capture radius in meters (default: 0.5)")

    # Reward tuning
    parser.add_argument("--distance_scale", type=float, default=1.0,
                        help="Dense distance reward scale factor (default: 1.0)")
    parser.add_argument("--pursuer_v_max", type=float, default=1.0,
                        help="Pursuer max linear velocity (default: 1.0)")
    parser.add_argument("--fixed_speed", action="store_true", default=False,
                        help="Fix v=v_max, only learn omega (1D action space)")

    # Curriculum
    parser.add_argument("--curriculum", action="store_true", default=False,
                        help="Enable 4-level curriculum (distance + obstacles)")

    # Opponent pool
    parser.add_argument("--opponent_pool_size", type=int, default=0,
                        help="Opponent pool size (0=disabled, e.g. 5 for diverse self-play)")

    # Reward shaping
    parser.add_argument("--w_occlusion", type=float, default=0.0,
                        help="Evader occlusion bonus weight (Mode A, e.g. 0.05). "
                             "Rewards evader for hiding behind obstacles.")
    parser.add_argument("--use_visibility_reward", action="store_true", default=False,
                        help="Enable visibility-based evader reward (Mode B, OpenAI H&S style). "
                             "Evader gets +1/-1 per step based on LOS occlusion.")
    parser.add_argument("--visibility_weight", type=float, default=1.0,
                        help="Scale for visibility reward signal (default: 1.0)")
    parser.add_argument("--survival_bonus", type=float, default=0.0,
                        help="Per-step survival bonus for evader (e.g. 1.0)")
    parser.add_argument("--timeout_penalty", type=float, default=-100.0,
                        help="Pursuer penalty on timeout / evader reward on escape (default: -100.0)")

    # Preparation phase
    parser.add_argument("--prep_steps", type=int, default=0,
                        help="Freeze pursuer for first N steps per episode (default: 0, off). "
                             "Gives evader time to reach obstacles before chase begins.")

    # Obstacle collision penalty
    parser.add_argument("--w_collision", type=float, default=0.0,
                        help="Obstacle collision penalty weight (default: 0.0, off)")

    # Wall collision penalty
    parser.add_argument("--w_wall", type=float, default=0.0,
                        help="Wall collision penalty weight (default: 0.0, off)")

    # PBRS obstacle-seeking
    parser.add_argument("--w_obs_approach", type=float, default=0.0,
                        help="PBRS obstacle-seeking weight for evader (default: 0.0, off). "
                             "Provides gradient toward nearest obstacle. Recommended: 50.0")

    # Asymmetric training
    parser.add_argument("--evader_multiplier", type=float, default=1.0,
                        help="Evader training step multiplier at obstacle levels (default: 1.0). "
                             "E.g. 3.0 gives evader 3x more steps when obstacles are present.")

    # Curriculum tuning
    parser.add_argument("--min_phases_per_level", type=int, default=1,
                        help="Minimum phases at each curriculum level before advancement (default: 1)")
    parser.add_argument("--min_escape_rate", type=float, default=0.0,
                        help="Minimum evader escape rate required for curriculum advancement (default: 0.0)")

    # L2 collapse countermeasures (S52)
    parser.add_argument("--bilateral_rollback", action="store_true", default=False,
                        help="Rollback both agents (not just active) on domination detection")
    parser.add_argument("--evader_first_on_advance", action="store_true", default=False,
                        help="Force evader training first when curriculum advances")
    parser.add_argument("--warm_start_evader", action="store_true", default=False,
                        help="Restore evader to best milestone on curriculum advance")
    parser.add_argument("--warm_start_timesteps", type=int, default=50000,
                        help="Timesteps for evader warm-start pre-training (default: 50000)")
    parser.add_argument("--mixed_level_ratio", type=float, default=0.0,
                        help="Fraction of envs using previous level distances (default: 0.0, off). "
                             "E.g. 0.2 uses 20%% prev-level envs to prevent forgetting.")
    parser.add_argument("--smooth_curriculum", action="store_true", default=False,
                        help="Use smooth (continuous) curriculum instead of discrete levels")
    parser.add_argument("--smooth_curriculum_increment", type=float, default=0.5,
                        help="Distance increment per advancement in smooth curriculum (default: 0.5)")

    # Tier 3: EWC (catastrophic forgetting prevention)
    parser.add_argument("--ewc_lambda", type=float, default=0.0,
                        help="EWC regularization strength (0=disabled, e.g. 1000.0)")
    parser.add_argument("--ewc_fisher_samples", type=int, default=1024,
                        help="Observations for Fisher info estimation (default: 1024)")

    # Tier 3: RND (intrinsic exploration motivation)
    parser.add_argument("--rnd_coef", type=float, default=0.0,
                        help="RND intrinsic reward coefficient (0=disabled, e.g. 0.1)")
    parser.add_argument("--rnd_embed_dim", type=int, default=64,
                        help="RND embedding dimension (default: 64)")
    parser.add_argument("--rnd_hidden_dim", type=int, default=128,
                        help="RND hidden layer width (default: 128)")
    parser.add_argument("--rnd_update_freq", type=int, default=256,
                        help="RND predictor training frequency in steps (default: 256)")

    # Safety
    parser.add_argument("--use_dcbf", action="store_true", default=True,
                        help="Use DCBF safety filter for pursuer (default: True)")
    parser.add_argument("--no_dcbf", dest="use_dcbf", action="store_false",
                        help="Disable DCBF safety filter")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="DCBF decay rate (default: 0.2)")

    # Observability
    parser.add_argument("--full_obs", action="store_true", default=False,
                        help="Use full observability (diagnostic mode, no partial obs)")

    # Model
    parser.add_argument("--encoder_type", type=str, default="bimdn",
                        choices=["bimdn", "lstm", "mlp"],
                        help="Belief encoder type (default: bimdn)")
    parser.add_argument("--history_length", type=int, default=10,
                        help="Observation history length K (default: 10)")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="PPO learning rate (default: 3e-4)")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--n_steps", type=int, default=512,
                        help="PPO rollout steps per env (default: 512)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="PPO mini-batch size (default: 256)")

    # Training
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="results/amsdrl",
                        help="Output directory (default: results/amsdrl)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create and run AMS-DRL
    amsdrl = AMSDRLSelfPlay(
        output_dir=args.output,
        max_phases=args.max_phases,
        timesteps_per_phase=args.timesteps_per_phase,
        cold_start_timesteps=args.cold_start_timesteps,
        eta=args.eta,
        eval_episodes=args.eval_episodes,
        n_envs=args.n_envs,
        use_dcbf=args.use_dcbf,
        gamma=args.gamma,
        encoder_type=args.encoder_type,
        history_length=args.history_length,
        n_obstacles=args.n_obstacles,
        arena_width=args.arena_width,
        arena_height=args.arena_height,
        max_steps=args.max_steps,
        capture_radius=args.capture_radius,
        seed=args.seed,
        device=device,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        full_obs=args.full_obs,
        distance_scale=args.distance_scale,
        pursuer_v_max=args.pursuer_v_max,
        fixed_speed=args.fixed_speed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        curriculum=args.curriculum,
        opponent_pool_size=args.opponent_pool_size,
        w_occlusion=args.w_occlusion,
        use_visibility_reward=args.use_visibility_reward,
        visibility_weight=args.visibility_weight,
        survival_bonus=args.survival_bonus,
        prep_steps=args.prep_steps,
        w_obs_approach=args.w_obs_approach,
        timeout_penalty=args.timeout_penalty,
        w_collision=args.w_collision,
        w_wall=args.w_wall,
        evader_training_multiplier=args.evader_multiplier,
        min_escape_rate=args.min_escape_rate,
        min_phases_per_level=args.min_phases_per_level,
        bilateral_rollback=args.bilateral_rollback,
        evader_first_on_advance=args.evader_first_on_advance,
        warm_start_evader=args.warm_start_evader,
        warm_start_timesteps=args.warm_start_timesteps,
        mixed_level_ratio=args.mixed_level_ratio,
        smooth_curriculum=args.smooth_curriculum,
        smooth_curriculum_increment=args.smooth_curriculum_increment,
        ewc_lambda=args.ewc_lambda,
        ewc_fisher_samples=args.ewc_fisher_samples,
        rnd_coef=args.rnd_coef,
        rnd_embed_dim=args.rnd_embed_dim,
        rnd_hidden_dim=args.rnd_hidden_dim,
        rnd_update_freq=args.rnd_update_freq,
    )

    result = amsdrl.run()

    # Analyze convergence
    print("\n" + "=" * 60)
    print("Convergence Analysis")
    print("=" * 60)

    analysis = analyze_convergence(result["history"], eta=args.eta)
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    # Save analysis
    with open(Path(args.output) / "convergence_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Plot convergence
    try:
        plot_path = Path(args.output) / "ne_convergence.png"
        plot_ne_convergence(
            result["history"],
            eta=args.eta,
            save_path=str(plot_path),
        )
        print(f"\n  NE convergence plot saved to: {plot_path}")
    except Exception as e:
        print(f"\n  Warning: Could not generate plot: {e}")

    # Save config
    config = vars(args)
    config["device"] = device
    with open(Path(args.output) / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAll results saved to: {args.output}")
    return 0 if result["converged"] else 1


if __name__ == "__main__":
    sys.exit(main())
