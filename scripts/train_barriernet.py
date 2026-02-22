"""Train BarrierNet PPO agent on the pursuit-evasion environment.

Usage:
    python scripts/train_barriernet.py
    python scripts/train_barriernet.py --timesteps 50000 --obstacles 2
    python scripts/train_barriernet.py --seed 123 --hidden-dim 256
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from agents.barriernet_ppo import BarrierNetPPO, BarrierNetPPOConfig
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper
from training.barriernet_trainer import BarrierNetTrainer, BarrierNetTrainerConfig


def make_env(
    n_obstacles: int = 2,
    arena_size: float = 20.0,
    dt: float = 0.05,
    max_steps: int = 600,
    seed: int = 42,
) -> SingleAgentPEWrapper:
    """Create the PE environment with SingleAgentPEWrapper."""
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
    parser = argparse.ArgumentParser(description="Train BarrierNet PPO on PE")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--obstacles", type=int, default=2, help="Number of obstacles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--rollout-length", type=int, default=1024, help="Steps per rollout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="checkpoints/barriernet", help="Save dir")
    parser.add_argument("--arena-size", type=float, default=20.0, help="Arena size")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    env = make_env(
        n_obstacles=args.obstacles,
        arena_size=args.arena_size,
        seed=args.seed,
    )

    # Determine obs_dim
    obs_dim = env.observation_space.shape[0]
    n_constraints_max = 4 + args.obstacles + 1 + 1  # arena + obs + collision + slack

    print(f"BarrierNet PPO Training")
    print(f"  obs_dim={obs_dim}, n_constraints_max={n_constraints_max}")
    print(f"  obstacles={args.obstacles}, arena={args.arena_size}x{args.arena_size}")
    print(f"  timesteps={args.timesteps}, rollout_length={args.rollout_length}")
    print(f"  hidden_dim={args.hidden_dim}, lr={args.lr}")
    print(f"  device={device}")

    # Create agent
    ppo_config = BarrierNetPPOConfig(
        obs_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        n_constraints_max=n_constraints_max,
        lr_actor=args.lr,
        lr_critic=args.lr,
        v_max=1.0,
        omega_max=2.84,
        arena_half_w=args.arena_size / 2,
        arena_half_h=args.arena_size / 2,
    )
    agent = BarrierNetPPO(ppo_config)
    agent.to(device)

    # Create trainer
    trainer_config = BarrierNetTrainerConfig(
        rollout_length=args.rollout_length,
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        seed=args.seed,
    )
    trainer = BarrierNetTrainer(env, agent, trainer_config)

    # Train
    print(f"\nStarting training...")
    metrics = trainer.train()

    # Summary
    summary = trainer.get_training_summary()
    print(f"\nTraining complete!")
    print(f"  Iterations: {summary.get('total_iterations', 0)}")
    print(f"  Episodes: {summary.get('total_episodes', 0)}")
    print(f"  Mean reward (last 10): {summary.get('mean_reward_last10', 0):.2f}")
    print(f"  Mean QP correction (last 10): {summary.get('mean_qp_correction_last10', 0):.4f}")
    print(f"  Mean intervention rate (last 10): {summary.get('mean_intervention_rate_last10', 0):.3f}")

    # Save final model
    final_path = os.path.join(args.save_dir, "barriernet_final.pt")
    os.makedirs(args.save_dir, exist_ok=True)
    agent.save(final_path)
    print(f"  Model saved to: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
