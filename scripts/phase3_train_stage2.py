"""Stage 2: Single-Agent Validation — PPO with partial obs vs random evader.

Runs three configurations:
  2A: PPO + BiMDN + DCBF + partial obs  (full pipeline)
  2B: PPO + MLP encoder + DCBF          (BiMDN ablation)
  2C: PPO + BiMDN + no DCBF             (DCBF ablation)

Usage:
    # Run all three configs
    ./venv/bin/python scripts/phase3_train_stage2.py --run all --steps 500000

    # Run a single config
    ./venv/bin/python scripts/phase3_train_stage2.py --run 2A --steps 500000

    # Quick test
    ./venv/bin/python scripts/phase3_train_stage2.py --run 2A --steps 10000
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from agents.partial_obs_policy import PartialObsFeaturesExtractor
from envs.dcbf_action_wrapper import DCBFActionWrapper
from envs.partial_obs_wrapper import PartialObsWrapper
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper


class Phase3MetricsCallback(BaseCallback):
    """Log Phase 3-specific metrics during training."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_rewards = []
        self._episode_lengths = []
        self._captures = 0
        self._timeouts = 0
        self._dcbf_interventions = []
        self._episodes = 0

    def _on_step(self) -> bool:
        # Check for episode completions
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                infos = self.locals.get("infos", [{}])
                info = infos[i] if i < len(infos) else {}
                self._episodes += 1

                if info.get("is_capture", False):
                    self._captures += 1
                else:
                    self._timeouts += 1

                # DCBF metrics (if available)
                if "dcbf_intervention_rate" in info:
                    self._dcbf_interventions.append(
                        info["dcbf_intervention_rate"]
                    )

        # Log periodically
        if self.num_timesteps % self.log_freq == 0 and self._episodes > 0:
            cr = self._captures / self._episodes
            self.logger.record("phase3/capture_rate", cr)
            self.logger.record("phase3/episodes", self._episodes)

            if self._dcbf_interventions:
                avg_interv = np.mean(self._dcbf_interventions[-100:])
                self.logger.record("phase3/dcbf_intervention_rate", avg_interv)

        return True


def make_phase3_env(
    encoder_type: str = "bimdn",
    use_dcbf: bool = True,
    gamma: float = 0.2,
    history_length: int = 10,
    bimdn_weights_path: str | None = None,
    seed: int = 42,
    n_obstacles: int = 0,
):
    """Create a Phase 3 environment with partial obs + optional DCBF."""
    base_env = PursuitEvasionEnv(n_obstacles=n_obstacles)
    single_env = SingleAgentPEWrapper(base_env, role="pursuer")
    partial_env = PartialObsWrapper(
        single_env, role="pursuer", history_length=history_length,
    )

    if use_dcbf:
        env = DCBFActionWrapper(partial_env, role="pursuer", gamma=gamma)
    else:
        env = partial_env

    env.reset(seed=seed)
    return env


def make_vec_phase3_env(n_envs, seed=42, **kwargs):
    """Create vectorized Phase 3 environments."""
    def _make(env_seed):
        def _init():
            return make_phase3_env(seed=env_seed, **kwargs)
        return _init

    envs = DummyVecEnv([_make(seed + i) for i in range(n_envs)])
    envs = VecMonitor(envs)
    return envs


def get_policy_kwargs(encoder_type: str, history_length: int = 10):
    """Get SB3 policy kwargs for the given encoder type."""
    encoder_kwargs = {}
    if encoder_type == "mlp":
        encoder_kwargs["history_length"] = history_length

    return {
        "features_extractor_class": PartialObsFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "encoder_type": encoder_type,
            "encoder_kwargs": encoder_kwargs if encoder_kwargs else None,
        },
        "net_arch": [256, 256],
        "activation_fn": torch.nn.Tanh,
    }


def run_config(
    config_name: str,
    encoder_type: str,
    use_dcbf: bool,
    total_timesteps: int,
    seed: int,
    output_dir: str,
    n_envs: int = 4,
    gamma: float = 0.2,
    bimdn_weights_path: str | None = None,
    n_obstacles: int = 0,
):
    """Run a single training configuration."""
    print(f"\n{'='*60}")
    print(f"Running config {config_name}")
    print(f"  encoder: {encoder_type}, dcbf: {use_dcbf}, "
          f"steps: {total_timesteps}, seed: {seed}")
    print(f"{'='*60}")

    run_dir = os.path.join(output_dir, f"run_{config_name}")
    os.makedirs(run_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create environments
    env_kwargs = dict(
        encoder_type=encoder_type,
        use_dcbf=use_dcbf,
        gamma=gamma,
        bimdn_weights_path=bimdn_weights_path,
        n_obstacles=n_obstacles,
    )
    train_env = make_vec_phase3_env(n_envs, seed=seed, **env_kwargs)
    eval_env = make_phase3_env(seed=seed + 1000, **env_kwargs)

    # Create PPO
    policy_kwargs = get_policy_kwargs(encoder_type)
    model = PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=os.path.join(run_dir, "tb_logs"),
        seed=seed,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    # Callbacks
    callbacks = CallbackList([
        Phase3MetricsCallback(log_freq=2000),
        EvalCallback(
            eval_env,
            eval_freq=max(10000 // n_envs, 1),
            n_eval_episodes=20,
            best_model_save_path=os.path.join(run_dir, "best"),
            log_path=os.path.join(run_dir, "eval_logs"),
            deterministic=True,
        ),
    ])

    # Train
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    train_time = time.time() - t0

    # Save final model
    model.save(os.path.join(run_dir, "final_model"))

    # Save config
    config = {
        "config_name": config_name,
        "encoder_type": encoder_type,
        "use_dcbf": use_dcbf,
        "gamma": gamma,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "n_envs": n_envs,
        "train_time_seconds": train_time,
        "device": device,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfig {config_name} done in {train_time:.1f}s")
    print(f"Model saved to {run_dir}")

    train_env.close()
    eval_env.close()

    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Single-Agent Training")
    parser.add_argument("--run", type=str, default="all",
                        choices=["all", "2A", "2B", "2C"],
                        help="Which config to run")
    parser.add_argument("--steps", type=int, default=500000,
                        help="Total training timesteps per run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="DCBF gamma")
    parser.add_argument("--bimdn_weights", type=str, default=None,
                        help="Path to pre-trained BiMDN weights (Stage 1)")
    parser.add_argument("--n_obstacles", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/stage2")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    configs = {
        "2A": {"encoder_type": "bimdn", "use_dcbf": True},
        "2B": {"encoder_type": "mlp",   "use_dcbf": True},
        "2C": {"encoder_type": "bimdn", "use_dcbf": False},
    }

    runs_to_do = list(configs.keys()) if args.run == "all" else [args.run]

    t0 = time.time()
    run_dirs = {}
    for name in runs_to_do:
        cfg = configs[name]
        run_dirs[name] = run_config(
            config_name=name,
            total_timesteps=args.steps,
            seed=args.seed,
            output_dir=args.output,
            n_envs=args.n_envs,
            gamma=args.gamma,
            bimdn_weights_path=args.bimdn_weights,
            n_obstacles=args.n_obstacles,
            **cfg,
        )

    print(f"\n{'='*60}")
    print(f"Stage 2 complete. Total time: {time.time()-t0:.1f}s")
    for name, d in run_dirs.items():
        print(f"  {name}: {d}")
    print(f"\nNext: ./venv/bin/python scripts/phase3_evaluate_stage2.py "
          f"--input {args.output}")


if __name__ == "__main__":
    main()
