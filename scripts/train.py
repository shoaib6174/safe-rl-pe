"""Main training entry point with Hydra config management.

Usage:
    # Default training (PPO pursuer vs random evader)
    python scripts/train.py

    # Override from CLI
    python scripts/train.py algorithm.learning_rate=1e-4 seed=123

    # Disable wandb for quick local tests
    python scripts/train.py wandb.mode=disabled

    # Multi-run hyperparameter sweep
    python scripts/train.py --multirun algorithm.learning_rate=1e-3,3e-4,1e-4 seed=0,1,2,3,4
"""

import sys
import os

# Add project root to path so imports work regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from training.tracking import (
    HParamCallback,
    PursuitEvasionMetricsCallback,
    init_tracking,
)
from training.utils import make_pe_env, make_vec_env, setup_reproducibility


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Training config:\n{OmegaConf.to_yaml(cfg)}")

    # Setup reproducibility
    use_cuda = torch.cuda.is_available()
    setup_reproducibility(cfg.seed, use_cuda=use_cuda)
    device = "cuda" if use_cuda else "cpu"
    print(f"Device: {device}")

    # Init tracking
    run = init_tracking(cfg)
    run_id = run.id if run is not None else f"local_{cfg.seed}"

    # Create training envs (vectorized)
    train_env = make_vec_env(
        cfg, n_envs=cfg.n_envs, role="pursuer",
        opponent_policy=None, seed=cfg.seed,
    )

    # Create eval env (single, for evaluation callbacks)
    eval_env = make_pe_env(cfg, role="pursuer", opponent_policy=None, render_mode=None)

    # Build PPO config from Hydra
    algo_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)

    # Convert net_arch list properly for SB3
    policy_kwargs = algo_cfg.pop("policy_kwargs", {})
    if "net_arch" in policy_kwargs:
        policy_kwargs["net_arch"] = list(policy_kwargs["net_arch"])
    policy_kwargs["activation_fn"] = torch.nn.Tanh

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        tensorboard_log=f"runs/{run_id}",
        seed=cfg.seed,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
        **algo_cfg,
    )

    # Setup callbacks
    callbacks = [
        PursuitEvasionMetricsCallback(log_frequency=cfg.wandb.log_frequency),
        HParamCallback(),
        EvalCallback(
            eval_env,
            eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
            n_eval_episodes=cfg.n_eval_episodes,
            best_model_save_path=f"models/{run_id}/best",
            log_path=f"models/{run_id}/eval_logs",
            deterministic=True,
        ),
    ]

    # Add wandb callback if tracking is active
    if run is not None:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(
            WandbCallback(
                model_save_path=f"models/{run_id}",
                model_save_freq=cfg.save_freq,
                verbose=2,
            ),
        )

    callback_list = CallbackList(callbacks)

    # Train
    print(f"Starting training: {cfg.total_timesteps} timesteps, seed={cfg.seed}")
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callback_list,
        progress_bar=True,
    )

    # Save final model
    model.save(f"models/{run_id}/final_model")
    print(f"Model saved to models/{run_id}/final_model")

    # Cleanup
    train_env.close()
    eval_env.close()
    if run is not None:
        run.finish()

    print("Training complete.")


if __name__ == "__main__":
    main()
