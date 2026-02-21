"""Smoke tests for the training pipeline.

These tests verify that training starts, runs a few steps, and produces
valid outputs. They do NOT test convergence (that requires longer runs).
"""

import numpy as np
import pytest
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper
from training.utils import make_pe_env, make_vec_env, setup_reproducibility


def _make_test_cfg():
    """Create a minimal Hydra-like config for testing."""
    return OmegaConf.create({
        "seed": 42,
        "total_timesteps": 2048,
        "eval_freq": 1024,
        "n_eval_episodes": 2,
        "save_freq": 1024,
        "n_envs": 2,
        "experiment_group": "test",
        "env": {
            "arena_width": 20.0,
            "arena_height": 20.0,
            "dt": 0.05,
            "max_steps": 200,
            "capture_radius": 0.5,
            "collision_radius": 0.3,
            "robot_radius": 0.15,
            "min_init_distance": 3.0,
            "max_init_distance": 15.0,
            "pursuer": {"v_max": 1.0, "omega_max": 2.84},
            "evader": {"v_max": 1.0, "omega_max": 2.84},
            "reward": {
                "capture_bonus": 100.0,
                "timeout_penalty": -50.0,
                "distance_scale": 1.0,
            },
        },
        "algorithm": {
            "learning_rate": 3e-4,
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": [64, 64]},
        },
        "wandb": {
            "entity": None,
            "project": "test",
            "mode": "disabled",
            "sync_tensorboard": False,
            "save_code": False,
            "tags": ["test"],
            "log_frequency": 128,
            "video": {"enabled": False, "record_every_n_episodes": 100, "fps": 20},
            "model": {"save_freq": 1024, "save_best": False},
        },
    })


class TestTrainingSmoke:
    """Smoke tests for the training pipeline."""

    def test_make_pe_env(self):
        """Environment factory creates valid env."""
        cfg = _make_test_cfg()
        env = make_pe_env(cfg, role="pursuer")
        obs, info = env.reset(seed=42)
        assert obs.shape == (14,)
        env.close()

    def test_make_vec_env(self):
        """Vectorized env factory works."""
        cfg = _make_test_cfg()
        envs = make_vec_env(cfg, n_envs=2, role="pursuer", seed=42)
        obs = envs.reset()
        assert obs.shape == (2, 14)
        envs.close()

    def test_ppo_trains_pursuer(self):
        """PPO trains pursuer for a small number of steps without error."""
        cfg = _make_test_cfg()
        setup_reproducibility(cfg.seed)

        envs = make_vec_env(cfg, n_envs=2, role="pursuer", seed=cfg.seed)

        import torch
        algo_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
        policy_kwargs = algo_cfg.pop("policy_kwargs", {})
        policy_kwargs["net_arch"] = list(policy_kwargs["net_arch"])
        policy_kwargs["activation_fn"] = torch.nn.Tanh

        model = PPO(
            "MlpPolicy", envs, verbose=0, seed=cfg.seed,
            policy_kwargs=policy_kwargs, **algo_cfg,
        )
        model.learn(total_timesteps=512)

        # Model should be able to predict
        obs, _ = make_pe_env(cfg, role="pursuer").reset(seed=0)
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (2,)

        envs.close()

    def test_ppo_trains_evader(self):
        """PPO trains evader for a small number of steps without error."""
        cfg = _make_test_cfg()
        setup_reproducibility(cfg.seed)

        envs = make_vec_env(cfg, n_envs=2, role="evader", seed=cfg.seed)

        import torch
        algo_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
        policy_kwargs = algo_cfg.pop("policy_kwargs", {})
        policy_kwargs["net_arch"] = list(policy_kwargs["net_arch"])
        policy_kwargs["activation_fn"] = torch.nn.Tanh

        model = PPO(
            "MlpPolicy", envs, verbose=0, seed=cfg.seed,
            policy_kwargs=policy_kwargs, **algo_cfg,
        )
        model.learn(total_timesteps=512)

        obs, _ = make_pe_env(cfg, role="evader").reset(seed=0)
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (2,)

        envs.close()

    def test_reproducibility_same_seed(self):
        """Same seed produces identical first observations."""
        cfg = _make_test_cfg()
        env1 = make_pe_env(cfg, role="pursuer")
        env2 = make_pe_env(cfg, role="pursuer")

        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

        env1.close()
        env2.close()

    def test_metrics_callback(self):
        """PursuitEvasionMetricsCallback runs without error."""
        from training.tracking import PursuitEvasionMetricsCallback

        cfg = _make_test_cfg()
        envs = make_vec_env(cfg, n_envs=2, role="pursuer", seed=42)

        import torch
        algo_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
        policy_kwargs = algo_cfg.pop("policy_kwargs", {})
        policy_kwargs["net_arch"] = list(policy_kwargs["net_arch"])
        policy_kwargs["activation_fn"] = torch.nn.Tanh

        model = PPO(
            "MlpPolicy", envs, verbose=0, seed=42,
            policy_kwargs=policy_kwargs, **algo_cfg,
        )

        cb = PursuitEvasionMetricsCallback(log_frequency=128)
        model.learn(total_timesteps=512, callback=cb)

        envs.close()
