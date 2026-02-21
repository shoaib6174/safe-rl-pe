"""Training utilities: reproducibility, environment factory, etc."""

import os
import random

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper


def setup_reproducibility(seed: int, use_cuda: bool = False):
    """Full reproducibility setup. Call BEFORE creating envs or models."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_random_seed(seed, using_cuda=use_cuda)

    if use_cuda and torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def make_pe_env(cfg, role="pursuer", opponent_policy=None, render_mode=None, seed=None):
    """Create a SingleAgentPEWrapper from Hydra config."""
    base_env = PursuitEvasionEnv(
        arena_width=cfg.env.arena_width,
        arena_height=cfg.env.arena_height,
        dt=cfg.env.dt,
        max_steps=cfg.env.max_steps,
        capture_radius=cfg.env.capture_radius,
        collision_radius=cfg.env.collision_radius,
        robot_radius=cfg.env.robot_radius,
        pursuer_v_max=cfg.env.pursuer.v_max,
        pursuer_omega_max=cfg.env.pursuer.omega_max,
        evader_v_max=cfg.env.evader.v_max,
        evader_omega_max=cfg.env.evader.omega_max,
        min_init_distance=cfg.env.min_init_distance,
        max_init_distance=cfg.env.max_init_distance,
        distance_scale=cfg.env.reward.distance_scale,
        capture_bonus=cfg.env.reward.capture_bonus,
        timeout_penalty=cfg.env.reward.timeout_penalty,
        render_mode=render_mode,
    )
    wrapped = SingleAgentPEWrapper(base_env, role=role, opponent_policy=opponent_policy)
    if seed is not None:
        wrapped.reset(seed=seed)
    return wrapped


def make_vec_env(cfg, n_envs, role="pursuer", opponent_policy=None, seed=42):
    """Create vectorized training environment."""
    def _make_env(env_seed):
        def _init():
            env = make_pe_env(cfg, role=role, opponent_policy=opponent_policy)
            env.reset(seed=env_seed)
            return env
        return _init

    envs = DummyVecEnv([_make_env(seed + i) for i in range(n_envs)])
    envs = VecMonitor(envs)
    return envs
