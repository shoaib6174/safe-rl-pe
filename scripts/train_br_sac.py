"""Train a fresh SAC learner against a FROZEN SAC opponent for the BR
exploitability test (CoRL 2026 paper).

Spec: docs/superpowers/specs/2026-05-02-paper-strengthening-design.md

The frozen opponent is loaded from a `milestone_phase*` snapshot dir
(containing `ppo.zip` despite the legacy filename — it is SAC weights).
Wrapped with `PartialObsOpponentAdapter` so the frozen side observes the
world through its own sensors, identical to how the cohort trained it.

The trained side runs SAC with cohort-identical hyperparameters, p_full
fixed at 0.0 (full partial obs, no curriculum), no PFSP, no alternating
freeze, no forced switch — vanilla single-side training against a
fixed adversary.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from envs.opponent_adapter import PartialObsOpponentAdapter
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.partial_obs_wrapper import PartialObsWrapper
from envs.rewards import RewardComputer
from envs.wrappers import SingleAgentPEWrapper


# Cohort-identical hyperparameters (from scripts/launch_sp17b.sh).
COHORT_ENV_KWARGS = {
    "arena_width": 20.0,
    "arena_height": 20.0,
    "max_steps": 1200,
    "capture_radius": 0.5,
    "n_obstacles": 0,
    "n_obstacles_min": 0,
    "n_obstacles_max": 3,
    "n_obstacle_obs": 0,
    "pursuer_v_max": 1.0,
    "evader_v_max": 1.0,
    "distance_scale": 1.0,
    "use_visibility_reward": False,
    "visibility_weight": 1.0,
    "survival_bonus": 0.0,
    "timeout_penalty": -100.0,
    "capture_bonus": 100.0,
    "partial_obs": True,
    "sensing_radius": 8.0,
    "combined_masking": True,
    "fov_angle": 90.0,
    "w_search": 0.0001,
    "t_stale": 100,
    "w_vis_pursuer": 0.2,
    "history_length": 10,
}

COHORT_SAC_KWARGS = {
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "learning_starts": 10_000,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": 0.03,
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_kwargs": {"net_arch": [256, 256]},
}


@dataclass
class BRSetup:
    """Aggregate of the live objects the BR trainer needs."""
    base_env: PursuitEvasionEnv
    wrapper: SingleAgentPEWrapper
    frozen_model: SAC
    learner: SAC
    output_dir: Path
    trained_role: str
    frozen_role: str


def _make_reward_computer(env_kwargs: dict) -> RewardComputer:
    diagonal = float(np.sqrt(env_kwargs["arena_width"] ** 2
                             + env_kwargs["arena_height"] ** 2))
    return RewardComputer(
        distance_scale=env_kwargs.get("distance_scale", 1.0),
        d_max=diagonal,
        use_visibility_reward=env_kwargs.get("use_visibility_reward", False),
        visibility_weight=env_kwargs.get("visibility_weight", 1.0),
        survival_bonus=env_kwargs.get("survival_bonus", 0.0),
        timeout_penalty=env_kwargs.get("timeout_penalty", -100.0),
        capture_bonus=env_kwargs.get("capture_bonus", 100.0),
        w_vis_pursuer=env_kwargs.get("w_vis_pursuer", 0.0),
        sensing_radius=env_kwargs.get("sensing_radius", None),
    )


def _make_base_env(env_kwargs: dict) -> PursuitEvasionEnv:
    """Construct the raw PursuitEvasionEnv matching cohort training."""
    rc = _make_reward_computer(env_kwargs)
    return PursuitEvasionEnv(
        arena_width=env_kwargs["arena_width"],
        arena_height=env_kwargs["arena_height"],
        max_steps=env_kwargs["max_steps"],
        capture_radius=env_kwargs.get("capture_radius", 0.5),
        n_obstacles=env_kwargs.get("n_obstacles", 0),
        pursuer_v_max=env_kwargs["pursuer_v_max"],
        evader_v_max=env_kwargs["evader_v_max"],
        reward_computer=rc,
        partial_obs=env_kwargs["partial_obs"],
        n_obstacles_min=env_kwargs.get("n_obstacles_min"),
        n_obstacles_max=env_kwargs.get("n_obstacles_max"),
        asymmetric_obs=False,
        sensing_radius=env_kwargs.get("sensing_radius"),
        combined_masking=env_kwargs.get("combined_masking", False),
        fov_angle=env_kwargs.get("fov_angle", None),
        w_search=env_kwargs.get("w_search", 0.0),
        t_stale=env_kwargs.get("t_stale", 50),
    )


def build_br_setup(
    frozen_opponent_path: str,
    frozen_role: str,
    seed: int,
    output_dir: str,
    env_kwargs: dict | None = None,
) -> BRSetup:
    """Construct env + frozen opponent + fresh learner. No training yet.

    Args:
        frozen_opponent_path: Directory containing `ppo.zip` (cohort snapshot).
        frozen_role: 'pursuer' or 'evader' — the role the FROZEN side plays.
        seed: Random seed for the trainer.
        output_dir: Where eval log + checkpoints are written.
        env_kwargs: Override of COHORT_ENV_KWARGS (test/smoke-only). None => cohort.
    """
    if frozen_role not in ("pursuer", "evader"):
        raise ValueError(f"frozen_role must be 'pursuer' or 'evader', got {frozen_role!r}")
    trained_role = "evader" if frozen_role == "pursuer" else "pursuer"

    env_kwargs = dict(env_kwargs) if env_kwargs is not None else dict(COHORT_ENV_KWARGS)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Build base env (raw PursuitEvasionEnv).
    base_env = _make_base_env(env_kwargs)

    # 2) Load frozen SAC opponent.
    zip_path = Path(frozen_opponent_path) / "ppo.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"frozen opponent zip missing: {zip_path}")
    frozen_model = SAC.load(zip_path, device="cpu")

    # 3) Wrap frozen model into PartialObsOpponentAdapter.
    opp_adapter = PartialObsOpponentAdapter(
        model=frozen_model,
        role=frozen_role,
        base_env=base_env,
        deterministic=False,
    )

    # 4) Build SingleAgentPEWrapper (trained side + frozen opponent).
    #    p_full_obs=0.0 => always partial obs. NO CURRICULUM (per spec §4).
    single_env = SingleAgentPEWrapper(
        base_env,
        role=trained_role,
        opponent_policy=opp_adapter,
        p_full_obs=0.0,
    )

    # 5) Wrap with PartialObsWrapper (identical to cohort training stack).
    env = PartialObsWrapper(
        single_env,
        role=trained_role,
        history_length=env_kwargs.get("history_length", 10),
    )
    monitored = Monitor(env)

    # 6) Fresh SAC learner.
    learner = SAC("MultiInputPolicy", monitored, seed=seed, verbose=0,
                  **COHORT_SAC_KWARGS)

    return BRSetup(
        base_env=base_env,
        wrapper=monitored,
        frozen_model=frozen_model,
        learner=learner,
        output_dir=out,
        trained_role=trained_role,
        frozen_role=frozen_role,
    )


def run_one_update(setup: BRSetup, n_train_steps: int = 200) -> None:
    """Run a small `learn()` call. Used by tests to verify frozen invariance."""
    setup.learner.learn(total_timesteps=n_train_steps,
                        reset_num_timesteps=False, progress_bar=False)
