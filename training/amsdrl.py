"""AMS-DRL self-play protocol implementation.

Alternating Multi-agent Self-play with Deep Reinforcement Learning (Paper [18]).
Formal alternating self-play with Nash Equilibrium convergence tracking.

Protocol:
  S0: Cold-start — pre-train evader with NavigationEnv (goal-reaching + flee)
  S1: Train pursuer (evader frozen)
  S2: Train evader (pursuer frozen)
  S3: Train pursuer (evader frozen)
  ...repeat until NE convergence (|SR_P - SR_E| < eta) or max_phases reached

Each agent is a separate SB3 PPO instance. During a training phase, one agent
trains normally via model.learn() while the other is frozen as the opponent
inside the environment wrapper.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from agents.partial_obs_policy import PartialObsFeaturesExtractor
from envs.dcbf_action_wrapper import DCBFActionWrapper
from envs.rewards import RewardComputer
from envs.rnd import RNDModule, RNDRewardWrapper
from envs.navigation_env import NavigationEnv
from envs.opponent_adapter import PartialObsOpponentAdapter
from envs.partial_obs_wrapper import PartialObsWrapper
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import FixedSpeedModelAdapter, FixedSpeedWrapper, SingleAgentPEWrapper
from training.checkpoint_manager import CheckpointManager
from training.curriculum import CurriculumManager, SmoothCurriculumManager
from training.ewc import EWCRegularizer
from training.opponent_pool import OpponentPool
from training.selfplay_callbacks import (
    EntropyMonitorCallback,
    FixedBaselineEvalCallback,
    SelfPlayHealthMonitorCallback,
    flee_away_policy,
    flee_to_corner_policy,
    pure_pursuit_policy,
)


def _make_partial_obs_env(
    role: str,
    use_dcbf: bool = False,
    gamma: float = 0.2,
    history_length: int = 10,
    n_obstacles: int = 0,
    seed: int = 42,
    arena_width: float = 20.0,
    arena_height: float = 20.0,
    max_steps: int = 1200,
    capture_radius: float = 0.5,
    full_obs: bool = False,
    distance_scale: float = 1.0,
    pursuer_v_max: float = 1.0,
    evader_v_max: float = 1.0,
    fixed_speed: bool = False,
    min_init_distance: float = 3.0,
    max_init_distance: float = 15.0,
    w_occlusion: float = 0.0,
    use_visibility_reward: bool = False,
    visibility_weight: float = 1.0,
    survival_bonus: float = 0.0,
    prep_steps: int = 0,
    w_obs_approach: float = 0.0,
    timeout_penalty: float = -100.0,
    capture_bonus: float = 100.0,
    w_collision: float = 0.0,
    w_wall: float = 0.0,
    n_obstacle_obs: int = 0,
    partial_obs_los: bool = False,
) -> tuple:
    """Create an environment stack for one agent.

    Args:
        full_obs: If True, skip PartialObsWrapper (full-state observations).
        distance_scale: Scale factor for dense distance reward.
        pursuer_v_max: Maximum linear velocity for pursuer.
        evader_v_max: Maximum linear velocity for evader.
        fixed_speed: If True, fix v=v_max and only learn omega (1D action).
        min_init_distance: Minimum initial agent separation.
        max_init_distance: Maximum initial agent separation.
        w_occlusion: Weight on evader occlusion bonus (Mode A, 0 = off).
        use_visibility_reward: Enable visibility-based evader reward (Mode B).
        visibility_weight: Scale for +1/-1 visibility signal (Mode B).
        survival_bonus: Per-step survival bonus for evader (Mode B).
        prep_steps: Freeze pursuer for first N steps per episode (0 = off).
        w_obs_approach: PBRS obstacle-seeking weight for evader (0 = off).
        capture_bonus: Terminal capture reward magnitude.
        n_obstacle_obs: Number of nearest obstacles in observation (0 = none).
        partial_obs_los: Enable LOS-based partial obs (mask opponent when occluded).

    Returns:
        (env, base_env) where env is the fully-wrapped env and
        base_env is the underlying PursuitEvasionEnv.
    """
    # Build reward computer
    arena_diagonal = np.sqrt(arena_width**2 + arena_height**2)
    reward_computer = RewardComputer(
        distance_scale=distance_scale,
        d_max=arena_diagonal,
        w_occlusion=w_occlusion,
        use_visibility_reward=use_visibility_reward,
        visibility_weight=visibility_weight,
        survival_bonus=survival_bonus,
        w_obs_approach=w_obs_approach,
        timeout_penalty=timeout_penalty,
        capture_bonus=capture_bonus,
    )

    base_env = PursuitEvasionEnv(
        arena_width=arena_width,
        arena_height=arena_height,
        max_steps=max_steps,
        capture_radius=capture_radius,
        n_obstacles=n_obstacles,
        pursuer_v_max=pursuer_v_max,
        evader_v_max=evader_v_max,
        min_init_distance=min_init_distance,
        max_init_distance=max_init_distance,
        reward_computer=reward_computer,
        prep_steps=prep_steps,
        w_collision=w_collision,
        w_wall=w_wall,
        n_obstacle_obs=n_obstacle_obs,
        partial_obs=partial_obs_los,
    )
    single_env = SingleAgentPEWrapper(base_env, role=role)

    if full_obs:
        env = single_env
    else:
        env = PartialObsWrapper(single_env, role=role, history_length=history_length)

    if use_dcbf and role == "pursuer":
        env = DCBFActionWrapper(env, role=role, gamma=gamma)

    if fixed_speed:
        v_max = pursuer_v_max if role == "pursuer" else base_env.evader_v_max
        env = FixedSpeedWrapper(env, v_max=v_max)

    return env, base_env


def _make_vec_env(
    role: str,
    n_envs: int = 4,
    use_dcbf: bool = False,
    gamma: float = 0.2,
    history_length: int = 10,
    n_obstacles: int = 0,
    seed: int = 42,
    full_obs: bool = False,
    rnd_module: RNDModule | None = None,
    rnd_coef: float = 0.0,
    rnd_update_freq: int = 256,
    **env_kwargs,
) -> tuple:
    """Create vectorized environments.

    Returns:
        (vec_env, base_envs) where base_envs is a list of base PursuitEvasionEnv
        instances for opponent adapter access.
    """
    base_envs = []

    def make_env(env_seed):
        def _init():
            env, base = _make_partial_obs_env(
                role=role,
                use_dcbf=use_dcbf,
                gamma=gamma,
                history_length=history_length,
                n_obstacles=n_obstacles,
                seed=env_seed,
                full_obs=full_obs,
                **env_kwargs,
            )
            # RND wrapper for evader intrinsic motivation
            if rnd_module is not None and role == "evader":
                env = RNDRewardWrapper(
                    env, rnd_module=rnd_module,
                    rnd_coef=rnd_coef, update_freq=rnd_update_freq,
                )
            base_envs.append(base)
            return Monitor(env)
        return _init

    vec_env = DummyVecEnv([make_env(seed + i) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    return vec_env, base_envs


def _build_partial_obs_ppo(
    env,
    encoder_type: str = "bimdn",
    encoder_kwargs: dict | None = None,
    seed: int = 42,
    device: str = "cpu",
    tensorboard_log: str | None = None,
    learning_rate: float = 3e-4,
    n_steps: int = 512,
    batch_size: int = 256,
    ent_coef: float = 0.01,
) -> PPO:
    """Create a PPO model with partial-obs policy network."""
    policy_kwargs = {
        "features_extractor_class": PartialObsFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "encoder_type": encoder_type,
            **(encoder_kwargs or {}),
        },
        "net_arch": [256, 256],
        "activation_fn": torch.nn.Tanh,
    }

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=0,
        policy_kwargs=policy_kwargs,
    )
    return model


def _build_full_obs_ppo(
    env,
    seed: int = 42,
    device: str = "cpu",
    tensorboard_log: str | None = None,
    learning_rate: float = 3e-4,
    n_steps: int = 512,
    batch_size: int = 256,
    ent_coef: float = 0.01,
) -> PPO:
    """Create a PPO model with full-obs MLP policy (diagnostic mode)."""
    policy_kwargs = {
        "net_arch": [256, 256],
        "activation_fn": torch.nn.Tanh,
    }

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=0,
        policy_kwargs=policy_kwargs,
    )
    return model


def _update_opponent_in_vec_env(vec_env, adapter):
    """Set the opponent adapter in all sub-envs of a VecEnv.

    Traverses VecMonitor -> DummyVecEnv -> Monitor -> wrapper chain
    to find SingleAgentPEWrapper and set the opponent.
    """
    inner_vec = vec_env.venv if hasattr(vec_env, "venv") else vec_env
    for env in inner_vec.envs:
        # Unwrap Monitor if present
        base = env
        while hasattr(base, "env"):
            if isinstance(base, SingleAgentPEWrapper):
                break
            base = base.env
        if isinstance(base, SingleAgentPEWrapper):
            base.set_opponent(adapter)


def _evaluate_head_to_head(
    pursuer_model,
    evader_model,
    n_episodes: int = 100,
    seed: int = 0,
    arena_width: float = 20.0,
    arena_height: float = 20.0,
    max_steps: int = 1200,
    capture_radius: float = 0.5,
    n_obstacles: int = 0,
    distance_scale: float = 1.0,
    pursuer_v_max: float = 1.0,
    evader_v_max: float = 1.0,
    fixed_speed: bool = False,
    min_init_distance: float = 3.0,
    max_init_distance: float = 15.0,
    prep_steps: int = 0,
    w_collision: float = 0.0,
    w_wall: float = 0.0,
    n_obstacle_obs: int = 0,
    **kwargs,
) -> dict:
    """Evaluate pursuer vs evader head-to-head using full-state obs.

    Both agents use their own partial obs processing internally.
    We step the base env and feed observations through each agent's adapter.
    """
    base_env = PursuitEvasionEnv(
        arena_width=arena_width,
        arena_height=arena_height,
        max_steps=max_steps,
        capture_radius=capture_radius,
        n_obstacles=n_obstacles,
        distance_scale=distance_scale,
        pursuer_v_max=pursuer_v_max,
        evader_v_max=evader_v_max,
        min_init_distance=min_init_distance,
        max_init_distance=max_init_distance,
        prep_steps=prep_steps,
        w_collision=w_collision,
        w_wall=w_wall,
        n_obstacle_obs=n_obstacle_obs,
    )

    # Create adapters for both agents
    p_adapter = PartialObsOpponentAdapter(
        model=pursuer_model, role="pursuer", base_env=base_env, deterministic=True,
    )
    e_adapter = PartialObsOpponentAdapter(
        model=evader_model, role="evader", base_env=base_env, deterministic=True,
    )

    captures = 0
    timeouts = 0
    episode_lengths = []
    capture_times = []

    for ep in range(n_episodes):
        obs, info = base_env.reset(seed=seed + ep)
        p_adapter.reset()
        e_adapter.reset()
        done = False

        while not done:
            # Both agents predict through their partial obs adapters
            p_action, _ = p_adapter.predict(obs["pursuer"], deterministic=True)
            e_action, _ = e_adapter.predict(obs["evader"], deterministic=True)
            # If fixed_speed, model outputs 1D [omega] — expand to [v_max, omega]
            if fixed_speed:
                if p_action.shape[-1] == 1:
                    p_action = np.array([pursuer_v_max, p_action[0]], dtype=np.float32)
                if e_action.shape[-1] == 1:
                    e_action = np.array([base_env.evader_v_max, e_action[0]], dtype=np.float32)
            obs, rewards, terminated, truncated, info = base_env.step(p_action, e_action)
            done = terminated or truncated

        if "episode_metrics" in info:
            m = info["episode_metrics"]
            if m["captured"]:
                captures += 1
                capture_times.append(m["capture_time"])
            else:
                timeouts += 1
            episode_lengths.append(m["episode_length"])

    base_env.close()

    return {
        "capture_rate": captures / max(n_episodes, 1),
        "escape_rate": timeouts / max(n_episodes, 1),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0,
        "mean_capture_time": float(np.mean(capture_times)) if capture_times else float("nan"),
    }


def _evaluate_head_to_head_full_obs(
    pursuer_model,
    evader_model,
    n_episodes: int = 100,
    seed: int = 0,
    arena_width: float = 20.0,
    arena_height: float = 20.0,
    max_steps: int = 1200,
    capture_radius: float = 0.5,
    n_obstacles: int = 0,
    distance_scale: float = 1.0,
    pursuer_v_max: float = 1.0,
    evader_v_max: float = 1.0,
    fixed_speed: bool = False,
    min_init_distance: float = 3.0,
    max_init_distance: float = 15.0,
    prep_steps: int = 0,
    w_collision: float = 0.0,
    w_wall: float = 0.0,
    n_obstacle_obs: int = 0,
    **kwargs,
) -> dict:
    """Evaluate pursuer vs evader with full-state observations (diagnostic mode)."""
    base_env = PursuitEvasionEnv(
        arena_width=arena_width,
        arena_height=arena_height,
        max_steps=max_steps,
        capture_radius=capture_radius,
        n_obstacles=n_obstacles,
        distance_scale=distance_scale,
        pursuer_v_max=pursuer_v_max,
        evader_v_max=evader_v_max,
        min_init_distance=min_init_distance,
        max_init_distance=max_init_distance,
        prep_steps=prep_steps,
        w_collision=w_collision,
        w_wall=w_wall,
        n_obstacle_obs=n_obstacle_obs,
    )

    captures = 0
    timeouts = 0
    episode_lengths = []
    capture_times = []

    for ep in range(n_episodes):
        obs, info = base_env.reset(seed=seed + ep)
        done = False

        while not done:
            # Both agents use full-state obs directly
            p_action, _ = pursuer_model.predict(obs["pursuer"], deterministic=True)
            e_action, _ = evader_model.predict(obs["evader"], deterministic=True)
            # If fixed_speed, model outputs 1D [omega] — expand to [v_max, omega]
            if fixed_speed:
                if p_action.shape[-1] == 1:
                    p_action = np.array([pursuer_v_max, p_action[0]], dtype=np.float32)
                if e_action.shape[-1] == 1:
                    e_action = np.array([base_env.evader_v_max, e_action[0]], dtype=np.float32)
            obs, rewards, terminated, truncated, info = base_env.step(p_action, e_action)
            done = terminated or truncated

        if "episode_metrics" in info:
            m = info["episode_metrics"]
            if m["captured"]:
                captures += 1
                capture_times.append(m["capture_time"])
            else:
                timeouts += 1
            episode_lengths.append(m["episode_length"])

    base_env.close()

    return {
        "capture_rate": captures / max(n_episodes, 1),
        "escape_rate": timeouts / max(n_episodes, 1),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0,
        "mean_capture_time": float(np.mean(capture_times)) if capture_times else float("nan"),
    }


class AMSDRLSelfPlay:
    """AMS-DRL self-play protocol orchestrator.

    Manages the full alternating self-play training loop:
    1. Cold-start evader with NavigationEnv
    2. Alternate training pursuer and evader
    3. Track NE convergence
    4. Health monitoring and rollback

    Args:
        output_dir: Directory for checkpoints, logs, and results.
        max_phases: Maximum number of alternating phases.
        timesteps_per_phase: Training steps per phase.
        cold_start_timesteps: Training steps for cold-start (S0).
        eta: NE convergence threshold (|SR_P - SR_E| < eta).
        eval_episodes: Number of episodes for inter-phase evaluation.
        n_envs: Number of parallel environments.
        use_dcbf: Whether to use DCBF safety filter for pursuer.
        gamma: DCBF decay rate.
        encoder_type: Belief encoder type ('bimdn', 'lstm', 'mlp').
        encoder_kwargs: Additional encoder arguments.
        history_length: Observation history length (K).
        n_obstacles: Number of obstacles in arena.
        arena_width: Arena width.
        arena_height: Arena height.
        max_steps: Max episode steps.
        capture_radius: Capture distance.
        seed: Random seed.
        device: Torch device ('cpu' or 'cuda').
        learning_rate: PPO learning rate.
        ent_coef: Entropy coefficient.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        output_dir: str = "results/amsdrl",
        max_phases: int = 12,
        timesteps_per_phase: int = 500_000,
        cold_start_timesteps: int = 200_000,
        eta: float = 0.10,
        eval_episodes: int = 100,
        n_envs: int = 4,
        use_dcbf: bool = True,
        gamma: float = 0.2,
        encoder_type: str = "bimdn",
        encoder_kwargs: dict | None = None,
        history_length: int = 10,
        n_obstacles: int = 0,
        arena_width: float = 20.0,
        arena_height: float = 20.0,
        max_steps: int = 1200,
        capture_radius: float = 0.5,
        seed: int = 42,
        device: str = "cpu",
        learning_rate: float = 3e-4,
        ent_coef: float = 0.01,
        full_obs: bool = False,
        distance_scale: float = 1.0,
        pursuer_v_max: float = 1.0,
        evader_v_max: float = 1.0,
        fixed_speed: bool = False,
        n_steps: int = 512,
        batch_size: int = 256,
        curriculum: bool = False,
        opponent_pool_size: int = 0,
        w_occlusion: float = 0.0,
        use_visibility_reward: bool = False,
        visibility_weight: float = 1.0,
        survival_bonus: float = 0.0,
        prep_steps: int = 0,
        w_obs_approach: float = 0.0,
        timeout_penalty: float = -100.0,
        capture_bonus: float = 100.0,
        w_collision: float = 0.0,
        w_wall: float = 0.0,
        n_obstacle_obs: int = 0,
        evader_training_multiplier: float = 1.0,
        min_escape_rate: float = 0.0,
        min_phases_per_level: int = 1,
        bilateral_rollback: bool = False,
        evader_first_on_advance: bool = False,
        warm_start_evader: bool = False,
        warm_start_timesteps: int = 50_000,
        mixed_level_ratio: float = 0.0,
        smooth_curriculum: bool = False,
        smooth_curriculum_increment: float = 0.5,
        min_obstacles: int = 0,
        phase_warmup: bool = False,
        phase_warmup_schedule: list[tuple[int, int]] | None = None,
        ne_gap_advancement: bool = False,
        ne_gap_threshold: float = 0.15,
        ne_gap_consecutive: int = 2,
        micro_phase_steps: int = 0,
        eval_interval_micro: int = 50,
        snapshot_freq_micro: int = 5,
        max_total_steps: int = 0,
        convergence_consecutive: int = 5,
        adaptive_ratio_threshold: float = 0.0,
        adaptive_boost_phases: int = 20,
        lr_dampen_threshold: float = 0.0,
        collapse_threshold: float = 0.0,
        collapse_streak_limit: int = 3,
        pfsp_enabled: bool = False,
        ewc_lambda: float = 0.0,
        ewc_fisher_samples: int = 1024,
        rnd_coef: float = 0.0,
        rnd_embed_dim: int = 64,
        rnd_hidden_dim: int = 128,
        rnd_update_freq: int = 256,
        init_pursuer_path: str | None = None,
        init_evader_path: str | None = None,
        partial_obs_los: bool = False,
        verbose: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_phases = max_phases
        self.timesteps_per_phase = timesteps_per_phase
        self.cold_start_timesteps = cold_start_timesteps
        self.eta = eta
        self.eval_episodes = eval_episodes
        self.n_envs = n_envs
        self.use_dcbf = use_dcbf
        self.gamma = gamma
        self.encoder_type = encoder_type
        self.encoder_kwargs = encoder_kwargs or {}
        self.history_length = history_length
        self.n_obstacles = n_obstacles
        self.seed = seed
        self.device = device
        self.learning_rate = learning_rate
        self.ent_coef = ent_coef
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.full_obs = full_obs
        self.verbose = verbose

        # Common env kwargs
        self.env_kwargs = {
            "arena_width": arena_width,
            "arena_height": arena_height,
            "max_steps": max_steps,
            "capture_radius": capture_radius,
            "distance_scale": distance_scale,
            "pursuer_v_max": pursuer_v_max,
            "evader_v_max": evader_v_max,
            "w_occlusion": w_occlusion,
            "use_visibility_reward": use_visibility_reward,
            "visibility_weight": visibility_weight,
            "survival_bonus": survival_bonus,
            "prep_steps": prep_steps,
            "w_obs_approach": w_obs_approach,
            "timeout_penalty": timeout_penalty,
            "capture_bonus": capture_bonus,
            "w_collision": w_collision,
            "w_wall": w_wall,
            "n_obstacle_obs": n_obstacle_obs,
            "partial_obs_los": partial_obs_los,
        }
        self.fixed_speed = fixed_speed
        self.evader_training_multiplier = evader_training_multiplier
        self.bilateral_rollback = bilateral_rollback
        self.evader_first_on_advance = evader_first_on_advance
        self.warm_start_evader = warm_start_evader
        self.warm_start_timesteps = warm_start_timesteps
        self.mixed_level_ratio = mixed_level_ratio

        # Phase warmup: shorter phases early, ramp to full length
        self.phase_warmup = phase_warmup
        if phase_warmup and phase_warmup_schedule is None:
            # Default schedule: (phase_number, timesteps)
            # S1-S2: 100K, S3-S4: 200K, S5-S6: 300K, S7+: full
            phase_warmup_schedule = [
                (1, 100_000), (2, 100_000),
                (3, 200_000), (4, 200_000),
                (5, 300_000), (6, 300_000),
            ]
        self.phase_warmup_schedule = {
            p: t for p, t in (phase_warmup_schedule or [])
        }

        # NE-gap-based advancement: advance when balanced, not when dominated
        self.ne_gap_advancement = ne_gap_advancement
        self.ne_gap_threshold = ne_gap_threshold
        self.ne_gap_consecutive = ne_gap_consecutive
        self._ne_gap_streak = 0  # consecutive phases with NE gap below threshold

        # Micro-phase rapid alternation (RA redesign)
        self.micro_phase_steps = micro_phase_steps
        self.eval_interval_micro = eval_interval_micro
        self.snapshot_freq_micro = snapshot_freq_micro
        self.max_total_steps = max_total_steps
        self.convergence_consecutive = convergence_consecutive

        # Adaptive training ratio + LR dampening (anti-cycling)
        self.adaptive_ratio_threshold = adaptive_ratio_threshold
        self.adaptive_boost_phases = adaptive_boost_phases
        self.lr_dampen_threshold = lr_dampen_threshold

        # Collapse rollback: restore best checkpoint when agent collapses
        self.collapse_threshold = collapse_threshold  # SR below this = collapsed
        self.collapse_streak_limit = collapse_streak_limit  # consecutive evals

        # PFSP-lite: bias opponent pool sampling when losing
        self.pfsp_enabled = pfsp_enabled

        # Curriculum learning
        self.curriculum = None
        if smooth_curriculum:
            self.curriculum = SmoothCurriculumManager(
                arena_width=arena_width,
                arena_height=arena_height,
                distance_increment=smooth_curriculum_increment,
                min_escape_rate=min_escape_rate,
                min_phases_per_level=min_phases_per_level,
                min_obstacles=min_obstacles,
            )
            overrides = self.curriculum.get_env_overrides()
            self.env_kwargs["min_init_distance"] = overrides["min_init_distance"]
            self.env_kwargs["max_init_distance"] = overrides["max_init_distance"]
            self.n_obstacles = overrides["n_obstacles"]
        elif curriculum:
            self.curriculum = CurriculumManager(
                arena_width=arena_width,
                arena_height=arena_height,
                min_escape_rate=min_escape_rate,
                min_phases_per_level=min_phases_per_level,
            )
            # Apply Level 1 overrides immediately
            overrides = self.curriculum.get_env_overrides()
            self.env_kwargs["min_init_distance"] = overrides["min_init_distance"]
            self.env_kwargs["max_init_distance"] = overrides["max_init_distance"]
            self.n_obstacles = overrides["n_obstacles"]

        # Models (initialized in run())
        self.pursuer_model = None
        self.evader_model = None

        # History
        self.history: list[dict] = []

        # Checkpoint managers (separate for pursuer/evader)
        self.pursuer_ckpt = CheckpointManager(
            self.output_dir / "checkpoints" / "pursuer", max_rolling=15
        )
        self.evader_ckpt = CheckpointManager(
            self.output_dir / "checkpoints" / "evader", max_rolling=15
        )

        # Opponent pools for diverse self-play (0 = disabled)
        self.opponent_pool_size = opponent_pool_size
        if opponent_pool_size > 0:
            self.pursuer_pool = OpponentPool(
                max_size=opponent_pool_size, include_random=True,
                eviction_strategy="reservoir",
            )
            self.evader_pool = OpponentPool(
                max_size=opponent_pool_size, include_random=True,
                eviction_strategy="reservoir",
            )
        else:
            self.pursuer_pool = None
            self.evader_pool = None

        # EWC regularizer (Tier 3, Fix 5)
        self.ewc_lambda = ewc_lambda
        self.ewc: EWCRegularizer | None = None
        if ewc_lambda > 0:
            self.ewc = EWCRegularizer(
                lambda_=ewc_lambda,
                fisher_samples=ewc_fisher_samples,
            )

        # RND intrinsic motivation (Tier 3, Fix 9)
        self.rnd_coef = rnd_coef
        self.rnd_embed_dim = rnd_embed_dim
        self.rnd_hidden_dim = rnd_hidden_dim
        self.rnd_update_freq = rnd_update_freq
        self.rnd_module = None

        # Pre-trained model paths for warm-seeded self-play
        self.init_pursuer_path = init_pursuer_path
        self.init_evader_path = init_evader_path  # Created lazily when env obs_dim is known

    def run(self) -> dict:
        """Execute the full AMS-DRL protocol.

        Returns:
            Dict with 'pursuer_model', 'evader_model', 'history', 'converged'.
        """
        start_time = time.time()

        if self.verbose:
            print("=" * 60)
            print("AMS-DRL Self-Play Protocol")
            print(f"  Max phases: {self.max_phases}")
            print(f"  Steps/phase: {self.timesteps_per_phase}")
            print(f"  NE threshold (eta): {self.eta}")
            print(f"  Encoder: {self.encoder_type}")
            print(f"  DCBF: {self.use_dcbf}")
            print(f"  Full-obs: {self.full_obs}")
            print(f"  Distance scale: {self.env_kwargs['distance_scale']}")
            print(f"  Pursuer v_max: {self.env_kwargs['pursuer_v_max']}")
            print(f"  Evader v_max: {self.env_kwargs['evader_v_max']}")
            print(f"  Fixed speed: {self.fixed_speed}")
            print(f"  Curriculum: {self.curriculum is not None}")
            print(f"  Opponent pool: {self.opponent_pool_size if self.opponent_pool_size > 0 else 'disabled'}")
            if self.micro_phase_steps > 0:
                print(f"  Mode: MICRO-PHASE (steps={self.micro_phase_steps})")
            prep = self.env_kwargs.get("prep_steps", 0)
            if prep > 0:
                print(f"  Prep phase: {prep} steps (pursuer frozen)")
            if self.env_kwargs.get("use_visibility_reward", False):
                print(f"  Reward mode: VISIBILITY (OpenAI H&S style)")
                print(f"    Visibility weight: {self.env_kwargs.get('visibility_weight', 1.0)}")
                print(f"    Survival bonus: {self.env_kwargs.get('survival_bonus', 0.0)}")
                print(f"    Capture bonus: {self.env_kwargs.get('capture_bonus', 100.0)}")
                print(f"    Timeout penalty: {self.env_kwargs.get('timeout_penalty', -100.0)}")
            else:
                w_occ = self.env_kwargs.get("w_occlusion", 0.0)
                if w_occ > 0:
                    print(f"  Occlusion bonus: {w_occ}")
            n_obs_obs = self.env_kwargs.get("n_obstacle_obs", 0)
            if n_obs_obs > 0:
                print(f"  Obstacle obs: {n_obs_obs} nearest obstacles in obs vector")
            if self.env_kwargs.get("partial_obs_los", False):
                print(f"  Partial obs: LOS masking ENABLED (opponent masked when occluded)")
            if self.ewc_lambda > 0:
                print(f"  EWC: lambda={self.ewc_lambda}, fisher_samples={self.ewc.fisher_samples}")
            if self.rnd_coef > 0:
                print(f"  RND: coef={self.rnd_coef}, embed={self.rnd_embed_dim}, "
                      f"hidden={self.rnd_hidden_dim}, update_freq={self.rnd_update_freq}")
            if self.phase_warmup:
                schedule_str = ", ".join(
                    f"S{p}:{t//1000}K" for p, t in sorted(self.phase_warmup_schedule.items())
                )
                print(f"  Phase warmup: {schedule_str}, then {self.timesteps_per_phase//1000}K")
            if self.ne_gap_advancement:
                print(f"  NE-gap advancement: threshold={self.ne_gap_threshold}, "
                      f"consecutive={self.ne_gap_consecutive}")
            if self.init_pursuer_path and self.init_evader_path:
                print(f"  Init mode: WARM-SEED (pre-trained models)")
                print(f"    Pursuer: {self.init_pursuer_path}")
                print(f"    Evader: {self.init_evader_path}")
            print(f"  Seed: {self.seed}")
            print(f"  Device: {self.device}")
            print("=" * 60)

        # ─── Phase S0: Cold-Start or Warm-Seed ───
        if self.init_pursuer_path and self.init_evader_path:
            if self.verbose:
                print("\n=== Phase S0: Warm-Seed (Loading Pre-trained Models) ===")
            self._warm_seed()
        else:
            if self.verbose:
                print("\n=== Phase S0: Cold-Start (Evader Pre-training) ===")
            self._cold_start()

        # Evaluate cold-start
        cs_metrics = self._evaluate()
        cs_entry = {
            "phase": "S0",
            "role": "evader",
            **cs_metrics,
        }
        if self.curriculum:
            cs_entry.update(self.curriculum.get_status())
        self.history.append(cs_entry)
        if self.verbose:
            print(f"  Cold-start eval: capture_rate={cs_metrics['capture_rate']:.2f}, "
                  f"escape_rate={cs_metrics['escape_rate']:.2f}")

        # ─── Dispatch: Micro-Phase or Legacy ───
        if self.micro_phase_steps > 0:
            return self._run_micro_phases()

        # ─── Legacy: Alternating Training Phases ───
        converged = False
        force_next_role = None  # Override alternation for evader-first
        for phase in range(1, self.max_phases + 1):
            if force_next_role is not None:
                role = force_next_role
                force_next_role = None
            else:
                role = "pursuer" if phase % 2 == 1 else "evader"

            if self.verbose:
                print(f"\n=== Phase S{phase}: Training {role} ===")

            self._train_phase(phase, role)

            # Evaluate
            metrics = self._evaluate()
            metrics["phase"] = f"S{phase}"
            metrics["role"] = role

            # Curriculum: log level and check advancement / regression
            if self.curriculum:
                metrics.update(self.curriculum.get_status())

                # NE-gap advancement: advance when balanced for N consecutive phases
                if self.ne_gap_advancement:
                    _ne_gap = abs(metrics["capture_rate"] - metrics["escape_rate"])
                    self.curriculum.phases_at_level += 1
                    self.curriculum.level_history.append({
                        "level": self.curriculum.current_level,
                        "capture_rate": metrics["capture_rate"],
                        "escape_rate": metrics["escape_rate"],
                        "ne_gap": _ne_gap,
                        "advanced": False,
                    })
                    if _ne_gap < self.ne_gap_threshold:
                        self._ne_gap_streak += 1
                    else:
                        self._ne_gap_streak = 0
                    advanced = (
                        self._ne_gap_streak >= self.ne_gap_consecutive
                        and not self.curriculum.at_max_level
                    )
                    if advanced:
                        self.curriculum._advance()
                        self.curriculum.level_history[-1]["advanced"] = True
                        self._ne_gap_streak = 0
                        if self.verbose:
                            print(f"  [NE-GAP ADV] Balanced for {self.ne_gap_consecutive} phases "
                                  f"(gap={_ne_gap:.3f} < {self.ne_gap_threshold}), advancing curriculum")
                else:
                    advanced = self.curriculum.check_advancement(
                        metrics["capture_rate"], metrics["escape_rate"]
                    )
                if advanced:
                    # EWC: snapshot evader before changing level
                    if self.ewc is not None and self.evader_model is not None:
                        self._ewc_snapshot_evader()

                    # Apply new level's env overrides for subsequent phases
                    overrides = self.curriculum.get_env_overrides()
                    self.env_kwargs["min_init_distance"] = overrides["min_init_distance"]
                    self.env_kwargs["max_init_distance"] = overrides["max_init_distance"]
                    self.n_obstacles = overrides["n_obstacles"]

                    # Evader-first: force next phase to train evader
                    if self.evader_first_on_advance:
                        force_next_role = "evader"
                        if self.verbose:
                            print("  [EVADER-FIRST] Next phase will train evader at new level")

                    # Warm-start evader at new level
                    if self.warm_start_evader:
                        self._warm_start_evader_at_level()
                else:
                    # Check for regression (evader collapse)
                    regressed = self.curriculum.check_regression(metrics["escape_rate"])
                    if regressed:
                        overrides = self.curriculum.get_env_overrides()
                        self.env_kwargs["min_init_distance"] = overrides["min_init_distance"]
                        self.env_kwargs["max_init_distance"] = overrides["max_init_distance"]
                        self.n_obstacles = overrides["n_obstacles"]

            self.history.append(metrics)

            sr_p = metrics["capture_rate"]
            sr_e = metrics["escape_rate"]
            ne_gap = abs(sr_p - sr_e)

            if self.verbose:
                status = f"  SR_P={sr_p:.3f}, SR_E={sr_e:.3f}, NE gap={ne_gap:.3f}"
                if self.curriculum:
                    status += f", Level={self.curriculum.current_level}"
                print(status)

            # Check convergence (require max curriculum level if curriculum is active)
            curriculum_ready = (self.curriculum is None) or self.curriculum.at_max_level
            if ne_gap < self.eta and curriculum_ready:
                converged = True
                if self.verbose:
                    print(f"  *** Converged at phase {phase}! (NE gap {ne_gap:.3f} < {self.eta}) ***")
                break
            elif ne_gap < self.eta and not curriculum_ready:
                if self.verbose:
                    print(f"  NE gap {ne_gap:.3f} < {self.eta} but curriculum not at max level ({self.curriculum.current_level}/{self.curriculum.max_level}) — continuing")

        # Save final models
        self._save_final()

        # Save history
        elapsed = time.time() - start_time
        result = {
            "converged": converged,
            "total_phases": len(self.history) - 1,  # Exclude S0
            "history": self.history,
            "elapsed_seconds": elapsed,
        }
        if self.curriculum:
            result["curriculum_final_level"] = self.curriculum.current_level
            result["curriculum_history"] = self.curriculum.level_history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"AMS-DRL complete. Converged: {converged}")
            print(f"Total time: {elapsed / 3600:.1f}h")
            print(f"Results saved to: {self.output_dir}")
            print(f"{'=' * 60}")

        return {
            "pursuer_model": self.pursuer_model,
            "evader_model": self.evader_model,
            "history": self.history,
            "converged": converged,
        }

    def _warm_seed(self):
        """Phase S0 alternative: load pre-trained models and seed opponent pools.

        Used when init_pursuer_path and init_evader_path are provided.
        Models are loaded directly via PPO.load() to preserve their
        original architecture. Training hyperparameters (lr, ent_coef, etc.)
        are overridden to match the self-play config.

        The _run_micro_phases() method handles env attachment and n_envs
        mismatch via its save/reload cycle, so no environment setup is
        needed here.
        """
        # Load pre-trained models directly (preserves architecture)
        self.pursuer_model = PPO.load(
            self.init_pursuer_path, device=self.device)
        self.evader_model = PPO.load(
            self.init_evader_path, device=self.device)

        # Override training hyperparameters to match self-play config
        for model, role in [(self.pursuer_model, "pursuer"),
                            (self.evader_model, "evader")]:
            model.learning_rate = self.learning_rate
            model.ent_coef = self.ent_coef
            model.n_steps = self.n_steps
            model.batch_size = self.batch_size
            model.tensorboard_log = str(self.output_dir / "tb" / role)
            model.seed = self.seed if role == "pursuer" else self.seed + 1

        if self.verbose:
            print(f"  Loaded pursuer from: {self.init_pursuer_path}")
            print(f"  Loaded evader from: {self.init_evader_path}")
            print(f"  Overridden: lr={self.learning_rate}, "
                  f"ent_coef={self.ent_coef}, "
                  f"n_steps={self.n_steps}, batch_size={self.batch_size}")

        # Save initial checkpoints
        self.pursuer_ckpt.save_milestone(
            model=self.pursuer_model, phase=0, role="pursuer",
        )
        self.evader_ckpt.save_milestone(
            model=self.evader_model, phase=0, role="evader",
        )

        # Seed opponent pools with the pre-trained models
        if self.opponent_pool_size > 0:
            self._save_micro_snapshot("pursuer", 0)
            self._save_micro_snapshot("evader", 0)
            if self.verbose:
                print(f"  Seeded opponent pools (P:{len(self.pursuer_pool)}, "
                      f"E:{len(self.evader_pool)})")

        if self.verbose:
            print("  Warm-seed complete: both models loaded from checkpoints")

    def _cold_start(self):
        """Phase S0: Pre-train evader with NavigationEnv (or init both for full-obs)."""
        if self.full_obs:
            # Full-obs mode: no NavigationEnv, just create both models from scratch
            build_fn = _build_full_obs_ppo

            e_env, _ = _make_partial_obs_env(
                role="evader",
                use_dcbf=False,
                history_length=self.history_length,
                n_obstacles=self.n_obstacles,
                full_obs=True,
                fixed_speed=self.fixed_speed,
                **self.env_kwargs,
            )
            e_env = Monitor(e_env)
            self.evader_model = build_fn(
                e_env,
                seed=self.seed + 1,
                device=self.device,
                tensorboard_log=str(self.output_dir / "tb" / "evader"),
                learning_rate=self.learning_rate,
                ent_coef=self.ent_coef,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
            )
            e_env.close()

            p_env, _ = _make_partial_obs_env(
                role="pursuer",
                use_dcbf=self.use_dcbf,
                gamma=self.gamma,
                history_length=self.history_length,
                n_obstacles=self.n_obstacles,
                full_obs=True,
                fixed_speed=self.fixed_speed,
                **self.env_kwargs,
            )
            p_env = Monitor(p_env)
            self.pursuer_model = build_fn(
                p_env,
                seed=self.seed,
                device=self.device,
                tensorboard_log=str(self.output_dir / "tb" / "pursuer"),
                learning_rate=self.learning_rate,
                ent_coef=self.ent_coef,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
            )
            p_env.close()

            if self.verbose:
                print("  Full-obs mode: both models initialized (no cold-start training)")
            return

        # Partial-obs mode: pre-train evader with NavigationEnv
        # Filter env_kwargs: reward params are handled by RewardComputer, not PursuitEvasionEnv
        _reward_keys = {"w_occlusion", "use_visibility_reward", "visibility_weight", "survival_bonus", "w_obs_approach", "timeout_penalty"}
        _env_only_keys = {"partial_obs_los"}  # handled by _make_partial_obs_env, not PursuitEvasionEnv directly
        pe_kwargs = {k: v for k, v in self.env_kwargs.items() if k not in _reward_keys and k not in _env_only_keys}
        reward_params = {k: v for k, v in self.env_kwargs.items() if k in _reward_keys}
        if any(v for k, v in reward_params.items() if k != "w_occlusion" and v) or reward_params.get("w_occlusion", 0.0) > 0:
            arena_diag = np.sqrt(pe_kwargs["arena_width"]**2 + pe_kwargs["arena_height"]**2)
            rc = RewardComputer(
                distance_scale=pe_kwargs.get("distance_scale", 1.0),
                d_max=arena_diag,
                **reward_params,
            )
            pe_kwargs["reward_computer"] = rc
        base_env = PursuitEvasionEnv(**pe_kwargs, n_obstacles=self.n_obstacles)
        nav_env = NavigationEnv(
            base_env,
            role="evader",
            include_flee_phase=True,
            history_length=self.history_length,
            max_steps=300,
        )
        if self.fixed_speed:
            nav_env = FixedSpeedWrapper(nav_env, v_max=base_env.evader_v_max)
        nav_env = Monitor(nav_env)

        # Create evader model
        self.evader_model = _build_partial_obs_ppo(
            nav_env,
            encoder_type=self.encoder_type,
            encoder_kwargs=self.encoder_kwargs,
            seed=self.seed + 1,
            device=self.device,
            tensorboard_log=str(self.output_dir / "tb" / "evader_coldstart"),
            learning_rate=self.learning_rate,
            ent_coef=self.ent_coef,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
        )

        # Train
        self.evader_model.learn(
            total_timesteps=self.cold_start_timesteps,
            progress_bar=True,
        )

        nav_env.close()

        # Create pursuer model (untrained — will start from scratch in S1)
        # We need to create a temp env for model initialization
        p_env, _ = _make_partial_obs_env(
            role="pursuer",
            use_dcbf=self.use_dcbf,
            gamma=self.gamma,
            history_length=self.history_length,
            n_obstacles=self.n_obstacles,
            fixed_speed=self.fixed_speed,
            **self.env_kwargs,
        )
        p_env = Monitor(p_env)

        self.pursuer_model = _build_partial_obs_ppo(
            p_env,
            encoder_type=self.encoder_type,
            encoder_kwargs=self.encoder_kwargs,
            seed=self.seed,
            device=self.device,
            tensorboard_log=str(self.output_dir / "tb" / "pursuer"),
            learning_rate=self.learning_rate,
            ent_coef=self.ent_coef,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
        )
        p_env.close()

        # Save evader as rolling checkpoint so bilateral rollback works in S1
        self.evader_ckpt.save_rolling(
            model=self.evader_model,
            encoder=getattr(self.evader_model.policy, "features_extractor", None),
            step=0,
            meta={"phase": "S0", "role": "evader", "source": "cold_start"},
        )
        # Also save milestone for warm-start reference
        self.evader_ckpt.save_milestone(
            model=self.evader_model,
            encoder=getattr(self.evader_model.policy, "features_extractor", None),
            phase=0,
            role="evader",
            meta={"source": "cold_start"},
        )

        if self.verbose:
            print(f"  Cold-start complete: {self.cold_start_timesteps} steps")

    def _warm_start_evader_at_level(self):
        """Restore evader to best milestone weights and optionally pre-train.

        Called when curriculum advances to a new level. Prevents the evader
        from entering the new level with a policy that's been overfit to the
        previous level's distance distribution.
        """
        # Find the best evader milestone checkpoint
        milestone_dir = self.evader_ckpt.checkpoint_dir
        milestones = sorted(milestone_dir.glob("milestone_phase*_evader"))
        if not milestones:
            if self.verbose:
                print("  [WARM-START] No evader milestones found, skipping")
            return

        # Use the most recent evader milestone
        best_milestone = milestones[-1]
        if self.verbose:
            print(f"  [WARM-START] Restoring evader from {best_milestone.name}")

        try:
            loaded_model = type(self.evader_model).load(
                str(best_milestone / "ppo.zip")
            )
            self.evader_model.policy.load_state_dict(
                loaded_model.policy.state_dict()
            )
        except Exception as e:
            print(f"  [WARM-START] Failed to restore milestone: {e}")
            return

        # Solo pre-training: train evader at new level with random opponent
        if self.warm_start_timesteps > 0:
            if self.verbose:
                print(f"  [WARM-START] Solo pre-training evader for "
                      f"{self.warm_start_timesteps} steps at new level")

            train_vec_env, base_envs = _make_vec_env(
                role="evader",
                n_envs=self.n_envs,
                use_dcbf=False,
                history_length=self.history_length,
                n_obstacles=self.n_obstacles,
                seed=self.seed + 9999,
                full_obs=self.full_obs,
                fixed_speed=self.fixed_speed,
                **self.env_kwargs,
            )

            # Set random opponents (None = random policy)
            for base_env in base_envs:
                inner_vec = train_vec_env.venv if hasattr(train_vec_env, "venv") else train_vec_env
                for i, sub_env in enumerate(inner_vec.envs):
                    wrapper = sub_env
                    while hasattr(wrapper, "env"):
                        if isinstance(wrapper, SingleAgentPEWrapper):
                            break
                        wrapper = wrapper.env
                    if isinstance(wrapper, SingleAgentPEWrapper):
                        wrapper.set_opponent(None)  # Random opponent

            # Handle n_envs mismatch
            if self.evader_model.n_envs != train_vec_env.num_envs:
                import tempfile
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp) / "model.zip"
                    self.evader_model.save(tmp_path)
                    self.evader_model = PPO.load(
                        tmp_path,
                        env=train_vec_env,
                        device=self.device,
                    )
            else:
                self.evader_model.set_env(train_vec_env)

            self.evader_model.learn(
                total_timesteps=self.warm_start_timesteps,
                reset_num_timesteps=False,
                progress_bar=True,
            )
            train_vec_env.close()

            if self.verbose:
                print("  [WARM-START] Solo pre-training complete")

    def _ewc_snapshot_evader(self):
        """Take EWC snapshot of evader policy using observations from current level.

        Collects observations by running the evader's current policy in the
        current-level environment, then computes the Fisher information matrix.
        """
        # Create a temporary env at current level to collect observations
        tmp_env, _ = _make_partial_obs_env(
            role="evader",
            use_dcbf=False,
            history_length=self.history_length,
            n_obstacles=self.n_obstacles,
            full_obs=self.full_obs,
            fixed_speed=self.fixed_speed,
            **self.env_kwargs,
        )

        # Collect observations by rolling out the current evader policy
        obs_list = []
        obs, _ = tmp_env.reset()
        for _ in range(self.ewc.fisher_samples + 100):
            obs_list.append(obs)
            action, _ = self.evader_model.predict(obs, deterministic=False)
            obs, _, terminated, truncated, _ = tmp_env.step(action)
            if terminated or truncated:
                obs, _ = tmp_env.reset()
        tmp_env.close()

        # Flatten Dict observations (from PartialObsWrapper) to flat arrays
        samples = obs_list[:self.ewc.fisher_samples]
        if isinstance(samples[0], dict):
            flat = []
            for obs_dict in samples:
                parts = [np.asarray(obs_dict[k]).flatten()
                         for k in sorted(obs_dict.keys())]
                flat.append(np.concatenate(parts))
            obs_batch = torch.FloatTensor(np.array(flat))
        else:
            obs_batch = torch.FloatTensor(np.array(samples))
        obs_batch = obs_batch.to(self.evader_model.device)

        self.ewc.snapshot(self.evader_model, obs_batch)
        if self.verbose:
            print(f"  [EWC] Snapshot taken: {len(obs_batch)} observations, "
                  f"lambda={self.ewc.lambda_}")

    def _wrap_opponent_model(self, opp_model, opponent_role, base_env):
        """Wrap an opponent model with adapters (fixed-speed, partial-obs).

        Args:
            opp_model: PPO model to wrap (or None for random policy).
            opponent_role: 'pursuer' or 'evader'.
            base_env: The base PursuitEvasionEnv (for velocity info).

        Returns:
            Wrapped opponent suitable for SingleAgentPEWrapper.set_opponent().
            Returns None if opp_model is None (random policy).
        """
        if opp_model is None:
            return None  # Random policy — SingleAgentPEWrapper handles None

        if self.full_obs:
            if self.fixed_speed:
                opp_v_max = (base_env.evader_v_max if opponent_role == "evader"
                             else base_env.pursuer_v_max)
                return FixedSpeedModelAdapter(opp_model, v_max=opp_v_max)
            return opp_model
        else:
            model = opp_model
            if self.fixed_speed:
                opp_v_max = (base_env.evader_v_max if opponent_role == "evader"
                             else base_env.pursuer_v_max)
                model = FixedSpeedModelAdapter(model, v_max=opp_v_max)
            return PartialObsOpponentAdapter(
                model=model,
                role=opponent_role,
                base_env=base_env,
                history_length=self.history_length,
                deterministic=False,
            )

    def _train_phase(self, phase: int, role: str):
        """Train one agent while the other is frozen.

        Args:
            phase: Current phase number (1-indexed).
            role: Which agent to train ('pursuer' or 'evader').
        """
        # Create training env for this phase
        # Lazily create RND module on first evader training phase
        rnd_for_env = None
        if role == "evader" and self.rnd_coef > 0:
            if self.rnd_module is None:
                # Determine obs_dim from a temporary env
                tmp_env, _ = _make_partial_obs_env(
                    role="evader",
                    history_length=self.history_length,
                    n_obstacles=self.n_obstacles,
                    full_obs=self.full_obs,
                    fixed_speed=self.fixed_speed,
                    **self.env_kwargs,
                )
                # Compute flattened obs dim (handles Dict obs spaces)
                obs_space = tmp_env.observation_space
                if hasattr(obs_space, 'spaces'):
                    # Dict space: sum of flattened shapes
                    obs_dim = sum(
                        np.prod(s.shape) for s in obs_space.spaces.values()
                    )
                else:
                    obs_dim = obs_space.shape[0]
                tmp_env.close()
                self.rnd_module = RNDModule(
                    obs_dim=obs_dim,
                    embed_dim=self.rnd_embed_dim,
                    hidden_dim=self.rnd_hidden_dim,
                )
                if self.verbose:
                    print(f"  [RND] Created module: obs_dim={obs_dim}, "
                          f"embed={self.rnd_embed_dim}, hidden={self.rnd_hidden_dim}")
            rnd_for_env = self.rnd_module

        train_vec_env, base_envs = _make_vec_env(
            role=role,
            n_envs=self.n_envs,
            use_dcbf=(self.use_dcbf and role == "pursuer"),
            gamma=self.gamma,
            history_length=self.history_length,
            n_obstacles=self.n_obstacles,
            seed=self.seed + phase * 100,
            full_obs=self.full_obs,
            fixed_speed=self.fixed_speed,
            rnd_module=rnd_for_env,
            rnd_coef=self.rnd_coef,
            rnd_update_freq=self.rnd_update_freq,
            **self.env_kwargs,
        )

        # Mixed-level replay: override some sub-envs to use previous level
        if (self.mixed_level_ratio > 0
                and self.curriculum is not None
                and self.curriculum.current_level > 1):
            from training.curriculum import SmoothCurriculumManager
            if isinstance(self.curriculum, SmoothCurriculumManager):
                # Smooth curriculum: step back by one distance_increment
                prev_max = self.curriculum.max_init_distance - self.curriculum.distance_increment
                prev_min = self.curriculum.min_init_distance
                prev_label = f"max_dist={prev_max:.1f}"
            else:
                # Discrete curriculum: use the previous level's config
                prev_level = self.curriculum.levels[self.curriculum.current_level - 1]
                prev_max = prev_level["max_init_distance"]
                prev_min = prev_level["min_init_distance"]
                prev_label = f"L{self.curriculum.current_level - 1}"
            n_mixed = max(1, int(self.n_envs * self.mixed_level_ratio))
            for i in range(n_mixed):
                if i < len(base_envs):
                    base_envs[i].min_init_distance = prev_min
                    base_envs[i].max_init_distance = prev_max
            if self.verbose:
                print(f"  [MIXED-LEVEL] {n_mixed}/{self.n_envs} envs using "
                      f"{prev_label} distances")

        # Determine roles and models
        if role == "pursuer":
            active_model = self.pursuer_model
            opponent_model = self.evader_model
            opponent_role = "evader"
            ckpt_mgr = self.pursuer_ckpt
            opponent_pool = self.evader_pool  # Sample from evader checkpoints
        else:
            active_model = self.evader_model
            opponent_model = self.pursuer_model
            opponent_role = "pursuer"
            ckpt_mgr = self.evader_ckpt
            opponent_pool = self.pursuer_pool  # Sample from pursuer checkpoints

        # Decide per-sub-env opponents
        # If pool is enabled and has checkpoints, sample diverse opponents
        use_pool = (opponent_pool is not None and len(opponent_pool) > 0)
        if use_pool:
            sampled_ckpts = opponent_pool.sample(self.n_envs)
            if self.verbose:
                n_random = sum(1 for s in sampled_ckpts if s is None)
                n_model = self.n_envs - n_random
                print(f"  Opponent pool: {n_model} model(s), {n_random} random")

        # Set frozen opponent in each sub-env
        for i, base_env in enumerate(base_envs):
            if use_pool:
                ckpt_path = sampled_ckpts[i]
                if ckpt_path is None:
                    opp_model = None  # Random policy
                else:
                    opp_model = opponent_pool.get_model(ckpt_path, device=self.device)
            else:
                opp_model = opponent_model

            opponent = self._wrap_opponent_model(opp_model, opponent_role, base_env)

            # Set in the corresponding sub-env
            inner_vec = train_vec_env.venv if hasattr(train_vec_env, "venv") else train_vec_env
            sub_env = inner_vec.envs[i]
            # Traverse to SingleAgentPEWrapper
            wrapper = sub_env
            while hasattr(wrapper, "env"):
                if isinstance(wrapper, SingleAgentPEWrapper):
                    break
                wrapper = wrapper.env
            if isinstance(wrapper, SingleAgentPEWrapper):
                wrapper.set_opponent(opponent)

        # Update active model's environment
        # If n_envs changed (e.g. cold-start used 1 env, self-play uses N),
        # we must save+reload rather than set_env (SB3 requirement).
        if active_model.n_envs != train_vec_env.num_envs:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp) / "model.zip"
                active_model.save(tmp_path)
                active_model = PPO.load(
                    tmp_path,
                    env=train_vec_env,
                    device=self.device,
                )
            # Update reference
            if role == "pursuer":
                self.pursuer_model = active_model
            else:
                self.evader_model = active_model
        else:
            active_model.set_env(train_vec_env)

        # Build callbacks
        callbacks = self._build_callbacks(role, ckpt_mgr, phase)

        # Determine training timesteps
        phase_timesteps = self.timesteps_per_phase

        # Phase warmup: use shorter phases early in training
        if self.phase_warmup and phase in self.phase_warmup_schedule:
            phase_timesteps = self.phase_warmup_schedule[phase]
            if self.verbose:
                print(f"  [PHASE-WARMUP] {phase_timesteps} steps "
                      f"(ramps to {self.timesteps_per_phase} at S{max(self.phase_warmup_schedule.keys()) + 1}+)")

        # Asymmetric evader training at obstacle levels
        if role == "evader" and self.n_obstacles > 0 and self.evader_training_multiplier != 1.0:
            phase_timesteps = int(phase_timesteps * self.evader_training_multiplier)
            if self.verbose:
                print(f"  Asymmetric training: evader gets {phase_timesteps} steps "
                      f"({self.evader_training_multiplier}x at obstacle level)")

        # EWC: register gradient hooks for evader training
        ewc_hooks = []
        if role == "evader" and self.ewc is not None and self.ewc.has_snapshot:
            ewc_hooks = self.ewc.register_hooks(active_model)
            if self.verbose and ewc_hooks:
                penalty = self.ewc.penalty(active_model)
                print(f"  [EWC] Registered {len(ewc_hooks)} hooks, "
                      f"current penalty={penalty:.2f}")

        # Train
        active_model.learn(
            total_timesteps=phase_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        # EWC: remove hooks after training
        if ewc_hooks:
            EWCRegularizer.remove_hooks(ewc_hooks)
            if self.verbose:
                penalty = self.ewc.penalty(active_model)
                print(f"  [EWC] Post-training penalty={penalty:.2f}")

        # Save milestone checkpoint
        ckpt_mgr.save_milestone(
            model=active_model,
            phase=phase,
            role=role,
        )

        # Add milestone to opponent pool for the trained role
        if self.opponent_pool_size > 0:
            milestone_dir = ckpt_mgr.checkpoint_dir / f"milestone_phase{phase}_{role}"
            if role == "pursuer" and self.pursuer_pool is not None:
                self.pursuer_pool.add_checkpoint(str(milestone_dir))
            elif role == "evader" and self.evader_pool is not None:
                self.evader_pool.add_checkpoint(str(milestone_dir))

        train_vec_env.close()

    def _run_micro_phases(self):
        """Rapid alternation self-play with micro-phases (RA redesign).

        Instead of long frozen phases (100K-500K steps), alternate training
        every micro_phase_steps (e.g. 2048 = 1 PPO rollout). Environments
        are created once and reused. Opponent weights are synced in-place.
        """
        start_time = time.time()

        if self.verbose:
            rollout_steps = self.n_envs * self.n_steps
            print(f"\n{'=' * 60}")
            print("Micro-Phase Rapid Alternation Mode")
            print(f"  Steps per micro-phase: {self.micro_phase_steps}")
            print(f"  PPO rollout size: {rollout_steps} "
                  f"(n_envs={self.n_envs} x n_steps={self.n_steps})")
            print(f"  Eval interval: every {self.eval_interval_micro} micro-phases")
            print(f"  Snapshot interval: every {self.snapshot_freq_micro} micro-phases")
            print(f"  Convergence: {self.convergence_consecutive} consecutive "
                  f"evals with gap < {self.eta}")
            if self.max_total_steps > 0:
                print(f"  Max total steps: {self.max_total_steps:,}")
            if self.opponent_pool_size > 0:
                strategy = self.pursuer_pool.eviction_strategy
                print(f"  Opponent pool: size={self.opponent_pool_size}, "
                      f"strategy={strategy}")
            if self.adaptive_ratio_threshold > 0:
                print(f"  Adaptive ratio: threshold={self.adaptive_ratio_threshold}, "
                      f"boost_phases={self.adaptive_boost_phases}")
            if self.lr_dampen_threshold > 0:
                print(f"  LR dampening: threshold={self.lr_dampen_threshold}")
            if self.collapse_threshold > 0:
                print(f"  Collapse rollback: threshold={self.collapse_threshold}, "
                      f"streak_limit={self.collapse_streak_limit}")
            if self.pfsp_enabled:
                print(f"  PFSP-lite: enabled (bias toward weaker opponents when losing)")
            print(f"{'=' * 60}")

        # 1. Create persistent environment sets
        pursuer_vec_env, pursuer_base_envs = _make_vec_env(
            role="pursuer",
            n_envs=self.n_envs,
            use_dcbf=(self.use_dcbf),
            gamma=self.gamma,
            history_length=self.history_length,
            n_obstacles=self.n_obstacles,
            seed=self.seed + 1000,
            full_obs=self.full_obs,
            fixed_speed=self.fixed_speed,
            **self.env_kwargs,
        )
        evader_vec_env, evader_base_envs = _make_vec_env(
            role="evader",
            n_envs=self.n_envs,
            use_dcbf=False,
            gamma=self.gamma,
            history_length=self.history_length,
            n_obstacles=self.n_obstacles,
            seed=self.seed + 2000,
            full_obs=self.full_obs,
            fixed_speed=self.fixed_speed,
            **self.env_kwargs,
        )

        # 2. Handle n_envs mismatch from cold-start (which uses 1 env)
        #    SB3 requires save/reload when n_envs changes.
        for role_name, vec_env in [("pursuer", pursuer_vec_env),
                                    ("evader", evader_vec_env)]:
            model = self.pursuer_model if role_name == "pursuer" else self.evader_model
            if model.n_envs != vec_env.num_envs:
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp) / "model.zip"
                    model.save(tmp_path)
                    reloaded = PPO.load(tmp_path, env=vec_env, device=self.device)
                if role_name == "pursuer":
                    self.pursuer_model = reloaded
                else:
                    self.evader_model = reloaded
                if self.verbose:
                    print(f"  [{role_name}] Reloaded model for n_envs={vec_env.num_envs}")
            else:
                model.set_env(vec_env)

        # 3. Set initial opponents in each sub-env
        self._set_opponents_in_vec_env(
            pursuer_vec_env, pursuer_base_envs, self.evader_model, "evader")
        self._set_opponents_in_vec_env(
            evader_vec_env, evader_base_envs, self.pursuer_model, "pursuer")

        # 4. Main micro-phase loop
        total_steps = 0
        role = "pursuer"
        converged = False
        convergence_streak = 0  # consecutive evals with gap < eta

        # Adaptive training ratio state
        boost_remaining = 0
        boost_role = "pursuer"
        last_gap = None

        # Collapse rollback state
        best_sr_pursuer = 0.0
        best_sr_evader = 0.0
        best_ckpt_pursuer = None  # (micro, path)
        best_ckpt_evader = None
        collapse_streak_pursuer = 0
        collapse_streak_evader = 0

        # Calculate max micro-phases
        if self.max_total_steps > 0:
            max_micro = self.max_total_steps // self.micro_phase_steps
        else:
            max_micro = self.max_phases * (self.timesteps_per_phase // self.micro_phase_steps)

        if self.verbose:
            print(f"  Max micro-phases: {max_micro}")
            print(f"\n--- Starting micro-phase training ---")

        last_micro = 0
        for micro in range(1, max_micro + 1):
            last_micro = micro

            # Select active model
            if role == "pursuer":
                active_model = self.pursuer_model
            else:
                active_model = self.evader_model

            # LR dampening: reduce LR when gap is small (near equilibrium)
            lr_scale = 1.0
            if self.lr_dampen_threshold > 0 and last_gap is not None:
                if last_gap < self.lr_dampen_threshold:
                    lr_scale = max(0.1, last_gap / self.lr_dampen_threshold)
                    dampened_lr = self.learning_rate * lr_scale
                    active_model.lr_schedule = lambda _, lr=dampened_lr: lr
                else:
                    base_lr = self.learning_rate
                    active_model.lr_schedule = lambda _, lr=base_lr: lr

            # Train for one micro-phase
            active_model.learn(
                total_timesteps=self.micro_phase_steps,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            total_steps += self.micro_phase_steps

            # Restore full LR after training step
            if lr_scale < 1.0:
                base_lr = self.learning_rate
                active_model.lr_schedule = lambda _, lr=base_lr: lr

            # Sync opponent weights: copy just-trained model into the other env
            if role == "pursuer":
                self._sync_opponent_weights_vec(evader_vec_env, self.pursuer_model)
            else:
                self._sync_opponent_weights_vec(pursuer_vec_env, self.evader_model)

            # Snapshot to opponent pool
            if (self.opponent_pool_size > 0
                    and micro % self.snapshot_freq_micro == 0):
                self._save_micro_snapshot(role, micro)

                # Resample diverse opponents for both env sets
                # PFSP-lite: bias toward weaker opponents when agent is collapsing
                pfsp_pursuer = (self.pfsp_enabled
                                and collapse_streak_pursuer > 0)
                pfsp_evader = (self.pfsp_enabled
                               and collapse_streak_evader > 0)
                self._resample_pool_opponents_vec(
                    pursuer_vec_env, pursuer_base_envs,
                    self.evader_pool, "evader",
                    current_model=self.evader_model,
                    use_pfsp=pfsp_pursuer,
                )
                self._resample_pool_opponents_vec(
                    evader_vec_env, evader_base_envs,
                    self.pursuer_pool, "pursuer",
                    current_model=self.pursuer_model,
                    use_pfsp=pfsp_evader,
                )

            # Periodic evaluation
            if micro % self.eval_interval_micro == 0:
                metrics = self._evaluate()
                sr_p = metrics["capture_rate"]
                sr_e = metrics["escape_rate"]
                ne_gap = abs(sr_p - sr_e)

                elapsed = time.time() - start_time
                entry = {
                    "phase": f"M{micro}",
                    "role": role,
                    "total_steps": total_steps,
                    **metrics,
                }

                # Curriculum handling
                if self.curriculum:
                    entry.update(self.curriculum.get_status())

                    if self.ne_gap_advancement:
                        self.curriculum.phases_at_level += 1
                        self.curriculum.level_history.append({
                            "level": self.curriculum.current_level,
                            "capture_rate": sr_p,
                            "escape_rate": sr_e,
                            "ne_gap": ne_gap,
                            "advanced": False,
                        })
                        if ne_gap < self.ne_gap_threshold:
                            self._ne_gap_streak += 1
                        else:
                            self._ne_gap_streak = 0
                        if (self._ne_gap_streak >= self.ne_gap_consecutive
                                and not self.curriculum.at_max_level):
                            self.curriculum._advance()
                            self.curriculum.level_history[-1]["advanced"] = True
                            self._ne_gap_streak = 0
                            overrides = self.curriculum.get_env_overrides()
                            self.env_kwargs["min_init_distance"] = overrides["min_init_distance"]
                            self.env_kwargs["max_init_distance"] = overrides["max_init_distance"]
                            self.n_obstacles = overrides["n_obstacles"]
                            if self.verbose:
                                print(f"  [NE-GAP ADV] Curriculum advanced at M{micro}")
                                print(f"    New: dist={overrides['min_init_distance']:.1f}-"
                                      f"{overrides['max_init_distance']:.1f}m, "
                                      f"obstacles={overrides['n_obstacles']}")

                self.history.append(entry)

                if self.verbose:
                    status = (f"  [M{micro:>5d}] steps={total_steps:>9,} | "
                              f"SR_P={sr_p:.3f} SR_E={sr_e:.3f} gap={ne_gap:.3f} | "
                              f"t={elapsed/60:.0f}m")
                    if self.curriculum:
                        status += f" | L={self.curriculum.current_level}"
                    if self.opponent_pool_size > 0:
                        p_pool = len(self.pursuer_pool) if self.pursuer_pool else 0
                        e_pool = len(self.evader_pool) if self.evader_pool else 0
                        status += f" | pool=P{p_pool}/E{e_pool}"
                    if boost_remaining > 0:
                        status += f" | boost={boost_role}({boost_remaining})"
                    if lr_scale < 1.0:
                        status += f" | lr_scale={lr_scale:.2f}"
                    print(status)

                # Save checkpoint at eval intervals
                self.pursuer_ckpt.save_milestone(
                    model=self.pursuer_model, phase=micro, role="pursuer",
                )
                self.evader_ckpt.save_milestone(
                    model=self.evader_model, phase=micro, role="evader",
                )

                # Save history incrementally
                self._save_history_incremental(
                    converged=False, elapsed=time.time() - start_time)

                # Check convergence (require consecutive evals below threshold)
                # Uses convergence_consecutive (not ne_gap_consecutive, which
                # controls curriculum advancement). Default 5 prevents premature
                # convergence declaration from transient balanced evals.
                curriculum_ready = (self.curriculum is None) or self.curriculum.at_max_level
                if ne_gap < self.eta and curriculum_ready:
                    convergence_streak += 1
                    if convergence_streak >= self.convergence_consecutive:
                        converged = True
                        if self.verbose:
                            print(f"  *** Converged at M{micro}! "
                                  f"(NE gap < {self.eta} for "
                                  f"{self.convergence_consecutive} consecutive evals) ***")
                        break
                    elif self.verbose:
                        print(f"    (convergence streak: {convergence_streak}"
                              f"/{self.convergence_consecutive})")
                else:
                    convergence_streak = 0

                # Collapse rollback: restore best checkpoint when agent degenerates
                if self.collapse_threshold > 0:
                    # Track best performance for each agent
                    if sr_p > best_sr_pursuer:
                        best_sr_pursuer = sr_p
                        best_ckpt_pursuer = (micro, str(
                            self.pursuer_ckpt.checkpoint_dir
                            / f"milestone_phase{micro}_pursuer"))
                    if sr_e > best_sr_evader:
                        best_sr_evader = sr_e
                        best_ckpt_evader = (micro, str(
                            self.evader_ckpt.checkpoint_dir
                            / f"milestone_phase{micro}_evader"))

                    # Check pursuer collapse
                    if sr_p < self.collapse_threshold:
                        collapse_streak_pursuer += 1
                    else:
                        collapse_streak_pursuer = 0

                    # Check evader collapse
                    if sr_e < self.collapse_threshold:
                        collapse_streak_evader += 1
                    else:
                        collapse_streak_evader = 0

                    # Rollback pursuer if collapsed
                    if (collapse_streak_pursuer >= self.collapse_streak_limit
                            and best_ckpt_pursuer is not None):
                        ckpt_micro, ckpt_path = best_ckpt_pursuer
                        ckpt_file = Path(ckpt_path) / "ppo.zip"
                        if ckpt_file.exists():
                            if self.verbose:
                                print(f"    [ROLLBACK] Pursuer collapsed "
                                      f"(SR={sr_p:.3f} < {self.collapse_threshold} "
                                      f"for {collapse_streak_pursuer} evals). "
                                      f"Restoring M{ckpt_micro} "
                                      f"(best SR={best_sr_pursuer:.3f})")
                            self.pursuer_model = PPO.load(
                                ckpt_file, env=pursuer_vec_env,
                                device=self.device)
                            self._sync_opponent_weights_vec(
                                evader_vec_env, self.pursuer_model)
                            collapse_streak_pursuer = 0

                    # Rollback evader if collapsed
                    if (collapse_streak_evader >= self.collapse_streak_limit
                            and best_ckpt_evader is not None):
                        ckpt_micro, ckpt_path = best_ckpt_evader
                        ckpt_file = Path(ckpt_path) / "ppo.zip"
                        if ckpt_file.exists():
                            if self.verbose:
                                print(f"    [ROLLBACK] Evader collapsed "
                                      f"(SR={sr_e:.3f} < {self.collapse_threshold} "
                                      f"for {collapse_streak_evader} evals). "
                                      f"Restoring M{ckpt_micro} "
                                      f"(best SR={best_sr_evader:.3f})")
                            self.evader_model = PPO.load(
                                ckpt_file, env=evader_vec_env,
                                device=self.device)
                            self._sync_opponent_weights_vec(
                                pursuer_vec_env, self.evader_model)
                            collapse_streak_evader = 0

                # Adaptive training ratio: give loser extra phases
                if (self.adaptive_ratio_threshold > 0
                        and ne_gap > self.adaptive_ratio_threshold):
                    # Determine who is losing
                    if sr_p > sr_e:
                        boost_role = "evader"
                    else:
                        boost_role = "pursuer"
                    boost_remaining = self.adaptive_boost_phases
                    if self.verbose:
                        print(f"    [BOOST] {boost_role} gets "
                              f"{boost_remaining} extra phases "
                              f"(gap={ne_gap:.3f} > {self.adaptive_ratio_threshold})")

                # Track gap for LR dampening
                last_gap = ne_gap
            else:
                # Compact progress line every 10 micro-phases
                if micro % 10 == 0 and self.verbose:
                    elapsed = time.time() - start_time
                    steps_per_sec = total_steps / max(elapsed, 1)
                    print(f"  [M{micro:>5d}] steps={total_steps:>9,} "
                          f"role={role:>7s} | {steps_per_sec:.0f} steps/s")

            # Alternate role (with adaptive boost override)
            if boost_remaining > 0:
                role = boost_role
                boost_remaining -= 1
            else:
                role = "evader" if role == "pursuer" else "pursuer"

        # Cleanup
        pursuer_vec_env.close()
        evader_vec_env.close()

        # Final save
        self._save_final()
        elapsed = time.time() - start_time

        result = {
            "converged": converged,
            "total_micro_phases": last_micro,
            "total_steps": total_steps,
            "history": self.history,
            "elapsed_seconds": elapsed,
        }
        if self.curriculum:
            result["curriculum_final_level"] = self.curriculum.current_level
            result["curriculum_history"] = self.curriculum.level_history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Micro-phase training complete. Converged: {converged}")
            print(f"Total steps: {total_steps:,}")
            print(f"Total time: {elapsed / 3600:.1f}h")
            print(f"Results saved to: {self.output_dir}")
            print(f"{'=' * 60}")

        return {
            "pursuer_model": self.pursuer_model,
            "evader_model": self.evader_model,
            "history": self.history,
            "converged": converged,
        }

    def _set_opponents_in_envs(self, base_envs, opponent_model, opponent_role):
        """Deprecated stub — use _set_opponents_in_vec_env instead."""
        raise NotImplementedError("Use _set_opponents_in_vec_env")

    def _set_opponents_in_vec_env(self, vec_env, base_envs, opponent_model, opponent_role):
        """Set frozen opponent in each sub-env of a vectorized env.

        Args:
            vec_env: The DummyVecEnv/VecMonitor wrapper.
            base_envs: List of base PursuitEvasionEnv instances.
            opponent_model: Model to use as opponent (or None for random).
            opponent_role: 'pursuer' or 'evader'.
        """
        # Access underlying DummyVecEnv
        inner_vec = vec_env.venv if hasattr(vec_env, "venv") else vec_env

        use_pool = False
        opponent_pool = None
        if opponent_role == "evader" and self.evader_pool is not None and len(self.evader_pool) > 0:
            opponent_pool = self.evader_pool
            use_pool = True
        elif opponent_role == "pursuer" and self.pursuer_pool is not None and len(self.pursuer_pool) > 0:
            opponent_pool = self.pursuer_pool
            use_pool = True

        if use_pool:
            sampled_ckpts = opponent_pool.sample(len(base_envs))
        else:
            sampled_ckpts = [None] * len(base_envs)

        for i, base_env in enumerate(base_envs):
            if use_pool:
                ckpt_path = sampled_ckpts[i]
                if ckpt_path is None:
                    opp_model = opponent_model  # Use current model instead of random
                else:
                    opp_model = opponent_pool.get_model(ckpt_path, device=self.device)
            else:
                opp_model = opponent_model

            opponent = self._wrap_opponent_model(opp_model, opponent_role, base_env)

            # Traverse wrapper stack to find SingleAgentPEWrapper
            sub_env = inner_vec.envs[i]
            wrapper = sub_env
            while hasattr(wrapper, "env"):
                if isinstance(wrapper, SingleAgentPEWrapper):
                    break
                wrapper = wrapper.env
            if isinstance(wrapper, SingleAgentPEWrapper):
                wrapper.set_opponent(opponent)

    def _sync_opponent_weights(self, base_envs, updated_model):
        """Deprecated stub — use _sync_opponent_weights_vec instead."""
        raise NotImplementedError("Use _sync_opponent_weights_vec")

    def _sync_opponent_weights_vec(self, vec_env, updated_model):
        """Copy updated model weights into opponent adapters in a vec_env.

        Updates the policy state_dict of the opponent model in each sub-env
        without recreating the env or adapter objects.
        """
        inner_vec = vec_env.venv if hasattr(vec_env, "venv") else vec_env

        for i in range(len(inner_vec.envs)):
            sub_env = inner_vec.envs[i]
            # Traverse to SingleAgentPEWrapper
            wrapper = sub_env
            while hasattr(wrapper, "env"):
                if isinstance(wrapper, SingleAgentPEWrapper):
                    break
                wrapper = wrapper.env

            if not isinstance(wrapper, SingleAgentPEWrapper):
                continue

            opp = wrapper.opponent_policy
            if opp is None:
                continue

            # Extract the underlying PPO model from any adapter
            target_model = None
            if isinstance(opp, FixedSpeedModelAdapter):
                target_model = opp.model
            elif isinstance(opp, PartialObsOpponentAdapter):
                inner = opp.model
                if isinstance(inner, FixedSpeedModelAdapter):
                    target_model = inner.model
                else:
                    target_model = inner
            elif isinstance(opp, PPO):
                target_model = opp

            if target_model is not None and hasattr(target_model, "policy"):
                target_model.policy.load_state_dict(
                    updated_model.policy.state_dict()
                )

    def _save_micro_snapshot(self, role, micro):
        """Save a snapshot to the opponent pool during micro-phase training."""
        if role == "pursuer":
            model = self.pursuer_model
            ckpt_mgr = self.pursuer_ckpt
            pool = self.pursuer_pool
        else:
            model = self.evader_model
            ckpt_mgr = self.evader_ckpt
            pool = self.evader_pool

        if pool is None:
            return

        # Save as milestone checkpoint
        ckpt_mgr.save_milestone(model=model, phase=micro, role=role)
        milestone_dir = ckpt_mgr.checkpoint_dir / f"milestone_phase{micro}_{role}"
        pool.add_checkpoint(str(milestone_dir))

    def _resample_pool_opponents(self, base_envs_target, opponent_pool,
                                  opponent_role, current_model):
        """Deprecated stub — use _resample_pool_opponents_vec instead."""
        raise NotImplementedError("Use _resample_pool_opponents_vec")

    def _resample_pool_opponents_vec(self, vec_env, base_envs, opponent_pool,
                                      opponent_role, current_model,
                                      use_pfsp: bool = False):
        """Resample opponents from pool into vec_env sub-envs.

        50% of sub-envs get current model weights, 50% sample from pool.
        When use_pfsp=True, pool sampling biases toward older/weaker opponents.
        """
        if opponent_pool is None or len(opponent_pool) == 0:
            return

        inner_vec = vec_env.venv if hasattr(vec_env, "venv") else vec_env
        n = len(base_envs)

        for i, base_env in enumerate(base_envs):
            if i % 2 == 0:
                # Current model
                opp_model = current_model
            else:
                # Sample from pool (PFSP-lite or uniform)
                if use_pfsp:
                    sampled = opponent_pool.sample_pfsp(1)[0]
                else:
                    sampled = opponent_pool.sample(1)[0]
                if sampled is None:
                    opp_model = current_model
                else:
                    opp_model = opponent_pool.get_model(sampled, device=self.device)

            opponent = self._wrap_opponent_model(opp_model, opponent_role, base_env)

            sub_env = inner_vec.envs[i]
            wrapper = sub_env
            while hasattr(wrapper, "env"):
                if isinstance(wrapper, SingleAgentPEWrapper):
                    break
                wrapper = wrapper.env
            if isinstance(wrapper, SingleAgentPEWrapper):
                wrapper.set_opponent(opponent)

    def _save_history_incremental(self, converged, elapsed):
        """Save history to disk incrementally for crash recovery."""
        result = {
            "converged": converged,
            "history": self.history,
            "elapsed_seconds": elapsed,
        }
        if self.curriculum:
            result["curriculum_final_level"] = self.curriculum.current_level
            result["curriculum_history"] = self.curriculum.level_history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    def _build_callbacks(self, role: str, ckpt_mgr: CheckpointManager, phase: int):
        """Build callback list for a training phase."""
        # Determine opponent references for bilateral rollback
        if role == "pursuer":
            opp_ckpt_mgr = self.evader_ckpt
            opp_model_ref = [self.evader_model]
        else:
            opp_ckpt_mgr = self.pursuer_ckpt
            opp_model_ref = [self.pursuer_model]

        callback_list = [
            EntropyMonitorCallback(check_freq=2048, log_std_floor=-2.0),
            SelfPlayHealthMonitorCallback(
                checkpoint_manager=ckpt_mgr,
                checkpoint_freq=10_000,
                bilateral_rollback=self.bilateral_rollback,
                opponent_ckpt_mgr=opp_ckpt_mgr,
                opponent_model_ref=opp_model_ref,
                verbose=self.verbose,
            ),
        ]

        # Baseline evaluation (only for pursuer — evaluates against scripted evaders)
        if role == "pursuer":
            _rk = {"w_occlusion", "use_visibility_reward", "visibility_weight", "survival_bonus", "w_obs_approach", "partial_obs_los"}
            baseline_kwargs = {k: v for k, v in self.env_kwargs.items() if k not in _rk}
            eval_env = PursuitEvasionEnv(
                **baseline_kwargs, n_obstacles=self.n_obstacles
            )
            if self.full_obs:
                # Full-obs: model.predict works directly with flat obs
                agent_adapter = None
            else:
                # Partial-obs: need adapter to convert full-state → Dict obs
                agent_adapter = PartialObsOpponentAdapter(
                    model=self.pursuer_model,
                    role="pursuer",
                    base_env=eval_env,
                    history_length=self.history_length,
                    deterministic=True,
                )
            callback_list.append(
                FixedBaselineEvalCallback(
                    eval_env=eval_env,
                    baselines={
                        "random": None,
                        "flee_away": flee_away_policy,
                        "flee_to_corner": flee_to_corner_policy,
                    },
                    role="pursuer",
                    lightweight_eval_freq=10_000,
                    full_eval_freq=50_000,
                    n_eval_episodes=20,
                    agent_adapter=agent_adapter,
                    fixed_speed_v_max=(
                        self.env_kwargs["pursuer_v_max"] if self.fixed_speed else None
                    ),
                )
            )

        return CallbackList(callback_list)

    def _evaluate(self) -> dict:
        """Evaluate current pursuer vs evader head-to-head."""
        if self.full_obs:
            return _evaluate_head_to_head_full_obs(
                pursuer_model=self.pursuer_model,
                evader_model=self.evader_model,
                n_episodes=self.eval_episodes,
                seed=self.seed + len(self.history) * 1000,
                n_obstacles=self.n_obstacles,
                fixed_speed=self.fixed_speed,
                **self.env_kwargs,
            )
        return _evaluate_head_to_head(
            pursuer_model=self.pursuer_model,
            evader_model=self.evader_model,
            n_episodes=self.eval_episodes,
            seed=self.seed + len(self.history) * 1000,
            n_obstacles=self.n_obstacles,
            fixed_speed=self.fixed_speed,
            **self.env_kwargs,
        )

    def _save_final(self):
        """Save final models and metadata."""
        final_dir = self.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        self.pursuer_model.save(str(final_dir / "pursuer"))
        self.evader_model.save(str(final_dir / "evader"))

        if self.verbose:
            print(f"  Final models saved to: {final_dir}")
