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
from envs.navigation_env import NavigationEnv
from envs.opponent_adapter import PartialObsOpponentAdapter
from envs.partial_obs_wrapper import PartialObsWrapper
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import FixedSpeedModelAdapter, FixedSpeedWrapper, SingleAgentPEWrapper
from training.checkpoint_manager import CheckpointManager
from training.curriculum import CurriculumManager
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
    w_collision: float = 0.0,
    w_wall: float = 0.0,
) -> tuple:
    """Create an environment stack for one agent.

    Args:
        full_obs: If True, skip PartialObsWrapper (full-state observations).
        distance_scale: Scale factor for dense distance reward.
        pursuer_v_max: Maximum linear velocity for pursuer.
        fixed_speed: If True, fix v=v_max and only learn omega (1D action).
        min_init_distance: Minimum initial agent separation.
        max_init_distance: Maximum initial agent separation.
        w_occlusion: Weight on evader occlusion bonus (Mode A, 0 = off).
        use_visibility_reward: Enable visibility-based evader reward (Mode B).
        visibility_weight: Scale for +1/-1 visibility signal (Mode B).
        survival_bonus: Per-step survival bonus for evader (Mode B).
        prep_steps: Freeze pursuer for first N steps per episode (0 = off).
        w_obs_approach: PBRS obstacle-seeking weight for evader (0 = off).

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
    )

    base_env = PursuitEvasionEnv(
        arena_width=arena_width,
        arena_height=arena_height,
        max_steps=max_steps,
        capture_radius=capture_radius,
        n_obstacles=n_obstacles,
        pursuer_v_max=pursuer_v_max,
        min_init_distance=min_init_distance,
        max_init_distance=max_init_distance,
        reward_computer=reward_computer,
        prep_steps=prep_steps,
        w_collision=w_collision,
        w_wall=w_wall,
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
    fixed_speed: bool = False,
    min_init_distance: float = 3.0,
    max_init_distance: float = 15.0,
    prep_steps: int = 0,
    w_collision: float = 0.0,
    w_wall: float = 0.0,
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
        min_init_distance=min_init_distance,
        max_init_distance=max_init_distance,
        prep_steps=prep_steps,
        w_collision=w_collision,
        w_wall=w_wall,
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
    fixed_speed: bool = False,
    min_init_distance: float = 3.0,
    max_init_distance: float = 15.0,
    prep_steps: int = 0,
    w_collision: float = 0.0,
    w_wall: float = 0.0,
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
        min_init_distance=min_init_distance,
        max_init_distance=max_init_distance,
        prep_steps=prep_steps,
        w_collision=w_collision,
        w_wall=w_wall,
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
        w_collision: float = 0.0,
        w_wall: float = 0.0,
        evader_training_multiplier: float = 1.0,
        min_escape_rate: float = 0.0,
        min_phases_per_level: int = 1,
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
            "w_occlusion": w_occlusion,
            "use_visibility_reward": use_visibility_reward,
            "visibility_weight": visibility_weight,
            "survival_bonus": survival_bonus,
            "prep_steps": prep_steps,
            "w_obs_approach": w_obs_approach,
            "timeout_penalty": timeout_penalty,
            "w_collision": w_collision,
            "w_wall": w_wall,
        }
        self.fixed_speed = fixed_speed
        self.evader_training_multiplier = evader_training_multiplier

        # Curriculum learning
        self.curriculum = None
        if curriculum:
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
                max_size=opponent_pool_size, include_random=True
            )
            self.evader_pool = OpponentPool(
                max_size=opponent_pool_size, include_random=True
            )
        else:
            self.pursuer_pool = None
            self.evader_pool = None

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
            print(f"  Fixed speed: {self.fixed_speed}")
            print(f"  Curriculum: {self.curriculum is not None}")
            print(f"  Opponent pool: {self.opponent_pool_size if self.opponent_pool_size > 0 else 'disabled'}")
            prep = self.env_kwargs.get("prep_steps", 0)
            if prep > 0:
                print(f"  Prep phase: {prep} steps (pursuer frozen)")
            if self.env_kwargs.get("use_visibility_reward", False):
                print(f"  Reward mode: VISIBILITY (OpenAI H&S style)")
                print(f"    Visibility weight: {self.env_kwargs.get('visibility_weight', 1.0)}")
                print(f"    Survival bonus: {self.env_kwargs.get('survival_bonus', 0.0)}")
            else:
                w_occ = self.env_kwargs.get("w_occlusion", 0.0)
                if w_occ > 0:
                    print(f"  Occlusion bonus: {w_occ}")
            print(f"  Seed: {self.seed}")
            print(f"  Device: {self.device}")
            print("=" * 60)

        # ─── Phase S0: Cold-Start ───
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

        # ─── Alternating Training Phases ───
        converged = False
        for phase in range(1, self.max_phases + 1):
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
                advanced = self.curriculum.check_advancement(
                    metrics["capture_rate"], metrics["escape_rate"]
                )
                if advanced:
                    # Apply new level's env overrides for subsequent phases
                    overrides = self.curriculum.get_env_overrides()
                    self.env_kwargs["min_init_distance"] = overrides["min_init_distance"]
                    self.env_kwargs["max_init_distance"] = overrides["max_init_distance"]
                    self.n_obstacles = overrides["n_obstacles"]
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
        pe_kwargs = {k: v for k, v in self.env_kwargs.items() if k not in _reward_keys}
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

        if self.verbose:
            print(f"  Cold-start complete: {self.cold_start_timesteps} steps")

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
            **self.env_kwargs,
        )

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

        # Determine training timesteps (asymmetric for evader at obstacle levels)
        phase_timesteps = self.timesteps_per_phase
        if role == "evader" and self.n_obstacles > 0 and self.evader_training_multiplier != 1.0:
            phase_timesteps = int(self.timesteps_per_phase * self.evader_training_multiplier)
            if self.verbose:
                print(f"  Asymmetric training: evader gets {phase_timesteps} steps "
                      f"({self.evader_training_multiplier}x at obstacle level)")

        # Train
        active_model.learn(
            total_timesteps=phase_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=True,
        )

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

    def _build_callbacks(self, role: str, ckpt_mgr: CheckpointManager, phase: int):
        """Build callback list for a training phase."""
        callback_list = [
            EntropyMonitorCallback(check_freq=2048, log_std_floor=-2.0),
            SelfPlayHealthMonitorCallback(
                checkpoint_manager=ckpt_mgr,
                checkpoint_freq=10_000,
                verbose=self.verbose,
            ),
        ]

        # Baseline evaluation (only for pursuer — evaluates against scripted evaders)
        if role == "pursuer":
            _rk = {"w_occlusion", "use_visibility_reward", "visibility_weight", "survival_bonus", "w_obs_approach"}
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
