"""Vanilla alternating self-play training loop.

Protocol:
1. Initialize both agents randomly
2. Train Pursuer for N steps while Evader is frozen
3. Train Evader for N steps while Pursuer is frozen
4. Repeat until convergence

This does NOT guarantee NE convergence (requires AMS-DRL in Phase 3),
but is sufficient for learning basic PE behaviors.
"""

import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from envs.pursuit_evasion_env import PursuitEvasionEnv
from training.health_monitor import SelfPlayHealthMonitor
from training.self_play_eval import evaluate_both_agents, collect_trajectories
from training.tracking import PursuitEvasionMetricsCallback
from training.utils import make_vec_env, setup_reproducibility


def _build_ppo(env, cfg: DictConfig, seed: int, run_id: str, device: str = "cpu"):
    """Create a PPO model from config."""
    algo_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    policy_kwargs = algo_cfg.pop("policy_kwargs", {})
    policy_kwargs["net_arch"] = list(policy_kwargs["net_arch"])
    policy_kwargs["activation_fn"] = torch.nn.Tanh

    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log=f"runs/{run_id}",
        seed=seed,
        verbose=0,
        device=device,
        policy_kwargs=policy_kwargs,
        **algo_cfg,
    )
    return model


def alternating_self_play(cfg: DictConfig) -> dict:
    """Run vanilla alternating self-play training.

    Args:
        cfg: Full Hydra config with env, algorithm, self_play, wandb sections.

    Returns:
        Dict with 'pursuer_model', 'evader_model', 'history'.
    """
    use_cuda = torch.cuda.is_available()
    setup_reproducibility(cfg.seed, use_cuda=use_cuda)
    device = "cuda" if use_cuda else "cpu"

    run_id = f"selfplay_{cfg.seed}_{int(time.time())}"
    sp_cfg = cfg.self_play

    # Create base eval environment (not wrapped — we step both agents manually)
    eval_env = PursuitEvasionEnv(
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
        render_mode=None,
    )

    # Create training envs for each role (opponent will be updated each phase)
    pursuer_train_env = make_vec_env(
        cfg, n_envs=cfg.n_envs, role="pursuer", opponent_policy=None, seed=cfg.seed,
    )
    evader_train_env = make_vec_env(
        cfg, n_envs=cfg.n_envs, role="evader", opponent_policy=None, seed=cfg.seed + 1000,
    )

    # Create PPO models
    pursuer_model = _build_ppo(pursuer_train_env, cfg, cfg.seed, f"{run_id}/pursuer", device)
    evader_model = _build_ppo(evader_train_env, cfg, cfg.seed + 1, f"{run_id}/evader", device)

    # Health monitor
    health_monitor = SelfPlayHealthMonitor(
        min_entropy=sp_cfg.health.min_entropy,
        max_capture_rate=sp_cfg.health.max_capture_rate,
        min_capture_rate=sp_cfg.health.min_capture_rate,
        greedy_eval_interval=sp_cfg.health.greedy_eval_interval,
        max_checkpoints=sp_cfg.health.max_checkpoints,
    )

    # History tracking
    history = {
        "capture_rate": [],
        "escape_rate": [],
        "mean_episode_length": [],
        "health_alerts": [],
    }

    # Checkpoint storage
    checkpoint_dir = Path(f"models/{run_id}/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting self-play: {sp_cfg.n_phases} phases, "
          f"{sp_cfg.timesteps_per_phase} steps/phase, seed={cfg.seed}")

    for phase in range(sp_cfg.n_phases):
        role = "pursuer" if phase % 2 == 0 else "evader"
        print(f"\n--- Phase {phase}/{sp_cfg.n_phases - 1}: Training {role} ---")

        # Update opponent policy in training envs
        if phase % 2 == 0:
            # Training pursuer — freeze evader as opponent
            _update_opponent_in_vec_env(pursuer_train_env, evader_model)
            active_model = pursuer_model
            active_env = pursuer_train_env
        else:
            # Training evader — freeze pursuer as opponent
            _update_opponent_in_vec_env(evader_train_env, pursuer_model)
            active_model = evader_model
            active_env = evader_train_env

        # Train active agent
        callbacks = CallbackList([
            PursuitEvasionMetricsCallback(log_frequency=cfg.wandb.log_frequency),
        ])

        active_model.learn(
            total_timesteps=sp_cfg.timesteps_per_phase,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        # Evaluate both agents against each other
        metrics = evaluate_both_agents(
            pursuer_model, evader_model, eval_env,
            n_episodes=sp_cfg.eval_episodes, seed=cfg.seed + phase * 100,
        )
        history["capture_rate"].append(metrics["capture_rate"])
        history["escape_rate"].append(metrics["escape_rate"])
        history["mean_episode_length"].append(metrics["mean_episode_length"])

        print(f"  Capture rate: {metrics['capture_rate']:.2f}, "
              f"Escape rate: {metrics['escape_rate']:.2f}, "
              f"Mean ep len: {metrics['mean_episode_length']:.0f}")

        # Health checks
        alerts = []

        # Entropy check
        pursuer_entropy = health_monitor.check_entropy(pursuer_model)
        evader_entropy = health_monitor.check_entropy(evader_model)
        if pursuer_entropy is not None and pursuer_entropy < sp_cfg.health.min_entropy:
            alerts.append(f"Pursuer entropy low: {pursuer_entropy:.3f}")
        if evader_entropy is not None and evader_entropy < sp_cfg.health.min_entropy:
            alerts.append(f"Evader entropy low: {evader_entropy:.3f}")

        # Trajectory diversity check
        trajectories = collect_trajectories(
            pursuer_model, evader_model, eval_env, n_episodes=20, seed=cfg.seed + phase,
        )
        n_clusters = health_monitor.check_trajectory_diversity(trajectories)
        if n_clusters < 2:
            alerts.append(f"Low trajectory diversity: {n_clusters} clusters")

        # Capture rate balance check + potential rollback
        if health_monitor.should_rollback(history["capture_rate"]):
            alerts.append("Capture rate imbalanced for 2 consecutive phases — rollback recommended")

        history["health_alerts"].append(alerts)
        if alerts:
            print(f"  Health alerts: {alerts}")

        # Save checkpoint
        health_monitor.save_checkpoint(
            pursuer_model, evader_model,
            checkpoint_dir, phase,
        )

    # Save final models
    final_dir = Path(f"models/{run_id}/final")
    final_dir.mkdir(parents=True, exist_ok=True)
    pursuer_model.save(str(final_dir / "pursuer"))
    evader_model.save(str(final_dir / "evader"))

    # Cleanup
    pursuer_train_env.close()
    evader_train_env.close()
    eval_env.close()

    print(f"\nSelf-play complete. Models saved to {final_dir}")
    return {
        "pursuer_model": pursuer_model,
        "evader_model": evader_model,
        "history": history,
        "run_id": run_id,
    }


def _update_opponent_in_vec_env(vec_env, opponent_model):
    """Update the opponent policy in all sub-environments of a VecEnv.

    The VecMonitor wraps the DummyVecEnv, so we access the inner envs.
    """
    # VecMonitor -> DummyVecEnv -> list of SingleAgentPEWrapper
    inner_vec = vec_env.venv if hasattr(vec_env, "venv") else vec_env
    for env in inner_vec.envs:
        env.set_opponent(opponent_model)
