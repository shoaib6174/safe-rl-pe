"""Baseline comparison script.

Trains DQN, DDPG, PPO against random opponents and compares
with random and greedy heuristic baselines.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py total_timesteps=500000 wandb.mode=disabled
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, DQN, DDPG
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.noise import NormalActionNoise

from envs.discrete_action_wrapper import DiscreteActionWrapper
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper
from training.baselines import (
    GreedyEvaderPolicy,
    GreedyPursuerPolicy,
    RandomPolicy,
)
from training.tracking import PursuitEvasionMetricsCallback
from training.utils import make_pe_env, make_vec_env, setup_reproducibility


def evaluate_policy_pair(
    pursuer_policy,
    evader_policy,
    env: PursuitEvasionEnv,
    n_episodes: int = 100,
    seed: int = 0,
) -> dict:
    """Evaluate a pursuer policy against an evader policy."""
    captures = 0
    capture_times = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            p_action, _ = pursuer_policy.predict(obs["pursuer"], deterministic=True)
            e_action, _ = evader_policy.predict(obs["evader"], deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(p_action, e_action)
            done = terminated or truncated

        if "episode_metrics" in info:
            m = info["episode_metrics"]
            if m["captured"]:
                captures += 1
                capture_times.append(m["capture_time"])
            episode_lengths.append(m["episode_length"])

    return {
        "capture_rate": captures / n_episodes,
        "mean_capture_time": float(np.mean(capture_times)) if capture_times else float("nan"),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0,
    }


def train_ppo_baseline(cfg: DictConfig, seed: int) -> PPO:
    """Train PPO pursuer against random evader."""
    setup_reproducibility(seed)
    envs = make_vec_env(cfg, n_envs=cfg.n_envs, role="pursuer", seed=seed)

    algo_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    policy_kwargs = algo_cfg.pop("policy_kwargs", {})
    policy_kwargs["net_arch"] = list(policy_kwargs["net_arch"])
    policy_kwargs["activation_fn"] = torch.nn.Tanh

    model = PPO(
        "MlpPolicy", envs, verbose=0, seed=seed, device="cpu",
        policy_kwargs=policy_kwargs, **algo_cfg,
    )

    cb = PursuitEvasionMetricsCallback(log_frequency=cfg.wandb.log_frequency)
    model.learn(total_timesteps=cfg.total_timesteps, callback=cb, progress_bar=True)
    envs.close()
    return model


def train_dqn_baseline(cfg: DictConfig, seed: int) -> DQN:
    """Train DQN pursuer (discrete actions) against random evader."""
    setup_reproducibility(seed)

    base = PursuitEvasionEnv(
        arena_width=cfg.env.arena_width, arena_height=cfg.env.arena_height,
        dt=cfg.env.dt, max_steps=cfg.env.max_steps,
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
    wrapped = SingleAgentPEWrapper(base, role="pursuer", opponent_policy=None)
    env = DiscreteActionWrapper(wrapped, n_velocities=5, n_angular=7)

    model = DQN(
        "MlpPolicy", env, verbose=0, seed=seed, device="cpu",
        learning_rate=1e-4,
        buffer_size=50000,
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=500,
        policy_kwargs={"net_arch": [256, 256]},
    )

    cb = PursuitEvasionMetricsCallback(log_frequency=cfg.wandb.log_frequency)
    model.learn(total_timesteps=cfg.total_timesteps, callback=cb, progress_bar=True)
    env.close()
    return model


def train_ddpg_baseline(cfg: DictConfig, seed: int) -> DDPG:
    """Train DDPG pursuer against random evader."""
    setup_reproducibility(seed)
    env = make_pe_env(cfg, role="pursuer", opponent_policy=None, seed=seed)

    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions),
    )

    model = DDPG(
        "MlpPolicy", env, verbose=0, seed=seed, device="cpu",
        learning_rate=1e-4,
        buffer_size=50000,
        batch_size=128,
        gamma=0.99,
        tau=0.005,
        action_noise=action_noise,
        policy_kwargs={"net_arch": [256, 256]},
    )

    cb = PursuitEvasionMetricsCallback(log_frequency=cfg.wandb.log_frequency)
    model.learn(total_timesteps=cfg.total_timesteps, callback=cb, progress_bar=True)
    env.close()
    return model


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Baseline comparison config:\n{OmegaConf.to_yaml(cfg)}")

    eval_env = PursuitEvasionEnv(
        arena_width=cfg.env.arena_width, arena_height=cfg.env.arena_height,
        dt=cfg.env.dt, max_steps=cfg.env.max_steps,
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

    random_p = RandomPolicy(v_max=cfg.env.pursuer.v_max, omega_max=cfg.env.pursuer.omega_max)
    random_e = RandomPolicy(v_max=cfg.env.evader.v_max, omega_max=cfg.env.evader.omega_max)
    greedy_p = GreedyPursuerPolicy(
        v_max=cfg.env.pursuer.v_max, omega_max=cfg.env.pursuer.omega_max,
    )
    greedy_e = GreedyEvaderPolicy(
        v_max=cfg.env.evader.v_max, omega_max=cfg.env.evader.omega_max,
    )

    results = {}
    n_eval = 100

    # 1. Random vs Random
    print("\n=== Random vs Random ===")
    r = evaluate_policy_pair(random_p, random_e, eval_env, n_eval, seed=0)
    results["Random"] = r
    print(f"  Capture rate: {r['capture_rate']:.2f}")

    # 2. Greedy vs Greedy
    print("\n=== Greedy vs Greedy ===")
    r = evaluate_policy_pair(greedy_p, greedy_e, eval_env, n_eval, seed=0)
    results["Greedy"] = r
    print(f"  Capture rate: {r['capture_rate']:.2f}")

    # 3. PPO (vs random evader)
    print("\n=== Training PPO pursuer (vs random) ===")
    t0 = time.time()
    ppo_model = train_ppo_baseline(cfg, seed=cfg.seed)
    ppo_time = time.time() - t0
    r = evaluate_policy_pair(ppo_model, random_e, eval_env, n_eval, seed=0)
    results["PPO"] = {**r, "training_time": ppo_time}
    print(f"  Capture rate: {r['capture_rate']:.2f} (trained in {ppo_time:.0f}s)")

    # 4. DQN (vs random evader)
    print("\n=== Training DQN pursuer (vs random) ===")
    t0 = time.time()
    dqn_model = train_dqn_baseline(cfg, seed=cfg.seed)
    dqn_time = time.time() - t0
    # DQN needs discrete-to-continuous for eval
    r = evaluate_dqn(dqn_model, random_e, eval_env, cfg, n_eval, seed=0)
    results["DQN"] = {**r, "training_time": dqn_time}
    print(f"  Capture rate: {r['capture_rate']:.2f} (trained in {dqn_time:.0f}s)")

    # 5. DDPG (vs random evader)
    print("\n=== Training DDPG pursuer (vs random) ===")
    t0 = time.time()
    ddpg_model = train_ddpg_baseline(cfg, seed=cfg.seed)
    ddpg_time = time.time() - t0
    r = evaluate_policy_pair(ddpg_model, random_e, eval_env, n_eval, seed=0)
    results["DDPG"] = {**r, "training_time": ddpg_time}
    print(f"  Capture rate: {r['capture_rate']:.2f} (trained in {ddpg_time:.0f}s)")

    # Print comparison table
    print("\n" + "=" * 75)
    print(f"{'Method':<12} {'Capture Rate':>14} {'Mean Cap Time':>15} {'Train Time':>12}")
    print("-" * 75)
    for name, r in results.items():
        cr = f"{r['capture_rate']:.2f}"
        ct = f"{r['mean_capture_time']:.1f}s" if not np.isnan(r["mean_capture_time"]) else "N/A"
        tt = f"{r.get('training_time', 0):.0f}s"
        print(f"{name:<12} {cr:>14} {ct:>15} {tt:>12}")
    print("=" * 75)

    eval_env.close()


def evaluate_dqn(dqn_model, evader_policy, env, cfg, n_episodes, seed):
    """Evaluate DQN model by wrapping its discrete predictions to continuous."""
    from training.baselines import create_discrete_action_space
    action_table = create_discrete_action_space(
        v_max=cfg.env.pursuer.v_max, omega_max=cfg.env.pursuer.omega_max,
    )

    class DQNBridge:
        """Bridges DQN discrete output to continuous actions for eval."""
        def __init__(self, model, action_table):
            self.model = model
            self.table = action_table
        def predict(self, obs, deterministic=True):
            action_idx, _ = self.model.predict(obs, deterministic=deterministic)
            return self.table[int(action_idx)], None

    bridge = DQNBridge(dqn_model, action_table)
    return evaluate_policy_pair(bridge, evader_policy, env, n_episodes, seed)


if __name__ == "__main__":
    main()
