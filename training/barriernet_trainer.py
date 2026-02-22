"""BarrierNet PPO training loop for pursuit-evasion.

Integrates BarrierNet PPO agent with the PE environment, handling
rollout collection with QP-in-the-loop and training stability.

Key differences from standard PPO training:
1. Rollout collection extracts robot states and obstacles from env info
2. PPO update backprops through the differentiable QP layer
3. Additional logging for QP-specific metrics (correction, solve time, infeasibility)

Performance note: When the agent is on CUDA, a separate CPU copy is used for
rollout inference (batch=1) to avoid GPU transfer overhead. The GPU agent
handles batched PPO updates where the 34x QP speedup matters.

Usage:
    from training.barriernet_trainer import BarrierNetTrainer, BarrierNetTrainerConfig
    from agents.barriernet_ppo import BarrierNetPPO, BarrierNetPPOConfig

    trainer = BarrierNetTrainer(env, agent, BarrierNetTrainerConfig())
    metrics = trainer.train(total_timesteps=100000)
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from agents.barriernet_ppo import BarrierNetPPO, BarrierNetPPOConfig, RolloutBuffer


@dataclass
class BarrierNetTrainerConfig:
    """Configuration for BarrierNet training loop.

    Args:
        rollout_length: Steps per rollout before PPO update.
        total_timesteps: Total environment steps to train.
        log_interval: Iterations between logging.
        save_interval: Iterations between checkpoints.
        save_dir: Directory for saving checkpoints.
        greedy_opponent: If True, use greedy opponent; else random.
        seed: Random seed.
    """
    rollout_length: int = 1024
    total_timesteps: int = 100000
    log_interval: int = 1
    save_interval: int = 50
    save_dir: str = "checkpoints/barriernet"
    greedy_opponent: bool = False
    seed: int = 42


class BarrierNetTrainer:
    """Training loop for BarrierNet PPO on the PE environment.

    Handles rollout collection with QP-in-the-loop: at each step, the
    robot state is extracted from the env and passed to the BarrierNet
    actor for CBF constraint computation.

    Args:
        env: SingleAgentPEWrapper environment.
        agent: BarrierNetPPO agent.
        config: BarrierNetTrainerConfig.
    """

    def __init__(
        self,
        env,
        agent: BarrierNetPPO,
        config: BarrierNetTrainerConfig | None = None,
    ):
        self.env = env
        self.agent = agent
        self.config = config or BarrierNetTrainerConfig()

        # Unwrap to access PE env internals
        self._pe_env = self._unwrap_env(env)

        # Create CPU inference copy if agent is on GPU
        # (avoids CPU↔GPU transfer overhead for batch=1 rollout steps)
        self._cpu_agent = None
        if agent.device.type == "cuda":
            self._cpu_agent = copy.deepcopy(agent)
            self._cpu_agent.to(torch.device("cpu"))

        # Training metrics
        self.total_timesteps = 0
        self.iteration = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_log = []

    def _sync_cpu_agent(self):
        """Sync CPU inference agent weights from GPU agent after update."""
        if self._cpu_agent is not None:
            gpu_state = self.agent.actor.state_dict()
            cpu_state = {k: v.cpu() for k, v in gpu_state.items()}
            self._cpu_agent.actor.load_state_dict(cpu_state)

            gpu_critic_state = self.agent.critic.state_dict()
            cpu_critic_state = {k: v.cpu() for k, v in gpu_critic_state.items()}
            self._cpu_agent.critic.load_state_dict(cpu_critic_state)

            # Sync action_log_std
            self._cpu_agent.actor.action_log_std.data = (
                self.agent.actor.action_log_std.data.cpu()
            )

    @staticmethod
    def _unwrap_env(env):
        """Unwrap to get the base PursuitEvasionEnv."""
        # Handle VecEnv
        if hasattr(env, "envs"):
            env = env.envs[0]
        # Handle Gymnasium wrappers
        while hasattr(env, "env"):
            if hasattr(env, "pursuer_state"):
                return env
            env = env.env
        return env

    def _get_env_state(self) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Extract robot states and obstacles from the env.

        Returns:
            pursuer_state: [x, y, theta] of pursuer.
            evader_state: [x, y, theta] of evader.
            obstacles: List of obstacle dicts.
        """
        pe = self._pe_env
        return pe.pursuer_state.copy(), pe.evader_state.copy(), pe.obstacles

    def collect_rollout(self) -> tuple[RolloutBuffer, dict]:
        """Collect a rollout with QP-in-the-loop.

        Returns:
            buffer: Filled rollout buffer.
            info: Dict with rollout metrics.
        """
        buffer = RolloutBuffer()
        obs, env_info = self.env.reset()

        total_qp_time = 0.0
        n_interventions = 0
        n_infeasible = 0
        n_steps = 0
        episode_reward = 0.0
        episode_length = 0
        episode_rewards = []
        episode_lengths = []

        # Use CPU agent for rollout (avoids GPU transfer overhead for batch=1)
        infer_agent = self._cpu_agent if self._cpu_agent is not None else self.agent

        for step in range(self.config.rollout_length):
            # Get current state from env
            p_state, e_state, obstacles = self._get_env_state()

            # Convert to tensors (CPU)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            state_t = torch.tensor(p_state, dtype=torch.float32).unsqueeze(0)
            opp_t = torch.tensor(e_state, dtype=torch.float32).unsqueeze(0)

            # Get action from BarrierNet actor (CPU inference)
            t0 = time.perf_counter()
            with torch.no_grad():
                u_safe, log_prob, value, act_info = infer_agent.get_action(
                    obs_t, state_t,
                    obstacles=obstacles,
                    opponent_states=opp_t,
                )
            qp_time = time.perf_counter() - t0
            total_qp_time += qp_time

            # Track QP metrics
            if act_info["qp_correction"][0] > 0.01:
                n_interventions += 1
            if not act_info["qp_feasible"]:
                n_infeasible += 1

            # Step environment
            action = u_safe.squeeze(0).numpy()
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            done = terminated or truncated

            # Store in buffer (CPU tensors)
            buffer.add(
                obs=obs_t.squeeze(0),
                state=state_t.squeeze(0),
                action=u_safe.squeeze(0).detach(),
                reward=float(reward),
                log_prob=log_prob.squeeze(0).detach(),
                value=value.squeeze(0).detach(),
                done=done,
                obstacles=obstacles,
                opponent_state=opp_t.squeeze(0),
            )

            episode_reward += float(reward)
            episode_length += 1
            n_steps += 1

            obs = next_obs

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0.0
                episode_length = 0
                obs, env_info = self.env.reset()

        # Compute returns and advantages
        with torch.no_grad():
            last_obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            last_value = infer_agent.critic(last_obs_t).item()

        buffer.compute_returns_and_advantages(
            last_value,
            gamma=self.agent.config.gamma,
            gae_lambda=self.agent.config.gae_lambda,
        )

        # Store episode stats
        self.episode_rewards.extend(episode_rewards)
        self.episode_lengths.extend(episode_lengths)

        rollout_info = {
            "n_steps": n_steps,
            "mean_qp_time_ms": total_qp_time / max(n_steps, 1) * 1000,
            "intervention_rate": n_interventions / max(n_steps, 1),
            "infeasibility_rate": n_infeasible / max(n_steps, 1),
            "n_episodes": len(episode_rewards),
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        }

        return buffer, rollout_info

    def train(self, total_timesteps: int | None = None) -> list[dict]:
        """Main training loop.

        Args:
            total_timesteps: Override config total_timesteps if provided.

        Returns:
            List of per-iteration metric dicts.
        """
        target = total_timesteps or self.config.total_timesteps
        all_metrics = []

        while self.total_timesteps < target:
            # Collect rollout (CPU inference)
            buffer, rollout_info = self.collect_rollout()
            self.total_timesteps += rollout_info["n_steps"]
            self.iteration += 1

            # Get obstacles from buffer for PPO update
            obstacles = buffer.obstacles[0] if buffer.obstacles else None

            # PPO update on GPU (gradients flow through QP)
            update_info = self.agent.update(buffer, obstacles=obstacles)

            # Sync CPU agent weights after GPU update
            self._sync_cpu_agent()

            # Combine metrics
            metrics = {
                "iteration": self.iteration,
                "timesteps": self.total_timesteps,
                **rollout_info,
                **update_info,
            }
            all_metrics.append(metrics)
            self.training_log.append(metrics)

            # Logging
            if self.iteration % self.config.log_interval == 0:
                self._log_metrics(metrics)

            # Save checkpoint
            if self.iteration % self.config.save_interval == 0:
                self._save_checkpoint()

        return all_metrics

    def _log_metrics(self, metrics: dict):
        """Print training metrics."""
        print(
            f"[Iter {metrics['iteration']:4d}] "
            f"steps={metrics['timesteps']:7d}, "
            f"reward={metrics.get('mean_episode_reward', 0):.2f}, "
            f"policy_loss={metrics['policy_loss']:.4f}, "
            f"critic_loss={metrics['critic_loss']:.4f}, "
            f"qp_corr={metrics['mean_qp_correction']:.4f}, "
            f"qp_time={metrics['mean_qp_time_ms']:.1f}ms, "
            f"interv={metrics['intervention_rate']:.3f}, "
            f"infeas={metrics['infeasibility_rate']:.4f}, "
            f"entropy={metrics.get('entropy', 0):.3f}, "
            f"std_v={metrics.get('action_std_v', 0):.3f}, "
            f"std_w={metrics.get('action_std_omega', 0):.3f}"
        )

    def _save_checkpoint(self):
        """Save model checkpoint."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"barriernet_iter_{self.iteration}.pt"
        self.agent.save(str(path))

    def get_training_summary(self) -> dict:
        """Get summary statistics for the entire training run."""
        if not self.training_log:
            return {}

        recent = self.training_log[-10:]
        return {
            "total_iterations": self.iteration,
            "total_timesteps": self.total_timesteps,
            "total_episodes": len(self.episode_rewards),
            "mean_reward_last10": np.mean([m.get("mean_episode_reward", 0) for m in recent]),
            "mean_policy_loss_last10": np.mean([m["policy_loss"] for m in recent]),
            "mean_qp_correction_last10": np.mean([m["mean_qp_correction"] for m in recent]),
            "mean_qp_time_ms_last10": np.mean([m["mean_qp_time_ms"] for m in recent]),
            "mean_intervention_rate_last10": np.mean([m["intervention_rate"] for m in recent]),
        }
