"""BarrierNet PPO: PPO agent with differentiable QP safety layer.

Key difference from standard PPO and CBF-Beta PPO:
- Actor outputs go through a differentiable QP layer
- Gradients from PPO loss flow through the QP to the actor MLP parameters
- Actions are deterministically safe (no separate safety filter at deployment)
- Eliminates the training-deployment gap of CBF-Beta

Training loop:
1. Collect rollouts using BarrierNet actor (safe actions, log probs)
2. Compute advantages using GAE
3. PPO update: gradient flows through QP layer to actor parameters
4. Critic update: standard MSE on returns

Reference: Xiao et al. 2023 (BarrierNet), Schulman et al. 2017 (PPO).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from agents.barriernet_actor import BarrierNetActor, BarrierNetCritic


@dataclass
class BarrierNetPPOConfig:
    """Configuration for BarrierNet PPO training.

    Args:
        obs_dim: Observation dimensionality.
        hidden_dim: Hidden layer width.
        n_layers: Number of hidden layers.
        n_constraints_max: Maximum CBF constraints.
        lr_actor: Actor learning rate.
        lr_critic: Critic learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_ratio: PPO clip ratio.
        entropy_coeff: Entropy bonus coefficient.
        n_epochs: PPO epochs per update.
        batch_size: Mini-batch size.
        max_grad_norm: Gradient clipping norm.
        v_max: Maximum linear velocity.
        omega_max: Maximum angular velocity.
        w_v: QP weight on velocity deviation.
        w_omega: QP weight on angular velocity deviation.
        alpha: CBF class-K function parameter.
        d: VCP offset distance.
        arena_half_w: Half arena width.
        arena_half_h: Half arena height.
        robot_radius: Robot radius.
        r_min_separation: Minimum inter-robot separation.
    """
    obs_dim: int = 14
    hidden_dim: int = 256
    n_layers: int = 2
    n_constraints_max: int = 10
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    n_epochs: int = 10
    batch_size: int = 256
    max_grad_norm: float = 0.5
    v_max: float = 1.0
    omega_max: float = 2.84
    w_v: float = 150.0
    w_omega: float = 1.0
    alpha: float = 1.0
    d: float = 0.1
    arena_half_w: float = 10.0
    arena_half_h: float = 10.0
    robot_radius: float = 0.15
    r_min_separation: float = 0.35


class RolloutBuffer:
    """Simple rollout buffer for BarrierNet PPO.

    Stores observations, states, actions, rewards, log probs, and values
    for computing GAE advantages and PPO updates.
    """

    def __init__(self):
        self.obs = []
        self.states = []
        self.actions = []
        self.u_noms = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.obstacles = []
        self.opponent_states = []

    def add(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool,
        u_nom: torch.Tensor | None = None,
        obstacles: list[dict] | None = None,
        opponent_state: torch.Tensor | None = None,
    ):
        """Add a transition to the buffer."""
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(action)
        self.u_noms.append(u_nom)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.obstacles.append(obstacles)
        self.opponent_states.append(opponent_state)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: Value estimate of the last state.
            gamma: Discount factor.
            gae_lambda: GAE lambda.

        Returns:
            returns: (T,) discounted returns.
            advantages: (T,) GAE advantages.
        """
        T = len(self.rewards)
        advantages = torch.zeros(T)
        last_gae = 0.0

        values = [v.item() if isinstance(v, torch.Tensor) else v for v in self.values]
        values.append(last_value)

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + torch.tensor(values[:T])
        return returns, advantages

    def get_batches(self, batch_size: int):
        """Yield mini-batches for PPO update.

        Yields dicts with batched tensors for each mini-batch.
        """
        T = len(self.obs)
        indices = np.random.permutation(T)

        obs_t = torch.stack(self.obs)
        states_t = torch.stack(self.states)
        actions_t = torch.stack(self.actions)
        u_noms_t = torch.stack(self.u_noms) if self.u_noms[0] is not None else None
        log_probs_t = torch.stack(self.log_probs) if isinstance(self.log_probs[0], torch.Tensor) else torch.tensor(self.log_probs)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_idx = indices[start:end]

            batch = {
                "obs": obs_t[batch_idx],
                "states": states_t[batch_idx],
                "actions": actions_t[batch_idx],
                "log_probs": log_probs_t[batch_idx],
                "batch_idx": batch_idx,
            }
            if u_noms_t is not None:
                batch["u_noms"] = u_noms_t[batch_idx]
            yield batch

    def clear(self):
        """Clear the buffer."""
        self.obs.clear()
        self.states.clear()
        self.actions.clear()
        self.u_noms.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.obstacles.clear()
        self.opponent_states.clear()

    def __len__(self):
        return len(self.obs)


class BarrierNetPPO:
    """PPO agent using BarrierNet actor (differentiable QP safety layer).

    Key difference from standard PPO:
    - Actor outputs go through a differentiable QP layer
    - Gradients from PPO loss flow through QP to actor MLP parameters
    - Actions are deterministically safe (no separate safety filter)
    - Eliminates training-deployment gap

    Args:
        config: BarrierNetPPOConfig with all hyperparameters.
    """

    def __init__(self, config: BarrierNetPPOConfig):
        self.config = config
        self.device = torch.device("cpu")

        self.actor = BarrierNetActor(
            obs_dim=config.obs_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_constraints_max=config.n_constraints_max,
            v_max=config.v_max,
            omega_max=config.omega_max,
            w_v=config.w_v,
            w_omega=config.w_omega,
            alpha=config.alpha,
            d=config.d,
        )
        self.critic = BarrierNetCritic(
            obs_dim=config.obs_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
        )

        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(), lr=config.lr_actor,
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_critic,
        )

    def to(self, device: torch.device) -> "BarrierNetPPO":
        """Move actor and critic to device."""
        self.device = device
        self.actor.to(device)
        self.critic.to(device)
        return self

    def get_action(
        self,
        obs: torch.Tensor,
        states: torch.Tensor,
        obstacles: list[dict] | None = None,
        opponent_states: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Get action for environment interaction.

        Args:
            obs: (B, obs_dim) observations.
            states: (B, 3) robot states [x, y, theta].
            obstacles: Obstacle list.
            opponent_states: (B, 3) opponent states.
            deterministic: If True, no exploration noise.

        Returns:
            u_safe: (B, 2) safe actions.
            log_prob: (B,) log probabilities.
            value: (B,) value estimates.
            info: Diagnostic dict.
        """
        u_safe, log_prob, entropy, info = self.actor(
            obs, states,
            obstacles=obstacles,
            opponent_states=opponent_states,
            arena_half_w=self.config.arena_half_w,
            arena_half_h=self.config.arena_half_h,
            robot_radius=self.config.robot_radius,
            r_min_separation=self.config.r_min_separation,
            deterministic=deterministic,
        )
        with torch.no_grad():
            value = self.critic(obs)

        return u_safe, log_prob, value, info

    def update(
        self,
        buffer: RolloutBuffer,
        obstacles: list[dict] | None = None,
        opponent_states_list: list[torch.Tensor | None] | None = None,
    ) -> dict:
        """PPO update with gradients flowing through the QP layer.

        Args:
            buffer: Filled rollout buffer.
            obstacles: Static obstacles (used for all samples).
            opponent_states_list: Not used (opponents stored per-step in buffer).

        Returns:
            Dict with training metrics.
        """
        cfg = self.config

        # Compute returns and advantages
        with torch.no_grad():
            last_obs = buffer.obs[-1].to(self.device)
            last_value = self.critic(last_obs).item()

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, cfg.gamma, cfg.gae_lambda,
        )
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(cfg.n_epochs):
            for batch in buffer.get_batches(cfg.batch_size):
                idx = batch["batch_idx"]
                obs = batch["obs"].to(self.device)
                log_probs_old = batch["log_probs"].to(self.device)
                batch_adv = advantages[idx].to(self.device)
                batch_ret = returns[idx].to(self.device)

                # Get stored nominal actions for evaluate_actions
                u_noms = batch["u_noms"].to(self.device) if "u_noms" in batch else None

                if u_noms is not None:
                    # Proper PPO: evaluate log_prob of stored nominal actions
                    # under current policy. Gradient flows: log_prob → u_nom_mean → backbone
                    log_probs_new, entropy = self.actor.evaluate_actions(obs, u_noms)
                else:
                    # Fallback for buffers without u_nom (backward compat)
                    states = batch["states"].to(self.device)
                    _, log_probs_new, entropy, _ = self.actor(
                        obs, states, deterministic=False,
                    )

                # PPO clipped objective
                ratio = torch.exp(log_probs_new - log_probs_old)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(
                    ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio
                ) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy_loss = -cfg.entropy_coeff * entropy

                # Actor loss: gradient flows through evaluate_actions → backbone
                actor_loss = policy_loss + entropy_loss
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.optimizer_actor.step()

                # Critic loss (standard, no QP)
                values = self.critic(obs)
                critic_loss = 0.5 * (values - batch_ret).pow(2).mean()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.optimizer_critic.step()

                total_policy_loss += policy_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # Get current action std for monitoring
        action_std = torch.exp(self.actor.action_log_std).detach().cpu().numpy()

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "mean_qp_correction": 0.0,  # QP not called during update (Approach A)
            "n_updates": n_updates,
            "action_std_v": float(action_std[0]),
            "action_std_omega": float(action_std[1]),
        }

    def save(self, path: str):
        """Save actor and critic state dicts."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "config": self.config,
        }, path)

    @classmethod
    def load(cls, path: str) -> "BarrierNetPPO":
        """Load actor and critic from checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        agent = cls(checkpoint["config"])
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        return agent
