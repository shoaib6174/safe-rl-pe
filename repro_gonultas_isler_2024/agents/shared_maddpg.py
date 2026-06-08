import numpy as np
import torch
import torch.optim as optim
from torch import nn

try:
    from torch.amp import autocast as _autocast_raw, GradScaler
    def autocast():
        return _autocast_raw(device_type="cuda")
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from agents.maddpg import ActorNetwork, CriticNetwork


class SharedMADDPGAgent:
    """MADDPG with a shared actor for both pursuer and evader roles.

    The actor network is conditioned on a role input (0=pursuer, 1=evader).
    Both roles update the SAME actor weights, creating negative feedback
    toward competitive balance (CR=0.5).

    Critics remain separate (one per role) to reduce gradient conflict.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        total_obs_dim: int,
        total_action_dim: int,
        actor_lr: float = 0.0005,
        critic_lr: float = 0.0005,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden_dim: int = 128,
        device: str = "cpu",
        use_amp: bool = False,
        use_compile: bool = False,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"

        # Shared actor with role conditioning (role_dim=1)
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim, role_dim=1).to(self.device)
        self.actor_target = ActorNetwork(obs_dim, action_dim, hidden_dim, role_dim=1).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Pursuer critics (twin)
        self.critic_p = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_p_target = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_p_target.load_state_dict(self.critic_p.state_dict())
        self.critic_p_optimizer = optim.Adam(self.critic_p.parameters(), lr=critic_lr)

        self.critic_p2 = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_p2_target = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_p2_target.load_state_dict(self.critic_p2.state_dict())
        self.critic_p2_optimizer = optim.Adam(self.critic_p2.parameters(), lr=critic_lr)

        # Evader critics (twin)
        self.critic_e = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_e_target = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_e_target.load_state_dict(self.critic_e.state_dict())
        self.critic_e_optimizer = optim.Adam(self.critic_e.parameters(), lr=critic_lr)

        self.critic_e2 = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_e2_target = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_e2_target.load_state_dict(self.critic_e2.state_dict())
        self.critic_e2_optimizer = optim.Adam(self.critic_e2.parameters(), lr=critic_lr)

        # torch.compile
        if use_compile and self.device.type == "cuda":
            self.actor = torch.compile(self.actor, mode="reduce-overhead")
            self.actor_target = torch.compile(self.actor_target, mode="reduce-overhead")
            self.critic_p = torch.compile(self.critic_p, mode="reduce-overhead")
            self.critic_p_target = torch.compile(self.critic_p_target, mode="reduce-overhead")
            self.critic_p2 = torch.compile(self.critic_p2, mode="reduce-overhead")
            self.critic_p2_target = torch.compile(self.critic_p2_target, mode="reduce-overhead")
            self.critic_e = torch.compile(self.critic_e, mode="reduce-overhead")
            self.critic_e_target = torch.compile(self.critic_e_target, mode="reduce-overhead")
            self.critic_e2 = torch.compile(self.critic_e2, mode="reduce-overhead")
            self.critic_e2_target = torch.compile(self.critic_e2_target, mode="reduce-overhead")

        self.actor_scaler = GradScaler() if self.use_amp else None
        self.critic_scaler = GradScaler() if self.use_amp else None

    def _to_tensor(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.to(self.device)
        return torch.FloatTensor(arr).to(self.device)

    @staticmethod
    def _clear_adam_state(optimizer, parameters):
        for param in parameters:
            if param in optimizer.state:
                state = optimizer.state[param]
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key].zero_()

    def select_action(self, obs: np.ndarray, role: int, noise_sigma: float = 0.1) -> np.ndarray:
        """Select action for a given role.

        role: 0 = pursuer, 1 = evader
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        role_tensor = torch.FloatTensor([[float(role)]]).to(self.device)
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    action = self.actor(obs_tensor, role_tensor).cpu().numpy()[0]
            else:
                action = self.actor(obs_tensor, role_tensor).cpu().numpy()[0]
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def select_actions_batch(
        self, observations: list, role: int, noise_sigma: float = 0.1
    ) -> np.ndarray:
        """Batched action selection for a given role."""
        obs_batch = torch.FloatTensor(np.stack(observations)).to(self.device)
        role_batch = torch.full((len(observations), 1), float(role)).to(self.device)
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    actions = self.actor(obs_batch, role_batch).cpu().numpy()
            else:
                actions = self.actor(obs_batch, role_batch).cpu().numpy()
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, size=actions.shape)
            actions = np.clip(actions + noise, -1.0, 1.0)
        return actions

    def _compute_target_q(self, next_obs_all, next_actions_all, rewards, dones,
                          critic_target, critic2_target):
        """Compute clipped target Q for critic update."""
        if self.use_amp:
            with autocast():
                target_q1 = critic_target(next_obs_all, next_actions_all)
                target_q2 = critic2_target(next_obs_all, next_actions_all)
                target_q = rewards + self.gamma * (1 - dones) * torch.min(target_q1, target_q2)
        else:
            target_q1 = critic_target(next_obs_all, next_actions_all)
            target_q2 = critic2_target(next_obs_all, next_actions_all)
            target_q = rewards + self.gamma * (1 - dones) * torch.min(target_q1, target_q2)
        Q_MIN, Q_MAX = -1500.0, 1500.0
        return torch.clamp(target_q, min=Q_MIN, max=Q_MAX)

    def _update_critic(self, obs_all, actions_all, target_q, critic, critic_optimizer):
        """Update a single critic network."""
        if self.use_amp:
            with autocast():
                current_q = critic(obs_all, actions_all)
                loss = nn.MSELoss()(current_q, target_q)
            critic_optimizer.zero_grad()
            self.critic_scaler.scale(loss).backward()
            self.critic_scaler.unscale_(critic_optimizer)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            if torch.isnan(loss) or torch.isinf(loss):
                self._clear_adam_state(critic_optimizer, critic.parameters())
            self.critic_scaler.step(critic_optimizer)
            self.critic_scaler.update()
        else:
            current_q = critic(obs_all, actions_all)
            loss = nn.MSELoss()(current_q, target_q)
            if torch.isnan(loss) or torch.isinf(loss):
                self._clear_adam_state(critic_optimizer, critic.parameters())
            else:
                critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                critic_optimizer.step()
        return loss.item()

    def update_critics(self, batch: dict, ema_actor) -> tuple:
        """Update both pursuer and evader critics (separate).

        Returns (c_loss_p, c_loss_e).
        """
        obs_p = self._to_tensor(batch["obs_p"])
        obs_e = self._to_tensor(batch["obs_e"])
        actions_p = self._to_tensor(batch["actions_p"])
        actions_e = self._to_tensor(batch["actions_e"])
        rewards_p = self._to_tensor(batch["rewards_p"]).unsqueeze(1)
        rewards_e = self._to_tensor(batch["rewards_e"]).unsqueeze(1)
        next_obs_p = self._to_tensor(batch["next_obs_p"])
        next_obs_e = self._to_tensor(batch["next_obs_e"])
        dones = self._to_tensor(batch["dones"]).unsqueeze(1)

        obs_all = torch.cat([obs_p, obs_e], dim=-1)
        next_obs_all = torch.cat([next_obs_p, next_obs_e], dim=-1)
        actions_all = torch.cat([actions_p, actions_e], dim=-1)

        with torch.no_grad():
            # Target actions: use shared actor_target with appropriate roles
            role_p = torch.zeros(next_obs_p.size(0), 1).to(self.device)
            role_e = torch.ones(next_obs_e.size(0), 1).to(self.device)
            next_actions_p = self.actor_target(next_obs_p, role_p)
            next_actions_e = self.actor_target(next_obs_e, role_e)
            next_actions_all = torch.cat([next_actions_p, next_actions_e], dim=-1)

        # Update pursuer critics
        target_q_p = self._compute_target_q(next_obs_all, next_actions_all, rewards_p, dones,
                                            self.critic_p_target, self.critic_p2_target)
        c_loss_p = self._update_critic(obs_all, actions_all, target_q_p,
                                       self.critic_p, self.critic_p_optimizer)
        c_loss_p2 = self._update_critic(obs_all, actions_all, target_q_p.detach(),
                                        self.critic_p2, self.critic_p2_optimizer)

        # Update evader critics
        target_q_e = self._compute_target_q(next_obs_all, next_actions_all, rewards_e, dones,
                                            self.critic_e_target, self.critic_e2_target)
        c_loss_e = self._update_critic(obs_all, actions_all, target_q_e,
                                       self.critic_e, self.critic_e_optimizer)
        c_loss_e2 = self._update_critic(obs_all, actions_all, target_q_e.detach(),
                                        self.critic_e2, self.critic_e2_optimizer)

        return c_loss_p, c_loss_e

    def update_shared_actor(self, batch: dict, ema_actor) -> tuple:
        """Update the shared actor using combined pursuer + evader losses.

        The key mechanism: both roles backprop through the SAME actor weights.
        If pursuer is dominant (CR > 0.5), its gradient is weak; evader's is strong.
        This creates negative feedback toward CR=0.5.

        Returns (a_loss_p, a_loss_e).
        """
        obs_p = self._to_tensor(batch["obs_p"])
        obs_e = self._to_tensor(batch["obs_e"])
        actions_p = self._to_tensor(batch["actions_p"])
        actions_e = self._to_tensor(batch["actions_e"])

        obs_all = torch.cat([obs_p, obs_e], dim=-1)

        Q_MIN, Q_MAX = -1500.0, 1500.0

        role_p = torch.zeros(obs_p.size(0), 1).to(self.device)
        role_e = torch.ones(obs_e.size(0), 1).to(self.device)

        if self.use_amp:
            with autocast():
                # Pursuer role loss
                new_actions_p = self.actor(obs_p, role_p)
                new_actions_all_p = torch.cat([new_actions_p, actions_e.detach()], dim=-1)
                q1_p = torch.clamp(self.critic_p(obs_all, new_actions_all_p), min=Q_MIN, max=Q_MAX)
                q2_p = torch.clamp(self.critic_p2(obs_all, new_actions_all_p), min=Q_MIN, max=Q_MAX)
                loss_p = -torch.min(q1_p, q2_p).mean()

                # Evader role loss
                new_actions_e = self.actor(obs_e, role_e)
                new_actions_all_e = torch.cat([actions_p.detach(), new_actions_e], dim=-1)
                q1_e = torch.clamp(self.critic_e(obs_all, new_actions_all_e), min=Q_MIN, max=Q_MAX)
                q2_e = torch.clamp(self.critic_e2(obs_all, new_actions_all_e), min=Q_MIN, max=Q_MAX)
                loss_e = -torch.min(q1_e, q2_e).mean()

                total_loss = loss_p + loss_e

            self.actor_optimizer.zero_grad()
            self.actor_scaler.scale(total_loss).backward()
            self.actor_scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self._clear_adam_state(self.actor_optimizer, self.actor.parameters())
            self.actor_scaler.step(self.actor_optimizer)
            self.actor_scaler.update()
        else:
            # Pursuer role loss
            new_actions_p = self.actor(obs_p, role_p)
            new_actions_all_p = torch.cat([new_actions_p, actions_e.detach()], dim=-1)
            q1_p = torch.clamp(self.critic_p(obs_all, new_actions_all_p), min=Q_MIN, max=Q_MAX)
            q2_p = torch.clamp(self.critic_p2(obs_all, new_actions_all_p), min=Q_MIN, max=Q_MAX)
            loss_p = -torch.min(q1_p, q2_p).mean()

            # Evader role loss
            new_actions_e = self.actor(obs_e, role_e)
            new_actions_all_e = torch.cat([actions_p.detach(), new_actions_e], dim=-1)
            q1_e = torch.clamp(self.critic_e(obs_all, new_actions_all_e), min=Q_MIN, max=Q_MAX)
            q2_e = torch.clamp(self.critic_e2(obs_all, new_actions_all_e), min=Q_MIN, max=Q_MAX)
            loss_e = -torch.min(q1_e, q2_e).mean()

            total_loss = loss_p + loss_e

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self._clear_adam_state(self.actor_optimizer, self.actor.parameters())
                return 0.0, 0.0

            self.actor_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

        return loss_p.item(), loss_e.item()

    def soft_update(self, source: nn.Module, target: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def update_targets(self):
        """Soft update all target networks."""
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic_p, self.critic_p_target)
        self.soft_update(self.critic_p2, self.critic_p2_target)
        self.soft_update(self.critic_e, self.critic_e_target)
        self.soft_update(self.critic_e2, self.critic_e2_target)
