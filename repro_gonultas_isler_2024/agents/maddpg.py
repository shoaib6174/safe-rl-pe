import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.amp import autocast as _autocast_raw, GradScaler
    def autocast():
        return _autocast_raw(device_type="cuda")
except ImportError:
    from torch.cuda.amp import autocast, GradScaler


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        return self.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    def __init__(self, total_obs_dim: int, total_action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class MADDPGAgent:
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
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"

        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = ActorNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin critics (double centralized critics) to reduce overestimation bias
        self.critic = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_target = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.critic2 = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic2_target = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.actor_scaler = GradScaler() if self.use_amp else None
        self.critic_scaler = GradScaler() if self.use_amp else None

    def select_action(self, obs: np.ndarray, noise_sigma: float = 0.1) -> np.ndarray:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    action = self.actor(obs_tensor).cpu().numpy()[0]
            else:
                action = self.actor(obs_tensor).cpu().numpy()[0]
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def update(
        self,
        batch: dict,
        other_agent: "MADDPGAgent",
        agent_id: str = "pursuer",
    ) -> tuple:
        obs_p = torch.FloatTensor(batch["obs_p"]).to(self.device)
        obs_e = torch.FloatTensor(batch["obs_e"]).to(self.device)
        actions_p = torch.FloatTensor(batch["actions_p"]).to(self.device)
        actions_e = torch.FloatTensor(batch["actions_e"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards_p"]).unsqueeze(1).to(self.device)
        next_obs_p = torch.FloatTensor(batch["next_obs_p"]).to(self.device)
        next_obs_e = torch.FloatTensor(batch["next_obs_e"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).unsqueeze(1).to(self.device)

        obs_all = torch.cat([obs_p, obs_e], dim=-1)
        next_obs_all = torch.cat([next_obs_p, next_obs_e], dim=-1)
        actions_all = torch.cat([actions_p, actions_e], dim=-1)

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    next_actions_p = self.actor_target(next_obs_p)
                    next_actions_e = other_agent.actor_target(next_obs_e)
                    next_actions_all = torch.cat([next_actions_p, next_actions_e], dim=-1)
                    target_q1 = self.critic_target(next_obs_all, next_actions_all)
                    target_q2 = self.critic2_target(next_obs_all, next_actions_all)
                    target_q = rewards + self.gamma * (1 - dones) * torch.min(target_q1, target_q2)
            else:
                next_actions_p = self.actor_target(next_obs_p)
                next_actions_e = other_agent.actor_target(next_obs_e)
                next_actions_all = torch.cat([next_actions_p, next_actions_e], dim=-1)
                target_q1 = self.critic_target(next_obs_all, next_actions_all)
                target_q2 = self.critic2_target(next_obs_all, next_actions_all)
                target_q = rewards + self.gamma * (1 - dones) * torch.min(target_q1, target_q2)

        # Update first critic
        if self.use_amp:
            with autocast():
                current_q1 = self.critic(obs_all, actions_all)
                critic_loss = nn.MSELoss()(current_q1, target_q)
            self.critic_optimizer.zero_grad()
            self.critic_scaler.scale(critic_loss).backward()
            self.critic_scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_scaler.step(self.critic_optimizer)
            self.critic_scaler.update()
        else:
            current_q1 = self.critic(obs_all, actions_all)
            critic_loss = nn.MSELoss()(current_q1, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()

        # Update second critic
        if self.use_amp:
            with autocast():
                current_q2 = self.critic2(obs_all, actions_all)
                critic2_loss = nn.MSELoss()(current_q2, target_q)
            self.critic2_optimizer.zero_grad()
            self.critic_scaler.scale(critic2_loss).backward()
            self.critic_scaler.unscale_(self.critic2_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
            self.critic_scaler.step(self.critic2_optimizer)
            self.critic_scaler.update()
        else:
            current_q2 = self.critic2(obs_all, actions_all)
            critic2_loss = nn.MSELoss()(current_q2, target_q)
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
            self.critic2_optimizer.step()

        # Actor update uses min of both critics (conservative estimate)
        own_obs = obs_p if agent_id == "pursuer" else obs_e

        if self.use_amp:
            with autocast():
                new_actions = self.actor(own_obs)
                if agent_id == "pursuer":
                    new_actions_all = torch.cat([new_actions, actions_e.detach()], dim=-1)
                else:
                    new_actions_all = torch.cat([actions_p.detach(), new_actions], dim=-1)
                q1_actor = self.critic(obs_all, new_actions_all)
                q2_actor = self.critic2(obs_all, new_actions_all)
                actor_loss = -torch.min(q1_actor, q2_actor).mean()
            self.actor_optimizer.zero_grad()
            self.actor_scaler.scale(actor_loss).backward()
            self.actor_scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_scaler.step(self.actor_optimizer)
            self.actor_scaler.update()
        else:
            new_actions = self.actor(own_obs)
            if agent_id == "pursuer":
                new_actions_all = torch.cat([new_actions, actions_e.detach()], dim=-1)
            else:
                new_actions_all = torch.cat([actions_p.detach(), new_actions], dim=-1)
            q1_actor = self.critic(obs_all, new_actions_all)
            q2_actor = self.critic2(obs_all, new_actions_all)
            actor_loss = -torch.min(q1_actor, q2_actor).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, source: nn.Module, target: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def update_targets(self):
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.critic2, self.critic2_target)
