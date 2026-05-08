import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = ActorNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_target = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs: np.ndarray, noise_sigma: float = 0.1) -> np.ndarray:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def update(
        self,
        batch: dict,
        other_agent: "MADDPGAgent",
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
            next_actions_p = self.actor_target(next_obs_p)
            next_actions_e = other_agent.actor_target(next_obs_e)
            next_actions_all = torch.cat([next_actions_p, next_actions_e], dim=-1)
            target_q = rewards + self.gamma * (1 - dones) * self.critic_target(
                next_obs_all, next_actions_all
            )

        current_q = self.critic(obs_all, actions_all)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions_p = self.actor(obs_p)
        new_actions_all = torch.cat([new_actions_p, actions_e.detach()], dim=-1)
        actor_loss = -self.critic(obs_all, new_actions_all).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
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
