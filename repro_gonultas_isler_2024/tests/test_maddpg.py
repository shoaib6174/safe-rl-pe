import numpy as np
import pytest
import torch
from agents.maddpg import ActorNetwork, CriticNetwork, MADDPGAgent


def test_actor_output_shape():
    actor = ActorNetwork(obs_dim=13, action_dim=2, hidden_dim=128)
    obs = torch.randn(32, 13)
    action = actor(obs)
    assert action.shape == (32, 2)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_critic_output_shape():
    critic = CriticNetwork(total_obs_dim=26, total_action_dim=4, hidden_dim=128)
    obs = torch.randn(32, 26)
    actions = torch.randn(32, 4)
    q = critic(obs, actions)
    assert q.shape == (32, 1)


def test_agent_select_action():
    agent = MADDPGAgent(
        obs_dim=13,
        action_dim=2,
        total_obs_dim=26,
        total_action_dim=4,
        actor_lr=0.0005,
        critic_lr=0.0005,
        gamma=0.99,
        tau=0.005,
    )
    obs = np.ones(13, dtype=np.float32)
    action = agent.select_action(obs, noise_sigma=0.0)
    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_target_networks_update():
    agent = MADDPGAgent(
        obs_dim=13,
        action_dim=2,
        total_obs_dim=26,
        total_action_dim=4,
        actor_lr=0.0005,
        critic_lr=0.0005,
        gamma=0.99,
        tau=0.005,
    )
    old_target = agent.actor_target.fc1.weight.clone()
    agent.actor.fc1.weight.data += 1.0
    agent.soft_update(agent.actor, agent.actor_target)
    new_target = agent.actor_target.fc1.weight.clone()
    assert not torch.allclose(old_target, new_target)
