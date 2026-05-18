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


def test_update_accepts_opponent_actor():
    """Verify update() accepts an external actor module (for EMA / Pop-FSP)."""
    agent1 = MADDPGAgent(
        obs_dim=13,
        action_dim=2,
        total_obs_dim=26,
        total_action_dim=4,
    )
    agent2 = MADDPGAgent(
        obs_dim=13,
        action_dim=2,
        total_obs_dim=26,
        total_action_dim=4,
    )

    batch = {
        "obs_p": np.random.randn(32, 13).astype(np.float32),
        "obs_e": np.random.randn(32, 13).astype(np.float32),
        "actions_p": np.random.randn(32, 2).astype(np.float32),
        "actions_e": np.random.randn(32, 2).astype(np.float32),
        "rewards_p": np.random.randn(32).astype(np.float32),
        "rewards_e": np.random.randn(32).astype(np.float32),
        "next_obs_p": np.random.randn(32, 13).astype(np.float32),
        "next_obs_e": np.random.randn(32, 13).astype(np.float32),
        "dones": np.zeros(32, dtype=np.float32),
    }

    # Pass agent2's actor directly (simulating EMA opponent)
    c_loss, a_loss = agent1.update(batch, agent2.actor, agent_id="pursuer")
    assert isinstance(c_loss, float)
    assert isinstance(a_loss, float)


def test_ema_update_slowly():
    """Verify EMA tracking: after updating actor, EMA moves by approximately tau."""
    from copy import deepcopy

    agent = MADDPGAgent(
        obs_dim=13,
        action_dim=2,
        total_obs_dim=26,
        total_action_dim=4,
    )

    # Create EMA copy
    ema_actor = deepcopy(agent.actor)
    ema_actor.eval()
    for param in ema_actor.parameters():
        param.requires_grad = False

    # Record initial weights
    old_ema_weight = ema_actor.fc1.weight.data.clone()
    old_actor_weight = agent.actor.fc1.weight.data.clone()

    # Perturb actor weights
    agent.actor.fc1.weight.data += 1.0

    # Apply EMA update with tau=0.005
    tau = 0.005
    with torch.no_grad():
        for ema_param, param in zip(ema_actor.parameters(), agent.actor.parameters()):
            ema_param.data.mul_(1 - tau).add_(param.data, alpha=tau)

    # Verify EMA moved toward actor by approximately tau
    delta_ema = (ema_actor.fc1.weight.data - old_ema_weight).abs().mean().item()
    delta_actor = (agent.actor.fc1.weight.data - old_actor_weight).abs().mean().item()

    # EMA should have moved by tau * delta_actor ≈ 0.005 * 1.0 = 0.005
    expected_delta = tau * delta_actor
    assert abs(delta_ema - expected_delta) < 1e-6, f"EMA moved by {delta_ema}, expected {expected_delta}"

    # Verify EMA did NOT fully catch up
    assert not torch.allclose(ema_actor.fc1.weight.data, agent.actor.fc1.weight.data)
