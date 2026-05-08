import numpy as np
import torch
from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.maddpg import MADDPGAgent
from agents.bimdn import BiMDN
from training.replay_buffer import ReplayBuffer


def test_full_episode_with_agents():
    env = PursuitEvasionEnv()
    obs_dim = 22  # 11 * 2 frame stack
    action_dim = 2

    pursuer = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=obs_dim * 2,
        total_action_dim=action_dim * 2,
    )
    evader = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=obs_dim * 2,
        total_action_dim=action_dim * 2,
    )
    buffer = ReplayBuffer(capacity=1000, obs_dim=obs_dim, action_dim=action_dim)

    obs_p, obs_e = env.reset(seed=42)
    obs_p_buffer = [obs_p.copy(), obs_p.copy()]
    obs_e_buffer = [obs_e.copy(), obs_e.copy()]

    done = False
    steps = 0
    while not done and steps < 50:
        stacked_p = np.concatenate(obs_p_buffer[-2:])
        stacked_e = np.concatenate(obs_e_buffer[-2:])

        action_p = pursuer.select_action(stacked_p, noise_sigma=0.1)
        action_e = evader.select_action(stacked_e, noise_sigma=0.1)

        next_obs_p, next_obs_e, r_p, r_e, done = env.step(action_p, action_e)

        next_stacked_p = np.concatenate(obs_p_buffer[-1:] + [next_obs_p])
        next_stacked_e = np.concatenate(obs_e_buffer[-1:] + [next_obs_e])

        buffer.add(stacked_p, stacked_e, action_p, action_e, r_p, r_e,
                   next_stacked_p, next_stacked_e, done)

        obs_p_buffer.append(next_obs_p.copy())
        obs_e_buffer.append(next_obs_e.copy())
        steps += 1

    assert buffer.size > 0
    assert steps > 0


def test_bimdn_with_env_obs():
    bimdn = BiMDN(obs_dim=11, hidden_dim=128, feature_dim=32, num_components=3)
    env = PursuitEvasionEnv()
    obs_p, _ = env.reset(seed=42)

    # Create fake history (batch=1, seq=20, obs=11)
    history = torch.tensor(obs_p, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    history = history.repeat(1, 20, 1)

    pi, mu, sigma = bimdn(history)
    assert pi.shape == (1, 3)
    assert mu.shape == (1, 3, 2)
    assert sigma.shape == (1, 3, 2)

    weighted_mean = bimdn.get_weighted_mean(history)
    assert weighted_mean.shape == (1, 2)
