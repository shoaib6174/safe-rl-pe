import numpy as np
import pytest
from training.replay_buffer import ReplayBuffer


def test_buffer_add_and_sample():
    buffer = ReplayBuffer(capacity=100, obs_dim=11, action_dim=2)
    for i in range(10):
        buffer.add(
            obs_p=np.ones(11) * i,
            obs_e=np.ones(11) * i,
            action_p=np.ones(2) * i,
            action_e=np.ones(2) * i,
            reward_p=float(i),
            reward_e=float(-i),
            next_obs_p=np.ones(11) * (i + 1),
            next_obs_e=np.ones(11) * (i + 1),
            done=False,
        )
    batch = buffer.sample(5)
    assert batch["obs_p"].shape == (5, 11)
    assert batch["actions_p"].shape == (5, 2)
    assert batch["rewards_p"].shape == (5,)
    assert batch["dones"].shape == (5,)


def test_buffer_sample_all():
    buffer = ReplayBuffer(capacity=100, obs_dim=11, action_dim=2)
    for i in range(5):
        buffer.add(
            obs_p=np.ones(11),
            obs_e=np.ones(11),
            action_p=np.ones(2),
            action_e=np.ones(2),
            reward_p=1.0,
            reward_e=-1.0,
            next_obs_p=np.ones(11),
            next_obs_e=np.ones(11),
            done=False,
        )
    batch = buffer.sample(5)
    assert batch["obs_p"].shape == (5, 11)


def test_buffer_overwrite():
    buffer = ReplayBuffer(capacity=5, obs_dim=11, action_dim=2)
    for i in range(10):
        buffer.add(
            obs_p=np.ones(11) * i,
            obs_e=np.ones(11) * i,
            action_p=np.ones(2) * i,
            action_e=np.ones(2) * i,
            reward_p=float(i),
            reward_e=float(-i),
            next_obs_p=np.ones(11) * (i + 1),
            next_obs_e=np.ones(11) * (i + 1),
            done=False,
        )
    assert buffer.size == 5
