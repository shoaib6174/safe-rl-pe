import numpy as np
import pytest
from envs.pursuit_evasion_env import PursuitEvasionEnv


def test_env_reset():
    env = PursuitEvasionEnv()
    obs_p, obs_e = env.reset(seed=42)
    assert obs_p.shape == (11,)
    assert obs_e.shape == (11,)
    assert np.all(obs_p >= -1.0) and np.all(obs_p <= 1.0)
    assert np.all(obs_e >= -1.0) and np.all(obs_e <= 1.0)


def test_bicycle_dynamics_straight():
    env = PursuitEvasionEnv()
    env.reset(seed=42)
    env.pursuer_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    _, _, _, _, _ = env.step(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert env.pursuer_state[0] > 0.0
    assert abs(env.pursuer_state[1]) < 0.01


def test_point_mass_dynamics():
    env = PursuitEvasionEnv()
    env.reset(seed=42)
    env.evader_state = np.array([0.0, 0.0, 0.0, 0.0])
    _, _, _, _, _ = env.step(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert env.evader_state[0] > 0.0
    assert env.evader_state[2] > 0.0


def test_capture_condition():
    env = PursuitEvasionEnv()
    env.reset(seed=42)
    env.pursuer_state[:2] = np.array([0.0, 0.0])
    env.evader_state[:2] = np.array([0.1, 0.0])
    _, _, _, _, done = env.step(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert done


def test_reward_zero_sum():
    env = PursuitEvasionEnv()
    env.reset(seed=42)
    _, _, r_p, r_e, _ = env.step(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert abs(r_p + r_e) < 1e-6


def test_reward_capture_positive_for_pursuer():
    env = PursuitEvasionEnv()
    env.reset(seed=42)
    env.pursuer_state[:2] = np.array([0.0, 0.0])
    env.evader_state[:2] = np.array([0.1, 0.0])
    _, _, r_p, _, done = env.step(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert done
    assert r_p > 0


def test_observation_normalization():
    env = PursuitEvasionEnv()
    obs_p, obs_e = env.reset(seed=42)
    assert np.all(obs_p >= -1.0) and np.all(obs_p <= 1.0)
    assert np.all(obs_e >= -1.0) and np.all(obs_e <= 1.0)


def test_timeout():
    env = PursuitEvasionEnv(max_steps=5)
    env.reset(seed=42)
    done = False
    for _ in range(5):
        _, _, _, _, done = env.step(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert done
