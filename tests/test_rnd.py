"""Tests for RND (Random Network Distillation) intrinsic motivation.

Tests cover RunningMeanStd, RNDModule, RNDRewardWrapper, and integration
with the evader training pipeline.
"""

import numpy as np
import pytest
import torch
from stable_baselines3.common.monitor import Monitor

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper, FixedSpeedWrapper
from envs.rnd import RNDModule, RNDRewardWrapper, RunningMeanStd


# ─── Helpers ───


def _make_evader_env():
    """Create a minimal evader environment for testing (full-obs, flat Box space)."""
    base = PursuitEvasionEnv(
        arena_width=10.0, arena_height=10.0, max_steps=100,
    )
    single = SingleAgentPEWrapper(base, role="evader")
    fixed = FixedSpeedWrapper(single, v_max=1.0)
    return fixed, base


# ─── RunningMeanStd Tests ───


class TestRunningMeanStd:
    def test_initial_state(self):
        rms = RunningMeanStd()
        assert rms.mean == 0.0
        assert rms.var == 1.0

    def test_single_update(self):
        rms = RunningMeanStd()
        rms.update(np.array([5.0]))
        # After one observation, mean should approach 5.0
        assert rms.count > 1  # count starts at 1e-4

    def test_batch_update(self):
        rms = RunningMeanStd()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rms.update(data)
        # Mean should be close to 3.0
        assert abs(rms.mean - 3.0) < 0.1

    def test_std_positive(self):
        rms = RunningMeanStd()
        rms.update(np.array([1.0, 2.0, 3.0]))
        assert rms.std > 0

    def test_multiple_updates_converge(self):
        rms = RunningMeanStd()
        rng = np.random.RandomState(42)
        for _ in range(100):
            batch = rng.normal(10.0, 2.0, size=32)
            rms.update(batch)
        assert abs(rms.mean - 10.0) < 0.5
        assert abs(np.sqrt(rms.var) - 2.0) < 0.5

    def test_vector_shape(self):
        rms = RunningMeanStd(shape=(3,))
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rms.update(data)
        assert rms.mean.shape == (3,)
        assert rms.var.shape == (3,)


# ─── RNDModule Tests ───


class TestRNDModule:
    def test_init(self):
        rnd = RNDModule(obs_dim=43, embed_dim=64, hidden_dim=128)
        assert rnd.obs_dim == 43
        assert rnd.embed_dim == 64

    def test_target_is_frozen(self):
        rnd = RNDModule(obs_dim=10)
        for param in rnd.target.parameters():
            assert not param.requires_grad

    def test_predictor_is_trainable(self):
        rnd = RNDModule(obs_dim=10)
        for param in rnd.predictor.parameters():
            assert param.requires_grad

    def test_intrinsic_reward_positive(self):
        rnd = RNDModule(obs_dim=10, embed_dim=32, hidden_dim=64)
        obs = np.random.randn(10).astype(np.float32)
        reward = rnd.compute_intrinsic_reward(obs)
        assert reward >= 0

    def test_normalized_reward_finite(self):
        rnd = RNDModule(obs_dim=10, embed_dim=32, hidden_dim=64)
        obs = np.random.randn(10).astype(np.float32)
        reward = rnd.compute_normalized_reward(obs)
        assert np.isfinite(reward)

    def test_training_reduces_error(self):
        rnd = RNDModule(obs_dim=10, embed_dim=32, hidden_dim=64, lr=1e-3)
        rng = np.random.RandomState(42)

        # Fixed batch of observations
        obs_batch = rng.randn(256, 10).astype(np.float32)

        # Measure initial prediction error
        initial_error = rnd.compute_intrinsic_reward(obs_batch[0])

        # Train predictor multiple times on same batch
        for _ in range(50):
            rnd.train_predictor(obs_batch)

        # Error on same observations should decrease
        final_error = rnd.compute_intrinsic_reward(obs_batch[0])
        assert final_error < initial_error, (
            f"Training should reduce error: {final_error} >= {initial_error}"
        )

    def test_buffer_accumulation(self):
        rnd = RNDModule(obs_dim=5)
        obs1 = np.ones(5)
        obs2 = np.zeros(5)

        rnd.add_to_buffer(obs1)
        rnd.add_to_buffer(obs2)
        assert len(rnd._obs_buffer) == 2

    def test_train_from_buffer_clears(self):
        rnd = RNDModule(obs_dim=5, embed_dim=16, hidden_dim=32)
        for _ in range(10):
            rnd.add_to_buffer(np.random.randn(5).astype(np.float32))

        loss = rnd.train_from_buffer()
        assert loss is not None
        assert loss > 0
        assert len(rnd._obs_buffer) == 0

    def test_train_from_empty_buffer(self):
        rnd = RNDModule(obs_dim=5)
        loss = rnd.train_from_buffer()
        assert loss is None

    def test_target_weights_unchanged_after_training(self):
        rnd = RNDModule(obs_dim=10, embed_dim=32, hidden_dim=64)

        # Store target weights
        target_params = [p.clone() for p in rnd.target.parameters()]

        # Train predictor
        obs_batch = np.random.randn(64, 10).astype(np.float32)
        rnd.train_predictor(obs_batch)

        # Target should be unchanged
        for old, new in zip(target_params, rnd.target.parameters()):
            assert torch.allclose(old, new), "Target network weights should not change"


# ─── RNDRewardWrapper Tests ───


class TestRNDRewardWrapper:
    def test_wrapper_augments_reward(self):
        env, _ = _make_evader_env()
        obs_dim = env.observation_space.shape[0]
        rnd = RNDModule(obs_dim=obs_dim, embed_dim=32, hidden_dim=64)
        wrapped = RNDRewardWrapper(env, rnd_module=rnd, rnd_coef=1.0)

        obs, _ = wrapped.reset()
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)

        # Info should contain RND diagnostic keys
        assert "rnd_intrinsic" in info
        assert "rnd_extrinsic" in info
        assert "rnd_total" in info
        assert info["rnd_total"] == reward
        wrapped.close()

    def test_wrapper_reward_differs_from_base(self):
        env, base = _make_evader_env()
        obs_dim = env.observation_space.shape[0]
        rnd = RNDModule(obs_dim=obs_dim, embed_dim=32, hidden_dim=64)
        wrapped = RNDRewardWrapper(env, rnd_module=rnd, rnd_coef=1.0)

        obs, _ = wrapped.reset()
        action = wrapped.action_space.sample()
        _, reward, _, _, info = wrapped.step(action)

        # Intrinsic reward should be non-negative
        assert info["rnd_intrinsic"] >= 0
        # Total reward should differ from extrinsic (by intrinsic amount)
        diff = abs(reward - info["rnd_extrinsic"])
        # diff should be rnd_coef * intrinsic
        expected_diff = abs(1.0 * info["rnd_intrinsic"])
        assert abs(diff - expected_diff) < 1e-6
        wrapped.close()

    def test_wrapper_preserves_obs_space(self):
        env, _ = _make_evader_env()
        obs_dim = env.observation_space.shape[0]
        rnd = RNDModule(obs_dim=obs_dim, embed_dim=32, hidden_dim=64)
        wrapped = RNDRewardWrapper(env, rnd_module=rnd, rnd_coef=0.1)

        assert wrapped.observation_space == env.observation_space
        assert wrapped.action_space == env.action_space
        wrapped.close()

    def test_wrapper_with_zero_coef(self):
        env, _ = _make_evader_env()
        obs_dim = env.observation_space.shape[0]
        rnd = RNDModule(obs_dim=obs_dim, embed_dim=32, hidden_dim=64)
        wrapped = RNDRewardWrapper(env, rnd_module=rnd, rnd_coef=0.0)

        obs, _ = wrapped.reset()
        action = wrapped.action_space.sample()
        _, reward, _, _, info = wrapped.step(action)

        # With zero coefficient, reward should equal extrinsic
        assert abs(reward - info["rnd_extrinsic"]) < 1e-6
        wrapped.close()

    def test_wrapper_periodic_training(self):
        env, _ = _make_evader_env()
        obs_dim = env.observation_space.shape[0]
        rnd = RNDModule(obs_dim=obs_dim, embed_dim=32, hidden_dim=64)
        update_freq = 10
        wrapped = RNDRewardWrapper(
            env, rnd_module=rnd, rnd_coef=0.1, update_freq=update_freq,
        )

        obs, _ = wrapped.reset()
        for i in range(update_freq + 5):
            action = wrapped.action_space.sample()
            obs, _, terminated, truncated, _ = wrapped.step(action)
            if terminated or truncated:
                obs, _ = wrapped.reset()

        # Buffer should have been cleared at step `update_freq`
        # and have accumulated 5 more observations since
        assert len(rnd._obs_buffer) == 5
        wrapped.close()

    def test_wrapper_reset(self):
        env, _ = _make_evader_env()
        obs_dim = env.observation_space.shape[0]
        rnd = RNDModule(obs_dim=obs_dim, embed_dim=32, hidden_dim=64)
        wrapped = RNDRewardWrapper(env, rnd_module=rnd, rnd_coef=0.1)

        obs, info = wrapped.reset()
        assert obs.shape == (obs_dim,)
        wrapped.close()

    def test_monitor_compatible(self):
        """RNDRewardWrapper should work under Monitor wrapper."""
        env, _ = _make_evader_env()
        obs_dim = env.observation_space.shape[0]
        rnd = RNDModule(obs_dim=obs_dim, embed_dim=32, hidden_dim=64)
        wrapped = RNDRewardWrapper(env, rnd_module=rnd, rnd_coef=0.1)
        monitored = Monitor(wrapped)

        obs, _ = monitored.reset()
        for _ in range(10):
            action = monitored.action_space.sample()
            obs, _, terminated, truncated, _ = monitored.step(action)
            if terminated or truncated:
                obs, _ = monitored.reset()
        monitored.close()
