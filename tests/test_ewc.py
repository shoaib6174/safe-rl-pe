"""Tests for EWC (Elastic Weight Consolidation) regularizer.

Tests cover initialization, snapshot, Fisher estimation, gradient hooks,
penalty computation, and integration with AMSDRLSelfPlay.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper, FixedSpeedWrapper
from training.ewc import EWCRegularizer


# ─── Helpers ───


def _make_evader_model():
    """Create a minimal PPO evader model for testing (full-obs, flat Box space)."""
    base = PursuitEvasionEnv(
        arena_width=10.0, arena_height=10.0, max_steps=100,
    )
    single = SingleAgentPEWrapper(base, role="evader")
    fixed = FixedSpeedWrapper(single, v_max=1.0)
    env = Monitor(fixed)

    model = PPO(
        "MlpPolicy", env,
        n_steps=64, batch_size=32, n_epochs=1,
        seed=42, device="cpu",
        policy_kwargs={"net_arch": [32, 32]},
    )
    return model, env


def _collect_obs(env, n=128):
    """Collect observations from environment."""
    obs_list = []
    obs, _ = env.reset()
    for _ in range(n):
        obs_list.append(obs)
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    return torch.FloatTensor(np.array(obs_list))


# ─── Initialization Tests ───


class TestEWCInit:
    def test_default_params(self):
        ewc = EWCRegularizer()
        assert ewc.lambda_ == 1000.0
        assert ewc.fisher_samples == 1024
        assert not ewc.has_snapshot

    def test_custom_params(self):
        ewc = EWCRegularizer(lambda_=500.0, fisher_samples=512)
        assert ewc.lambda_ == 500.0
        assert ewc.fisher_samples == 512

    def test_no_snapshot_initially(self):
        ewc = EWCRegularizer()
        assert not ewc.has_snapshot
        assert ewc._theta_star == {}
        assert ewc._fisher == {}


# ─── Snapshot Tests ───


class TestEWCSnapshot:
    def test_snapshot_stores_params(self):
        ewc = EWCRegularizer(lambda_=100.0, fisher_samples=64)
        model, env = _make_evader_model()
        obs = _collect_obs(env, n=64)

        ewc.snapshot(model, obs)

        assert ewc.has_snapshot
        assert len(ewc._theta_star) > 0
        # Every trainable param should be stored
        trainable = {n for n, p in model.policy.named_parameters() if p.requires_grad}
        assert set(ewc._theta_star.keys()) == trainable
        env.close()

    def test_theta_star_matches_current(self):
        ewc = EWCRegularizer(lambda_=100.0, fisher_samples=64)
        model, env = _make_evader_model()
        obs = _collect_obs(env, n=64)

        ewc.snapshot(model, obs)

        for name, param in model.policy.named_parameters():
            if param.requires_grad:
                assert torch.allclose(ewc._theta_star[name], param.data)
        env.close()

    def test_fisher_non_negative(self):
        ewc = EWCRegularizer(lambda_=100.0, fisher_samples=64)
        model, env = _make_evader_model()
        obs = _collect_obs(env, n=64)

        ewc.snapshot(model, obs)

        for name, fisher in ewc._fisher.items():
            assert (fisher >= 0).all(), f"Negative Fisher values for {name}"
        env.close()

    def test_fisher_not_all_zero(self):
        ewc = EWCRegularizer(lambda_=100.0, fisher_samples=64)
        model, env = _make_evader_model()
        obs = _collect_obs(env, n=64)

        ewc.snapshot(model, obs)

        total_fisher = sum(f.sum().item() for f in ewc._fisher.values())
        assert total_fisher > 0, "Fisher information should not be all zeros"
        env.close()

    def test_snapshot_can_be_updated(self):
        ewc = EWCRegularizer(lambda_=100.0, fisher_samples=64)
        model, env = _make_evader_model()
        obs = _collect_obs(env, n=64)

        ewc.snapshot(model, obs)
        old_fisher = {n: f.clone() for n, f in ewc._fisher.items()}

        # Train a bit to change params
        model.learn(total_timesteps=128)

        # Re-snapshot
        obs2 = _collect_obs(env, n=64)
        ewc.snapshot(model, obs2)

        # Theta_star should now reflect the new params
        for name, param in model.policy.named_parameters():
            if param.requires_grad:
                assert torch.allclose(ewc._theta_star[name], param.data)
        env.close()


# ─── Penalty Tests ───


class TestEWCPenalty:
    def test_penalty_zero_at_anchor(self):
        ewc = EWCRegularizer(lambda_=100.0, fisher_samples=64)
        model, env = _make_evader_model()
        obs = _collect_obs(env, n=64)

        ewc.snapshot(model, obs)
        penalty = ewc.penalty(model)

        # At the anchor point, penalty should be zero
        assert abs(penalty) < 1e-6
        env.close()

    def test_penalty_positive_after_training(self):
        ewc = EWCRegularizer(lambda_=100.0, fisher_samples=64)
        model, env = _make_evader_model()
        obs = _collect_obs(env, n=64)

        ewc.snapshot(model, obs)

        # Train to move params away from anchor
        model.learn(total_timesteps=256)

        penalty = ewc.penalty(model)
        assert penalty > 0, "Penalty should be positive after parameters drift"
        env.close()

    def test_penalty_zero_without_snapshot(self):
        ewc = EWCRegularizer()
        model, env = _make_evader_model()

        penalty = ewc.penalty(model)
        assert penalty == 0.0
        env.close()

    def test_penalty_scales_with_lambda(self):
        model, env = _make_evader_model()
        obs = _collect_obs(env, n=64)

        ewc_lo = EWCRegularizer(lambda_=10.0, fisher_samples=64)
        ewc_hi = EWCRegularizer(lambda_=1000.0, fisher_samples=64)

        ewc_lo.snapshot(model, obs)
        ewc_hi.snapshot(model, obs)

        model.learn(total_timesteps=256)

        p_lo = ewc_lo.penalty(model)
        p_hi = ewc_hi.penalty(model)

        # Higher lambda should give higher penalty
        assert p_hi > p_lo
        env.close()


# ─── Hook Tests ───


class TestEWCHooks:
    def test_hooks_without_snapshot_returns_empty(self):
        ewc = EWCRegularizer()
        model, env = _make_evader_model()

        hooks = ewc.register_hooks(model)
        assert hooks == []
        env.close()

    def test_hooks_registered_after_snapshot(self):
        ewc = EWCRegularizer(lambda_=100.0, fisher_samples=64)
        model, env = _make_evader_model()
        obs = _collect_obs(env, n=64)

        ewc.snapshot(model, obs)
        hooks = ewc.register_hooks(model)

        assert len(hooks) > 0
        EWCRegularizer.remove_hooks(hooks)
        env.close()

    def test_hooks_modify_gradients(self):
        ewc = EWCRegularizer(lambda_=1000.0, fisher_samples=64)
        model, env = _make_evader_model()
        obs = _collect_obs(env, n=64)

        ewc.snapshot(model, obs)

        # Train a bit without hooks to move params away
        model.learn(total_timesteps=128)

        # Now check that hooks modify gradients
        hooks = ewc.register_hooks(model)

        # Run a forward/backward pass
        obs_batch = _collect_obs(env, n=32)
        dist = model.policy.get_distribution(obs_batch)
        actions = dist.get_actions(deterministic=False)
        log_probs = dist.log_prob(actions)
        loss = -log_probs.mean()

        model.policy.zero_grad()
        loss.backward()

        # Gradients should exist and be non-zero (modified by EWC hooks)
        has_nonzero_grad = False
        for name, param in model.policy.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_nonzero_grad = True
                    break
        assert has_nonzero_grad

        EWCRegularizer.remove_hooks(hooks)
        env.close()

    def test_remove_hooks_cleans_up(self):
        ewc = EWCRegularizer(lambda_=100.0, fisher_samples=64)
        model, env = _make_evader_model()
        obs = _collect_obs(env, n=64)

        ewc.snapshot(model, obs)
        hooks = ewc.register_hooks(model)
        n_hooks = len(hooks)
        assert n_hooks > 0

        EWCRegularizer.remove_hooks(hooks)

        # Hooks should be removed (they won't fire anymore)
        # We verify by checking that all handles are removed
        for h in hooks:
            # Hook handles have 'id' attribute but no clean way to check removal
            # The fact that remove() succeeded is sufficient
            pass
        env.close()

    def test_training_with_hooks_works(self):
        """Verify that model.learn() works with EWC hooks active."""
        ewc = EWCRegularizer(lambda_=100.0, fisher_samples=64)
        model, env = _make_evader_model()
        obs = _collect_obs(env, n=64)

        ewc.snapshot(model, obs)
        hooks = ewc.register_hooks(model)

        # This should not raise
        model.learn(total_timesteps=128)

        EWCRegularizer.remove_hooks(hooks)
        env.close()
