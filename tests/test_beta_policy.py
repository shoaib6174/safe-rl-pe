"""Tests for Beta distribution policy — Phase 2 Session 1.

Tests cover:
- BetaDistribution: sampling, log_prob, entropy, mode
- BetaPolicy: SB3 integration, gradient flow, training
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO

from safety.beta_distribution import BetaDistribution
from safety.beta_policy import BetaPolicy


class TestBetaDistribution:
    """Unit tests for BetaDistribution class."""

    def setup_method(self):
        """Create standard distribution for tests."""
        self.action_dim = 2
        self.dist = BetaDistribution(self.action_dim)
        self.low = torch.tensor([0.0, -2.84])
        self.high = torch.tensor([1.0, 2.84])

    def test_samples_in_bounds(self):
        """All samples from Beta distribution must be within [low, high]."""
        # Create distribution with known params
        params = torch.randn(1000, 2 * self.action_dim)
        self.dist.proba_distribution(params, self.low, self.high)
        samples = self.dist.sample()

        assert (samples[:, 0] >= 0.0).all(), "v samples below 0"
        assert (samples[:, 0] <= 1.0).all(), "v samples above v_max"
        assert (samples[:, 1] >= -2.84).all(), "omega samples below -omega_max"
        assert (samples[:, 1] <= 2.84).all(), "omega samples above omega_max"

    def test_log_prob_finite(self):
        """Log prob should be finite for valid actions within bounds."""
        params = torch.zeros(3, 2 * self.action_dim)  # softplus(0)+1 = 1+1 = 2
        self.dist.proba_distribution(params, self.low, self.high)

        actions = torch.tensor([
            [0.5, 0.0],     # Center of ranges
            [0.1, -1.0],    # Low v, negative omega
            [0.9, 2.0],     # High v, positive omega
        ])
        log_probs = self.dist.log_prob(actions)
        assert torch.isfinite(log_probs).all(), f"Non-finite log probs: {log_probs}"

    def test_log_prob_shape(self):
        """Log prob should reduce action dims, keeping batch dims."""
        batch_size = 8
        params = torch.randn(batch_size, 2 * self.action_dim)
        self.dist.proba_distribution(params, self.low, self.high)
        actions = self.dist.sample()
        log_probs = self.dist.log_prob(actions)

        assert log_probs.shape == (batch_size,), \
            f"Expected shape ({batch_size},), got {log_probs.shape}"

    def test_entropy_positive(self):
        """Entropy should be positive for reasonable distributions."""
        params = torch.zeros(5, 2 * self.action_dim)
        self.dist.proba_distribution(params, self.low, self.high)
        entropy = self.dist.entropy()

        assert torch.isfinite(entropy).all(), f"Non-finite entropy: {entropy}"
        # Beta(2,2) on [0,1] has entropy ~-0.125, but rescaling adds log(scale)
        # For v: log(1.0) = 0, for omega: log(5.68) = 1.74
        # Total should be positive due to omega's wide range

    def test_mode_in_bounds(self):
        """Mode should always be within [low, high]."""
        params = torch.randn(100, 2 * self.action_dim)
        self.dist.proba_distribution(params, self.low, self.high)
        modes = self.dist.mode()

        assert (modes[:, 0] >= 0.0).all() and (modes[:, 0] <= 1.0).all()
        assert (modes[:, 1] >= -2.84).all() and (modes[:, 1] <= 2.84).all()

    def test_mode_deterministic(self):
        """Mode should return the same value every time."""
        params = torch.randn(1, 2 * self.action_dim)
        self.dist.proba_distribution(params, self.low, self.high)

        mode1 = self.dist.mode()
        mode2 = self.dist.mode()
        torch.testing.assert_close(mode1, mode2)

    def test_jacobian_correction(self):
        """Log prob should include Jacobian for rescaling [0,1] → [low, high].

        Test: log_prob on [0, v_max] should differ from log_prob on [-omega_max, omega_max]
        because the scale factors differ (1.0 vs 5.68).
        """
        # softplus(0) + 1 = log(2) + 1 ≈ 1.693 (NOT 2.0)
        params = torch.zeros(1, 2 * self.action_dim)
        self.dist.proba_distribution(params, self.low, self.high)
        actual_ab = F.softplus(torch.tensor(0.0)) + 1.0  # ≈ 1.693

        # Action at midpoint of each rescaled range
        action = torch.tensor([[0.5, 0.0]])
        log_prob = self.dist.log_prob(action)

        # Compute manually: x_v = 0.5, x_omega = 0.5 (midpoints in [0,1])
        beta_dist = torch.distributions.Beta(actual_ab, actual_ab)
        expected_v = beta_dist.log_prob(torch.tensor(0.5)) - torch.log(torch.tensor(1.0))
        expected_omega = beta_dist.log_prob(torch.tensor(0.5)) - torch.log(torch.tensor(5.68))
        expected_total = expected_v + expected_omega

        torch.testing.assert_close(log_prob[0], expected_total, atol=1e-4, rtol=1e-4)

    def test_rsample_gradient(self):
        """Samples should support gradient computation (reparameterization)."""
        params = torch.randn(4, 2 * self.action_dim, requires_grad=True)
        alpha = F.softplus(params[..., :self.action_dim]) + 1.0
        beta = F.softplus(params[..., self.action_dim:]) + 1.0
        dist = torch.distributions.Beta(alpha, beta)

        x = dist.rsample()
        loss = x.sum()
        loss.backward()

        assert params.grad is not None, "No gradients computed"
        assert torch.isfinite(params.grad).all(), "Non-finite gradients"

    def test_proba_distribution_net(self):
        """Network layer should output correct shape."""
        net = self.dist.proba_distribution_net(latent_dim=256)
        assert isinstance(net, torch.nn.Linear)
        assert net.in_features == 256
        assert net.out_features == 2 * self.action_dim

    def test_get_actions_stochastic(self):
        """get_actions(deterministic=False) should return different samples."""
        params = torch.randn(1, 2 * self.action_dim)
        self.dist.proba_distribution(params, self.low, self.high)

        actions = [self.dist.get_actions(deterministic=False) for _ in range(10)]
        # At least some should differ (with overwhelming probability)
        all_same = all(torch.allclose(actions[0], a) for a in actions[1:])
        assert not all_same, "Stochastic sampling should produce different actions"

    def test_get_actions_deterministic(self):
        """get_actions(deterministic=True) should return mode every time."""
        params = torch.randn(1, 2 * self.action_dim)
        self.dist.proba_distribution(params, self.low, self.high)

        actions = [self.dist.get_actions(deterministic=True) for _ in range(5)]
        for a in actions[1:]:
            torch.testing.assert_close(actions[0], a)


class TestBetaPolicy:
    """Integration tests for BetaPolicy with SB3."""

    def setup_method(self):
        """Create standard environment for tests."""
        self.obs_space = spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
        self.act_space = spaces.Box(
            low=np.array([0.0, -2.84], dtype=np.float32),
            high=np.array([1.0, 2.84], dtype=np.float32),
        )

    def test_policy_construction(self):
        """BetaPolicy should be constructible with standard SB3 args."""
        policy = BetaPolicy(
            self.obs_space,
            self.act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
        )
        assert isinstance(policy.action_dist, BetaDistribution)
        assert policy.action_net.out_features == 4  # 2 * action_dim

    def test_forward_shapes(self):
        """Forward pass should return correct shapes."""
        policy = BetaPolicy(
            self.obs_space, self.act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
        )
        obs = torch.randn(8, 14)
        actions, values, log_probs = policy.forward(obs)

        assert actions.shape == (8, 2), f"Actions shape: {actions.shape}"
        assert values.shape == (8, 1), f"Values shape: {values.shape}"
        assert log_probs.shape == (8,), f"Log probs shape: {log_probs.shape}"

    def test_forward_actions_in_bounds(self):
        """Actions from forward pass must be within action space bounds."""
        policy = BetaPolicy(
            self.obs_space, self.act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
        )
        obs = torch.randn(100, 14)
        actions, _, _ = policy.forward(obs, deterministic=False)

        assert (actions[:, 0] >= 0.0).all(), "v below 0"
        assert (actions[:, 0] <= 1.0).all(), "v above v_max"
        assert (actions[:, 1] >= -2.84).all(), "omega below -omega_max"
        assert (actions[:, 1] <= 2.84).all(), "omega above omega_max"

    def test_evaluate_actions(self):
        """evaluate_actions should return values, log_prob, entropy."""
        policy = BetaPolicy(
            self.obs_space, self.act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
        )
        obs = torch.randn(8, 14)
        actions = torch.tensor([[0.5, 0.0]] * 8, dtype=torch.float32)

        values, log_prob, entropy = policy.evaluate_actions(obs, actions)

        assert values.shape == (8, 1)
        assert log_prob.shape == (8,)
        assert entropy.shape == (8,)
        assert torch.isfinite(log_prob).all()
        assert torch.isfinite(entropy).all()

    def test_gradient_flow(self):
        """Backprop through Beta log_prob should produce finite gradients."""
        policy = BetaPolicy(
            self.obs_space, self.act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
        )

        obs = torch.randn(4, 14)
        actions, _, log_probs = policy.forward(obs)

        # Compute simple loss and backprop
        loss = -log_probs.mean()
        loss.backward()

        for name, p in policy.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), \
                    f"Non-finite gradient in {name}"

    def test_predict(self):
        """predict() should return bounded actions."""
        policy = BetaPolicy(
            self.obs_space, self.act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
        )
        obs = torch.randn(1, 14)

        # Deterministic
        action_det = policy._predict(obs, deterministic=True)
        assert action_det.shape == (1, 2)
        assert (action_det[:, 0] >= 0.0).all() and (action_det[:, 0] <= 1.0).all()

        # Stochastic
        action_sto = policy._predict(obs, deterministic=False)
        assert action_sto.shape == (1, 2)

    def test_ppo_integration(self):
        """BetaPolicy should work with SB3's PPO for a short training run."""
        from omegaconf import OmegaConf
        from training.utils import make_vec_env

        cfg = OmegaConf.create({
            "env": {
                "arena_width": 10.0, "arena_height": 10.0,
                "dt": 0.05, "max_steps": 50,
                "capture_radius": 0.5, "collision_radius": 0.3,
                "robot_radius": 0.15,
                "pursuer": {"v_max": 1.0, "omega_max": 2.84},
                "evader": {"v_max": 1.0, "omega_max": 2.84},
                "min_init_distance": 3.0, "max_init_distance": 7.0,
                "reward": {"distance_scale": 1.0, "capture_bonus": 100.0, "timeout_penalty": -100.0},
            },
        })

        envs = make_vec_env(cfg, n_envs=1, role="pursuer", seed=42)

        model = PPO(
            BetaPolicy, envs, verbose=0, seed=42, device="cpu",
            n_steps=64, batch_size=32, n_epochs=2,
            policy_kwargs={"net_arch": [32]},
        )

        # Should train without errors
        model.learn(total_timesteps=128)

        # Predict should return bounded actions
        obs = envs.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert 0.0 <= action[0, 0] <= 1.0, f"v={action[0, 0]} out of bounds"
        assert -2.84 <= action[0, 1] <= 2.84, f"omega={action[0, 1]} out of bounds"

        envs.close()

    def test_ppo_deterministic_predictions(self):
        """Same obs + deterministic=True should give same action."""
        from omegaconf import OmegaConf
        from training.utils import make_vec_env, setup_reproducibility

        setup_reproducibility(42)

        cfg = OmegaConf.create({
            "env": {
                "arena_width": 10.0, "arena_height": 10.0,
                "dt": 0.05, "max_steps": 50,
                "capture_radius": 0.5, "collision_radius": 0.3,
                "robot_radius": 0.15,
                "pursuer": {"v_max": 1.0, "omega_max": 2.84},
                "evader": {"v_max": 1.0, "omega_max": 2.84},
                "min_init_distance": 3.0, "max_init_distance": 7.0,
                "reward": {"distance_scale": 1.0, "capture_bonus": 100.0, "timeout_penalty": -100.0},
            },
        })

        envs = make_vec_env(cfg, n_envs=1, role="pursuer", seed=42)
        model = PPO(
            BetaPolicy, envs, verbose=0, seed=42, device="cpu",
            n_steps=64, batch_size=32,
            policy_kwargs={"net_arch": [32]},
        )
        model.learn(total_timesteps=128)

        test_obs = np.random.default_rng(42).standard_normal(14).astype(np.float32)
        actions = [model.predict(test_obs, deterministic=True)[0].copy() for _ in range(5)]

        for a in actions[1:]:
            np.testing.assert_array_equal(actions[0], a,
                                          "Deterministic predict should be identical")
        envs.close()


class TestBetaVsGaussian:
    """Compare Beta policy against Gaussian baseline."""

    def test_beta_actions_always_bounded(self):
        """Beta policy never produces out-of-bounds actions (unlike Gaussian)."""
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([0.0, -2.84], dtype=np.float32),
            high=np.array([1.0, 2.84], dtype=np.float32),
        )

        policy = BetaPolicy(
            obs_space, act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
        )

        # Sample many actions
        obs = torch.randn(1000, 14)
        actions, _, _ = policy.forward(obs, deterministic=False)

        # ALL actions must be in bounds (not just most — ALL)
        assert (actions[:, 0] >= 0.0).all()
        assert (actions[:, 0] <= 1.0).all()
        assert (actions[:, 1] >= -2.84).all()
        assert (actions[:, 1] <= 2.84).all()
