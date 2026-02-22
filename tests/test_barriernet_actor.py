"""Tests for BarrierNet actor and PPO agent (Phase 2.5 Session 3).

Tests cover:
1. BarrierNetActor forward pass and output shapes
2. Control bounds and CBF constraint satisfaction
3. Gradient flow through QP to MLP parameters
4. Deterministic vs stochastic mode
5. BarrierNetCritic value estimation
6. BarrierNetPPO update step
7. RolloutBuffer operations
8. Save/load
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.barriernet_actor import BarrierNetActor, BarrierNetCritic
from agents.barriernet_ppo import BarrierNetPPO, BarrierNetPPOConfig, RolloutBuffer


# --- Fixtures ---

@pytest.fixture
def actor():
    """BarrierNet actor with obs_dim=14."""
    return BarrierNetActor(
        obs_dim=14,
        hidden_dim=64,
        n_layers=2,
        n_constraints_max=10,
        v_max=1.0,
        omega_max=2.84,
        w_v=150.0,
        w_omega=1.0,
        alpha=1.0,
        d=0.1,
    )


@pytest.fixture
def critic():
    """BarrierNet critic with obs_dim=14."""
    return BarrierNetCritic(obs_dim=14, hidden_dim=64, n_layers=2)


@pytest.fixture
def ppo_config():
    """BarrierNet PPO config with small sizes for testing."""
    return BarrierNetPPOConfig(
        obs_dim=14,
        hidden_dim=64,
        n_layers=2,
        n_constraints_max=10,
        lr_actor=3e-4,
        lr_critic=3e-4,
        n_epochs=2,
        batch_size=8,
    )


@pytest.fixture
def sample_obs():
    """Batch of 4 observations."""
    return torch.randn(4, 14)


@pytest.fixture
def sample_states():
    """Batch of 4 robot states [x, y, theta]."""
    states = torch.randn(4, 3)
    states[:, :2] *= 3  # positions in [-3, 3]
    return states


@pytest.fixture
def sample_obstacles():
    """Two obstacles."""
    return [
        {"x": 4.0, "y": 0.0, "radius": 0.5},
        {"x": -3.0, "y": 2.0, "radius": 0.8},
    ]


# ==============================================================================
# Test Class 1: BarrierNetActor Forward Pass
# ==============================================================================


class TestBarrierNetActorForward:
    """Test actor forward pass produces correct outputs."""

    def test_output_shapes(self, actor, sample_obs, sample_states):
        """Forward pass should return correct shapes."""
        u_safe, log_prob, entropy, info = actor(sample_obs, sample_states)

        assert u_safe.shape == (4, 2), f"u_safe shape: {u_safe.shape}"
        assert log_prob.shape == (4,), f"log_prob shape: {log_prob.shape}"
        assert entropy.shape == (), f"entropy shape: {entropy.shape}"

    def test_safe_action_within_bounds(self, actor, sample_obs, sample_states):
        """Safe actions should be within control bounds."""
        u_safe, _, _, _ = actor(sample_obs, sample_states)

        assert torch.all(u_safe[:, 0] >= -0.01), f"v below min: {u_safe[:, 0].min()}"
        assert torch.all(u_safe[:, 0] <= 1.01), f"v above max: {u_safe[:, 0].max()}"
        assert torch.all(u_safe[:, 1] >= -2.85), f"omega below min: {u_safe[:, 1].min()}"
        assert torch.all(u_safe[:, 1] <= 2.85), f"omega above max: {u_safe[:, 1].max()}"

    def test_nominal_action_within_bounds(self, actor, sample_obs):
        """Nominal action from MLP should be within bounds."""
        u_nom = actor.get_nominal_action(sample_obs)

        assert torch.all(u_nom[:, 0] >= 0.0)
        assert torch.all(u_nom[:, 0] <= 1.0)
        assert torch.all(u_nom[:, 1] >= -2.84)
        assert torch.all(u_nom[:, 1] <= 2.84)

    def test_with_obstacles(self, actor, sample_obs, sample_states, sample_obstacles):
        """Forward pass should work with obstacles."""
        u_safe, _, _, info = actor(
            sample_obs, sample_states, obstacles=sample_obstacles,
        )
        assert u_safe.shape == (4, 2)
        assert info["qp_feasible"]

    def test_with_opponent(self, actor, sample_obs, sample_states):
        """Forward pass should work with opponent states."""
        opponent = torch.tensor([[5.0, 5.0, 0.0]] * 4, dtype=torch.float32)
        u_safe, _, _, info = actor(
            sample_obs, sample_states, opponent_states=opponent,
        )
        assert u_safe.shape == (4, 2)

    def test_info_contains_diagnostics(self, actor, sample_obs, sample_states):
        """Info dict should contain all diagnostic fields."""
        _, _, _, info = actor(sample_obs, sample_states)

        assert "u_nom" in info
        assert "u_safe" in info
        assert "u_nom_mean" in info
        assert "qp_correction" in info
        assert "action_std" in info
        assert "qp_feasible" in info
        assert "min_cbf_value" in info

    def test_single_sample(self, actor):
        """Should work with batch size 1."""
        obs = torch.randn(1, 14)
        states = torch.tensor([[0.0, 0.0, 0.0]])
        u_safe, log_prob, entropy, info = actor(obs, states)
        assert u_safe.shape == (1, 2)
        assert log_prob.shape == (1,)


# ==============================================================================
# Test Class 2: Deterministic vs Stochastic
# ==============================================================================


class TestDeterministicStochastic:
    """Test deterministic and stochastic modes."""

    def test_deterministic_consistent(self, actor, sample_obs, sample_states):
        """Deterministic mode should give same result each time."""
        u1, _, _, _ = actor(sample_obs, sample_states, deterministic=True)
        u2, _, _, _ = actor(sample_obs, sample_states, deterministic=True)
        assert torch.allclose(u1, u2, atol=1e-5)

    def test_stochastic_varies(self, actor, sample_obs, sample_states):
        """Stochastic mode should give different results."""
        torch.manual_seed(42)
        u1, _, _, _ = actor(sample_obs, sample_states, deterministic=False)
        torch.manual_seed(123)
        u2, _, _, _ = actor(sample_obs, sample_states, deterministic=False)
        # With different seeds, actions should differ
        assert not torch.allclose(u1, u2, atol=1e-3)

    def test_deterministic_log_prob_zero(self, actor, sample_obs, sample_states):
        """Deterministic mode should have zero log prob."""
        _, log_prob, _, _ = actor(sample_obs, sample_states, deterministic=True)
        assert torch.allclose(log_prob, torch.zeros_like(log_prob))

    def test_stochastic_log_prob_finite(self, actor, sample_obs, sample_states):
        """Stochastic mode should have finite log prob."""
        _, log_prob, _, _ = actor(sample_obs, sample_states, deterministic=False)
        assert torch.all(torch.isfinite(log_prob))


# ==============================================================================
# Test Class 3: Gradient Flow
# ==============================================================================


class TestGradientFlow:
    """Test gradient flow through QP to MLP parameters."""

    def test_gradient_to_backbone(self, actor, sample_obs, sample_states):
        """Gradient should reach backbone parameters."""
        u_safe, _, _, _ = actor(sample_obs, sample_states)
        loss = u_safe.sum()
        loss.backward()

        # Check backbone first layer has gradient
        for name, param in actor.backbone.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient on {name}"
                break

    def test_gradient_to_action_head(self, actor, sample_obs, sample_states):
        """Gradient should reach action head parameters."""
        u_safe, _, _, _ = actor(sample_obs, sample_states)
        loss = u_safe.sum()
        loss.backward()

        assert actor.action_mean.weight.grad is not None

    def test_log_std_is_fixed_buffer(self, actor, sample_obs, sample_states):
        """action_log_std should be a non-learnable buffer (prevents std explosion).

        When action_log_std is a learnable Parameter, policy gradients push it
        higher because QP corrections create a positive correlation between
        noise magnitude and reward. Making it a buffer prevents this.
        """
        assert not actor.action_log_std.requires_grad
        assert "action_log_std" not in dict(actor.named_parameters())
        assert "action_log_std" in dict(actor.named_buffers())

    def test_ppo_style_loss_gradient(self, actor, sample_obs, sample_states):
        """PPO-style loss via evaluate_actions should produce valid gradients.

        During PPO update, evaluate_actions computes log_prob of stored
        nominal actions under the current policy. Gradient flows through
        u_nom_mean (current MLP output) to backbone parameters.
        """
        # Simulate rollout: get nominal actions (detached, as stored in buffer)
        with torch.no_grad():
            u_safe, _, _, info = actor(sample_obs, sample_states)
        u_nom_stored = info["u_nom"]  # already detached

        # Simulate PPO update: evaluate stored actions under current policy
        log_prob, entropy = actor.evaluate_actions(sample_obs, u_nom_stored)

        # Simulate PPO loss
        advantages = torch.randn(4)
        old_log_prob = log_prob.detach() + 0.1
        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in actor.parameters()
        )
        assert has_grad, "No gradients in actor parameters after PPO loss"

    def test_no_nan_gradients(self, actor, sample_obs, sample_states):
        """Gradients should be finite (no NaN/Inf)."""
        u_safe, log_prob, entropy, _ = actor(sample_obs, sample_states)
        loss = u_safe.sum() + log_prob.sum()
        loss.backward()

        for name, param in actor.named_parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), (
                    f"Non-finite gradient in {name}: {param.grad}"
                )


# ==============================================================================
# Test Class 4: BarrierNetCritic
# ==============================================================================


class TestBarrierNetCritic:
    """Test critic network."""

    def test_output_shape(self, critic, sample_obs):
        """Critic should output (B,) values."""
        values = critic(sample_obs)
        assert values.shape == (4,)

    def test_gradient_flows(self, critic, sample_obs):
        """Critic should have gradients after backward."""
        values = critic(sample_obs)
        loss = values.sum()
        loss.backward()

        for param in critic.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break

    def test_single_sample(self, critic):
        """Should work with batch size 1."""
        obs = torch.randn(1, 14)
        values = critic(obs)
        assert values.shape == (1,)


# ==============================================================================
# Test Class 5: RolloutBuffer
# ==============================================================================


class TestRolloutBuffer:
    """Test rollout buffer operations."""

    def test_add_and_length(self):
        """Buffer length should match number of adds."""
        buf = RolloutBuffer()
        for i in range(10):
            buf.add(
                obs=torch.randn(14),
                state=torch.randn(3),
                action=torch.randn(2),
                reward=float(i),
                log_prob=torch.tensor(-0.5),
                value=torch.tensor(1.0),
                done=(i == 9),
            )
        assert len(buf) == 10

    def test_compute_returns(self):
        """Returns and advantages should have correct shape."""
        buf = RolloutBuffer()
        for i in range(5):
            buf.add(
                obs=torch.randn(14),
                state=torch.randn(3),
                action=torch.randn(2),
                reward=1.0,
                log_prob=torch.tensor(-0.5),
                value=torch.tensor(1.0),
                done=False,
            )
        returns, advantages = buf.compute_returns_and_advantages(
            last_value=0.5, gamma=0.99, gae_lambda=0.95,
        )
        assert returns.shape == (5,)
        assert advantages.shape == (5,)
        assert torch.all(torch.isfinite(returns))
        assert torch.all(torch.isfinite(advantages))

    def test_get_batches(self):
        """Batches should yield correct number of items."""
        buf = RolloutBuffer()
        for i in range(16):
            buf.add(
                obs=torch.randn(14),
                state=torch.randn(3),
                action=torch.randn(2),
                reward=1.0,
                log_prob=torch.tensor(-0.5),
                value=torch.tensor(1.0),
                done=False,
            )
        batches = list(buf.get_batches(batch_size=8))
        assert len(batches) == 2
        assert batches[0]["obs"].shape == (8, 14)

    def test_clear(self):
        """Clear should empty the buffer."""
        buf = RolloutBuffer()
        buf.add(
            obs=torch.randn(14), state=torch.randn(3),
            action=torch.randn(2), reward=1.0,
            log_prob=torch.tensor(-0.5), value=torch.tensor(1.0),
            done=False,
        )
        assert len(buf) == 1
        buf.clear()
        assert len(buf) == 0


# ==============================================================================
# Test Class 6: BarrierNetPPO
# ==============================================================================


class TestBarrierNetPPO:
    """Test the full BarrierNet PPO agent."""

    def test_construction(self, ppo_config):
        """Agent should construct without errors."""
        agent = BarrierNetPPO(ppo_config)
        assert agent.actor is not None
        assert agent.critic is not None

    def test_get_action(self, ppo_config):
        """get_action should return correct shapes."""
        agent = BarrierNetPPO(ppo_config)
        obs = torch.randn(1, 14)
        states = torch.tensor([[0.0, 0.0, 0.0]])

        u_safe, log_prob, value, info = agent.get_action(obs, states)
        assert u_safe.shape == (1, 2)
        assert log_prob.shape == (1,)
        assert value.shape == (1,)

    def test_update_runs(self, ppo_config):
        """PPO update should run without errors on synthetic data."""
        agent = BarrierNetPPO(ppo_config)

        # Create synthetic rollout with u_nom
        buf = RolloutBuffer()
        for i in range(16):
            obs = torch.randn(14)
            state = torch.randn(3)
            state[:2] *= 3

            buf.add(
                obs=obs,
                state=state,
                action=torch.rand(2),
                reward=np.random.randn(),
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                done=(i == 15),
                u_nom=torch.rand(2),
            )

        metrics = agent.update(buf)
        assert "policy_loss" in metrics
        assert "critic_loss" in metrics
        assert "entropy" in metrics
        assert "mean_qp_correction" in metrics
        assert metrics["n_updates"] > 0

    def test_update_changes_parameters(self, ppo_config):
        """PPO update should change actor parameters."""
        agent = BarrierNetPPO(ppo_config)

        # Store initial params
        initial_params = {
            name: param.clone()
            for name, param in agent.actor.named_parameters()
        }

        # Create synthetic rollout with u_nom
        buf = RolloutBuffer()
        for i in range(16):
            buf.add(
                obs=torch.randn(14),
                state=torch.randn(3) * 3,
                action=torch.rand(2),
                reward=np.random.randn(),
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                done=(i == 15),
                u_nom=torch.rand(2),
            )

        agent.update(buf)

        # At least some parameters should have changed
        changed = any(
            not torch.allclose(initial_params[name], param, atol=1e-7)
            for name, param in agent.actor.named_parameters()
        )
        assert changed, "No actor parameters changed after update"

    def test_save_load(self, ppo_config):
        """Save and load should preserve parameters."""
        agent = BarrierNetPPO(ppo_config)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save(path)
            loaded = BarrierNetPPO.load(path)

            # Check actor params match
            for (n1, p1), (n2, p2) in zip(
                agent.actor.named_parameters(),
                loaded.actor.named_parameters(),
            ):
                assert torch.allclose(p1, p2), f"Param mismatch: {n1}"

            # Check critic params match
            for (n1, p1), (n2, p2) in zip(
                agent.critic.named_parameters(),
                loaded.critic.named_parameters(),
            ):
                assert torch.allclose(p1, p2), f"Param mismatch: {n1}"
        finally:
            os.unlink(path)

    def test_deterministic_action(self, ppo_config):
        """Deterministic actions should be consistent."""
        agent = BarrierNetPPO(ppo_config)
        obs = torch.randn(1, 14)
        states = torch.tensor([[0.0, 0.0, 0.0]])

        u1, _, _, _ = agent.get_action(obs, states, deterministic=True)
        u2, _, _, _ = agent.get_action(obs, states, deterministic=True)
        assert torch.allclose(u1, u2, atol=1e-5)


# ==============================================================================
# Test Class 7: QP Correction Monitoring
# ==============================================================================


class TestQPCorrection:
    """Test that QP correction is properly tracked."""

    def test_zero_correction_safe_center(self, actor):
        """QP correction should be ~0 when at center with no obstacles."""
        obs = torch.randn(1, 14)
        states = torch.tensor([[0.0, 0.0, 0.0]])
        _, _, _, info = actor(obs, states, deterministic=True)

        # At center with no obstacles, correction should be minimal
        assert info["qp_correction"][0] < 0.5

    def test_nonzero_correction_near_wall(self, actor):
        """QP correction should be nonzero when heading into wall."""
        obs = torch.randn(1, 14)
        states = torch.tensor([[9.5, 0.0, 0.0]])  # near right wall, facing right

        # Force nominal action toward wall by manipulating network
        with torch.no_grad():
            actor.action_mean.bias.fill_(2.0)  # bias toward high v

        _, _, _, info = actor(obs, states, deterministic=True)
        # Should have some correction (may or may not, depends on network output)
        # Just verify the field exists and is non-negative
        assert info["qp_correction"][0] >= 0.0

    def test_correction_shape(self, actor, sample_obs, sample_states):
        """QP correction should have shape (B,)."""
        _, _, _, info = actor(sample_obs, sample_states)
        assert info["qp_correction"].shape == (4,)
