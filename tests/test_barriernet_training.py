"""Tests for BarrierNet training integration (Phase 2.5 Session 4).

Tests cover:
1. Rollout collection with QP-in-the-loop
2. Training loop runs without errors
3. Gradient stability (no NaN/Inf)
4. QP metrics tracking
5. Checkpoint save/load during training
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.barriernet_ppo import BarrierNetPPO, BarrierNetPPOConfig
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper
from training.barriernet_trainer import BarrierNetTrainer, BarrierNetTrainerConfig


# --- Fixtures ---

@pytest.fixture
def pe_env():
    """PE environment with 2 obstacles, wrapped for single agent."""
    env = PursuitEvasionEnv(
        arena_width=20.0,
        arena_height=20.0,
        dt=0.05,
        max_steps=100,
        capture_radius=0.5,
        collision_radius=0.3,
        robot_radius=0.15,
        pursuer_v_max=1.0,
        pursuer_omega_max=2.84,
        evader_v_max=1.0,
        evader_omega_max=2.84,
        n_obstacles=2,
        obstacle_radius_range=(0.3, 0.8),
        obstacle_margin=0.5,
        n_obstacle_obs=2,
    )
    wrapped = SingleAgentPEWrapper(env, role="pursuer", opponent_policy=None)
    yield wrapped
    wrapped.close()


@pytest.fixture
def agent(pe_env):
    """BarrierNet PPO agent matching env obs_dim."""
    obs_dim = pe_env.observation_space.shape[0]
    config = BarrierNetPPOConfig(
        obs_dim=obs_dim,
        hidden_dim=64,
        n_layers=2,
        n_constraints_max=8,
        lr_actor=3e-4,
        lr_critic=3e-4,
        n_epochs=2,
        batch_size=16,
    )
    return BarrierNetPPO(config)


@pytest.fixture
def trainer(pe_env, agent):
    """BarrierNet trainer with short rollouts."""
    config = BarrierNetTrainerConfig(
        rollout_length=32,
        total_timesteps=64,
        log_interval=1,
        save_interval=100,
    )
    return BarrierNetTrainer(pe_env, agent, config)


# ==============================================================================
# Test Class 1: Rollout Collection
# ==============================================================================


class TestRolloutCollection:
    """Test rollout collection with QP-in-the-loop."""

    def test_collect_rollout_runs(self, trainer):
        """Rollout collection should complete without errors."""
        buffer, info = trainer.collect_rollout()
        assert len(buffer) == 32
        assert info["n_steps"] == 32

    def test_rollout_returns_valid_metrics(self, trainer):
        """Rollout info should contain all expected metrics."""
        _, info = trainer.collect_rollout()

        assert "n_steps" in info
        assert "mean_qp_time_ms" in info
        assert "intervention_rate" in info
        assert "infeasibility_rate" in info
        assert info["mean_qp_time_ms"] >= 0
        assert 0 <= info["intervention_rate"] <= 1
        assert 0 <= info["infeasibility_rate"] <= 1

    def test_rollout_buffer_has_states(self, trainer):
        """Buffer should contain robot states for QP reconstruction."""
        buffer, _ = trainer.collect_rollout()

        assert len(buffer.states) == 32
        assert buffer.states[0].shape == (3,)  # [x, y, theta]

    def test_rollout_buffer_has_obstacles(self, trainer):
        """Buffer should store obstacle info."""
        buffer, _ = trainer.collect_rollout()

        assert len(buffer.obstacles) == 32
        # Obstacles should be a list of dicts
        if buffer.obstacles[0] is not None:
            assert isinstance(buffer.obstacles[0], list)

    def test_rollout_actions_within_bounds(self, trainer):
        """Collected actions should be within control bounds."""
        buffer, _ = trainer.collect_rollout()

        for action in buffer.actions:
            assert action[0] >= -0.01, f"v below min: {action[0]}"
            assert action[0] <= 1.01, f"v above max: {action[0]}"
            assert action[1] >= -2.85, f"omega below min: {action[1]}"
            assert action[1] <= 2.85, f"omega above max: {action[1]}"


# ==============================================================================
# Test Class 2: Training Loop
# ==============================================================================


class TestTrainingLoop:
    """Test the full training loop."""

    def test_train_runs(self, trainer):
        """Training should complete without errors."""
        metrics = trainer.train()
        assert len(metrics) > 0
        assert trainer.total_timesteps >= 64

    def test_train_produces_metrics(self, trainer):
        """Each iteration should produce complete metrics."""
        metrics = trainer.train()

        for m in metrics:
            assert "iteration" in m
            assert "timesteps" in m
            assert "policy_loss" in m
            assert "critic_loss" in m
            assert "entropy" in m
            assert "mean_qp_correction" in m

    def test_training_changes_parameters(self, pe_env, agent):
        """Training should update actor parameters."""
        config = BarrierNetTrainerConfig(
            rollout_length=32,
            total_timesteps=32,
        )
        trainer = BarrierNetTrainer(pe_env, agent, config)

        # Store initial params
        initial_params = {
            name: param.clone()
            for name, param in agent.actor.named_parameters()
        }

        trainer.train()

        # Check some params changed
        changed = any(
            not torch.allclose(initial_params[name], param, atol=1e-7)
            for name, param in agent.actor.named_parameters()
        )
        assert changed, "No actor parameters changed after training"

    def test_gradient_stability(self, pe_env, agent):
        """Gradients should remain finite during training."""
        config = BarrierNetTrainerConfig(
            rollout_length=32,
            total_timesteps=32,
        )
        trainer = BarrierNetTrainer(pe_env, agent, config)
        trainer.train()

        # Check for NaN/Inf in parameters
        for name, param in agent.actor.named_parameters():
            assert torch.all(torch.isfinite(param)), f"Non-finite param: {name}"

    def test_multiple_iterations(self, pe_env, agent):
        """Training should run multiple iterations."""
        config = BarrierNetTrainerConfig(
            rollout_length=32,
            total_timesteps=128,
        )
        trainer = BarrierNetTrainer(pe_env, agent, config)
        metrics = trainer.train()

        assert len(metrics) >= 3  # At least 3 iterations (128/32 = 4)


# ==============================================================================
# Test Class 3: QP Metrics
# ==============================================================================


class TestQPMetrics:
    """Test QP-specific metrics during training."""

    def test_qp_time_reasonable(self, trainer):
        """QP solve time should be reasonable (<50ms per step on CPU)."""
        _, info = trainer.collect_rollout()
        assert info["mean_qp_time_ms"] < 50, (
            f"QP too slow: {info['mean_qp_time_ms']:.1f}ms"
        )

    def test_infeasibility_rate_low(self, trainer):
        """Infeasibility rate should be low (<50% — generous for testing)."""
        _, info = trainer.collect_rollout()
        assert info["infeasibility_rate"] < 0.5

    def test_qp_correction_tracked(self, trainer):
        """Training metrics should include QP correction magnitude."""
        metrics = trainer.train()
        for m in metrics:
            assert "mean_qp_correction" in m
            assert m["mean_qp_correction"] >= 0


# ==============================================================================
# Test Class 4: Checkpoint Save/Load
# ==============================================================================


class TestCheckpointing:
    """Test checkpoint save/load during training."""

    def test_save_during_training(self, pe_env, agent):
        """Checkpoints should save during training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BarrierNetTrainerConfig(
                rollout_length=32,
                total_timesteps=64,
                save_interval=1,
                save_dir=tmpdir,
            )
            trainer = BarrierNetTrainer(pe_env, agent, config)
            trainer.train()

            # Check checkpoint exists
            files = os.listdir(tmpdir)
            assert any(f.endswith(".pt") for f in files), f"No checkpoint: {files}"

    def test_load_after_save(self, pe_env, agent):
        """Loaded model should produce same actions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            agent.save(path)

            loaded = BarrierNetPPO.load(path)

            # Compare actions
            obs = torch.randn(1, agent.config.obs_dim)
            states = torch.tensor([[0.0, 0.0, 0.0]])

            u1, _, _, _ = agent.get_action(obs, states, deterministic=True)
            u2, _, _, _ = loaded.get_action(obs, states, deterministic=True)

            assert torch.allclose(u1, u2, atol=1e-5)


# ==============================================================================
# Test Class 5: Training Summary
# ==============================================================================


class TestTrainingSummary:
    """Test training summary generation."""

    def test_summary_after_training(self, trainer):
        """Summary should be available after training."""
        trainer.train()
        summary = trainer.get_training_summary()

        assert "total_iterations" in summary
        assert "total_timesteps" in summary
        assert summary["total_timesteps"] >= 64

    def test_summary_empty_before_training(self, trainer):
        """Summary should be empty before training."""
        summary = trainer.get_training_summary()
        assert summary == {}


# ==============================================================================
# Test Class 6: Environment Integration
# ==============================================================================


class TestEnvIntegration:
    """Test integration with the PE environment."""

    def test_env_reset_provides_state(self, pe_env):
        """Env should provide robot states via internal attributes."""
        pe_env.reset(seed=42)
        pe = pe_env.env  # unwrap SingleAgentPEWrapper

        assert pe.pursuer_state is not None
        assert pe.evader_state is not None
        assert pe.pursuer_state.shape == (3,)
        assert pe.evader_state.shape == (3,)

    def test_env_step_updates_state(self, pe_env):
        """Env step should update robot states."""
        pe_env.reset(seed=42)
        pe = pe_env.env

        old_state = pe.pursuer_state.copy()
        action = np.array([0.5, 0.5], dtype=np.float32)
        pe_env.step(action)
        new_state = pe.pursuer_state

        assert not np.allclose(old_state, new_state), (
            "State didn't change after step"
        )

    def test_env_has_obstacles(self, pe_env):
        """Env should have obstacles available."""
        pe_env.reset(seed=42)
        pe = pe_env.env

        assert hasattr(pe, "obstacles")
        assert len(pe.obstacles) == 2  # configured with 2 obstacles
        assert "x" in pe.obstacles[0]
        assert "y" in pe.obstacles[0]
        assert "radius" in pe.obstacles[0]

    def test_obs_dim_matches_config(self, pe_env, agent):
        """Observation dim should match agent config."""
        obs, _ = pe_env.reset()
        assert obs.shape[0] == agent.config.obs_dim
