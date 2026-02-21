"""Tests for Phase 2.5 comparison framework (Session 6 preparation).

Tests cover:
1. EvaluationResult dataclass
2. CBF margin computation
3. BarrierNet evaluation agent wrapper
4. SB3 evaluation agent wrapper
5. evaluate_approach function
6. Comparison report generation
7. Table formatting
"""

import os
import sys
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.comparison_framework import (
    EvaluationResult,
    ComparisonReport,
    BarrierNetEvalAgent,
    SB3EvalAgent,
    compute_cbf_margins,
    evaluate_approach,
    compute_comparison,
    format_comparison_table,
)


# --- Fixtures ---


@pytest.fixture
def dummy_result():
    """Create a dummy EvaluationResult."""
    n = 50
    return EvaluationResult(
        approach_name="test",
        n_episodes=n,
        safety_violation_rate=0.01,
        mean_min_cbf_margin=0.5,
        cbf_margin_values=np.random.uniform(0.1, 1.0, n),
        capture_rate=0.6,
        mean_capture_time=15.0,
        mean_episode_reward=-20.0,
        episode_rewards=np.random.normal(-20, 5, n),
        mean_inference_time_ms=5.0,
        training_wall_clock_hours=10.0,
        qp_infeasibility_rate=0.0,
        mean_qp_correction=0.1,
        intervention_rate=0.3,
        episode_lengths=np.random.randint(50, 600, n),
        capture_times=np.random.uniform(5, 25, 30),
        min_distances=np.random.uniform(0.0, 2.0, n),
    )


@pytest.fixture
def pe_env():
    """PE environment for evaluation tests."""
    from envs.pursuit_evasion_env import PursuitEvasionEnv
    from envs.wrappers import SingleAgentPEWrapper

    env = PursuitEvasionEnv(
        arena_width=20.0,
        arena_height=20.0,
        dt=0.05,
        max_steps=50,  # short episodes for tests
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
def barriernet_agent(pe_env):
    """BarrierNet agent for evaluation."""
    from agents.barriernet_ppo import BarrierNetPPO, BarrierNetPPOConfig

    obs_dim = pe_env.observation_space.shape[0]
    config = BarrierNetPPOConfig(
        obs_dim=obs_dim,
        hidden_dim=64,
        n_layers=2,
        n_constraints_max=8,
    )
    return BarrierNetPPO(config)


# ==============================================================================
# Test Class 1: EvaluationResult
# ==============================================================================


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_create_result(self, dummy_result):
        """Should create a valid result."""
        assert dummy_result.approach_name == "test"
        assert dummy_result.n_episodes == 50
        assert 0 <= dummy_result.capture_rate <= 1

    def test_result_arrays(self, dummy_result):
        """Arrays should have correct shapes."""
        assert len(dummy_result.episode_rewards) == 50
        assert len(dummy_result.cbf_margin_values) == 50
        assert len(dummy_result.episode_lengths) == 50
        assert len(dummy_result.min_distances) == 50

    def test_result_metrics_ranges(self, dummy_result):
        """Metrics should be in valid ranges."""
        assert 0 <= dummy_result.safety_violation_rate <= 1
        assert 0 <= dummy_result.capture_rate <= 1
        assert 0 <= dummy_result.qp_infeasibility_rate <= 1
        assert 0 <= dummy_result.intervention_rate <= 1
        assert dummy_result.mean_inference_time_ms >= 0


# ==============================================================================
# Test Class 2: CBF Margin Computation
# ==============================================================================


class TestCBFMargins:
    """Test CBF margin computation."""

    def test_margins_center_safe(self):
        """Robot at center should have positive margins."""
        state = np.array([0.0, 0.0, 0.0])
        obstacles = []
        margins = compute_cbf_margins(state, obstacles, opponent_state=None)
        assert len(margins) == 4  # 4 arena walls
        assert all(m > 0 for m in margins), f"Not all positive: {margins}"

    def test_margins_near_wall_low(self):
        """Robot near wall should have low margin."""
        state = np.array([9.5, 0.0, 0.0])  # near right wall (half_w = 10)
        margins = compute_cbf_margins(state, [], opponent_state=None)
        # Right wall margin should be small
        assert min(margins) < 2.0

    def test_margins_with_obstacle(self):
        """Obstacle should add constraint."""
        state = np.array([0.0, 0.0, 0.0])
        obstacles = [{"x": 2.0, "y": 0.0, "radius": 0.5}]
        margins = compute_cbf_margins(state, obstacles, opponent_state=None)
        assert len(margins) == 5  # 4 walls + 1 obstacle

    def test_margins_with_opponent(self):
        """Opponent should add constraint."""
        state = np.array([0.0, 0.0, 0.0])
        opponent = np.array([3.0, 0.0, 0.0])
        margins = compute_cbf_margins(state, [], opponent_state=opponent)
        assert len(margins) == 5  # 4 walls + 1 collision

    def test_margins_full(self):
        """Full constraints (walls + obstacles + opponent)."""
        state = np.array([0.0, 0.0, 0.0])
        obstacles = [
            {"x": 3.0, "y": 0.0, "radius": 0.5},
            {"x": -3.0, "y": 2.0, "radius": 0.7},
        ]
        opponent = np.array([4.0, 1.0, 0.5])
        margins = compute_cbf_margins(state, obstacles, opponent_state=opponent)
        assert len(margins) == 7  # 4 walls + 2 obstacles + 1 collision

    def test_margins_far_from_boundary_all_positive(self):
        """Well-separated state should have all positive margins."""
        state = np.array([0.0, 0.0, 0.0])
        obstacles = [{"x": 5.0, "y": 5.0, "radius": 0.5}]
        opponent = np.array([5.0, -5.0, 0.0])
        margins = compute_cbf_margins(state, obstacles, opponent_state=opponent)
        assert all(m > 0 for m in margins)


# ==============================================================================
# Test Class 3: BarrierNet Evaluation Agent
# ==============================================================================


class TestBarrierNetEvalAgent:
    """Test BarrierNet evaluation agent wrapper."""

    def test_get_eval_action_returns_valid(self, barriernet_agent):
        """Should return valid action and info."""
        eval_agent = BarrierNetEvalAgent(barriernet_agent)
        obs = np.random.randn(barriernet_agent.config.obs_dim).astype(np.float32)
        state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obstacles = [{"x": 3.0, "y": 3.0, "radius": 0.5}]

        action, info = eval_agent.get_eval_action(
            obs, state, obstacles, None, 10.0, 10.0,
        )
        assert action.shape == (2,)
        assert "qp_correction" in info
        assert "qp_feasible" in info

    def test_action_within_bounds(self, barriernet_agent):
        """Actions should be within control limits."""
        eval_agent = BarrierNetEvalAgent(barriernet_agent)
        obs = np.random.randn(barriernet_agent.config.obs_dim).astype(np.float32)
        state = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        action, _ = eval_agent.get_eval_action(
            obs, state, [], None, 10.0, 10.0,
        )
        assert -0.01 <= action[0] <= 1.01  # v in [0, v_max]
        assert -2.85 <= action[1] <= 2.85  # omega in [-omega_max, omega_max]

    def test_deterministic(self, barriernet_agent):
        """Should produce deterministic actions."""
        eval_agent = BarrierNetEvalAgent(barriernet_agent)
        obs = np.random.randn(barriernet_agent.config.obs_dim).astype(np.float32)
        state = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        a1, _ = eval_agent.get_eval_action(obs, state, [], None, 10.0, 10.0)
        a2, _ = eval_agent.get_eval_action(obs, state, [], None, 10.0, 10.0)
        np.testing.assert_allclose(a1, a2, atol=1e-5)


# ==============================================================================
# Test Class 4: Evaluate Approach
# ==============================================================================


class TestEvaluateApproach:
    """Test the evaluate_approach function."""

    def test_runs_barriernet(self, pe_env, barriernet_agent):
        """Should complete evaluation."""
        eval_agent = BarrierNetEvalAgent(barriernet_agent)
        result = evaluate_approach(
            eval_agent, pe_env,
            n_episodes=3,
            approach_name="test_bn",
            verbose=False,
        )
        assert result.n_episodes == 3
        assert len(result.episode_rewards) == 3
        assert len(result.cbf_margin_values) == 3
        assert result.mean_inference_time_ms > 0

    def test_safety_metrics_computed(self, pe_env, barriernet_agent):
        """Safety metrics should be computed."""
        eval_agent = BarrierNetEvalAgent(barriernet_agent)
        result = evaluate_approach(
            eval_agent, pe_env,
            n_episodes=3,
            verbose=False,
        )
        assert 0 <= result.safety_violation_rate <= 1
        assert result.mean_min_cbf_margin != 0  # Should be nonzero

    def test_qp_metrics_tracked(self, pe_env, barriernet_agent):
        """QP metrics should be tracked."""
        eval_agent = BarrierNetEvalAgent(barriernet_agent)
        result = evaluate_approach(
            eval_agent, pe_env,
            n_episodes=3,
            verbose=False,
        )
        assert result.mean_qp_correction >= 0
        assert result.qp_infeasibility_rate >= 0


# ==============================================================================
# Test Class 5: Comparison Report
# ==============================================================================


class TestComparisonReport:
    """Test comparison report generation."""

    def _make_result(self, name, capture_rate, reward, margins):
        n = len(margins)
        return EvaluationResult(
            approach_name=name,
            n_episodes=n,
            safety_violation_rate=0.0,
            mean_min_cbf_margin=float(np.mean(margins)),
            cbf_margin_values=np.array(margins),
            capture_rate=capture_rate,
            mean_capture_time=15.0,
            mean_episode_reward=reward,
            episode_rewards=np.full(n, reward),
            mean_inference_time_ms=5.0,
            training_wall_clock_hours=10.0,
            qp_infeasibility_rate=0.0,
            mean_qp_correction=0.1,
            intervention_rate=0.2,
            episode_lengths=np.full(n, 100),
            capture_times=np.array([]),
            min_distances=np.full(n, 1.0),
        )

    def test_compute_comparison(self):
        """Should compute valid comparison."""
        bn = self._make_result("bn", 0.7, -15.0, np.random.uniform(0.1, 1.0, 50))
        ct = self._make_result("ct", 0.8, -10.0, np.random.uniform(0.1, 1.0, 50))
        cd = self._make_result("cd", 0.65, -18.0, np.random.uniform(0.05, 0.8, 50))

        report = compute_comparison(bn, ct, cd)
        assert isinstance(report, ComparisonReport)
        assert report.train_deploy_gap_capture > 0  # 0.8 vs 0.65
        assert report.train_deploy_gap_reward > 0

    def test_train_deploy_gap_calculation(self):
        """Gap should be correctly computed."""
        bn = self._make_result("bn", 0.7, -15.0, np.random.uniform(0.1, 1.0, 50))
        ct = self._make_result("ct", 0.8, -10.0, np.random.uniform(0.1, 1.0, 50))
        cd = self._make_result("cd", 0.6, -14.0, np.random.uniform(0.1, 1.0, 50))

        report = compute_comparison(bn, ct, cd)
        expected_gap = abs(0.8 - 0.6) / 0.8  # 0.25
        assert abs(report.train_deploy_gap_capture - expected_gap) < 0.01

    def test_p_values_valid(self):
        """P-values should be between 0 and 1."""
        bn = self._make_result("bn", 0.7, -15.0, np.random.uniform(0.1, 1.0, 50))
        ct = self._make_result("ct", 0.8, -10.0, np.random.uniform(0.1, 1.0, 50))
        cd = self._make_result("cd", 0.65, -18.0, np.random.uniform(0.05, 0.8, 50))

        report = compute_comparison(bn, ct, cd)
        assert 0 <= report.capture_rate_p_value <= 1
        assert 0 <= report.reward_p_value <= 1
        assert 0 <= report.safety_p_value <= 1


# ==============================================================================
# Test Class 6: Table Formatting
# ==============================================================================


class TestTableFormatting:
    """Test comparison table formatting."""

    def _make_result(self, name, capture_rate=0.5):
        n = 50
        return EvaluationResult(
            approach_name=name,
            n_episodes=n,
            safety_violation_rate=0.01,
            mean_min_cbf_margin=0.5,
            cbf_margin_values=np.random.uniform(0.1, 1.0, n),
            capture_rate=capture_rate,
            mean_capture_time=15.0,
            mean_episode_reward=-20.0,
            episode_rewards=np.random.normal(-20, 5, n),
            mean_inference_time_ms=5.0,
            training_wall_clock_hours=10.0,
            qp_infeasibility_rate=0.0,
            mean_qp_correction=0.1,
            intervention_rate=0.3,
            episode_lengths=np.full(n, 100),
            capture_times=np.array([]),
            min_distances=np.full(n, 1.0),
        )

    def test_table_is_string(self):
        """Table should be a non-empty string."""
        bn = self._make_result("bn")
        ct = self._make_result("ct")
        cd = self._make_result("cd")
        report = compute_comparison(bn, ct, cd)
        table = format_comparison_table(report)
        assert isinstance(table, str)
        assert len(table) > 100

    def test_table_has_metrics(self):
        """Table should contain key metrics."""
        bn = self._make_result("bn")
        ct = self._make_result("ct")
        cd = self._make_result("cd")
        report = compute_comparison(bn, ct, cd)
        table = format_comparison_table(report)

        assert "Safety violations" in table
        assert "Capture rate" in table
        assert "inference time" in table.lower() or "Inference" in table
        assert "Train-Deploy Gap" in table

    def test_table_has_p_values(self):
        """Table should include statistical significance."""
        bn = self._make_result("bn", 0.7)
        ct = self._make_result("ct", 0.8)
        cd = self._make_result("cd", 0.6)
        report = compute_comparison(bn, ct, cd)
        table = format_comparison_table(report)

        assert "p =" in table
        assert "significant" in table.lower()
