"""Tests for Phase 2 Session 7: Safety-Constrained Self-Play Training.

Tests cover:
1. SafetyMetricsTracker
2. SafeSelfPlayConfig
3. Environment creation helpers
4. Safe rollout (SB3-independent)
5. Comparison run configurations
"""

import numpy as np
import pytest

from training.safety_metrics import SafetyMetricsTracker
from training.safe_self_play import (
    SafeSelfPlayConfig,
    get_run_configs,
    make_cbf_filter,
    make_resolver,
    make_safe_env,
    safe_self_play_rollout,
)


# =============================================================================
# SafetyMetricsTracker
# =============================================================================

class TestSafetyMetricsTracker:
    """Tests for safety metrics tracking."""

    def test_initial_state(self):
        tracker = SafetyMetricsTracker()
        assert tracker.total_steps == 0
        assert tracker.total_violations == 0

    def test_record_step(self):
        tracker = SafetyMetricsTracker()
        tracker.record_step(method="exact", min_h=5.0)
        assert tracker.total_steps == 1

    def test_record_violation(self):
        tracker = SafetyMetricsTracker()
        tracker.record_step(method="exact", min_h=0.0, violation=True)
        assert tracker.total_violations == 1

    def test_record_backup(self):
        tracker = SafetyMetricsTracker()
        tracker.record_step(method="backup", min_h=0.0)
        assert tracker.total_backups == 1

    def test_record_intervention(self):
        tracker = SafetyMetricsTracker()
        tracker.record_step(method="exact", min_h=0.0, intervention=True)
        assert tracker.total_interventions == 1

    def test_end_phase(self):
        tracker = SafetyMetricsTracker()
        for _ in range(100):
            tracker.record_step(method="exact", min_h=5.0)
        metrics = tracker.end_phase()
        assert metrics["phase"] == 0
        assert metrics["steps"] == 100
        assert metrics["violations"] == 0
        assert metrics["exact_rate"] == 1.0

    def test_multiple_phases(self):
        tracker = SafetyMetricsTracker()
        # Phase 0
        for _ in range(50):
            tracker.record_step(method="exact", min_h=5.0)
        tracker.end_phase()
        # Phase 1
        for _ in range(50):
            tracker.record_step(method="backup", min_h=0.0)
        metrics = tracker.end_phase()
        assert metrics["phase"] == 1
        assert len(tracker.phase_metrics) == 2

    def test_get_recent_metrics(self):
        tracker = SafetyMetricsTracker(window_size=50)
        for _ in range(100):
            tracker.record_step(method="exact", min_h=3.0)
        recent = tracker.get_recent_metrics()
        assert recent["window_size"] == 50  # capped at window
        assert recent["exact_rate"] == 1.0

    def test_get_summary(self):
        tracker = SafetyMetricsTracker()
        for _ in range(50):
            tracker.record_step(method="exact", min_h=5.0)
        for _ in range(5):
            tracker.record_step(method="backup", min_h=0.0)
        summary = tracker.get_summary()
        assert summary["total_steps"] == 55
        assert summary["total_backups"] == 5
        assert summary["backup_rate"] == pytest.approx(5 / 55, abs=0.01)

    def test_check_safety_targets_met(self):
        tracker = SafetyMetricsTracker()
        for _ in range(1000):
            tracker.record_step(method="exact", min_h=5.0)
        targets = tracker.check_safety_targets()
        assert targets["zero_violations"] is True
        assert targets["feasibility_above_99"] is True
        assert targets["backup_below_1pct"] is True
        assert targets["all_targets_met"] is True

    def test_check_safety_targets_failed(self):
        tracker = SafetyMetricsTracker()
        for _ in range(100):
            tracker.record_step(method="exact", min_h=5.0, violation=True)
        targets = tracker.check_safety_targets()
        assert targets["zero_violations"] is False
        assert targets["all_targets_met"] is False

    def test_cbf_margin_statistics(self):
        tracker = SafetyMetricsTracker()
        margins = [1.0, 2.0, 3.0, 0.5, 4.0]
        for m in margins:
            tracker.record_step(method="exact", min_h=m)
        metrics = tracker.end_phase()
        assert metrics["min_cbf_margin"] == pytest.approx(0.5, abs=1e-6)
        assert metrics["mean_cbf_margin"] == pytest.approx(np.mean(margins), abs=1e-6)

    def test_method_counts(self):
        tracker = SafetyMetricsTracker()
        tracker.record_step(method="exact", min_h=5.0)
        tracker.record_step(method="exact", min_h=5.0)
        tracker.record_step(method="relaxed_obstacles", min_h=1.0)
        tracker.record_step(method="backup", min_h=0.0)
        metrics = tracker.end_phase()
        assert metrics["method_counts"]["exact"] == 2
        assert metrics["method_counts"]["relaxed_obstacles"] == 1
        assert metrics["method_counts"]["backup"] == 1


# =============================================================================
# SafeSelfPlayConfig
# =============================================================================

class TestSafeSelfPlayConfig:
    """Tests for the config dataclass."""

    def test_default_config(self):
        config = SafeSelfPlayConfig()
        assert config.n_phases == 10
        assert config.n_obstacles == 4
        assert config.cbf_alpha == 1.0
        assert config.w_safety == 0.05
        assert config.use_cbf is True

    def test_custom_config(self):
        config = SafeSelfPlayConfig(
            n_phases=5,
            n_obstacles=8,
            seed=123,
        )
        assert config.n_phases == 5
        assert config.n_obstacles == 8
        assert config.seed == 123

    def test_run_configs(self):
        configs = get_run_configs()
        assert len(configs) == 4
        assert "A_full_safe" in configs
        assert "B_unsafe" in configs
        assert "C_cbf_qp_only" in configs
        assert "D_no_safety_reward" in configs

    def test_run_a_config(self):
        configs = get_run_configs()
        a = configs["A_full_safe"]
        assert a.use_cbf is True
        assert a.use_beta_policy is True
        assert a.use_safety_reward is True

    def test_run_b_config(self):
        configs = get_run_configs()
        b = configs["B_unsafe"]
        assert b.use_cbf is False
        assert b.n_obstacles == 0

    def test_run_c_config(self):
        configs = get_run_configs()
        c = configs["C_cbf_qp_only"]
        assert c.use_cbf is True
        assert c.use_beta_policy is False
        assert c.use_safety_reward is False

    def test_run_d_config(self):
        configs = get_run_configs()
        d = configs["D_no_safety_reward"]
        assert d.use_cbf is True
        assert d.use_safety_reward is False


# =============================================================================
# Environment & Component Creation
# =============================================================================

class TestMakeComponents:
    """Tests for helper functions that create components."""

    def test_make_safe_env(self):
        config = SafeSelfPlayConfig()
        env = make_safe_env(config)
        assert env.n_obstacles == 4
        obs, _ = env.reset(seed=42)
        assert obs["pursuer"].shape == (20,)  # 14 + 2*3 obstacle obs
        env.close()

    def test_make_safe_env_no_safety_reward(self):
        config = SafeSelfPlayConfig(use_safety_reward=False)
        env = make_safe_env(config)
        assert env.reward_computer is not None  # default RewardComputer
        env.close()

    def test_make_cbf_filter(self):
        config = SafeSelfPlayConfig()
        cbf = make_cbf_filter(config)
        assert cbf.v_max == config.v_max
        assert cbf.alpha == config.cbf_alpha

    def test_make_resolver(self):
        config = SafeSelfPlayConfig()
        cbf = make_cbf_filter(config)
        resolver = make_resolver(config, cbf)
        assert resolver.cbf_filter is cbf


# =============================================================================
# Safe Rollout (SB3-independent)
# =============================================================================

class TestSafeRollout:
    """Tests for safe_self_play_rollout."""

    def test_basic_rollout(self):
        config = SafeSelfPlayConfig(
            n_obstacles=2, n_obstacle_obs=2,
            max_steps=50, seed=42,
        )
        results = safe_self_play_rollout(config, n_episodes=3)
        assert "episodes" in results
        assert len(results["episodes"]) == 3
        assert "safety_metrics" in results
        assert "safety_targets" in results

    def test_rollout_with_safety(self):
        config = SafeSelfPlayConfig(
            n_obstacles=3, n_obstacle_obs=3,
            max_steps=100, seed=42, use_cbf=True,
        )
        results = safe_self_play_rollout(config, n_episodes=5)
        # With CBF, should have zero violations (ideally)
        metrics = results["safety_metrics"]
        assert metrics["violations"] == 0

    def test_rollout_without_cbf(self):
        config = SafeSelfPlayConfig(
            n_obstacles=0, n_obstacle_obs=0,
            max_steps=50, seed=42, use_cbf=False,
        )
        results = safe_self_play_rollout(config, n_episodes=3)
        assert len(results["episodes"]) == 3

    def test_rollout_episode_results(self):
        config = SafeSelfPlayConfig(
            n_obstacles=2, n_obstacle_obs=2,
            max_steps=50, seed=42,
        )
        results = safe_self_play_rollout(config, n_episodes=2)
        for ep in results["episodes"]:
            assert "reward" in ep
            assert "steps" in ep
            assert "captured" in ep
            assert ep["steps"] > 0

    def test_resolver_metrics_populated(self):
        config = SafeSelfPlayConfig(
            n_obstacles=2, max_steps=50, seed=42,
        )
        results = safe_self_play_rollout(config, n_episodes=3)
        rm = results["resolver_metrics"]
        assert rm["n_total"] > 0

    def test_custom_policy(self):
        """Custom policy function should be used."""
        config = SafeSelfPlayConfig(
            n_obstacles=0, n_obstacle_obs=0,
            max_steps=20, seed=42,
        )

        def zero_policy(obs):
            return np.array([0.0, 0.0], dtype=np.float32)

        results = safe_self_play_rollout(
            config, n_episodes=2,
            pursuer_policy=zero_policy,
        )
        assert len(results["episodes"]) == 2

    def test_all_run_types(self):
        """All 4 comparison run configs should produce valid rollouts."""
        configs = get_run_configs()
        for label, config in configs.items():
            config.max_steps = 30  # Keep fast
            results = safe_self_play_rollout(config, n_episodes=2)
            assert len(results["episodes"]) == 2, (
                f"Run {label} failed to produce episodes"
            )


# =============================================================================
# End-to-end safety validation
# =============================================================================

class TestSafetyValidation:
    """End-to-end safety validation tests."""

    def test_safe_rollout_no_violations(self):
        """Full safe rollout should produce zero violations."""
        config = SafeSelfPlayConfig(
            n_obstacles=3, n_obstacle_obs=3,
            max_steps=200, seed=42, use_cbf=True,
        )
        results = safe_self_play_rollout(config, n_episodes=5)
        targets = results["safety_targets"]
        assert targets["zero_violations"], (
            f"Violations occurred: {results['summary']['total_violations']}"
        )

    def test_exact_rate_reasonable(self):
        """Most steps should resolve as 'exact' for normal scenarios."""
        config = SafeSelfPlayConfig(
            n_obstacles=2, n_obstacle_obs=2,
            max_steps=100, seed=42, use_cbf=True,
        )
        results = safe_self_play_rollout(config, n_episodes=5)
        metrics = results["safety_metrics"]
        assert metrics["exact_rate"] > 0.8, (
            f"Exact rate {metrics['exact_rate']:.2f} too low"
        )

    def test_speed_benchmark(self):
        """Safe rollout should be fast enough for training."""
        import time
        config = SafeSelfPlayConfig(
            n_obstacles=3, n_obstacle_obs=3,
            max_steps=200, seed=42, use_cbf=True,
        )
        start = time.perf_counter()
        results = safe_self_play_rollout(config, n_episodes=3)
        elapsed = time.perf_counter() - start

        total_steps = sum(ep["steps"] for ep in results["episodes"])
        steps_per_sec = total_steps / elapsed
        assert steps_per_sec > 100, (
            f"Too slow: {steps_per_sec:.0f} steps/s with CBF (need >100)"
        )
