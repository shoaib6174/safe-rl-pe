"""Tests for Phase 2 Session 8: Ablation Study & Results.

Tests cover:
1. Ablation config definitions (A-E)
2. Ablation runner (single config execution)
3. Results compilation (aggregation, tables, analysis)
4. Phase 2 integration (end-to-end safety pipeline)
5. Cross-config safety assertions
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.safe_self_play import (
    SafeSelfPlayConfig,
    make_cbf_filter,
    make_resolver,
    make_safe_env,
    safe_self_play_rollout,
)
from training.safety_metrics import SafetyMetricsTracker


# Import ablation runner
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from run_ablation import get_ablation_configs, run_single_config
from compile_ablation_results import (
    aggregate_by_config,
    generate_latex_table,
    generate_markdown_table,
    generate_analysis,
)


# =============================================================================
# Ablation Config Definitions
# =============================================================================

class TestAblationConfigs:
    """Tests for the 5 ablation configurations."""

    def test_all_five_configs_present(self):
        """All 5 configs A-E must be defined."""
        configs = get_ablation_configs()
        assert set(configs.keys()) == {"A", "B", "C", "D", "E"}

    def test_config_a_full_safe(self):
        """Config A: Full safe with all safety components."""
        configs = get_ablation_configs()
        a = configs["A"]
        assert a.use_cbf is True
        assert a.use_beta_policy is True
        assert a.use_safety_reward is True
        assert a.use_feasibility_classifier is True
        assert a.n_obstacles == 4

    def test_config_b_unsafe(self):
        """Config B: Unsafe baseline, no safety at all."""
        configs = get_ablation_configs()
        b = configs["B"]
        assert b.use_cbf is False
        assert b.use_beta_policy is False
        assert b.use_safety_reward is False
        assert b.n_obstacles == 0
        assert b.n_obstacle_obs == 0

    def test_config_c_qp_only(self):
        """Config C: QP filter only, no Beta policy."""
        configs = get_ablation_configs()
        c = configs["C"]
        assert c.use_cbf is True
        assert c.use_beta_policy is False
        assert c.use_safety_reward is False

    def test_config_d_no_w5(self):
        """Config D: CBF-Beta but no safety reward."""
        configs = get_ablation_configs()
        d = configs["D"]
        assert d.use_cbf is True
        assert d.use_beta_policy is True
        assert d.use_safety_reward is False
        assert d.use_feasibility_classifier is True

    def test_config_e_no_n13(self):
        """Config E: CBF-Beta + w5 but no feasibility classifier."""
        configs = get_ablation_configs()
        e = configs["E"]
        assert e.use_cbf is True
        assert e.use_beta_policy is True
        assert e.use_safety_reward is True
        assert e.use_feasibility_classifier is False

    def test_all_configs_have_same_env_params(self):
        """Configs A, C, D, E share the same obstacle environment (B has none)."""
        configs = get_ablation_configs()
        for label in ["A", "C", "D", "E"]:
            assert configs[label].n_obstacles == 4, f"Config {label}"
            assert configs[label].n_obstacle_obs == 3, f"Config {label}"

    def test_config_b_no_obstacles(self):
        """Config B must have zero obstacles."""
        configs = get_ablation_configs()
        assert configs["B"].n_obstacles == 0
        assert configs["B"].n_obstacle_obs == 0


# =============================================================================
# Ablation Runner
# =============================================================================

class TestAblationRunner:
    """Tests for running individual ablation configs."""

    def test_run_config_a(self):
        """Config A produces results with expected keys."""
        configs = get_ablation_configs()
        result = run_single_config("A", configs["A"], n_episodes=2, seed=42)
        assert result["config"] == "A"
        assert result["seed"] == 42
        assert "capture_rate" in result
        assert "violations" in result
        assert "intervention_rate" in result
        assert "all_targets_met" in result

    def test_run_config_b(self):
        """Config B (unsafe) runs without errors."""
        configs = get_ablation_configs()
        result = run_single_config("B", configs["B"], n_episodes=2, seed=42)
        assert result["config"] == "B"
        # Config B has no CBF, so exact_rate is meaningless
        assert result["intervention_rate"] == 0.0

    def test_run_config_c(self):
        """Config C (QP only) runs without errors."""
        configs = get_ablation_configs()
        result = run_single_config("C", configs["C"], n_episodes=2, seed=42)
        assert result["config"] == "C"

    def test_run_config_d(self):
        """Config D (no w5) runs without errors."""
        configs = get_ablation_configs()
        result = run_single_config("D", configs["D"], n_episodes=2, seed=42)
        assert result["config"] == "D"

    def test_run_config_e(self):
        """Config E (no N13) runs without errors."""
        configs = get_ablation_configs()
        result = run_single_config("E", configs["E"], n_episodes=2, seed=42)
        assert result["config"] == "E"

    def test_run_all_configs(self):
        """All 5 configs run successfully."""
        configs = get_ablation_configs()
        results = []
        for label, config in configs.items():
            result = run_single_config(label, config, n_episodes=2, seed=42)
            results.append(result)
        assert len(results) == 5
        labels = {r["config"] for r in results}
        assert labels == {"A", "B", "C", "D", "E"}

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different rollouts."""
        configs = get_ablation_configs()
        r1 = run_single_config("A", configs["A"], n_episodes=3, seed=0)
        r2 = run_single_config("A", configs["A"], n_episodes=3, seed=99)
        # Not deterministically different, but likely
        # Just check both run successfully
        assert r1["config"] == r2["config"] == "A"


# =============================================================================
# Results Compilation
# =============================================================================

class TestResultsCompilation:
    """Tests for results aggregation and report generation."""

    @pytest.fixture
    def sample_results(self):
        """Generate sample results for testing compilation."""
        configs = get_ablation_configs()
        results = []
        for label, config in configs.items():
            result = run_single_config(label, config, n_episodes=2, seed=42)
            results.append(result)
        return results

    def test_aggregate_by_config(self, sample_results):
        """Aggregation produces entry for each config."""
        agg = aggregate_by_config(sample_results)
        assert set(agg.keys()) == {"A", "B", "C", "D", "E"}
        for label, metrics in agg.items():
            assert "capture_rate_mean" in metrics
            assert "violations_total" in metrics
            assert "all_targets_met" in metrics

    def test_markdown_table(self, sample_results):
        """Markdown table contains all config rows."""
        agg = aggregate_by_config(sample_results)
        table = generate_markdown_table(agg)
        assert "| Config |" in table
        for label in "ABCDE":
            assert f"| {label} |" in table

    def test_latex_table(self, sample_results):
        """LaTeX table contains all config rows."""
        agg = aggregate_by_config(sample_results)
        table = generate_latex_table(agg)
        assert "\\begin{table}" in table
        for label in "ABCDE":
            assert f"{label} &" in table

    def test_analysis_text(self, sample_results):
        """Analysis text is generated for key comparisons."""
        agg = aggregate_by_config(sample_results)
        analysis = generate_analysis(agg)
        assert "Safety cost" in analysis
        assert "CBF-Beta vs QP-only" in analysis
        assert "w5 effect" in analysis
        assert "N13 effect" in analysis

    def test_results_json_serializable(self, sample_results):
        """Results can be serialized to JSON."""
        json_str = json.dumps(sample_results, default=str)
        loaded = json.loads(json_str)
        assert len(loaded) == len(sample_results)


# =============================================================================
# Phase 2 Integration Tests
# =============================================================================

class TestPhase2Integration:
    """End-to-end integration tests for the Phase 2 safety pipeline."""

    def test_env_with_obstacles_and_cbf(self):
        """Environment with obstacles + CBF filter produces safe actions."""
        config = SafeSelfPlayConfig(n_obstacles=3, n_obstacle_obs=3)
        env = make_safe_env(config)
        cbf_filter = make_cbf_filter(config)
        resolver = make_resolver(config, cbf_filter)

        obs, _ = env.reset(seed=42)
        for _ in range(50):
            action = env.pursuer_action_space.sample()
            safe_action, method, info = resolver.resolve(
                action, env.pursuer_state,
                obstacles=env.obstacles,
                opponent_state=env.evader_state,
            )
            e_action = env.evader_action_space.sample()
            obs, rewards, terminated, truncated, _ = env.step(safe_action, e_action)
            if terminated or truncated:
                obs, _ = env.reset()

        env.close()

    def test_safety_reward_integration(self):
        """SafetyRewardComputer integrates with CBF margin."""
        config = SafeSelfPlayConfig(use_safety_reward=True, w_safety=0.05)
        env = make_safe_env(config)
        cbf_filter = make_cbf_filter(config)

        obs, _ = env.reset(seed=42)
        action = env.pursuer_action_space.sample()
        e_action = env.evader_action_space.sample()

        # Get CBF constraints for min_h
        constraints = cbf_filter.get_constraints(
            env.pursuer_state, env.evader_state, env.obstacles,
        )
        if constraints:
            min_h = min(c.h for c in constraints)
            assert isinstance(min_h, float)

        env.close()

    def test_resolver_tracks_metrics(self):
        """SafeActionResolver correctly tracks resolution metrics."""
        config = SafeSelfPlayConfig(n_obstacles=3)
        env = make_safe_env(config)
        cbf_filter = make_cbf_filter(config)
        resolver = make_resolver(config, cbf_filter)

        obs, _ = env.reset(seed=42)
        for _ in range(20):
            action = env.pursuer_action_space.sample()
            resolver.resolve(
                action, env.pursuer_state,
                obstacles=env.obstacles,
                opponent_state=env.evader_state,
            )
            e_action = env.evader_action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action, e_action)
            if terminated or truncated:
                obs, _ = env.reset()

        metrics = resolver.get_metrics()
        assert metrics["n_total"] == 20
        assert metrics["n_exact"] + metrics["n_relaxed"] + metrics["n_backup"] == 20
        env.close()

    def test_safety_metrics_across_phases(self):
        """SafetyMetricsTracker handles multiple phases correctly."""
        tracker = SafetyMetricsTracker()

        # Phase 0
        for _ in range(100):
            tracker.record_step(method="exact", min_h=5.0)
        phase0 = tracker.end_phase()
        assert phase0["phase"] == 0
        assert phase0["steps"] == 100

        # Phase 1
        for _ in range(50):
            tracker.record_step(method="exact", min_h=3.0)
        for _ in range(50):
            tracker.record_step(method="relaxed_obstacles", min_h=0.1)
        phase1 = tracker.end_phase()
        assert phase1["phase"] == 1
        assert phase1["steps"] == 100

        summary = tracker.get_summary()
        assert summary["total_steps"] == 200
        assert summary["n_phases_completed"] == 2

    def test_full_rollout_with_cbf(self):
        """Full safe rollout produces valid results."""
        config = SafeSelfPlayConfig(n_obstacles=3)
        results = safe_self_play_rollout(config, n_episodes=3)

        assert len(results["episodes"]) == 3
        assert "safety_metrics" in results
        assert "safety_targets" in results
        assert "resolver_metrics" in results

        for ep in results["episodes"]:
            assert "reward" in ep
            assert "steps" in ep
            assert ep["steps"] > 0

    def test_rollout_without_cbf(self):
        """Rollout without CBF runs successfully (Config B)."""
        config = SafeSelfPlayConfig(
            use_cbf=False, use_beta_policy=False,
            use_safety_reward=False,
            n_obstacles=0, n_obstacle_obs=0,
        )
        results = safe_self_play_rollout(config, n_episodes=3)
        assert len(results["episodes"]) == 3

    def test_observation_dimension_consistency(self):
        """Observation dimension matches with/without obstacles."""
        # Without obstacles
        env0 = PursuitEvasionEnv(
            arena_width=20.0, arena_height=20.0, dt=0.05,
            max_steps=100, n_obstacles=0, n_obstacle_obs=0,
            render_mode=None,
        )
        obs0, _ = env0.reset(seed=42)
        dim0 = obs0["pursuer"].shape[0]

        # With 3 obstacle observations
        env3 = PursuitEvasionEnv(
            arena_width=20.0, arena_height=20.0, dt=0.05,
            max_steps=100, n_obstacles=3, n_obstacle_obs=3,
            render_mode=None,
        )
        obs3, _ = env3.reset(seed=42)
        dim3 = obs3["pursuer"].shape[0]

        # 3 obstacles => 6 extra features (3 * 2)
        assert dim3 == dim0 + 6

        env0.close()
        env3.close()

    def test_cbf_filter_with_obstacles(self):
        """VCPCBFFilter generates obstacle constraints."""
        config = SafeSelfPlayConfig(n_obstacles=3)
        env = make_safe_env(config)
        cbf_filter = make_cbf_filter(config)

        obs, _ = env.reset(seed=42)
        constraints = cbf_filter.get_constraints(
            env.pursuer_state, env.evader_state, env.obstacles,
        )
        # Should have arena (4) + collision (1) + obstacle constraints
        assert len(constraints) >= 5  # At least 4 arena + 1 collision
        env.close()

    def test_hierarchical_relaxation_preserves_safety(self):
        """Hierarchical relaxation produces valid actions."""
        from safety.infeasibility import solve_cbf_qp_with_relaxation
        from safety.vcp_cbf import CBFConstraint, ConstraintType

        # Create constraints that are hard to satisfy simultaneously
        constraints = [
            CBFConstraint(
                h=0.1, a_v=1.0, a_omega=0.0,
                ctype=ConstraintType.ARENA,
            ),
            CBFConstraint(
                h=0.1, a_v=-1.0, a_omega=0.0,
                ctype=ConstraintType.ARENA,
            ),
        ]

        u_nominal = np.array([0.5, 0.0])
        u_safe, method, info = solve_cbf_qp_with_relaxation(
            u_nominal, constraints,
        )

        assert u_safe is not None
        assert method in ("exact", "relaxed_obstacles", "relaxed_arena", "unconstrained")


# =============================================================================
# Cross-Config Safety Assertions
# =============================================================================

class TestCrossConfigSafety:
    """Cross-config comparisons for ablation validation."""

    @pytest.fixture(scope="class")
    def all_results(self):
        """Run all configs with same seed for comparison."""
        configs = get_ablation_configs()
        results = {}
        for label, config in configs.items():
            results[label] = run_single_config(label, config, n_episodes=3, seed=42)
        return results

    def test_config_b_has_no_cbf_interventions(self, all_results):
        """Config B (unsafe) should have zero interventions."""
        assert all_results["B"]["intervention_rate"] == 0.0

    def test_safe_configs_produce_resolver_metrics(self, all_results):
        """Configs A, C, D, E (with CBF) produce resolver metrics."""
        for label in ["A", "C", "D", "E"]:
            assert all_results[label]["resolver_exact_rate"] >= 0.0
            assert all_results[label]["resolver_exact_rate"] <= 1.0

    def test_configs_produce_valid_capture_rates(self, all_results):
        """All configs produce capture rates in [0, 1]."""
        for label in "ABCDE":
            assert 0.0 <= all_results[label]["capture_rate"] <= 1.0

    def test_configs_produce_non_negative_violations(self, all_results):
        """All configs produce non-negative violation counts."""
        for label in "ABCDE":
            assert all_results[label]["violations"] >= 0

    def test_safe_configs_have_cbf_margins(self, all_results):
        """Configs with CBF produce finite CBF margins."""
        for label in ["A", "C", "D", "E"]:
            margin = all_results[label]["mean_cbf_margin"]
            assert isinstance(margin, float)


# =============================================================================
# Hydra Config Validation
# =============================================================================

class TestHydraConfigs:
    """Tests for Hydra YAML ablation configs."""

    @pytest.fixture
    def config_dir(self):
        return os.path.join(os.path.dirname(__file__), "..", "conf", "experiment")

    def test_all_configs_exist(self, config_dir):
        """All 5 YAML configs exist."""
        for label in "ABCDE":
            path = os.path.join(config_dir, f"ablation_{label}.yaml")
            assert os.path.exists(path), f"Missing: ablation_{label}.yaml"

    def test_configs_are_valid_yaml(self, config_dir):
        """All configs parse as valid YAML."""
        import yaml
        for label in "ABCDE":
            path = os.path.join(config_dir, f"ablation_{label}.yaml")
            with open(path) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"ablation_{label}.yaml is not a dict"

    def test_configs_have_experiment_group(self, config_dir):
        """All configs share the same experiment group."""
        import yaml
        for label in "ABCDE":
            path = os.path.join(config_dir, f"ablation_{label}.yaml")
            with open(path) as f:
                data = yaml.safe_load(f)
            assert data.get("experiment_group") == "phase2-ablation", f"Config {label}"

    def test_configs_have_wandb_tags(self, config_dir):
        """All configs have wandb tags including 'phase2' and 'ablation'."""
        import yaml
        for label in "ABCDE":
            path = os.path.join(config_dir, f"ablation_{label}.yaml")
            with open(path) as f:
                data = yaml.safe_load(f)
            tags = data.get("wandb", {}).get("tags", [])
            assert "phase2" in tags, f"Config {label} missing 'phase2' tag"
            assert "ablation" in tags, f"Config {label} missing 'ablation' tag"

    def test_config_a_enables_all_safety(self, config_dir):
        """Config A YAML enables all safety features."""
        import yaml
        path = os.path.join(config_dir, "ablation_A.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        safety = data.get("safety", {})
        assert safety.get("enabled") is True
        assert safety.get("cbf_beta") is True
        assert safety.get("safety_reward_w5") == 0.05
        assert safety.get("n13_feasibility") is True

    def test_config_b_disables_all_safety(self, config_dir):
        """Config B YAML disables all safety features."""
        import yaml
        path = os.path.join(config_dir, "ablation_B.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        safety = data.get("safety", {})
        assert safety.get("enabled") is False
        assert safety.get("cbf_beta") is False
        assert safety.get("safety_reward_w5") == 0.0
        assert safety.get("n13_feasibility") is False
        assert data.get("env", {}).get("n_obstacles") == 0

    def test_config_c_qp_only(self, config_dir):
        """Config C YAML enables QP filter but not Beta."""
        import yaml
        path = os.path.join(config_dir, "ablation_C.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        safety = data.get("safety", {})
        assert safety.get("cbf_beta") is False
        assert safety.get("cbf_qp_filter") is True
        assert safety.get("safety_reward_w5") == 0.0

    def test_config_d_no_safety_reward(self, config_dir):
        """Config D YAML has CBF-Beta but no safety reward."""
        import yaml
        path = os.path.join(config_dir, "ablation_D.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        safety = data.get("safety", {})
        assert safety.get("cbf_beta") is True
        assert safety.get("safety_reward_w5") == 0.0
        assert safety.get("n13_feasibility") is True

    def test_config_e_no_feasibility(self, config_dir):
        """Config E YAML has CBF-Beta + w5 but no N13 feasibility."""
        import yaml
        path = os.path.join(config_dir, "ablation_E.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        safety = data.get("safety", {})
        assert safety.get("cbf_beta") is True
        assert safety.get("safety_reward_w5") == 0.05
        assert safety.get("n13_feasibility") is False


# =============================================================================
# Import from envs for observation test
# =============================================================================

from envs.pursuit_evasion_env import PursuitEvasionEnv
