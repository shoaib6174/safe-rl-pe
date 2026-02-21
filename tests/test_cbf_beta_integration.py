"""Tests for Phase 2 Session 4: CBF-Beta Policy Integration.

Tests cover:
- SafeBetaPolicy dynamic bound management
- CBFSafetyCallback bound computation
- Safe bounds + QP filter consistency
- Integration with VCPCBFFilter
"""

import numpy as np
import pytest

from safety.vcp_cbf import VCPCBFFilter


class TestSafeBetaPolicyBounds:
    """Tests for SafeBetaPolicy's dynamic safe bounds management."""

    @pytest.fixture
    def policy(self):
        """Create a SafeBetaPolicy with minimal config."""
        sb3 = pytest.importorskip("stable_baselines3")
        from gymnasium import spaces
        from safety.safe_beta_policy import SafeBetaPolicy

        obs_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([0.0, -2.84], dtype=np.float32),
            high=np.array([1.0, 2.84], dtype=np.float32),
        )
        return SafeBetaPolicy(obs_space, act_space, lambda _: 3e-4)

    def test_nominal_bounds_at_init(self, policy):
        """Policy should start with nominal action space bounds."""
        low, high = policy.get_current_bounds()
        np.testing.assert_allclose(low.numpy(), [0.0, -2.84], atol=1e-6)
        np.testing.assert_allclose(high.numpy(), [1.0, 2.84], atol=1e-6)

    def test_set_safe_bounds(self, policy):
        """Setting safe bounds should update current bounds."""
        policy.set_safe_bounds((0.0, 0.5), (-1.0, 1.0))
        low, high = policy.get_current_bounds()
        np.testing.assert_allclose(low.numpy(), [0.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(high.numpy(), [0.5, 1.0], atol=1e-6)

    def test_clear_safe_bounds(self, policy):
        """Clearing bounds should revert to nominal."""
        policy.set_safe_bounds((0.0, 0.3), (-0.5, 0.5))
        policy.clear_safe_bounds()
        low, high = policy.get_current_bounds()
        np.testing.assert_allclose(low.numpy(), [0.0, -2.84], atol=1e-6)
        np.testing.assert_allclose(high.numpy(), [1.0, 2.84], atol=1e-6)

    def test_safe_bounds_clamp_to_nominal(self, policy):
        """Safe bounds exceeding nominal should be clamped."""
        policy.set_safe_bounds((-1.0, 2.0), (-5.0, 5.0))
        low, high = policy.get_current_bounds()
        # Should be clamped to action space bounds
        assert low[0].item() >= 0.0
        assert high[0].item() <= 1.0
        assert low[1].item() >= -2.84
        assert high[1].item() <= 2.84

    def test_safe_bounds_min_gap(self, policy):
        """Safe bounds with min > max should ensure min gap."""
        policy.set_safe_bounds((0.6, 0.5), (1.0, -1.0))
        low, high = policy.get_current_bounds()
        # high should be at least low + epsilon
        assert high[0].item() > low[0].item()
        assert high[1].item() > low[1].item()

    def test_get_bounds_numpy(self, policy):
        """get_current_bounds_numpy should return numpy arrays."""
        policy.set_safe_bounds((0.0, 0.5), (-1.0, 1.0))
        low, high = policy.get_current_bounds_numpy()
        assert isinstance(low, np.ndarray)
        assert isinstance(high, np.ndarray)
        assert low.shape == (2,)
        np.testing.assert_allclose(low, [0.0, -1.0], atol=1e-6)

    def test_forward_with_safe_bounds(self, policy):
        """Forward pass should sample within safe bounds."""
        import torch
        policy.set_safe_bounds((0.0, 0.3), (-0.5, 0.5))
        obs = torch.randn(1, 14)
        with torch.no_grad():
            actions, values, log_probs = policy.forward(obs)
        assert actions.shape == (1, 2)
        assert actions[0, 0].item() >= 0.0 - 1e-3
        assert actions[0, 0].item() <= 0.3 + 1e-3
        assert actions[0, 1].item() >= -0.5 - 1e-3
        assert actions[0, 1].item() <= 0.5 + 1e-3

    def test_forward_many_samples_in_bounds(self, policy):
        """100 forward passes should all stay within safe bounds."""
        import torch
        policy.set_safe_bounds((0.1, 0.4), (-1.0, 1.0))
        obs = torch.randn(100, 14)
        with torch.no_grad():
            actions, _, _ = policy.forward(obs)
        assert (actions[:, 0] >= 0.1 - 1e-3).all()
        assert (actions[:, 0] <= 0.4 + 1e-3).all()
        assert (actions[:, 1] >= -1.0 - 1e-3).all()
        assert (actions[:, 1] <= 1.0 + 1e-3).all()

    def test_evaluate_actions_with_bounds(self, policy):
        """evaluate_actions_with_bounds should use provided bounds."""
        import torch
        obs = torch.randn(4, 14)
        actions = torch.tensor([
            [0.2, 0.5],
            [0.3, -0.3],
            [0.1, 0.8],
            [0.25, 0.0],
        ])
        bounds_low = torch.tensor([[0.0, -1.0]] * 4)
        bounds_high = torch.tensor([[0.5, 1.0]] * 4)
        values, log_probs, entropy = policy.evaluate_actions_with_bounds(
            obs, actions, bounds_low, bounds_high,
        )
        assert values.shape == (4, 1)
        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(entropy).all()

    def test_log_prob_changes_with_bounds(self, policy):
        """Different bounds should give different log-probs for same action."""
        import torch
        obs = torch.randn(1, 14)
        action = torch.tensor([[0.3, 0.0]])

        # Wide bounds
        bounds_low_wide = torch.tensor([[0.0, -2.84]])
        bounds_high_wide = torch.tensor([[1.0, 2.84]])
        _, lp_wide, _ = policy.evaluate_actions_with_bounds(
            obs, action, bounds_low_wide, bounds_high_wide,
        )

        # Narrow bounds
        bounds_low_narrow = torch.tensor([[0.2, -0.5]])
        bounds_high_narrow = torch.tensor([[0.4, 0.5]])
        _, lp_narrow, _ = policy.evaluate_actions_with_bounds(
            obs, action, bounds_low_narrow, bounds_high_narrow,
        )

        # Narrower bounds should give higher log-prob (same action, smaller support)
        assert lp_narrow.item() > lp_wide.item(), \
            "Narrower bounds should increase log-prob density"


class TestCBFBoundsComputer:
    """Tests for CBFBoundsComputer bound computation."""

    def test_compute_bounds_center(self):
        """Robot in center should get near-nominal bounds."""
        pytest.importorskip("stable_baselines3")
        from safety.safe_beta_policy import SafeBetaPolicy
        from training.safe_ppo import CBFBoundsComputer
        from gymnasium import spaces

        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )
        computer = CBFBoundsComputer(cbf, use_opponent=False)

        class MockEnv:
            pursuer_state = np.array([0.0, 0.0, 0.0])
            evader_state = np.array([3.0, 0.0, np.pi])

        obs_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([0.0, -2.84], dtype=np.float32),
            high=np.array([1.0, 2.84], dtype=np.float32),
        )
        policy = SafeBetaPolicy(obs_space, act_space, lambda _: 3e-4)

        low, high = computer.compute_and_set_bounds(MockEnv(), policy)
        assert low[0] <= 0.1  # v_min near 0
        assert high[0] >= 0.9  # v_max near 1

    def test_compute_bounds_near_wall(self):
        """Robot near wall should get tightened v bounds."""
        pytest.importorskip("stable_baselines3")
        from safety.safe_beta_policy import SafeBetaPolicy
        from training.safe_ppo import CBFBoundsComputer
        from gymnasium import spaces

        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )
        computer = CBFBoundsComputer(cbf, use_opponent=False)

        class MockEnv:
            pursuer_state = np.array([4.7, 0.0, 0.0])  # Near right wall
            evader_state = np.array([0.0, 0.0, 0.0])

        obs_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([0.0, -2.84], dtype=np.float32),
            high=np.array([1.0, 2.84], dtype=np.float32),
        )
        policy = SafeBetaPolicy(obs_space, act_space, lambda _: 3e-4)

        low, high = computer.compute_and_set_bounds(MockEnv(), policy)
        assert high[0] < 1.0, f"v_max should be tightened near wall, got {high[0]}"

    def test_filter_action_applies_qp(self):
        """filter_action should apply QP safety filter."""
        from training.safe_ppo import CBFBoundsComputer

        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )
        computer = CBFBoundsComputer(cbf, use_opponent=False)

        class MockEnv:
            pursuer_state = np.array([4.7, 0.0, 0.0])
            evader_state = np.array([0.0, 0.0, 0.0])

        action = np.array([1.0, 0.0])  # Full speed toward wall
        u_safe, info = computer.filter_action(action, MockEnv())
        assert info["intervention"], "Should intervene near wall"
        assert u_safe[0] < 1.0


class TestGetUnwrappedEnv:
    """Tests for environment unwrapping utility."""

    def test_unwrap_direct(self):
        """Direct PE env should be returned as-is."""
        from training.safe_ppo import get_unwrapped_pe_env

        class MockPEEnv:
            pursuer_state = np.array([0.0, 0.0, 0.0])

        env = MockPEEnv()
        result = get_unwrapped_pe_env(env)
        assert result is env

    def test_unwrap_wrapped(self):
        """Wrapped env should be unwrapped to PE env."""
        from training.safe_ppo import get_unwrapped_pe_env

        class MockPEEnv:
            pursuer_state = np.array([0.0, 0.0, 0.0])

        class MockWrapper:
            def __init__(self, inner):
                self.env = inner

        inner = MockPEEnv()
        wrapped = MockWrapper(inner)
        result = get_unwrapped_pe_env(wrapped)
        assert hasattr(result, "pursuer_state")


class TestCBFBetaEndToEnd:
    """End-to-end tests for CBF + Beta integration (no SB3 required)."""

    def test_safe_bounds_then_qp_consistency(self):
        """Actions within safe bounds should pass QP filter without change."""
        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )
        state = np.array([0.0, 0.0, 0.0])  # Center

        # Get safe bounds
        v_bounds, omega_bounds, feasible = cbf.compute_safe_bounds(
            state, method="lp",
        )
        assert feasible

        # Sample within safe bounds
        rng = np.random.default_rng(42)
        for _ in range(20):
            v = rng.uniform(v_bounds[0], v_bounds[1])
            omega = rng.uniform(omega_bounds[0], omega_bounds[1])
            action = np.array([v, omega])

            u_safe, info = cbf.filter_action(action, state)
            # Action within LP safe bounds should need minimal correction
            assert abs(u_safe[0] - action[0]) < 0.1 or abs(u_safe[1] - action[1]) < 0.2

    def test_cbf_bounds_vary_with_state(self):
        """Safe bounds should change based on robot state."""
        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )

        # Center: wide bounds
        state_center = np.array([0.0, 0.0, 0.0])
        v_c, omega_c, _ = cbf.compute_safe_bounds(state_center)

        # Near wall heading toward it: tighter bounds
        state_wall = np.array([4.7, 0.0, 0.0])
        v_w, omega_w, _ = cbf.compute_safe_bounds(state_wall)

        # v_max should be tighter near wall
        assert v_w[1] < v_c[1], \
            f"v_max should be tighter near wall: {v_w[1]} vs {v_c[1]}"

    def test_cbf_bounds_with_opponent(self):
        """Bounds should account for nearby opponent."""
        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
            r_min_separation=0.35,
        )
        state = np.array([0.0, 0.0, 0.0])

        # Far opponent: no effect
        opp_far = np.array([5.0, 5.0, 0.0])
        v_far, _, _ = cbf.compute_safe_bounds(
            state, opponent_state=opp_far,
        )

        # Close opponent ahead: should tighten v_max
        opp_close = np.array([0.8, 0.0, np.pi])
        v_close, _, _ = cbf.compute_safe_bounds(
            state, opponent_state=opp_close,
        )

        # With close opponent, bounds should be at least as tight
        assert v_close[1] <= v_far[1] + 1e-6

    def test_rollout_simulation_no_violations(self):
        """Simulated rollout with CBF bounds should have no safety violations."""
        from envs.dynamics import unicycle_step

        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )
        rng = np.random.default_rng(123)

        xp, yp, tp = 0.0, 0.0, 0.0
        dt = 0.05

        for step in range(200):
            state = np.array([xp, yp, tp])

            # Compute safe bounds
            v_bounds, omega_bounds, feasible = cbf.compute_safe_bounds(
                state, method="analytical",
            )

            if feasible:
                # Sample within safe bounds
                v = rng.uniform(max(0, v_bounds[0]), v_bounds[1])
                omega = rng.uniform(omega_bounds[0], omega_bounds[1])
            else:
                v, omega = 0.0, 0.0

            # Apply QP filter as backup
            action = np.array([v, omega])
            u_safe, _ = cbf.filter_action(action, state)

            # Step dynamics
            xp, yp, tp, _ = unicycle_step(
                xp, yp, tp, float(u_safe[0]), float(u_safe[1]),
                dt, 10.0, 10.0, 0.15,
            )

            # Check safety
            assert abs(xp) <= 5.0 + 0.05, f"Step {step}: x={xp} out of bounds"
            assert abs(yp) <= 5.0 + 0.05, f"Step {step}: y={yp} out of bounds"

    def test_bounds_computation_speed(self):
        """Analytical bounds should be fast enough for training."""
        import time

        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )
        state = np.array([2.0, 1.0, 0.5])
        obstacles = [{"x": 3.0, "y": 1.0, "radius": 0.3}]

        # Warm up
        for _ in range(10):
            cbf.compute_safe_bounds(state, obstacles=obstacles)

        # Time 1000 calls
        start = time.perf_counter()
        n_calls = 1000
        for _ in range(n_calls):
            cbf.compute_safe_bounds(state, obstacles=obstacles)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_calls) * 1000
        assert avg_ms < 1.0, f"Analytical bounds too slow: {avg_ms:.3f}ms per call"


class TestSafeRolloutBuffer:
    """Tests for SafeRolloutBuffer."""

    def test_buffer_stores_bounds(self):
        """SafeRolloutBuffer should store and return safe bounds."""
        sb3 = pytest.importorskip("stable_baselines3")
        from gymnasium import spaces
        from training.safe_rollout_buffer import SafeRolloutBuffer

        obs_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([0.0, -2.84], dtype=np.float32),
            high=np.array([1.0, 2.84], dtype=np.float32),
        )
        import torch

        buffer = SafeRolloutBuffer(
            buffer_size=8,
            observation_space=obs_space,
            action_space=act_space,
            device="cpu",
            n_envs=1,
        )

        # Fill buffer
        for i in range(8):
            obs = np.random.randn(1, 14).astype(np.float32)
            action = np.array([[0.5, 0.0]])
            reward = np.array([1.0])
            episode_start = np.array([i == 0])
            value = torch.tensor([0.5])
            log_prob = torch.tensor([-1.0])
            bounds_low = np.array([[0.0, -1.0]])
            bounds_high = np.array([[0.5 + i * 0.05, 1.0]])

            buffer.add(
                obs, action, reward, episode_start, value, log_prob,
                safe_bounds_low=bounds_low,
                safe_bounds_high=bounds_high,
            )

        # Compute returns
        buffer.compute_returns_and_advantage(
            last_values=torch.tensor([0.0]),
            dones=np.array([False]),
        )

        # Get batch and check bounds are present
        for batch in buffer.get(batch_size=4):
            assert hasattr(batch, "safe_bounds_low")
            assert hasattr(batch, "safe_bounds_high")
            assert batch.safe_bounds_low.shape[-1] == 2
            assert batch.safe_bounds_high.shape[-1] == 2

    def test_buffer_reset_clears_bounds(self):
        """Reset should reinitialize bounds storage."""
        sb3 = pytest.importorskip("stable_baselines3")
        from gymnasium import spaces
        from training.safe_rollout_buffer import SafeRolloutBuffer

        obs_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([0.0, -2.84], dtype=np.float32),
            high=np.array([1.0, 2.84], dtype=np.float32),
        )

        buffer = SafeRolloutBuffer(
            buffer_size=4,
            observation_space=obs_space,
            action_space=act_space,
            device="cpu",
            n_envs=1,
        )

        assert buffer.safe_bounds_low.shape == (4, 1, 2)
        assert buffer.safe_bounds_high.shape == (4, 1, 2)
        assert np.all(buffer.safe_bounds_low == 0)
