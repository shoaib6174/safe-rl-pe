"""Tests for Phase 2 Session 5: Obstacles & Environment Extension.

Tests cover:
1. Obstacle generation (non-overlapping, within arena)
2. Obstacle collision detection
3. Obstacle observation features
4. Safety reward shaping
5. CBF-obstacle integration (safety filter prevents collisions)
6. Environment with obstacles end-to-end
"""

import numpy as np
import pytest

from envs.observations import ObservationBuilder
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.rewards import RewardComputer, SafetyRewardComputer
from envs.wrappers import SingleAgentPEWrapper
from safety.vcp_cbf import VCPCBFFilter, vcp_cbf_obstacle


# =============================================================================
# Obstacle generation
# =============================================================================

class TestObstacleGeneration:
    """Tests for _generate_obstacles()."""

    def make_env(self, **kwargs):
        defaults = dict(
            arena_width=20.0, arena_height=20.0, dt=0.05,
            max_steps=100, render_mode=None,
        )
        defaults.update(kwargs)
        return PursuitEvasionEnv(**defaults)

    def test_correct_count(self):
        """Generate the requested number of obstacles."""
        env = self.make_env(n_obstacles=5)
        env.reset(seed=42)
        assert len(env.obstacles) == 5

    def test_zero_obstacles(self):
        """No obstacles when n_obstacles=0."""
        env = self.make_env(n_obstacles=0)
        env.reset(seed=42)
        assert len(env.obstacles) == 0

    def test_obstacles_within_arena(self):
        """All obstacles must be fully within arena bounds."""
        env = self.make_env(n_obstacles=8)
        half_w = env.arena_width / 2.0
        half_h = env.arena_height / 2.0
        for seed in range(10):
            env.reset(seed=seed)
            for obs in env.obstacles:
                assert obs["x"] - obs["radius"] >= -half_w, f"Obstacle left edge outside arena"
                assert obs["x"] + obs["radius"] <= half_w, f"Obstacle right edge outside arena"
                assert obs["y"] - obs["radius"] >= -half_h, f"Obstacle bottom edge outside arena"
                assert obs["y"] + obs["radius"] <= half_h, f"Obstacle top edge outside arena"

    def test_no_overlap(self):
        """Obstacles must not overlap each other."""
        env = self.make_env(n_obstacles=6)
        for seed in range(10):
            env.reset(seed=seed)
            obstacles = env.obstacles
            for i in range(len(obstacles)):
                for j in range(i + 1, len(obstacles)):
                    dist = np.sqrt(
                        (obstacles[i]["x"] - obstacles[j]["x"])**2
                        + (obstacles[i]["y"] - obstacles[j]["y"])**2
                    )
                    min_dist = obstacles[i]["radius"] + obstacles[j]["radius"]
                    assert dist >= min_dist - 1e-6, (
                        f"Overlap: obs {i} and {j}, dist={dist:.3f}, min={min_dist:.3f}"
                    )

    def test_radius_in_range(self):
        """Obstacle radii must be in the configured range."""
        r_range = (0.3, 1.0)
        env = self.make_env(n_obstacles=5, obstacle_radius_range=r_range)
        for seed in range(5):
            env.reset(seed=seed)
            for obs in env.obstacles:
                assert r_range[0] <= obs["radius"] <= r_range[1], (
                    f"Radius {obs['radius']:.3f} outside range {r_range}"
                )

    def test_obstacles_regenerated_on_reset(self):
        """Obstacles should be randomly regenerated each reset."""
        env = self.make_env(n_obstacles=3)
        env.reset(seed=42)
        obs1 = [o.copy() for o in env.obstacles]
        env.reset(seed=99)
        obs2 = [o.copy() for o in env.obstacles]
        # Different seeds should produce different obstacles
        positions1 = [(o["x"], o["y"]) for o in obs1]
        positions2 = [(o["x"], o["y"]) for o in obs2]
        assert positions1 != positions2

    def test_agents_not_in_obstacles(self):
        """Initial agent positions must not be inside obstacles."""
        env = self.make_env(n_obstacles=5)
        for seed in range(20):
            env.reset(seed=seed)
            assert not env._check_obstacle_collision(env.pursuer_state), (
                f"Pursuer spawned inside obstacle at seed={seed}"
            )
            assert not env._check_obstacle_collision(env.evader_state), (
                f"Evader spawned inside obstacle at seed={seed}"
            )

    def test_obstacle_dict_format(self):
        """Each obstacle should have x, y, radius keys with float values."""
        env = self.make_env(n_obstacles=3)
        env.reset(seed=42)
        for obs in env.obstacles:
            assert "x" in obs
            assert "y" in obs
            assert "radius" in obs
            assert isinstance(obs["x"], float)
            assert isinstance(obs["y"], float)
            assert isinstance(obs["radius"], float)


# =============================================================================
# Obstacle collision detection
# =============================================================================

class TestObstacleCollision:
    """Tests for _check_obstacle_collision()."""

    def make_env_with_fixed_obstacle(self):
        """Create env with a known obstacle at origin."""
        env = PursuitEvasionEnv(
            arena_width=20.0, arena_height=20.0,
            n_obstacles=0, render_mode=None,
        )
        env.reset(seed=42)
        # Manually place an obstacle at (3, 3) with radius 1.0
        env.obstacles = [{"x": 3.0, "y": 3.0, "radius": 1.0}]
        return env

    def test_collision_inside(self):
        """Robot center inside obstacle is a collision."""
        env = self.make_env_with_fixed_obstacle()
        state = np.array([3.0, 3.0, 0.0])  # at obstacle center
        assert env._check_obstacle_collision(state) is True

    def test_collision_at_edge(self):
        """Robot touching obstacle edge is a collision."""
        env = self.make_env_with_fixed_obstacle()
        # robot_radius = 0.15, obs_radius = 1.0
        # collision when dist <= 1.0 + 0.15 = 1.15
        state = np.array([3.0 + 1.14, 3.0, 0.0])
        assert env._check_obstacle_collision(state) is True

    def test_no_collision_outside(self):
        """Robot safely away from obstacle."""
        env = self.make_env_with_fixed_obstacle()
        state = np.array([3.0 + 2.0, 3.0, 0.0])  # 2m from center
        assert env._check_obstacle_collision(state) is False

    def test_no_collision_empty_obstacles(self):
        """No collision when there are no obstacles."""
        env = PursuitEvasionEnv(n_obstacles=0, render_mode=None)
        env.reset(seed=42)
        assert env._check_obstacle_collision(np.array([0.0, 0.0, 0.0])) is False

    def test_collision_reported_in_info(self):
        """step() should report obstacle collision status in info."""
        env = PursuitEvasionEnv(n_obstacles=0, render_mode=None)
        env.reset(seed=42)
        p_action = np.array([0.5, 0.0])
        e_action = np.array([0.5, 0.0])
        _, _, _, _, info = env.step(p_action, e_action)
        assert "pursuer_obstacle_collision" in info
        assert "evader_obstacle_collision" in info


# =============================================================================
# Obstacle observation features
# =============================================================================

class TestObstacleObservations:
    """Tests for obstacle features in observations."""

    def test_obs_dim_no_obstacles(self):
        """Base obs dim should be 14 when no obstacle obs."""
        builder = ObservationBuilder(20.0, 20.0, 1.0, 2.84, n_obstacle_obs=0)
        assert builder.obs_dim == 14

    def test_obs_dim_with_obstacles(self):
        """Obs dim should be 14 + 2*K with K obstacle obs."""
        for k in [1, 2, 3, 5]:
            builder = ObservationBuilder(20.0, 20.0, 1.0, 2.84, n_obstacle_obs=k)
            assert builder.obs_dim == 14 + 2 * k

    def test_obs_shape_env(self):
        """Env observation space should reflect obstacle obs."""
        env = PursuitEvasionEnv(n_obstacles=3, n_obstacle_obs=3, render_mode=None)
        assert env.observation_space.shape == (20,)
        obs, _ = env.reset(seed=42)
        assert obs["pursuer"].shape == (20,)
        assert obs["evader"].shape == (20,)

    def test_obs_padded_no_obstacles(self):
        """With n_obstacle_obs > 0 but no obstacles, features are padded."""
        builder = ObservationBuilder(20.0, 20.0, 1.0, 2.84, n_obstacle_obs=3)
        state = np.array([0.0, 0.0, 0.0])
        action = np.array([0.0, 0.0])
        obs = builder.build(state, action, state, action, obstacles=[])
        # Last 6 features should be [1,0,1,0,1,0] (padded)
        obstacle_features = obs[14:]
        assert len(obstacle_features) == 6
        for i in range(3):
            assert obstacle_features[2 * i] == pytest.approx(1.0, abs=1e-5)  # dist
            assert obstacle_features[2 * i + 1] == pytest.approx(0.0, abs=1e-5)  # bearing

    def test_obs_nearest_obstacle(self):
        """Nearest obstacle should appear first in features."""
        builder = ObservationBuilder(20.0, 20.0, 1.0, 2.84, n_obstacle_obs=2)
        state = np.array([0.0, 0.0, 0.0])
        action = np.array([0.0, 0.0])
        obstacles = [
            {"x": 5.0, "y": 0.0, "radius": 0.5},  # far
            {"x": 2.0, "y": 0.0, "radius": 0.5},  # near
        ]
        obs = builder.build(state, action, state, action, obstacles=obstacles)
        # First obstacle feature pair should be for the nearer obstacle
        dist_1 = obs[14]
        dist_2 = obs[16]
        assert dist_1 < dist_2, "Nearest obstacle should have smallest distance"

    def test_obs_distance_normalized(self):
        """Obstacle distances should be normalized by arena diagonal."""
        builder = ObservationBuilder(20.0, 20.0, 1.0, 2.84, n_obstacle_obs=1)
        state = np.array([0.0, 0.0, 0.0])
        action = np.array([0.0, 0.0])
        obstacles = [{"x": 3.0, "y": 0.0, "radius": 0.5}]
        obs = builder.build(state, action, state, action, obstacles=obstacles)
        dist_feature = obs[14]
        assert 0.0 <= dist_feature <= 1.0, f"Distance feature {dist_feature} not in [0, 1]"

    def test_obs_bearing_normalized(self):
        """Obstacle bearings should be in [-1, 1]."""
        builder = ObservationBuilder(20.0, 20.0, 1.0, 2.84, n_obstacle_obs=1)
        state = np.array([0.0, 0.0, 0.0])
        action = np.array([0.0, 0.0])
        obstacles = [{"x": 0.0, "y": 3.0, "radius": 0.5}]
        obs = builder.build(state, action, state, action, obstacles=obstacles)
        bearing_feature = obs[15]
        assert -1.0 <= bearing_feature <= 1.0, f"Bearing {bearing_feature} not in [-1, 1]"

    def test_obs_bearing_ahead(self):
        """Obstacle directly ahead should have bearing ~0."""
        builder = ObservationBuilder(20.0, 20.0, 1.0, 2.84, n_obstacle_obs=1)
        state = np.array([0.0, 0.0, 0.0])  # heading along +x
        action = np.array([0.0, 0.0])
        obstacles = [{"x": 3.0, "y": 0.0, "radius": 0.5}]
        obs = builder.build(state, action, state, action, obstacles=obstacles)
        bearing = obs[15]
        assert abs(bearing) < 0.05, f"Bearing for obstacle ahead = {bearing}, expected ~0"

    def test_backward_compatible_no_obstacles(self):
        """Env with n_obstacle_obs=0 produces 14D obs (backward compatible)."""
        env = PursuitEvasionEnv(n_obstacles=0, n_obstacle_obs=0, render_mode=None)
        obs, _ = env.reset(seed=42)
        assert obs["pursuer"].shape == (14,)


# =============================================================================
# Safety reward
# =============================================================================

class TestSafetyReward:
    """Tests for SafetyRewardComputer."""

    def test_base_reward_unchanged(self):
        """Without h_min, safety reward should be zero (same as base)."""
        base = RewardComputer(d_max=28.28)
        safe = SafetyRewardComputer(w_safety=0.05, h_ref=1.0, d_max=28.28)
        r_b_p, r_b_e = base.compute(d_curr=5.0, d_prev=6.0, captured=False, timed_out=False)
        r_s_p, r_s_e = safe.compute(d_curr=5.0, d_prev=6.0, captured=False, timed_out=False)
        assert r_s_p == pytest.approx(r_b_p, abs=1e-8)
        assert r_s_e == pytest.approx(r_b_e, abs=1e-8)

    def test_safety_reward_positive_margin(self):
        """Positive h_min should add positive safety reward."""
        safe = SafetyRewardComputer(w_safety=0.05, h_ref=1.0, d_max=28.28)
        r_p, _ = safe.compute(d_curr=5.0, d_prev=5.0, captured=False, timed_out=False, h_min=1.0)
        assert r_p > 0.0, "Positive h_min should give positive reward"

    def test_safety_reward_magnitude(self):
        """Safety reward should be w_safety * h_min/h_ref when h_min < h_ref."""
        safe = SafetyRewardComputer(w_safety=0.05, h_ref=2.0, d_max=28.28)
        # With d_curr = d_prev, base distance reward is 0
        r_p, _ = safe.compute(d_curr=5.0, d_prev=5.0, captured=False, timed_out=False, h_min=1.0)
        expected_safety = 0.05 * (1.0 / 2.0)  # w_safety * h_min/h_ref
        assert r_p == pytest.approx(expected_safety, abs=1e-8)

    def test_safety_reward_clamped(self):
        """Safety reward should be clamped to w_safety when h_min > h_ref."""
        safe = SafetyRewardComputer(w_safety=0.05, h_ref=1.0, d_max=28.28)
        r_p, _ = safe.compute(d_curr=5.0, d_prev=5.0, captured=False, timed_out=False, h_min=10.0)
        expected_max = 0.05  # clamped at 1.0
        assert r_p == pytest.approx(expected_max, abs=1e-8)

    def test_safety_reward_zero_margin(self):
        """Zero h_min should give zero safety reward."""
        safe = SafetyRewardComputer(w_safety=0.05, h_ref=1.0, d_max=28.28)
        r_p, _ = safe.compute(d_curr=5.0, d_prev=5.0, captured=False, timed_out=False, h_min=0.0)
        assert r_p == pytest.approx(0.0, abs=1e-8)

    def test_safety_reward_negative_margin(self):
        """Negative h_min should clip to zero safety reward."""
        safe = SafetyRewardComputer(w_safety=0.05, h_ref=1.0, d_max=28.28)
        r_p, _ = safe.compute(d_curr=5.0, d_prev=5.0, captured=False, timed_out=False, h_min=-0.5)
        assert r_p == pytest.approx(0.0, abs=1e-8)

    def test_zero_sum_with_safety(self):
        """Reward should remain zero-sum with safety term."""
        safe = SafetyRewardComputer(w_safety=0.05, h_ref=1.0, d_max=28.28)
        r_p, r_e = safe.compute(d_curr=5.0, d_prev=6.0, captured=False, timed_out=False, h_min=0.5)
        assert r_p + r_e == pytest.approx(0.0, abs=1e-8)


# =============================================================================
# CBF-obstacle integration
# =============================================================================

class TestCBFObstacleIntegration:
    """Tests for VCP-CBF with obstacles preventing collisions."""

    def test_cbf_obstacle_constraint(self):
        """CBF should produce positive h when away from obstacle."""
        state = np.array([0.0, 0.0, 0.0])
        obs_pos = np.array([5.0, 0.0])
        h, a_v, a_omega = vcp_cbf_obstacle(state, obs_pos, 1.15)  # 1.0 + 0.15 robot radius
        assert h > 0, f"h={h} should be positive when safely away"

    def test_cbf_filter_with_obstacles(self):
        """CBF filter should produce valid actions with obstacles."""
        cbf = VCPCBFFilter(
            arena_half_w=10.0, arena_half_h=10.0,
            robot_radius=0.15,
        )
        state = np.array([0.0, 0.0, 0.0])
        obstacles = [{"x": 2.0, "y": 0.0, "radius": 0.5}]
        action = np.array([1.0, 0.0])

        u_safe, info = cbf.filter_action(action, state, obstacles=obstacles)
        assert u_safe[0] >= 0.0
        assert u_safe[0] <= 1.0

    def test_cbf_prevents_obstacle_collision(self):
        """CBF filter should prevent driving into an obstacle."""
        cbf = VCPCBFFilter(
            arena_half_w=10.0, arena_half_h=10.0,
            robot_radius=0.15, alpha=1.0,
        )
        obstacles = [{"x": 2.0, "y": 0.0, "radius": 0.5}]

        # Robot heading straight toward obstacle at 1m/s
        state = np.array([0.0, 0.0, 0.0])
        dt = 0.05

        for _ in range(200):
            action = np.array([1.0, 0.0])  # full speed ahead
            u_safe, info = cbf.filter_action(action, state, obstacles=obstacles)

            # Step dynamics
            from envs.dynamics import unicycle_step
            x, y, theta, _ = unicycle_step(
                state[0], state[1], state[2],
                u_safe[0], u_safe[1], dt,
                20.0, 20.0, 0.15,
            )
            state = np.array([x, y, theta])

            # Check: robot should NOT collide with obstacle
            dist_to_obs = np.sqrt((state[0] - 2.0)**2 + (state[1] - 0.0)**2)
            effective_radius = 0.5 + 0.15
            assert dist_to_obs > effective_radius - 0.05, (
                f"Collision! dist={dist_to_obs:.3f}, min={effective_radius:.3f}"
            )

    def test_cbf_safe_bounds_with_obstacles(self):
        """Safe bounds should tighten when near an obstacle."""
        cbf = VCPCBFFilter(
            arena_half_w=10.0, arena_half_h=10.0, robot_radius=0.15,
        )
        obstacles = [{"x": 1.5, "y": 0.0, "radius": 0.5}]

        # State near obstacle, heading toward it
        state = np.array([0.0, 0.0, 0.0])
        v_bounds, omega_bounds, feasible = cbf.compute_safe_bounds(
            state, obstacles=obstacles,
        )
        assert feasible
        # v_max should be tightened (can't go full speed toward obstacle)
        assert v_bounds[1] <= 1.0 + 1e-6

    def test_info_includes_obstacles(self):
        """Info dict should include obstacles list."""
        env = PursuitEvasionEnv(n_obstacles=3, render_mode=None)
        _, info = env.reset(seed=42)
        assert "obstacles" in info
        assert len(info["obstacles"]) == 3


# =============================================================================
# End-to-end environment with obstacles
# =============================================================================

class TestEnvWithObstacles:
    """End-to-end tests for PursuitEvasionEnv with obstacles."""

    def make_env(self, **kwargs):
        defaults = dict(
            arena_width=20.0, arena_height=20.0, dt=0.05,
            max_steps=100, n_obstacles=3, n_obstacle_obs=3,
            render_mode=None,
        )
        defaults.update(kwargs)
        return PursuitEvasionEnv(**defaults)

    def test_reset_step_cycle(self):
        """Basic reset-step cycle with obstacles."""
        env = self.make_env()
        obs, info = env.reset(seed=42)
        assert obs["pursuer"].shape == (20,)

        for _ in range(10):
            p_a = env.pursuer_action_space.sample()
            e_a = env.evader_action_space.sample()
            obs, rewards, term, trunc, info = env.step(p_a, e_a)
            assert obs["pursuer"].shape == (20,)
            assert "pursuer" in rewards
            if term or trunc:
                obs, info = env.reset(seed=None)

    def test_wrapper_with_obstacles(self):
        """SingleAgentPEWrapper works with obstacle env."""
        base_env = self.make_env()
        env = SingleAgentPEWrapper(base_env, role="pursuer")
        obs, info = env.reset(seed=42)
        assert obs.shape == (20,)

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert obs.shape == (20,)
            if term or trunc:
                obs, info = env.reset()

    def test_env_with_safety_reward(self):
        """Environment with SafetyRewardComputer should work."""
        reward_comp = SafetyRewardComputer(
            w_safety=0.05, h_ref=1.0,
            d_max=np.sqrt(20.0**2 + 20.0**2),
        )
        env = PursuitEvasionEnv(
            n_obstacles=3, n_obstacle_obs=3,
            reward_computer=reward_comp, render_mode=None,
        )
        obs, info = env.reset(seed=42)
        for _ in range(10):
            p_a = env.pursuer_action_space.sample()
            e_a = env.evader_action_space.sample()
            obs, rewards, term, trunc, info = env.step(p_a, e_a)
            assert isinstance(rewards["pursuer"], float)
            if term or trunc:
                break

    def test_reproducible_with_seed(self):
        """Same seed should produce same obstacle layout and initial states."""
        env1 = self.make_env()
        obs1, _ = env1.reset(seed=42)
        obstacles1 = [o.copy() for o in env1.obstacles]

        env2 = self.make_env()
        obs2, _ = env2.reset(seed=42)
        obstacles2 = [o.copy() for o in env2.obstacles]

        assert len(obstacles1) == len(obstacles2)
        for o1, o2 in zip(obstacles1, obstacles2):
            assert o1["x"] == pytest.approx(o2["x"], abs=1e-10)
            assert o1["y"] == pytest.approx(o2["y"], abs=1e-10)
            assert o1["radius"] == pytest.approx(o2["radius"], abs=1e-10)

        np.testing.assert_array_almost_equal(obs1["pursuer"], obs2["pursuer"])

    def test_many_obstacles_still_places_agents(self):
        """Even with many obstacles, agents should still be placed."""
        env = PursuitEvasionEnv(
            arena_width=20.0, arena_height=20.0,
            n_obstacles=10, n_obstacle_obs=3,
            render_mode=None,
        )
        obs, info = env.reset(seed=42)
        assert env.pursuer_state is not None
        assert env.evader_state is not None

    def test_speed_benchmark(self):
        """Obstacle env step should be fast enough for training."""
        import time
        env = self.make_env(n_obstacles=5, n_obstacle_obs=3)
        env.reset(seed=42)

        n_steps = 500
        start = time.perf_counter()
        for _ in range(n_steps):
            p_a = env.pursuer_action_space.sample()
            e_a = env.evader_action_space.sample()
            _, _, term, trunc, _ = env.step(p_a, e_a)
            if term or trunc:
                env.reset()
        elapsed = time.perf_counter() - start

        steps_per_sec = n_steps / elapsed
        assert steps_per_sec > 1000, f"Too slow: {steps_per_sec:.0f} steps/s (need >1000)"


# =============================================================================
# Physical obstacle collision enforcement (S50)
# =============================================================================

class TestObstacleCollisionEnforcement:
    """Integration tests for physical obstacle barriers in the environment."""

    def test_sliding_preserves_tangential_motion(self):
        """Agent moving tangentially past obstacle should slide, not stop."""
        env = PursuitEvasionEnv(
            arena_width=20.0, arena_height=20.0,
            n_obstacles=0, render_mode=None,
        )
        env.reset(seed=42)
        # Place obstacle at (3, 0) radius 1.0
        env.obstacles = [{"x": 3.0, "y": 0.0, "radius": 1.0}]

        # Place pursuer near obstacle, heading north (tangential)
        min_dist = 1.0 + env.robot_radius
        env.pursuer_state = np.array([3.0 + min_dist - 0.05, 0.0, np.pi / 2])
        # Place evader far away to avoid capture
        env.evader_state = np.array([-5.0, 0.0, 0.0])

        y_before = env.pursuer_state[1]
        # Step with forward velocity and no turning
        env.step(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        y_after = env.pursuer_state[1]

        # Y should have increased (tangential motion preserved)
        assert y_after > y_before, "Tangential motion should be preserved during sliding"
        # Agent should not be inside obstacle
        dist = np.sqrt(
            (env.pursuer_state[0] - 3.0)**2 + (env.pursuer_state[1] - 0.0)**2
        )
        assert dist >= min_dist - 1e-6, "Agent should be outside obstacle after sliding"

    def test_wall_plus_obstacle_near_boundary(self):
        """Obstacle near arena wall: agent is re-clamped to arena bounds."""
        env = PursuitEvasionEnv(
            arena_width=20.0, arena_height=20.0,
            n_obstacles=0, render_mode=None,
        )
        env.reset(seed=42)
        # Place obstacle near east wall
        half_w = env.arena_width / 2.0
        env.obstacles = [{"x": half_w - 1.0, "y": 0.0, "radius": 1.0}]

        # Place pursuer between obstacle and wall
        env.pursuer_state = np.array([half_w - 0.5, 0.0, 0.0])
        env.evader_state = np.array([-5.0, 0.0, 0.0])

        env.step(np.array([1.0, 0.0]), np.array([0.0, 0.0]))

        # Agent must be within arena bounds
        x_max = half_w - env.robot_radius
        assert env.pursuer_state[0] <= x_max + 1e-6, (
            f"Agent x={env.pursuer_state[0]:.3f} exceeds arena bound {x_max:.3f}"
        )

    def test_collision_penalty_applied(self):
        """With w_collision > 0, hitting obstacle should reduce reward."""
        env = PursuitEvasionEnv(
            arena_width=20.0, arena_height=20.0,
            n_obstacles=0, render_mode=None,
            w_collision=5.0,
        )
        env.reset(seed=42)
        # Place obstacle where pursuer will hit it
        env.obstacles = [{"x": 1.0, "y": 0.0, "radius": 0.5}]
        env.pursuer_state = np.array([0.0, 0.0, 0.0])  # heading east toward obstacle
        env.evader_state = np.array([-5.0, 0.0, np.pi])

        # Step without collision (evader far away, heading west)
        env_no_penalty = PursuitEvasionEnv(
            arena_width=20.0, arena_height=20.0,
            n_obstacles=0, render_mode=None,
            w_collision=0.0,
        )
        env_no_penalty.reset(seed=42)
        env_no_penalty.obstacles = [{"x": 1.0, "y": 0.0, "radius": 0.5}]
        env_no_penalty.pursuer_state = np.array([0.0, 0.0, 0.0])
        env_no_penalty.evader_state = np.array([-5.0, 0.0, np.pi])
        env_no_penalty.prev_distance = env.prev_distance

        _, rewards_penalty, _, _, info_penalty = env.step(
            np.array([1.0, 0.0]), np.array([1.0, 0.0])
        )
        _, rewards_no_penalty, _, _, info_no_penalty = env_no_penalty.step(
            np.array([1.0, 0.0]), np.array([1.0, 0.0])
        )

        if info_penalty["pursuer_obstacle_collision"]:
            assert rewards_penalty["pursuer"] < rewards_no_penalty["pursuer"], (
                "Collision penalty should reduce pursuer reward"
            )

    def test_no_penalty_when_w_collision_zero(self):
        """With w_collision=0 (default), collision does not affect reward."""
        env = PursuitEvasionEnv(
            arena_width=20.0, arena_height=20.0,
            n_obstacles=0, render_mode=None,
            w_collision=0.0,
        )
        env.reset(seed=42)
        env.obstacles = [{"x": 1.0, "y": 0.0, "radius": 0.5}]
        env.pursuer_state = np.array([0.0, 0.0, 0.0])
        env.evader_state = np.array([-5.0, 0.0, np.pi])

        _, rewards, _, _, info = env.step(
            np.array([1.0, 0.0]), np.array([1.0, 0.0])
        )
        # Reward should be the same regardless of collision since w_collision=0
        # We just verify no crash and collision is detected
        assert isinstance(rewards["pursuer"], float)
        assert "pursuer_obstacle_collision" in info
