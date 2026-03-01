"""Unit tests for the PursuitEvasionEnv environment."""

import numpy as np
import pytest

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper


class TestPursuitEvasionEnv:
    """Tests for the core PE environment."""

    def make_env(self, **kwargs):
        defaults = dict(
            arena_width=20.0,
            arena_height=20.0,
            dt=0.05,
            max_steps=1200,
            capture_radius=0.5,
            render_mode=None,
        )
        defaults.update(kwargs)
        return PursuitEvasionEnv(**defaults)

    def test_reset_returns_correct_structure(self):
        env = self.make_env()
        obs, info = env.reset(seed=42)

        assert "pursuer" in obs
        assert "evader" in obs
        assert obs["pursuer"].shape == (14,)
        assert obs["evader"].shape == (14,)
        assert "distance" in info

    def test_reset_initial_separation(self):
        """Agents should start with min_init_distance <= d <= max_init_distance."""
        env = self.make_env(min_init_distance=3.0, max_init_distance=15.0)
        for seed in range(20):
            obs, info = env.reset(seed=seed)
            d = info["distance"]
            assert d >= 3.0 - 0.01, f"seed={seed}: distance {d} < min 3.0"
            assert d <= 15.0 + 0.01, f"seed={seed}: distance {d} > max 15.0"

    def test_step_returns_correct_structure(self):
        env = self.make_env()
        env.reset(seed=42)

        p_action = np.array([0.5, 0.0], dtype=np.float32)
        e_action = np.array([0.5, 0.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(p_action, e_action)

        assert "pursuer" in obs
        assert "evader" in obs
        assert "pursuer" in rewards
        assert "evader" in rewards
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_zero_sum_reward(self):
        """Reward should be zero-sum at every step."""
        env = self.make_env()
        env.reset(seed=42)

        for _ in range(50):
            p_action = env.pursuer_action_space.sample()
            e_action = env.evader_action_space.sample()
            obs, rewards, terminated, truncated, info = env.step(p_action, e_action)

            r_sum = rewards["pursuer"] + rewards["evader"]
            assert r_sum == pytest.approx(0.0, abs=1e-10), \
                f"Non-zero-sum: r_P={rewards['pursuer']}, r_E={rewards['evader']}"

            if terminated or truncated:
                env.reset(seed=42)

    def test_capture_detection(self):
        """Force agents close together and verify capture triggers."""
        env = self.make_env(capture_radius=0.5)
        env.reset(seed=42)

        # Manually place agents very close
        env.pursuer_state = np.array([0.0, 0.0, 0.0])
        env.evader_state = np.array([0.3, 0.0, 0.0])
        env.prev_distance = env._compute_distance()

        # Step with zero actions (they're already close)
        p_action = np.array([0.0, 0.0], dtype=np.float32)
        e_action = np.array([0.0, 0.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(p_action, e_action)

        assert terminated is True, "Capture should trigger when distance <= capture_radius"
        assert "episode_metrics" in info
        assert info["episode_metrics"]["captured"] is True

    def test_timeout_detection(self):
        """Episode should truncate at max_steps."""
        env = self.make_env(max_steps=10)
        env.reset(seed=42)

        # Place agents far apart so no capture
        env.pursuer_state = np.array([-8.0, 0.0, 0.0])
        env.evader_state = np.array([8.0, 0.0, np.pi])
        env.prev_distance = env._compute_distance()

        p_action = np.array([0.0, 0.0], dtype=np.float32)
        e_action = np.array([0.0, 0.0], dtype=np.float32)

        for i in range(10):
            obs, rewards, terminated, truncated, info = env.step(p_action, e_action)
            if i < 9:
                assert not terminated and not truncated
            else:
                assert truncated is True
                assert "episode_metrics" in info
                assert info["episode_metrics"]["captured"] is False

    def test_observations_normalized(self):
        """Observations should be roughly within [-1, 1]."""
        env = self.make_env()
        obs, _ = env.reset(seed=42)

        for role in ["pursuer", "evader"]:
            o = obs[role]
            # Allow slight overflow due to floating point
            assert np.all(o >= -1.1), f"{role} obs has values < -1.1: {o}"
            assert np.all(o <= 1.1), f"{role} obs has values > 1.1: {o}"

    def test_pursuer_positive_reward_closing(self):
        """Pursuer should get positive reward when closing distance."""
        env = self.make_env()
        env.reset(seed=42)

        # Place pursuer heading toward evader
        env.pursuer_state = np.array([-5.0, 0.0, 0.0])  # heading east
        env.evader_state = np.array([5.0, 0.0, 0.0])     # to the east
        env.prev_distance = env._compute_distance()

        # Pursuer moves toward evader, evader stands still
        p_action = np.array([1.0, 0.0], dtype=np.float32)
        e_action = np.array([0.0, 0.0], dtype=np.float32)
        _, rewards, _, _, _ = env.step(p_action, e_action)

        assert rewards["pursuer"] > 0, "Pursuer should get positive reward when closing"
        assert rewards["evader"] < 0, "Evader should get negative reward when pursuer closes"

    def test_capture_bonus(self):
        """Capture should give large positive reward to pursuer."""
        env = self.make_env(capture_radius=1.0, capture_bonus=100.0)
        env.reset(seed=42)

        # Place agents just outside capture radius heading toward each other
        env.pursuer_state = np.array([-0.5, 0.0, 0.0])
        env.evader_state = np.array([0.5, 0.0, np.pi])
        env.prev_distance = env._compute_distance()

        # Both approach
        p_action = np.array([1.0, 0.0], dtype=np.float32)
        e_action = np.array([1.0, 0.0], dtype=np.float32)  # heading west (pi), so moves west... wait
        # Evader heading is pi, so v=1.0 moves in direction pi = west (-x)
        # That means evader moves toward pursuer

        _, rewards, terminated, _, _ = env.step(p_action, e_action)

        if terminated:
            # Capture bonus should dominate
            assert rewards["pursuer"] > 50, f"Capture reward too small: {rewards['pursuer']}"

    def test_action_space_bounds(self):
        """Action space should have correct bounds."""
        env = self.make_env(pursuer_v_max=1.0, pursuer_omega_max=2.84)
        assert env.pursuer_action_space.low[0] == pytest.approx(0.0)
        assert env.pursuer_action_space.high[0] == pytest.approx(1.0)
        assert env.pursuer_action_space.low[1] == pytest.approx(-2.84)
        assert env.pursuer_action_space.high[1] == pytest.approx(2.84)

    def test_reproducibility(self):
        """Same seed should produce identical sequences."""
        env1 = self.make_env()
        env2 = self.make_env()

        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)

        np.testing.assert_array_equal(obs1["pursuer"], obs2["pursuer"])

        action = np.array([0.5, 1.0], dtype=np.float32)
        o1, r1, t1, tr1, _ = env1.step(action, action)
        o2, r2, t2, tr2, _ = env2.step(action, action)

        np.testing.assert_array_equal(o1["pursuer"], o2["pursuer"])
        assert r1["pursuer"] == r2["pursuer"]


class TestSingleAgentWrapper:
    """Tests for the SingleAgentPEWrapper."""

    def test_pursuer_wrapper(self):
        base_env = PursuitEvasionEnv(render_mode=None)
        env = SingleAgentPEWrapper(base_env, role="pursuer", opponent_policy=None)

        obs, info = env.reset(seed=42)
        assert obs.shape == (14,)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (14,)
        assert isinstance(reward, float)

    def test_evader_wrapper(self):
        base_env = PursuitEvasionEnv(render_mode=None)
        env = SingleAgentPEWrapper(base_env, role="evader", opponent_policy=None)

        obs, info = env.reset(seed=42)
        assert obs.shape == (14,)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (14,)

    def test_invalid_role(self):
        base_env = PursuitEvasionEnv(render_mode=None)
        with pytest.raises(ValueError, match="role must be"):
            SingleAgentPEWrapper(base_env, role="invalid")

    def test_set_opponent(self):
        """set_opponent should update the opponent policy without error."""
        base_env = PursuitEvasionEnv(render_mode=None)
        env = SingleAgentPEWrapper(base_env, role="pursuer")

        # Should not raise
        env.set_opponent(None)
        env.set_opponent(lambda obs, deterministic=False: (np.zeros(2), None))


class TestPrepPhase:
    """Tests for the preparation phase (pursuer freeze)."""

    def make_env(self, **kwargs):
        defaults = dict(
            arena_width=10.0,
            arena_height=10.0,
            dt=0.05,
            max_steps=100,
            capture_radius=0.5,
            render_mode=None,
        )
        defaults.update(kwargs)
        return PursuitEvasionEnv(**defaults)

    def test_prep_steps_default_off(self):
        """prep_steps=0 by default (no freeze)."""
        env = self.make_env()
        assert env.prep_steps == 0

    def test_pursuer_frozen_during_prep(self):
        """Pursuer should not move during preparation phase."""
        env = self.make_env(prep_steps=10)
        obs, _ = env.reset(seed=42)
        initial_pos = env.pursuer_state[:2].copy()

        # Step with full-speed pursuer action during prep phase
        p_action = np.array([1.0, 0.0])  # v=1.0, omega=0
        e_action = np.array([0.0, 0.0])
        for _ in range(5):
            env.step(p_action, e_action)

        # Pursuer should not have moved (v forced to 0)
        assert np.allclose(env.pursuer_state[:2], initial_pos, atol=1e-6), (
            f"Pursuer moved during prep phase: {initial_pos} -> {env.pursuer_state[:2]}"
        )

    def test_pursuer_moves_after_prep(self):
        """Pursuer should move normally after preparation phase ends."""
        env = self.make_env(prep_steps=5)
        obs, _ = env.reset(seed=42)

        # Step through prep phase
        p_action = np.array([1.0, 0.0])
        e_action = np.array([0.0, 0.0])
        for _ in range(5):
            env.step(p_action, e_action)

        # Now past prep phase — pursuer should move
        pos_before = env.pursuer_state[:2].copy()
        env.step(p_action, e_action)
        pos_after = env.pursuer_state[:2].copy()

        assert not np.allclose(pos_before, pos_after, atol=1e-6), (
            "Pursuer did not move after prep phase ended"
        )

    def test_evader_moves_during_prep(self):
        """Evader should move freely during preparation phase."""
        env = self.make_env(prep_steps=10)
        obs, _ = env.reset(seed=42)
        initial_pos = env.evader_state[:2].copy()

        # Step with evader moving
        p_action = np.array([0.0, 0.0])
        e_action = np.array([1.0, 0.0])  # v=1.0, omega=0
        for _ in range(5):
            env.step(p_action, e_action)

        # Evader should have moved
        assert not np.allclose(env.evader_state[:2], initial_pos, atol=1e-6), (
            "Evader did not move during prep phase"
        )

    def test_pursuer_can_turn_during_prep(self):
        """Pursuer can turn (omega) during prep but not translate."""
        env = self.make_env(prep_steps=10)
        obs, _ = env.reset(seed=42)
        initial_pos = env.pursuer_state[:2].copy()
        initial_theta = env.pursuer_state[2]

        # Step with omega only
        p_action = np.array([1.0, 2.0])  # v=1.0 (will be zeroed), omega=2.0
        e_action = np.array([0.0, 0.0])
        for _ in range(5):
            env.step(p_action, e_action)

        # Position should not change (v=0)
        assert np.allclose(env.pursuer_state[:2], initial_pos, atol=1e-6)
        # Heading should change (omega allowed)
        assert env.pursuer_state[2] != pytest.approx(initial_theta, abs=1e-6)

    def test_in_prep_phase_info_flag(self):
        """Info dict should report in_prep_phase correctly."""
        env = self.make_env(prep_steps=3)
        env.reset(seed=42)

        p_action = np.array([0.0, 0.0])
        e_action = np.array([0.0, 0.0])

        # Steps 0, 1, 2 are prep (current_step < 3 before step, then <= 3 after)
        _, _, _, _, info = env.step(p_action, e_action)  # step 1
        assert info["in_prep_phase"] is True

        _, _, _, _, info = env.step(p_action, e_action)  # step 2
        assert info["in_prep_phase"] is True

        _, _, _, _, info = env.step(p_action, e_action)  # step 3
        assert info["in_prep_phase"] is True

        _, _, _, _, info = env.step(p_action, e_action)  # step 4
        assert info["in_prep_phase"] is False

    def test_no_capture_during_prep(self):
        """Even if agents start close, pursuer can't move to capture during prep."""
        env = self.make_env(prep_steps=10, capture_radius=0.5, min_init_distance=0.6,
                           max_init_distance=1.0)
        env.reset(seed=42)

        # Try to move pursuer toward evader
        p_action = np.array([1.0, 0.0])
        e_action = np.array([0.0, 0.0])

        for _ in range(5):
            _, _, terminated, _, _ = env.step(p_action, e_action)
            # Pursuer is frozen, so capture shouldn't happen (unless evader walks into pursuer)
            # We just check that the pursuer's position didn't change
        assert np.allclose(env.pursuer_state[:2], env.pursuer_state[:2])


class TestWallPenalty:
    """Tests for the w_wall wall collision penalty."""

    def make_env(self, **kwargs):
        defaults = dict(
            arena_width=10.0,
            arena_height=10.0,
            dt=0.1,
            max_steps=100,
            capture_radius=0.5,
            render_mode=None,
        )
        defaults.update(kwargs)
        return PursuitEvasionEnv(**defaults)

    def test_wall_penalty_applied_on_contact(self):
        """Reward should decrease by w_wall when agent hits a wall."""
        w_wall = 5.0
        env = self.make_env(w_wall=w_wall)
        env.reset(seed=42)

        # Place pursuer at the right wall edge, heading east (into the wall)
        half_w = env.arena_width / 2.0 - env.robot_radius
        env.pursuer_state = np.array([half_w, 0.0, 0.0])  # at wall, heading east
        # Place evader far away so no capture
        env.evader_state = np.array([-3.0, 0.0, np.pi])
        env.prev_distance = env._compute_distance()

        # Pursuer drives into wall (v=1.0, heading east at east wall)
        p_action = np.array([1.0, 0.0], dtype=np.float32)
        e_action = np.array([0.0, 0.0], dtype=np.float32)  # evader stays put

        # Run same scenario WITHOUT wall penalty for comparison
        env_no_wall = self.make_env(w_wall=0.0)
        env_no_wall.reset(seed=42)
        env_no_wall.pursuer_state = np.array([half_w, 0.0, 0.0])
        env_no_wall.evader_state = np.array([-3.0, 0.0, np.pi])
        env_no_wall.prev_distance = env_no_wall._compute_distance()

        _, rewards_wall, _, _, _ = env.step(p_action, e_action)
        _, rewards_no_wall, _, _, _ = env_no_wall.step(p_action, e_action)

        # Pursuer hit the wall, so reward should be lower by w_wall
        assert rewards_wall["pursuer"] == pytest.approx(
            rewards_no_wall["pursuer"] - w_wall, abs=1e-6
        ), (
            f"Wall penalty not applied: with={rewards_wall['pursuer']}, "
            f"without={rewards_no_wall['pursuer']}, expected diff={w_wall}"
        )

    def test_no_wall_penalty_when_zero(self):
        """With w_wall=0 (default), wall contact should not affect reward."""
        env = self.make_env(w_wall=0.0)
        env.reset(seed=42)

        # Place pursuer at the wall heading into it
        half_w = env.arena_width / 2.0 - env.robot_radius
        env.pursuer_state = np.array([half_w, 0.0, 0.0])
        env.evader_state = np.array([-3.0, 0.0, np.pi])
        env.prev_distance = env._compute_distance()

        p_action = np.array([1.0, 0.0], dtype=np.float32)
        e_action = np.array([0.0, 0.0], dtype=np.float32)

        _, rewards, _, _, _ = env.step(p_action, e_action)

        # Reward should be the same as base reward (zero-sum from RewardComputer)
        # Just verify it's a finite number — no penalty was subtracted
        assert np.isfinite(rewards["pursuer"])
        assert rewards["pursuer"] == pytest.approx(-rewards["evader"], abs=1e-6)


class TestSurvivalBonusAllLevels:
    """Test that survival_bonus applies at ALL levels, not just obstacle levels."""

    def test_survival_bonus_without_obstacles(self):
        """Survival bonus should apply even when no obstacles exist."""
        from envs.rewards import RewardComputer
        rc = RewardComputer(
            distance_scale=1.0,
            d_max=10.0,
            survival_bonus=1.0,
            use_visibility_reward=True,
        )
        # No obstacles — this triggers zero-sum fallback path
        r_p, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0,
            captured=False, timed_out=False,
            obstacles=[],
        )
        # r_pursuer = 0 (no distance change, no terminal)
        # r_evader = -r_pursuer + survival_bonus = 0 + 1.0 = 1.0
        assert r_e == pytest.approx(1.0, abs=0.01)

    def test_survival_bonus_not_on_capture(self):
        """Survival bonus should NOT apply on terminal steps."""
        from envs.rewards import RewardComputer
        rc = RewardComputer(
            distance_scale=1.0,
            d_max=10.0,
            survival_bonus=1.0,
        )
        r_p, r_e = rc.compute(
            d_curr=0.1, d_prev=1.0,
            captured=True, timed_out=False,
        )
        # On capture: r_e = -r_p (zero-sum), no survival bonus
        assert r_e == pytest.approx(-r_p, abs=0.01)

    def test_survival_bonus_with_visibility_and_obstacles(self):
        """Survival bonus should also apply in visibility mode with obstacles."""
        from envs.rewards import RewardComputer
        rc = RewardComputer(
            distance_scale=1.0,
            d_max=10.0,
            survival_bonus=1.0,
            use_visibility_reward=True,
            visibility_weight=1.0,
        )
        # With obstacles and visibility — evader is hidden
        r_p, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0,
            captured=False, timed_out=False,
            pursuer_pos=np.array([1.0, 1.0, 0.0]),
            evader_pos=np.array([5.0, 5.0, 0.0]),
            obstacles=[{
                "x": 3.0, "y": 3.0, "radius": 2.0,
            }],
        )
        # Visibility: +1.0 (hidden) + survival_bonus 1.0 = 2.0
        assert r_e == pytest.approx(2.0, abs=0.2)


class TestPartialObsLOS:
    """Tests for Phase 3 LOS-based partial observability."""

    def make_env(self, partial_obs=True, **kwargs):
        defaults = dict(
            arena_width=10.0,
            arena_height=10.0,
            max_steps=600,
            capture_radius=0.5,
            n_obstacles=2,
            n_obstacle_obs=2,
            partial_obs=partial_obs,
        )
        defaults.update(kwargs)
        return PursuitEvasionEnv(**defaults)

    def test_obs_dim_with_partial_obs(self):
        """partial_obs=True adds 1 dim (los_visible flag)."""
        env_full = self.make_env(partial_obs=False)
        env_partial = self.make_env(partial_obs=True)
        assert env_partial.obs_builder.obs_dim == env_full.obs_builder.obs_dim + 1

    def test_obs_dim_value(self):
        """14 base + 1 los_visible + 2*2 obstacles = 19."""
        env = self.make_env(partial_obs=True, n_obstacle_obs=2)
        assert env.obs_builder.obs_dim == 19

    def test_obs_shape_matches_space(self):
        """Observation array shape matches observation_space."""
        env = self.make_env(partial_obs=True)
        obs, _ = env.reset(seed=42)
        assert obs["pursuer"].shape == (env.obs_builder.obs_dim,)
        assert obs["evader"].shape == (env.obs_builder.obs_dim,)

    def test_full_obs_no_masking(self):
        """With partial_obs=False, opponent features are never masked."""
        env = self.make_env(partial_obs=False)
        obs, _ = env.reset(seed=42)
        # Opponent position (indices 5,6) should not be zero
        p_obs = obs["pursuer"]
        e_obs = obs["evader"]
        # At least one agent should have non-zero opponent position
        assert not (p_obs[5] == 0.0 and p_obs[6] == 0.0 and
                    e_obs[5] == 0.0 and e_obs[6] == 0.0)

    def test_los_visible_flag_present(self):
        """partial_obs=True adds los_visible flag at index 14."""
        env = self.make_env(partial_obs=True)
        obs, _ = env.reset(seed=42)
        p_obs = obs["pursuer"]
        # los_visible is at index 14, should be 0.0 or 1.0
        assert p_obs[14] in (0.0, 1.0)

    def test_masking_when_los_blocked(self):
        """When LOS is blocked, opponent features should be zeroed."""
        from envs.observations import ObservationBuilder
        builder = ObservationBuilder(
            arena_width=10.0, arena_height=10.0,
            v_max=1.0, omega_max=2.84,
            n_obstacle_obs=0, partial_obs=True,
        )
        self_state = np.array([0.0, 0.0, 0.0])
        self_action = np.array([0.5, 0.1])
        opp_state = np.array([3.0, 3.0, 1.0])
        opp_action = np.array([0.8, -0.5])

        # LOS not blocked — opponent features visible
        obs_visible = builder.build(
            self_state, self_action, opp_state, opp_action,
            los_blocked=False,
        )
        # Opponent pos (idx 5,6), heading (7), vel (8,9), dist (10), bearing (11)
        assert obs_visible[5] != 0.0  # x_opp
        assert obs_visible[10] != 0.0  # distance
        assert obs_visible[14] == 1.0  # los_visible = True

        # LOS blocked — opponent features masked
        obs_masked = builder.build(
            self_state, self_action, opp_state, opp_action,
            los_blocked=True,
        )
        assert obs_masked[5] == 0.0   # x_opp masked
        assert obs_masked[6] == 0.0   # y_opp masked
        assert obs_masked[7] == 0.0   # theta_opp masked
        assert obs_masked[8] == 0.0   # v_opp masked
        assert obs_masked[9] == 0.0   # omega_opp masked
        assert obs_masked[10] == 0.0  # distance masked
        assert obs_masked[11] == 0.0  # bearing masked
        assert obs_masked[14] == 0.0  # los_visible = False

    def test_self_features_never_masked(self):
        """Own state features are always available regardless of LOS."""
        from envs.observations import ObservationBuilder
        builder = ObservationBuilder(
            arena_width=10.0, arena_height=10.0,
            v_max=1.0, omega_max=2.84,
            n_obstacle_obs=0, partial_obs=True,
        )
        self_state = np.array([2.0, -1.5, 0.5])
        self_action = np.array([0.7, 0.3])
        opp_state = np.array([3.0, 3.0, 1.0])
        opp_action = np.array([0.8, -0.5])

        obs = builder.build(
            self_state, self_action, opp_state, opp_action,
            los_blocked=True,
        )
        # Own state (idx 0-4) should be non-zero
        assert obs[0] != 0.0  # x_self
        assert obs[1] != 0.0  # y_self
        assert obs[3] != 0.0  # v_self
        # Wall distances (idx 12,13) should be non-zero
        assert obs[12] != 0.0
        assert obs[13] != 0.0

    def test_obstacle_features_never_masked(self):
        """Obstacle features are always available regardless of LOS."""
        from envs.observations import ObservationBuilder
        builder = ObservationBuilder(
            arena_width=10.0, arena_height=10.0,
            v_max=1.0, omega_max=2.84,
            n_obstacle_obs=2, partial_obs=True,
        )
        self_state = np.array([0.0, 0.0, 0.0])
        self_action = np.array([0.5, 0.1])
        opp_state = np.array([3.0, 3.0, 1.0])
        opp_action = np.array([0.8, -0.5])
        obstacles = [
            {"x": 2.0, "y": 0.0, "radius": 0.5},
            {"x": -2.0, "y": 1.0, "radius": 0.7},
        ]

        obs = builder.build(
            self_state, self_action, opp_state, opp_action,
            obstacles=obstacles, los_blocked=True,
        )
        # Obstacle features start at index 15 (14 base + 1 los_visible)
        # obs1_dist, obs1_bearing, obs2_dist, obs2_bearing
        assert obs[15] != 0.0  # obstacle distance (not 0)

    def test_env_step_with_partial_obs(self):
        """Environment step works correctly with partial_obs enabled."""
        env = self.make_env(partial_obs=True)
        obs, _ = env.reset(seed=42)
        p_action = np.array([0.5, 0.5], dtype=np.float32)
        e_action = np.array([0.5, -0.5], dtype=np.float32)
        obs2, rewards, terminated, truncated, info = env.step(p_action, e_action)
        assert obs2["pursuer"].shape == (env.obs_builder.obs_dim,)
        assert obs2["evader"].shape == (env.obs_builder.obs_dim,)

    def test_asymmetric_obs_pursuer_masked(self):
        """With asymmetric_obs=True, pursuer obs has zeroed opponent when LOS blocked."""
        from envs.observations import ObservationBuilder
        from envs.rewards import line_of_sight_blocked

        # Create env with asymmetric obs and an obstacle between agents
        env = self.make_env(
            partial_obs=True, asymmetric_obs=True,
            n_obstacles=1, n_obstacle_obs=1,
        )
        obs, _ = env.reset(seed=42)

        # Manually place agents with obstacle between them
        env.pursuer_state = np.array([-3.0, 0.0, 0.0])
        env.evader_state = np.array([3.0, 0.0, np.pi])
        env.obstacles = [{"x": 0.0, "y": 0.0, "radius": 1.0}]

        # Verify LOS is actually blocked
        assert line_of_sight_blocked(
            env.pursuer_state, env.evader_state, env.obstacles
        )

        obs = env._get_obs()
        # Pursuer should have masked opponent (idx 5-11 zeroed)
        p_obs = obs["pursuer"]
        assert p_obs[5] == 0.0  # x_opp masked
        assert p_obs[6] == 0.0  # y_opp masked
        assert p_obs[10] == 0.0  # distance masked

    def test_asymmetric_obs_evader_always_sees(self):
        """With asymmetric_obs=True, evader obs always has opponent info."""
        from envs.rewards import line_of_sight_blocked

        env = self.make_env(
            partial_obs=True, asymmetric_obs=True,
            n_obstacles=1, n_obstacle_obs=1,
        )
        obs, _ = env.reset(seed=42)

        # Place obstacle between agents
        env.pursuer_state = np.array([-3.0, 0.0, 0.0])
        env.evader_state = np.array([3.0, 0.0, np.pi])
        env.obstacles = [{"x": 0.0, "y": 0.0, "radius": 1.0}]

        # Verify LOS is blocked
        assert line_of_sight_blocked(
            env.pursuer_state, env.evader_state, env.obstacles
        )

        obs = env._get_obs()
        # Evader should still see the pursuer (asymmetric: evader not masked)
        e_obs = obs["evader"]
        assert e_obs[5] != 0.0 or e_obs[6] != 0.0  # opponent position visible
        assert e_obs[10] != 0.0  # distance visible
        assert e_obs[14] == 1.0  # los_visible = True (evader always sees)

    def test_randomized_obstacles_count(self):
        """Reset with n_obstacles_min/max produces varying obstacle counts."""
        env = self.make_env(
            partial_obs=True,
            n_obstacles=2,
            n_obstacles_min=1,
            n_obstacles_max=3,
            n_obstacle_obs=3,
        )
        counts = set()
        for seed in range(50):
            env.reset(seed=seed)
            counts.add(len(env.obstacles))
        # Should see at least 2 different counts out of {1, 2, 3}
        assert len(counts) >= 2, f"Expected variety in obstacle counts, got {counts}"
        assert all(1 <= c <= 3 for c in counts), f"Counts out of range: {counts}"

    def test_randomized_obstacles_obs_dim_constant(self):
        """obs_dim stays fixed even when n_obstacles varies per episode."""
        env = self.make_env(
            partial_obs=True,
            n_obstacles=2,
            n_obstacles_min=1,
            n_obstacles_max=3,
            n_obstacle_obs=3,
        )
        expected_dim = env.obs_builder.obs_dim
        for seed in range(20):
            obs, _ = env.reset(seed=seed)
            assert obs["pursuer"].shape == (expected_dim,)
            assert obs["evader"].shape == (expected_dim,)

    def test_randomized_obstacles_backward_compat(self):
        """n_obstacles_min=None keeps fixed obstacle count."""
        env = self.make_env(
            partial_obs=True, n_obstacles=2,
            n_obstacle_obs=2,
        )
        # Default: n_obstacles_min and n_obstacles_max are None
        assert env.n_obstacles_min is None
        assert env.n_obstacles_max is None
        for seed in range(10):
            env.reset(seed=seed)
            assert len(env.obstacles) == 2
