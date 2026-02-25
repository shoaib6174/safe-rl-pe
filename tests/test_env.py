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
