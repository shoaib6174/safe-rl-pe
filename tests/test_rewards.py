"""Unit tests for the reward computation."""

import numpy as np
import pytest

from envs.rewards import RewardComputer, line_of_sight_blocked, nearest_obstacle_distance


class TestRewardComputer:
    """Tests for the reward function."""

    def make_reward(self, **kwargs):
        defaults = dict(
            distance_scale=1.0,
            capture_bonus=100.0,
            timeout_penalty=-100.0,
            d_max=28.28,
        )
        defaults.update(kwargs)
        return RewardComputer(**defaults)

    def test_zero_sum(self):
        """r_P + r_E should always be 0."""
        rc = self.make_reward()

        # Normal step
        r_p, r_e = rc.compute(d_curr=8.0, d_prev=9.0, captured=False, timed_out=False)
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

        # Capture
        r_p, r_e = rc.compute(d_curr=0.3, d_prev=0.6, captured=True, timed_out=False)
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

        # Timeout
        r_p, r_e = rc.compute(d_curr=10.0, d_prev=10.0, captured=False, timed_out=True)
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

    def test_closing_positive(self):
        """Pursuer gets positive reward when closing distance."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(d_curr=8.0, d_prev=9.0, captured=False, timed_out=False)
        assert r_p > 0
        assert r_e < 0

    def test_retreating_negative(self):
        """Pursuer gets negative reward when distance increases."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(d_curr=10.0, d_prev=9.0, captured=False, timed_out=False)
        assert r_p < 0
        assert r_e > 0

    def test_no_change_zero(self):
        """No distance change, no terminal -> zero reward."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(d_curr=5.0, d_prev=5.0, captured=False, timed_out=False)
        assert r_p == pytest.approx(0.0, abs=1e-10)
        assert r_e == pytest.approx(0.0, abs=1e-10)

    def test_capture_bonus(self):
        """Capture should add large positive reward for pursuer."""
        rc = self.make_reward(capture_bonus=100.0)
        r_p, r_e = rc.compute(d_curr=0.3, d_prev=0.5, captured=True, timed_out=False)
        assert r_p > 90  # capture bonus dominates
        assert r_e < -90

    def test_timeout_penalty(self):
        """Timeout should penalize pursuer."""
        rc = self.make_reward(timeout_penalty=-50.0)
        r_p, r_e = rc.compute(d_curr=10.0, d_prev=10.0, captured=False, timed_out=True)
        assert r_p < 0
        assert r_e > 0

    def test_distance_scale(self):
        """Distance scale should proportionally affect shaping reward."""
        rc1 = self.make_reward(distance_scale=1.0)
        rc2 = self.make_reward(distance_scale=2.0)

        r1, _ = rc1.compute(d_curr=8.0, d_prev=9.0, captured=False, timed_out=False)
        r2, _ = rc2.compute(d_curr=8.0, d_prev=9.0, captured=False, timed_out=False)

        assert r2 == pytest.approx(2 * r1, abs=1e-10)


class TestLineOfSightBlocked:
    """Tests for the line-of-sight occlusion check."""

    def test_no_obstacles_not_blocked(self):
        p = np.array([0.0, 0.0])
        e = np.array([5.0, 0.0])
        assert not line_of_sight_blocked(p, e, [])

    def test_obstacle_on_line_blocks(self):
        """Obstacle directly between agents should block LOS."""
        p = np.array([0.0, 0.0])
        e = np.array([6.0, 0.0])
        obs = [{"x": 3.0, "y": 0.0, "radius": 0.5}]
        assert line_of_sight_blocked(p, e, obs)

    def test_obstacle_off_line_no_block(self):
        """Obstacle far from the line should not block."""
        p = np.array([0.0, 0.0])
        e = np.array([6.0, 0.0])
        obs = [{"x": 3.0, "y": 5.0, "radius": 0.5}]
        assert not line_of_sight_blocked(p, e, obs)

    def test_obstacle_just_touching_line(self):
        """Obstacle whose edge touches the line should block."""
        p = np.array([0.0, 0.0])
        e = np.array([6.0, 0.0])
        # Obstacle at y=0.5 with radius 0.5 — edge touches y=0
        obs = [{"x": 3.0, "y": 0.5, "radius": 0.501}]
        assert line_of_sight_blocked(p, e, obs)

    def test_obstacle_just_missing_line(self):
        """Obstacle whose edge barely misses the line should not block."""
        p = np.array([0.0, 0.0])
        e = np.array([6.0, 0.0])
        obs = [{"x": 3.0, "y": 0.5, "radius": 0.49}]
        assert not line_of_sight_blocked(p, e, obs)

    def test_obstacle_behind_pursuer(self):
        """Obstacle behind the pursuer (not between agents) should not block."""
        p = np.array([3.0, 0.0])
        e = np.array([6.0, 0.0])
        obs = [{"x": 1.0, "y": 0.0, "radius": 0.5}]
        assert not line_of_sight_blocked(p, e, obs)

    def test_obstacle_beyond_evader(self):
        """Obstacle beyond the evader should not block."""
        p = np.array([0.0, 0.0])
        e = np.array([3.0, 0.0])
        obs = [{"x": 5.0, "y": 0.0, "radius": 0.5}]
        assert not line_of_sight_blocked(p, e, obs)

    def test_diagonal_line(self):
        """Test with diagonal line between agents."""
        p = np.array([0.0, 0.0])
        e = np.array([4.0, 4.0])
        obs = [{"x": 2.0, "y": 2.0, "radius": 0.3}]
        assert line_of_sight_blocked(p, e, obs)

    def test_multiple_obstacles_one_blocks(self):
        """Only one obstacle needs to block for LOS to be blocked."""
        p = np.array([0.0, 0.0])
        e = np.array([6.0, 0.0])
        obs = [
            {"x": 3.0, "y": 5.0, "radius": 0.5},  # far off
            {"x": 3.0, "y": 0.0, "radius": 0.5},  # blocks
        ]
        assert line_of_sight_blocked(p, e, obs)

    def test_same_position_not_blocked(self):
        """Same position should return False (no segment)."""
        p = np.array([3.0, 3.0])
        obs = [{"x": 3.0, "y": 3.0, "radius": 1.0}]
        assert not line_of_sight_blocked(p, p, obs)

    def test_works_with_3d_state(self):
        """Should work with [x, y, theta] state vectors."""
        p = np.array([0.0, 0.0, 1.57])
        e = np.array([6.0, 0.0, 0.0])
        obs = [{"x": 3.0, "y": 0.0, "radius": 0.5}]
        assert line_of_sight_blocked(p, e, obs)


class TestOcclusionReward:
    """Tests for the occlusion bonus in RewardComputer."""

    def make_reward(self, **kwargs):
        defaults = dict(
            distance_scale=1.0,
            capture_bonus=100.0,
            timeout_penalty=-100.0,
            d_max=28.28,
            w_occlusion=0.0,
        )
        defaults.update(kwargs)
        return RewardComputer(**defaults)

    def test_occlusion_disabled_by_default(self):
        """w_occlusion=0 should produce strict zero-sum."""
        rc = self.make_reward(w_occlusion=0.0)
        r_p, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[{"x": 3.0, "y": 0.0, "radius": 0.5}],
        )
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

    def test_occlusion_bonus_when_hidden(self):
        """Evader behind obstacle should get bonus."""
        rc = self.make_reward(w_occlusion=0.05)
        r_p, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[{"x": 3.0, "y": 0.0, "radius": 0.5}],
        )
        # Pursuer reward unchanged (zero distance change)
        assert r_p == pytest.approx(0.0, abs=1e-10)
        # Evader gets occlusion bonus on top of zero-sum base
        assert r_e == pytest.approx(0.05, abs=1e-10)
        # No longer zero-sum
        assert r_p + r_e != pytest.approx(0.0, abs=1e-3)

    def test_no_bonus_when_exposed(self):
        """Evader not behind obstacle should get no bonus."""
        rc = self.make_reward(w_occlusion=0.05)
        r_p, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[{"x": 3.0, "y": 5.0, "radius": 0.5}],  # far off
        )
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

    def test_no_bonus_on_capture(self):
        """No occlusion bonus when captured."""
        rc = self.make_reward(w_occlusion=0.05)
        r_p, r_e = rc.compute(
            d_curr=0.3, d_prev=0.5, captured=True, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[{"x": 3.0, "y": 0.0, "radius": 0.5}],
        )
        # Capture bonus dominates, no occlusion bonus on capture step
        assert r_p > 90

    def test_no_bonus_without_obstacles(self):
        """No obstacles means no occlusion bonus even with w_occlusion > 0."""
        rc = self.make_reward(w_occlusion=0.05)
        r_p, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[],
        )
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

    def test_no_bonus_without_positions(self):
        """Missing positions should gracefully produce no bonus."""
        rc = self.make_reward(w_occlusion=0.05)
        r_p, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
        )
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

    def test_backward_compatible_no_kwargs(self):
        """Old-style calls without position kwargs should still work."""
        rc = self.make_reward(w_occlusion=0.05)
        r_p, r_e = rc.compute(d_curr=8.0, d_prev=9.0, captured=False, timed_out=False)
        assert r_p > 0  # closing distance
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)  # no occlusion without positions


class TestVisibilityReward:
    """Tests for visibility-based evader reward (Mode B, OpenAI H&S style)."""

    def make_reward(self, **kwargs):
        defaults = dict(
            distance_scale=1.0,
            capture_bonus=100.0,
            timeout_penalty=-100.0,
            d_max=28.28,
            use_visibility_reward=True,
            visibility_weight=1.0,
            survival_bonus=1.0,
        )
        defaults.update(kwargs)
        return RewardComputer(**defaults)

    def test_evader_positive_when_hidden(self):
        """Hidden evader should get positive reward (visibility + survival)."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[{"x": 3.0, "y": 0.0, "radius": 0.5}],
        )
        # Evader: +1.0 (hidden) + 1.0 (survival) = +2.0
        assert r_e == pytest.approx(2.0, abs=1e-10)

    def test_evader_negative_when_visible(self):
        """Visible evader should get negative visibility but positive survival."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[{"x": 3.0, "y": 5.0, "radius": 0.5}],  # far off
        )
        # Evader: -1.0 (visible) + 1.0 (survival) = 0.0
        assert r_e == pytest.approx(0.0, abs=1e-10)

    def test_pursuer_always_distance_based(self):
        """Pursuer should always get distance-based reward regardless of mode."""
        rc = self.make_reward()
        # Closing distance
        r_p, _ = rc.compute(
            d_curr=8.0, d_prev=9.0, captured=False, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[{"x": 3.0, "y": 0.0, "radius": 0.5}],
        )
        assert r_p > 0  # pursuer rewarded for closing distance

    def test_not_zero_sum(self):
        """Visibility mode should NOT be zero-sum during normal steps."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[{"x": 3.0, "y": 0.0, "radius": 0.5}],
        )
        # Pursuer: 0.0 (no distance change). Evader: +2.0 (hidden + survival)
        assert r_p + r_e != pytest.approx(0.0, abs=0.1)

    def test_fallback_zero_sum_no_obstacles(self):
        """Without obstacles, should fall back to zero-sum distance-based."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(
            d_curr=8.0, d_prev=9.0, captured=False, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[],  # no obstacles
        )
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

    def test_fallback_zero_sum_no_positions(self):
        """Without positions, should fall back to zero-sum."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(
            d_curr=8.0, d_prev=9.0, captured=False, timed_out=False,
        )
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

    def test_capture_zero_sum(self):
        """Capture step should be zero-sum even in visibility mode."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(
            d_curr=0.3, d_prev=0.5, captured=True, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[{"x": 3.0, "y": 0.0, "radius": 0.5}],
        )
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)
        assert r_p > 90  # capture bonus dominates

    def test_timeout_zero_sum(self):
        """Timeout step should be zero-sum even in visibility mode."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(
            d_curr=10.0, d_prev=10.0, captured=False, timed_out=True,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[{"x": 3.0, "y": 0.0, "radius": 0.5}],
        )
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)
        assert r_p < 0  # timeout penalty for pursuer
        assert r_e > 0  # timeout is good for evader (survived)

    def test_visibility_weight_scales(self):
        """Visibility weight should scale the +1/-1 signal."""
        rc = self.make_reward(visibility_weight=2.0, survival_bonus=0.0)
        r_p, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[{"x": 3.0, "y": 0.0, "radius": 0.5}],
        )
        assert r_e == pytest.approx(2.0, abs=1e-10)  # 2.0 * 1.0

    def test_survival_only_no_visibility(self):
        """Survival bonus without visibility weight."""
        rc = self.make_reward(visibility_weight=0.0, survival_bonus=0.5)
        r_p, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[{"x": 3.0, "y": 0.0, "radius": 0.5}],
        )
        assert r_e == pytest.approx(0.5, abs=1e-10)  # survival only

    def test_accumulation_advantage(self):
        """Over many steps, hidden evader accumulates much more reward."""
        rc = self.make_reward(visibility_weight=1.0, survival_bonus=1.0)
        total_hidden = 0.0
        total_visible = 0.0
        for _ in range(100):
            _, r_hidden = rc.compute(
                d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
                pursuer_pos=np.array([0.0, 0.0, 0.0]),
                evader_pos=np.array([6.0, 0.0, 0.0]),
                obstacles=[{"x": 3.0, "y": 0.0, "radius": 0.5}],
            )
            total_hidden += r_hidden
            _, r_visible = rc.compute(
                d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
                pursuer_pos=np.array([0.0, 0.0, 0.0]),
                evader_pos=np.array([6.0, 0.0, 0.0]),
                obstacles=[{"x": 3.0, "y": 5.0, "radius": 0.5}],
            )
            total_visible += r_visible
        # Hidden: 100 * (+1 + 1) = 200
        # Visible: 100 * (-1 + 1) = 0
        assert total_hidden == pytest.approx(200.0, abs=1e-6)
        assert total_visible == pytest.approx(0.0, abs=1e-6)
        assert total_hidden > total_visible


class TestNearestObstacleDistance:
    """Tests for the nearest_obstacle_distance helper."""

    def test_no_obstacles_returns_inf(self):
        pos = np.array([0.0, 0.0])
        assert nearest_obstacle_distance(pos, []) == float("inf")

    def test_single_obstacle_surface_distance(self):
        """Surface distance = center distance - radius."""
        pos = np.array([0.0, 0.0])
        obs = [{"x": 3.0, "y": 0.0, "radius": 1.0}]
        # center dist = 3.0, surface dist = 3.0 - 1.0 = 2.0
        assert nearest_obstacle_distance(pos, obs) == pytest.approx(2.0, abs=1e-6)

    def test_multiple_obstacles_returns_nearest(self):
        pos = np.array([0.0, 0.0])
        obs = [
            {"x": 5.0, "y": 0.0, "radius": 1.0},  # surface dist = 4.0
            {"x": 2.0, "y": 0.0, "radius": 0.5},   # surface dist = 1.5
            {"x": 8.0, "y": 0.0, "radius": 2.0},   # surface dist = 6.0
        ]
        assert nearest_obstacle_distance(pos, obs) == pytest.approx(1.5, abs=1e-6)

    def test_inside_obstacle_clamped_to_zero(self):
        """Agent inside obstacle should get distance 0, not negative."""
        pos = np.array([3.0, 0.0])
        obs = [{"x": 3.0, "y": 0.0, "radius": 1.0}]
        assert nearest_obstacle_distance(pos, obs) == pytest.approx(0.0, abs=1e-6)

    def test_on_obstacle_surface_zero(self):
        """Agent exactly on obstacle surface = distance 0."""
        pos = np.array([2.0, 0.0])
        obs = [{"x": 3.0, "y": 0.0, "radius": 1.0}]
        assert nearest_obstacle_distance(pos, obs) == pytest.approx(0.0, abs=1e-6)

    def test_works_with_3d_state(self):
        """Should work with [x, y, theta] state vectors (ignores theta)."""
        pos = np.array([0.0, 0.0, 1.57])
        obs = [{"x": 3.0, "y": 0.0, "radius": 1.0}]
        assert nearest_obstacle_distance(pos, obs) == pytest.approx(2.0, abs=1e-6)

    def test_diagonal_distance(self):
        pos = np.array([0.0, 0.0])
        obs = [{"x": 3.0, "y": 4.0, "radius": 0.5}]
        # center dist = sqrt(9+16) = 5.0, surface dist = 4.5
        assert nearest_obstacle_distance(pos, obs) == pytest.approx(4.5, abs=1e-6)


class TestPBRSObstacleSeeking:
    """Tests for PBRS obstacle-seeking reward shaping."""

    def make_reward(self, **kwargs):
        defaults = dict(
            distance_scale=1.0,
            capture_bonus=100.0,
            timeout_penalty=-100.0,
            d_max=14.14,  # diagonal of 10x10 arena
            w_obs_approach=10.0,
        )
        defaults.update(kwargs)
        return RewardComputer(**defaults)

    def test_positive_reward_when_approaching(self):
        """Evader moving closer to obstacle should get positive PBRS."""
        rc = self.make_reward()
        _, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            d_obs_curr=2.0, d_obs_prev=3.0,  # moved 1m closer
        )
        # PBRS = 10.0 * (3.0 - 2.0) / 14.14 ≈ 0.707
        # Zero-sum base = 0.0, so evader gets PBRS only
        expected_pbrs = 10.0 * (3.0 - 2.0) / 14.14
        assert r_e == pytest.approx(expected_pbrs, abs=1e-3)

    def test_negative_reward_when_retreating(self):
        """Evader moving away from obstacle should get negative PBRS."""
        rc = self.make_reward()
        _, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            d_obs_curr=4.0, d_obs_prev=3.0,  # moved 1m away
        )
        expected_pbrs = 10.0 * (3.0 - 4.0) / 14.14
        assert r_e == pytest.approx(expected_pbrs, abs=1e-3)
        assert r_e < 0

    def test_zero_reward_when_stationary(self):
        """No change in obstacle distance = no PBRS."""
        rc = self.make_reward()
        _, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            d_obs_curr=3.0, d_obs_prev=3.0,
        )
        assert r_e == pytest.approx(0.0, abs=1e-10)

    def test_disabled_when_weight_zero(self):
        """w_obs_approach=0 should produce no PBRS term."""
        rc = self.make_reward(w_obs_approach=0.0)
        _, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            d_obs_curr=2.0, d_obs_prev=3.0,
        )
        assert r_e == pytest.approx(0.0, abs=1e-10)  # zero-sum base only

    def test_no_pbrs_on_capture(self):
        """PBRS should not apply on capture step."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(
            d_curr=0.3, d_prev=0.5, captured=True, timed_out=False,
            d_obs_curr=1.0, d_obs_prev=2.0,
        )
        # Capture is zero-sum, no PBRS added
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

    def test_no_pbrs_on_timeout(self):
        """PBRS should not apply on timeout step."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(
            d_curr=10.0, d_prev=10.0, captured=False, timed_out=True,
            d_obs_curr=1.0, d_obs_prev=2.0,
        )
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

    def test_no_pbrs_without_obs_distances(self):
        """Missing d_obs_curr/d_obs_prev should gracefully skip PBRS."""
        rc = self.make_reward()
        _, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
        )
        assert r_e == pytest.approx(0.0, abs=1e-10)

    def test_no_pbrs_when_no_obstacles_inf(self):
        """When d_obs is inf (no obstacles), PBRS should be skipped (no NaN)."""
        rc = self.make_reward()
        _, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            d_obs_curr=float("inf"), d_obs_prev=float("inf"),
        )
        assert np.isfinite(r_e), f"Got non-finite reward: {r_e}"
        assert r_e == pytest.approx(0.0, abs=1e-10)

    def test_pbrs_combines_with_visibility(self):
        """PBRS should add on top of visibility reward."""
        rc = self.make_reward(
            use_visibility_reward=True,
            visibility_weight=1.0,
            survival_bonus=1.0,
        )
        _, r_e = rc.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            pursuer_pos=np.array([0.0, 0.0, 0.0]),
            evader_pos=np.array([6.0, 0.0, 0.0]),
            obstacles=[{"x": 3.0, "y": 0.0, "radius": 0.5}],
            d_obs_curr=2.0, d_obs_prev=3.0,
        )
        # Visibility: +1.0 (hidden) + 1.0 (survival) = +2.0
        # PBRS: 10.0 * (3.0 - 2.0) / 14.14 ≈ 0.707
        expected = 2.0 + 10.0 * 1.0 / 14.14
        assert r_e == pytest.approx(expected, abs=1e-3)

    def test_pbrs_weight_scales_linearly(self):
        """Doubling weight should double PBRS term."""
        rc1 = self.make_reward(w_obs_approach=10.0)
        rc2 = self.make_reward(w_obs_approach=20.0)

        _, r_e1 = rc1.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            d_obs_curr=2.0, d_obs_prev=3.0,
        )
        _, r_e2 = rc2.compute(
            d_curr=5.0, d_prev=5.0, captured=False, timed_out=False,
            d_obs_curr=2.0, d_obs_prev=3.0,
        )
        assert r_e2 == pytest.approx(2 * r_e1, abs=1e-10)

    def test_pursuer_reward_unaffected(self):
        """PBRS should NOT affect pursuer reward."""
        rc = self.make_reward()
        rc_no_pbrs = self.make_reward(w_obs_approach=0.0)

        r_p_pbrs, _ = rc.compute(
            d_curr=8.0, d_prev=9.0, captured=False, timed_out=False,
            d_obs_curr=2.0, d_obs_prev=3.0,
        )
        r_p_base, _ = rc_no_pbrs.compute(
            d_curr=8.0, d_prev=9.0, captured=False, timed_out=False,
            d_obs_curr=2.0, d_obs_prev=3.0,
        )
        assert r_p_pbrs == pytest.approx(r_p_base, abs=1e-10)
