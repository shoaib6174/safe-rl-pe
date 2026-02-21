"""Unit tests for unicycle dynamics."""

import numpy as np
import pytest

from envs.dynamics import clip_action, unicycle_step, wrap_angle


class TestWrapAngle:
    """Tests for angle normalization."""

    def test_already_in_range(self):
        assert wrap_angle(0.0) == pytest.approx(0.0)
        assert wrap_angle(1.0) == pytest.approx(1.0)
        assert wrap_angle(-1.0) == pytest.approx(-1.0)

    def test_positive_overflow(self):
        # 2*pi should wrap to ~0
        assert wrap_angle(2 * np.pi) == pytest.approx(0.0, abs=1e-10)

    def test_negative_overflow(self):
        # -2*pi should wrap to ~0
        assert wrap_angle(-2 * np.pi) == pytest.approx(0.0, abs=1e-10)

    def test_pi_boundary(self):
        # pi should remain pi (or -pi, they're equivalent)
        result = wrap_angle(np.pi)
        assert abs(result) == pytest.approx(np.pi, abs=1e-10)

    def test_large_positive(self):
        result = wrap_angle(3 * np.pi)
        assert -np.pi <= result <= np.pi
        assert result == pytest.approx(-np.pi, abs=1e-10)

    def test_large_negative(self):
        result = wrap_angle(-3 * np.pi)
        assert -np.pi <= result <= np.pi
        assert result == pytest.approx(-np.pi, abs=1e-10)


class TestUnicycleStep:
    """Tests for unicycle dynamics integration."""

    ARENA_W = 20.0
    ARENA_H = 20.0
    ROBOT_R = 0.15

    def test_straight_forward(self):
        """Robot at origin heading east, v=1.0 for dt=1.0 should move 1.0m east."""
        x, y, theta, wall = unicycle_step(
            0.0, 0.0, 0.0,  # state: origin, heading east
            1.0, 0.0,        # action: v=1.0, omega=0
            1.0,              # dt=1.0s
            self.ARENA_W, self.ARENA_H, self.ROBOT_R,
        )
        assert x == pytest.approx(1.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)
        assert theta == pytest.approx(0.0, abs=1e-10)
        assert wall is False

    def test_pure_rotation(self):
        """omega=pi/2 for dt=1.0 should rotate 90 degrees."""
        x, y, theta, wall = unicycle_step(
            0.0, 0.0, 0.0,
            0.0, np.pi / 2,  # v=0, omega=pi/2
            1.0,
            self.ARENA_W, self.ARENA_H, self.ROBOT_R,
        )
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)
        assert theta == pytest.approx(np.pi / 2, abs=1e-10)
        assert wall is False

    def test_angle_wrapping(self):
        """Large rotation should wrap angle to [-pi, pi]."""
        # Start at theta=3.0, rotate by 1.0 -> 4.0, should wrap
        x, y, theta, wall = unicycle_step(
            0.0, 0.0, 3.0,
            0.0, 1.0,
            1.0,
            self.ARENA_W, self.ARENA_H, self.ROBOT_R,
        )
        assert -np.pi <= theta <= np.pi

    def test_wall_clipping_east(self):
        """Robot near east wall heading east should be clipped."""
        x_start = 9.9  # Near east boundary (arena half width = 10.0)
        x, y, theta, wall = unicycle_step(
            x_start, 0.0, 0.0,  # heading east
            1.0, 0.0,            # v=1.0
            0.05,                 # dt=0.05
            self.ARENA_W, self.ARENA_H, self.ROBOT_R,
        )
        x_max = self.ARENA_W / 2 - self.ROBOT_R
        assert x == pytest.approx(x_max)
        assert wall is True

    def test_wall_clipping_west(self):
        """Robot near west wall heading west should be clipped."""
        x, y, theta, wall = unicycle_step(
            -9.9, 0.0, np.pi,  # heading west
            1.0, 0.0,
            0.05,
            self.ARENA_W, self.ARENA_H, self.ROBOT_R,
        )
        x_min = -self.ARENA_W / 2 + self.ROBOT_R
        assert x == pytest.approx(x_min)
        assert wall is True

    def test_wall_clipping_north(self):
        """Robot near north wall heading north should be clipped."""
        x, y, theta, wall = unicycle_step(
            0.0, 9.9, np.pi / 2,  # heading north
            1.0, 0.0,
            0.05,
            self.ARENA_W, self.ARENA_H, self.ROBOT_R,
        )
        y_max = self.ARENA_H / 2 - self.ROBOT_R
        assert y == pytest.approx(y_max)
        assert wall is True

    def test_no_wall_contact_interior(self):
        """Robot in the interior should not trigger wall contact."""
        x, y, theta, wall = unicycle_step(
            0.0, 0.0, 0.0,
            1.0, 0.0,
            0.05,
            self.ARENA_W, self.ARENA_H, self.ROBOT_R,
        )
        assert wall is False

    def test_diagonal_movement(self):
        """Robot heading northeast at 45 degrees."""
        theta = np.pi / 4
        v = 1.0
        dt = 1.0
        x, y, _, _ = unicycle_step(
            0.0, 0.0, theta,
            v, 0.0,
            dt,
            self.ARENA_W, self.ARENA_H, self.ROBOT_R,
        )
        expected = v * np.cos(theta) * dt
        assert x == pytest.approx(expected, abs=1e-10)
        assert y == pytest.approx(expected, abs=1e-10)

    def test_small_timestep(self):
        """Verify Euler integration with small dt."""
        x, y, theta, _ = unicycle_step(
            5.0, 3.0, 0.5,
            0.8, 1.0,
            0.01,
            self.ARENA_W, self.ARENA_H, self.ROBOT_R,
        )
        expected_x = 5.0 + 0.8 * np.cos(0.5) * 0.01
        expected_y = 3.0 + 0.8 * np.sin(0.5) * 0.01
        expected_theta = wrap_angle(0.5 + 1.0 * 0.01)
        assert x == pytest.approx(expected_x, abs=1e-10)
        assert y == pytest.approx(expected_y, abs=1e-10)
        assert theta == pytest.approx(expected_theta, abs=1e-10)


class TestClipAction:
    """Tests for action clipping."""

    def test_within_bounds(self):
        v, omega = clip_action(0.5, 1.0, 1.0, 2.84)
        assert v == pytest.approx(0.5)
        assert omega == pytest.approx(1.0)

    def test_clip_v_negative(self):
        """Negative v should be clipped to 0 (no reverse)."""
        v, omega = clip_action(-0.5, 0.0, 1.0, 2.84)
        assert v == pytest.approx(0.0)

    def test_clip_v_over(self):
        v, omega = clip_action(2.0, 0.0, 1.0, 2.84)
        assert v == pytest.approx(1.0)

    def test_clip_omega(self):
        v, omega = clip_action(0.5, 5.0, 1.0, 2.84)
        assert omega == pytest.approx(2.84)
        v, omega = clip_action(0.5, -5.0, 1.0, 2.84)
        assert omega == pytest.approx(-2.84)
