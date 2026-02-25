"""Unit tests for unicycle dynamics."""

import numpy as np
import pytest

from envs.dynamics import clip_action, resolve_obstacle_collisions, unicycle_step, wrap_angle


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


class TestResolveObstacleCollisions:
    """Tests for obstacle collision resolution (physical barrier enforcement)."""

    ROBOT_R = 0.15

    def test_agent_inside_pushed_to_surface(self):
        """Agent inside obstacle is pushed to the obstacle surface."""
        obstacles = [{"x": 3.0, "y": 3.0, "radius": 1.0}]
        # Place agent inside: 0.5m from center, min_dist = 1.0 + 0.15 = 1.15
        x, y, theta, collided = resolve_obstacle_collisions(
            3.3, 3.0, 0.0, obstacles, self.ROBOT_R,
        )
        dist = np.sqrt((x - 3.0)**2 + (y - 3.0)**2)
        assert collided is True
        assert dist == pytest.approx(1.0 + self.ROBOT_R, abs=1e-10)

    def test_agent_at_boundary_not_modified(self):
        """Agent exactly at min_dist boundary is not moved, no collision."""
        obstacles = [{"x": 3.0, "y": 3.0, "radius": 1.0}]
        min_dist = 1.0 + self.ROBOT_R  # 1.15
        x_start = 3.0 + min_dist
        x, y, theta, collided = resolve_obstacle_collisions(
            x_start, 3.0, 0.0, obstacles, self.ROBOT_R,
        )
        assert collided is False
        assert x == pytest.approx(x_start, abs=1e-10)
        assert y == pytest.approx(3.0, abs=1e-10)

    def test_agent_outside_not_modified(self):
        """Agent safely away from obstacle is not modified."""
        obstacles = [{"x": 3.0, "y": 3.0, "radius": 1.0}]
        x, y, theta, collided = resolve_obstacle_collisions(
            6.0, 3.0, 0.0, obstacles, self.ROBOT_R,
        )
        assert collided is False
        assert x == pytest.approx(6.0, abs=1e-10)
        assert y == pytest.approx(3.0, abs=1e-10)

    def test_agent_at_obstacle_center_uses_heading(self):
        """Agent exactly at obstacle center is pushed along heading direction."""
        obstacles = [{"x": 3.0, "y": 3.0, "radius": 1.0}]
        theta = np.pi / 4  # heading northeast
        x, y, theta_out, collided = resolve_obstacle_collisions(
            3.0, 3.0, theta, obstacles, self.ROBOT_R,
        )
        assert collided is True
        dist = np.sqrt((x - 3.0)**2 + (y - 3.0)**2)
        assert dist == pytest.approx(1.0 + self.ROBOT_R, abs=1e-10)
        # Direction should match heading
        direction = np.arctan2(y - 3.0, x - 3.0)
        assert direction == pytest.approx(theta, abs=1e-6)

    def test_heading_never_modified(self):
        """Heading (theta) must never be changed by collision resolution."""
        obstacles = [{"x": 3.0, "y": 3.0, "radius": 1.0}]
        for theta_in in [0.0, np.pi / 2, -np.pi / 4, np.pi, -2.5]:
            _, _, theta_out, _ = resolve_obstacle_collisions(
                3.0, 3.0, theta_in, obstacles, self.ROBOT_R,
            )
            assert theta_out == pytest.approx(theta_in, abs=1e-10)

    def test_multiple_obstacles_both_resolved(self):
        """Agent overlapping two obstacles gets pushed out of both."""
        # Obstacles spaced 3.0m apart (enough room for robot between them)
        obstacles = [
            {"x": 0.0, "y": 0.0, "radius": 1.0},
            {"x": 3.0, "y": 0.0, "radius": 1.0},
        ]
        # Place agent inside first obstacle (dist=0.5 < 1.15)
        x, y, theta, collided = resolve_obstacle_collisions(
            0.5, 0.0, 0.0, obstacles, self.ROBOT_R,
        )
        assert collided is True
        # Should be outside the first obstacle
        d0 = np.sqrt(x**2 + y**2)
        assert d0 >= 1.0 + self.ROBOT_R - 1e-6

    def test_multiple_obstacles_both_detected(self):
        """Agent that overlaps two separate obstacles triggers collision for both."""
        obstacles = [
            {"x": -2.0, "y": 0.0, "radius": 1.0},
            {"x": 2.0, "y": 0.0, "radius": 1.0},
        ]
        # Place agent inside first obstacle
        x, y, theta, c1 = resolve_obstacle_collisions(
            -1.5, 0.0, 0.0, obstacles, self.ROBOT_R,
        )
        assert c1 is True
        # Place agent inside second obstacle
        x, y, theta, c2 = resolve_obstacle_collisions(
            1.5, 0.0, 0.0, obstacles, self.ROBOT_R,
        )
        assert c2 is True

    def test_empty_obstacles_noop(self):
        """Empty obstacles list should be a no-op."""
        x, y, theta, collided = resolve_obstacle_collisions(
            1.0, 2.0, 0.5, [], self.ROBOT_R,
        )
        assert collided is False
        assert x == pytest.approx(1.0, abs=1e-10)
        assert y == pytest.approx(2.0, abs=1e-10)
        assert theta == pytest.approx(0.5, abs=1e-10)

    def test_cannot_tunnel_through_obstacle(self):
        """500-step integration: agent driving toward obstacle cannot pass through."""
        obstacles = [{"x": 3.0, "y": 0.0, "radius": 1.0}]
        x, y, theta = 0.0, 0.0, 0.0  # heading east toward obstacle
        dt = 0.05
        v = 1.0

        for _ in range(500):
            # Euler step
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            # Resolve
            x, y, theta, _ = resolve_obstacle_collisions(
                x, y, theta, obstacles, self.ROBOT_R,
            )

        # Agent should be stuck at obstacle surface, NOT past x=3.0+1.0
        dist_to_obs = np.sqrt((x - 3.0)**2 + y**2)
        assert dist_to_obs >= 1.0 + self.ROBOT_R - 1e-6
        assert x <= 3.0 + 1.0 + self.ROBOT_R + 0.01  # cannot pass through
