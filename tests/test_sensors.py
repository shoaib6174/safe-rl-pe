"""Tests for Phase 3 sensor models (FOV, Lidar, WallSegment).

Test Index:
  A: test_fov_detects_target_in_cone
  B: test_fov_misses_target_outside_cone
  B2: test_fov_boundary_conditions
  C: test_lidar_obstacle_distance
  D: test_lidar_arena_boundary
"""

import numpy as np
import pytest

from envs.sensors import FOVSensor, LidarSensor, ObservationHistoryBuffer, WallSegment


# --- Test A: FOV detection when target is in cone ---

def test_fov_detects_target_in_cone():
    """Target at 5m, bearing=30 deg (within 60 deg half-angle): should detect."""
    sensor = FOVSensor(fov_angle=60, fov_range=10.0)
    own = [0, 0, 0]  # facing right (+x)
    # Target at ~5m, ~30 deg bearing
    target = [5.0 * np.cos(np.radians(30)), 5.0 * np.sin(np.radians(30)), 0]
    d, b = sensor.detect(own, target)
    assert d > 0, f"Target should be detected, got d={d}"
    assert abs(d - 5.0) < 0.1, f"Distance should be ~5m, got {d}"
    assert abs(b - np.radians(30)) < 0.05, f"Bearing should be ~30 deg, got {np.degrees(b)}"


# --- Test B: FOV misses target outside cone ---

def test_fov_misses_target_outside_cone():
    """Target at 5m but bearing=90 deg (outside 60 deg half-angle): miss."""
    sensor = FOVSensor(fov_angle=60, fov_range=10.0)
    own = [0, 0, 0]  # facing right
    target = [0, 5, 0]  # 90 deg bearing (straight up)
    d, b = sensor.detect(own, target)
    assert d == -1.0, "Target at 90 deg should NOT be detected (FOV half-angle is 60 deg)"


# --- Test B2: FOV boundary conditions ---

def test_fov_boundary_conditions():
    """Test edge cases: at max range, directly ahead, behind."""
    sensor = FOVSensor(fov_angle=60, fov_range=10.0)
    own = [0, 0, 0]

    # Target exactly at fov_range (strict inequality: NOT detected)
    target_at_range = [10.0, 0, 0]
    d, b = sensor.detect(own, target_at_range)
    assert d == -1.0, "Target at exactly fov_range should NOT be detected"

    # Target just inside fov_range
    target_inside = [9.99, 0, 0]
    d, b = sensor.detect(own, target_inside)
    assert d > 0, "Target just inside fov_range should be detected"

    # Target directly ahead (bearing = 0)
    target_ahead = [5.0, 0, 0]
    d, b = sensor.detect(own, target_ahead)
    assert d > 0, "Target directly ahead should be detected"
    assert abs(b) < 0.01, f"Bearing should be ~0, got {b}"

    # Target directly behind (bearing = pi)
    target_behind = [-5.0, 0, 0]
    d, b = sensor.detect(own, target_behind)
    assert d == -1.0, "Target directly behind should NOT be detected"

    # Target at zero distance (coincident)
    target_zero = [0, 0, 0]
    d, b = sensor.detect(own, target_zero)
    assert d == 0.0, "Zero-distance target should be detected"


# --- Test C: Lidar detects obstacle at correct distance ---

def test_lidar_obstacle_distance():
    """Obstacle at 3m directly ahead: lidar forward ray should read ~3m."""
    sensor = LidarSensor(n_rays=36, max_range=5.0, noise_std=0.0)
    own = [0, 0, 0]  # facing right (+x)
    obs = [{"x": 3.5, "y": 0, "radius": 0.5}]  # Edge at 3.0m
    arena_bounds = {"x_min": -20, "x_max": 20, "y_min": -20, "y_max": 20}
    readings = sensor.scan(own, obs, arena_bounds)

    # The first ray angle is -pi (pointing left), need to find the forward ray
    # Rays are at angles from -pi to pi, so forward (angle=0) is at index n_rays//2
    # Actually, linspace(-pi, pi, 36, endpoint=False) puts 0 at index 18
    # Let's find the forward ray index
    angles = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    forward_idx = np.argmin(np.abs(angles))  # Closest to 0

    assert abs(readings[forward_idx] - 3.0) < 0.15, (
        f"Forward ray should read ~3.0m, got {readings[forward_idx]}"
    )


# --- Test D: Lidar detects arena boundary ---

def test_lidar_arena_boundary():
    """Robot facing right wall at 2m: forward lidar should read 2m."""
    sensor = LidarSensor(n_rays=36, max_range=5.0, noise_std=0.0)
    own = [8.0, 5.0, 0]  # facing right (+x), 2m from x_max=10
    arena_bounds = {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10}
    readings = sensor.scan(own, [], arena_bounds)

    # Find forward ray index (closest to angle=0)
    angles = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    forward_idx = np.argmin(np.abs(angles))

    assert abs(readings[forward_idx] - 2.0) < 0.15, (
        f"Forward ray should read ~2.0m, got {readings[forward_idx]}"
    )


# --- Additional sensor tests ---

class TestFOVSensorAdditional:
    """Additional FOV sensor tests for robustness."""

    def test_detection_with_rotated_heading(self):
        """Test FOV with non-zero heading."""
        sensor = FOVSensor(fov_angle=60, fov_range=10.0)
        # Robot facing up (+y), target is at bearing 0 (directly ahead)
        own = [5.0, 5.0, np.pi / 2]  # heading = 90 deg
        target = [5.0, 10.0, 0]  # directly ahead (north)
        d, b = sensor.detect(own, target)
        assert d > 0, "Target ahead of rotated robot should be detected"
        assert abs(b) < 0.05, f"Bearing should be ~0, got {b}"

    def test_detection_symmetric(self):
        """Targets at +/- 45 deg should both be detected."""
        sensor = FOVSensor(fov_angle=60, fov_range=10.0)
        own = [0, 0, 0]
        d_pos = 5.0
        for sign in [1, -1]:
            target = [
                d_pos * np.cos(np.radians(45 * sign)),
                d_pos * np.sin(np.radians(45 * sign)),
                0,
            ]
            d, b = sensor.detect(own, target)
            assert d > 0, f"Target at {45*sign} deg should be detected"


class TestLidarSensorAdditional:
    """Additional Lidar sensor tests."""

    def test_all_rays_capped_at_max_range(self):
        """In open space, all rays should read max_range."""
        sensor = LidarSensor(n_rays=36, max_range=5.0, noise_std=0.0)
        own = [50, 50, 0]  # far from all walls
        arena_bounds = {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100}
        readings = sensor.scan(own, [], arena_bounds)
        assert np.allclose(readings, 5.0, atol=0.01), "All readings should be max_range"

    def test_wall_segment_intersection(self):
        """Lidar should detect wall segments."""
        sensor = LidarSensor(n_rays=36, max_range=10.0, noise_std=0.0)
        own = [5.0, 5.0, 0]  # facing right
        wall = WallSegment(8.0, 0.0, 8.0, 10.0)  # vertical wall at x=8
        arena_bounds = {"x_min": 0, "x_max": 20, "y_min": 0, "y_max": 20}
        readings = sensor.scan(own, [], arena_bounds, wall_segments=[wall])

        # Forward ray should detect the wall at ~3m
        angles = np.linspace(-np.pi, np.pi, 36, endpoint=False)
        forward_idx = np.argmin(np.abs(angles))
        assert readings[forward_idx] < 5.0, (
            f"Forward ray should detect wall at 3m, got {readings[forward_idx]}"
        )
        assert abs(readings[forward_idx] - 3.0) < 0.5, (
            f"Wall distance should be ~3m, got {readings[forward_idx]}"
        )

    def test_noise_adds_variability(self):
        """Noisy lidar should produce different readings across calls."""
        sensor = LidarSensor(n_rays=36, max_range=5.0, noise_std=0.1)
        own = [5.0, 5.0, 0]
        arena_bounds = {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10}
        r1 = sensor.scan(own, [], arena_bounds)
        r2 = sensor.scan(own, [], arena_bounds)
        # With noise_std=0.1, readings should differ slightly
        assert not np.allclose(r1, r2), "Noisy lidar should give different readings"


class TestWallSegment:
    """Tests for WallSegment."""

    def test_distance_to_point_midpoint(self):
        """Point perpendicular to wall midpoint."""
        wall = WallSegment(0, 0, 10, 0)
        assert abs(wall.distance_to_point(np.array([5, 3])) - 3.0) < 0.01

    def test_distance_to_point_endpoint(self):
        """Point closest to an endpoint."""
        wall = WallSegment(0, 0, 10, 0)
        # Point at (-3, 4): closest to p1=(0,0), distance = 5
        d = wall.distance_to_point(np.array([-3, 4]))
        assert abs(d - 5.0) < 0.01

    def test_zero_length_raises(self):
        """Zero-length wall should raise ValueError."""
        with pytest.raises(ValueError, match="zero length"):
            WallSegment(5, 5, 5, 5)

    def test_closest_point(self):
        """Closest point should be on the segment."""
        wall = WallSegment(0, 0, 10, 0)
        cp = wall.closest_point(np.array([5, 3]))
        np.testing.assert_allclose(cp, [5, 0], atol=0.01)


class TestObservationHistoryBuffer:
    """Tests for ObservationHistoryBuffer."""

    def test_fill_and_get(self):
        """Buffer should maintain FIFO order."""
        buf = ObservationHistoryBuffer(K=3, obs_dim=2)
        buf.add(np.array([1.0, 2.0]))
        buf.add(np.array([3.0, 4.0]))
        buf.add(np.array([5.0, 6.0]))
        hist = buf.get_history()
        assert hist.shape == (3, 2)
        np.testing.assert_allclose(hist[-1], [5.0, 6.0])
        np.testing.assert_allclose(hist[0], [1.0, 2.0])
        assert buf.is_full

    def test_zero_padded_initially(self):
        """Before K observations, buffer is zero-padded."""
        buf = ObservationHistoryBuffer(K=5, obs_dim=3)
        buf.add(np.array([1.0, 1.0, 1.0]))
        hist = buf.get_history()
        assert hist.shape == (5, 3)
        # First 4 should be zero
        np.testing.assert_allclose(hist[:4], 0.0)
        np.testing.assert_allclose(hist[4], [1.0, 1.0, 1.0])
        assert not buf.is_full

    def test_reset_clears(self):
        """Reset should zero the buffer."""
        buf = ObservationHistoryBuffer(K=3, obs_dim=2)
        buf.add(np.array([1.0, 2.0]))
        buf.reset()
        np.testing.assert_allclose(buf.get_history(), 0.0)
        assert buf.filled == 0
