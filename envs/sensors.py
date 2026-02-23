"""Sensor models for partial observability in pursuit-evasion.

Implements:
- FOVSensor: Limited field-of-view detection of opponent
- LidarSensor: 360-degree range sensing for obstacles/boundaries
- WallSegment: Linear wall obstacle for complex environments
"""

from __future__ import annotations

import numpy as np

from envs.dynamics import wrap_angle


class FOVSensor:
    """Limited field-of-view sensor for detecting the opponent.

    Models a forward-facing detection cone with configurable half-angle
    and maximum range. Returns (distance, bearing) if target is within
    the cone, or (-1, 0) if not detected.

    Args:
        fov_angle: Half-angle of FOV cone in degrees (e.g., 60 for 120 deg total).
        fov_range: Maximum detection range in meters.
    """

    def __init__(self, fov_angle: float = 60.0, fov_range: float = 10.0):
        self.fov_half_angle = np.radians(fov_angle)
        self.fov_range = fov_range

    def detect(
        self,
        own_state: np.ndarray | list,
        target_state: np.ndarray | list,
    ) -> tuple[float, float]:
        """Detect target if within FOV cone.

        Args:
            own_state: [x, y, theta] of the sensing agent.
            target_state: [x, y, ...] of the target (only x, y used).

        Returns:
            (distance, bearing) if detected, (-1.0, 0.0) if not.
            Bearing is relative to own heading, in [-pi, pi].
        """
        own_x, own_y, own_theta = float(own_state[0]), float(own_state[1]), float(own_state[2])
        target_x, target_y = float(target_state[0]), float(target_state[1])

        dx = target_x - own_x
        dy = target_y - own_y
        dist = np.sqrt(dx * dx + dy * dy)

        # Check range (strict inequality: at exactly fov_range, not detected)
        if dist >= self.fov_range:
            return -1.0, 0.0

        # Zero distance edge case
        if dist < 1e-10:
            return 0.0, 0.0

        # Bearing relative to own heading
        absolute_angle = np.arctan2(dy, dx)
        bearing = wrap_angle(absolute_angle - own_theta)

        # Check if within FOV cone (strict inequality on boundary)
        if abs(bearing) > self.fov_half_angle:
            return -1.0, 0.0

        return float(dist), float(bearing)


class LidarSensor:
    """360-degree lidar sensor for obstacle and boundary detection.

    Casts uniformly-spaced rays and returns distance to the nearest
    obstacle or boundary along each ray direction.

    Args:
        n_rays: Number of uniformly spaced rays.
        max_range: Maximum lidar range in meters.
        noise_std: Gaussian noise standard deviation (meters). 0 for noiseless.
    """

    def __init__(
        self,
        n_rays: int = 36,
        max_range: float = 5.0,
        noise_std: float = 0.0,
    ):
        self.n_rays = n_rays
        self.max_range = max_range
        self.noise_std = noise_std
        # Ray angles relative to robot heading: uniformly from -pi to pi
        self.angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)

    def scan(
        self,
        own_state: np.ndarray | list,
        obstacles: list[dict],
        arena_bounds: dict,
        wall_segments: list | None = None,
    ) -> np.ndarray:
        """Perform a lidar scan.

        Args:
            own_state: [x, y, theta] of the sensing agent.
            obstacles: List of circular obstacles [{'x', 'y', 'radius'}, ...].
            arena_bounds: Dict with 'x_min', 'x_max', 'y_min', 'y_max'.
            wall_segments: Optional list of WallSegment objects.

        Returns:
            Array of distances [n_rays], clipped to [0, max_range].
        """
        own_x = float(own_state[0])
        own_y = float(own_state[1])
        own_theta = float(own_state[2])

        readings = np.full(self.n_rays, self.max_range, dtype=np.float64)

        for i, angle_offset in enumerate(self.angles):
            ray_angle = own_theta + angle_offset

            # Check arena boundaries
            boundary_dist = self._ray_boundary_intersect(
                own_x, own_y, ray_angle, arena_bounds,
            )
            readings[i] = min(readings[i], boundary_dist)

            # Check circular obstacles
            for obs in obstacles:
                obs_dist = self._ray_circle_intersect(
                    own_x, own_y, ray_angle,
                    obs["x"], obs["y"], obs["radius"],
                )
                if obs_dist > 0:
                    readings[i] = min(readings[i], obs_dist)

            # Check wall segments
            if wall_segments is not None:
                for wall in wall_segments:
                    wall_dist = self._ray_segment_intersect(
                        own_x, own_y, ray_angle, wall,
                    )
                    if wall_dist > 0:
                        readings[i] = min(readings[i], wall_dist)

        # Clip to valid range
        readings = np.clip(readings, 0.0, self.max_range)

        # Add noise if configured
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=self.n_rays)
            readings = np.clip(readings + noise, 0.0, self.max_range)

        return readings.astype(np.float32)

    def _ray_circle_intersect(
        self,
        rx: float, ry: float, angle: float,
        cx: float, cy: float, radius: float,
    ) -> float:
        """Ray-circle intersection. Returns distance or inf if no hit."""
        dx = np.cos(angle)
        dy = np.sin(angle)
        fx = rx - cx
        fy = ry - cy

        # Quadratic: a*t^2 + b*t + c = 0 where a = 1 (unit direction)
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius

        discriminant = b * b - 4.0 * c
        if discriminant < 0:
            return float("inf")

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0

        # Return nearest positive intersection
        if t1 > 1e-6:
            return t1
        if t2 > 1e-6:
            return t2
        return float("inf")

    def _ray_boundary_intersect(
        self,
        rx: float, ry: float, angle: float,
        bounds: dict,
    ) -> float:
        """Ray-axis-aligned-boundary intersection. Returns nearest hit distance."""
        dx = np.cos(angle)
        dy = np.sin(angle)
        min_t = float("inf")

        # Check all 4 boundaries
        # Right wall: x = x_max
        if abs(dx) > 1e-10:
            t = (bounds["x_max"] - rx) / dx
            if t > 1e-6:
                hit_y = ry + t * dy
                if bounds["y_min"] <= hit_y <= bounds["y_max"]:
                    min_t = min(min_t, t)

            # Left wall: x = x_min
            t = (bounds["x_min"] - rx) / dx
            if t > 1e-6:
                hit_y = ry + t * dy
                if bounds["y_min"] <= hit_y <= bounds["y_max"]:
                    min_t = min(min_t, t)

        if abs(dy) > 1e-10:
            # Top wall: y = y_max
            t = (bounds["y_max"] - ry) / dy
            if t > 1e-6:
                hit_x = rx + t * dx
                if bounds["x_min"] <= hit_x <= bounds["x_max"]:
                    min_t = min(min_t, t)

            # Bottom wall: y = y_min
            t = (bounds["y_min"] - ry) / dy
            if t > 1e-6:
                hit_x = rx + t * dx
                if bounds["x_min"] <= hit_x <= bounds["x_max"]:
                    min_t = min(min_t, t)

        return min_t

    def _ray_segment_intersect(
        self,
        rx: float, ry: float, angle: float,
        wall: "WallSegment",
    ) -> float:
        """Ray-line-segment intersection for WallSegment obstacles.

        Uses parametric intersection of ray with line segment.
        Returns distance or inf if no hit.
        """
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Wall segment: p1 to p2
        sx = wall.p2[0] - wall.p1[0]
        sy = wall.p2[1] - wall.p1[1]

        denom = dx * sy - dy * sx
        if abs(denom) < 1e-10:
            return float("inf")  # Parallel

        # Solve for t (ray parameter) and u (segment parameter)
        qx = wall.p1[0] - rx
        qy = wall.p1[1] - ry

        t = (qx * sy - qy * sx) / denom
        u = (qx * dy - qy * dx) / denom

        if t > 1e-6 and 0.0 <= u <= 1.0:
            return t
        return float("inf")


class WallSegment:
    """A linear wall segment defined by two endpoints.

    Used for complex environment layouts (corridor, L-shaped, warehouse).
    These are for the generalization study (evaluation only, not core training).

    Args:
        x1, y1: First endpoint.
        x2, y2: Second endpoint.
        thickness: Wall thickness for collision checking.
    """

    def __init__(self, x1: float, y1: float, x2: float, y2: float, thickness: float = 0.1):
        self.p1 = np.array([x1, y1], dtype=np.float64)
        self.p2 = np.array([x2, y2], dtype=np.float64)
        self.thickness = thickness
        self.length = float(np.linalg.norm(self.p2 - self.p1))
        if self.length < 1e-10:
            raise ValueError("Wall segment has zero length")
        self.direction = (self.p2 - self.p1) / self.length
        self.normal = np.array([-self.direction[1], self.direction[0]])

    def distance_to_point(self, point: np.ndarray) -> float:
        """Minimum distance from a point to the wall segment.

        Projects point onto the segment line and clips to [0, 1] to handle
        endpoint regions correctly.

        Args:
            point: [x, y] position.

        Returns:
            Minimum distance (always >= 0).
        """
        v = point - self.p1
        t = np.clip(np.dot(v, self.direction) / self.length, 0.0, 1.0)
        closest = self.p1 + t * (self.p2 - self.p1)
        return float(np.linalg.norm(point - closest))

    def closest_point(self, point: np.ndarray) -> np.ndarray:
        """Find the closest point on the segment to the given point.

        Args:
            point: [x, y] position.

        Returns:
            Closest point [x, y] on the segment.
        """
        v = point - self.p1
        t = np.clip(np.dot(v, self.direction) / self.length, 0.0, 1.0)
        return self.p1 + t * (self.p2 - self.p1)


class ObservationHistoryBuffer:
    """Rolling window buffer for observation history.

    Maintains the last K observations for temporal processing
    (e.g., BiMDN belief encoder). Zero-padded before K observations
    are collected.

    Args:
        K: History window length.
        obs_dim: Dimension of each observation.
    """

    def __init__(self, K: int = 10, obs_dim: int = 40):
        self.K = K
        self.obs_dim = obs_dim
        self.buffer = np.zeros((K, obs_dim), dtype=np.float32)
        self.filled = 0

    def add(self, obs: np.ndarray) -> None:
        """Add an observation to the buffer (FIFO)."""
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = obs[:self.obs_dim]
        self.filled = min(self.filled + 1, self.K)

    def get_history(self) -> np.ndarray:
        """Return the full history buffer (K, obs_dim). Zero-padded if < K obs."""
        return self.buffer.copy()

    def reset(self) -> None:
        """Clear the buffer."""
        self.buffer = np.zeros((self.K, self.obs_dim), dtype=np.float32)
        self.filled = 0

    @property
    def is_full(self) -> bool:
        """Whether K observations have been collected."""
        return self.filled >= self.K
