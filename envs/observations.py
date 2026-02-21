"""Observation builder for the pursuit-evasion environment.

Phase 1: Full state observations (14D per agent).
Phase 2: Adds obstacle distance/bearing features (14 + 2*K per agent).
Phase 3 will add: lidar, partial observability, belief encoding.

Per-agent observation (Phase 2 — with obstacles):
    obs = [
        x_self, y_self, theta_self,         # own state (normalized)
        v_self, omega_self,                  # own velocity (normalized)
        x_opp, y_opp, theta_opp,            # opponent state (normalized)
        v_opp, omega_opp,                   # opponent velocity (normalized)
        distance_to_opponent,               # normalized by arena diagonal
        bearing_to_opponent,                # normalized to [-1, 1]
        d_to_nearest_wall_x,               # normalized by half arena width
        d_to_nearest_wall_y,               # normalized by half arena height
        obs1_dist, obs1_bearing,           # nearest obstacle (if n_obstacle_obs > 0)
        obs2_dist, obs2_bearing,           # second nearest
        ...                                # up to K obstacles
    ]
"""

import numpy as np

from envs.dynamics import wrap_angle


class ObservationBuilder:
    """Builds normalized observations for each agent.

    Args:
        arena_width: Arena width in meters.
        arena_height: Arena height in meters.
        v_max: Maximum linear velocity for normalization.
        omega_max: Maximum angular velocity for normalization.
        full_state: Whether to include full opponent state (Phase 1).
        n_obstacle_obs: Number of nearest obstacles to include in obs (0 = none).
    """

    def __init__(
        self,
        arena_width: float,
        arena_height: float,
        v_max: float,
        omega_max: float,
        full_state: bool = True,
        n_obstacle_obs: int = 0,
    ):
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.half_w = arena_width / 2.0
        self.half_h = arena_height / 2.0
        self.v_max = v_max
        self.omega_max = omega_max
        self.arena_diagonal = np.sqrt(arena_width**2 + arena_height**2)
        self.full_state = full_state
        self.n_obstacle_obs = n_obstacle_obs
        self.obs_dim = 14 + 2 * n_obstacle_obs

    def build(
        self,
        self_state: np.ndarray,
        self_action: np.ndarray,
        opp_state: np.ndarray,
        opp_action: np.ndarray,
        obstacles: list[dict] | None = None,
    ) -> np.ndarray:
        """Build normalized observation for one agent.

        Args:
            self_state: [x, y, theta] of the observing agent.
            self_action: [v, omega] last action of the observing agent.
            opp_state: [x, y, theta] of the opponent.
            opp_action: [v, omega] last action of the opponent.
            obstacles: List of obstacle dicts with 'x', 'y', 'radius' keys.

        Returns:
            Normalized observation array of shape (obs_dim,).
        """
        x_s, y_s, theta_s = self_state
        v_s, omega_s = self_action
        x_o, y_o, theta_o = opp_state
        v_o, omega_o = opp_action

        # Normalize positions to [-1, 1]
        x_s_norm = x_s / self.half_w
        y_s_norm = y_s / self.half_h
        x_o_norm = x_o / self.half_w
        y_o_norm = y_o / self.half_h

        # Normalize angles to [-1, 1] (theta in [-pi, pi] -> divide by pi)
        theta_s_norm = theta_s / np.pi
        theta_o_norm = theta_o / np.pi

        # Normalize velocities
        v_s_norm = v_s / self.v_max
        omega_s_norm = omega_s / self.omega_max
        v_o_norm = v_o / self.v_max
        omega_o_norm = omega_o / self.omega_max

        # Distance to opponent (normalized by arena diagonal)
        dx = x_o - x_s
        dy = y_o - y_s
        distance = np.sqrt(dx**2 + dy**2)
        distance_norm = distance / self.arena_diagonal

        # Bearing to opponent (relative to own heading, normalized to [-1, 1])
        bearing_abs = np.arctan2(dy, dx)
        bearing_rel = wrap_angle(bearing_abs - theta_s)
        bearing_norm = bearing_rel / np.pi

        # Distance to nearest wall (normalized)
        d_wall_x = min(x_s - (-self.half_w), self.half_w - x_s) / self.half_w
        d_wall_y = min(y_s - (-self.half_h), self.half_h - y_s) / self.half_h

        base_obs = [
            x_s_norm, y_s_norm, theta_s_norm,
            v_s_norm, omega_s_norm,
            x_o_norm, y_o_norm, theta_o_norm,
            v_o_norm, omega_o_norm,
            distance_norm,
            bearing_norm,
            d_wall_x,
            d_wall_y,
        ]

        # Obstacle features: distance and bearing to K nearest obstacles
        if self.n_obstacle_obs > 0:
            obs_features = self._compute_obstacle_features(
                x_s, y_s, theta_s, obstacles,
            )
            base_obs.extend(obs_features)

        return np.array(base_obs, dtype=np.float32)

    def _compute_obstacle_features(
        self,
        x: float,
        y: float,
        theta: float,
        obstacles: list[dict] | None,
    ) -> list[float]:
        """Compute distance and bearing to K nearest obstacles.

        Returns:
            Flat list of [dist1, bearing1, dist2, bearing2, ...] for K nearest.
            Padded with (1.0, 0.0) if fewer than K obstacles exist.
        """
        k = self.n_obstacle_obs

        if not obstacles:
            # No obstacles: pad with max distance, zero bearing
            return [1.0, 0.0] * k

        # Compute distances to all obstacles (surface distance, not center)
        dists_bearings = []
        for obs in obstacles:
            dx = obs["x"] - x
            dy = obs["y"] - y
            center_dist = np.sqrt(dx**2 + dy**2)
            # Surface distance (distance to obstacle edge)
            surface_dist = max(center_dist - obs["radius"], 0.0)
            # Normalize by arena diagonal
            dist_norm = min(surface_dist / self.arena_diagonal, 1.0)
            # Bearing (relative to heading)
            bearing_abs = np.arctan2(dy, dx)
            bearing_rel = wrap_angle(bearing_abs - theta)
            bearing_norm = bearing_rel / np.pi
            dists_bearings.append((dist_norm, bearing_norm, center_dist))

        # Sort by center distance, take K nearest
        dists_bearings.sort(key=lambda x: x[2])
        features = []
        for i in range(k):
            if i < len(dists_bearings):
                features.extend([dists_bearings[i][0], dists_bearings[i][1]])
            else:
                features.extend([1.0, 0.0])

        return features
