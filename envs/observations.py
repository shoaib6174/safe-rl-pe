"""Observation builder for the pursuit-evasion environment.

Phase 1: Full state observations (14D per agent).
Phase 3 will add: lidar, partial observability, belief encoding.

Per-agent observation (Phase 1 — full state):
    obs = [
        x_self, y_self, theta_self,         # own state (normalized)
        v_self, omega_self,                  # own velocity (normalized)
        x_opp, y_opp, theta_opp,            # opponent state (normalized)
        v_opp, omega_opp,                   # opponent velocity (normalized)
        distance_to_opponent,               # normalized by arena diagonal
        bearing_to_opponent,                # normalized to [-1, 1]
        d_to_nearest_wall_x,               # normalized by half arena width
        d_to_nearest_wall_y,               # normalized by half arena height
    ]
"""

import numpy as np

from envs.dynamics import wrap_angle


class ObservationBuilder:
    """Builds normalized observations for each agent.

    Phase 3 extension: override build() to add lidar, partial obs, belief encoding.
    """

    def __init__(
        self,
        arena_width: float,
        arena_height: float,
        v_max: float,
        omega_max: float,
        full_state: bool = True,
    ):
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.half_w = arena_width / 2.0
        self.half_h = arena_height / 2.0
        self.v_max = v_max
        self.omega_max = omega_max
        self.arena_diagonal = np.sqrt(arena_width**2 + arena_height**2)
        self.full_state = full_state
        self.obs_dim = 14

    def build(
        self,
        self_state: np.ndarray,
        self_action: np.ndarray,
        opp_state: np.ndarray,
        opp_action: np.ndarray,
    ) -> np.ndarray:
        """Build normalized observation for one agent.

        Args:
            self_state: [x, y, theta] of the observing agent.
            self_action: [v, omega] last action of the observing agent.
            opp_state: [x, y, theta] of the opponent.
            opp_action: [v, omega] last action of the opponent.

        Returns:
            Normalized observation array of shape (14,).
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

        obs = np.array([
            x_s_norm, y_s_norm, theta_s_norm,
            v_s_norm, omega_s_norm,
            x_o_norm, y_o_norm, theta_o_norm,
            v_o_norm, omega_o_norm,
            distance_norm,
            bearing_norm,
            d_wall_x,
            d_wall_y,
        ], dtype=np.float32)

        return obs
