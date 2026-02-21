"""Baseline agents for comparison: random, greedy, DQN, DDPG.

These provide reference points for evaluating PPO self-play performance.
"""

import numpy as np

from envs.dynamics import wrap_angle


class RandomPolicy:
    """Random action policy. Both v and omega sampled uniformly."""

    def __init__(self, v_max: float = 1.0, omega_max: float = 2.84):
        self.v_max = v_max
        self.omega_max = omega_max

    def predict(self, obs, deterministic=False):
        v = np.random.uniform(0, self.v_max)
        omega = np.random.uniform(-self.omega_max, self.omega_max)
        return np.array([v, omega], dtype=np.float32), None


class GreedyPursuerPolicy:
    """Proportional navigation pursuer: steer toward evader.

    v = v_max * max(0, cos(heading_error))  — slow down when turning
    omega = clip(K_p * heading_error, -omega_max, omega_max)

    K_p = 3.0 justified by Dovrat et al. (2022, IEEE CSL): sufficiently
    large K guarantees capture for unicycle bearing pursuit.
    """

    def __init__(
        self,
        v_max: float = 1.0,
        omega_max: float = 2.84,
        K_p: float = 3.0,
        arena_half_w: float = 10.0,
        arena_half_h: float = 10.0,
    ):
        self.v_max = v_max
        self.omega_max = omega_max
        self.K_p = K_p
        self.arena_half_w = arena_half_w
        self.arena_half_h = arena_half_h

    def predict(self, obs, deterministic=False):
        """Predict action from normalized observation.

        Obs layout (14D):
            [x_s_norm, y_s_norm, theta_s_norm,
             v_s_norm, omega_s_norm,
             x_o_norm, y_o_norm, theta_o_norm,
             v_o_norm, omega_o_norm,
             distance_norm, bearing_norm,
             d_wall_x_norm, d_wall_y_norm]
        """
        # Denormalize relevant fields
        x_self = obs[0] * self.arena_half_w
        y_self = obs[1] * self.arena_half_h
        theta_self = obs[2] * np.pi
        x_opp = obs[5] * self.arena_half_w
        y_opp = obs[6] * self.arena_half_h

        # Bearing to opponent
        dx = x_opp - x_self
        dy = y_opp - y_self
        bearing = np.arctan2(dy, dx)
        heading_error = wrap_angle(bearing - theta_self)

        # Proportional navigation
        omega = np.clip(self.K_p * heading_error, -self.omega_max, self.omega_max)
        v = self.v_max * max(0.0, np.cos(heading_error))

        return np.array([v, omega], dtype=np.float32), None


class GreedyEvaderPolicy:
    """Flee-from-pursuer evader: steer away from pursuer at max speed.

    v = v_max (always max speed)
    omega = clip(K_p * heading_error_away, -omega_max, omega_max)
    """

    def __init__(
        self,
        v_max: float = 1.0,
        omega_max: float = 2.84,
        K_p: float = 3.0,
        arena_half_w: float = 10.0,
        arena_half_h: float = 10.0,
    ):
        self.v_max = v_max
        self.omega_max = omega_max
        self.K_p = K_p
        self.arena_half_w = arena_half_w
        self.arena_half_h = arena_half_h

    def predict(self, obs, deterministic=False):
        """Predict action from normalized observation (evader perspective)."""
        x_self = obs[0] * self.arena_half_w
        y_self = obs[1] * self.arena_half_h
        theta_self = obs[2] * np.pi
        x_opp = obs[5] * self.arena_half_w
        y_opp = obs[6] * self.arena_half_h

        # Bearing to pursuer
        dx = x_opp - x_self
        dy = y_opp - y_self
        bearing_to_pursuer = np.arctan2(dy, dx)

        # Away bearing = opposite direction
        away_bearing = bearing_to_pursuer + np.pi
        heading_error = wrap_angle(away_bearing - theta_self)

        # Proportional flee
        omega = np.clip(self.K_p * heading_error, -self.omega_max, self.omega_max)
        v = self.v_max  # Always max speed

        return np.array([v, omega], dtype=np.float32), None


def create_discrete_action_space(
    v_max: float = 1.0,
    omega_max: float = 2.84,
    n_velocities: int = 5,
    n_angular: int = 7,
) -> np.ndarray:
    """Create discrete action lookup table for DQN.

    Returns array of shape (n_velocities * n_angular, 2) mapping
    discrete action index to [v, omega].
    """
    velocities = np.linspace(0, v_max, n_velocities)
    angulars = np.linspace(-omega_max, omega_max, n_angular)

    actions = []
    for v in velocities:
        for omega in angulars:
            actions.append([v, omega])

    return np.array(actions, dtype=np.float32)
