"""Unicycle (differential-drive) dynamics model for ground robots.

The unicycle model:
    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    theta_dot = omega

State:   s = [x, y, theta]        (position + heading)
Control: u = [v, omega]           (linear velocity, angular velocity)
Bounds:  v in [0, v_max],  omega in [-omega_max, omega_max]

Uses Euler integration with configurable timestep.
Arena wall collision model: position clipping with velocity zeroing.
"""

import numpy as np


def wrap_angle(theta: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def unicycle_step(
    x: float,
    y: float,
    theta: float,
    v: float,
    omega: float,
    dt: float,
    arena_width: float,
    arena_height: float,
    robot_radius: float,
) -> tuple[float, float, float, bool]:
    """Advance unicycle state by one timestep using Euler integration.

    Arena is centered at (0, 0), spanning [-W/2, W/2] x [-H/2, H/2].

    Args:
        x, y, theta: Current state (position and heading).
        v: Linear velocity command (will be clipped to [0, v_max] by caller).
        omega: Angular velocity command.
        dt: Integration timestep in seconds.
        arena_width: Arena width in meters.
        arena_height: Arena height in meters.
        robot_radius: Robot radius for wall collision offset.

    Returns:
        (x_new, y_new, theta_new, wall_contact): Updated state and whether
        the robot hit a wall (position was clipped).
    """
    # Euler integration
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = wrap_angle(theta + omega * dt)

    # Arena boundary clipping
    half_w = arena_width / 2.0
    half_h = arena_height / 2.0

    x_min = -half_w + robot_radius
    x_max = half_w - robot_radius
    y_min = -half_h + robot_radius
    y_max = half_h - robot_radius

    wall_contact = False

    if x_new < x_min:
        x_new = x_min
        wall_contact = True
    elif x_new > x_max:
        x_new = x_max
        wall_contact = True

    if y_new < y_min:
        y_new = y_min
        wall_contact = True
    elif y_new > y_max:
        y_new = y_max
        wall_contact = True

    return x_new, y_new, theta_new, wall_contact


def resolve_obstacle_collisions(
    x: float,
    y: float,
    theta: float,
    obstacles: list[dict],
    robot_radius: float,
    max_iterations: int = 3,
) -> tuple[float, float, float, bool]:
    """Resolve physical collisions with circular obstacles.

    If the robot's body (center + robot_radius) overlaps any obstacle,
    push the robot radially outward to the obstacle surface. Iterates
    to handle chain reactions (pushing away from one obstacle may push
    into another).

    Args:
        x, y, theta: Current robot state.
        obstacles: List of obstacle dicts with 'x', 'y', 'radius' keys.
        robot_radius: Robot body radius.
        max_iterations: Max resolution passes (handles chain collisions).

    Returns:
        (x, y, theta, collision_occurred): Corrected position and whether
        any collision was resolved. Heading (theta) is never modified.
    """
    if not obstacles:
        return x, y, theta, False

    collision_occurred = False

    for _ in range(max_iterations):
        resolved_any = False
        for obs in obstacles:
            dx = x - obs["x"]
            dy = y - obs["y"]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = obs["radius"] + robot_radius

            if dist < min_dist:
                collision_occurred = True
                resolved_any = True

                if dist < 1e-10:
                    # Agent exactly at obstacle center — push along heading
                    nx = np.cos(theta)
                    ny = np.sin(theta)
                else:
                    # Push radially outward
                    nx = dx / dist
                    ny = dy / dist

                x = obs["x"] + nx * min_dist
                y = obs["y"] + ny * min_dist

        if not resolved_any:
            break

    return x, y, theta, collision_occurred


def clip_action(
    v: float,
    omega: float,
    v_max: float,
    omega_max: float,
) -> tuple[float, float]:
    """Clip control inputs to physical bounds.

    v is clipped to [0, v_max] (no reverse).
    omega is clipped to [-omega_max, omega_max].
    """
    v_clipped = np.clip(v, 0.0, v_max)
    omega_clipped = np.clip(omega, -omega_max, omega_max)
    return float(v_clipped), float(omega_clipped)
