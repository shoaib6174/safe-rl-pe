"""Step 2: FOV + line-of-sight sensing geometry.

Combined masking (D6): a target is visible iff
    (a) inside the FOV cone (angle AND range), AND
    (b) the line of sight is not blocked by an obstacle.

Walls: the arena is convex and both agents are always inside it, so the
segment between two agents can never cross a wall — wall occlusion is a
no-op here by geometry (obstacle occlusion is the whole problem). If
non-convex wall layouts are ever added, this module needs a wall segment
intersection check.

Tangency convention: a segment exactly tangent to an obstacle disk does NOT
block LOS (strict inequality). This makes grazing passes stable instead of
flickering (plan Step 2 checklist).
"""

import numpy as np

from pe2.config import SensingConfig


def wrap_angle(theta: float) -> float:
    """Normalize angle to [-pi, pi)."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def in_fov(
    own_pose: np.ndarray,
    target_xy: tuple[float, float],
    fov_angle: float,
    fov_range: float,
) -> bool:
    """True if target is inside the FOV cone centered on own heading."""
    dx = target_xy[0] - own_pose[0]
    dy = target_xy[1] - own_pose[1]
    if np.hypot(dx, dy) > fov_range:
        return False
    bearing = np.arctan2(dy, dx)
    rel = wrap_angle(bearing - own_pose[2])
    return abs(rel) <= fov_angle / 2.0


def segment_intersects_disk(
    p1: tuple[float, float],
    p2: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> bool:
    """True if segment p1->p2 passes through the open disk (center, radius).

    Tangent (distance == radius exactly) returns False — stable grazing.
    """
    d = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    f = np.array([p1[0] - center[0], p1[1] - center[1]])
    seg_len_sq = float(d @ d)
    if seg_len_sq < 1e-18:
        # Degenerate segment: p1 == p2
        return bool(f @ f < radius * radius)
    # Closest point on the segment to the center, clamped to [0, 1]
    t = float(np.clip(-(f @ d) / seg_len_sq, 0.0, 1.0))
    closest = f + t * d
    return bool(closest @ closest < radius * radius)


def los_clear(
    p1: tuple[float, float],
    p2: tuple[float, float],
    obstacles: list[dict],
) -> bool:
    """True if no obstacle blocks the line of sight between two points."""
    return not any(
        segment_intersects_disk(p1, p2, (obs["x"], obs["y"]), obs["radius"])
        for obs in obstacles
    )


def is_visible(
    own_pose: np.ndarray,
    target_xy: tuple[float, float],
    obstacles: list[dict],
    sensing: SensingConfig,
) -> bool:
    """Combined masking (D6): FOV cone AND (optionally) LOS unblocked."""
    if not in_fov(own_pose, target_xy, sensing.fov_angle, sensing.fov_range):
        return False
    return not sensing.los_occlusion or los_clear(
        (own_pose[0], own_pose[1]), target_xy, obstacles
    )
