"""Step 2: observation vector under partial observability.

Layout (all values normalized to O(1); float32):

    [0]  own x / half_arena                — own pose is MANDATORY (L4:
    [1]  own y / half_arena                  partial observability is about
    [2]  own theta / pi                      the opponent, never self-state)
    [3]  opponent dx_body / arena_size     } opponent pose in OWN BODY FRAME,
    [4]  opponent dy_body / arena_size     } zeroed when not visible
    [5]  opponent dtheta / pi            }
    [6]  los_visible in {0.0, 1.0}         — explicit flag, so "never seen"
                                             is distinguishable from "at origin"
    [7 + 2k, 8 + 2k]  k-th nearest obstacle (dx_body, dy_body) / arena_size,
                      k = 0..n_obstacles-1

Obstacle features are NOT masked: the static map is known (only the opponent
is partially observed). This matches the shield architecture (the safety
filter sees the map/true state while the policy sees sensors) and avoids the
"blind near walls" failure of sensor-masked static features.
"""

import numpy as np
from gymnasium import spaces

from pe2.config import EnvConfig
from pe2.sensing import is_visible, wrap_angle


def obs_dim(cfg: EnvConfig) -> int:
    return 7 + 2 * cfg.n_obstacles


def own_pose_from_obs(obs: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    """Decode own pose [x, y, theta] (unnormalized) from an observation."""
    half = cfg.arena_size / 2.0
    return np.array([obs[0] * half, obs[1] * half, obs[2] * np.pi])


def opponent_world_from_obs(obs: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    """Decode the opponent's world-frame xy from a VISIBLE observation
    (obs[6] == 1.0). Meaningless when the opponent is not visible."""
    own = own_pose_from_obs(obs, cfg)
    dx_b = obs[3] * cfg.arena_size
    dy_b = obs[4] * cfg.arena_size
    c, s = np.cos(own[2]), np.sin(own[2])
    return own[:2] + np.array([c * dx_b - s * dy_b, s * dx_b + c * dy_b])


def observation_space(cfg: EnvConfig) -> spaces.Box:
    # Normalized features can slightly exceed 1 (body-frame dx of a corner
    # is sqrt(2) * half_arena / arena_size); +-2 is a safe bound.
    return spaces.Box(low=-2.0, high=2.0, shape=(obs_dim(cfg),), dtype=np.float32)


def _body_frame(
    own_pose: np.ndarray, point_xy: tuple[float, float]
) -> tuple[float, float]:
    """Transform a world-frame point into own body frame (x forward, y left)."""
    dx = point_xy[0] - own_pose[0]
    dy = point_xy[1] - own_pose[1]
    c = np.cos(own_pose[2])
    s = np.sin(own_pose[2])
    return c * dx + s * dy, -s * dx + c * dy


def build_observation(
    own_pose: np.ndarray,
    opponent_pose: np.ndarray,
    obstacles: list[dict],
    cfg: EnvConfig,
) -> np.ndarray:
    """PO observation for one agent. Pure function of (poses, obstacles, cfg)."""
    half = cfg.arena_size / 2.0
    out = np.zeros(obs_dim(cfg), dtype=np.float32)

    out[0] = own_pose[0] / half
    out[1] = own_pose[1] / half
    out[2] = own_pose[2] / np.pi

    visible = is_visible(
        own_pose, (opponent_pose[0], opponent_pose[1]), obstacles, cfg.sensing
    )
    out[6] = 1.0 if visible else 0.0
    if visible:
        dx_b, dy_b = _body_frame(own_pose, (opponent_pose[0], opponent_pose[1]))
        out[3] = dx_b / cfg.arena_size
        out[4] = dy_b / cfg.arena_size
        out[5] = wrap_angle(opponent_pose[2] - own_pose[2]) / np.pi

    nearest = sorted(
        obstacles,
        key=lambda o: (o["x"] - own_pose[0]) ** 2 + (o["y"] - own_pose[1]) ** 2,
    )[: cfg.n_obstacles]
    for k, obs in enumerate(nearest):
        dx_b, dy_b = _body_frame(own_pose, (obs["x"], obs["y"]))
        out[7 + 2 * k] = dx_b / cfg.arena_size
        out[8 + 2 * k] = dy_b / cfg.arena_size

    return out
