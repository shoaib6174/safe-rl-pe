"""Step 3: scripted baseline agents + duel runner.

L1.5: evaluation baselines see truth — the full-obs greedy agents read the
env state directly. The FOV-limited variants act only on the 11-D observation
vector (plus their own last-seen memory) and exist to (a) demonstrate that
occlusion changes outcomes (Step 3 exit gate) and (b) serve as visual-gate
foils. Only learned policies are sensor-limited in headline comparisons.

Policies are stateful (last-seen memory for the FOV-limited variants); the
duel runner calls `reset()` on both policies at the start of every episode.
"""

from __future__ import annotations

import numpy as np

from pe2.env import PursuitEvasionEnv, wrap_angle
from pe2.obs import opponent_world_from_obs, own_pose_from_obs

K_OMEGA = 3.0  # P-gain on heading error (rad/s per rad)
SHADOW_MARGIN = 0.6  # shadow point sits this far past the obstacle surface
SHADOW_SEEK_RANGE = 5.0  # only seek a shadow point within this range (m)
THREAT_RANGE = 6.0  # only obstacle-seek when the pursuer is this close (m)
LAST_SEEN_TOL = 0.5  # arrived-at-last-seen distance (m)


def _clip_omega(value: float, cfg) -> float:
    return float(np.clip(value, -cfg.omega_max, cfg.omega_max))


class GreedyPursuer:
    """Full-obs baseline: turn toward the evader, always full speed."""

    uses_truth = True

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, env: PursuitEvasionEnv) -> tuple[float, float]:
        p, e = env.pursuer_pose, env.evader_pose
        err = wrap_angle(np.arctan2(e[1] - p[1], e[0] - p[0]) - p[2])
        return env.cfg.v_max, _clip_omega(K_OMEGA * err, env.cfg)


class GreedyEvader:
    """Full-obs baseline: flee the pursuer; when threatened, steer for the
    shadow point behind the nearest usable obstacle (LOS breaking)."""

    uses_truth = True

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, env: PursuitEvasionEnv) -> tuple[float, float]:
        p, e = env.pursuer_pose, env.evader_pose
        direction = e[:2] - p[:2]  # default: flee directly away
        pursuer_dist = float(np.hypot(*direction))
        if pursuer_dist < THREAT_RANGE:
            shadow = self._nearest_shadow_point(env)
            if shadow is not None:
                direction = shadow - e[:2]
        err = wrap_angle(np.arctan2(direction[1], direction[0]) - e[2])
        return env.cfg.evader_v_max, _clip_omega(K_OMEGA * err, env.cfg)

    @staticmethod
    def _nearest_shadow_point(env: PursuitEvasionEnv) -> np.ndarray | None:
        """Point past the obstacle surface on the line away from the pursuer."""
        p, e = env.pursuer_pose[:2], env.evader_pose[:2]
        best = None
        for ob in env.obstacles:
            c = np.array([ob["x"], ob["y"]])
            away = c - p
            d = float(np.hypot(*away))
            if d < 1e-6:
                continue
            point = c + (ob["radius"] + SHADOW_MARGIN) * away / d
            dist = float(np.hypot(*(point - e)))
            if dist < SHADOW_SEEK_RANGE and (best is None or dist < best[0]):
                best = (dist, point)
        return None if best is None else best[1]


class FovLimitedPursuer:
    """Acts on the observation only. Chases while visible; on losing sight,
    drives to the last-seen position, then scans in place (searching — it
    never acts on the true opponent state)."""

    uses_truth = False

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_seen: np.ndarray | None = None

    def act(self, obs: np.ndarray, env: PursuitEvasionEnv) -> tuple[float, float]:
        cfg = env.cfg
        own = own_pose_from_obs(obs, cfg)
        if obs[6] > 0.5:  # opponent visible
            self.last_seen = opponent_world_from_obs(obs, cfg)
            bearing_b = np.arctan2(obs[4], obs[3])  # body-frame bearing
            return cfg.v_max, _clip_omega(K_OMEGA * bearing_b, cfg)
        if self.last_seen is not None:
            to = self.last_seen - own[:2]
            if float(np.hypot(*to)) > LAST_SEEN_TOL:
                err = wrap_angle(np.arctan2(to[1], to[0]) - own[2])
                return cfg.v_max, _clip_omega(K_OMEGA * err, cfg)
            self.last_seen = None  # arrived, nothing here — start scanning
        return 0.0, 0.5 * cfg.omega_max  # scan in place


class FovLimitedEvader:
    """Acts on the observation only. Flees the observed pursuer; when the
    pursuer is not visible, keeps fleeing along the last-seen away-direction
    (stale dead-reckoning). Never seen: cruises a slow arc."""

    uses_truth = False

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_away: np.ndarray | None = None

    def act(self, obs: np.ndarray, env: PursuitEvasionEnv) -> tuple[float, float]:
        cfg = env.cfg
        own = own_pose_from_obs(obs, cfg)
        if obs[6] > 0.5:
            away = own[:2] - opponent_world_from_obs(obs, cfg)
            norm = float(np.hypot(*away))
            if norm > 1e-6:
                self.last_away = away / norm
        if self.last_away is None:
            return cfg.evader_v_max, 0.3 * cfg.omega_max
        err = wrap_angle(np.arctan2(self.last_away[1], self.last_away[0]) - own[2])
        return cfg.evader_v_max, _clip_omega(K_OMEGA * err, cfg)


def run_duel(
    env: PursuitEvasionEnv,
    pursuer,
    evader,
    seed: int,
) -> dict:
    """Run one episode; return the per-episode outcome log.

    `los_breaks` counts pursuer-view visible -> not-visible transitions
    (occlusion events) from the observation flag.
    """
    pursuer.reset()
    evader.reset()
    obs, _ = env.reset(seed=seed)
    los_prev = bool(obs["pursuer"][6] > 0.5)
    los_breaks = 0
    while True:
        v_p, w_p = pursuer.act(obs["pursuer"], env)
        v_e, w_e = evader.act(obs["evader"], env)
        obs, _, terminated, truncated, info = env.step([v_p, w_p, v_e, w_e])
        los = bool(obs["pursuer"][6] > 0.5)
        if los_prev and not los:
            los_breaks += 1
        los_prev = los
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "pursuer": type(pursuer).__name__,
        "evader": type(evader).__name__,
        "outcome": info["outcome"],
        "steps": env.current_step,
        "final_distance": float(info["distance"]),
        "los_breaks": los_breaks,
    }
