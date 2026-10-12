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
from pe2.metrics import ShieldMetrics
from pe2.obs import opponent_world_from_obs, own_pose_from_obs

K_OMEGA = 3.0  # P-gain on heading error (rad/s per rad)
SHADOW_MARGIN = 0.6  # shadow point sits this far past the obstacle surface
SHADOW_SEEK_RANGE = 5.0  # only seek a shadow point within this range (m)
THREAT_RANGE = 6.0  # only obstacle-seek when the pursuer is this close (m)
LAST_SEEN_TOL = 0.5  # arrived-at-last-seen distance (m)

# ObstacleHuggingEvader (Step 6 G3/G4 adversarial probe) constants.
HUG_CLEARANCE = 0.05  # standoff past (obstacle radius + agent radius) when hugging (m)
HUG_RANGE = 4.0  # only hug an obstacle whose SURFACE is this close (m)
HUG_THREAT_RANGE = 7.0  # hug at all only while the pursuer is this close (m)
WALL_HUG_BAND = 0.35  # distance from the wall the boundary hug drives to (m)
RADIAL_GAIN = 2.0  # weight of the radial correction against the tangential slide
RADIAL_CLIP = 1.0  # cap on the radial error (m) so the slide is never lost


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


class BoxRoundTripEvader(GreedyEvader):
    """`GreedyEvader` whose command takes the SAME float32 Box round-trip that
    `SingleAgentWrapper._scale_action` applies to a learner's action.

    V1-P compares two routes that put the scripted evader in different seats:
    in `SearchMDP` it is the opponent (its float64 (v, omega) reaches the env
    untouched), while in `SingleAgentWrapper(role="evader")` it stands in for
    the learner, so its omega passes through a float32 Box. That is a ~4e-8
    difference in the EVADER's command — nothing to do with the pursuer under
    test — and it compounds over 1200 steps (measured: seed 9001003 diverged
    at env step 14 with the pursuer commands still bit-identical). Giving the
    reference side this evader equalises that one path, so any remaining
    difference is the pursuer's, which is what V1-P is for.

    It lives here rather than in the tests because the runner's 200-seed V1-P
    preflight needs it too."""

    def act(self, obs: np.ndarray, env: PursuitEvasionEnv) -> tuple[float, float]:
        v, w = super().act(obs, env)
        cfg = env.cfg
        # exactly SingleAgentWrapper._scale_action for the evader role
        a = np.array(
            [2.0 * v / cfg.evader_v_max - 1.0, w / cfg.omega_max], dtype=np.float32
        )
        a64 = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
        return (a64[0] + 1.0) / 2.0 * cfg.evader_v_max, a64[1] * cfg.omega_max


class StationaryEvader:
    """An evader that never moves — the reference opponent for CG-S1
    (docs/pathway/pursuit_strategy_plan_2026-10-12.md section 4): with the
    evader frozen, every member of the S23 strategy family must act
    bit-identically to `PlannerInterceptLaw`."""

    uses_truth = True

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, env: PursuitEvasionEnv) -> tuple[float, float]:
        return 0.0, 0.0


OPEN_SPACE_HEADINGS = 16  # candidate directions, evenly spaced
OPEN_SPACE_HORIZON = 1.0  # seconds of straight-line lookahead


class OpenSpaceEvader:
    """Full-obs evader that flees WITHOUT cornering itself (F1 spec §2).

    D104: `GreedyEvader` flees directly away from the pursuer with no wall
    awareness, so it pins itself against the arena edge and a slow constant
    arc collects it — a randomly initialised network scored 0.312 against it
    while the hand-written `FovLimitedPursuer` scored 0.234. That makes a
    single-opponent capture rate useless for ordering pursuers.

    This evader considers OPEN_SPACE_HEADINGS evenly spaced directions,
    projects each one OPEN_SPACE_HORIZON seconds ahead at top speed, discards
    any whose endpoint leaves the arena or enters an obstacle, and takes the
    survivor that ends furthest from the pursuer. Deterministic, stateless,
    and it never chooses a heading that ends in a wall while a free one
    exists -- which is exactly the failure a circling pursuer exploits.
    """

    uses_truth = True

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, env: PursuitEvasionEnv) -> tuple[float, float]:
        cfg = env.cfg
        e, p = env.evader_pose, env.pursuer_pose
        reach = cfg.evader_v_max * OPEN_SPACE_HORIZON
        lim = cfg.arena_size / 2.0 - cfg.agent_radius
        best_free, best_any = None, None
        for i in range(OPEN_SPACE_HEADINGS):
            theta = -np.pi + 2.0 * np.pi * i / OPEN_SPACE_HEADINGS
            end = e[:2] + reach * np.array([np.cos(theta), np.sin(theta)])
            score = float(np.hypot(*(end - p[:2])))  # distance from the pursuer
            if best_any is None or score > best_any[0]:
                best_any = (score, theta)
            if abs(end[0]) > lim or abs(end[1]) > lim:
                continue  # ends outside the arena
            if any(
                np.hypot(end[0] - o["x"], end[1] - o["y"])
                < o["radius"] + cfg.agent_radius
                for o in env.obstacles
            ):
                continue  # ends inside an obstacle
            if best_free is None or score > best_free[0]:
                best_free = (score, theta)
        # Every heading blocked (pinned in a corner): fall back to the best of
        # them rather than stopping, so the evader still runs.
        _score, theta = best_free if best_free is not None else best_any
        err = wrap_angle(theta - e[2])
        return cfg.evader_v_max, _clip_omega(K_OMEGA * err, cfg)


class ObstacleHuggingEvader:
    """Step 6 G3/G4 adversarial probe: flee with something solid in between.

    Full-obs (L1.5: scripted baselines see truth). Built to stress the
    shield, not to be a good evader: while the pursuer is within
    `HUG_THREAT_RANGE` it drives onto a hug circle `HUG_CLEARANCE` outside
    the nearest obstacle's swept radius and orbits toward the point
    diametrically opposite the pursuer, so the obstacle stays between the
    two; with no obstacle in range it presses against the nearest arena wall
    and slides along it away from the pursuer. Both modes are a tangential
    slide plus a clipped radial correction, so the approach is a spiral that
    is already orbiting when it arrives rather than a head-on dive.

    The point of the probe is what it does to the OTHER robot: a pursuer
    that beelines at an evader glued to an obstacle or a wall drives itself
    into that obstacle or wall. G4 (pre-registered) requires exactly that of
    the unshielded arm; G3 requires the shielded pursuer to survive it with
    zero contact steps.
    """

    uses_truth = True

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, env: PursuitEvasionEnv) -> tuple[float, float]:
        cfg = env.cfg
        p, e = env.pursuer_pose[:2], env.evader_pose[:2]
        away = e - p
        direction = None
        if float(np.hypot(*away)) < HUG_THREAT_RANGE:
            direction = self._obstacle_hug(e, p, env)
            if direction is None:
                direction = self._boundary_hug(e, p, env)
        if direction is None:
            direction = away  # nothing worth hiding behind: plain flee
        err = wrap_angle(np.arctan2(direction[1], direction[0]) - env.evader_pose[2])
        return cfg.evader_v_max, _clip_omega(K_OMEGA * err, cfg)

    @staticmethod
    def _obstacle_hug(e, p, env: PursuitEvasionEnv):
        """Orbit the nearest in-range obstacle toward the pursuer's antipode."""
        best = None
        for ob in env.obstacles:
            surface = float(np.hypot(e[0] - ob["x"], e[1] - ob["y"])) - ob["radius"]
            if surface < HUG_RANGE and (best is None or surface < best[0]):
                best = (surface, ob)
        if best is None:
            return None
        ob = best[1]
        c = np.array([ob["x"], ob["y"]])
        r_vec = e - c
        r = float(np.hypot(*r_vec))
        if r < 1e-9:
            return None  # degenerate: standing on the obstacle centre
        r_hat = r_vec / r
        hug_r = ob["radius"] + env.cfg.agent_radius + HUG_CLEARANCE
        antipode = np.arctan2(p[1] - c[1], p[0] - c[0]) + np.pi
        delta = wrap_angle(antipode - np.arctan2(r_vec[1], r_vec[0]))
        sign = 1.0 if delta >= 0.0 else -1.0  # exact tie orbits CCW
        tangent = sign * np.array([-r_hat[1], r_hat[0]])
        radial = np.clip(hug_r - r, -RADIAL_CLIP, RADIAL_CLIP) * r_hat
        return tangent + RADIAL_GAIN * radial

    @staticmethod
    def _boundary_hug(e, p, env: PursuitEvasionEnv):
        """Press against the nearest wall and slide along it away from p."""
        cfg = env.cfg
        lim = cfg.arena_size / 2.0 - cfg.agent_radius
        walls = [
            (np.array([1.0, 0.0]), lim - e[0]),
            (np.array([-1.0, 0.0]), e[0] + lim),
            (np.array([0.0, 1.0]), lim - e[1]),
            (np.array([0.0, -1.0]), e[1] + lim),
        ]
        outward, dist = min(walls, key=lambda w: w[1])
        tangent = np.array([-outward[1], outward[0]])
        # Slide the way that opens distance from the pursuer; exact tie -> +.
        sign = 1.0 if float(np.dot(tangent, e - p)) >= 0.0 else -1.0
        radial = np.clip(dist - WALL_HUG_BAND, -RADIAL_CLIP, RADIAL_CLIP) * outward
        return sign * tangent + RADIAL_GAIN * radial


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


PATROL_RADIUS = 2.5  # m — an orbit at this radius sweeps a 10x10 arena with
PATROL_LEAD = 0.6  # rad — a 90 deg cone at 8 m range; lead keeps it moving on


class SearchingPursuer:
    """`FovLimitedPursuer` with ONE branch changed: it searches by MOVING.

    D105 measured the old teacher standing still on 92.6 % of its steps, and
    its demonstrations at 83.6 % 'stand still and rotate' with a mean
    commanded speed of 0.164 m/s. Every learner in four rounds cloned that.
    Chasing and driving to the last-seen point are copied unchanged; only
    the fallback differs.

    The orbit is a pure function of OWN POSE, which the observation carries
    in full. That is deliberate (T2 spec §2): steering toward the stalest
    cell of a coverage map would search better but would add a second hidden
    variable, and a memoryless clone already cannot follow this teacher's
    `last_seen` (D79). This round fixes standing still, not hidden state,
    and keeping them separate is what makes the outcome readable.
    """

    uses_truth = False

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_seen: np.ndarray | None = None

    def act(self, obs: np.ndarray, env: PursuitEvasionEnv) -> tuple[float, float]:
        cfg = env.cfg
        own = own_pose_from_obs(obs, cfg)
        if obs[6] > 0.5:  # opponent visible — unchanged
            self.last_seen = opponent_world_from_obs(obs, cfg)
            bearing_b = np.arctan2(obs[4], obs[3])
            return cfg.v_max, _clip_omega(K_OMEGA * bearing_b, cfg)
        if self.last_seen is not None:  # go to last-seen — unchanged
            to = self.last_seen - own[:2]
            if float(np.hypot(*to)) > LAST_SEEN_TOL:
                err = wrap_angle(np.arctan2(to[1], to[0]) - own[2])
                return cfg.v_max, _clip_omega(K_OMEGA * err, cfg)
            self.last_seen = None
        # THE ONE CHANGED BRANCH: orbit instead of standing still.
        angle = float(np.arctan2(own[1], own[0]))
        target = PATROL_RADIUS * np.array(
            [np.cos(angle + PATROL_LEAD), np.sin(angle + PATROL_LEAD)]
        )
        to = target - own[:2]
        if float(np.hypot(*to)) < 1e-9:
            return cfg.v_max, 0.0
        err = wrap_angle(np.arctan2(to[1], to[0]) - own[2])
        return cfg.v_max, _clip_omega(K_OMEGA * err, cfg)


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

    Violations use the pre-registered Step 6 semantics: an env contact step
    (wall or obstacle) for the role in question. Unlike `run_shielded_duel`,
    this runner has no shielded role, so BOTH roles are counted separately
    (`pursuer_violations`, `evader_violations`) plus their sum
    (`total_violations`) — an unshielded arm has no privileged role, and the
    gate's negative control picks whichever role the shielded arm shielded.
    """
    pursuer.reset()
    evader.reset()
    obs, _ = env.reset(seed=seed)
    los_prev = bool(obs["pursuer"][6] > 0.5)
    los_breaks = 0
    pursuer_violations = 0
    evader_violations = 0
    while True:
        v_p, w_p = pursuer.act(obs["pursuer"], env)
        v_e, w_e = evader.act(obs["evader"], env)
        obs, _, terminated, truncated, info = env.step([v_p, w_p, v_e, w_e])
        pursuer_violations += int(
            info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]
        )
        evader_violations += int(
            info["evader_wall_contact"] or info["evader_obstacle_contact"]
        )
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
        "pursuer_violations": pursuer_violations,
        "evader_violations": evader_violations,
        "total_violations": pursuer_violations + evader_violations,
    }


def run_shielded_duel(
    env: PursuitEvasionEnv,
    pursuer,
    evader,
    shield,
    seed: int,
    shielded_role: str = "pursuer",
) -> dict:
    """Run one episode with `shield` filtering one role's actions.

    Returns the `run_duel` log (including its per-role violation counts, so a
    shielded log and an unshielded log can be compared key by key) plus
    `shielded_role`, `violations` (env contact steps of the SHIELDED role —
    the pre-registered Step 6 semantics) and `shield`
    (ShieldMetrics.summary() for the episode). The shield sees the TRUE pose
    and the static map; the policies see their observations.

    For the pre-registered unshielded negative control, pass a
    `pe2.shield.NoOpShield`: the log then has the same shape with a shield
    block whose tier histogram is entirely "none".
    """
    if shielded_role not in ("pursuer", "evader"):
        raise ValueError(
            f"shielded_role must be 'pursuer' or 'evader', got {shielded_role!r}"
        )
    pursuer.reset()
    evader.reset()
    obs, _ = env.reset(seed=seed)
    los_prev = bool(obs["pursuer"][6] > 0.5)
    los_breaks = 0
    metrics = ShieldMetrics()
    pursuer_violations = 0
    evader_violations = 0
    v_cap = env.cfg.v_max if shielded_role == "pursuer" else env.cfg.evader_v_max
    while True:
        v_p, w_p = pursuer.act(obs["pursuer"], env)
        v_e, w_e = evader.act(obs["evader"], env)
        if shielded_role == "pursuer":
            (v_p, w_p), sinfo = shield.filter(
                (v_p, w_p), env.pursuer_pose, env.obstacles, v_max=v_cap
            )
        else:
            (v_e, w_e), sinfo = shield.filter(
                (v_e, w_e), env.evader_pose, env.obstacles, v_max=v_cap
            )
        obs, _, terminated, truncated, info = env.step([v_p, w_p, v_e, w_e])
        pursuer_violations += int(
            info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]
        )
        evader_violations += int(
            info["evader_wall_contact"] or info["evader_obstacle_contact"]
        )
        contact = bool(
            info[f"{shielded_role}_wall_contact"]
            or info[f"{shielded_role}_obstacle_contact"]
        )
        metrics.record(sinfo, violation=contact)
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
        "pursuer_violations": pursuer_violations,
        "evader_violations": evader_violations,
        "total_violations": pursuer_violations + evader_violations,
        "shielded_role": shielded_role,
        # The pre-registered gate quantity: contact steps of the SHIELDED role.
        "violations": (
            pursuer_violations if shielded_role == "pursuer" else evader_violations
        ),
        "shield": metrics.summary(),
    }
