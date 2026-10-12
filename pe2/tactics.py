"""S24.0 Step 1 — the tactic library for both roles.

Spec: `docs/pathway/tactics_plan_2026-10-13.md` sections 1 and 5.

A **tactic** is a closed-loop scripted controller with a name, a precondition
(`applicable`) and a termination predicate: `act` returns `None` to say DONE.
Nothing here selects, learns or scores; the v0 rule table and the learned
selector are later steps, and `select` is injected into the two adapters.

Every tactic WRAPS a primitive this repo already has (`GreedyEvader`,
`ObstacleHuggingEvader`'s hug-circle mode, `PlannerInterceptLaw` /
`LeadPlannerInterceptLaw` and their A* planning, `OrbitSearch`, `is_visible`)
and adds only the precondition, the termination predicate and — for the two
mirrored-goal tactics — the goal point.

The binding rule (plan section 1): **a tactic never reads the speed ratio or
any speed of its opponent.** It commands its OWN top speed and is written to
work at any ratio and to fail honestly where it cannot. `evader_speed_ratio`
does not appear in this module, and no pursuer tactic reads `evader_v_max`;
`tests/pe2/test_tactics.py` asserts both on the source and at runtime.

L1.5: scripted controllers are full-state. Evader tactics take
`act(obs, env)` like `GreedyEvader` (the `SingleAgentWrapper` opponent
interface); pursuer tactics take `act(obs)` and read the env they were handed
at `reset` like `LeadPlannerInterceptLaw` (the law interface). That split is
the two call signatures the two seats require, not two notions of a tactic.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np

from pe2.config import EnvConfig
from pe2.planner import LeadPlannerInterceptLaw, PlannerInterceptLaw
from pe2.scripted import (
    HUG_CLEARANCE,
    HUG_THREAT_RANGE,
    RADIAL_CLIP,
    RADIAL_GAIN,
    GreedyEvader,
    ObstacleHuggingEvader,
)
from pe2.sensing import is_visible, segment_intersects_disk, wrap_angle
from pe2.structured import K_OMEGA, OrbitSearch, _clip_omega, _normalize

ARRIVAL_TOL = 0.3  # m — "arrived at the goal" for HOP and AMBUSH
COUNTER_ORBIT_MAX_STEPS = 80  # then the selector decides again
COUNTER_ORBIT_CLOSE = 1.5  # m — clear LOS inside this ends COUNTER-ORBIT
AMBUSH_MAX_STEPS = 120
AMBUSH_CLOSE = 2.0  # m — the evader walking into the gap ends AMBUSH
AMBUSH_GAP_RANGE = 5.0  # m — only ambush a gap this close
# S24.0 Step 4 (additive): `Sweep` steers at a dead-reckoned point this recent
# (5 s at dt = 0.05) and falls back to `OrbitSearch` beyond it — the brief's
# "while its age is < 100 steps, else OrbitSearch".
BELIEF_FRESH_STEPS = 100
# Fix round 1 (D149): steps the search spends on one stale cell before giving
# it up for the next-stalest — the guard against a target that can be reached
# but, from the bearings the route allows, never actually observed.
STALE_TARGET_MAX_STEPS = 120


# ---------------------------------------------------------------- geometry
def obstacle_between(own, opp, obstacle: dict, inflate: float = 0.0) -> bool:
    """The segment own->opp passes through obstacle's inflated disc.

    `inflate` is normally `agent_radius`, so "between" means the same disc the
    planner and the shield treat as solid. Tangency does not count (the
    `pe2.sensing` convention, so grazing passes do not flicker)."""
    return segment_intersects_disk(
        (float(own[0]), float(own[1])),
        (float(opp[0]), float(opp[1])),
        (obstacle["x"], obstacle["y"]),
        obstacle["radius"] + inflate,
    )


def hug_radius(obstacle: dict, agent_radius: float) -> float:
    """`ObstacleHuggingEvader`'s hug circle: HUG_CLEARANCE past the inflated
    radius. The same expression as `_obstacle_hug`, named once."""
    return obstacle["radius"] + agent_radius + HUG_CLEARANCE


def far_side_point(obstacle: dict, from_xy, agent_radius: float) -> np.ndarray:
    """The point on obstacle's hug circle diametrically away from `from_xy` —
    where an evader stands to put the obstacle between the two."""
    c = np.array([obstacle["x"], obstacle["y"]], dtype=float)
    away = c - np.asarray(from_xy, dtype=float)[:2]
    n = float(np.hypot(*away))
    unit = np.array([1.0, 0.0]) if n < 1e-9 else away / n
    return c + hug_radius(obstacle, agent_radius) * unit


def gap_point(obstacle: dict, others: list[dict], lim: float) -> np.ndarray:
    """Midpoint of the NARROWEST gap between `obstacle` and either another
    obstacle or an arena wall.

    `lim` is the inset arena bound (`arena_size / 2 - agent_radius`, the env's
    own `_x_max`). The gap width is surface to surface and may be negative when
    the two already overlap; the midpoint is still well defined, so the caller —
    not this function — decides whether such a gap is worth standing in.
    Candidates are scanned in list order and then +x, -x, +y, -y, so ties
    resolve deterministically."""
    c = np.array([obstacle["x"], obstacle["y"]], dtype=float)
    r = float(obstacle["radius"])
    best: tuple[float, np.ndarray] | None = None
    for o in others:
        d = np.array([o["x"], o["y"]], dtype=float) - c
        n = float(np.hypot(*d))
        if n < 1e-9:
            continue  # concentric: no gap between them
        width = n - r - float(o["radius"])
        point = c + (r + width / 2.0) * d / n
        if best is None or width < best[0]:
            best = (width, point)
    for outward in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        out = np.array(outward, dtype=float)
        width = lim - float(out @ c) - r
        point = c + (r + width / 2.0) * out
        if best is None or width < best[0]:
            best = (width, point)
    return best[1]


def counter_orbit_direction(own_xy, opp_xy, obstacle: dict) -> float:
    """The angular direction about the obstacle AWAY from the shorter way
    round toward the opponent: +1 counter-clockwise, -1 clockwise.

    The shorter way is the one whose angular step takes own position toward
    the opponent's, which is what `Intercept`'s planner already does.

    This is a ONE-OFF choice, not a per-step one. Its input
    `wrap(phi_opp - phi_own)` sits exactly at the +-pi branch cut when the
    obstacle is between the two -- which is `CounterOrbit`'s own precondition
    -- so recomputing it every step flips the direction on alternating steps
    and the manoeuvre never progresses (measured: 0/50 starts reached the
    evader, including against a STATIONARY one). `CounterOrbit` therefore
    commits to the direction when the span begins and holds it."""
    c = np.array([obstacle["x"], obstacle["y"]], dtype=float)
    own = np.asarray(own_xy, dtype=float)[:2]
    opp = np.asarray(opp_xy, dtype=float)[:2]
    phi_own = float(np.arctan2(own[1] - c[1], own[0] - c[0]))
    phi_opp = float(np.arctan2(opp[1] - c[1], opp[0] - c[0]))
    short = 1.0 if wrap_angle(phi_opp - phi_own) >= 0.0 else -1.0
    return -short


def bearing_side(own_xy, opp_xy, point) -> float:
    """Which side of the direct bearing own->opp the bearing own->point is on:
    +1 left, -1 right, 0 exactly on it."""
    own = np.asarray(own_xy, dtype=float)[:2]
    to_opp = np.asarray(opp_xy, dtype=float)[:2] - own
    to_pt = np.asarray(point, dtype=float)[:2] - own
    delta = wrap_angle(
        float(np.arctan2(to_pt[1], to_pt[0])) - float(np.arctan2(to_opp[1], to_opp[0]))
    )
    return float(np.sign(delta))


def unwrap_angle(previous: float, phi: float) -> float:
    """`phi` continued from `previous` without a 2pi jump, so a running total
    of angular progress has no branch cut."""
    return float(previous) + wrap_angle(float(phi) - float(previous))


def counter_orbit_goal(
    obstacle: dict,
    agent_radius: float,
    phi_from: float,
    direction: float,
    phi_target: float,
) -> tuple[np.ndarray, float]:
    """The next waypoint on obstacle's hug circle on the way round from
    `phi_from` to `phi_target` in `direction`, and the arc still to go.

    All three angles are UNWRAPPED (see `unwrap_angle`). `phi_target` is the
    angle diametrically opposite own position when the span began -- where an
    evader keeping the obstacle between the two has to be. Both ways round to
    it are the same length, so A* alone cannot be told which to take; the
    waypoint is therefore capped a quarter turn ahead in the committed
    `direction`, which forces one way and still closes onto the terminal point
    as the remaining arc shrinks.

    Taking the remaining arc modulo 2pi instead of tracking it unwrapped does
    not work here: it starts at exactly pi, so the first step either way
    decides whether it reads as pi - d or pi + d, and the pursuer that drifts
    the wrong way while driving onto the circle reads as already arrived and
    steers at its own position (measured: it swept the committed way on only
    13 of 30 starts, and the chatter before the direction was committed at all
    reached the evader on 0 of 50)."""
    remaining = max(0.0, float(direction) * (float(phi_target) - float(phi_from)))
    phi = float(phi_from) + float(direction) * min(np.pi / 2.0, remaining)
    c = np.array([obstacle["x"], obstacle["y"]], dtype=float)
    goal = c + hug_radius(obstacle, agent_radius) * np.array([np.cos(phi), np.sin(phi)])
    return goal, remaining


def wall_slide_direction(own_xy, opp_xy, lim: float, band: float) -> np.ndarray:
    """Press toward the nearest arena wall and slide along it away from the
    opponent: the tangent the opponent is NOT on, plus a clipped radial pull
    toward a standoff of `band` past the inset bound `lim`.

    The same shape as `ObstacleHuggingEvader._boundary_hug` (tangent plus
    RADIAL_GAIN times a RADIAL_CLIP-limited radial error, exact tie slides +),
    but with the band as an argument instead of that law's `WALL_HUG_BAND`.
    The law is a gate opponent in three completed rounds and stays frozen;
    its 0.35 m standoff from a bound already inset by `agent_radius` is 0.50 m
    from the wall, which is not "pressed against" it."""
    own = np.asarray(own_xy, dtype=float)[:2]
    opp = np.asarray(opp_xy, dtype=float)[:2]
    walls = (
        (np.array([1.0, 0.0]), lim - own[0]),
        (np.array([-1.0, 0.0]), own[0] + lim),
        (np.array([0.0, 1.0]), lim - own[1]),
        (np.array([0.0, -1.0]), own[1] + lim),
    )
    outward, dist = min(walls, key=lambda w: w[1])
    tangent = np.array([-outward[1], outward[0]])
    sign = 1.0 if float(tangent @ (own - opp)) >= 0.0 else -1.0
    radial = np.clip(dist - band, -RADIAL_CLIP, RADIAL_CLIP) * outward
    return sign * tangent + RADIAL_GAIN * radial


def _evader_command(direction, pose, cfg: EnvConfig) -> tuple[float, float]:
    """`ObstacleHuggingEvader.act`'s final two lines: full own speed, P-gain on
    the heading error toward `direction`."""
    err = wrap_angle(float(np.arctan2(direction[1], direction[0])) - pose[2])
    return cfg.evader_v_max, _clip_omega(K_OMEGA * err, cfg)


class _OneObstacle:
    """An env view carrying exactly one obstacle. `_obstacle_hug` reads only
    `cfg` and `obstacles`, so this pins the hug law to obstacle k instead of
    letting it pick the nearest — the hug law itself is unchanged."""

    __slots__ = ("cfg", "obstacles")

    def __init__(self, cfg: EnvConfig, obstacles: list[dict]) -> None:
        self.cfg = cfg
        self.obstacles = obstacles


# ------------------------------------------------------------------- bases
class Tactic:
    """Name, precondition, controller, termination. `steps` counts the steps
    of the CURRENT span (the adapters zero it on entry); `act` returning None
    is DONE."""

    name: str = "TACTIC"

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.steps = 0

    def reset(self, env=None) -> None:
        """Per-episode. `env` is the static map, known by convention."""
        self.steps = 0

    def enter(self) -> None:
        """Called by an adapter when this tactic becomes current. It zeroes the
        SPAN counter only: a wrapped law's last-seen memory is memory of the
        world, not of the span, and survives a switch."""
        self.steps = 0

    def applicable(self, obs, env) -> bool:
        return True


class EvaderTactic(Tactic):
    """`act(obs, env) -> (v, omega) | None` — the opponent interface of
    `SingleAgentWrapper(role="pursuer")`, as `GreedyEvader` has."""

    uses_truth = True  # L1.5

    def act(self, obs, env) -> tuple[float, float] | None:
        raise NotImplementedError


class PursuerTactic(Tactic):
    """`act(obs) -> np.ndarray | None`, normalised like `InterceptLaw`. The env
    is read at `reset` for the map and the true state (L1.5), exactly as
    `LeadPlannerInterceptLaw` keeps `self._env`."""

    def __init__(self, cfg: EnvConfig) -> None:
        super().__init__(cfg)
        self.env = None

    def reset(self, env=None) -> None:
        super().reset(env)
        self.env = env

    def act(self, obs) -> np.ndarray | None:
        raise NotImplementedError


# ---------------------------------------------------------- evader tactics
class Flee(EvaderTactic):
    """`GreedyEvader`, unchanged. The default: never DONE."""

    name = "FLEE"

    def __init__(self, cfg: EnvConfig) -> None:
        super().__init__(cfg)
        self.inner = GreedyEvader()

    def reset(self, env=None) -> None:
        super().reset(env)
        self.inner.reset()

    def act(self, obs, env):
        self.steps += 1
        return self.inner.act(obs, env)


class Orbit(EvaderTactic):
    """Keep obstacle k between itself and the pursuer: `ObstacleHuggingEvader`'s
    hug-circle mode PINNED to obstacle k rather than the nearest one.

    DONE when the pursuer is farther than HUG_THREAT_RANGE or k is no longer
    within HUG_RANGE of the evader's position (the hug law's own range test,
    which is also the precondition)."""

    def __init__(self, cfg: EnvConfig, k: int) -> None:
        super().__init__(cfg)
        self.k = int(k)
        self.name = f"ORBIT({self.k})"

    def _direction(self, env):
        if self.k >= len(env.obstacles):
            return None
        view = _OneObstacle(env.cfg, [env.obstacles[self.k]])
        return ObstacleHuggingEvader._obstacle_hug(
            env.evader_pose[:2], env.pursuer_pose[:2], view
        )

    def applicable(self, obs, env) -> bool:
        return self._direction(env) is not None

    def act(self, obs, env):
        gap = env.evader_pose[:2] - env.pursuer_pose[:2]
        if float(np.hypot(*gap)) > HUG_THREAT_RANGE:
            return None
        direction = self._direction(env)
        if direction is None:
            return None
        self.steps += 1
        return _evader_command(direction, env.evader_pose, env.cfg)


class Hop(EvaderTactic):
    """Plan to the far side of obstacle j from the pursuer so LOS breaks.

    A* on the known static map (`PlannerInterceptLaw.plan`), steering at the
    smoothed path's lookahead point; the goal is recomputed from the pursuer's
    CURRENT position every step, so this is closed loop. DONE on arrival
    (ARRIVAL_TOL) or the step the pursuer can no longer see the evader,
    whichever comes first.

    `applicable` iff obstacle j is not already the one between the two and the
    planner has a free goal cell for the far-side point (amendment A2.1's
    notion of a free goal, i.e. exactly the condition under which `plan` will
    not fall back to direct steering)."""

    def __init__(self, cfg: EnvConfig, j: int) -> None:
        super().__init__(cfg)
        self.j = int(j)
        self.name = f"HOP({self.j})"
        self.planner = PlannerInterceptLaw(cfg)
        self._map_key: tuple | None = None
        self.goal: np.ndarray | None = None

    def reset(self, env=None) -> None:
        super().reset(env)
        self.goal = None
        self._sync(env)

    def _sync(self, env) -> None:
        """Load the static map into the planner when it changes. The evader
        seat (`SingleAgentWrapper`) calls `reset()` with no env, so the map
        arrives on the first `act` instead."""
        key = (
            None
            if env is None
            else tuple((o["x"], o["y"], o["radius"]) for o in env.obstacles)
        )
        if key != self._map_key:
            self.planner.set_map([] if env is None else env.obstacles)
            self._map_key = key

    def _goal_point(self, env) -> np.ndarray:
        return far_side_point(
            env.obstacles[self.j], env.pursuer_pose, env.cfg.agent_radius
        )

    def _seen(self, env) -> bool:
        return is_visible(
            env.pursuer_pose, env.evader_pose[:2], env.obstacles, env.cfg.sensing
        )

    def applicable(self, obs, env) -> bool:
        if self.j >= len(env.obstacles):
            return False
        self._sync(env)
        if obstacle_between(
            env.evader_pose,
            env.pursuer_pose,
            env.obstacles[self.j],
            env.cfg.agent_radius,
        ):
            return False
        return self.planner._snap_goal(self._goal_point(env)) is not None

    def act(self, obs, env):
        self._sync(env)
        if self.j >= len(env.obstacles) or not self._seen(env):
            return None  # LOS broken: the tactic's goal, reached
        own = env.evader_pose
        goal = self._goal_point(env)
        self.goal = goal
        if float(np.hypot(*(goal - own[:2]))) <= ARRIVAL_TOL:
            return None
        path = self.planner.plan(own[:2], goal)
        point = goal if path is None else self.planner.lookahead_point(path)
        direction = point - own[:2]
        if float(np.hypot(*direction)) < 1e-9:
            direction = goal - own[:2]
        self.steps += 1
        return _evader_command(direction, own, env.cfg)


class WallSlide(EvaderTactic):
    """Press against the nearest wall and slide along it away from the pursuer
    (`wall_slide_direction`). DONE when the pursuer is farther than
    HUG_THREAT_RANGE, which is also the precondition: there is always a nearest
    wall, so only the threat makes hugging one a tactic.

    `band` is this tactic's OWN standoff past the arena bound already inset by
    `agent_radius` — the default 0.10 m puts the body 0.25 m from the wall,
    which is what "pressed against it" means here. It is a constructor
    argument rather than `ObstacleHuggingEvader`'s `WALL_HUG_BAND` because that
    law is a frozen gate opponent and its 0.35 m leaves a 0.50 m standoff."""

    name = "WALL-SLIDE"

    def __init__(self, cfg: EnvConfig, band: float = 0.10) -> None:
        super().__init__(cfg)
        self.band = float(band)

    def _threatened(self, env) -> bool:
        gap = env.evader_pose[:2] - env.pursuer_pose[:2]
        return float(np.hypot(*gap)) <= HUG_THREAT_RANGE

    def applicable(self, obs, env) -> bool:
        return self._threatened(env)

    def act(self, obs, env):
        if not self._threatened(env):
            return None
        cfg = env.cfg
        direction = wall_slide_direction(
            env.evader_pose[:2],
            env.pursuer_pose[:2],
            cfg.arena_size / 2.0 - cfg.agent_radius,
            self.band,
        )
        self.steps += 1
        return _evader_command(direction, env.evader_pose, cfg)


class HoldHidden(EvaderTactic):
    """Stand still while the pursuer cannot see you. DONE the step visibility
    returns. Costs nothing and gives the pursuer's search nothing to follow."""

    name = "HOLD-HIDDEN"

    def _seen(self, env) -> bool:
        return is_visible(
            env.pursuer_pose, env.evader_pose[:2], env.obstacles, env.cfg.sensing
        )

    def applicable(self, obs, env) -> bool:
        return not self._seen(env)

    def act(self, obs, env):
        if self._seen(env):
            return None
        self.steps += 1
        return 0.0, 0.0


# --------------------------------------------------------- pursuer tactics
class _LawTactic(PursuerTactic):
    """A pursuer tactic that IS one of the S19/S23 laws: delegate `act`
    verbatim so the action is the law's bit for bit, and DONE is exactly the
    law's own None (visible branch gone, last-seen exhausted)."""

    def __init__(self, cfg: EnvConfig, law) -> None:
        super().__init__(cfg)
        self.law = law

    def reset(self, env=None) -> None:
        super().reset(env)
        self.law.reset(env)

    def act(self, obs):
        action = self.law.act(obs)
        if action is None:
            return None
        self.steps += 1
        return action


class Intercept(_LawTactic):
    """`PlannerInterceptLaw` (S19 `PLAN`)."""

    name = "INTERCEPT"

    def __init__(self, cfg: EnvConfig) -> None:
        super().__init__(cfg, PlannerInterceptLaw(cfg))


class Lead(_LawTactic):
    """`LeadPlannerInterceptLaw(tau=1.0)` (S23 `LEAD-1.0`, the best measured)."""

    name = "LEAD"

    def __init__(self, cfg: EnvConfig, tau: float = 1.0) -> None:
        super().__init__(cfg, LeadPlannerInterceptLaw(cfg, float(tau)))


class CounterOrbit(PursuerTactic):
    """Go the OTHER way round the obstacle the evader is keeping between you.

    `Intercept`'s planner already takes the shorter way round, so the goal here
    is the quarter-turn waypoint on k's hug circle on the far side of the
    direct bearing (`counter_orbit_goal`); `last_turn_sign` records which side
    that is and `last_steer_point` the first planned segment, so "it turns away
    from the direct bearing" is measurable rather than asserted.

    DONE when the line to the evader is clear and it is within
    COUNTER_ORBIT_CLOSE, or after COUNTER_ORBIT_MAX_STEPS (then the selector
    decides again). `applicable` iff obstacle k is between the two."""

    def __init__(self, cfg: EnvConfig, k: int) -> None:
        super().__init__(cfg)
        self.k = int(k)
        self.name = f"COUNTER-ORBIT({self.k})"
        self.law = PlannerInterceptLaw(cfg)  # planning + clearance geometry
        self.direction = 0.0  # committed at the start of each span
        self.phi_target = 0.0  # ditto: the antipode of the entry position
        self.phi_acc = 0.0  # own unwrapped angle about k, integrated
        self.last_remaining = 0.0
        self.last_goal: np.ndarray | None = None
        self.last_steer_point: np.ndarray | None = None
        self.last_turn_sign = 0.0

    def reset(self, env=None) -> None:
        super().reset(env)
        self.law.reset(env)
        self.direction = 0.0
        self.phi_target = 0.0
        self.phi_acc = 0.0
        self.last_remaining = 0.0
        self.last_goal = None
        self.last_steer_point = None
        self.last_turn_sign = 0.0

    def enter(self) -> None:
        super().enter()
        self.direction = 0.0  # re-chosen on the first act of the new span

    def applicable(self, obs, env) -> bool:
        if self.k >= len(env.obstacles):
            return False
        return obstacle_between(
            env.pursuer_pose,
            env.evader_pose,
            env.obstacles[self.k],
            env.cfg.agent_radius,
        )

    def act(self, obs):
        env = self.env
        own, evader = env.pursuer_pose, env.evader_pose
        if self.steps >= COUNTER_ORBIT_MAX_STEPS or self.k >= len(env.obstacles):
            return None
        close = float(np.hypot(*(evader[:2] - own[:2]))) <= COUNTER_ORBIT_CLOSE
        if close and self.law.is_clear(own[:2], evader[:2]):
            return None
        obstacle = env.obstacles[self.k]
        phi = float(np.arctan2(own[1] - obstacle["y"], own[0] - obstacle["x"]))
        if self.steps == 0:  # commit the way round and the terminal point
            self.direction = counter_orbit_direction(own[:2], evader[:2], obstacle)
            self.phi_acc = phi
            self.phi_target = phi + self.direction * np.pi
        else:
            self.phi_acc = unwrap_angle(self.phi_acc, phi)
        goal, remaining = counter_orbit_goal(
            obstacle,
            env.cfg.agent_radius,
            self.phi_acc,
            self.direction,
            self.phi_target,
        )
        self.last_goal = goal
        self.last_remaining = remaining
        self.last_turn_sign = bearing_side(own[:2], evader[:2], goal)
        # `PlannerInterceptLaw._steer_toward`, with the point it steers at
        # recorded: direct when the line is clear, else the planned lookahead.
        point = goal
        if not self.law.is_clear(own[:2], goal):
            path = self.law.plan(own[:2], goal)
            if path is None:
                self.law.n_unreachable += 1
            else:
                step_to = self.law.lookahead_point(path)
                if float(np.hypot(*(step_to - own[:2]))) >= 1e-9:
                    point = step_to
        self.last_steer_point = point
        self.steps += 1
        return self.law._steer_at(own, point)


class Ambush(PursuerTactic):
    """Stand in the gap beside the evader's obstacle and wait.

    The gap is `gap_point(obstacle k, the other obstacles, the inset bound)` —
    the narrowest passage out of k's neighbourhood. Plan there; on arrival hold
    position (v = 0) and turn to face the evader's side, so the hold is a
    watch, not a blind stop. DONE when the evader becomes visible within
    AMBUSH_CLOSE or after AMBUSH_MAX_STEPS. `applicable` iff the gap is within
    AMBUSH_GAP_RANGE.

    Reading: the plan writes `AMBUSH(gap)`; the gap is derived, so the tactic is
    parameterised by the obstacle k whose gap it is."""

    def __init__(self, cfg: EnvConfig, k: int) -> None:
        super().__init__(cfg)
        self.k = int(k)
        self.name = f"AMBUSH({self.k})"
        self.law = PlannerInterceptLaw(cfg)
        self.last_gap: np.ndarray | None = None

    def reset(self, env=None) -> None:
        super().reset(env)
        self.law.reset(env)
        self.last_gap = None

    def _gap(self, env) -> np.ndarray:
        others = [o for i, o in enumerate(env.obstacles) if i != self.k]
        lim = env.cfg.arena_size / 2.0 - env.cfg.agent_radius
        return gap_point(env.obstacles[self.k], others, lim)

    def applicable(self, obs, env) -> bool:
        if self.k >= len(env.obstacles):
            return False
        gap = self._gap(env)
        return float(np.hypot(*(gap - env.pursuer_pose[:2]))) <= AMBUSH_GAP_RANGE

    def act(self, obs):
        env, cfg = self.env, self.cfg
        own, evader = env.pursuer_pose, env.evader_pose
        if self.steps >= AMBUSH_MAX_STEPS or self.k >= len(env.obstacles):
            return None
        seen = obs[6] > 0.5
        if seen and float(np.hypot(*(evader[:2] - own[:2]))) <= AMBUSH_CLOSE:
            return None
        gap = self._gap(env)
        self.last_gap = gap
        self.steps += 1
        if float(np.hypot(*(gap - own[:2]))) <= ARRIVAL_TOL:
            to = evader[:2] - own[:2]  # hold, facing the evader's side (L1.5)
            err = wrap_angle(float(np.arctan2(to[1], to[0])) - own[2])
            return _normalize(0.0, _clip_omega(K_OMEGA * err, cfg), cfg)
        return self.law._steer_toward(own, gap)


class Sweep(PursuerTactic):
    """`OrbitSearch` — the teacher's orbit fallback as a search. Belief-driven
    once the belief exists (plan section 1). DONE when the evader is visible.

    S24.0 Step 4, ADDITIVE: the belief now exists and is dead reckoning
    (`pe2.belief.LastSeenBelief`; D146 shelved the particle filter), so `belief`
    is an optional constructor argument and, while its point estimate is
    younger than `BELIEF_FRESH_STEPS`, the search STEERS AT that point —
    through the planner, so the route round an obstacle is planned rather than
    driven into. Older than that, or absent, and this is `OrbitSearch`
    unchanged: with `belief=None` the class is bit-for-bit the Step 1 one
    (`tests/pe2/test_tactics.py` holds it to `OrbitSearch` by identity).

    S24.0 Step 4 fix round 1 (D149): with no fresh estimate — never seen, or
    `BELIEF_FRESH_STEPS` old — the search is no longer the fixed orbit. Given a
    `staleness` map (`pe2.rewards.SearchStalenessTracker`, updated once per step
    by whoever owns the observation stream), it steers at the STALEST reachable
    grid-cell centre and re-targets when that cell is observed, when it is
    reached, or after `STALE_TARGET_MAX_STEPS` on one target. `OrbitSearch`
    remains the fallback when there is no map to be stale (every candidate cell
    blocked) and when no `staleness` is given at all.

    Both hooks read only their object's own API — the belief's (`seen`, `mean`,
    `t_since_seen`) and the tracker's (`centres`, `last_observed`) — no speed
    and no speed ratio, as every tactic here.
    """

    name = "SWEEP"

    def __init__(self, cfg: EnvConfig, belief=None, staleness=None) -> None:
        super().__init__(cfg)
        self.search = OrbitSearch(cfg)
        self.belief = belief
        self.staleness = staleness
        self.law = PlannerInterceptLaw(cfg)  # the planned route to the estimate
        self.last_goal: np.ndarray | None = None
        self.target: int | None = None  # index into the staleness grid
        self._target_since = 0

    def reset(self, env=None) -> None:
        super().reset(env)
        self.law.reset(env)
        self.last_goal = None
        self.target = None
        self._target_since = 0

    def _fresh_point(self) -> np.ndarray | None:
        """The dead-reckoning estimate while it is fresh enough to steer at."""
        belief = self.belief
        if belief is None or not getattr(belief, "seen", False):
            return None
        age = getattr(belief, "t_since_seen", None)
        if age is None or int(age) >= BELIEF_FRESH_STEPS:
            return None
        return np.asarray(belief.mean, dtype=float)[:2]

    def _reachable_cell(self, index: int) -> bool:
        """A cell whose centre is inside an INFLATED obstacle can never be
        stood on or (from most bearings) seen, so it is not a search target —
        the planner's own blocked grid decides, not a second copy of it."""
        centre = self.staleness.centres[index]
        return not bool(self.law._blocked[self.law._cell(centre)])

    def _stale_point(self) -> np.ndarray | None:
        """The stalest reachable cell centre, held until it is observed,
        reached, or `STALE_TARGET_MAX_STEPS` old. Ties go to the nearest cell,
        so the search does not cross the arena for an equally stale one."""
        tracker, env = self.staleness, self.env
        if tracker is None or env is None:
            return None
        own = np.asarray(env.pursuer_pose, dtype=float)[:2]
        step = int(env.current_step)
        if self.target is not None:
            centre = np.asarray(tracker.centres[self.target], dtype=float)
            expired = step - self._target_since >= STALE_TARGET_MAX_STEPS
            observed = int(tracker.last_observed[self.target]) >= self._target_since
            arrived = float(np.hypot(*(centre - own))) <= ARRIVAL_TOL
            if not (expired or observed or arrived):
                return centre
        giving_up = self.target
        stale = step - np.asarray(tracker.last_observed, dtype=float)
        centres = np.asarray(tracker.centres, dtype=float)
        dist = np.hypot(centres[:, 0] - own[0], centres[:, 1] - own[1])
        for index in np.lexsort((dist, -stale)):
            index = int(index)
            if index == giving_up or not self._reachable_cell(index):
                continue
            self.target = index
            self._target_since = step
            return centres[index]
        self.target = None
        return None

    def applicable(self, obs, env) -> bool:
        return obs[6] <= 0.5

    def act(self, obs):
        if obs[6] > 0.5:
            return None
        self.steps += 1
        point = self._fresh_point()  # a fresh sighting outranks the map
        if point is not None:
            self.target = None
        else:
            point = self._stale_point()
        self.last_goal = point
        if point is None:
            return self.search.predict(obs)[0]
        return self.law._steer_toward(self.env.pursuer_pose, point)


# ------------------------------------------------------------------ adapters
class _TacticHolder:
    """Holds ONE current tactic, calls it until DONE, then asks `select` for
    the next. `select(obs, env, history)` is injected — the v0 rule table and
    the learned selector are later steps.

    `history` is the span log `(name, start_step, end_step)` with the OPEN span
    last and its `end_step` None, so the log and `current.name` never disagree.
    `step_index` counts this adapter's own calls."""

    def __init__(self, tactics: list, select: Callable) -> None:
        if not tactics:
            raise ValueError("a tactic library needs at least one tactic")
        self.tactics = list(tactics)
        self.select = select
        self.current = None
        self.history: list[tuple[str, int, int | None]] = []
        self.step_index = 0

    def _reset_common(self, env) -> None:
        for tactic in self.tactics:
            tactic.reset(env)
        self.current = None
        self.history = []
        self.step_index = 0

    def _enter(self, tactic) -> None:
        if tactic is None:
            raise ValueError("select returned None; it must return a tactic")
        tactic.enter()
        self.current = tactic
        self.history.append((tactic.name, self.step_index, None))

    def _close(self) -> None:
        name, start, _ = self.history[-1]
        self.history[-1] = (name, start, self.step_index)


class TacticOpponent(_TacticHolder):
    """Evader-side adapter: the `SingleAgentWrapper` opponent interface
    (`reset()`, `act(obs, env)`, `uses_truth`). A tactic that is DONE on
    arrival leaves the evader standing still for that step."""

    uses_truth = True

    def reset(self, env=None) -> None:
        self._reset_common(env)

    def act(self, obs, env) -> tuple[float, float]:
        if self.current is None:
            self._enter(self.select(obs, env, self.history))
        action = self.current.act(obs, env)
        if action is None:  # DONE: close the span and ask for the next tactic
            self._close()
            self._enter(self.select(obs, env, self.history))
            action = self.current.act(obs, env)
        self.step_index += 1
        return (0.0, 0.0) if action is None else action


class TacticLaw(_TacticHolder):
    """Pursuer-side adapter: the law interface (`reset(env=None)`,
    `act(obs) -> np.ndarray | None`). None propagates, meaning what it means
    for `InterceptLaw`: nothing to act on."""

    def __init__(self, tactics: list, select: Callable) -> None:
        super().__init__(tactics, select)
        self.env = None

    def reset(self, env=None) -> None:
        self.env = env
        self._reset_common(env)

    def act(self, obs) -> np.ndarray | None:
        if self.current is None:
            self._enter(self.select(obs, self.env, self.history))
        action = self.current.act(obs)
        if action is None:  # DONE: close the span and ask for the next tactic
            self._close()
            self._enter(self.select(obs, self.env, self.history))
            action = self.current.act(obs)
        self.step_index += 1
        return action


def round_robin(tactics: list) -> Callable:
    """A placeholder `select` that cycles the library in order."""
    counter = itertools.count()
    library = list(tactics)

    def select(obs, env, history):
        return library[next(counter) % len(library)]

    return select


def fixed(tactic) -> Callable:
    """A placeholder `select` that always returns the same tactic."""

    def select(obs, env, history):
        return tactic

    return select
