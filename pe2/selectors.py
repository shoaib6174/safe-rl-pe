"""S24.0 Step 4 — the scripted selectors v0 for both roles.

Spec: `docs/pathway/tactics_plan_2026-10-13.md` section 1 ("Selector" / "v0,
scripted": a readable rule table that "exists so the tactic game can be WATCHED
before anything is learned") and section 2 (S24.0).

A selector is `select(obs, env, memory) -> Tactic`, called by the Step 1
adapters at decision points — the current tactic's DONE, or every
`DECISION_PERIOD` = 40 steps, which is the plan's T and is imposed by offering
`select` the tactics through `_Capped` (the adapters themselves switch on DONE
only). The rules are ORDERED, one line each, with the reason as a comment.

Three rulings from the gates that preceded this file are binding here:

- **D145/D146: both particle-belief gates failed.** The memory is
  `pe2.belief.LastSeenBelief` (dead reckoning); `ParticleBelief` is shelved and
  is not imported.
- **D147/D148: both recogniser gates failed, so NO selector consumes recogniser
  LABELS** — for either role. Every rule that reads the opponent is written over
  `TacticRecogniser.features()`: the dimensionless 40-step window quantities,
  observable and speed-agnostic, with `valid` False below 10 visible steps. The
  brief's conditional "if gate 2 passed the evader may use pursuer labels"
  branch is therefore dead and is not written.
- **The Step 1 review's watch item: neither adapter checks `applicable()`.** So
  the selector returns only tactics whose `applicable(obs, env)` is True, and
  falls back to `Flee` / `Intercept`, which always are. An inapplicable pick is
  impossible by construction, not caught afterwards.

The brief names its feature conditions in the recogniser's PLANNED vocabulary
(`angular_sweep` in radians, `closing_rate` > 0). `features()` ships the
dimensionless window quantities instead, so each condition is re-expressed on
the key that carries that quantity and the reading is named at the constant:
`angular_sweep` about obstacle k -> `sweep_ratio` (the TANGENTIAL FRACTION of
the path about k, not an angle), `closing_rate > 0` -> `observer_trend < 0`
(the recogniser's trend is the distance to the observer, so closing is
negative), `dist_obstacle_trend` -> `obstacle_trend`, `dist_wall_trend` ->
`wall_trend`.

**The v0 rules are speed-agnostic: they never read the speed ratio.** The
memory carries it (the learned selector of S24.1 takes it as an input, plan
section 1) and v0 must not branch on it; `tests/pe2/test_selectors.py` asserts
that on the source of both rule tables and at runtime at 0.8 / 1.0 / 1.2.

L1.5 (scripted controllers are full-state) is unchanged and is where the rules
read geometry: the evader's rules read `env` poses as every scripted evader in
this repo does, and the pursuer's rules read the evader's position from its own
OBSERVATION (`opponent_world_from_obs`) and `env` only for the static map. The
pursuer tactics' own `applicable()` tests are Step 1's and full-state; they are
consulted here only on steps where the evader is visible, where truth and
observation agree.
"""

from __future__ import annotations

import numpy as np

from pe2.belief import LastSeenBelief
from pe2.config import EnvConfig
from pe2.obs import opponent_world_from_obs
from pe2.recogniser import TacticRecogniser
from pe2.scripted import HUG_RANGE, HUG_THREAT_RANGE
from pe2.tactics import (
    AMBUSH_CLOSE,
    Ambush,
    CounterOrbit,
    Flee,
    HoldHidden,
    Hop,
    Intercept,
    Lead,
    Orbit,
    Sweep,
    WallSlide,
)

WINDOW = 40  # the recogniser's window, ~2 s at dt = 0.05 (plan section 1)

# --- thresholds. Module constants, one comment each; a test pins them. ---
# `sweep_ratio` is the tangential fraction of the opponent's path about
# obstacle k, 1 for a circular arc and 0 for a radial run. The brief's
# "angular_sweep > pi/2" (coming round) and "> pi" (orbiting) are swept ANGLES,
# which `features()` does not report; they are re-expressed as the two shares of
# tangential motion that distinguish a flank in progress from a sustained orbit.
COMING_ROUND_SWEEP = 0.50  # half the path tangential about MY obstacle: coming round
ORBITING_SWEEP = 0.70  # most of it, at a flat radius: circling that obstacle
# `observer_trend` is d(distance to the observer)/dstep over the window's own
# step length, so the brief's "closing_rate > 0" is a NEGATIVE trend here. The
# bar is a tenth of a step length per step: closing, not merely not opening.
CLOSING_TREND = -0.10
# `obstacle_trend` ~ 0 is the brief's test for "on k's circle rather than
# arriving at or leaving it"; +-0.15 step lengths per step is 6 cm over a
# 40-step window at the evader's top speed.
FLAT_OBSTACLE_TREND = 0.15
# `tortuosity` is net / path displacement, 1 for a straight line. The brief's
# "tortuosity ~ 1" for a transit: 0.90 admits the gentle arc an evader steering
# with a heading P-gain actually flies.
STRAIGHT_RUN = 0.90
APPROACHING_TREND = -0.15  # closing on that obstacle at the same rate scale
# `obstacle_level` is the mean surface distance to k over the half arena
# (5 m), so 0.10 is 0.50 m: inside this the evader is already AT the obstacle
# and is not in transit to it.
AT_OBSTACLE_LEVEL = 0.10
# `speed_frac` is the mean step over the largest step observed this episode.
# The recogniser's own standing bar: below it the window holds no motion to
# lead, so there is nothing for `LEAD` to aim ahead of.
MOVING_SPEED = 0.15

# Plan section 1: "chooses a tactic at decision points (the current tactic's
# `DONE`, or every T = 40 steps)". The Step 1 adapters switch on DONE ONLY, so
# the period is a property of the tactic as offered to `select` — the same
# wrapper the Step 3 gate used, at the plan's T instead of that gate's 80.
# Measured without it (pilot seeds 2 291 000+, ratio 0.8): a single ORBIT span
# of 1197 steps and a single HOLD-HIDDEN span of 1200, i.e. a game in which no
# selector decision is ever taken after the first.
DECISION_PERIOD = 40


class _Capped:
    """`inner`, offered to `select` for at most `DECISION_PERIOD` steps of one
    span, then DONE so the selector decides again.

    Everything else is forwarded, so the controller, the precondition, the name
    and whatever the GIF overlays read off a tactic are Step 1's unchanged. A
    re-selection of the same tactic re-`enter`s it, which is that tactic's own
    documented behaviour at a span boundary.
    """

    __slots__ = ("inner", "limit", "steps")

    def __init__(self, inner, limit: int = DECISION_PERIOD) -> None:
        self.inner = inner
        self.limit = int(limit)
        self.steps = 0

    @property
    def name(self) -> str:
        return self.inner.name

    def __getattr__(self, attr):  # the wrapped tactic's own fields (overlays)
        return getattr(self.inner, attr)

    def reset(self, env=None) -> None:
        self.inner.reset(env)
        self.steps = 0

    def enter(self) -> None:
        self.inner.enter()
        self.steps = 0

    def applicable(self, obs, env) -> bool:
        return self.inner.applicable(obs, env)

    def act(self, *args):
        if self.steps >= self.limit:
            return None
        action = self.inner.act(*args)
        if action is None:
            return None
        self.steps += 1
        return action


# ---------------------------------------------------------------- memory
class TacticMemory:
    """One role's memory, reset per episode (plan section 1, as D146/D148 left
    it): the dead-reckoning estimate of the opponent (point + age), the
    opponent's MOTION FEATURES and label history from `TacticRecogniser`, own
    span history, and the speed ratio.

    `speed_ratio` is carried because the plan's section 1 lists it as an input
    of the LEARNED selector. The v0 rule tables must not branch on it.

    `observe` is called once per env step by whoever drives the episode, with
    the observation of this role and the poses after that step — so the window
    a selector reads at a decision point ends at the state it is deciding from.
    `full_state` is the L1.5 observer (the evader watching the pursuer): the
    opponent's position enters on every step rather than only the visible ones.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        opponent_role: str,
        window: int = WINDOW,
        with_belief: bool = True,
        full_state: bool = False,
    ) -> None:
        self.cfg = cfg
        self.opponent_role = opponent_role
        self.window = int(window)
        self.full_state = bool(full_state)
        self.belief = LastSeenBelief(cfg) if with_belief else None
        self.recogniser = TacticRecogniser(cfg, opponent_role, int(window))
        self.speed_ratio = float(cfg.evader_speed_ratio)  # NOT read by v0
        self.own_history: list = []  # the adapter's span log, refreshed per call

    def reset(self) -> None:
        if self.belief is not None:
            self.belief.reset()
        self.recogniser.reset()
        self.own_history = []

    def observe(self, obs, own_pose, opponent_pose, obstacles) -> None:
        if self.belief is not None:
            self.belief.update(np.asarray(obs))
        seen = self.full_state or bool(obs[6] > 0.5)
        self.recogniser.observe(
            own_pose, np.asarray(opponent_pose, dtype=float)[:2] if seen else None,
            obstacles,
        )

    def features(self) -> dict:
        """The opponent's window features (D147: features, never labels)."""
        return self.recogniser.features()

    @property
    def opponent_history(self) -> list:
        """The recogniser's run log of the opponent — reported, consumed by no
        rule (D147/D148)."""
        return self.recogniser.history

    @property
    def point(self) -> np.ndarray | None:
        """The dead-reckoning point estimate, or None when never seen."""
        if self.belief is None or not self.belief.seen:
            return None
        return np.asarray(self.belief.mean, dtype=float)

    @property
    def age(self) -> int | None:
        """Steps since the last sighting (0 while visible), None if never."""
        if self.belief is None or not self.belief.seen:
            return None
        return int(self.belief.t_since_seen)


# ----------------------------------------------------------------- helpers
def _surface_distances(point, obstacles) -> list[float]:
    p = np.asarray(point, dtype=float)[:2]
    return [
        float(np.hypot(p[0] - o["x"], p[1] - o["y"])) - float(o["radius"])
        for o in obstacles
    ]


def _nearest_obstacle(point, obstacles) -> int | None:
    """The obstacle whose SURFACE is nearest `point` — the recogniser's own
    choice of obstacle k, so a rule over `sweep_ratio` and `obstacle_trend`
    talks about the obstacle those features were computed about."""
    d = _surface_distances(point, obstacles)
    return None if not d else int(np.argmin(d))


def _wall_distance(point, cfg: EnvConfig) -> float:
    p = np.asarray(point, dtype=float)[:2]
    return float(cfg.arena_size / 2.0 - max(abs(p[0]), abs(p[1])))


def _first_applicable(candidates, obs, env):
    """The first candidate whose precondition holds, or None. The ONLY way a
    rule returns a tactic, so an inapplicable pick cannot be returned."""
    for tactic in candidates:
        if tactic.applicable(obs, env):
            return tactic
    return None


class _RulesV0:
    """Shared plumbing: the library, and the adapter's `select(obs, env,
    history)` call signature mapped onto the brief's `select(obs, env,
    memory)`. The memory is this object's own, built for the role unless one is
    supplied, and the adapter's span log is handed to it on every call."""

    def __init__(self, cfg: EnvConfig, memory: TacticMemory | None = None) -> None:
        self.cfg = cfg
        self.memory = self._default_memory(cfg) if memory is None else memory
        self.tactics: list = []

    def _default_memory(self, cfg: EnvConfig) -> TacticMemory:
        raise NotImplementedError

    def reset(self) -> None:
        self.memory.reset()

    def __call__(self, obs, env, history):
        self.memory.own_history = history
        return self.select(obs, env, self.memory)

    def select(self, obs, env, memory):
        raise NotImplementedError


# ------------------------------------------------------------ evader rules
class EvaderRulesV0(_RulesV0):
    """The evader's v0 rule table (brief, order matters).

    1. hidden from the pursuer                              -> HOLD-HIDDEN
    2. the pursuer is coming round MY obstacle and closing   -> HOP to the
       obstacle farthest from the pursuer
    3. pursuer within HUG_THREAT_RANGE, obstacle in HUG_RANGE-> ORBIT(nearest)
    4. pursuer within range, a wall nearer than any obstacle -> WALL-SLIDE
    5. otherwise                                             -> FLEE
    """

    def __init__(self, cfg: EnvConfig, memory: TacticMemory | None = None) -> None:
        super().__init__(cfg, memory)
        self.hold_hidden = _Capped(HoldHidden(cfg))
        self.hops = [_Capped(Hop(cfg, j)) for j in range(cfg.n_obstacles)]
        self.orbits = [_Capped(Orbit(cfg, k)) for k in range(cfg.n_obstacles)]
        self.wall_slide = _Capped(WallSlide(cfg))
        self.flee = _Capped(Flee(cfg))
        self.tactics = [
            self.flee,
            *self.orbits,
            *self.hops,
            self.wall_slide,
            self.hold_hidden,
        ]

    def _default_memory(self, cfg: EnvConfig) -> TacticMemory:
        # The evader watches the pursuer with the full-state view L1.5 gives
        # every scripted evader, and needs no belief: it is never the occluded
        # observer of this game.
        return TacticMemory(cfg, "pursuer", with_belief=False, full_state=True)

    def select(self, obs, env, memory):
        f = memory.features()  # the PURSUER's motion features (D147)
        threat = float(np.hypot(*(env.evader_pose[:2] - env.pursuer_pose[:2])))
        surfaces = _surface_distances(env.evader_pose, env.obstacles)
        # 1. Out of the pursuer's sight: standing still costs nothing and gives
        #    its search nothing to follow.
        if self.hold_hidden.applicable(obs, env):
            return self.hold_hidden
        # 2. The pursuer's path about MY obstacle is mostly tangential while it
        #    closes — it is coming round the thing I am hiding behind, so leave
        #    for the obstacle farthest from it before the line of sight opens.
        if self._coming_round(f, env):
            hop = _first_applicable(self._hops_from_farthest(env), obs, env)
            if hop is not None:
                return hop
        # 3. Pursuer close and an obstacle in reach: put the obstacle between us
        #    (the hug law's own two ranges, so the tactic can actually run).
        if threat <= HUG_THREAT_RANGE and surfaces and min(surfaces) < HUG_RANGE:
            orbit = _first_applicable(
                [self.orbits[int(np.argmin(surfaces))]], obs, env
            )
            if orbit is not None:
                return orbit
        # 4. Pursuer close and the nearest wall nearer than any obstacle: the
        #    wall is the only thing to put a flank against.
        if threat <= HUG_THREAT_RANGE and _wall_distance(
            env.evader_pose, env.cfg
        ) < (min(surfaces) if surfaces else float("inf")):
            slide = _first_applicable([self.wall_slide], obs, env)
            if slide is not None:
                return slide
        # 5. Nothing worth hiding behind: run (always applicable).
        return self.flee

    def _coming_round(self, f: dict, env) -> bool:
        """Rule 2's condition. `sweep_ratio` and the trends are computed about
        the obstacle nearest the PURSUER (the recogniser's k), so "MY obstacle"
        is ENFORCED by requiring that to be the evader's nearest too, rather
        than assumed from the pursuer being close."""
        if not f["valid"]:
            return False
        if f["sweep_ratio"] < COMING_ROUND_SWEEP or f["observer_trend"] > CLOSING_TREND:
            return False
        k = _nearest_obstacle(env.pursuer_pose, env.obstacles)
        return k is not None and k == _nearest_obstacle(
            env.evader_pose, env.obstacles
        )

    def _hops_from_farthest(self, env) -> list:
        """The HOP instances ordered by their obstacle's distance from the
        pursuer, farthest first — "HOP to the obstacle farthest from the
        pursuer", with the next-best as the fallback if that one is not
        applicable (already between us, or no free goal cell)."""
        order = np.argsort(_surface_distances(env.pursuer_pose, env.obstacles))[::-1]
        return [self.hops[int(j)] for j in order if int(j) < len(self.hops)]


# ----------------------------------------------------------- pursuer rules
class PursuerRulesV0(_RulesV0):
    """The pursuer's v0 rule table (brief, order matters).

    1. evader not visible                                    -> SWEEP
       (dead-reckoning-driven: `Sweep` steers at `LastSeenBelief`'s point while
       its age is under `pe2.tactics.BELIEF_FRESH_STEPS` = 100 steps, else
       `OrbitSearch`. That hook was added to `Sweep` additively for this step.)
    2. the evader is circling obstacle k                      -> COUNTER-ORBIT(k)
    3. a straight run closing on an obstacle j it is not at   -> AMBUSH(j),
       else LEAD
    4. visible and moving                                     -> LEAD
    5. otherwise                                              -> INTERCEPT
    """

    def __init__(self, cfg: EnvConfig, memory: TacticMemory | None = None) -> None:
        super().__init__(cfg, memory)
        self.sweep = _Capped(Sweep(cfg, belief=self.memory.belief))
        self.counter_orbits = [
            _Capped(CounterOrbit(cfg, k)) for k in range(cfg.n_obstacles)
        ]
        self.ambushes = [_Capped(Ambush(cfg, k)) for k in range(cfg.n_obstacles)]
        self.lead = _Capped(Lead(cfg))
        self.intercept = _Capped(Intercept(cfg))
        self.tactics = [
            self.intercept,
            self.lead,
            *self.counter_orbits,
            *self.ambushes,
            self.sweep,
        ]

    def _default_memory(self, cfg: EnvConfig) -> TacticMemory:
        # The pursuer is the FOV-limited observer: it carries the dead-reckoning
        # belief (D146) and watches the evader only on the steps it can see it.
        return TacticMemory(cfg, "evader", with_belief=True, full_state=False)

    def select(self, obs, env, memory):
        f = memory.features()  # the EVADER's motion features (D147)
        # 1. Nothing to chase: search, steered by the dead-reckoning estimate
        #    while it is fresh (the belief is this tactic's, see `Sweep`).
        if self.sweep.applicable(obs, env):
            return self.sweep
        # From here the evader is visible, so its position is observed, not
        # assumed, and the obstacle the features were computed about is known.
        seen_xy = opponent_world_from_obs(np.asarray(obs), self.cfg)
        k = _nearest_obstacle(seen_xy, env.obstacles)
        gap = float(np.hypot(*(seen_xy - np.asarray(env.pursuer_pose)[:2])))
        # 2. Its path about k is mostly tangential at a flat distance from it:
        #    it is keeping k between us, so take the other way round.
        if (
            k is not None
            and f["valid"]
            and f["sweep_ratio"] >= ORBITING_SWEEP
            and abs(f["obstacle_trend"]) <= FLAT_OBSTACLE_TREND
        ):
            flank = _first_applicable([self.counter_orbits[k]], obs, env)
            if flank is not None:
                return flank
        # 3. A straight run closing on an obstacle it has not reached: it is
        #    hopping to k, so stand in that obstacle's gap and wait. Only while
        #    it is still farther than AMBUSH_CLOSE — inside that the ambush is
        #    DONE on entry and waiting is pointless.
        if (
            k is not None
            and f["valid"]
            and f["tortuosity"] >= STRAIGHT_RUN
            and f["obstacle_trend"] <= APPROACHING_TREND
            and f["obstacle_level"] > AT_OBSTACLE_LEVEL
            and gap > AMBUSH_CLOSE
        ):
            trap = _first_applicable([self.ambushes[k]], obs, env)
            if trap is not None:
                return trap
            lead = _first_applicable([self.lead], obs, env)
            if lead is not None:
                return lead
        # 4. Visible and moving: aim a second ahead of it (S23's best law).
        if f["valid"] and f["speed_frac"] >= MOVING_SPEED:
            lead = _first_applicable([self.lead], obs, env)
            if lead is not None:
                return lead
        # 5. No motion to lead (just acquired, or standing): go at it.
        return self.intercept
