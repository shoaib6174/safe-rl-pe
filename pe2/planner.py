"""pe2.planner — `PlannerInterceptLaw`, the path-planning intercept law.

Spec: docs/pathway/planner_intercept_prereg_2026-10-06.md section 2.

`InterceptLaw` steers at a visible evader along a straight bearing. When an
obstacle or a wall lies on that bearing the CBF shield pins the pursuer at
zero clearance (S126 CG4; S18's behavioural gate watched ~1100 steps of it in
one episode), so every "cost of the shield" this project has reported was
measured on a controller that drives into walls. This law changes ONLY the
point steered toward: it plans around the known static map when the straight
line is obstructed (goal-snapped per amendment A2.1), and it is
**bit-identical to `InterceptLaw` whenever the line is clear** (prereg section 2, invariant). Without that identity the S19
round would compare two different laws rather than one law against
itself-plus-planning.
"""

from __future__ import annotations

import hashlib
import heapq
from itertools import pairwise

import numpy as np

from pe2.config import EnvConfig
from pe2.env import pursuer_step_batch
from pe2.structured import (
    K_OMEGA,
    LAST_SEEN_TOL,
    _clip_omega,
    _normalize,
    opponent_world_from_obs,
    own_pose_from_obs,
    wrap_angle,
)

MARGIN = 0.0  # prereg section 2: the S18 shield's margin, so the planner's
# notion of "clear" is exactly the filter's notion of "no contact".
CELL = 0.1  # m, A* grid resolution (prereg section 2)
LOOKAHEAD = 0.5  # m along the smoothed path (prereg section 2)
_SQRT2 = float(np.sqrt(2.0))
# The law's inputs are DECODED from a float32 observation, so the arena-bound
# test can only be evaluated to that precision: an agent clipped exactly onto
# the inset bound comes back a few float32 ulps outside it (own = obs[0]*half
# and the opponent adds a normalised body-frame offset times arena_size,
# rotated — measured worst case 1.9e-6 m over 50 open-arena episodes). The
# tolerance is that precision, 8 float32 eps of the arena size: 9.5e-6 m, four
# orders below the 0.1 m grid cell and five below the smallest inflated
# obstacle (0.45 m), so it cannot admit an untraversable line.
_BOUND_REL_TOL = 8.0 * float(np.finfo(np.float32).eps)

_NEIGHBOURS = tuple(
    (dr, dc, 1.0 if dr == 0 or dc == 0 else _SQRT2)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


class PlannerInterceptLaw:
    """`InterceptLaw` with the straight bearing replaced by a planned one.

    Same interface, same two branches in the same order (visible -> chase;
    else last-seen; else None) and the same `last_seen` state. `reset(env=...)`
    reads the static map once per episode; scripted laws are full-state by
    project convention (L1.5) and the map does not change within an episode.

    `n_unreachable` is CUMULATIVE over the life of the object (not cleared by
    `reset`): it counts the residual calls that still fall back to direct
    steering after goal-snapping (amendment A2.1), so CG-C can read how often
    it happens over a whole cell.
    """

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.last_seen: np.ndarray | None = None
        self.n_unreachable = 0
        self._lim = cfg.arena_size / 2.0 - cfg.agent_radius
        self._bound_tol = _BOUND_REL_TOL * cfg.arena_size
        self._n = round(2.0 * self._lim / CELL)
        self.set_map([])

    # -- map -------------------------------------------------------------
    def set_map(self, obstacles: list[dict]) -> None:
        """Inflate the circles and mark the blocked cells. Called by `reset`."""
        self.obstacles = [dict(o) for o in obstacles]
        self._oc = np.array(
            [[o["x"], o["y"]] for o in self.obstacles], dtype=float
        ).reshape(-1, 2)
        self._or = np.array(
            [o["radius"] + self.cfg.agent_radius + MARGIN for o in self.obstacles],
            dtype=float,
        )
        # Cell centres: -lim + (i + 0.5) * CELL, so every centre lies inside
        # the inset arena by construction and only the discs can block one.
        axis = -self._lim + (np.arange(self._n) + 0.5) * CELL
        gx, gy = np.meshgrid(axis, axis, indexing="ij")
        blocked = np.zeros((self._n, self._n), dtype=bool)
        for (ox, oy), r in zip(self._oc, self._or, strict=True):
            blocked |= (gx - ox) ** 2 + (gy - oy) ** 2 <= r * r
        self._blocked = blocked
        self._axis = axis

    def reset(self, env=None) -> None:
        self.last_seen = None
        self.set_map([] if env is None else env.obstacles)

    # -- geometry --------------------------------------------------------
    def in_arena(self, p) -> bool:
        return bool(
            abs(p[0]) <= self._lim + self._bound_tol
            and abs(p[1]) <= self._lim + self._bound_tol
        )

    def segment_clearance(self, a, b) -> float:
        """min over obstacles of (distance from the centre to the segment
        minus the inflated radius). <= 0 means the segment touches or enters
        an inflated disc; +inf on an empty map."""
        if not self.obstacles:
            return float("inf")
        a = np.asarray(a, dtype=float)
        ab = np.asarray(b, dtype=float) - a
        l2 = float(ab @ ab)
        if l2 <= 0.0:
            proj = np.broadcast_to(a, self._oc.shape)
        else:
            t = np.clip((self._oc - a) @ ab / l2, 0.0, 1.0)
            proj = a + t[:, None] * ab
        return float(np.min(np.linalg.norm(self._oc - proj, axis=1) - self._or))

    def is_clear(self, a, b) -> bool:
        """The segment a->b stays inside the inset arena and misses every
        inflated disc. The inset arena is a convex box, so both endpoints
        being inside it is exactly the segment being inside it."""
        if not (self.in_arena(a) and self.in_arena(b)):
            return False
        return self.segment_clearance(a, b) > 0.0

    # -- planning --------------------------------------------------------
    def _cell(self, p) -> tuple[int, int]:
        idx = np.clip(
            np.floor((np.asarray(p, dtype=float) + self._lim) / CELL).astype(int),
            0,
            self._n - 1,
        )
        return int(idx[0]), int(idx[1])

    def _centre(self, r: int, c: int) -> np.ndarray:
        return np.array([self._axis[r], self._axis[c]], dtype=float)

    def _snap_goal(self, target) -> tuple[int, int] | None:
        """The A* goal cell (amendment A2.1): the target's own cell when it is
        free, else the NEAREST FREE CELL whose centre lies within one inflated
        obstacle radius of the target point. None when no such cell exists --
        the only case the law still falls back to direct steering.

        The radius is that of the obstacle nearest the target: the one whose
        inflated disc blocked the cell. The original rule fell back whenever the
        target's 0.1 m CELL CENTRE was blocked, which on the default arena fired
        on ~14 % of planner calls -- mostly pure discretisation (the target point
        itself clear) and otherwise an evader in contact with an obstacle, i.e.
        exactly the HUG / learned-attacker behaviour the round is about.
        """
        cell = self._cell(target)
        if not self._blocked[cell]:
            return cell
        if not self.obstacles:  # no obstacles, no blocked cells
            return None
        target = np.asarray(target, dtype=float)
        k = int(np.argmin(np.linalg.norm(self._oc - target, axis=1) - self._or))
        radius = float(self._or[k])
        span = int(np.ceil(radius / CELL))
        lo_r, hi_r = max(0, cell[0] - span), min(self._n, cell[0] + span + 1)
        lo_c, hi_c = max(0, cell[1] - span), min(self._n, cell[1] + span + 1)
        free = ~self._blocked[lo_r:hi_r, lo_c:hi_c]
        d2 = (self._axis[lo_r:hi_r, None] - target[0]) ** 2 + (
            self._axis[None, lo_c:hi_c] - target[1]
        ) ** 2
        d2 = np.where(free, d2, np.inf)
        if float(d2.min()) > radius * radius:
            return None
        i = int(np.argmin(d2))  # row-major, so ties resolve deterministically
        return lo_r + i // d2.shape[1], lo_c + i % d2.shape[1]

    def _astar(self, start: tuple[int, int], goal: tuple[int, int]) -> list | None:
        """8-connected grid A* over the inflated map, to a goal cell that
        `_snap_goal` has already shown to be free. The start cell is always
        expandable (a pursuer in contact with an obstacle sits in a blocked
        cell); diagonal moves may not cut a blocked corner."""
        n, blocked = self._n, self._blocked
        si, gi = start[0] * n + start[1], goal[0] * n + goal[1]
        g = np.full(n * n, np.inf)
        closed = np.zeros(n * n, dtype=bool)
        came: dict[int, int] = {}
        g[si] = 0.0
        heap = [(0.0, si)]
        while heap:
            _f, i = heapq.heappop(heap)
            if closed[i]:
                continue
            closed[i] = True
            if i == gi:
                break
            r, c = divmod(i, n)
            for dr, dc, w in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < n) or blocked[nr, nc]:
                    continue
                if dr and dc and (blocked[r, nc] or blocked[nr, c]):
                    continue
                j = nr * n + nc
                if closed[j]:
                    continue
                ng = g[i] + w
                if ng < g[j]:
                    g[j] = ng
                    came[j] = i
                    dx, dy = abs(nr - goal[0]), abs(nc - goal[1])
                    h = (dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)
                    heapq.heappush(heap, (ng + h, j))
        if not closed[gi]:
            return None
        cells, i = [gi], gi
        while i != si:
            i = came[i]
            cells.append(i)
        cells.reverse()
        return [self._centre(*divmod(i, n)) for i in cells]

    def _shortcut(self, pts: list) -> list:
        """Greedy line-of-sight shortcutting: from the current point keep the
        farthest vertex reachable by a clear segment, always advancing by at
        least one vertex so a start in contact with an obstacle still moves."""
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = i + 1
            for k in range(len(pts) - 1, i + 1, -1):
                if self.is_clear(pts[i], pts[k]):
                    j = k
                    break
            out.append(pts[j])
            i = j
        return out

    def plan(self, own_xy, target) -> list | None:
        """The smoothed path own->target, or None when there is no free goal
        cell within one inflated obstacle radius of the target or A* cannot
        reach the one there is."""
        own_xy = np.asarray(own_xy, dtype=float)
        target = np.asarray(target, dtype=float)
        goal = self._snap_goal(target)
        if goal is None:
            return None
        cells = self._astar(self._cell(own_xy), goal)
        if cells is None:
            return None
        pts = [own_xy, *cells]
        if self.is_clear(pts[-1], target):
            pts.append(target)
        return self._shortcut(pts)

    def lookahead_point(self, path: list, distance: float = LOOKAHEAD) -> np.ndarray:
        rem = distance
        for a, b in pairwise(path):
            seg = b - a
            length = float(np.hypot(*seg))
            if length <= 0.0:
                continue
            if rem <= length:
                return a + seg * (rem / length)
            rem -= length
        return path[-1]

    # -- the law ---------------------------------------------------------
    def _steer_at(self, own: np.ndarray, point) -> np.ndarray:
        cfg = self.cfg
        to = np.asarray(point, dtype=float) - own[:2]
        err = wrap_angle(float(np.arctan2(to[1], to[0])) - own[2])
        return _normalize(cfg.v_max, _clip_omega(K_OMEGA * err, cfg), cfg)

    def _steer_toward(self, own: np.ndarray, point) -> np.ndarray:
        """Steer at a WORLD point: directly when the line to it is clear, else
        through the planned path. Shared by the last-seen branch and by
        `LeadPlannerInterceptLaw`'s aim point so there is one such path, not two."""
        if self.is_clear(own[:2], point):
            return self._steer_at(own, point)
        return self._steer_planned(own, point)

    def _steer_planned(self, own: np.ndarray, target: np.ndarray) -> np.ndarray:
        path = self.plan(own[:2], target)
        if path is None:  # no free goal cell within one radius (A2.1 residual)
            self.n_unreachable += 1
            return self._steer_at(own, target)
        point = self.lookahead_point(path)
        if float(np.hypot(*(point - own[:2]))) < 1e-9:
            return self._steer_at(own, target)
        return self._steer_at(own, point)

    def act(self, obs: np.ndarray) -> np.ndarray | None:
        cfg = self.cfg
        if obs[6] > 0.5:  # opponent visible — same branch, same order
            self.last_seen = opponent_world_from_obs(obs, cfg)
            own = own_pose_from_obs(obs, cfg)
            if self.is_clear(own[:2], self.last_seen):
                # The identical expression on the identical inputs as
                # InterceptLaw: body-frame bearing, never via world coords.
                bearing_b = np.arctan2(obs[4], obs[3])
                return _normalize(cfg.v_max, _clip_omega(K_OMEGA * bearing_b, cfg), cfg)
            return self._steer_planned(own, self.last_seen)
        if self.last_seen is not None:  # go to last-seen
            own = own_pose_from_obs(obs, cfg)
            to = self.last_seen - own[:2]
            if float(np.hypot(*to)) > LAST_SEEN_TOL:
                return self._steer_toward(own, self.last_seen)
            self.last_seen = None
        return None


# A stationary evader does NOT decode to a bit-identical world position twice:
# the reconstruction is own[:2] + R(own_theta) @ body_offset from a float32
# observation, and own moves even when the evader does not, so the apparent
# displacement wanders inside the decoding precision this module already
# quantifies (`_BOUND_REL_TOL * arena_size` = 9.5e-6 m; the measured worst case
# for the position itself is 1.9e-6 m, see the module docstring above).  An
# apparent displacement at or below that precision is therefore read as exactly
# zero -- which is what makes CG-S1's identity exact rather than approximate.
# The cost is a blind spot below 1.9e-4 m/s: four orders under the evader's
# 1.2 m/s and 1600 s to cross the 0.30 m capture radius.


class LeadPlannerInterceptLaw(PlannerInterceptLaw):
    """`PlannerInterceptLaw` aiming `tau` seconds AHEAD of the visible evader.

    Spec: docs/pathway/pursuit_strategy_plan_2026-10-12.md sections 1 and 4.
    D142 showed the planner's chase is "steer at where the evader IS"; against
    an evader 1.2x faster and free to orbit an obstacle with contact forbidden,
    following cannot close. This law changes ONLY the aim point: the evader's
    decoded world position plus `tau` times its estimated velocity, clipped
    into the inset arena and goal-snapped out of an inflated disc.

    ONE variable (CG-S1): whenever the estimated velocity is zero -- a
    stationary evader, or no pair of consecutive sightings to difference -- the
    call DELEGATES to `PlannerInterceptLaw.act`, the same code path, so the
    action is bit-identical to `PLAN` rather than merely close to it.

    `last_aim` / `last_v_est` are per-call records for CG-S4's recorder.
    """

    def __init__(self, cfg: EnvConfig, tau: float) -> None:
        super().__init__(cfg)
        self.tau = float(tau)
        self._env = None
        self._calls = 0
        self._prev: tuple[np.ndarray, int] | None = None
        self.last_aim: np.ndarray | None = None
        self.last_v_est = np.zeros(2)

    def reset(self, env=None) -> None:
        super().reset(env)
        self._env = env
        self._calls = 0
        self._prev = None
        self.last_aim = None
        self.last_v_est = np.zeros(2)

    def _step_index(self) -> int:
        """The env step index the current observation describes.

        `SearchMDP` calls the law once per env step inside `_advance`, but NOT
        during a learner's `hold` span -- `step()` drives those itself -- so
        counting calls would make the two sightings either side of a span look
        consecutive and invent a velocity over `hold` steps. The env's
        `current_step` (not yet incremented for the step this call produces) is
        therefore the source whenever `reset` was handed an env; counting calls
        is the fallback for the map-only unit path where there is no env.
        """
        self._calls += 1
        return self._calls if self._env is None else int(self._env.current_step)

    def _velocity(self, pos: np.ndarray, step: int) -> np.ndarray:
        """Backward difference over the IMMEDIATELY preceding env step, or zero
        when there is no such sighting. Updates the memory either way."""
        prev, self._prev = self._prev, (pos, step)
        if prev is None or step - prev[1] != 1:
            return np.zeros(2)
        d = pos - prev[0]
        if float(np.hypot(*d)) <= self._bound_tol:  # decoding noise, see above
            return np.zeros(2)
        return d / self.cfg.dt

    def _aim_point(self, pos: np.ndarray, v_est: np.ndarray) -> np.ndarray:
        aim = np.clip(pos + self.tau * v_est, -self._lim, self._lim)
        if self.segment_clearance(aim, aim) <= 0.0:  # inside an inflated disc
            goal = self._snap_goal(aim)  # the same rule as the A* goal (A2.1)
            if goal is not None:
                aim = self._centre(*goal)
        return aim

    def act(self, obs: np.ndarray) -> np.ndarray | None:
        step = self._step_index()
        if obs[6] <= 0.5:  # not visible: nothing to lead; parent owns last-seen
            self.last_aim = None
            self.last_v_est = np.zeros(2)
            return super().act(obs)
        pos = opponent_world_from_obs(obs, self.cfg)
        v_est = self._velocity(pos, step)
        self.last_v_est = v_est
        if v_est[0] == 0.0 and v_est[1] == 0.0:  # CG-S1: the SAME code path
            self.last_aim = None
            return super().act(obs)
        self.last_seen = pos  # as the parent's visible branch would have set it
        self.last_aim = self._aim_point(pos, v_est)
        return self._steer_toward(own_pose_from_obs(obs, self.cfg), self.last_aim)


# -- MPC-H (plan sections 1 and 2) -------------------------------------------
MPC_HORIZON = 2.0  # s, H -- 40 env steps at dt = 0.05
MPC_HOLD = 10  # env steps one sampled (v, omega) is held for
MPC_K = 64  # sampled action sequences per call, INCLUDING the direct one
MPC_PATH_WEIGHT = 0.01  # cost per metre of predicted path


def _episode_entropy(env) -> int:
    """A 64-bit digest of the env's RNG state as `reset(env=...)` finds it.

    The law's sampling must be a deterministic function of the episode, and
    Gymnasium does not keep the integer seed `reset(seed=...)` was given. The
    bit generator's state after the episode's obstacles and initial poses were
    drawn IS such a function (`SearchMDP._begin` resets the env, then the law),
    and reading it consumes nothing, so the episode itself is unchanged.
    0 on the map-only unit path, where there is no env.
    """
    rng = getattr(env, "np_random", None)
    if rng is None:
        return 0
    digest = hashlib.blake2b(
        repr(rng.bit_generator.state).encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


class MpcInterceptLaw(LeadPlannerInterceptLaw):
    """`LeadPlannerInterceptLaw` with the fixed-tau aim point replaced by a
    sampled short-horizon plan over the environment's OWN dynamics.

    Spec: docs/pathway/pursuit_strategy_plan_2026-10-12.md sections 1, 2 and 4.
    `LEAD` aims one fixed tau ahead; this law asks whether H = 2 s of reasoning
    over `pe2.env.pursuer_step` finds the cut-off, the reversal or the
    wall-cornering by itself. The opponent model is inertia and nothing else
    (section 2): the evader is predicted at constant velocity from the
    inherited estimate, clipped into the inset arena.

    Per call: K action sequences of `n_seg` segments of `hold` constant steps,
    K - 1 drawn uniformly from the action box and one -- row 0 -- the parent's
    own action held throughout, so the plan can never cost more than the
    parent's on its own cost and exact ties resolve to the parent. Cost is the
    minimum predicted pursuer-evader distance over the horizon, zeroed at
    capture and tie-broken by the earliest capture step, plus
    MPC_PATH_WEIGHT x path length; +inf if any predicted state contacts a wall
    or an inflated obstacle. The first segment of the winner is executed for
    THIS step only -- replanned every call, no commitment.

    ONE variable (CG-S1): when the evader is not visible, or the inherited
    velocity estimate reads as zero under the module's decoding tolerance,
    the call is the parent's -- which is `PlannerInterceptLaw`'s -- so the
    action is bit-identical to `PLAN`. The sampling runs only when `v_est` is
    nonzero. Since the path term is at most MPC_PATH_WEIGHT x v_max x H = 0.02
    and a non-capturing cost is at least the 0.30 m capture distance,
    `last_cost < cfg.capture_distance` iff the chosen plan captures.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        horizon: float = MPC_HORIZON,
        k: int = MPC_K,
        hold: int = MPC_HOLD,
    ) -> None:
        # tau = the horizon: row 0 is the parent's lead action, and the lead a
        # constant-velocity plan is implicitly compared against is the one over
        # the same H. `_aim_point` is otherwise unused by this law.
        super().__init__(cfg, tau=horizon)
        self.k = int(k)
        self.hold = int(hold)
        self.h_steps = round(float(horizon) / cfg.dt)
        if self.h_steps % self.hold:
            raise ValueError(f"horizon {horizon} s is not {self.hold} | steps")
        self.n_seg = self.h_steps // self.hold
        # The pursuer bounds the ROLLOUT must use are the env's own expressions
        # (`PursuitEvasionEnv.__init__`), not -self._lim, so the predicted clip
        # is the executed one to the last ulp.
        half = cfg.arena_size / 2.0
        self._x_lim = (-half + cfg.agent_radius, half - cfg.agent_radius)
        self._y_lim = self._x_lim
        self._ep_seed = 0
        self._clear_plan_records()

    def _clear_plan_records(self) -> None:
        """No plan ran this call (the CG-S1 delegation, or a fresh episode).
        `last_aim is None` is the flag; these three say the sampling was idle."""
        self.last_cost = float("inf")
        self.last_feasible_count = 0
        self.last_chose_direct = False

    def reset(self, env=None) -> None:
        super().reset(env)
        self._ep_seed = _episode_entropy(env)
        self._clear_plan_records()

    # -- the plan --------------------------------------------------------
    def _sequences(self, a_direct, rng) -> np.ndarray:
        """(K, n_seg, 2) Box actions: K - 1 uniform over the action box, plus
        the parent's own action held for every segment at row 0."""
        acts = rng.uniform(-1.0, 1.0, size=(self.k, self.n_seg, 2)).astype(np.float32)
        acts[0] = np.asarray(a_direct, dtype=np.float32)
        return acts

    def _rollout(self, own, pos, v_est, acts: np.ndarray):
        """Roll `acts` over the horizon through `pursuer_step_batch`.

        Returns the per-sequence cost (+inf on any wall or obstacle contact),
        the first predicted capture step (+inf when there is none) and the
        (h_steps, 2) predicted evader path. Commands are decoded from the Box
        exactly as `SingleAgentWrapper._scale_action` decodes them, so what is
        planned is what the harness executes.
        """
        cfg = self.cfg
        k = acts.shape[0]
        v = (acts[:, :, 0].astype(float) + 1.0) / 2.0 * cfg.v_max
        w = acts[:, :, 1].astype(float) * cfg.omega_max
        poses = np.repeat(np.asarray(own, dtype=float)[None, :3], k, axis=0)
        # Constant velocity, clipped into the inset arena (as `_aim_point`).
        e_pred = np.clip(
            np.asarray(pos, dtype=float)
            + np.outer(np.arange(1, self.h_steps + 1) * cfg.dt, v_est),
            -self._lim,
            self._lim,
        )
        # The loop body is kept to the dynamics itself -- the cost is formed
        # afterwards over the whole (h_steps, k) trace, which is what keeps a
        # call inside the 2 ms budget.
        bad = np.zeros(k, dtype=bool)
        xy = np.empty((self.h_steps, k, 2))
        for t in range(self.h_steps):
            seg = t // self.hold
            poses, wall, obstacle = pursuer_step_batch(
                poses,
                v[:, seg],
                w[:, seg],
                cfg,
                self.obstacles,
                self._x_lim,
                self._y_lim,
            )
            bad |= wall | obstacle  # infeasible from this step on
            xy[t] = poses[:, :2]
        start = np.broadcast_to(np.asarray(own, dtype=float)[None, None, :2], (1, k, 2))
        legs = np.diff(xy, axis=0, prepend=start)
        path = np.hypot(legs[:, :, 0], legs[:, :, 1]).sum(axis=0)
        d = np.hypot(xy[:, :, 0] - e_pred[:, None, 0], xy[:, :, 1] - e_pred[:, None, 1])
        captured = d <= cfg.capture_distance
        any_cap = captured.any(axis=0)
        cap = np.where(any_cap, captured.argmax(axis=0), np.inf)  # first such step
        cost = np.where(any_cap, 0.0, d.min(axis=0))
        cost = np.where(bad, np.inf, cost + MPC_PATH_WEIGHT * path)
        return cost, np.where(bad, np.inf, cap), e_pred

    def act(self, obs: np.ndarray) -> np.ndarray | None:
        # The parent owns the step index, the velocity estimate, `last_seen`
        # and both non-sampling branches; its return is also the direct
        # sequence. Calling it is therefore the CG-S1 delegation AND the
        # baseline the plan is scored against -- one code path, not two.
        a_direct = super().act(obs)
        if self.last_aim is None:  # not visible, or v_est zero: this IS `PLAN`
            self._clear_plan_records()
            return a_direct
        own = own_pose_from_obs(obs, self.cfg)
        pos = opponent_world_from_obs(obs, self.cfg)
        # `_velocity` just recorded this call's env step index there.
        step = self._prev[1]
        rng = np.random.default_rng((self._ep_seed, int(step)))
        acts = self._sequences(a_direct, rng)
        cost, cap, e_pred = self._rollout(own, pos, self.last_v_est, acts)
        # Keys last to first: cost, then earliest capture step, then the index
        # -- so an exact tie goes to row 0, the parent's own action.
        best = int(np.lexsort((np.arange(self.k), cap, cost))[0])
        t_aim = int(cap[best]) if np.isfinite(cap[best]) else self.h_steps - 1
        self.last_aim = e_pred[t_aim]
        self.last_cost = float(cost[best])
        self.last_feasible_count = int(np.isfinite(cost).sum())
        self.last_chose_direct = best == 0
        return acts[best, 0]  # first segment only: replanned every call
