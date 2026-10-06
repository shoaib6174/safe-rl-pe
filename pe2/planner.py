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

import heapq
from itertools import pairwise

import numpy as np

from pe2.config import EnvConfig
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
                if self.is_clear(own[:2], self.last_seen):
                    return self._steer_at(own, self.last_seen)
                return self._steer_planned(own, self.last_seen)
            self.last_seen = None
        return None
