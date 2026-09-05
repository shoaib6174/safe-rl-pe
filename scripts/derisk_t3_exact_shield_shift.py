#!/usr/bin/env python3
"""T3 -- exact-game shield-shift twin (de-risk plan 2026-09-04, Tier 2).

QUESTION (R1, contra Jia et al., Control Engineering Practice 171, 2026):
does a hard projection safety shield move the *value* of an adversarial
pursuit-evasion game?  Jia et al. assert, as motivation and without measuring
it, that projecting each player's action into a safe set "restricts the
exploration space of both players, alters the iterative learning dynamics, and
may prevent convergence to equilibrium strategies".  An exact finite tabular
game can settle it: no function approximation, no seeds, no training budget.

PRE-REGISTERED CRITERION (fixed before any number was produced; not to be
edited afterwards):

  * The effect counts as PRESENT iff |V_shielded - V_unshielded| > 0.02 (two
    percentage points of capture probability) on at least 2 of the 3 hazard
    layouts, measured on the exact Nash value of the game.
  * The SIGN of the shift is recorded as the direction prediction for the deep
    R1 experiment.
  * A NULL result is a publishable finding ("projection shields are benign in
    the exact game") and is reported as such.  Layouts, hazard counts and the
    shield definition are not to be tuned after results are seen.
  * PRIMARY shield definition is `nearest_cell` (below).  `motion_preserving`
    is a secondary robustness read only and never decides the criterion.
  * The three shield configurations are: none / both / pursuer-only.  The
    pursuer-only arm exists because shield asymmetry is a confound in the deep
    version of this experiment.

GAME.  A 7x7 gridworld simultaneous-move zero-sum pursuit-evasion game,
extending `scripts/experiment_1_tabular_forall_exists.py` (which is paper-1
evidence and is NOT modified).  Both players see the full state, move one of
{stay, up, down, left, right} per step, and move simultaneously.  Capture =
Chebyshev distance <= 1 after the joint move; capture is absorbing and pays
the pursuer 1.  The horizon is finite and the discount is 1.0, so the value of
the game IS the capture probability -- which is the unit the pre-registered
0.02 threshold is stated in.  (The parent script uses gamma = 0.95; the change
is deliberate and is the only semantic deviation.  A gamma = 0.95 shadow value
is still computed, and used *only* to break ties between equally-optimal
actions in the best-response solvers, so that "capture sooner" beats "capture
later" and best responses are not lazy.  It never enters a reported number.)

The zero-sum matrix solver `solve_zero_sum_nash` is imported from the parent
script rather than re-implemented.

HAZARDS AND THE SHIELD.  Hazard cells are cells a safety filter must keep a
player out of.  The shield is a projection applied to the *move*:

    nearest_cell (PRIMARY): if the action's target cell is a hazard, replace
      the move with the reachable non-hazard cell nearest -- in L1 distance --
      to the intended target; ties broken by lowest action index in the order
      (stay, up, down, left, right).  "Reachable" means the target of one of
      the five actions from the current cell.  Since the current cell is never
      a hazard under shielded play, "stay" is always in the candidate set.

      Consequence, stated because it is a property of the discrete action set
      and not a modelling choice: on a 4-neighbourhood grid the L1-nearest
      safe cell to a blocked target is always the cell you are standing on, so
      `nearest_cell` coincides exactly with "hazards act as walls for the
      shielded player".  That is the strongest possible projection shield
      here, which makes a null result correspondingly strong.

    motion_preserving (SECONDARY, robustness only): as above but the candidate
      set excludes staying put whenever any other safe move exists, so the
      shield deflects the player sideways instead of stopping it.

If a shielded player has no safe candidate at all the shield fails open (the
intended move executes) and the occurrence is counted; this never happens on
the layouts used here and the counter is asserted to be zero.

WHY THE UNSHIELDED ARM STILL HAS HAZARDS.  If the unshielded arm simply had no
hazards, the comparison would measure the hazards, not the shield.  So within a
layout BOTH arms carry the identical hazard set, and the hazards define the
identical initial distribution (uniform over ordered pairs of distinct
non-hazard, non-wall cells at Chebyshev distance > 1, i.e. not already
captured).  The ONLY difference between the arms is whether the projection is
applied to a move.  The unshielded arm may walk straight through hazards --
that is what being unshielded means, and it is the unsafe behaviour the shield
exists to prevent.  The hazard-free game is also reported, as a reference
point only; it never enters a shield delta.

NEGATIVE CONTROLS.
  NC-A (placement): a layout whose hazard sits on a cell that is provably never
    occupied -- a pocket sealed off by two wall cells.  Walls are part of the
    environment and are identical in both arms of that layout.  The shield can
    never fire, so the shielded value must reproduce the unshielded value to
    numerical precision.  A uniform initial distribution on a fully connected
    grid puts positive probability on EVERY cell at t = 0, so a never-occupied
    cell cannot exist without the sealed pocket; that is a property of the
    initial distribution and is reported rather than worked around.
  NC-B (code path): shield enabled with an empty hazard set, on each layout's
    geometry.  Must reproduce the unshielded value bit for bit.

WHAT IS COMPUTED PER (layout, shield configuration).
  * The exact Nash value of the Markov game, by backward induction with a
    matrix-game (minimax) solve at every state.  This is the exact equilibrium
    value in behavioural strategies; no double oracle or policy population is
    involved and no approximation enters.
  * The fixed point reached by alternating exact best-response dynamics from
    (greedy_catch, greedy_escape), with cycle detection.  Deterministic
    policies over a finite state space must eventually cycle; the reported
    br_value is the mean value over the detected cycle.
  * Structural diagnostics that decide whether the two arms are comparable at
    all: safe-set connectivity, the number of joint states reachable from the
    initial distribution, whether capture remains possible, and the expected
    number of shield activations per episode under equilibrium play (a shield
    that never fires cannot produce an effect, and a null from such a layout
    would be vacuous).

Outputs (names begin with t3_ so as not to collide with the concurrent GPU job
writing into results/derisk/):
  results/derisk/t3_exact_shield_shift.json
  results/derisk/t3_exact_shift.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "derisk"


def _load_parent_module():
    """Import the paper-1 tabular script for its zero-sum matrix solver."""
    path = PROJECT_ROOT / "scripts" / "experiment_1_tabular_forall_exists.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_1_tabular_forall_exists", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PARENT = _load_parent_module()
solve_zero_sum_nash = _PARENT.solve_zero_sum_nash

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------

# GRID and HORIZON were calibrated on the hazard-free UNSHIELDED game alone,
# before any shielded arm was run (see `calibrate`). The requirement was an
# interior Nash value so that a 0.02 shift is measurable in EITHER direction: a
# value pinned at 0.0 or 1.0 cannot move by 0.02 whatever the shield does. On a
# 5x5 grid the pursuer captures everything by horizon 6 (V = 1.000) and only a
# 4-step horizon is interior; a 7x7 grid with horizon 7 gives V = 0.562745,
# which is both interior and deep enough for the best-response dynamics to
# develop. No shielded information entered this choice.
GRID = 7
ACTIONS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES = ("stay", "up", "down", "left", "right")
N_ACTIONS = len(ACTIONS)
STAY = 0
CAPTURE_RADIUS = 1  # Chebyshev
HORIZON = 7  # see the note above GRID
GAMMA_TIEBREAK = 0.95  # shadow discount, tie-breaking only, never reported
VALUE_ROUND = 12  # decimals; keeps the Nash solver cache effective

NEAREST_CELL = "nearest_cell"
MOTION_PRESERVING = "motion_preserving"

# Pre-registered decision threshold.
SHIFT_THRESHOLD = 0.02
MIN_LAYOUTS = 2


def idx(r: int, c: int) -> int:
    return r * GRID + c


def rc(i: int) -> tuple[int, int]:
    return divmod(i, GRID)


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    name: str
    hazards: frozenset
    walls: frozenset
    rationale: str


LAYOUTS = [
    Layout(
        name="sparse",
        hazards=frozenset({(1, 5), (3, 3), (5, 1)}),
        walls=frozenset(),
        rationale=(
            "Three cells, 6 % of the 49-cell grid: the smallest perturbation "
            "that still lets the shield fire in more than one region. Placed "
            "on the anti-diagonal and spread across the arena rather than "
            "clustered, so the layout is invariant under 180-degree rotation "
            "and does not privilege any spatial region."
        ),
    ),
    Layout(
        name="dense",
        hazards=frozenset((r, c) for r in (1, 3, 5) for c in (1, 3, 5)),
        walls=frozenset(),
        rationale=(
            "Nine cells, 18 % of the grid, a regular lattice of pillars at the "
            "odd coordinates. Three times the density of `sparse` with the "
            "same symmetries, so density is the only thing that changes "
            "between them. Every remaining cell has an even row or an even "
            "column, so the safe set is provably still connected."
        ),
    ),
    Layout(
        name="corridor",
        hazards=frozenset({(3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6)}),
        walls=frozenset(),
        rationale=(
            "A hazard wall across the middle row with a single gap at (3,3). A "
            "shielded player must funnel through the chokepoint while an "
            "unshielded one crosses anywhere, so this is the layout where a "
            "projection shield restricts the reachable set most. Mirror- and "
            "rotation-symmetric; the safe set stays connected through the gap."
        ),
    ),
]

# NC-A: (0,6) is sealed off by walls at (0,5) and (1,6), so no player can ever
# occupy it or move into it, in either arm.
NEG_CONTROL = Layout(
    name="neg_control",
    hazards=frozenset({(0, 6)}),
    walls=frozenset({(0, 5), (1, 6)}),
    rationale=(
        "Negative control. The single hazard sits in a pocket sealed by two "
        "wall cells, so it is provably never occupied and the projection can "
        "never fire. The walls are environment geometry and are present "
        "identically in both arms, so the arms differ only by the projection. "
        "The shielded value must therefore equal the unshielded value exactly."
    ),
)

HAZARD_FREE = Layout(
    name="hazard_free",
    hazards=frozenset(),
    walls=frozenset(),
    rationale="Reference game. Reported for context only; never enters a delta.",
)

SHIELD_CONFIGS = {
    "none": (False, False),
    "both": (True, True),
    "pursuer_only": (True, False),
}


# ---------------------------------------------------------------------------
# Transition tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Game:
    layout: Layout
    shield_pursuer: bool
    shield_evader: bool
    projection: str = NEAREST_CELL
    horizon: int = HORIZON


class Tables:
    """Precomputed transition, capture and initial-distribution tables."""

    def __init__(self, game: Game):
        self.game = game
        walls = {idx(r, c) for (r, c) in game.layout.walls}
        hazards = {idx(r, c) for (r, c) in game.layout.hazards}
        self.walls = walls
        self.hazards = hazards
        self.n = GRID * GRID
        self.cells = [i for i in range(self.n) if i not in walls]
        self.init_cells = [i for i in self.cells if i not in hazards]

        self.mv_raw = self._raw_table()
        self.mv_safe, self.fired, self.fail_open = self._safe_table(game.projection)

        self.mv_p = self.mv_safe if game.shield_pursuer else self.mv_raw
        self.mv_e = self.mv_safe if game.shield_evader else self.mv_raw
        self.fired_p = self.fired if game.shield_pursuer else np.zeros_like(self.fired)
        self.fired_e = self.fired if game.shield_evader else np.zeros_like(self.fired)

        self.cap = np.zeros((self.n, self.n), dtype=bool)
        for p in self.cells:
            pr, pc = rc(p)
            for e in self.cells:
                er, ec = rc(e)
                self.cap[p, e] = max(abs(pr - er), abs(pc - ec)) <= CAPTURE_RADIUS

        # How much the projection binds, counted only on cells a shielded
        # player can actually occupy (it never starts on or enters a hazard).
        occupiable = np.array(sorted(self.init_cells), dtype=np.int64)
        self.projected_entries = int(self.fired[occupiable].sum())
        no_safe = self.mv_safe == self.mv_raw
        self.fail_open_occupiable = int(
            sum(
                1
                for i in self.init_cells
                for a in range(N_ACTIONS)
                if self.mv_raw[i, a] in self.hazards and no_safe[i, a]
            )
        )

        self.init_pairs = [
            (p, e)
            for p in self.init_cells
            for e in self.init_cells
            if not self.cap[p, e]
        ]

    def _raw_table(self) -> np.ndarray:
        table = np.zeros((self.n, N_ACTIONS), dtype=np.int64)
        for i in range(self.n):
            r, c = rc(i)
            for a, (dr, dc) in enumerate(ACTIONS):
                nr = min(max(r + dr, 0), GRID - 1)
                nc = min(max(c + dc, 0), GRID - 1)
                j = idx(nr, nc)
                table[i, a] = i if j in self.walls else j
        return table

    def _safe_table(self, mode: str):
        """Projection shield. See the module docstring for the definitions."""
        table = self.mv_raw.copy()
        fired = np.zeros((self.n, N_ACTIONS), dtype=bool)
        fail_open = 0
        for i in range(self.n):
            for a in range(N_ACTIONS):
                target = self.mv_raw[i, a]
                if target not in self.hazards:
                    continue
                tr, tc = rc(target)

                def cost(b, tr=tr, tc=tc, i=i):
                    j = self.mv_raw[i, b]
                    jr, jc = rc(j)
                    return (abs(jr - tr) + abs(jc - tc), b)

                safe = [
                    b for b in range(N_ACTIONS) if self.mv_raw[i, b] not in self.hazards
                ]
                if mode == MOTION_PRESERVING:
                    moving = [b for b in safe if self.mv_raw[i, b] != i]
                    candidates = moving if moving else safe
                elif mode == NEAREST_CELL:
                    candidates = safe
                else:
                    raise ValueError(f"unknown projection mode: {mode}")

                if not candidates:
                    fail_open += 1
                    continue
                best = min(candidates, key=cost)
                table[i, a] = self.mv_raw[i, best]
                fired[i, a] = True
        return table, fired, fail_open


# ---------------------------------------------------------------------------
# Exact Nash value of the Markov game
# ---------------------------------------------------------------------------


class _NashCache:
    def __init__(self):
        self.store = {}
        self.hits = 0
        self.misses = 0

    def solve(self, M: np.ndarray):
        key = M.tobytes()
        cached = self.store.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        p_dist, e_dist, value = solve_zero_sum_nash(M)
        p_dist = np.clip(np.asarray(p_dist, dtype=float), 0.0, None)
        e_dist = np.clip(np.asarray(e_dist, dtype=float), 0.0, None)
        p_dist = p_dist / p_dist.sum()
        e_dist = e_dist / e_dist.sum()
        out = (p_dist, e_dist, float(value))
        self.store[key] = out
        return out


def nash_solve(tab: Tables, collect_strategies: bool = False):
    """Exact Nash value by backward induction with a minimax solve per state.

    Returns (value, strategies) where `value` is the expected capture
    probability under the initial distribution and `strategies`, when
    requested, is a (H, n*n, 2, N_ACTIONS) array of behavioural strategies.
    """
    n = tab.n
    H = tab.game.horizon
    cache = _NashCache()
    V_next = np.zeros(n * n, dtype=float)
    strategies = (
        np.zeros((H, n * n, 2, N_ACTIONS), dtype=float) if collect_strategies else None
    )

    for t in range(H - 1, -1, -1):
        V = np.zeros(n * n, dtype=float)
        for p in tab.cells:
            for e in tab.cells:
                if tab.cap[p, e]:
                    continue  # absorbing, value already collected
                M = np.empty((N_ACTIONS, N_ACTIONS), dtype=float)
                for ap in range(N_ACTIONS):
                    np_ = tab.mv_p[p, ap]
                    for ae in range(N_ACTIONS):
                        ne_ = tab.mv_e[e, ae]
                        M[ap, ae] = 1.0 if tab.cap[np_, ne_] else V_next[np_ * n + ne_]
                p_dist, e_dist, value = cache.solve(M)
                V[p * n + e] = round(value, VALUE_ROUND)
                if collect_strategies:
                    strategies[t, p * n + e, 0] = p_dist
                    strategies[t, p * n + e, 1] = e_dist
        V_next = V

    value = float(np.mean([V_next[p * n + e] for (p, e) in tab.init_pairs]))
    return value, strategies, {"lp_solves": cache.misses, "cache_hits": cache.hits}


def verify_equilibrium(tab: Tables, strategies: np.ndarray, value: float) -> dict:
    """Unilateral-deviation saddle check on the computed equilibrium.

    Computes each player's EXACT best response against the opponent's stored
    behavioural strategy. In a zero-sum game a correct equilibrium admits no
    profitable unilateral deviation, so both best-response values must equal
    the Nash value. This is exact arithmetic, so any gap above solver tolerance
    is a bug, not noise.
    """
    n = tab.n
    H = tab.game.horizon
    out = {}
    for role in ("pursuer", "evader"):
        V_next = np.zeros(n * n, dtype=float)
        for t in range(H - 1, -1, -1):
            V = np.zeros(n * n, dtype=float)
            for p in tab.cells:
                for e in tab.cells:
                    s = p * n + e
                    if tab.cap[p, e]:
                        continue
                    opp = strategies[t, s, 1 if role == "pursuer" else 0]
                    best = None
                    for a in range(N_ACTIONS):
                        acc = 0.0
                        for b in range(N_ACTIONS):
                            w = opp[b]
                            if w <= 0.0:
                                continue
                            if role == "pursuer":
                                np_, ne_ = tab.mv_p[p, a], tab.mv_e[e, b]
                            else:
                                np_, ne_ = tab.mv_p[p, b], tab.mv_e[e, a]
                            ns = np_ * n + ne_
                            acc += w * (1.0 if tab.cap[np_, ne_] else V_next[ns])
                        if best is None or (
                            acc > best if role == "pursuer" else acc < best
                        ):
                            best = acc
                    V[s] = best
            V_next = V
        out[role] = float(np.mean([V_next[p * n + e] for (p, e) in tab.init_pairs]))
    out["max_gap"] = max(abs(out["pursuer"] - value), abs(out["evader"] - value))
    out["passes"] = out["max_gap"] < 1e-9
    return out


def nash_occupancy(tab: Tables, strategies: np.ndarray):
    """Forward-propagate equilibrium play; return occupancy diagnostics.

    Yields the set of cells each player occupies with positive probability and
    the expected number of shield activations per episode.
    """
    n = tab.n
    H = tab.game.horizon
    occ = {}
    w0 = 1.0 / len(tab.init_pairs)
    for pair in tab.init_pairs:
        occ[pair] = occ.get(pair, 0.0) + w0

    p_cells, e_cells = set(), set()
    fires = 0.0
    for t in range(H):
        nxt = {}
        for (p, e), w in occ.items():
            p_cells.add(p)
            e_cells.add(e)
            pd = strategies[t, p * n + e, 0]
            ed = strategies[t, p * n + e, 1]
            fires += w * (
                float(pd @ tab.fired_p[p].astype(float))
                + float(ed @ tab.fired_e[e].astype(float))
            )
            for ap in range(N_ACTIONS):
                if pd[ap] <= 0.0:
                    continue
                np_ = tab.mv_p[p, ap]
                for ae in range(N_ACTIONS):
                    if ed[ae] <= 0.0:
                        continue
                    ne_ = tab.mv_e[e, ae]
                    if tab.cap[np_, ne_]:
                        continue  # absorbed
                    nxt[(np_, ne_)] = nxt.get((np_, ne_), 0.0) + w * pd[ap] * ed[ae]
        occ = nxt
    for p, e in occ:
        p_cells.add(p)
        e_cells.add(e)
    return p_cells, e_cells, fires


# ---------------------------------------------------------------------------
# Best-response dynamics
# ---------------------------------------------------------------------------


def _greedy_policy(tab: Tables, role: str) -> np.ndarray:
    """Manhattan-greedy chase / flee, used as the single BR-dynamics start."""
    n = tab.n
    H = tab.game.horizon
    pol = np.zeros((H, n * n), dtype=np.int8)
    mv = tab.mv_p if role == "pursuer" else tab.mv_e
    for p in tab.cells:
        for e in tab.cells:
            if tab.cap[p, e]:
                continue
            best_a, best_d = STAY, None
            for a in range(N_ACTIONS):
                if role == "pursuer":
                    nr, nc = rc(mv[p, a])
                    er, ec = rc(e)
                else:
                    nr, nc = rc(mv[e, a])
                    er, ec = rc(p)
                d = abs(nr - er) + abs(nc - ec)
                if best_d is None or (d < best_d if role == "pursuer" else d > best_d):
                    best_d, best_a = d, a
            pol[:, p * n + e] = best_a
    return pol


def best_response(tab: Tables, opponent: np.ndarray, role: str) -> np.ndarray:
    """Exact best response to a deterministic opponent policy.

    Primary objective is the undiscounted capture probability. Ties are broken
    on a gamma = 0.95 shadow value so that the pursuer prefers capturing sooner
    and the evader prefers being captured later; the shadow value is never
    reported.
    """
    n = tab.n
    H = tab.game.horizon
    pol = np.zeros((H, n * n), dtype=np.int8)
    V_next = np.zeros(n * n, dtype=float)
    W_next = np.zeros(n * n, dtype=float)
    maximize = role == "pursuer"

    for t in range(H - 1, -1, -1):
        V = np.zeros(n * n, dtype=float)
        W = np.zeros(n * n, dtype=float)
        for p in tab.cells:
            for e in tab.cells:
                s = p * n + e
                if tab.cap[p, e]:
                    continue
                a_opp = int(opponent[t, s])
                best = None
                for a in range(N_ACTIONS):
                    if maximize:
                        np_, ne_ = tab.mv_p[p, a], tab.mv_e[e, a_opp]
                    else:
                        np_, ne_ = tab.mv_p[p, a_opp], tab.mv_e[e, a]
                    ns = np_ * n + ne_
                    if tab.cap[np_, ne_]:
                        v, w = 1.0, 1.0
                    else:
                        v, w = V_next[ns], GAMMA_TIEBREAK * W_next[ns]
                    key = (v, w) if maximize else (-v, -w)
                    if best is None or key > best[0]:
                        best = (key, a, v, w)
                pol[t, s] = best[1]
                V[s] = best[2]
                W[s] = best[3]
        V_next, W_next = V, W
    return pol


def evaluate_pair(tab: Tables, pol_p: np.ndarray, pol_e: np.ndarray) -> float:
    n = tab.n
    H = tab.game.horizon
    V_next = np.zeros(n * n, dtype=float)
    for t in range(H - 1, -1, -1):
        V = np.zeros(n * n, dtype=float)
        for p in tab.cells:
            for e in tab.cells:
                s = p * n + e
                if tab.cap[p, e]:
                    continue
                np_ = tab.mv_p[p, int(pol_p[t, s])]
                ne_ = tab.mv_e[e, int(pol_e[t, s])]
                V[s] = 1.0 if tab.cap[np_, ne_] else V_next[np_ * n + ne_]
        V_next = V
    return float(np.mean([V_next[p * n + e] for (p, e) in tab.init_pairs]))


def br_dynamics(tab: Tables, max_iters: int = 40):
    """Alternating exact best response from (greedy_catch, greedy_escape)."""
    pol_e = _greedy_policy(tab, "evader")
    seen = {}
    values = []
    cycle_start = None
    for it in range(max_iters):
        pol_p = best_response(tab, pol_e, "pursuer")
        pol_e = best_response(tab, pol_p, "evader")
        values.append(evaluate_pair(tab, pol_p, pol_e))
        key = (pol_p.tobytes(), pol_e.tobytes())
        if key in seen:
            cycle_start = seen[key]
            break
        seen[key] = it
    if cycle_start is None:
        cycle_values = values[-1:]
        cycle_len = None
    else:
        cycle_values = values[cycle_start:]
        cycle_len = len(values) - cycle_start
    return {
        "iterations": len(values),
        "converged": cycle_start is not None,
        "cycle_start": cycle_start,
        "cycle_length": cycle_len,
        "cycle_values": [round(v, 10) for v in cycle_values],
        "br_value": float(np.mean(cycle_values)),
        "trajectory": [round(v, 10) for v in values],
    }


# ---------------------------------------------------------------------------
# Structural diagnostics (are the two arms comparable at all?)
# ---------------------------------------------------------------------------


def safe_set_connected(tab: Tables) -> bool:
    """Strong connectivity of the cells a shielded player may occupy."""
    nodes = set(tab.init_cells)
    if not nodes:
        return False
    start = next(iter(nodes))

    def reach(table, forward=True):
        seen = {start}
        stack = [start]
        adj = {i: set() for i in nodes}
        for i in nodes:
            for a in range(N_ACTIONS):
                j = int(table[i, a])
                if j in nodes:
                    if forward:
                        adj[i].add(j)
                    else:
                        adj[j].add(i)
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        return seen

    return len(reach(tab.mv_safe, True)) == len(nodes) and len(
        reach(tab.mv_safe, False)
    ) == len(nodes)


def reachable_joint_states(tab: Tables) -> int:
    """Joint (pursuer, evader) states reachable from the initial set."""
    seen = set(tab.init_pairs)
    frontier = list(tab.init_pairs)
    for _ in range(tab.game.horizon):
        nxt = []
        for p, e in frontier:
            if tab.cap[p, e]:
                continue
            for ap in range(N_ACTIONS):
                np_ = int(tab.mv_p[p, ap])
                for ae in range(N_ACTIONS):
                    ne_ = int(tab.mv_e[e, ae])
                    if (np_, ne_) not in seen:
                        seen.add((np_, ne_))
                        nxt.append((np_, ne_))
        frontier = nxt
        if not frontier:
            break
    return len(seen)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_config(layout: Layout, config: str, projection: str = NEAREST_CELL) -> dict:
    sp, se = SHIELD_CONFIGS[config]
    game = Game(layout, sp, se, projection)
    tab = Tables(game)
    t0 = time.time()
    value, strategies, solver_stats = nash_solve(tab, collect_strategies=True)
    p_cells, e_cells, fires = nash_occupancy(tab, strategies)
    saddle = verify_equilibrium(tab, strategies, value)
    br = br_dynamics(tab)
    out = {
        "layout": layout.name,
        "shield": config,
        "projection": projection,
        "horizon": game.horizon,
        "n_hazards": len(layout.hazards),
        "n_walls": len(layout.walls),
        "n_init_cells": len(tab.init_cells),
        "n_init_pairs": len(tab.init_pairs),
        "nash_value": value,
        "br": br,
        "diagnostics": {
            "safe_set_strongly_connected": safe_set_connected(tab),
            "reachable_joint_states": reachable_joint_states(tab),
            "capture_possible": value > 0.0,
            "n_occupiable_cells": len(tab.init_cells),
            "shield_applied_to": [
                r
                for r, on in (
                    ("pursuer", tab.game.shield_pursuer),
                    ("evader", tab.game.shield_evader),
                )
                if on
            ],
            "layout_projectable_action_entries": tab.projected_entries,
            "projected_action_entries_in_force": (
                tab.projected_entries
                if (tab.game.shield_pursuer or tab.game.shield_evader)
                else 0
            ),
            "shield_activations_per_episode_at_equilibrium": round(fires, 8),
            "shield_activations_note": (
                "Expected to be near zero even where the shield binds hard: an "
                "exact equilibrium already knows the projection and simply does "
                "not select actions that get projected. The structural measures "
                "of how much the shield binds are "
                "projected_action_entries_on_occupiable_cells and "
                "reachable_joint_states."
            ),
            "pursuer_cells_occupied": len(p_cells),
            "evader_cells_occupied": len(e_cells),
            "hazard_cells_occupied": sorted(
                rc(i) for i in (p_cells | e_cells) & set(tab.hazards)
            ),
            "shield_fail_open_entries_all_cells": tab.fail_open,
            "shield_fail_open_entries_occupiable": tab.fail_open_occupiable,
            "equilibrium_saddle_check": saddle,
        },
        "solver": solver_stats,
        "seconds": round(time.time() - t0, 2),
    }
    return out


def calibrate(horizons=(4, 5, 6, 7, 8, 9, 10)) -> dict:
    """Choose the horizon on the hazard-free UNSHIELDED game only.

    The horizon is a sensitivity choice: a value pinned at 0 or 1 cannot move
    by 0.02 whatever the shield does. It is fixed here, before any shielded
    arm is run, using no shielded information.
    """
    out = {}
    for H in horizons:
        tab = Tables(Game(HAZARD_FREE, False, False, NEAREST_CELL, H))
        value, _, _ = nash_solve(tab)
        out[H] = value
        print(f"  horizon {H:2d}  hazard-free unshielded Nash value = {value:.6f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--quick", action="store_true", help="sparse layout only")
    args = ap.parse_args()

    if args.calibrate:
        calibrate()
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    layouts = LAYOUTS[:1] if args.quick else LAYOUTS
    results = []

    print(
        f"Horizon = {HORIZON}, grid = {GRID}x{GRID}, capture radius = {CAPTURE_RADIUS}"
    )
    print("Horizon calibration (hazard-free UNSHIELDED game only):")
    calibration = calibrate()

    ref = run_config(HAZARD_FREE, "none")
    print(f"[reference] hazard-free unshielded Nash = {ref['nash_value']:.6f}")
    results.append(ref)

    for layout in layouts:
        for config in ("none", "both", "pursuer_only"):
            row = run_config(layout, config)
            results.append(row)
            print(
                f"[{layout.name:9s} {config:12s}] Nash = {row['nash_value']:.6f}  "
                f"BR = {row['br']['br_value']:.6f}  "
                f"proj = {row['diagnostics']['projected_action_entries_in_force']:3d}  "
                f"saddle = {row['diagnostics']['equilibrium_saddle_check']['passes']}  "
                f"({row['seconds']}s)"
            )

    # Secondary projection (robustness read only; never decides the criterion).
    secondary = []
    for layout in layouts:
        for config in ("both", "pursuer_only"):
            row = run_config(layout, config, projection=MOTION_PRESERVING)
            secondary.append(row)
            print(
                f"[{layout.name:9s} {config:12s} motion] Nash = {row['nash_value']:.6f}"
            )

    # Negative controls.
    controls = {}
    nc_a = {c: run_config(NEG_CONTROL, c) for c in ("none", "both", "pursuer_only")}
    controls["NC_A_sealed_pocket"] = {
        "rationale": NEG_CONTROL.rationale,
        "arms": nc_a,
        "delta_both": nc_a["both"]["nash_value"] - nc_a["none"]["nash_value"],
        "delta_pursuer_only": nc_a["pursuer_only"]["nash_value"]
        - nc_a["none"]["nash_value"],
        "hazard_ever_occupied": bool(
            nc_a["none"]["diagnostics"]["hazard_cells_occupied"]
        ),
        "shield_activations": nc_a["both"]["diagnostics"][
            "shield_activations_per_episode_at_equilibrium"
        ],
        "projected_entries_on_occupiable_cells": nc_a["both"]["diagnostics"][
            "projected_action_entries_in_force"
        ],
    }
    for k in ("delta_both", "delta_pursuer_only"):
        controls["NC_A_sealed_pocket"][k + "_passes"] = (
            abs(controls["NC_A_sealed_pocket"][k]) < 1e-12
        )

    empty = Layout("empty_hazards", frozenset(), frozenset(), "NC-B code-path control")
    nc_b_none = run_config(empty, "none")
    nc_b_both = run_config(empty, "both")
    controls["NC_B_empty_hazard_set"] = {
        "rationale": "Shield enabled with no hazards must be a bit-identical no-op.",
        "unshielded": nc_b_none["nash_value"],
        "shielded": nc_b_both["nash_value"],
        "delta": nc_b_both["nash_value"] - nc_b_none["nash_value"],
        "passes": nc_b_both["nash_value"] == nc_b_none["nash_value"],
    }

    verdict = evaluate_criterion(results, layouts)

    payload = {
        "experiment": "T3 exact-game shield-shift twin",
        "spec": "docs/pathway/derisk_plan_2026-09-04.md, Tier 2 / T3",
        "preregistered": {
            "threshold_abs_value_shift": SHIFT_THRESHOLD,
            "min_layouts": MIN_LAYOUTS,
            "n_layouts": len(layouts),
            "primary_metric": "exact Nash value of the Markov game",
            "primary_projection": NEAREST_CELL,
            "null_is_publishable": True,
        },
        "game": {
            "grid": GRID,
            "horizon": HORIZON,
            "capture_radius_chebyshev": CAPTURE_RADIUS,
            "discount": 1.0,
            "tiebreak_shadow_discount": GAMMA_TIEBREAK,
            "actions": list(ACTION_NAMES),
            "initial_distribution": (
                "uniform over ordered pairs of distinct non-hazard, non-wall "
                "cells at Chebyshev distance > capture radius; identical in "
                "both arms of a layout"
            ),
        },
        "horizon_calibration": {
            "note": (
                "Grid and horizon were fixed on the hazard-free UNSHIELDED "
                "game alone, requiring an interior value so a 0.02 shift is "
                "measurable in either direction. No shielded information "
                "entered the choice."
            ),
            "hazard_free_unshielded_nash_by_horizon": {
                str(k): v for k, v in calibration.items()
            },
        },
        "layout_rationale": {ly.name: ly.rationale for ly in LAYOUTS},
        "results": results,
        "secondary_projection": secondary,
        "negative_controls": controls,
        "verdict": verdict,
    }
    out_path = RESULTS_DIR / "t3_exact_shield_shift.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")

    print("\n--- PRE-REGISTERED VERDICT ---")
    for row in verdict["per_layout"]:
        print(
            f"  {row['layout']:9s} both: {row['delta_both']:+.4f}  "
            f"pursuer_only: {row['delta_pursuer_only']:+.4f}"
        )
    print(f"  criterion met (both-shielded): {verdict['criterion_met_both']}")
    print(f"  sign prediction: {verdict['sign_prediction_both']}")

    make_figure(payload)
    return payload


def evaluate_criterion(results: list, layouts: list) -> dict:
    by = {(r["layout"], r["shield"]): r for r in results}
    per_layout = []
    for ly in layouts:
        base = by[(ly.name, "none")]["nash_value"]
        row = {
            "layout": ly.name,
            "nash_unshielded": base,
            "nash_both": by[(ly.name, "both")]["nash_value"],
            "nash_pursuer_only": by[(ly.name, "pursuer_only")]["nash_value"],
            "br_unshielded": by[(ly.name, "none")]["br"]["br_value"],
            "br_both": by[(ly.name, "both")]["br"]["br_value"],
            "br_pursuer_only": by[(ly.name, "pursuer_only")]["br"]["br_value"],
        }
        row["delta_both"] = row["nash_both"] - base
        row["delta_pursuer_only"] = row["nash_pursuer_only"] - base
        row["br_delta_both"] = row["br_both"] - row["br_unshielded"]
        row["br_delta_pursuer_only"] = row["br_pursuer_only"] - row["br_unshielded"]
        row["exceeds_both"] = abs(row["delta_both"]) > SHIFT_THRESHOLD
        row["exceeds_pursuer_only"] = abs(row["delta_pursuer_only"]) > SHIFT_THRESHOLD
        per_layout.append(row)

    def summarize(key_exceeds, key_delta):
        n = sum(r[key_exceeds] for r in per_layout)
        met = n >= MIN_LAYOUTS
        signs = {np.sign(r[key_delta]) for r in per_layout if r[key_exceeds]}
        if not met:
            sign = "none (criterion not met)"
        elif len(signs) == 1:
            sign = (
                "positive (shield RAISES capture probability)"
                if signs == {1.0}
                else "negative (shield LOWERS capture probability)"
            )
        else:
            sign = "mixed (layout-dependent sign)"
        return n, met, sign

    n_b, met_b, sign_b = summarize("exceeds_both", "delta_both")
    n_p, met_p, sign_p = summarize("exceeds_pursuer_only", "delta_pursuer_only")
    return {
        "per_layout": per_layout,
        "n_layouts_exceeding_both": int(n_b),
        "criterion_met_both": bool(met_b),
        "sign_prediction_both": sign_b,
        "n_layouts_exceeding_pursuer_only": int(n_p),
        "criterion_met_pursuer_only": bool(met_p),
        "sign_prediction_pursuer_only": sign_p,
    }


def make_figure(payload: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verdict = payload["verdict"]
    rows = verdict["per_layout"]
    names = [r["layout"] for r in rows]
    show = LAYOUTS + [NEG_CONTROL]

    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.35], hspace=0.32, wspace=1.6)

    for k, ly in enumerate(show):
        ax = fig.add_subplot(gs[0, 3 * k : 3 * k + 3])
        grid = np.zeros((GRID, GRID))
        for r, c in ly.hazards:
            grid[r, c] = 1.0
        for r, c in ly.walls:
            grid[r, c] = 0.5
        ax.imshow(grid, cmap="Reds", vmin=0, vmax=1.4)
        for r, c in ly.walls:
            ax.text(c, r, "W", ha="center", va="center", fontsize=8, color="0.2")
        for r, c in ly.hazards:
            ax.text(c, r, "H", ha="center", va="center", fontsize=8, color="white")
        ax.set_xticks(np.arange(-0.5, GRID, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, GRID, 1), minor=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(which="minor", color="0.75", linewidth=0.5)
        ax.set_title(f"{ly.name}\n{len(ly.hazards)} hazard(s)", fontsize=9)

    x = np.arange(len(names))
    w = 0.26

    ax = fig.add_subplot(gs[1, 0:4])
    ax.bar(x - w, [r["nash_unshielded"] for r in rows], w, label="unshielded")
    ax.bar(x, [r["nash_both"] for r in rows], w, label="both shielded")
    ax.bar(x + w, [r["nash_pursuer_only"] for r in rows], w, label="pursuer only")
    ax.set_ylabel("capture probability")
    ax.set_title("Exact Nash value", fontsize=11)

    ax_br = fig.add_subplot(gs[1, 4:8])
    ax_br.bar(x - w, [r["br_unshielded"] for r in rows], w, label="unshielded")
    ax_br.bar(x, [r["br_both"] for r in rows], w, label="both shielded")
    ax_br.bar(x + w, [r["br_pursuer_only"] for r in rows], w, label="pursuer only")
    ax_br.set_ylabel("capture probability")
    ax_br.set_title("Best-response-dynamics fixed point", fontsize=11)

    ax2 = fig.add_subplot(gs[1, 8:12])
    ax2.bar(x - w / 2, [r["delta_both"] for r in rows], w, label="both shielded")
    ax2.bar(x + w / 2, [r["delta_pursuer_only"] for r in rows], w, label="pursuer only")
    ax2.axhline(SHIFT_THRESHOLD, color="crimson", ls="--", lw=1)
    ax2.axhline(-SHIFT_THRESHOLD, color="crimson", ls="--", lw=1)
    ax2.axhline(0.0, color="0.3", lw=0.8)
    ax2.text(
        0.99,
        0.03,
        f"pre-registered threshold  |dV| > {SHIFT_THRESHOLD}",
        transform=ax2.transAxes,
        ha="right",
        fontsize=8,
        color="crimson",
    )
    ax2.set_ylabel(r"$V_{\rm shielded} - V_{\rm unshielded}$")
    ax2.set_title(
        f"Shift in Nash value — criterion met: {verdict['criterion_met_both']}",
        fontsize=11,
    )

    for a in (ax, ax_br, ax2):
        a.set_xticks(x)
        a.set_xticklabels(names)
        a.legend(fontsize=8, frameon=False)
        a.grid(True, axis="y", alpha=0.3)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.suptitle(
        f"T3 — exact tabular shield-shift twin ({GRID}x{GRID} grid, horizon "
        f"{HORIZON}, exact Nash, no learning)",
        fontsize=12,
    )
    path = RESULTS_DIR / "t3_exact_shift.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
