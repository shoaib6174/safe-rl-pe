"""Step 6: VCP-CBF safety shield — discrete-time, γ=0.2 (D31).

Constraints: obstacles + arena boundaries ONLY. The inter-agent constraint is
NEVER applied to the pursuer — capture requires contact (plan Step 6, root
codebase_reference §3 note). The shield sees the TRUE state and the static map
while the policy sees sensors; that split is what keeps the CBF well-posed
under partial observability (root dcbf_action_wrapper architecture).

DCBF condition (root vcp_cbf.dcbf_filter_action): h(x_{k+1}) ≥ (1−γ)h(x_k),
linearized around the current state so it stays a QP:
    dt·a_v·v + dt·a_ω·ω + γ·h ≥ 0

D85 (deviation from the root pattern): the root solves this QP with scipy
SLSQP for n-robot generality. pe2's action space is strictly 2-D (v, ω) with
box bounds, so the exact optimum is enumerated instead: unconstrained point,
weighted projection onto each active constraint line, pairwise line
intersections, line∩box-edge intersections, box corners — the optimum of a
convex 2-D QP is always one of these. Exact, deterministic, and drops the
scipy dependency (pe2's CI installs only numpy/gymnasium/matplotlib/imageio).

Each surface (wall or obstacle) contributes two rows. The `hard` row is the
env's own contact predicate evaluated on the BODY (wall: x_lim − x; obstacle:
hypot(body − centre) − r_eff), so h ≥ 0 there is bitwise equivalent to no
contact at margin 0 — it is exact, not an early warning. The `soft` row is the
virtual-control-point barrier evaluated d = 0.1 m ahead of the body, which
supplies steering (uniform relative degree 1, unlike the body row) and
reaches zero roughly d before the body touches anything.

Three tiers, all selecting from ONE candidate set (D85 enumeration + u = 0 +
the clipped nominal) judged by the exact one-step barrier on the true Euler
successor. Nothing is ever dropped from the row set — no tier discards a
constraint (spec B3):
  exact   — some candidate satisfies every row → the one nearest the nominal.
  relaxed — none does, but some satisfy every HARD row → among those,
            maximise the worst normalised soft-row progress.
  backup  — no candidate satisfies every hard row → maximise the worst
            normalised hard-row progress, then the worst soft-row progress.
Ties break toward the nominal, so every tier is deterministic. A violation is
defined at env level (wall/obstacle contact flags); this module only maintains
h > 0.

h > 0 is a CONSERVATIVE early warning only for the SOFT rows — they are
evaluated at the virtual control point d = 0.1 m ahead of the body, so in the
heading direction the barrier reaches zero roughly d BEFORE the body touches
anything (on top of the agent_radius already folded into the constraint
radii); h = 0 there is not the contact surface and h < 0 does not imply
contact. The HARD rows are exact: h = 0 is the contact surface and h < 0
means the env's own contact predicate has fired.

Units: `Constraint.h` is a barrier value whose units depend on the kind
(obstacle barriers are m², boundary barriers are m), so it is a barrier margin
and never a distance. `Constraint.clearance` is the signed distance in metres
for both kinds and is the only field that may be reported as a distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pe2.config import EnvConfig
from pe2.env import pursuer_step, wrap_angle

INTERVENTION_ATOL = 1e-4  # |Δu| above this counts as an intervention (root convention)
STALL_V_FRAC = 0.05  # executed speed <= this fraction of v_max counts as stopped
MOVE_V_FRAC = 0.5  # nominal speed >= this fraction of v_max counts as "wanted to move"
RECOVERY_AFTER = 20  # consecutive immobile steps before RecoveryShield fires (D125)
RECOVERY_OMEGA_MIN = 0.1  # rad/s a pure turn must clear to be worth taking


@dataclass
class Constraint:
    """One barrier row: h ≥ 0 safe; linearised rate h_dot ≈ a_v·v + a_ω·ω.

    `hard` rows are the env's own contact predicates evaluated on the BODY
    (wall: x_lim − x; obstacle: hypot(b − p) − r_eff), both in metres, so
    h ≥ 0 ⟺ no contact, bitwise, at margin 0. Soft rows are the virtual-
    control-point barriers that supply steering; their units follow `kind`
    (obstacle m², boundary m). `scale` converts a row's progress to metres
    (2·r_eff for the obstacle VCP row, 1 otherwise). `clearance` is the signed
    distance in metres for every row and is the only reportable distance.
    """

    h: float
    a_v: float
    a_omega: float
    kind: str  # "obstacle" | "boundary"
    clearance: float  # signed distance to the constraint surface, metres
    hard: bool
    scale: float


def compute_vcp(x: float, y: float, theta: float, d: float) -> tuple[float, float]:
    """Virtual control point d ahead of the robot (uniform relative degree 1)."""
    return x + d * np.cos(theta), y + d * np.sin(theta)


def _obstacle_constraint(
    state: np.ndarray, obs_x: float, obs_y: float, eff_radius: float, d: float
) -> Constraint:
    """h = ||q − p_obs||² − r_eff², q = VCP. Coefficients from root vcp_cbf_obstacle."""
    x, y, theta = float(state[0]), float(state[1]), float(state[2])
    qx, qy = compute_vcp(x, y, theta, d)
    dx, dy = qx - obs_x, qy - obs_y
    h = dx * dx + dy * dy - eff_radius * eff_radius
    # h + r_eff² = ||q − p_obs||² ≥ 0, so this sqrt is exact, not an
    # approximation, and gives the signed clearance in metres.
    clearance = np.sqrt(h + eff_radius * eff_radius) - eff_radius
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    a_v = 2.0 * dx * cos_t + 2.0 * dy * sin_t
    a_omega = 2.0 * dx * (-d * sin_t) + 2.0 * dy * (d * cos_t)
    return Constraint(
        float(h),
        float(a_v),
        float(a_omega),
        "obstacle",
        float(clearance),
        False,
        2.0 * eff_radius,
    )


def _boundary_constraints(
    state: np.ndarray, x_lim: float, y_lim: float, d: float
) -> list[Constraint]:
    """Linear barriers h = x_lim − qx etc. Coefficients from root vcp_cbf_boundary."""
    x, y, theta = float(state[0]), float(state[1]), float(state[2])
    qx, qy = compute_vcp(x, y, theta, d)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def wall(h: float, a_v: float, a_omega: float) -> Constraint:
        # A boundary barrier is already a signed distance in metres, so the
        # clearance passes through unchanged.
        return Constraint(
            float(h), float(a_v), float(a_omega), "boundary", float(h), False, 1.0
        )

    return [
        wall(x_lim - qx, -cos_t, d * sin_t),
        wall(qx + x_lim, cos_t, -d * sin_t),
        wall(y_lim - qy, -sin_t, -d * cos_t),
        wall(qy + y_lim, sin_t, d * cos_t),
    ]


def _body_wall_constraints(
    state: np.ndarray, x_lim: float, y_lim: float
) -> list[Constraint]:
    """Hard rows: the env's wall clip predicate (`x > x_max` etc.) as barriers."""
    x, y, theta = float(state[0]), float(state[1]), float(state[2])
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def wall(h: float, a_v: float) -> Constraint:
        return Constraint(float(h), float(a_v), 0.0, "boundary", float(h), True, 1.0)

    return [
        wall(x_lim - x, -cos_t),
        wall(x + x_lim, cos_t),
        wall(y_lim - y, -sin_t),
        wall(y + y_lim, sin_t),
    ]


def _body_obstacle_constraint(
    state: np.ndarray, obs_x: float, obs_y: float, eff_radius: float
) -> Constraint:
    """Hard row: the env's obstacle predicate (`hypot < r_obs + r_agent`)."""
    x, y, theta = float(state[0]), float(state[1]), float(state[2])
    dx, dy = x - obs_x, y - obs_y
    dist = float(np.hypot(dx, dy))
    h = dist - eff_radius
    a_v = (dx * np.cos(theta) + dy * np.sin(theta)) / dist if dist > 0.0 else 0.0
    return Constraint(float(h), float(a_v), 0.0, "obstacle", float(h), True, 1.0)


def _qp_candidates(
    v_nom: float,
    w_nom: float,
    rows: list[tuple[float, float, float]],
    v_max: float,
    omega_max: float,
    w_v: float,
    w_omega: float,
) -> np.ndarray:
    """Every point the weighted projection onto {c0·v + c1·ω ≥ c2} ∩ box could
    pick (D85): nominal, weighted projection onto each line, pairwise line
    intersections, line ∩ box-edge, box corners — plus `u = 0`, which is the
    action that keeps every hard row bit-identical (spec B4) and must always
    be on the table. Returns a (K, 2) float64 array, every row inside the box,
    in a deterministic order. The rows are only used to LOCATE candidates; the
    exact one-step check in `CBFShield` decides which are admissible.
    """
    box = [(0.0, v_max), (-omega_max, omega_max)]

    def clip_box(v: float, w: float) -> list[float]:
        return [float(np.clip(v, *box[0])), float(np.clip(w, *box[1]))]

    cands = [clip_box(v_nom, w_nom), [0.0, 0.0]]
    for c0, c1, c2 in rows:
        denom = c0 * c0 / w_v + c1 * c1 / w_omega
        if denom < 1e-18:
            continue
        lam = (c0 * v_nom + c1 * w_nom - c2) / denom
        cands.append(clip_box(v_nom - lam * c0 / w_v, w_nom - lam * c1 / w_omega))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            c0i, c1i, c2i = rows[i]
            c0j, c1j, c2j = rows[j]
            det = c0i * c1j - c0j * c1i
            if abs(det) < 1e-12:
                continue
            cands.append(
                clip_box((c2i * c1j - c2j * c1i) / det, (c0i * c2j - c0j * c2i) / det)
            )
    for c0, c1, c2 in rows:
        for v_edge in box[0]:
            if abs(c1) > 1e-12:
                cands.append(clip_box(v_edge, (c2 - c0 * v_edge) / c1))
        for w_edge in box[1]:
            if abs(c0) > 1e-12:
                cands.append(clip_box((c2 - c1 * w_edge) / c0, w_edge))
    for v_c in box[0]:
        for w_c in box[1]:
            cands.append([v_c, w_c])
    return np.asarray(cands, dtype=np.float64)


class CBFShield:
    """Discrete-time VCP-CBF action filter for one robot (γ=0.2, D31).

    Usage:
        shield = CBFShield(env.cfg)
        safe_action, info = shield.filter(nominal, env.pursuer_pose, env.obstacles)

    `margin` (delta, metres) is the D87 robust arm: every constraint radius is
    inflated and every boundary limit shrunk by delta, so a barrier reaches
    zero delta metres of clearance earlier than the true contact surface. It
    exists because the hardware shield runs on an ESTIMATED state, and it is
    NOT a tuned number: `docs/phases/phase4_sim_to_real_transfer.md` requires
    localization error below 5 cm at the 95th percentile
    (`test_localization_accuracy`) and its §8.7 prescribes adding that
    position uncertainty to the obstacle radii in the CBF, which is the
    measurement-robust CBF construction cited in D87 (Cosner/Ames
    arXiv 2104.14030; Dean/Taylor CoRL 2020). The default is 0.0, so the arm
    gated by G1-G4 is unchanged; the G5 arm uses delta = 0.05 m.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        gamma: float = 0.2,
        vcp_d: float = 0.1,
        w_v: float = 150.0,
        w_omega: float = 1.0,
        margin: float = 0.0,
    ):
        self.cfg = cfg
        self.gamma = gamma
        self.d = vcp_d
        self.w_v = w_v
        self.w_omega = w_omega
        self.margin = margin
        half = cfg.arena_size / 2.0
        self._x_lim = half - cfg.agent_radius - margin
        self._y_lim = half - cfg.agent_radius - margin

    def constraints(self, state: np.ndarray, obstacles: list[dict]) -> list[Constraint]:
        """Hard body rows, then soft VCP rows, walls first then obstacles.

        Order is a contract: `successor_barriers` evaluates the same rows in
        the same order on the predicted successor, and the two must agree
        column for column. Effective radius = obs + agent + margin; boundary
        limits already carry the margin (D87 robust arm).
        """
        cons = _body_wall_constraints(state, self._x_lim, self._y_lim)
        cons += _boundary_constraints(state, self._x_lim, self._y_lim, self.d)
        for obs in obstacles:
            r_eff = obs["radius"] + self.cfg.agent_radius + self.margin
            cons.append(_body_obstacle_constraint(state, obs["x"], obs["y"], r_eff))
            cons.append(_obstacle_constraint(state, obs["x"], obs["y"], r_eff, self.d))
        # Recompute h through the same vectorised path `successor_barriers`
        # uses, so the current-pose values and the predicted-successor values
        # can never drift apart into two independent implementations.
        H = self._h_columns(
            np.array([float(state[0])]),
            np.array([float(state[1])]),
            np.array([float(state[2])]),
            obstacles,
        )[0]
        for c, h in zip(cons, H, strict=True):
            c.h = float(h)
            c.clearance = float(h) if c.hard or c.kind == "boundary" else c.clearance
        return cons

    def _h_columns(
        self, X: np.ndarray, Y: np.ndarray, TH: np.ndarray, obstacles: list[dict]
    ) -> np.ndarray:
        """Barrier values for K poses, one column per row of `constraints()`,
        in the same order. Hard rows are the env's contact predicates and use
        the same ufuncs the env uses (`np.hypot`, subtraction), so their
        values at the predicted successor are the env's values."""
        d = self.d
        QX = X + d * np.cos(TH)
        QY = Y + d * np.sin(TH)
        cols = [
            self._x_lim - X,
            X + self._x_lim,
            self._y_lim - Y,
            Y + self._y_lim,
            self._x_lim - QX,
            QX + self._x_lim,
            self._y_lim - QY,
            QY + self._y_lim,
        ]
        for obs in obstacles:
            r_eff = obs["radius"] + self.cfg.agent_radius + self.margin
            cols.append(np.hypot(X - obs["x"], Y - obs["y"]) - r_eff)
            dqx, dqy = QX - obs["x"], QY - obs["y"]
            cols.append(dqx * dqx + dqy * dqy - r_eff * r_eff)
        return np.stack(cols, axis=1)

    def successor_barriers(
        self, state: np.ndarray, obstacles: list[dict], U: np.ndarray
    ) -> np.ndarray:
        """h_i at euler_step(state, u) for every candidate u in U (K, 2) -> (K, R).

        Vectorised transcription of `pe2.env.euler_step`: identical operation
        order, scalar cos/sin of the CURRENT heading, so the hard rows are
        the env's next-step predicates bit for bit (spec B4)."""
        U = np.asarray(U, dtype=np.float64)
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        dt = self.cfg.dt
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        X = x + U[:, 0] * cos_t * dt
        Y = y + U[:, 0] * sin_t * dt
        TH = wrap_angle(theta + U[:, 1] * dt)
        return self._h_columns(X, Y, TH, obstacles)

    def progress(self, state: np.ndarray, obstacles: list[dict], action) -> np.ndarray:
        """Normalised one-step DCBF progress of every row under `action`:
        (h_i(x⁺) − (1−γ) h_i(x)) / scale_i. Every entry ≥ 0 ⟺ the exact
        tier accepts the action. Public so a gate can re-derive C1 from the
        definition rather than from another copy of the filter.

        Two things this does NOT model, both benign today: it does not clip
        `action` to the box (the caller must pass an already box-clipped
        action -- `_qp_candidates` clips every candidate before this is
        called), and the successor prediction ignores `cfg.prep_steps`
        (under which the env forces the pursuer's speed to zero for the
        first steps after reset; the default is 0, where this is moot). A
        gate built directly on this method should account for both.
        """
        cons = self.constraints(state, obstacles)
        H1 = self.successor_barriers(state, obstacles, np.asarray(action)[None, :])[0]
        H0 = np.array([c.h for c in cons])
        scale = np.array([c.scale for c in cons])
        return (H1 - (1.0 - self.gamma) * H0) / scale

    def _dcbf_rows(self, cons: list[Constraint]) -> list[tuple[float, float, float]]:
        """Linearized DCBF: dt·a_v·v + dt·a_ω·ω ≥ −γ·h → (c0, c1, c2)."""
        dt = self.cfg.dt
        return [(dt * c.a_v, dt * c.a_omega, -self.gamma * c.h) for c in cons]

    def filter(
        self,
        action: tuple[float, float] | np.ndarray,
        state: np.ndarray,
        obstacles: list[dict],
        v_max: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Project a nominal action into the safe set; never blocks capture.

        One candidate set (D85 enumeration + u = 0 + nominal) is judged by the
        exact one-step barrier on the true Euler successor (spec B2). Tiers:
          exact   — some candidate satisfies every row → nearest to nominal.
          relaxed — none does, some satisfies every HARD row → among those,
                    maximise the worst normalised soft-row progress.
          backup  — no candidate satisfies every hard row → maximise the
                    worst normalised hard-row progress, then the worst
                    soft-row progress.
        Ties break toward the nominal. Nothing is ever dropped (spec B3).

        Returns (safe_action, info); see `_info` for the keys. `feasible` is
        True only on the exact tier.
        """
        v_max = self.cfg.v_max if v_max is None else v_max
        v_nom, w_nom = self._clip_nominal(action, v_max)
        cons = self.constraints(state, obstacles)
        U = _qp_candidates(
            v_nom,
            w_nom,
            self._dcbf_rows(cons),
            v_max,
            self.cfg.omega_max,
            self.w_v,
            self.w_omega,
        )
        H0 = np.array([c.h for c in cons])
        scale = np.array([c.scale for c in cons])
        hard = np.array([c.hard for c in cons])
        H1 = self.successor_barriers(state, obstacles, U)
        target = (1.0 - self.gamma) * H0
        sat = H1 >= target  # (K, R) — exact comparison, no tolerance
        prog = (H1 - target) / scale
        obj = self.w_v * (U[:, 0] - v_nom) ** 2 + self.w_omega * (U[:, 1] - w_nom) ** 2

        all_ok = sat.all(axis=1)
        hard_ok = sat[:, hard].all(axis=1)
        if all_ok.any():
            idx = int(np.flatnonzero(all_ok)[np.argmin(obj[all_ok])])
            method = "exact"
        elif hard_ok.any():
            worst_soft = prog[:, ~hard].min(axis=1)
            pool = np.flatnonzero(hard_ok)
            order = np.lexsort((obj[pool], -worst_soft[pool]))
            idx = int(pool[order[0]])
            method = "relaxed"
        else:
            worst_hard = prog[:, hard].min(axis=1)
            worst_soft = prog[:, ~hard].min(axis=1)
            # Hard progress first, then soft (so a body stuck heading at the
            # surface still turns its VCP out), then the nominal.
            order = np.lexsort((obj, -worst_soft, -worst_hard))
            idx = int(order[0])
            method = "backup"
        u = U[idx].copy()
        return u, self._info(method, method == "exact", u, v_nom, w_nom, cons, v_max)

    def _clip_nominal(
        self, action: tuple[float, float] | np.ndarray, v_max: float
    ) -> tuple[float, float]:
        """The nominal (v, ω) the candidate set is built around, clipped to the box."""
        return (
            float(np.clip(float(action[0]), 0.0, v_max)),
            float(np.clip(float(action[1]), -self.cfg.omega_max, self.cfg.omega_max)),
        )

    def _info(
        self,
        method: str,
        feasible: bool,
        u: np.ndarray,
        v_nom: float,
        w_nom: float,
        cons: list[Constraint],
        v_max: float,
    ) -> dict:
        """Build the per-step info dict. Shared with NoOpShield so the shielded
        and unshielded arms report exactly the same keys.

        Keys: method ("exact" | "relaxed" | "backup" | "none"), feasible,
        intervention, min_h, min_clearance, min_clearance_hard, min_h_obstacle,
        min_h_boundary, h_obstacle, h_boundary, n_constraints, v_nom, v_exec,
        stalled, immobile.

        `h_obstacle` / `h_boundary` (and their minima) are the SOFT (VCP) rows
        only, so each list stays single-unit and matches what the docstrings
        say it is: obstacle m², boundary m. Mixing in the hard body rows, which
        are metres for both kinds, would silently make `h_obstacle` a list of
        two different units.

        `min_h` is the worst margin over ALL rows, hard and soft, and is
        therefore mixed-unit; it must never be reported as a distance.

        `min_clearance` is the worst signed clearance in metres over ALL rows:
        the minimum of the BODY clearances (hard rows) and the VIRTUAL CONTROL
        POINT clearances (soft rows, measured d = 0.1 m ahead of the body). It
        is therefore a conservative lower bound on safety and NOT by itself a
        body-to-surface distance — the soft rows win whenever the agent heads
        toward a surface, so a body a centimetre clear of an obstacle can
        report a `min_clearance` of −0.09 m with no contact anywhere. The
        body-to-surface distance is the minimum over the HARD rows alone
        (`min(c.clearance for c in cons if c.hard)`); that, not this, is the
        quantity that goes to zero exactly when the env's contact flag fires.
        It is reported separately as `min_clearance_hard` (reported, never
        gated: a violation is still the env's own contact flag, D89).

        `immobile` is `v_exec <= STALL_V_FRAC * v_max` with no condition on the
        nominal; `stalled` additionally requires `v_nom >= MOVE_V_FRAC * v_max`
        ("the policy wanted to move"). A policy that lowers its own commanded
        speed can therefore be immobile without ever being stalled, which is
        why both are reported.

        `v_exec` is the speed the filter RETURNS, not necessarily the speed the
        environment applies: with `cfg.prep_steps > 0` the env zeroes the
        pursuer's speed for the first prep steps after the filter has run. The
        default is `prep_steps = 0`, where the two coincide.
        """
        h_obstacle = [c.h for c in cons if c.kind == "obstacle" and not c.hard]
        h_boundary = [c.h for c in cons if c.kind == "boundary" and not c.hard]
        v_exec = float(u[0])
        immobile = v_exec <= STALL_V_FRAC * v_max
        stalled = immobile and v_nom >= MOVE_V_FRAC * v_max
        return {
            "method": method,
            "feasible": bool(feasible),
            "intervention": bool(
                not np.allclose(u, [v_nom, w_nom], atol=INTERVENTION_ATOL)
            ),
            "min_h": float(min(c.h for c in cons)),  # mixed-unit, NOT a distance
            "min_clearance": float(min(c.clearance for c in cons)),  # metres
            # Body-to-surface distance alone (hard rows only); reported, never
            # gated -- see the docstring above.
            "min_clearance_hard": float(min(c.clearance for c in cons if c.hard)),
            "min_h_obstacle": min(h_obstacle) if h_obstacle else None,
            "min_h_boundary": min(h_boundary) if h_boundary else None,
            "h_obstacle": h_obstacle,
            "h_boundary": h_boundary,
            "n_constraints": len(cons),
            "v_nom": float(v_nom),
            "v_exec": v_exec,
            "stalled": bool(stalled),
            "immobile": bool(immobile),
        }


class NoOpShield(CBFShield):
    """Pass-through filter for the pre-registered UNSHIELDED negative control.

    D83 requires the Step 6 gate to carry an unshielded arm. Running one
    through `run_shielded_duel` with a hand-rolled pass-through object would
    write a shield block that looks like a measurement but is not, so the
    control gets an explicit class here instead.

    It returns the nominal action unchanged (clipped only to the action box,
    which the env would clip anyway) and reports `method="none"` — a distinct
    tier in the histogram, so this arm's zero infeasibility rate and zero
    intervention rate can never be read as a run on which every step really did
    satisfy every barrier row.

    It DOES still measure the barriers: `min_h` / `min_clearance` are
    properties of the state and the static map, not of the filter, and the
    unshielded arm's clearance floor is precisely the number the shielded
    arm's floor is compared against. Nothing it reports is invented — the one
    value that has no meaning here, tier feasibility, is True because no
    candidate was ever tested and therefore none failed.
    """

    def filter(
        self,
        action: tuple[float, float] | np.ndarray,
        state: np.ndarray,
        obstacles: list[dict],
        v_max: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Return the nominal unchanged with `method="none"`. Same info keys as
        `CBFShield.filter`, so the two arms are comparable step by step."""
        v_max = self.cfg.v_max if v_max is None else v_max
        v_nom, w_nom = self._clip_nominal(action, v_max)
        u = np.array([v_nom, w_nom], dtype=np.float64)
        cons = self.constraints(state, obstacles)
        return u, self._info("none", True, u, v_nom, w_nom, cons, v_max)


class RecoveryShield(CBFShield):
    """`CBFShield` plus the scripted rotation recovery of S16 arm `REC`.

    D125-D128: a shielded pursuer can spend hundreds of steps asking for a
    forward command the filter refuses while a pure turn is admissible the
    whole time, and nothing in its observation or its discounted return tells
    it to take that turn. This subclass takes it for it: after
    `recovery_after` consecutive steps whose filtered result is `immobile`,
    the nominal is replaced by a pure turn `(0, ±ω_max)` -- both signs are
    filtered and the one with the larger executed `|ω|` wins (ties go to the
    positive sign, so the rule is deterministic) -- provided that `|ω|`
    clears `RECOVERY_OMEGA_MIN`. Otherwise the normal result stands.

    **The recovery REQUESTS a pure turn; it does not execute one.** What it
    executes is whatever `CBFShield.filter` returns for that request, and
    that is occasionally a TRANSLATION: `filter` scores an enumerated
    candidate set by `w_v·(v − v_nom)² + w_ω·(ω − ω_nom)²`, so a candidate
    carrying `v < √(w_ω·ω_max² / w_v) ≈ 0.23 m/s` can beat `(0, 0)` when its
    ω is nearer `±ω_max`. A deterministic instance is pinned by
    `test_a_firing_recovery_may_execute_forward_motion` (v = 0.2308 m/s at a
    contact-free pose). The base rate is ~4e-6 of contact-free states, but it
    is not zero, so **the safety argument is the ordinary one: a recovery
    action is CBF-filtered exactly like any other action, so its safety rests
    on the filter, not on a zero speed** (prereg amendments S16-A1/A2). The
    behaviour is deliberately KEPT rather than forced to `v = 0`: it is what
    §2 pre-registers, it is bit-for-bit the path that produced D125, and
    admissible forward motion escapes the immobility better than a turn.
    `n_recoveries` / `n_recoveries_translating` count firings and the subset
    that translated, so a round can report that fraction.

    The reported info carries every key `CBFShield` reports plus
    `recovery_fired`. On a recovery step the *barrier* and *executed* fields
    describe the turn that was executed, but `v_nom`, `intervention` and
    `stalled` are still measured against the POLICY's own nominal -- the
    recovery must not launder away the fact that the policy asked to move and
    was refused, or the stall metrics the round is judged on would read as
    solved wherever the recovery merely papered over them.

    The immobile run is shield state, not episode state, and a shield has no
    episode hook: a run that is still open when an episode ends continues
    into the next one. Build one `RecoveryShield` per episode if that matters.
    The two counters span episodes for the same reason, which is what a
    per-run total wants.
    """

    def __init__(
        self, cfg: EnvConfig, recovery_after: int = RECOVERY_AFTER, **cbf_kwargs
    ):
        super().__init__(cfg, **cbf_kwargs)
        self.recovery_after = int(recovery_after)
        self._immobile_run = 0
        self.n_recoveries = 0  # steps on which the recovery fired
        self.n_recoveries_translating = 0  # of those, the ones with v_exec > 0

    def filter(
        self,
        action: tuple[float, float] | np.ndarray,
        state: np.ndarray,
        obstacles: list[dict],
        v_max: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        """`CBFShield.filter`, with the rotation recovery on top."""
        u, info = super().filter(action, state, obstacles, v_max)
        if not info["immobile"]:
            self._immobile_run = 0
            info["recovery_fired"] = False
            return u, info
        self._immobile_run += 1
        if self._immobile_run < self.recovery_after:
            info["recovery_fired"] = False
            return u, info
        best_u, best_info = None, None
        for omega in (self.cfg.omega_max, -self.cfg.omega_max):
            u_try, info_try = super().filter((0.0, omega), state, obstacles, v_max)
            if best_u is None or abs(u_try[1]) > abs(best_u[1]):
                best_u, best_info = u_try, info_try
        if abs(best_u[1]) <= RECOVERY_OMEGA_MIN:
            info["recovery_fired"] = False
            return u, info
        v_max = self.cfg.v_max if v_max is None else v_max
        v_nom, w_nom = self._clip_nominal(action, v_max)
        rec = self._info(
            best_info["method"],
            best_info["feasible"],
            best_u,
            v_nom,
            w_nom,
            self.constraints(state, obstacles),
            v_max,
        )
        rec["recovery_fired"] = True
        self.n_recoveries += 1
        self.n_recoveries_translating += int(float(best_u[0]) > 0.0)
        return best_u, rec


def escape_omega_bounds(
    shield: CBFShield, state: np.ndarray, obstacles: list[dict]
) -> tuple[float, float] | None:
    """The interval of ω `shield`'s DCBF rows admit at v = 0, ∩ the action box.

    Each row `c0·v + c1·ω ≥ c2` collapses at v = 0 to `c1·ω ≥ c2`: a lower
    bound when c1 > 0, an upper bound when c1 < 0, and a feasibility test
    (`0 ≥ c2`) when c1 is exactly zero -- which is what every HARD body row
    is, since translating is the only way to change a body barrier. Returns
    None when the rows admit no ω at all.

    This is the LINEARISED admissible set, not the filter's own exact
    one-step test, so it is the set the QP's candidates are located from and
    not a promise about what `CBFShield.filter` will return.
    """
    lo, hi = -shield.cfg.omega_max, shield.cfg.omega_max
    for _c0, c1, c2 in shield._dcbf_rows(shield.constraints(state, obstacles)):
        if c1 == 0.0:
            if c2 > 0.0:
                return None
            continue
        bound = c2 / c1
        if c1 > 0.0:
            lo = max(lo, bound)
        else:
            hi = min(hi, bound)
    return None if lo > hi else (lo, hi)


def escape_features(
    shield: CBFShield | None, state: np.ndarray, obstacles: list[dict]
) -> np.ndarray:
    """`escape_omega_bounds` normalised by ω_max and clipped to [−1, 1].

    The two observation features of S16 arm `OBS`: which way the filter will
    let the robot turn if it stops. `(0, 0)` means no turn is admissible;
    `(−1, 1)` means every turn is, which is also what `shield=None` reports —
    an unshielded run refuses nothing.
    """
    if shield is None:
        return np.array([-1.0, 1.0], dtype=np.float32)
    bounds = escape_omega_bounds(shield, state, obstacles)
    if bounds is None:
        return np.zeros(2, dtype=np.float32)
    scaled = np.asarray(bounds, dtype=np.float64) / shield.cfg.omega_max
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def _backup_choice(contact_free: np.ndarray, worst: np.ndarray, obj: np.ndarray) -> int:
    """Index of the backup candidate: contact-free first, then the largest
    worst-case clearance, then nearest the nominal.

    Split out of `PredictiveShield.filter` so the ordering is unit-testable
    on its own. The contact-free key is primary because a candidate whose
    rollout fired a contact flag has already been projected back OUT of the
    surface by the env, so its post-projection clearance reads as exactly 0 —
    the same value a contact-free candidate resting on the surface reports.
    Without this key the tie between the two could go to the contacting one
    on `obj` alone. (Unreachable while the shield itself keeps the pose
    strictly off every surface, but free to close.)
    """
    return int(
        np.lexsort((obj, -np.asarray(worst), -np.asarray(contact_free, dtype=np.int8)))[
            0
        ]
    )


class PredictiveShield(CBFShield):
    """S11's SECOND hard filter: sampling lookahead through the env's own step.

    Same interface, same info keys and the same guarantee as `CBFShield`, built
    a different way — the pre-registered test of whether the stall surface is a
    property of hard shields in general or of the CBF-QP construction
    (`docs/pathway/second_filter_prereg_2026-09-18.md` §1).

    Where `CBFShield` accepts a candidate by a LINEARISED barrier condition,
    this one accepts it by SIMULATION: hold the candidate for `horizon` steps,
    then one full-stop step, and accept iff no simulated step raised a contact
    flag and every simulated body clearance stayed at or above `margin` (δ).
    Among the accepted candidates it executes the one nearest the nominal in
    the same weighted metric `_qp_candidates` uses, `w_v (Δv)² + w_ω (Δω)²`
    (`method = "exact"`). If none is accepted it executes the candidate whose
    rollout stayed contact-free and whose WORST simulated clearance is largest
    (`method = "backup"`, `_backup_choice`), ties toward the nominal.
    `feasible` is True on the exact tier only. There is no `relaxed` tier: the
    acceptance test is a single predicate, not a row set.

    **What is guaranteed.** Every candidate — the clipped nominal and `u = 0`
    included — is first SNAPPED onto the float32 action grid (`_snap`) and only
    then simulated, and the snapped command is what `filter` returns. The
    rollout then calls `pe2.env.pursuer_step`, which is literally the function
    `PursuitEvasionEnv.step` applies to each agent, on the env's own inset
    bounds and contact radii (not a transcription of it — contrast
    `CBFShield.successor_barriers`, which must transcribe `euler_step` to
    vectorise and is bound to it by a bitwise test). The environment is
    deterministic and the pursuer's successor pose and contact flags depend on
    nothing but its own pose and command, `cfg`, the obstacles and the bounds —
    not on the evader, not on the step index. Therefore:

    * on the float64 execution path (`pe2.scripted.run_shielded_duel`, which
      hands the returned doubles straight to `env.step`) the first simulated
      step IS the executed step, bit for bit;
    * on the float32 path (`SearchMDP._step_inner` re-normalises the returned
      command into a float32 Box action, which `SingleAgentWrapper._scale_action`
      de-normalises again) the round-trip is the identity on a snapped command,
      because `_snap` applies exactly that arithmetic and is idempotent — so the
      first simulated step is again the executed step, bit for bit.

    Without the snap the two would differ by up to ~1.5e-8 m/s of commanded
    speed (7.5e-10 m of position after `dt`), which this filter — unlike the
    γ-contracted CBF shield — routinely has less clearance than, since its
    exact tier accepts ANY strictly contact-free successor.

    **The one exception is the prep phase.** `PursuitEvasionEnv.step` forces
    `v_p = 0` while `current_step < cfg.prep_steps`, so there the env executes a
    full stop instead of the accepted command. The filter ignores that rule. It
    is sound — a full stop from a contact-free pose is contact-free (see below)
    — but it is the one case where the executed step is not the simulated one.
    It is moot in practice: `cfg.prep_steps` is 0 in `EnvConfig`'s locked
    default and in every config this project runs.

    **`u = 0` is always a candidate** (`_qp_candidates` puts it in the set
    unconditionally, and `_snap` maps it to itself) and from a contact-free pose
    it is always accepted at δ = 0: a zero command leaves the pose bit-identical
    (`euler_step` adds `0 * cos(θ) * dt`), so neither the wall clip nor the
    obstacle projection fires and the clearance is unchanged. Hence at δ = 0 the
    backup tier is unreachable from a contact-free state, and even when it is
    reached (δ > 0, or a state already in contact) it can never execute INTO
    contact while a contact-free candidate exists, because `u = 0` is one.
    Asserted by `TestPredictiveShieldTiers` in `tests/pe2/test_shield.py`.

    **δ (`margin`) is the clearance threshold, applied once.** The rollout runs
    on the ENV's bounds and the ENV's contact radii, so the simulated poses are
    the env's poses for any δ; δ enters only as the acceptance threshold on the
    body clearance (`_body_clearance`, an exact distance identical to the
    base class's hard-row clearance at margin 0). Feeding the rollout the
    base class's already-inset `self._x_lim` instead would apply δ twice.
    `self.margin` is still passed to `CBFShield.__init__`, so the reported
    barrier rows (and therefore `min_clearance`, which `shield_obs` reads)
    keep exactly the meaning they have under the CBF shield.

    **Horizon.** `horizon = 0` degenerates to a clip (V-P (i)): the only
    simulated step is the full stop, which every candidate survives from a
    contact-free pose, so the nominal — distance 0 in the metric — is returned
    unchanged (snapped). Candidates are deduplicated and tested in increasing
    distance from the nominal, and the first acceptable one is returned, so an
    uncontested step costs `horizon + 1` `pursuer_step` calls in total; a step
    where the nominal is rejected costs up to one early-exiting rollout per
    distinct candidate, and only a step where NOTHING is acceptable pays a
    second, full pass to rank the backup tier.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        horizon: int = 4,
        margin: float = 0.0,
        **cbf_kwargs,
    ):
        super().__init__(cfg, margin=margin, **cbf_kwargs)
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, (int, np.integer))
            or horizon < 0
        ):
            # Loud on `PredictiveShield(cfg, 0.2)` — a caller assuming the base
            # class's positional order, where the third argument is `vcp_d`,
            # would otherwise silently get horizon = 0, a filter that is a clip.
            raise ValueError(f"horizon must be a non-negative int, got {horizon!r}")
        self.horizon = int(horizon)
        # The env's OWN inset bounds (`PursuitEvasionEnv._x_min/_x_max`), which
        # carry no δ — see the δ note in the class docstring. Identical to
        # (−self._x_lim, self._x_lim) at the default margin = 0, and pinned to
        # a live env by `test_sim_bounds_are_the_envs_bounds`.
        lim = cfg.arena_size / 2.0 - cfg.agent_radius
        self._sim_x_lim = (-lim, lim)
        self._sim_y_lim = (-lim, lim)

    def _snap(self, v: float, omega: float, v_max: float) -> tuple[float, float]:
        """Round `(v, ω)` onto the float32 action grid the env is driven on.

        Replicates, in order, `SearchMDP._step_inner`'s normalisation into a
        float32 Box action and `SingleAgentWrapper._scale_action`'s clip and
        de-normalisation. Idempotent, so a snapped command is a fixed point of
        that round-trip and the simulated step is the executed step (see the
        class docstring). The perturbation is ≤ 1.5e-8 m/s, far below any
        readout, and the box endpoints (0, `v_max`, ±`omega_max`) map to
        themselves exactly.

        `v_max` is the caller's, not `cfg.v_max`: the wrapper normalises by the
        ACTING ROLE's speed cap, so a shielded evader (`run_shielded_duel` with
        `shielded_role="evader"`, cap `evader_v_max`) must snap on its own grid.
        Hard-coding `cfg.v_max` would clip such an evader to the pursuer's cap.
        """
        a = np.array(
            [2.0 * v / v_max - 1.0, omega / self.cfg.omega_max], dtype=np.float32
        )
        a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
        return float((a[0] + 1.0) / 2.0 * v_max), float(a[1] * self.cfg.omega_max)

    def _body_clearance(self, pose, obstacles: list[dict]) -> float:
        """Signed body-to-surface distance in metres, min over walls and
        obstacles. Exact: the same quantity as `min(c.clearance for c in
        CBFShield(cfg, margin=0).constraints(pose, obstacles) if c.hard)`, and
        zero exactly where the env's contact predicates change sign."""
        x, y = float(pose[0]), float(pose[1])
        clearance = min(
            self._sim_x_lim[1] - x,
            x - self._sim_x_lim[0],
            self._sim_y_lim[1] - y,
            y - self._sim_y_lim[0],
        )
        for obs in obstacles:
            dist = float(np.hypot(x - obs["x"], y - obs["y"]))
            clearance = min(clearance, dist - (obs["radius"] + self.cfg.agent_radius))
        return float(clearance)

    def _accept(self, u, state: np.ndarray, obstacles: list[dict]) -> bool:
        """True iff the rollout of `u` is contact-free with clearance ≥ δ.

        Early-exits on the first rejecting step — the common case is a rejection
        on step 1, and the exact tier needs nothing but this boolean. Equal by
        construction to `all(_rollout(u, ...))`-style evaluation:
        `_accept == (contact_free and worst >= margin)` of `_rollout`, which
        `test_accept_agrees_with_the_full_rollout` pins.

        `u` must already lie in the action box (`_qp_candidates` clips every
        candidate); the env would clip it, the rollout does not.
        """
        pose = np.array(state, dtype=float)
        v, omega = float(u[0]), float(u[1])
        for k in range(self.horizon + 1):
            v_k, w_k = (v, omega) if k < self.horizon else (0.0, 0.0)
            pose, wall_contact, obstacle_contact = pursuer_step(
                pose, v_k, w_k, self.cfg, obstacles, self._sim_x_lim, self._sim_y_lim
            )
            if wall_contact or obstacle_contact:
                return False
            if self._body_clearance(pose, obstacles) < self.margin:
                return False
        return True

    def _rollout(
        self, u, state: np.ndarray, obstacles: list[dict]
    ) -> tuple[bool, float]:
        """Full rollout, no early exit: (contact-free, worst body clearance).

        Only the backup tier needs this — it ranks candidates by how much
        clearance their WHOLE rollout keeps, so it cannot stop at the first
        rejection. At δ = 0 it is never called (`u = 0` is always accepted).
        """
        pose = np.array(state, dtype=float)
        v, omega = float(u[0]), float(u[1])
        contact_free = True
        worst = np.inf
        for k in range(self.horizon + 1):
            v_k, w_k = (v, omega) if k < self.horizon else (0.0, 0.0)
            pose, wall_contact, obstacle_contact = pursuer_step(
                pose, v_k, w_k, self.cfg, obstacles, self._sim_x_lim, self._sim_y_lim
            )
            contact_free = contact_free and not (wall_contact or obstacle_contact)
            worst = min(worst, self._body_clearance(pose, obstacles))
        return contact_free, float(worst)

    def _candidates(
        self, v_nom: float, w_nom: float, cons: list[Constraint], v_max: float
    ) -> np.ndarray:
        """The CBF shield's candidate grid, snapped to the float32 action grid
        and deduplicated, first occurrence first.

        The barrier rows LOCATE the candidates (same set as `CBFShield`, so the
        two filters are compared on the same actions); the rollout, not the
        rows, decides which are admissible. Duplicates — `_qp_candidates`
        returns ~61 points of which ~13 are distinct, and snapping can merge a
        few more — have identical `obj` and identical rollouts, so dropping them
        cannot change the returned command, only the work done.
        """
        U = _qp_candidates(
            v_nom,
            w_nom,
            self._dcbf_rows(cons),
            v_max,
            self.cfg.omega_max,
            self.w_v,
            self.w_omega,
        )
        for i in range(len(U)):
            U[i] = self._snap(U[i, 0], U[i, 1], v_max)
        _uniq, first = np.unique(U, axis=0, return_index=True)
        return U[np.sort(first)]

    def filter(
        self,
        action: tuple[float, float] | np.ndarray,
        state: np.ndarray,
        obstacles: list[dict],
        v_max: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Execute the accepted candidate nearest the nominal; see the class
        docstring for the guarantee. Returns (safe_action, info) with exactly
        `CBFShield.filter`'s info keys, `method` ∈ {"exact", "backup"}."""
        v_max = self.cfg.v_max if v_max is None else v_max
        v_nom, w_nom = self._snap(*self._clip_nominal(action, v_max), v_max)
        cons = self.constraints(state, obstacles)
        U = self._candidates(v_nom, w_nom, cons, v_max)
        obj = self.w_v * (U[:, 0] - v_nom) ** 2 + self.w_omega * (U[:, 1] - w_nom) ** 2
        # Test candidates nearest-first and stop at the first acceptable one:
        # identical to testing all and taking the argmin over the accepted set,
        # but the common case (the nominal is safe) costs one rollout. A stable
        # sort keeps `_qp_candidates`' order among equal-distance candidates,
        # so the choice is deterministic.
        for i in np.argsort(obj, kind="stable"):
            if self._accept(U[i], state, obstacles):
                u = U[i].copy()
                return u, self._info("exact", True, u, v_nom, w_nom, cons, v_max)
        # Nothing acceptable: rank the backup tier, which needs the whole
        # rollout of every candidate rather than its first rejecting step.
        rolled = [self._rollout(u, state, obstacles) for u in U]
        idx = _backup_choice(
            np.array([r[0] for r in rolled]), np.array([r[1] for r in rolled]), obj
        )
        u = U[idx].copy()
        return u, self._info("backup", False, u, v_nom, w_nom, cons, v_max)
