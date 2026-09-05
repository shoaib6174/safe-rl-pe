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

3-tier fallback (root SafeActionResolver pattern, minus the unused Tier-1 SVM
classifier) selects from one candidate set that always includes every row,
hard and soft alike: exact QP (all constraints) → relaxed QP (boundary only;
obstacles are lowest priority) → backup controller (brake + turn away from
nearest danger). Nothing is ever dropped from the candidate set ahead of a
tier trying it. A violation is defined at env level (wall/obstacle contact
flags); this module only maintains h > 0.

h > 0 is a CONSERVATIVE early warning only for the SOFT rows — they are
evaluated at the virtual control point d = 0.1 m ahead of the body, so in the
heading direction the barrier reaches zero roughly d BEFORE the body touches
anything (on top of the agent_radius already folded into the constraint
radii); h = 0 there is not the contact surface and h < 0 does not imply
contact. The HARD rows are exact: h = 0 is the contact surface and h < 0
means the env's own contact predicate has fired.

Units: `Constraint.h` is a barrier value whose units depend on the kind
(obstacle barriers are m², boundary barriers are m), so it is a QP margin and
never a distance. `Constraint.clearance` is the signed distance in metres for
both kinds and is the only field that may be reported as a distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pe2.config import EnvConfig
from pe2.env import wrap_angle

INTERVENTION_ATOL = 1e-4  # |Δu| above this counts as an intervention (root convention)
HEAD_ON_TIE_TOL = 1e-9  # |error| within this of pi counts as an exact head-on tie
STALL_V_FRAC = 0.05  # executed speed <= this fraction of v_max counts as stopped
MOVE_V_FRAC = 0.5  # nominal speed >= this fraction of v_max counts as "wanted to move"
BACKUP_V_FRAC = 0.1  # backup-tier crawl speed, as a fraction of the effective v_max
# NOT independent of STALL_V_FRAC: BACKUP_V_FRAC > STALL_V_FRAC by construction,
# so a backup-tier step can never be flagged `stalled` or `immobile`. Tier-3
# immobility is therefore invisible to those two flags and is counted
# separately as a backup run (ShieldMetrics.longest_backup_run / `pinned`).
# Changing either constant must preserve that ordering, or update both metrics.


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

    def backup(
        self, state: np.ndarray, obstacles: list[dict], v_max: float | None = None
    ) -> np.ndarray:
        """Tier 3: brake + turn away from the nearest danger (root pattern).

        Exact head-on ties turn CCW. The crawl speed is BACKUP_V_FRAC of the
        EFFECTIVE cap `v_max` (the caller's per-role cap, as in `filter`), so
        the returned action never exceeds the cap it was given.
        """
        v_max = self.cfg.v_max if v_max is None else v_max
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        dangers = [
            (0.0, self._x_lim - x),
            (np.pi, x + self._x_lim),
            (np.pi / 2, self._y_lim - y),
            (-np.pi / 2, y + self._y_lim),
        ]
        for obs in obstacles:
            dx, dy = obs["x"] - x, obs["y"] - y
            dist = max(np.hypot(dx, dy) - obs["radius"], 0.01)
            dangers.append((float(np.arctan2(dy, dx)), float(dist)))
        away = min(dangers, key=lambda d: d[1])[0] + np.pi
        err = wrap_angle(away - theta)
        # wrap_angle maps an exact head-on error (±pi) to -pi, which would make
        # the turn direction depend on float rounding. Convention: ties turn CCW.
        if err <= -np.pi + HEAD_ON_TIE_TOL:
            err = np.pi
        omega = float(np.clip(3.0 * err, -self.cfg.omega_max, self.cfg.omega_max))
        return np.array([BACKUP_V_FRAC * v_max, omega], dtype=np.float64)

    def filter(
        self,
        action: tuple[float, float] | np.ndarray,
        state: np.ndarray,
        obstacles: list[dict],
        v_max: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Project a nominal action into the safe set; never blocks capture.

        Returns (safe_action, info). info keys: method ("exact" | "relaxed" |
        "backup"), feasible, intervention (bool), min_h, min_clearance,
        min_h_obstacle, min_h_boundary, h_obstacle, h_boundary, n_constraints,
        v_nom, v_exec, stalled, immobile.

        `min_h` is the worst quadratic-program margin over constraints of BOTH
        kinds and is therefore mixed-unit (obstacle m², boundary m). It must
        never be reported as a distance. `min_clearance` is the worst signed
        distance in metres and is the reportable safety margin.

        `immobile` is `v_exec <= STALL_V_FRAC * v_max` with no condition on the
        nominal; `stalled` additionally requires `v_nom >= MOVE_V_FRAC * v_max`
        ("the policy wanted to move"). A policy that lowers its own commanded
        speed can therefore be immobile without ever being stalled, which is
        why both are reported.

        `v_exec` is the speed this filter RETURNS, not necessarily the speed
        the environment applies: with `cfg.prep_steps > 0` the env zeroes the
        pursuer's speed for the first prep steps after the filter has run.
        The default is `prep_steps = 0`, where the two coincide.
        """
        v_max = self.cfg.v_max if v_max is None else v_max
        v_nom, w_nom = self._clip_nominal(action, v_max)
        cons = self.constraints(state, obstacles)

        # Tier 1: exact — all constraints.
        u, feasible = self._solve(v_nom, w_nom, cons, v_max)
        method = "exact"

        # Tier 2: relaxed — boundary only (obstacles lowest priority, root).
        if not feasible:
            boundary = [c for c in cons if c.kind == "boundary"]
            u, feasible = self._solve(v_nom, w_nom, boundary, v_max)
            method = "relaxed"

        # Tier 3: backup controller.
        if not feasible:
            u = self.backup(state, obstacles, v_max)
            method = "backup"

        # info["feasible"] is derived from the tier, not from the local
        # `feasible` above: only the exact tier means the full constraint set
        # was satisfied, whereas the local merely records whether the most
        # recent _solve() call returned a point (true again after a
        # successful relaxed solve, when the reported feasibility must stay
        # False). ShieldMetrics.record reads info["feasible"] rather than
        # re-deriving it from the method string, so a new tier cannot
        # disagree with itself.
        return u, self._info(method, method == "exact", u, v_nom, w_nom, cons, v_max)

    def _clip_nominal(
        self, action: tuple[float, float] | np.ndarray, v_max: float
    ) -> tuple[float, float]:
        """The nominal (v, ω) actually fed to the program, clipped to the box."""
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
        and unshielded arms report exactly the same keys."""
        h_obstacle = [c.h for c in cons if c.kind == "obstacle"]
        h_boundary = [c.h for c in cons if c.kind == "boundary"]
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

    def _solve(
        self, v_nom: float, w_nom: float, cons: list[Constraint], v_max: float
    ) -> tuple[np.ndarray, bool]:
        raise NotImplementedError("replaced in Task 5")


class NoOpShield(CBFShield):
    """Pass-through filter for the pre-registered UNSHIELDED negative control.

    D83 requires the Step 6 gate to carry an unshielded arm. Running one
    through `run_shielded_duel` with a hand-rolled pass-through object would
    write a shield block that looks like a measurement but is not, so the
    control gets an explicit class here instead.

    It returns the nominal action unchanged (clipped only to the action box,
    which the env would clip anyway) and reports `method="none"` — a distinct
    tier in the histogram, so this arm's zero infeasibility rate and zero
    intervention rate can never be read as an all-exact QP measurement.

    It DOES still measure the barriers: `min_h` / `min_clearance` are
    properties of the state and the static map, not of the filter, and the
    unshielded arm's clearance floor is precisely the number the shielded
    arm's floor is compared against. Nothing it reports is invented — the one
    value that has no meaning here, QP feasibility, is True because no program
    was solved and therefore none failed.
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
