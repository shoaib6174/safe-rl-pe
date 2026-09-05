# Step 6 Shield v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the pe2 safety shield so it never lets the shielded body touch a wall or obstacle at δ = 0 (provably, in floating point), then re-run the pre-registered Step 6 gates on fresh seed streams.

**Architecture:** Each surface gets a hard *body* row (the env's own contact predicate) and a soft *VCP* row (steering). Candidate actions come from the existing D85 enumeration over linearized rows plus `u = 0` and the nominal; every candidate is then judged by the exact one-step barrier on the true Euler successor (shared `pe2.env.euler_step`), with zero tolerance. Three tiers select from the same candidate set: exact (all rows), relaxed (hard rows + best-effort soft), backup (best-effort hard). The brake-and-turn heuristic is deleted.

**Tech Stack:** Python 3, numpy, gymnasium, pytest (`./venv/bin/python -m pytest`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), ruff.

**Spec:** `docs/pathway/step6_shield_v2_prereg_2026-09-05.md` (design Part B, pre-registration Part C). The S119 criteria are in `scripts/pe2_step06_gate.py`'s docstring.

## Global Constraints

- Run everything with `./venv/bin/python`; tests as `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q`.
- Lint only the files you touch: `./venv/bin/ruff check <files>` and `./venv/bin/ruff format --check <files>`.
- **No pre-registered criterion, threshold, definition, N, or δ may change.** Seed streams are fixed by the spec: `MAIN_SEED0 = 800000`, `PROBE_SEED0 = 810000`, `N_EPISODES = 1000`, `G2_CAPTURE_FACTOR = 0.5`, `G4_MIN_VIOLATION_EPISODE_RATE = 0.05`, `ROBUST_DELTA = 0.05`.
- Violation = env contact flag of the shielded role, magnitude-blind (unchanged).
- Tier names stay `("exact", "relaxed", "backup", "none")` (`pe2.metrics.TIERS`).
- Shield objective weights unchanged: `w_v = 150.0`, `w_omega = 1.0`, `gamma = 0.2`, `vcp_d = 0.1`.
- Bit-identity requirement: the shield's hard-row values at the predicted successor must equal what the env computes for that state. Vectorisation is optional; identity is not. If the bitwise test in Task 4 fails with numpy array ufuncs, evaluate hard rows with scalar `np.hypot` in a loop over candidates.
- Commit after every task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Work on branch `step6-shield-v2` (create from `main`). Never run training. Gate runs are CPU only.
- The `results/` tree is git-ignored; never delete v1 artifacts (Task 8 moves them).

---

## File map

| file | responsibility after this plan |
|---|---|
| `pe2/env.py` | adds module-level `euler_step(pose, v, omega, dt)`; `_advance_agent` calls it |
| `pe2/shield.py` | `Constraint` gains `hard`, `scale`; `CBFShield.constraints` emits body+VCP rows; `_qp_candidates` (enumeration, returns `(K,2)` array); `CBFShield.successor_barriers` (vectorised true one-step barriers); `CBFShield.progress` (public, one action); `filter` with the three tiers; `backup()`, `BACKUP_V_FRAC`, `HEAD_ON_TIE_TOL`, `FEAS_TOL`, `_solve_qp_2d` removed |
| `pe2/metrics.py` | docstring/comment updates only (backup tier meaning) |
| `tests/pe2/test_env.py` | `euler_step` bit-identity test |
| `tests/pe2/test_shield.py` | rows, candidates, exact check, tiers, guarantee tests; old backup/relaxed-semantics tests replaced |
| `scripts/pe2_step06_gate.py` | seeds 800000/810000; pre-registration pointer; per-episode `start_in_barrier_set` / `start_hard_feasible`; constraint dump includes `hard`/`scale` |
| `scripts/pe2_step06_visual_gate.py` | C1 via `CBFShield.progress`; bound docstring; episodes from new streams |
| `tests/pe2/test_step06_gate.py` | seed constants updated |

---

### Task 1: Shared Euler kinematics `euler_step`

**Files:**
- Modify: `pe2/env.py` (module level, above `class PursuitEvasionEnv`; and `_advance_agent` at lines 136–158)
- Test: `tests/pe2/test_env.py`

**Interfaces:**
- Produces: `pe2.env.euler_step(pose, v: float, omega: float, dt: float) -> tuple[float, float, float]` returning the UNCLIPPED `(x, y, theta)` with `theta` wrapped by `wrap_angle`. Operation order is part of the contract.

- [ ] **Step 1: Write the failing test**

Append to `tests/pe2/test_env.py`:

```python
class TestEulerStepIdentity:
    """The shield predicts the env's next pose with this function; the
    zero-contact argument in the Step 6 v2 spec (B4) needs bit identity."""

    def test_advance_agent_uses_euler_step_bitwise(self):
        from pe2.env import euler_step

        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        rng = np.random.default_rng(12345)
        for _ in range(10_000):
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            # Keep away from the walls so the clip never engages.
            env.pursuer_pose[:] = (
                rng.uniform(-3.0, 3.0),
                rng.uniform(-3.0, 3.0),
                rng.uniform(-np.pi, np.pi),
            )
            env.obstacles = []
            v = float(rng.uniform(0.0, cfg.v_max))
            w = float(rng.uniform(-cfg.omega_max, cfg.omega_max))
            predicted = euler_step(env.pursuer_pose.copy(), v, w, cfg.dt)
            env._advance_agent("pursuer", v, w)
            assert tuple(env.pursuer_pose) == predicted  # bitwise, no approx

    def test_zero_action_is_the_identity_on_position(self):
        from pe2.env import euler_step

        pose = np.array([1.2345678901234567, -2.3456789012345678, 0.7])
        x, y, _ = euler_step(pose, 0.0, 0.0, 0.05)
        assert x == pose[0] and y == pose[1]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_env.py::TestEulerStepIdentity -q`
Expected: FAIL with `ImportError: cannot import name 'euler_step'`.

- [ ] **Step 3: Implement**

In `pe2/env.py`, after the `OUTCOME_*` constants add:

```python
def euler_step(pose, v: float, omega: float, dt: float) -> tuple[float, float, float]:
    """One UNCLIPPED unicycle Euler step -- the kinematics of this env.

    Shared with pe2.shield so the shield's one-step prediction is the env's
    next state bit for bit; the shield's zero-contact argument (Step 6 v2
    spec, B4) rests on that identity. Keep the operation order exactly as
    written: (v * cos) * dt, then + x.
    """
    x = pose[0] + v * np.cos(pose[2]) * dt
    y = pose[1] + v * np.sin(pose[2]) * dt
    theta = wrap_angle(pose[2] + omega * dt)
    return x, y, theta
```

Replace the three assignment lines at the top of `_advance_agent` with:

```python
        x, y, theta = euler_step(pose, v, omega, c.dt)
```

(leave the wall clip and `pose[:] = (x, y, theta)` untouched).

- [ ] **Step 4: Run the env tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_env.py -q`
Expected: all pass (the pre-existing hand-computed trajectory tests prove the refactor is behaviour-preserving).

- [ ] **Step 5: Commit**

```bash
git add pe2/env.py tests/pe2/test_env.py
git commit -m "feat(pe2): share euler_step between env and shield" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Hard body rows and soft VCP rows

**Files:**
- Modify: `pe2/shield.py` (`Constraint` dataclass; `_obstacle_constraint`; `_boundary_constraints`; `CBFShield.constraints`; the module docstring paragraph starting "3-tier fallback")
- Test: `tests/pe2/test_shield.py` (class `TestVCPMath` gains tests; nothing removed yet)

**Interfaces:**
- Produces: `Constraint(h, a_v, a_omega, kind, clearance, hard: bool, scale: float)`. Row ORDER from `CBFShield.constraints(state, obstacles)` is fixed and Task 4 depends on it: `[wall+x hard, wall−x hard, wall+y hard, wall−y hard, wall+x soft, wall−x soft, wall+y soft, wall−y soft]` then per obstacle `[hard, soft]`.
- Produces: `_body_wall_constraints(state, x_lim, y_lim) -> list[Constraint]`, `_body_obstacle_constraint(state, obs_x, obs_y, eff_radius) -> Constraint`.

- [ ] **Step 1: Write the failing tests**

Add to `TestVCPMath` in `tests/pe2/test_shield.py`:

```python
    def test_body_obstacle_row_is_the_env_contact_predicate(self):
        from pe2.shield import _body_obstacle_constraint

        # Body at origin heading +x; obstacle at (1.0, 0), r_eff 0.5:
        # h = hypot(1, 0) - 0.5 = 0.5 m, a_v = cos(angle to outward normal)
        # = -1 (moving straight at it), a_omega = 0 (body row ignores heading).
        c = _body_obstacle_constraint(np.array([0.0, 0.0, 0.0]), 1.0, 0.0, 0.5)
        assert c.h == pytest.approx(0.5)
        assert c.clearance == c.h
        assert c.a_v == pytest.approx(-1.0)
        assert c.a_omega == 0.0
        assert c.kind == "obstacle" and c.hard is True and c.scale == 1.0

    def test_body_wall_rows_ignore_heading(self):
        from pe2.shield import _body_wall_constraints

        rows = _body_wall_constraints(np.array([4.0, -1.0, 1.0]), 4.85, 4.85)
        assert [r.h for r in rows] == pytest.approx([0.85, 8.85, 5.85, 3.85])
        assert all(r.a_omega == 0.0 and r.hard and r.scale == 1.0 for r in rows)
        assert rows[0].a_v == pytest.approx(-np.cos(1.0))

    def test_vcp_rows_are_soft_with_obstacle_scale(self):
        c = _obstacle_constraint(np.array([0.0, 0.0, 0.0]), 1.1, 0.0, 0.5, 0.1)
        assert c.hard is False and c.scale == pytest.approx(1.0)  # 2 * r_eff
        w = _boundary_constraints(np.array([0.0, 0.0, 0.0]), 4.85, 4.85, 0.1)
        assert all((not r.hard) and r.scale == 1.0 for r in w)

    def test_constraints_row_order_and_count(self):
        shield = CBFShield(CFG)
        obstacles = [{"x": 2.0, "y": 0.0, "radius": 0.4}, {"x": -2.0, "y": 1.0, "radius": 0.3}]
        cons = shield.constraints(np.array([0.0, 0.0, 0.0]), obstacles)
        assert len(cons) == 12
        assert [c.hard for c in cons] == [True] * 4 + [False] * 4 + [True, False, True, False]
        assert [c.kind for c in cons] == ["boundary"] * 8 + ["obstacle"] * 4
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py::TestVCPMath -q`
Expected: FAIL (`ImportError` for the new helpers, `TypeError` on `hard`).

- [ ] **Step 3: Implement**

In `pe2/shield.py`:

```python
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
    clearance: float
    hard: bool
    scale: float
```

Keep `_obstacle_constraint` and `_boundary_constraints` as they are but construct with `hard=False` and `scale=2.0 * eff_radius` (obstacle) / `scale=1.0` (boundary). Add:

```python
def _body_wall_constraints(state: np.ndarray, x_lim: float, y_lim: float) -> list[Constraint]:
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
```

`CBFShield.constraints` becomes:

```python
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
        return cons
```

Update the module docstring: replace the "3-tier fallback …" paragraph with a two-paragraph summary of B1–B3 from the spec (hard body rows = env predicates; exact one-step check; tiers exact/relaxed/backup selecting from one candidate set; nothing dropped). Update the `h > 0 is a CONSERVATIVE early warning` paragraph to say that is true of the *soft* rows only; hard rows are exact.

- [ ] **Step 4: Run the whole shield test file**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py -q`
Expected: the four new tests pass. Existing tests that construct `Constraint(...)` positionally with 5 args or assert `n_constraints == 6` will now FAIL — that is expected at this point; list them in the report. Do NOT fix them here; Task 5 replaces the tier tests and fixes the counts. `TestVCPMath`, `TestShieldMetrics`, `TestSummarizeShield` must pass.

- [ ] **Step 5: Commit**

```bash
git add pe2/shield.py tests/pe2/test_shield.py
git commit -m "feat(shield): hard body rows beside soft VCP rows" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Candidate enumeration `_qp_candidates`

**Files:**
- Modify: `pe2/shield.py` (replace `_solve_qp_2d` at lines 125–198)
- Test: `tests/pe2/test_shield.py` (class `TestExactQP` rewritten as `TestCandidates`)

**Interfaces:**
- Produces: `_qp_candidates(v_nom, w_nom, rows, v_max, omega_max, w_v, w_omega) -> np.ndarray` of shape `(K, 2)`, float64, every row inside the box `[0, v_max] × [−omega_max, omega_max]`, containing (in this order) the clipped nominal, `[0.0, 0.0]`, then the D85 enumeration. Deterministic order.
- `FEAS_TOL` and `_solve_qp_2d` no longer exist.

- [ ] **Step 1: Replace `TestExactQP`**

Delete the class `TestExactQP` and add:

```python
def _linear_optimum(v_nom, w_nom, rows, v_max, omega_max, w_v, w_omega):
    """Test helper: exact optimum of the LINEAR program over the candidates.

    The shield itself judges candidates by the exact one-step barrier
    (Task 4); this helper only checks that the enumeration still contains
    the vertex the convex 2-D QP would pick (D85).
    """
    from pe2.shield import _qp_candidates

    cands = _qp_candidates(v_nom, w_nom, rows, v_max, omega_max, w_v, w_omega)
    ok = [
        u for u in cands
        if all(c0 * u[0] + c1 * u[1] >= c2 - 1e-12 for c0, c1, c2 in rows)
    ]
    if not ok:
        return None
    return min(ok, key=lambda u: w_v * (u[0] - v_nom) ** 2 + w_omega * (u[1] - w_nom) ** 2)


class TestCandidates:
    def test_nominal_and_zero_are_always_present(self):
        from pe2.shield import _qp_candidates

        cands = _qp_candidates(0.5, 0.5, [], 1.0, 2.84, 150.0, 1.0)
        assert np.array_equal(cands[0], [0.5, 0.5])
        assert np.array_equal(cands[1], [0.0, 0.0])

    def test_every_candidate_is_inside_the_box(self):
        from pe2.shield import _qp_candidates

        rows = [(-0.06, -0.003, -0.04), (1.0, 1.0, 2.0), (1.0, -1.0, 2.0)]
        cands = _qp_candidates(3.0, -9.0, rows, 1.0, 2.84, 150.0, 1.0)
        assert cands.dtype == np.float64 and cands.shape[1] == 2
        assert (cands[:, 0] >= 0.0).all() and (cands[:, 0] <= 1.0).all()
        assert (np.abs(cands[:, 1]) <= 2.84).all()

    def test_single_constraint_hand_computed(self):
        # dt*a_v = -0.05, dt*a_omega = 0, c2 = -0.01 -> v <= 0.2.
        u = _linear_optimum(1.0, 0.0, [(-0.05, 0.0, -0.01)], 1.0, 2.84, 150.0, 1.0)
        assert u[0] == pytest.approx(0.2)
        assert u[1] == pytest.approx(0.0, abs=1e-9)

    def test_infeasible_rows_leave_no_linear_candidate(self):
        assert _linear_optimum(
            0.0, 0.0, [(1.0, 0.0, 10.0), (-1.0, 0.0, 10.0)], 1.0, 2.84, 150.0, 1.0
        ) is None

    def test_vertex_optimum_two_constraints(self):
        u = _linear_optimum(
            0.0, 0.0, [(1.0, 1.0, 2.0), (1.0, -1.0, 2.0)], 5.0, 5.0, 1.0, 1.0
        )
        assert u[0] == pytest.approx(2.0)
        assert u[1] == pytest.approx(0.0, abs=1e-9)

    def test_anisotropic_weight_prefers_steering(self):
        u = _linear_optimum(1.0, 0.0, [(-0.06, -0.003, -0.04)], 1.0, 2.84, 150.0, 1.0)
        assert -0.06 * u[0] - 0.003 * u[1] >= -0.04 - 1e-9
        assert abs(u[1]) > 1.0 and u[0] > 0.7

    def test_enumeration_is_deterministic(self):
        from pe2.shield import _qp_candidates

        rows = [(-0.06, -0.003, -0.04), (1.0, 1.0, 2.0)]
        a = _qp_candidates(0.7, 0.1, rows, 1.0, 2.84, 150.0, 1.0)
        b = _qp_candidates(0.7, 0.1, rows, 1.0, 2.84, 150.0, 1.0)
        assert np.array_equal(a, b)
```

Also change the import block at the top of the file: remove `_solve_qp_2d` from the `from pe2.shield import (...)` list.

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py::TestCandidates -q`
Expected: FAIL with `ImportError: cannot import name '_qp_candidates'`.

- [ ] **Step 3: Implement**

Replace `_solve_qp_2d` (and delete the `FEAS_TOL` constant and its comment) with:

```python
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
```

`CBFShield._solve` still references `_solve_qp_2d`; leave `filter` broken until Task 5 — but the module must import, so temporarily make `_solve` raise `NotImplementedError("replaced in Task 5")`. (Task 5 deletes `_solve`.)

- [ ] **Step 4: Run the candidate tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py::TestCandidates tests/pe2/test_shield.py::TestVCPMath -q`
Expected: PASS. (Other classes in the file are expected red until Task 5.)

- [ ] **Step 5: Commit**

```bash
git add pe2/shield.py tests/pe2/test_shield.py
git commit -m "refactor(shield): candidate enumeration returns the D85 set incl. u=0" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Exact one-step barriers on the true successor

**Files:**
- Modify: `pe2/shield.py` (`CBFShield`: add `successor_barriers`, `progress`; import `euler_step`)
- Test: `tests/pe2/test_shield.py` (new class `TestSuccessorBarriers`)

**Interfaces:**
- Consumes: `pe2.env.euler_step` (Task 1); row order from Task 2.
- Produces: `CBFShield.successor_barriers(state, obstacles, U: np.ndarray) -> np.ndarray` of shape `(K, R)`: row values `h_i` evaluated at `euler_step(state, U[k])` for every candidate `k`, column order identical to `constraints()`.
- Produces: `CBFShield.progress(state, obstacles, action) -> np.ndarray` of shape `(R,)`: `(h_i(x⁺) − (1 − γ)·h_i(x)) / scale_i` for one action. `progress ≥ 0` for every row ⟺ the exact tier accepts the action.
- Produces: `CBFShield._h_columns(X, Y, TH, obstacles) -> np.ndarray (K, R)` — the vectorised barrier evaluation used by both.

- [ ] **Step 1: Write the failing tests**

```python
class TestSuccessorBarriers:
    def _random_setup(self, rng, margin=0.0):
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        shield = CBFShield(cfg, margin=margin)
        state = np.array(env.pursuer_pose, dtype=np.float64)
        return cfg, env, shield, state

    def test_columns_match_constraints_at_the_current_pose(self):
        rng = np.random.default_rng(7)
        cfg, env, shield, state = self._random_setup(rng)
        cons = shield.constraints(state, env.obstacles)
        U = np.zeros((1, 2))
        H = shield.successor_barriers(state, env.obstacles, U)
        assert H.shape == (1, len(cons))
        # Hard rows are bit-identical under u = 0 (spec B4 ii).
        for j, c in enumerate(cons):
            if c.hard:
                assert H[0, j] == c.h
            else:
                assert H[0, j] == pytest.approx(c.h, abs=1e-12)

    def test_hard_rows_equal_env_predicates_bitwise(self):
        """Predicted hard rows at the successor == what the env computes
        after it actually steps. 10k random (pose, action) pairs."""
        rng = np.random.default_rng(99)
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        shield = CBFShield(cfg)
        for _ in range(10_000):
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            state = np.array(env.pursuer_pose, dtype=np.float64)
            u = np.array([[rng.uniform(0, cfg.v_max), rng.uniform(-cfg.omega_max, cfg.omega_max)]])
            H = shield.successor_barriers(state, env.obstacles, u)
            cons = shield.constraints(state, env.obstacles)
            hard_idx = [j for j, c in enumerate(cons) if c.hard]
            # Step the env with the evader frozen; take the pursuer's new pose.
            _, _, _, _, info = env.step([u[0, 0], u[0, 1], 0.0, 0.0])
            if info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]:
                continue  # the env clipped/projected; the sign test covers that
            after = np.array(env.pursuer_pose, dtype=np.float64)
            hard_after = [c for c in shield.constraints(after, env.obstacles) if c.hard]
            for j, c_env in zip(hard_idx, hard_after, strict=True):
                assert H[0, j] == c_env.h, (j, H[0, j], c_env.h)

    def test_hard_row_sign_is_the_contact_predicate(self):
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=3)
        shield = CBFShield(cfg)
        lim = cfg.arena_size / 2.0 - cfg.agent_radius
        # Body exactly on the +x wall, heading into it, full speed: the
        # successor has x > x_max, so the hard +x row must be < 0.
        state = np.array([lim, 0.0, 0.0])
        H = shield.successor_barriers(state, [], np.array([[1.0, 0.0]]))
        assert H[0, 0] < 0.0
        # Same pose, u = 0: row exactly 0 (x_lim - x with x == x_lim).
        H0 = shield.successor_barriers(state, [], np.array([[0.0, 0.0]]))
        assert H0[0, 0] == 0.0

    def test_progress_sign_matches_the_dcbf_condition(self):
        rng = np.random.default_rng(5)
        cfg, env, shield, state = self._random_setup(rng)
        u = np.array([0.6, -0.4])
        p = shield.progress(state, env.obstacles, u)
        cons = shield.constraints(state, env.obstacles)
        H = shield.successor_barriers(state, env.obstacles, u[None, :])[0]
        for j, c in enumerate(cons):
            expected = (H[j] - (1.0 - shield.gamma) * c.h) / c.scale
            assert p[j] == pytest.approx(expected)
            assert (p[j] >= 0) == (H[j] >= (1.0 - shield.gamma) * c.h)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py::TestSuccessorBarriers -q`
Expected: FAIL with `AttributeError: 'CBFShield' object has no attribute 'successor_barriers'`.

- [ ] **Step 3: Implement**

In `pe2/shield.py` add `from pe2.env import euler_step, wrap_angle` and inside `CBFShield`:

```python
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
            self._x_lim - X, X + self._x_lim, self._y_lim - Y, Y + self._y_lim,
            self._x_lim - QX, QX + self._x_lim, self._y_lim - QY, QY + self._y_lim,
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

    def progress(
        self, state: np.ndarray, obstacles: list[dict], action
    ) -> np.ndarray:
        """Normalised one-step DCBF progress of every row under `action`:
        (h_i(x⁺) − (1−γ) h_i(x)) / scale_i. Every entry ≥ 0 ⟺ the exact
        tier accepts the action. Public so a gate can re-derive C1 from the
        definition rather than from another copy of the filter."""
        cons = self.constraints(state, obstacles)
        H1 = self.successor_barriers(state, obstacles, np.asarray(action)[None, :])[0]
        H0 = np.array([c.h for c in cons])
        scale = np.array([c.scale for c in cons])
        return (H1 - (1.0 - self.gamma) * H0) / scale
```

Also make `constraints()` compute `h` for the current pose THROUGH `_h_columns` so the two paths cannot drift: after building the list, do

```python
        H = self._h_columns(
            np.array([float(state[0])]), np.array([float(state[1])]),
            np.array([float(state[2])]), obstacles,
        )[0]
        for c, h in zip(cons, H, strict=True):
            c.h = float(h)
            c.clearance = float(h) if c.hard or c.kind == "boundary" else c.clearance
```

(the obstacle VCP row's `clearance` stays `sqrt(h + r_eff²) − r_eff`, computed in `_obstacle_constraint`).

If `test_hard_rows_equal_env_predicates_bitwise` fails by last-bit differences, replace the two hard-row lines in `_h_columns` with a scalar loop (`np.array([np.hypot(float(xi) - obs["x"], float(yi) - obs["y"]) for xi, yi in zip(X, Y)]) - r_eff`) and re-run; record which path was needed in the report.

- [ ] **Step 4: Run**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py::TestSuccessorBarriers tests/pe2/test_shield.py::TestVCPMath tests/pe2/test_shield.py::TestCandidates -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pe2/shield.py tests/pe2/test_shield.py
git commit -m "feat(shield): exact one-step barriers on the shared Euler successor" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: The three tiers in `filter`; delete the heuristic backup

**Files:**
- Modify: `pe2/shield.py` (`filter`, `_info`, delete `backup`, `_solve`, `_dcbf_rows` stays; delete `BACKUP_V_FRAC`, `HEAD_ON_TIE_TOL` and the comment block at lines 54–59)
- Modify: `pe2/metrics.py` (docstring lines 31–34; comment above the `if method == "backup"` block in `record`; the `n_backup` field comment if any)
- Test: `tests/pe2/test_shield.py` (`TestCBFShieldFilter`, `TestRobustMargin`, `TestShieldMetrics` where they encode old tier semantics)

**Interfaces:**
- Consumes: `_qp_candidates` (Task 3), `successor_barriers` (Task 4), `constraints` (Task 2).
- Produces: `CBFShield.filter(action, state, obstacles, v_max=None) -> (u, info)` with the SAME info keys as before (`method`, `feasible`, `intervention`, `min_h`, `min_clearance`, `min_h_obstacle`, `min_h_boundary`, `h_obstacle`, `h_boundary`, `n_constraints`, `v_nom`, `v_exec`, `stalled`, `immobile`). `feasible` is `method == "exact"`.
- `NoOpShield` unchanged in behaviour (`n_constraints` is now 12 with two obstacles).

- [ ] **Step 1: Replace the old tier tests**

Delete these tests in `tests/pe2/test_shield.py`: `test_relaxed_tier_when_obstacle_infeasible`, `test_backup_tier_when_wall_infeasible`, `test_backup_turns_away_from_nearest_danger`, `test_backup_head_on_north_wall_tie_breaks_ccw`, `test_backup_turns_the_short_way_when_not_a_tie`, `test_backup_speed_sits_above_the_stall_band_by_construction`, `test_backup_honours_the_effective_speed_cap`, `test_relaxed_tier_through_filter_reports_infeasible`, `test_relaxed_tier_info_is_classified_by_the_accumulator`, `TestRobustMargin::test_margin_moves_the_backup_turn_away_boundary`. Remove `HEAD_ON_TIE_TOL`/`BACKUP_V_FRAC` imports if present. Any remaining test asserting `n_constraints == 6` becomes `== 4 + 4 + 2 * n_obstacles` for its fixture. Add to `TestCBFShieldFilter`:

```python
    def test_safe_nominal_passes_unchanged_on_the_exact_tier(self):
        shield = CBFShield(CFG)
        u, info = shield.filter((0.8, 0.3), np.array([0.0, 0.0, 0.0]), [])
        assert info["method"] == "exact" and info["feasible"]
        assert np.allclose(u, [0.8, 0.3]) and not info["intervention"]

    def test_head_on_at_the_wall_stops_and_never_contacts(self):
        shield = CBFShield(CFG)
        state = np.array([X_LIM, 0.0, 0.0])  # body on the +x wall, heading into it
        u, info = shield.filter((1.0, 0.0), state, [])
        assert u[0] < 1e-12  # sub-ulp speeds cannot move x past x_lim
        assert info["method"] in ("exact", "relaxed")  # the VCP row is already < 0
        assert shield.progress(state, [], u)[0] >= 0.0  # hard +x row kept

    def test_tangent_slide_into_obstacle_is_braked_not_penetrated(self):
        """RC2 from the spec: at the surface with the heading 96° from the
        outward normal the VCP row alone would allow ~3 mm penetration."""
        shield = CBFShield(CFG)
        r_eff = 0.3 + CFG.agent_radius  # 0.45
        obstacles = [{"x": 0.0, "y": 0.0, "radius": 0.3}]
        state = np.array([r_eff, 0.0, np.pi / 2 + np.deg2rad(6.0)])  # slightly inward
        u, info = shield.filter((1.0, 0.0), state, obstacles)
        env = PursuitEvasionEnv(CFG)
        env.reset(seed=0)
        env.obstacles = obstacles
        env.pursuer_pose[:] = state
        env.evader_pose[:] = (4.0, 4.0, 0.0)
        _, _, _, _, step_info = env.step([u[0], u[1], 0.0, 0.0])
        assert not step_info["pursuer_obstacle_contact"]
        assert info["intervention"]

    def test_reset_with_vcp_inside_uses_relaxed_tier_and_keeps_hard_rows(self):
        """RC1: body just outside the surface, VCP already inside (h_vcp < 0)."""
        shield = CBFShield(CFG)
        obstacles = [{"x": 0.0, "y": 0.0, "radius": 0.3}]
        state = np.array([0.46, 0.0, np.pi])  # heading straight at the centre
        cons = shield.constraints(state, obstacles)
        assert any(c.h < 0 for c in cons if not c.hard)
        assert all(c.h >= 0 for c in cons if c.hard)
        u, info = shield.filter((1.0, 0.0), state, obstacles)
        assert info["method"] == "relaxed" and not info["feasible"]
        p = shield.progress(state, obstacles, u)
        assert all(p[j] >= 0 for j, c in enumerate(cons) if c.hard)

    def test_backup_tier_maximises_worst_hard_progress_then_turns(self):
        """Body inside the INFLATED set (robust arm), heading at the centre:
        no candidate keeps the hard row (any v > 0 goes deeper, v = 0 holds
        h < (1-γ)h), so the tier is backup; the best hard progress is v = 0,
        and the soft-row tie-break turns the VCP out at full rate so the
        next step can drive away."""
        shield = CBFShield(CFG, margin=0.05)
        obstacles = [{"x": 0.0, "y": 0.0, "radius": 0.3}]
        state = np.array([0.47, 0.0, np.pi])  # inside 0.50 but outside 0.45
        cons = shield.constraints(state, obstacles)
        assert any(c.h < 0 for c in cons if c.hard)
        u, info = shield.filter((1.0, 0.0), state, obstacles)
        assert info["method"] == "backup" and not info["feasible"]
        assert u[0] == 0.0 and abs(u[1]) == CFG.omega_max
        p = shield.progress(state, obstacles, u)
        hard = [j for j, c in enumerate(cons) if c.hard]
        worst = min(p[hard])
        # No other candidate does better on the worst hard row.
        from pe2.shield import _qp_candidates

        rows = shield._dcbf_rows(cons)
        cands = _qp_candidates(1.0, 0.0, rows, CFG.v_max, CFG.omega_max, shield.w_v, shield.w_omega)
        for cand in cands:
            assert min(shield.progress(state, obstacles, cand)[hard]) <= worst + 1e-12

    def test_backup_tier_drives_out_when_heading_away(self):
        shield = CBFShield(CFG, margin=0.05)
        obstacles = [{"x": 0.0, "y": 0.0, "radius": 0.3}]
        state = np.array([0.47, 0.0, 0.0])  # heading away from the centre
        u, info = shield.filter((0.0, 0.0), state, obstacles)
        # Full speed away satisfies the hard row (0.52 - 0.50 >= 0.8 * -0.03),
        # so this is NOT the backup tier: the exact/relaxed tiers handle it.
        assert info["method"] in ("exact", "relaxed")
        assert u[0] > 0.0 and info["intervention"]

    def test_filter_is_deterministic(self):
        shield = CBFShield(CFG)
        obstacles = [{"x": 1.0, "y": 0.2, "radius": 0.4}]
        state = np.array([0.3, 0.0, 0.1])
        a, ia = shield.filter((1.0, 1.0), state, obstacles)
        b, ib = shield.filter((1.0, 1.0), state, obstacles)
        assert np.array_equal(a, b) and ia == ib

    def test_the_heuristic_backup_is_gone(self):
        import pe2.shield as m

        assert not hasattr(m, "BACKUP_V_FRAC") and not hasattr(m, "HEAD_ON_TIE_TOL")
        assert not hasattr(m, "FEAS_TOL") and not hasattr(CBFShield, "backup")
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py::TestCBFShieldFilter -q`
Expected: FAIL (`NotImplementedError` from `_solve`, attribute assertions).

- [ ] **Step 3: Implement `filter`**

Replace `filter`, delete `backup`, `_solve`, the two constants and their comment block. New `filter`:

```python
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
            v_nom, w_nom, self._dcbf_rows(cons), v_max, self.cfg.omega_max,
            self.w_v, self.w_omega,
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
```

Edge case: with no obstacles `~hard` still has the four soft wall rows, so `prog[:, ~hard]` is never empty. Keep `_dcbf_rows` as is (it now emits 12 rows; hard rows have `a_omega = 0`).

In `_info`, keep `h_obstacle` / `h_boundary` / `min_h_obstacle` / `min_h_boundary` as the SOFT (VCP) rows only — `[c.h for c in cons if c.kind == "obstacle" and not c.hard]` etc. — so their units stay what the docstrings say (obstacle m², boundary m). `min_h` stays the minimum over ALL rows (mixed-unit, as documented) and `min_clearance` the minimum clearance over ALL rows, which now includes the body clearance and is therefore the true reportable margin. Say so in the `_info` docstring. `n_constraints = len(cons)`.

In `pe2/metrics.py` change the docstring bullet for `pinned` to: "`pinned`: DEADLOCK_STEPS consecutive BACKUP-tier steps. The backup tier means no candidate action could keep every hard (body) row, i.e. the state is outside the invariant set — at δ = 0 this cannot happen after a valid reset, on the robust arm it is the signature of an episode that started inside the inflated set and cannot get out." Replace the comment above `if method == "backup":` in `record` with "Tier-3 pinning: consecutive steps on which the hard rows were unsatisfiable. Same threshold as deadlock." Do not touch any field name.

- [ ] **Step 4: Run the full shield test file and fix remaining count assertions**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py -q`
Expected: PASS. Tests that fail only on `n_constraints` counts or on `Constraint(...)` positional construction are updated to the new count / keyword form; any other failure is a defect in the implementation, not the test — diagnose it.

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff check pe2/shield.py pe2/metrics.py tests/pe2/test_shield.py && ./venv/bin/ruff format --check pe2/shield.py pe2/metrics.py tests/pe2/test_shield.py
git add pe2/shield.py pe2/metrics.py tests/pe2/test_shield.py
git commit -m "feat(shield): exact/relaxed/backup tiers on the exact one-step check; delete heuristic backup" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: The zero-contact property test and a timing probe

**Files:**
- Test: `tests/pe2/test_shield.py` (new class `TestZeroContactGuarantee`)
- Create: `scripts/pe2_step06_timing_probe.py`

**Interfaces:**
- Consumes: `run_shielded_duel`, `GreedyPursuer`, `GreedyEvader`, `ObstacleHuggingEvader`, `CBFShield`, `NoOpShield`.

- [ ] **Step 1: Write the property tests**

```python
class _WallSeeker:
    """Adversarial nominal: always full speed, always steering toward the
    nearest wall or obstacle. The shield alone must keep the body off it."""

    def reset(self):
        pass

    def act(self, obs, env):
        x, y, th = env.pursuer_pose
        lim = env.cfg.arena_size / 2.0
        targets = [(lim, y), (-lim, y), (x, lim), (x, -lim)] + [
            (o["x"], o["y"]) for o in env.obstacles
        ]
        tx, ty = min(targets, key=lambda t: np.hypot(t[0] - x, t[1] - y))
        err = wrap_angle(np.arctan2(ty - y, tx - x) - th)
        return 1.0, float(np.clip(3.0 * err, -env.cfg.omega_max, env.cfg.omega_max))


class TestZeroContactGuarantee:
    """Spec B4: at δ = 0 the shielded role never produces an env contact step,
    for ANY nominal and ANY opponent. Not the gate (that is N = 1000 on the
    pre-registered streams); a local witness on unrelated seeds."""

    @pytest.mark.parametrize("seed", range(900000, 900040))
    def test_wall_seeker_never_contacts(self, seed):
        env = PursuitEvasionEnv(CFG)
        log = run_shielded_duel(env, _WallSeeker(), GreedyEvader(), CBFShield(CFG), seed)
        assert log["violations"] == 0, log["shield"]["tier_histogram"]
        assert log["shield"]["tier_histogram"]["backup"] == 0

    @pytest.mark.parametrize("seed", range(900100, 900130))
    def test_greedy_vs_hugger_never_contacts(self, seed):
        from pe2.scripted import ObstacleHuggingEvader

        env = PursuitEvasionEnv(CFG)
        log = run_shielded_duel(
            env, GreedyPursuer(), ObstacleHuggingEvader(), CBFShield(CFG), seed
        )
        assert log["violations"] == 0

    def test_wall_seeker_does_contact_without_the_shield(self):
        """Negative control for the property: the adversarial nominal is real."""
        env = PursuitEvasionEnv(CFG)
        hits = sum(
            run_shielded_duel(env, _WallSeeker(), GreedyEvader(), NoOpShield(CFG), s)["violations"] > 0
            for s in range(900000, 900010)
        )
        assert hits >= 8
```

`wrap_angle` is imported from `pe2.env` at the top of the test file.

- [ ] **Step 2: Run them**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py::TestZeroContactGuarantee -q`
Expected: PASS. A failure here is a defect in Tasks 1–5 (the guarantee is a theorem given bit identity); report the offending seed, step, tier and hard-row values, do not weaken the test.

- [ ] **Step 3: Timing probe**

Create `scripts/pe2_step06_timing_probe.py`:

```python
"""How long will the v2 gate take? 50 episodes per arm on the pre-registered
streams' first seeds is NOT a gate run; it only extrapolates wall-clock.

Usage: ./venv/bin/python scripts/pe2_step06_timing_probe.py
"""

from __future__ import annotations

import time

from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.scripted import GreedyEvader, GreedyPursuer, ObstacleHuggingEvader, run_shielded_duel
from pe2.shield import CBFShield

N = 50


def main() -> None:
    cfg = EnvConfig()
    for label, evader_cls, seed0 in (("main", GreedyEvader, 800000), ("probe", ObstacleHuggingEvader, 810000)):
        env = PursuitEvasionEnv(cfg)
        shield = CBFShield(cfg)
        t0 = time.perf_counter()
        steps = sum(
            run_shielded_duel(env, GreedyPursuer(), evader_cls(), shield, seed0 + i)["steps"]
            for i in range(N)
        )
        dt = time.perf_counter() - t0
        print(f"{label}: {N} eps, {steps} steps, {dt:.1f} s, {1e6 * dt / steps:.0f} us/step, "
              f"-> ~{dt / N * 1000 / 60:.1f} min per 1000-episode arm")


if __name__ == "__main__":
    main()
```

Run: `./venv/bin/python scripts/pe2_step06_timing_probe.py`
Expected: prints two lines; record the µs/step and the per-arm estimate in the report. If a 1000-episode shielded arm extrapolates above 60 min, vectorise `_qp_candidates` (it is the only pure-Python loop left) — do not reduce N.

- [ ] **Step 4: Commit**

```bash
./venv/bin/ruff check tests/pe2/test_shield.py scripts/pe2_step06_timing_probe.py
git add tests/pe2/test_shield.py scripts/pe2_step06_timing_probe.py
git commit -m "test(shield): zero-contact property witness + timing probe" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Gate runner and visual gate for v2

**Files:**
- Modify: `scripts/pe2_step06_gate.py` (docstring header; constants lines 80–81; `episode_row`; `forensic_replay` constraint dump; `pre_registration` block in `main`)
- Modify: `scripts/pe2_step06_visual_gate.py` (`GATE_EPISODES` comment + placeholder; `intervention_clearance_bound` docstring; `nominal_violates_a_row`)
- Modify: `tests/pe2/test_step06_gate.py` (`test_seed_streams_and_episode_count`)
- Test: `tests/pe2/test_step06_gate.py` (new tests for the start flags)

**Interfaces:**
- Consumes: `CBFShield.progress` (Task 4), `Constraint.hard/scale` (Task 2), `run_shielded_duel` log shape.
- Produces: per-episode row fields `start_in_barrier_set: bool`, `start_hard_feasible: bool` computed in `run_sweep` from `shield.constraints(env.pursuer_pose, env.obstacles)` right after `env.reset(seed)` — i.e. before the first `filter` call. Since `run_shielded_duel` resets internally, add a helper in the gate script:

```python
def start_flags(cfg, shield, seed: int) -> dict:
    """Whether the RESET state already satisfies every row / every hard row.

    RC1 in the v2 spec: 1.45% of resets start with a VCP row < 0; the report
    must be able to separate those episodes without re-running."""
    env = PursuitEvasionEnv(cfg)
    env.reset(seed=seed)
    cons = shield.constraints(np.array(env.pursuer_pose), env.obstacles)
    return {
        "start_in_barrier_set": bool(all(c.h >= 0.0 for c in cons)),
        "start_hard_feasible": bool(all(c.h >= 0.0 for c in cons if c.hard)),
    }
```

and merge its result into each log in `run_sweep` (`log.update(start_flags(cfg, shield, seed0 + i))`), then include both keys in `episode_row` and add aggregate counts `episodes_starting_outside_barrier_set` / `episodes_starting_hard_infeasible` to `reported_not_gated`'s block (computed from the logs by the caller and passed in — keep `reported_not_gated(summary, is_control)` signature and add the two counts in `main` next to it).

- [ ] **Step 1: Tests first**

In `tests/pe2/test_step06_gate.py` change:

```python
    def test_seed_streams_and_episode_count(self):
        assert GATE.MAIN_SEED0 == 800000
        assert GATE.PROBE_SEED0 == 810000
        assert GATE.N_EPISODES == 1000
```

and add:

```python
class TestStartFlags:
    def test_flags_are_booleans_and_hard_implies_nothing_about_soft(self):
        from pe2.shield import CBFShield

        cfg = EnvConfig()
        flags = GATE.start_flags(cfg, CBFShield(cfg), 800000)
        assert set(flags) == {"start_in_barrier_set", "start_hard_feasible"}
        assert isinstance(flags["start_in_barrier_set"], bool)
        assert flags["start_hard_feasible"] is True  # δ = 0: the env guarantees it

    def test_a_known_outside_start_is_flagged(self):
        """Seed 760372 started with the VCP inside an obstacle in the v1 run
        (robust_main first offence at step 1); at δ = 0.05 its BODY is inside."""
        from pe2.shield import CBFShield

        cfg = EnvConfig()
        flags = GATE.start_flags(cfg, CBFShield(cfg, margin=0.05), 760372)
        assert flags["start_hard_feasible"] is False
```

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_step06_gate.py -q` → expected FAIL on the seed constants and `AttributeError: start_flags`.

- [ ] **Step 2: Gate runner edits**

- Constants: `MAIN_SEED0 = 800000  # v2 stream (v1's 760000+ consumed, spec Part C)` and `PROBE_SEED0 = 810000`.
- Docstring: after the criteria list add one paragraph: "v2 (2026-09-05): criteria unchanged, seed streams fresh, shield redesigned — see `docs/pathway/step6_shield_v2_prereg_2026-09-05.md`. The violation definition is unchanged and magnitude-blind; the contact-depth census is reported, never gated."
- `forensic_replay`: in the `"constraints"` dump add `"hard": c.hard, "scale": c.scale`.
- `run_sweep`: merge `start_flags`; `episode_row`: add the two keys.
- `main`: in the `pre_registration` block add `"v2_spec": "docs/pathway/step6_shield_v2_prereg_2026-09-05.md"`; per arm add `"episodes_starting_outside_barrier_set"` and `"episodes_starting_hard_infeasible"` next to the `reported_not_gated` block.
- Nothing else. Do not touch thresholds, N, δ, the G1–G5 verdict code, or the census.

- [ ] **Step 3: Visual gate edits**

- `nominal_violates_a_row(shield, cons, v_nom, w_nom)` → replace body and signature with:

```python
def nominal_violates_a_row(shield, state, obstacles, v_nom: float, w_nom: float) -> bool:
    """True iff the nominal action fails the exact one-step check of at least
    one row — re-derived from `CBFShield.progress`, the constraint DEFINITION,
    not from another copy of the filter (C1)."""
    return bool((shield.progress(state, obstacles, np.array([v_nom, w_nom])) < 0.0).any())
```

  and update its call site in `play` (it needs the state and obstacles the filter saw, which `play` already has in hand — pass `np.array(env.pursuer_pose)` captured BEFORE `env.step`).
- `intervention_clearance_bound`: keep the code; rewrite the docstring: a row can bind only if the nominal fails the exact one-step check; the body moves ≤ v_max·dt and the VCP ≤ dt(v_max + d·ω_max) per step (chord ≤ arc), so `h_i(x⁺) − h_i(x) ≥ −(that reach)` for metre rows and `≥ −2·dist·reach` for the m² row; binding needs `γ·h < reach`-scale, which is the same bound as before, and the body rows bind below `dt·v_max/γ = 0.25 m` which is inside it.
- `GATE_EPISODES`: replace the three seeds with `None` placeholders and a comment: "Filled in Task 8 from the v2 streams (800000+/810000+) after the gate run, by the same criteria as v1 (a capture with a ≥ MIN_SLIDE_STEPS contact-free slide and a visible intervention rate); the seeds and the reason are recorded in the JSON." Make `main` raise `SystemExit("GATE_EPISODES not yet chosen — see Task 8")` if any seed is `None`.

- [ ] **Step 4: Run tests + lint**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_step06_gate.py tests/pe2/test_shield.py -q` → PASS.
`./venv/bin/ruff check scripts/pe2_step06_gate.py scripts/pe2_step06_visual_gate.py tests/pe2/test_step06_gate.py`.

- [ ] **Step 5: Commit**

```bash
git add scripts/pe2_step06_gate.py scripts/pe2_step06_visual_gate.py tests/pe2/test_step06_gate.py
git commit -m "feat(step06): v2 gate runner — fresh seed streams, start-state flags, C1 via exact check" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Run the pre-registered gates, then the visual gate; record

**Files:**
- Move: `results/visual_gates/step_06/` → `results/visual_gates/step_06_v1_failed_2026-09-05/` (before running)
- Create (by the scripts): `results/visual_gates/step_06/step_06_gate.json`, `step_06_visual_gate.json`, `step_06_shield.gif`, `gate_v2.log`
- Modify: `scripts/pe2_step06_visual_gate.py` (`GATE_EPISODES` seeds)
- Modify: `docs/worklogs/2026-09-05_S122.md` (results section), `docs/workflow_tracker.md` (S122 row + Current Blockers), `docs/decisions.md` (D97 outcome line)

- [ ] **Step 1: Preserve v1 and run the gate**

```bash
mv results/visual_gates/step_06 results/visual_gates/step_06_v1_failed_2026-09-05
mkdir -p results/visual_gates/step_06
nohup ./venv/bin/python scripts/pe2_step06_gate.py > results/visual_gates/step_06/gate_v2.log 2>&1 &
echo $!
```

Poll `tail -3 results/visual_gates/step_06/gate_v2.log` every ≥ 5 minutes (use the timing-probe estimate; do not busy-wait). The script exits non-zero on any gate failure — that is a result, not an error.

- [ ] **Step 2: Read the verdicts**

```bash
./venv/bin/python - <<'PY'
import json
g = json.load(open("results/visual_gates/step_06/step_06_gate.json"))
for name, gate in g["gates"].items():
    print(name, "PASS" if gate["pass"] else "FAIL", {k: v for k, v in gate.items() if k not in ("pass", "criterion")})
for arm, blk in g["arms"].items():
    r = blk["reported_not_gated"]
    print(arm, "tiers", r["tier_histogram"], "interv", round(r["intervention_rate"], 4),
          "start_outside", blk.get("episodes_starting_outside_barrier_set"),
          "hard_infeasible_start", blk.get("episodes_starting_hard_infeasible"))
PY
```

If G1 or G3 fails: STOP after Step 4's recording. Do not touch the shield, the gate, or the seeds. The census + `start_*` flags go into the worklog verbatim.

- [ ] **Step 3: Choose GIF episodes and run the visual gate (only if G1 and G3 pass)**

Pick from `step_06_gate.json` `arms.shielded_probe.episodes` the two lowest seeds with `outcome == "capture"`, `steps >= 120`, `n_interventions / n_steps >= 0.10`; from `arms.shielded_main.episodes` the lowest seed with the same properties. Write them into `GATE_EPISODES` with a comment giving the numbers, then:

```bash
./venv/bin/python scripts/pe2_step06_visual_gate.py && ./venv/bin/python scripts/pe2_step06_visual_gate.py --md5-only
```

Expected: C1–C4 asserted, md5 printed twice identical. If C2 (slide ≥ 15 steps) is not found in the three chosen episodes, take the next qualifying probe seed and re-run; record every attempt in the JSON's `episode_selection` note (add that key in the script's output dict if absent). Commit the seed choice:

```bash
git add scripts/pe2_step06_visual_gate.py
git commit -m "chore(step06): v2 visual-gate episodes from the 800000/810000 streams" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 4: Record**

Append to `docs/worklogs/2026-09-05_S122.md` a section `### Step 6 v2 gate run` with: the five verdicts in a table (criterion, result, numbers), per-arm tier histogram / intervention / infeasibility / clearance floor / deadlock / pinned, the two start-state counts per arm, the contact-depth census if any contact occurred, the visual-gate md5, and the artifact paths. Update the S122 row in `docs/workflow_tracker.md` and the Current Blockers row ("Step 6 shield fails its kill criteria" → resolved or updated with the v2 outcome). Add the outcome line to D97 in `docs/decisions.md`. Commit:

```bash
git add docs/worklogs/2026-09-05_S122.md docs/workflow_tracker.md docs/decisions.md
git commit -m "docs(S122): Step 6 v2 gate outcome" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 5: Final full-suite run**

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q` → all pass; record the count in the report.

---

## Self-review

- **Spec coverage:** B1 → Task 2; B2 → Tasks 1, 3, 4; B3 → Task 5; B4 → Task 6 (witness) + Task 8 (gate); B5 → reported fields already in the runner; Part C seeds/flags/pointer/v1 preservation/visual gate → Tasks 7–8; stop rule → Task 8 Step 2.
- **Placeholders:** the only deliberate placeholder is `GATE_EPISODES = None` in Task 7, filled by rule in Task 8 (the seeds cannot exist before the run).
- **Type consistency:** `Constraint(h, a_v, a_omega, kind, clearance, hard, scale)` is used positionally in Task 2's helpers and by keyword nowhere else; `successor_barriers(state, obstacles, U)` takes `(K, 2)` in Tasks 4–5; `progress(state, obstacles, action)` takes a 1-D action in Tasks 4, 5, 7; `_qp_candidates` signature identical in Tasks 3, 5; `start_flags(cfg, shield, seed)` identical in Task 7's helper and tests.
