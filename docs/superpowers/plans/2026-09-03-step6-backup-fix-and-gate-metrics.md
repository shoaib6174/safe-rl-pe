# Step 6 (part 1+2): Backup-Tier Fix + Gate Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pe2 shield test suite green and extend the shield's per-step / per-episode / per-sweep metrics with the four fields paper 2 needs — infeasibility rate, per-episode minimum barrier value, fallback-tier histogram, and a deadlock diagnostic — so the Step 6 gates (G1–G4) can be run against a complete gate JSON.

**Architecture:** Three small, additive changes to existing modules. (1) `CBFShield.backup` gets an explicit counter-clockwise tie-break for exact head-on approaches (the failing test's root cause is `wrap_angle(π) = −π`). (2) `CBFShield.filter` info grows four cheap fields; `ShieldMetrics` becomes a per-episode accumulator that tracks feasibility, stalls and deadlock. (3) A `run_shielded_duel` runner mirrors `run_duel` and a `summarize_shield` aggregator mirrors `eval.summarize`, producing the sweep-level numbers the gate JSON will carry.

**Tech Stack:** Python 3.10+, numpy, pytest (run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` — ROS plugin contamination), ruff. No new dependencies (pe2 CI installs only numpy/gymnasium/matplotlib/imageio).

**Spec:** `docs/pathway/from_scratch_build_plan.md` §5 "Step 6" (v3 additions) and the pre-registered Step 6 gate criteria in `docs/worklogs/2026-08-24_S119.md` ("Step 6 — Safety shield"). Decisions: D31 (γ=0.2), D83 (gate rules), D85 (exact 2-D QP), D87 (hardware → robust arm required — NOT in this plan; it is item 3).

## Global Constraints

- Always use the project venv: `./venv/bin/python -m pytest tests/pe2 ...` with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` in the environment.
- `pe2/` and `tests/pe2/` are currently **untracked in git** (S119 never committed). The commit steps below assume the user has authorized committing pe2; if not, skip every "Commit" step and leave the working tree for the user.
- Violation semantics (S119, pre-registered): a violation is an env-level contact flag (`pursuer_wall_contact` / `pursuer_obstacle_contact`), never `h < 0` inside the shield.
- Infeasibility semantics (new, this plan): a step is **infeasible** when the exact all-constraints QP has no solution, i.e. `info["method"] != "exact"`. `"backup"` is the hard case (boundary-only relaxation also infeasible).
- Deadlock semantics (new, this plan; mechanism per Grover, Liu & Sycara — CBF controllers stall at force-equilibria): a step is **stalled** when the executed speed is ≤ 5 % of `v_max` while the nominal speed is ≥ 50 % of `v_max`; an episode is **deadlocked** when the longest consecutive stall run is ≥ 40 steps (2.0 s at dt = 0.05).
- Every new number in a test is hand-computed in a comment. No `assert x is not None` tests.
- Do not touch the S119 gate criteria G1–G4 or `test_shield.py` tests that already pass, except `test_backup_turns_away_from_nearest_danger`, which is fixed by changing the implementation, not the test.
- Ruff must stay clean: `./venv/bin/ruff check pe2 tests/pe2 && ./venv/bin/ruff format --check pe2 tests/pe2`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `pe2/shield.py` | VCP-DCBF filter, 3-tier fallback, backup controller | Task 1: tie-break in `backup`. Task 2: four extra keys in `filter` info (`v_nom`, `v_exec`, `feasible`, `stalled`) + two module constants |
| `pe2/metrics.py` | Per-episode shield accumulator + sweep aggregator | Task 3: extend `ShieldMetrics` (infeasibility, stalls, deadlock, tier histogram). Task 5: add `summarize_shield(logs)` |
| `pe2/scripted.py` | Scripted agents + duel runner | Task 4: add `run_shielded_duel` next to `run_duel` |
| `tests/pe2/test_shield.py` | Shield + metrics tests | Tasks 1–5: new tests appended to existing classes / new classes |

---

### Task 1: Fix the backup-tier tie-break (failing test)

**Files:**
- Modify: `pe2/shield.py` — `CBFShield.backup` (currently lines ~233–252) and module constants (~line 38)
- Test: `tests/pe2/test_shield.py` — class `TestCBFShieldFilter`

**Interfaces:**
- Consumes: `pe2.env.wrap_angle(theta) -> float` (normalizes to `[-π, π)`; maps exactly `±π` to `−π`).
- Produces: `CBFShield.backup(state, obstacles) -> np.ndarray([v, ω])` with the documented convention: exact head-on danger ⇒ turn counter-clockwise (`+ω_max`).

Root cause (verified): at state `(4.8, 0, 0)` the nearest danger is the east wall at angle 0; `away = π`; `wrap_angle(π − 0)` returns `−π`, so `ω = clip(3·(−π)) = −ω_max`. Both rotation directions are physically valid for a head-on approach; the test pins the CCW convention, so the implementation must break the tie deterministically.

- [ ] **Step 1: Run the failing test to confirm the failure mode**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py::TestCBFShieldFilter::test_backup_turns_away_from_nearest_danger -v --no-header`
Expected: FAIL with `assert np.float64(-2.84) > 0`

- [ ] **Step 2: Add two convention-pinning tests (they fail for the same reason)**

Append inside `class TestCBFShieldFilter` in `tests/pe2/test_shield.py`, directly after `test_backup_turns_away_from_nearest_danger`:

```python
    def test_backup_head_on_north_wall_tie_breaks_ccw(self):
        # Heading +y at y=4.8: nearest danger is the north wall (angle pi/2),
        # away = -pi/2, error = wrap(-pi/2 - pi/2) = wrap(-pi) = -pi -> an exact
        # tie. Convention: ties turn CCW (+omega).
        shield = CBFShield(CFG)
        u = shield.backup(np.array([0.0, 4.8, np.pi / 2]), [])
        assert u[0] == pytest.approx(0.1 * CFG.v_max)
        assert u[1] == pytest.approx(CFG.omega_max)

    def test_backup_turns_the_short_way_when_not_a_tie(self):
        # Wall east, heading 0.5 rad north of east: away = pi, error =
        # wrap(pi - 0.5) = +2.64 -> CCW. Heading -0.5: error = wrap(pi + 0.5)
        # = -2.64 -> CW. Both saturate at omega_max (3 * 2.64 > 2.84).
        shield = CBFShield(CFG)
        u_ccw = shield.backup(np.array([4.8, 0.0, 0.5]), [])
        u_cw = shield.backup(np.array([4.8, 0.0, -0.5]), [])
        assert u_ccw[1] == pytest.approx(CFG.omega_max)
        assert u_cw[1] == pytest.approx(-CFG.omega_max)
```

- [ ] **Step 3: Run the three backup tests to verify two fail, one passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py -k backup -v --no-header`
Expected: `test_backup_turns_away_from_nearest_danger` FAIL, `test_backup_head_on_north_wall_tie_breaks_ccw` FAIL (`-2.84 == approx(2.84)`), `test_backup_turns_the_short_way_when_not_a_tie` PASS, `test_backup_tier_when_wall_infeasible` PASS.

- [ ] **Step 4: Implement the tie-break**

In `pe2/shield.py`, add a constant after `INTERVENTION_ATOL`:

```python
HEAD_ON_TIE_TOL = 1e-9  # |error| within this of pi counts as an exact head-on tie
```

Replace the body of `CBFShield.backup` from `away = ...` to the end with:

```python
        away = min(dangers, key=lambda d: d[1])[0] + np.pi
        err = wrap_angle(away - theta)
        # wrap_angle maps an exact head-on error (±pi) to -pi, which would make
        # the turn direction depend on float rounding. Convention: ties turn CCW.
        if err <= -np.pi + HEAD_ON_TIE_TOL:
            err = np.pi
        omega = float(np.clip(3.0 * err, -self.cfg.omega_max, self.cfg.omega_max))
        return np.array([0.1 * self.cfg.v_max, omega], dtype=np.float64)
```

Update the method docstring to: `"""Tier 3: brake + turn away from the nearest danger (root pattern). Exact head-on ties turn CCW."""`

- [ ] **Step 5: Run the whole shield file to verify green**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py -v --no-header`
Expected: all PASS (24 tests: the original 22 + 2 new).

- [ ] **Step 6: Lint**

Run: `./venv/bin/ruff check pe2/shield.py tests/pe2/test_shield.py && ./venv/bin/ruff format --check pe2/shield.py tests/pe2/test_shield.py`
Expected: no output / "All checks passed".

- [ ] **Step 7: Commit**

```bash
git add pe2/shield.py tests/pe2/test_shield.py
git commit -m "fix(pe2/shield): CCW tie-break for exact head-on backup turns

wrap_angle maps an error of exactly pi to -pi, so a head-on approach
turned clockwise while the test pinned counter-clockwise. Break the tie
explicitly so the backup direction never depends on float rounding.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Expose feasibility and stall signals from `CBFShield.filter`

**Files:**
- Modify: `pe2/shield.py` — module constants and `CBFShield.filter` (info dict, ~lines 254–302)
- Test: `tests/pe2/test_shield.py` — class `TestCBFShieldFilter`

**Interfaces:**
- Consumes: existing `filter` info keys `method`, `intervention`, `min_h`, `h_obstacle`, `h_boundary`, `n_constraints`.
- Produces (new info keys, all consumed by Task 3):
  - `"feasible": bool` — `method == "exact"`.
  - `"v_nom": float` — clipped nominal speed actually fed to the QP.
  - `"v_exec": float` — executed speed `u[0]`.
  - `"stalled": bool` — `v_exec <= STALL_V_FRAC * v_max and v_nom >= MOVE_V_FRAC * v_max`.
  - Module constants `STALL_V_FRAC = 0.05`, `MOVE_V_FRAC = 0.5`.

- [ ] **Step 1: Write the failing tests**

Append inside `class TestCBFShieldFilter`:

```python
    def test_filter_info_reports_feasibility_and_speeds(self):
        shield = CBFShield(CFG)
        # Safe state: exact tier, no change -> feasible, v_exec == v_nom.
        u, info = shield.filter((0.5, 0.0), np.array([0.0, 0.0, 0.0]), [])
        assert info["feasible"] is True
        assert info["v_nom"] == pytest.approx(0.5)
        assert info["v_exec"] == pytest.approx(0.5)
        assert info["stalled"] is False
        # Outside the wall: backup tier -> not feasible.
        _, info = shield.filter((1.0, 0.0), np.array([5.0, 0.0, 0.0]), [])
        assert info["feasible"] is False
        assert info["method"] == "backup"

    def test_filter_flags_stall_when_shield_stops_a_moving_nominal(self):
        # At x=4.85 (= x_lim) heading +x: h = x_lim - qx = -0.1 for the +x wall,
        # but the boundary-only QP is still feasible (v = 0 satisfies
        # -0.05 v >= -0.2 * (-0.1) = 0.02? no: -0.05v >= 0.02 has no v >= 0
        # solution) -> backup tier at v = 0.1 v_max = 0.1 > 0.05 -> NOT stalled.
        # So stall must be exercised where the exact tier brakes to ~0:
        # x = 4.849, h = 0.001 -> v <= gamma*h/(dt*1) = 0.2*0.001/0.05 = 0.004
        # <= 0.05 * v_max = 0.05, nominal 1.0 >= 0.5 -> stalled.
        shield = CBFShield(CFG)
        u, info = shield.filter((1.0, 0.0), np.array([4.849, 0.0, 0.0]), [])
        assert info["method"] == "exact"
        assert u[0] == pytest.approx(0.004, abs=1e-6)
        assert info["stalled"] is True

    def test_filter_not_stalled_when_nominal_is_slow(self):
        # Same wall squeeze but the nominal itself is slow (0.2 < 0.5 v_max):
        # braking a slow nominal is not a stall.
        shield = CBFShield(CFG)
        _, info = shield.filter((0.2, 0.0), np.array([4.849, 0.0, 0.0]), [])
        assert info["stalled"] is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py -k "filter_info or stall" -v --no-header`
Expected: 3 FAIL with `KeyError: 'feasible'` / `KeyError: 'stalled'`.

- [ ] **Step 3: Implement**

In `pe2/shield.py` add after `HEAD_ON_TIE_TOL`:

```python
STALL_V_FRAC = 0.05  # executed speed <= this fraction of v_max counts as stopped
MOVE_V_FRAC = 0.5  # nominal speed >= this fraction of v_max counts as "wanted to move"
```

In `CBFShield.filter`, replace the `info = {...}` block with:

```python
        intervention = not np.allclose(u, [v_nom, w_nom], atol=INTERVENTION_ATOL)
        v_exec = float(u[0])
        stalled = v_exec <= STALL_V_FRAC * v_max and v_nom >= MOVE_V_FRAC * v_max
        info = {
            "method": method,
            "feasible": method == "exact",
            "intervention": bool(intervention),
            "min_h": float(min_h),
            "h_obstacle": [c.h for c in cons if c.kind == "obstacle"],
            "h_boundary": [c.h for c in cons if c.kind == "boundary"],
            "n_constraints": len(cons),
            "v_nom": float(v_nom),
            "v_exec": v_exec,
            "stalled": bool(stalled),
        }
        return u, info
```

Update the `filter` docstring's "info keys" sentence to list: `method ("exact" | "relaxed" | "backup"), feasible, intervention, min_h, h_obstacle, h_boundary, n_constraints, v_nom, v_exec, stalled`.

- [ ] **Step 4: Run the shield tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py -v --no-header`
Expected: all PASS (27 tests). If `test_filter_flags_stall_when_shield_stops_a_moving_nominal` fails on the `0.004` value, print `u` and `info["h_boundary"]` and recompute `γ·h/(dt·|a_v|)` with the printed `h`; the constraint arithmetic is `dt·a_v·v + γ·h ≥ 0` with `a_v = −1`, so `v ≤ γ·h/dt`. Do not loosen the tolerance beyond `1e-6` — fix the hand computation instead.

- [ ] **Step 5: Lint and commit**

Run: `./venv/bin/ruff check pe2/shield.py tests/pe2/test_shield.py && ./venv/bin/ruff format --check pe2/shield.py tests/pe2/test_shield.py`

```bash
git add pe2/shield.py tests/pe2/test_shield.py
git commit -m "feat(pe2/shield): report feasibility, speeds and stall flag in filter info

Adds feasible / v_nom / v_exec / stalled so the Step 6 gate can compute
infeasibility rate and a deadlock diagnostic (Grover, Liu & Sycara:
CBF controllers stall at force-equilibria).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Extend `ShieldMetrics` with infeasibility, stalls, deadlock and tier histogram

**Files:**
- Modify: `pe2/metrics.py` — `ShieldMetrics` (whole class)
- Test: `tests/pe2/test_shield.py` — class `TestShieldMetrics`

**Interfaces:**
- Consumes: `filter` info keys from Task 2 (`method`, `intervention`, `min_h`, `stalled`).
- Produces:
  - `ShieldMetrics.record(shield_info: dict, violation: bool) -> None` (unchanged signature).
  - `ShieldMetrics.summary() -> dict` with the existing keys **plus**: `infeasibility_rate`, `backup_rate`, `tier_histogram` (dict with keys `exact`, `relaxed`, `backup` always present, integer counts), `stall_step_rate`, `longest_stall_run` (int), `deadlock` (bool).
  - Module constants `DEADLOCK_STEPS = 40`, `TIERS = ("exact", "relaxed", "backup")`.
  - Backward compatibility: `method_counts` stays (only tiers seen), `min_h_floor` / `min_h_mean` unchanged.

- [ ] **Step 1: Write the failing tests**

Replace `class TestShieldMetrics` in `tests/pe2/test_shield.py` with:

```python
def _info(method="exact", intervention=False, min_h=1.0, stalled=False):
    return {
        "method": method,
        "intervention": intervention,
        "min_h": min_h,
        "stalled": stalled,
    }


class TestShieldMetrics:
    def test_rates_hand_computed(self):
        m = ShieldMetrics()
        m.record(_info(method="exact", intervention=True, min_h=0.5), violation=False)
        m.record(_info(method="exact", intervention=False, min_h=0.3), violation=True)
        m.record(_info(method="backup", intervention=True, min_h=0.1), violation=False)
        s = m.summary()
        assert s["n_steps"] == 3
        assert s["violation_rate"] == pytest.approx(1 / 3)
        assert s["intervention_rate"] == pytest.approx(2 / 3)
        assert s["method_counts"] == {"exact": 2, "backup": 1}
        assert s["min_h_floor"] == pytest.approx(0.1)
        assert s["min_h_mean"] == pytest.approx(0.3)

    def test_empty_summary(self):
        s = ShieldMetrics().summary()
        assert s["n_steps"] == 0
        assert s["violation_rate"] == 0.0
        assert s["min_h_floor"] is None
        assert s["infeasibility_rate"] == 0.0
        assert s["tier_histogram"] == {"exact": 0, "relaxed": 0, "backup": 0}
        assert s["longest_stall_run"] == 0
        assert s["deadlock"] is False

    def test_infeasibility_and_tier_histogram(self):
        # 4 steps: exact, relaxed, backup, exact -> infeasible = 2/4,
        # backup = 1/4, histogram exact 2 / relaxed 1 / backup 1.
        m = ShieldMetrics()
        for method in ("exact", "relaxed", "backup", "exact"):
            m.record(_info(method=method), violation=False)
        s = m.summary()
        assert s["infeasibility_rate"] == pytest.approx(0.5)
        assert s["backup_rate"] == pytest.approx(0.25)
        assert s["tier_histogram"] == {"exact": 2, "relaxed": 1, "backup": 1}

    def test_stall_runs_and_deadlock_threshold(self):
        # Pattern: 3 stalled, 1 free, 5 stalled -> longest run 5, stall rate 8/9,
        # deadlock False (5 < 40).
        m = ShieldMetrics()
        for stalled in [True, True, True, False, True, True, True, True, True]:
            m.record(_info(stalled=stalled), violation=False)
        s = m.summary()
        assert s["stall_step_rate"] == pytest.approx(8 / 9)
        assert s["longest_stall_run"] == 5
        assert s["deadlock"] is False

    def test_deadlock_when_stall_run_reaches_threshold(self):
        # Exactly DEADLOCK_STEPS consecutive stalled steps -> deadlock True.
        m = ShieldMetrics()
        for _ in range(DEADLOCK_STEPS):
            m.record(_info(stalled=True), violation=False)
        assert m.summary()["deadlock"] is True
        # One fewer, broken by a free step in the middle -> False.
        m2 = ShieldMetrics()
        for i in range(DEADLOCK_STEPS):
            m2.record(_info(stalled=(i != DEADLOCK_STEPS // 2)), violation=False)
        assert m2.summary()["deadlock"] is False
```

Update the import line at the top of the test file:

```python
from pe2.metrics import DEADLOCK_STEPS, ShieldMetrics
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py::TestShieldMetrics -v --no-header`
Expected: ImportError on `DEADLOCK_STEPS` (collection error). That is the expected red state.

- [ ] **Step 3: Implement**

Replace the whole of `pe2/metrics.py` with:

```python
"""Step 6: shield metrics — violation/intervention rates, CBF margins,
infeasibility, fallback-tier histogram and a deadlock diagnostic.

A violation is an env-level event (wall_contact or obstacle_contact in the env
info dict), NOT h < 0 in the shield — the two coincide when constraint radii
include agent_radius, but the env flag is the ground truth the gate counts.

Infeasibility: the exact all-constraints QP had no solution (method != "exact").
Deadlock: CBF controllers can settle at a force-equilibrium and stall short of
the goal (Grover, Liu & Sycara, WAFR 2020 / IJRR 2023). We detect the
behavioral signature: the shield holds executed speed near zero while the
nominal policy wants to move, for DEADLOCK_STEPS consecutive steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

TIERS = ("exact", "relaxed", "backup")
DEADLOCK_STEPS = 40  # 2.0 s at dt = 0.05


@dataclass
class ShieldMetrics:
    """Accumulates per-step shield stats for one episode (or a whole sweep)."""

    n_steps: int = 0
    n_violations: int = 0
    n_interventions: int = 0
    n_infeasible: int = 0
    n_backup: int = 0
    n_stalled: int = 0
    longest_stall_run: int = 0
    _current_stall_run: int = 0
    method_counts: dict[str, int] = field(default_factory=dict)
    min_h_floor: float = math.inf  # worst margin seen anywhere
    min_h_sum: float = 0.0  # for the mean margin

    def record(self, shield_info: dict, violation: bool) -> None:
        """One step: shield_info from CBFShield.filter, violation from env info."""
        self.n_steps += 1
        self.n_violations += int(violation)
        self.n_interventions += int(shield_info["intervention"])
        method = shield_info["method"]
        self.method_counts[method] = self.method_counts.get(method, 0) + 1
        self.n_infeasible += int(method != "exact")
        self.n_backup += int(method == "backup")
        self.min_h_floor = min(self.min_h_floor, shield_info["min_h"])
        self.min_h_sum += shield_info["min_h"]
        if shield_info["stalled"]:
            self.n_stalled += 1
            self._current_stall_run += 1
            self.longest_stall_run = max(self.longest_stall_run, self._current_stall_run)
        else:
            self._current_stall_run = 0

    def summary(self) -> dict:
        """Rates + margins. min_h_floor is None if no steps were recorded."""
        n = self.n_steps
        return {
            "n_steps": n,
            "violation_rate": self.n_violations / n if n else 0.0,
            "intervention_rate": self.n_interventions / n if n else 0.0,
            "infeasibility_rate": self.n_infeasible / n if n else 0.0,
            "backup_rate": self.n_backup / n if n else 0.0,
            "method_counts": dict(self.method_counts),
            "tier_histogram": {t: self.method_counts.get(t, 0) for t in TIERS},
            "min_h_floor": None if n == 0 else self.min_h_floor,
            "min_h_mean": self.min_h_sum / n if n else 0.0,
            "stall_step_rate": self.n_stalled / n if n else 0.0,
            "longest_stall_run": self.longest_stall_run,
            "deadlock": self.longest_stall_run >= DEADLOCK_STEPS,
        }
```

- [ ] **Step 4: Run the metrics tests, then the whole file**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py -v --no-header`
Expected: all PASS (30 tests).

- [ ] **Step 5: Lint and commit**

Run: `./venv/bin/ruff check pe2/metrics.py tests/pe2/test_shield.py && ./venv/bin/ruff format --check pe2/metrics.py tests/pe2/test_shield.py`

```bash
git add pe2/metrics.py tests/pe2/test_shield.py
git commit -m "feat(pe2/metrics): infeasibility rate, tier histogram, stall/deadlock diagnostic

Extends ShieldMetrics with the per-episode fields the Step 6 gate JSON
needs for paper 2 (R2 shield-exploitation claim).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `run_shielded_duel` — one shielded episode → per-episode log

**Files:**
- Modify: `pe2/scripted.py` — add `run_shielded_duel` directly after `run_duel` (~line 173); add import of `ShieldMetrics`
- Test: `tests/pe2/test_shield.py` — new class `TestRunShieldedDuel`

**Interfaces:**
- Consumes: `run_duel`'s log layout (`seed`, `pursuer`, `evader`, `outcome`, `steps`, `final_distance`, `los_breaks`); `CBFShield.filter(action, state, obstacles, v_max=None) -> (u, info)`; `ShieldMetrics` from Task 3; env info flags `pursuer_wall_contact`, `pursuer_obstacle_contact`, `evader_wall_contact`, `evader_obstacle_contact`; `env.pursuer_pose`, `env.evader_pose`, `env.obstacles`, `cfg.evader_v_max`.
- Produces:

```python
def run_shielded_duel(
    env: PursuitEvasionEnv,
    pursuer,
    evader,
    shield,            # pe2.shield.CBFShield
    seed: int,
    shielded_role: str = "pursuer",   # "pursuer" | "evader"
) -> dict
```
  returning the `run_duel` log **plus** `"shielded_role": str`, `"violations": int` (contact steps of the shielded role), and `"shield": dict` (= `ShieldMetrics.summary()` for the episode).

- [ ] **Step 1: Write the failing tests**

Append to `tests/pe2/test_shield.py`:

```python
from pe2.scripted import run_shielded_duel  # noqa: E402  (keep with other imports at top in practice)


class TestRunShieldedDuel:
    def test_obstacle_free_shielded_pursuer_log_shape(self):
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        log = run_shielded_duel(
            env, GreedyPursuer(), GreedyEvader(), CBFShield(cfg), seed=760000
        )
        # run_duel fields preserved.
        for key in ("seed", "pursuer", "evader", "outcome", "steps", "final_distance", "los_breaks"):
            assert key in log
        assert log["seed"] == 760000
        assert log["shielded_role"] == "pursuer"
        # One shield record per env step.
        assert log["shield"]["n_steps"] == log["steps"]
        # No obstacles and a boundary shield: no contacts for the shielded role.
        assert log["violations"] == 0
        assert log["shield"]["violation_rate"] == 0.0
        assert set(log["shield"]["tier_histogram"]) == {"exact", "relaxed", "backup"}

    def test_shielded_evader_uses_evader_speed_cap(self):
        # The evader's v_max is 1.2x; the shield must clip the nominal to the
        # evader cap, not the pursuer's, or a fast evader is silently slowed.
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        log = run_shielded_duel(
            env, GreedyPursuer(), GreedyEvader(), CBFShield(cfg), seed=760001,
            shielded_role="evader",
        )
        assert log["shielded_role"] == "evader"
        assert log["shield"]["n_steps"] == log["steps"]
        assert log["violations"] == 0

    def test_matches_unshielded_duel_when_shield_never_intervenes(self):
        # Deterministic: same seed, obstacle-free, and both agents start far
        # from walls for the first step -> identical first-step outcome fields
        # are not guaranteed for the whole episode (the shield brakes at walls),
        # so only pin determinism of the shielded run itself.
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        a = run_shielded_duel(env, GreedyPursuer(), GreedyEvader(), CBFShield(cfg), seed=760002)
        b = run_shielded_duel(env, GreedyPursuer(), GreedyEvader(), CBFShield(cfg), seed=760002)
        assert a == b

    def test_rejects_unknown_role(self):
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        with pytest.raises(ValueError):
            run_shielded_duel(env, GreedyPursuer(), GreedyEvader(), CBFShield(cfg), seed=1, shielded_role="both")
```

(Move the `run_shielded_duel` import up to the existing `from pe2.scripted import ...` line and drop the `noqa` — the snippet shows it inline only for readability.)

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py::TestRunShieldedDuel -v --no-header`
Expected: ImportError `cannot import name 'run_shielded_duel'`.

- [ ] **Step 3: Implement**

In `pe2/scripted.py`, add to the imports:

```python
from pe2.metrics import ShieldMetrics
```

Append after `run_duel`:

```python
def run_shielded_duel(
    env: PursuitEvasionEnv,
    pursuer,
    evader,
    shield,
    seed: int,
    shielded_role: str = "pursuer",
) -> dict:
    """Run one episode with `shield` filtering one role's actions.

    Returns the `run_duel` log plus `shielded_role`, `violations` (env contact
    steps of the shielded role — the pre-registered Step 6 semantics) and
    `shield` (ShieldMetrics.summary() for the episode). The shield sees the
    TRUE pose and the static map; the policies see their observations.
    """
    if shielded_role not in ("pursuer", "evader"):
        raise ValueError(f"shielded_role must be 'pursuer' or 'evader', got {shielded_role!r}")
    pursuer.reset()
    evader.reset()
    obs, _ = env.reset(seed=seed)
    los_prev = bool(obs["pursuer"][6] > 0.5)
    los_breaks = 0
    metrics = ShieldMetrics()
    violations = 0
    v_cap = env.cfg.v_max if shielded_role == "pursuer" else env.cfg.evader_v_max
    while True:
        v_p, w_p = pursuer.act(obs["pursuer"], env)
        v_e, w_e = evader.act(obs["evader"], env)
        if shielded_role == "pursuer":
            (v_p, w_p), sinfo = shield.filter((v_p, w_p), env.pursuer_pose, env.obstacles, v_max=v_cap)
        else:
            (v_e, w_e), sinfo = shield.filter((v_e, w_e), env.evader_pose, env.obstacles, v_max=v_cap)
        obs, _, terminated, truncated, info = env.step([v_p, w_p, v_e, w_e])
        contact = bool(
            info[f"{shielded_role}_wall_contact"] or info[f"{shielded_role}_obstacle_contact"]
        )
        violations += int(contact)
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
        "shielded_role": shielded_role,
        "violations": violations,
        "shield": metrics.summary(),
    }
```

Note: `shield.filter` returns a numpy array; unpacking `(v_p, w_p), sinfo = ...` works because a 2-element array iterates. Keep it — it avoids a temporary.

- [ ] **Step 4: Run the new tests, then the whole pe2 suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q --no-header`
Expected: all PASS (existing 166 + 2 + 3 + 3 + 4 = 178 tests; count may differ by ±1 if a class was split — the requirement is zero failures).

- [ ] **Step 5: Lint and commit**

Run: `./venv/bin/ruff check pe2 tests/pe2 && ./venv/bin/ruff format --check pe2 tests/pe2`

```bash
git add pe2/scripted.py tests/pe2/test_shield.py
git commit -m "feat(pe2): run_shielded_duel — per-episode shielded log with violations and shield metrics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `summarize_shield` — sweep-level aggregation for the gate JSON

**Files:**
- Modify: `pe2/metrics.py` — add `summarize_shield(logs)` after `ShieldMetrics`
- Test: `tests/pe2/test_shield.py` — new class `TestSummarizeShield`

**Interfaces:**
- Consumes: a list of `run_shielded_duel` logs (Task 4 layout).
- Produces:

```python
def summarize_shield(logs: list[dict]) -> dict
```
  with keys: `n_episodes`, `n_steps` (pooled), `violation_steps`, `episodes_with_violation`, `violation_episode_rate`, `intervention_rate` (pooled over steps), `infeasibility_rate` (pooled), `backup_rate` (pooled), `tier_histogram` (pooled counts, all three tiers), `min_h_per_episode` (list of floats, episode order), `min_h_floor`, `deadlock_episodes`, `deadlock_episode_rate`, `timeout_rate`, `capture_rate`. This is what the Step 6 gate script (item 4, not in this plan) will write per matchup, next to `eval.summarize`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pe2/test_shield.py` (and add `summarize_shield` to the `from pe2.metrics import ...` line):

```python
def _log(outcome, steps, violations, shield):
    return {
        "outcome": outcome,
        "steps": steps,
        "violations": violations,
        "shield": shield,
    }


class TestSummarizeShield:
    def test_pooled_rates_hand_computed(self):
        # Episode A: 100 steps, 10 interventions, 4 infeasible (3 relaxed + 1
        # backup), min_h 0.2, no stall, 0 violations, capture.
        # Episode B: 300 steps, 30 interventions, 0 infeasible, min_h -0.05,
        # deadlock, 2 violations, timeout.
        # Pooled: steps 400, intervention 40/400 = 0.1, infeasible 4/400 = 0.01,
        # backup 1/400, tiers exact 396 / relaxed 3 / backup 1,
        # violation steps 2, episodes with violation 1/2, deadlock 1/2,
        # timeout 1/2, capture 1/2, floor -0.05.
        a = _log("capture", 100, 0, {
            "n_steps": 100, "n_interventions_unused": 0,
            "intervention_rate": 0.10, "infeasibility_rate": 0.04, "backup_rate": 0.01,
            "tier_histogram": {"exact": 96, "relaxed": 3, "backup": 1},
            "min_h_floor": 0.2, "deadlock": False,
        })
        b = _log("timeout", 300, 2, {
            "n_steps": 300,
            "intervention_rate": 0.10, "infeasibility_rate": 0.0, "backup_rate": 0.0,
            "tier_histogram": {"exact": 300, "relaxed": 0, "backup": 0},
            "min_h_floor": -0.05, "deadlock": True,
        })
        s = summarize_shield([a, b])
        assert s["n_episodes"] == 2
        assert s["n_steps"] == 400
        assert s["violation_steps"] == 2
        assert s["episodes_with_violation"] == 1
        assert s["violation_episode_rate"] == pytest.approx(0.5)
        assert s["intervention_rate"] == pytest.approx(0.1)
        assert s["infeasibility_rate"] == pytest.approx(0.01)
        assert s["backup_rate"] == pytest.approx(1 / 400)
        assert s["tier_histogram"] == {"exact": 396, "relaxed": 3, "backup": 1}
        assert s["min_h_per_episode"] == [0.2, -0.05]
        assert s["min_h_floor"] == pytest.approx(-0.05)
        assert s["deadlock_episodes"] == 1
        assert s["deadlock_episode_rate"] == pytest.approx(0.5)
        assert s["timeout_rate"] == pytest.approx(0.5)
        assert s["capture_rate"] == pytest.approx(0.5)

    def test_empty(self):
        s = summarize_shield([])
        assert s["n_episodes"] == 0
        assert s["n_steps"] == 0
        assert s["min_h_per_episode"] == []
        assert s["min_h_floor"] is None
        assert s["tier_histogram"] == {"exact": 0, "relaxed": 0, "backup": 0}

    def test_from_real_shielded_duels(self):
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        logs = [
            run_shielded_duel(env, GreedyPursuer(), GreedyEvader(), CBFShield(cfg), seed=760010 + i)
            for i in range(3)
        ]
        s = summarize_shield(logs)
        assert s["n_episodes"] == 3
        assert s["n_steps"] == sum(log["steps"] for log in logs)
        assert s["violation_steps"] == 0
        assert len(s["min_h_per_episode"]) == 3
        assert sum(s["tier_histogram"].values()) == s["n_steps"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_shield.py::TestSummarizeShield -v --no-header`
Expected: ImportError `cannot import name 'summarize_shield'`.

- [ ] **Step 3: Implement**

Append to `pe2/metrics.py`:

```python
def summarize_shield(logs: list[dict]) -> dict:
    """Aggregate `run_shielded_duel` logs into the sweep-level gate numbers.

    Step-level rates are pooled over all steps (weighted by episode length),
    matching how G1 counts "violation steps over N episodes". Per-episode
    minima are kept as a list so the gate can report their distribution.
    """
    n_episodes = len(logs)
    n_steps = sum(log["shield"]["n_steps"] for log in logs)
    tiers = {t: 0 for t in TIERS}
    interventions = 0.0
    infeasible = 0.0
    backup = 0.0
    for log in logs:
        sh = log["shield"]
        for t in TIERS:
            tiers[t] += sh["tier_histogram"].get(t, 0)
        interventions += sh["intervention_rate"] * sh["n_steps"]
        infeasible += sh["infeasibility_rate"] * sh["n_steps"]
        backup += sh["backup_rate"] * sh["n_steps"]
    min_h = [log["shield"]["min_h_floor"] for log in logs]
    deadlocks = sum(bool(log["shield"]["deadlock"]) for log in logs)
    with_violation = sum(log["violations"] > 0 for log in logs)
    return {
        "n_episodes": n_episodes,
        "n_steps": n_steps,
        "violation_steps": sum(log["violations"] for log in logs),
        "episodes_with_violation": with_violation,
        "violation_episode_rate": with_violation / n_episodes if n_episodes else 0.0,
        "intervention_rate": interventions / n_steps if n_steps else 0.0,
        "infeasibility_rate": infeasible / n_steps if n_steps else 0.0,
        "backup_rate": backup / n_steps if n_steps else 0.0,
        "tier_histogram": tiers,
        "min_h_per_episode": min_h,
        "min_h_floor": min(min_h) if min_h else None,
        "deadlock_episodes": deadlocks,
        "deadlock_episode_rate": deadlocks / n_episodes if n_episodes else 0.0,
        "timeout_rate": (
            sum(log["outcome"] == "timeout" for log in logs) / n_episodes if n_episodes else 0.0
        ),
        "capture_rate": (
            sum(log["outcome"] == "capture" for log in logs) / n_episodes if n_episodes else 0.0
        ),
    }
```

Pooling note for the reviewer: `intervention_rate * n_steps` recovers the integer count exactly up to float rounding (`0.10 * 100 = 10.000000000000002`); `pytest.approx` absorbs it, and the gate script will store counts as floats. If exact integers are wanted later, add `n_interventions` to `ShieldMetrics.summary()` — not needed for G1–G4.

- [ ] **Step 4: Run the whole pe2 suite and lint**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q --no-header && ./venv/bin/ruff check pe2 tests/pe2 && ./venv/bin/ruff format --check pe2 tests/pe2`
Expected: 0 failed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add pe2/metrics.py tests/pe2/test_shield.py
git commit -m "feat(pe2/metrics): summarize_shield — pooled gate numbers for Step 6

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Record the semantics in the worklog (pre-registration hygiene, D83)

**Files:**
- Modify: `docs/worklogs/2026-09-03_S120.md` — "Work Done" and "Files Changed"

No code. D83 requires the gate's *definitions* to be on record before the gate runs; the three new definitions (infeasibility, stall, deadlock) were not in the S119 pre-registration.

- [ ] **Step 1: Append to "Work Done"**

```markdown
- **Step 6 part 1+2 executed** (plan `docs/superpowers/plans/2026-09-03-step6-backup-fix-and-gate-metrics.md`): backup-tier CCW tie-break (root cause: `wrap_angle(π) = −π`); `filter` info now carries `feasible / v_nom / v_exec / stalled`; `ShieldMetrics` adds `infeasibility_rate`, `backup_rate`, `tier_histogram`, `stall_step_rate`, `longest_stall_run`, `deadlock`; new `run_shielded_duel` + `summarize_shield`. **Definitions added to the Step 6 pre-registration (before G1–G4 run)**: infeasible step = exact QP has no solution (`method != "exact"`); stalled step = executed speed ≤ 5 % v_max while nominal ≥ 50 % v_max; deadlocked episode = ≥ 40 consecutive stalled steps (2.0 s; behavioral signature of the CBF force-equilibrium stall of Grover, Liu & Sycara). These are **reported, not gated** in G1–G4.
```

- [ ] **Step 2: Add rows to "Files Changed"**

```markdown
| `pe2/shield.py` | Modified | Backup CCW tie-break; feasible/v_nom/v_exec/stalled in filter info |
| `pe2/metrics.py` | Modified | Infeasibility, tier histogram, stall/deadlock; `summarize_shield` |
| `pe2/scripted.py` | Modified | `run_shielded_duel` |
| `tests/pe2/test_shield.py` | Modified | +12 tests (tie-break ×2, info ×3, metrics ×3, duel ×4 → see file) |
```

- [ ] **Step 3: Commit the worklog**

```bash
git add docs/worklogs/2026-09-03_S120.md
git commit -m "docs(S120): pre-register Step 6 infeasibility/stall/deadlock definitions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review (done while writing)

- **Spec coverage**: plan v3 Step 6 "record infeasibility rate, min-h, fallback-tier histogram in the gate JSON" → Tasks 3 + 5; "deadlock diagnostics" → Tasks 2 + 3; failing test → Task 1. The robust-margin arm (item 3) and running G1–G4 (item 4) are deliberately out of scope and will be separate plans.
- **Placeholder scan**: none; every step has code or an exact command.
- **Type consistency**: `filter` info keys (`feasible`, `v_nom`, `v_exec`, `stalled`) are the ones `ShieldMetrics.record` reads; `run_shielded_duel` log keys (`violations`, `shield`, `outcome`, `steps`) are the ones `summarize_shield` reads; `TIERS`/`DEADLOCK_STEPS` are defined in Task 3 before use in Task 5 tests.
- **Known caveat**: `test_filter_flags_stall_when_shield_stops_a_moving_nominal` depends on the exact boundary arithmetic (`v ≤ γ·h/dt` with `a_v = −1` at heading 0); its comment shows the derivation so a wrong constant is caught as a math error, not silenced.
