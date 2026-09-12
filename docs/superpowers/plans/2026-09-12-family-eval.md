# F1 Family Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace a yardstick that D104 showed is degenerate. Build the one missing evader, evaluate every existing pursuer against a four-evader family with a worst-case readout, and run the two pre-registered validity checks that decide whether the new measure is adopted.

**Architecture:** One new scripted evader in `pe2/scripted.py`, one evaluation module in `pe2/family.py` that scores a policy set against an evader set on paired seeds, and one script that assembles the policy zoo, runs the matrix and evaluates the validity checks. No training anywhere.

**Tech Stack:** Python 3.10+, numpy, gymnasium, stable-baselines3 2.7.1, matplotlib, pytest, ruff.

**Spec:** `docs/pathway/family_eval_prereg_2026-09-12.md` — §2 (the family), §3 (protocol), §4 (the policy set), §5 (the validity checks) and §6 (direction) are binding. Decisions: D83, D104.

## Global Constraints

- Always `./venv/bin/python`; tests `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest <target> -q`. Full `tests/pe2` takes ~7 min; run it once, in Task 3.
- Lint only touched files (`./venv/bin/ruff check` + `ruff format --check`).
- **Pre-registered constants, verbatim from spec §2/§3/§4 — never tuned:** `N_EPISODES = 200`; seed streams `GreedyEvader` 930_000, `FovLimitedEvader` 940_000, `ObstacleHuggingEvader` 950_000, `OpenSpaceEvader` 960_000; `EnvConfig()` at shipped defaults; `OpenSpaceEvader` uses 16 candidate headings, a 1.0 s projection at the evader's top speed, and the existing `K_OMEGA = 3.0` steering gain.
- **This round trains nothing.** Any code path that calls `learn()`, `train_bc_actor`, `finetune_*` or writes a checkpoint is out of scope.
- **The validity checks in spec §5 are the round's product.** They are computed by the code, recorded in the JSON, and reported whether they pass or fail. The family membership must NOT be adjusted to make them pass.
- `results/derisk/t0*`, `results/derisk/t1*` and `results/visual_gates/**` are read-only evidence. The earlier round runners are frozen artefacts: do not edit them.
- Branch `family-eval` from `main`. Commit only the files each task names. Trailer: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

---

## File map

| file | responsibility |
|---|---|
| `pe2/scripted.py` | `OpenSpaceEvader` — flees without cornering itself (Task 1) |
| `tests/pe2/test_scripted.py` | its tests, incl. that it beats `GreedyEvader` against a constant arc (Task 1) |
| `pe2/family.py` | `EVADER_FAMILY`, `family_matrix`, `worst_case`, `validity_checks` (Task 2) |
| `tests/pe2/test_family.py` | the matrix and the checks (Task 2) |
| `scripts/derisk_f1_family.py` | the policy zoo, the run, the figure, the CLI (Task 3) |
| `docs/worklogs/2026-09-12_S125.md`, `docs/decisions.md`, `docs/workflow_tracker.md` | Task 3 record |

---

### Task 1: `OpenSpaceEvader`

**Files:**
- Modify: `pe2/scripted.py` (append after `GreedyEvader`; touch nothing else)
- Test: `tests/pe2/test_scripted.py` (append)

**Interfaces:**
- Produces: `OpenSpaceEvader` with `uses_truth = True`, `reset()`, `act(obs, env) -> (v, omega)`. Module constants `OPEN_SPACE_HEADINGS = 16`, `OPEN_SPACE_HORIZON = 1.0` (seconds).

- [ ] **Step 1: Write the failing tests**

Append to `tests/pe2/test_scripted.py`:

```python
class TestOpenSpaceEvader:
    """F1 spec §2: the evader that punishes circling.

    D104: `GreedyEvader` flees directly away with no wall awareness, corners
    itself, and is therefore collected by a slow constant arc — a randomly
    initialised net scored 0.312 against it while the hand-written pursuer
    scored 0.234. This evader refuses headings that end in a wall or an
    obstacle, so it cannot be cornered that way.
    """

    def _env(self, seed=930_000):
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=seed)
        return cfg, env

    def test_it_flees_the_pursuer_in_the_open(self):
        from pe2.scripted import OpenSpaceEvader

        cfg, env = self._env()
        env.obstacles = []
        env.pursuer_pose[:] = (0.0, 0.0, 0.0)
        env.evader_pose[:] = (1.0, 0.0, 0.0)  # due east of the pursuer, facing east
        v, omega = OpenSpaceEvader().act(env._get_obs()["evader"], env)
        assert v == pytest.approx(cfg.evader_v_max)
        assert abs(omega) < 0.5  # already pointing away: little correction

    def test_it_refuses_to_drive_into_a_wall(self):
        from pe2.scripted import OpenSpaceEvader

        cfg, env = self._env()
        env.obstacles = []
        lim = cfg.arena_size / 2.0 - cfg.agent_radius
        # Evader pinned at the +x wall, pursuer due west: fleeing "away" is +x,
        # straight into the wall. It must pick a heading along the wall instead.
        env.evader_pose[:] = (lim - 0.05, 0.0, 0.0)
        env.pursuer_pose[:] = (lim - 3.0, 0.0, 0.0)
        ev = OpenSpaceEvader()
        _v, omega = ev.act(env._get_obs()["evader"], env)
        assert abs(omega) > 1.0  # it turns hard, rather than driving on into +x

    def test_it_does_not_get_cornered_by_a_constant_arc(self):
        """The behavioural point of the whole family (spec §5 check 1)."""
        from pe2.exploitability import ConstantPolicy
        from pe2.scripted import GreedyEvader, OpenSpaceEvader, run_duel

        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)

        class _Arc:
            """The D104 non-policy: half speed, a lazy left arc."""

            def reset(self):
                pass

            def act(self, obs, env):
                return 0.5 * env.cfg.v_max, -0.078

        greedy = sum(
            run_duel(env, _Arc(), GreedyEvader(), 930_000 + i)["outcome"] == "capture"
            for i in range(60)
        )
        openp = sum(
            run_duel(env, _Arc(), OpenSpaceEvader(), 960_000 + i)["outcome"] == "capture"
            for i in range(60)
        )
        assert openp < greedy, (
            f"the arc caught OpenSpaceEvader {openp}/60 vs GreedyEvader {greedy}/60 — "
            "this evader does not punish circling and the family will not fix D104"
        )
        assert ConstantPolicy  # imported for the zoo; keeps the import used

    def test_it_is_deterministic_and_stateless(self):
        from pe2.scripted import OpenSpaceEvader

        _cfg, env = self._env()
        a = OpenSpaceEvader()
        b = OpenSpaceEvader()
        b.reset()
        obs = env._get_obs()["evader"]
        assert a.act(obs, env) == b.act(obs, env)
```

- [ ] **Step 2: Run to verify failure**

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_scripted.py::TestOpenSpaceEvader -q` → FAIL, `ImportError: cannot import name 'OpenSpaceEvader'`.

- [ ] **Step 3: Implement**

Append to `pe2/scripted.py`, after `GreedyEvader`:

```python
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
    exists — which is exactly the failure a circling pursuer exploits.
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
                np.hypot(end[0] - o["x"], end[1] - o["y"]) < o["radius"] + cfg.agent_radius
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
```

- [ ] **Step 4: Run**

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_scripted.py -q` → all pass. The cornering test is the one that matters; if it fails, **stop and report** — the family cannot fix D104 without it and the spec forbids tuning the membership to rescue it.

- [ ] **Step 5: Commit**

```bash
./venv/bin/ruff check pe2/scripted.py tests/pe2/test_scripted.py && ./venv/bin/ruff format --check pe2/scripted.py tests/pe2/test_scripted.py
git add pe2/scripted.py tests/pe2/test_scripted.py
git commit -m "feat(scripted): OpenSpaceEvader — flees without cornering itself (D104)" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `pe2/family.py`

**Files:**
- Create: `pe2/family.py`
- Test: `tests/pe2/test_family.py`

**Interfaces:**
- Consumes: `pe2.exploitability.evaluate_sb3_policy`, `pe2.eval.{wilson_ci, win_count}`, `pe2.scripted.{GreedyEvader, FovLimitedEvader, ObstacleHuggingEvader, OpenSpaceEvader}`.
- Produces:
  ```python
  EVADER_FAMILY: dict[str, tuple[type, int]]   # name -> (factory, seed0), spec §2/§3
  N_EPISODES = 200
  def evaluate_against(policy, cfg, evader_factory, seed0, n=N_EPISODES) -> dict
  def family_matrix(policies: dict, cfg, n=N_EPISODES) -> dict
  def worst_case(row: dict) -> tuple[str, float]
  def validity_checks(matrix: dict) -> dict
  ```
  `evaluate_against` returns `{"k","n","rate","wilson"}`; `family_matrix` returns
  `{policy: {evader: cell}}`; `worst_case` returns the (evader, rate) of the minimum;
  `validity_checks` returns `{"arc_collapses": {...}, "ordering_restored": {...}, "pass": bool}`
  with each check carrying the numbers it compared and a worded reason.

- [ ] **Step 1: Write the failing tests**

Create `tests/pe2/test_family.py`:

```python
"""Tests for pe2.family — the worst-case-over-family yardstick (D104, F1 spec)."""

import pytest

from pe2.config import EnvConfig


class TestFamilyMembership:
    """Spec §2/§3 fix the family and its seed streams before any measurement."""

    def test_the_four_evaders_and_their_streams(self):
        from pe2.family import EVADER_FAMILY
        from pe2.scripted import (
            FovLimitedEvader,
            GreedyEvader,
            ObstacleHuggingEvader,
            OpenSpaceEvader,
        )

        assert {k: v[1] for k, v in EVADER_FAMILY.items()} == {
            "GreedyEvader": 930_000,
            "FovLimitedEvader": 940_000,
            "ObstacleHuggingEvader": 950_000,
            "OpenSpaceEvader": 960_000,
        }
        assert [v[0] for v in EVADER_FAMILY.values()] == [
            GreedyEvader,
            FovLimitedEvader,
            ObstacleHuggingEvader,
            OpenSpaceEvader,
        ]

    def test_streams_are_fresh_and_clear_of_the_test_range(self):
        from pe2.family import EVADER_FAMILY

        for _f, seed0 in EVADER_FAMILY.values():
            assert 890_999 < seed0 < 900_000 or seed0 >= 930_000

    def test_the_episode_count_is_the_preregistered_one(self):
        from pe2.family import N_EPISODES

        assert N_EPISODES == 200


class TestWorstCase:
    def test_it_returns_the_minimum_cell_and_its_evader(self):
        from pe2.family import worst_case

        row = {
            "GreedyEvader": {"rate": 0.30},
            "OpenSpaceEvader": {"rate": 0.02},
            "FovLimitedEvader": {"rate": 0.55},
        }
        assert worst_case(row) == ("OpenSpaceEvader", 0.02)

    def test_a_single_opponent_row_is_its_own_worst_case(self):
        from pe2.family import worst_case

        assert worst_case({"GreedyEvader": {"rate": 0.19}}) == ("GreedyEvader", 0.19)


class TestValidityChecks:
    """Spec §5: both checks are computed and reported, pass or fail."""

    def _matrix(self, arc_worst, teacher_worst, full_worst, clone_worst, arc3=0.0, still=0.0):
        def row(w):
            return {"GreedyEvader": {"rate": w}, "OpenSpaceEvader": {"rate": w}}

        return {
            "GreedyPursuer_fullstate": row(full_worst),
            "FovLimitedPursuer_teacher": row(teacher_worst),
            "bc_clone": row(clone_worst),
            "randinit_seed4_arc": row(arc_worst),
            "randinit_seed3_arc": row(arc3),
            "stand_still": row(still),
        }

    def test_both_checks_pass_when_the_family_works(self):
        from pe2.family import validity_checks

        out = validity_checks(self._matrix(0.02, 0.20, 0.40, 0.15))
        assert out["arc_collapses"]["pass"] is True
        assert out["ordering_restored"]["pass"] is True
        assert out["pass"] is True

    def test_it_fails_when_the_arc_survives(self):
        from pe2.family import validity_checks

        out = validity_checks(self._matrix(0.25, 0.20, 0.40, 0.15))
        assert out["arc_collapses"]["pass"] is False and out["pass"] is False
        assert "0.25" in out["arc_collapses"]["reason"]

    def test_it_fails_when_the_ordering_is_broken(self):
        from pe2.family import validity_checks

        # clone above the teacher: the ordering spec §5 requires is violated
        out = validity_checks(self._matrix(0.02, 0.20, 0.40, 0.30))
        assert out["ordering_restored"]["pass"] is False and out["pass"] is False

    def test_it_fails_when_an_arc_beats_the_clone(self):
        from pe2.family import validity_checks

        out = validity_checks(self._matrix(0.02, 0.20, 0.40, 0.15, arc3=0.18))
        assert out["ordering_restored"]["pass"] is False

    def test_every_check_carries_its_numbers_and_a_reason(self):
        from pe2.family import validity_checks

        out = validity_checks(self._matrix(0.02, 0.20, 0.40, 0.15))
        for key in ("arc_collapses", "ordering_restored"):
            assert isinstance(out[key]["reason"], str) and out[key]["reason"]


class TestEvaluateAgainst:
    def test_one_cell_has_the_expected_shape(self):
        from pe2.exploitability import ConstantPolicy
        from pe2.family import evaluate_against
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=30)
        cell = evaluate_against(
            ConstantPolicy((-1.0, 0.0)), cfg, GreedyEvader, seed0=930_000, n=4
        )
        assert set(cell) == {"k", "n", "rate", "wilson"}
        assert cell["n"] == 4 and cell["k"] == 0 and cell["rate"] == 0.0
        assert cell["wilson"][0] == 0.0

    def test_the_matrix_covers_every_policy_and_evader(self):
        from pe2.exploitability import ConstantPolicy
        from pe2.family import EVADER_FAMILY, family_matrix

        cfg = EnvConfig(max_steps=30)
        m = family_matrix({"still": ConstantPolicy((-1.0, 0.0))}, cfg, n=2)
        assert set(m) == {"still"}
        assert set(m["still"]) == set(EVADER_FAMILY)
```

- [ ] **Step 2: Run to verify failure** → `ModuleNotFoundError: No module named 'pe2.family'`.

- [ ] **Step 3: Implement `pe2/family.py`**

Module docstring must state: why a single-opponent score is degenerate (D104's numbers), that the primary readout is the worst case, and that the validity checks are computed and reported whether or not they pass.

```python
EVADER_FAMILY = {
    "GreedyEvader": (GreedyEvader, 930_000),
    "FovLimitedEvader": (FovLimitedEvader, 940_000),
    "ObstacleHuggingEvader": (ObstacleHuggingEvader, 950_000),
    "OpenSpaceEvader": (OpenSpaceEvader, 960_000),
}
N_EPISODES = 200
```

`evaluate_against(policy, cfg, evader_factory, seed0, n)` calls `evaluate_sb3_policy(policy, cfg, "pursuer", evader_factory, n_episodes=n, seed0=seed0)`, then `win_count(logs, "pursuer")` and `wilson_ci`, returning `{"k","n","rate","wilson"}`. It works for any object with `predict` — the scripted policies are adapted by the zoo in Task 3, not here.

`family_matrix(policies, cfg, n)` returns `{name: {evader: cell}}`, printing one line per cell as it goes (these runs take minutes).

`worst_case(row)` returns `min(row.items(), key=lambda kv: kv[1]["rate"])` as `(evader, rate)`.

`validity_checks(matrix)` computes each policy's worst case, then:
- `arc_collapses`: `randinit_seed4_arc`'s worst case **strictly below** `FovLimitedPursuer_teacher`'s. Record both numbers and their evaders.
- `ordering_restored`: `GreedyPursuer_fullstate` ≥ `FovLimitedPursuer_teacher` ≥ `bc_clone`, **and** each of `randinit_seed4_arc`, `randinit_seed3_arc`, `stand_still` strictly below `bc_clone`. Record every comparison.
- `pass` is the conjunction. Each check carries a worded `reason` naming the numbers, so a reader never meets a bare boolean. A policy missing from the matrix makes that check `None` with a reason, never silently `True`.

- [ ] **Step 4: Run** → `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_family.py -q` → all pass.

- [ ] **Step 5: Commit** — `git add pe2/family.py tests/pe2/test_family.py`; message `feat(family): worst-case-over-family yardstick with its validity checks`.

---

### Task 3: Run the matrix and record

**Files:**
- Create: `scripts/derisk_f1_family.py`
- Create (by the script): `results/derisk/f1_family.{json,png}`
- Create: `docs/worklogs/2026-09-12_S125.md`; Modify: `docs/decisions.md` (D105 — the outcome), `docs/workflow_tracker.md`, and the memory file

- [ ] **Step 1: Write the script**

`scripts/derisk_f1_family.py` assembles the spec §4 zoo and runs the matrix. The zoo:

| key | object |
|---|---|
| `GreedyPursuer_fullstate` | `_Scripted(GreedyPursuer)` |
| `FovLimitedPursuer_teacher` | `_Scripted(FovLimitedPursuer)` |
| `bc_clone` | `SAC.load("results/visual_gates/step_05/sac_bc_pursuer.zip")` |
| `randinit_seed4_arc` | `SAC("MlpPolicy", SingleAgentWrapper(cfg,"pursuer",GreedyEvader), seed=4, device="cpu")` |
| `randinit_seed3_arc` | same with `seed=3` |
| `t05_RE_final` | `SAC.load("results/derisk/ckpts_RE/RE_1000000.zip")` if present, else skipped with a recorded note |
| `stand_still` | `ConstantPolicy((-1.0, 0.0))` |

`_Scripted` is a thin `predict()` shim wrapping a scripted policy so `evaluate_sb3_policy` can drive it. It must hold one instance per episode — scripted policies carry per-episode state (`FovLimitedPursuer.last_seen`) — so it resets its inner policy when the wrapper's episode changes. **Simplest correct approach:** give `_Scripted` a `reset()` and have it construct a fresh inner policy on the first `predict` of each episode, detected by the evaluator's own reset. If that cannot be detected cleanly, do NOT bodge it — use `pe2.scripted.run_duel` for the scripted rows instead, which already handles per-episode policy lifetime, and note in the report that scripted rows go through `run_duel` while network rows go through `evaluate_sb3_policy`. Both use the same seeds and the same env, so the cells remain comparable.

Then: write `f1_family.json` with the full matrix, every worst case, and the `validity_checks` block; draw the grouped bar chart to `f1_family.png`; print the two checks and the verdict.

- [ ] **Step 2: Full suite once** — `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q` → all pass (expect ~615). Record the count.

- [ ] **Step 3: Run** — `./venv/bin/python scripts/derisk_f1_family.py | tee results/derisk/f1_family.log`. Expect 2–3 h. No training; if anything calls `learn()`, stop.

- [ ] **Step 4: Record**

Create `docs/worklogs/2026-09-12_S125.md` per the Session Protocol with: the full policy × evader matrix with Wilson intervals; the worst-case column; **both validity checks with their numbers, pass or fail**; the figure signed as the visual gate; and the direction per spec §6.

Add **D105** with the outcome. State plainly which of spec §6's two cases obtains:
- both checks pass → the worst-case-over-family score is the project's yardstick; Route C's negative claim is restated against it before any paper text; the belief-augmented clone is the next pre-registration and is judged on this metric;
- either fails → the family metric is not the fix; record it and **stop** — no further evaluation or learning work until a different measurement idea is pre-registered.

Update the tracker and the memory file. **Do not adjust the family to make a check pass** — spec §5 forbids it, and doing so would reproduce exactly the error D104 exposed.

- [ ] **Step 5: Stop.** The direction is the user's to act on.

---

## Self-review

- **Spec coverage:** §1 premise → D104, already recorded; §2 family → Task 1 + `EVADER_FAMILY`; §3 protocol → `N_EPISODES`, the seed streams, `EnvConfig()`; §4 policy set → Task 3's zoo; §5 validity checks → `validity_checks` + Task 3 Step 4; §6 direction → Task 3 Step 4; §7 (no training, belief work deferred) → the Global Constraint and Task 3 Step 3; §8 artefacts → Task 3.
- **Placeholders:** none. The one open construction decision (how scripted policies reach the evaluator) is given a primary approach and a named fallback, with the instruction not to bodge it.
- **Type consistency:** `EVADER_FAMILY: dict[str, tuple[factory, int]]` used identically in Tasks 2 and 3; `evaluate_against(...) -> {"k","n","rate","wilson"}` consumed by `family_matrix`, `worst_case` and `validity_checks`; the zoo's key strings (`randinit_seed4_arc`, `FovLimitedPursuer_teacher`, `bc_clone`, `randinit_seed3_arc`, `stand_still`, `GreedyPursuer_fullstate`) are exactly the names `validity_checks` looks up.
