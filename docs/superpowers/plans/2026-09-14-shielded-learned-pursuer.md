# Shielded Learned Pursuer (S4 / Step 8-alt) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the v2 CBF shield inside `SearchMDP` (deployment and training), and build the S4 runner: 21 paired deployment cells across an opponent family, three shield-in-training arms, the H-SAFE / H1 / H1-family / H-COST / H3-lite verdicts, a figure and the visual-gate GIF.

**Architecture:** `SearchMDP` gains an optional `shield`: every executed pursuer command (law or learner) is de-normalised, filtered by `CBFShield.filter` with the true pose and static map, re-normalised and executed; `ShieldMetrics` records each step with the env's contact flag, and `play_episode`'s log carries `shielded_role`, `violations` and `shield` so `pe2.metrics.summarize_shield` applies unchanged. A cumulative `violations_total` on the env gives training-time counts. `scripts/derisk_s4_shield.py` runs one deployment cell per process (`--cell POLICY:OPPONENT:ARM`), the three training arms (`--arms STRSH16 …`), the summary and the GIF, modelled on `scripts/derisk_s3_confirm.py` (guards, provenance, crash record) and `scripts/pe2_step06_gate.py` (shield arms, `summarize_shield`).

**Tech Stack:** Python 3.12, `./venv/bin/python`, stable-baselines3 2.7, gymnasium, numpy, matplotlib, imageio (`pe2.render.save_gif`), pytest (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), ruff.

**Spec:** `docs/pathway/shielded_learned_pursuer_prereg_2026-09-14.md` (binding; read it first).

## Global Constraints

- Always `./venv/bin/python`; tests `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest <files> -q`; lint `./venv/bin/ruff format <files> && ./venv/bin/ruff check <files>` clean before each commit.
- Pre-registered constants, never tunable: `N_CELL = 500`, `TOTAL_LONG = 10_000_000`, `EVERY_LONG = 1_000_000`, `EVAL_N = 200`, `PROBE_SEED0 = 1_100_000`, `SEEDS = (16, 17, 18)`, `PASS_MIN = 2`, `P_GATE = 0.05`, `ROBUST_DELTA = 0.05`, `GAMMA_S = 0.99`, `HOLD` (library), `r_acq = 0.0`; opponent streams `GREEDY12 1_090_000`, `FOV 1_091_000`, `HUG 1_092_000`, `OPEN 1_093_000`, `GREEDY10 1_094_000`, `GREEDY14 1_095_000`; opponents: `GreedyEvader` at `evader_speed_ratio` 1.2 / 1.0 / 1.4, `FovLimitedEvader`, `ObstacleHuggingEvader`, `OpenSpaceEvader` (all at 1.2); shield arms `NONE = NoOpShield(cfg)`, `SHIELD = CBFShield(cfg)`, `ROBUST = CBFShield(cfg, margin=0.05)`; policies `STR14` (`results/derisk/ckpts_STR14/STR14_10000000.zip`, hold 10) and `TEACHER` (`OrbitSearch`, hold 1); TEACHER only on GREEDY12.
- Violation = an env contact step of the pursuer (`info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]`), magnitude-blind. H-SAFE = 0 such steps on every SHIELD/ROBUST cell and on every STRSH arm's training (cumulative) and probes.
- The shield filters with the TRUE pursuer pose and `env.obstacles` (as Step 6); the learner's observation is never changed; the training reward is never touched.
- Training arms = S3's configuration exactly plus `shield=CBFShield(cfg)` inside the training env and every probe/gate rollout; `OMP_NUM_THREADS=2` everywhere; provenance recorded.
- Never run a cell, an arm, the summary or the GIF against the real `results/derisk/`; tests and smoke runs redirect every output path into `tmp_path`/scratch and may only read the STR14 checkpoint.
- Commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; lazy SB3/torch imports in library functions; docstrings say why with decision numbers (D97, D89, D111).

---

### Task 1: the shield inside `SearchMDP`

**Files:**
- Modify: `pe2/structured.py` (`SearchMDP.__init__`, `_begin`, `_step_inner`, `play_episode`; `evaluate_structured`)
- Test: `tests/pe2/test_structured.py` (append)

**Interfaces:**
- Consumes: `pe2.shield.CBFShield(cfg, margin=…)`, `NoOpShield(cfg)`, `shield.filter((v, w), pose, obstacles, v_max=) -> (np.ndarray[v, w], info)`; `pe2.metrics.ShieldMetrics().record(info, violation=bool)`, `.summary()`; `SingleAgentWrapper._scale_action` semantics (`v = (a+1)/2·v_max`, `w = a·omega_max`); env info flags `pursuer_wall_contact`, `pursuer_obstacle_contact`.
- Produces:
  - `SearchMDP(cfg, opponent_factory, seed0=0, hold=HOLD, r_acq=R_ACQ, zero_reward=False, shield=None)`; attributes `.shield`, `.shield_metrics` (a fresh `ShieldMetrics` per episode when `shield` is set, else `None`), `.violations_total` (cumulative pursuer contact steps over the env's life, counted with or without a shield), `.episode_violations`.
  - `play_episode` log gains, when `shield` is set: `"shielded_role": "pursuer"`, `"violations": <int>`, `"shield": metrics.summary()`; without a shield the log is unchanged.
  - `evaluate_structured(model, cfg, opponent_factory, n_episodes, seed0, hold=HOLD, shield_factory=None)` — `shield_factory()` builds ONE shield used for the whole call (shields are stateless across episodes; `ShieldMetrics` is per episode).

- [ ] **Step 1: Write the failing tests**

Append to `tests/pe2/test_structured.py`:

```python
class TestShieldInSearchMDP:
    def test_no_shield_leaves_logs_unchanged(self):
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader)
        log = m.play_episode(OrbitSearch(CFG).predict, 900_930)
        assert "shield" not in log and "shielded_role" not in log
        assert m.shield is None and m.shield_metrics is None
        assert m.violations_total >= 0

    def test_noop_shield_reproduces_the_unshielded_episode_and_measures(self):
        from pe2.metrics import summarize_shield
        from pe2.shield import NoOpShield
        from pe2.structured import SearchMDP

        plain = SearchMDP(CFG, GreedyEvader, hold=1).play_episode(OrbitSearch(CFG).predict, 900_931)
        m = SearchMDP(CFG, GreedyEvader, hold=1, shield=NoOpShield(CFG))
        log = m.play_episode(OrbitSearch(CFG).predict, 900_931)
        assert (log["outcome"], log["steps"], log["los_breaks"]) == (plain["outcome"], plain["steps"], plain["los_breaks"])
        assert log["shielded_role"] == "pursuer"
        assert log["shield"]["n_steps"] == log["steps"]
        assert log["shield"]["tier_histogram"]["none"] == log["steps"]
        assert log["violations"] == m.episode_violations
        s = summarize_shield([log])
        assert s["n_episodes"] == 1 and s["violation_steps"] == log["violations"]

    def test_cbf_shield_filters_every_step_and_counts_no_violation(self):
        from pe2.shield import CBFShield
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader, hold=10, shield=CBFShield(CFG))
        for i in range(3):
            log = m.play_episode(RandomSearch(i).predict, 900_940 + i)
            assert log["shield"]["n_steps"] == log["steps"]
            assert log["shield"]["tier_histogram"]["none"] == 0
            assert log["violations"] == 0  # D97: zero contact steps at delta = 0
        assert m.violations_total == 0

    def test_unshielded_random_search_does_hit_walls(self):
        # the contact counter must be live without a shield, and a full-speed
        # random searcher collides somewhere in a few episodes (G4's logic)
        from pe2.structured import SearchMDP

        class Straight:
            def predict(self, obs, deterministic=True):
                return np.array([1.0, 0.0], dtype=np.float32), None

        m = SearchMDP(CFG, GreedyEvader, hold=10)
        for i in range(5):
            m.play_episode(Straight().predict, 900_950 + i)
        assert m.violations_total > 0

    def test_shield_is_applied_to_the_law_branch_too(self):
        # every executed command goes through the filter: n_steps equals env steps
        # even on an episode where the law (chase) plays most of the steps
        from pe2.shield import CBFShield
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader, hold=1, shield=CBFShield(CFG))
        m.trace = []
        log = m.play_episode(OrbitSearch(CFG).predict, 900_960)
        assert any(b == "law" for _, b, _, _ in m.trace)
        assert log["shield"]["n_steps"] == len(m.trace) == log["steps"]

    def test_evaluate_structured_shield_factory(self):
        from pe2.shield import CBFShield
        from pe2.structured import evaluate_structured

        logs = evaluate_structured(RandomSearch(0), CFG, GreedyEvader, 2, 900_970, hold=10, shield_factory=lambda: CBFShield(CFG))
        assert all("shield" in lg and lg["shielded_role"] == "pursuer" for lg in logs)
        plain = evaluate_structured(RandomSearch(0), CFG, GreedyEvader, 2, 900_970, hold=10)
        assert all("shield" not in lg for lg in plain)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_structured.py -k ShieldInSearchMDP -q`
Expected: FAIL — `TypeError: unexpected keyword 'shield'` / missing attributes.

- [ ] **Step 3: Implement**

In `pe2/structured.py`:

`__init__` gains `shield=None` (last parameter) and sets:

```python
        # S4 (spec section 2): optional CBF shield on EVERY executed pursuer
        # command — the law's and the learner's alike — using the true pose and
        # the static map, as in Step 6 (D97). The learner's observation is
        # untouched; only execution differs. ShieldMetrics is per episode.
        self.shield = shield
        self.shield_metrics = None
        self.violations_total = 0  # cumulative pursuer contact steps, shield or not
        self.episode_violations = 0
```

`_begin` adds, after `self._info = {}`:

```python
        self.episode_violations = 0
        self.shield_metrics = ShieldMetrics() if self.shield is not None else None
```

(`from pe2.metrics import ShieldMetrics` at module top — it is a pure dataclass module.)

`_step_inner` becomes:

```python
    def _step_inner(self, action: np.ndarray, branch: str = "law"):
        prev = self._visible
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        sinfo = None
        if self.shield is not None:
            cfg = self.cfg
            env = self.inner.env
            v_nom, w_nom = (a[0] + 1.0) / 2.0 * cfg.v_max, a[1] * cfg.omega_max
            (v, w), sinfo = self.shield.filter(
                (float(v_nom), float(w_nom)), env.pursuer_pose, env.obstacles, v_max=cfg.v_max
            )
            a = np.array([2.0 * v / cfg.v_max - 1.0, w / cfg.omega_max], dtype=np.float32)
        obs, r, term, trunc, info = self.inner.step(a)
        self.env_steps += 1
        contact = bool(info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"])
        self.violations_total += int(contact)
        self.episode_violations += int(contact)
        if sinfo is not None:
            self.shield_metrics.record(sinfo, violation=contact)
        self._visible = bool(obs[6] > 0.5)
        self._los_breaks += int(prev and not self._visible)
        self._info = info
        if self.trace is not None:
            self.trace.append((self._visible, branch, float(a[0]), float(a[1])))
        onset = int((not prev) and self._visible)
        return obs, float(r), onset, term, trunc
```

Note the trace now records the EXECUTED (post-shield) normalised action; without a shield it is the clipped nominal, which is what the env executes anyway (the earlier tests still pass because the orbit/still actions are inside the box).

`play_episode` returns, when `self.shield is not None`, the log dict `| {"shielded_role": "pursuer", "violations": self.episode_violations, "shield": self.shield_metrics.summary()}`.

`evaluate_structured` gains `shield_factory=None` and builds `SearchMDP(cfg, opponent_factory, hold=hold, shield=shield_factory() if shield_factory is not None else None)`.

- [ ] **Step 4: Run the whole file**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_structured.py -q`
Expected: all PASS, including V1 (`hold=1` orbit == teacher — the no-shield path is byte-identical apart from the clip, which the orbit never triggers).

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/structured.py tests/pe2/test_structured.py && ./venv/bin/ruff check pe2/structured.py tests/pe2/test_structured.py
git add pe2/structured.py tests/pe2/test_structured.py
git commit -m "feat(pe2): CBF shield inside SearchMDP — filters law and learner commands, per-episode ShieldMetrics (S4)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `scripts/derisk_s4_shield.py` — cells, training arms, verdicts, figure, GIF

**Files:**
- Create: `scripts/derisk_s4_shield.py`
- Test: `tests/pe2/test_derisk_s4_round.py`

**Interfaces:**
- Consumes: Task 1; `pe2.derisk.gate_pass`; `pe2.metrics.summarize_shield`; `pe2.eval.mcnemar_pvalue, wilson_ci, win_count`; `pe2.structured` (`SearchMDP`, `StructuredEvalProbe`, `evaluate_structured`, `OrbitSearch`, `HOLD`); `pe2.shield.CBFShield, NoOpShield`; `pe2.render.Renderer, Trails, FovOverlay, ShieldOverlay, save_gif`; `pe2.scripted` evaders.
- Produces: CLI `--cell POLICY:OPP:ARM [...] | --arms STRSH16 … | --summarize | --gif [--device cpu]`; module-level `CELLS`, `OPPONENTS`, `ARMS`, `paired_shift`, `h1_verdict`, `hsafe_verdict`, `h3_verdict`, `direction`, `summarize`; artefacts per spec §8.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_derisk_s4_round.py
"""scripts/derisk_s4_shield.py — the shielded-learned-pursuer round's own decisions."""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s4_shield.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s4_shield", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _logs(wins, n, seed0=0):
    return [{"seed": seed0 + i, "role": "pursuer", "outcome": "capture" if i < wins else "timeout"} for i in range(n)]


def _cell(policy, opp, arm, wins, violations=0, n=500):
    return {"policy": policy, "opponent": opp, "arm": arm, "logs": _logs(wins, n),
            "shield": {"violation_steps": violations, "intervention_rate": 0.1, "infeasibility_rate": 0.0,
                       "backup_rate": 0.0, "stall_step_rate": 0.2, "deadlock_episode_rate": 0.1,
                       "min_clearance_floor": 0.01, "tier_histogram": {"exact": n, "relaxed": 0, "backup": 0, "none": 0}},
            "rate": wins / n}


class TestProtocol:
    def test_constants(self):
        from pe2.structured import HOLD

        assert (R.N_CELL, R.TOTAL_LONG, R.EVERY_LONG, R.EVAL_N) == (500, 10_000_000, 1_000_000, 200)
        assert R.PROBE_SEED0 == 1_100_000 and R.SEEDS == (16, 17, 18) and R.PASS_MIN == 2
        assert R.P_GATE == 0.05 and R.ROBUST_DELTA == 0.05 and R.GAMMA_S == 0.99 and R.HOLD == HOLD
        assert R.OPPONENTS["GREEDY12"]["seed0"] == 1_090_000 and R.OPPONENTS["GREEDY14"]["speed"] == 1.4
        assert set(R.OPPONENTS) == {"GREEDY12", "FOV", "HUG", "OPEN", "GREEDY10", "GREEDY14"}
        assert set(R.SHIELD_ARMS) == {"NONE", "SHIELD", "ROBUST"}

    def test_cells_are_21_and_teacher_only_on_greedy12(self):
        assert len(R.CELLS) == 21
        teacher = [c for c in R.CELLS if c.startswith("TEACHER:")]
        assert teacher == ["TEACHER:GREEDY12:NONE", "TEACHER:GREEDY12:SHIELD", "TEACHER:GREEDY12:ROBUST"]
        assert "STR14:OPEN:ROBUST" in R.CELLS

    def test_training_arms(self):
        assert set(R.ARMS) == {"STRSH16", "STRSH17", "STRSH18"}
        assert R.ARMS["STRSH17"] == {"seed": 17}

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s4_cell_STR14_GREEDY12_NONE.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_cell_not_run("STR14:GREEDY12:NONE")
        R.require_cell_not_run("STR14:GREEDY12:SHIELD")
        (tmp_path / "ckpts_STRSH16").mkdir()
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("STRSH16")


class TestShift:
    def test_paired_shift_direction_and_ci(self):
        s = R.paired_shift(_logs(300, 500), _logs(340, 500))  # shielded 0.60 vs unshielded 0.68: 40 losses, 0 wins
        assert s["delta"] == pytest.approx(-0.08) and s["b"] == 0 and s["c"] == 40 and s["p"] < 0.05
        assert s["ci"][0] <= s["delta"] <= s["ci"][1]

    def test_h1_confirmed(self):
        cells = {"STR14:GREEDY12:NONE": _cell("STR14", "GREEDY12", "NONE", 340),
                 "STR14:GREEDY12:SHIELD": _cell("STR14", "GREEDY12", "SHIELD", 300)}
        assert R.h1_verdict(cells)["verdict"] == "CONFIRMED"

    def test_h1_no_shift(self):
        cells = {"STR14:GREEDY12:NONE": _cell("STR14", "GREEDY12", "NONE", 340),
                 "STR14:GREEDY12:SHIELD": _cell("STR14", "GREEDY12", "SHIELD", 338)}
        assert R.h1_verdict(cells)["verdict"] == "NO SHIFT"

    def test_h1_reversed(self):
        cells = {"STR14:GREEDY12:NONE": _cell("STR14", "GREEDY12", "NONE", 300),
                 "STR14:GREEDY12:SHIELD": _cell("STR14", "GREEDY12", "SHIELD", 340)}
        assert R.h1_verdict(cells)["verdict"] == "REVERSED"

    def test_h1_incomplete(self):
        assert R.h1_verdict({})["verdict"] == "INCOMPLETE"


class TestSafety:
    def test_hsafe_holds_when_all_shielded_cells_and_arms_are_clean(self):
        cells = {"STR14:GREEDY12:SHIELD": _cell("STR14", "GREEDY12", "SHIELD", 300),
                 "STR14:GREEDY12:ROBUST": _cell("STR14", "GREEDY12", "ROBUST", 290),
                 "STR14:GREEDY12:NONE": _cell("STR14", "GREEDY12", "NONE", 340, violations=30)}
        arms = {"STRSH16": {"train_violations": 0, "probe_violations": 0, "final": {"violations": 0}}}
        h = R.hsafe_verdict(cells, arms)
        assert h["verdict"] == "HOLDS" and h["shielded_violation_steps"] == 0

    def test_hsafe_fails_on_any_shielded_violation(self):
        cells = {"STR14:FOV:SHIELD": _cell("STR14", "FOV", "SHIELD", 300, violations=1)}
        assert R.hsafe_verdict(cells, {})["verdict"] == "FAILS"
        cells = {"STR14:FOV:SHIELD": _cell("STR14", "FOV", "SHIELD", 300)}
        arms = {"STRSH16": {"train_violations": 2, "probe_violations": 0, "final": {"violations": 0}}}
        assert R.hsafe_verdict(cells, arms)["verdict"] == "FAILS"

    def test_none_arm_weakness_is_reported(self):
        cells = {"STR14:GREEDY12:NONE": _cell("STR14", "GREEDY12", "NONE", 340, violations=0)}
        h = R.hsafe_verdict(cells, {})
        assert h["none_greedy12_violation_episode_rate"] is not None


class TestH3:
    def _arm(self, k, pass_vs_shield):
        return {"final": {"violations": 0, "gate": {"STR14_SHIELD": {"pass": pass_vs_shield, "rate": k / 500},
                                                    "TEACHER_NONE": {"pass": True, "rate": k / 500}}}}

    def test_recovered_needs_two_of_three(self):
        arms = {"STRSH16": self._arm(330, True), "STRSH17": self._arm(320, True), "STRSH18": self._arm(280, False)}
        assert R.h3_verdict(arms)["verdict"] == "RECOVERED"
        arms["STRSH17"] = self._arm(280, False)
        assert R.h3_verdict(arms)["verdict"] == "NOT RECOVERED"

    def test_missing_seed_is_incomplete(self):
        assert R.h3_verdict({"STRSH16": self._arm(330, True)})["verdict"] == "INCOMPLETE"


class TestDirection:
    def test_rows(self):
        assert "central result" in R.direction({"hsafe": "HOLDS", "h1": "CONFIRMED", "h3": "RECOVERED"})
        assert "free for the learned pursuer" in R.direction({"hsafe": "HOLDS", "h1": "NO SHIFT", "h3": "NOT RECOVERED"})
        assert "refutation" in R.direction({"hsafe": "HOLDS", "h1": "REVERSED", "h3": "INCOMPLETE"})
        assert "Stop" in R.direction({"hsafe": "FAILS", "h1": "CONFIRMED", "h3": "RECOVERED"})
        assert "incomplete" in R.direction({"hsafe": "HOLDS", "h1": "INCOMPLETE", "h3": "INCOMPLETE"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s4_round.py -q`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Write the runner**

```python
# scripts/derisk_s4_shield.py
"""S4 — Shielded Games on the learned pursuer (plan v3 Step 8-alt).
Spec: docs/pathway/shielded_learned_pursuer_prereg_2026-09-14.md. Nothing here
may change a threshold, a seed, a stream, a rule or a budget after a number is
seen.

Deployment cells (one process each): POLICY:OPPONENT:ARM with
  POLICY   STR14 (D111 artefact, hold 10) | TEACHER (OrbitSearch, hold 1 — V1-identical to SearchingPursuer)
  OPPONENT GREEDY12 (primary) FOV HUG OPEN GREEDY10 GREEDY14
  ARM      NONE (NoOpShield) | SHIELD (CBFShield, delta 0) | ROBUST (CBFShield, delta 0.05)
TEACHER runs on GREEDY12 only -> 6*3 + 3 = 21 cells, N = 500 paired episodes each.
Training arms: STRSH16/17/18 = S3's structured learner with the shield INSIDE the
loop (R3-lite), 10M env steps, gated shielded on the primary stream.

Usage:
  ./venv/bin/python scripts/derisk_s4_shield.py --cell STR14:GREEDY12:NONE   (one per process)
  ./venv/bin/python scripts/derisk_s4_shield.py --arms STRSH16
  ./venv/bin/python scripts/derisk_s4_shield.py --summarize
  ./venv/bin/python scripts/derisk_s4_shield.py --gif
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import gate_pass
from pe2.eval import mcnemar_pvalue, wilson_ci, win_count
from pe2.metrics import summarize_shield
from pe2.scripted import FovLimitedEvader, GreedyEvader, ObstacleHuggingEvader, OpenSpaceEvader
from pe2.structured import HOLD

OUT = pathlib.Path("results/derisk")
STR14_CKPT = OUT / "ckpts_STR14" / "STR14_10000000.zip"

# --- pre-registered (spec sections 3-4). Do not tune. ---
N_CELL = 500
TOTAL_LONG = 10_000_000
EVERY_LONG = 1_000_000
EVAL_N = 200
PROBE_SEED0 = 1_100_000
SEEDS = (16, 17, 18)
PASS_MIN = 2
P_GATE = 0.05
ROBUST_DELTA = 0.05
GAMMA_S = 0.99
OPPONENTS = {
    "GREEDY12": {"factory": GreedyEvader, "speed": 1.2, "seed0": 1_090_000},
    "FOV": {"factory": FovLimitedEvader, "speed": 1.2, "seed0": 1_091_000},
    "HUG": {"factory": ObstacleHuggingEvader, "speed": 1.2, "seed0": 1_092_000},
    "OPEN": {"factory": OpenSpaceEvader, "speed": 1.2, "seed0": 1_093_000},
    "GREEDY10": {"factory": GreedyEvader, "speed": 1.0, "seed0": 1_094_000},
    "GREEDY14": {"factory": GreedyEvader, "speed": 1.4, "seed0": 1_095_000},
}
SHIELD_ARMS = ("NONE", "SHIELD", "ROBUST")
POLICIES = {"STR14": {"hold": HOLD}, "TEACHER": {"hold": 1}}
CELLS = [f"STR14:{o}:{a}" for o in OPPONENTS for a in SHIELD_ARMS] + [f"TEACHER:GREEDY12:{a}" for a in SHIELD_ARMS]
ARMS = {f"STRSH{s}": {"seed": s} for s in SEEDS}


def cfg_for(opp: str) -> EnvConfig:
    return dataclasses.replace(EnvConfig(), evader_speed_ratio=OPPONENTS[opp]["speed"])


def make_shield(arm: str, cfg: EnvConfig):
    from pe2.shield import CBFShield, NoOpShield

    if arm == "NONE":
        return NoOpShield(cfg)
    if arm == "SHIELD":
        return CBFShield(cfg)
    if arm == "ROBUST":
        return CBFShield(cfg, margin=ROBUST_DELTA)
    raise ValueError(arm)


def _ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


def provenance() -> dict:
    import os
    import subprocess

    import stable_baselines3
    import torch

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    return {"git_commit": sha, "sb3_version": stable_baselines3.__version__,
            "torch_threads": torch.get_num_threads(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}


def _cell_file(cell: str) -> pathlib.Path:
    return OUT / f"s4_cell_{cell.replace(':', '_')}.json"


def require_cell_not_run(cell: str) -> None:
    if _cell_file(cell).exists():
        raise SystemExit(f"cell {cell} has already run; the stop rule (spec section 5) forbids re-running it.")


def require_arm_not_run(arm: str) -> None:
    if (OUT / f"s4_arm_{arm}.json").exists() or (OUT / f"ckpts_{arm}").exists():
        raise SystemExit(f"arm {arm} has already run; the stop rule (spec section 5) forbids re-running it.")


# ----------------------------------------------------------------------------
# deployment cells
# ----------------------------------------------------------------------------
def policy_predict(policy: str, cfg: EnvConfig, device: str):
    from stable_baselines3 import PPO

    from pe2.structured import OrbitSearch

    if policy == "STR14":
        return PPO.load(STR14_CKPT, device=device).predict
    if policy == "TEACHER":
        return OrbitSearch(cfg).predict
    raise ValueError(policy)


class _P:
    def __init__(self, f):
        self.predict = f


def run_cell(cell: str, device: str) -> dict:
    from pe2.structured import evaluate_structured

    require_cell_not_run(cell)
    policy, opp, arm = cell.split(":")
    if policy not in POLICIES or opp not in OPPONENTS or arm not in SHIELD_ARMS or cell not in CELLS:
        raise SystemExit(f"unknown cell {cell}")
    cfg = cfg_for(opp)
    logs = evaluate_structured(
        _P(policy_predict(policy, cfg, device)), cfg, OPPONENTS[opp]["factory"], N_CELL,
        OPPONENTS[opp]["seed0"], hold=POLICIES[policy]["hold"], shield_factory=lambda: make_shield(arm, cfg),
    )
    result = {
        "cell": cell, "policy": policy, "opponent": opp, "arm": arm, "n": N_CELL, "seed0": OPPONENTS[opp]["seed0"],
        "speed": OPPONENTS[opp]["speed"], "hold": POLICIES[policy]["hold"], "provenance": provenance(),
        "score": _ci(*win_count(logs, "pursuer")), "shield": summarize_shield(logs), "logs": logs,
    }
    result["rate"] = result["score"]["rate"]
    _cell_file(cell).write_text(json.dumps(result, indent=2))
    sh = result["shield"]
    print(f"[{cell}] capture {result['rate']:.3f}; violation steps {sh['violation_steps']}; intervention {sh['intervention_rate']:.3f}; "
          f"stall {sh['stall_step_rate']:.3f}; deadlock eps {sh['deadlock_episode_rate']:.3f}; clearance floor {sh['min_clearance_floor']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# shield-in-training arms (R3-lite)
# ----------------------------------------------------------------------------
def run_train_arm(arm: str, device: str, holder: dict | None = None) -> dict:
    from stable_baselines3 import PPO

    from pe2.shield import CBFShield
    from pe2.structured import SearchMDP, StructuredEvalProbe, evaluate_structured

    require_arm_not_run(arm)
    seed = ARMS[arm]["seed"]
    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    # probes are SHIELDED rollouts (spec section 3): StructuredEvalProbe builds its
    # own SearchMDP through evaluate_structured, so pass the shield factory via hold+shield
    probe = ShieldedStructuredProbe(cfg, GreedyEvader, every=EVERY_LONG, total=TOTAL_LONG, eval_n=EVAL_N,
                                    seed0=PROBE_SEED0, ckpt_dir=d, report_path=OUT / f"s4_probe_{arm}.json", label=arm)
    if holder is not None:
        holder["rows"] = probe.rows
    env = SearchMDP(cfg, GreedyEvader, seed0=seed * 1000, hold=HOLD, r_acq=0.0, zero_reward=False, shield=CBFShield(cfg))
    model = PPO("MlpPolicy", env, seed=seed, gamma=GAMMA_S, verbose=0, device=device)
    model.save(OUT / f"s4_init_{seed}.zip")
    model.learn(total_timesteps=TOTAL_LONG, callback=probe.callback)
    result = {
        "arm": arm, "seed": seed, "env_steps": int(env.env_steps), "decisions": int(model.num_timesteps),
        "train_violations": int(env.violations_total), "probe_violations": int(sum(r.get("violations", 0) for r in probe.rows)),
        "checkpoints": probe.rows, "provenance": provenance(),
    }
    ckpt = d / f"{arm}_{TOTAL_LONG}.zip"
    if not any(r["mark"] == TOTAL_LONG for r in probe.rows) or not ckpt.exists():
        result["verdict"], result["reason"] = "UNDETERMINED", f"final probe at mark {TOTAL_LONG} never fired (env steps {env.env_steps})"
        return result
    # gate: shielded, on the primary stream, vs SHIELD STR14 and vs NONE TEACHER (their paired logs)
    refs = {}
    for name, cell in (("STR14_SHIELD", "STR14:GREEDY12:SHIELD"), ("TEACHER_NONE", "TEACHER:GREEDY12:NONE")):
        f = _cell_file(cell)
        if not f.exists():
            raise SystemExit(f"{arm}: reference cell {cell} has not run; run the deployment cells first.")
        refs[name] = json.loads(f.read_text())["logs"]
    gcfg = cfg_for("GREEDY12")
    logs = evaluate_structured(PPO.load(ckpt, device=device), gcfg, GreedyEvader, N_CELL, OPPONENTS["GREEDY12"]["seed0"],
                               hold=HOLD, shield_factory=lambda: CBFShield(gcfg))
    result["final"] = {
        "ckpt": str(ckpt), "score": _ci(*win_count(logs, "pursuer")), "violations": int(sum(lg["violations"] for lg in logs)),
        "shield": summarize_shield(logs), "gate": {name: gate_pass(logs, ref, "pursuer", P_GATE) for name, ref in refs.items()},
    }
    g = result["final"]["gate"]["STR14_SHIELD"]
    result["verdict"] = f"final {'PASS' if g['pass'] else 'FAIL'} vs STR14_SHIELD"
    result["reason"] = (f"final {g['k']}/{g['n']} = {g['rate']:.3f}, McNemar p={g['p']:.2e} (b={g['b']}, c={g['c']}) vs shielded STR14 "
                        f"{g['ref_rate']:.3f}; vs unshielded teacher {'PASS' if result['final']['gate']['TEACHER_NONE']['pass'] else 'fail'}; "
                        f"training violations {result['train_violations']}, probe violations {result['probe_violations']}, gate violations {result['final']['violations']}")
    return result


class ShieldedStructuredProbe:
    """StructuredEvalProbe whose evaluations run SHIELDED (spec section 3) and whose
    rows carry the probe's violation count. Same env-step grid, same stop rule."""

    def __init__(self, cfg, opponent_factory, every, total, eval_n, seed0, ckpt_dir, report_path, label, hold=HOLD):
        from stable_baselines3.common.callbacks import BaseCallback

        from pe2.shield import CBFShield
        from pe2.structured import evaluate_structured

        if total % every != 0:
            raise ValueError(f"total ({total}) must be a multiple of every ({every})")
        rows: list[dict] = []

        def evaluate_at(model, env_steps: int, mark: int) -> dict:
            path = ckpt_dir / f"{label}_{mark}.zip"
            model.save(path)
            logs = evaluate_structured(model, cfg, opponent_factory, eval_n, seed0, hold=hold, shield_factory=lambda: CBFShield(cfg))
            k, n = win_count(logs, "pursuer")
            row = {"steps": env_steps, "mark": mark, "ckpt": str(path), "win_rate": k / n, "k": k, "n": n,
                   "violations": int(sum(lg["violations"] for lg in logs)),
                   "intervention_rate": summarize_shield(logs)["intervention_rate"]}
            rows.append(row)
            with open(report_path, "w") as f:
                json.dump({"checkpoints": rows}, f, indent=2)
            print(f"  @{env_steps:>9} env steps (mark {mark}): shielded win rate {k}/{n} = {k / n:.3f}, violations {row['violations']}", flush=True)
            return row

        class _CB(BaseCallback):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.next_mark = every

            def _on_step(inner_self) -> bool:
                counts = inner_self.training_env.get_attr("env_steps")
                assert len(counts) == 1
                env_steps = int(counts[0])
                while inner_self.next_mark <= total and env_steps >= inner_self.next_mark:
                    evaluate_at(inner_self.model, env_steps, inner_self.next_mark)
                    inner_self.next_mark += every
                return env_steps < total

        self.rows = rows
        self.callback = _CB()
        self.evaluate_at = evaluate_at


def run_arm(arm: str, device: str) -> dict:
    holder: dict = {}
    try:
        result = run_train_arm(arm, device, holder)
    except Exception as e:
        (OUT / f"s4_arm_{arm}.json").write_text(json.dumps({
            "arm": arm, "verdict": "UNDETERMINED", "reason": f"crashed: {type(e).__name__}: {e}",
            "checkpoints": holder.get("rows", []), "provenance": provenance()}, indent=2))
        raise
    (OUT / f"s4_arm_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# verdicts (spec section 4)
# ----------------------------------------------------------------------------
def paired_shift(logs_shielded: list[dict], logs_unshielded: list[dict]) -> dict:
    """Delta = rate(shielded) - rate(unshielded) with a paired CI from the discordant
    counts (b = shielded wins where unshielded lost, c = the reverse): the difference
    of paired proportions is (b - c)/n with variance (b + c - (b - c)^2/n)/n^2."""
    import math

    p, b, c = mcnemar_pvalue(logs_shielded, logs_unshielded, "pursuer")
    ks, n = win_count(logs_shielded, "pursuer")
    ku, _ = win_count(logs_unshielded, "pursuer")
    delta = (ks - ku) / n
    var = (b + c - (b - c) ** 2 / n) / n**2
    half = 1.96 * math.sqrt(max(var, 0.0))
    return {"delta": delta, "ci": [delta - half, delta + half], "p": p, "b": b, "c": c,
            "rate_shielded": ks / n, "rate_unshielded": ku / n, "n": n}


def h1_verdict(cells: dict, policy: str = "STR14", opp: str = "GREEDY12", arm: str = "SHIELD") -> dict:
    s, u = cells.get(f"{policy}:{opp}:{arm}"), cells.get(f"{policy}:{opp}:NONE")
    if s is None or u is None:
        return {"verdict": "INCOMPLETE"}
    sh = paired_shift(s["logs"], u["logs"])
    lo_s, hi_s = wilson_ci(*win_count(s["logs"], "pursuer"))
    if sh["p"] < P_GATE and sh["c"] > sh["b"] and hi_s < sh["rate_unshielded"]:
        v = "CONFIRMED"
    elif sh["p"] < P_GATE and sh["b"] > sh["c"]:
        v = "REVERSED"
    else:
        v = "NO SHIFT"
    return {"verdict": v, "cell": f"{policy}:{opp}:{arm}"} | sh


def hsafe_verdict(cells: dict, arms: dict) -> dict:
    shielded = {k: v for k, v in cells.items() if v["arm"] in ("SHIELD", "ROBUST")}
    steps = sum(v["shield"]["violation_steps"] for v in shielded.values())
    offenders = [k for k, v in shielded.items() if v["shield"]["violation_steps"] > 0]
    for a, r in arms.items():
        tv = int(r.get("train_violations", 0)) + int(r.get("probe_violations", 0)) + int((r.get("final") or {}).get("violations", 0))
        steps += tv
        if tv:
            offenders.append(a)
    none = cells.get("STR14:GREEDY12:NONE")
    none_rate = None
    if none is not None:
        n = none["n"] if "n" in none else len(none["logs"])
        eps = none["shield"].get("episodes_with_violation")
        none_rate = (eps / n) if eps is not None else None
    return {"verdict": "HOLDS" if steps == 0 else "FAILS", "shielded_violation_steps": steps, "offenders": offenders,
            "none_greedy12_violation_episode_rate": none_rate,
            "none_probe_too_benign": (none_rate is not None and none_rate < 0.05)}


def h3_verdict(arms: dict) -> dict:
    per = {}
    for a in ARMS:
        r = arms.get(a)
        per[a] = None if r is None or "final" not in r else bool(r["final"]["gate"]["STR14_SHIELD"]["pass"])
    if any(v is None for v in per.values()):
        return {"verdict": "INCOMPLETE", "per_seed": per}
    n = sum(per.values())
    return {"verdict": "RECOVERED" if n >= PASS_MIN else "NOT RECOVERED", "passes": n, "per_seed": per}


def direction(h: dict) -> str:
    if h["hsafe"] == "FAILS":
        return ("H-SAFE FAILS: a shielded arm recorded a contact step. Stop; the census is written; no further round "
                "without a user ruling (Step 6 stop rule).")
    if h["h1"] == "INCOMPLETE":
        return "incomplete: the primary cells STR14:GREEDY12:{NONE,SHIELD} must both run"
    if h["h1"] == "CONFIRMED":
        d = ("H1 CONFIRMED at zero violations: R1 is measured on a learned agent with the sign the exact twin predicted — "
             "the paper's central result. Magnitude and costs go to the results section; next: Step 9 (shield-attacker) on STR14-SHIELD.")
    elif h["h1"] == "NO SHIFT":
        d = ("H1 NO SHIFT: the shield is free for the learned pursuer at this margin, contradicting Step 6's scripted shift and "
             "T3's prediction on a learned policy; report with the cost table.")
    else:
        d = ("H1 REVERSED: the shield helps the learned pursuer — a refutation of T3's direction on this policy class; "
             "report with the trajectories that explain it (visual gate).")
    if h["h3"] == "RECOVERED":
        d += " H3-lite RECOVERED: shield-in-training closes the deployment gap; STRSH's best seed is the shielded artefact for Step 9."
    elif h["h3"] == "NOT RECOVERED":
        d += " H3-lite NOT RECOVERED: Step 10 needs the feasibility-constraint / margin-shaping components."
    return d


def load_cells() -> dict:
    out = {}
    for p in sorted(OUT.glob("s4_cell_*.json")):
        r = json.loads(p.read_text())
        out[r["cell"]] = r
    return out


def load_arms() -> dict:
    return {p.stem.removeprefix("s4_arm_"): json.loads(p.read_text()) for p in sorted(OUT.glob("s4_arm_*.json"))}


def summarize(cells: dict, arms: dict) -> dict:
    h1 = h1_verdict(cells)
    fam = {}
    for opp in OPPONENTS:
        for arm in ("SHIELD", "ROBUST"):
            r = h1_verdict(cells, "STR14", opp, arm)
            fam[f"{opp}:{arm}"] = {k: r[k] for k in ("verdict", "delta", "ci", "p", "b", "c", "rate_shielded", "rate_unshielded") if k in r}
    teacher = {arm: h1_verdict(cells, "TEACHER", "GREEDY12", arm) for arm in ("SHIELD", "ROBUST")}
    cost_keys = ("intervention_rate", "infeasibility_rate", "backup_rate", "stall_step_rate", "deadlock_episode_rate", "min_clearance_floor")
    cost = {k: {ck: v["shield"].get(ck) for ck in cost_keys} | {"rate": v["rate"]} for k, v in cells.items()}
    hs = hsafe_verdict(cells, arms)
    h3 = h3_verdict(arms)
    h = {"hsafe": hs["verdict"], "h1": h1["verdict"], "h3": h3["verdict"]}
    return {
        "spec": "docs/pathway/shielded_learned_pursuer_prereg_2026-09-14.md",
        "hsafe": hs, "h1": h1, "h1_family": fam, "teacher_shift": teacher, "h3": h3, "cost": cost,
        "arms": {a: {"verdict": r.get("verdict"), "reason": r.get("reason")} for a, r in arms.items()},
        "complete": all(c in cells for c in CELLS) and all(a in arms for a in ARMS),
        "direction": direction(h),
    }


def write_figure(cells: dict, arms: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax = axes[0]
    opps = list(OPPONENTS)
    for j, (arm, c) in enumerate((("SHIELD", "tab:blue"), ("ROBUST", "tab:orange"))):
        xs, ys, err = [], [], []
        for i, opp in enumerate(opps):
            r = summary["h1_family"].get(f"{opp}:{arm}", {})
            if "delta" in r:
                xs.append(i + (j - 0.5) * 0.25); ys.append(r["delta"]); err.append(r["delta"] - r["ci"][0])
        if xs:
            ax.errorbar(xs, ys, yerr=err, fmt="o", color=c, capsize=3, label=f"STR14 {arm} − NONE")
    t = summary["teacher_shift"].get("SHIELD", {})
    if "delta" in t:
        ax.errorbar([-0.6], [t["delta"]], yerr=[t["delta"] - t["ci"][0]], fmt="s", color="k", capsize=3, label="TEACHER SHIELD − NONE (GREEDY12)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(opps))); ax.set_xticklabels(opps, fontsize=8)
    ax.set_ylabel("Δ capture rate (paired, 95 % CI)"); ax.set_title(f"S4 — shift (H1: {summary['h1']['verdict']})"); ax.legend(fontsize=7)
    ax = axes[1]
    keys = ("intervention_rate", "stall_step_rate", "deadlock_episode_rate")
    for j, cell in enumerate(("STR14:GREEDY12:SHIELD", "TEACHER:GREEDY12:SHIELD", "STR14:GREEDY12:ROBUST", "TEACHER:GREEDY12:ROBUST")):
        c = summary["cost"].get(cell)
        if c:
            ax.bar([i + j * 0.2 for i in range(len(keys))], [c[k] or 0 for k in keys], width=0.2, label=cell)
    ax.set_xticks([i + 0.3 for i in range(len(keys))]); ax.set_xticklabels(keys, fontsize=8); ax.set_title("cost (reported)"); ax.legend(fontsize=6)
    ax = axes[2]
    for a, r in arms.items():
        rows = r.get("checkpoints") or []
        if rows:
            ax.plot([x["steps"] for x in rows], [x["win_rate"] for x in rows], marker="o", label=f"{a} ({r.get('verdict')})")
    for cell, c in (("STR14:GREEDY12:SHIELD", "tab:blue"), ("STR14:GREEDY12:NONE", "tab:green"), ("TEACHER:GREEDY12:NONE", "k")):
        if cell in cells:
            ax.axhline(cells[cell]["rate"], ls=":", color=c, lw=1.2, label=f"{cell} = {cells[cell]['rate']:.3f}")
    ax.set_xlabel("env steps"); ax.set_title(f"shield-in-training (H3-lite: {summary['h3']['verdict']})")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6)
    fig.suptitle("S4 — Shielded Games on the learned pursuer" + ("" if summary["complete"] else " — INCOMPLETE"))
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def write_gif(path, device: str, every: int = 4) -> None:
    """Visual gate (spec section 6): STR14 unshielded vs shielded on the same
    GREEDY12 episodes — one where the shield changes the outcome, one where it
    does not — with the FOV cone and the ShieldOverlay (nominal vs executed)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from stable_baselines3 import PPO

    from pe2.render import FovOverlay, Renderer, ShieldOverlay, Trails, save_gif
    from pe2.shield import CBFShield, NoOpShield
    from pe2.structured import SearchMDP

    cfg = cfg_for("GREEDY12")
    predict = PPO.load(STR14_CKPT, device=device).predict

    class Rec(SearchMDP):
        def __init__(self, *a, label, **k):
            super().__init__(*a, **k)
            self.label = label; self.frames = []; self.r = Renderer(cfg, dpi=80, figsize=(5, 5)); self.t = Trails()
            self.live = {"nominal": None, "executed": None, "info": None}
            self.ov = [FovOverlay("pursuer"), ShieldOverlay("pursuer", self.shield, self.live)]

        def _step_inner(self, action, branch="law"):
            env = self.inner.env
            self.t.append("pursuer", *env.pursuer_pose[:2]); self.t.append("evader", *env.evader_pose[:2])
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            nominal = ((a[0] + 1) / 2 * cfg.v_max, a[1] * cfg.omega_max)
            out = super()._step_inner(action, branch)
            v, w = self.trace[-1][2], self.trace[-1][3]
            self.live.update(nominal=nominal, executed=((v + 1) / 2 * cfg.v_max, w * cfg.omega_max),
                             info=self.shield_metrics and self._last_sinfo)
            if env.current_step % every == 0:
                self.frames.append(self.r.draw(env, trails=self.t, overlays=self.ov, title_prefix=f"{self.label} [{branch}]"))
            return out

        def finish(self, outcome, step):
            for _ in range(10):
                self.frames.append(self.r.draw(self.inner.env, trails=self.t, overlays=self.ov, title_prefix=f"{self.label}  {outcome.upper()} @ {step}"))
            plt.close(self.r.fig)

    def play(seed, shield, label):
        m = Rec(cfg, GreedyEvader, hold=HOLD, shield=shield, label=label); m.trace = []
        log = m.play_episode(predict, seed); m.finish(log["outcome"], log["steps"]); return m.frames, log

    def find(pred):
        for s in range(OPPONENTS["GREEDY12"]["seed0"], OPPONENTS["GREEDY12"]["seed0"] + 80):
            u = SearchMDP(cfg, GreedyEvader, hold=HOLD, shield=NoOpShield(cfg)).play_episode(predict, s)
            sh = SearchMDP(cfg, GreedyEvader, hold=HOLD, shield=CBFShield(cfg)).play_episode(predict, s)
            if pred(u, sh):
                return s
        return None

    changed = find(lambda u, sh: u["outcome"] != sh["outcome"])
    same = find(lambda u, sh: u["outcome"] == sh["outcome"] == "capture")
    frames = []
    for s in (changed, same):
        if s is None:
            continue
        fu, _ = play(s, NoOpShield(cfg), "STR14 unshielded"); fs, _ = play(s, CBFShield(cfg), "STR14 shielded (δ=0)")
        n = max(len(fu), len(fs)); fu += [fu[-1]] * (n - len(fu)); fs += [fs[-1]] * (n - len(fs))
        frames += [np.concatenate([a, b], axis=1) for a, b in zip(fu, fs, strict=True)]
    save_gif(frames, str(path), fps=15)
    print(f"wrote {path} ({len(frames)} frames; seeds changed={changed}, same={same})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", nargs="*", default=[])
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for cell in args.cell:
        run_cell(cell, args.device)
    for arm in args.arms:
        run_arm(arm, args.device)
    if args.summarize:
        cells, arms = load_cells(), load_arms()
        summary = summarize(cells, arms)
        (OUT / "s4_round.json").write_text(json.dumps({"summary": summary, "arms": arms}, indent=2))
        write_figure(cells, arms, summary, OUT / "s4_round.png")
        print(json.dumps({k: summary[k] for k in ("hsafe", "h1", "h3")}, indent=2, default=str))
        print(json.dumps(summary["h1_family"], indent=2))
        print(f"\n{summary['direction']}")
    if args.gif:
        write_gif(OUT / "step_08_shield.gif", args.device)


if __name__ == "__main__":
    main()
```

Implementation notes for the implementer: (1) `write_gif`'s `Rec` reads `self._last_sinfo` — add to Task 1's `_step_inner` the line `self._last_sinfo = sinfo` (initialised to `None` in `__init__`), so the overlay draws exactly what was logged; if Task 1 is already committed, add it in this task with a one-line test asserting `m._last_sinfo["method"] in TIERS` after a shielded step. (2) The trace records the executed normalised action, which is what `Rec` converts back to `(v, w)` for the overlay's "executed" arc. (3) `dataclasses.replace(EnvConfig(), evader_speed_ratio=…)` requires `EnvConfig` to be a dataclass — it is. (4) The deployment cells never train; `run_cell` is pure evaluation and is safe to run 21-wide.

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s4_round.py -q`
Expected: all PASS.

- [ ] **Step 5: Smoke the wiring at a tiny budget in a scratch directory**

Throwaway script under `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/ff03fc68-2116-4512-a227-312f3099d143/scratchpad/s4smoke/`: import the runner via `importlib`, monkeypatch `OUT` to the scratch dir and `N_CELL=6, TOTAL_LONG=4000, EVERY_LONG=2000, EVAL_N=4`, then `run_cell` for `STR14:GREEDY12:NONE`, `STR14:GREEDY12:SHIELD`, `STR14:GREEDY14:ROBUST`, `TEACHER:GREEDY12:NONE`, `TEACHER:GREEDY12:SHIELD`, then `run_arm("STRSH16", "cpu")`, then `summarize(load_cells(), load_arms())`, `write_figure(...)`, and `write_gif(scratch/"g.gif", "cpu", every=20)`. Expected: no exception; every SHIELD/ROBUST cell has `shield.violation_steps == 0`; the arm record has `train_violations == 0`; the figure and GIF exist. Paste the console output into your report. Nothing under the real `results/derisk/` may change.

- [ ] **Step 6: Lint, full suite, commit**

```bash
./venv/bin/ruff format scripts/derisk_s4_shield.py tests/pe2/test_derisk_s4_round.py && ./venv/bin/ruff check scripts/derisk_s4_shield.py tests/pe2/test_derisk_s4_round.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q -x
git add scripts/derisk_s4_shield.py tests/pe2/test_derisk_s4_round.py pe2/structured.py tests/pe2/test_structured.py
git commit -m "feat(derisk): S4 runner — 21 shielded deployment cells, shield-in-training arms, H-SAFE/H1/H3 verdicts, figure, visual-gate GIF" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** §2 components → Task 1 (`shield=`, metrics, log keys, `evaluate_structured(shield_factory=)`); §3 policies/opponents/arms/streams/N/training arms → Task 2 constants, `CELLS`, `run_cell`, `run_train_arm`, `ShieldedStructuredProbe`; §4 H-SAFE (cells + training + probes + gate violations, NONE-arm benign check) → `hsafe_verdict`; H1 rule (p, c > b, Wilson-hi < unshielded point) → `h1_verdict`; H1-family and teacher shift → `summarize`; H-COST → `cost`; H3-lite (≥ 2/3 vs SHIELD STR14, teacher reported, violations) → `run_train_arm` gate + `h3_verdict`; §5 direction → `direction`; stop rule → guards; §6 GIF → `write_gif`; §8 artefacts → paths.
- **Placeholders:** none.
- **Type consistency:** `evaluate_structured(model, cfg, opp, n, seed0, hold=, shield_factory=)` used identically in Tasks 1–2; cell records carry `logs`, `shield` (summarize_shield block), `rate`, `arm` as the tests' `_cell` fixture mirrors; `gate_pass` returns `pass/rate/ref_rate/k/n/p/b/c` as read by `run_train_arm` and `h3_verdict`.
