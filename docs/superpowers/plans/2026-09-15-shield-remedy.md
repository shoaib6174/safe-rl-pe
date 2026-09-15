# Shield-Aware Remedy (S5 / Step 10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `SearchMDP` two shield-aware training ingredients — three shield-state observation features and a per-step stall penalty — and build the S5 runner (references from S4's cells and STRSH checkpoints → ten training arms → H3 per arm, RZ harness, artefact rule, cost table vs STRSH, figure).

**Architecture:** `SearchMDP` gains `shield_obs` (appends `[intervention, min_clearance clipped, stall_run/DEADLOCK_STEPS clipped]` from the last executed step's shield info to the observation the LEARNER sees; the scripted law keeps the raw 11-dim observation) and `stall_penalty` (subtracts `w` per stalled env step within a learner's span, off under `zero_reward`). `evaluate_structured` forwards both. `scripts/derisk_s5_remedy.py` mirrors `scripts/derisk_s4_shield.py`'s training arm, probe, guards, crash record and provenance; its `--measure-start` collects the paired reference logs (S4 cell files + the three STRSH final checkpoints re-scored shielded on the same stream).

**Tech Stack:** Python 3.12, `./venv/bin/python`, stable-baselines3 2.7, gymnasium, numpy, matplotlib, pytest (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), ruff.

**Spec:** `docs/pathway/shield_remedy_prereg_2026-09-15.md` (binding; read it first).

## Global Constraints

- Always `./venv/bin/python`; tests `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest <files> -q`; ruff format + check clean before each commit.
- Pre-registered, never tunable: `STALL_W = 1e-3`; features = last step's `intervention` (0/1), `min_clearance` clipped to [−0.5, 2.0], `stall_run / DEADLOCK_STEPS` clipped to [0, 2]; without a shield or before any step: (0.0, 2.0, 0.0); `TOTAL_LONG = 10_000_000`, `EVERY_LONG = 1_000_000`, `EVAL_N = 200`, `GATE_N = 500`, `PROBE_SEED0 = 1_110_000`, `GATE_SEED0 = 1_090_000` (S4's primary stream), `SEEDS = (19, 20, 21)`, `PASS_MIN = 2`, `P_GATE = 0.05`, `GAMMA_S = 0.99`, `HOLD`, `r_acq = 0.0`; arms `RA` (shield_obs, w=0), `RB` (no obs, w=1e-3), `RC` (both), `RZ` (shield_obs, zero reward, seed 19 only).
- Training env = S4's STRSH configuration exactly plus the arm's ingredients: `SearchMDP(cfg, GreedyEvader, seed0=seed*1000, hold=HOLD, r_acq=0.0, zero_reward=<RZ>, shield=CBFShield(cfg), shield_obs=<arm>, stall_penalty=<arm>)`, `PPO("MlpPolicy", env, seed=seed, gamma=0.99, verbose=0, device)`; probes and gate are SHIELDED rollouts with the arm's `shield_obs` (the policy's input width must match) and `stall_penalty=0` (evaluation is unshaped).
- References (paired on 1 090 000+): `TEACHER_SHIELD` = S4 cell `TEACHER:GREEDY12:SHIELD` logs (verdict bar), `STR14_SHIELD`, `TEACHER_NONE` (S4 cell logs), and `STRSH16/17/18` = the S4 checkpoints `ckpts_STRSH{16,17,18}/STRSH*_10000000.zip` re-scored shielded on the same stream (must reproduce 210/500, 216/500, 208/500 — reported).
- H-SAFE: 0 contact steps across every arm's training, probes and gate — kill criterion. H3(arm) iff ≥ 2/3 seeds pass vs `TEACHER_SHIELD` by `gate_pass`; H3 overall iff any arm passes; RZ passing ⇒ all `UNDETERMINED (harness)`; missing seed ⇒ that arm `INCOMPLETE`; artefact = highest gate rate among passing arms' passing seeds, ties → arm order RC, RA, RB then lowest seed.
- Never run any arm or `--measure-start` against the real `results/derisk/`; tests and smoke runs redirect `OUT` and read only the existing S4 artefacts (cells, STRSH checkpoints) and the STR14 checkpoint.
- Commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; docstrings say why with decision numbers (D112).

---

### Task 1: `shield_obs` and `stall_penalty` in `SearchMDP`

**Files:**
- Modify: `pe2/structured.py` (`SearchMDP.__init__`, `_begin`, `_step_inner`, `_shaped`, `reset`, `step`, `play_episode`; `evaluate_structured`)
- Test: `tests/pe2/test_structured.py` (append)

**Interfaces:**
- Consumes: existing `SearchMDP` (with `shield`, `_last_sinfo`, `shield_metrics`), `pe2.metrics.DEADLOCK_STEPS`, `ShieldMetrics._current_stall_run`, `gymnasium.spaces.Box`, `pe2.obs.obs_dim`.
- Produces:
  - `SearchMDP(…, shield=None, shield_obs=False, stall_penalty=0.0)`; `observation_space` is Box(−2, 2, (obs_dim+3,)) when `shield_obs`; `.shield_features() -> np.ndarray[3]` (float32); `.span_stalled` (stalled steps in the last learner span); info gains `stalled_steps`.
  - `evaluate_structured(model, cfg, opponent_factory, n_episodes, seed0, hold=HOLD, shield_factory=None, shield_obs=False)`.
  - Reward per span = `_shaped(reward, onsets) − stall_penalty × span_stalled`, and 0.0 under `zero_reward`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pe2/test_structured.py`:

```python
class TestShieldAwareIngredients:
    def test_features_default_without_a_shield(self):
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader, shield_obs=True)
        np.testing.assert_allclose(m.shield_features(), [0.0, 2.0, 0.0])
        assert m.observation_space.shape == (14,)
        obs, _ = m.reset(seed=900_980)
        assert obs.shape == (14,) and obs.dtype == np.float32
        np.testing.assert_allclose(obs[-3:], [0.0, 2.0, 0.0])

    def test_features_come_from_the_last_executed_step(self):
        from pe2.metrics import DEADLOCK_STEPS
        from pe2.shield import CBFShield
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader, hold=10, shield=CBFShield(CFG), shield_obs=True)
        obs, _ = m.reset(seed=900_981)
        assert obs.shape == (14,)
        f = m.shield_features()
        s = m._last_sinfo
        assert f[0] == float(bool(s["intervention"]))
        assert f[1] == pytest.approx(float(np.clip(s["min_clearance"], -0.5, 2.0)))
        assert f[2] == pytest.approx(float(np.clip(m.shield_metrics._current_stall_run / DEADLOCK_STEPS, 0.0, 2.0)))
        # the law still sees the raw 11-dim observation: play a full episode without error
        log = m.play_episode(RandomSearch(1).predict, 900_981)
        assert log["violations"] == 0

    def test_shield_obs_off_leaves_the_observation_unchanged(self):
        from pe2.shield import CBFShield
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader, hold=10, shield=CBFShield(CFG))
        obs, _ = m.reset(seed=900_982)
        assert obs.shape == (11,) and m.observation_space.shape == (11,)

    def test_stall_penalty_subtracts_per_stalled_step_in_the_span(self):
        from pe2.shield import CBFShield
        from pe2.structured import SearchMDP

        class Wall:  # drive straight into the nearest wall: the shield must stall it
            def predict(self, obs, deterministic=True):
                return np.array([1.0, 0.0], dtype=np.float32), None

        m = SearchMDP(CFG, GreedyEvader, hold=10, shield=CBFShield(CFG), stall_penalty=1e-3)
        obs, _ = m.reset(seed=900_983)
        total_stalled = 0
        total_penalty = 0.0
        done = False
        while not done:
            obs, r, term, trunc, info = m.step(Wall().predict(obs)[0])
            done = term or trunc
            assert info["stalled_steps"] == m.span_stalled
            total_stalled += info["stalled_steps"]
            total_penalty += r - info["raw_reward"]
        assert total_stalled > 0
        assert total_penalty == pytest.approx(-1e-3 * total_stalled)

    def test_stall_penalty_is_zero_under_zero_reward(self):
        from pe2.shield import CBFShield
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader, hold=10, shield=CBFShield(CFG), stall_penalty=1e-3, zero_reward=True)
        obs, _ = m.reset(seed=900_984)
        _, r, _, _, info = m.step(np.array([1.0, 0.0], dtype=np.float32))
        assert r == 0.0

    def test_evaluate_structured_with_shield_obs(self):
        from pe2.shield import CBFShield
        from pe2.structured import evaluate_structured

        class Echo:
            def predict(self, obs, deterministic=True):
                assert obs.shape == (14,)
                return np.array([0.5, 0.0], dtype=np.float32), None

        logs = evaluate_structured(Echo(), CFG, GreedyEvader, 2, 900_990, hold=10, shield_factory=lambda: CBFShield(CFG), shield_obs=True)
        assert len(logs) == 2 and all("shield" in lg for lg in logs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_structured.py -k ShieldAware -q`
Expected: FAIL — `TypeError: unexpected keyword 'shield_obs'`.

- [ ] **Step 3: Implement**

In `pe2/structured.py`:

Imports: add `from gymnasium import spaces` and `from pe2.metrics import DEADLOCK_STEPS, ShieldMetrics` (replace the existing `ShieldMetrics` import) and `from pe2.obs import obs_dim` (extend the existing `pe2.obs` import).

`__init__` gains `shield_obs: bool = False, stall_penalty: float = 0.0` after `shield=None`, and:

```python
        # S5 (spec section 2, D112): shield-aware ingredients. `shield_obs`
        # appends the last executed step's shield state to the LEARNER's
        # observation — intervention flag, min_clearance (m, clipped), stall
        # run / DEADLOCK_STEPS (clipped) — because S4 showed a learner that
        # never sees the filter acting cannot learn around the states where it
        # is pinned. The scripted law keeps the raw observation. `stall_penalty`
        # subtracts w per stalled env step in a learner's span; NOT
        # potential-based by design (the objective must change).
        self.shield_obs = bool(shield_obs)
        self.stall_penalty = float(stall_penalty)
        self.span_stalled = 0
        if self.shield_obs:
            self.observation_space = spaces.Box(
                low=-2.0, high=2.0, shape=(obs_dim(cfg) + 3,), dtype=np.float32
            )
```

Add methods:

```python
    def shield_features(self) -> np.ndarray:
        """[intervention, min_clearance clipped to [-0.5, 2], stall_run /
        DEADLOCK_STEPS clipped to [0, 2]] of the last executed step; the
        defaults (0, 2, 0) mean 'no shield acted'."""
        s = self._last_sinfo
        if s is None or self.shield_metrics is None:
            return np.array([0.0, 2.0, 0.0], dtype=np.float32)
        run = self.shield_metrics._current_stall_run / DEADLOCK_STEPS
        return np.array(
            [
                float(bool(s["intervention"])),
                float(np.clip(s["min_clearance"], -0.5, 2.0)),
                float(np.clip(run, 0.0, 2.0)),
            ],
            dtype=np.float32,
        )

    def _learner_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.shield_obs:
            return obs
        return np.concatenate([np.asarray(obs, dtype=np.float32), self.shield_features()])
```

In `_step_inner`, after `self.shield_metrics.record(...)` inside the `if sinfo is not None:` block add `self.span_stalled += int(sinfo["stalled"])`.

`_shaped` becomes:

```python
    def _shaped(self, reward: float, onsets: int) -> float:
        if self.zero_reward:
            return 0.0
        return reward + self.r_acq * onsets - self.stall_penalty * self.span_stalled
```

`reset` returns `self._learner_obs(obs), {...}` (and resets `self.span_stalled = 0` in `_begin`). `step` sets `self.span_stalled = 0` as its first statement, adds `stalled_steps=self.span_stalled` to `info`, and returns `self._learner_obs(obs)`. `play_episode` passes `self._learner_obs(obs)` to `predict` (both the first decision after `_advance` and each subsequent one — simplest: `a, _ = predict(self._learner_obs(obs), deterministic=True)` and keep `obs` raw from `self.step`… note `self.step` now returns the learner obs; so in `play_episode` keep the loop as `obs, _r, term, trunc, _i = self.step(a)` and call `predict(obs, ...)` directly on it, but the FIRST obs after `_advance` is raw and must go through `_learner_obs`). Concretely:

```python
        obs = self._begin(seed)
        obs, _r, _o, term, trunc = self._advance(obs)
        obs = self._learner_obs(obs)
        while not (term or trunc):
            a, _ = predict(obs, deterministic=True)
            obs, _r, term, trunc, _i = self.step(a)
```

`evaluate_structured` gains `shield_obs: bool = False` and passes `shield_obs=shield_obs` to `SearchMDP` (never `stall_penalty`: evaluation is unshaped).

- [ ] **Step 4: Run the whole file**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_structured.py -q`
Expected: all PASS (V1 unchanged: `shield_obs=False`, `stall_penalty=0` are the defaults and touch nothing).

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/structured.py tests/pe2/test_structured.py && ./venv/bin/ruff check pe2/structured.py tests/pe2/test_structured.py
git add pe2/structured.py tests/pe2/test_structured.py
git commit -m "feat(pe2): shield-aware observation features and stall penalty in SearchMDP (S5)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `scripts/derisk_s5_remedy.py`

**Files:**
- Create: `scripts/derisk_s5_remedy.py`
- Test: `tests/pe2/test_derisk_s5_round.py`

**Interfaces:**
- Consumes: Task 1; `pe2.derisk.gate_pass`; `pe2.metrics.summarize_shield`; `pe2.eval.wilson_ci, win_count`; `pe2.structured` (`SearchMDP`, `evaluate_structured`, `HOLD`); `pe2.shield.CBFShield`; the S4 cell files `results/derisk/s4_cell_{TEACHER_GREEDY12_SHIELD,STR14_GREEDY12_SHIELD,TEACHER_GREEDY12_NONE}.json` (each has `logs`) and `results/derisk/ckpts_STRSH{16,17,18}/STRSH*_10000000.zip`.
- Produces: CLI `--measure-start | --arms <ARM...> | --summarize [--device cpu]`; module-level `ARMS`, `hypothesis_verdicts`, `choose_artefact`, `hsafe_verdict`, `direction`, `summarize`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_derisk_s5_round.py
"""scripts/derisk_s5_remedy.py — the remedy round's own decisions."""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s5_remedy.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s5_remedy", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _arm(name, teacher_pass, rate, viol=0):
    return {"arm": name, "family": name[:2], "seed": int(name[2:]), "train_violations": viol, "probe_violations": 0,
            "final": {"violations": 0, "score": {"rate": rate}, "gate": {"TEACHER_SHIELD": {"pass": teacher_pass, "rate": rate},
                      "STR14_SHIELD": {"pass": True, "rate": rate}, "TEACHER_NONE": {"pass": False, "rate": rate}},
                      "shield": {"stall_step_rate": 0.3, "deadlock_episode_rate": 0.2, "intervention_rate": 0.5, "infeasibility_rate": 0.0}},
            "checkpoints": []}


def _full(ra=(True, True, False), rb=(False, False, False), rc=(True, True, True), rz=False, rates=None):
    res = {}
    for fam, passes in (("RA", ra), ("RB", rb), ("RC", rc)):
        for s, p in zip(R.SEEDS, passes, strict=True):
            res[f"{fam}{s}"] = _arm(f"{fam}{s}", p, (rates or {}).get(f"{fam}{s}", 0.5 if p else 0.4))
    res["RZ19"] = _arm("RZ19", rz, 0.3)
    return res


class TestProtocol:
    def test_constants(self):
        from pe2.structured import HOLD

        assert R.STALL_W == 1e-3 and R.SEEDS == (19, 20, 21) and R.PASS_MIN == 2 and R.P_GATE == 0.05
        assert (R.TOTAL_LONG, R.EVERY_LONG, R.EVAL_N, R.GATE_N) == (10_000_000, 1_000_000, 200, 500)
        assert R.PROBE_SEED0 == 1_110_000 and R.GATE_SEED0 == 1_090_000 and R.GAMMA_S == 0.99 and R.HOLD == HOLD

    def test_arm_table(self):
        assert set(R.ARMS) == {"RA19", "RA20", "RA21", "RB19", "RB20", "RB21", "RC19", "RC20", "RC21", "RZ19"}
        assert R.ARMS["RA20"] == {"family": "RA", "seed": 20, "shield_obs": True, "stall_penalty": 0.0, "zero_reward": False}
        assert R.ARMS["RB21"] == {"family": "RB", "seed": 21, "shield_obs": False, "stall_penalty": 1e-3, "zero_reward": False}
        assert R.ARMS["RC19"] == {"family": "RC", "seed": 19, "shield_obs": True, "stall_penalty": 1e-3, "zero_reward": False}
        assert R.ARMS["RZ19"] == {"family": "RZ", "seed": 19, "shield_obs": True, "stall_penalty": 0.0, "zero_reward": True}

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s5_start.json")
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()
        (tmp_path / "s5_start.json").write_text("{}")
        with pytest.raises(SystemExit, match="already"):
            R.require_not_measured()
        (tmp_path / "s5_arm_RA19.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("RA19")
        R.require_arm_not_run("RA20")


class TestVerdicts:
    def test_h3_per_arm_and_overall(self):
        h = R.hypothesis_verdicts(_full())
        assert h["H3"] == {"RA": "PASS", "RB": "FAIL", "RC": "PASS"} and h["H3_overall"] == "PASS"

    def test_rz_pass_invalidates(self):
        h = R.hypothesis_verdicts(_full(rz=True))
        assert all(v.startswith("UNDETERMINED") for v in h["H3"].values()) and h["H3_overall"].startswith("UNDETERMINED")

    def test_missing_seed_is_incomplete(self):
        res = _full()
        del res["RB20"]
        h = R.hypothesis_verdicts(res)
        assert h["H3"]["RB"] == "INCOMPLETE" and h["H3"]["RA"] == "PASS"
        del res["RZ19"]
        assert R.hypothesis_verdicts(res)["H3_overall"] == "INCOMPLETE"

    def test_artefact_rule_order_rc_ra_rb_then_lowest_seed(self):
        res = _full(rates={"RA19": 0.60, "RA20": 0.60, "RC19": 0.60, "RC21": 0.55})
        h = R.hypothesis_verdicts(res)
        assert h["artefact"] == "RC19"  # tie at 0.60 between RA19, RA20, RC19 -> RC first, then lowest seed
        res = _full(rc=(False, False, False), rates={"RA19": 0.55, "RA20": 0.60})
        assert R.hypothesis_verdicts(res)["artefact"] == "RA20"
        res = _full(ra=(False,) * 3, rc=(False,) * 3)
        assert R.hypothesis_verdicts(res)["artefact"] is None


class TestSafety:
    def test_hsafe(self):
        assert R.hsafe_verdict(_full())["verdict"] == "HOLDS"
        res = _full()
        res["RB19"]["train_violations"] = 1
        assert R.hsafe_verdict(res)["verdict"] == "FAILS"
        res = _full()
        del res["RC21"]
        assert R.hsafe_verdict(res)["verdict"] == "INCOMPLETE"


class TestDirection:
    def test_rows(self):
        assert "safe learned pursuer" in R.direction({"hsafe": "HOLDS", "H3_overall": "PASS", "RZ": "FAIL"})
        assert "feasibility constraint" in R.direction({"hsafe": "HOLDS", "H3_overall": "FAIL", "RZ": "FAIL"})
        assert "harness" in R.direction({"hsafe": "HOLDS", "H3_overall": "UNDETERMINED (harness)", "RZ": "PASS"})
        assert "Stop" in R.direction({"hsafe": "FAILS", "H3_overall": "PASS", "RZ": "FAIL"})
        assert "incomplete" in R.direction({"hsafe": "INCOMPLETE", "H3_overall": "INCOMPLETE", "RZ": "INCOMPLETE"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s5_round.py -q`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Write the runner**

```python
# scripts/derisk_s5_remedy.py
"""S5 — Step 10 remedy: a shield-aware structured learner.
Spec: docs/pathway/shield_remedy_prereg_2026-09-15.md. Nothing here may change a
threshold, a seed, a stream, a rule or a budget after a number is seen.

Arms (S4's STRSH configuration + the ingredient; 10M env steps, shield in the loop):
  RA{19,20,21}  shield_obs                    RB{19,20,21}  stall penalty 1e-3
  RC{19,20,21}  both                          RZ19          shield_obs, all rewards zeroed (control)
Gate: final checkpoint, SHIELDED, on S4's primary stream 1 090 000+, paired with the S4
cells (TEACHER_SHIELD = the verdict bar, STR14_SHIELD, TEACHER_NONE) and the three S4
STRSH finals re-scored on that stream.

Usage:
  ./venv/bin/python scripts/derisk_s5_remedy.py --measure-start
  ./venv/bin/python scripts/derisk_s5_remedy.py --arms RA19        (one per process)
  ./venv/bin/python scripts/derisk_s5_remedy.py --summarize
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import gate_pass
from pe2.eval import wilson_ci, win_count
from pe2.metrics import summarize_shield
from pe2.scripted import GreedyEvader
from pe2.structured import HOLD

OUT = pathlib.Path("results/derisk")
START_PATH = OUT / "s5_start.json"
S4_CELL = {
    "TEACHER_SHIELD": OUT / "s4_cell_TEACHER_GREEDY12_SHIELD.json",
    "STR14_SHIELD": OUT / "s4_cell_STR14_GREEDY12_SHIELD.json",
    "TEACHER_NONE": OUT / "s4_cell_TEACHER_GREEDY12_NONE.json",
}
STRSH_CKPTS = {f"STRSH{s}": OUT / f"ckpts_STRSH{s}" / f"STRSH{s}_10000000.zip" for s in (16, 17, 18)}
STRSH_EXPECTED = {"STRSH16": 210, "STRSH17": 216, "STRSH18": 208}  # S4 finals, k of 500 — reproduction check

# --- pre-registered (spec sections 2-5). Do not tune. ---
STALL_W = 1e-3
TOTAL_LONG = 10_000_000
EVERY_LONG = 1_000_000
EVAL_N = 200
GATE_N = 500
PROBE_SEED0 = 1_110_000
GATE_SEED0 = 1_090_000
SEEDS = (19, 20, 21)
PASS_MIN = 2
P_GATE = 0.05
GAMMA_S = 0.99
FAMILIES = {
    "RA": {"shield_obs": True, "stall_penalty": 0.0, "zero_reward": False},
    "RB": {"shield_obs": False, "stall_penalty": STALL_W, "zero_reward": False},
    "RC": {"shield_obs": True, "stall_penalty": STALL_W, "zero_reward": False},
}
ARMS: dict = {f"{f}{s}": {"family": f, "seed": s, **cfg} for f, cfg in FAMILIES.items() for s in SEEDS}
ARMS["RZ19"] = {"family": "RZ", "seed": 19, "shield_obs": True, "stall_penalty": 0.0, "zero_reward": True}
ARTEFACT_ORDER = ("RC", "RA", "RB")
REFS = ("TEACHER_SHIELD", "STR14_SHIELD", "TEACHER_NONE", "STRSH16", "STRSH17", "STRSH18")


def _ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


def _write_json(path: pathlib.Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def provenance() -> dict:
    import subprocess

    import stable_baselines3
    import torch

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    return {"git_commit": sha, "sb3_version": stable_baselines3.__version__,
            "torch_threads": torch.get_num_threads(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}


# ----------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------
def require_not_measured() -> None:
    if START_PATH.exists():
        raise SystemExit(f"{START_PATH} already exists: references already measured (stop rule, spec section 6).")
    if any(OUT.glob("s5_arm_*.json")) or any((OUT / f"ckpts_{a}").exists() for a in ARMS):
        raise SystemExit("S5 arms have already run; the stop rule forbids re-measuring.")


def require_arm_not_run(arm: str) -> None:
    if (OUT / f"s5_arm_{arm}.json").exists() or (OUT / f"ckpts_{arm}").exists():
        raise SystemExit(f"arm {arm} has already run; the stop rule (spec section 6) forbids re-running it.")


def load_start() -> dict:
    if not START_PATH.exists():
        raise SystemExit(f"references missing: {START_PATH}. Run --measure-start FIRST.")
    s = json.loads(START_PATH.read_text())
    for k in REFS:
        if k not in s or "logs" not in s[k]:
            raise SystemExit(f"{k} (with paired logs) missing from {START_PATH}; re-run --measure-start.")
    return s


# ----------------------------------------------------------------------------
# references (spec section 3), BEFORE any arm
# ----------------------------------------------------------------------------
def measure_start(device: str) -> dict:
    from stable_baselines3 import PPO

    from pe2.shield import CBFShield
    from pe2.structured import evaluate_structured

    require_not_measured()
    cfg = EnvConfig()
    out: dict = {"gate_seed0": GATE_SEED0, "gate_n": GATE_N, "provenance": provenance()}
    for name, path in S4_CELL.items():
        if not path.exists():
            raise SystemExit(f"S4 cell missing: {path}")
        cell = json.loads(path.read_text())
        if cell.get("seed0") != GATE_SEED0 or len(cell["logs"]) != GATE_N:
            raise SystemExit(f"{name}: S4 cell is not the {GATE_N}-episode stream {GATE_SEED0}")
        out[name] = _ci(*win_count(cell["logs"], "pursuer")) | {"logs": cell["logs"], "source": str(path)}
        print(f"{name} = {out[name]['k']}/{GATE_N} = {out[name]['rate']:.4f} (S4 cell)", flush=True)
    for name, ckpt in STRSH_CKPTS.items():
        if not ckpt.exists():
            raise SystemExit(f"STRSH checkpoint missing: {ckpt}")
        logs = evaluate_structured(PPO.load(ckpt, device=device), cfg, GreedyEvader, GATE_N, GATE_SEED0, hold=HOLD,
                                   shield_factory=lambda: CBFShield(cfg))
        k, n = win_count(logs, "pursuer")
        out[name] = _ci(k, n) | {"logs": logs, "ckpt": str(ckpt), "expected_k": STRSH_EXPECTED[name],
                                 "reproduced": k == STRSH_EXPECTED[name]}
        print(f"{name} = {k}/{n} = {k / n:.4f} (S4 recorded {STRSH_EXPECTED[name]}/500: "
              f"{'reproduced' if k == STRSH_EXPECTED[name] else 'NOT reproduced'})", flush=True)
    _write_json(START_PATH, out)
    return out


# ----------------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------------
class ShieldedStructuredProbe:
    """StructuredEvalProbe whose evaluations run SHIELDED with the arm's
    shield_obs; rows carry the probe's violation count. Env-step grid; stops learn."""

    def __init__(self, cfg, opponent_factory, every, total, eval_n, seed0, ckpt_dir, report_path, label, shield_obs, hold=HOLD):
        from stable_baselines3.common.callbacks import BaseCallback

        from pe2.shield import CBFShield
        from pe2.structured import evaluate_structured

        if total % every != 0:
            raise ValueError(f"total ({total}) must be a multiple of every ({every})")
        rows: list[dict] = []

        def evaluate_at(model, env_steps: int, mark: int) -> dict:
            path = ckpt_dir / f"{label}_{mark}.zip"
            model.save(path)
            logs = evaluate_structured(model, cfg, opponent_factory, eval_n, seed0, hold=hold,
                                       shield_factory=lambda: CBFShield(cfg), shield_obs=shield_obs)
            k, n = win_count(logs, "pursuer")
            sh = summarize_shield(logs)
            row = {"steps": env_steps, "mark": mark, "ckpt": str(path), "win_rate": k / n, "k": k, "n": n,
                   "violations": int(sum(lg["violations"] for lg in logs)),
                   "intervention_rate": sh["intervention_rate"], "stall_step_rate": sh["stall_step_rate"],
                   "deadlock_episode_rate": sh["deadlock_episode_rate"]}
            rows.append(row)
            _write_json(report_path, {"checkpoints": rows})
            print(f"  @{env_steps:>9} env steps (mark {mark}): shielded win rate {k}/{n} = {k / n:.3f}, "
                  f"violations {row['violations']}, stall {row['stall_step_rate']:.3f}, deadlock eps {row['deadlock_episode_rate']:.3f}", flush=True)
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


def run_train_arm(arm: str, starts: dict, device: str, holder: dict | None = None) -> dict:
    from stable_baselines3 import PPO

    from pe2.shield import CBFShield
    from pe2.structured import SearchMDP, evaluate_structured

    require_arm_not_run(arm)
    spec = ARMS[arm]
    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = ShieldedStructuredProbe(cfg, GreedyEvader, every=EVERY_LONG, total=TOTAL_LONG, eval_n=EVAL_N, seed0=PROBE_SEED0,
                                    ckpt_dir=d, report_path=OUT / f"s5_probe_{arm}.json", label=arm, shield_obs=spec["shield_obs"])
    if holder is not None:
        holder["rows"] = probe.rows
    env = SearchMDP(cfg, GreedyEvader, seed0=spec["seed"] * 1000, hold=HOLD, r_acq=0.0, zero_reward=spec["zero_reward"],
                    shield=CBFShield(cfg), shield_obs=spec["shield_obs"], stall_penalty=spec["stall_penalty"])
    if holder is not None:
        holder["env"] = env
    model = PPO("MlpPolicy", env, seed=spec["seed"], gamma=GAMMA_S, verbose=0, device=device)
    model.save(OUT / f"s5_init_{arm}.zip")
    model.learn(total_timesteps=TOTAL_LONG, callback=probe.callback)
    result = {
        "arm": arm, "family": spec["family"], "seed": spec["seed"], "settings": spec, "env_steps": int(env.env_steps),
        "decisions": int(model.num_timesteps), "train_violations": int(env.violations_total),
        "probe_violations": int(sum(r.get("violations", 0) for r in probe.rows)), "checkpoints": probe.rows,
        "provenance": provenance(),
    }
    ckpt = d / f"{arm}_{TOTAL_LONG}.zip"
    if not any(r["mark"] == TOTAL_LONG for r in probe.rows) or not ckpt.exists():
        result["verdict"], result["reason"] = "UNDETERMINED", f"final probe at mark {TOTAL_LONG} never fired (env steps {env.env_steps})"
        return result
    logs = evaluate_structured(PPO.load(ckpt, device=device), cfg, GreedyEvader, GATE_N, GATE_SEED0, hold=HOLD,
                               shield_factory=lambda: CBFShield(cfg), shield_obs=spec["shield_obs"])
    result["final"] = {
        "ckpt": str(ckpt), "score": _ci(*win_count(logs, "pursuer")), "violations": int(sum(lg["violations"] for lg in logs)),
        "shield": summarize_shield(logs), "gate": {ref: gate_pass(logs, starts[ref]["logs"], "pursuer", P_GATE) for ref in REFS},
    }
    g = result["final"]["gate"]["TEACHER_SHIELD"]
    result["verdict"] = f"final {'PASS' if g['pass'] else 'FAIL'} vs TEACHER_SHIELD"
    result["reason"] = (f"final {g['k']}/{g['n']} = {g['rate']:.3f}, McNemar p={g['p']:.2e} (b={g['b']}, c={g['c']}) vs shielded teacher "
                        f"{g['ref_rate']:.3f}; " + "; ".join(f"vs {r} {'PASS' if result['final']['gate'][r]['pass'] else 'fail'}" for r in REFS[1:])
                        + f"; violations train {result['train_violations']} probe {result['probe_violations']} gate {result['final']['violations']}")
    return result


def run_arm(arm: str, device: str) -> dict:
    starts = load_start()
    holder: dict = {}
    try:
        result = run_train_arm(arm, starts, device, holder)
    except BaseException as e:
        _write_json(OUT / f"s5_arm_{arm}.json", {
            "arm": arm, "family": ARMS[arm]["family"], "seed": ARMS[arm]["seed"], "verdict": "UNDETERMINED",
            "reason": f"crashed: {type(e).__name__}: {e}", "checkpoints": holder.get("rows", []),
            "train_violations": int(getattr(holder.get("env"), "violations_total", 0)),
            "probe_violations": int(sum(r.get("violations", 0) for r in holder.get("rows", []))), "provenance": provenance()})
        raise
    _write_json(OUT / f"s5_arm_{arm}.json", result)
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# verdicts (spec section 5)
# ----------------------------------------------------------------------------
def _pass(r: dict | None) -> bool | None:
    try:
        return bool(r["final"]["gate"]["TEACHER_SHIELD"]["pass"])
    except (KeyError, TypeError):
        return None


def hypothesis_verdicts(results: dict) -> dict:
    rz = _pass(results.get("RZ19"))
    h3 = {}
    for fam in FAMILIES:
        per = {s: _pass(results.get(f"{fam}{s}")) for s in SEEDS}
        if any(v is None for v in per.values()):
            h3[fam] = "INCOMPLETE"
        elif rz:
            h3[fam] = "UNDETERMINED (harness)"
        else:
            h3[fam] = "PASS" if sum(per.values()) >= PASS_MIN else "FAIL"
    if rz is None or any(v == "INCOMPLETE" for v in h3.values()):
        overall = "INCOMPLETE"
    elif rz:
        overall = "UNDETERMINED (harness)"
    else:
        overall = "PASS" if any(v == "PASS" for v in h3.values()) else "FAIL"
    return {"H3": h3, "H3_overall": overall, "RZ": "INCOMPLETE" if rz is None else ("PASS" if rz else "FAIL"),
            "artefact": choose_artefact(results, h3) if overall == "PASS" else None}


def choose_artefact(results: dict, h3: dict) -> str | None:
    """Spec section 5: among PASSING arms, the passing seed with the highest
    gate rate vs TEACHER_SHIELD; ties -> arm order RC, RA, RB, then lowest seed."""
    cands = []
    for rank, fam in enumerate(ARTEFACT_ORDER):
        if h3.get(fam) != "PASS":
            continue
        for s in SEEDS:
            r = results.get(f"{fam}{s}")
            if r and _pass(r):
                cands.append((-r["final"]["gate"]["TEACHER_SHIELD"]["rate"], rank, s, f"{fam}{s}"))
    return min(cands)[3] if cands else None


def hsafe_verdict(results: dict) -> dict:
    steps = 0
    offenders = []
    missing = [a for a in ARMS if a not in results or "final" not in results[a]]
    for a, r in results.items():
        v = int(r.get("train_violations", 0)) + int(r.get("probe_violations", 0)) + int((r.get("final") or {}).get("violations", 0))
        if "probe_violations" not in r:
            v += int(sum(x.get("violations", 0) for x in r.get("checkpoints", [])))
        steps += v
        if v:
            offenders.append(a)
    if steps:
        verdict = "FAILS"
    elif missing:
        verdict = "INCOMPLETE"
    else:
        verdict = "HOLDS"
    return {"verdict": verdict, "shielded_violation_steps": steps, "offenders": offenders, "missing": missing}


def direction(h: dict) -> str:
    if h["hsafe"] == "FAILS":
        return "H-SAFE FAILS: a shielded arm recorded a contact step. Stop; census; no further round without a user ruling."
    if h["hsafe"] == "INCOMPLETE" or h["H3_overall"] == "INCOMPLETE":
        return "incomplete: all nine remedy arms and the RZ control must run"
    if h["RZ"] == "PASS":
        return "RZ PASSES: the augmented observation with no signal beats the shielded teacher — harness invalid; diagnose."
    if h["H3_overall"] == "PASS":
        return ("H3 PASSES: a safe learned pursuer that beats the shielded teacher exists — R3's first evidence in the "
                "adversarial setting, and the artefact for Step 9 (the attacker faces both STR14-SHIELD and this policy). "
                "Which ingredient did it is the reported reading.")
    return ("H3 FAILS (RZ valid): neither cheap ingredient recovers the loss at this budget; Step 10 needs the learned "
            "feasibility constraint or a search policy that plans around the barrier surfaces explicitly. Step 9 proceeds "
            "against STR14-SHIELD alone.")


def load_results() -> dict:
    return {p.stem.removeprefix("s5_arm_"): json.loads(p.read_text()) for p in sorted(OUT.glob("s5_arm_*.json"))}


def summarize(results: dict, starts: dict) -> dict:
    h = hypothesis_verdicts(results)
    hs = hsafe_verdict(results)
    cost_keys = ("intervention_rate", "infeasibility_rate", "stall_step_rate", "deadlock_episode_rate")
    cost = {a: {k: r["final"]["shield"].get(k) for k in cost_keys} | {"rate": r["final"]["score"]["rate"]}
            for a, r in results.items() if "final" in r}
    ingredient = {fam: {"passes": sum(bool(_pass(results.get(f"{fam}{s}"))) for s in SEEDS),
                        "mean_rate": (sum(results[f"{fam}{s}"]["final"]["score"]["rate"] for s in SEEDS if f"{fam}{s}" in results and "final" in results[f"{fam}{s}"])
                                      / max(1, sum(1 for s in SEEDS if f"{fam}{s}" in results and "final" in results[f"{fam}{s}"])))}
                  for fam in FAMILIES}
    d = direction({"hsafe": hs["verdict"], "H3_overall": h["H3_overall"], "RZ": h["RZ"]})
    return {
        "spec": "docs/pathway/shield_remedy_prereg_2026-09-15.md",
        "hsafe": hs, "hypotheses": h, "bars": {k: starts[k]["rate"] for k in REFS},
        "strsh_reproduced": {k: starts[k].get("reproduced") for k in ("STRSH16", "STRSH17", "STRSH18")},
        "arms": {a: {"verdict": r.get("verdict"), "reason": r.get("reason")} for a, r in results.items()},
        "cost": cost, "ingredient": ingredient,
        "complete": all(a in results and "final" in results[a] for a in ARMS),
        "direction": d,
    }


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    col = {"RA": "tab:blue", "RB": "tab:orange", "RC": "tab:green", "RZ": "tab:red"}
    for a, r in results.items():
        rows = r.get("checkpoints") or []
        if rows:
            ax.plot([x["steps"] for x in rows], [x["win_rate"] for x in rows], marker=".", color=col[r["family"]], alpha=0.8, label=f"{a} ({summary['arms'][a]['verdict']})")
    for ref, c in (("TEACHER_SHIELD", "k"), ("STR14_SHIELD", "tab:purple"), ("TEACHER_NONE", "tab:gray")):
        ax.axhline(starts[ref]["rate"], ls=":", color=c, lw=1.2, label=f"{ref} = {starts[ref]['rate']:.3f}")
    for s in ("STRSH16", "STRSH17", "STRSH18"):
        ax.axhline(starts[s]["rate"], ls="--", color="tab:brown", lw=0.6, label=f"{s} = {starts[s]['rate']:.3f}")
    ax.set_xlabel("env steps"); ax.set_ylabel(f"shielded probe capture rate (seeds {PROBE_SEED0}+)")
    ax.set_title(f"S5 — remedy arms (H3: {summary['hypotheses']['H3_overall']})"); ax.legend(fontsize=5, ncol=2)
    ax = axes[1]
    names = [a for a in results if "final" in results[a]]
    if names:
        xs = range(len(names))
        ax.bar([x - 0.2 for x in xs], [summary["cost"][a]["stall_step_rate"] for a in names], width=0.2, label="stall rate")
        ax.bar(xs, [summary["cost"][a]["deadlock_episode_rate"] for a in names], width=0.2, label="deadlock episode rate")
        ax.bar([x + 0.2 for x in xs], [summary["cost"][a]["rate"] for a in names], width=0.2, label="gate capture rate")
        ax.set_xticks(list(xs)); ax.set_xticklabels(names, fontsize=7, rotation=45)
        ax.set_title("final checkpoints: cost vs capture (shielded, gate stream)"); ax.legend(fontsize=7)
    fig.suptitle("S5 — shield-aware remedy" + ("" if summary["complete"] else " — INCOMPLETE"))
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.measure_start:
        measure_start(args.device)
        return
    for arm in args.arms:
        run_arm(arm, args.device)
    if args.summarize:
        results, starts = load_results(), load_start()
        summary = summarize(results, starts)
        _write_json(OUT / "s5_round.json", {"summary": summary, "arms": results})
        write_figure(results, starts, summary, OUT / "s5_round.png")
        print(json.dumps({k: summary[k] for k in ("hsafe", "hypotheses", "ingredient", "strsh_reproduced")}, indent=2, default=str))
        print(json.dumps(summary["arms"], indent=2))
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s5_round.py -q`
Expected: all PASS.

- [ ] **Step 5: Smoke the wiring at a tiny budget in a scratch directory**

Throwaway script under `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/ff03fc68-2116-4512-a227-312f3099d143/scratchpad/s5smoke/`: import via `importlib`, monkeypatch `OUT`, `START_PATH` to the scratch dir and `TOTAL_LONG=4000, EVERY_LONG=2000, EVAL_N=4, GATE_N=6`; ALSO monkeypatch `S4_CELL` to point at copies of the real S4 cell files truncated to their first 6 logs with `seed0` kept (write the truncated copies into the scratch dir) and `STRSH_EXPECTED` to whatever the 6-episode re-score gives (or simply assert the `reproduced` flag is a bool). Then `measure_start("cpu")`, `run_arm("RA19","cpu")`, `run_arm("RB19","cpu")`, `run_arm("RZ19","cpu")`, `summarize(load_results(), load_start())`, `write_figure(...)`. Expected: no exception; RA/RZ arms' probes and gate run with 14-dim observations, RB with 11-dim; every arm has `train_violations == 0`; the PNG exists. Paste the console output into your report. Nothing under the real `results/derisk/` may change.

- [ ] **Step 6: Lint, full suite, commit**

```bash
./venv/bin/ruff format scripts/derisk_s5_remedy.py tests/pe2/test_derisk_s5_round.py && ./venv/bin/ruff check scripts/derisk_s5_remedy.py tests/pe2/test_derisk_s5_round.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q -x
git add scripts/derisk_s5_remedy.py tests/pe2/test_derisk_s5_round.py
git commit -m "feat(derisk): S5 runner — shield-aware remedy arms RA/RB/RC + RZ control, S4-paired gate, H3/artefact/cost" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** §2 ingredients → Task 1 (`shield_obs`, `stall_penalty`, `evaluate_structured(shield_obs=)`); §3 config, streams, references incl. STRSH re-score with reproduction flag → `measure_start`, `run_train_arm`; §4 arms → `ARMS`; §5 verdicts (per-arm ≥ 2/3, overall, RZ harness, INCOMPLETE, artefact order RC/RA/RB then lowest seed, cost and ingredient readings) → `hypothesis_verdicts`, `choose_artefact`, `hsafe_verdict`, `summarize`; §6 direction → `direction`; stop rule → guards; §8 artefacts → paths + figure.
- **Placeholders:** none.
- **Type consistency:** `evaluate_structured(..., shield_factory=, shield_obs=)` used identically; arm records carry `final.gate[REF].{pass,rate,...}`, `final.score.rate`, `final.shield`, `train_violations`, `probe_violations`, `checkpoints` — the tests' `_arm` fixture mirrors that shape; `_pass` reads `TEACHER_SHIELD`.
