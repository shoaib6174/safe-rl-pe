# Structured Confirmatory Round (S3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument `SearchMDP` so a learned search policy can be characterised, and build the S3 runner (gate bars → five fresh-seed structured arms + zero-reward control + SBLONG gate → verdicts, artefact rule, characterisation, opponent-family table, figure).

**Architecture:** `pe2/structured.py` gains an optional per-step trace and a `characterise()` function computing the behavioural statistics the spec §5 names, for any search policy (the teacher is characterised through `OrbitSearch` at hold 1, which V1 proved identical to `SearchingPursuer`). `scripts/derisk_s3_confirm.py` is the runner, modelled on `scripts/derisk_s2_cp_round.py` (its `run_sblong`, `provenance`, guards, crash record) and `scripts/derisk_s1_structured.py` (its structured-arm loop); it reuses `pe2.derisk.gate_pass` for every gate and `evaluate_structured` / `evaluate_scripted` / `evaluate_sb3_policy` for scoring.

**Tech Stack:** Python 3.12, `./venv/bin/python`, stable-baselines3 2.7 (PPO), gymnasium, numpy, matplotlib, pytest (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), ruff.

**Spec:** `docs/pathway/structured_confirmatory_prereg_2026-09-14.md` (binding; read it first).

## Global Constraints

- Always `./venv/bin/python`; tests `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest <files> -q`; lint `./venv/bin/ruff format <files> && ./venv/bin/ruff check <files>` clean before each commit.
- Pre-registered constants, never tunable: `TOTAL_LONG = 10_000_000`, `EVERY_LONG = 1_000_000`, `EVAL_N = 200`, `GATE_N = 500`, `PROBE_SEED0 = 1_060_000`, `GATE_SEED0 = 1_070_000`, `FAMILY_SEED0 = 1_080_000`, `FAMILY_N = 200`, `SEEDS = (11, 12, 13, 14, 15)`, `PASS_MIN = 4`, `P_GATE = 0.05`, `GAMMA_S = 0.99`, `HOLD` (library, 10), `r_acq = 0.0`.
- Every training arm = SBLONG's configuration exactly: `SearchMDP(cfg, GreedyEvader, seed0=seed*1000, hold=HOLD, r_acq=0.0, zero_reward=<arm>)`, `PPO("MlpPolicy", env, seed=seed, gamma=0.99, verbose=0, device=device)` with SB3 defaults otherwise, init saved as `s3_init_{seed}.zip` before `learn`, budget enforced in env steps by `StructuredEvalProbe`, no selection — the final checkpoint `{ARM}_{TOTAL_LONG}.zip` is the policy.
- Gate rule = `pe2.derisk.gate_pass` (McNemar p < 0.05 with b > c AND Wilson-lo > reference point), paired on `GATE_SEED0`; verdict bar = `TEACHER`.
- H-STR passes iff ≥ `PASS_MIN` of the five fresh seeds' finals pass vs `TEACHER`; `SZLONG11` passing ⇒ `UNDETERMINED (harness)`; a missing seed record ⇒ `INCOMPLETE`; artefact = passing fresh seed with the highest gate rate vs `TEACHER`, ties → lowest seed; SBLONG reported, never counted or eligible.
- Never run `--measure-start` or any arm; tests and smoke runs redirect every output path into `tmp_path`/scratch; the real `results/derisk/` (S1/S2/R1 artefacts, clone zips, `ckpts_SBLONG`, `ckpts_SEL7`) is read-only for this plan.
- Commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; lazy SB3/torch imports inside functions; docstrings say why with decision numbers (D109, D110).

---

### Task 1: `SearchMDP` trace + `characterise()`

**Files:**
- Modify: `pe2/structured.py` (`SearchMDP._step_inner`, `_advance`, `step`, `_begin`; append `characterise`)
- Test: `tests/pe2/test_structured.py` (append)

**Interfaces:**
- Consumes: existing `SearchMDP`, `OrbitSearch`, `evaluate_structured`, `HOLD`, `_normalize`.
- Produces:
  - `SearchMDP.trace: list[tuple[bool, str, float, float]] | None` — when a list is assigned, every env step appends `(visible_after_step, branch, v_action, w_action)` with `branch ∈ {"law", "search"}` and the actions as the normalised Box values; `_begin` does NOT clear it (callers own it).
  - `characterise(predict, cfg, opponent_factory, n_episodes, seed0, hold=HOLD) -> dict` with keys `visible_frac, first_sight_median, first_sight_p90, found_frac, stretch_median, stretch_p90, search_frac, search_speed_mean, search_abs_omega_mean, capture_rate, n_episodes` (speeds in m/s and rad/s: `v = (a+1)/2*cfg.v_max`, `omega = a*cfg.omega_max`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/pe2/test_structured.py`:

```python
class TestTraceAndCharacterise:
    def test_trace_records_every_env_step_with_its_branch(self):
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader, hold=10)
        m.trace = []
        log = m.play_episode(OrbitSearch(CFG).predict, 900_900)
        assert len(m.trace) == log["steps"]
        branches = {b for _, b, _, _ in m.trace}
        assert branches <= {"law", "search"} and "search" in branches
        vis = [v for v, _, _, _ in m.trace]
        # every recorded "law" step follows a visible or last-seen state; the
        # count of visible->invisible transitions must equal the log's los_breaks
        breaks = sum(1 for a, b in zip(vis, vis[1:], strict=False) if a and not b)
        assert breaks == log["los_breaks"]
        for _, _, v, w in m.trace:
            assert -1.0 <= v <= 1.0 and -1.0 <= w <= 1.0

    def test_trace_is_off_by_default(self):
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader)
        assert m.trace is None
        m.play_episode(OrbitSearch(CFG).predict, 900_901)
        assert m.trace is None

    def test_characterise_orbit_matches_the_design_numbers_in_shape(self):
        from pe2.structured import characterise

        c = characterise(OrbitSearch(CFG).predict, CFG, GreedyEvader, 12, 900_910, hold=1)
        assert c["n_episodes"] == 12
        assert 0.0 < c["visible_frac"] < 1.0
        assert 0.0 < c["search_frac"] < 1.0
        assert c["first_sight_median"] <= c["first_sight_p90"]
        assert c["stretch_median"] <= c["stretch_p90"]
        assert c["search_speed_mean"] == pytest.approx(CFG.v_max, abs=1e-6)  # the orbit drives full speed
        assert 0.0 <= c["search_abs_omega_mean"] <= CFG.omega_max
        assert 0.0 <= c["capture_rate"] <= 1.0 and 0.0 <= c["found_frac"] <= 1.0

    def test_characterise_stand_still_has_zero_search_speed(self):
        from pe2.structured import characterise

        class Still:
            def predict(self, obs, deterministic=True):
                return np.array([-1.0, 0.5], dtype=np.float32), None

        c = characterise(Still().predict, CFG, GreedyEvader, 4, 900_920, hold=10)
        assert c["search_speed_mean"] == pytest.approx(0.0, abs=1e-6)
        assert c["search_abs_omega_mean"] == pytest.approx(0.5 * CFG.omega_max, abs=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_structured.py -k "TraceAndCharacterise" -q`
Expected: FAIL — `AttributeError: trace` / `ImportError: characterise`.

- [ ] **Step 3: Implement**

In `SearchMDP.__init__` add `self.trace: list | None = None` (after `self._info = {}`), with the comment `# optional per-step record (visible, branch, v, w) for characterise(); off by default`.

Change `_step_inner` to take the branch and record:

```python
    def _step_inner(self, action: np.ndarray, branch: str = "law"):
        prev = self._visible
        obs, r, term, trunc, info = self.inner.step(action)
        self.env_steps += 1
        self._visible = bool(obs[6] > 0.5)
        self._los_breaks += int(prev and not self._visible)
        self._info = info
        if self.trace is not None:
            self.trace.append((self._visible, branch, float(action[0]), float(action[1])))
        onset = int((not prev) and self._visible)
        return obs, float(r), onset, term, trunc
```

In `_advance` keep `self._step_inner(a)` (branch defaults to `"law"`). In `step`, the held loop becomes `obs, r, onset, term, trunc = self._step_inner(a, "search")`.

Append to `pe2/structured.py`:

```python
def characterise(
    predict,
    cfg: EnvConfig,
    opponent_factory: Callable[[], object],
    n_episodes: int,
    seed0: int,
    hold: int = HOLD,
) -> dict:
    """Behavioural statistics of a search policy inside SearchMDP (spec S3
    section 5, reported only): what the learned search actually does, on the
    same footing as the teacher (characterise OrbitSearch at hold=1 for the
    teacher — V1 proved that identical to SearchingPursuer). Speeds are
    physical: v = (a+1)/2 * v_max, omega = a * omega_max."""
    mdp = SearchMDP(cfg, opponent_factory, hold=hold)
    first_sight: list[int] = []
    stretches: list[int] = []
    n_vis = n_search = n_steps = captures = 0
    speed_sum = omega_sum = 0.0
    for i in range(n_episodes):
        mdp.trace = []
        log = mdp.play_episode(predict, seed0 + i)
        captures += log["outcome"] == "capture"
        cur = 0
        seen = False
        for t, (vis, branch, v, w) in enumerate(mdp.trace, start=1):
            n_steps += 1
            n_vis += vis
            if vis and not seen:
                seen = True
                first_sight.append(t)
            if branch == "search":
                n_search += 1
                cur += 1
                speed_sum += (v + 1.0) / 2.0 * cfg.v_max
                omega_sum += abs(w) * cfg.omega_max
            elif cur:
                stretches.append(cur)
                cur = 0
        if cur:
            stretches.append(cur)
    mdp.trace = None

    def _q(xs: list[int], q: float) -> float:
        return float(np.percentile(xs, q)) if xs else float("nan")

    return {
        "n_episodes": n_episodes,
        "capture_rate": captures / n_episodes,
        "visible_frac": n_vis / n_steps if n_steps else float("nan"),
        "found_frac": len(first_sight) / n_episodes,
        "first_sight_median": _q(first_sight, 50),
        "first_sight_p90": _q(first_sight, 90),
        "stretch_median": _q(stretches, 50),
        "stretch_p90": _q(stretches, 90),
        "search_frac": n_search / n_steps if n_steps else float("nan"),
        "search_speed_mean": speed_sum / n_search if n_search else float("nan"),
        "search_abs_omega_mean": omega_sum / n_search if n_search else float("nan"),
    }
```

- [ ] **Step 4: Run the tests** — the whole file, since `_step_inner`'s signature changed:

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_structured.py -q`
Expected: all PASS (the V1 test in particular must still pass — the trace never changes control flow).

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/structured.py tests/pe2/test_structured.py && ./venv/bin/ruff check pe2/structured.py tests/pe2/test_structured.py
git add pe2/structured.py tests/pe2/test_structured.py
git commit -m "feat(pe2): SearchMDP trace + characterise() for the S3 behavioural table" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `scripts/derisk_s3_confirm.py` — bars, seven arms, verdicts, artefact, characterisation, family, figure

**Files:**
- Create: `scripts/derisk_s3_confirm.py`
- Test: `tests/pe2/test_derisk_s3_round.py`

**Interfaces:**
- Consumes: `pe2.structured` (`SearchMDP`, `StructuredEvalProbe`, `evaluate_structured`, `evaluate_scripted`, `OrbitSearch`, `characterise`, `HOLD`); `pe2.derisk.gate_pass`; `pe2.exploitability.evaluate_sb3_policy`; `pe2.eval.wilson_ci, win_count, mcnemar_pvalue`; `pe2.scripted` (`GreedyEvader`, `FovLimitedEvader`, `ObstacleHuggingEvader`, `OpenSpaceEvader`, `SearchingPursuer`, `GreedyPursuer`).
- Produces: CLI `--measure-start | --arms <ARM...> | --summarize [--device cpu]`; module-level `ARMS`, `hypothesis_verdict`, `choose_artefact`, `direction`, `summarize`, `provenance`; artefacts per spec §8.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_derisk_s3_round.py
"""scripts/derisk_s3_confirm.py — the confirmatory round's own decisions."""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s3_confirm.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s3_confirm", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _arm(name, seed, teacher_pass, rate, kind="train"):
    return {
        "arm": name, "kind": kind, "seed": seed,
        "final": {"score": {"rate": rate, "k": int(rate * 500), "n": 500},
                  "gate": {"TEACHER": {"pass": teacher_pass, "rate": rate}, "PPO_CLONE": {"pass": teacher_pass, "rate": rate},
                           "SAC_CLONE": {"pass": False, "rate": rate}, "SEL7_100K": {"pass": False, "rate": rate},
                           "GREEDY_FULLSTATE": {"pass": False, "rate": rate}}},
        "checkpoints": [],
    }


def _full(passes, sz_pass=False, rates=(0.66, 0.70, 0.62, 0.68, 0.64)):
    res = {f"STR{s}": _arm(f"STR{s}", s, p, r) for s, p, r in zip(R.SEEDS, passes, rates, strict=True)}
    res["SZLONG11"] = _arm("SZLONG11", 11, sz_pass, 0.30, kind="control")
    res["SBLONG"] = _arm("SBLONG", 7, True, 0.69, kind="existing")
    return res


class TestProtocol:
    def test_pre_registered_constants(self):
        from pe2.structured import HOLD

        assert (R.TOTAL_LONG, R.EVERY_LONG, R.EVAL_N, R.GATE_N) == (10_000_000, 1_000_000, 200, 500)
        assert (R.PROBE_SEED0, R.GATE_SEED0, R.FAMILY_SEED0, R.FAMILY_N) == (1_060_000, 1_070_000, 1_080_000, 200)
        assert R.SEEDS == (11, 12, 13, 14, 15) and R.PASS_MIN == 4 and R.P_GATE == 0.05
        assert R.GAMMA_S == 0.99 and R.HOLD == HOLD == 10
        assert R.REFS == ("TEACHER", "PPO_CLONE", "SAC_CLONE", "SEL7_100K", "GREEDY_FULLSTATE")

    def test_arm_table(self):
        assert set(R.ARMS) == {"STR11", "STR12", "STR13", "STR14", "STR15", "SZLONG11", "SBLONG"}
        assert R.ARMS["STR13"] == {"kind": "train", "seed": 13, "zero_reward": False}
        assert R.ARMS["SZLONG11"] == {"kind": "control", "seed": 11, "zero_reward": True}
        assert R.ARMS["SBLONG"] == {"kind": "existing", "seed": 7, "ckpt": "results/derisk/ckpts_SBLONG/SBLONG_10000000.zip"}

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s3_start.json")
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()
        (tmp_path / "s3_start.json").write_text("{}")
        with pytest.raises(SystemExit, match="already"):
            R.require_not_measured()
        (tmp_path / "s3_arm_STR12.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("STR12")
        R.require_arm_not_run("STR13")


class TestVerdicts:
    def test_pass_needs_four_of_five(self):
        h = R.hypothesis_verdict(_full([True, True, True, True, False]))
        assert h["H_STR"] == "PASS" and h["passes"] == 4
        h = R.hypothesis_verdict(_full([True, True, True, False, False]))
        assert h["H_STR"] == "FAIL" and h["passes"] == 3

    def test_control_passing_invalidates(self):
        h = R.hypothesis_verdict(_full([True] * 5, sz_pass=True))
        assert h["H_STR"].startswith("UNDETERMINED") and h["harness"] == "INVALID"

    def test_missing_seed_is_incomplete(self):
        res = _full([True] * 5)
        del res["STR14"]
        assert R.hypothesis_verdict(res)["H_STR"] == "INCOMPLETE"
        res = _full([True] * 5)
        del res["SZLONG11"]
        assert R.hypothesis_verdict(res)["H_STR"] == "INCOMPLETE"

    def test_artefact_is_highest_passing_fresh_seed_ties_to_lowest(self):
        res = _full([True, True, True, True, False], rates=(0.66, 0.70, 0.70, 0.68, 0.99))
        h = R.hypothesis_verdict(res)
        assert h["artefact"] == "STR12"  # 0.70 tie between 12 and 13 -> lowest seed; STR15 fails so 0.99 ignored
        res = _full([True, True, True, False, False])
        assert R.hypothesis_verdict(res)["artefact"] is None  # no artefact without a pass

    def test_sblong_is_never_the_artefact(self):
        res = _full([True] * 5, rates=(0.60, 0.61, 0.62, 0.63, 0.64))
        assert R.hypothesis_verdict(res)["artefact"] == "STR15"


class TestDirection:
    def test_pass(self):
        assert "paper's learned pursuer" in R.direction({"H_STR": "PASS", "harness": "VALID"})

    def test_fail(self):
        assert "seed-specific" in R.direction({"H_STR": "FAIL", "harness": "VALID"})

    def test_invalid(self):
        assert "harness" in R.direction({"H_STR": "UNDETERMINED (harness)", "harness": "INVALID"})

    def test_incomplete(self):
        assert "incomplete" in R.direction({"H_STR": "INCOMPLETE", "harness": "VALID"})


class TestFamilyTable:
    def test_rows_have_ci_and_mcnemar(self):
        logs_a = [{"seed": i, "role": "pursuer", "outcome": "capture" if i < 120 else "timeout"} for i in range(200)]
        logs_t = [{"seed": i, "role": "pursuer", "outcome": "capture" if i < 100 else "timeout"} for i in range(200)]
        row = R.family_row(logs_a, logs_t)
        assert row["rate"] == 0.6 and row["teacher_rate"] == 0.5 and row["b"] == 20 and row["c"] == 0
        assert row["wilson"][0] < 0.6 < row["wilson"][1] and 0.0 <= row["p"] <= 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s3_round.py -q`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Write the runner**

```python
# scripts/derisk_s3_confirm.py
"""S3 — confirmatory round for the structured learner (spec:
docs/pathway/structured_confirmatory_prereg_2026-09-14.md). Nothing here may
change a threshold, a seed, a stream, a rule or a budget after a number is seen.

Arms:
  STR11..STR15  SBLONG's configuration, five fresh seeds, 10M env steps  <- the test
  SZLONG11      same, every reward zeroed                                  <- negative control
  SBLONG        the existing final checkpoint, gated on the new stream     <- reported, not counted

Usage:
  ./venv/bin/python scripts/derisk_s3_confirm.py --measure-start
  ./venv/bin/python scripts/derisk_s3_confirm.py --arms STR11        (one process per arm)
  ./venv/bin/python scripts/derisk_s3_confirm.py --summarize
"""

from __future__ import annotations

import argparse
import json
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import gate_pass
from pe2.eval import mcnemar_pvalue, wilson_ci, win_count
from pe2.scripted import (
    FovLimitedEvader,
    GreedyEvader,
    GreedyPursuer,
    ObstacleHuggingEvader,
    OpenSpaceEvader,
    SearchingPursuer,
)
from pe2.structured import HOLD

OUT = pathlib.Path("results/derisk")
START_PATH = OUT / "s3_start.json"
PPO_CLONE = OUT / "ppo_bc_searching_pursuer.zip"
SAC_CLONE = OUT / "sac_bc_searching_pursuer.zip"
SEL7_100K = OUT / "ckpts_SEL7" / "SEL7_100000.zip"
SBLONG_CKPT = OUT / "ckpts_SBLONG" / "SBLONG_10000000.zip"

# --- pre-registered (spec sections 2-5). Do not tune. ---
TOTAL_LONG = 10_000_000
EVERY_LONG = 1_000_000
EVAL_N = 200
GATE_N = 500
PROBE_SEED0 = 1_060_000
GATE_SEED0 = 1_070_000
FAMILY_SEED0 = 1_080_000
FAMILY_N = 200
SEEDS = (11, 12, 13, 14, 15)
PASS_MIN = 4
P_GATE = 0.05
GAMMA_S = 0.99
REFS = ("TEACHER", "PPO_CLONE", "SAC_CLONE", "SEL7_100K", "GREEDY_FULLSTATE")
FAMILY = {
    "GreedyEvader": GreedyEvader,
    "FovLimitedEvader": FovLimitedEvader,
    "ObstacleHuggingEvader": ObstacleHuggingEvader,
    "OpenSpaceEvader": OpenSpaceEvader,
}

ARMS: dict = {f"STR{s}": {"kind": "train", "seed": s, "zero_reward": False} for s in SEEDS}
ARMS["SZLONG11"] = {"kind": "control", "seed": 11, "zero_reward": True}
ARMS["SBLONG"] = {"kind": "existing", "seed": 7, "ckpt": str(SBLONG_CKPT)}


def _ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


def provenance() -> dict:
    import os
    import subprocess

    import stable_baselines3
    import torch

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    return {
        "git_commit": sha,
        "sb3_version": stable_baselines3.__version__,
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


# ----------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------
def require_not_measured() -> None:
    if START_PATH.exists():
        raise SystemExit(f"{START_PATH} already exists: the bars are already measured (stop rule, spec section 6).")
    if any(OUT.glob("s3_arm_*.json")) or any((OUT / f"ckpts_{a}").exists() for a in ARMS if ARMS[a]["kind"] != "existing"):
        raise SystemExit("S3 arms have already run; the stop rule forbids re-measuring.")


def require_arm_not_run(arm: str) -> None:
    if (OUT / f"s3_arm_{arm}.json").exists() or (ARMS[arm]["kind"] != "existing" and (OUT / f"ckpts_{arm}").exists()):
        raise SystemExit(f"arm {arm} has already run; the stop rule (spec section 6) forbids re-running an arm.")


def load_start() -> dict:
    if not START_PATH.exists():
        raise SystemExit(f"gate bars missing: {START_PATH}. Run --measure-start FIRST.")
    s = json.loads(START_PATH.read_text())
    for k in REFS:
        if k not in s or "logs" not in s[k]:
            raise SystemExit(f"{k} (with paired logs) missing from {START_PATH}; re-run --measure-start.")
    return s


# ----------------------------------------------------------------------------
# bars (spec section 2), BEFORE any arm
# ----------------------------------------------------------------------------
def measure_start(device: str) -> dict:
    from stable_baselines3 import PPO, SAC

    from pe2.exploitability import evaluate_sb3_policy
    from pe2.structured import evaluate_scripted

    require_not_measured()
    cfg = EnvConfig()
    refs = {
        "TEACHER": lambda: evaluate_scripted(cfg, SearchingPursuer, GreedyEvader, GATE_N, GATE_SEED0),
        "PPO_CLONE": lambda: evaluate_sb3_policy(PPO.load(PPO_CLONE, device=device), cfg, "pursuer", GreedyEvader, GATE_N, GATE_SEED0),
        "SAC_CLONE": lambda: evaluate_sb3_policy(SAC.load(SAC_CLONE, device=device), cfg, "pursuer", GreedyEvader, GATE_N, GATE_SEED0),
        "SEL7_100K": lambda: evaluate_sb3_policy(PPO.load(SEL7_100K, device=device), cfg, "pursuer", GreedyEvader, GATE_N, GATE_SEED0),
        "GREEDY_FULLSTATE": lambda: evaluate_scripted(cfg, GreedyPursuer, GreedyEvader, GATE_N, GATE_SEED0),
    }
    out: dict = {"gate_seed0": GATE_SEED0, "gate_n": GATE_N, "provenance": provenance()}
    for name, fn in refs.items():
        logs = fn()
        out[name] = _ci(*win_count(logs, "pursuer")) | {"logs": logs}
        lo, hi = out[name]["wilson"]
        print(f"{name} = {out[name]['k']}/{GATE_N} = {out[name]['rate']:.4f}, Wilson [{lo:.4f}, {hi:.4f}]", flush=True)
    START_PATH.write_text(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------------
def gate_final(ckpt, starts: dict, cfg, device: str) -> dict:
    from stable_baselines3 import PPO

    from pe2.structured import evaluate_structured

    model = PPO.load(ckpt, device=device)
    logs = evaluate_structured(model, cfg, GreedyEvader, GATE_N, GATE_SEED0, hold=HOLD)
    return {
        "ckpt": str(ckpt),
        "score": _ci(*win_count(logs, "pursuer")),
        "gate": {ref: gate_pass(logs, starts[ref]["logs"], "pursuer", P_GATE) for ref in REFS},
    }


def run_train_arm(arm: str, starts: dict, device: str, holder: dict | None = None) -> dict:
    from stable_baselines3 import PPO

    from pe2.structured import SearchMDP, StructuredEvalProbe

    spec = ARMS[arm]
    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = StructuredEvalProbe(
        cfg, GreedyEvader, every=EVERY_LONG, total=TOTAL_LONG, eval_n=EVAL_N, seed0=PROBE_SEED0,
        ckpt_dir=d, report_path=OUT / f"s3_probe_{arm}.json", label=arm, hold=HOLD,
    )
    if holder is not None:
        holder["rows"] = probe.rows
    env = SearchMDP(cfg, GreedyEvader, seed0=spec["seed"] * 1000, hold=HOLD, r_acq=0.0, zero_reward=spec["zero_reward"])
    model = PPO("MlpPolicy", env, seed=spec["seed"], gamma=GAMMA_S, verbose=0, device=device)
    init_path = OUT / f"s3_init_{spec['seed']}.zip"
    if not init_path.exists():
        model.save(init_path)  # STR11 and SZLONG11 share seed 11: literally the same init
    model.learn(total_timesteps=TOTAL_LONG, callback=probe.callback)
    result = {
        "arm": arm, "kind": spec["kind"], "seed": spec["seed"], "settings": spec, "init": str(init_path),
        "env_steps": int(env.env_steps), "decisions": int(model.num_timesteps),
        "probe_seed0": PROBE_SEED0, "gate_seed0": GATE_SEED0, "checkpoints": probe.rows, "provenance": provenance(),
    }
    ckpt = d / f"{arm}_{TOTAL_LONG}.zip"
    if not any(r["mark"] == TOTAL_LONG for r in probe.rows) or not ckpt.exists():
        result["verdict"] = "UNDETERMINED"
        result["reason"] = f"final probe at mark {TOTAL_LONG} never fired (env steps reached {env.env_steps})"
        return result
    result["final"] = gate_final(ckpt, starts, cfg, device)
    t = result["final"]["gate"]["TEACHER"]
    result["verdict"] = f"final {'PASS' if t['pass'] else 'FAIL'} vs TEACHER"
    result["reason"] = (
        f"final {t['k']}/{t['n']} = {t['rate']:.3f}, McNemar p={t['p']:.2e} (b={t['b']}, c={t['c']}) vs teacher {t['ref_rate']:.3f}; "
        + "; ".join(f"vs {r} {'PASS' if result['final']['gate'][r]['pass'] else 'fail'}" for r in REFS[1:])
    )
    return result


def run_existing_arm(arm: str, starts: dict, device: str) -> dict:
    spec = ARMS[arm]
    ckpt = pathlib.Path(spec["ckpt"])
    if not ckpt.exists():
        raise SystemExit(f"{arm}: checkpoint missing: {ckpt}")
    cfg = EnvConfig()
    result = {"arm": arm, "kind": spec["kind"], "seed": spec["seed"], "settings": spec, "gate_seed0": GATE_SEED0,
              "checkpoints": [], "provenance": provenance(), "final": gate_final(ckpt, starts, cfg, device)}
    t = result["final"]["gate"]["TEACHER"]
    result["verdict"] = f"final {'PASS' if t['pass'] else 'FAIL'} vs TEACHER (reported, not counted)"
    result["reason"] = f"final {t['k']}/{t['n']} = {t['rate']:.3f}, McNemar p={t['p']:.2e} (b={t['b']}, c={t['c']}) vs teacher {t['ref_rate']:.3f}"
    return result


def run_arm(arm: str, device: str) -> dict:
    starts = load_start()
    require_arm_not_run(arm)
    holder: dict = {}
    try:
        result = run_existing_arm(arm, starts, device) if ARMS[arm]["kind"] == "existing" else run_train_arm(arm, starts, device, holder)
    except Exception as e:
        (OUT / f"s3_arm_{arm}.json").write_text(json.dumps({
            "arm": arm, "kind": ARMS[arm]["kind"], "seed": ARMS[arm]["seed"], "verdict": "UNDETERMINED",
            "reason": f"crashed: {type(e).__name__}: {e}", "checkpoints": holder.get("rows", []), "provenance": provenance(),
        }, indent=2))
        raise
    (OUT / f"s3_arm_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# verdicts (spec section 4), artefact rule, direction (section 6)
# ----------------------------------------------------------------------------
def _teacher_pass(r: dict) -> bool | None:
    try:
        return bool(r["final"]["gate"]["TEACHER"]["pass"])
    except (KeyError, TypeError):
        return None


def hypothesis_verdict(results: dict) -> dict:
    fresh = {f"STR{s}": results.get(f"STR{s}") for s in SEEDS}
    passes = {a: _teacher_pass(r) if r else None for a, r in fresh.items()}
    ctrl = results.get("SZLONG11")
    ctrl_pass = _teacher_pass(ctrl) if ctrl else None
    if any(v is None for v in passes.values()) or ctrl_pass is None:
        return {"H_STR": "INCOMPLETE", "passes": sum(bool(v) for v in passes.values()), "harness": "INCOMPLETE", "artefact": None, "per_seed": passes}
    harness = "INVALID" if ctrl_pass else "VALID"
    n = sum(passes.values())
    if harness == "INVALID":
        h = "UNDETERMINED (harness)"
    else:
        h = "PASS" if n >= PASS_MIN else "FAIL"
    artefact = choose_artefact(results) if h == "PASS" else None
    return {"H_STR": h, "passes": n, "harness": harness, "artefact": artefact, "per_seed": passes}


def choose_artefact(results: dict) -> str | None:
    """Spec section 4: the passing FRESH seed with the highest gate rate vs
    TEACHER; ties -> lowest seed. SBLONG is never eligible."""
    cands = []
    for s in SEEDS:
        r = results.get(f"STR{s}")
        if r and _teacher_pass(r):
            cands.append((-r["final"]["gate"]["TEACHER"]["rate"], s))
    if not cands:
        return None
    return f"STR{min(cands)[1]}"


def direction(h: dict) -> str:
    if h["H_STR"] == "INCOMPLETE":
        return "incomplete: all five fresh seeds and the SZLONG11 control must run"
    if h["harness"] == "INVALID":
        return ("SZLONG11 PASSES vs the teacher: the harness is invalid at this budget (unrewarded drift beats the "
                "teacher inside the fixed law); nothing is read; diagnose before any further round.")
    if h["H_STR"] == "PASS":
        return (f"H-STR PASSES ({h['passes']}/5): the structured learner is the paper's learned pursuer — improves on its "
                f"scripted teacher without selection, anchoring, or the possibility of destroying the capture law. Artefact: "
                f"{h['artefact']}. Next pre-registration: the shield on the artefact policy (Step 8).")
    return (f"H-STR FAILS ({h['passes']}/5, control valid): SBLONG's 0.68 was seed-specific; report the seed distribution; "
            "the selection-based PPO artefact (D110 H-SEL) is the fallback pursuer for the shield round.")


# ----------------------------------------------------------------------------
# characterisation + family (spec section 5), reported only
# ----------------------------------------------------------------------------
def family_row(logs_policy: list[dict], logs_teacher: list[dict]) -> dict:
    k, n = win_count(logs_policy, "pursuer")
    kt, nt = win_count(logs_teacher, "pursuer")
    p, b, c = mcnemar_pvalue(logs_policy, logs_teacher, "pursuer")
    return _ci(k, n) | {"teacher_rate": kt / nt, "p": p, "b": b, "c": c}


def characterise_and_family(results: dict, h: dict, device: str) -> dict:
    from stable_baselines3 import PPO

    from pe2.structured import OrbitSearch, characterise, evaluate_scripted, evaluate_structured

    cfg = EnvConfig()
    policies = {"TEACHER": (OrbitSearch(cfg).predict, 1)}
    if "SBLONG" in results:
        policies["SBLONG"] = (PPO.load(SBLONG_CKPT, device=device).predict, HOLD)
    if h.get("artefact"):
        policies[h["artefact"]] = (PPO.load(results[h["artefact"]]["final"]["ckpt"], device=device).predict, HOLD)
    out = {"characterisation": {}, "family": {}}
    for name, (predict, hold) in policies.items():
        out["characterisation"][name] = characterise(predict, cfg, GreedyEvader, GATE_N, GATE_SEED0, hold=hold)
    for ev_name, ev in FAMILY.items():
        teacher_logs = evaluate_scripted(cfg, SearchingPursuer, ev, FAMILY_N, FAMILY_SEED0)
        out["family"][ev_name] = {"TEACHER": _ci(*win_count(teacher_logs, "pursuer"))}
        for name, (predict, hold) in policies.items():
            if name == "TEACHER":
                continue
            class _P:  # predict shim for evaluate_structured
                def __init__(self, f): self.predict = f
            logs = evaluate_structured(_P(predict), cfg, ev, FAMILY_N, FAMILY_SEED0, hold=hold)
            out["family"][ev_name][name] = family_row(logs, teacher_logs)
    return out


def summarize(results: dict, starts: dict, device: str = "cpu", extras: bool = True) -> dict:
    h = hypothesis_verdict(results)
    d = direction(h)
    summary = {
        "spec": "docs/pathway/structured_confirmatory_prereg_2026-09-14.md",
        "hypothesis": h,
        "bars": {k: starts[k]["rate"] for k in REFS},
        "arms": {a: {"verdict": r.get("verdict"), "reason": r.get("reason"),
                     "final_rate": (r.get("final") or {}).get("score", {}).get("rate")} for a, r in results.items()},
        "secondary": {a: {ref: bool(results[a]["final"]["gate"][ref]["pass"]) for ref in REFS}
                      for a in results if "final" in results[a]},
        "complete": all(a in results for a in ARMS),
        "direction": d,
    }
    if extras:
        summary |= characterise_and_family(results, h, device)
    return summary


def load_results() -> dict:
    return {p.stem.removeprefix("s3_arm_"): json.loads(p.read_text()) for p in sorted(OUT.glob("s3_arm_*.json"))}


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    for arm, r in results.items():
        rows = r.get("checkpoints") or []
        if rows:
            ax.plot([x["steps"] for x in rows], [x["win_rate"] for x in rows], marker="o",
                    color="tab:red" if r.get("kind") == "control" else None, label=f"{arm} ({summary['arms'][arm]['verdict']})")
    for ref, c in zip(REFS, ("k", "tab:red", "tab:purple", "tab:orange", "tab:gray"), strict=True):
        ax.axhline(starts[ref]["rate"], ls=":", color=c, lw=1.2, label=f"{ref} (gate) = {starts[ref]['rate']:.3f}")
    ax.set_xlabel("env steps"); ax.set_ylabel(f"probe capture rate (seeds {PROBE_SEED0}+)")
    ax.set_title("S3 — structured learner, five fresh seeds + zero-reward control"); ax.legend(fontsize=6, ncol=2)
    ax = axes[1]
    fam = summary.get("family", {})
    if fam:
        names = list(fam)
        pols = [p for p in next(iter(fam.values())) if p != "TEACHER"]
        width = 0.8 / (len(pols) + 1)
        for j, pol in enumerate(["TEACHER", *pols]):
            xs = [i + j * width for i in range(len(names))]
            ys = [fam[n][pol]["rate"] for n in names]
            ax.bar(xs, ys, width=width, label=pol)
        ax.set_xticks([i + 0.4 - width / 2 for i in range(len(names))]); ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel(f"capture rate (N={FAMILY_N}, seeds {FAMILY_SEED0}+)"); ax.set_title("opponent family (reported only)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.suptitle("S3 — confirmatory round" + ("" if summary["complete"] else " — INCOMPLETE"))
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
        summary = summarize(results, starts, args.device)
        (OUT / "s3_round.json").write_text(json.dumps({"summary": summary, "arms": results}, indent=2))
        (OUT / "s3_characterisation.json").write_text(json.dumps(
            {"characterisation": summary.get("characterisation"), "family": summary.get("family")}, indent=2))
        write_figure(results, starts, summary, OUT / "s3_round.png")
        print(json.dumps(summary["hypothesis"], indent=2))
        print(json.dumps(summary["arms"], indent=2))
        print(json.dumps(summary.get("characterisation"), indent=2))
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
```

Notes for the implementer: `evaluate_structured(model, ...)` calls `model.predict`, so the `_P` shim (an object with a `.predict` attribute) is what the family loop passes; `characterise(predict, ...)` takes the bound function directly. `run_train_arm` builds PPO directly on the `SearchMDP` (not via `PPO.load` of an init) — the init is saved for the record and for the shared-seed property of STR11/SZLONG11. `ruff` will want the `_P` class body on separate lines; that is fine.

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s3_round.py -q`
Expected: all PASS.

- [ ] **Step 5: Smoke the wiring at a tiny budget in a scratch directory**

A throwaway script under `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/ff03fc68-2116-4512-a227-312f3099d143/scratchpad/s3smoke/` that imports the runner via `importlib`, monkeypatches `OUT`/`START_PATH` to the scratch dir and `TOTAL_LONG=4000, EVERY_LONG=2000, EVAL_N=4, GATE_N=6, FAMILY_N=4`, then runs `measure_start("cpu")`, `run_arm("STR11","cpu")`, `run_arm("SZLONG11","cpu")`, `run_arm("SBLONG","cpu")` (reads the real `ckpts_SBLONG` zip; writes only to scratch), then `summarize(load_results(), load_start(), "cpu")` and `write_figure(...)`. Expected: no exception; `s3_arm_*.json` carry `final.gate` with five refs; `s3_init_11.zip` written once; the characterisation dict has TEACHER and SBLONG entries; the family dict has four evaders; the PNG exists. Paste the console output into your report. Nothing under the real `results/derisk/` may change.

- [ ] **Step 6: Lint, full suite, commit**

```bash
./venv/bin/ruff format scripts/derisk_s3_confirm.py tests/pe2/test_derisk_s3_round.py && ./venv/bin/ruff check scripts/derisk_s3_confirm.py tests/pe2/test_derisk_s3_round.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q -x
git add scripts/derisk_s3_confirm.py tests/pe2/test_derisk_s3_round.py
git commit -m "feat(derisk): S3 runner — five-seed confirmatory round, control, SBLONG gate, artefact rule, characterisation, family" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** §2 learner/streams/bars/gate → Task 2 (`run_train_arm`, `measure_start`, `gate_final`); §3 arms → `ARMS`, `run_existing_arm`; §4 verdicts, harness, INCOMPLETE, artefact rule (fresh seeds only, ties → lowest) → `hypothesis_verdict`, `choose_artefact`; §5 characterisation → Task 1 `characterise` + `characterise_and_family`; family table → `FAMILY`, `family_row`; §6 direction → `direction`; stop rule → guards; §8 artefacts → paths, `s3_characterisation.json`, figure.
- **Placeholders:** none.
- **Type consistency:** `characterise(predict, cfg, opp, n, seed0, hold)` matches; `gate_pass(logs, ref_logs, role, p_gate)` matches; result dicts carry `final.gate[REF].{pass, rate, k, n, p, b, c, ref_rate}` which `hypothesis_verdict`/`choose_artefact`/tests read; `_arm` fixture mirrors that shape.
