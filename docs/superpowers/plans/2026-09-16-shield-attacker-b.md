# Shield Attacker, Round B (S6b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A thin S6b runner that reuses the S6 runner's machinery (defender factory, `EvadeMDP` arms, probe, atomic writes, provenance) with the S6b constants, arms, comparators and verdict rules.

**Architecture:** `scripts/derisk_s6b_attacker.py` loads `scripts/derisk_s6_attacker.py` as a module via `importlib`, overrides its module-level paths/constants/`ARMS` (so `run_train_arm`, `run_arm`, `_entry`, `defender_factory`, `_write_json`, `provenance` behave with S6b settings), and defines its own `measure_start` (HUG7 + GREEDY only), `h2_verdicts`, `hsafe_verdict`, `direction`, `summarize`, `write_figure`, `write_gif` (CTL vs EZ twin). No library changes.

**Tech Stack:** as S6.

**Spec:** `docs/pathway/shield_attacker_b_prereg_2026-09-16.md` (binding; read it first). S6's spec and runner are the reference for everything not restated.

## Global Constraints

- Always `./venv/bin/python`; tests `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest <files> -q`; ruff format + check clean.
- Pre-registered, never tunable: `SEEDS = (25, 26, 27)`, `PROBE_SEED0 = 1_140_000`, `GATE_SEED0 = 1_150_000`, `GATE_N = 500`, `EVAL_N = 200`, `TOTAL_LONG = 10_000_000`, `EVERY_LONG = 1_000_000`, `PASS_MIN = 2`, `P_GATE = 0.05`, `GAMMA = 0.99`, `HOLD = 10`, `D_POS = 4.0`, `BETA = 0.0` for every arm; arms CTL/EZ × D1/D2 × 3 seeds = 12; comparators HUG7 (bar) and GREEDY (reported); defenders as S6.
- H2(D) iff CTL-D-s beats EZ-D-s on ≥ 2/3 (seed-paired) AND CTL-D-s beats HUG7 on ≥ 2/3; harness flag if EZ-D-s beats HUG7 on ≥ 2/3 (then `PASS (vs EZ only)` if only the EZ clause holds); H-SAFE counts train + probe + gate + comparator pursuer violations; missing seed ⇒ INCOMPLETE; stop-rule guards; crash records.
- Never run comparators/arms/summary/GIF against the real `results/derisk/`; tests and the smoke redirect every S6 module path (`OUT`, `START_PATH`) AND the S6b ones into scratch. The S6 module's `PREFLIGHT_PATH` is not used (V1-P already passed in S6 and the defenders are unchanged).
- Commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

### Task 1: `scripts/derisk_s6b_attacker.py`

**Files:**
- Create: `scripts/derisk_s6b_attacker.py`
- Test: `tests/pe2/test_derisk_s6b_round.py`

**Interfaces:**
- Consumes (from the S6 module `S6`): `S6.defender_factory(d, device)`, `S6.run_arm(arm, device)` (which uses `S6.ARMS`, `S6.OUT`, `S6.load_start`, `S6.run_train_arm`, `S6.PROBE_SEED0/GATE_SEED0/GATE_N/EVAL_N/TOTAL_LONG/EVERY_LONG`, `S6._entry`, `S6._write_json`, `S6.provenance`, `S6.require_arm_not_run`), `pe2.evade.play_scripted_evader`, `pe2.derisk.paired_indicator_test`.
- Produces: CLI `--measure-start | --arms <ARM...> | --summarize | --gif [--device cpu]`; module-level `ARMS`, `h2_verdicts`, `hsafe_verdict`, `direction`, `summarize`, `apply_overrides()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_derisk_s6b_round.py
"""scripts/derisk_s6b_attacker.py — the second attacker round's own decisions."""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s6b_attacker.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s6b_attacker", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _ind(rate, n=500):
    k = int(round(rate * n))
    return [i < k for i in range(n)]


def _arm(name, deadlocks):
    fam, rest = name.split("-")
    return {"arm": name, "family": fam, "defender": rest[:2], "seed": int(rest[2:]), "train_pursuer_violations": 0,
            "probe_pursuer_violations": 0,
            "final": {"pursuer_violations": 0, "deadlock_ind": deadlocks, "escape_ind": _ind(0.8),
                      "pursuer": {"deadlock_episode_rate": sum(deadlocks) / len(deadlocks), "escape_rate": 0.8,
                                  "stall_step_rate": 0.4, "intervention_rate": 0.5}}, "checkpoints": []}


def _starts(hug7=(0.53, 0.34), greedy=(0.49, 0.32)):
    return {d: {"HUG7": {"deadlock_ind": _ind(h), "escape_ind": _ind(0.6), "pursuer_violations": 0,
                         "pursuer": {"deadlock_episode_rate": h, "escape_rate": 0.6}},
                "GREEDY": {"deadlock_ind": _ind(g), "escape_ind": _ind(0.5), "pursuer_violations": 0,
                           "pursuer": {"deadlock_episode_rate": g, "escape_rate": 0.5}}}
            for d, h, g in zip(("D1", "D2"), hug7, greedy, strict=True)}


def _full(ctl=((0.66, 0.68, 0.64), (0.62, 0.65, 0.61)), ez=((0.48, 0.47, 0.50), (0.34, 0.35, 0.33))):
    res = {}
    for d, c3, e3 in zip(("D1", "D2"), ctl, ez, strict=True):
        for s, c, e in zip(R.SEEDS, c3, e3, strict=True):
            res[f"CTL-{d}{s}"] = _arm(f"CTL-{d}{s}", _ind(c))
            res[f"EZ-{d}{s}"] = _arm(f"EZ-{d}{s}", _ind(e))
    return res


class TestProtocol:
    def test_constants_and_overrides(self):
        assert R.SEEDS == (25, 26, 27) and (R.PROBE_SEED0, R.GATE_SEED0) == (1_140_000, 1_150_000)
        assert (R.GATE_N, R.EVAL_N, R.TOTAL_LONG, R.EVERY_LONG) == (500, 200, 10_000_000, 1_000_000)
        assert R.PASS_MIN == 2 and R.P_GATE == 0.05
        assert R.S6.PROBE_SEED0 == 1_140_000 and R.S6.GATE_SEED0 == 1_150_000  # overrides applied to the S6 module
        assert R.S6.OUT == R.OUT and R.S6.START_PATH == R.START_PATH == R.OUT / "s6b_start.json"
        assert R.S6.ARMS is R.ARMS

    def test_arm_table(self):
        assert len(R.ARMS) == 12 and set(a.split("-")[0] for a in R.ARMS) == {"CTL", "EZ"}
        assert R.ARMS["CTL-D125"] == {"family": "CTL", "defender": "D1", "seed": 25, "beta": 0.0, "zero_reward": False}
        assert R.ARMS["EZ-D227"] == {"family": "EZ", "defender": "D2", "seed": 27, "beta": 0.0, "zero_reward": True}
        assert all(v["beta"] == 0.0 for v in R.ARMS.values())

    def test_artefact_names_are_s6b(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path); monkeypatch.setattr(R.S6, "OUT", tmp_path)
        assert R.S6._arm_file("CTL-D125").name == "s6b_arm_CTL-D125.json"
        assert R.S6._probe_file("CTL-D125").name == "s6b_probe_CTL-D125.json"

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path); monkeypatch.setattr(R.S6, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s6b_start.json"); monkeypatch.setattr(R.S6, "START_PATH", tmp_path / "s6b_start.json")
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()
        (tmp_path / "s6b_arm_CTL-D125.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.S6.require_arm_not_run("CTL-D125")


class TestVerdicts:
    def test_pass_both_clauses(self):
        h = R.h2_verdicts(_full(), _starts())
        assert h["H2"] == {"D1": "PASS", "D2": "PASS"} and h["H2_overall"] == "PASS"
        assert h["clauses"]["D1"]["vs_ez"] == 3 and h["clauses"]["D1"]["vs_hug7"] == 3
        assert h["harness"] == {"D1": "VALID", "D2": "VALID"}

    def test_fail_when_ctl_equals_ez(self):
        h = R.h2_verdicts(_full(ctl=((0.48, 0.47, 0.50), (0.34, 0.35, 0.33))), _starts())
        assert h["H2"]["D1"] == "FAIL" and h["H2"]["D2"] == "FAIL"

    def test_pass_vs_ez_only_when_structure_beats_hug7(self):
        h = R.h2_verdicts(_full(ez=((0.60, 0.61, 0.62), (0.34, 0.35, 0.33))), _starts())
        assert h["harness"]["D1"] == "STRUCTURE-DRIVEN" and h["H2"]["D1"] == "PASS (vs EZ only)"
        assert h["harness"]["D2"] == "VALID" and h["H2"]["D2"] == "PASS"

    def test_missing_seed_incomplete(self):
        res = _full(); del res["EZ-D226"]
        assert R.h2_verdicts(res, _starts())["H2"]["D2"] == "INCOMPLETE"

    def test_r3b(self):
        h = R.h2_verdicts(_full(), _starts())
        assert set(h["R3b"]) >= {"gains_D1", "gains_D2", "mean_gain_D1", "mean_gain_D2", "ctl_mean_D1", "ctl_mean_D2", "remedy_reduces_gain"}


class TestSafety:
    def test_hsafe(self):
        assert R.hsafe_verdict(_full(), _starts())["verdict"] == "HOLDS"
        st = _starts(); st["D1"]["HUG7"]["pursuer_violations"] = 1
        assert R.hsafe_verdict(_full(), st)["verdict"] == "FAILS"
        res = _full(); del res["CTL-D127"]
        assert R.hsafe_verdict(res, _starts())["verdict"] == "INCOMPLETE"


class TestDirection:
    def test_rows(self):
        assert "weaponises" in R.direction({"hsafe": "HOLDS", "H2_overall": "PASS"})
        assert "no R2" in R.direction({"hsafe": "HOLDS", "H2_overall": "FAIL"})
        assert "Stop" in R.direction({"hsafe": "FAILS", "H2_overall": "PASS"})
        assert "incomplete" in R.direction({"hsafe": "INCOMPLETE", "H2_overall": "INCOMPLETE"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s6b_round.py -q`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Write the runner**

First, two tiny refactors in `scripts/derisk_s6_attacker.py` so the artefact names are overridable: add module-level `ARTEFACT_PREFIX = "s6"` and helpers `_arm_file(arm) -> OUT / f"{ARTEFACT_PREFIX}_arm_{arm}.json"`, `_probe_file(arm) -> OUT / f"{ARTEFACT_PREFIX}_probe_{arm}.json"`, `_init_file(arm) -> OUT / f"{ARTEFACT_PREFIX}_init_{arm}.zip"`, and use them everywhere the S6 runner builds those paths (`run_train_arm`, `run_arm`, `require_arm_not_run`, `load_results`, `require_not_measured`'s glob — use `f"{ARTEFACT_PREFIX}_arm_*.json"`). Behaviour for S6 is unchanged (prefix "s6"); re-run `tests/pe2/test_derisk_s6_round.py` to confirm.

```python
# scripts/derisk_s6b_attacker.py
"""S6b — Step 9 shield-attacker, second round: the learned evader vs its zero-reward twin
and vs the strongest scripted probe (spec: docs/pathway/shield_attacker_b_prereg_2026-09-16.md).
Nothing here may change a threshold, a seed, a stream, a rule or a budget after a number is seen.

Reuses the S6 runner's machinery (loaded as a module) with S6b constants and arms:
  CTL-{D1,D2}{25,26,27}  terminal reward, no bonus      <- the learned evader
  EZ-{D1,D2}{25,26,27}   all rewards zeroed             <- seed-paired harness comparator
Comparators on the gate stream: HUG7 (the bar), GREEDY (reported).

Usage:
  ./venv/bin/python scripts/derisk_s6b_attacker.py --measure-start
  ./venv/bin/python scripts/derisk_s6b_attacker.py --arms CTL-D125   (one per process)
  ./venv/bin/python scripts/derisk_s6b_attacker.py --summarize
  ./venv/bin/python scripts/derisk_s6b_attacker.py --gif
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import paired_indicator_test
from pe2.scripted import GreedyEvader, ObstacleHuggingEvader

_S6_PATH = pathlib.Path(__file__).resolve().parent / "derisk_s6_attacker.py"
_spec = importlib.util.spec_from_file_location("derisk_s6_attacker", _S6_PATH)
S6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S6)

OUT = pathlib.Path("results/derisk")
START_PATH = OUT / "s6b_start.json"

# --- pre-registered (spec sections 2-4). Do not tune. ---
SEEDS = (25, 26, 27)
PROBE_SEED0 = 1_140_000
GATE_SEED0 = 1_150_000
GATE_N = 500
EVAL_N = 200
TOTAL_LONG = 10_000_000
EVERY_LONG = 1_000_000
PASS_MIN = 2
P_GATE = 0.05
DEFENDERS = S6.DEFENDERS
COMPARATORS = ("HUG7", "GREEDY")
ARMS: dict = {}
for _d in DEFENDERS:
    for _s in SEEDS:
        ARMS[f"CTL-{_d}{_s}"] = {"family": "CTL", "defender": _d, "seed": _s, "beta": 0.0, "zero_reward": False}
        ARMS[f"EZ-{_d}{_s}"] = {"family": "EZ", "defender": _d, "seed": _s, "beta": 0.0, "zero_reward": True}


def apply_overrides() -> None:
    """Point the S6 machinery at S6b's paths, streams, seeds and arms. The S6
    functions read these as module globals at call time."""
    S6.OUT = OUT
    S6.START_PATH = START_PATH
    S6.ARTEFACT_PREFIX = "s6b"
    S6.PROBE_SEED0 = PROBE_SEED0
    S6.GATE_SEED0 = GATE_SEED0
    S6.GATE_N = GATE_N
    S6.EVAL_N = EVAL_N
    S6.TOTAL_LONG = TOTAL_LONG
    S6.EVERY_LONG = EVERY_LONG
    S6.SEEDS = SEEDS
    S6.ARMS = ARMS
    S6.COMPARATORS = COMPARATORS


apply_overrides()


def load_start() -> dict:
    if not START_PATH.exists():
        raise SystemExit(f"comparators missing: {START_PATH}. Run --measure-start FIRST.")
    s = json.loads(START_PATH.read_text())
    for d in DEFENDERS:
        for c in COMPARATORS:
            if c not in s.get(d, {}) or "deadlock_ind" not in s[d][c]:
                raise SystemExit(f"{d}/{c} missing from {START_PATH}; re-run --measure-start.")
    return s


def measure_start(device: str) -> dict:
    from pe2.evade import play_scripted_evader

    if START_PATH.exists():
        raise SystemExit(f"{START_PATH} already exists (stop rule, spec section 5).")
    if any(OUT.glob("s6b_arm_*.json")) or any((OUT / f"ckpts_{a}").exists() for a in ARMS):
        raise SystemExit("S6b arms have already run; the stop rule forbids re-measuring.")
    cfg = EnvConfig()
    out: dict = {"gate_seed0": GATE_SEED0, "gate_n": GATE_N, "provenance": S6.provenance()}
    for d in DEFENDERS:
        f = S6.defender_factory(d, device)
        out[d] = {"HUG7": S6._entry(play_scripted_evader(cfg, ObstacleHuggingEvader, f, GATE_N, GATE_SEED0)),
                  "GREEDY": S6._entry(play_scripted_evader(cfg, GreedyEvader, f, GATE_N, GATE_SEED0))}
        for c, e in out[d].items():
            p = e["pursuer"]
            print(f"{d} {c:6s}: pursuer deadlock eps {p['deadlock_episode_rate']:.3f} stall {p['stall_step_rate']:.3f} "
                  f"violations {e['pursuer_violations']} | evader escape {p['escape_rate']:.3f}", flush=True)
    S6._write_json(START_PATH, out)
    return out


def run_arm(arm: str, device: str) -> dict:
    S6.load_start = load_start  # the S6 run_arm validates comparators through this name
    return S6.run_arm(arm, device)


def _ind(r):
    try:
        return r["final"]["deadlock_ind"]
    except (KeyError, TypeError):
        return None


def h2_verdicts(results: dict, starts: dict) -> dict:
    h2, harness, clauses, tests, r3, ctl_means = {}, {}, {}, {}, {}, {}
    for d in DEFENDERS:
        hug7 = starts[d]["HUG7"]["deadlock_ind"]
        per_ez, per_h7, per_ez_h7, gains, ctl_rates = {}, {}, {}, [], []
        for s in SEEDS:
            c, e = _ind(results.get(f"CTL-{d}{s}")), _ind(results.get(f"EZ-{d}{s}"))
            if c is None or e is None:
                per_ez[s] = per_h7[s] = per_ez_h7[s] = None
                continue
            per_ez[s] = paired_indicator_test(c, e, P_GATE)
            per_h7[s] = paired_indicator_test(c, hug7, P_GATE)
            per_ez_h7[s] = paired_indicator_test(e, hug7, P_GATE)
            gains.append(per_ez[s]["rate_a"] - per_ez[s]["rate_b"])
            ctl_rates.append(per_ez[s]["rate_a"])
        tests[d] = {"ctl_vs_ez": per_ez, "ctl_vs_hug7": per_h7, "ez_vs_hug7": per_ez_h7}
        if any(v is None for v in per_ez.values()):
            h2[d], harness[d] = "INCOMPLETE", "INCOMPLETE"
            clauses[d] = {"vs_ez": None, "vs_hug7": None}
        else:
            n_ez = sum(v["pass"] for v in per_ez.values())
            n_h7 = sum(v["pass"] for v in per_h7.values())
            n_ez_h7 = sum(v["pass"] for v in per_ez_h7.values())
            clauses[d] = {"vs_ez": n_ez, "vs_hug7": n_h7, "ez_vs_hug7": n_ez_h7}
            harness[d] = "STRUCTURE-DRIVEN" if n_ez_h7 >= PASS_MIN else "VALID"
            if n_ez >= PASS_MIN and n_h7 >= PASS_MIN:
                h2[d] = "PASS"
            elif n_ez >= PASS_MIN and harness[d] == "STRUCTURE-DRIVEN":
                h2[d] = "PASS (vs EZ only)"
            else:
                h2[d] = "FAIL"
        r3[d] = gains
        ctl_means[d] = (sum(ctl_rates) / len(ctl_rates)) if ctl_rates else None
    vals = list(h2.values())
    overall = "INCOMPLETE" if "INCOMPLETE" in vals else ("PASS" if any(v.startswith("PASS") for v in vals) else "FAIL")
    r3b = {"gains_D1": r3["D1"], "gains_D2": r3["D2"],
           "mean_gain_D1": (sum(r3["D1"]) / len(r3["D1"])) if r3["D1"] else None,
           "mean_gain_D2": (sum(r3["D2"]) / len(r3["D2"])) if r3["D2"] else None,
           "ctl_mean_D1": ctl_means["D1"], "ctl_mean_D2": ctl_means["D2"]}
    r3b["remedy_reduces_gain"] = (r3b["mean_gain_D1"] is not None and r3b["mean_gain_D2"] is not None and r3b["mean_gain_D2"] < r3b["mean_gain_D1"])
    return {"H2": h2, "H2_overall": overall, "clauses": clauses, "harness": harness, "tests": tests, "R3b": r3b}


def hsafe_verdict(results: dict, starts: dict | None = None) -> dict:
    steps, offenders = 0, []
    missing = [a for a in ARMS if a not in results or "final" not in results[a]]
    for a, r in results.items():
        v = int(r.get("train_pursuer_violations", 0)) + int(r.get("probe_pursuer_violations", 0)) + int((r.get("final") or {}).get("pursuer_violations", 0))
        if "probe_pursuer_violations" not in r:
            v += int(sum(x.get("pursuer_violations", 0) for x in r.get("checkpoints", [])))
        steps += v
        if v:
            offenders.append(a)
    if starts is not None:
        for d in DEFENDERS:
            for c in COMPARATORS:
                v = int(((starts.get(d) or {}).get(c) or {}).get("pursuer_violations") or 0)
                steps += v
                if v:
                    offenders.append(f"comparator:{d}/{c}")
    return {"verdict": "FAILS" if steps else ("INCOMPLETE" if missing else "HOLDS"), "pursuer_violation_steps": steps,
            "offenders": offenders, "missing": missing}


def direction(h: dict) -> str:
    if h["hsafe"] == "FAILS":
        return "H-SAFE FAILS: the shielded defender recorded a contact step. Stop; census; no further round without a user ruling."
    if h["hsafe"] == "INCOMPLETE" or h["H2_overall"] == "INCOMPLETE":
        return "incomplete: all twelve arms must run"
    if h["H2_overall"] == "PASS":
        return ("H2 PASSES with a valid control: a learned opponent weaponises a hard shield beyond its unrewarded twin and beyond the "
                "best scripted probe — R2 measured. Figure 1 candidate; R3-b says whether the S5 remedy resists it.")
    return "H2 FAILS: learned positioning at this budget does not exceed unrewarded positioning and/or the scripted probe — no R2 beyond the hugging geometry."


def load_results() -> dict:
    return {p.stem.removeprefix("s6b_arm_"): json.loads(p.read_text()) for p in sorted(OUT.glob("s6b_arm_*.json"))}


def summarize(results: dict, starts: dict) -> dict:
    h = h2_verdicts(results, starts)
    hs = hsafe_verdict(results, starts)
    table = {a: {"defender": r.get("defender"), "family": r.get("family"), "seed": r.get("seed"),
                 "deadlock_rate": (r.get("final") or {}).get("pursuer", {}).get("deadlock_episode_rate"),
                 "stall_rate": (r.get("final") or {}).get("pursuer", {}).get("stall_step_rate"),
                 "escape_rate": (r.get("final") or {}).get("pursuer", {}).get("escape_rate")} for a, r in results.items()}
    comps = {d: {c: {"deadlock_rate": starts[d][c]["pursuer"]["deadlock_episode_rate"], "escape_rate": starts[d][c]["pursuer"]["escape_rate"],
                     "pursuer_violations": starts[d][c].get("pursuer_violations")} for c in COMPARATORS} for d in DEFENDERS}
    return {"spec": "docs/pathway/shield_attacker_b_prereg_2026-09-16.md", "hsafe": hs, "hypotheses": h, "arms": table,
            "comparators": comps, "complete": all(a in results and "final" in results[a] for a in ARMS),
            "direction": direction({"hsafe": hs["verdict"], "H2_overall": h["H2_overall"]})}


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, d in zip(axes, DEFENDERS, strict=True):
        names, ys, cols = [], [], []
        for c in COMPARATORS:
            names.append(c); ys.append(summary["comparators"][d][c]["deadlock_rate"]); cols.append("tab:gray")
        for fam, col in (("EZ", "tab:green"), ("CTL", "tab:red")):
            for a, r in summary["arms"].items():
                if r["defender"] == d and r["family"] == fam and r["deadlock_rate"] is not None:
                    names.append(a); ys.append(r["deadlock_rate"]); cols.append(col)
        ax.bar(range(len(names)), ys, color=cols)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, fontsize=7)
        ax.set_ylabel("pursuer deadlock-episode rate (gate)"); ax.set_title(f"{d}: H2 {summary['hypotheses']['H2'][d]}")
    fig.suptitle("S6b — shield attacker vs its zero-reward twin" + ("" if summary["complete"] else " — INCOMPLETE"))
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def write_gif(path, results: dict, device: str) -> None:
    """Best CTL-D1 seed vs its EZ twin on a seed where CTL deadlocks the pursuer and EZ does not."""
    S6.GATE_SEED0 = GATE_SEED0
    ctl = max((a for a in results if a.startswith("CTL-D1") and "final" in results[a]), key=lambda a: results[a]["final"]["pursuer"]["deadlock_episode_rate"])
    ez = "EZ-D1" + ctl[len("CTL-D1"):]
    S6.write_gif_pair(path, results, device, left=ez, right=ctl)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.measure_start:
        measure_start(args.device); return
    for arm in args.arms:
        run_arm(arm, args.device)
    if args.summarize:
        results, starts = load_results(), load_start()
        summary = summarize(results, starts)
        S6._write_json(OUT / "s6b_round.json", {"summary": summary, "arms": results})
        write_figure(results, starts, summary, OUT / "s6b_round.png")
        print(json.dumps({k: summary[k] for k in ("hsafe", "hypotheses", "comparators")}, indent=2, default=str))
        print(json.dumps(summary["arms"], indent=2))
        print(f"\n{summary['direction']}")
    if args.gif:
        write_gif(OUT / "step_09b_attack.gif", load_results(), args.device)


if __name__ == "__main__":
    main()
```

`write_gif` needs one more S6 refactor: split S6's `write_gif` into `write_gif_pair(path, results, device, left, right, every=4)` (the rendering of two named arms on a discordant seed found by scanning `GATE_SEED0..+60` where `right` deadlocks and `left` does not, raising `SystemExit` if none) and have S6's `write_gif` compute `left = CTL twin, right = best ATK` and call it. S6's behaviour is unchanged.

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s6b_round.py tests/pe2/test_derisk_s6_round.py -q`
Expected: all PASS (S6's tests must still pass after the prefix/gif refactors).

- [ ] **Step 5: Smoke in scratch**

Scratch script under `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/ff03fc68-2116-4512-a227-312f3099d143/scratchpad/s6bsmoke/`: import the S6b runner via importlib; monkeypatch `R.OUT`, `R.START_PATH` AND `R.S6.OUT`, `R.S6.START_PATH` to scratch; set `R.GATE_N = R.S6.GATE_N = 6`, `R.EVAL_N = R.S6.EVAL_N = 4`, `R.TOTAL_LONG = R.S6.TOTAL_LONG = 4000`, `R.EVERY_LONG = R.S6.EVERY_LONG = 2000`; run `measure_start("cpu")`, `run_arm("CTL-D125", "cpu")`, `run_arm("EZ-D125", "cpu")`, `run_arm("CTL-D225", "cpu")`, `summarize(load_results(), load_start())`, `write_figure(...)`. Expected: artefacts named `s6b_*` in scratch, pursuer violations 0, PNG written, nothing under the real `results/derisk/` changed (assert with a before/after listing). Paste the output into your report.

- [ ] **Step 6: Lint, full suite, commit**

```bash
./venv/bin/ruff format scripts/derisk_s6_attacker.py scripts/derisk_s6b_attacker.py tests/pe2/test_derisk_s6b_round.py && ./venv/bin/ruff check scripts/derisk_s6_attacker.py scripts/derisk_s6b_attacker.py tests/pe2/test_derisk_s6b_round.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q -x
git add scripts/derisk_s6_attacker.py scripts/derisk_s6b_attacker.py tests/pe2/test_derisk_s6b_round.py
git commit -m "feat(derisk): S6b runner — learned evader vs zero-reward twin and HUG7, reusing the S6 machinery" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

## Self-review

- **Spec coverage:** §2 constants/streams/seeds/defenders/comparators → constants + `measure_start`; §3 arms → `ARMS`; §4 deadlock test, H2 two clauses ≥ 2/3, harness `STRUCTURE-DRIVEN` → `PASS (vs EZ only)`, R3-b with CTL means, INCOMPLETE, H-SAFE incl. comparators, GIF → `h2_verdicts`, `hsafe_verdict`, `write_gif`; §5 direction → `direction`; stop rule → S6 guards with the s6b prefix + `measure_start` guard; §6 artefacts → prefix "s6b".
- **Placeholders:** none; the two S6 refactors (artefact prefix helpers; `write_gif_pair`) are specified.
- **Type consistency:** `_entry` output (`deadlock_ind`, `escape_ind`, `pursuer`, `pursuer_violations`) is what `h2_verdicts`/`hsafe_verdict`/`summarize` read; the tests' fixtures mirror it.
