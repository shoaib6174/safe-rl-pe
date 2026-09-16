# Shield Remedy, Iteration 2 (S7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The structured evader as an opponent object plus a per-episode mixture factory, and the S7 runner: V1-E preflight, RC19 reference on the gate stream, three mixture-trained shield-aware pursuers, the H3-2 verdict with capture guard and artefact rule, and the stage-2 race (fresh attackers vs the artefact).

**Architecture:** `pe2/evade.py` gains `StructuredEvaderPolicy` (mirror of `StructuredPursuerPolicy` on the evader side, reproducing `EvadeMDP` bit-for-bit incl. the Box round-trip) and `MixtureOpponentFactory`. `scripts/derisk_s7_remedy2.py` follows the S5 runner for stage 1 (S5's `ShieldedStructuredProbe` pattern with `shield_obs=True`) and reuses the S6 machinery via importlib for stage 2 (like S6b), with S7 constants.

**Tech Stack:** as S6b.

**Spec:** `docs/pathway/shield_remedy_2_prereg_2026-09-16.md` (binding; read it first).

## Global Constraints

- Always `./venv/bin/python`; tests `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest <files> -q`; ruff format + check clean.
- Pre-registered, never tunable: stage 1 `SEEDS = (28, 29, 30)`, `PROBE_SEED0 = 1_160_000`, `GATE_SEED0 = 1_170_000`, `GATE_N = 500`, `EVAL_N = 200`, `TOTAL_LONG = 10_000_000`, `EVERY_LONG = 1_000_000`, `PASS_MIN = 2`, `P_GATE = 0.05`, RC config (`shield_obs=True`, `stall_penalty=1e-3`, `r_acq=0`, `shield=CBFShield(cfg)`, γ 0.99, hold 10); mixture members = `GreedyEvader`, `ObstacleHuggingEvader`, structured `SplitHugEvader`, structured `CTL-D125/126/127` (`ckpts_CTL-D12{5,6,7}/CTL-D12{5,6,7}_10000000.zip`), uniform per episode; probes vs `GreedyEvader` shielded; gate opponents = held-out `CTL-D225/226/227` (`ckpts_CTL-D22{5,6,7}/…_10000000.zip`), `GreedyEvader`, `ObstacleHuggingEvader`; reference `RC19` = `ckpts_RC19/RC19_10000000.zip` (`shield_obs=True`). Stage 2: `SEEDS2 = (31, 32, 33)`, `PROBE2_SEED0 = 1_180_000`, `GATE2_SEED0 = 1_190_000`, `RACE_BAR = 0.58`.
- V1-E: `StructuredEvaderPolicy` inside `SingleAgentWrapper(role="pursuer")` driven by `StructuredPursuerPolicy(D1)` must reproduce `EvadeMDP(D1).play_episode(model)` (outcome, steps, pursuer `n_stalled`, `longest_stall_run`) on 200 seeds (preflight) / 20 (test); the pursuer seat applies the float32 Box round-trip, so the reference side's pursuer must too — `StructuredPursuerPolicy.act` already returns the post-round-trip command, and re-normalising it is idempotent.
- H-SAFE counts pursuer contact steps in every arm's training, probes and gates, in the RC19 reference runs, and in stage 2.
- Never run preflight/reference/arms/summary against the real `results/derisk/` except the REAL `--preflight` at the end of Task 2 (writes only `s7_preflight.json`); tests and smokes redirect `OUT` etc. into scratch.
- Commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

### Task 1: `StructuredEvaderPolicy` and `MixtureOpponentFactory`

**Files:**
- Modify: `pe2/evade.py` (append)
- Test: `tests/pe2/test_evade.py` (append)

**Interfaces:**
- Consumes: `EvadeMDP` (its `_law_applies`/`_advance`/`step` semantics), `ObstacleHuggingEvader`, `full_state_observation`, `_normalize_evader`, `D_POS`; `SingleAgentWrapper._scale_action` (evader role: `evader_v_max`).
- Produces:
  - `StructuredEvaderPolicy(model, cfg, hold=10, d_pos=D_POS)` with `uses_truth = True`, `.reset()`, `.act(obs, env) -> tuple[float, float]`, `.decisions`.
  - `MixtureOpponentFactory(members: list[Callable[[], object]], seed: int)` — callable; each call returns `members[rng.integers(len(members))]()`; `.draws` (list of chosen indices) for tests.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pe2/test_evade.py`:

```python
class TestStructuredEvaderPolicy:
    def test_v1e_reproduces_evademdp_against_a_frozen_defender(self):
        from stable_baselines3 import PPO

        from pe2.evade import EvadeMDP, StructuredEvaderPolicy
        from pe2.exploitability import SingleAgentWrapper

        d1 = _str14()
        ev_model = PPO.load("results/derisk/ckpts_CTL-D125/CTL-D125_10000000.zip", device="cpu")
        for i in range(20):
            seed = 900_1400 + i
            ref = EvadeMDP(CFG, _d1_factory(d1), hold=10).play_episode(ev_model.predict, seed)
            w = SingleAgentWrapper(CFG, "pursuer", lambda: StructuredEvaderPolicy(ev_model, CFG, hold=10))
            w.reset(seed=seed)
            pol = StructuredPursuerPolicy(d1, CFG, CBFShield(CFG), HOLD, False)
            pol.reset()
            while True:
                v, om = pol.act(w._last_joint_obs["pursuer"], w.env)
                _, _, term, trunc, info = w.step(np.array([2 * v / CFG.v_max - 1, om / CFG.omega_max], dtype=np.float32))
                pol.record(bool(info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]))
                if term or trunc:
                    break
            got = (info["outcome"], w.env.current_step, pol.metrics.n_stalled, pol.metrics.longest_stall_run)
            exp = (ref["outcome"], ref["steps"], ref["pursuer_shield"]["n_stalled"], ref["pursuer_shield"]["longest_stall_run"])
            assert got == exp, f"seed {seed}: {got} != {exp}"

    def test_decisions_only_beyond_d_pos(self):
        from stable_baselines3 import PPO

        from pe2.evade import D_POS, StructuredEvaderPolicy
        from pe2.exploitability import SingleAgentWrapper

        ev_model = PPO.load("results/derisk/ckpts_CTL-D125/CTL-D125_10000000.zip", device="cpu")
        w = SingleAgentWrapper(CFG, "pursuer", lambda: StructuredEvaderPolicy(ev_model, CFG, hold=10))
        w.reset(seed=900_1450)
        pol = w._opponent
        n_before = pol.decisions
        # a stationary pursuer far away: the law never applies, decisions tick every `hold` steps
        far = float(np.hypot(*(w.env.pursuer_pose[:2] - w.env.evader_pose[:2]))) > D_POS
        for _ in range(30):
            _, _, term, trunc, _ = w.step(np.array([-1.0, 0.0], dtype=np.float32))
            if term or trunc:
                break
        if far:
            assert pol.decisions - n_before >= 1


class TestMixtureOpponentFactory:
    def test_draws_are_seeded_and_uniform_ish(self):
        from pe2.evade import MixtureOpponentFactory
        from pe2.scripted import FovLimitedEvader, GreedyEvader, ObstacleHuggingEvader

        members = [GreedyEvader, ObstacleHuggingEvader, FovLimitedEvader]
        f1 = MixtureOpponentFactory(members, seed=5)
        f2 = MixtureOpponentFactory(members, seed=5)
        kinds1 = [type(f1()).__name__ for _ in range(300)]
        kinds2 = [type(f2()).__name__ for _ in range(300)]
        assert kinds1 == kinds2
        counts = {k: kinds1.count(k) for k in set(kinds1)}
        assert set(counts) == {"GreedyEvader", "ObstacleHuggingEvader", "FovLimitedEvader"}
        assert all(60 <= c <= 140 for c in counts.values())
        assert f1.draws[:3] == f2.draws[:3]

    def test_works_as_a_training_opponent(self):
        from pe2.evade import MixtureOpponentFactory
        from pe2.scripted import GreedyEvader, ObstacleHuggingEvader
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, MixtureOpponentFactory([GreedyEvader, ObstacleHuggingEvader], seed=1), hold=10)
        obs, _ = m.reset()
        for _ in range(3):
            obs, _, term, trunc, _ = m.step(np.array([1.0, 0.0], dtype=np.float32))
            if term or trunc:
                obs, _ = m.reset()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_evade.py -k "StructuredEvaderPolicy or Mixture" -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement**

Append to `pe2/evade.py`:

```python
class StructuredEvaderPolicy:
    """The structured evader as an OPPONENT object for
    SingleAgentWrapper(role="pursuer") (S7, Step 10 iteration 2). The hug law
    (full state) whenever the pursuer is within `d_pos`; else the model's
    decision on the full-state observation held `hold` env steps, ending early
    when the law applies again — exactly EvadeMDP's semantics, so a learned
    attacker can be a training/gate opponent for a pursuer. The executed
    command takes the same float32 Box round-trip the learner seat applies
    (V1-E)."""

    uses_truth = True

    def __init__(self, model, cfg: EnvConfig, hold: int = 10, d_pos: float = D_POS):
        self.model = model
        self.cfg = cfg
        self.hold = int(hold)
        self.d_pos = float(d_pos)
        self.law = ObstacleHuggingEvader()
        self.decisions = 0
        self._held = None
        self._hold_left = 0
        self.reset()

    def reset(self) -> None:
        self.law.reset()
        self.decisions = 0
        self._held = None
        self._hold_left = 0

    def act(self, obs: np.ndarray, env) -> tuple[float, float]:
        cfg = self.cfg
        d = float(np.hypot(env.pursuer_pose[0] - env.evader_pose[0], env.pursuer_pose[1] - env.evader_pose[1]))
        if d <= self.d_pos:
            self._hold_left = 0
            v, w = self.law.act(None, env)
            a = _normalize_evader(v, w, cfg)
        else:
            if self._hold_left <= 0:
                lobs = full_state_observation(env.evader_pose, env.pursuer_pose, env.obstacles, cfg)
                self._held, _ = self.model.predict(lobs, deterministic=True)
                self._hold_left = self.hold
                self.decisions += 1
            a = self._held
            self._hold_left -= 1
        a64 = np.clip(np.asarray(np.asarray(a, dtype=np.float32), dtype=float), -1.0, 1.0)
        return (a64[0] + 1.0) / 2.0 * cfg.evader_v_max, a64[1] * cfg.omega_max


class MixtureOpponentFactory:
    """Per-episode opponent sampler for training (S7): each call draws one
    member factory uniformly with a seeded RNG and returns a fresh opponent.
    SingleAgentWrapper calls the factory at every reset, so the training
    stream sees the whole family in a reproducible order."""

    def __init__(self, members, seed: int):
        self.members = list(members)
        self.rng = np.random.default_rng(seed)
        self.draws: list[int] = []

    def __call__(self):
        i = int(self.rng.integers(len(self.members)))
        self.draws.append(i)
        return self.members[i]()
```

Subtlety for V1-E (read `EvadeMDP.step`): inside `EvadeMDP` a held decision ends early when `_law_applies()` becomes true AFTER a step; the object checks the law at the START of each `act` — same step index, because the law check on step t+1's `act` sees the post-step-t distance. The first decision: `EvadeMDP.play_episode` runs `_advance` (law while within `d_pos`) then decides; the object, starting with `_hold_left = 0`, decides on the first `act` where the law does not apply. Identical.

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_evade.py -q`
Expected: all PASS. If V1-E fails, compare per-step commands of both routes — do not add tolerances.

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/evade.py tests/pe2/test_evade.py && ./venv/bin/ruff check pe2/evade.py tests/pe2/test_evade.py
git add pe2/evade.py tests/pe2/test_evade.py
git commit -m "feat(pe2): StructuredEvaderPolicy (the structured attacker as an opponent) and MixtureOpponentFactory (S7)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `scripts/derisk_s7_remedy2.py`

**Files:**
- Create: `scripts/derisk_s7_remedy2.py`
- Test: `tests/pe2/test_derisk_s7_round.py`

**Interfaces:**
- Consumes: Task 1; `scripts/derisk_s5_remedy.py`'s patterns (`ShieldedStructuredProbe` — copy it into this runner with `shield_obs=True` fixed; `provenance`, `_write_json`, guards, crash record); `scripts/derisk_s6_attacker.py` loaded via importlib for stage 2 (`S6.run_arm` with overrides as S6b does: `OUT`, `START_PATH`, `ARTEFACT_PREFIX="s7race"`, `PROBE_SEED0`, `GATE_SEED0`, `SEEDS`, `ARMS`, `DEFENDERS` pointing the "D3" defender at the S7 artefact checkpoint with `shield_obs=True`, `load_start` rebound to a stage-2 loader); `pe2.derisk.paired_indicator_test`, `gate_pass`; `pe2.evade.evaluate_evader`, `play_scripted_evader`, `StructuredEvaderPolicy`, `MixtureOpponentFactory`; `pe2.structured.evaluate_structured(..., shield_factory, shield_obs)`, `StructuredPursuerPolicy`.
- Produces: CLI `--preflight | --measure-start | --arms RCM28.. | --summarize | --race-start | --race-arms ATK2-31.. | --race-summarize [--device cpu]`; module-level `ARMS`, `h32_verdicts`, `hsafe_verdict`, `direction`, `summarize`, `race_summary`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_derisk_s7_round.py
"""scripts/derisk_s7_remedy2.py — the second remedy round's own decisions."""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s7_remedy2.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s7_remedy2", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _ind(rate, n=500):
    k = int(round(rate * n))
    return [i < k for i in range(n)]


def _gate(deadlock, capture):
    return {"deadlock_ind": _ind(deadlock), "capture_ind": _ind(capture), "pursuer_violations": 0,
            "pursuer": {"deadlock_episode_rate": deadlock, "stall_step_rate": 0.3, "intervention_rate": 0.4}, "capture_rate": capture}


def _starts(att=(0.60, 0.58, 0.61), greedy=0.56, hug=0.40):
    return {"RC19": {"ATT25": _gate(att[0], 0.15), "ATT26": _gate(att[1], 0.15), "ATT27": _gate(att[2], 0.15),
                     "GREEDY": _gate(0.30, greedy), "HUG7": _gate(0.34, hug)}}


def _arm(name, att, greedy_capture=0.56, viol=0):
    return {"arm": name, "seed": int(name[3:]), "train_violations": viol, "probe_violations": 0,
            "final": {"pursuer_violations": 0, "gate": {"ATT25": _gate(att[0], 0.2), "ATT26": _gate(att[1], 0.2), "ATT27": _gate(att[2], 0.2),
                                                          "GREEDY": _gate(0.28, greedy_capture), "HUG7": _gate(0.30, 0.42)}}, "checkpoints": []}


def _full(seeds=((0.40, 0.42, 0.45), (0.38, 0.41, 0.44), (0.62, 0.60, 0.59)), greedy=(0.56, 0.55, 0.57)):
    return {f"RCM{s}": _arm(f"RCM{s}", a, g) for s, a, g in zip(R.SEEDS, seeds, greedy, strict=True)}


class TestProtocol:
    def test_constants(self):
        assert R.SEEDS == (28, 29, 30) and (R.PROBE_SEED0, R.GATE_SEED0) == (1_160_000, 1_170_000)
        assert (R.GATE_N, R.EVAL_N, R.TOTAL_LONG, R.EVERY_LONG, R.PASS_MIN, R.P_GATE) == (500, 200, 10_000_000, 1_000_000, 2, 0.05)
        assert R.RC_SETTINGS == {"shield_obs": True, "stall_penalty": 1e-3, "r_acq": 0.0}
        assert R.SEEDS2 == (31, 32, 33) and (R.PROBE2_SEED0, R.GATE2_SEED0, R.RACE_BAR) == (1_180_000, 1_190_000, 0.58)
        assert set(R.ARMS) == {"RCM28", "RCM29", "RCM30"}
        assert R.HELD_OUT == ("ATT25", "ATT26", "ATT27") and R.GATE_OPPONENTS == ("ATT25", "ATT26", "ATT27", "GREEDY", "HUG7")
        assert [m for m in R.MIXTURE] == ["GREEDY", "HUG7", "SPLITHUG", "CTL-D125", "CTL-D126", "CTL-D127"]

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s7_start.json")
        monkeypatch.setattr(R, "PREFLIGHT_PATH", tmp_path / "s7_preflight.json")
        with pytest.raises(SystemExit, match="preflight"):
            R.require_preflight()
        (tmp_path / "s7_preflight.json").write_text('{"pass": true}')
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()
        (tmp_path / "s7_arm_RCM28.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("RCM28")


class TestVerdicts:
    def test_h32_pass_two_of_three_seeds(self):
        h = R.h32_verdicts(_full(), _starts())
        assert h["per_seed"][28]["verdict"] == "PASS" and h["per_seed"][29]["verdict"] == "PASS" and h["per_seed"][30]["verdict"] == "FAIL"
        assert h["H32"] == "PASS" and h["artefact"] in ("RCM28", "RCM29")

    def test_artefact_is_lowest_mean_deadlock_among_passing(self):
        h = R.h32_verdicts(_full(), _starts())
        assert h["artefact"] == "RCM29"  # mean 0.41 < 0.423

    def test_capture_guard_label(self):
        h = R.h32_verdicts(_full(greedy=(0.40, 0.55, 0.57)), _starts())  # seed 28 captures far less than RC19's 0.56
        assert h["per_seed"][28]["verdict"] == "PASS (robust but weaker)"

    def test_fail_and_incomplete(self):
        h = R.h32_verdicts(_full(seeds=((0.60, 0.58, 0.61),) * 3), _starts())
        assert h["H32"] == "FAIL" and h["artefact"] is None
        res = _full(); del res["RCM29"]
        assert R.h32_verdicts(res, _starts())["H32"] == "INCOMPLETE"


class TestSafety:
    def test_hsafe(self):
        assert R.hsafe_verdict(_full(), _starts())["verdict"] == "HOLDS"
        res = _full(); res["RCM28"]["train_violations"] = 1
        assert R.hsafe_verdict(res, _starts())["verdict"] == "FAILS"
        st = _starts(); st["RC19"]["HUG7"]["pursuer_violations"] = 2
        assert R.hsafe_verdict(_full(), st)["verdict"] == "FAILS"
        res = _full(); del res["RCM30"]
        assert R.hsafe_verdict(res, _starts())["verdict"] == "INCOMPLETE"


class TestRace:
    def test_race_summary(self):
        arms = {f"ATK2-{s}": {"final": {"deadlock_ind": _ind(r), "pursuer": {"deadlock_episode_rate": r, "escape_rate": 0.7}, "pursuer_violations": 0}}
                for s, r in zip(R.SEEDS2, (0.45, 0.50, 0.48), strict=True)}
        r = R.race_summary(arms)
        assert r["pooled_rate"] == pytest.approx((0.45 + 0.50 + 0.48) / 3, abs=0.01) and r["H_RACE"] in ("HOLDS", "FAILS")
        assert r["pooled_wilson"][1] < R.RACE_BAR and r["H_RACE"] == "HOLDS"
        arms2 = {f"ATK2-{s}": {"final": {"deadlock_ind": _ind(0.62), "pursuer": {"deadlock_episode_rate": 0.62, "escape_rate": 0.9}, "pursuer_violations": 0}} for s in R.SEEDS2}
        assert R.race_summary(arms2)["H_RACE"] == "FAILS"


class TestDirection:
    def test_rows(self):
        assert "converges" in R.direction({"hsafe": "HOLDS", "H32": "PASS", "H_RACE": "HOLDS"})
        assert "open arms race" in R.direction({"hsafe": "HOLDS", "H32": "PASS", "H_RACE": "FAILS"})
        assert "does not buy" in R.direction({"hsafe": "HOLDS", "H32": "FAIL", "H_RACE": None})
        assert "Stop" in R.direction({"hsafe": "FAILS", "H32": "PASS", "H_RACE": None})
        assert "incomplete" in R.direction({"hsafe": "INCOMPLETE", "H32": "INCOMPLETE", "H_RACE": None})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s7_round.py -q`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Write the runner**

```python
# scripts/derisk_s7_remedy2.py
"""S7 — Step 10 iteration 2: a shield-aware pursuer trained against the structured attacker family
(spec: docs/pathway/shield_remedy_2_prereg_2026-09-16.md). Nothing here may change a threshold, a seed,
a stream, a rule or a budget after a number is seen.

Stage 1: RCM{28,29,30} = S5's RC configuration trained against the MIXTURE (Greedy, Hug7, structured
SplitHug, structured CTL-D125/126/127); gate vs the held-out attackers CTL-D225/226/227, Greedy, Hug7,
paired with RC19 on the same stream.
Stage 2 (only if H3-2 passes): ATK2-{31,32,33} fresh structured attackers trained against the S7 artefact
(S6 machinery via importlib), gate on a fresh stream; H-RACE reported vs RACE_BAR = 0.58.

Usage:
  ./venv/bin/python scripts/derisk_s7_remedy2.py --preflight        (V1-E on 200 seeds)
  ./venv/bin/python scripts/derisk_s7_remedy2.py --measure-start    (RC19 reference on the gate stream)
  ./venv/bin/python scripts/derisk_s7_remedy2.py --arms RCM28       (one per process)
  ./venv/bin/python scripts/derisk_s7_remedy2.py --summarize
  ./venv/bin/python scripts/derisk_s7_remedy2.py --race-start | --race-arms ATK2-31 | --race-summarize
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import gate_pass, paired_indicator_test
from pe2.eval import wilson_ci, win_count
from pe2.evade import D_POS
from pe2.metrics import summarize_shield
from pe2.scripted import GreedyEvader, ObstacleHuggingEvader
from pe2.structured import HOLD

OUT = pathlib.Path("results/derisk")
START_PATH = OUT / "s7_start.json"
PREFLIGHT_PATH = OUT / "s7_preflight.json"
RACE_START_PATH = OUT / "s7_race_start.json"
RC19_CKPT = OUT / "ckpts_RC19" / "RC19_10000000.zip"
STR14_CKPT = OUT / "ckpts_STR14" / "STR14_10000000.zip"
ATT_CKPT = {f"CTL-D1{s}": OUT / f"ckpts_CTL-D1{s}" / f"CTL-D1{s}_10000000.zip" for s in (25, 26, 27)}
HELD_OUT_CKPT = {f"ATT{s}": OUT / f"ckpts_CTL-D2{s}" / f"CTL-D2{s}_10000000.zip" for s in (25, 26, 27)}

# --- pre-registered (spec sections 3-5). Do not tune. ---
SEEDS = (28, 29, 30)
PROBE_SEED0 = 1_160_000
GATE_SEED0 = 1_170_000
GATE_N = 500
EVAL_N = 200
TOTAL_LONG = 10_000_000
EVERY_LONG = 1_000_000
PASS_MIN = 2
P_GATE = 0.05
GAMMA_S = 0.99
RC_SETTINGS = {"shield_obs": True, "stall_penalty": 1e-3, "r_acq": 0.0}
MIXTURE = ("GREEDY", "HUG7", "SPLITHUG", "CTL-D125", "CTL-D126", "CTL-D127")
HELD_OUT = ("ATT25", "ATT26", "ATT27")
GATE_OPPONENTS = HELD_OUT + ("GREEDY", "HUG7")
ARMS = {f"RCM{s}": {"seed": s} for s in SEEDS}
SEEDS2 = (31, 32, 33)
PROBE2_SEED0 = 1_180_000
GATE2_SEED0 = 1_190_000
RACE_BAR = 0.58
PREFLIGHT_N = 200

_MODELS: dict = {}


def _model(path, device):
    from stable_baselines3 import PPO

    key = str(path)
    if key not in _MODELS:
        _MODELS[key] = PPO.load(path, device=device)
    return _MODELS[key]


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
    return {"git_commit": sha, "sb3_version": stable_baselines3.__version__, "torch_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}


def _ci(k, n):
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


# ----------------------------------------------------------------------------
# opponents
# ----------------------------------------------------------------------------
def structured_evader_factory(ckpt, device):
    from pe2.evade import StructuredEvaderPolicy

    cfg = EnvConfig()
    model = _model(ckpt, device)
    return lambda: StructuredEvaderPolicy(model, cfg, hold=10, d_pos=D_POS)


def splithug_factory(device):
    from pe2.evade import SplitHugEvader, StructuredEvaderPolicy

    cfg = EnvConfig()
    return lambda: StructuredEvaderPolicy(SplitHugEvader(cfg), cfg, hold=10, d_pos=D_POS)


def mixture_factory(seed: int, device: str):
    from pe2.evade import MixtureOpponentFactory

    members = [GreedyEvader, ObstacleHuggingEvader, splithug_factory(device)] + [structured_evader_factory(ATT_CKPT[k], device) for k in ("CTL-D125", "CTL-D126", "CTL-D127")]
    return MixtureOpponentFactory(members, seed=seed)


def gate_opponent_factory(name: str, device: str):
    if name == "GREEDY":
        return GreedyEvader
    if name == "HUG7":
        return ObstacleHuggingEvader
    return structured_evader_factory(HELD_OUT_CKPT[name], device)


# ----------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------
def require_preflight() -> None:
    if not PREFLIGHT_PATH.exists():
        raise SystemExit(f"preflight missing: {PREFLIGHT_PATH}. Run --preflight FIRST (spec V1-E).")
    if not json.loads(PREFLIGHT_PATH.read_text()).get("pass"):
        raise SystemExit("V1-E preflight FAILED; the round may not proceed.")


def require_not_measured() -> None:
    if START_PATH.exists():
        raise SystemExit(f"{START_PATH} already exists (stop rule).")
    if any(OUT.glob("s7_arm_*.json")) or any((OUT / f"ckpts_{a}").exists() for a in ARMS):
        raise SystemExit("S7 arms have already run; the stop rule forbids re-measuring.")


def require_arm_not_run(arm: str) -> None:
    if (OUT / f"s7_arm_{arm}.json").exists() or (OUT / f"ckpts_{arm}").exists():
        raise SystemExit(f"arm {arm} has already run; the stop rule forbids re-running it.")


def load_start() -> dict:
    if not START_PATH.exists():
        raise SystemExit(f"reference missing: {START_PATH}. Run --measure-start FIRST.")
    s = json.loads(START_PATH.read_text())
    for o in GATE_OPPONENTS:
        if o not in s.get("RC19", {}) or "deadlock_ind" not in s["RC19"][o]:
            raise SystemExit(f"RC19/{o} missing from {START_PATH}; re-run --measure-start.")
    return s


# ----------------------------------------------------------------------------
# preflight (V1-E) and reference
# ----------------------------------------------------------------------------
def preflight(device: str) -> dict:
    import numpy as np

    from pe2.evade import EvadeMDP, StructuredEvaderPolicy
    from pe2.exploitability import SingleAgentWrapper
    from pe2.shield import CBFShield
    from pe2.structured import StructuredPursuerPolicy

    cfg = EnvConfig()
    d1 = _model(STR14_CKPT, device)
    ev = _model(ATT_CKPT["CTL-D125"], device)
    mism = []
    for i in range(PREFLIGHT_N):
        seed = GATE_SEED0 + i
        ref = EvadeMDP(cfg, lambda: StructuredPursuerPolicy(d1, cfg, CBFShield(cfg), HOLD, False), hold=10).play_episode(ev.predict, seed)
        w = SingleAgentWrapper(cfg, "pursuer", lambda: StructuredEvaderPolicy(ev, cfg, hold=10))
        w.reset(seed=seed)
        pol = StructuredPursuerPolicy(d1, cfg, CBFShield(cfg), HOLD, False)
        pol.reset()
        while True:
            v, om = pol.act(w._last_joint_obs["pursuer"], w.env)
            _, _, term, trunc, info = w.step(np.array([2 * v / cfg.v_max - 1, om / cfg.omega_max], dtype=np.float32))
            pol.record(bool(info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]))
            if term or trunc:
                break
        got = (info["outcome"], w.env.current_step, pol.metrics.n_stalled, pol.metrics.longest_stall_run)
        exp = (ref["outcome"], ref["steps"], ref["pursuer_shield"]["n_stalled"], ref["pursuer_shield"]["longest_stall_run"])
        if got != exp:
            mism.append({"seed": seed, "got": got, "expected": exp})
    out = {"pass": not mism, "n": PREFLIGHT_N, "seed0": GATE_SEED0, "mismatches": len(mism), "examples": mism[:10], "provenance": provenance()}
    _write_json(PREFLIGHT_PATH, out)
    print(f"V1-E {'PASS' if out['pass'] else 'FAIL'}: {len(mism)} mismatches in {PREFLIGHT_N} episodes", flush=True)
    if mism:
        raise SystemExit("V1-E FAILED: the evader object does not reproduce EvadeMDP; halting.")
    return out


def gate_pursuer(model, opponents: dict, device: str) -> dict:
    """Shielded, shield_obs=True gate of a pursuer against each gate opponent on GATE_SEED0."""
    from pe2.shield import CBFShield
    from pe2.structured import evaluate_structured

    cfg = EnvConfig()
    out = {}
    for name, factory in opponents.items():
        logs = evaluate_structured(model, cfg, factory, GATE_N, GATE_SEED0, hold=HOLD, shield_factory=lambda: CBFShield(cfg), shield_obs=True)
        k, n = win_count(logs, "pursuer")
        sh = summarize_shield(logs)
        out[name] = {"deadlock_ind": [bool(lg["shield"]["deadlock"]) for lg in logs], "capture_ind": [lg["outcome"] == "capture" for lg in logs],
                     "capture_rate": k / n, "capture": _ci(k, n), "pursuer": {kk: sh[kk] for kk in ("deadlock_episode_rate", "stall_step_rate", "intervention_rate", "infeasibility_rate", "backup_rate", "min_clearance_floor")},
                     "pursuer_violations": int(sh["violation_steps"])}
        print(f"    vs {name:7s}: capture {k / n:.3f}, deadlock eps {sh['deadlock_episode_rate']:.3f}, stall {sh['stall_step_rate']:.3f}, violations {sh['violation_steps']}", flush=True)
    return out


def measure_start(device: str) -> dict:
    require_preflight()
    require_not_measured()
    opps = {name: gate_opponent_factory(name, device) for name in GATE_OPPONENTS}
    print("RC19 reference on the gate stream:", flush=True)
    out = {"gate_seed0": GATE_SEED0, "gate_n": GATE_N, "provenance": provenance(), "RC19": gate_pursuer(_model(RC19_CKPT, device), opps, device)}
    _write_json(START_PATH, out)
    return out


# ----------------------------------------------------------------------------
# stage-1 arms
# ----------------------------------------------------------------------------
class ShieldedStructuredProbe:
    def __init__(self, cfg, opponent_factory, every, total, eval_n, seed0, ckpt_dir, report_path, label, hold=HOLD):
        from stable_baselines3.common.callbacks import BaseCallback

        from pe2.shield import CBFShield
        from pe2.structured import evaluate_structured

        if total % every != 0:
            raise ValueError(f"total ({total}) must be a multiple of every ({every})")
        rows: list[dict] = []

        def evaluate_at(model, env_steps, mark, train_violations):
            path = ckpt_dir / f"{label}_{mark}.zip"
            model.save(path)
            logs = evaluate_structured(model, cfg, opponent_factory, eval_n, seed0, hold=hold, shield_factory=lambda: CBFShield(cfg), shield_obs=True)
            k, n = win_count(logs, "pursuer")
            sh = summarize_shield(logs)
            row = {"steps": env_steps, "mark": mark, "ckpt": str(path), "win_rate": k / n, "k": k, "n": n, "violations": int(sh["violation_steps"]),
                   "stall_step_rate": sh["stall_step_rate"], "deadlock_episode_rate": sh["deadlock_episode_rate"], "train_violations": int(train_violations)}
            rows.append(row)
            _write_json(report_path, {"checkpoints": rows})
            print(f"  @{env_steps:>9} env steps (mark {mark}): shielded win rate vs Greedy {k}/{n} = {k / n:.3f}, violations {row['violations']}, deadlock eps {row['deadlock_episode_rate']:.3f}", flush=True)
            return row

        class _CB(BaseCallback):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.next_mark = every

            def _on_step(inner_self) -> bool:
                counts = inner_self.training_env.get_attr("env_steps")
                assert len(counts) == 1, "single env assumed (budget in env steps)"
                env_steps = int(counts[0])
                viol = int(inner_self.training_env.get_attr("violations_total")[0])
                while inner_self.next_mark <= total and env_steps >= inner_self.next_mark:
                    evaluate_at(inner_self.model, env_steps, inner_self.next_mark, viol)
                    inner_self.next_mark += every
                return env_steps < total

        self.rows = rows
        self.callback = _CB()


def run_train_arm(arm: str, starts: dict, device: str, holder: dict | None = None) -> dict:
    from stable_baselines3 import PPO

    from pe2.shield import CBFShield
    from pe2.structured import SearchMDP

    seed = ARMS[arm]["seed"]
    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = ShieldedStructuredProbe(cfg, GreedyEvader, every=EVERY_LONG, total=TOTAL_LONG, eval_n=EVAL_N, seed0=PROBE_SEED0,
                                    ckpt_dir=d, report_path=OUT / f"s7_probe_{arm}.json", label=arm)
    if holder is not None:
        holder["rows"] = probe.rows
    env = SearchMDP(cfg, mixture_factory(seed, device), seed0=seed * 1000, hold=HOLD, r_acq=RC_SETTINGS["r_acq"], zero_reward=False,
                    shield=CBFShield(cfg), shield_obs=RC_SETTINGS["shield_obs"], stall_penalty=RC_SETTINGS["stall_penalty"])
    if holder is not None:
        holder["env"] = env
    model = PPO("MlpPolicy", env, seed=seed, gamma=GAMMA_S, verbose=0, device=device)
    model.save(OUT / f"s7_init_{arm}.zip")
    model.learn(total_timesteps=TOTAL_LONG, callback=probe.callback)
    result = {"arm": arm, "seed": seed, "settings": RC_SETTINGS, "mixture": list(MIXTURE), "env_steps": int(env.env_steps), "decisions": int(model.num_timesteps),
              "train_violations": int(env.violations_total), "probe_violations": int(sum(r.get("violations", 0) for r in probe.rows)),
              "mixture_draws": env.inner.opponent_factory.draws[:2000], "checkpoints": probe.rows, "provenance": provenance()}
    ckpt = d / f"{arm}_{TOTAL_LONG}.zip"
    if not any(r["mark"] == TOTAL_LONG for r in probe.rows) or not ckpt.exists():
        result["verdict"], result["reason"] = "UNDETERMINED", f"final probe at mark {TOTAL_LONG} never fired (env steps {env.env_steps})"
        return result
    opps = {name: gate_opponent_factory(name, device) for name in GATE_OPPONENTS}
    print(f"[{arm}] gate:", flush=True)
    g = gate_pursuer(PPO.load(ckpt, device=device), opps, device)
    result["final"] = {"ckpt": str(ckpt), "gate": g, "pursuer_violations": int(sum(v["pursuer_violations"] for v in g.values()))}
    per = {o: paired_indicator_test(starts["RC19"][o]["deadlock_ind"], g[o]["deadlock_ind"], P_GATE) for o in HELD_OUT}
    n_pass = sum(v["pass"] for v in per.values())
    result["verdict"] = f"deadlock lower than RC19 on {n_pass}/3 held-out attackers"
    result["reason"] = "; ".join(f"{o}: RC19 {per[o]['rate_a']:.3f} vs S7 {per[o]['rate_b']:.3f} p={per[o]['p']:.1e} {'PASS' if per[o]['pass'] else 'fail'}" for o in HELD_OUT) + \
        f"; capture vs Greedy {g['GREEDY']['capture_rate']:.3f} (RC19 {starts['RC19']['GREEDY']['capture_rate']:.3f}); violations train {result['train_violations']} probe {result['probe_violations']} gate {result['final']['pursuer_violations']}"
    return result


def run_arm(arm: str, device: str) -> dict:
    require_arm_not_run(arm)
    starts = load_start()
    holder: dict = {}
    try:
        result = run_train_arm(arm, starts, device, holder)
    except BaseException as e:
        _write_json(OUT / f"s7_arm_{arm}.json", {"arm": arm, "seed": ARMS[arm]["seed"], "verdict": "UNDETERMINED", "reason": f"crashed: {type(e).__name__}: {e}",
                    "checkpoints": holder.get("rows", []), "train_violations": int(getattr(holder.get("env"), "violations_total", 0)),
                    "probe_violations": int(sum(r.get("violations", 0) for r in holder.get("rows", []))), "provenance": provenance()})
        raise
    _write_json(OUT / f"s7_arm_{arm}.json", result)
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# stage-1 verdicts
# ----------------------------------------------------------------------------
def h32_verdicts(results: dict, starts: dict) -> dict:
    per_seed, passing = {}, []
    for s in SEEDS:
        r = results.get(f"RCM{s}")
        if r is None or "final" not in r:
            per_seed[s] = {"verdict": "INCOMPLETE"}
            continue
        g = r["final"]["gate"]
        tests = {o: paired_indicator_test(starts["RC19"][o]["deadlock_ind"], g[o]["deadlock_ind"], P_GATE) for o in HELD_OUT}
        n = sum(v["pass"] for v in tests.values())
        cap = gate_pass(  # is RC19 significantly better at capture vs Greedy than the S7 pursuer?
            [{"seed": i, "role": "pursuer", "outcome": "capture" if c else "timeout"} for i, c in enumerate(starts["RC19"]["GREEDY"]["capture_ind"])],
            [{"seed": i, "role": "pursuer", "outcome": "capture" if c else "timeout"} for i, c in enumerate(g["GREEDY"]["capture_ind"])], "pursuer", P_GATE)
        weaker = bool(cap["pass"])
        mean_dl = sum(g[o]["pursuer"]["deadlock_episode_rate"] for o in HELD_OUT) / len(HELD_OUT)
        if n >= PASS_MIN:
            v = "PASS (robust but weaker)" if weaker else "PASS"
            passing.append((mean_dl, s))
        else:
            v = "FAIL"
        per_seed[s] = {"verdict": v, "passes": n, "tests": tests, "capture_guard": {"rc19_better_p": cap["p"], "weaker": weaker, "s7_capture": g["GREEDY"]["capture_rate"], "rc19_capture": starts["RC19"]["GREEDY"]["capture_rate"]}, "mean_deadlock_heldout": mean_dl}
    if any(v["verdict"] == "INCOMPLETE" for v in per_seed.values()):
        h = "INCOMPLETE"
    else:
        h = "PASS" if sum(v["verdict"].startswith("PASS") for v in per_seed.values()) >= PASS_MIN else "FAIL"
    artefact = f"RCM{min(passing)[1]}" if (h == "PASS" and passing) else None
    return {"H32": h, "per_seed": per_seed, "artefact": artefact}


def hsafe_verdict(results: dict, starts: dict | None = None) -> dict:
    steps, offenders = 0, []
    missing = [a for a in ARMS if a not in results or "final" not in results[a]]
    for a, r in results.items():
        v = int(r.get("train_violations", 0)) + int(r.get("probe_violations", 0)) + int((r.get("final") or {}).get("pursuer_violations", 0))
        if "probe_violations" not in r:
            v += int(sum(x.get("violations", 0) for x in r.get("checkpoints", [])))
        steps += v
        if v:
            offenders.append(a)
    if starts is not None:
        for o in GATE_OPPONENTS:
            v = int(((starts.get("RC19") or {}).get(o) or {}).get("pursuer_violations") or 0)
            steps += v
            if v:
                offenders.append(f"reference:RC19/{o}")
    return {"verdict": "FAILS" if steps else ("INCOMPLETE" if missing else "HOLDS"), "pursuer_violation_steps": steps, "offenders": offenders, "missing": missing}


def direction(h: dict) -> str:
    if h["hsafe"] == "FAILS":
        return "H-SAFE FAILS: a shielded pursuer recorded a contact step. Stop; census; no further round without a user ruling."
    if h["hsafe"] == "INCOMPLETE" or h["H32"] == "INCOMPLETE":
        return "incomplete: all three RCM arms must run"
    if h["H32"] == "FAIL":
        return "H3-2 FAILS: family training does not buy resistance at this budget — the structured attack is a property of the shield's geometry, not of the pursuer's policy."
    if h.get("H_RACE") == "HOLDS":
        return "H3-2 PASSES and H-RACE holds: training against the structured family closes most of the attacker's gain — the arms race converges toward the defender (R3's strongest form)."
    if h.get("H_RACE") == "FAILS":
        return "H3-2 PASSES but fresh attackers adapt (H-RACE fails): robust to unseen attackers of the old kind, an open arms race against new ones — reported as such."
    return "H3-2 PASSES: stage 2 (fresh attackers vs the artefact) decides whether the race converges — run --race-start / --race-arms / --race-summarize."


def load_results() -> dict:
    return {p.stem.removeprefix("s7_arm_"): json.loads(p.read_text()) for p in sorted(OUT.glob("s7_arm_RCM*.json"))}


def summarize(results: dict, starts: dict, race: dict | None = None) -> dict:
    h = h32_verdicts(results, starts)
    hs = hsafe_verdict(results, starts)
    table = {}
    for a, r in results.items():
        g = (r.get("final") or {}).get("gate") or {}
        table[a] = {o: {"deadlock": g[o]["pursuer"]["deadlock_episode_rate"], "capture": g[o]["capture_rate"]} for o in g}
    ref = {o: {"deadlock": starts["RC19"][o]["pursuer"]["deadlock_episode_rate"], "capture": starts["RC19"][o]["capture_rate"]} for o in GATE_OPPONENTS}
    hr = race["H_RACE"] if race else None
    return {"spec": "docs/pathway/shield_remedy_2_prereg_2026-09-16.md", "hsafe": hs, "hypotheses": h, "arms": table, "rc19": ref, "race": race,
            "complete": all(a in results and "final" in results[a] for a in ARMS),
            "direction": direction({"hsafe": hs["verdict"], "H32": h["H32"], "H_RACE": hr})}


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    opps = list(GATE_OPPONENTS)
    x = range(len(opps))
    ax.bar([i - 0.3 for i in x], [summary["rc19"][o]["deadlock"] for o in opps], width=0.15, color="k", label="RC19 (S5)")
    for j, a in enumerate(sorted(summary["arms"])):
        ax.bar([i - 0.15 + 0.15 * j for i in x], [summary["arms"][a].get(o, {}).get("deadlock", 0) for o in opps], width=0.15, label=f"{a} ({summary['hypotheses']['per_seed'][int(a[3:])]['verdict']})")
    ax.set_xticks(list(x)); ax.set_xticklabels(opps); ax.set_ylabel("pursuer deadlock-episode rate (gate)"); ax.set_title(f"S7 stage 1: H3-2 {summary['hypotheses']['H32']}"); ax.legend(fontsize=7)
    ax = axes[1]
    ax.bar([i - 0.3 for i in x], [summary["rc19"][o]["capture"] for o in opps], width=0.15, color="k", label="RC19")
    for j, a in enumerate(sorted(summary["arms"])):
        ax.bar([i - 0.15 + 0.15 * j for i in x], [summary["arms"][a].get(o, {}).get("capture", 0) for o in opps], width=0.15, label=a)
    ax.set_xticks(list(x)); ax.set_xticklabels(opps); ax.set_ylabel("capture rate (gate)"); ax.set_title("capture"); ax.legend(fontsize=7)
    fig.suptitle("S7 — shield-aware pursuer trained against the structured family" + ("" if summary["complete"] else " — INCOMPLETE"))
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------
# stage 2: the race (S6 machinery)
# ----------------------------------------------------------------------------
def _s6():
    p = pathlib.Path(__file__).resolve().parent / "derisk_s6_attacker.py"
    spec = importlib.util.spec_from_file_location("derisk_s6_attacker", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def race_config(artefact_ckpt) -> dict:
    return {"D3": {"ckpt": pathlib.Path(artefact_ckpt), "shield_obs": True, "hold": HOLD}}


def race_setup(S6, artefact_ckpt):
    S6.OUT = OUT
    S6.START_PATH = RACE_START_PATH
    S6.ARTEFACT_PREFIX = "s7race"
    S6.PROBE_SEED0 = PROBE2_SEED0
    S6.GATE_SEED0 = GATE2_SEED0
    S6.GATE_N = GATE_N
    S6.EVAL_N = EVAL_N
    S6.TOTAL_LONG = TOTAL_LONG
    S6.EVERY_LONG = EVERY_LONG
    S6.SEEDS = SEEDS2
    S6.DEFENDERS = race_config(artefact_ckpt)
    S6.COMPARATORS = ("HUG7", "GREEDY")
    S6.ARMS = {f"ATK2-{s}": {"family": "CTL", "defender": "D3", "seed": s, "beta": 0.0, "zero_reward": False} for s in SEEDS2}

    def load_race_start():
        if not RACE_START_PATH.exists():
            raise SystemExit(f"race comparators missing: {RACE_START_PATH}. Run --race-start FIRST.")
        return json.loads(RACE_START_PATH.read_text())

    S6.load_start = load_race_start


def _artefact_ckpt() -> pathlib.Path:
    s = json.loads((OUT / "s7_round.json").read_text())["summary"]
    art = s["hypotheses"]["artefact"]
    if not art:
        raise SystemExit("no S7 artefact: H3-2 did not pass; stage 2 does not run (spec section 5).")
    return pathlib.Path(json.loads((OUT / f"s7_arm_{art}.json").read_text())["final"]["ckpt"])


def race_start(device: str) -> dict:
    from pe2.evade import play_scripted_evader

    S6 = _s6()
    race_setup(S6, _artefact_ckpt())
    if RACE_START_PATH.exists():
        raise SystemExit(f"{RACE_START_PATH} already exists (stop rule).")
    cfg = EnvConfig()
    f = S6.defender_factory("D3", device)
    out = {"gate_seed0": GATE2_SEED0, "gate_n": GATE_N, "artefact": str(S6.DEFENDERS["D3"]["ckpt"]), "provenance": provenance(),
           "D3": {"HUG7": S6._entry(play_scripted_evader(cfg, ObstacleHuggingEvader, f, GATE_N, GATE2_SEED0)),
                  "GREEDY": S6._entry(play_scripted_evader(cfg, GreedyEvader, f, GATE_N, GATE2_SEED0))}}
    for c, e in out["D3"].items():
        print(f"D3 {c:6s}: pursuer deadlock eps {e['pursuer']['deadlock_episode_rate']:.3f} violations {e['pursuer_violations']} | escape {e['pursuer']['escape_rate']:.3f}", flush=True)
    _write_json(RACE_START_PATH, out)
    return out


def race_arm(arm: str, device: str) -> dict:
    S6 = _s6()
    race_setup(S6, _artefact_ckpt())
    return S6.run_arm(arm, device)


def race_summary(arms: dict) -> dict:
    inds = [arms[a]["final"]["deadlock_ind"] for a in sorted(arms) if "final" in arms[a]]
    pooled = [x for ind in inds for x in ind]
    k, n = sum(map(bool, pooled)), len(pooled)
    lo, hi = wilson_ci(k, n) if n else (None, None)
    per = {a: {"deadlock_rate": arms[a]["final"]["pursuer"]["deadlock_episode_rate"], "escape_rate": arms[a]["final"]["pursuer"]["escape_rate"],
               "pursuer_violations": arms[a]["final"]["pursuer_violations"]} for a in sorted(arms) if "final" in arms[a]}
    complete = len(inds) == len(SEEDS2)
    return {"per_arm": per, "pooled_rate": (k / n) if n else None, "pooled_wilson": [lo, hi], "race_bar": RACE_BAR,
            "H_RACE": ("INCOMPLETE" if not complete else ("HOLDS" if hi < RACE_BAR else "FAILS"))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--race-start", action="store_true")
    p.add_argument("--race-arms", nargs="*", default=[])
    p.add_argument("--race-summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        preflight(args.device); return
    if args.measure_start:
        measure_start(args.device); return
    for arm in args.arms:
        run_arm(arm, args.device)
    if args.summarize:
        results, starts = load_results(), load_start()
        race = None
        if (OUT / "s7_race.json").exists():
            race = json.loads((OUT / "s7_race.json").read_text())
        summary = summarize(results, starts, race)
        _write_json(OUT / "s7_round.json", {"summary": summary, "arms": results})
        write_figure(results, starts, summary, OUT / "s7_round.png")
        print(json.dumps({k: summary[k] for k in ("hsafe", "hypotheses", "rc19", "arms")}, indent=2, default=str))
        print(f"\n{summary['direction']}")
    if args.race_start:
        race_start(args.device); return
    for arm in args.race_arms:
        race_arm(arm, args.device)
    if args.race_summarize:
        arms = {p_.stem.removeprefix("s7race_arm_"): json.loads(p_.read_text()) for p_ in sorted(OUT.glob("s7race_arm_*.json"))}
        r = race_summary(arms)
        _write_json(OUT / "s7_race.json", r)
        print(json.dumps(r, indent=2, default=str))
        results, starts = load_results(), load_start()
        summary = summarize(results, starts, r)
        _write_json(OUT / "s7_round.json", {"summary": summary, "arms": results})
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
```

Notes: `race_arm` names come from `S6.ARMS` after `race_setup`, so `--race-arms` validates against them inside `S6.run_arm` (`ARMS[arm]` KeyError → add an explicit `SystemExit` if `arm not in S6.ARMS`). `gate_pursuer` uses `shield_obs=True` because every S7 pursuer and RC19 are shield-aware.

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s7_round.py -q`
Expected: all PASS.

- [ ] **Step 5: Smoke in scratch, then the real V1-E preflight**

Scratch smoke: monkeypatch `OUT`, `START_PATH`, `PREFLIGHT_PATH`, `RACE_START_PATH` to scratch; `PREFLIGHT_N=5`, `GATE_N=6`, `EVAL_N=4`, `TOTAL_LONG=4000`, `EVERY_LONG=2000`; run `preflight`, `measure_start`, `run_arm("RCM28")`, `summarize(load_results(), load_start())`, `write_figure`; then (stage 2 at the same tiny budget) `race_start`, `race_arm("ATK2-31")` (with `S6` overrides ALSO pointed at scratch — `race_setup` sets `S6.OUT = OUT`, which the monkeypatch already redirected), `race_summary`. Expected: no exception, 0 violations, `s7_*`/`s7race_*` artefacts only in scratch. Then the REAL preflight: `OMP_NUM_THREADS=2 ./venv/bin/python scripts/derisk_s7_remedy2.py --preflight` → `results/derisk/s7_preflight.json` `pass: true` (≈ 8 min: each episode runs two policies + the QP). Paste both outputs into your report.

- [ ] **Step 6: Lint, full suite, commit**

```bash
./venv/bin/ruff format scripts/derisk_s7_remedy2.py tests/pe2/test_derisk_s7_round.py && ./venv/bin/ruff check scripts/derisk_s7_remedy2.py tests/pe2/test_derisk_s7_round.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q -x
git add scripts/derisk_s7_remedy2.py tests/pe2/test_derisk_s7_round.py
git commit -m "feat(derisk): S7 runner — mixture-trained shield-aware pursuer, held-out attacker gate vs RC19, stage-2 race" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

## Self-review

- **Spec coverage:** §2 components → Task 1; V1-E → Task 1 test + Task 2 `--preflight`; §3 learner/mixture/probes/gate/reference → Task 2 `mixture_factory`, `run_train_arm`, `gate_pursuer`, `measure_start`; §4 deadlock test direction (RC19 more), ≥2/3 per seed, ≥2/3 seeds, capture guard label, artefact rule, INCOMPLETE, H-SAFE incl. reference → `h32_verdicts`, `hsafe_verdict`; §5 stage 2 via S6 machinery with `D3` = artefact, `RACE_BAR`, pooled Wilson → `race_*`; §6 direction → `direction`; stop rule → guards; §7 artefacts → paths.
- **Placeholders:** none.
- **Type consistency:** gate records `gate[o].{deadlock_ind, capture_ind, capture_rate, pursuer, pursuer_violations}` used identically in `measure_start` (RC19), `run_train_arm`, `h32_verdicts`, `summarize`, tests' `_gate`; race arms use S6's record shape (`final.deadlock_ind`, `final.pursuer.deadlock_episode_rate/escape_rate`, `final.pursuer_violations`) as `race_summary` and its test read.
