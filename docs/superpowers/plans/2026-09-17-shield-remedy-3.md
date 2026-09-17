# Shield Remedy, Iteration 3 (S8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Weighted mixture sampling, an anchored fine-tune of RC19 on `SearchMDP`, and the S8 runner: V-A0 preflight, RC19 + RCM28 references on a fresh gate stream, seven arms, the two-sided H4 verdict, artefact rule, and the stage-2 race.

**Architecture:** `MixtureOpponentFactory` gains an optional `weights`. `scripts/derisk_s8_remedy3.py` imports S7's runner via importlib and reuses `gate_pursuer`, `structured_evader_factory`, `splithug_factory`, `gate_opponent_factory`, `ShieldedStructuredProbe`, `provenance`, `_write_json`, `_ci`, `_model`, and the S6 race machinery through S7's `_s6`/`race_setup`, overriding only what S8 changes (paths, prefixes, streams, seeds, mixture weights, arms, learner construction, verdicts).

**Tech Stack:** as S7.

**Spec:** `docs/pathway/shield_remedy_3_prereg_2026-09-17.md` (binding; read it first).

## Global Constraints

- Always `./venv/bin/python`; tests `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest <files> -q`; ruff format + check clean.
- Pre-registered, never tunable: `SEEDS = (34, 35, 36)`, `BETAS = {"FT-A1": 0.1, "FT-A2": 0.01}`, control `SCR-W34` (scratch, seed 34), `WEIGHTS = (0.50, 0.20, 0.10, 1/15, 1/15, 1/15)` over `MIXTURE = ("GREEDY", "HUG7", "SPLITHUG", "CTL-D125", "CTL-D126", "CTL-D127")`, `TOTAL = 5_000_000`, `EVERY = 500_000`, `PROBE_SEED0 = 1_210_000`, `GATE_SEED0 = 1_200_000`, `GATE_N = 500`, `EVAL_N = 200`, `PASS_MIN = 2`, `P_GATE = 0.05`, RC config (`shield_obs=True`, `stall_penalty=1e-3`, `r_acq=0.0`, `shield=CBFShield(cfg)`, γ 0.99, hold 10), init `RC19 = results/derisk/ckpts_RC19/RC19_10000000.zip`, second reference `RCM28 = results/derisk/ckpts_RCM28/RCM28_10000000.zip`; stage 2 `SEEDS2 = (37, 38, 39)`, `PROBE2_SEED0 = 1_220_000`, `GATE2_SEED0 = 1_230_000`, `RACE_BAR = 0.58`, `S7_RACE_RATE = 0.147`.
- V-A0: `KLAnchoredPPO.load(RC19, env=SearchMDP(...), seed=34, device="cpu")` with `beta=0` and `set_anchor(model.policy)`, trained for one rollout (`n_steps=2048` decisions, SB3 default) must end with policy parameters bit-identical to `PPO.load(RC19, env=<fresh identical SearchMDP>, seed=34, device="cpu")` trained for one rollout; compare every parameter tensor with `torch.equal`, and `num_timesteps`. The two envs must be constructed identically (same mixture seed, same `seed0`).
- H-SAFE counts pursuer contact steps in every arm's training, probes and gates, both references, and stage 2 (train, probe, gate, comparators).
- Never write under `results/derisk/` except the REAL `--preflight` at the end of Task 2 (`s8_preflight.json` only); tests and smokes redirect every path into scratch. Never touch S7's artefacts.
- Commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

### Task 1: weighted `MixtureOpponentFactory`

**Files:**
- Modify: `pe2/evade.py` (`MixtureOpponentFactory`)
- Test: `tests/pe2/test_evade.py` (append)

**Interfaces:**
- Produces: `MixtureOpponentFactory(members, seed, weights=None)`; with `weights=None` the draw is `rng.integers(len(members))` exactly as before (bit-identical draws for a given seed); with weights, `rng.choice(len(members), p=w)` where `w = weights / sum(weights)`; `ValueError` on a length mismatch or a non-positive weight; attribute `.weights` (normalised tuple or `None`).

- [ ] **Step 1: Write the failing tests**

```python
class TestMixtureWeights:
    def test_uniform_draws_unchanged_when_weights_none(self):
        from pe2.evade import MixtureOpponentFactory
        from pe2.scripted import GreedyEvader, ObstacleHuggingEvader

        f = MixtureOpponentFactory([GreedyEvader, ObstacleHuggingEvader], seed=5)
        draws = [type(f()).__name__ for _ in range(50)]
        g = MixtureOpponentFactory([GreedyEvader, ObstacleHuggingEvader], seed=5, weights=None)
        assert [type(g()).__name__ for _ in range(50)] == draws
        rng = np.random.default_rng(5)
        assert f.draws == [int(rng.integers(2)) for _ in range(50)]

    def test_weighted_draws(self):
        from pe2.evade import MixtureOpponentFactory
        from pe2.scripted import FovLimitedEvader, GreedyEvader, ObstacleHuggingEvader

        f = MixtureOpponentFactory([GreedyEvader, ObstacleHuggingEvader, FovLimitedEvader], seed=1, weights=(5, 4, 1))
        kinds = [type(f()).__name__ for _ in range(1000)]
        c = {k: kinds.count(k) for k in ("GreedyEvader", "ObstacleHuggingEvader", "FovLimitedEvader")}
        assert 440 <= c["GreedyEvader"] <= 560 and 340 <= c["ObstacleHuggingEvader"] <= 460 and 60 <= c["FovLimitedEvader"] <= 140
        assert f.weights == pytest.approx((0.5, 0.4, 0.1))
        g = MixtureOpponentFactory([GreedyEvader, ObstacleHuggingEvader, FovLimitedEvader], seed=1, weights=(5, 4, 1))
        assert [type(g()).__name__ for _ in range(1000)] == kinds

    def test_bad_weights(self):
        from pe2.evade import MixtureOpponentFactory
        from pe2.scripted import GreedyEvader, ObstacleHuggingEvader

        with pytest.raises(ValueError):
            MixtureOpponentFactory([GreedyEvader, ObstacleHuggingEvader], seed=1, weights=(1,))
        with pytest.raises(ValueError):
            MixtureOpponentFactory([GreedyEvader, ObstacleHuggingEvader], seed=1, weights=(1, 0))
```

- [ ] **Step 2: Run to verify failure** — `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_evade.py -k MixtureWeights -q` → TypeError (unexpected kwarg).

- [ ] **Step 3: Implement**

```python
class MixtureOpponentFactory:
    """Per-episode opponent sampler for training (S7/S8): each call draws one
    member factory — uniformly (`weights=None`, S7) or with the given sampling
    weights (S8) — with a seeded RNG and returns a fresh opponent. ..."""

    def __init__(self, members, seed: int, weights=None):
        self.members = list(members)
        self.rng = np.random.default_rng(seed)
        self.draws: list[int] = []
        if weights is None:
            self.weights = None
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (len(self.members),):
                raise ValueError(f"weights has shape {w.shape}, expected ({len(self.members)},)")
            if np.any(w <= 0):
                raise ValueError("every weight must be positive")
            self.weights = tuple(float(x) for x in w / w.sum())

    def __call__(self):
        if self.weights is None:
            i = int(self.rng.integers(len(self.members)))
        else:
            i = int(self.rng.choice(len(self.members), p=self.weights))
        self.draws.append(i)
        return self.members[i]()
```

- [ ] **Step 4: Run the whole test file** — all pass (the S7 tests included).
- [ ] **Step 5: Lint and commit** — `git commit -m "feat(pe2): weighted MixtureOpponentFactory (S8)"`.

---

### Task 2: `scripts/derisk_s8_remedy3.py`

**Files:**
- Create: `scripts/derisk_s8_remedy3.py`
- Test: `tests/pe2/test_derisk_s8_round.py`

**Interfaces:**
- Consumes: `scripts/derisk_s7_remedy2.py` loaded via importlib as `S7` (`gate_pursuer(model, opponents, device, holder=None)`, `structured_evader_factory(ckpt, device)`, `splithug_factory(device)`, `gate_opponent_factory(name, device)` — uses S7's `HELD_OUT_CKPT` for ATT names, `ShieldedStructuredProbe`, `provenance`, `_write_json`, `_ci`, `_model`, `ATT_CKPT`, `HELD_OUT_CKPT`, `RC19_CKPT`, `_s6`, `race_setup(S6, artefact_ckpt)`, `race_summary(arms, race_start=None, before=None)`); `pe2.anchored_ppo.KLAnchoredPPO`; `pe2.derisk.paired_indicator_test`, `gate_pass`; `pe2.evade.MixtureOpponentFactory`.
- Produces: CLI `--preflight | --measure-start | --arms <names> | --summarize | --race-start | --race-arms | --race-summarize [--device]`; module-level constants above; `ARMS` (7 entries: `{"seed", "family": "FT-A1"|"FT-A2"|"SCR", "beta": float|None}`), `h4_verdicts(results, starts)`, `hsafe_verdict(results, starts, race=None)`, `direction(h)`, `summarize`, `write_figure`.

- [ ] **Step 1: Write the failing tests** (`tests/pe2/test_derisk_s8_round.py`; same `_ind/_gate` helpers as S7's tests):

```python
R = _runner()  # importlib of scripts/derisk_s8_remedy3.py

def _starts(att=(0.63, 0.62, 0.62), greedy=0.55):
    g = lambda d, c: _gate(d, c)
    return {"RC19": {"ATT25": g(att[0], 0.12), "ATT26": g(att[1], 0.12), "ATT27": g(att[2], 0.10), "GREEDY": g(0.33, greedy), "HUG7": g(0.33, 0.41)},
            "RCM28": {"ATT25": g(0.10, 0.10), "ATT26": g(0.09, 0.09), "ATT27": g(0.10, 0.10), "GREEDY": g(0.04, 0.13), "HUG7": g(0.05, 0.15)}}

def _arm(name, att, greedy_capture):
    return {"arm": name, "seed": R.ARMS[name]["seed"], "family": R.ARMS[name]["family"], "train_violations": 0, "probe_violations": 0,
            "final": {"pursuer_violations": 0, "gate": {"ATT25": _gate(att[0], 0.1), "ATT26": _gate(att[1], 0.1), "ATT27": _gate(att[2], 0.1),
                                                          "GREEDY": _gate(0.2, greedy_capture), "HUG7": _gate(0.2, 0.35)}}, "checkpoints": []}

def _full(primary=((0.30, 0.32, 0.31), (0.28, 0.30, 0.29), (0.60, 0.61, 0.62)), primary_cap=(0.54, 0.53, 0.55),
          secondary=((0.20, 0.22, 0.21),) * 3, secondary_cap=(0.30, 0.31, 0.29), ctl=(0.10, 0.09, 0.10), ctl_cap=0.13):
    r = {}
    for s, a, c in zip(R.SEEDS, primary, primary_cap, strict=True): r[f"FT-A1-{s}"] = _arm(f"FT-A1-{s}", a, c)
    for s, a, c in zip(R.SEEDS, secondary, secondary_cap, strict=True): r[f"FT-A2-{s}"] = _arm(f"FT-A2-{s}", a, c)
    r["SCR-W34"] = _arm("SCR-W34", ctl, ctl_cap)
    return r

class TestProtocol:
    def test_constants(self):
        assert R.SEEDS == (34, 35, 36) and R.BETAS == {"FT-A1": 0.1, "FT-A2": 0.01}
        assert R.TOTAL == 5_000_000 and R.EVERY == 500_000 and (R.PROBE_SEED0, R.GATE_SEED0) == (1_210_000, 1_200_000)
        assert R.WEIGHTS == pytest.approx((0.5, 0.2, 0.1, 1 / 15, 1 / 15, 1 / 15)) and R.MIXTURE == ("GREEDY", "HUG7", "SPLITHUG", "CTL-D125", "CTL-D126", "CTL-D127")
        assert set(R.ARMS) == {"FT-A1-34", "FT-A1-35", "FT-A1-36", "FT-A2-34", "FT-A2-35", "FT-A2-36", "SCR-W34"}
        assert R.ARMS["FT-A1-34"] == {"seed": 34, "family": "FT-A1", "beta": 0.1} and R.ARMS["SCR-W34"] == {"seed": 34, "family": "SCR", "beta": None}
        assert (R.SEEDS2, R.PROBE2_SEED0, R.GATE2_SEED0, R.RACE_BAR, R.S7_RACE_RATE) == ((37, 38, 39), 1_220_000, 1_230_000, 0.58, 0.147)
    def test_guards(self, tmp_path, monkeypatch): ...  # as S7: preflight missing, measure-start missing, arm already run, start lacking RCM28 → SystemExit

class TestVerdicts:
    def test_h4_two_sided_primary_passes(self):
        h = R.h4_verdicts(_full(), _starts())
        assert h["per_arm"]["FT-A1-34"]["verdict"] == "PASS" and h["per_arm"]["FT-A1-36"]["verdict"] == "FAIL (still pinned)"
        assert h["per_arm"]["FT-A2-34"]["verdict"] == "FAIL (capture lost)" and h["per_arm"]["SCR-W34"]["verdict"] == "FAIL (capture lost)"
        assert h["H4"] == "PASS" and h["artefact"] == "FT-A1-35"   # lowest mean held-out deadlock among passing primaries
    def test_secondary_rescue(self):
        h = R.h4_verdicts(_full(primary_cap=(0.30, 0.30, 0.30), secondary=((0.30, 0.31, 0.32),) * 3, secondary_cap=(0.54, 0.54, 0.54)), _starts())
        assert h["H4"] == "PASS (β = 0.01)" and h["artefact"].startswith("FT-A2-")
    def test_fail_and_incomplete(self):
        assert R.h4_verdicts(_full(primary_cap=(0.3, 0.3, 0.3)), _starts())["H4"] == "FAIL"
        res = _full(); del res["FT-A1-35"]; assert R.h4_verdicts(res, _starts())["H4"] == "INCOMPLETE"
    def test_artefact_tiebreak_capture_then_seed(self):
        h = R.h4_verdicts(_full(primary=((0.30, 0.30, 0.30),) * 3, primary_cap=(0.53, 0.55, 0.55)), _starts())
        assert h["artefact"] == "FT-A1-35"

class TestSafety: ...   # as S7, plus references RC19 and RCM28 both counted, plus race
class TestDirection: ... # rows of spec §6 incl. "capture lost" vs "still pinned" wording
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Write the runner.** Load S7 with importlib (`_s7()`), then:

```python
SEEDS = (34, 35, 36); BETAS = {"FT-A1": 0.1, "FT-A2": 0.01}
ARMS = {**{f"{fam}-{s}": {"seed": s, "family": fam, "beta": b} for fam, b in BETAS.items() for s in SEEDS}, "SCR-W34": {"seed": 34, "family": "SCR", "beta": None}}
MIXTURE = S7.MIXTURE; WEIGHTS = (0.50, 0.20, 0.10, 1/15, 1/15, 1/15)
TOTAL, EVERY, PROBE_SEED0, GATE_SEED0, GATE_N, EVAL_N, PASS_MIN, P_GATE = 5_000_000, 500_000, 1_210_000, 1_200_000, 500, 200, 2, 0.05
SEEDS2, PROBE2_SEED0, GATE2_SEED0, RACE_BAR, S7_RACE_RATE = (37, 38, 39), 1_220_000, 1_230_000, 0.58, 0.147
RC19_CKPT = S7.RC19_CKPT; RCM28_CKPT = OUT / "ckpts_RCM28" / "RCM28_10000000.zip"
```

Key functions:
- `mixture_factory(seed, device)` — S7's members in `MIXTURE` order with `weights=WEIGHTS`.
- `make_env(seed, device)` — `SearchMDP(cfg, mixture_factory(seed, device), seed0=seed*1000, hold=HOLD, r_acq=0.0, zero_reward=False, shield=CBFShield(cfg), shield_obs=True, stall_penalty=1e-3)`.
- `make_learner(arm, env, device)` — `SCR`: `PPO("MlpPolicy", env, seed=seed, gamma=0.99, verbose=0, device=device)`; `FT-*`: `m = KLAnchoredPPO.load(RC19_CKPT, env=env, seed=seed, device=device); m.beta = beta; m.set_anchor(m.policy); return m`. (Check `KLAnchoredPPO.load` sets `self.beta` default 0 and that `set_anchor` deep-copies — `pe2/anchored_ppo.py`.)
- `preflight(device)` — V-A0 exactly as Global Constraints: two identical envs (`make_env(34, device)` each — the mixture factory RNG is per-env so both see the same opponent order), anchored β=0 vs plain PPO, `learn(total_timesteps=2048)` each with NO callback, compare `num_timesteps` and every `state_dict()` tensor with `torch.equal`; write `s8_preflight.json` `{pass, n_params, mismatched: [...]}`; SystemExit on mismatch.
- `gate_pursuer` must use S8's `GATE_SEED0`: S7's function reads its own module global — set `S7.GATE_SEED0 = GATE_SEED0` and `S7.GATE_N = GATE_N` at import time in `_s7()` (and `S7.OUT = OUT` for path safety), and document that in a comment.
- `measure_start(device)` — both references: `{"RC19": gate_pursuer(RC19 model,...), "RCM28": gate_pursuer(RCM28 model, ...)}` on `GATE_OPPONENTS = S7.GATE_OPPONENTS`.
- `run_train_arm(arm, starts, device, holder)` — S7's structure with `make_env`/`make_learner`, S7's `ShieldedStructuredProbe(cfg, GreedyEvader, every=EVERY, total=TOTAL, eval_n=EVAL_N, seed0=PROBE_SEED0, ckpt_dir, report_path=OUT/f"s8_probe_{arm}.json", label=arm)`; record `anchor_kl` trace if the learner exposes `last_anchor_kl` (append per probe: read `model.last_anchor_kl`); final gate; per-arm verdict string with both tests.
- `h4_verdicts(results, starts)` — per arm: deadlock tests vs RC19 on `HELD_OUT`, capture guard = `gate_pass(rc19_logs, s8_logs, "pursuer", P_GATE)["pass"]` (RC19 significantly better ⇒ guard fails); verdict `PASS` / `FAIL (capture lost)` (deadlock ok, guard failed) / `FAIL (still pinned)` (deadlock < 2/3, guard ok) / `FAIL (both)`; H4 from primary (≥ PASS_MIN of 3 → `PASS`), else secondary (≥ PASS_MIN → `PASS (β = 0.01)`), else `FAIL`; `INCOMPLETE` if any of the 7 arms lacks `final`; artefact rule per spec §4 (mean held-out deadlock ascending, Greedy capture descending, seed ascending) among passing arms of the family that passed.
- `hsafe_verdict(results, starts, race=None)` — as S7 plus `starts["RCM28"]`.
- `direction(h)` — spec §6 rows.
- `summarize`, `write_figure` (three reference bars RC19/RCM28 + 7 arms; two panels deadlock/capture as S7), race via `S7._s6()` + `S7.race_setup(S6, artefact)` then override `S6.START_PATH = OUT/"s8_race_start.json"`, `S6.ARTEFACT_PREFIX = "s8race"`, `S6.PROBE_SEED0 = PROBE2_SEED0`, `S6.GATE_SEED0 = GATE2_SEED0`, `S6.SEEDS = SEEDS2`, `S6.ARMS = {f"ATK3-{s}": {...CTL...} for s in SEEDS2}` and rebind `S6.load_start` to read the S8 race start; `race_summary` = `S7.race_summary(arms, race_start, before)` plus `"s7_race_rate": S7_RACE_RATE` and `"below_s7": pooled_rate < S7_RACE_RATE`.

- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Scratch smoke** (all paths redirected; `TOTAL=4096`, `EVERY=2048`, `GATE_N=6`, `EVAL_N=4`; one FT arm and the SCR arm; summarize; race-start + one race arm at tiny budget; race-summarize), then the REAL `OMP_NUM_THREADS=2 ./venv/bin/python scripts/derisk_s8_remedy3.py --preflight` (writes only `s8_preflight.json`). Paste outputs in the report.
- [ ] **Step 6: Lint, full `tests/pe2` suite, commit.**

## Self-review
- Spec §2 components → T1 (weights), T2 (`make_learner`, V-A0 preflight); §3 protocol → T2 constants/`mixture_factory`/`make_env`/arms/probe/gate/references; §4 verdicts → `h4_verdicts` (two-sided, families, artefact tiebreaks, INCOMPLETE), `hsafe_verdict` with both references; §5 → race overrides; §6 → `direction`; §7 → paths.
- Type consistency: gate record shape identical to S7 (`gate_pursuer` reused); `starts` has two keys; tests' `_gate` matches.
