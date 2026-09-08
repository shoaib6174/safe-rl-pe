# T1.0 Horizon Round Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the recorded learning failure is caused by the credit-assignment horizon rather than the learner — by running one fixed learner at a 400-step episode, at the original 1,200-step episode as a matched control, at 400 with a matched discount, and with a negated reward as the negative control.

**Architecture:** One small addition to `pe2` (a reward-negating training wrapper for the control) plus a new runner modelled on `scripts/derisk_t0c_round.py`. Nothing about the learner changes: every arm calls the existing `finetune_br` with the proven recipe, differing only in `EnvConfig.max_steps`, `gamma`, and (for the control) the reward sign.

**Tech Stack:** Python 3.10+, numpy, gymnasium, stable-baselines3 2.7.1, torch (CUDA on niro-1), matplotlib, pytest, ruff.

**Spec:** `docs/pathway/t1_horizon_round_prereg_2026-09-08.md` — §2 (protocol), §3 (arms), §5 (direction) are binding. §0 records the user's override of D101 §5. Decisions: D83, D102, D103.

## Global Constraints

- Always `./venv/bin/python`; tests `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest <target> -q`. Full `tests/pe2` takes ~13 min; run it once, in Task 3.
- Lint only touched files (`./venv/bin/ruff check` + `ruff format --check`).
- **Pre-registered constants, verbatim from spec §2/§3 — never tuned:** `TOTAL = 1_000_000`, `EVERY = 100_000`, `EVAL_N = 200`, `FINAL_EVAL_N = 500`, `PASS_FROM = 500_000`, `CRITIC_WARMUP = 50_000`, `CLONE_CI_LO = 0.137` (existing), `SHORT_STEPS = 400`, `LONG_STEPS = 1200`, `GAMMA_ORIG = 0.999`, `GAMMA_MATCHED = 0.9975`, `SEED = 7` (every arm), `SHORT_SEED0 = 870_000`, `LONG_SEED0 = 880_000`.
- **Every arm uses the same learner**: `pe2.exploitability.finetune_br` with `critic_warmup=CRITIC_WARMUP`. No arm varies the mechanism — that is the round's whole design.
- **Two start bars, measured before any training**, into `results/derisk/t1_start.json`; an arm that cannot read its own bar refuses to run.
- **Evaluation is on true outcomes only.** The reward negation of the control is a TRAINING wrapper; the probe and the final evaluation must see the unmodified environment. A test must pin this.
- Verdict rules are the existing `pe2.derisk.round_verdict` / `control_collapsed`; do not reimplement them.
- `results/derisk/t0*_*` and `results/visual_gates/**` are read-only evidence. `scripts/derisk_t0b_round.py` and `scripts/derisk_t0c_round.py` are frozen artefacts: do not edit them.
- Never train in a test: tiny budgets (≤ 3,000 steps), `pytest.importorskip`.
- Branch `t1-horizon-round` from `main`. Commit only the files each task names. Trailer: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

---

## File map

| file | responsibility |
|---|---|
| `pe2/rewards.py` | `NegatedRewardWrapper` — training-only reward sign flip for the negative control (Task 1) |
| `tests/pe2/test_rewards.py` | its tests, including that it never reaches an eval path (Task 1) |
| `scripts/derisk_t1_round.py` | constants, `arm_config`, `measure_starts`, `load_starts`, `run_arm`, `summarize`, `write_figure`, CLI (Task 2) |
| `tests/pe2/test_derisk_t1_round.py` | the runner's decisions (Task 2) |
| `docs/worklogs/2026-09-08_S124.md`, `docs/decisions.md`, `docs/workflow_tracker.md` | Task 3 record |

---

### Task 1: `NegatedRewardWrapper`

**Files:**
- Modify: `pe2/rewards.py` (append; do not touch `DenseRewardWrapper` or `SearchStalenessTracker`)
- Test: `tests/pe2/test_rewards.py` (append)

**Interfaces:**
- Produces: `NegatedRewardWrapper(env)` — a `gymnasium.Wrapper` whose `step` returns the environment's reward multiplied by −1, everything else untouched. Exposes `info["true_reward"]` carrying the unnegated value.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pe2/test_rewards.py`:

```python
class TestNegatedRewardWrapper:
    """T1.0 spec §3's negative control: the pursuer is paid for NOT capturing.

    Horizon-independent by construction, which is why it is the control for a
    round whose hypothesis is about the horizon — the usual naive-fine-tune
    control might legitimately SUCCEED here if the hypothesis is true, and a
    control the hypothesis predicts may pass is no control at all.
    """

    def _wrapped(self, max_steps=40):
        from pe2.config import EnvConfig
        from pe2.exploitability import SingleAgentWrapper
        from pe2.rewards import NegatedRewardWrapper
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=max_steps)
        return NegatedRewardWrapper(SingleAgentWrapper(cfg, "pursuer", GreedyEvader))

    def test_reward_sign_is_flipped_and_the_true_value_is_kept(self):
        import numpy as np

        env = self._wrapped()
        env.reset(seed=970_000)
        seen = []
        while True:
            _obs, reward, terminated, truncated, info = env.step(
                np.array([1.0, 0.0], dtype=np.float32)
            )
            seen.append((reward, info["true_reward"]))
            if terminated or truncated:
                break
        assert all(r == -t for r, t in seen)
        # A terminal-only env pays exactly once, at the end.
        assert seen[-1][1] != 0.0
        assert all(t == 0.0 for _, t in seen[:-1])

    def test_everything_but_the_reward_passes_through(self):
        import numpy as np

        from pe2.config import EnvConfig
        from pe2.exploitability import SingleAgentWrapper
        from pe2.rewards import NegatedRewardWrapper
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=40)
        plain = SingleAgentWrapper(cfg, "pursuer", GreedyEvader)
        wrapped = NegatedRewardWrapper(SingleAgentWrapper(cfg, "pursuer", GreedyEvader))
        a = np.array([0.6, -0.2], dtype=np.float32)
        o1, _ = plain.reset(seed=970_001)
        o2, _ = wrapped.reset(seed=970_001)
        assert np.array_equal(o1, o2)
        for _ in range(20):
            s1 = plain.step(a)
            s2 = wrapped.step(a)
            assert np.array_equal(s1[0], s2[0])  # obs
            assert s1[2] == s2[2] and s1[3] == s2[3]  # terminated / truncated
            assert s1[1] == -s2[1]  # reward, and only the reward
            if s1[2] or s1[3]:
                break

    def test_it_is_a_training_wrapper_only(self):
        """The eval path builds its own unshaped wrapper; a negated reward must
        never reach a probe or the final evaluation."""
        import inspect

        from pe2 import exploitability

        for fn in (exploitability.evaluate_sb3_policy,):
            assert "Negated" not in inspect.getsource(fn)
        assert "Negated" not in inspect.getsource(exploitability.CheckpointEvalProbe)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_rewards.py::TestNegatedRewardWrapper -q`
Expected: FAIL, `ImportError: cannot import name 'NegatedRewardWrapper'`.

- [ ] **Step 3: Implement**

Append to `pe2/rewards.py`:

```python
class NegatedRewardWrapper(gym.Wrapper):
    """TRAINING-ONLY reward sign flip — the T1.0 round's negative control.

    The pursuer is paid for NOT capturing, so a working learner must drive
    the capture rate down. It is horizon-independent by construction, which
    is exactly why this round uses it: T1.0's hypothesis is that the naive
    configurations failed because of the credit-assignment horizon, so the
    usual D78 naive-fine-tune control might legitimately SUCCEED at the
    short horizon, and a control the hypothesis predicts may pass validates
    nothing.

    `info["true_reward"]` carries the unnegated value so a diagnostic can
    still read what the environment actually paid. Evaluation never uses
    this wrapper: `evaluate_sb3_policy` and `CheckpointEvalProbe` build
    their own unshaped `SingleAgentWrapper`, so the probes and the final
    N=500 measurement always see true outcomes (D68's rule).
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info, true_reward=reward)
        return obs, -reward, terminated, truncated, info
```

`gym` is already imported in that module (check the top of the file; if it is imported as `import gymnasium as gym`, use that name).

- [ ] **Step 4: Run**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_rewards.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
./venv/bin/ruff check pe2/rewards.py tests/pe2/test_rewards.py && ./venv/bin/ruff format --check pe2/rewards.py tests/pe2/test_rewards.py
git add pe2/rewards.py tests/pe2/test_rewards.py
git commit -m "feat(rewards): NegatedRewardWrapper for the T1.0 negative control" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The T1.0 runner `scripts/derisk_t1_round.py`

**Files:**
- Create: `scripts/derisk_t1_round.py`
- Test: `tests/pe2/test_derisk_t1_round.py`

**Interfaces:**
- Consumes: `pe2.derisk.{round_verdict, control_collapsed, CLONE_CI_LO, TrainingCaptureCounter}`, `pe2.eval.{wilson_ci, win_count}`, `pe2.exploitability.{CheckpointEvalProbe, finetune_br, evaluate_sb3_policy}`, `pe2.rewards.NegatedRewardWrapper`, `pe2.scripted.GreedyEvader`.
- Produces: `arm_config(arm) -> (EnvConfig, gamma, seed0, bar_key)`; `measure_starts(device) -> dict`; `load_starts() -> dict`; `run_arm(arm, device) -> dict`; `load_results()`; `summarize(results) -> dict`; `write_figure(results, summary, path)`; CLI `--measure-starts` / `--arms` / `--summarize` / `--device`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pe2/test_derisk_t1_round.py`:

```python
"""scripts/derisk_t1_round.py — the horizon round's own decisions."""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_t1_round.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_t1_round", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _arm(name, verdict, final=(120, 500), bar=0.179, steps=400):
    return {
        "arm": name, "verdict": verdict, "reason": "test", "timesteps": 1_000_000,
        "max_steps": steps, "gamma": 0.999,
        "checkpoints": [
            {"steps": s, "win_rate": 0.2, "k": 40, "n": 200}
            for s in range(100_000, 1_000_001, 100_000)
        ],
        "final": {
            "k": final[0], "n": final[1], "win_rate": final[0] / final[1],
            "wilson": list(R.wilson_ci(final[0], final[1])),
        },
        "start": {"k": 90, "n": 500, "rate": bar, "bar": "START_400"},
    }


class TestPreRegisteredConstants:
    def test_protocol(self):
        assert (R.TOTAL, R.EVERY, R.EVAL_N, R.FINAL_EVAL_N, R.PASS_FROM) == (
            1_000_000, 100_000, 200, 500, 500_000,
        )
        assert R.CRITIC_WARMUP == 50_000 and R.SEED == 7

    def test_horizons_and_discounts(self):
        assert (R.SHORT_STEPS, R.LONG_STEPS) == (400, 1200)
        assert (R.GAMMA_ORIG, R.GAMMA_MATCHED) == (0.999, 0.9975)
        # The matched discount sets the effective horizon to the episode length.
        assert 1.0 / (1.0 - R.GAMMA_MATCHED) == pytest.approx(R.SHORT_STEPS, rel=1e-6)

    def test_fresh_seed_streams(self):
        assert (R.SHORT_SEED0, R.LONG_SEED0) == (870_000, 880_000)
        assert R.SHORT_SEED0 > 860_999 and R.LONG_SEED0 > 860_999

    def test_pass_window_is_bound_to_the_library_constant(self):
        from pe2.derisk import ROUND_PASS_FROM

        assert R.PASS_FROM == ROUND_PASS_FROM


class TestArmConfig:
    """Spec §3: every arm shares the learner and differs only in the task."""

    def test_ha_is_the_short_horizon_at_the_original_discount(self):
        cfg, gamma, seed0, bar = R.arm_config("HA")
        assert (cfg.max_steps, gamma, seed0, bar) == (400, 0.999, 870_000, "START_400")

    def test_hc_is_the_matched_control_at_the_original_length(self):
        cfg, gamma, seed0, bar = R.arm_config("HC")
        assert (cfg.max_steps, gamma, seed0, bar) == (1200, 0.999, 880_000, "START_1200")

    def test_hb_adds_the_matched_discount(self):
        cfg, gamma, seed0, bar = R.arm_config("HB")
        assert (cfg.max_steps, gamma, seed0, bar) == (400, 0.9975, 870_000, "START_400")

    def test_the_control_matches_ha_exactly(self):
        ha = R.arm_config("HA")
        neg = R.arm_config("HNEG")
        assert (neg[0].max_steps, neg[1], neg[2], neg[3]) == (
            ha[0].max_steps, ha[1], ha[2], ha[3]
        )

    def test_only_the_horizon_differs_between_ha_and_hc(self):
        import dataclasses

        a = dataclasses.asdict(R.arm_config("HA")[0])
        c = dataclasses.asdict(R.arm_config("HC")[0])
        assert {k for k in a if a[k] != c[k]} == {"max_steps"}


class TestStartBars:
    def test_arms_refuse_to_run_without_measured_bars(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t1_start.json")
        with pytest.raises(SystemExit, match="start bar"):
            R.load_starts()

    def test_load_starts_reads_both_bars(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t1_start.json")
        payload = {
            "START_400": {"k": 90, "n": 500, "rate": 0.18, "wilson": [0.15, 0.22], "eval_seed0": 870_000},
            "START_1200": {"k": 97, "n": 500, "rate": 0.194, "wilson": [0.16, 0.23], "eval_seed0": 880_000},
        }
        (tmp_path / "t1_start.json").write_text(json.dumps(payload))
        assert R.load_starts() == payload

    def test_a_missing_single_bar_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t1_start.json")
        (tmp_path / "t1_start.json").write_text(json.dumps({"START_400": {}}))
        with pytest.raises(SystemExit, match="START_1200"):
            R.load_starts()


class TestSummarize:
    def _ctrl(self, verdict):
        return {"arm": "HNEG", "verdict": verdict, "reason": "test"}

    def test_the_horizon_finding_names_both_arms(self):
        s = R.summarize({
            "HA": _arm("HA", "IMPROVED"), "HB": _arm("HB", "PRESERVED"),
            "HC": _arm("HC", "COLLAPSED", bar=0.194, steps=1200), "HNEG": self._ctrl(True),
        })
        assert s["survivors"] == ["HA"]
        assert "horizon was the binding constraint" in s["direction"]

    def test_both_horizons_improving_is_reported_as_a_coverage_gap(self):
        s = R.summarize({
            "HA": _arm("HA", "IMPROVED"), "HB": _arm("HB", "PRESERVED"),
            "HC": _arm("HC", "IMPROVED", bar=0.194, steps=1200), "HNEG": self._ctrl(True),
        })
        assert "not what changed" in s["direction"] or "coverage" in s["direction"]

    def test_zero_improved_leaves_route_c_standing(self):
        s = R.summarize({
            "HA": _arm("HA", "COLLAPSED"), "HB": _arm("HB", "PRESERVED"),
            "HC": _arm("HC", "COLLAPSED", bar=0.194, steps=1200), "HNEG": self._ctrl(True),
        })
        assert s["survivors"] == []
        assert "Route C stands" in s["direction"]

    def test_every_arm_is_withheld_when_the_control_does_not_collapse(self):
        s = R.summarize({
            "HA": _arm("HA", "IMPROVED"), "HB": _arm("HB", "IMPROVED"),
            "HC": _arm("HC", "COLLAPSED", bar=0.194, steps=1200), "HNEG": self._ctrl(False),
        })
        assert s["survivors"] == []
        assert all(s["arms"][a]["verdict"] == "UNDETERMINED" for a in ("HA", "HB", "HC"))
        assert "unvalidated" in s["arms"]["HA"]["reason"]

    def test_a_crashed_learner_does_not_reach_a_direction(self):
        s = R.summarize({
            "HA": _arm("HA", "UNDETERMINED"), "HB": _arm("HB", "COLLAPSED"),
            "HC": _arm("HC", "COLLAPSED", bar=0.194, steps=1200), "HNEG": self._ctrl(True),
        })
        assert s["complete"] is False and "incomplete" in s["direction"].lower()


class TestFigure:
    def test_writes_with_partial_arms(self, tmp_path):
        pytest.importorskip("matplotlib")
        results = {"HA": _arm("HA", "PRESERVED")}
        R.write_figure(results, R.summarize(results), tmp_path / "f.png")
        assert (tmp_path / "f.png").stat().st_size > 0


class TestRunArmWiring:
    """Pins that every arm uses the SAME learner and differs only in the task."""

    def test_per_arm_call_arguments(self, tmp_path, monkeypatch):
        pytest.importorskip("stable_baselines3")
        calls = {}
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t1_start.json")
        (tmp_path / "t1_start.json").write_text(json.dumps({
            "START_400": {"k": 90, "n": 500, "rate": 0.18, "wilson": [0.15, 0.22], "eval_seed0": 870_000},
            "START_1200": {"k": 97, "n": 500, "rate": 0.194, "wilson": [0.16, 0.23], "eval_seed0": 880_000},
        }))
        current = {}

        def fake_finetune(cfg, role, opp, path, total, **kw):
            calls[current["arm"]] = {
                "max_steps": cfg.max_steps, "gamma": kw["gamma"], "seed": kw["seed"],
                "critic_warmup": kw["critic_warmup"], "total": total,
                "env_wrap": kw.get("env_wrap"),
            }

        import pe2.exploitability as expl

        monkeypatch.setattr(expl, "finetune_br", fake_finetune)
        monkeypatch.setattr(R, "final_eval", lambda *a, **k: {
            "k": 100, "n": 500, "win_rate": 0.2, "wilson": [0.17, 0.24], "ckpt": "x"})

        for arm in ("HA", "HB", "HC", "HNEG"):
            current["arm"] = arm
            R.run_arm(arm, "cpu")

        assert all(c["seed"] == 7 for c in calls.values())
        assert all(c["critic_warmup"] == 50_000 for c in calls.values())
        assert all(c["total"] == 1_000_000 for c in calls.values())
        assert (calls["HA"]["max_steps"], calls["HA"]["gamma"]) == (400, 0.999)
        assert (calls["HC"]["max_steps"], calls["HC"]["gamma"]) == (1200, 0.999)
        assert (calls["HB"]["max_steps"], calls["HB"]["gamma"]) == (400, 0.9975)
        assert (calls["HNEG"]["max_steps"], calls["HNEG"]["gamma"]) == (400, 0.999)
        # Only the control wraps the training env.
        assert calls["HNEG"]["env_wrap"] is not None
        assert all(calls[a]["env_wrap"] is None for a in ("HA", "HB", "HC"))

    def test_the_control_wrap_negates_the_reward(self, tmp_path, monkeypatch):
        pytest.importorskip("stable_baselines3")
        captured = {}
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t1_start.json")
        (tmp_path / "t1_start.json").write_text(json.dumps({
            "START_400": {"k": 90, "n": 500, "rate": 0.18, "wilson": [0.15, 0.22], "eval_seed0": 870_000},
            "START_1200": {"k": 97, "n": 500, "rate": 0.194, "wilson": [0.16, 0.23], "eval_seed0": 880_000},
        }))

        def fake_finetune(cfg, role, opp, path, total, **kw):
            captured["wrap"] = kw.get("env_wrap")

        import pe2.exploitability as expl

        monkeypatch.setattr(expl, "finetune_br", fake_finetune)
        monkeypatch.setattr(R, "final_eval", lambda *a, **k: {
            "k": 1, "n": 500, "win_rate": 0.002, "wilson": [0.0, 0.01], "ckpt": "x"})
        R.run_arm("HNEG", "cpu")

        from pe2.config import EnvConfig
        from pe2.exploitability import SingleAgentWrapper
        from pe2.rewards import NegatedRewardWrapper
        from pe2.scripted import GreedyEvader

        wrapped = captured["wrap"](
            SingleAgentWrapper(EnvConfig(max_steps=40), "pursuer", GreedyEvader)
        )
        assert isinstance(wrapped, NegatedRewardWrapper)
```

Run it: FAIL (`FileNotFoundError` for the script).

- [ ] **Step 2: Implement the runner**

Create `scripts/derisk_t1_round.py`, modelled on `scripts/derisk_t0c_round.py` — read that file first and follow its structure, docstring discipline and comment style. Duplication is deliberate: the earlier runners are artefacts of their own pre-registrations and are frozen.

Module docstring: state the arms, the hypothesis, the pass and improvement rules, the two bars, and that the direction on 0 IMPROVED is "Route C stands as D102 left it, strengthened". Reference the spec path.

Constants exactly as the Global Constraints list them, plus `OUT = pathlib.Path("results/derisk")`, `CLONE = pathlib.Path("results/visual_gates/step_05/sac_bc_pursuer.zip")`, `START_PATH = OUT / "t1_start.json"`, `LEARNER_ARMS = ("HA", "HB", "HC")`, `DETERMINATE_VERDICTS = ("COLLAPSED", "PRESERVED", "IMPROVED")`.

```python
ARMS = {
    # arm: (max_steps, gamma, eval seed stream, which bar it is judged against)
    "HA": (SHORT_STEPS, GAMMA_ORIG, SHORT_SEED0, "START_400"),
    "HB": (SHORT_STEPS, GAMMA_MATCHED, SHORT_SEED0, "START_400"),
    "HC": (LONG_STEPS, GAMMA_ORIG, LONG_SEED0, "START_1200"),
    "HNEG": (SHORT_STEPS, GAMMA_ORIG, SHORT_SEED0, "START_400"),
}


def arm_config(arm: str):
    """The ONLY thing that differs between arms is the task (and, for the
    control, the training reward's sign). The learner is a constant."""
    steps, gamma, seed0, bar = ARMS[arm]
    return EnvConfig(max_steps=steps), gamma, seed0, bar
```

`measure_starts(device)` evaluates the gated clone twice — `EnvConfig(max_steps=400)` on `SHORT_SEED0` and `EnvConfig(max_steps=1200)` on `LONG_SEED0`, both `FINAL_EVAL_N` episodes, deterministic — and writes both under their bar names with `k`, `n`, `rate`, `wilson`, `eval_seed0`, `max_steps`. `load_starts()` raises `SystemExit` naming the missing bar if either is absent (its message must contain the words "start bar" and the missing key).

`final_eval(cfg, arm, seed0, device)` loads `OUT / f"ckpts_{arm}" / f"{arm}_{TOTAL}.zip"` with `SAC.load` and evaluates on `cfg` at `seed0`, N = `FINAL_EVAL_N`.

`run_arm(arm, device)`: `starts = load_starts()` FIRST; `cfg, gamma, seed0, bar = arm_config(arm)`; probe with `every=EVERY, eval_n=EVAL_N, seed0=seed0`, `ckpt_dir=OUT/f"ckpts_{arm}"`, `report_path=OUT/f"t1_probe_{arm}.json"`; `counter = TrainingCaptureCounter(snapshot_every=CRITIC_WARMUP)`; then

```python
    import pe2.exploitability as expl

    env_wrap = NegatedRewardWrapper if arm == "HNEG" else None
    expl.finetune_br(
        cfg, "pursuer", GreedyEvader, CLONE, TOTAL,
        seed=SEED, device=device, gamma=gamma,
        critic_warmup=CRITIC_WARMUP, callback=cb, env_wrap=env_wrap,
    )
```

Record `max_steps`, `gamma`, `eval_seed0`, `bar`, the start block (`k`, `n`, `rate`, `bar`), `checkpoints`, `training` (counter state + `positives_per_1M` + `segments`), and — per spec §4 — `training["capture_episode_fraction"] = counter.state["captures"] / counter.state["episodes"]` guarding a zero denominator. Verdict: `HNEG` → `control_collapsed(pr.rows, from_step=EVERY)`; others → the crashed-arm guard (missing final checkpoint → `UNDETERMINED` naming the path, arm JSON still written), else `final_eval` then `round_verdict(pr.rows, final["k"], final["n"], starts[bar]["rate"], from_step=PASS_FROM)`. Write `OUT / f"t1_arm_{arm}.json"`.

`summarize(results)`: build `arms` from each result's verdict/reason. If `HNEG` is present and its verdict is not `True`, **withhold every learner arm** (`UNDETERMINED`, reason containing "unvalidated") — unlike the earlier rounds, the control here validates the harness at a NEW episode length, so a failed control invalidates all three learners, not one. If `HNEG` is absent, withhold likewise with a reason saying it has not run. `survivors` = learner arms verdicted `IMPROVED`. `complete` = all four present AND every learner determinate (no withheld exemption in this round — see above). Directions, per spec §5:

- `HA` improved and `HC` did not → `"HA IMPROVED and HC did not: the horizon was the binding constraint. Re-plan Track 2 (belief-augmented evader) and Track 3 (the pair, shielded vs unshielded) at the short horizon, each with its own pre-registration."`
- both `HA` and `HC` improved → `"HA and HC both IMPROVED: the horizon is not what changed — this configuration was simply never run at 1200. Report as a coverage gap in the earlier rounds and re-plan Track 2/3 at the ORIGINAL horizon."`
- `HB` improved and `HA` did not → `"HB IMPROVED where HA did not: shortening alone is insufficient and the matched discount is doing the work. Carry gamma as a matched parameter into Track 2/3."`
- any other non-empty survivor set → name the survivors and say the pattern matches none of the spec's named cases; report it, do not force it.
- no survivors and complete → `"No arm IMPROVED. Route C stands as D102 left it, strengthened: the learning failure now survives reward shaping, algorithm family, anchoring at both strengths, bootstrapping, weighting calibration, AND a threefold reduction in the credit-assignment horizon with a 2.6x denser reward."`
- otherwise → `"incomplete: not every learner arm has a determinate verdict"`.

`write_figure`: curves per arm labelled with verdict, the 0.137 floor, the pass-window line, final-probe error bars, and **both** start bars drawn as distinct horizontal lines labelled `START_400` and `START_1200`; "INCOMPLETE" in the title when not complete.

`main()`: `--measure-starts` runs `measure_starts` and returns; `--arms` runs each (refusing if `CLONE` is absent); `--summarize` writes `t1_round.{json,png}` only.

- [ ] **Step 3: Run the tests**

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_t1_round.py -q` → PASS. Smoke the refusal path: `./venv/bin/python scripts/derisk_t1_round.py --summarize` with no arm files prints an incomplete summary and exits 0; then delete the two files it wrote.

- [ ] **Step 4: Lint and commit**

```bash
./venv/bin/ruff check scripts/derisk_t1_round.py tests/pe2/test_derisk_t1_round.py && ./venv/bin/ruff format --check scripts/derisk_t1_round.py tests/pe2/test_derisk_t1_round.py
git add scripts/derisk_t1_round.py tests/pe2/test_derisk_t1_round.py
git commit -m "feat(derisk): T1.0 horizon-round runner (HA/HB/HC/HNEG, two start bars)" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Run the round, summarise, record

**Files:**
- Create (by the runner): `results/derisk/t1_start.json`, `t1_arm_*.json`, `t1_round.{json,png}`, `ckpts_{HA,HB,HC,HNEG}/`
- Create: `docs/worklogs/2026-09-08_S124.md` (new session; follow `docs/worklogs/TEMPLATE.md` and the Session Protocol in CLAUDE.md)
- Modify: `docs/decisions.md` (D103: the override AND the outcome), `docs/workflow_tracker.md`, and the memory file `~/.claude/projects/-home-niro-1-Codes-safe-rl-pe/memory/pathway-b-plan.md`

- [ ] **Step 1: Full suite once, on the tree that will run**

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q` → all pass (expect ~585). Record the count.

- [ ] **Step 2: Measure both bars, then launch**

```bash
./venv/bin/python scripts/derisk_t1_round.py --measure-starts | tee results/derisk/t1_start.log
for A in HNEG HB HC HA; do
  nohup ./venv/bin/python scripts/derisk_t1_round.py --arms $A > results/derisk/t1_$A.log 2>&1 &
done
```

The bars MUST be written before the arms launch — their file mtime preceding every checkpoint is the auditable evidence for spec §2. Poll `tail -2 results/derisk/t1_*.log` no more than every 10 minutes. Expect ≈ 6 h.

- [ ] **Step 3: Summarise**

`./venv/bin/python scripts/derisk_t1_round.py --summarize`, then report per arm: verdict, reason, every probe, the final N=500 with its interval, its bar, `positives_per_1M`, and the realised `capture_episode_fraction` and mean episode length. **Check the premise held**: HA/HB/HNEG's training capture fraction should be near the truncation table's 15.8–17.9 % and their positives-per-1M near 485, against HC's ~184. If it is not, say so — the round's premise is then not what the audit predicted and that is itself a finding.

- [ ] **Step 4: Record**

Create `docs/worklogs/2026-09-08_S124.md` per the Session Protocol with: the per-arm table (verdict, every probe, final with interval, which bar); **both measured bars against the ≈ 0.179 / ≈ 0.19 the truncation table predicted**; the realised reward-density check from Step 3; the figure signed as the round's visual gate; and the direction sentence quoted VERBATIM. Add D103 to `docs/decisions.md` covering BOTH the user's override of D101 §5 (recorded as an override, per spec §0) AND the round's outcome. Update the tracker and the memory file.

**State the outcome plainly**, whichever it is. If 0 IMPROVED, Route C stands as D102 left it and the spec §6 risk — the clone sees its opponent on 5.2 % of steps against the teacher's ~22 % — is named as the next place to look, not as a further round.

- [ ] **Step 5: Stop.** The direction is the user's to act on.

---

## Self-review

- **Spec coverage:** §0 override → Task 3's D103; §1 premise → Task 3 Step 3's density check; §2 protocol → Task 2 constants, `measure_starts`, existing verdict rules; §3 arms → `ARMS` / `arm_config` / `run_arm` + Task 1's wrapper; §4 reported → `run_arm`'s training block; §5 direction → `summarize`; §6 risk → Task 3 Step 4; §7 artefacts → Task 2 paths.
- **Placeholders:** none. Task 2 Step 2 describes the runner against a named existing file it mirrors, with every constant, key and behavioural difference stated explicitly.
- **Type consistency:** `arm_config(arm) -> (EnvConfig, float, int, str)` identical in the runner and its tests; `load_starts() -> dict` keyed by the same `"START_400"` / `"START_1200"` strings `measure_starts` writes and `ARMS` names; `finetune_br(cfg, role, opponent_factory, model_path, total_timesteps, seed, device, gamma, critic_warmup, callback, env_wrap)` matches the existing signature; `round_verdict` and `control_collapsed` unchanged.
