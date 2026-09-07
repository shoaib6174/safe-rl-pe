# T0.5 Final Learner Round Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the final pre-registered learner round — the strong anchor (β = 1.0) at the full 1M budget, advantage-weighted regression at an adaptive temperature that keeps the weighting engaged, and that method's sign-flipped control — and decide, by the pre-registered rules, whether any mechanism improves on the cloned prior.

**Architecture:** One change to `pe2/awr.py` (an adaptive temperature, off by default so the frozen T0.4 code path is bit-identical) plus a new standalone runner modelled on `scripts/derisk_t0b_round.py`. The T0.4 runner is NOT refactored: it is the artefact that produced a recorded result and must keep producing it.

**Tech Stack:** Python 3.10+, numpy, gymnasium, stable-baselines3 2.7.1, torch (CUDA on niro-1), matplotlib, pytest, ruff.

**Spec:** `docs/pathway/t0_final_round_prereg_2026-09-07.md` — §2 (protocol), §3 (arms), §4 (reported), §5 (direction and the terminal commitment) are binding. Decisions: D83, D99, D100, D101.

## Global Constraints

- Always `./venv/bin/python`; tests `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest <target> -q`. Full `tests/pe2` takes ~12 min; run it once, in Task 3.
- Lint only touched files (`./venv/bin/ruff check` + `ruff format --check`).
- **Pre-registered constants, verbatim from spec §2/§3 — never tuned:** `TOTAL = 1_000_000`, `EVERY = 100_000`, `EVAL_N = 200`, `FINAL_EVAL_N = 500`, `PASS_FROM = 500_000`, `EVAL_SEED0 = 840_000`, `NEG_TOTAL = 500_000`, `RF_SEED0 = 850_000`, `RFNEG_SEED0 = 860_000`, `SEEDS = {"RE": 7, "RF": 13, "RFNEG": 13}`, `BETA_RE = 1.0`, `CRITIC_WARMUP = 50_000`, `CLONE_CI_LO = 0.137` (existing), AWR `w_max = 20.0`, `gamma = 0.999`, `initial_episodes = 300`, `buffer_episodes = 300`, `steps_per_iter = 50_000`, `lam_floor = 0.01`.
- **Verdict rules** are the existing `pe2.derisk.round_verdict` / `pilot_passes` / `control_collapsed`; do not reimplement them.
- **The start bar is measured, never hardcoded.** `START_500` comes from evaluating the gated clone at N = 500 on seeds 840000–840499 and is written to `results/derisk/t0c_start.json` BEFORE any arm trains. An arm that cannot read that file must refuse to run.
- **`AwrConfig()` with its defaults must remain bit-identical to T0.4's behaviour.** The adaptive temperature is opt-in.
- **Scripts must never be imported by the test suite at collection time** beyond the existing `importlib` pattern in `tests/pe2/test_derisk_t0b_round.py`.
- `results/derisk/t0b_*`, `results/derisk/t0_*` and `results/visual_gates/**` are read-only evidence. `scripts/derisk_t0b_round.py` is a frozen artefact: do not edit it.
- Never train in a test: tiny budgets (≤ 3,000 steps), `pytest.importorskip` for `stable_baselines3` / `torch`.
- Branch `t0-final-round` from `main`. Commit only the files each task names. Trailer: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

---

## File map

| file | responsibility |
|---|---|
| `pe2/awr.py` | `LAM_FLOOR`; `AwrConfig.adaptive_lam` + `.lam_floor`; `resolve_lam`; `weight_ratio`; per-iteration `lam_used` / `weight_ratio_1sigma` / `weight_ratio_2sigma` (Task 1) |
| `tests/pe2/test_awr.py` | tests for the above, including bit-identical default behaviour (Task 1) |
| `scripts/derisk_t0c_round.py` | constants, `measure_start`, `run_arm`, `anchor_pattern`, `spec_check`, `summarize`, `write_figure`, CLI (Task 2) |
| `tests/pe2/test_derisk_t0c_round.py` | the runner's decisions (Task 2) |
| `docs/worklogs/…`, `docs/decisions.md`, `docs/workflow_tracker.md` | Task 3 record |

---

### Task 1: Adaptive temperature in `pe2/awr.py`

**Files:**
- Modify: `pe2/awr.py`
- Test: `tests/pe2/test_awr.py`

**Interfaces:**
- Produces: `LAM_FLOOR = 0.01`; `AwrConfig` gains `adaptive_lam: bool = False` and `lam_floor: float = LAM_FLOOR`; `resolve_lam(advantages, cfg) -> float`; `weight_ratio(sigmas, adv_std, lam) -> float`. Each iteration dict in `train_awr`'s stats gains `lam_used`, `weight_ratio_1sigma`, `weight_ratio_2sigma`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pe2/test_awr.py`:

```python
class TestAdaptiveTemperature:
    """T0.5 spec §3.2: lambda_t = max(std(A_t), lam_floor) per iteration.

    The point is scale-freeness — the weight distribution must be the same
    at every iteration regardless of how far the value net has driven the
    advantage spread down. T0.4 used a fixed 0.5 against a spread that
    decayed 0.186 -> 0.095, so the +/-1 sigma weight ratio fell from 2.10
    to 1.46 and the weighting was inert.
    """

    def test_fixed_is_the_default_and_returns_the_configured_value(self):
        from pe2.awr import AwrConfig, resolve_lam

        cfg = AwrConfig()
        assert cfg.adaptive_lam is False and cfg.lam_floor == 0.01
        adv = np.random.default_rng(0).normal(0.0, 0.2, size=5000)
        assert resolve_lam(adv, cfg) == 0.5

    def test_adaptive_returns_the_measured_spread(self):
        from pe2.awr import AwrConfig, resolve_lam

        cfg = AwrConfig(adaptive_lam=True)
        adv = np.random.default_rng(1).normal(0.0, 0.2, size=200_000)
        assert resolve_lam(adv, cfg) == pytest.approx(np.std(adv))
        assert resolve_lam(adv, cfg) == pytest.approx(0.2, abs=0.01)

    def test_floor_engages_on_a_degenerate_spread(self):
        from pe2.awr import AwrConfig, resolve_lam

        cfg = AwrConfig(adaptive_lam=True)
        assert resolve_lam(np.full(100, 0.3), cfg) == 0.01  # zero spread
        assert resolve_lam(np.random.default_rng(2).normal(0, 1e-6, 100), cfg) == 0.01

    def test_the_weight_ratio_is_constant_under_adaptive_lambda(self):
        """The property the whole change exists for."""
        from pe2.awr import AwrConfig, resolve_lam, weight_ratio

        cfg = AwrConfig(adaptive_lam=True)
        rng = np.random.default_rng(3)
        # The spread T0.4 actually traversed, first iteration to last.
        for std in (0.1857, 0.1489, 0.1203, 0.0953):
            adv = rng.normal(0.0, std, size=200_000)
            lam = resolve_lam(adv, cfg)
            assert weight_ratio(1.0, float(np.std(adv)), lam) == pytest.approx(
                np.exp(2.0), rel=1e-9
            )
            assert weight_ratio(2.0, float(np.std(adv)), lam) == pytest.approx(
                np.exp(4.0), rel=1e-9
            )

    def test_the_fixed_temperature_reproduces_the_t04_numbers(self):
        """Guards the claim in the spec's own comparison table."""
        from pe2.awr import weight_ratio

        assert weight_ratio(1.0, 0.1857, 0.5) == pytest.approx(2.10, abs=0.01)
        assert weight_ratio(1.0, 0.0953, 0.5) == pytest.approx(1.46, abs=0.01)

    def test_predicted_clip_fraction_under_adaptive_lambda(self):
        """Spec §3.2 predicts ~0.0014 of weights reach the w_max=20 ceiling."""
        from pe2.awr import AwrConfig, advantage_weights, resolve_lam

        cfg = AwrConfig(adaptive_lam=True)
        adv = np.random.default_rng(4).normal(0.0, 0.15, size=400_000)
        lam = resolve_lam(adv, cfg)
        w = advantage_weights(adv, np.zeros_like(adv), lam, cfg.w_max, 1.0)
        assert float((w >= cfg.w_max).mean()) == pytest.approx(0.00137, abs=0.0006)


class TestAdaptiveTemperatureInTraining:
    def _clone(self, tmp_path, cfg):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import SAC

        from pe2.exploitability import SingleAgentWrapper
        from pe2.scripted import GreedyEvader

        path = tmp_path / "clone.zip"
        SAC(
            "MlpPolicy",
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader),
            seed=0,
            verbose=0,
            device="cpu",
        ).save(path)
        return path

    def test_iterations_record_the_temperature_and_the_ratios(self, tmp_path):
        from pe2.awr import AwrConfig, train_awr
        from pe2.config import EnvConfig
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=40)
        clone = self._clone(tmp_path, cfg)
        awr = AwrConfig(
            adaptive_lam=True,
            initial_episodes=6,
            buffer_episodes=8,
            steps_per_iter=100,
            value_epochs=2,
            actor_epochs=2,
            batch_size=64,
        )
        _, stats = train_awr(
            cfg, "pursuer", GreedyEvader, clone, 400, seed=3, awr=awr,
            probe_every=200, probe=lambda m, s: None, device="cpu",
            rollout_seed0=990200,
        )
        assert stats["config"]["adaptive_lam"] is True
        for it in stats["iterations"]:
            assert it["lam_used"] > 0.0
            assert set(it) >= {"lam_used", "weight_ratio_1sigma", "weight_ratio_2sigma"}
            if it["lam_used"] > awr.lam_floor:  # floor not engaged
                assert it["weight_ratio_1sigma"] == pytest.approx(np.exp(2.0), rel=1e-6)

    def test_the_default_config_is_unchanged_by_this_feature(self, tmp_path):
        """T0.4's runs are a frozen record; the fixed path must not move."""
        from pe2.awr import AwrConfig, train_awr
        from pe2.config import EnvConfig
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=40)
        clone = self._clone(tmp_path, cfg)
        kw = dict(
            initial_episodes=5, buffer_episodes=6, steps_per_iter=80,
            value_epochs=1, actor_epochs=1, batch_size=64,
        )
        runs = [
            train_awr(
                cfg, "pursuer", GreedyEvader, clone, 300, seed=5,
                awr=AwrConfig(**kw), probe_every=150, probe=lambda m, s: None,
                device="cpu", rollout_seed0=990300,
            )[1]
            for _ in range(2)
        ]
        assert runs[0]["first_action"] == runs[1]["first_action"]
        for it in runs[0]["iterations"]:
            assert it["lam_used"] == 0.5  # the fixed default, not the spread
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_awr.py::TestAdaptiveTemperature -q`
Expected: FAIL, `ImportError: cannot import name 'resolve_lam'`.

- [ ] **Step 3: Implement**

In `pe2/awr.py`, beside `LAM` and `W_MAX`:

```python
LAM_FLOOR = 0.01  # smallest admissible temperature; guards a degenerate spread
```

Add the two fields to `AwrConfig`, directly after `lam`:

```python
    adaptive_lam: bool = False  # True: lambda_t = max(std(A_t), lam_floor)
    lam_floor: float = LAM_FLOOR
```

Add two module-level functions after `advantage_weights`:

```python
def resolve_lam(advantages, cfg: "AwrConfig") -> float:
    """The temperature this iteration uses.

    Fixed by default, so T0.4's runs stay reproducible. Under
    `adaptive_lam` it is the measured advantage spread (T0.5 spec §3.2):
    dividing by the spread makes the weight distribution scale-free, so the
    weighting is exercised identically at every iteration however far the
    value net has driven the spread down. The floor guards the degenerate
    case where the spread is ~0 and the weights would otherwise blow up.
    """
    if not cfg.adaptive_lam:
        return cfg.lam
    return max(float(np.std(np.asarray(advantages))), cfg.lam_floor)


def weight_ratio(sigmas: float, adv_std: float, lam: float) -> float:
    """Weight ratio between a +k sigma and a -k sigma sample: exp(2k*std/lam).

    The reportable measure of whether the weighting is engaged at all. Under
    `adaptive_lam` (lam == std) it is exp(2k) regardless of the spread; at
    T0.4's fixed lam = 0.5 it fell from 2.10 to 1.46 as the spread decayed,
    which is why that arm's weights were indistinguishable from uniform.
    """
    return float(np.exp(2.0 * sigmas * adv_std / lam))
```

In `train_awr`'s `update()`, replace the two lines that compute `w` with:

```python
        adv_np = rets_np - values
        lam_used = resolve_lam(adv_np, awr)
        w = advantage_weights(rets_np, values, lam_used, awr.w_max, awr.advantage_sign)
```

and add three keys to the returned dict, next to `advantage_std`:

```python
            "lam_used": lam_used,
            "weight_ratio_1sigma": weight_ratio(1.0, float(adv_np.std()), lam_used),
            "weight_ratio_2sigma": weight_ratio(2.0, float(adv_np.std()), lam_used),
```

Leave `advantage_mean` / `advantage_std` computing from `rets_np - values` as they are, or reuse `adv_np` — either is fine, but do not change their values.

Update the module docstring's temperature sentence to state both modes in one line.

- [ ] **Step 4: Run**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_awr.py -q`
Expected: all pass (previous AWR tests included — they exercise the default path, which must be unchanged).

- [ ] **Step 5: Commit**

```bash
./venv/bin/ruff check pe2/awr.py tests/pe2/test_awr.py && ./venv/bin/ruff format --check pe2/awr.py tests/pe2/test_awr.py
git add pe2/awr.py tests/pe2/test_awr.py
git commit -m "feat(awr): adaptive temperature (lambda = advantage spread), off by default" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The T0.5 runner `scripts/derisk_t0c_round.py`

**Files:**
- Create: `scripts/derisk_t0c_round.py`
- Test: `tests/pe2/test_derisk_t0c_round.py`

**Interfaces:**
- Consumes: `pe2.derisk.{round_verdict, control_collapsed, CLONE_CI_LO, TrainingCaptureCounter}`, `pe2.eval.{wilson_ci, win_count}`, `pe2.exploitability.{CheckpointEvalProbe, finetune_br_anchored, evaluate_sb3_policy}`, `pe2.anchored_sac.{AnchorKLProbe, anchor_kl_grew}`, `pe2.awr.{AwrConfig, train_awr}`, `pe2.scripted.GreedyEvader`.
- Produces: `measure_start(cfg, device) -> dict`; `load_start() -> dict`; `anchor_pattern(verdict, kl_grew) -> tuple[str, str]`; `spec_check(iterations) -> dict`; `run_arm(arm, cfg, device) -> dict`; `load_results()`; `summarize(results) -> dict`; `write_figure(results, summary, path)`; CLI `--measure-start` / `--arms` / `--summarize` / `--device`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pe2/test_derisk_t0c_round.py`:

```python
"""scripts/derisk_t0c_round.py — the final round's own decisions."""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_t0c_round.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_t0c_round", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _arm(name, verdict, final=(120, 500), rows=None):
    return {
        "arm": name, "verdict": verdict, "reason": "test", "timesteps": 1_000_000,
        "checkpoints": rows or [
            {"steps": s, "win_rate": 0.2, "k": 40, "n": 200}
            for s in range(100_000, 1_000_001, 100_000)
        ],
        "final": {
            "k": final[0], "n": final[1], "win_rate": final[0] / final[1],
            "wilson": list(R.wilson_ci(final[0], final[1])),
        },
        "start": {"k": 37, "n": 500, "rate": 0.185},
    }


class TestPreRegisteredConstants:
    def test_protocol(self):
        assert (R.TOTAL, R.EVERY, R.EVAL_N, R.FINAL_EVAL_N, R.PASS_FROM) == (
            1_000_000, 100_000, 200, 500, 500_000,
        )
        assert (R.NEG_TOTAL, R.BETA_RE, R.CRITIC_WARMUP) == (500_000, 1.0, 50_000)

    def test_fresh_seed_streams(self):
        assert (R.EVAL_SEED0, R.RF_SEED0, R.RFNEG_SEED0) == (840_000, 850_000, 860_000)
        # T0.4 and earlier rounds must not be reused.
        assert R.EVAL_SEED0 not in (700_000, 780_000, 800_000, 810_000)

    def test_seeds_and_learners(self):
        assert R.SEEDS == {"RE": 7, "RF": 13, "RFNEG": 13}
        assert R.LEARNER_ARMS == ("RE", "RF")

    def test_pass_window_is_bound_to_the_library_constant(self):
        from pe2.derisk import ROUND_PASS_FROM

        assert R.PASS_FROM == ROUND_PASS_FROM


class TestStartBar:
    def test_arms_refuse_to_run_without_a_measured_start(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t0c_start.json")
        with pytest.raises(SystemExit, match="start bar"):
            R.load_start()

    def test_load_start_reads_what_measure_start_wrote(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t0c_start.json")
        payload = {"k": 96, "n": 500, "rate": 0.192, "wilson": [0.16, 0.23],
                   "eval_seed0": 840_000, "clone": "x.zip"}
        (tmp_path / "t0c_start.json").write_text(json.dumps(payload))
        assert R.load_start() == payload


class TestAnchorPattern:
    """Spec §3.1 fixes the reading of each outcome pattern in advance."""

    def test_improved(self):
        assert R.anchor_pattern("IMPROVED", True)[0] == "improved"
        assert R.anchor_pattern("IMPROVED", False)[0] == "improved"

    def test_pinned(self):
        pattern, reading = R.anchor_pattern("PRESERVED", False)
        assert pattern == "pinned"
        assert "cannot learn" in reading

    def test_late_collapse(self):
        pattern, reading = R.anchor_pattern("COLLAPSED", True)
        assert pattern == "late_collapse"
        assert "continuum" in reading

    def test_anything_else_is_named_not_forced(self):
        pattern, reading = R.anchor_pattern("COLLAPSED", False)
        assert pattern == "other" and reading


class TestSpecCheck:
    """Spec §3.2: the temperature's predicted diagnostics are falsifiable."""

    def _its(self, ratio, lam=0.15, n=5):
        return [
            {"lam_used": lam, "weight_ratio_1sigma": ratio, "weight_ratio_2sigma": ratio**2}
            for _ in range(n)
        ]

    def test_matching_iterations_pass(self):
        import math

        assert R.spec_check(self._its(math.exp(2.0)))["pass"] is True

    def test_a_deviating_iteration_fails(self):
        out = R.spec_check(self._its(2.10))  # T0.4's inert value
        assert out["pass"] is False and "2.1" in out["reason"]

    def test_floored_iterations_are_excluded_not_failed(self):
        import math

        its = self._its(math.exp(2.0)) + [
            {"lam_used": R.LAM_FLOOR, "weight_ratio_1sigma": 1.02, "weight_ratio_2sigma": 1.04}
        ]
        out = R.spec_check(its)
        assert out["pass"] is True and out["n_floored"] == 1


class TestSummarize:
    def test_survivors_are_improved_only(self):
        s = R.summarize({
            "RE": _arm("RE", "IMPROVED"), "RF": _arm("RF", "PRESERVED"),
            "RFNEG": {"arm": "RFNEG", "verdict": True, "reason": "collapsed"},
        })
        assert s["survivors"] == ["RE"] and s["complete"] is True

    def test_rf_withheld_when_its_control_did_not_collapse(self):
        s = R.summarize({
            "RE": _arm("RE", "PRESERVED"), "RF": _arm("RF", "IMPROVED"),
            "RFNEG": {"arm": "RFNEG", "verdict": None, "reason": "not separated"},
        })
        assert s["arms"]["RF"]["verdict"] == "UNDETERMINED"
        assert s["survivors"] == []
        # Spec 3.3: a second non-discriminating control is itself a finding.
        assert "insensitive" in s["control_finding"]

    def test_route_c_is_declared_final(self):
        s = R.summarize({
            "RE": _arm("RE", "COLLAPSED"), "RF": _arm("RF", "COLLAPSED"),
            "RFNEG": {"arm": "RFNEG", "verdict": True, "reason": "collapsed"},
        })
        assert s["survivors"] == []
        assert "final" in s["direction"].lower()
        assert "no further learner pre-registration" in s["direction"].lower()

    def test_a_crashed_learner_does_not_reach_route_c(self):
        s = R.summarize({
            "RE": _arm("RE", "UNDETERMINED"), "RF": _arm("RF", "COLLAPSED"),
            "RFNEG": {"arm": "RFNEG", "verdict": True, "reason": "collapsed"},
        })
        assert s["complete"] is False and "incomplete" in s["direction"].lower()

    def test_re_does_not_depend_on_the_awr_control(self):
        s = R.summarize({"RE": _arm("RE", "IMPROVED")})
        assert s["arms"]["RE"]["verdict"] == "IMPROVED" and s["survivors"] == ["RE"]


class TestFigure:
    def test_writes_with_partial_arms(self, tmp_path):
        pytest.importorskip("matplotlib")
        results = {"RE": _arm("RE", "PRESERVED")}
        R.write_figure(results, R.summarize(results), tmp_path / "f.png")
        assert (tmp_path / "f.png").stat().st_size > 0


class TestRunArmWiring:
    """Pins the USE of the constants, not merely their values."""

    def test_per_arm_call_arguments(self, tmp_path, monkeypatch):
        pytest.importorskip("stable_baselines3")
        calls = {}
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t0c_start.json")
        (tmp_path / "t0c_start.json").write_text(json.dumps(
            {"k": 96, "n": 500, "rate": 0.192, "wilson": [0.16, 0.23],
             "eval_seed0": 840_000, "clone": "x.zip"}))

        def fake_anchored(cfg, role, opp, path, total, **kw):
            calls["RE"] = {"path": str(path), "total": total, **kw}

        def fake_awr(cfg, role, opp, path, total, **kw):
            arm = "RFNEG" if kw["awr"].advantage_sign < 0 else "RF"
            calls[arm] = {"path": str(path), "total": total,
                          "seed": kw["seed"], "rollout_seed0": kw["rollout_seed0"],
                          "adaptive_lam": kw["awr"].adaptive_lam,
                          "sign": kw["awr"].advantage_sign}
            return None, {"total_steps": total, "n_episodes": 1, "positives": 0,
                          "positives_per_1M": 0.0, "iterations": [], "first_action": [0.0, 0.0],
                          "probe_marks": [], "config": {}}

        import pe2.exploitability as expl
        import pe2.awr as awr_mod
        monkeypatch.setattr(expl, "finetune_br_anchored", fake_anchored)
        monkeypatch.setattr(awr_mod, "train_awr", fake_awr)
        monkeypatch.setattr(R, "final_eval", lambda *a, **k: {
            "k": 100, "n": 500, "win_rate": 0.2, "wilson": [0.17, 0.24], "ckpt": "x"})

        from pe2.config import EnvConfig

        for arm in ("RE", "RF", "RFNEG"):
            R.run_arm(arm, EnvConfig(), "cpu")

        assert calls["RE"]["beta"] == 1.0 and calls["RE"]["seed"] == 7
        assert calls["RE"]["total"] == 1_000_000
        assert calls["RE"]["critic_warmup"] == 50_000
        assert calls["RF"]["adaptive_lam"] is True and calls["RF"]["sign"] == 1.0
        assert calls["RF"]["rollout_seed0"] == 850_000 and calls["RF"]["total"] == 1_000_000
        assert calls["RFNEG"]["sign"] == -1.0
        assert calls["RFNEG"]["rollout_seed0"] == 860_000
        assert calls["RFNEG"]["total"] == 500_000
        assert calls["RF"]["seed"] == calls["RFNEG"]["seed"] == 13
```

Run it: FAIL (`FileNotFoundError` for the script).

- [ ] **Step 2: Implement the runner**

Create `scripts/derisk_t0c_round.py`. It is modelled on `scripts/derisk_t0b_round.py` — read that file first and follow its structure, docstring discipline and comment style. Deliberate duplication: T0.4's runner is the artefact that produced a recorded result and is frozen; this is a separate round with its own constants and its own direction table.

```python
"""T0.5 final learner round — the last learner pre-registration (D101).

Spec: docs/pathway/t0_final_round_prereg_2026-09-07.md, reproduced here
where it binds. Nothing here may change a threshold, a seed, a budget or a
definition after a number is seen.

Arms (pursuer vs frozen GreedyEvader, probes every 100k on 200 paired seeds
840000+, final probe N=500):
  RE     the untested anchor cell: KL-anchored SAC at beta = 1.0, 1M steps
  RF     the untested AWR cell: adaptive temperature lambda_t = std(A_t), 1M
  RFNEG  RF with the advantage sign flipped, 500k: MUST collapse (control)

Pass = win rate >= 0.137 at every probe from 500k. Improved = final N=500
Wilson lower bound > START_500, the clone's OWN score measured on this
round's eval stream before any arm trains (spec 2; T0.4's bar came from a
different stream, which its final review flagged). Survivor = IMPROVED.
RF is withheld unless RFNEG collapses.

TERMINAL (spec 5): 0 IMPROVED makes Route C final and NO further learner
pre-registration follows.

Usage:
  ./venv/bin/python scripts/derisk_t0c_round.py --measure-start   (first, once)
  ./venv/bin/python scripts/derisk_t0c_round.py --arms RE         (one per arm)
  ./venv/bin/python scripts/derisk_t0c_round.py --summarize       (after all)
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

from pe2.awr import LAM_FLOOR
from pe2.config import EnvConfig
from pe2.derisk import (
    CLONE_CI_LO,
    ROUND_PASS_FROM,
    TrainingCaptureCounter,
    control_collapsed,
    round_verdict,
)
from pe2.eval import wilson_ci, win_count
from pe2.scripted import GreedyEvader

OUT = pathlib.Path("results/derisk")
CLONE = pathlib.Path("results/visual_gates/step_05/sac_bc_pursuer.zip")
START_PATH = OUT / "t0c_start.json"

# --- pre-registered (spec 2/3). Do not tune. ---
TOTAL = 1_000_000
EVERY = 100_000
EVAL_N = 200
FINAL_EVAL_N = 500
PASS_FROM = ROUND_PASS_FROM  # 500_000, bound to the library constant
EVAL_SEED0 = 840_000  # fresh: 700000-790999, 800000-830999 are consumed
NEG_TOTAL = 500_000  # C3-a's budget: offline ~306k + four online iterations
BETA_RE = 1.0
CRITIC_WARMUP = 50_000
RF_SEED0 = 850_000
RFNEG_SEED0 = 860_000
SEEDS = {"RE": 7, "RF": 13, "RFNEG": 13}
LEARNER_ARMS = ("RE", "RF")
DETERMINATE_VERDICTS = ("COLLAPSED", "PRESERVED", "IMPROVED")

# Spec 3.2's predicted diagnostics, stated BEFORE the run so the temperature
# specification is falsifiable separately from the learning hypothesis.
PREDICTED_RATIO_1SIGMA = math.exp(2.0)  # 7.389
SPEC_TOL = 0.05  # relative


def probe(cfg, arm):
    from pe2.exploitability import CheckpointEvalProbe

    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    return CheckpointEvalProbe(
        cfg, "pursuer", GreedyEvader, every=EVERY, eval_n=EVAL_N, seed0=EVAL_SEED0,
        ckpt_dir=d, report_path=OUT / f"t0c_probe_{arm}.json", label=arm,
    )


def measure_start(cfg, device) -> dict:
    """The bar every arm is judged against: the gated clone's OWN score at
    N=500 on THIS round's eval stream, measured once before any training.

    T0.4 judged its arms against a bar taken from stream 720000+ while its
    probes ran on 780000+; its final review flagged the mismatch (it happened
    to disfavour the arms). Measuring it here on the same seeds the probes
    use makes the comparison paired by construction.
    """
    from stable_baselines3 import SAC

    from pe2.exploitability import evaluate_sb3_policy

    model = SAC.load(CLONE, device=device)
    logs = evaluate_sb3_policy(
        model, cfg, "pursuer", GreedyEvader, n_episodes=FINAL_EVAL_N, seed0=EVAL_SEED0
    )
    k, n = win_count(logs, "pursuer")
    lo, hi = wilson_ci(k, n)
    payload = {
        "k": k, "n": n, "rate": k / n, "wilson": [lo, hi],
        "eval_seed0": EVAL_SEED0, "clone": str(CLONE),
    }
    START_PATH.write_text(json.dumps(payload, indent=2))
    print(f"START_500 = {k}/{n} = {k / n:.4f}, Wilson [{lo:.4f}, {hi:.4f}]", flush=True)
    return payload


def load_start() -> dict:
    """Refuse to judge an arm against a bar that was never measured."""
    if not START_PATH.exists():
        raise SystemExit(
            f"start bar missing: {START_PATH}. Run --measure-start FIRST; the "
            "pre-registration requires it to exist before any arm trains."
        )
    return json.loads(START_PATH.read_text())


def anchor_pattern(verdict: str, kl_grew) -> tuple[str, str]:
    """Spec 3.1's pre-committed reading of RE's outcome pattern.

    Fixed before the run so the mechanism's interpretation is not chosen
    after seeing which pattern occurred. The VERDICT is decided by the
    protocol's rules; this only names what the round concludes about the
    mechanism.
    """
    if verdict == "IMPROVED":
        return "improved", (
            "a survivor: beta = 1.0 at 1M is the Track-1 learner and Track 2 "
            "is planned for it"
        )
    if verdict == "PRESERVED" and kl_grew is False:
        return "pinned", (
            "the strong anchor genuinely cannot learn even given 5x the budget; "
            "with RB's collapse at beta = 0.1 this closes the dichotomy D100 "
            "correctly refused to claim on the evidence then available"
        )
    if verdict == "COLLAPSED" and kl_grew is True:
        return "late_collapse", (
            "supports the beta-continuum reading: anchor strength trades "
            "time-to-collapse against learning rate and no strength tested "
            "escapes, so Route C's claim strengthens from 'these settings "
            "failed' to 'the mechanism has no operating point'"
        )
    return "other", (
        f"verdict {verdict} with kl_grew={kl_grew} matches none of the three "
        "patterns the pre-registration named; report the pattern, do not force "
        "it into one of them"
    )


def spec_check(iterations: list[dict]) -> dict:
    """Did the adaptive temperature behave as spec 3.2 predicted?

    Falsifies the TEMPERATURE SPECIFICATION independently of the learning
    hypothesis: if the observed weight ratios do not match, the
    implementation is wrong and the arm is UNDETERMINED on those grounds,
    not evidence about AWR. Iterations where the floor engaged are excluded
    (the ratio is legitimately smaller there) and counted.
    """
    active = [it for it in iterations if it["lam_used"] > LAM_FLOOR]
    floored = len(iterations) - len(active)
    if not active:
        return {"pass": None, "n_active": 0, "n_floored": floored,
                "reason": "no iteration ran above the temperature floor"}
    worst = max(active, key=lambda it: abs(it["weight_ratio_1sigma"] - PREDICTED_RATIO_1SIGMA))
    dev = abs(worst["weight_ratio_1sigma"] - PREDICTED_RATIO_1SIGMA) / PREDICTED_RATIO_1SIGMA
    ok = dev <= SPEC_TOL
    return {
        "pass": ok, "n_active": len(active), "n_floored": floored,
        "predicted_ratio_1sigma": PREDICTED_RATIO_1SIGMA,
        "worst_observed_ratio_1sigma": worst["weight_ratio_1sigma"],
        "worst_relative_deviation": dev,
        "reason": (
            f"worst +/-1 sigma weight ratio {worst['weight_ratio_1sigma']:.4f} vs "
            f"predicted {PREDICTED_RATIO_1SIGMA:.4f} "
            f"({dev:.1%} deviation, tolerance {SPEC_TOL:.0%})"
        ),
    }
```

Then `final_eval`, `run_arm`, `load_results`, `summarize`, `write_figure` and `main`, following `derisk_t0b_round.py`'s equivalents with these differences:

- `final_eval(cfg, arm, total, device)`: always `SAC.load` (every arm here is SAC-family — RE is `KLAnchoredSAC`, which loads as SAC; RF/RFNEG update the clone's SAC actor in place). Path `OUT / f"ckpts_{arm}" / f"{arm}_{total}.zip"`.
- `run_arm(arm, cfg, device)`: `start = load_start()` FIRST (so a missing bar refuses before any training). `total = NEG_TOTAL if arm == "RFNEG" else TOTAL`. `counter = TrainingCaptureCounter(snapshot_every=CRITIC_WARMUP)`.
  - `RE`: `finetune_br_anchored(..., beta=BETA_RE, critic_warmup=CRITIC_WARMUP, callback=CallbackList([pr.callback, counter.callback, AnchorKLProbe(pr.rows, EVERY).callback]))`; `extra["anchor"] = {"beta": BETA_RE, "kl_grew": grew, "kl_reason": why}`.
  - `RF` / `RFNEG`: `AwrConfig(adaptive_lam=True, advantage_sign=1.0 if arm == "RF" else -1.0)`; `rollout_seed0=RF_SEED0 if arm == "RF" else RFNEG_SEED0`; `probe=pr.evaluate_at`; `extra["awr"] = stats`; `extra["spec_check"] = spec_check(stats["iterations"])`.
  - Verdict: `RFNEG` → `control_collapsed(pr.rows, from_step=EVERY)`. Others → the crashed-arm guard first (if the final checkpoint is missing, `UNDETERMINED` with the missing path named, arm JSON still written), else `final_eval` then `round_verdict(pr.rows, final["k"], final["n"], start["rate"], from_step=PASS_FROM)`.
  - `RE` additionally records `extra["anchor_pattern"], extra["anchor_reading"] = anchor_pattern(verdict, grew)` AFTER the verdict is known.
  - `result["start"] = {"k": start["k"], "n": start["n"], "rate": start["rate"], "eval_seed0": start["eval_seed0"], "provenance": "measured on this round's eval stream before any training (spec 2); results/derisk/t0c_start.json"}`.
  - Write `OUT / f"t0c_arm_{arm}.json"`.
- `load_results()`: glob `t0c_arm_*.json`.
- `summarize(results)`: same shape as T0.4's, with `LEARNER_ARMS = ("RE", "RF")`, `ctrl = results.get("RFNEG")`, RF withheld exactly as RD was, `complete` requiring every learner judged with the withheld-RF exemption, and:
  - `spec` key → `"docs/pathway/t0_final_round_prereg_2026-09-07.md"`;
  - a `control_finding` key, non-empty ONLY when RFNEG ran and did not collapse: `"the control did not discriminate even with the weighting engaged (spec 3.3): the policy class is INSENSITIVE to the sign of the advantage on this task, i.e. self-imitation dominates the update regardless of weighting. Report this as a finding, not as another null."`;
  - direction when survivors: as T0.4's;
  - direction when `complete` and no survivors: `"No arm IMPROVED. Route C is FINAL: the characterisation paper is written (shield guarantee D97, exact-game shift T3, and a learning failure independent of reward shaping, algorithm family, anchoring, bootstrapping and the weighting's calibration). No further learner pre-registration follows (spec 5)."`;
  - otherwise `"incomplete: not every learner arm has a determinate verdict"`.
  - Also surface `results["RE"]["anchor_pattern"]` as `summary["anchor_pattern"]` when present.
- `write_figure`: as T0.4's (curves per arm, the 0.137 line, the pass-window line, the final-probe error bar, an "INCOMPLETE" title when not complete), plus a horizontal line at `START_500` labelled as the improvement bar.
- `main()`: `--measure-start` runs `measure_start` and returns; `--arms` runs each arm (refusing if `CLONE` is missing); `--summarize` writes `t0c_round.{json,png}` only.

- [ ] **Step 3: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_t0c_round.py -q` → PASS.
Also smoke the CLI's refusal path: `./venv/bin/python scripts/derisk_t0c_round.py --summarize` with no arm files present must print an incomplete summary and exit 0.

- [ ] **Step 4: Lint and commit**

```bash
./venv/bin/ruff check scripts/derisk_t0c_round.py tests/pe2/test_derisk_t0c_round.py && ./venv/bin/ruff format --check scripts/derisk_t0c_round.py tests/pe2/test_derisk_t0c_round.py
git add scripts/derisk_t0c_round.py tests/pe2/test_derisk_t0c_round.py
git commit -m "feat(derisk): T0.5 final-round runner (RE/RF/RFNEG, measured start bar, spec check)" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Run the round, summarise, record

**Files:**
- Create (by the runner): `results/derisk/t0c_start.json`, `t0c_arm_*.json`, `t0c_round.{json,png}`, `ckpts_{RE,RF,RFNEG}/`
- Modify: a NEW worklog `docs/worklogs/2026-09-07_S123.md` (this is a new session; follow `docs/worklogs/TEMPLATE.md` and the Session Protocol in CLAUDE.md), `docs/decisions.md` (D102: the outcome), `docs/workflow_tracker.md`

- [ ] **Step 1: Full suite once, on the tree that will run**

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q` → all pass (expect ~560). Record the count.

- [ ] **Step 2: Measure the start bar, then launch**

```bash
./venv/bin/python scripts/derisk_t0c_round.py --measure-start | tee results/derisk/t0c_start.log
for A in RFNEG RF RE; do
  nohup ./venv/bin/python scripts/derisk_t0c_round.py --arms $A > results/derisk/t0c_$A.log 2>&1 &
done
```

The start measurement MUST complete before the arms launch — its file is what they read, and its existence before any checkpoint is the auditable evidence for the pre-registration's claim. Poll `tail -2 results/derisk/t0c_*.log` no more than every 10 minutes. Expect ≈ 6 h (RE is the long pole).

- [ ] **Step 3: Summarise**

`./venv/bin/python scripts/derisk_t0c_round.py --summarize`, then report per arm: verdict, reason, every probe, the final N=500 with its interval, `training.positives_per_1M`, RE's `anchor_kl` trajectory with its `anchor_pattern` and reading, RF's `spec_check` block, and both AWR arms' per-iteration `lam_used` / `weight_ratio_1sigma` / `advantage_std` / `weight_clipped_fraction`.

**If RF's `spec_check.pass` is False**, RF is UNDETERMINED on implementation grounds — record it that way and do NOT reinterpret it as evidence about AWR.

- [ ] **Step 4: Record**

Create `docs/worklogs/2026-09-07_S123.md` per the Session Protocol, with: the per-arm table (verdict, every probe, final with interval, the measured `START_500` and its provenance); RE's pattern and its pre-committed reading quoted from the spec; RF's spec-check numbers against the predicted 7.389; the AWR grid-versus-actual probe mapping (no AWR probe before the offline phase completes may be presented as a time point); `positives_per_1M` per arm; the figure signed as the round's visual gate; and the direction sentence quoted VERBATIM from the summary JSON. Add D102 with the outcome and the direction it fixes. Update the tracker's header, session index and Current Blockers. Update the memory file `~/.claude/projects/-home-niro-1-Codes-safe-rl-pe/memory/pathway-b-plan.md`.

**State plainly whichever of these applies**, and do not soften it: if no arm improved, Route C is final by the terminal commitment in spec §5 and no further learner pre-registration follows; if an arm improved, name it as the Track-1 mechanism.

- [ ] **Step 5: Stop.** The direction is the user's to act on. Do not start Track 2 or the Route C write-up.

---

## Self-review

- **Spec coverage:** §2 protocol → Task 2 constants + `measure_start` + existing verdict rules; §3.1 RE → Task 2 `run_arm` + `anchor_pattern`; §3.2 RF → Task 1 (adaptive temperature) + Task 2 `spec_check`; §3.3 RFNEG → Task 2 `run_arm` + `summarize`'s `control_finding`; §4 reported → Task 1's new iteration keys + Task 3 Step 3; §5 direction and terminal commitment → `summarize`'s direction strings + Task 3 Step 4; §6 artefacts → Task 2 paths + Task 3; §7 limitation → Task 3's record.
- **Placeholders:** none. Task 2's Step 2 describes the second half of the runner against a named existing file rather than restating ~200 lines of near-identical code; every value, key and behavioural difference is given explicitly.
- **Type consistency:** `resolve_lam(advantages, cfg) -> float` and `weight_ratio(sigmas, adv_std, lam) -> float` identical in Tasks 1 and 2; `anchor_pattern(verdict, kl_grew) -> (str, str)` and `spec_check(iterations) -> dict` identical between the runner and its tests; `load_start() -> dict` returns the payload `measure_start` writes; `round_verdict(rows, final_k, final_n, start_rate, threshold, from_step)` and `control_collapsed(rows, threshold, from_step)` match their existing signatures.
