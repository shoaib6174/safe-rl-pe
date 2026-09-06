# T0.4 Retry Round Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the pre-registered retry round — three pursuer learners (KL-anchored SAC at β = 0.1, PPO+GAE at 1M, advantage-weighted regression on Monte-Carlo returns) plus AWR's sign-flipped negative control — and decide, by the pre-registered verdicts, whether any mechanism IMPROVES on the cloned prior.

**Architecture:** Two small additions to `pe2/` (a `round_verdict` in `pe2/derisk.py`; a new `pe2/awr.py` whose actor lives inside the clone's SB3 model so every probe/eval/checkpoint path is reused), one small refactor (`CheckpointEvalProbe.evaluate_at` so a non-SB3 loop can probe on the same grid), and one runner script that runs arms as independent processes and summarises from their JSON files.

**Tech Stack:** Python 3.10+, numpy, gymnasium, stable-baselines3 2.7.1, torch (CUDA on niro-1), matplotlib, pytest, ruff.

**Spec:** `docs/pathway/t0_retry_round_prereg_2026-09-06.md` — Part C is binding (protocol C2, arms C3, direction C4, artefacts C5). Governing decisions D83, D95 (A1, A2, A4), D98.

## Global Constraints

- Always `./venv/bin/python`; tests as `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest <target> -q`. The full `tests/pe2` suite takes ~12 minutes; run it once, in Task 5.
- Lint only touched files: `./venv/bin/ruff check <files>` and `./venv/bin/ruff format --check <files>`.
- **Pre-registered constants, verbatim from spec C2/C3 — never tuned** (with one exception, marked): `TOTAL = 1_000_000`, `EVERY = 100_000`, `EVAL_N = 200`, `FINAL_EVAL_N = 500`, `PASS_FROM = 500_000`, `CLONE_CI_LO = 0.137` (existing), `BETA_RB = 0.1`, `CRITIC_WARMUP = 50_000`, ~~`NEG_TOTAL = 200_000`~~ → **`NEG_TOTAL = 500_000` per amendment C3-a** (recorded in `docs/pathway/t0_retry_round_prereg_2026-09-06.md` §C3 at 19:24:31 on 2026-09-06, twenty minutes before the first checkpoint of any arm existed; `train_awr`'s unconditional 300-episode offline phase alone costs ≈306k steps, so at 200k the control's online loop would never run). **This plan line is superseded; the spec, the code and the results are all 500,000.** `EVAL_SEED0 = 780_000`, `AWR_SEED0 = 820_000`, `AWRNEG_SEED0 = 830_000`, `SEEDS = {"RB": 7, "RC": 9, "RD": 11, "RDNEG": 11}`, `STARTS = {"RB": (37, 200), "RC": (41, 200), "RD": (37, 200), "RDNEG": (37, 200)}`, AWR `LAM = 0.5`, `W_MAX = 20.0`, `initial_episodes = 300`, `buffer_episodes = 300`, `steps_per_iter = 50_000`, `gamma = 0.999`.
- **Verdict rules, verbatim:** pass = win rate ≥ 0.137 at every probe with `steps >= PASS_FROM`; improved = Wilson lower bound (z = 1.96) of the final N = 500 probe > the arm's start point estimate; verdict ∈ {COLLAPSED, PRESERVED, IMPROVED, UNDETERMINED}; survivor = IMPROVED only. `RDNEG` must satisfy `control_collapsed(rows, from_step=100_000)` is `True` or `RD` is withheld.
- **Evaluation is on true outcomes only** — `evaluate_sb3_policy` / `CheckpointEvalProbe` build their own unshaped wrapper. Nothing shaped touches an eval path.
- SAC-family recipe is D80/D81 binding: `finetune_br_anchored` unchanged. PPO kwargs are the existing `PPO_KWARGS`, unchanged.
- No pre-registered value, criterion, seed or budget changes after a number is seen. A verdict that cannot be reached is reported UNDETERMINED with its reason, never adjusted.
- Never run training in a test: unit tests use tiny budgets (≤ 3,000 steps) and skip when `stable_baselines3` / `torch` are absent (`pytest.importorskip`), matching `tests/pe2/test_derisk.py`.
- Work on branch `t0-retry-round` from `main`. Commit only the files each task names. Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- `results/` is git-ignored. `results/derisk/t0_*.json` (Tier 0) and `results/visual_gates/**` are read-only evidence.

---

## File map

| file | responsibility |
|---|---|
| `pe2/derisk.py` | `improved`, `round_verdict`, `VERDICTS` (Task 1) |
| `pe2/exploitability.py` | `CheckpointEvalProbe.evaluate_at(model, steps)`; the callback calls it (Task 2) |
| `pe2/awr.py` | `AwrConfig`, `Episode`, `rollout_episode`, `collect`, `advantage_weights`, `make_value_net`, `fit_value`, `awr_actor_update`, `train_awr` (Task 3) |
| `scripts/derisk_t0b_round.py` | constants, `run_arm`, `final_eval`, `summarize`, `write_figure`, CLI `--arms` / `--summarize` (Task 4) |
| `tests/pe2/test_derisk.py`, `tests/pe2/test_exploitability.py`, `tests/pe2/test_awr.py`, `tests/pe2/test_derisk_t0b_round.py` | tests |
| `docs/worklogs/2026-09-05_S122.md`, `docs/workflow_tracker.md`, `docs/decisions.md` | Task 5 record |

---

### Task 1: Verdict rules in `pe2/derisk.py`

**Files:**
- Modify: `pe2/derisk.py` (append after `start_is_adequate`)
- Test: `tests/pe2/test_derisk.py`

**Interfaces:**
- Consumes: `pilot_passes(rows, threshold, from_step)`, `pe2.eval.wilson_ci(k, n)`.
- Produces:
  ```python
  VERDICTS = ("COLLAPSED", "PRESERVED", "IMPROVED", "UNDETERMINED")
  def improved(k: int, n: int, start_rate: float) -> bool
  def round_verdict(rows: list[dict], final_k: int, final_n: int, start_rate: float,
                    threshold: float = CLONE_CI_LO, from_step: int = 500_000) -> tuple[str, str]
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/pe2/test_derisk.py`:

```python
class TestRoundVerdict:
    """T0.4 retry-round rules (spec C2): pass = >= 0.137 at every probe from
    500k; improved = final N=500 Wilson lower bound above the start."""

    def _rows(self, rates, every=100_000):
        return [
            {"steps": (i + 1) * every, "win_rate": r, "k": int(round(r * 200)), "n": 200}
            for i, r in enumerate(rates)
        ]

    def test_improvement_needs_the_lower_bound_not_the_point(self):
        from pe2.derisk import improved

        # 110/500 = 0.220 has Wilson lo 0.1859 > 0.185; 109/500 does not.
        assert improved(110, 500, 0.185)
        assert not improved(109, 500, 0.185)
        # PPO's start is 0.205: needs 121/500.
        assert improved(121, 500, 0.205)
        assert not improved(120, 500, 0.205)

    def test_collapse_inside_the_window(self):
        from pe2.derisk import round_verdict

        rows = self._rows([0.2] * 4 + [0.10] + [0.2] * 5)  # dip at 500k
        verdict, reason = round_verdict(rows, 150, 500, 0.185)
        assert verdict == "COLLAPSED"
        assert "500000" in reason and "0.100" in reason

    def test_dip_before_the_window_is_forgiven(self):
        from pe2.derisk import round_verdict

        rows = self._rows([0.2, 0.03, 0.09, 0.14] + [0.2] * 6)  # P-C's shape
        verdict, _ = round_verdict(rows, 150, 500, 0.185)
        assert verdict == "IMPROVED"

    def test_preserved_when_final_is_inside_the_start_interval(self):
        from pe2.derisk import round_verdict

        rows = self._rows([0.2] * 10)
        verdict, reason = round_verdict(rows, 100, 500, 0.185)  # 0.200, lo 0.168
        assert verdict == "PRESERVED"
        assert "0.185" in reason

    def test_empty_window_is_undetermined_not_collapsed(self):
        from pe2.derisk import round_verdict

        rows = self._rows([0.2] * 4)  # run died before 500k
        verdict, reason = round_verdict(rows, 150, 500, 0.185)
        assert verdict == "UNDETERMINED"
        assert "500000" in reason

    def test_every_verdict_is_in_the_registry(self):
        from pe2.derisk import VERDICTS, round_verdict

        for rows, k in (
            (self._rows([0.2] * 10), 150),
            (self._rows([0.2] * 10), 100),
            (self._rows([0.2] * 4 + [0.05] * 6), 150),
            (self._rows([0.2]), 150),
        ):
            assert round_verdict(rows, k, 500, 0.185)[0] in VERDICTS
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk.py::TestRoundVerdict -q`
Expected: FAIL with `ImportError: cannot import name 'improved'`.

- [ ] **Step 3: Implement**

Append to `pe2/derisk.py`:

```python
# ---------------------------------------------------------------------------
# T0.4 retry-round verdicts (docs/pathway/t0_retry_round_prereg_2026-09-06.md
# §C2). Pre-registered before any arm ran.
# ---------------------------------------------------------------------------

VERDICTS = ("COLLAPSED", "PRESERVED", "IMPROVED", "UNDETERMINED")
ROUND_PASS_FROM = 500_000


def improved(k: int, n: int, start_rate: float) -> bool:
    """Improvement rule: the final probe's Wilson LOWER bound exceeds the
    arm's own START point estimate. The lower bound, not the point, so a
    survivor is statistically separated from where it began."""
    from pe2.eval import wilson_ci

    return wilson_ci(k, n)[0] > start_rate


def round_verdict(
    rows: list[dict],
    final_k: int,
    final_n: int,
    start_rate: float,
    threshold: float = CLONE_CI_LO,
    from_step: int = ROUND_PASS_FROM,
) -> tuple[str, str]:
    """(verdict, reason) for one retry-round arm.

    COLLAPSED: a probe inside the judging window fell below `threshold`.
    PRESERVED: passed the window, but the final probe is not separated from
    the start. IMPROVED: passed AND `improved`. UNDETERMINED: the window is
    empty (a crashed run is neither a pass nor a collapse). The reason is
    always worded so a reader never meets a bare label.
    """
    from pe2.eval import wilson_ci

    window = [r for r in rows if r["steps"] >= from_step]
    if not window:
        return "UNDETERMINED", (
            f"judging window empty: no probe at or after {from_step} steps "
            f"(probes seen: {[r['steps'] for r in rows]})"
        )
    if not pilot_passes(rows, threshold, from_step):
        worst = min(window, key=lambda r: r["win_rate"])
        return "COLLAPSED", (
            f"win rate {worst['win_rate']:.3f} ({worst['k']}/{worst['n']} "
            f"@{worst['steps']}) below {threshold} inside the window from {from_step}"
        )
    lo, hi = wilson_ci(final_k, final_n)
    final = f"final {final_k}/{final_n} = {final_k / final_n:.3f}, Wilson [{lo:.3f}, {hi:.3f}]"
    if lo > start_rate:
        return "IMPROVED", f"{final}: lower bound above the start {start_rate:.3f}"
    return "PRESERVED", (
        f"{final}: lower bound does not exceed the start {start_rate:.3f} — "
        "no collapse, no separated improvement"
    )
```

- [ ] **Step 4: Run**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk.py -q` → PASS (the file's SB3-dependent tests skip or pass as before).

- [ ] **Step 5: Commit**

```bash
./venv/bin/ruff check pe2/derisk.py tests/pe2/test_derisk.py && ./venv/bin/ruff format --check pe2/derisk.py tests/pe2/test_derisk.py
git add pe2/derisk.py tests/pe2/test_derisk.py
git commit -m "feat(derisk): retry-round verdict rules (pass window, improvement, UNDETERMINED)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `CheckpointEvalProbe.evaluate_at`

**Files:**
- Modify: `pe2/exploitability.py` (`CheckpointEvalProbe.__init__`, lines ~782–846)
- Test: `tests/pe2/test_exploitability.py`

**Interfaces:**
- Produces: `probe.evaluate_at(model, steps: int) -> dict` — saves `ckpt_dir/{label}_{steps}.zip`, evaluates on `eval_n` paired seeds, appends and flushes the row `{"steps", "ckpt", "win_rate", "k", "n"}`, returns it. `probe.callback._on_step` now calls it; behaviour of the callback path is unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/pe2/test_exploitability.py` (next to the existing probe tests; reuse their fixture style):

```python
class TestProbeEvaluateAt:
    def test_manual_probe_appends_the_same_row_shape_as_the_callback(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import SAC

        from pe2.config import EnvConfig
        from pe2.exploitability import CheckpointEvalProbe, SingleAgentWrapper
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=30)
        wrapper = SingleAgentWrapper(cfg, "pursuer", GreedyEvader)
        model = SAC("MlpPolicy", wrapper, seed=0, verbose=0, device="cpu")
        probe = CheckpointEvalProbe(
            cfg, "pursuer", GreedyEvader, every=1000, eval_n=2, seed0=990000,
            ckpt_dir=tmp_path, report_path=tmp_path / "r.json", label="T",
        )
        row = probe.evaluate_at(model, 1000)
        row2 = probe.evaluate_at(model, 2000)
        assert probe.rows == [row, row2]
        assert set(row) == {"steps", "ckpt", "win_rate", "k", "n"}
        assert row["steps"] == 1000 and row["n"] == 2
        assert (tmp_path / "T_1000.zip").exists() and (tmp_path / "T_2000.zip").exists()
        import json

        assert json.load(open(tmp_path / "r.json"))["checkpoints"] == probe.rows
```

- [ ] **Step 2: Run to verify failure** — `AttributeError: 'CheckpointEvalProbe' object has no attribute 'evaluate_at'`.

- [ ] **Step 3: Implement**

Inside `CheckpointEvalProbe.__init__`, replace the `_CB` class body with a closure plus a thin callback:

```python
        rows: list[dict] = []

        def evaluate_at(model, steps: int) -> dict:
            """Checkpoint + paired eval at `steps`; appends and flushes the row.
            Public so a training loop that is not an SB3 `learn` (pe2.awr)
            can probe on exactly the same grid, seeds and artefacts."""
            path = ckpt_dir / f"{label}_{steps}.zip"
            model.save(path)
            logs = evaluate_sb3_policy(
                model, cfg, role, opponent_factory, n_episodes=eval_n, seed0=seed0
            )
            k, n = win_count(logs, role)
            row = {"steps": steps, "ckpt": str(path), "win_rate": k / n, "k": k, "n": n}
            rows.append(row)
            with open(report_path, "w") as f:
                json.dump({"checkpoints": rows}, f, indent=2)
            print(f"  @{steps:>8}: selection win rate {k}/{n} = {k / n:.3f}", flush=True)
            return row

        class _CB(BaseCallback):
            def _on_step(inner_self) -> bool:
                if inner_self.num_timesteps % every == 0:
                    evaluate_at(inner_self.model, inner_self.num_timesteps)
                return True

        self.rows = rows
        self.callback = _CB()
        self.evaluate_at = evaluate_at
```

- [ ] **Step 4: Run** — `tests/pe2/test_exploitability.py -q` → PASS (existing probe tests prove the callback path is unchanged).

- [ ] **Step 5: Commit** — `git add pe2/exploitability.py tests/pe2/test_exploitability.py`; message `refactor(exploitability): CheckpointEvalProbe.evaluate_at for non-SB3 loops`.

---

### Task 3: Advantage-weighted regression `pe2/awr.py`

**Files:**
- Create: `pe2/awr.py`
- Test: `tests/pe2/test_awr.py`

**Interfaces:**
- Consumes: `pe2.derisk.mc_returns`, `pe2.derisk.seed_torch`, `pe2.exploitability.SingleAgentWrapper`, `pe2.obs.obs_dim`, SB3 `SAC.load`, `model.predict`, `model.actor.get_action_dist_params`, `model.device`.
- Produces (all in `pe2/awr.py`):
  ```python
  LAM = 0.5; W_MAX = 20.0
  @dataclass(frozen=True) class AwrConfig(lam=0.5, w_max=20.0, advantage_sign=1.0, gamma=0.999,
      initial_episodes=300, buffer_episodes=300, steps_per_iter=50_000, value_epochs=20,
      value_lr=1e-3, actor_epochs=10, actor_lr=3e-4, batch_size=512, hidden=256)
  @dataclass class Episode(obs, actions, returns, won, length, seed)
  def rollout_episode(model, wrapper, role, seed, gamma, deterministic=False) -> Episode
  def collect(model, wrapper, role, seed0, min_steps, gamma, min_episodes=1) -> list[Episode]
  def advantage_weights(returns, values, lam, w_max, sign) -> np.ndarray
  def make_value_net(obs_dim, hidden, device)
  def fit_value(value_net, opt, obs, returns, epochs, batch_size, rng) -> list[float]
  def awr_actor_update(model, opt, obs, actions, weights, epochs, batch_size, rng) -> list[float]
  def train_awr(cfg, role, opponent_factory, clone_path, total_timesteps, seed, awr, probe_every,
                probe, device="auto", rollout_seed0=820_000) -> tuple[model, dict]
  ```
  `train_awr` calls `probe(model, grid_steps)` the first time the cumulative env-step count reaches each multiple of `probe_every` up to and including `total_timesteps`; `stats` has `torch_seed, seed, rollout_seed0, total_steps, n_episodes, positives, positives_per_1M, probe_marks (list of {grid, actual, iteration}), iterations (list of per-iteration dicts), config`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pe2/test_awr.py`:

```python
"""Tests for pe2.awr — advantage-weighted regression on MC returns (T0.4 R-D)."""

import numpy as np
import pytest

from pe2.config import EnvConfig


class TestAdvantageWeights:
    def test_formula_clip_and_sign(self):
        from pe2.awr import advantage_weights

        r = np.array([1.0, -1.0, 0.0])
        v = np.array([0.0, 0.0, 0.0])
        w = advantage_weights(r, v, lam=0.5, w_max=20.0, sign=1.0)
        assert w[0] == pytest.approx(np.exp(2.0))  # 7.39, under the clip
        assert w[1] == pytest.approx(np.exp(-2.0))
        assert w[2] == pytest.approx(1.0)
        flipped = advantage_weights(r, v, lam=0.5, w_max=20.0, sign=-1.0)
        assert flipped[0] == pytest.approx(w[1]) and flipped[1] == pytest.approx(w[0])
        big = advantage_weights(np.array([10.0]), np.array([0.0]), 0.5, 20.0, 1.0)
        assert big[0] == 20.0

    def test_defaults_are_the_preregistered_values(self):
        from pe2.awr import LAM, W_MAX, AwrConfig

        c = AwrConfig()
        assert (LAM, W_MAX) == (0.5, 20.0)
        assert (c.lam, c.w_max, c.advantage_sign, c.gamma) == (0.5, 20.0, 1.0, 0.999)
        assert (c.initial_episodes, c.buffer_episodes, c.steps_per_iter) == (300, 300, 50_000)


class TestValueFit:
    def test_regression_descends_on_a_linear_target(self):
        torch = pytest.importorskip("torch")
        from pe2.awr import fit_value, make_value_net

        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        obs = torch.as_tensor(rng.normal(size=(2000, 4)), dtype=torch.float32)
        target = obs[:, 0] * 0.5 - obs[:, 1] * 0.25
        net = make_value_net(4, 32, "cpu")
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        losses = fit_value(net, opt, obs, target, epochs=30, batch_size=256, rng=rng)
        assert losses[-1] < 0.1 * losses[0]


class TestActorUpdate:
    def _model(self, cfg):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import SAC

        from pe2.exploitability import SingleAgentWrapper
        from pe2.scripted import GreedyEvader

        return SAC("MlpPolicy", SingleAgentWrapper(cfg, "pursuer", GreedyEvader), seed=0, verbose=0, device="cpu")

    def test_uniform_weights_move_tanh_mu_toward_the_actions(self):
        torch = pytest.importorskip("torch")
        from pe2.awr import awr_actor_update
        from pe2.obs import obs_dim

        cfg = EnvConfig()
        model = self._model(cfg)
        rng = np.random.default_rng(1)
        obs = torch.as_tensor(rng.uniform(-1, 1, size=(512, obs_dim(cfg))), dtype=torch.float32)
        acts = torch.full((512, 2), 0.7)
        with torch.no_grad():
            before = torch.tanh(model.actor.get_action_dist_params(obs)[0])
        opt = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
        awr_actor_update(model, opt, obs, acts, torch.ones(512), epochs=20, batch_size=128, rng=rng)
        with torch.no_grad():
            after = torch.tanh(model.actor.get_action_dist_params(obs)[0])
        assert ((after - acts) ** 2).mean() < ((before - acts) ** 2).mean()

    def test_weights_steer_which_actions_are_matched(self):
        torch = pytest.importorskip("torch")
        from pe2.awr import awr_actor_update
        from pe2.obs import obs_dim

        cfg = EnvConfig()
        model = self._model(cfg)
        rng = np.random.default_rng(2)
        obs = torch.as_tensor(rng.uniform(-1, 1, size=(1, obs_dim(cfg))), dtype=torch.float32).repeat(400, 1)
        # Same state, two candidate actions: +0.8 (heavily weighted) vs -0.8.
        acts = torch.cat([torch.full((200, 2), 0.8), torch.full((200, 2), -0.8)])
        w = torch.cat([torch.full((200,), 20.0), torch.full((200,), 0.05)])
        opt = torch.optim.Adam(model.actor.parameters(), lr=3e-3)
        awr_actor_update(model, opt, obs, acts, w, epochs=40, batch_size=400, rng=rng)
        with torch.no_grad():
            out = torch.tanh(model.actor.get_action_dist_params(obs[:1])[0])
        assert (out > 0.4).all()  # pulled to the heavily weighted action


class TestTrainAwr:
    def test_tiny_run_probes_on_the_grid_and_returns_a_usable_model(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import SAC

        from pe2.awr import AwrConfig, train_awr
        from pe2.exploitability import SingleAgentWrapper
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=40)
        clone = tmp_path / "clone.zip"
        SAC("MlpPolicy", SingleAgentWrapper(cfg, "pursuer", GreedyEvader), seed=0, verbose=0, device="cpu").save(clone)
        awr = AwrConfig(initial_episodes=5, buffer_episodes=8, steps_per_iter=100, value_epochs=2, actor_epochs=2, batch_size=64)
        marks = []
        model, stats = train_awr(
            cfg, "pursuer", GreedyEvader, clone, total_timesteps=800, seed=3, awr=awr,
            probe_every=200, probe=lambda m, s: marks.append(s), device="cpu", rollout_seed0=990000,
        )
        assert marks == [200, 400, 600, 800]
        assert stats["total_steps"] >= 800 and stats["n_episodes"] >= 5
        assert stats["iterations"][0]["phase"] == "offline"
        assert all(it["phase"] == "online" for it in stats["iterations"][1:])
        assert [m["grid"] for m in stats["probe_marks"]] == marks
        assert stats["config"]["advantage_sign"] == 1.0
        action, _ = model.predict(np.zeros(SingleAgentWrapper(cfg, "pursuer", GreedyEvader).observation_space.shape, dtype=np.float32), deterministic=True)
        assert action.shape == (2,)

    def test_same_seed_same_stats(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import SAC

        from pe2.awr import AwrConfig, train_awr
        from pe2.exploitability import SingleAgentWrapper
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=40)
        clone = tmp_path / "clone.zip"
        SAC("MlpPolicy", SingleAgentWrapper(cfg, "pursuer", GreedyEvader), seed=0, verbose=0, device="cpu").save(clone)
        awr = AwrConfig(initial_episodes=4, buffer_episodes=6, steps_per_iter=80, value_epochs=1, actor_epochs=1, batch_size=64)
        runs = [
            train_awr(cfg, "pursuer", GreedyEvader, clone, 300, seed=5, awr=awr, probe_every=150, probe=lambda m, s: None, device="cpu", rollout_seed0=990100)[1]
            for _ in range(2)
        ]
        assert runs[0]["iterations"] == runs[1]["iterations"]
        assert runs[0]["torch_seed"] == 5
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: No module named 'pe2.awr'`.

- [ ] **Step 3: Implement `pe2/awr.py`**

```python
"""Advantage-weighted regression (Peng, Kumar, Zhang & Levine 2019) on
Monte-Carlo returns — the retry round's family-independent arm (R-D).

Why this and not another actor-critic: every Tier-0 arm that bootstrapped a
critic either collapsed (D82's TD drift) or had to be pinned to the clone to
survive. AWR bootstraps NOTHING. The baseline V(s) is a regression onto the
Monte-Carlo return G_t = gamma^(L-t)·R of the episode the step came from
(`pe2.derisk.mc_returns`), and the actor is updated by weighted supervised
regression onto the actions it actually executed, with weights
exp((G_t − V(s_t)) / lam) clipped at w_max. Weights favour the actions that
led to better-than-expected outcomes; a flipped sign (the round's control)
favours the worse ones and must degrade the policy.

The actor is the SB3 SAC actor inside the clone's own model object, updated
the way `train_bc_actor` updates it (regression of tanh(mu), which is what
predict(deterministic=True) executes), so checkpoints, probes and the
evaluator are the ones every other arm uses. Rollouts are STOCHASTIC
(predict(deterministic=False)) — the training-time regime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from pe2.config import EnvConfig
from pe2.derisk import mc_returns, seed_torch
from pe2.exploitability import SingleAgentWrapper
from pe2.obs import obs_dim

LAM = 0.5  # temperature on our ±1 return range (AWR's 0.05 is for returns of tens)
W_MAX = 20.0  # AWR's weight clip


@dataclass(frozen=True)
class AwrConfig:
    lam: float = LAM
    w_max: float = W_MAX
    advantage_sign: float = 1.0  # −1.0 is the negative control (R-D-neg)
    gamma: float = 0.999
    initial_episodes: int = 300  # the offline phase: the clone's own rollouts
    buffer_episodes: int = 300  # FIFO, in episodes
    steps_per_iter: int = 50_000  # online: at least this many fresh steps per iteration
    value_epochs: int = 20
    value_lr: float = 1e-3
    actor_epochs: int = 10
    actor_lr: float = 3e-4
    batch_size: int = 512
    hidden: int = 256


@dataclass
class Episode:
    obs: np.ndarray  # (T, obs_dim) float32
    actions: np.ndarray  # (T, 2) float32 — the EXECUTED Box actions in [−1, 1]
    returns: np.ndarray  # (T,) float32 — G_t under the terminal-only reward
    won: bool
    length: int
    seed: int


def rollout_episode(model, wrapper, role, seed, gamma, deterministic=False) -> Episode:
    obs, _ = wrapper.reset(seed=seed)
    obs_l, act_l = [], []
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs_l.append(np.asarray(obs, dtype=np.float32))
        act_l.append(np.asarray(action, dtype=np.float32))
        obs, _reward, terminated, truncated, info = wrapper.step(action)
        if terminated or truncated:
            break
    length = wrapper.env.current_step
    won = (info["outcome"] == "capture") == (role == "pursuer")
    returns = np.asarray(mc_returns(length, 1.0 if won else -1.0, gamma), dtype=np.float32)
    return Episode(np.stack(obs_l), np.stack(act_l), returns, won, length, seed)


def collect(model, wrapper, role, seed0, min_steps, gamma, min_episodes=1) -> list[Episode]:
    """Episodes on seeds seed0, seed0+1, … until >= min_steps AND >= min_episodes."""
    episodes, steps, i = [], 0, 0
    while steps < min_steps or len(episodes) < min_episodes:
        ep = rollout_episode(model, wrapper, role, seed0 + i, gamma)
        i += 1
        episodes.append(ep)
        steps += ep.length
    return episodes


def advantage_weights(returns, values, lam, w_max, sign) -> np.ndarray:
    adv = sign * (np.asarray(returns) - np.asarray(values))
    return np.minimum(np.exp(adv / lam), w_max)


def make_value_net(obs_dim_: int, hidden: int, device):
    from torch import nn  # lazy: optional dependency

    return nn.Sequential(
        nn.Linear(obs_dim_, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(device)


def fit_value(value_net, opt, obs, returns, epochs, batch_size, rng) -> list[float]:
    import torch

    n = len(obs)
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = torch.as_tensor(perm[i : i + batch_size], device=obs.device)
            loss = torch.nn.functional.mse_loss(value_net(obs[idx]).squeeze(-1), returns[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        losses.append(total / n)
    return losses


def awr_actor_update(model, opt, obs, actions, weights, epochs, batch_size, rng) -> list[float]:
    """Weighted regression of tanh(mu(s)) onto the executed actions —
    `train_bc_actor`'s step with per-sample weights (normalised per batch)."""
    import torch

    n = len(obs)
    losses = []
    model.actor.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = torch.as_tensor(perm[i : i + batch_size], device=obs.device)
            mean_actions, _log_std, _kwargs = model.actor.get_action_dist_params(obs[idx])
            se = ((torch.tanh(mean_actions) - actions[idx]) ** 2).sum(dim=1)
            loss = (weights[idx] * se).sum() / weights[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        losses.append(total / n)
    model.actor.eval()
    return losses


def train_awr(
    cfg: EnvConfig,
    role: str,
    opponent_factory,
    clone_path,
    total_timesteps: int,
    seed: int,
    awr: AwrConfig,
    probe_every: int,
    probe,
    device: str = "auto",
    rollout_seed0: int = 820_000,
):
    """Offline AWR on the clone's rollouts, then iterated online AWR, until
    `total_timesteps` env steps have been collected (the offline rollouts
    count). `probe(model, grid_steps)` fires the first time the cumulative
    count reaches each multiple of `probe_every` (≤ total). Returns
    (model, stats)."""
    import torch
    from stable_baselines3 import SAC

    torch_seed = seed_torch(seed)
    rng = np.random.default_rng(seed)
    model = SAC.load(clone_path, device=device)
    dev = model.device
    wrapper = SingleAgentWrapper(cfg, role, opponent_factory)
    value_net = make_value_net(obs_dim(cfg), awr.hidden, dev)
    v_opt = torch.optim.Adam(value_net.parameters(), lr=awr.value_lr)
    a_opt = torch.optim.Adam(model.actor.parameters(), lr=awr.actor_lr)

    buffer: list[Episode] = []
    steps = 0
    n_episodes = 0
    positives = 0
    next_seed = rollout_seed0
    next_probe = probe_every
    probe_marks: list[dict] = []
    iterations: list[dict] = []

    def absorb(episodes: list[Episode]) -> None:
        nonlocal steps, n_episodes, positives, next_seed
        for ep in episodes:
            steps += ep.length
            n_episodes += 1
            positives += int(ep.won)
        next_seed += len(episodes)
        buffer.extend(episodes)
        del buffer[: max(0, len(buffer) - awr.buffer_episodes)]

    def update(phase: str) -> dict:
        obs = torch.as_tensor(np.concatenate([e.obs for e in buffer]), device=dev)
        acts = torch.as_tensor(np.concatenate([e.actions for e in buffer]), device=dev)
        rets_np = np.concatenate([e.returns for e in buffer])
        rets = torch.as_tensor(rets_np, device=dev)
        v_losses = fit_value(value_net, v_opt, obs, rets, awr.value_epochs, awr.batch_size, rng)
        with torch.no_grad():
            values = value_net(obs).squeeze(-1).cpu().numpy()
        w = advantage_weights(rets_np, values, awr.lam, awr.w_max, awr.advantage_sign)
        a_losses = awr_actor_update(
            model, a_opt, obs, acts, torch.as_tensor(w, dtype=torch.float32, device=dev),
            awr.actor_epochs, awr.batch_size, rng,
        )
        return {
            "phase": phase,
            "steps": steps,
            "n_samples": int(len(obs)),
            "buffer_episodes": len(buffer),
            "value_loss_first": v_losses[0],
            "value_loss_last": v_losses[-1],
            "actor_loss_first": a_losses[0],
            "actor_loss_last": a_losses[-1],
            "advantage_mean": float((rets_np - values).mean()),
            "weight_mean": float(w.mean()),
            "weight_clipped_fraction": float((w >= awr.w_max).mean()),
        }

    def fire_probes(iteration: int) -> None:
        nonlocal next_probe
        while next_probe <= total_timesteps and steps >= next_probe:
            probe(model, next_probe)
            probe_marks.append({"grid": next_probe, "actual": steps, "iteration": iteration})
            next_probe += probe_every

    absorb(collect(model, wrapper, role, next_seed, 0, awr.gamma, min_episodes=awr.initial_episodes))
    iterations.append(update("offline"))
    fire_probes(0)
    while steps < total_timesteps:
        absorb(collect(model, wrapper, role, next_seed, awr.steps_per_iter, awr.gamma))
        iterations.append(update("online"))
        fire_probes(len(iterations) - 1)

    stats = {
        "torch_seed": torch_seed,
        "seed": seed,
        "rollout_seed0": rollout_seed0,
        "total_steps": steps,
        "n_episodes": n_episodes,
        "positives": positives,
        "positives_per_1M": positives / steps * 1e6,
        "probe_marks": probe_marks,
        "iterations": iterations,
        "config": asdict(awr),
    }
    return model, stats
```

- [ ] **Step 4: Run** — `tests/pe2/test_awr.py -q` → PASS (allow ~1–2 min; the tiny runs roll real episodes at `max_steps=40`).

- [ ] **Step 5: Commit** — `git add pe2/awr.py tests/pe2/test_awr.py`; message `feat(awr): advantage-weighted regression on MC returns (retry-round R-D)`.

---

### Task 4: The round runner `scripts/derisk_t0b_round.py`

**Files:**
- Create: `scripts/derisk_t0b_round.py`
- Test: `tests/pe2/test_derisk_t0b_round.py`

**Interfaces:**
- Consumes: `finetune_br_anchored`, `finetune_ppo`, `CheckpointEvalProbe` (+ `evaluate_at`), `TrainingCaptureCounter`, `AnchorKLProbe`, `anchor_kl_grew`, `train_awr`, `AwrConfig`, `evaluate_sb3_policy`, `win_count`, `wilson_ci`, `round_verdict`, `control_collapsed`, `SAC.load` (final eval is ALWAYS from the saved `ckpts_{ARM}/{ARM}_{total}.zip`, so the number is tied to an artefact).
- Produces: `results/derisk/t0b_arm_{ARM}.json`, `results/derisk/t0b_round.{json,png}`; pure functions `arm_result_verdict(result, control) -> (verdict, reason)`, `summarize(results: dict) -> dict`, `write_figure(results, summary, path)`.

- [ ] **Step 1: Tests first**

Create `tests/pe2/test_derisk_t0b_round.py`:

```python
"""scripts/derisk_t0b_round.py — the retry round's own decisions."""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_t0b_round.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_t0b_round", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _arm(name, verdict, rows=None, final=(120, 500)):
    return {
        "arm": name, "verdict": verdict, "reason": "test",
        "checkpoints": rows or [{"steps": s, "win_rate": 0.2, "k": 40, "n": 200} for s in range(100_000, 1_000_001, 100_000)],
        "final": {"k": final[0], "n": final[1], "win_rate": final[0] / final[1]},
        "start": {"k": 37, "n": 200, "rate": 0.185},
    }


class TestPreRegisteredConstants:
    def test_protocol(self):
        assert (R.TOTAL, R.EVERY, R.EVAL_N, R.FINAL_EVAL_N, R.PASS_FROM) == (1_000_000, 100_000, 200, 500, 500_000)
        # SUPERSEDED by amendment C3-a: NEG_TOTAL is 500_000, not 200_000.
        # The shipped test asserts 500_000; see tests/pe2/test_derisk_t0b_round.py.
        assert (R.BETA_RB, R.CRITIC_WARMUP, R.NEG_TOTAL) == (0.1, 50_000, 500_000)
        assert (R.EVAL_SEED0, R.AWR_SEED0, R.AWRNEG_SEED0) == (780_000, 820_000, 830_000)

    def test_seeds_and_starts(self):
        assert R.SEEDS == {"RB": 7, "RC": 9, "RD": 11, "RDNEG": 11}
        assert R.STARTS == {"RB": (37, 200), "RC": (41, 200), "RD": (37, 200), "RDNEG": (37, 200)}


class TestSummarize:
    def test_survivors_are_improved_only(self):
        s = R.summarize({"RB": _arm("RB", "PRESERVED"), "RC": _arm("RC", "IMPROVED"), "RD": _arm("RD", "IMPROVED"), "RDNEG": {"arm": "RDNEG", "verdict": True, "reason": "collapsed"}})
        assert s["survivors"] == ["RC", "RD"] and s["complete"] is True

    def test_rd_is_withheld_when_its_control_did_not_collapse(self):
        s = R.summarize({"RD": _arm("RD", "IMPROVED"), "RDNEG": {"arm": "RDNEG", "verdict": None, "reason": "not separated"}})
        assert s["arms"]["RD"]["verdict"] == "UNDETERMINED"
        assert "control" in s["arms"]["RD"]["reason"]
        assert s["survivors"] == []

    def test_rd_is_withheld_when_its_control_is_missing(self):
        s = R.summarize({"RD": _arm("RD", "IMPROVED")})
        assert s["arms"]["RD"]["verdict"] == "UNDETERMINED" and s["complete"] is False

    def test_rb_rc_do_not_depend_on_rdneg(self):
        s = R.summarize({"RB": _arm("RB", "IMPROVED")})
        assert s["arms"]["RB"]["verdict"] == "IMPROVED" and s["survivors"] == ["RB"]

    def test_direction_text_follows_c4(self):
        assert "Route C" in R.summarize({"RB": _arm("RB", "PRESERVED"), "RC": _arm("RC", "COLLAPSED"), "RD": _arm("RD", "PRESERVED"), "RDNEG": {"arm": "RDNEG", "verdict": True, "reason": ""}})["direction"]
        assert "Track 2" in R.summarize({"RB": _arm("RB", "IMPROVED")})["direction"]


class TestFigure:
    def test_writes_with_partial_arms(self, tmp_path):
        pytest.importorskip("matplotlib")
        results = {"RB": _arm("RB", "PRESERVED")}
        R.write_figure(results, R.summarize(results), tmp_path / "f.png")
        assert (tmp_path / "f.png").stat().st_size > 0


class TestArmJsonRoundTrip:
    def test_load_results_reads_only_arm_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "t0b_arm_RB.json").write_text(json.dumps(_arm("RB", "PRESERVED")))
        (tmp_path / "t0_pilots.json").write_text("{}")
        assert set(R.load_results()) == {"RB"}
```

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_t0b_round.py -q` → FAIL (`FileNotFoundError` for the script).

- [ ] **Step 2: Implement the runner**

```python
"""T0.4 retry round — the pre-registered learner round after Tier 0 (D98).

Spec: docs/pathway/t0_retry_round_prereg_2026-09-06.md, Part C, reproduced
here verbatim where it binds. Nothing here may change a threshold, a seed, a
budget or a definition after a number is seen.

Arms (pursuer vs frozen GreedyEvader, 1M steps, probes every 100k on 200
paired seeds 780000+, final probe N=500):
  RB    KL-anchored SAC fine-tune, beta = 0.1 (the A4 pre-commitment)
  RC    PPO-BC clone -> PPO+GAE at 1M (the only Tier-0 arm rising at cutoff)
  RD    advantage-weighted regression on MC returns, offline then online
  RDNEG RD with the advantage sign flipped, 200k: MUST collapse (control)
Pass = win rate >= 0.137 at every probe from 500k. Improved = final N=500
Wilson lower bound > the arm's own start (37/200 = 0.185 SAC-family;
41/200 = 0.205 PPO). Survivor = IMPROVED. RD is withheld unless RDNEG
collapses by `control_collapsed`.

Usage:
  ./venv/bin/python scripts/derisk_t0b_round.py --arms RB        (one process per arm)
  ./venv/bin/python scripts/derisk_t0b_round.py --summarize      (after all four)
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from pe2.config import EnvConfig
from pe2.derisk import (
    CLONE_CI_LO,
    TrainingCaptureCounter,
    control_collapsed,
    round_verdict,
)
from pe2.eval import wilson_ci, win_count
from pe2.scripted import GreedyEvader

OUT = pathlib.Path("results/derisk")
CLONE = pathlib.Path("results/visual_gates/step_05/sac_bc_pursuer.zip")
PPO_BC = OUT / "ppo_bc_pursuer.zip"  # Tier 0's PPO-BC clone, start 41/200

# --- pre-registered (spec C2/C3). Do not tune. ---
TOTAL = 1_000_000
EVERY = 100_000
EVAL_N = 200
FINAL_EVAL_N = 500
PASS_FROM = 500_000
EVAL_SEED0 = 780_000  # Tier 0's probe stream: every probe is paired with Tier 0's
BETA_RB = 0.1
CRITIC_WARMUP = 50_000
NEG_TOTAL = 500_000  # amendment C3-a (was 200_000); see the Global Constraints note
AWR_SEED0 = 820_000
AWRNEG_SEED0 = 830_000
SEEDS = {"RB": 7, "RC": 9, "RD": 11, "RDNEG": 11}
STARTS = {"RB": (37, 200), "RC": (41, 200), "RD": (37, 200), "RDNEG": (37, 200)}
START_PROVENANCE = {
    "RB": "gated SAC clone, D76 battery (S119): 37/200 = 0.185, Wilson [0.137, 0.245]",
    "RC": "Tier 0 arms.PC.bc_start (S122): 41/200 = 0.205",
    "RD": "gated SAC clone, D76 battery (S119): 37/200 = 0.185",
    "RDNEG": "gated SAC clone, D76 battery (S119): 37/200 = 0.185",
}
TIER0_PC_EXPECTED = {100_000: 7, 200_000: 28}  # RC must reproduce these (determinism)
LEARNER_ARMS = ("RB", "RC", "RD")


def start_rate(arm: str) -> float:
    k, n = STARTS[arm]
    return k / n


def probe(cfg, arm):
    from pe2.exploitability import CheckpointEvalProbe

    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    return CheckpointEvalProbe(
        cfg, "pursuer", GreedyEvader, every=EVERY, eval_n=EVAL_N, seed0=EVAL_SEED0,
        ckpt_dir=d, report_path=OUT / f"t0b_probe_{arm}.json", label=arm,
    )


def final_eval(cfg, arm, total, device) -> dict:
    """The N=500 improvement probe, ALWAYS from the saved final checkpoint so
    the number is tied to an artefact a reader can re-run."""
    from stable_baselines3 import SAC, PPO
    from pe2.exploitability import evaluate_sb3_policy

    path = OUT / f"ckpts_{arm}" / f"{arm}_{total}.zip"
    model = (PPO if arm == "RC" else SAC).load(path, device=device)
    logs = evaluate_sb3_policy(model, cfg, "pursuer", GreedyEvader, n_episodes=FINAL_EVAL_N, seed0=EVAL_SEED0)
    k, n = win_count(logs, "pursuer")
    lo, hi = wilson_ci(k, n)
    return {"ckpt": str(path), "k": k, "n": n, "win_rate": k / n, "wilson": [lo, hi], "eval_seed0": EVAL_SEED0}


def run_arm(arm: str, cfg: EnvConfig, device: str) -> dict:
    from stable_baselines3.common.callbacks import CallbackList
    from pe2.anchored_sac import AnchorKLProbe, anchor_kl_grew
    from pe2.awr import AwrConfig, train_awr
    from pe2.exploitability import finetune_br_anchored, finetune_ppo

    total = NEG_TOTAL if arm == "RDNEG" else TOTAL
    pr = probe(cfg, arm)
    counter = TrainingCaptureCounter(snapshot_every=EVERY)
    extra: dict = {}
    if arm == "RB":
        cb = CallbackList([pr.callback, counter.callback, AnchorKLProbe(pr.rows, EVERY).callback])
        finetune_br_anchored(cfg, "pursuer", GreedyEvader, CLONE, total, seed=SEEDS[arm], beta=BETA_RB, device=device, critic_warmup=CRITIC_WARMUP, callback=cb)
        training = dict(counter.state, positives_per_1M=counter.positives_per_1M, segments=counter.segments)
        grew, why = anchor_kl_grew(pr.rows)
        extra["anchor"] = {"beta": BETA_RB, "kl_grew": grew, "kl_reason": why}
    elif arm == "RC":
        cb = CallbackList([pr.callback, counter.callback])
        finetune_ppo(cfg, "pursuer", GreedyEvader, PPO_BC, total, seed=SEEDS[arm], device=device, callback=cb)
        training = dict(counter.state, positives_per_1M=counter.positives_per_1M, segments=counter.segments)
        observed = {r["steps"]: r["k"] for r in pr.rows if r["steps"] in TIER0_PC_EXPECTED}
        extra["tier0_replication"] = {"expected_k": TIER0_PC_EXPECTED, "observed_k": observed, "reproduced": observed == TIER0_PC_EXPECTED}
    elif arm in ("RD", "RDNEG"):
        awr = AwrConfig(advantage_sign=1.0 if arm == "RD" else -1.0)
        _, stats = train_awr(cfg, "pursuer", GreedyEvader, CLONE, total, seed=SEEDS[arm], awr=awr, probe_every=EVERY, probe=pr.evaluate_at, device=device, rollout_seed0=AWR_SEED0 if arm == "RD" else AWRNEG_SEED0)
        training = {k: stats[k] for k in ("total_steps", "n_episodes", "positives", "positives_per_1M")}
        extra["awr"] = stats
    else:
        raise SystemExit(f"unknown arm {arm}")

    result = {
        "arm": arm, "seed": SEEDS[arm], "timesteps": total, "eval_n": EVAL_N, "eval_seed0": EVAL_SEED0,
        "start": {"k": STARTS[arm][0], "n": STARTS[arm][1], "rate": start_rate(arm), "provenance": START_PROVENANCE[arm]},
        "checkpoints": pr.rows, "training": training, **extra,
    }
    if arm == "RDNEG":
        verdict, reason = control_collapsed(pr.rows, from_step=100_000)
    else:
        result["final"] = final_eval(cfg, arm, total, device)
        verdict, reason = round_verdict(pr.rows, result["final"]["k"], result["final"]["n"], start_rate(arm), from_step=PASS_FROM)
    result["verdict"], result["reason"] = verdict, reason
    (OUT / f"t0b_arm_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] verdict={verdict}: {reason}", flush=True)
    return result


def load_results() -> dict:
    return {p.stem.removeprefix("t0b_arm_"): json.loads(p.read_text()) for p in sorted(OUT.glob("t0b_arm_*.json"))}


def summarize(results: dict) -> dict:
    """Apply the withhold rule and the C4 direction. Pure: no I/O."""
    arms = {a: {"verdict": r["verdict"], "reason": r["reason"]} for a, r in results.items()}
    ctrl = results.get("RDNEG")
    if "RD" in arms:
        if ctrl is None:
            arms["RD"] = {"verdict": "UNDETERMINED", "reason": "withheld: the AWR control (RDNEG) has not run, so the AWR loop and its probe path are unvalidated"}
        elif ctrl["verdict"] is not True:
            arms["RD"] = {"verdict": "UNDETERMINED", "reason": f"withheld: the AWR control did not demonstrably collapse ({ctrl['reason']})"}
    survivors = [a for a in LEARNER_ARMS if a in arms and arms[a]["verdict"] == "IMPROVED"]
    complete = all(a in results for a in (*LEARNER_ARMS, "RDNEG"))
    if survivors:
        best = max(survivors, key=lambda a: wilson_ci(results[a]["final"]["k"], results[a]["final"]["n"])[0])
        direction = f"IMPROVED: {survivors}. Track-1 mechanism = {best} (highest final Wilson lower bound). Write the Track 2 (evader) plan for it; two survivors run as Track-2 arms."
    elif complete:
        direction = "No arm IMPROVED. Stabilisation results are reported as method findings; an RL-trained pair that improves on its priors is not reachable at this budget with these families -> Route C (spec C4)."
    else:
        direction = "incomplete: not every arm has run"
    return {"spec": "docs/pathway/t0_retry_round_prereg_2026-09-06.md", "clone_ci_lo": CLONE_CI_LO, "pass_from": PASS_FROM, "final_eval_n": FINAL_EVAL_N, "arms": arms, "survivors": survivors, "complete": complete, "direction": direction}


def write_figure(results: dict, summary: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for arm, r in results.items():
        rows = r["checkpoints"]
        if rows:
            ax.plot([x["steps"] for x in rows], [x["win_rate"] for x in rows], marker="o", label=f"{arm} ({summary['arms'][arm]['verdict']})")
        if "final" in r:
            lo, hi = r["final"]["wilson"]
            ax.errorbar([r["timesteps"] + 20_000], [r["final"]["win_rate"]], yerr=[[r["final"]["win_rate"] - lo], [hi - r["final"]["win_rate"]]], fmt="s", capsize=3)
        ax.axhline(r["start"]["rate"], ls=":", lw=0.8)
    ax.axhline(CLONE_CI_LO, color="k", ls="--", label=f"clone CI-lo {CLONE_CI_LO}")
    ax.axvline(PASS_FROM, color="grey", ls="-.", lw=0.8, label="pass window from 500k")
    ax.set_xlabel("env steps"); ax.set_ylabel("win rate vs GreedyEvader (paired seeds 780000+)")
    title = "T0.4 retry round" + ("" if summary["complete"] else " — INCOMPLETE, do not read as a result")
    ax.set_title(title); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="*", choices=list(SEEDS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig()
    for arm in args.arms:
        if arm in ("RB", "RD", "RDNEG") and not CLONE.exists():
            raise SystemExit(f"gated clone missing: {CLONE}")
        if arm == "RC" and not PPO_BC.exists():
            raise SystemExit(f"Tier 0 PPO-BC clone missing: {PPO_BC}")
        run_arm(arm, cfg, args.device)
    if args.summarize or args.arms:
        results = load_results()
        summary = summarize(results)
        (OUT / "t0b_round.json").write_text(json.dumps({"summary": summary, "arms": results}, indent=2))
        write_figure(results, summary, OUT / "t0b_round.png")
        print(json.dumps(summary["arms"], indent=2))
        print(f"survivors={summary['survivors']}\n{summary['direction']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the tests** — `tests/pe2/test_derisk_t0b_round.py -q` → PASS. Also a smoke of the CLI's argument surface: `./venv/bin/python scripts/derisk_t0b_round.py --summarize` with no arm files present must print an incomplete summary and exit 0.

- [ ] **Step 4: Lint + commit** — `git add scripts/derisk_t0b_round.py tests/pe2/test_derisk_t0b_round.py`; message `feat(derisk): T0.4 retry-round runner (RB/RC/RD/RDNEG, verdicts, withhold rule)`.

---

### Task 5: Run the round, summarise, record

**Files:**
- Create (by the runner): `results/derisk/t0b_arm_*.json`, `t0b_round.{json,png}`, `ckpts_{RB,RC,RD,RDNEG}/`
- Modify: `docs/worklogs/2026-09-05_S122.md`, `docs/workflow_tracker.md`, `docs/decisions.md` (new D100: the round's outcome and the direction it fixes)

- [ ] **Step 1: Full suite once, on the tree that will run** — `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q` → all pass (expect ~520). Record the count.

- [ ] **Step 2: Launch four concurrent processes** (24 cores, the GPU is idle):

```bash
for A in RDNEG RD RC RB; do
  nohup ./venv/bin/python scripts/derisk_t0b_round.py --arms $A > results/derisk/t0b_$A.log 2>&1 &
done
```

Poll `tail -2 results/derisk/t0b_*.log` no more than every 10 minutes. Expected wall time ≈ 6 h (RB is the long pole). RDNEG finishes first (~20 min): read its verdict immediately — if it is not `True`, RD is already withheld by rule; do NOT stop RD (its data is still reported) and do NOT re-run RDNEG.

- [ ] **Step 3: Summarise** — `./venv/bin/python scripts/derisk_t0b_round.py --summarize`, then print per arm: verdict, reason, checkpoints, final, `training.positives_per_1M`, RB's `anchor`, RC's `tier0_replication.reproduced`, RD's last iteration stats (`weight_clipped_fraction`, `advantage_mean`). If RC's replication is `False`, that is a determinism finding to record — the verdict still stands on its own probes.

- [ ] **Step 4: Record** — Append `### T0.4 retry round` to the S122 worklog: the arm table (verdict, every probe, final N=500 with CI, start), RB's divergence vs 1e-3, RC's replication, RD's control verdict and the withhold decision if any, `positives_per_1M` per arm, the figure signed as the visual gate, and the direction sentence from the summary VERBATIM. Add D100 to `docs/decisions.md`: the outcome and the direction it fixes (C4), quoting the spec. Update the S122 tracker row and Current Blockers (add "learner mechanism undecided" → resolved or Route C). Commit the three docs.

- [ ] **Step 5: Stop.** The direction is the user's to act on: Track 2's plan (survivor) or the Route C write-up. Do not start either.

---

## Self-review

- **Spec coverage:** C2 protocol → Task 4 constants + Task 1 rules; C3 RB/RC → Task 4; C3 RD/RDNEG → Tasks 3–4; C4 direction → `summarize` + Task 5; C5 artefacts → Task 4 paths + Task 5. Track 2/3 deliberately out of scope (spec Part D).
- **Placeholders:** none. Every constant is a number from the spec; every function has its body.
- **Type consistency:** `round_verdict(rows, final_k, final_n, start_rate, threshold, from_step)` identical in Tasks 1 and 4; `probe.evaluate_at(model, steps)` in Tasks 2, 3 (as `probe=pr.evaluate_at`) and 4; `train_awr(..., probe_every, probe, device, rollout_seed0) -> (model, stats)` identical in Tasks 3 and 4; `control_collapsed(rows, from_step=100_000)` matches its existing signature `(rows, threshold, from_step)`.
