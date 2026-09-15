# Shield Attacker (S6 / Step 9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the frozen shielded defender as an opponent object, the structured full-state evader environment with the hugging law and a stall bonus, the scripted comparators, and the S6 runner (comparators → fourteen arms → H2 / H2-implicit / R3-b / harness verdicts, figure, GIF).

**Architecture:** `pe2/structured.py` gains `StructuredPursuerPolicy` (law + PPO decision held `hold` steps + shield + optional shield-state features), reproducing `SearchMDP`'s executed commands bit-for-bit so it can serve as the opponent inside `SingleAgentWrapper(role="evader")`. `pe2/obs.py` gains `full_state_observation`. New module `pe2/evade.py` holds `EvadeMDP` (fixed `ObstacleHuggingEvader` law within `d_pos`, learned positioning beyond, stall bonus from the defender's shield metrics), `SplitHugEvader` (the structure's scripted positioning rule), `evaluate_evader`, `play_scripted_evader`, and `EvaderEvalProbe`. `pe2/derisk.py` gains `paired_indicator_test`. `scripts/derisk_s6_attacker.py` is the runner, modelled on `scripts/derisk_s5_remedy.py`.

**Tech Stack:** Python 3.12, `./venv/bin/python`, stable-baselines3 2.7, gymnasium, numpy, matplotlib, imageio, pytest (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), ruff.

**Spec:** `docs/pathway/shield_attacker_prereg_2026-09-15.md` (binding; read it first).

## Global Constraints

- Always `./venv/bin/python`; tests `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest <files> -q`; ruff format + check clean before each commit.
- Pre-registered, never tunable: `BETA = 1e-3` per pursuer STALLED step (the shield's `stalled` flag; not interventions); `D_POS = 4.0` (= `HUG_RANGE`); `HOLD = 10`; `TOTAL_LONG = 10_000_000`, `EVERY_LONG = 1_000_000`, `EVAL_N = 200`, `GATE_N = 500`, `PROBE_SEED0 = 1_120_000`, `GATE_SEED0 = 1_130_000`, `SEEDS = (22, 23, 24)`, `PASS_MIN = 2`, `P_GATE = 0.05`, `GAMMA = 0.99`; defenders D1 = STR14-SHIELD (`ckpts_STR14/STR14_10000000.zip`, hold 10, no features), D2 = RC19-SHIELD (`ckpts_RC19/RC19_10000000.zip`, hold 10, `shield_obs=True`), both `CBFShield(cfg)` δ = 0, `EnvConfig()` defaults; arms ATK/CTL × D1/D2 × 3 seeds + EZ-D1/EZ-D2 (seed 22, zero reward).
- The attacker uses FULL STATE (`full_state_observation`); the defender uses its own partial observation and the shield reads the true pose (as Step 6). The defender's `ShieldMetrics` is recorded every env step with the pursuer's contact flag; H-SAFE = 0 pursuer contact steps in every arm's training, probes and gate.
- V1-P: `StructuredPursuerPolicy` inside `SingleAgentWrapper(role="evader")` must reproduce `SearchMDP(shield=CBFShield(cfg)).play_episode(model)` against the same scripted evader episode-for-episode (outcome, steps, pursuer violations, stall count) — the plan's Task 1 test on 200 seeds and the runner's `--preflight`.
- Deadlock test = paired exact two-sided binomial on the per-episode deadlock indicator (b = a-only, c = b-only), p < 0.05 with b > c; H2(D) ≥ 2/3 seeds ATK vs CTL seed-paired; H2-implicit(D) CTL vs SPLITHUG ≥ 2/3; harness: EZ-D vs SPLITHUG must not pass; R3-b reported.
- Never run comparators, arms, summary or GIF against the real `results/derisk/`; tests and smokes redirect `OUT` and read only existing checkpoints.
- Commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; docstrings say why with decision numbers (D112, D113).

---

### Task 1: `full_state_observation` and `StructuredPursuerPolicy`

**Files:**
- Modify: `pe2/obs.py` (append `full_state_observation`)
- Modify: `pe2/structured.py` (append `StructuredPursuerPolicy`)
- Test: `tests/pe2/test_obs.py` (append), `tests/pe2/test_structured.py` (append)

**Interfaces:**
- Consumes: `pe2.obs.build_observation`, `_body_frame`, `obs_dim`; `pe2.structured.InterceptLaw`, `HOLD`; `pe2.metrics.ShieldMetrics`, `DEADLOCK_STEPS`; `SingleAgentWrapper._scale_action` arithmetic (`a = np.clip(np.asarray(action, dtype=float), -1, 1)`; `v = (a[0]+1)/2*v_max`; `w = a[1]*omega_max`).
- Produces:
  - `pe2.obs.full_state_observation(own_pose, opponent_pose, obstacles, cfg) -> np.ndarray` (11-dim float32; opponent features always filled; `out[6] = 1.0`).
  - `pe2.structured.StructuredPursuerPolicy(model, cfg, shield=None, hold=HOLD, shield_obs=False)` with `uses_truth = False`, `.reset()`, `.act(obs, env) -> tuple[float, float]` (executed command), `.last_sinfo`, `.metrics` (`ShieldMetrics | None`, fresh per `reset`), `.record(contact: bool) -> None`, `.decisions` (count of PPO decisions this episode).

- [ ] **Step 1: Write the failing tests**

Append to `tests/pe2/test_obs.py`:

```python
class TestFullStateObservation:
    def test_equals_build_observation_when_visible(self):
        from pe2.obs import build_observation, full_state_observation

        cfg = EnvConfig()
        own = np.array([0.0, 0.0, 0.0]); opp = np.array([2.0, 0.5, 1.0])
        obstacles = [{"x": 3.0, "y": 3.0, "radius": 0.5}, {"x": -2.0, "y": 1.0, "radius": 0.4}]
        a = build_observation(own, opp, obstacles, cfg)
        b = full_state_observation(own, opp, obstacles, cfg)
        assert a[6] == 1.0
        np.testing.assert_array_equal(a, b)

    def test_fills_opponent_features_when_invisible(self):
        from pe2.obs import build_observation, full_state_observation

        cfg = EnvConfig()
        own = np.array([0.0, 0.0, 0.0]); opp = np.array([-3.0, 0.0, 0.0])  # directly behind
        obstacles = []
        a = build_observation(own, opp, obstacles, cfg)
        b = full_state_observation(own, opp, obstacles, cfg)
        assert a[6] == 0.0 and a[3] == 0.0
        assert b[6] == 1.0 and b[3] == pytest.approx(-3.0 / cfg.arena_size)
        assert b.dtype == np.float32 and b.shape == a.shape
```

(`EnvConfig`, `np`, `pytest` are already imported in that file; check.)

Append to `tests/pe2/test_structured.py`:

```python
class TestStructuredPursuerPolicy:
    @staticmethod
    def _model():
        from stable_baselines3 import PPO

        return PPO.load("results/derisk/ckpts_STR14/STR14_10000000.zip", device="cpu")

    def test_v1p_reproduces_searchmdp_shielded_against_a_scripted_evader(self):
        # Spec section 2, V1-P, on 20 seeds here (the runner's preflight does 200).
        from pe2.shield import CBFShield
        from pe2.structured import SearchMDP, StructuredPursuerPolicy

        model = self._model()
        for i in range(20):
            seed = 900_1000 + i
            m = SearchMDP(CFG, GreedyEvader, hold=HOLD, shield=CBFShield(CFG))
            ref = m.play_episode(model.predict, seed)
            w = SingleAgentWrapper(CFG, "evader", lambda: StructuredPursuerPolicy(model, CFG, CBFShield(CFG), HOLD, False))
            obs, _ = w.reset(seed=seed)
            pol = w._opponent
            ev = GreedyEvader()
            ev.reset()
            while True:
                v, om = ev.act(w._last_joint_obs["evader"], w.env)
                obs, _, term, trunc, info = w.step(np.array([2 * v / CFG.evader_v_max - 1, om / CFG.omega_max], dtype=np.float32))
                pol.record(bool(info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]))
                if term or trunc:
                    break
            got = (info["outcome"], w.env.current_step, pol.metrics.n_violations, pol.metrics.n_stalled)
            exp = (ref["outcome"], ref["steps"], ref["violations"], ref["shield"]["n_stalled"])
            assert got == exp, f"seed {seed}: {got} != {exp}"

    def test_unshielded_policy_reproduces_searchmdp_too(self):
        from pe2.structured import SearchMDP, StructuredPursuerPolicy

        model = self._model()
        seed = 900_1100
        ref = SearchMDP(CFG, GreedyEvader, hold=HOLD).play_episode(model.predict, seed)
        w = SingleAgentWrapper(CFG, "evader", lambda: StructuredPursuerPolicy(model, CFG, None, HOLD, False))
        obs, _ = w.reset(seed=seed)
        ev = GreedyEvader(); ev.reset()
        while True:
            v, om = ev.act(w._last_joint_obs["evader"], w.env)
            obs, _, term, trunc, info = w.step(np.array([2 * v / CFG.evader_v_max - 1, om / CFG.omega_max], dtype=np.float32))
            if term or trunc:
                break
        assert (info["outcome"], w.env.current_step) == (ref["outcome"], ref["steps"])
        assert w._opponent.metrics is None and w._opponent.last_sinfo is None

    def test_shield_obs_policy_feeds_14_dim_features(self):
        from pe2.shield import CBFShield
        from pe2.structured import StructuredPursuerPolicy

        class Spy:
            def __init__(self):
                self.shapes = []

            def predict(self, obs, deterministic=True):
                self.shapes.append(obs.shape)
                return np.array([1.0, 0.0], dtype=np.float32), None

        spy = Spy()
        w = SingleAgentWrapper(CFG, "evader", lambda: StructuredPursuerPolicy(spy, CFG, CBFShield(CFG), HOLD, True))
        w.reset(seed=900_1200)
        for _ in range(60):
            _, _, term, trunc, info = w.step(np.array([1.0, 0.0], dtype=np.float32))
            w._opponent.record(bool(info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]))
            if term or trunc:
                break
        assert spy.shapes and all(s == (14,) for s in spy.shapes)
        assert w._opponent.decisions == len(spy.shapes)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_obs.py -k FullState tests/pe2/test_structured.py -k StructuredPursuerPolicy -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement**

Append to `pe2/obs.py`:

```python
def full_state_observation(
    own_pose: np.ndarray,
    opponent_pose: np.ndarray,
    obstacles: list[dict],
    cfg: EnvConfig,
) -> np.ndarray:
    """FULL-STATE observation for one agent: `build_observation` with the
    opponent features always filled and the flag set to 1. S6 (Step 9)
    grants the attacking evader full state — the worst-case adversary is
    the informative one for a safety guarantee — so this is what its
    learner sees. Equal to `build_observation` whenever the opponent is
    visible."""
    out = build_observation(own_pose, opponent_pose, obstacles, cfg)
    dx_b, dy_b = _body_frame(own_pose, (opponent_pose[0], opponent_pose[1]))
    out[3] = dx_b / cfg.arena_size
    out[4] = dy_b / cfg.arena_size
    out[5] = wrap_angle(opponent_pose[2] - own_pose[2]) / np.pi
    out[6] = 1.0
    return out
```

Append to `pe2/structured.py`:

```python
class StructuredPursuerPolicy:
    """The frozen structured pursuer as an OPPONENT object (S6, Step 9): the
    intercept law when it applies, else the PPO decision held `hold` env
    steps, ending early when the law applies again; the shield filters the
    executed command with the true pose — exactly `SearchMDP`'s semantics,
    so a shielded defender can sit inside `SingleAgentWrapper(role="evader")`
    while an attacker learns against it. Validity check V1-P: episode-for-
    episode equal to `SearchMDP.play_episode` against the same scripted
    evader (the executed command goes through the identical float32 Box
    round-trip that `SingleAgentWrapper._scale_action` applies).

    The environment must call `record(contact)` after every env step with
    the pursuer's contact flag, so `metrics` matches what `SearchMDP` would
    have logged (D97/D89 semantics)."""

    uses_truth = False  # acts on its own observation; the shield reads truth as in Step 6

    def __init__(self, model, cfg: EnvConfig, shield=None, hold: int = HOLD, shield_obs: bool = False):
        self.model = model
        self.cfg = cfg
        self.shield = shield
        self.hold = int(hold)
        self.shield_obs = bool(shield_obs)
        self.law = InterceptLaw(cfg)
        self.metrics = None
        self.last_sinfo = None
        self.decisions = 0
        self._held = None
        self._hold_left = 0
        self.reset()

    def reset(self) -> None:
        self.law.reset()
        self.metrics = ShieldMetrics() if self.shield is not None else None
        self.last_sinfo = None
        self.decisions = 0
        self._held = None
        self._hold_left = 0

    def _features(self) -> np.ndarray:
        s = self.last_sinfo
        if s is None or self.metrics is None:
            return np.array([0.0, 2.0, 0.0], dtype=np.float32)
        run = self.metrics._current_stall_run / DEADLOCK_STEPS
        return np.array(
            [float(bool(s["intervention"])), float(np.clip(s["min_clearance"], -0.5, 2.0)), float(np.clip(run, 0.0, 2.0))],
            dtype=np.float32,
        )

    def act(self, obs: np.ndarray, env) -> tuple[float, float]:
        a = self.law.act(obs)
        if a is None:
            if self._hold_left <= 0:
                lobs = np.concatenate([np.asarray(obs, dtype=np.float32), self._features()]) if self.shield_obs else obs
                self._held, _ = self.model.predict(lobs, deterministic=True)
                self._hold_left = self.hold
                self.decisions += 1
            a = self._held
            self._hold_left -= 1
        else:
            self._hold_left = 0  # the law acted: the next decision is fresh (SearchMDP's hold ends on visibility)
        cfg = self.cfg
        a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
        if self.shield is not None:
            v_nom = (float(a[0]) + 1.0) / 2.0 * cfg.v_max
            w_nom = float(a[1]) * cfg.omega_max
            (v, w), sinfo = self.shield.filter((v_nom, w_nom), env.pursuer_pose, env.obstacles, v_max=cfg.v_max)
            self.last_sinfo = sinfo
            a = np.array([2.0 * v / cfg.v_max - 1.0, w / cfg.omega_max], dtype=np.float32)
        else:
            self.last_sinfo = None
        # the identical Box round-trip SingleAgentWrapper._scale_action performs, so the executed
        # command is bit-identical to SearchMDP's route
        a64 = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
        return (a64[0] + 1.0) / 2.0 * cfg.v_max, a64[1] * cfg.omega_max

    def record(self, contact: bool) -> None:
        if self.last_sinfo is not None and self.metrics is not None:
            self.metrics.record(self.last_sinfo, violation=bool(contact))
```

(`DEADLOCK_STEPS` is already imported in `pe2/structured.py` from Task S5; confirm.)

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_obs.py tests/pe2/test_structured.py -q`
Expected: all PASS. If V1-P fails on some seed, compare step-by-step: the most likely cause is a difference in WHEN a new decision is taken (hold expiry vs the law having acted) or a float path difference in the executed command; the arithmetic above is what makes it exact — do not add tolerances.

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/obs.py pe2/structured.py tests/pe2/test_obs.py tests/pe2/test_structured.py && ./venv/bin/ruff check pe2/obs.py pe2/structured.py tests/pe2/test_obs.py tests/pe2/test_structured.py
git add pe2/obs.py pe2/structured.py tests/pe2/test_obs.py tests/pe2/test_structured.py
git commit -m "feat(pe2): full_state_observation; StructuredPursuerPolicy — the frozen shielded defender as an opponent (S6)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `pe2/evade.py` — `EvadeMDP`, `SplitHugEvader`, evaluators, probe; `paired_indicator_test`

**Files:**
- Create: `pe2/evade.py`
- Modify: `pe2/derisk.py` (append `paired_indicator_test`)
- Test: `tests/pe2/test_evade.py`, `tests/pe2/test_derisk.py` (append)

**Interfaces:**
- Consumes: Task 1; `pe2.scripted.ObstacleHuggingEvader`, `K_OMEGA`, `_clip_omega`, `wrap_angle`; `pe2.exploitability.SingleAgentWrapper`; `pe2.metrics.summarize_shield`; `pe2.eval.binomial_pvalue`, `win_count`.
- Produces:
  - `D_POS = 4.0`
  - `EvadeMDP(cfg, defender_factory, seed0=0, hold=10, beta=0.0, zero_reward=False, d_pos=D_POS)` — gym env; `.env_steps`, `.pursuer_violations_total`, `.span_stalled`, `.defender` (the live `StructuredPursuerPolicy`), `play_episode(predict, seed) -> dict` with keys `seed, role="evader", outcome, steps, escaped (bool), pursuer_violations, pursuer_deadlock (bool), pursuer_shield (summary dict)`.
  - `SplitHugEvader(cfg)` with `predict(obs, deterministic=True)` — plain flee computed from the full-state body-frame opponent features.
  - `evaluate_evader(model, cfg, defender_factory, n_episodes, seed0, hold=10, d_pos=D_POS) -> list[dict]`.
  - `play_scripted_evader(cfg, evader_factory, defender_factory, n_episodes, seed0) -> list[dict]` (same log schema; the scripted evader plays every step, e.g. `ObstacleHuggingEvader`, `GreedyEvader`).
  - `EvaderEvalProbe(cfg, defender_factory, every, total, eval_n, seed0, ckpt_dir, report_path, label, hold, d_pos)` — rows `steps, mark, ckpt, escape_rate, deadlock_rate, k, n, pursuer_violations`; stops `learn` at `total` env steps.
  - `pe2.derisk.paired_indicator_test(ind_a: list[bool], ind_b: list[bool], p_gate=0.05) -> dict` with `pass, p, b, c, rate_a, rate_b`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_evade.py
"""pe2.evade — the structured full-state attacker (S6, Step 9)."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.scripted import GreedyEvader, GreedyPursuer, ObstacleHuggingEvader
from pe2.shield import CBFShield
from pe2.structured import HOLD, StructuredPursuerPolicy

CFG = EnvConfig()
LOG_KEYS = {"seed", "role", "outcome", "steps", "escaped", "pursuer_violations", "pursuer_deadlock", "pursuer_shield"}


def _str14():
    from stable_baselines3 import PPO

    return PPO.load("results/derisk/ckpts_STR14/STR14_10000000.zip", device="cpu")


def _d1_factory(model):
    return lambda: StructuredPursuerPolicy(model, CFG, CBFShield(CFG), HOLD, False)


class TestEvadeMDP:
    def test_spaces_and_full_state_obs(self):
        from pe2.evade import EvadeMDP

        m = EvadeMDP(CFG, _d1_factory(_str14()))
        assert m.observation_space.shape == (11,)
        obs, _ = m.reset(seed=900_1300)
        assert obs.shape == (11,) and obs[6] == 1.0  # full state: the pursuer is always 'visible'

    def test_law_acts_within_d_pos_and_learner_beyond(self):
        from pe2.evade import D_POS, EvadeMDP

        assert D_POS == 4.0
        m = EvadeMDP(CFG, _d1_factory(_str14()), hold=10)
        m.trace = []
        log = m.play_episode(lambda o, deterministic=True: (np.array([1.0, 0.0], dtype=np.float32), None), 900_1301)
        assert set(b for b, _ in m.trace) <= {"law", "search"}
        for branch, dist in m.trace:
            if branch == "law":
                assert dist <= D_POS + 1e-9
            else:
                assert dist > D_POS
        assert log["steps"] == len(m.trace)

    def test_stall_bonus_counts_pursuer_stalls_in_the_span(self):
        from pe2.evade import EvadeMDP

        m = EvadeMDP(CFG, _d1_factory(_str14()), hold=10, beta=1e-3)
        obs, _ = m.reset(seed=900_1302)
        total_stalled = 0; total_bonus = 0.0; done = False
        while not done:
            obs, r, term, trunc, info = m.step(np.array([1.0, 0.0], dtype=np.float32))
            done = term or trunc
            total_stalled += info["pursuer_stalled_steps"]
            total_bonus += r - info["raw_reward"]
        assert total_bonus == pytest.approx(1e-3 * total_stalled)

    def test_zero_reward(self):
        from pe2.evade import EvadeMDP

        m = EvadeMDP(CFG, _d1_factory(_str14()), hold=10, beta=1e-3, zero_reward=True)
        m.reset(seed=900_1303)
        _, r, _, _, _ = m.step(np.array([1.0, 0.0], dtype=np.float32))
        assert r == 0.0

    def test_play_episode_log_and_pursuer_metrics(self):
        from pe2.evade import EvadeMDP, SplitHugEvader

        m = EvadeMDP(CFG, _d1_factory(_str14()), hold=10)
        log = m.play_episode(SplitHugEvader(CFG).predict, 900_1304)
        assert set(log) >= LOG_KEYS and log["role"] == "evader"
        assert log["escaped"] == (log["outcome"] == "timeout")
        assert log["pursuer_shield"]["n_steps"] == log["steps"]
        assert log["pursuer_violations"] == 0  # D97: shielded defender never contacts
        assert isinstance(log["pursuer_deadlock"], bool)

    def test_terminal_reward_is_the_evaders(self):
        from pe2.evade import EvadeMDP

        m = EvadeMDP(CFG, _d1_factory(_str14()), hold=10)
        obs, _ = m.reset(seed=900_1305)
        total = 0.0; done = False
        while not done:
            obs, r, term, trunc, info = m.step(np.array([1.0, 0.0], dtype=np.float32))
            done = term or trunc; total += info["raw_reward"]
        assert total == (1.0 if info["outcome"] == "timeout" else -1.0)


class TestSplitHugEvader:
    def test_flees_directly_away_in_body_frame(self):
        from pe2.evade import SplitHugEvader
        from pe2.obs import full_state_observation

        own = np.array([0.0, 0.0, 0.0]); opp = np.array([2.0, 0.0, 0.0])  # pursuer dead ahead
        obs = full_state_observation(own, opp, [], CFG)
        a, _ = SplitHugEvader(CFG).predict(obs)
        assert a[0] == pytest.approx(1.0)  # full speed
        assert abs(a[1]) == pytest.approx(1.0)  # turn hard: flee direction is behind


class TestEvaluators:
    def test_evaluate_evader_and_scripted_share_the_schema(self):
        from pe2.evade import SplitHugEvader, evaluate_evader, play_scripted_evader

        model = _str14()
        a = evaluate_evader(SplitHugEvader(CFG), CFG, _d1_factory(model), 2, 900_1310)
        b = play_scripted_evader(CFG, ObstacleHuggingEvader, _d1_factory(model), 2, 900_1310)
        c = play_scripted_evader(CFG, GreedyEvader, _d1_factory(model), 2, 900_1310)
        for logs in (a, b, c):
            assert len(logs) == 2 and all(set(lg) >= LOG_KEYS for lg in logs)
            assert [lg["seed"] for lg in logs] == [900_1310, 900_1311]
            assert all(lg["pursuer_violations"] == 0 for lg in logs)

    def test_probe_grid_and_stop(self, tmp_path):
        from pe2.evade import EvaderEvalProbe

        class _Model:
            def save(self, path):
                import pathlib
                pathlib.Path(path).write_bytes(b"x")

            def predict(self, obs, deterministic=True):
                return np.array([1.0, 0.0], dtype=np.float32), None

        class _VecEnv:
            env_steps = 0

            def get_attr(self, name):
                return [getattr(self, name)]

        p = EvaderEvalProbe(CFG, _d1_factory(_str14()), every=100, total=200, eval_n=2, seed0=900_1320, ckpt_dir=tmp_path, report_path=tmp_path / "r.json", label="T", hold=10, d_pos=4.0)
        cb, venv = p.callback, _VecEnv()
        cb.model = _Model(); cb.__dict__["_venv"] = venv
        import types
        type(cb).training_env = property(lambda self: self.__dict__["_venv"])
        venv.env_steps = 150
        assert cb._on_step() is True and [r["mark"] for r in p.rows] == [100]
        venv.env_steps = 200
        assert cb._on_step() is False and [r["mark"] for r in p.rows] == [100, 200]
        assert {"steps", "mark", "ckpt", "escape_rate", "deadlock_rate", "k", "n", "pursuer_violations"} <= set(p.rows[0])
```

Append to `tests/pe2/test_derisk.py`:

```python
class TestPairedIndicatorTest:
    def test_direction_and_significance(self):
        from pe2.derisk import paired_indicator_test

        a = [True] * 60 + [False] * 40
        b = [True] * 40 + [False] * 60  # a-only on 20, b-only on 0
        r = paired_indicator_test(a, b)
        assert r["b"] == 20 and r["c"] == 0 and r["p"] < 0.05 and r["pass"] is True
        assert r["rate_a"] == 0.6 and r["rate_b"] == 0.4
        assert paired_indicator_test(b, a)["pass"] is False  # reversed direction never passes
        assert paired_indicator_test(a, a)["p"] == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_evade.py tests/pe2/test_derisk.py -k "evade or PairedIndicator" -q`
Expected: FAIL — `No module named 'pe2.evade'`, `ImportError`.

- [ ] **Step 3: Implement**

`pe2/evade.py`:

```python
"""pe2.evade — the structured full-state attacker (S6, Step 9 / R2).

D112 measured what a hard shield does to the learned pursuer; D113 found the
remedy. R2 asks whether a LEARNING opponent can weaponise the shield — drive
the shielded pursuer into stalls and deadlocks — and by how much. The evader
side uses the same trick that produced the pursuer: a fixed law where the
geometry matters (ObstacleHuggingEvader's hug within D_POS of the pursuer,
full state) and a learned POSITIONING policy beyond it, one decision held
`hold` env steps, ending early when the pursuer comes within D_POS. The
attacker is granted full state (the worst-case adversary is the informative
one for a safety guarantee). Reward per span = the evader's terminal sum
+ beta * (pursuer stalled steps in the span) — stalls only, not
interventions, because interventions are cheap and common (0.69 of steps for
STR14-SHIELD) while stalls are the exploit.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import ClassVar

import gymnasium as gym
import numpy as np

from pe2.config import EnvConfig
from pe2.eval import win_count
from pe2.exploitability import SingleAgentWrapper
from pe2.metrics import summarize_shield
from pe2.obs import full_state_observation, observation_space
from pe2.scripted import K_OMEGA, ObstacleHuggingEvader, _clip_omega

D_POS = 4.0  # m: the hug law engages within this pursuer distance (= HUG_RANGE)


def _normalize_evader(v: float, w: float, cfg: EnvConfig) -> np.ndarray:
    return np.array([2.0 * v / cfg.evader_v_max - 1.0, w / cfg.omega_max], dtype=np.float32)


class SplitHugEvader:
    """predict() shim: the structure's own scripted positioning rule — plain
    flee directly away from the pursuer, computed from the full-state
    body-frame opponent features (obs[3], obs[4])."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg

    def predict(self, obs, deterministic: bool = True):
        err = float(np.arctan2(-obs[4], -obs[3]))  # body-frame heading of 'away'
        return _normalize_evader(self.cfg.evader_v_max, _clip_omega(K_OMEGA * err, self.cfg), self.cfg), None


class EvadeMDP(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, cfg: EnvConfig, defender_factory: Callable[[], object], seed0: int = 0, hold: int = 10,
                 beta: float = 0.0, zero_reward: bool = False, d_pos: float = D_POS):
        super().__init__()
        self.cfg = cfg
        self.hold = int(hold)
        self.beta = float(beta)
        self.zero_reward = bool(zero_reward)
        self.d_pos = float(d_pos)
        self.inner = SingleAgentWrapper(cfg, "evader", defender_factory, seed0=seed0)
        self.law = ObstacleHuggingEvader()
        self.observation_space = observation_space(cfg)
        self.action_space = self.inner.action_space
        self.env_steps = 0
        self.pursuer_violations_total = 0
        self.span_stalled = 0
        self.trace: list | None = None  # (branch, distance) per env step when a list is assigned
        self._info: dict = {}

    @property
    def defender(self):
        return self.inner._opponent

    def _distance(self) -> float:
        env = self.inner.env
        return float(np.hypot(env.pursuer_pose[0] - env.evader_pose[0], env.pursuer_pose[1] - env.evader_pose[1]))

    def _law_applies(self) -> bool:
        return self._distance() <= self.d_pos

    def _learner_obs(self) -> np.ndarray:
        env = self.inner.env
        return full_state_observation(env.evader_pose, env.pursuer_pose, env.obstacles, self.cfg)

    def _begin(self, seed: int | None) -> None:
        self.inner.reset(seed=seed)
        self.law.reset()
        self.span_stalled = 0
        self._info = {}

    def _step_inner(self, a_norm: np.ndarray, branch: str):
        if self.trace is not None:
            self.trace.append((branch, self._distance()))
        obs, r, term, trunc, info = self.inner.step(a_norm)
        self.env_steps += 1
        contact = bool(info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"])
        self.pursuer_violations_total += int(contact)
        d = self.defender
        d.record(contact)
        if d.last_sinfo is not None and d.last_sinfo["stalled"]:
            self.span_stalled += 1
        self._info = info
        return float(r), term, trunc

    def _advance(self):
        reward, term, trunc = 0.0, False, False
        while not (term or trunc) and self._law_applies():
            v, w = self.law.act(None, self.inner.env)  # full-state law reads env truth
            r, term, trunc = self._step_inner(_normalize_evader(v, w, self.cfg), "law")
            reward += r
        return reward, term, trunc

    def _shaped(self, reward: float) -> float:
        return 0.0 if self.zero_reward else reward + self.beta * self.span_stalled

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        while True:
            self._begin(seed)
            _r, term, trunc = self._advance()
            if not (term or trunc):
                return self._learner_obs(), {"env_steps": self.env_steps}
            seed = None

    def step(self, action):
        self.span_stalled = 0
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        reward, term, trunc = 0.0, False, False
        for _ in range(self.hold):
            r, term, trunc = self._step_inner(a, "search")
            reward += r
            if term or trunc or self._law_applies():
                break
        if not (term or trunc):
            r2, term, trunc = self._advance()
            reward += r2
        info = dict(self._info, env_steps=self.env_steps, pursuer_stalled_steps=self.span_stalled, raw_reward=reward)
        return self._learner_obs(), self._shaped(reward), term, trunc, info

    def play_episode(self, predict, seed: int) -> dict:
        self._begin(seed)
        _r, term, trunc = self._advance()
        obs = self._learner_obs()
        while not (term or trunc):
            a, _ = predict(obs, deterministic=True)
            obs, _r, term, trunc, _i = self.step(a)
        return self._log(seed)

    def _log(self, seed: int) -> dict:
        d = self.defender
        summary = d.metrics.summary() if d.metrics is not None else None
        return {
            "seed": seed, "role": "evader", "outcome": self._info["outcome"], "steps": self.inner.env.current_step,
            "escaped": self._info["outcome"] == "timeout",
            "pursuer_violations": int(d.metrics.n_violations) if d.metrics is not None else None,
            "pursuer_deadlock": bool(summary["deadlock"]) if summary else False,
            "pursuer_shield": summary,
        }


def evaluate_evader(model, cfg: EnvConfig, defender_factory, n_episodes: int, seed0: int, hold: int = 10, d_pos: float = D_POS) -> list[dict]:
    mdp = EvadeMDP(cfg, defender_factory, hold=hold, d_pos=d_pos)
    return [mdp.play_episode(model.predict, seed0 + i) for i in range(n_episodes)]


def play_scripted_evader(cfg: EnvConfig, evader_factory, defender_factory, n_episodes: int, seed0: int) -> list[dict]:
    """A scripted evader (acting on env truth or its own observation as it
    declares) against the frozen defender, every step; same log schema."""
    logs = []
    for i in range(n_episodes):
        w = SingleAgentWrapper(cfg, "evader", defender_factory)
        w.reset(seed=seed0 + i)
        ev = evader_factory()
        ev.reset()
        while True:
            v, om = ev.act(w._last_joint_obs["evader"], w.env)
            _, _, term, trunc, info = w.step(_normalize_evader(v, om, cfg))
            w._opponent.record(bool(info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]))
            if term or trunc:
                break
        d = w._opponent
        summary = d.metrics.summary() if d.metrics is not None else None
        logs.append({
            "seed": seed0 + i, "role": "evader", "outcome": info["outcome"], "steps": w.env.current_step,
            "escaped": info["outcome"] == "timeout",
            "pursuer_violations": int(d.metrics.n_violations) if d.metrics is not None else None,
            "pursuer_deadlock": bool(summary["deadlock"]) if summary else False, "pursuer_shield": summary,
        })
    return logs


def pursuer_summary(logs: list[dict]) -> dict:
    """Pooled defender numbers over evader logs (the summarize_shield block
    plus the escape rate)."""
    shield_logs = [{"shielded_role": "pursuer", "violations": lg["pursuer_violations"], "shield": lg["pursuer_shield"]} for lg in logs]
    s = summarize_shield(shield_logs)
    esc = sum(lg["escaped"] for lg in logs)
    return s | {"escape_rate": esc / len(logs), "n_episodes": len(logs)}


class EvaderEvalProbe:
    """Env-step-grid probe for EvadeMDP training (mirrors StructuredEvalProbe)."""

    def __init__(self, cfg, defender_factory, every, total, eval_n, seed0, ckpt_dir, report_path, label, hold=10, d_pos=D_POS):
        from stable_baselines3.common.callbacks import BaseCallback

        if total % every != 0:
            raise ValueError(f"total ({total}) must be a multiple of every ({every})")
        rows: list[dict] = []

        def evaluate_at(model, env_steps: int, mark: int) -> dict:
            path = ckpt_dir / f"{label}_{mark}.zip"
            model.save(path)
            logs = evaluate_evader(model, cfg, defender_factory, eval_n, seed0, hold=hold, d_pos=d_pos)
            k = sum(lg["escaped"] for lg in logs)
            dl = sum(lg["pursuer_deadlock"] for lg in logs)
            row = {"steps": env_steps, "mark": mark, "ckpt": str(path), "escape_rate": k / eval_n, "deadlock_rate": dl / eval_n,
                   "k": k, "n": eval_n, "pursuer_violations": int(sum(lg["pursuer_violations"] or 0 for lg in logs))}
            rows.append(row)
            with open(report_path, "w") as f:
                json.dump({"checkpoints": rows}, f, indent=2)
            print(f"  @{env_steps:>9} env steps (mark {mark}): escape {k}/{eval_n} = {k / eval_n:.3f}, pursuer deadlock eps {dl / eval_n:.3f}, pursuer violations {row['pursuer_violations']}", flush=True)
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
```

Note for the probe test: `BaseCallback.training_env` is a read-only property; the test above patches the class property — if that proves fragile, give `_Model` a `get_env()` returning the fake VecEnv and set only `cb.model` (the S2 pattern), and adjust the test accordingly; say so in the report.

Append to `pe2/derisk.py`:

```python
def paired_indicator_test(ind_a: list[bool], ind_b: list[bool], p_gate: float = 0.05) -> dict:
    """S6 deadlock test: paired exact two-sided binomial on the discordant
    pairs of a per-episode indicator (b = a-only, c = b-only). PASS iff
    p < p_gate and b > c — 'a' produces the event more often than 'b' on
    the same episodes."""
    from pe2.eval import binomial_pvalue

    if len(ind_a) != len(ind_b):
        raise ValueError("paired indicators must have equal length")
    b = sum(1 for x, y in zip(ind_a, ind_b, strict=True) if x and not y)
    c = sum(1 for x, y in zip(ind_a, ind_b, strict=True) if y and not x)
    p = 1.0 if b + c == 0 else binomial_pvalue(b, b + c, p0=0.5, alternative="two-sided")
    n = len(ind_a)
    return {"pass": bool(p < p_gate and b > c), "p": p, "b": b, "c": c,
            "rate_a": sum(map(bool, ind_a)) / n, "rate_b": sum(map(bool, ind_b)) / n, "n": n}
```

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_evade.py tests/pe2/test_derisk.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/evade.py pe2/derisk.py tests/pe2/test_evade.py tests/pe2/test_derisk.py && ./venv/bin/ruff check pe2/evade.py pe2/derisk.py tests/pe2/test_evade.py tests/pe2/test_derisk.py
git add pe2/evade.py pe2/derisk.py tests/pe2/test_evade.py tests/pe2/test_derisk.py
git commit -m "feat(pe2): EvadeMDP — structured full-state attacker with stall bonus; SplitHugEvader; evaluators; paired_indicator_test (S6)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `scripts/derisk_s6_attacker.py`

**Files:**
- Create: `scripts/derisk_s6_attacker.py`
- Test: `tests/pe2/test_derisk_s6_round.py`

**Interfaces:**
- Consumes: Tasks 1–2; `pe2.derisk.paired_indicator_test`; `pe2.render` (`Renderer`, `Trails`, `FovOverlay`, `ShieldOverlay`, `save_gif`); `pe2.shield.CBFShield`; `pe2.scripted` evaders.
- Produces: CLI `--preflight | --measure-start | --arms <ARM...> | --summarize | --gif [--device cpu]`; module-level `ARMS`, `DEFENDERS`, `h2_verdicts`, `hsafe_verdict`, `direction`, `summarize`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_derisk_s6_round.py
"""scripts/derisk_s6_attacker.py — the attacker round's own decisions."""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s6_attacker.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s6_attacker", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _ind(rate, n=500, seed0=0):
    k = int(round(rate * n))
    return [i < k for i in range(n)]


def _arm(name, deadlocks, viol=0):
    fam, d = name.split("-")[0], name.split("-")[1][:2]
    seed = int(name.split("-")[1][2:]) if len(name.split("-")[1]) > 2 else 22
    return {"arm": name, "family": fam, "defender": d, "seed": seed, "train_pursuer_violations": viol, "probe_pursuer_violations": 0,
            "final": {"pursuer_violations": 0, "deadlock_ind": deadlocks, "escape_ind": _ind(0.5), "pursuer": {"deadlock_episode_rate": sum(deadlocks) / len(deadlocks), "stall_step_rate": 0.4, "intervention_rate": 0.6, "backup_rate": 0.0, "infeasibility_rate": 0.0}},
            "checkpoints": []}


def _starts():
    out = {}
    for d in ("D1", "D2"):
        out[d] = {"SPLITHUG": {"deadlock_ind": _ind(0.50), "escape_ind": _ind(0.4)},
                  "HUG7": {"deadlock_ind": _ind(0.61), "escape_ind": _ind(0.5)},
                  "GREEDY": {"deadlock_ind": _ind(0.54), "escape_ind": _ind(0.6)}}
    return out


def _full(atk=(0.70, 0.68, 0.66), ctl=(0.55, 0.56, 0.54), ez=0.50, d2_shift=-0.10):
    res = {}
    for d, shift in (("D1", 0.0), ("D2", d2_shift)):
        for s, ra, rc in zip(R.SEEDS, atk, ctl, strict=True):
            res[f"ATK-{d}{s}"] = _arm(f"ATK-{d}{s}", _ind(ra + shift))
            res[f"CTL-{d}{s}"] = _arm(f"CTL-{d}{s}", _ind(rc + shift))
        res[f"EZ-{d}22"] = _arm(f"EZ-{d}22", _ind(ez + shift))
    return res


class TestProtocol:
    def test_constants(self):
        assert (R.BETA, R.D_POS, R.HOLD) == (1e-3, 4.0, 10)
        assert (R.TOTAL_LONG, R.EVERY_LONG, R.EVAL_N, R.GATE_N) == (10_000_000, 1_000_000, 200, 500)
        assert (R.PROBE_SEED0, R.GATE_SEED0) == (1_120_000, 1_130_000)
        assert R.SEEDS == (22, 23, 24) and R.PASS_MIN == 2 and R.P_GATE == 0.05 and R.GAMMA == 0.99
        assert set(R.DEFENDERS) == {"D1", "D2"} and R.DEFENDERS["D2"]["shield_obs"] is True and R.DEFENDERS["D1"]["shield_obs"] is False

    def test_arm_table(self):
        assert len(R.ARMS) == 14
        assert R.ARMS["ATK-D122"] == {"family": "ATK", "defender": "D1", "seed": 22, "beta": 1e-3, "zero_reward": False}
        assert R.ARMS["CTL-D224"] == {"family": "CTL", "defender": "D2", "seed": 24, "beta": 0.0, "zero_reward": False}
        assert R.ARMS["EZ-D122"] == {"family": "EZ", "defender": "D1", "seed": 22, "beta": 0.0, "zero_reward": True}

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s6_start.json")
        monkeypatch.setattr(R, "PREFLIGHT_PATH", tmp_path / "s6_preflight.json")
        with pytest.raises(SystemExit, match="preflight"):
            R.require_preflight()
        (tmp_path / "s6_preflight.json").write_text('{"pass": false}')
        with pytest.raises(SystemExit, match="V1-P"):
            R.require_preflight()
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()
        (tmp_path / "s6_arm_ATK-D122.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("ATK-D122")


class TestVerdicts:
    def test_h2_pass_on_both_defenders(self):
        h = R.h2_verdicts(_full(), _starts())
        assert h["H2"] == {"D1": "PASS", "D2": "PASS"} and h["H2_overall"] == "PASS"
        assert h["H2_implicit"]["D1"] in ("PASS", "FAIL") and h["harness"] == {"D1": "VALID", "D2": "VALID"}

    def test_h2_fail_when_atk_equals_ctl(self):
        h = R.h2_verdicts(_full(atk=(0.55, 0.56, 0.54)), _starts())
        assert h["H2"]["D1"] == "FAIL"

    def test_harness_invalid_when_ez_beats_splithug(self):
        h = R.h2_verdicts(_full(ez=0.70), _starts())
        assert h["harness"]["D1"] == "INVALID" and h["H2"]["D1"].startswith("UNDETERMINED")

    def test_missing_seed_incomplete(self):
        res = _full()
        del res["CTL-D223"]
        assert R.h2_verdicts(res, _starts())["H2"]["D2"] == "INCOMPLETE"

    def test_r3b_is_reported(self):
        h = R.h2_verdicts(_full(d2_shift=-0.10), _starts())
        assert "R3b" in h and abs(h["R3b"]["mean_gain_D1"] - h["R3b"]["mean_gain_D2"]) < 0.02 + 1e-9  # same gain shape under a pure shift


class TestSafety:
    def test_hsafe(self):
        assert R.hsafe_verdict(_full())["verdict"] == "HOLDS"
        res = _full(); res["ATK-D122"]["train_pursuer_violations"] = 1
        assert R.hsafe_verdict(res)["verdict"] == "FAILS"
        res = _full(); del res["EZ-D222"]
        assert R.hsafe_verdict(res)["verdict"] == "INCOMPLETE"


class TestDirection:
    def test_rows(self):
        assert "weaponises" in R.direction({"hsafe": "HOLDS", "H2_overall": "PASS", "H2_implicit_any": True})
        assert "implicit" in R.direction({"hsafe": "HOLDS", "H2_overall": "FAIL", "H2_implicit_any": True})
        assert "no exploitation" in R.direction({"hsafe": "HOLDS", "H2_overall": "FAIL", "H2_implicit_any": False})
        assert "Stop" in R.direction({"hsafe": "FAILS", "H2_overall": "PASS", "H2_implicit_any": True})
        assert "incomplete" in R.direction({"hsafe": "INCOMPLETE", "H2_overall": "INCOMPLETE", "H2_implicit_any": False})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s6_round.py -q`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Write the runner**

```python
# scripts/derisk_s6_attacker.py
"""S6 — Step 9 shield-attacker (spec: docs/pathway/shield_attacker_prereg_2026-09-15.md).
Nothing here may change a threshold, a seed, a stream, a rule or a budget after a number is seen.

Arms: ATK-{D1,D2}{22,23,24} (stall bonus 1e-3), CTL-{D1,D2}{22,23,24} (no bonus), EZ-{D1,D2}22 (zero reward).
Defenders: D1 = STR14-SHIELD, D2 = RC19-SHIELD (frozen, delta = 0). Attacker = EvadeMDP (full state).

Usage:
  ./venv/bin/python scripts/derisk_s6_attacker.py --preflight      (V1-P on 200 seeds)
  ./venv/bin/python scripts/derisk_s6_attacker.py --measure-start  (scripted comparators per defender)
  ./venv/bin/python scripts/derisk_s6_attacker.py --arms ATK-D122  (one per process)
  ./venv/bin/python scripts/derisk_s6_attacker.py --summarize
  ./venv/bin/python scripts/derisk_s6_attacker.py --gif
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import paired_indicator_test
from pe2.evade import D_POS
from pe2.scripted import GreedyEvader, ObstacleHuggingEvader
from pe2.structured import HOLD

OUT = pathlib.Path("results/derisk")
START_PATH = OUT / "s6_start.json"
PREFLIGHT_PATH = OUT / "s6_preflight.json"

# --- pre-registered (spec sections 3-5). Do not tune. ---
BETA = 1e-3
TOTAL_LONG = 10_000_000
EVERY_LONG = 1_000_000
EVAL_N = 200
GATE_N = 500
PROBE_SEED0 = 1_120_000
GATE_SEED0 = 1_130_000
SEEDS = (22, 23, 24)
PASS_MIN = 2
P_GATE = 0.05
GAMMA = 0.99
DEFENDERS = {
    "D1": {"ckpt": OUT / "ckpts_STR14" / "STR14_10000000.zip", "shield_obs": False, "hold": HOLD},
    "D2": {"ckpt": OUT / "ckpts_RC19" / "RC19_10000000.zip", "shield_obs": True, "hold": HOLD},
}
COMPARATORS = ("SPLITHUG", "HUG7", "GREEDY")
ARMS: dict = {}
for _d in DEFENDERS:
    for _s in SEEDS:
        ARMS[f"ATK-{_d}{_s}"] = {"family": "ATK", "defender": _d, "seed": _s, "beta": BETA, "zero_reward": False}
        ARMS[f"CTL-{_d}{_s}"] = {"family": "CTL", "defender": _d, "seed": _s, "beta": 0.0, "zero_reward": False}
    ARMS[f"EZ-{_d}22"] = {"family": "EZ", "defender": _d, "seed": 22, "beta": 0.0, "zero_reward": True}

_MODELS: dict = {}


def defender_factory(d: str, device: str):
    """One loaded model per process, a fresh policy object per episode."""
    from stable_baselines3 import PPO

    from pe2.shield import CBFShield
    from pe2.structured import StructuredPursuerPolicy

    cfg = EnvConfig()
    if d not in _MODELS:
        _MODELS[d] = PPO.load(DEFENDERS[d]["ckpt"], device=device)
    model, spec = _MODELS[d], DEFENDERS[d]
    return lambda: StructuredPursuerPolicy(model, cfg, CBFShield(cfg), spec["hold"], spec["shield_obs"])


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


# ----------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------
def require_preflight() -> None:
    if not PREFLIGHT_PATH.exists():
        raise SystemExit(f"preflight missing: {PREFLIGHT_PATH}. Run --preflight FIRST (spec V1-P).")
    if not json.loads(PREFLIGHT_PATH.read_text()).get("pass"):
        raise SystemExit("V1-P preflight FAILED; the round may not proceed.")


def require_not_measured() -> None:
    if START_PATH.exists():
        raise SystemExit(f"{START_PATH} already exists (stop rule, spec section 6).")
    if any(OUT.glob("s6_arm_*.json")) or any((OUT / f"ckpts_{a}").exists() for a in ARMS):
        raise SystemExit("S6 arms have already run; the stop rule forbids re-measuring.")


def require_arm_not_run(arm: str) -> None:
    if (OUT / f"s6_arm_{arm}.json").exists() or (OUT / f"ckpts_{arm}").exists():
        raise SystemExit(f"arm {arm} has already run; the stop rule (spec section 6) forbids re-running it.")


def load_start() -> dict:
    if not START_PATH.exists():
        raise SystemExit(f"comparators missing: {START_PATH}. Run --measure-start FIRST.")
    s = json.loads(START_PATH.read_text())
    for d in DEFENDERS:
        for c in COMPARATORS:
            if c not in s.get(d, {}) or "deadlock_ind" not in s[d][c]:
                raise SystemExit(f"{d}/{c} missing from {START_PATH}; re-run --measure-start.")
    return s


# ----------------------------------------------------------------------------
# preflight (V1-P) and comparators
# ----------------------------------------------------------------------------
def preflight(device: str) -> dict:
    import numpy as np
    from stable_baselines3 import PPO

    from pe2.exploitability import SingleAgentWrapper
    from pe2.shield import CBFShield
    from pe2.structured import SearchMDP, StructuredPursuerPolicy

    cfg = EnvConfig()
    model = PPO.load(DEFENDERS["D1"]["ckpt"], device=device)
    mism = []
    for i in range(200):
        seed = GATE_SEED0 + i
        ref = SearchMDP(cfg, GreedyEvader, hold=HOLD, shield=CBFShield(cfg)).play_episode(model.predict, seed)
        w = SingleAgentWrapper(cfg, "evader", lambda: StructuredPursuerPolicy(model, cfg, CBFShield(cfg), HOLD, False))
        w.reset(seed=seed)
        ev = GreedyEvader(); ev.reset()
        while True:
            v, om = ev.act(w._last_joint_obs["evader"], w.env)
            _, _, term, trunc, info = w.step(np.array([2 * v / cfg.evader_v_max - 1, om / cfg.omega_max], dtype=np.float32))
            w._opponent.record(bool(info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]))
            if term or trunc:
                break
        got = (info["outcome"], w.env.current_step, w._opponent.metrics.n_violations, w._opponent.metrics.n_stalled)
        exp = (ref["outcome"], ref["steps"], ref["violations"], ref["shield"]["n_stalled"])
        if got != exp:
            mism.append({"seed": seed, "got": got, "expected": exp})
    out = {"pass": not mism, "n": 200, "seed0": GATE_SEED0, "mismatches": len(mism), "examples": mism[:10]}
    _write_json(PREFLIGHT_PATH, out)
    print(f"V1-P {'PASS' if out['pass'] else 'FAIL'}: {len(mism)} mismatches in 200 episodes", flush=True)
    if mism:
        raise SystemExit("V1-P FAILED: the opponent object does not reproduce SearchMDP; halting.")
    return out


def _entry(logs: list[dict]) -> dict:
    from pe2.evade import pursuer_summary

    return {"deadlock_ind": [bool(lg["pursuer_deadlock"]) for lg in logs], "escape_ind": [bool(lg["escaped"]) for lg in logs],
            "pursuer": pursuer_summary(logs), "pursuer_violations": int(sum(lg["pursuer_violations"] or 0 for lg in logs)), "logs": logs}


def measure_start(device: str) -> dict:
    from pe2.evade import SplitHugEvader, evaluate_evader, play_scripted_evader

    require_preflight()
    require_not_measured()
    cfg = EnvConfig()
    out: dict = {"gate_seed0": GATE_SEED0, "gate_n": GATE_N, "provenance": provenance()}
    for d in DEFENDERS:
        f = defender_factory(d, device)
        out[d] = {
            "SPLITHUG": _entry(evaluate_evader(SplitHugEvader(cfg), cfg, f, GATE_N, GATE_SEED0, hold=HOLD, d_pos=D_POS)),
            "HUG7": _entry(play_scripted_evader(cfg, ObstacleHuggingEvader, f, GATE_N, GATE_SEED0)),
            "GREEDY": _entry(play_scripted_evader(cfg, GreedyEvader, f, GATE_N, GATE_SEED0)),
        }
        for c, e in out[d].items():
            p = e["pursuer"]
            print(f"{d} {c:8s}: pursuer deadlock eps {p['deadlock_episode_rate']:.3f} stall {p['stall_step_rate']:.3f} interv {p['intervention_rate']:.3f} "
                  f"violations {e['pursuer_violations']} | evader escape {p['escape_rate']:.3f}", flush=True)
    _write_json(START_PATH, out)
    return out


# ----------------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------------
def run_train_arm(arm: str, starts: dict, device: str, holder: dict | None = None) -> dict:
    from stable_baselines3 import PPO

    from pe2.evade import EvadeMDP, EvaderEvalProbe, evaluate_evader

    spec = ARMS[arm]
    cfg = EnvConfig()
    f = defender_factory(spec["defender"], device)
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = EvaderEvalProbe(cfg, f, every=EVERY_LONG, total=TOTAL_LONG, eval_n=EVAL_N, seed0=PROBE_SEED0, ckpt_dir=d,
                            report_path=OUT / f"s6_probe_{arm}.json", label=arm, hold=HOLD, d_pos=D_POS)
    if holder is not None:
        holder["rows"] = probe.rows
    env = EvadeMDP(cfg, f, seed0=spec["seed"] * 1000, hold=HOLD, beta=spec["beta"], zero_reward=spec["zero_reward"], d_pos=D_POS)
    if holder is not None:
        holder["env"] = env
    model = PPO("MlpPolicy", env, seed=spec["seed"], gamma=GAMMA, verbose=0, device=device)
    model.save(OUT / f"s6_init_{arm}.zip")
    model.learn(total_timesteps=TOTAL_LONG, callback=probe.callback)
    result = {"arm": arm, **spec, "settings": spec, "env_steps": int(env.env_steps), "decisions": int(model.num_timesteps),
              "train_pursuer_violations": int(env.pursuer_violations_total),
              "probe_pursuer_violations": int(sum(r.get("pursuer_violations", 0) for r in probe.rows)),
              "checkpoints": probe.rows, "provenance": provenance()}
    ckpt = d / f"{arm}_{TOTAL_LONG}.zip"
    if not any(r["mark"] == TOTAL_LONG for r in probe.rows) or not ckpt.exists():
        result["verdict"], result["reason"] = "UNDETERMINED", f"final probe at mark {TOTAL_LONG} never fired (env steps {env.env_steps})"
        return result
    logs = evaluate_evader(PPO.load(ckpt, device=device), cfg, f, GATE_N, GATE_SEED0, hold=HOLD, d_pos=D_POS)
    e = _entry(logs)
    e.pop("logs")
    result["final"] = e | {"ckpt": str(ckpt)}
    p = e["pursuer"]
    result["verdict"] = f"final deadlock eps {p['deadlock_episode_rate']:.3f}"
    result["reason"] = (f"pursuer deadlock eps {p['deadlock_episode_rate']:.3f}, stall {p['stall_step_rate']:.3f}, interv {p['intervention_rate']:.3f}, "
                        f"escape {p['escape_rate']:.3f}; pursuer violations train {result['train_pursuer_violations']} probe {result['probe_pursuer_violations']} gate {e['pursuer_violations']}")
    return result


def run_arm(arm: str, device: str) -> dict:
    require_arm_not_run(arm)
    starts = load_start()
    holder: dict = {}
    try:
        result = run_train_arm(arm, starts, device, holder)
    except BaseException as e:
        _write_json(OUT / f"s6_arm_{arm}.json", {"arm": arm, **ARMS[arm], "verdict": "UNDETERMINED", "reason": f"crashed: {type(e).__name__}: {e}",
                    "checkpoints": holder.get("rows", []), "train_pursuer_violations": int(getattr(holder.get("env"), "pursuer_violations_total", 0)),
                    "probe_pursuer_violations": int(sum(r.get("pursuer_violations", 0) for r in holder.get("rows", []))), "provenance": provenance()})
        raise
    _write_json(OUT / f"s6_arm_{arm}.json", result)
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# verdicts (spec section 5)
# ----------------------------------------------------------------------------
def _ind(r: dict | None):
    try:
        return r["final"]["deadlock_ind"]
    except (KeyError, TypeError):
        return None


def h2_verdicts(results: dict, starts: dict) -> dict:
    h2, h2i, harness, tests, r3 = {}, {}, {}, {}, {}
    for d in DEFENDERS:
        split = starts[d]["SPLITHUG"]["deadlock_ind"]
        hug7 = starts[d]["HUG7"]["deadlock_ind"]
        ez = _ind(results.get(f"EZ-{d}22"))
        if ez is None:
            harness[d] = "INCOMPLETE"
        else:
            harness[d] = "INVALID" if paired_indicator_test(ez, split, P_GATE)["pass"] else "VALID"
        per, per_i, per_h7, gains = {}, {}, {}, []
        for s in SEEDS:
            a, c = _ind(results.get(f"ATK-{d}{s}")), _ind(results.get(f"CTL-{d}{s}"))
            if a is None or c is None:
                per[s] = per_i[s] = per_h7[s] = None
                continue
            per[s] = paired_indicator_test(a, c, P_GATE)
            per_i[s] = paired_indicator_test(c, split, P_GATE)
            per_h7[s] = paired_indicator_test(a, hug7, P_GATE)
            gains.append(per[s]["rate_a"] - per[s]["rate_b"])
        tests[d] = {"atk_vs_ctl": per, "ctl_vs_splithug": per_i, "atk_vs_hug7": per_h7}
        if any(v is None for v in per.values()) or harness[d] == "INCOMPLETE":
            h2[d] = h2i[d] = "INCOMPLETE"
        elif harness[d] == "INVALID":
            h2[d] = h2i[d] = "UNDETERMINED (harness)"
        else:
            h2[d] = "PASS" if sum(v["pass"] for v in per.values()) >= PASS_MIN else "FAIL"
            h2i[d] = "PASS" if sum(v["pass"] for v in per_i.values()) >= PASS_MIN else "FAIL"
        r3[d] = gains
    vals = list(h2.values())
    overall = "INCOMPLETE" if "INCOMPLETE" in vals else ("UNDETERMINED (harness)" if all(v.startswith("UNDETERMINED") for v in vals)
              else ("PASS" if "PASS" in vals else "FAIL"))
    r3b = {"gains_D1": r3["D1"], "gains_D2": r3["D2"],
           "mean_gain_D1": (sum(r3["D1"]) / len(r3["D1"])) if r3["D1"] else None,
           "mean_gain_D2": (sum(r3["D2"]) / len(r3["D2"])) if r3["D2"] else None}
    r3b["remedy_reduces_gain"] = (r3b["mean_gain_D1"] is not None and r3b["mean_gain_D2"] is not None and r3b["mean_gain_D2"] < r3b["mean_gain_D1"])
    return {"H2": h2, "H2_overall": overall, "H2_implicit": h2i, "H2_implicit_any": any(v == "PASS" for v in h2i.values()),
            "harness": harness, "tests": tests, "R3b": r3b}


def hsafe_verdict(results: dict) -> dict:
    steps, offenders = 0, []
    missing = [a for a in ARMS if a not in results or "final" not in results[a]]
    for a, r in results.items():
        v = int(r.get("train_pursuer_violations", 0)) + int(r.get("probe_pursuer_violations", 0)) + int((r.get("final") or {}).get("pursuer_violations", 0))
        if "probe_pursuer_violations" not in r:
            v += int(sum(x.get("pursuer_violations", 0) for x in r.get("checkpoints", [])))
        steps += v
        if v:
            offenders.append(a)
    return {"verdict": "FAILS" if steps else ("INCOMPLETE" if missing else "HOLDS"), "pursuer_violation_steps": steps, "offenders": offenders, "missing": missing}


def direction(h: dict) -> str:
    if h["hsafe"] == "FAILS":
        return "H-SAFE FAILS: the shielded defender recorded a contact step. Stop; census; no further round without a user ruling."
    if h["hsafe"] == "INCOMPLETE" or h["H2_overall"] == "INCOMPLETE":
        return "incomplete: all fourteen arms must run"
    if h["H2_overall"].startswith("UNDETERMINED"):
        return "harness invalid on every defender: the zero-reward evader beats the scripted split-hug rule; diagnose."
    if h["H2_overall"] == "PASS":
        return ("H2 PASSES: a learned opponent weaponises a hard shield — R2 measured; magnitude vs the scripted probe and the remedy's "
                "effect (R3-b) go to the paper. Figure 1 candidate.")
    if h["H2_implicit_any"]:
        return "H2 FAILS but H2-implicit PASSES: the bonus adds nothing, yet learning to escape already exploits the shield — R2 holds in its implicit form."
    return "Both fail, harness valid: learned positioning at this budget does not exceed the scripted probe — no exploitation beyond the hugging geometry found."


def load_results() -> dict:
    return {p.stem.removeprefix("s6_arm_"): json.loads(p.read_text()) for p in sorted(OUT.glob("s6_arm_*.json"))}


def summarize(results: dict, starts: dict) -> dict:
    h = h2_verdicts(results, starts)
    hs = hsafe_verdict(results)
    d = direction({"hsafe": hs["verdict"], "H2_overall": h["H2_overall"], "H2_implicit_any": h["H2_implicit_any"]})
    table = {a: {"defender": r.get("defender"), "family": r.get("family"), "seed": r.get("seed"),
                 "deadlock_rate": (r.get("final") or {}).get("pursuer", {}).get("deadlock_episode_rate"),
                 "stall_rate": (r.get("final") or {}).get("pursuer", {}).get("stall_step_rate"),
                 "escape_rate": (r.get("final") or {}).get("pursuer", {}).get("escape_rate")} for a, r in results.items()}
    comps = {d_: {c: {"deadlock_rate": starts[d_][c]["pursuer"]["deadlock_episode_rate"], "escape_rate": starts[d_][c]["pursuer"]["escape_rate"]}
                  for c in COMPARATORS} for d_ in DEFENDERS}
    return {"spec": "docs/pathway/shield_attacker_prereg_2026-09-15.md", "hsafe": hs, "hypotheses": h, "arms": table, "comparators": comps,
            "complete": all(a in results and "final" in results[a] for a in ARMS), "direction": d}


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, d in zip(axes, DEFENDERS, strict=True):
        names, ys, cols = [], [], []
        for c in COMPARATORS:
            names.append(c); ys.append(summary["comparators"][d][c]["deadlock_rate"]); cols.append("tab:gray")
        for fam, col in (("CTL", "tab:blue"), ("ATK", "tab:red"), ("EZ", "tab:green")):
            for a, r in summary["arms"].items():
                if r["defender"] == d and r["family"] == fam and r["deadlock_rate"] is not None:
                    names.append(a); ys.append(r["deadlock_rate"]); cols.append(col)
        ax.bar(range(len(names)), ys, color=cols)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, fontsize=7)
        ax.set_ylabel("pursuer deadlock-episode rate (gate)"); ax.set_title(f"{d}: H2 {summary['hypotheses']['H2'][d]}")
    fig.suptitle("S6 — shield attacker" + ("" if summary["complete"] else " — INCOMPLETE"))
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def write_gif(path, results: dict, device: str, every: int = 4) -> None:
    """Visual gate (spec section 5): the best ATK-D1 seed vs CTL on the same seed,
    on an episode where the attacker produced a pursuer deadlock; pursuer FOV +
    ShieldOverlay on the pursuer."""
    import numpy as np
    from pe2.render import FovOverlay, Renderer, ShieldOverlay, Trails, save_gif  # noqa: I001
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO

    from pe2.evade import EvadeMDP

    cfg = EnvConfig()
    atk = max((a for a in results if a.startswith("ATK-D1") and "final" in results[a]), key=lambda a: results[a]["final"]["pursuer"]["deadlock_episode_rate"])
    ctl = "CTL-D1" + atk[len("ATK-D1"):]
    models = {n: PPO.load(results[n]["final"]["ckpt"], device=device) for n in (atk, ctl)}
    f = defender_factory("D1", device)

    class Rec(EvadeMDP):
        def __init__(self, *a, label, **k):
            super().__init__(*a, **k)
            self.label = label; self.frames = []; self.r = Renderer(cfg, dpi=80, figsize=(5, 5)); self.t = Trails()
            self.live = {"nominal": None, "executed": None, "info": None}; self.ov = None

        def _step_inner(self, a_norm, branch):
            env = self.inner.env
            if self.ov is None:
                self.ov = [FovOverlay("pursuer"), ShieldOverlay("pursuer", self.defender.shield, self.live)]
            self.t.append("pursuer", *env.pursuer_pose[:2]); self.t.append("evader", *env.evader_pose[:2])
            out = super()._step_inner(a_norm, branch)
            s = self.defender.last_sinfo
            self.live.update(info=s, nominal=(s["v_nom"], None) if s else None, executed=(s["v_exec"], None) if s else None)
            if env.current_step % every == 0:
                self.frames.append(self.r.draw(env, trails=self.t, overlays=self.ov, title_prefix=f"{self.label} [{branch}]"))
            return out

        def finish(self, outcome, step):
            for _ in range(10):
                self.frames.append(self.r.draw(self.inner.env, trails=self.t, overlays=self.ov, title_prefix=f"{self.label}  {outcome.upper()} @ {step}"))
            plt.close(self.r.fig)

    scan = {}
    for s in range(GATE_SEED0, GATE_SEED0 + 60):
        la = EvadeMDP(cfg, f, hold=HOLD, d_pos=D_POS).play_episode(models[atk].predict, s)
        lc = EvadeMDP(cfg, f, hold=HOLD, d_pos=D_POS).play_episode(models[ctl].predict, s)
        scan[s] = (la, lc)
    seed = next((s for s, (la, lc) in scan.items() if la["pursuer_deadlock"] and not lc["pursuer_deadlock"]), None)
    if seed is None:
        raise SystemExit("visual gate: no seed in the first 60 where the attacker deadlocks the pursuer and the control does not")
    frames = []
    pair = []
    for name in (ctl, atk):
        m = Rec(cfg, f, hold=HOLD, d_pos=D_POS, label=name)
        log = m.play_episode(models[name].predict, seed); m.finish(log["outcome"], log["steps"]); pair.append(m.frames)
    n = max(map(len, pair)); pair = [p + [p[-1]] * (n - len(p)) for p in pair]
    frames = [np.concatenate([a, b], axis=1) for a, b in zip(*pair, strict=True)]
    save_gif(frames, str(path), fps=15)
    print(f"wrote {path} ({len(frames)} frames; seed {seed}; {ctl} left, {atk} right)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--gif", action="store_true")
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
        summary = summarize(results, starts)
        _write_json(OUT / "s6_round.json", {"summary": summary, "arms": results})
        write_figure(results, starts, summary, OUT / "s6_round.png")
        print(json.dumps({k: summary[k] for k in ("hsafe", "hypotheses", "comparators")}, indent=2, default=str))
        print(json.dumps(summary["arms"], indent=2))
        print(f"\n{summary['direction']}")
    if args.gif:
        write_gif(OUT / "step_09_attack.gif", load_results(), args.device)


if __name__ == "__main__":
    main()
```

Notes: `ShieldOverlay`'s `live["nominal"]`/`["executed"]` expect `(v, ω)` pairs; the defender's `last_sinfo` carries `v_nom` and `v_exec` but not ω — pass `(v_nom, w_nom)` and `(v_exec, w_exec)` if the info dict has them (check `CBFShield._info` keys in `pe2/shield.py` ~437–515; it has `v_nom`, `v_exec`; if no ω keys exist, store the executed `(v, w)` on the `StructuredPursuerPolicy` as `.last_exec` and the nominal as `.last_nom` in Task 1's `act` — add those two attributes there if missing and use them here). Test file for the runner does not exercise `write_gif`.

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s6_round.py -q`
Expected: all PASS.

- [ ] **Step 5: Smoke in scratch, then the V1-P preflight**

Scratch smoke under `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/ff03fc68-2116-4512-a227-312f3099d143/scratchpad/s6smoke/`: monkeypatch `OUT`, `START_PATH`, `PREFLIGHT_PATH` to scratch and `TOTAL_LONG=4000, EVERY_LONG=2000, EVAL_N=4, GATE_N=6`; run `preflight` with the loop shortened to 5 seeds (monkeypatch a `PREFLIGHT_N` constant — add `PREFLIGHT_N = 200` to the runner and use it in the loop), `measure_start`, `run_arm("ATK-D122")`, `run_arm("CTL-D222")`, `run_arm("EZ-D122")`, `summarize`, `write_figure`. Expected: no exception; every arm's pursuer violations 0; D2's defender consumes 14-dim features (the spy pattern is not needed — just that it runs); PNG written. Then run the REAL preflight (this is the only real-path command in the plan): `OMP_NUM_THREADS=2 ./venv/bin/python scripts/derisk_s6_attacker.py --preflight` → `results/derisk/s6_preflight.json` with `pass: true` (≈ 6 min). Paste both outputs into your report.

- [ ] **Step 6: Lint, full suite, commit**

```bash
./venv/bin/ruff format scripts/derisk_s6_attacker.py tests/pe2/test_derisk_s6_round.py && ./venv/bin/ruff check scripts/derisk_s6_attacker.py tests/pe2/test_derisk_s6_round.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q -x
git add scripts/derisk_s6_attacker.py tests/pe2/test_derisk_s6_round.py
git commit -m "feat(derisk): S6 runner — V1-P preflight, comparators, ATK/CTL/EZ arms vs two shielded defenders, H2/H2-implicit/R3-b, figure, GIF" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** §2 components → Task 1 (`StructuredPursuerPolicy`, `full_state_observation`), Task 2 (`EvadeMDP`, `SplitHugEvader`, `evaluate_evader`, `play_scripted_evader`), V1-P → Task 1 test + Task 3 `--preflight`; §3 defenders, learner, streams, β, comparators → Task 3; §4 arms → `ARMS`; §5 deadlock test → `paired_indicator_test`; H2 ≥ 2/3 seed-paired, H2-implicit, vs HUG7, R3-b, harness, escape trade-off, INCOMPLETE, H-SAFE, GIF → Task 3; §6 direction → `direction`; stop rule → guards; §8 artefacts → paths.
- **Placeholders:** none; the ω-key question for the overlay is resolved inline (add `.last_nom`/`.last_exec` in Task 1 if the info dict lacks ω).
- **Type consistency:** `evaluate_evader(model, cfg, defender_factory, n, seed0, hold, d_pos)` and `play_scripted_evader(cfg, evader_factory, defender_factory, n, seed0)` used identically; arm records carry `final.deadlock_ind`, `final.escape_ind`, `final.pursuer` (pursuer_summary block incl. `escape_rate`), `train_pursuer_violations`, `probe_pursuer_violations`; the tests' `_arm` fixture mirrors that shape; `paired_indicator_test` returns `pass, p, b, c, rate_a, rate_b, n`.
