# Structured Search (S1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `SearchMDP` structured pursuer (fixed intercept law, learned search sub-policy), its evaluator and env-step probe, and the pre-registered S1 runner with four arms (SA / SB / SZ / CP), then pass the V1 pre-flight so the round can run.

**Architecture:** `pe2/structured.py` holds the intercept law copied verbatim from `SearchingPursuer`, a `gym.Env` (`SearchMDP`) that consults the learner only when the law has nothing to act on and holds each decision for `HOLD` env steps, `predict()` shims for the orbit and random searches, an evaluator in `evaluate_sb3_policy`'s log schema, and an SB3 callback that probes on an env-step grid. `pe2/derisk.py` gains the structured verdict rule. `scripts/derisk_s1_structured.py` is the runner (pre-flight → bars → arms → summary), modelled on `scripts/derisk_r1_requal.py`.

**Tech Stack:** Python 3.12, `./venv/bin/python`, gymnasium, stable-baselines3 2.x (PPO), numpy, pytest (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), ruff.

**Spec:** `docs/pathway/structured_search_prereg_2026-09-12.md` (read it first; it is the binding authority).

## Global Constraints

- Always `./venv/bin/python`; tests as `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/<file> -q`.
- Lint: `./venv/bin/ruff format <files> && ./venv/bin/ruff check <files>` must be clean before each commit.
- Pre-registered constants (spec §2/§3, do not tune): `HOLD = 10`, `R_ACQ = 0.2`, structured PPO `gamma = 0.99` with every other PPO hyper-parameter at the SB3 default, `TOTAL = 1_000_000` env steps, `EVERY = 100_000`, `EVAL_N = 200`, `FINAL_EVAL_N = 500`, `EVAL_SEED0 = 1_010_000`, `DEMO_SEED0 = 1_020_000`, `DEMO_EPISODES = 300`, `BC_EPOCHS = 50`, `SEED = 7`, control `gae_lambda = 1.0`, V2 threshold `0.35`.
- The intercept law must be `SearchingPursuer.act` minus its fallback, **verbatim** (same constants `K_OMEGA`, `LAST_SEEN_TOL`, same branch order, same `last_seen` semantics).
- Never run any arm. The plan ends at the V1 pre-flight; bars and arms are launched by the controller after the branch is green.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Match existing style: lazy SB3/torch imports inside functions in library code; docstrings explain *why* with decision numbers.

---

### Task 1: `pe2/structured.py` — intercept law, `SearchMDP`, search shims, evaluators

**Files:**
- Create: `pe2/structured.py`
- Test: `tests/pe2/test_structured.py`

**Interfaces:**
- Consumes: `pe2.exploitability.SingleAgentWrapper(cfg, "pursuer", opponent_factory, seed0)`, whose `.step` returns `(obs, reward, terminated, truncated, info)` with `info["outcome"]`, `info["distance"]`, and whose `.env.current_step` is the episode step; `pe2.scripted` constants `K_OMEGA`, `LAST_SEEN_TOL`, `PATROL_RADIUS`, `PATROL_LEAD`, helpers `_clip_omega`, `wrap_angle`; `pe2.obs.own_pose_from_obs`, `opponent_world_from_obs`. Observation index 6 is the LOS/visible flag; indices 3, 4 are the body-frame opponent dx, dy.
- Produces (later tasks rely on these exact names):
  - `HOLD: int = 10`, `R_ACQ: float = 0.2`
  - `class InterceptLaw(cfg)` with `.reset()`, `.act(obs) -> np.ndarray | None`, attribute `.last_seen`
  - `class OrbitSearch(cfg)` and `class RandomSearch(seed=0)`, each with `.predict(obs, deterministic=True) -> (np.ndarray, None)`
  - `class SearchMDP(cfg, opponent_factory, seed0=0, hold=HOLD, r_acq=R_ACQ, zero_reward=False)` — `gym.Env`; attributes `.env_steps` (int, cumulative), `.inner` (the `SingleAgentWrapper`), `.law`; methods `reset`, `step`, `play_episode(predict, seed) -> dict`
  - `evaluate_structured(model, cfg, opponent_factory, n_episodes, seed0, hold=HOLD) -> list[dict]`
  - `evaluate_scripted(cfg, teacher_factory, opponent_factory, n_episodes, seed0) -> list[dict]`
  - Log schema (both evaluators): `{"seed", "role", "outcome", "steps", "final_distance", "los_breaks"}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_structured.py
"""pe2.structured — the S1 structured pursuer (spec:
docs/pathway/structured_search_prereg_2026-09-12.md)."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.exploitability import SingleAgentWrapper
from pe2.scripted import GreedyEvader, SearchingPursuer
from pe2.structured import (
    HOLD,
    R_ACQ,
    InterceptLaw,
    OrbitSearch,
    RandomSearch,
    SearchMDP,
    evaluate_scripted,
    evaluate_structured,
)

CFG = EnvConfig()
LOG_KEYS = {"seed", "role", "outcome", "steps", "final_distance", "los_breaks"}


def _teacher_action(obs, wrapper):
    """SearchingPursuer's (v, w) on this obs, normalized to the Box."""
    t = SearchingPursuer()
    t.reset()
    v, w = t.act(obs, wrapper.env)
    return np.array([2.0 * v / CFG.v_max - 1.0, w / CFG.omega_max], dtype=np.float32), t


def _reset_with_evader(wrapper, pursuer_pose, evader_pose, seed=1):
    obs, _ = wrapper.reset(seed=seed, init_poses=(pursuer_pose, evader_pose))
    return obs


class TestInterceptLaw:
    def test_constants_are_the_spec_values(self):
        assert HOLD == 10
        assert R_ACQ == 0.2

    def test_visible_branch_matches_the_teacher(self):
        w = SingleAgentWrapper(CFG, "pursuer", GreedyEvader)
        # evader 2 m dead ahead: visible
        obs = _reset_with_evader(w, (5.0, 5.0, 0.0), (7.0, 5.0, 0.0))
        assert obs[6] > 0.5
        law = InterceptLaw(CFG)
        expected, teacher = _teacher_action(obs, w)
        got = law.act(obs)
        assert got is not None
        np.testing.assert_allclose(got, expected, atol=1e-6)
        np.testing.assert_allclose(law.last_seen, teacher.last_seen)

    def test_nothing_to_act_on_returns_none(self):
        w = SingleAgentWrapper(CFG, "pursuer", GreedyEvader)
        # evader 3 m directly behind: invisible, no last_seen
        obs = _reset_with_evader(w, (5.0, 5.0, 0.0), (2.0, 5.0, 0.0))
        assert obs[6] < 0.5
        law = InterceptLaw(CFG)
        assert law.act(obs) is None
        assert law.last_seen is None

    def test_last_seen_branch_matches_the_teacher_then_clears(self):
        w = SingleAgentWrapper(CFG, "pursuer", GreedyEvader)
        obs_vis = _reset_with_evader(w, (5.0, 5.0, 0.0), (7.0, 5.0, 0.0))
        law = InterceptLaw(CFG)
        law.act(obs_vis)  # sets last_seen
        teacher = SearchingPursuer()
        teacher.reset()
        teacher.act(obs_vis, w.env)
        # now an invisible observation from a pose far from last_seen
        obs_inv = _reset_with_evader(w, (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), seed=2)
        assert obs_inv[6] < 0.5
        v, om = teacher.act(obs_inv, w.env)
        expected = np.array([2.0 * v / CFG.v_max - 1.0, om / CFG.omega_max], dtype=np.float32)
        got = law.act(obs_inv)
        assert got is not None
        np.testing.assert_allclose(got, expected, atol=1e-6)
        # standing on the last-seen point: the law clears it and has nothing to act on
        law.last_seen = np.array([1.0, 1.0])
        assert law.act(obs_inv) is None
        assert law.last_seen is None


class TestSearchShims:
    def test_orbit_search_is_the_teacher_fallback(self):
        w = SingleAgentWrapper(CFG, "pursuer", GreedyEvader)
        obs = _reset_with_evader(w, (5.0, 5.0, 0.0), (2.0, 5.0, 0.0))
        expected, _ = _teacher_action(obs, w)  # teacher falls through to the orbit
        got, _ = OrbitSearch(CFG).predict(obs, deterministic=True)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_random_search_is_seeded_and_in_the_box(self):
        a, _ = RandomSearch(seed=0).predict(np.zeros(11))
        b, _ = RandomSearch(seed=0).predict(np.zeros(11))
        np.testing.assert_array_equal(a, b)
        assert a.shape == (2,) and np.all(np.abs(a) <= 1.0)


class TestSearchMDP:
    def test_spaces_match_the_inner_wrapper(self):
        m = SearchMDP(CFG, GreedyEvader)
        assert m.observation_space == m.inner.observation_space
        assert m.action_space == m.inner.action_space

    def test_v1_orbit_at_hold_1_reproduces_the_teacher_episode_for_episode(self):
        # Spec §2 validity check V1 on a small seed set.
        seeds0, n = 900_500, 12
        teacher_logs = evaluate_scripted(CFG, SearchingPursuer, GreedyEvader, n, seeds0)
        mdp_logs = evaluate_structured(OrbitSearch(CFG), CFG, GreedyEvader, n, seeds0, hold=1)
        assert [(l["outcome"], l["steps"]) for l in teacher_logs] == [
            (l["outcome"], l["steps"]) for l in mdp_logs
        ]
        assert [l["los_breaks"] for l in teacher_logs] == [l["los_breaks"] for l in mdp_logs]

    def test_a_decision_is_held_for_hold_env_steps_while_invisible(self):
        m = SearchMDP(CFG, GreedyEvader, hold=10)
        # evader behind; learner drives straight ahead so it stays invisible
        m.inner.reset(seed=3, init_poses=((5.0, 5.0, 0.0), (2.0, 5.0, np.pi)))
        m.law.reset()
        m._visible = False
        before = m.env_steps
        obs, r, term, trunc, info = m.step(np.array([1.0, 0.0], dtype=np.float32))
        assert m.env_steps - before == 10
        assert info["env_steps"] == m.env_steps
        assert info["onsets"] == 0
        assert r == 0.0 and not term and not trunc

    def test_hold_ends_early_on_sight_and_the_law_then_plays(self):
        m = SearchMDP(CFG, GreedyEvader, hold=10)
        # evader 2.5 m behind; spinning in place brings it into the 90 deg cone
        # well before 10 steps, then the chase law runs until sight is lost or
        # the episode ends — so far more than 10 env steps elapse.
        m.inner.reset(seed=4, init_poses=((5.0, 5.0, 0.0), (2.5, 5.0, 0.0)))
        m.law.reset()
        m._visible = False
        obs, r, term, trunc, info = m.step(np.array([-1.0, 1.0], dtype=np.float32))
        assert info["onsets"] >= 1
        assert m.env_steps > 10
        assert r == pytest.approx(info["raw_reward"] + R_ACQ * info["onsets"])

    def test_zero_reward_arm_returns_zero_but_reports_the_raw_reward(self):
        m = SearchMDP(CFG, GreedyEvader, hold=10, zero_reward=True)
        m.inner.reset(seed=4, init_poses=((5.0, 5.0, 0.0), (2.5, 5.0, 0.0)))
        m.law.reset()
        m._visible = False
        _, r, _, _, info = m.step(np.array([-1.0, 1.0], dtype=np.float32))
        assert r == 0.0
        assert info["onsets"] >= 1

    def test_terminal_only_arm_has_no_bonus(self):
        m = SearchMDP(CFG, GreedyEvader, hold=10, r_acq=0.0)
        m.inner.reset(seed=4, init_poses=((5.0, 5.0, 0.0), (2.5, 5.0, 0.0)))
        m.law.reset()
        m._visible = False
        _, r, _, _, info = m.step(np.array([-1.0, 1.0], dtype=np.float32))
        assert r == pytest.approx(info["raw_reward"])

    def test_gym_reset_and_step_run_a_whole_episode(self):
        m = SearchMDP(CFG, GreedyEvader, seed0=7000)
        obs, info = m.reset()
        assert obs.shape == (11,)
        assert obs[6] < 0.5  # a decision point is always one the law cannot act on
        done = False
        decisions = 0
        while not done:
            obs, r, term, trunc, info = m.step(m.action_space.sample())
            done = term or trunc
            decisions += 1
        assert decisions <= 1200
        assert info["outcome"] in ("capture", "timeout")

    def test_evaluators_share_the_log_schema(self):
        s = evaluate_scripted(CFG, SearchingPursuer, GreedyEvader, 2, 900_600)
        e = evaluate_structured(RandomSearch(0), CFG, GreedyEvader, 2, 900_600)
        assert len(s) == len(e) == 2
        assert set(s[0]) == set(e[0]) == LOG_KEYS
        assert s[0]["seed"] == e[0]["seed"] == 900_600
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_structured.py -q`
Expected: FAIL at import — `No module named 'pe2.structured'`.

- [ ] **Step 3: Write `pe2/structured.py`**

```python
"""pe2.structured — the S1 structured pursuer.

Spec: docs/pathway/structured_search_prereg_2026-09-12.md.

Every learner round through R1 collapsed because one-step TD at gamma=0.999
cannot form a calibrated value over a 1200-step chain from a +/-1 terminal
(D82; R1's critic sat at +3.4 -> +21.5 on a problem whose values lie in
[-0.3, 1], with no action dependence). This module changes WHAT has to be
credited. The teacher's chase and last-seen branches (`SearchingPursuer`,
D106) are a five-line feedback law that provably captures; they stay fixed
inside the environment. RL learns only the SEARCH sub-policy: what to do
when that law has nothing to act on. One decision is held for HOLD env
steps (ending early when the evader comes into view), then the law plays
until it again has nothing to act on. From the learner's side the game is a
short semi-MDP (~20 decisions per ~200-step search stretch), and the capture
ability cannot be destroyed because it is not in the learned parameters.
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
from pe2.obs import opponent_world_from_obs, own_pose_from_obs
from pe2.scripted import (
    K_OMEGA,
    LAST_SEEN_TOL,
    PATROL_LEAD,
    PATROL_RADIUS,
    _clip_omega,
    wrap_angle,
)

HOLD = 10  # env steps one search decision is held (spec section 2)
R_ACQ = 0.2  # acquisition bonus per invisible->visible onset (arm SA)


def _normalize(v: float, w: float, cfg: EnvConfig) -> np.ndarray:
    """(v, omega) -> the Box action that SingleAgentWrapper._scale_action
    maps back to exactly (v, omega). Same inverse as collect_demonstrations."""
    return np.array([2.0 * v / cfg.v_max - 1.0, w / cfg.omega_max], dtype=np.float32)


class InterceptLaw:
    """`SearchingPursuer.act` MINUS its fallback: chase while visible, drive
    to the last-seen point, else None (nothing to act on). Owns `last_seen`
    — it is the scripted law's state, not the learner's (spec section 7).
    The two branches are copied verbatim so that V1 (orbit at HOLD=1
    reproduces the teacher episode-for-episode) holds bit-exactly."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.last_seen: np.ndarray | None = None

    def reset(self) -> None:
        self.last_seen = None

    def act(self, obs: np.ndarray) -> np.ndarray | None:
        cfg = self.cfg
        if obs[6] > 0.5:  # opponent visible — unchanged from SearchingPursuer
            self.last_seen = opponent_world_from_obs(obs, cfg)
            bearing_b = np.arctan2(obs[4], obs[3])
            return _normalize(cfg.v_max, _clip_omega(K_OMEGA * bearing_b, cfg), cfg)
        if self.last_seen is not None:  # go to last-seen — unchanged
            own = own_pose_from_obs(obs, cfg)
            to = self.last_seen - own[:2]
            if float(np.hypot(*to)) > LAST_SEEN_TOL:
                err = wrap_angle(np.arctan2(to[1], to[0]) - own[2])
                return _normalize(cfg.v_max, _clip_omega(K_OMEGA * err, cfg), cfg)
            self.last_seen = None
        return None


class OrbitSearch:
    """predict() shim: the teacher's orbit fallback as a search policy. Inside
    SearchMDP at hold=1 this IS SearchingPursuer (validity check V1) and at
    the spec's hold it is BAR_ORBIT."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg

    def predict(self, obs, deterministic: bool = True):
        cfg = self.cfg
        own = own_pose_from_obs(obs, cfg)
        angle = float(np.arctan2(own[1], own[0]))
        target = PATROL_RADIUS * np.array(
            [np.cos(angle + PATROL_LEAD), np.sin(angle + PATROL_LEAD)]
        )
        to = target - own[:2]
        if float(np.hypot(*to)) < 1e-9:
            return _normalize(cfg.v_max, 0.0, cfg), None
        err = wrap_angle(np.arctan2(to[1], to[0]) - own[2])
        return _normalize(cfg.v_max, _clip_omega(K_OMEGA * err, cfg), cfg), None


class RandomSearch:
    """predict() shim: a uniform Box action — FLOOR_RAND, what the fixed law
    achieves with no search competence at all."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic: bool = True):
        return self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32), None


class SearchMDP(gym.Env):
    """The structured pursuer as a single-agent gym env (spec section 2).

    The learner is consulted only at states where `InterceptLaw` returns
    None. Its action is held for `hold` env steps, ending early if the
    evader becomes visible or the episode ends; the law then plays until it
    again has nothing to act on. The reward of one learner step is the SUM of
    env rewards over that whole span plus `r_acq` per acquisition onset
    (obs[6] crossing from <= 0.5 to > 0.5 between consecutive env steps) —
    no discounting inside a span (semi-MDP, gamma per decision).
    `zero_reward=True` is arm SZ: the same dynamics with every reward zeroed.

    `env_steps` counts underlying env steps so budgets and probes stay in
    env steps. `play_episode` is the evaluation path: it plays one seeded
    episode end-to-end, including episodes the law finishes before any
    decision, and returns a log in evaluate_sb3_policy's schema.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        cfg: EnvConfig,
        opponent_factory: Callable[[], object],
        seed0: int = 0,
        hold: int = HOLD,
        r_acq: float = R_ACQ,
        zero_reward: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.hold = int(hold)
        self.r_acq = float(r_acq)
        self.zero_reward = bool(zero_reward)
        self.inner = SingleAgentWrapper(cfg, "pursuer", opponent_factory, seed0=seed0)
        self.law = InterceptLaw(cfg)
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space
        self.env_steps = 0
        self._visible = False
        self._los_breaks = 0
        self._info: dict = {}

    # -- internals -------------------------------------------------------
    def _begin(self, seed: int | None) -> np.ndarray:
        obs, _ = self.inner.reset(seed=seed)
        self.law.reset()
        self._visible = bool(obs[6] > 0.5)
        self._los_breaks = 0
        self._info = {}
        return obs

    def _step_inner(self, action: np.ndarray):
        prev = self._visible
        obs, r, term, trunc, info = self.inner.step(action)
        self.env_steps += 1
        self._visible = bool(obs[6] > 0.5)
        self._los_breaks += int(prev and not self._visible)
        self._info = info
        onset = int((not prev) and self._visible)
        return obs, float(r), onset, term, trunc

    def _advance(self, obs: np.ndarray):
        """Play the law until it has nothing to act on or the episode ends.
        Returns (obs, reward_sum, onsets, terminated, truncated)."""
        reward, onsets, term, trunc = 0.0, 0, False, False
        while not (term or trunc):
            a = self.law.act(obs)
            if a is None:
                break
            obs, r, onset, term, trunc = self._step_inner(a)
            reward += r
            onsets += onset
        return obs, reward, onsets, term, trunc

    def _shaped(self, reward: float, onsets: int) -> float:
        return 0.0 if self.zero_reward else reward + self.r_acq * onsets

    # -- gym API (training) ---------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        while True:
            obs = self._begin(seed)
            obs, _r, _o, term, trunc = self._advance(obs)
            if not (term or trunc):
                return obs, {"env_steps": self.env_steps}
            # The law finished this episode before any decision (evader
            # visible at reset, then captured or timed out): the learner had
            # no influence, so start the next episode from the stream.
            seed = None

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        reward, onsets, term, trunc = 0.0, 0, False, False
        obs = None
        for _ in range(self.hold):
            obs, r, onset, term, trunc = self._step_inner(a)
            reward += r
            onsets += onset
            if term or trunc or self._visible:
                break
        if not (term or trunc):
            obs, r2, o2, term, trunc = self._advance(obs)
            reward += r2
            onsets += o2
        info = dict(
            self._info, env_steps=self.env_steps, onsets=onsets, raw_reward=reward
        )
        return obs, self._shaped(reward, onsets), term, trunc, info

    # -- evaluation ------------------------------------------------------
    def play_episode(self, predict, seed: int) -> dict:
        """One seeded episode: `predict(obs, deterministic=True) -> (action, _)`
        decides at decision points, the law plays the rest."""
        obs = self._begin(seed)
        obs, _r, _o, term, trunc = self._advance(obs)
        while not (term or trunc):
            a, _ = predict(obs, deterministic=True)
            obs, _r, term, trunc, _i = self.step(a)
        return {
            "seed": seed,
            "role": "pursuer",
            "outcome": self._info["outcome"],
            "steps": self.inner.env.current_step,
            "final_distance": float(self._info["distance"]),
            "los_breaks": self._los_breaks,
        }


def evaluate_structured(
    model,
    cfg: EnvConfig,
    opponent_factory: Callable[[], object],
    n_episodes: int,
    seed0: int,
    hold: int = HOLD,
) -> list[dict]:
    """Paired-seed evaluation of a search policy inside SearchMDP (seeds
    seed0 + i, deterministic predict, frozen opponent). Same log schema as
    evaluate_sb3_policy so win_count / Wilson / McNemar apply unchanged."""
    mdp = SearchMDP(cfg, opponent_factory, hold=hold)
    return [mdp.play_episode(model.predict, seed0 + i) for i in range(n_episodes)]


def evaluate_scripted(
    cfg: EnvConfig,
    teacher_factory: Callable[[], object],
    opponent_factory: Callable[[], object],
    n_episodes: int,
    seed0: int,
) -> list[dict]:
    """Paired-seed evaluation of a scripted pursuer (acts on its partial
    observation) in the plain SingleAgentWrapper — the reference side of V1."""
    wrapper = SingleAgentWrapper(cfg, "pursuer", opponent_factory)
    logs = []
    for i in range(n_episodes):
        obs, _ = wrapper.reset(seed=seed0 + i)
        teacher = teacher_factory()
        teacher.reset()
        visible = obs[6] > 0.5
        los_breaks = 0
        while True:
            v, w = teacher.act(wrapper._last_joint_obs["pursuer"], wrapper.env)
            obs, _, term, trunc, info = wrapper.step(_normalize(v, w, cfg))
            now = obs[6] > 0.5
            los_breaks += int(visible and not now)
            visible = now
            if term or trunc:
                break
        logs.append(
            {
                "seed": seed0 + i,
                "role": "pursuer",
                "outcome": info["outcome"],
                "steps": wrapper.env.current_step,
                "final_distance": float(info["distance"]),
                "los_breaks": los_breaks,
            }
        )
    return logs


class StructuredEvalProbe:
    """SB3 callback for SearchMDP training: checkpoint + paired eval at every
    `every` ENV steps (not learner decisions), rows flushed to `report_path`
    after each probe, and `learn` is STOPPED (the callback returns False)
    once `total` env steps have been reached — so the budget is in env steps
    exactly as for every prior round. A probe fires at the first decision
    boundary at or after each mark; the row records the true env-step count
    and the mark, and the checkpoint is named by the mark
    (`{label}_{mark}.zip`) so the final one is `{label}_{total}.zip`.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        opponent_factory: Callable[[], object],
        every: int,
        total: int,
        eval_n: int,
        seed0: int,
        ckpt_dir,
        report_path,
        label: str,
        hold: int = HOLD,
    ):
        from stable_baselines3.common.callbacks import BaseCallback  # lazy

        rows: list[dict] = []

        def evaluate_at(model, env_steps: int, mark: int) -> dict:
            path = ckpt_dir / f"{label}_{mark}.zip"
            model.save(path)
            logs = evaluate_structured(
                model, cfg, opponent_factory, eval_n, seed0, hold=hold
            )
            k, n = win_count(logs, "pursuer")
            row = {
                "steps": env_steps,
                "mark": mark,
                "ckpt": str(path),
                "win_rate": k / n,
                "k": k,
                "n": n,
            }
            rows.append(row)
            with open(report_path, "w") as f:
                json.dump({"checkpoints": rows}, f, indent=2)
            print(
                f"  @{env_steps:>8} env steps (mark {mark}): selection win rate "
                f"{k}/{n} = {k / n:.3f}",
                flush=True,
            )
            return row

        class _CB(BaseCallback):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.next_mark = every

            def _on_step(inner_self) -> bool:
                env_steps = int(inner_self.training_env.get_attr("env_steps")[0])
                while inner_self.next_mark <= total and env_steps >= inner_self.next_mark:
                    evaluate_at(inner_self.model, env_steps, inner_self.next_mark)
                    inner_self.next_mark += every
                return env_steps < total

        self.rows = rows
        self.callback = _CB()
        self.evaluate_at = evaluate_at
```

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_structured.py -q`
Expected: all PASS. If `test_hold_ends_early_on_sight_and_the_law_then_plays` fails on `onsets >= 1`, check the pose: the evader must sit inside 8 m and the pursuer must be spinning at full `omega_max` (`action[1] = 1.0`), which sweeps the 90° cone across it within a few steps at `dt = 0.05`.

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/structured.py tests/pe2/test_structured.py && ./venv/bin/ruff check pe2/structured.py tests/pe2/test_structured.py
git add pe2/structured.py tests/pe2/test_structured.py
git commit -m "feat(pe2): SearchMDP — structured pursuer with a fixed intercept law (S1)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `StructuredEvalProbe` behaviour test, `structured_verdict`, and `finetune_ppo` overrides

**Files:**
- Modify: `pe2/derisk.py` (append after `round_verdict`, around line 330)
- Modify: `pe2/exploitability.py:532-550` (`finetune_ppo`)
- Test: `tests/pe2/test_structured.py` (append), `tests/pe2/test_derisk.py` (append), `tests/pe2/test_exploitability.py` (append)

**Interfaces:**
- Consumes: `pe2.structured.StructuredEvalProbe` (Task 1), `pe2.eval.wilson_ci(k, n) -> (lo, hi)`.
- Produces:
  - `pe2.derisk.STRUCTURED_VERDICTS = ("IMPROVED", "LEARNED", "FAILED", "UNDETERMINED")`
  - `pe2.derisk.structured_verdict(final_k: int, final_n: int, bar_rate: float, floor_hi: float, start_hi: float) -> tuple[str, str]`
  - `pe2.exploitability.finetune_ppo(..., custom_objects: dict | None = None)` — forwarded to `PPO.load`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pe2/test_structured.py`:

```python
class TestStructuredEvalProbe:
    def test_probes_on_the_env_step_grid_and_stops_at_total(self, tmp_path):
        from pe2.structured import StructuredEvalProbe

        class _Model:
            def save(self, path):
                pathlib.Path(path).write_bytes(b"ckpt")

            def predict(self, obs, deterministic=True):
                return np.array([1.0, 0.0], dtype=np.float32), None

        class _VecEnv:
            def __init__(self):
                self.env_steps = 0

            def get_attr(self, name):
                return [getattr(self, name)]

        probe = StructuredEvalProbe(
            CFG, GreedyEvader, every=100, total=300, eval_n=2, seed0=900_700,
            ckpt_dir=tmp_path, report_path=tmp_path / "r.json", label="T",
        )
        cb, venv = probe.callback, _VecEnv()
        cb.model, cb.training_env = _Model(), venv  # what SB3's init_callback sets
        venv.env_steps = 57
        assert cb._on_step() is True and probe.rows == []
        venv.env_steps = 130  # first decision boundary past the 100 mark
        assert cb._on_step() is True
        assert [(r["mark"], r["steps"]) for r in probe.rows] == [(100, 130)]
        venv.env_steps = 310  # one stretch jumped over two marks
        assert cb._on_step() is False  # total reached: learn() must stop
        assert [(r["mark"], r["steps"]) for r in probe.rows] == [(100, 130), (200, 310), (300, 310)]
        assert (tmp_path / "T_300.zip").exists()
        report = json.loads((tmp_path / "r.json").read_text())
        assert [r["mark"] for r in report["checkpoints"]] == [100, 200, 300]
        assert {"steps", "mark", "ckpt", "win_rate", "k", "n"} <= set(probe.rows[0])
```

(add `import json` and `import pathlib` at the top of the test file.)

Append to `tests/pe2/test_derisk.py`:

```python
class TestStructuredVerdict:
    def test_improved_when_lower_bound_beats_the_orbit_bar(self):
        from pe2.derisk import structured_verdict

        v, reason = structured_verdict(300, 500, bar_rate=0.49, floor_hi=0.38, start_hi=0.30)
        assert v == "IMPROVED" and "0.490" in reason

    def test_learned_when_above_floor_and_start_but_not_the_bar(self):
        from pe2.derisk import structured_verdict

        v, reason = structured_verdict(220, 500, bar_rate=0.49, floor_hi=0.38, start_hi=0.30)
        assert v == "LEARNED" and "0.380" in reason

    def test_failed_when_not_separated_from_the_floor_or_the_start(self):
        from pe2.derisk import structured_verdict

        assert structured_verdict(180, 500, 0.49, 0.38, 0.30)[0] == "FAILED"
        # a high START_S upper bound is as binding as the floor
        assert structured_verdict(220, 500, 0.49, 0.30, 0.45)[0] == "FAILED"

    def test_verdict_vocabulary(self):
        from pe2.derisk import STRUCTURED_VERDICTS

        assert STRUCTURED_VERDICTS == ("IMPROVED", "LEARNED", "FAILED", "UNDETERMINED")
```

Append to `tests/pe2/test_exploitability.py`:

```python
def test_finetune_ppo_custom_objects_override_the_saved_gae_lambda(tmp_path):
    from stable_baselines3 import PPO

    from pe2.exploitability import PPO_KWARGS, SingleAgentWrapper, finetune_ppo
    from pe2.scripted import GreedyEvader

    cfg = EnvConfig()
    path = tmp_path / "ppo.zip"
    PPO("MlpPolicy", SingleAgentWrapper(cfg, "pursuer", GreedyEvader), seed=1, verbose=0, device="cpu", **PPO_KWARGS).save(path)
    m = finetune_ppo(cfg, "pursuer", GreedyEvader, path, total_timesteps=0, seed=1, device="cpu", custom_objects={"gae_lambda": 1.0})
    assert m.gae_lambda == 1.0
    assert PPO.load(path, device="cpu").gae_lambda == PPO_KWARGS["gae_lambda"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_structured.py::TestStructuredEvalProbe tests/pe2/test_derisk.py::TestStructuredVerdict tests/pe2/test_exploitability.py::test_finetune_ppo_custom_objects_override_the_saved_gae_lambda -q`
Expected: the probe test PASSES already (Task 1 shipped the class); the verdict tests FAIL with `ImportError`; the PPO test FAILS with `TypeError: unexpected keyword argument 'custom_objects'`.

- [ ] **Step 3: Implement**

Append to `pe2/derisk.py`:

```python
STRUCTURED_VERDICTS = ("IMPROVED", "LEARNED", "FAILED", "UNDETERMINED")


def structured_verdict(
    final_k: int,
    final_n: int,
    bar_rate: float,
    floor_hi: float,
    start_hi: float,
) -> tuple[str, str]:
    """(verdict, reason) for an S1 structured arm (spec section 5) on its
    final probe. IMPROVED: Wilson lower bound above the orbit bar's point
    estimate. LEARNED: above BOTH the random floor's and the untrained
    start's Wilson upper bounds, but not the bar. FAILED otherwise. The
    reason names the numbers so a reader never meets a bare label."""
    from pe2.eval import wilson_ci

    lo, hi = wilson_ci(final_k, final_n)
    final = (
        f"final {final_k}/{final_n} = {final_k / final_n:.3f}, "
        f"Wilson [{lo:.3f}, {hi:.3f}]"
    )
    if lo > bar_rate:
        return "IMPROVED", f"{final}: lower bound above the orbit bar {bar_rate:.3f}"
    ref = max(floor_hi, start_hi)
    if lo > ref:
        return "LEARNED", (
            f"{final}: lower bound above the floor/start upper bounds "
            f"(max {ref:.3f}) but not the orbit bar {bar_rate:.3f}"
        )
    return "FAILED", (
        f"{final}: lower bound does not clear the floor/start upper bounds "
        f"(max {ref:.3f})"
    )
```

Modify `finetune_ppo` in `pe2/exploitability.py`: add the parameter `custom_objects: dict | None = None` after `callback=None`, extend the docstring with one sentence — `custom_objects` is forwarded to `PPO.load` so a round can pre-register a hyper-parameter that differs from the clone's saved one (S1's control sets `gae_lambda = 1.0`) — and change the load line to:

```python
    model = PPO.load(
        model_path, env=wrapper, device=device, seed=seed, custom_objects=custom_objects
    )
```

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_structured.py tests/pe2/test_derisk.py tests/pe2/test_exploitability.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/derisk.py pe2/exploitability.py tests/pe2/test_structured.py tests/pe2/test_derisk.py tests/pe2/test_exploitability.py && ./venv/bin/ruff check pe2 tests/pe2
git add pe2/derisk.py pe2/exploitability.py tests/pe2/test_structured.py tests/pe2/test_derisk.py tests/pe2/test_exploitability.py
git commit -m "feat(derisk): structured_verdict; finetune_ppo custom_objects; probe test (S1)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `scripts/derisk_s1_structured.py` — the round runner, its tests, and the V1 pre-flight

**Files:**
- Create: `scripts/derisk_s1_structured.py`
- Test: `tests/pe2/test_derisk_s1_round.py`

**Interfaces:**
- Consumes: everything Task 1 and Task 2 produce; `pe2.exploitability.collect_demonstrations`, `train_bc_ppo`, `finetune_ppo`, `evaluate_sb3_policy`, `CheckpointEvalProbe`; `pe2.derisk.round_verdict`, `CLONE_CI_LO`, `ROUND_PASS_FROM`; `pe2.eval.wilson_ci`, `win_count`.
- Produces: CLI `--preflight | --measure-start | --arms {SA,SB,SZ,CP} | --summarize [--device cpu]`; artefacts exactly as spec §8; module-level functions `summarize(results, starts) -> dict` and `direction(verdicts) -> str` (tested).

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_derisk_s1_round.py
"""scripts/derisk_s1_structured.py — the structured round's own decisions."""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s1_structured.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s1_structured", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()

_STARTS = {
    "BAR_ORBIT": {"k": 245, "n": 500, "rate": 0.49, "wilson": [0.447, 0.533]},
    "FLOOR_RAND": {"k": 167, "n": 500, "rate": 0.334, "wilson": [0.294, 0.377]},
    "START_S": {"k": 100, "n": 500, "rate": 0.2, "wilson": [0.167, 0.237]},
    "START_CP": {"k": 230, "n": 500, "rate": 0.46, "wilson": [0.417, 0.504], "v2_pass": True},
}


def _arm(name, verdict, rate=0.5):
    return {"arm": name, "verdict": verdict, "reason": "r", "checkpoints": [], "final": {"rate": rate, "k": int(rate * 500), "n": 500, "wilson": [rate - 0.04, rate + 0.04]}}


class TestProtocol:
    def test_pre_registered_constants(self):
        from pe2.structured import HOLD, R_ACQ

        assert (R.TOTAL, R.EVERY, R.EVAL_N, R.FINAL_EVAL_N) == (1_000_000, 100_000, 200, 500)
        assert R.EVAL_SEED0 == 1_010_000 and R.DEMO_SEED0 == 1_020_000
        assert (R.DEMO_EPISODES, R.BC_EPOCHS, R.SEED) == (300, 50, 7)
        assert R.GAMMA_S == 0.99 and R.CP_LAMBDA == 1.0 and R.V2_MIN_LO == 0.35
        assert (R.HOLD, R.R_ACQ) == (HOLD, R_ACQ) == (10, 0.2)
        assert set(R.ARMS) == {"SA", "SB", "SZ", "CP"}

    def test_arm_reward_settings(self):
        assert R.ARMS["SA"] == {"r_acq": 0.2, "zero_reward": False}
        assert R.ARMS["SB"] == {"r_acq": 0.0, "zero_reward": False}
        assert R.ARMS["SZ"] == {"r_acq": 0.0, "zero_reward": True}
        assert R.ARMS["CP"] == "control"

    def test_arms_refuse_to_run_without_measured_bars(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "START_PATH", tmp_path / "missing.json")
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()

    def test_measure_start_refuses_without_a_passed_preflight(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "PREFLIGHT_PATH", tmp_path / "missing.json")
        with pytest.raises(SystemExit, match="preflight"):
            R.require_preflight()
        (tmp_path / "missing.json").write_text(json.dumps({"pass": False, "mismatches": 3}))
        with pytest.raises(SystemExit, match="V1"):
            R.require_preflight()


class TestDirection:
    def test_incomplete(self):
        assert "incomplete" in R.direction({"SA": "IMPROVED"})

    def test_sz_learned_invalidates_the_harness(self):
        d = R.direction({"SA": "IMPROVED", "SB": "LEARNED", "SZ": "LEARNED", "CP": "COLLAPSED"})
        assert "harness invalid" in d

    def test_sa_improved_cp_not(self):
        for cp in ("COLLAPSED", "PRESERVED", "UNDETERMINED"):
            d = R.direction({"SA": "IMPROVED", "SB": "FAILED", "SZ": "FAILED", "CP": cp})
            assert "structure is what makes learning work" in d and "Route C withdrawn" in d

    def test_both_improved(self):
        d = R.direction({"SA": "IMPROVED", "SB": "FAILED", "SZ": "FAILED", "CP": "IMPROVED"})
        assert "non-bootstrapping learner suffices" in d and "ablation" in d

    def test_sa_learned_not_improved(self):
        d = R.direction({"SA": "LEARNED", "SB": "FAILED", "SZ": "FAILED", "CP": "COLLAPSED"})
        assert "observation" in d and "not the learner" in d

    def test_sa_failed(self):
        d = R.direction({"SA": "FAILED", "SB": "FAILED", "SZ": "FAILED", "CP": "COLLAPSED"})
        assert "about the game" in d


class TestSummarize:
    def test_sz_learned_marks_structured_arms_undetermined(self):
        results = {"SA": _arm("SA", "IMPROVED"), "SB": _arm("SB", "LEARNED"), "SZ": _arm("SZ", "LEARNED"), "CP": _arm("CP", "COLLAPSED")}
        s = R.summarize(results, _STARTS)
        assert s["arms"]["SA"]["verdict"] == "UNDETERMINED (harness)"
        assert s["arms"]["SB"]["verdict"] == "UNDETERMINED (harness)"
        assert s["arms"]["CP"]["verdict"] == "COLLAPSED"
        assert s["complete"] is True

    def test_sb_at_least_sa_is_reported(self):
        results = {"SA": _arm("SA", "LEARNED", 0.40), "SB": _arm("SB", "LEARNED", 0.42), "SZ": _arm("SZ", "FAILED", 0.2), "CP": _arm("CP", "COLLAPSED", 0.0)}
        s = R.summarize(results, _STARTS)
        assert "event reward added nothing" in s["direction"]

    def test_weak_prior_caveat_survives_into_the_summary(self):
        starts = dict(_STARTS, START_CP=dict(_STARTS["START_CP"], v2_pass=False))
        results = {"SA": _arm("SA", "FAILED"), "SB": _arm("SB", "FAILED"), "SZ": _arm("SZ", "FAILED"), "CP": _arm("CP", "COLLAPSED (weak prior)")}
        s = R.summarize(results, starts)
        assert "weak prior" in s["arms"]["CP"]["verdict"]
        assert s["v2_pass"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s1_round.py -q`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Write the runner**

```python
# scripts/derisk_s1_structured.py
"""S1 — structured search: RL learns the search sub-policy under a fixed
intercept law (spec: docs/pathway/structured_search_prereg_2026-09-12.md).
Nothing here may change a threshold, a seed or a budget after a number is
seen.

Four arms, concurrent, 1M env steps each:
  SA  PPO from scratch on SearchMDP, terminal +/-1 + 0.2 per acquisition  <- the test
  SB  same, terminal only                                                 <- density ablation
  SZ  same, every reward zeroed                                           <- negative control
  CP  PPO-GAE lambda=1.0 from a PPO clone of SearchingPursuer, per-step   <- option-1 control

Usage (in this order):
  ./venv/bin/python scripts/derisk_s1_structured.py --preflight       (V1: wrapper == teacher)
  ./venv/bin/python scripts/derisk_s1_structured.py --measure-start   (bars + PPO clone + init)
  ./venv/bin/python scripts/derisk_s1_structured.py --arms SA         (one process per arm)
  ./venv/bin/python scripts/derisk_s1_structured.py --summarize
"""

from __future__ import annotations

import argparse
import json
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import (
    CLONE_CI_LO,
    ROUND_PASS_FROM,
    round_verdict,
    structured_verdict,
)
from pe2.eval import wilson_ci, win_count
from pe2.scripted import GreedyEvader, SearchingPursuer
from pe2.structured import HOLD, R_ACQ

OUT = pathlib.Path("results/derisk")
START_PATH = OUT / "s1_start.json"
PREFLIGHT_PATH = OUT / "s1_preflight.json"
INIT_PATH = OUT / "s1_ppo_init.zip"
CP_CLONE = OUT / "ppo_bc_searching_pursuer.zip"

# --- pre-registered (spec sections 2-5). Do not tune. ---
TOTAL = 1_000_000  # env steps per arm
EVERY = 100_000
EVAL_N = 200
FINAL_EVAL_N = 500
PASS_FROM = ROUND_PASS_FROM  # CP only: 500_000, bound to the library constant
EVAL_SEED0 = 1_010_000  # fresh: <= 1_000_999 consumed
DEMO_SEED0 = 1_020_000
DEMO_EPISODES = 300
BC_EPOCHS = 50
SEED = 7
GAMMA_S = 0.99  # structured arms, per decision
CP_LAMBDA = 1.0  # the control's GAE lambda (deviates from T0.4's 0.95 on purpose)
V2_MIN_LO = 0.35  # the PPO clone must be a competent prior
ARMS = {
    "SA": {"r_acq": R_ACQ, "zero_reward": False},
    "SB": {"r_acq": 0.0, "zero_reward": False},
    "SZ": {"r_acq": 0.0, "zero_reward": True},
    "CP": "control",
}
STRUCTURED = ("SA", "SB", "SZ")


def _ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


def _print(name: str, entry: dict) -> None:
    lo, hi = entry["wilson"]
    print(
        f"{name} = {entry['k']}/{entry['n']} = {entry['rate']:.4f}, "
        f"Wilson [{lo:.4f}, {hi:.4f}]",
        flush=True,
    )


# ----------------------------------------------------------------------------
# pre-flight (spec section 2, V1)
# ----------------------------------------------------------------------------
def preflight() -> dict:
    """V1: SearchMDP + orbit at hold=1 must reproduce SearchingPursuer
    episode-for-episode on all FINAL_EVAL_N bar seeds. A mismatch halts the
    round before any bar or arm exists."""
    from pe2.structured import OrbitSearch, evaluate_scripted, evaluate_structured

    cfg = EnvConfig()
    teacher = evaluate_scripted(cfg, SearchingPursuer, GreedyEvader, FINAL_EVAL_N, EVAL_SEED0)
    wrapped = evaluate_structured(
        OrbitSearch(cfg), cfg, GreedyEvader, FINAL_EVAL_N, EVAL_SEED0, hold=1
    )
    mism = [
        t["seed"]
        for t, w in zip(teacher, wrapped, strict=True)
        if (t["outcome"], t["steps"], t["los_breaks"])
        != (w["outcome"], w["steps"], w["los_breaks"])
    ]
    k, n = win_count(teacher, "pursuer")
    out = {
        "pass": not mism,
        "n": FINAL_EVAL_N,
        "eval_seed0": EVAL_SEED0,
        "mismatches": len(mism),
        "mismatch_seeds": mism[:20],
        "teacher_rate": k / n,
    }
    PREFLIGHT_PATH.write_text(json.dumps(out, indent=2))
    print(
        f"V1 {'PASS' if out['pass'] else 'FAIL'}: {len(mism)} mismatches in {n} "
        f"episodes (teacher rate {k / n:.3f})",
        flush=True,
    )
    if mism:
        raise SystemExit("V1 FAILED: the wrapper does not reproduce the teacher; halting.")
    return out


def require_preflight() -> None:
    if not PREFLIGHT_PATH.exists():
        raise SystemExit(
            f"preflight missing: {PREFLIGHT_PATH}. Run --preflight FIRST (spec V1)."
        )
    if not json.loads(PREFLIGHT_PATH.read_text()).get("pass"):
        raise SystemExit("V1 preflight FAILED; the round may not proceed.")


# ----------------------------------------------------------------------------
# bars (spec section 3), measured BEFORE any arm trains
# ----------------------------------------------------------------------------
def make_structured_env(cfg, arm: str):
    from pe2.structured import SearchMDP

    return SearchMDP(cfg, GreedyEvader, seed0=SEED * 1000, hold=HOLD, **ARMS[arm])


def measure_start(device: str) -> dict:
    from stable_baselines3 import PPO

    from pe2.exploitability import (
        collect_demonstrations,
        evaluate_sb3_policy,
        train_bc_ppo,
    )
    from pe2.structured import OrbitSearch, RandomSearch, evaluate_structured

    require_preflight()
    cfg = EnvConfig()
    out: dict = {}

    logs = evaluate_structured(OrbitSearch(cfg), cfg, GreedyEvader, FINAL_EVAL_N, EVAL_SEED0, hold=1)
    out["BAR_ORBIT"] = _ci(*win_count(logs, "pursuer")) | {"hold": 1}
    _print("BAR_ORBIT", out["BAR_ORBIT"])

    logs = evaluate_structured(RandomSearch(0), cfg, GreedyEvader, FINAL_EVAL_N, EVAL_SEED0, hold=HOLD)
    out["FLOOR_RAND"] = _ci(*win_count(logs, "pursuer")) | {"hold": HOLD, "rng_seed": 0}
    _print("FLOOR_RAND", out["FLOOR_RAND"])

    # The untrained init of SA/SB/SZ, saved so each arm loads literally this policy.
    init = PPO(
        "MlpPolicy", make_structured_env(cfg, "SA"), seed=SEED, gamma=GAMMA_S,
        verbose=0, device=device,
    )
    init.save(INIT_PATH)
    logs = evaluate_structured(init, cfg, GreedyEvader, FINAL_EVAL_N, EVAL_SEED0, hold=HOLD)
    out["START_S"] = _ci(*win_count(logs, "pursuer")) | {"init": str(INIT_PATH), "seed": SEED}
    _print("START_S", out["START_S"])

    # The control's prior: PPO clone of the searching teacher (T2 recipe).
    demos = collect_demonstrations(
        cfg, "pursuer", SearchingPursuer, GreedyEvader, DEMO_EPISODES, DEMO_SEED0
    )
    bc, losses = train_bc_ppo(
        cfg, "pursuer", GreedyEvader, demos, seed=SEED, device=device, epochs=BC_EPOCHS
    )
    bc.save(CP_CLONE)
    logs = evaluate_sb3_policy(
        bc, cfg, "pursuer", GreedyEvader, n_episodes=FINAL_EVAL_N, seed0=EVAL_SEED0
    )
    entry = _ci(*win_count(logs, "pursuer"))
    entry |= {
        "clone": str(CP_CLONE),
        "final_bc_loss": losses[-1],
        "demo_episodes": DEMO_EPISODES,
        "demo_seed0": DEMO_SEED0,
        "v2_pass": entry["wilson"][0] >= V2_MIN_LO,
    }
    out["START_CP"] = entry
    _print("START_CP", entry)
    print(
        f"V2 {'PASS' if entry['v2_pass'] else 'FAIL'}: START_CP Wilson lo "
        f"{entry['wilson'][0]:.4f} vs {V2_MIN_LO}",
        flush=True,
    )
    out["eval_seed0"] = EVAL_SEED0
    START_PATH.write_text(json.dumps(out, indent=2))
    return out


def load_start() -> dict:
    if not START_PATH.exists():
        raise SystemExit(
            f"start bars missing: {START_PATH}. Run --measure-start FIRST; the "
            "pre-registration requires it before any arm trains."
        )
    s = json.loads(START_PATH.read_text())
    for k in ("BAR_ORBIT", "FLOOR_RAND", "START_S", "START_CP"):
        if k not in s:
            raise SystemExit(f"{k} missing from {START_PATH}; re-run --measure-start.")
    return s


# ----------------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------------
def run_structured_arm(arm: str, starts: dict, device: str) -> dict:
    from stable_baselines3 import PPO

    from pe2.structured import StructuredEvalProbe, evaluate_structured

    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = StructuredEvalProbe(
        cfg, GreedyEvader, every=EVERY, total=TOTAL, eval_n=EVAL_N, seed0=EVAL_SEED0,
        ckpt_dir=d, report_path=OUT / f"s1_probe_{arm}.json", label=arm, hold=HOLD,
    )
    env = make_structured_env(cfg, arm)
    model = PPO.load(INIT_PATH, env=env, device=device, seed=SEED)
    # The budget is enforced in ENV steps by the probe (it stops learn());
    # the decision budget here is only a ceiling that can never bind first.
    model.learn(total_timesteps=TOTAL, callback=probe.callback)
    result = {
        "arm": arm,
        "settings": ARMS[arm],
        "seed": SEED,
        "gamma": GAMMA_S,
        "hold": HOLD,
        "env_steps": int(env.env_steps),
        "decisions": int(model.num_timesteps),
        "eval_n": EVAL_N,
        "eval_seed0": EVAL_SEED0,
        "checkpoints": probe.rows,
    }
    ckpt = d / f"{arm}_{TOTAL}.zip"
    if not any(r["mark"] == TOTAL for r in probe.rows) or not ckpt.exists():
        verdict = "UNDETERMINED"
        reason = (
            f"final probe at mark {TOTAL} never fired (env steps reached "
            f"{env.env_steps}); judged with no final probe rather than losing the "
            "probe history gathered before the crash"
        )
    else:
        final = PPO.load(ckpt, device=device)
        logs = evaluate_structured(final, cfg, GreedyEvader, FINAL_EVAL_N, EVAL_SEED0, hold=HOLD)
        result["final"] = _ci(*win_count(logs, "pursuer")) | {"ckpt": str(ckpt)}
        verdict, reason = structured_verdict(
            result["final"]["k"],
            result["final"]["n"],
            starts["BAR_ORBIT"]["rate"],
            starts["FLOOR_RAND"]["wilson"][1],
            starts["START_S"]["wilson"][1],
        )
    result["verdict"], result["reason"] = verdict, reason
    return result


def run_control_arm(starts: dict, device: str) -> dict:
    from stable_baselines3 import PPO

    from pe2.exploitability import CheckpointEvalProbe, evaluate_sb3_policy, finetune_ppo

    arm = "CP"
    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = CheckpointEvalProbe(
        cfg, "pursuer", GreedyEvader, every=EVERY, eval_n=EVAL_N, seed0=EVAL_SEED0,
        ckpt_dir=d, report_path=OUT / f"s1_probe_{arm}.json", label=arm,
    )
    finetune_ppo(
        cfg, "pursuer", GreedyEvader, CP_CLONE, TOTAL, seed=SEED, device=device,
        callback=probe.callback, custom_objects={"gae_lambda": CP_LAMBDA},
    )
    bar = starts["START_CP"]["rate"]
    result = {
        "arm": arm,
        "prior": str(CP_CLONE),
        "gae_lambda": CP_LAMBDA,
        "seed": SEED,
        "timesteps": TOTAL,
        "eval_n": EVAL_N,
        "eval_seed0": EVAL_SEED0,
        "bar": {"name": "START_CP", "rate": bar},
        "checkpoints": probe.rows,
    }
    ckpt = d / f"{arm}_{TOTAL}.zip"
    if not ckpt.exists():
        verdict = "UNDETERMINED"
        reason = f"final checkpoint missing: {ckpt} was never written"
    else:
        final = PPO.load(ckpt, device=device)
        logs = evaluate_sb3_policy(
            final, cfg, "pursuer", GreedyEvader, n_episodes=FINAL_EVAL_N, seed0=EVAL_SEED0
        )
        result["final"] = _ci(*win_count(logs, "pursuer")) | {"ckpt": str(ckpt)}
        verdict, reason = round_verdict(
            probe.rows, result["final"]["k"], result["final"]["n"], bar, from_step=PASS_FROM
        )
    if not starts["START_CP"].get("v2_pass", True):
        verdict = f"{verdict} (weak prior)"
        reason += f"; V2 failed: START_CP Wilson lo below {V2_MIN_LO}, not the competent prior"
    result["verdict"], result["reason"] = verdict, reason
    return result


def run_arm(arm: str, device: str) -> dict:
    starts = load_start()  # FIRST: refuse before training if the bars are missing
    if ARMS[arm] == "control":
        result = run_control_arm(starts, device)
    else:
        if not INIT_PATH.exists():
            raise SystemExit(f"init missing: {INIT_PATH}; re-run --measure-start.")
        result = run_structured_arm(arm, starts, device)
    (OUT / f"s1_arm_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] verdict={result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# summary (spec section 6), fixed before the run
# ----------------------------------------------------------------------------
_RANK = {"FAILED": 0, "LEARNED": 1, "IMPROVED": 2}


def direction(v: dict) -> str:
    sa, sb, sz, cp = (v.get(a) for a in ("SA", "SB", "SZ", "CP"))
    if any(x is None for x in (sa, sb, sz, cp)):
        return "incomplete: all four arms must run"
    if sz in ("LEARNED", "IMPROVED"):
        return (
            "SZ LEARNED: harness invalid — the zero-reward control moved, so "
            "something other than the reward is moving the policy; nothing else "
            "is read (spec section 6)."
        )
    cp_base = cp.split(" ")[0]
    if sa == "IMPROVED" and cp_base == "IMPROVED":
        return (
            "SA IMPROVED and CP IMPROVED: a non-bootstrapping learner suffices on "
            "the per-step problem. The unstructured learned pursuer is the simpler "
            "artefact; the structure is reported as an ablation, not the "
            "contribution. Route C withdrawn either way."
        )
    if sa == "IMPROVED":
        return (
            f"SA IMPROVED, CP {cp}: hypothesis S confirmed, C refuted — the "
            "structure is what makes learning work here. Route C withdrawn. The "
            "paper becomes learned search under a fixed intercept law with a hard "
            "CBF shield; next pre-registration adds the shield to the structured "
            "pursuer. SB says whether density was needed."
        )
    if sa == "LEARNED":
        return (
            f"SA LEARNED, not IMPROVED (CP {cp}): the search MDP is learnable but "
            "the orbit is hard to beat from an 11-dim memoryless observation. The "
            "next question is the search policy's observation (coverage / "
            "staleness memory), not the learner. Route C re-scoped to per-step "
            "fine-tuning; not withdrawn."
        )
    return (
        f"SA FAILED (SZ valid, CP {cp}): PPO cannot learn even a ~20-decision "
        "search MDP at this budget. The negative result is, for the first time, "
        "about the game and not about a critic — Route C stands on a stronger "
        "footing; D82/D100/D102/D103 are re-scoped as one-step-TD results."
    )


def summarize(results: dict, starts: dict) -> dict:
    v = {a: r["verdict"] for a, r in results.items()}
    arms = {a: {"verdict": r["verdict"], "reason": r["reason"]} for a, r in results.items()}
    if v.get("SZ") in ("LEARNED", "IMPROVED"):
        for a in STRUCTURED:
            if a in arms and a != "SZ":
                arms[a]["verdict"] = "UNDETERMINED (harness)"
    d = direction(v)
    sa, sb = results.get("SA"), results.get("SB")
    if sa and sb and "final" in sa and "final" in sb:
        if _RANK.get(v["SB"], -1) > _RANK.get(v["SA"], -1) or (
            _RANK.get(v["SB"], -1) == _RANK.get(v["SA"], -1)
            and sb["final"]["rate"] >= sa["final"]["rate"]
        ):
            d += (
                " SB >= SA: the event reward added nothing; prefer SB as the "
                "simpler arm."
            )
    return {
        "spec": "docs/pathway/structured_search_prereg_2026-09-12.md",
        "arms": arms,
        "bars": {k: starts[k]["rate"] for k in ("BAR_ORBIT", "FLOOR_RAND", "START_S", "START_CP")},
        "v2_pass": bool(starts["START_CP"].get("v2_pass", True)),
        "complete": all(a in results for a in ARMS),
        "direction": d,
    }


def load_results() -> dict:
    return {
        p.stem.removeprefix("s1_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob("s1_arm_*.json"))
    }


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for arm, r in results.items():
        rows = r["checkpoints"]
        if rows:
            ax.plot(
                [x["steps"] for x in rows],
                [x["win_rate"] for x in rows],
                marker="o",
                label=f"{arm} ({summary['arms'][arm]['verdict']})",
            )
        if "final" in r:
            lo, hi = r["final"]["wilson"]
            rate = r["final"]["rate"]
            ax.errorbar([TOTAL + 20_000], [rate], yerr=[[rate - lo], [hi - rate]], fmt="s", capsize=3)
    refs = (
        ("BAR_ORBIT", "tab:green", "the bar"),
        ("FLOOR_RAND", "tab:orange", "random floor"),
        ("START_S", "tab:purple", "untrained init"),
        ("START_CP", "tab:red", "PPO clone"),
    )
    for key, color, what in refs:
        ax.axhline(starts[key]["rate"], color=color, ls=":", lw=1.3, label=f"{key} = {starts[key]['rate']:.3f} ({what})")
    ax.axhline(CLONE_CI_LO, color="k", ls="--", lw=0.8, label=f"CP collapse floor {CLONE_CI_LO}")
    ax.axvline(PASS_FROM, color="grey", ls="-.", lw=0.8, label=f"CP pass window from {PASS_FROM}")
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"capture rate vs GreedyEvader (paired seeds {EVAL_SEED0}+)")
    ax.set_title(
        "S1 — structured search under a fixed intercept law"
        + ("" if summary["complete"] else " — INCOMPLETE")
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        preflight()
        return
    if args.measure_start:
        measure_start(args.device)
        return
    for arm in args.arms:
        run_arm(arm, args.device)
    if args.summarize:
        results = load_results()
        starts = load_start()
        summary = summarize(results, starts)
        (OUT / "s1_round.json").write_text(
            json.dumps({"summary": summary, "bars": starts, "arms": results}, indent=2)
        )
        write_figure(results, starts, summary, OUT / "s1_round.png")
        print(json.dumps(summary["arms"], indent=2))
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
```

Note on `model.learn(total_timesteps=TOTAL, ...)` in `run_structured_arm`: SB3 counts learner decisions there, and each decision consumes ≥ 1 env step, so `TOTAL` decisions can never be reached before `TOTAL` env steps — the probe's `return env_steps < total` is what actually stops training.

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s1_round.py -q`
Expected: all PASS.

- [ ] **Step 5: Run the V1 pre-flight (the plan's only run; ~5 min, CPU)**

Run: `./venv/bin/python scripts/derisk_s1_structured.py --preflight`
Expected: `V1 PASS: 0 mismatches in 500 episodes (teacher rate 0.4xx)` and `results/derisk/s1_preflight.json` with `"pass": true`. If it FAILS, do not touch the teacher: the wrapper is wrong. The usual cause is a divergence in when `last_seen` is updated or cleared relative to `SearchingPursuer.act` (it must be set on every visible step and cleared only when within `LAST_SEEN_TOL`), or the reset-time visible stretch being handled differently. Fix the wrapper until V1 passes.

- [ ] **Step 6: Lint, full suite, commit**

```bash
./venv/bin/ruff format scripts/derisk_s1_structured.py tests/pe2/test_derisk_s1_round.py && ./venv/bin/ruff check scripts/derisk_s1_structured.py tests/pe2/test_derisk_s1_round.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q -x   # full suite must be green
git add scripts/derisk_s1_structured.py tests/pe2/test_derisk_s1_round.py results/derisk/s1_preflight.json
git commit -m "feat(derisk): S1 runner — preflight, bars, four arms, summary; V1 preflight PASS" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** §2 wrapper, hold, onset reward, env-step budget, V1 → Task 1 (+ Task 3 Step 5). §3 bars incl. V2, PPO clone recipe, init shared by SA/SB/SZ, λ = 1.0 control → Task 3 (`measure_start`, `run_control_arm`), Task 2 (`custom_objects`). §4 arms → `ARMS`. §5 verdicts → Task 2 (`structured_verdict`), CP via `round_verdict`; SZ-harness rule and weak-prior caveat → `summarize`. §6 direction table → `direction`. §8 artefacts → runner paths and figure.
- **Placeholders:** none; every code step is complete.
- **Type consistency:** `evaluate_structured(model, cfg, opponent_factory, n_episodes, seed0, hold)` used identically in Tasks 1 and 3; probe rows carry `steps`/`mark`/`ckpt`/`win_rate`/`k`/`n` everywhere; `structured_verdict(final_k, final_n, bar_rate, floor_hi, start_hi)` matches its call site.
