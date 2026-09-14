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
        # optional per-step record (visible, branch, v, w) for characterise(); off by default
        self.trace: list | None = None

    # -- internals -------------------------------------------------------
    def _begin(self, seed: int | None) -> np.ndarray:
        obs, _ = self.inner.reset(seed=seed)
        self.law.reset()
        self._visible = bool(obs[6] > 0.5)
        self._los_breaks = 0
        self._info = {}
        return obs

    def _step_inner(self, action: np.ndarray, branch: str = "law"):
        prev = self._visible
        obs, r, term, trunc, info = self.inner.step(action)
        self.env_steps += 1
        self._visible = bool(obs[6] > 0.5)
        self._los_breaks += int(prev and not self._visible)
        self._info = info
        if self.trace is not None:
            self.trace.append(
                (self._visible, branch, float(action[0]), float(action[1]))
            )
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
            obs, r, onset, term, trunc = self._step_inner(a, "search")
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

        if total % every != 0:
            raise ValueError(
                f"total ({total}) must be a multiple of every ({every}) so the "
                f"final mark is the final checkpoint"
            )

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
                counts = inner_self.training_env.get_attr("env_steps")
                assert len(counts) == 1, (
                    "StructuredEvalProbe assumes a single env (budget is in env steps)"
                )
                env_steps = int(counts[0])
                while (
                    inner_self.next_mark <= total and env_steps >= inner_self.next_mark
                ):
                    evaluate_at(inner_self.model, env_steps, inner_self.next_mark)
                    inner_self.next_mark += every
                return env_steps < total

        self.rows = rows
        self.callback = _CB()
        self.evaluate_at = evaluate_at


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
