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
from pe2.exploitability import SingleAgentWrapper
from pe2.metrics import summarize_shield
from pe2.obs import full_state_observation, observation_space
from pe2.scripted import K_OMEGA, ObstacleHuggingEvader, _clip_omega

D_POS = 4.0  # m: the hug law engages within this pursuer distance (= HUG_RANGE)


def _normalize_evader(v: float, w: float, cfg: EnvConfig) -> np.ndarray:
    return np.array(
        [2.0 * v / cfg.evader_v_max - 1.0, w / cfg.omega_max], dtype=np.float32
    )


class SplitHugEvader:
    """predict() shim: the structure's own scripted positioning rule — plain
    flee directly away from the pursuer, computed from the full-state
    body-frame opponent features (obs[3], obs[4])."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg

    def predict(self, obs, deterministic: bool = True):
        err = float(np.arctan2(-obs[4], -obs[3]))  # body-frame heading of 'away'
        return (
            _normalize_evader(
                self.cfg.evader_v_max, _clip_omega(K_OMEGA * err, self.cfg), self.cfg
            ),
            None,
        )


class EvadeMDP(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        cfg: EnvConfig,
        defender_factory: Callable[[], object],
        seed0: int = 0,
        hold: int = 10,
        beta: float = 0.0,
        zero_reward: bool = False,
        d_pos: float = D_POS,
    ):
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
        self.trace: list | None = None  # (branch, distance) per env step when assigned
        self._info: dict = {}

    @property
    def defender(self):
        return self.inner._opponent

    def _distance(self) -> float:
        env = self.inner.env
        return float(
            np.hypot(
                env.pursuer_pose[0] - env.evader_pose[0],
                env.pursuer_pose[1] - env.evader_pose[1],
            )
        )

    def _law_applies(self) -> bool:
        return self._distance() <= self.d_pos

    def _learner_obs(self) -> np.ndarray:
        env = self.inner.env
        return full_state_observation(
            env.evader_pose, env.pursuer_pose, env.obstacles, self.cfg
        )

    def _begin(self, seed: int | None) -> None:
        self.inner.reset(seed=seed)
        self.law.reset()
        self.span_stalled = 0
        self._info = {}

    def _step_inner(self, a_norm: np.ndarray, branch: str):
        if self.trace is not None:
            self.trace.append((branch, self._distance()))
        _obs, r, term, trunc, info = self.inner.step(a_norm)
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
        info = dict(
            self._info,
            env_steps=self.env_steps,
            pursuer_stalled_steps=self.span_stalled,
            raw_reward=reward,
        )
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
            "seed": seed,
            "role": "evader",
            "outcome": self._info["outcome"],
            "steps": self.inner.env.current_step,
            "escaped": self._info["outcome"] == "timeout",
            "pursuer_violations": int(d.metrics.n_violations)
            if d.metrics is not None
            else None,
            "pursuer_deadlock": bool(summary["deadlock"]) if summary else False,
            "pursuer_shield": summary,
        }


def evaluate_evader(
    model,
    cfg: EnvConfig,
    defender_factory,
    n_episodes: int,
    seed0: int,
    hold: int = 10,
    d_pos: float = D_POS,
) -> list[dict]:
    mdp = EvadeMDP(cfg, defender_factory, hold=hold, d_pos=d_pos)
    return [mdp.play_episode(model.predict, seed0 + i) for i in range(n_episodes)]


def play_scripted_evader(
    cfg: EnvConfig, evader_factory, defender_factory, n_episodes: int, seed0: int
) -> list[dict]:
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
            w._opponent.record(
                bool(info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"])
            )
            if term or trunc:
                break
        d = w._opponent
        summary = d.metrics.summary() if d.metrics is not None else None
        logs.append(
            {
                "seed": seed0 + i,
                "role": "evader",
                "outcome": info["outcome"],
                "steps": w.env.current_step,
                "escaped": info["outcome"] == "timeout",
                "pursuer_violations": int(d.metrics.n_violations)
                if d.metrics is not None
                else None,
                "pursuer_deadlock": bool(summary["deadlock"]) if summary else False,
                "pursuer_shield": summary,
            }
        )
    return logs


def pursuer_summary(logs: list[dict]) -> dict:
    """Pooled defender numbers over evader logs (the summarize_shield block
    plus the escape rate)."""
    shield_logs = [
        {
            "shielded_role": "pursuer",
            "violations": lg["pursuer_violations"],
            "shield": lg["pursuer_shield"],
            # summarize_shield also pools the game outcome (timeout_rate /
            # capture_rate), so the evader log's outcome has to travel with it.
            "outcome": lg["outcome"],
        }
        for lg in logs
    ]
    s = summarize_shield(shield_logs)
    esc = sum(lg["escaped"] for lg in logs)
    return s | {"escape_rate": esc / len(logs), "n_episodes": len(logs)}


class EvaderEvalProbe:
    """Env-step-grid probe for EvadeMDP training (mirrors StructuredEvalProbe)."""

    def __init__(
        self,
        cfg,
        defender_factory,
        every,
        total,
        eval_n,
        seed0,
        ckpt_dir,
        report_path,
        label,
        hold=10,
        d_pos=D_POS,
    ):
        from stable_baselines3.common.callbacks import BaseCallback

        if total % every != 0:
            raise ValueError(f"total ({total}) must be a multiple of every ({every})")
        rows: list[dict] = []

        def evaluate_at(model, env_steps: int, mark: int) -> dict:
            path = ckpt_dir / f"{label}_{mark}.zip"
            model.save(path)
            logs = evaluate_evader(
                model, cfg, defender_factory, eval_n, seed0, hold=hold, d_pos=d_pos
            )
            k = sum(lg["escaped"] for lg in logs)
            dl = sum(lg["pursuer_deadlock"] for lg in logs)
            row = {
                "steps": env_steps,
                "mark": mark,
                "ckpt": str(path),
                "escape_rate": k / eval_n,
                "deadlock_rate": dl / eval_n,
                "k": k,
                "n": eval_n,
                "pursuer_violations": int(
                    sum(lg["pursuer_violations"] or 0 for lg in logs)
                ),
            }
            rows.append(row)
            with open(report_path, "w") as f:
                json.dump({"checkpoints": rows}, f, indent=2)
            print(
                f"  @{env_steps:>9} env steps (mark {mark}): escape {k}/{eval_n} = "
                f"{k / eval_n:.3f}, pursuer deadlock eps {dl / eval_n:.3f}, "
                f"pursuer violations {row['pursuer_violations']}",
                flush=True,
            )
            return row

        class _CB(BaseCallback):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.next_mark = every

            def _on_step(inner_self) -> bool:
                counts = inner_self.training_env.get_attr("env_steps")
                assert len(counts) == 1
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
        d = float(
            np.hypot(
                env.pursuer_pose[0] - env.evader_pose[0],
                env.pursuer_pose[1] - env.evader_pose[1],
            )
        )
        if d <= self.d_pos:
            self._hold_left = 0
            v, w = self.law.act(None, env)
            a = _normalize_evader(v, w, cfg)
        else:
            if self._hold_left <= 0:
                lobs = full_state_observation(
                    env.evader_pose, env.pursuer_pose, env.obstacles, cfg
                )
                self._held, _ = self.model.predict(lobs, deterministic=True)
                self._hold_left = self.hold
                self.decisions += 1
            a = self._held
            self._hold_left -= 1
        a64 = np.clip(
            np.asarray(np.asarray(a, dtype=np.float32), dtype=float), -1.0, 1.0
        )
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
