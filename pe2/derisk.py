"""Tier 0 diagnostics (docs/pathway/derisk_plan_2026-09-04.md, T0.1).

signal_audit rolls a saved policy in the training-time regime and measures
the learning signal the terminal-only reward actually provides: positive
terminals per million steps (the exploration bottleneck), the capture-
length distribution, the fraction of steps with the opponent visible, and
the variance of the Monte-Carlo return by step index. "Positive" is
role-aware (a win for the audited role, `positives_per_1M`/`positive_lengths`)
and is NOT the same as a literal capture event (`captures`/`capture_lengths`):
for the pursuer the two coincide, but for the evader a capture is a LOSS, so
the kill rule below is gated on `positives_per_1M`, never on `captures`. The
pre-registered kill rule (< 50 positives per 1M even with dense reward ->
Route C) is evaluated on the P-A pilot's training episodes, not here; this
audit gives the terminal-only baseline number.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pe2.config import EnvConfig
from pe2.exploitability import SingleAgentWrapper


def mc_returns(length: int, terminal_reward: float, gamma: float) -> list[float]:
    """Monte-Carlo return of every step of a terminal-only episode:
    G_t = gamma**(length - t) * R for t = 0..length-1."""
    return [gamma ** (length - t) * terminal_reward for t in range(length)]


def signal_audit(
    model,
    cfg: EnvConfig,
    role: str,
    opponent_factory: Callable[[], object],
    min_steps: int,
    seed0: int,
    gamma: float = 0.999,
    deterministic: bool = False,
    bucket: int = 200,
) -> dict:
    """Roll `model` (anything with .predict) until >= min_steps env steps."""
    wrapper = SingleAgentWrapper(cfg, role, opponent_factory)
    total_steps = 0
    visible_steps = 0
    episode_lengths: list[int] = []
    capture_lengths: list[int] = []
    positive_lengths: list[int] = []
    per_step_returns: list[tuple[int, float]] = []  # (t, G_t)
    ep = 0
    while total_steps < min_steps:
        obs, _ = wrapper.reset(seed=seed0 + ep)
        ep += 1
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, info = wrapper.step(action)
            visible_steps += int(wrapper._last_joint_obs[role][6] > 0.5)
            if terminated or truncated:
                break
        length = wrapper.env.current_step
        total_steps += length
        episode_lengths.append(length)
        won = (info["outcome"] == "capture") == (role == "pursuer")
        if info["outcome"] == "capture":
            capture_lengths.append(length)
        if won:
            positive_lengths.append(length)
        terminal = 1.0 if won else -1.0
        per_step_returns.extend(enumerate(mc_returns(length, terminal, gamma)))

    captures = len(capture_lengths)
    positives = len(positive_lengths)
    ts = np.array([t for t, _ in per_step_returns])
    gs = np.array([g for _, g in per_step_returns])
    buckets = []
    for lo in range(0, cfg.max_steps, bucket):
        mask = (ts >= lo) & (ts < lo + bucket)
        n = int(mask.sum())
        buckets.append(
            {
                "t_lo": lo,
                "t_hi": lo + bucket,
                "n": n,
                "variance": float(gs[mask].var()) if n else 0.0,
                "mean": float(gs[mask].mean()) if n else 0.0,
            }
        )
    return {
        "role": role,
        "seed0": seed0,
        "deterministic": deterministic,
        "gamma": gamma,
        "total_steps": total_steps,
        "n_episodes": len(episode_lengths),
        "episode_lengths": episode_lengths,
        # captures: literal capture events (may be a LOSS for the evader).
        "captures": captures,
        "capture_lengths": capture_lengths,
        # positives: terminal WINS for the audited role (the kill-rule metric).
        "positives": positives,
        "positive_lengths": positive_lengths,
        "positives_per_1M": positives / total_steps * 1e6,
        "visible_fraction": visible_steps / total_steps,
        "return_variance_overall": float(gs.var()),
        "return_variance_by_bucket": buckets,
    }


# ---------------------------------------------------------------------------
# T0.3 pilot pass rule and training-signal counter (docs/pathway/
# derisk_plan_2026-09-04.md). Pre-registered before any pilot was run.
# ---------------------------------------------------------------------------

# Measured, NOT a tunable: the gated SAC behaviour clone scored 0.185 with a
# Wilson 95% interval of [0.137, 0.245] at N=200 (S119, D76 gate battery).
# 0.137 is that interval's LOWER bound, so "win rate >= CLONE_CI_LO" is the
# weakest claim that is still consistent with not having damaged the clone.
# Changing this number invalidates the pre-registration.
CLONE_CI_LO = 0.137
PASS_FROM_STEP = 100_000


def pilot_passes(
    rows: list[dict], threshold: float = CLONE_CI_LO, from_step: int = PASS_FROM_STEP
) -> bool:
    """Pre-registered T0.3 rule: win rate >= threshold at EVERY probe from
    `from_step` onward; no probe in the window -> fail.

    `rows` are `CheckpointEvalProbe` rows ({"steps", "win_rate", ...}). The
    comparison is `>=`, so a probe exactly at the bound passes; a run that
    never reached `from_step` has nothing to judge and therefore fails.
    """
    window = [r for r in rows if r["steps"] >= from_step]
    return bool(window) and all(r["win_rate"] >= threshold for r in window)


class TrainingCaptureCounter:
    """SB3 callback: count training episodes ending in capture, and env
    steps, so the T0.1 kill rule (positives per 1M) can be read off the P-A
    run's own rollouts.

    Pursuer-only metric. For the pursuer a capture IS a positive terminal,
    so `positives_per_1M` here matches `signal_audit`'s definition; for the
    evader a capture is a LOSS and this class would count the wrong thing.

    VERSION-COUPLED: `_on_step` reads `self.locals["dones"]` and
    `self.locals["infos"]`, which are SB3 rollout-loop internals. Both
    `OffPolicyAlgorithm.collect_rollouts` (SAC/`finetune_br`) and
    `OnPolicyAlgorithm.collect_rollouts` (PPO/`finetune_ppo`) publish them
    via `callback.update_locals(locals())` immediately after `env.step` in
    SB3 2.x; a future SB3 could rename them. Counting `len(dones)` steps per
    call is the VecEnv's `num_envs` — 1 for every `SingleAgentWrapper` path
    here, since SB3 auto-wraps the single Gym env in a `DummyVecEnv`.
    """

    def __init__(self):
        from stable_baselines3.common.callbacks import BaseCallback  # lazy

        state = {"steps": 0, "captures": 0, "episodes": 0}

        class _CB(BaseCallback):
            def _on_step(inner) -> bool:
                dones = inner.locals["dones"]
                infos = inner.locals["infos"]
                state["steps"] += len(dones)
                for done, info in zip(dones, infos):
                    if done:
                        state["episodes"] += 1
                        state["captures"] += int(info.get("outcome") == "capture")
                return True

        self.state = state
        self.callback = _CB()

    @property
    def positives_per_1M(self) -> float:
        return self.state["captures"] / max(self.state["steps"], 1) * 1e6
