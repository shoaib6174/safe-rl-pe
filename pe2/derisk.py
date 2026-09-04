"""Tier 0 diagnostics (docs/pathway/derisk_plan_2026-09-04.md, T0.1).

signal_audit rolls a saved policy in the training-time regime and measures
the learning signal the terminal-only reward actually provides: positive
terminals per million steps (the exploration bottleneck), the capture-
length distribution, the fraction of steps with the opponent visible, and
the variance of the Monte-Carlo return by step index. The pre-registered
kill rule (< 50 positives per 1M even with dense reward -> Route C) is
evaluated on the P-A pilot's training episodes, not here; this audit gives
the terminal-only baseline number.
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
        terminal = 1.0 if won else -1.0
        per_step_returns.extend(enumerate(mc_returns(length, terminal, gamma)))

    captures = len(capture_lengths)
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
        "captures": captures,
        "capture_lengths": capture_lengths,
        "positives_per_1M": captures / total_steps * 1e6,
        "visible_fraction": visible_steps / total_steps,
        "return_variance_overall": float(gs.var()),
        "return_variance_by_bucket": buckets,
    }
