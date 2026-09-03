"""Step 3+: evaluation harness.

Step 3 shipped the core statistics (Wilson CI, battery runner over scripted
duels). Step 5 adds the binomial test (D56), role-aware win rates, and
best-checkpoint selection (D14). The paired-seed protocol (D57) is inherent
in `run_battery` (seed0 + i across all matchups); the frozen-opponent
protocol (D55) is inherent in evaluating against fresh scripted/policy
instances that never learn.

Capture rate (CR) is from the pursuer's perspective: fraction of episodes
ending in capture. Win rate is role-relative.
"""

from __future__ import annotations

from collections.abc import Callable
from math import comb

import numpy as np

from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.scripted import run_duel

BATTERY_SEED0 = 700000  # D57 eval-seed convention: reset(seed=seed0 + i)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (D56)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return float(center - half), float(center + half)


def binomial_pvalue(
    k: int, n: int, p0: float = 0.5, alternative: str = "two-sided"
) -> float:
    """Exact binomial test p-value (D56).

    alternative: "less" (P[X <= k]), "greater" (P[X >= k]), or "two-sided"
    (probability-weighted, like scipy.stats.binomtest: sum of probabilities
    of all outcomes at most as likely as k).
    """
    probs = [comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(n + 1)]
    if alternative == "less":
        return float(sum(probs[: k + 1]))
    if alternative == "greater":
        return float(sum(probs[k:]))
    if alternative == "two-sided":
        pk = probs[k]
        return float(min(1.0, sum(p for p in probs if p <= pk * (1 + 1e-9))))
    raise ValueError(f"unknown alternative: {alternative!r}")


def win_count(logs: list[dict], role: str) -> tuple[int, int]:
    """(wins, n) from the given role's perspective (capture = pursuer win)."""
    if role == "pursuer":
        return sum(log["outcome"] == "capture" for log in logs), len(logs)
    if role == "evader":
        return sum(log["outcome"] == "timeout" for log in logs), len(logs)
    raise ValueError(f"unknown role: {role!r}")


def win_ci(logs: list[dict], role: str, z: float = 1.96) -> tuple[float, float]:
    """Wilson CI on the WIN rate from `role`'s perspective (D56)."""
    k, n = win_count(logs, role)
    return wilson_ci(k, n, z)


def mcnemar_pvalue(
    logs_a: list[dict], logs_b: list[dict], role: str
) -> tuple[float, int, int]:
    """Exact two-sided McNemar test on PAIRED episodes (D76).

    Both log lists must cover the same episode seeds (D57 paired protocol);
    pairing is by the "seed" field. Only discordant pairs inform the test:
    b = episodes A wins and B loses, c = episodes B wins and A loses. Under
    the null (policies equally good), b ~ Binomial(b + c, 0.5).

    Returns (p_value, b, c). If b + c == 0 the policies agree everywhere
    and p = 1.0."""
    by_seed_b = {log["seed"]: log for log in logs_b}
    win_outcome = {"pursuer": "capture", "evader": "timeout"}.get(role)
    if win_outcome is None:
        raise ValueError(f"unknown role: {role!r}")
    b = c = 0
    for la in logs_a:
        lb = by_seed_b.get(la["seed"])
        if lb is None:
            raise ValueError(f"unpaired episode seed: {la['seed']}")
        wa = la["outcome"] == win_outcome
        wb = lb["outcome"] == win_outcome
        if wa and not wb:
            b += 1
        elif wb and not wa:
            c += 1
    if b + c == 0:
        return 1.0, 0, 0
    return binomial_pvalue(b, b + c, p0=0.5, alternative="two-sided"), b, c


def select_best_checkpoint(
    candidates: dict[str, object],
    eval_fn: Callable[[object], list[dict]],
    role: str,
) -> tuple[str, dict[str, dict]]:
    """Best-checkpoint selection (D14): evaluate every candidate on the SAME
    paired protocol (eval_fn must be seeded/deterministic) and pick the best
    win rate from `role`'s perspective — never the last checkpoint by default.

    Returns (best_name, {name: summary}) — the full table so the selection
    is auditable (CIs included; overlapping CIs mean the ranking is noisy —
    report, don't hide).
    """
    if not candidates:
        raise ValueError("no candidates")
    summaries = {}
    for name, candidate in candidates.items():
        logs = eval_fn(candidate)
        k, n = win_count(logs, role)
        summaries[name] = {**summarize(logs), "win_rate": k / n if n else float("nan")}
    best = max(summaries, key=lambda name: summaries[name]["win_rate"])
    return best, summaries


def summarize(logs: list[dict]) -> dict:
    """Aggregate a list of duel logs: CR + Wilson CI, episode length, LOS."""
    n = len(logs)
    captures = sum(log["outcome"] == "capture" for log in logs)
    lo, hi = wilson_ci(captures, n)
    return {
        "n": n,
        "captures": captures,
        "cr": captures / n if n else float("nan"),
        "cr_ci": (lo, hi),
        "mean_steps": float(np.mean([log["steps"] for log in logs])) if n else 0.0,
        "mean_los_breaks": float(np.mean([log["los_breaks"] for log in logs]))
        if n
        else 0.0,
        "frac_with_los_break": float(np.mean([log["los_breaks"] > 0 for log in logs]))
        if n
        else 0.0,
    }


def run_battery(
    cfg: EnvConfig,
    matchups: dict[str, tuple[Callable[[], object], Callable[[], object]]],
    n_episodes: int,
    seed0: int = BATTERY_SEED0,
) -> dict[str, list[dict]]:
    """Run each matchup for `n_episodes` seeded duels.

    `matchups` maps a name to (pursuer_factory, evader_factory); factories
    return fresh policy instances. Seeds follow D57: seed0 + i.
    """
    env = PursuitEvasionEnv(cfg)
    results = {}
    for name, (p_factory, e_factory) in matchups.items():
        results[name] = [
            run_duel(env, p_factory(), e_factory(), seed0 + i)
            for i in range(n_episodes)
        ]
    return results
