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

import itertools
from collections.abc import Callable

import numpy as np

from pe2.config import EnvConfig
from pe2.exploitability import SingleAgentWrapper


def seed_torch(seed: int) -> int | None:
    """Seed torch's GLOBAL generator and return the seed that was applied.

    `model.predict(deterministic=False)` samples the squashed Gaussian from
    torch's global generator, so the stochastic regime — which is the
    TRAINING-TIME regime the kill rule speaks about — is otherwise a
    measurement with no reproducible provenance. Two `signal_audit` calls
    on the same model at the same `seed0` returned visible fractions 0.0348
    and 0.0258 before this existed.

    A run could look reproducible anyway, by accident: `SAC.load` restores
    the `seed` stored in the zip and re-applies it through `_setup_model`,
    so a fresh process that loads a clone and audits it once lands on the
    same numbers. That seed belongs to whoever trained the clone, is not
    recorded in the audit's report, and stops governing the moment the
    regimes are reordered, a second audit runs in the same process, or the
    clone is retrained. The audit's own `seed0` must be the authority.

    Returns None only when torch is absent, in which case no SB3 policy can
    be running either and nothing stochastic needed seeding.
    """
    try:
        import torch  # lazy: optional dependency, as everywhere else in pe2
    except ImportError:
        return None
    torch.manual_seed(seed)
    return seed


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
    """Roll `model` (anything with .predict) until >= min_steps env steps.

    Seeds torch's global generator from `seed0` before the first rollout —
    once per call, i.e. once per regime — so the stochastic regime's number
    is reproducible from the seed the report records (see `seed_torch`).
    """
    torch_seed = seed_torch(seed0)
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
        # what was seeded, so the recorded seed is a claim a reader can check:
        # the env stream is seed0 + episode index, torch's global generator is
        # seed0 itself (None only if torch is not installed).
        "torch_seed": torch_seed,
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


def control_collapsed(
    rows: list[dict], threshold: float = CLONE_CI_LO, from_step: int = PASS_FROM_STEP
) -> tuple[bool | None, str]:
    """Did the negative control REPRODUCE the documented collapse?

    Returns `(verdict, reason)`, verdict in {True, False, None}. The whole
    justification for reading the other arms is that this control
    DISCRIMINATES, so `True` is a strong claim and is only made when it is
    earned. Failing `pilot_passes` is necessary but nowhere near sufficient:

    - the pass rule is "at or above the bound at EVERY probe", so a control
      that ends at 0.130 fails it while having very nearly preserved the
      clone (the measured spread across seeds is 0.010-0.280 — not contrived);
    - a run that never reaches `from_step` gives EVERY arm an empty window,
      so every arm "fails" and the control would certify a harness that
      judged zero probes.

    So three things are required for True: a non-empty judging window, a
    failed pass rule, AND a final win rate that is statistically SEPARATED
    from the clone — the final probe's Wilson upper bound below `threshold`.
    That is the same machinery `threshold` itself came from, and it is what
    distinguishes "collapsed" from "merely under the bound". Reference point,
    on record in `finetune_br`'s docstring: the naive fine-tune ran
    0.185 -> 0.060 @100k -> 0.020 @200k. At N=200 a final 0.020 has Wilson
    hi 0.050 (separated, True) while 0.130 has 0.184 (not separated, None).

    `None` means "cannot claim a collapse"; the reason says which case in
    words, so a reader never meets a bare null.
    """
    from pe2.eval import wilson_ci

    window = [r for r in rows if r["steps"] >= from_step]
    if not window:
        return None, (
            f"judging window empty: no probe at or after {from_step} steps, so "
            f"nothing was judged and the harness is unvalidated (probes seen: "
            f"{[r['steps'] for r in rows]})"
        )
    if pilot_passes(rows, threshold, from_step):
        return False, (
            f"the control held at or above {threshold} at every probe from "
            f"{from_step}: it did NOT collapse, so the harness is not measuring "
            f"what it claims"
        )
    final = max(window, key=lambda r: r["steps"])
    lo, hi = wilson_ci(final["k"], final["n"])
    if hi < threshold:
        return True, (
            f"collapse reproduced: final win rate {final['win_rate']:.3f} "
            f"({final['k']}/{final['n']} @{final['steps']}) has Wilson upper "
            f"bound {hi:.3f} < {threshold}, separated from the clone"
        )
    return None, (
        f"final win rate {final['win_rate']:.3f} ({final['k']}/{final['n']} "
        f"@{final['steps']}) is below {threshold} but not separated from the "
        f"clone (Wilson [{lo:.3f}, {hi:.3f}] reaches it): under the bound is "
        f"not the same as collapsed, so no collapse can be claimed"
    )


def start_is_adequate(
    k: int, n: int, threshold: float = CLONE_CI_LO
) -> tuple[bool | None, str]:
    """Can an arm that starts from its OWN clone be judged against `threshold`?

    Returns `(verdict, reason)`, verdict in {True, None}, in the same shape
    `control_collapsed` uses. P-A/P-B/CTRL all start from the GATED clone,
    whose 0.185 [0.137, 0.245] is where `threshold` came from. P-C starts
    from a freshly trained PPO clone of unknown quality and is judged
    against the same number. If the Wilson UPPER bound on that starting
    rate is already below `threshold`, the arm could not have cleared the
    bound however well the mechanism worked, so its failure says nothing
    about the horizon hypothesis and must not be reported as refuting it.

    `None` means "cannot be judged", never a bare null: the reason says so
    in words, and the caller forces the arm's verdict to undetermined.
    """
    from pe2.eval import wilson_ci

    lo, hi = wilson_ci(k, n)
    if hi < threshold:
        return None, (
            f"starting win rate {k}/{n} = {k / n:.3f} (Wilson [{lo:.3f}, "
            f"{hi:.3f}]) has an upper bound below {threshold}: this arm never "
            "had a competent starting point, so neither its pass nor its "
            "failure is evidence about the mechanism it tests"
        )
    return True, (
        f"starting win rate {k}/{n} = {k / n:.3f} (Wilson [{lo:.3f}, "
        f"{hi:.3f}]) reaches {threshold}: the start is consistent with "
        "clearing the bound, so the arm's verdict is about the mechanism"
    )


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

    `snapshot_every` records the cumulative counts every N steps. The
    aggregate rate alone BLENDS REGIMES: `finetune_br` freezes the actor for
    `critic_warmup` steps and then trains it, so one number averages the
    clone's own rollouts with whatever the policy became afterwards. The kill
    rule is meant to say "the reward signal is too sparse to learn from", but
    the aggregate also moves when the policy degrades — so a phase-B collapse
    could fire a rule that then gets read as a statement about the reward.
    `segments` differences the snapshots so the phases can be told apart.
    """

    def __init__(self, snapshot_every: int | None = None):
        from stable_baselines3.common.callbacks import BaseCallback  # lazy

        state = {"steps": 0, "captures": 0, "episodes": 0}
        snapshots: list[dict] = []
        next_at = [snapshot_every] if snapshot_every else []

        class _CB(BaseCallback):
            def _on_step(inner) -> bool:
                dones = inner.locals["dones"]
                infos = inner.locals["infos"]
                state["steps"] += len(dones)
                for done, info in zip(dones, infos):
                    if done:
                        state["episodes"] += 1
                        state["captures"] += int(info.get("outcome") == "capture")
                while next_at and state["steps"] >= next_at[0]:
                    snapshots.append(dict(state))
                    next_at[0] += snapshot_every
                return True

        self.state = state
        self.snapshots = snapshots
        self.callback = _CB()

    @property
    def positives_per_1M(self) -> float | None:
        """None, NOT 0.0, when nothing was collected: a rate off zero steps is
        undefined, and returning 0.0 would fire the kill rule ("the signal is
        too sparse to learn from") having observed no rollouts at all."""
        if self.state["steps"] == 0:
            return None
        return self.state["captures"] / self.state["steps"] * 1e6

    @property
    def segments(self) -> list[dict]:
        """Per-interval breakdown, differenced from the snapshots (plus the
        final state). Segments sum back to the aggregate exactly."""
        points = [{"steps": 0, "captures": 0, "episodes": 0}, *self.snapshots]
        if points[-1]["steps"] != self.state["steps"]:
            points.append(dict(self.state))
        out = []
        for a, b in itertools.pairwise(points):
            d = b["steps"] - a["steps"]
            captures = b["captures"] - a["captures"]
            out.append(
                {
                    "from_step": a["steps"],
                    "to_step": b["steps"],
                    "steps": d,
                    "episodes": b["episodes"] - a["episodes"],
                    "captures": captures,
                    "positives_per_1M": captures / d * 1e6 if d else None,
                }
            )
        return out
