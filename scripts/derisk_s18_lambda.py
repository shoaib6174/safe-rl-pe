"""S18 — the three calibrated weights, measured before any arm exists
(spec: docs/pathway/boundary_condition_prereg_2026-09-23.md section 3 + A1.2).

No policy is trained and nothing is selected here. One reference policy is
rolled out under the shield and three per-episode sums are measured, so that
the round's three weights are DERIVED from that measurement rather than chosen:

  lambda    the intervention penalty of arXiv:2410.11671, set so that the
            expected per-episode penalty of the reference policy equals 1.0 —
            the magnitude of the terminal reward. One lambda, shared by all
            three penalty cells, so the column comparison is not confounded by
            penalty strength (spec section 3).
  w_vis     DENSE-FLAT's weight, set so that the expected per-episode dense
            return equals 0.25.
  w_search  DENSE-VARY's weight, set so that the expected per-episode dense
            return equals 0.25 — the SAME 0.25.

A1.2 is why the two dense weights are calibrated at all: their historical
defaults (0.1 and 1e-4, D24/D4) differ by orders of magnitude in expected
per-episode return, so a difference between the columns would confound
gradient-on-`I` with reward scale. Calibrated to the same 0.25, the columns
differ in WHERE the reward has gradient and in nothing else.

Reference policy (A1.2): `RandomSearch` under the shield at the round's
`EnvConfig()` — the behaviour class of the untrained policy, and the only
reference available before any arm is trained. Section 3's "frozen BASE
policy" is superseded by this.

Stream (A1.1): 1 860 000+, disjoint from the construct gate (1 880 000+), the
training probes (1 950 000+) and the gate (1 900 000+).

What is summed, and over which steps. The learner is paid the SUM of env
rewards over its span, so the calibration measures exactly the steps the
learner is paid for:

  intervention  `SearchMDP.span_intervention` is reset at the top of every
                `step()` and reported as `info["intervention_magnitude"]`; the
                episode sum of those values IS the penalty the agent would
                receive, term for term. The law's opening span (played before
                the first decision) is deliberately NOT in it: it is not in the
                penalty either.
  vis, search   measured over the same windows — the per-env-step
                `info["dense_terms"]` of a `DenseRewardWrapper` at w = 1.0,
                accumulated between the clear at the top of one learner step
                and the end of that step. Both terms come from ONE pass: the
                wrapper reports them separately, so measuring them together
                cannot change either.

Usage:
  ./venv/bin/python scripts/derisk_s18_lambda.py
  ./venv/bin/python scripts/derisk_s18_lambda.py --out results/derisk

Stop rule (spec section 7): nothing here is re-run, extended or re-seeded after
a number is seen. An existing `s18_lambda.json` is NEVER overwritten silently —
the script refuses and says so.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import gymnasium as gym
import numpy as np

from pe2.config import EnvConfig
from pe2.rewards import DenseRewardWrapper
from pe2.scripted import GreedyEvader
from pe2.shield import CBFShield
from pe2.structured import RandomSearch, SearchMDP

OUT = pathlib.Path("results/derisk")
RESULT_NAME = "s18_lambda.json"

# --- pre-registered (spec section 3, amendment A1). Do not tune. ---
SEED0 = 1_860_000  # A1.1: the calibration stream
N_EPISODES = 200  # A1.2
HOLD = 10  # spec section 4
R_ACQ = 0.0  # the round's reward columns carry no acquisition bonus
TARGET_PENALTY = 1.0  # section 3: the magnitude of the terminal reward
TARGET_DENSE = 0.25  # A1.2: a quarter of it, in BOTH dense columns
PROBE_WEIGHT = 1.0  # the dense terms are measured at w = 1 and scaled after
POLICY_SEED = 0  # RandomSearch's own stream (as in derisk_s1_structured.py)
OMP_THREADS = "2"

# The three quantities, in the order they are reported everywhere below.
KEYS = ("intervention", "vis", "search")


def _write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def provenance() -> dict:
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    return {
        "git_commit": sha,
        "numpy_version": np.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


# ----------------------------------------------------------------------------
# the weights (spec section 3 + A1.2) — pure arithmetic on the measured means
# ----------------------------------------------------------------------------
def weights(means: dict) -> dict:
    """`{lambda, w_vis, w_search}` from the three measured per-episode means.

    REFUSES a non-positive or non-finite mean. A mean of zero means the term
    never fired over 200 episodes of the reference policy: there is no weight
    that makes its expected episode return 0.25, and a division would hand the
    round an infinity dressed up as a calibration.
    """
    for key in KEYS:
        m = float(means[key])
        if not np.isfinite(m) or m <= 0.0:
            raise ValueError(
                f"measured mean per-episode {key} is {m!r}; the calibration "
                "needs a strictly positive, finite mean (a zero means the term "
                "never fired over the reference rollout, and no weight can "
                "make its expected episode return the target)"
            )
    return {
        "lambda": TARGET_PENALTY / float(means["intervention"]),
        "w_vis": TARGET_DENSE / float(means["vis"]),
        "w_search": TARGET_DENSE / float(means["search"]),
    }


# ----------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------
class SpanTermRecorder(gym.Wrapper):
    """Accumulates the per-env-step dense terms of the wrapper beneath it.

    It is an OBSERVER: reward, observation, termination and info pass through
    untouched, so wrapping cannot change a step. `clear()` at the top of a
    learner step and the sums afterwards are that learner step's span — the
    same window `SearchMDP` resets `span_intervention` over, so the three
    measured quantities cover exactly the same env steps.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.clear()

    def clear(self) -> None:
        self.vis = 0.0
        self.search = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        terms = info.get("dense_terms", {})
        self.vis += float(terms.get("vis", 0.0))
        self.search += float(terms.get("search", 0.0))
        return obs, reward, terminated, truncated, info


def build_mdp(cfg: EnvConfig, seed0: int = SEED0) -> tuple[SearchMDP, SpanTermRecorder]:
    """The round's `SearchMDP` (spec section 4) with both dense terms measured
    at w = 1.0 in one pass, and the recorder that reads them.

    `intervention_penalty` stays 0: the reference rollout must be the reference
    policy's behaviour, not the behaviour of a policy already being penalised.
    (It could not change a trajectory anyway — the reward is discarded by the
    evaluation path — which is why one rollout can calibrate all three.)
    """
    box: dict = {}

    def factory(inner):
        box["rec"] = SpanTermRecorder(
            DenseRewardWrapper(inner, w_vis=PROBE_WEIGHT, w_search=PROBE_WEIGHT)
        )
        return box["rec"]

    mdp = SearchMDP(
        cfg,
        GreedyEvader,
        seed0=seed0,
        hold=HOLD,
        r_acq=R_ACQ,
        shield=CBFShield(cfg),
        shield_obs=True,
        escape_obs=True,
        reward_wrapper=factory,
    )
    return mdp, box["rec"]


def play_calibration_episode(
    mdp: SearchMDP, rec: SpanTermRecorder, predict, seed: int
) -> dict:
    """One seeded episode, with the three per-episode sums recorded.

    A step-for-step transcription of `SearchMDP.play_episode` (the round's
    evaluation path): `_begin`, the law's opening span, then one held decision
    per learner step until the episode ends. The only additions are the
    `rec.clear()` at the top of each learner step and the three accumulators —
    neither touches the dynamics, so the episode is the one `play_episode`
    would have produced for this seed.
    """
    obs = mdp._begin(seed)
    obs, _r, _o, term, trunc = mdp._advance(obs)
    obs = mdp._learner_obs(obs)
    sums = dict.fromkeys(KEYS, 0.0)
    n_decisions = 0
    while not (term or trunc):
        action, _ = predict(obs, deterministic=True)
        rec.clear()
        obs, _r, term, trunc, info = mdp.step(action)
        n_decisions += 1
        sums["intervention"] += float(info["intervention_magnitude"])
        sums["vis"] += rec.vis
        sums["search"] += rec.search
    return {
        "seed": seed,
        "outcome": mdp._info["outcome"],
        "steps": int(mdp.inner.env.current_step),
        "n_decisions": n_decisions,
        "violations": int(mdp.episode_violations),
        **sums,
    }


def measure(
    n_episodes: int = N_EPISODES, seed0: int = SEED0, cfg: EnvConfig | None = None
) -> list[dict]:
    """`n_episodes` seeded episodes of the reference policy from `seed0`."""
    cfg = EnvConfig() if cfg is None else cfg
    mdp, rec = build_mdp(cfg, seed0=seed0)
    policy = RandomSearch(POLICY_SEED)
    rows = []
    for i in range(n_episodes):
        row = play_calibration_episode(mdp, rec, policy.predict, seed0 + i)
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(
                f"  {i + 1}/{n_episodes} episodes "
                f"(intervention {np.mean([r['intervention'] for r in rows]):.4g}, "
                f"vis {np.mean([r['vis'] for r in rows]):.4g}, "
                f"search {np.mean([r['search'] for r in rows]):.4g})",
                flush=True,
            )
    return rows


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def build_record(rows: list[dict], seed0: int = SEED0) -> dict:
    """The artefact every arm of this round reads before it is allowed to run.

    `means` are over ALL episodes of the stream, which is what section 3 and
    A1.2 say ("200 episodes ... the mean per-episode sum"). Episodes the fixed
    law finishes before the learner is ever consulted contribute a sum of 0 to
    all three and are kept in that mean; they are counted separately under
    `diagnostics`, together with the means conditioned on at least one
    decision, so the effect of that reading is visible rather than argued.
    """
    means = {k: _mean(rows, k) for k in KEYS}
    decided = [r for r in rows if r["n_decisions"] > 0]
    return {
        **weights(means),
        "means": means,
        "n_episodes": len(rows),
        "seed0": seed0,
        "targets": {"penalty": TARGET_PENALTY, "dense": TARGET_DENSE},
        "reference": {
            "policy": "RandomSearch",
            "policy_seed": POLICY_SEED,
            "opponent": "GreedyEvader",
            "hold": HOLD,
            "r_acq": R_ACQ,
            "shield": "CBFShield",
            "shield_obs": True,
            "escape_obs": True,
            "probe_weight": PROBE_WEIGHT,
            "config": "EnvConfig()",
        },
        "diagnostics": {
            "n_zero_decision_episodes": len(rows) - len(decided),
            "means_given_a_decision": {k: _mean(decided, k) for k in KEYS},
            "mean_decisions_per_episode": _mean(rows, "n_decisions"),
            "mean_steps": _mean(rows, "steps"),
            "capture_rate": (
                float(np.mean([r["outcome"] == "capture" for r in rows]))
                if rows
                else float("nan")
            ),
            "violation_steps": int(sum(r["violations"] for r in rows)),
        },
        "per_episode": rows,
        "provenance": provenance(),
    }


def print_record(rec: dict) -> None:
    m, d = rec["means"], rec["diagnostics"]
    print(
        f"\nS18 calibration — {rec['n_episodes']} episodes, seeds {rec['seed0']}+, "
        f"{rec['reference']['policy']} under {rec['reference']['shield']}\n"
    )
    print(f"{'quantity':<14}{'mean/episode':>16}{'target':>10}{'weight':>16}")
    for key, weight, target in (
        ("intervention", "lambda", rec["targets"]["penalty"]),
        ("vis", "w_vis", rec["targets"]["dense"]),
        ("search", "w_search", rec["targets"]["dense"]),
    ):
        print(f"{key:<14}{m[key]:>16.6g}{target:>10}{rec[weight]:>16.6g}")
    print(
        f"\nzero-decision episodes {d['n_zero_decision_episodes']}/"
        f"{rec['n_episodes']}, decisions/episode {d['mean_decisions_per_episode']:.1f}, "
        f"capture {d['capture_rate']:.3f}, violation steps {d['violation_steps']}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--seed0", type=int, default=SEED0)
    args = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)
    path = pathlib.Path(args.out) / RESULT_NAME
    if path.exists():
        raise SystemExit(
            f"{path} already exists; the stop rule (spec section 7) forbids "
            "re-running a calibration whose numbers have been seen. Move the "
            "file aside deliberately if it is genuinely to be redone."
        )
    print(
        f"S18 calibration: {args.episodes} episodes from seed {args.seed0} "
        f"(spec section 3 + A1.2)",
        flush=True,
    )
    rec = build_record(measure(args.episodes, args.seed0), args.seed0)
    _write_json(path, rec)
    print_record(rec)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
