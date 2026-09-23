"""S18 — the BLOCKING construct gate, run before any outcome is read
(spec: docs/pathway/boundary_condition_prereg_2026-09-23.md section 6).

The retrospective's §4 rule: four of our five collapses would have been caught
by a gate like this one. Nothing here trains, and nothing here reads a capture
rate that belongs to an arm. Three checks:

  CG1  the column labels are true. At 150 intervened states drawn from real
       episodes, the VARIANCE of the per-step task reward across the admissible
       action set, per column. Required: SPARSE ~ 0, DENSE-RARE ~ 0,
       DENSE-VARY > 0. If DENSE-RARE is not flat on `I`, the column is
       mislabelled and the columns are renamed before any capture rate is
       looked at.
  CG2  the primary metric is not satisfiable by inaction. A policy that always
       commands (v = 0, omega = 0) must not obtain the capture rate.
  CG3  the signal exists. `pe2.rewards.signal_nondegeneracy` on each reward
       variant's rollout stream.

CG4 (behavioural: render and watch 3 unselected episodes per cell) is not this
script's.

The admissible action set is the shield's OWN candidate enumeration
(`pe2.shield._qp_candidates`) judged by the shield's OWN exact one-step test —
the same two steps `CBFShield.filter` takes to reach its exact tier. D126
settled that the loop is not an artefact of our discretisation, and inventing a
second one here would put that back in play.

Choices this script had to make because neither section 6 nor the task brief
fixes them. Stated here rather than buried:
  - The rollout policy is `RandomSearch` under the shield, the reference policy
    amendment A1.2 names for the calibrations: before any arm is trained it is
    the only policy the round has.
  - The opponent is `GreedyEvader`, the task benchmark of spec section 4.
  - "150 intervened states sampled from real episodes" is read as ONE state per
    episode, drawn uniformly at random (a seeded RNG) from that episode's
    intervened states, over 150 consecutive episodes of the stream. Taking the
    first 150 states in stream order would have drawn nearly all of them from a
    single episode — interventions run to the hundreds per episode — and a mean
    variance over one episode's pocket is not a measurement of the column.
  - CG3 runs `signal_nondegeneracy`, which returns descriptive statistics and
    no verdict, and section 6 names no threshold for it. It is therefore
    REPORTED, not gated; `pe2.rewards.nondegeneracy_report` carries the T0.2
    per-term criteria and is deliberately NOT substituted for the function the
    spec names.

Usage:
  ./venv/bin/python scripts/derisk_s18_construct.py

Exits non-zero when a gate fails, after writing the record: a crash would lose
the evidence the gate exists to produce.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import gymnasium as gym
import numpy as np

from pe2.config import EnvConfig
from pe2.env import pursuer_step
from pe2.obs import build_observation
from pe2.rewards import SearchStalenessTracker, signal_nondegeneracy, visibility_term
from pe2.scripted import GreedyEvader
from pe2.shield import CBFShield, _qp_candidates
from pe2.structured import RandomSearch, SearchMDP, _normalize

OUT = pathlib.Path("results/derisk")
LAMBDA_NAME = "s18_lambda.json"
RESULT_NAME = "s18_construct_gate.json"

# --- pre-registered (spec section 6, amendment A1.1). Do not tune. ---
SEED0 = 1_880_000  # A1.1: the construct-gate stream
CG1_N_STATES = 150  # section 6
CG2_N_EPISODES = 50  # task brief
HOLD = 10  # spec section 4
R_ACQ = 0.0
OMP_THREADS = "2"

# The only tolerance in this file, and it is the brief's, not one of mine:
# DENSE-RARE is NOT FLAT when its mean variance exceeds SPARSE's by more than
# this. SPARSE's is 0 by construction, so this is "DENSE-RARE varies at all".
FLAT_ATOL = 1e-9
# Sampling RNG for CG1's one-state-per-episode draw (see the module docstring).
CG1_RNG_SEED = 0

COLUMNS = ("SPARSE", "DRARE", "DVARY")
COLUMN_LABEL = {"SPARSE": "SPARSE", "DRARE": "DENSE-RARE", "DVARY": "DENSE-VARY"}


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


def load_weights(out: pathlib.Path = OUT) -> dict:
    """The calibrated weights. REFUSES to run without them (spec section 3:
    "An arm that cannot read that file refuses to run"); the gate measures the
    columns as they will be trained, which means at their calibrated weights."""
    path = pathlib.Path(out) / LAMBDA_NAME
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. The construct gate measures the reward "
            "columns at their CALIBRATED weights (spec section 3 + A1.2), so "
            "it cannot run before scripts/derisk_s18_lambda.py has written "
            "them."
        )
    rec = json.loads(path.read_text())
    missing = [k for k in ("lambda", "w_vis", "w_search") if k not in rec]
    if missing:
        raise SystemExit(f"{path} is missing {missing}: it is not a calibration.")
    return rec


# ----------------------------------------------------------------------------
# the admissible action set (spec section 6, CG1)
# ----------------------------------------------------------------------------
class NominalTap(CBFShield):
    """`CBFShield` that remembers what it was last asked to filter.

    A subclass, not a wrapper, so it IS the round's filter: `escape_features`
    and `SearchMDP` reach into a shield's attributes and methods, and a
    duck-typed stand-in would have to reproduce them. `last` carries the
    BOX-CLIPPED nominal — the point the candidate set is built around, which is
    what CG1 needs — beside the step's info dict, because the info dict reports
    `v_nom` as a scalar speed and carries no omega at all.
    """

    def __init__(self, cfg: EnvConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        self.last: dict | None = None

    def filter(self, action, state, obstacles, v_max=None):
        u, info = super().filter(action, state, obstacles, v_max)
        limit = self.cfg.v_max if v_max is None else v_max
        v_nom, w_nom = self._clip_nominal(action, limit)
        self.last = {
            "nominal": (v_nom, w_nom),
            "intervention": bool(info["intervention"]),
            "method": info["method"],
            "v_max": float(limit),
        }
        return u, info


def admissible_actions(
    shield: CBFShield,
    nominal: tuple[float, float],
    state: np.ndarray,
    obstacles: list[dict],
    v_max: float | None = None,
) -> np.ndarray:
    """Every action `CBFShield.filter` would accept at `state` on its exact
    tier, deduplicated, first occurrence first.

    Both halves are the filter's own: `_qp_candidates` locates the actions
    (D85's enumeration plus u = 0 and the nominal), and the exact one-step
    barrier test on the true Euler successor decides which are admissible —
    the same `H1 >= (1 - gamma) * H0` comparison, with no tolerance, that
    `filter` uses to select its exact tier. `_qp_candidates` returns ~61 points
    of which ~13 are distinct; the duplicates would weight a variance over the
    set by an artefact of the enumeration, so they are dropped.

    An empty return means no candidate satisfied every row: the filter fell to
    its relaxed or backup tier at this state, where "the admissible set" is
    empty by definition rather than small.
    """
    v_max = shield.cfg.v_max if v_max is None else v_max
    v_nom, w_nom = shield._clip_nominal(nominal, v_max)
    cons = shield.constraints(state, obstacles)
    U = _qp_candidates(
        v_nom,
        w_nom,
        shield._dcbf_rows(cons),
        v_max,
        shield.cfg.omega_max,
        shield.w_v,
        shield.w_omega,
    )
    _uniq, first = np.unique(U, axis=0, return_index=True)
    U = U[np.sort(first)]
    H1 = shield.successor_barriers(state, obstacles, U)
    target = (1.0 - shield.gamma) * np.array([c.h for c in cons])
    return U[(H1 >= target).all(axis=1)]


# ----------------------------------------------------------------------------
# the per-step task reward of each column, as a function of the action (CG1)
# ----------------------------------------------------------------------------
def column_values(
    st: dict,
    U: np.ndarray,
    cfg: EnvConfig,
    w_vis: float,
    w_search: float,
    tracker: SearchStalenessTracker,
) -> dict[str, list[float]]:
    """The per-step task reward of each column at `st` under every action in `U`.

    The counterfactual varies the PURSUER'S action and nothing else. The
    evader's command is chosen from the joint observation before the pursuer
    acts, so its successor pose is common to every candidate and `st` carries
    the one the episode actually produced.

    SPARSE is the terminal +/-1 alone: it pays nothing at a non-terminal step,
    identically, for every action. That is by construction and is recorded as a
    measured 0 so the gate's own table shows it rather than asserting it.

    DENSE-RARE and DENSE-VARY are evaluated through the same two functions
    `DenseRewardWrapper` pays — `visibility_term` on the successor observation,
    and `SearchStalenessTracker.observe_and_reward` from the staleness state
    `st` recorded before the step — so the gate cannot pass on a second copy of
    a reward that trains differently. `tracker` is a scratch instance whose
    `last_observed` is overwritten per candidate; it must not be the one
    driving a rollout.
    """
    vals: dict[str, list[float]] = {c: [] for c in COLUMNS}
    for v, omega in np.asarray(U, dtype=float):
        pose, _wall, _obs = pursuer_step(
            st["pursuer_pose"],
            float(v),
            float(omega),
            cfg,
            st["obstacles"],
            st["x_lim"],
            st["y_lim"],
        )
        vals["SPARSE"].append(0.0)
        vals["DRARE"].append(
            visibility_term(
                w_vis, build_observation(pose, st["evader_pose"], st["obstacles"], cfg)
            )
        )
        tracker.last_observed[:] = st["staleness"]
        vals["DVARY"].append(
            w_search
            * tracker.observe_and_reward(pose, st["obstacles"], st["current_step"] + 1)
        )
    return vals


def state_variances(
    st: dict,
    shield: CBFShield,
    cfg: EnvConfig,
    w_vis: float,
    w_search: float,
    tracker: SearchStalenessTracker,
) -> dict:
    """CG1 at one intervened state: the admissible set, and the variance of
    each column's per-step task reward across it."""
    U = admissible_actions(
        shield, st["nominal"], st["pursuer_pose"], st["obstacles"], st["v_max"]
    )
    head = {
        "state": st["id"],
        "episode": st["episode"],
        "step": st["current_step"],
        "method": st["method"],
        "n_admissible": len(U),
    }
    if not len(U):
        return {**head, "variance": None, "mean": None}
    vals = column_values(st, U, cfg, w_vis, w_search, tracker)
    return {
        **head,
        "variance": {c: float(np.var(vals[c])) for c in COLUMNS},
        "mean": {c: float(np.mean(vals[c])) for c in COLUMNS},
    }


def cg1_verdict(per_state: list[dict]) -> dict:
    """SPARSE ~ 0, DENSE-RARE ~ 0, DENSE-VARY > 0 (spec section 6).

    The only tolerance is the brief's: DENSE-RARE is NOT FLAT when its mean
    variance exceeds SPARSE's by more than `FLAT_ATOL`. SPARSE's is 0 by
    construction, so that is "DENSE-RARE varies at all". DENSE-VARY's criterion
    is the spec's own words, `> 0`, and no bar is invented beneath it.

    A state where the filter fell below its exact tier has an EMPTY admissible
    set and contributes no variance; it is counted, not silently dropped.
    """
    scored = [s for s in per_state if s["variance"] is not None]
    if not scored:
        return {
            "verdict": "INCOMPLETE",
            "n_states": len(per_state),
            "n_scored": 0,
            "reason": "no state had a non-empty admissible action set",
        }
    means = {c: float(np.mean([s["variance"][c] for s in scored])) for c in COLUMNS}
    excess = {c: means[c] - means["SPARSE"] for c in COLUMNS}
    # How many states the column varies at ALL, beside the mean of the variance
    # over them. A mean can be lifted off zero by one state in a hundred, and
    # "flat at 146 of 148 intervened states" and "slightly non-flat at every
    # one" are different findings that section 6's rename decision turns on.
    n_varying = {c: int(sum(s["variance"][c] > 0.0 for s in scored)) for c in COLUMNS}
    per_column = {
        "SPARSE": {
            "mean_variance": means["SPARSE"],
            "n_states_varying": n_varying["SPARSE"],
            "requirement": "~ 0 (terminal-only: constant across actions)",
            "pass": bool(means["SPARSE"] == 0.0),
        },
        "DRARE": {
            "mean_variance": means["DRARE"],
            "n_states_varying": n_varying["DRARE"],
            "excess_over_sparse": excess["DRARE"],
            "requirement": f"~ 0: excess over SPARSE <= {FLAT_ATOL}",
            "pass": bool(excess["DRARE"] <= FLAT_ATOL),
            "label": "FLAT" if excess["DRARE"] <= FLAT_ATOL else "NOT-FLAT",
        },
        "DVARY": {
            "mean_variance": means["DVARY"],
            "n_states_varying": n_varying["DVARY"],
            "requirement": "> 0",
            "pass": bool(means["DVARY"] > 0.0),
        },
    }
    ok = all(b["pass"] for b in per_column.values())
    return {
        "verdict": "PASS" if ok else "FAIL",
        "n_states": len(per_state),
        "n_scored": len(scored),
        "n_episodes_spanned": len({s["episode"] for s in scored}),
        "mean_n_admissible": float(np.mean([s["n_admissible"] for s in scored])),
        "mean_variance": means,
        "per_column": per_column,
        "note": (
            "DENSE-RARE not flat on I means the column is mislabelled: spec "
            "section 6 requires the columns to be renamed before any capture "
            "rate is looked at."
        ),
    }


# ----------------------------------------------------------------------------
# rollouts
# ----------------------------------------------------------------------------
class RolloutCollector(gym.Wrapper):
    """Records what CG1 and CG3 read, without changing a step.

    An OBSERVER on the inner `SingleAgentWrapper`: reward, observation,
    termination and info pass through untouched (A1.3's property is not put at
    risk to measure it). Per env step it

      - advances its own `SearchStalenessTracker` exactly as
        `DenseRewardWrapper` does (post-step pose, post-step step index), and
      - appends the three columns' per-step task reward to `streams` (CG3), and
      - when the filter intervened, records the PRE-step state CG1 needs: the
        pursuer's pose, the nominal the filter was given, the staleness state
        as it stood before the step, and the evader's resulting pose.

    `tap` is the shield that supplies the nominal; the collector reads its
    `last` before stepping, because `SearchMDP._step_inner` filters and then
    steps, so at this point `tap.last` is this step's filter call.
    """

    def __init__(
        self,
        env: gym.Env,
        cfg: EnvConfig,
        tap: NominalTap,
        w_vis: float,
        w_search: float,
    ):
        super().__init__(env)
        self.cfg = cfg
        self.tap = tap
        self.w_vis = w_vis
        self.w_search = w_search
        self.tracker = SearchStalenessTracker(cfg)
        self.streams: dict[str, list[float]] = {c: [] for c in COLUMNS}
        self.episode = -1
        self.episode_states: list[dict] = []
        self._next_id = 0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.tracker.reset()
        self.episode += 1
        self.episode_states = []
        return out

    def step(self, action):
        sim = self.env.env
        pre = {
            "pursuer_pose": sim.pursuer_pose.copy(),
            "current_step": int(sim.current_step),
            "staleness": self.tracker.last_observed.copy(),
            "obstacles": [dict(o) for o in sim.obstacles],
            "x_lim": (sim._x_min, sim._x_max),
            "y_lim": (sim._y_min, sim._y_max),
        }
        tapped = self.tap.last
        obs, reward, terminated, truncated, info = self.env.step(action)
        search = self.tracker.observe_and_reward(
            sim.pursuer_pose, sim.obstacles, sim.current_step
        )
        self.streams["SPARSE"].append(float(reward))
        self.streams["DRARE"].append(
            visibility_term(self.w_vis, self.env._last_joint_obs["pursuer"])
        )
        self.streams["DVARY"].append(self.w_search * search)
        if tapped is not None and tapped["intervention"]:
            self.episode_states.append(
                {
                    **pre,
                    "id": self._next_id,
                    "episode": self.episode,
                    "evader_pose": sim.evader_pose.copy(),
                    "nominal": tapped["nominal"],
                    "v_max": tapped["v_max"],
                    "method": tapped["method"],
                }
            )
            self._next_id += 1
        return obs, reward, terminated, truncated, info


class ZeroMotion:
    """CG2's policy: every decision is (v = 0, omega = 0), in the normalised
    action space, through the same inverse `pe2.structured` trains with."""

    def __init__(self, cfg: EnvConfig):
        self.action = _normalize(0.0, 0.0, cfg)

    def predict(self, obs, deterministic: bool = True):
        return self.action, None


def build_mdp(cfg: EnvConfig, shield, seed0: int, reward_wrapper=None) -> SearchMDP:
    """The round's `SearchMDP` (spec section 4)."""
    return SearchMDP(
        cfg,
        GreedyEvader,
        seed0=seed0,
        hold=HOLD,
        r_acq=R_ACQ,
        shield=shield,
        shield_obs=True,
        escape_obs=True,
        reward_wrapper=reward_wrapper,
    )


def collect(
    cfg: EnvConfig,
    w_vis: float,
    w_search: float,
    n_states: int = CG1_N_STATES,
    seed0: int = SEED0,
) -> tuple[list[dict], dict[str, list[float]], NominalTap]:
    """`n_states` intervened states — one per episode, drawn uniformly at
    random from that episode's intervened states — over `n_states` consecutive
    episodes of the stream, plus the three CG3 reward streams over all of them.

    An episode in which the filter never intervened contributes no state, and
    the stream is walked further until `n_states` are collected."""
    tap = NominalTap(cfg)
    box: dict = {}

    def factory(inner):
        box["rec"] = RolloutCollector(inner, cfg, tap, w_vis, w_search)
        return box["rec"]

    mdp = build_mdp(cfg, tap, seed0, reward_wrapper=factory)
    rec: RolloutCollector = box["rec"]
    policy = RandomSearch(0)
    rng = np.random.default_rng(CG1_RNG_SEED)
    states: list[dict] = []
    episode = 0
    while len(states) < n_states:
        mdp.play_episode(policy.predict, seed0 + episode)
        episode += 1
        if rec.episode_states:
            states.append(rec.episode_states[rng.integers(len(rec.episode_states))])
        if episode % 25 == 0:
            print(
                f"  {episode} episodes, {len(states)}/{n_states} states",
                flush=True,
            )
    return states, rec.streams, tap


def cg1(
    cfg: EnvConfig, states: list[dict], shield: CBFShield, w_vis: float, w_search: float
) -> tuple[dict, list[dict]]:
    tracker = SearchStalenessTracker(cfg)  # scratch: never drives a rollout
    per_state = [
        state_variances(st, shield, cfg, w_vis, w_search, tracker) for st in states
    ]
    return cg1_verdict(per_state), per_state


def cg2_rollout(
    cfg: EnvConfig, n_episodes: int = CG2_N_EPISODES, seed0: int = SEED0
) -> list[dict]:
    """`n_episodes` of the round's `SearchMDP` driven by `ZeroMotion`."""
    mdp = build_mdp(cfg, CBFShield(cfg), seed0)
    policy = ZeroMotion(cfg)
    return [mdp.play_episode(policy.predict, seed0 + i) for i in range(n_episodes)]


def cg2_verdict(logs: list[dict]) -> dict:
    """The primary metric (capture rate) must not be obtainable by inaction.

    The mechanism metrics of the same rollout are recorded beside it, report
    only, as the other half of section 6's CG2: a mechanism metric that a
    do-nothing policy satisfies is the D129 error, and these are the numbers
    that show whether one does.
    """
    n = len(logs)
    capture = (
        float(np.mean([log["outcome"] == "capture" for log in logs])) if n else 1.0
    )
    return {
        "verdict": "PASS" if capture == 0.0 else "FAIL",
        "requirement": "capture rate == 0 for a policy that never moves",
        "capture_rate": capture,
        "n_captures": int(sum(log["outcome"] == "capture" for log in logs)),
        "n_episodes": n,
        "policy": "ZeroMotion (v = 0, omega = 0 at every decision)",
        "mechanism_metrics_report_only": {
            "deadlock_rate": (
                float(np.mean([bool(log["shield"]["deadlock"]) for log in logs]))
                if n
                else float("nan")
            ),
            "intervention_rate": (
                float(np.mean([log["shield"]["intervention_rate"] for log in logs]))
                if n
                else float("nan")
            ),
            "immobile_step_rate": (
                float(np.mean([log["shield"]["immobile_step_rate"] for log in logs]))
                if n
                else float("nan")
            ),
            "violation_steps": int(sum(log["violations"] for log in logs)),
            "mean_steps": float(np.mean([log["steps"] for log in logs])) if n else 0.0,
        },
        "note": (
            "The fixed intercept law captures whenever it has something to act "
            "on, so this capture rate is what the LAW obtains while the search "
            "policy does nothing — which is precisely the quantity section 6 "
            "asks about: how much of the primary metric is available without "
            "searching."
        ),
    }


def cg3_report(streams: dict[str, list[float]], weights_: dict) -> dict:
    """`signal_nondegeneracy` per column (spec section 6, CG3).

    REPORTED, not gated: the function returns descriptive statistics and no
    verdict, and section 6 names no threshold for it. `nondegeneracy_report`
    carries the T0.2 per-term criteria and is deliberately not substituted for
    the function the spec names.
    """
    return {
        "verdict": "REPORTED (no threshold is pre-registered for CG3)",
        "n_steps": len(streams["SPARSE"]),
        "weights": {"w_vis": weights_["w_vis"], "w_search": weights_["w_search"]},
        "columns": {c: signal_nondegeneracy(streams[c]) for c in COLUMNS},
    }


# ----------------------------------------------------------------------------
# artefact
# ----------------------------------------------------------------------------
def build_record(
    cg1_block: dict,
    per_state: list[dict],
    cg2_block: dict,
    cg3_block: dict,
    weights_: dict,
    seed0: int = SEED0,
) -> dict:
    blocking = ("CG1", "CG2")
    verdicts = {"CG1": cg1_block["verdict"], "CG2": cg2_block["verdict"]}
    return {
        "gate": "S18 construct gate (spec section 6); CG4 is rendered separately",
        "verdict": ("PASS" if all(verdicts[k] == "PASS" for k in blocking) else "FAIL"),
        "blocking": list(blocking),
        "seed0": seed0,
        "weights": {k: weights_[k] for k in ("lambda", "w_vis", "w_search")},
        "CG1": cg1_block,
        "CG2": cg2_block,
        "CG3": cg3_block,
        "cg1_per_state": per_state,
        "provenance": provenance(),
    }


def print_record(rec: dict) -> None:
    print(f"\nS18 construct gate — seeds {rec['seed0']}+\n")
    c1 = rec["CG1"]
    print(f"CG1  the column labels are true                       {c1['verdict']}")
    if "per_column" in c1:
        print(
            f"     {'column':<12}{'mean variance on I':>22}{'varies at':>12}"
            f"{'pass':>7}  requirement"
        )
        for col in COLUMNS:
            b = c1["per_column"][col]
            varies = f"{b['n_states_varying']}/{c1['n_scored']}"
            print(
                f"     {COLUMN_LABEL[col]:<12}{b['mean_variance']:>22.6g}{varies:>12}"
                f"{'yes' if b['pass'] else 'NO':>7}  {b['requirement']}"
            )
        print(
            f"     {c1['n_scored']}/{c1['n_states']} states scored over "
            f"{c1['n_episodes_spanned']} episodes, "
            f"{c1['mean_n_admissible']:.1f} admissible actions each"
        )
    c2 = rec["CG2"]
    print(
        f"\nCG2  the metric is not satisfiable by inaction        {c2['verdict']}\n"
        f"     capture rate {c2['capture_rate']:.3f} "
        f"({c2['n_captures']}/{c2['n_episodes']}) of {c2['policy']}"
    )
    m = c2["mechanism_metrics_report_only"]
    print(
        f"     report only: deadlock {m['deadlock_rate']:.3f}, intervention "
        f"{m['intervention_rate']:.3f}, immobile {m['immobile_step_rate']:.3f}, "
        f"violations {m['violation_steps']}"
    )
    c3 = rec["CG3"]
    print(f"\nCG3  the signal exists                                {c3['verdict']}")
    print(f"     {'column':<12}{'mean':>14}{'std':>14}{'frac>0':>10}{'frac modal':>12}")
    for col in COLUMNS:
        s = c3["columns"][col]
        print(
            f"     {COLUMN_LABEL[col]:<12}{s['mean']:>14.6g}{s['std']:>14.6g}"
            f"{s['frac_positive']:>10.3f}{s['frac_modal']:>12.3f}"
        )
    print(f"\nGATE: {rec['verdict']}  (blocking: {', '.join(rec['blocking'])})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--states", type=int, default=CG1_N_STATES)
    p.add_argument("--cg2-episodes", type=int, default=CG2_N_EPISODES)
    p.add_argument("--seed0", type=int, default=SEED0)
    args = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)
    out = pathlib.Path(args.out)
    w = load_weights(out)
    cfg = EnvConfig()
    print(
        f"S18 construct gate: w_vis {w['w_vis']:.6g}, w_search {w['w_search']:.6g} "
        f"(lambda {w['lambda']:.6g}), seeds {args.seed0}+",
        flush=True,
    )
    print(f"CG1/CG3 rollout: collecting {args.states} intervened states", flush=True)
    states, streams, _tap = collect(
        cfg, w["w_vis"], w["w_search"], args.states, args.seed0
    )
    print("CG1: enumerating the admissible set at each state", flush=True)
    cg1_block, per_state = cg1(cfg, states, CBFShield(cfg), w["w_vis"], w["w_search"])
    print(f"CG2: {args.cg2_episodes} zero-motion episodes", flush=True)
    cg2_block = cg2_verdict(cg2_rollout(cfg, args.cg2_episodes, args.seed0))
    cg3_block = cg3_report(streams, w)
    rec = build_record(cg1_block, per_state, cg2_block, cg3_block, w, args.seed0)
    _write_json(out / RESULT_NAME, rec)
    print_record(rec)
    print(f"\nwrote {out / RESULT_NAME}")
    if rec["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
