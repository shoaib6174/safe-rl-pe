"""pe2.recorder — the branch-tagged gate recorder (shared instrument, S19+).

Plan: `docs/pathway/agent_intelligence_plan_2026-10-05.md` section 1.

The structured pursuer is a COMPOSITE. `InterceptLaw` drives whenever it has
something to act on (branch `"law"`: the evader is visible, or a last-seen
point is still fresh); the learned search policy drives otherwise (branch
`"search"`). S18's gate records cannot tell the two apart: its behavioural
gate watched episodes where the LAW drove into a wall and sat pinned at
clearance 0.000 for ~1100 steps, and the record attributed that to "the
pursuer". Every S19+ claim about "the law" versus "the learner" — and the
paper's Case 5 — needs per-step attribution, so it is built once, here.

Nothing in this module touches the dynamics. It overrides `_step_inner` to
read what the base class already computed and `step`/`_begin`/`play_episode`
to scope the record to one episode; the episode a `BranchTaggedSearchMDP`
plays is bit-identical to the one plain `SearchMDP` plays from the same seed
(`tests/pe2/test_recorder.py`).

EVALUATION ONLY, for the same reason as `derisk_s18_boundary.RecordingSearchMDP`:
`play_episode` calls `_begin` exactly once per episode, whereas `reset()`
retries `_begin` in a loop when the scripted law finishes an episode before any
decision. The per-episode record is therefore the record of the LAST `_begin`,
which under `play_episode` is the episode in the log. Training uses plain
`SearchMDP`.

Two intervention totals, and the difference between them matters
-----------------------------------------------------------------
`SearchMDP.span_intervention` is zeroed at the top of every `step()` call, so
`info["intervention_magnitude"]` — the quantity S18 recorded, summed over
`step()` calls — covers only the env steps at or after the FIRST learner
decision. The law steps `play_episode` plays before that first decision (and
the whole of an episode the law finishes on its own) are not in it. This module
records EVERY env step of the episode and reports both:

- `intervention_total` — all steps, and the sum the branch split reconciles to;
- `span_intervention_total` — accumulated exactly as S18 accumulated it
  (`+= info["intervention_magnitude"]` per `step()` call), so an S18 number can
  be reproduced bit-for-bit;
- `pre_decision_intervention` — the difference, i.e. what the S18 metric could
  not see. It is not a correction to S18; it is the part of the episode that
  metric was never defined over.
"""

from __future__ import annotations

import math

import numpy as np

from pe2.structured import SearchMDP

BRANCHES = ("law", "search")

# "Pinned" is a RUN, not a step: `pinned_runs` keeps every immobile run at
# least this long. 10 steps = 0.5 s at dt = 0.05 -- short enough that a run
# broken into fragments by a branch change is still visible, far below
# `DEADLOCK_STEPS` (40) so the record is not pre-filtered by the deadlock
# threshold it will be read against.
MIN_PINNED_RUN = 10

# Shield fields on a step taken with no shield. `min_clearance` is 0.0 rather
# than `shield_features`' "nothing acted" default of 2.0: these are the RECORD
# of a step, not an observation, and a clearance that was never measured must
# not read as a comfortable one.
_NO_SHIELD = {
    "intervention_magnitude": 0.0,
    "stalled": False,
    "immobile": False,
    "intervention": False,
    "min_clearance": 0.0,
}


def _runs(steps: list[dict], split_on_branch: bool) -> list[dict]:
    """Maximal runs of consecutive `immobile` steps, as
    `{"branch", "start", "length"}`. `start` is the 0-based index of the run's
    first step within the episode.

    With `split_on_branch` a branch change ENDS the run (the law being pinned
    for 30 steps and the learner for 30 more is not one 60-step law pin);
    without it the run spans branches and `branch` is the branch at its start.
    """
    out: list[dict] = []
    cur: dict | None = None
    for i, s in enumerate(steps):
        if not s["immobile"]:
            cur = None
            continue
        if cur is not None and (not split_on_branch or cur["branch"] == s["branch"]):
            cur["length"] += 1
            continue
        cur = {"branch": s["branch"], "start": i, "length": 1}
        out.append(cur)
    return out


def branch_summary(steps: list[dict], span_intervention_total: float) -> dict:
    """The per-episode summary from a list of per-step records.

    A free function so the summary can be recomputed from a saved step record
    (and tested) without an env.
    """
    by_branch = {b: [s for s in steps if s["branch"] == b] for b in BRANCHES}
    split = _runs(steps, split_on_branch=True)
    whole = _runs(steps, split_on_branch=False)
    longest = max(whole, key=lambda r: r["length"]) if whole else None
    total = math.fsum(s["intervention_magnitude"] for s in steps)
    return {
        "n_steps": len(steps),
        "intervention_by_branch": {
            b: math.fsum(s["intervention_magnitude"] for s in by_branch[b])
            for b in BRANCHES
        },
        "steps_by_branch": {b: len(by_branch[b]) for b in BRANCHES},
        "stalled_by_branch": {
            b: sum(int(s["stalled"]) for s in by_branch[b]) for b in BRANCHES
        },
        "immobile_by_branch": {
            b: sum(int(s["immobile"]) for s in by_branch[b]) for b in BRANCHES
        },
        "longest_pinned_run_by_branch": {
            b: max((r["length"] for r in split if r["branch"] == b), default=0)
            for b in BRANCHES
        },
        "longest_pinned_run": longest["length"] if longest else 0,
        "branch_at_start": longest["branch"] if longest else None,
        "pinned_runs": [r for r in split if r["length"] >= MIN_PINNED_RUN],
        "intervention_total": total,
        "span_intervention_total": float(span_intervention_total),
        "pre_decision_intervention": total - float(span_intervention_total),
    }


class BranchTaggedSearchMDP(SearchMDP):
    """`SearchMDP` that records, per env step, which component commanded it and
    what the filter did to that command.

    Per step (`self.step_records`, reset at every `_begin`):
    `branch`, `nominal` / `executed` normalised action, `intervention_magnitude`,
    `stalled`, `immobile`, `intervention`, `min_clearance`, and `in_span`
    (whether the step falls inside a learner decision's span, i.e. at or after
    the first decision -- the scope of S18's `intervention_magnitude`).

    `intervention_magnitude` is NOT a second copy of `||a_nominal - a_safe||_2`
    computed here: it is the INCREMENT of `SearchMDP.span_intervention` across
    the step, so the split cannot drift from the quantity the base class
    accumulates and the penalty is built from.

    `self.trace` is owned by this class (it is `None` -- off -- in `SearchMDP`),
    because the base class records the EXECUTED normalised command there and
    nowhere else. It is cleared per episode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []
        self.step_records: list[dict] = []
        self._span_total = 0.0
        self._in_span = False

    # -- recording --------------------------------------------------------
    def _begin(self, seed: int | None) -> np.ndarray:
        obs = super()._begin(seed)
        self.trace = []
        self.step_records = []
        self._span_total = 0.0
        self._in_span = False
        return obs

    def _step_inner(self, action, branch: str = "law"):
        if branch not in BRANCHES:
            raise ValueError(f"unknown branch {branch!r}; expected one of {BRANCHES}")
        nominal = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        before = self.span_intervention
        out = super()._step_inner(action, branch)
        executed = self.trace[-1]
        sinfo = self._last_sinfo
        rec = {
            "branch": branch,
            "in_span": self._in_span,
            "nominal": [float(nominal[0]), float(nominal[1])],
            "executed": [float(executed[2]), float(executed[3])],
        }
        if sinfo is None:
            rec |= _NO_SHIELD
        else:
            rec |= {
                # The increment of the base class's own accumulator.
                "intervention_magnitude": float(self.span_intervention - before),
                "stalled": bool(sinfo["stalled"]),
                "immobile": bool(sinfo["immobile"]),
                "intervention": bool(sinfo["intervention"]),
                "min_clearance": float(sinfo["min_clearance"]),
            }
        self.step_records.append(rec)
        return out

    def step(self, action):
        self._in_span = True
        out = super().step(action)
        # Accumulated exactly as `RecordingSearchMDP` accumulates it, so the
        # reproduction check compares the same arithmetic and not merely the
        # same quantity.
        self._span_total += float(out[4]["intervention_magnitude"])
        return out

    # -- summary ----------------------------------------------------------
    def branch_summary(self) -> dict:
        """The current episode's summary (plan section 1)."""
        return branch_summary(self.step_records, self._span_total)

    def play_episode(self, predict, seed: int) -> dict:
        log = super().play_episode(predict, seed)
        log["branch"] = self.branch_summary()
        return log
