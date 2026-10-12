"""S24.0 Step 2 — the belief construct gate: does the particle filter beat dead
reckoning while the opponent is out of sight?

Spec: `docs/pathway/tactics_plan_2026-10-13.md` sections 1 ("Memory": the belief
is "validated against dead-reckoning (its median occlusion error must be
lower)") and 2 (S24.0's construct gates). The quantity is S20's H-BEL-3
(`docs/pathway/agent_intelligence_plan_2026-10-05.md` section 3), which that
round pre-registered and never ran.

**Gate 1, written before it ran (and before the class was finished):**

    the particle belief's MEDIAN error to the true evader position, pooled over
    all occluded steps, must be LOWER than dead reckoning's, on BOTH opponents.

Gate 1 FAILED on both opponents (D145). Its table and its verdict stand here
unchanged, and this module's job is now to carry both numbers.

**Gate 2, pre-registered in the plan AFTER gate 1's failure and BEFORE this
replay ran** (plan section "S24.0 Step 2 — belief gate FAILED (D145); gate 2
pre-registered"): gate 1 scored the belief MEAN; the consumers of the belief
(`SWEEP`, `AMBUSH`) read the strongest MODE. On the SAME 200 paired episodes
and the SAME occluded steps, the error of `summary()["mode"]` to the true
position must have a LOWER median than dead reckoning's point estimate (which
is unchanged) on both opponents.

Nothing is re-seeded: the cells, the stream, the particle count and the
filter's seed are the same, so the replay reproduces gate 1's numbers
BIT-FOR-BIT — asserted against the previously written record before anything is
overwritten, and a mismatch STOPS the script instead of replacing a number.

Protocol: the S19 gate stream (2 000 000+), N = 200 paired episodes, the cells
`PLAN-SHIELD` vs GREEDY and vs HUG, built by S19's own `build_cell_env` (so the
law, the shield, the search policy and the opponent are the round's, not a
look-alike). Both beliefs ride along PASSIVELY: they are updated from the
pursuer's observation after every env step and they never touch a command. The
true evader position is read for the ERROR METRIC ONLY, outside both beliefs.

Information contract (what makes the comparison fair): both beliefs see exactly
the same stream. Dead reckoning takes the 11-D observation; the particle filter
takes that observation DECODED (own pose, the visibility flag, the observed
opponent position when visible) plus the static map, which is known to both
roles everywhere in pe2 (`pe2/obs.py`: obstacle features are not masked). The
opponent's speed bound is passed in by this script — the class never reads
`cfg.evader_speed_ratio` (plan section 1, the speed-agnostic rule).

Reported beside the gate, gating nothing: the p90 of each error, the fraction of
steps that were occluded, the particle cloud's mean spread, the per-episode
medians, and the same medians restricted to occluded steps AFTER the first
sighting — because before any sighting dead reckoning has no estimate at all
and reports the origin, and a reader is entitled to know whether the gate rests
on those steps. The gate itself is the pooled all-occluded-steps median above.

Usage:
  ./venv/bin/python scripts/s24_belief_gate.py
  ./venv/bin/python scripts/s24_belief_gate.py --summarize
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # plan section 4 / S19 section 7; BEFORE numpy builds pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:  # so `scripts.` resolves when run as a script
    sys.path.insert(0, str(_REPO))

import pe2
import scripts.derisk_s19_planner as S19
from pe2.belief import LastSeenBelief, ParticleBelief
from pe2.obs import opponent_world_from_obs, own_pose_from_obs
from pe2.structured import OrbitSearch

PLAN = "docs/pathway/tactics_plan_2026-10-13.md"
OUT = pathlib.Path("results/derisk") / "s24_belief_gate.json"

# --- fixed here before the run. Do not tune. -------------------------------
GATE_SEED0 = S19.GATE_SEED0  # 2 000 000, the S19 gate stream
N_EPISODES = 200
CELLS: tuple[str, ...] = ("PLAN-SHIELD_GREEDY", "PLAN-SHIELD_HUG")
N_PARTICLES = 500  # S20 section 3 ("~500 particles")
PF_SEED = 24_000  # the filter's own RNG seed; the cells share it
GATE_1 = (
    "the particle belief's median error to the true evader position, pooled "
    "over all occluded steps, is LOWER than dead reckoning's on BOTH opponents"
)
# Verbatim, as the plan writes it.
GATE_2 = (
    "GATE 2: the particle belief's MODE median occlusion error must be LOWER "
    "than dead-reckoning's on both opponents."
)


class BeliefProbe:
    """Both beliefs, ridden alongside one episode, plus the error record.

    `observe` is called once per env step, AFTER the step, with the pursuer's
    observation and the env. It updates both beliefs from the observation and —
    only when the pursuer cannot see the evader — records each belief mean's
    distance to the true evader position.
    """

    def __init__(self, cfg, v_max_opp: float, n: int = N_PARTICLES, seed: int = 0):
        self.cfg = cfg
        self.dr = LastSeenBelief(cfg)
        self.pf = ParticleBelief(cfg, v_max_opp, n=n, seed=seed)
        self.steps = 0
        self.occluded = 0
        self.dr_err: list[float] = []
        self.pf_err: list[float] = []  # gate 1: the belief MEAN
        self.pf_mode_err: list[float] = []  # gate 2: the strongest MODE
        self.after_sighting: list[bool] = []
        self.pf_spread: list[float] = []
        self.episode_median: list[dict] = []
        self._ep: tuple[list[float], list[float], list[float]] = ([], [], [])

    def reset(self, env) -> None:
        if any(self._ep):
            self._close_episode()
        self.dr.reset()
        self.pf.reset(env)
        self._ep = ([], [], [])

    def _close_episode(self) -> None:
        dr, pf, mode = self._ep
        self.episode_median.append(
            {
                "n_occluded": len(dr),
                "dr_median": float(np.median(dr)) if dr else None,
                "pf_median": float(np.median(pf)) if pf else None,
                "pf_mode_median": float(np.median(mode)) if mode else None,
            }
        )
        self._ep = ([], [], [])

    def observe(self, obs, env) -> None:
        visible = bool(obs[6] > 0.5)
        own = own_pose_from_obs(obs, self.cfg)
        opp = opponent_world_from_obs(obs, self.cfg) if visible else None
        self.dr.update(obs)
        self.pf.update(own, visible, opp, env.obstacles)
        self.steps += 1
        if visible:
            return
        self.occluded += 1
        truth = np.asarray(env.evader_pose[:2], dtype=float)
        summary = self.pf.summary()
        dr_e = float(np.linalg.norm(self.dr.mean - truth))
        pf_e = float(np.linalg.norm(self.pf.mean - truth))
        mode_e = float(np.linalg.norm(summary["mode"] - truth))
        self.dr_err.append(dr_e)
        self.pf_err.append(pf_e)
        self.pf_mode_err.append(mode_e)
        self.after_sighting.append(bool(self.dr.seen))
        self.pf_spread.append(summary["spread"])
        self._ep[0].append(dr_e)
        self._ep[1].append(pf_e)
        self._ep[2].append(mode_e)

    # -- readout ---------------------------------------------------------
    def stats(self) -> dict:
        if any(self._ep):
            self._close_episode()
        dr = np.asarray(self.dr_err)
        pf = np.asarray(self.pf_err)
        mode = np.asarray(self.pf_mode_err)
        aft = np.asarray(self.after_sighting, dtype=bool)
        eps = [e for e in self.episode_median if e["dr_median"] is not None]
        return {
            "n_steps": self.steps,
            "n_occluded": self.occluded,
            "occluded_fraction": self.occluded / self.steps if self.steps else None,
            "dr_median": _med(dr),
            "dr_p90": _p90(dr),
            "pf_median": _med(pf),
            "pf_p90": _p90(pf),
            "pf_spread_mean": (
                float(np.mean(self.pf_spread)) if self.pf_spread else None
            ),
            # reported only (see the module docstring): the same medians on the
            # occluded steps after the first sighting, where dead reckoning has
            # something to propagate.
            "after_sighting_fraction": float(aft.mean()) if len(aft) else None,
            "dr_median_after_sighting": _med(dr[aft]),
            "pf_median_after_sighting": _med(pf[aft]),
            "n_episodes_with_occlusion": len(eps),
            "episode_pf_better_frac": (
                sum(e["pf_median"] < e["dr_median"] for e in eps) / len(eps)
                if eps
                else None
            ),
            # gate 2 (same steps, same episodes): the strongest MODE
            "pf_mode_median": _med(mode),
            "pf_mode_p90": _p90(mode),
            "pf_mode_median_after_sighting": _med(mode[aft]),
            "episode_pf_mode_better_frac": (
                sum(e["pf_mode_median"] < e["dr_median"] for e in eps) / len(eps)
                if eps
                else None
            ),
            "episode_median": self.episode_median,
        }


def _med(a) -> float | None:
    a = np.asarray(a, dtype=float)
    return float(np.median(a)) if a.size else None


def _p90(a) -> float | None:
    a = np.asarray(a, dtype=float)
    return float(np.percentile(a, 90)) if a.size else None


def attach(mdp, probe: BeliefProbe) -> None:
    """Wrap the cell env's two hooks so the beliefs ride the episode.

    `_begin` and `_step_inner` are `SearchMDP`'s ONLY episode-start and
    per-env-step paths (the law branch, the search branch and the shield all go
    through them), so wrapping them records every step exactly once and nothing
    in the cell's construction changes. The wrappers are read-only.
    """
    begin, step_inner = mdp._begin, mdp._step_inner

    def _begin(seed):
        obs = begin(seed)
        probe.reset(mdp.inner.env)
        return obs

    def _step(action, branch: str = "law"):
        out = step_inner(action, branch)
        probe.observe(out[0], mdp.inner.env)
        return out

    mdp._begin = _begin
    mdp._step_inner = _step


def run_cell(name: str, n_episodes: int = N_EPISODES) -> dict:
    cfg = S19.cell_config()
    mdp = S19.build_cell_env(name, cfg)
    policy = OrbitSearch(cfg)
    # The caller supplies the opponent's speed bound (plan section 1): the
    # class is speed-agnostic, this script is not.
    probe = BeliefProbe(cfg, cfg.evader_v_max, seed=PF_SEED)
    attach(mdp, probe)
    outcomes: dict[str, int] = {}
    for i in range(n_episodes):
        if S19.CELLS[name]["shielded"]:
            mdp.shield = S19.build_shield(cfg)
        log = mdp.play_episode(policy.predict, GATE_SEED0 + i)
        outcomes[log["outcome"]] = outcomes.get(log["outcome"], 0) + 1
    return {
        "cell": name,
        "opponent": S19.CELLS[name]["opponent"],
        "n_episodes": n_episodes,
        "seed0": GATE_SEED0,
        "n_particles": N_PARTICLES,
        "pf_seed": PF_SEED,
        "v_max_opp": float(cfg.evader_v_max),
        "outcomes": outcomes,
        **probe.stats(),
    }


def _verdict(cells: list[dict], gate: str, key: str) -> dict:
    per = {
        c["opponent"]: bool(c[key] < c["dr_median"])
        for c in cells
        if c[key] is not None
    }
    return {
        "gate": gate,
        "statistic": key,
        "per_opponent_pass": per,
        "n_opponents": len(cells),
        "passed": bool(per) and len(per) == len(cells) and all(per.values()),
    }


# The fields gate 1 rests on. The reproduction check compares EXACTLY these,
# because the replay legitimately adds gate 2's fields beside them.
G1_CELL_KEYS: tuple[str, ...] = (
    "cell",
    "opponent",
    "n_episodes",
    "seed0",
    "n_particles",
    "pf_seed",
    "v_max_opp",
    "outcomes",
    "n_steps",
    "n_occluded",
    "occluded_fraction",
    "dr_median",
    "dr_p90",
    "pf_median",
    "pf_p90",
    "pf_spread_mean",
    "after_sighting_fraction",
    "dr_median_after_sighting",
    "pf_median_after_sighting",
    "n_episodes_with_occlusion",
    "episode_pf_better_frac",
)
G1_EPISODE_KEYS: tuple[str, ...] = ("n_occluded", "dr_median", "pf_median")
G2_CELL_KEYS: tuple[str, ...] = (
    "cell",
    "opponent",
    "n_episodes",
    "seed0",
    "n_occluded",
    "occluded_fraction",
    "dr_median",
    "dr_p90",
    "pf_mode_median",
    "pf_mode_p90",
    "after_sighting_fraction",
    "dr_median_after_sighting",
    "pf_mode_median_after_sighting",
    "episode_pf_mode_better_frac",
)


def gate_1_view(cell: dict) -> dict:
    return {k: cell[k] for k in G1_CELL_KEYS} | {
        "episode_median": [
            {k: e[k] for k in G1_EPISODE_KEYS} for e in cell["episode_median"]
        ]
    }


def gate_2_view(cell: dict) -> dict:
    return {k: cell[k] for k in G2_CELL_KEYS} | {
        "episode_median": [
            {"n_occluded": e["n_occluded"], "pf_mode_median": e["pf_mode_median"]}
            for e in cell["episode_median"]
        ]
    }


def old_gate_1_cells(old: dict) -> dict:
    """Gate 1's cells from a previously written record, under either schema
    (the first run wrote them at the top level; this one writes `gate_1`)."""
    cells = old["gate_1"]["cells"] if "gate_1" in old else old["cells"]
    return {c["cell"]: c for c in cells}


def check_reproduction(old: dict, cells: list[dict]) -> dict:
    """Gate 1 must come back BIT-FOR-BIT off the replay, or nothing is written.

    Same cells, same stream, same particle count, same filter seed, and gate 2
    consumes no randomness (`summary()["mode"]` is a readout), so every gate-1
    number is reproducible. A mismatch means the replay is not the run that
    produced D145's numbers, and then the right move is to report the
    discrepancy, not to overwrite a recorded result.
    """
    previous = old_gate_1_cells(old)
    diffs = []
    for c in cells:
        was, now = previous.get(c["cell"]), gate_1_view(c)
        if was is None:
            diffs.append({"cell": c["cell"], "field": "<cell>", "was": None})
            continue
        was = gate_1_view(was)
        for k in was.keys() | now.keys():
            if was.get(k) != now.get(k):
                diffs.append(
                    {
                        "cell": c["cell"],
                        "field": k,
                        "was": was.get(k) if k != "episode_median" else "<list>",
                        "now": now.get(k) if k != "episode_median" else "<list>",
                    }
                )
    return {
        "checked_cells": [c["cell"] for c in cells],
        "n_fields": len(G1_CELL_KEYS) + 1,
        "exact": not diffs,
        "diffs": diffs,
    }


def run(n_episodes: int = N_EPISODES) -> dict:
    old = json.loads(OUT.read_text()) if OUT.exists() else None
    if old is not None and "gate_2" in old:
        raise SystemExit(
            f"{OUT} already carries gate_2: this gate runs ONCE (plan, "
            '"One run; no other change"). Use --summarize to re-print it.'
        )
    cells = [run_cell(name, n_episodes) for name in CELLS]
    reproduction = (
        check_reproduction(old, cells)
        if old is not None
        else {"exact": None, "note": "no previous record to reproduce"}
    )
    if reproduction["exact"] is False:
        raise SystemExit(
            "STOP: the replay did NOT reproduce gate 1 bit-for-bit, so nothing "
            "was written and the recorded gate-1 numbers stand untouched. "
            f"Discrepancies: {json.dumps(reproduction['diffs'], indent=2)}"
        )
    record = {
        "spec": PLAN,
        "spec_section": (
            "1 (Memory) and 2 (S24.0); S20 H-BEL-3; "
            '"S24.0 Step 2 — belief gate FAILED (D145); gate 2 pre-registered"'
        ),
        "gate_1": {
            "gate": GATE_1,
            "status": "pre-registered before the first run; FAILED (D145)",
            "cells": [gate_1_view(c) for c in cells],
            "verdict": _verdict(cells, GATE_1, "pf_median"),
        },
        "gate_2": {
            "gate": GATE_2,
            "status": (
                "pre-registered in the plan after gate 1 failed and before "
                "this replay ran; same 200 paired episodes, same occluded "
                "steps, nothing re-seeded"
            ),
            "cells": [gate_2_view(c) for c in cells],
            "verdict": _verdict(cells, GATE_2, "pf_mode_median"),
        },
        "gate_1_reproduction": reproduction,
        "provenance": provenance(),
    }
    _write_json(OUT, record)
    record["path"] = str(OUT)
    return record


# ----------------------------------------------------------------------------
# provenance, output, table
# ----------------------------------------------------------------------------
def provenance() -> dict:
    try:
        sha = subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    plan = _REPO / PLAN
    return {
        "git_commit": sha,
        "pe2_file": pe2.__file__,
        "spec": PLAN,
        "spec_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def _write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _f(x, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def table(record: dict) -> str:
    """Each gate's verdict line FIRST, then the table it rests on (plan section
    4). Gate 1's table is the recorded one, unchanged."""
    g1, g2 = record["gate_1"], record["gate_2"]
    rep = record["gate_1_reproduction"]
    lines = [
        f"S24.0 Step 2 gate 1 — {g1['gate']}",
        "",
        _verdict_line("GATE", g1["verdict"]),
        "",
    ]
    head = (
        f"{'opponent':<9} {'DR med':>8} {'DR p90':>8} {'PF med':>8} {'PF p90':>8} "
        f"{'occl frac':>10} {'PF spread':>10} {'eps PF<DR':>10}"
    )
    lines += [head, "-" * len(head)]
    for c in g1["cells"]:
        lines.append(
            f"{c['opponent']:<9} {_f(c['dr_median']):>8} {_f(c['dr_p90']):>8} "
            f"{_f(c['pf_median']):>8} {_f(c['pf_p90']):>8} "
            f"{_f(c['occluded_fraction']):>10} {_f(c['pf_spread_mean']):>10} "
            f"{_f(c['episode_pf_better_frac'], 2):>10}"
        )
    lines += [
        "",
        "reported only (gates nothing) — occluded steps after the first sighting:",
    ]
    for c in g1["cells"]:
        lines.append(
            f"  {c['opponent']:<9} share {_f(c['after_sighting_fraction'], 2)}  "
            f"DR med {_f(c['dr_median_after_sighting'])}  "
            f"PF med {_f(c['pf_median_after_sighting'])}  "
            f"(n_occluded {c['n_occluded']} of {c['n_steps']} steps, "
            f"{c['n_episodes']} episodes from {c['seed0']})"
        )

    lines += [
        "",
        "",
        (
            "S24.0 Step 2 gate 2 — the same 200 paired episodes, the same "
            "occluded steps, nothing re-seeded."
        ),
        (
            "gate 1 reproduction on the replay: "
            + {True: "EXACT (bit-for-bit)", False: "MISMATCH", None: "n/a"}[
                rep.get("exact")
            ]
            + f" over {len(rep.get('checked_cells', []))} cells"
        ),
        "",
        g2["gate"],
        _verdict_line("GATE 2", g2["verdict"]),
        "",
    ]
    head2 = (
        f"{'opponent':<9} {'DR med':>8} {'DR p90':>8} {'PFmode med':>11} "
        f"{'PFmode p90':>11} {'occl frac':>10} {'eps mode<DR':>12}"
    )
    lines += [head2, "-" * len(head2)]
    for c in g2["cells"]:
        lines.append(
            f"{c['opponent']:<9} {_f(c['dr_median']):>8} {_f(c['dr_p90']):>8} "
            f"{_f(c['pf_mode_median']):>11} {_f(c['pf_mode_p90']):>11} "
            f"{_f(c['occluded_fraction']):>10} "
            f"{_f(c['episode_pf_mode_better_frac'], 2):>12}"
        )
    lines += [
        "",
        "reported only (gates nothing) — occluded steps after the first sighting:",
    ]
    for c in g2["cells"]:
        lines.append(
            f"  {c['opponent']:<9} share {_f(c['after_sighting_fraction'], 2)}  "
            f"DR med {_f(c['dr_median_after_sighting'])}  "
            f"PFmode med {_f(c['pf_mode_median_after_sighting'])}"
        )
    return "\n".join(lines)


def _verdict_line(label: str, verdict: dict) -> str:
    return f"{label} {'PASS' if verdict['passed'] else 'FAIL'}: " + ", ".join(
        f"{o} {'lower' if p else 'NOT lower'}"
        for o, p in verdict["per_opponent_pass"].items()
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summarize", action="store_true", help="re-print the existing record"
    )
    ap.add_argument("--episodes", type=int, default=N_EPISODES, help="N (default 200)")
    args = ap.parse_args(argv)
    if args.summarize:
        record = json.loads(OUT.read_text())
    else:
        record = run(args.episodes)
        print(f"wrote {record['path']}")
    print(table(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
