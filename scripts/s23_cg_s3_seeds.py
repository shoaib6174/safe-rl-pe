"""S23 CG-S3 — the watched seeds, chosen by a rule on the baseline's geometry.

Spec: `docs/pathway/pursuit_strategy_plan_2026-10-12.md` section 4 (CG-S3).

S19's CG-C spent two of its three watched seeds on episodes where the straight
line to the evader was never obstructed -- episodes in which the planner cannot
differ from the line law at all, so the behavioural gate watched the wrong
thing. The rule, fixed here before any treatment runs and blind to every
outcome, is the plan's:

  The three watched seeds are the FIRST THREE construct seeds on which
  `PLAN-SHIELD`'s law branch fires with the line obstructed for >= 1 step.

"the law branch fires with the line obstructed" is `PlannerInterceptLaw`'s own
clear-line test, counted by the law itself: `n_planned_steps` increments on
every visible-branch call where `is_clear(own, evader)` is false and the law
plans instead of steering. The scan reads it as a per-episode difference, in
seed order, on the construct stream, and STOPS at the third hit -- so the rule
cannot be influenced by anything that happens on later seeds.

The scan runs `PLAN-SHIELD` vs GREEDY: the reference cell of the round's primary
metric, built by the S19 runner's own `build_cell_env`, unmodified.

Usage:
  ./venv/bin/python scripts/s23_cg_s3_seeds.py
  ./venv/bin/python scripts/s23_cg_s3_seeds.py --summarize
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # plan section 7; set BEFORE numpy / torch build their pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:  # so `scripts.` resolves when run as a script
    sys.path.insert(0, str(_REPO))

import pe2
import scripts.derisk_s19_planner as S19
from pe2.planner import PlannerInterceptLaw
from pe2.structured import OrbitSearch

PLAN = "docs/pathway/pursuit_strategy_plan_2026-10-12.md"
OUT = pathlib.Path("results/derisk") / "s23_cg_s3_seeds.json"

# --- fixed by the plan (sections 4 and 5). Do not tune. ---------------------
CONSTRUCT_SEED0 = 2_090_000  # section 5, the construct stream
OPPONENT = "GREEDY"  # section 5, the primary metric's opponent
ROW, COL = "PLAN", "SHIELD"
N_SEEDS = 3  # "the first three construct seeds"
MAX_SCAN = 200  # the scan's cap; fewer than N_SEEDS hits is a refusal

RULE = (
    "the first three seeds of the construct stream (2 090 000+), in seed order, "
    "on which PLAN-SHIELD's law branch fires with the straight line to the "
    "evader obstructed for >= 1 step -- i.e. PlannerInterceptLaw.n_planned_steps "
    "increases during the episode (its visible branch planned instead of "
    "steering). Fixed before any treatment ran; blind to every outcome. The scan "
    "stops at the third hit and is capped at 200 seeds."
)


def scan(
    mdp,
    predict,
    *,
    shield_factory=None,
    seed0: int = CONSTRUCT_SEED0,
    n_seeds: int = N_SEEDS,
    max_scan: int = MAX_SCAN,
) -> dict:
    """Play `mdp` in seed order from `seed0` and return the rule's record.

    `mdp` must expose `law.n_planned_steps` and `play_episode(predict, seed)`;
    `shield_factory`, when given, rebuilds the shield before every episode
    exactly as `S19.run_cell` does. The loop RETURNS at the `n_seeds`-th hit --
    it does not scan the cap and then take the first three -- so no later
    episode can be read, and `scanned` is the number of episodes played.

    Raises `SystemExit` when the cap is reached with fewer than `n_seeds` hits:
    a watched set chosen by a weaker rule is not the pre-registered one.
    """
    seeds: list[int] = []
    episodes: list[dict] = []
    for i in range(max_scan):
        seed = seed0 + i
        if shield_factory is not None:
            mdp.shield = shield_factory()
        before = mdp.law.n_planned_steps
        log = mdp.play_episode(predict, seed)
        planned = mdp.law.n_planned_steps - before
        hit = planned >= 1
        episodes.append(
            {
                "seed": seed,
                "n_planned_steps": int(planned),
                "hit": bool(hit),
                "outcome": log["outcome"],
                "steps": log["steps"],
                "law_steps": int(log["branch"]["steps_by_branch"]["law"]),
            }
        )
        if hit:
            seeds.append(seed)
            if len(seeds) == n_seeds:
                return {
                    "seeds": seeds,
                    "rule": RULE,
                    "scanned": i + 1,
                    "episodes": episodes,
                }
    raise SystemExit(
        f"CG-S3 REFUSES: only {len(seeds)} of the required {n_seeds} obstructed "
        f"episodes in {max_scan} construct seeds from {seed0} "
        f"({', '.join(str(s) for s in seeds) or 'none'}). The watched episodes "
        "are the rule's or they are not pre-registered; nothing was written."
    )


def run(max_scan: int = MAX_SCAN) -> dict:
    cfg = S19.cell_config()
    cell = S19.cell_name(ROW, COL, OPPONENT)
    mdp = S19.build_cell_env(cell, cfg)
    if type(mdp.law) is not PlannerInterceptLaw:
        raise SystemExit(
            f"the {cell} cell's law is {type(mdp.law).__name__}, not "
            "PlannerInterceptLaw; CG-S3's rule is that law's clear-line test "
            "and this scan refuses to run on another."
        )
    record = scan(
        mdp,
        OrbitSearch(cfg).predict,
        shield_factory=lambda: S19.build_shield(cfg),
        max_scan=max_scan,
    )
    record |= {
        "spec": PLAN,
        "spec_section": "4 (CG-S3)",
        "cell": cell,
        "opponent": OPPONENT,
        "seed0": CONSTRUCT_SEED0,
        "max_scan": max_scan,
        "n_seeds": N_SEEDS,
        "law": type(mdp.law).__name__,
        "provenance": provenance(),
    }
    _write_json(OUT, record)
    record["path"] = str(OUT)
    return record


# ----------------------------------------------------------------------------
# provenance, output, table
# ----------------------------------------------------------------------------
def plan_path() -> pathlib.Path:
    return _REPO / PLAN


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
    return {
        "git_commit": sha,
        "pe2_file": pe2.__file__,
        "spec": PLAN,
        "spec_sha256": hashlib.sha256(plan_path().read_bytes()).hexdigest(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def _write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def table(record: dict) -> str:
    """The rule, then the seeds it selected and the episodes it scanned. No
    capture rate: these are per-episode geometry, chosen blind to outcome."""
    head = f"{'seed':>9} {'planned steps':>14} {'law steps':>10} {'hit':>5}"
    lines = [
        "CG-S3 rule (fixed before any treatment ran):",
        "",
        f"  {record['rule']}",
        "",
        (
            f"S23 step 4 — CG-S3 on {record['cell']}: scanned "
            f"{record['scanned']} seeds from {record['seed0']}, "
            f"watched seeds {record['seeds']}"
        ),
        "",
        head,
        "-" * len(head),
    ]
    for e in record["episodes"]:
        lines.append(
            f"{e['seed']:9d} {e['n_planned_steps']:14d} {e['law_steps']:10d} "
            f"{('yes' if e['hit'] else 'no'):>5}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summarize",
        action="store_true",
        help="re-print the rule and the scan from the existing record",
    )
    args = ap.parse_args(argv)
    if args.summarize:
        record = json.loads(OUT.read_text())
    else:
        record = run()
        print(f"wrote {record['path']}")
    print(table(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
