"""Replay an S18 gate cell through the branch-tagged recorder.

Plan: `docs/pathway/agent_intelligence_plan_2026-10-05.md` section 1 — the
construct check on the instrument itself, and the paper's Phase 0.

S18's gate records cannot say WHICH component of the composite pursuer the
filter pinned: `InterceptLaw` drives on branch `"law"`, the learned search
policy on branch `"search"`, and the recorded `intervention/ep` pools them.
This script replays a recorded gate cell — same checkpoint, same opponent, same
`EnvConfig()`, `hold`, `CBFShield`, `shield_obs`, `escape_obs`,
`reward_wrapper=None`, same seed stream, same N — through
`pe2.recorder.BranchTaggedSearchMDP`, and asks one question before any number
is read:

    does the instrument reproduce the recorded totals EXACTLY?

Same code path, same seeds, a deterministic policy and a memoryless filter, so
the answer is yes or the instrument changed something. The comparison is on
`==` over the floats and there is no tolerance to fall back to: a mismatch is
REPORTED, and everything built on the recorder is then suspect.

The constants, the opponent factories and the width check are IMPORTED from the
frozen `scripts/derisk_s18_boundary.py` rather than restated here — a restated
constant is a second source of truth, and the whole point is that the gate env
is the same env.

`pe2` must resolve to the tree that TRAINED the checkpoints (the S18 worktree),
so this script never imports `pe2.recorder` by package name: the recorder is new
and lives only in the main tree, and is loaded BY FILE PATH from beside this
script. The S18 runner is loaded from whichever tree `pe2` came from, so the
runner and the package can never disagree. All three paths are printed and
recorded in the artefact.

Usage (from the frozen worktree, so the checkpoints are loaded by the code that
trained them):
  cd /home/niro-1/s18_worktree && PYTHONPATH=/home/niro-1/s18_worktree \\
    OMP_NUM_THREADS=4 /home/niro-1/Codes/safe-rl-pe/venv/bin/python \\
    /home/niro-1/Codes/safe-rl-pe/scripts/replay_s18_branch_split.py \\
    --arm SPARSE-PEN-70 --opponent GREEDY
  ... --all          (every arm x both opponents; skips existing outputs)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve()
MAIN_TREE = HERE.parents[1]
# The real artefact directory. The worktree's `results` is a symlink to this
# one, so resolving it absolutely rather than from the cwd keeps the inputs and
# the output in one place whichever tree the script is driven from.
RESULTS = MAIN_TREE / "results" / "derisk"
PLAN = "docs/pathway/agent_intelligence_plan_2026-10-05.md"

# Plan section 4's trigger: S21 (learner takeover) exists only if law-pinned
# runs of at least this many steps occur in at least `TAKEOVER_EPISODE_FRAC` of
# shielded episodes. Measured here, decided later.
TAKEOVER_RUN_STEPS = 40


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _trees() -> tuple[object, object, pathlib.Path]:
    """`(s18_runner, recorder_module, pe2_tree)`.

    `pe2` is imported first and its tree decides where the S18 runner is read
    from; the recorder is read from beside THIS file. Nothing else may import
    `pe2.recorder` by package name — in the worktree there is no such module.
    """
    import pe2

    pe2_tree = pathlib.Path(pe2.__file__).resolve().parents[1]
    runner = _load(
        "derisk_s18_boundary", pe2_tree / "scripts" / "derisk_s18_boundary.py"
    )
    recorder = _load("pe2_recorder_main", MAIN_TREE / "pe2" / "recorder.py")
    return runner, recorder, pe2_tree


def paths(arm: str, opponent: str, runner) -> dict:
    return {
        "ckpt": RESULTS / f"ckpts_S18-{arm}" / f"{arm}_{runner.TOTAL}.zip",
        "recorded": RESULTS / f"{runner.ARTEFACT_PREFIX}_arm_{arm}.json",
        "out": RESULTS / f"s18_branch_split_{arm}_{opponent}.json",
    }


# ----------------------------------------------------------------------------
# the reproduction check
# ----------------------------------------------------------------------------
def compare_reproduction(replayed: dict, recorded: dict) -> dict:
    """Exact comparison of the replay against a recorded gate entry.

    `recorded` is `s18_arm_{ARM}.json -> final.gate.{OPP}`. Both quantities are
    compared with `==`: the replay runs the same code over the same seeds with a
    deterministic policy and a memoryless filter, so equality is a property of
    the setup, not a hope about floating point. A failure is reported with its
    difference and NEVER relaxed to a tolerance — the discrepancy is the finding.
    """
    checks = {}
    for key, got, want in (
        (
            "intervention_total",
            replayed["intervention_total"],
            recorded["intervention"]["total"],
        ),
        ("capture_rate", replayed["capture_rate"], recorded["capture_rate"]),
        ("n_episodes", replayed["n_episodes"], recorded["intervention"]["n_episodes"]),
    ):
        checks[key] = {
            "replayed": got,
            "recorded": want,
            "exact": got == want,
            "difference": got - want,
        }
    return {
        "checks": checks,
        "verdict": "PASS" if all(c["exact"] for c in checks.values()) else "FAIL",
    }


# ----------------------------------------------------------------------------
# aggregation (reported, never gated — the gate is the reproduction check)
# ----------------------------------------------------------------------------
def aggregate(summaries: list[dict]) -> dict:
    """Pooled branch split over the replayed episodes."""
    branches = ("law", "search")

    def pooled(key: str) -> dict:
        return {b: sum(s[key][b] for s in summaries) for b in branches}

    law_runs = [s["longest_pinned_run_by_branch"]["law"] for s in summaries]
    search_runs = [s["longest_pinned_run_by_branch"]["search"] for s in summaries]
    n = len(summaries)
    intervention = pooled("intervention_by_branch")
    total = intervention["law"] + intervention["search"]
    return {
        "n_episodes": n,
        "intervention_by_branch": intervention,
        "intervention_share_by_branch": {
            b: (intervention[b] / total if total else 0.0) for b in branches
        },
        "steps_by_branch": pooled("steps_by_branch"),
        "stalled_by_branch": pooled("stalled_by_branch"),
        "immobile_by_branch": pooled("immobile_by_branch"),
        "longest_pinned_run_by_branch": {
            "law": {
                "median": statistics.median(law_runs) if n else 0,
                "max": max(law_runs) if n else 0,
            },
            "search": {
                "median": statistics.median(search_runs) if n else 0,
                "max": max(search_runs) if n else 0,
            },
        },
        # Plan section 4's trigger for S21, measured.
        "episode_frac_law_pin_ge_40": (
            sum(r >= TAKEOVER_RUN_STEPS for r in law_runs) / n if n else 0.0
        ),
        "episode_frac_search_pin_ge_40": (
            sum(r >= TAKEOVER_RUN_STEPS for r in search_runs) / n if n else 0.0
        ),
        "pre_decision_intervention": sum(
            s["pre_decision_intervention"] for s in summaries
        ),
    }


# ----------------------------------------------------------------------------
# the replay
# ----------------------------------------------------------------------------
def replay(
    arm: str, opponent: str, *, device: str = "cpu", quiet: bool = False
) -> dict:
    runner, recorder, pe2_tree = _trees()
    from pe2.config import EnvConfig
    from pe2.eval import win_count
    from pe2.shield import CBFShield

    if arm not in runner.ARMS:
        raise SystemExit(f"unknown arm {arm!r}; expected one of {sorted(runner.ARMS)}")
    if opponent not in runner.GATE_OPPONENTS:
        raise SystemExit(
            f"unknown opponent {opponent!r}; expected one of {runner.GATE_OPPONENTS}"
        )
    p = paths(arm, opponent, runner)
    if not p["ckpt"].exists():
        raise SystemExit(f"no final checkpoint for {arm}: {p['ckpt']} is missing.")
    if not p["recorded"].exists():
        raise SystemExit(
            f"no recorded gate for {arm}: {p['recorded']} is missing — there is "
            "nothing to reproduce."
        )
    recorded = json.loads(p["recorded"].read_text())["final"]["gate"][opponent]

    from stable_baselines3 import PPO

    model = PPO.load(p["ckpt"], device=device)
    runner._check_width(arm, model)

    cfg = EnvConfig()
    # Exactly `derisk_s18_boundary.evaluate_arm_policy`: unwrapped reward, a
    # FRESH shield per episode, the gate's hold and both observation extras.
    mdp = recorder.BranchTaggedSearchMDP(
        cfg,
        runner.gate_opponent_factory(opponent, device),
        hold=runner.HOLD,
        shield=None,
        shield_obs=runner.SHIELD_OBS,
        escape_obs=runner.ESCAPE_OBS,
        reward_wrapper=None,
    )
    logs = []
    for i in range(runner.GATE_N):
        mdp.shield = CBFShield(cfg)
        logs.append(mdp.play_episode(model.predict, runner.GATE_SEED0 + i))
        if not quiet and (i + 1) % 100 == 0:
            print(f"    {i + 1}/{runner.GATE_N} episodes", flush=True)

    k, n = win_count(logs, "pursuer")
    summaries = [lg["branch"] for lg in logs]
    # Summed in episode order with the builtin `sum`, as `request_summary` sums
    # it, so the arithmetic and not merely the quantity is reproduced.
    replayed = {
        "intervention_total": float(
            sum(s["span_intervention_total"] for s in summaries)
        ),
        "capture_rate": k / n,
        "n_episodes": n,
    }
    out = {
        "plan": PLAN,
        "arm": arm,
        "cell": runner.ARMS[arm]["cell"],
        "seed": runner.ARMS[arm]["seed"],
        "opponent": opponent,
        "n_episodes": runner.GATE_N,
        "seed0": runner.GATE_SEED0,
        "ckpt": str(p["ckpt"]),
        "ckpt_sha256": hashlib.sha256(p["ckpt"].read_bytes()).hexdigest(),
        "pe2_file": str(pathlib.Path(sys.modules["pe2"].__file__).resolve()),
        "pe2_tree": str(pe2_tree),
        "s18_runner_file": str(pe2_tree / "scripts" / "derisk_s18_boundary.py"),
        "recorder_file": str(MAIN_TREE / "pe2" / "recorder.py"),
        "reproduction": compare_reproduction(replayed, recorded),
        "aggregate": aggregate(summaries),
        "episodes": [
            {
                "seed": lg["seed"],
                "outcome": lg["outcome"],
                "steps": lg["steps"],
                "violations": lg["violations"],
                **lg["branch"],
            }
            for lg in logs
        ],
        "provenance": runner.provenance(),
    }
    p["out"].parent.mkdir(parents=True, exist_ok=True)
    tmp = p["out"].with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    os.replace(tmp, p["out"])
    return out


def report(out: dict) -> None:
    rep, agg = out["reproduction"], out["aggregate"]
    print(f"  {out['arm']} vs {out['opponent']}: reproduction {rep['verdict']}")
    for key, c in rep["checks"].items():
        mark = "==" if c["exact"] else "!="
        print(
            f"    {key:18s} replayed {c['replayed']!r} {mark} recorded "
            f"{c['recorded']!r}  (diff {c['difference']!r})"
        )
    iv, sh = agg["intervention_by_branch"], agg["intervention_share_by_branch"]
    print(
        f"    intervention  law {iv['law']:.3f} ({sh['law']:.1%})  "
        f"search {iv['search']:.3f} ({sh['search']:.1%})"
    )
    print(
        f"    steps         law {agg['steps_by_branch']['law']}  "
        f"search {agg['steps_by_branch']['search']}"
    )
    print(
        f"    immobile      law {agg['immobile_by_branch']['law']}  "
        f"search {agg['immobile_by_branch']['search']}"
    )
    lr = agg["longest_pinned_run_by_branch"]
    print(
        f"    longest pin   law median {lr['law']['median']} max {lr['law']['max']}  "
        f"search median {lr['search']['median']} max {lr['search']['max']}"
    )
    print(
        f"    law pin >= {TAKEOVER_RUN_STEPS}: "
        f"{agg['episode_frac_law_pin_ge_40']:.3f} of episodes; "
        f"intervention outside the S18 metric's scope "
        f"{agg['pre_decision_intervention']:.3f}"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm")
    ap.add_argument("--opponent", choices=("GREEDY", "HUG"))
    ap.add_argument(
        "--all",
        action="store_true",
        help="every arm x both opponents, resumable: existing outputs are skipped",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    runner, _recorder, _pe2_tree = _trees()
    print(f"pe2 resolved to {sys.modules['pe2'].__file__}", flush=True)
    print(f"recorder loaded from {MAIN_TREE / 'pe2' / 'recorder.py'}", flush=True)

    if args.all:
        jobs = [(a, o) for a in runner.ARMS for o in runner.GATE_OPPONENTS]
    elif args.arm and args.opponent:
        jobs = [(args.arm, args.opponent)]
    else:
        ap.error("pass --arm and --opponent, or --all")

    failures = []
    for arm, opponent in jobs:
        out_path = paths(arm, opponent, runner)["out"]
        if args.all and out_path.exists():
            print(f"  {arm} vs {opponent}: already written, skipping", flush=True)
            continue
        out = replay(arm, opponent, device=args.device)
        report(out)
        if out["reproduction"]["verdict"] != "PASS":
            failures.append(f"{arm}/{opponent}")
    if failures:
        raise SystemExit(
            "reproduction FAILED for " + ", ".join(failures) + " — the recorder "
            "changed something; nothing built on it may be read until that is "
            "explained."
        )


if __name__ == "__main__":
    main()
