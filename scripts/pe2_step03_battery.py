"""Step 3 battery: seeded duel battery N=500 over 4 scripted matchups.

Verify (automated) per plan Step 3:
- report CR with Wilson CI per matchup (harness: pe2/eval.py)
- LOS-break event rate > 0 (occlusion is in play)

Exit gate (viability) per plan Step 3:
- full-obs greedy x greedy CR in (0, 1), non-degenerate — if CR~1 or ~0 the
  game is broken -> adjust speed ratio/arena (D5) BEFORE any learning
- occlusion demonstrably changes outcomes: CR(Gp x Ge) > CR(FLp x Ge)

Writes results/visual_gates/step_03/step_03_battery.json (full per-episode
logs + summaries) — the visual gate cross-checks its rendered episodes
against these rows.

Usage: ./venv/bin/python scripts/pe2_step03_battery.py [--n 500]
"""

import argparse
import json
import pathlib

from pe2.config import EnvConfig
from pe2.eval import BATTERY_SEED0, run_battery, summarize
from pe2.scripted import (
    FovLimitedEvader,
    FovLimitedPursuer,
    GreedyEvader,
    GreedyPursuer,
)

OUT_DIR = pathlib.Path("results/visual_gates/step_03")

MATCHUPS = {
    "Gp_x_Ge": (GreedyPursuer, GreedyEvader),
    "Gp_x_FLe": (GreedyPursuer, FovLimitedEvader),
    "FLp_x_Ge": (FovLimitedPursuer, GreedyEvader),
    "FLp_x_FLe": (FovLimitedPursuer, FovLimitedEvader),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_battery(EnvConfig(), MATCHUPS, n_episodes=args.n, seed0=BATTERY_SEED0)

    summaries = {}
    print(
        f"battery: N={args.n} per matchup, seeds {BATTERY_SEED0}..{BATTERY_SEED0 + args.n - 1}"
    )
    print(
        f"{'matchup':<12} {'CR':>6} {'Wilson 95% CI':>18} {'mean steps':>10} "
        f"{'mean LOS breaks':>15} {'frac w/ break':>13}"
    )
    for name, logs in results.items():
        s = summarize(logs)
        summaries[name] = s
        print(
            f"{name:<12} {s['cr']:>6.3f} "
            f"[{s['cr_ci'][0]:.3f}, {s['cr_ci'][1]:.3f}]{'':>4}"
            f"{s['mean_steps']:>10.1f} {s['mean_los_breaks']:>15.2f} "
            f"{s['frac_with_los_break']:>13.3f}"
        )

    with open(OUT_DIR / "step_03_battery.json", "w") as f:
        json.dump(
            {
                "n": args.n,
                "seed0": BATTERY_SEED0,
                "summaries": summaries,
                "logs": results,
            },
            f,
        )

    # --- viability exit gate ---
    cr_full = summaries["Gp_x_Ge"]["cr"]
    lo, hi = summaries["Gp_x_Ge"]["cr_ci"]
    assert 0.0 < cr_full < 1.0, (
        f"game degenerate: greedy x greedy CR={cr_full:.3f} "
        "-> adjust speed ratio/arena (D5) before any learning"
    )
    print(f"exit gate: Gp x Ge CR={cr_full:.3f} in (0,1) [CI ({lo:.3f}, {hi:.3f})] OK")

    gap = cr_full - summaries["FLp_x_Ge"]["cr"]
    assert gap > 0.0, (
        f"occlusion does not change outcomes: CR(Gp x Ge) - CR(FLp x Ge) = {gap:.3f}"
    )
    print(f"exit gate: occlusion gap CR(Gp x Ge) - CR(FLp x Ge) = {gap:.3f} > 0 OK")

    los_rate = sum(s["mean_los_breaks"] for s in summaries.values()) / len(summaries)
    assert los_rate > 0.0, "no LOS-break events — occlusion is not in play"
    print(f"verify: mean LOS-break rate across matchups = {los_rate:.2f} > 0 OK")


if __name__ == "__main__":
    main()
