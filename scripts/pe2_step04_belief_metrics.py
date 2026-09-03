"""Step 4 belief metrics: visible / just-lost / never-seen RMSE split (L4).

Runs FLp x Ge scripted duels (the matchup where the pursuer loses sight
most), feeds the pursuer's obs stream through LastSeenBelief, and scores
belief error against env truth OFFLINE (the tracker never sees truth).

Verify (automated) per plan Step 4:
- in-FOV RMSE < 2m (trivially ~0 here: observations are noiseless in sim)
- "just lost" vs "never seen" metrics split (never-seen is unpredictable in
  principle -> reported as a COUNT, excluded from RMSE)
- belief never leaks truth during occlusion is tested in test_belief.py

Writes results/visual_gates/step_04/step_04_belief_metrics.json.

Usage: ./venv/bin/python scripts/pe2_step04_belief_metrics.py [--n 100]
"""

import argparse
import json
import pathlib

import numpy as np

from pe2.belief import LastSeenBelief
from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.eval import BATTERY_SEED0
from pe2.scripted import FovLimitedPursuer, GreedyEvader

OUT_DIR = pathlib.Path("results/visual_gates/step_04")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    cfg = EnvConfig()
    env = PursuitEvasionEnv(cfg)
    pursuer, evader = FovLimitedPursuer(), GreedyEvader()

    errors = {"visible": [], "just_lost": []}
    never_seen_steps = 0
    for i in range(args.n):
        pursuer.reset()
        evader.reset()
        belief = LastSeenBelief(cfg)
        obs, _ = env.reset(seed=BATTERY_SEED0 + i)
        while True:
            belief.update(obs["pursuer"])
            if not belief.seen:
                never_seen_steps += 1
            else:
                err = float(np.hypot(*(belief.mean - env.evader_pose[:2])))
                errors["visible" if obs["pursuer"][6] > 0.5 else "just_lost"].append(
                    err
                )
            v_p, w_p = pursuer.act(obs["pursuer"], env)
            v_e, w_e = evader.act(obs["evader"], env)
            obs, _, terminated, truncated, _ = env.step([v_p, w_p, v_e, w_e])
            if terminated or truncated:
                break

    def rmse(vals):
        return float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")

    report = {
        "episodes": args.n,
        "seed0": BATTERY_SEED0,
        "visible": {"steps": len(errors["visible"]), "rmse": rmse(errors["visible"])},
        "just_lost": {
            "steps": len(errors["just_lost"]),
            "rmse": rmse(errors["just_lost"]),
            "mean": float(np.mean(errors["just_lost"]))
            if errors["just_lost"]
            else float("nan"),
            "max": float(np.max(errors["just_lost"]))
            if errors["just_lost"]
            else float("nan"),
        },
        "never_seen": {"steps": never_seen_steps},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step_04_belief_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(
        f"episodes: {args.n} (FLp x Ge, seeds {BATTERY_SEED0}..{BATTERY_SEED0 + args.n - 1})"
    )
    print(
        f"visible:    {report['visible']['steps']:>7} steps, RMSE = {report['visible']['rmse']:.4f} m"
    )
    print(
        f"just lost:  {report['just_lost']['steps']:>7} steps, RMSE = {report['just_lost']['rmse']:.3f} m "
        f"(mean {report['just_lost']['mean']:.3f}, max {report['just_lost']['max']:.3f})"
    )
    print(
        f"never seen: {report['never_seen']['steps']:>7} steps (excluded from RMSE, L4)"
    )

    assert report["visible"]["rmse"] < 2.0, "in-FOV RMSE target missed"
    assert report["just_lost"]["steps"] > 0, "no occlusion steps — split is meaningless"
    print("verify: in-FOV RMSE < 2m OK; just-lost/never-seen split reported OK")


if __name__ == "__main__":
    main()
