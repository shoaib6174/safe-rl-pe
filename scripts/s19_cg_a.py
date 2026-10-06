"""CG-A (docs/pathway/planner_intercept_prereg_2026-10-06.md section 5).

200 planned paths on 200 random maps from the construct stream (seeds
1 990 000+) never enter an inflated obstacle or leave the inset arena, plus a
figure of 12 of them. BLOCKING: no capture rate of the S19 round is read
before this passes.

    OMP_NUM_THREADS=4 ./venv/bin/python scripts/s19_cg_a.py

`sample_cases` is imported by tests/pe2/test_planner.py so the test and the
figure are the same 200 cases.
"""

from __future__ import annotations

import argparse
import pathlib
from itertools import pairwise

import numpy as np

from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.planner import LOOKAHEAD, PlannerInterceptLaw

CONSTRUCT_SEED0 = 1_990_000
N_CASES = 200
TOL = 1e-9  # a tangent path touches an inflated disc; it does not enter it


def _free_point(rng, law, lim):
    """A point a real agent could occupy: inside the inset arena and not
    inside any inflated obstacle (the env's own placement rule)."""
    while True:
        p = rng.uniform(-lim, lim, size=2)
        if law.segment_clearance(p, p) > 0.0:
            return p


def sample_cases(
    cfg: EnvConfig | None = None, n: int = N_CASES, seed0: int = CONSTRUCT_SEED0
):
    """n (obstacles, own, target, smoothed path) cases from the construct
    stream. `path` is None when A* found no path."""
    cfg = cfg or EnvConfig()
    env = PursuitEvasionEnv(cfg)
    law = PlannerInterceptLaw(cfg)
    lim = cfg.arena_size / 2.0 - cfg.agent_radius
    cases = []
    for i in range(n):
        env.reset(seed=seed0 + i)
        law.set_map(env.obstacles)
        rng = np.random.default_rng(seed0 + i)
        own = _free_point(rng, law, lim)
        target = _free_point(rng, law, lim)
        cases.append(
            {
                "seed": seed0 + i,
                "obstacles": [dict(o) for o in env.obstacles],
                "own": own,
                "target": target,
                "line_clear": law.is_clear(own, target),
                "path": law.plan(own, target),
            }
        )
    return cases


def check(cases) -> dict:
    """Per the prereg: every vertex, every segment and the lookahead point of
    every planned path stays inside the inset arena and out of every inflated
    disc. Returns the violation count and the offending seeds."""
    cfg = EnvConfig()
    law = PlannerInterceptLaw(cfg)
    bad, n_paths, n_unreachable, n_blocked_line = [], 0, 0, 0
    for case in cases:
        n_blocked_line += int(not case["line_clear"])
        if case["path"] is None:
            n_unreachable += 1
            continue
        n_paths += 1
        law.set_map(case["obstacles"])
        path = case["path"]
        points = [*path, law.lookahead_point(path, LOOKAHEAD)]
        ok = all(law.in_arena(p) for p in points)
        ok = ok and all(law.segment_clearance(p, p) >= -TOL for p in points)
        ok = ok and all(law.segment_clearance(a, b) >= -TOL for a, b in pairwise(path))
        if not ok:
            bad.append(case["seed"])
    return {
        "n_cases": len(cases),
        "n_paths": n_paths,
        "n_unreachable": n_unreachable,
        "n_line_blocked": n_blocked_line,
        "violations": len(bad),
        "bad_seeds": bad,
    }


def figure(cases, out: pathlib.Path, k: int = 12) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = EnvConfig()
    lim = cfg.arena_size / 2.0 - cfg.agent_radius
    law = PlannerInterceptLaw(cfg)
    # The 12 most interesting cases: obstructed lines first (a clear line
    # plans the straight segment and shows nothing), then in seed order.
    chosen = sorted(cases, key=lambda c: (c["line_clear"], c["seed"]))[:k]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    for ax, case in zip(axes.ravel(), chosen, strict=True):
        law.set_map(case["obstacles"])
        for o in case["obstacles"]:
            ax.add_patch(
                plt.Circle(
                    (o["x"], o["y"]),
                    o["radius"] + cfg.agent_radius,
                    fc="0.85",
                    ec="0.6",
                    lw=0.8,
                )
            )
            ax.add_patch(plt.Circle((o["x"], o["y"]), o["radius"], fc="0.5", ec="none"))
        own, target = case["own"], case["target"]
        ax.plot(*zip(own, target, strict=True), ":", c="tab:red", lw=1.0)
        if case["path"] is not None:
            p = np.array(case["path"])
            ax.plot(p[:, 0], p[:, 1], "-o", c="tab:blue", ms=3, lw=1.4)
            la = law.lookahead_point(case["path"], LOOKAHEAD)
            ax.plot(*la, "*", c="tab:orange", ms=11)
        ax.plot(*own, "s", c="k", ms=5)
        ax.plot(*target, "x", c="tab:green", ms=8, mew=2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{case['seed']} {'clear' if case['line_clear'] else 'planned'}", fontsize=9
        )
    fig.suptitle(
        "S19 CG-A — planned paths (blue) vs the straight line (red dotted); "
        "grey = obstacle inflated by agent_radius; star = 0.5 m lookahead",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/derisk/s19_paths.png")
    args = ap.parse_args()
    cases = sample_cases()
    report = check(cases)
    figure(cases, pathlib.Path(args.out))
    print(report)
    print("CG-A", "PASS" if report["violations"] == 0 else "FAIL")


if __name__ == "__main__":
    main()
