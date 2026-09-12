"""F1 — is the new clone competent, or just exploiting one weak evader?

Spec: docs/pathway/family_eval_prereg_2026-09-12.md (§2 family, §3 protocol,
§4 policy set, §5 validity checks). Re-promoted by D106: T2's 0.476 is
measured against the SINGLE evader whose weakness let a randomly-initialised
network score 0.312 (D104/D105), so it is not yet evidence of pursuit skill.

Primary readout: the WORST CASE across four evaders. Also reported per
opponent, never replaced by the mean. No training happens here.

Usage: ./venv/bin/python scripts/derisk_f1_family.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.eval import wilson_ci, win_count
from pe2.scripted import (
    FovLimitedEvader,
    FovLimitedPursuer,
    GreedyEvader,
    GreedyPursuer,
    ObstacleHuggingEvader,
    OpenSpaceEvader,
    SearchingPursuer,
    run_duel,
)

OUT = pathlib.Path("results/derisk")
OLD_CLONE = pathlib.Path("results/visual_gates/step_05/sac_bc_pursuer.zip")
NEW_CLONE = OUT / "sac_bc_searching_pursuer.zip"

# --- pre-registered (spec §2/§3). Do not tune. ---
N_EPISODES = 200
EVADER_FAMILY = {
    "GreedyEvader": (GreedyEvader, 930_000),
    "FovLimitedEvader": (FovLimitedEvader, 940_000),
    "ObstacleHuggingEvader": (ObstacleHuggingEvader, 950_000),
    "OpenSpaceEvader": (OpenSpaceEvader, 960_000),
}


def score_scripted(cfg, pursuer_cls, evader_cls, seed0) -> dict:
    env = PursuitEvasionEnv(cfg)
    k = sum(
        run_duel(env, pursuer_cls(), evader_cls(), seed0 + i)["outcome"] == "capture"
        for i in range(N_EPISODES)
    )
    lo, hi = wilson_ci(k, N_EPISODES)
    return {"k": k, "n": N_EPISODES, "rate": k / N_EPISODES, "wilson": [lo, hi]}


def score_network(cfg, model, evader_cls, seed0) -> dict:
    from pe2.exploitability import evaluate_sb3_policy

    logs = evaluate_sb3_policy(
        model, cfg, "pursuer", evader_cls, n_episodes=N_EPISODES, seed0=seed0
    )
    k, n = win_count(logs, "pursuer")
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


def worst_case(row: dict) -> tuple[str, float]:
    name = min(row, key=lambda e: row[e]["rate"])
    return name, row[name]["rate"]


def validity_checks(matrix: dict) -> dict:
    """Spec §5, fixed before the run. Both must hold for the family metric to
    be adopted; the membership must NOT be adjusted to make them pass."""
    w = {p: worst_case(r)[1] for p, r in matrix.items()}
    need = ("randinit_seed4_arc", "FovLimitedPursuer_teacher", "bc_clone")
    if any(k not in w for k in need):
        return {
            "pass": None,
            "reason": f"missing policies: {[k for k in need if k not in w]}",
        }
    arc = {
        "pass": bool(w["randinit_seed4_arc"] < w["FovLimitedPursuer_teacher"]),
        "arc_worst": w["randinit_seed4_arc"],
        "teacher_worst": w["FovLimitedPursuer_teacher"],
        "reason": (
            f"the 0.312 arc's worst case {w['randinit_seed4_arc']:.3f} vs the old "
            f"teacher's {w['FovLimitedPursuer_teacher']:.3f}"
        ),
    }
    below = [
        p
        for p in ("randinit_seed4_arc", "randinit_seed3_arc", "stand_still")
        if p in w and w[p] >= w["bc_clone"]
    ]
    order = {
        "pass": bool(
            w.get("GreedyPursuer_fullstate", 1.0) >= w["FovLimitedPursuer_teacher"]
            and w["FovLimitedPursuer_teacher"] >= w["bc_clone"]
            and not below
        ),
        "worst_cases": w,
        "reason": (
            f"full-state {w.get('GreedyPursuer_fullstate')} >= teacher "
            f"{w['FovLimitedPursuer_teacher']:.3f} >= clone {w['bc_clone']:.3f}; "
            f"policies not below the clone: {below or 'none'}"
        ),
    }
    return {
        "arc_collapses": arc,
        "ordering_restored": order,
        "pass": bool(arc["pass"] and order["pass"]),
    }


def write_figure(matrix: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pols, evs = list(matrix), list(EVADER_FAMILY)
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / len(evs)
    for j, e in enumerate(evs):
        ax.bar(
            np.arange(len(pols)) + j * width - 0.4,
            [matrix[p][e]["rate"] for p in pols],
            width=width,
            label=e,
        )
    ax.plot(
        np.arange(len(pols)),
        [worst_case(matrix[p])[1] for p in pols],
        "k*",
        markersize=13,
        label="worst case (the readout)",
    )
    ax.set_xticks(np.arange(len(pols)))
    ax.set_xticklabels(pols, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"capture rate (N={N_EPISODES} per cell)")
    ax.set_title("F1 — pursuers against the evader family")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    from stable_baselines3 import SAC

    from pe2.exploitability import ConstantPolicy, SingleAgentWrapper

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig()
    scripted = {
        "GreedyPursuer_fullstate": GreedyPursuer,
        "FovLimitedPursuer_teacher": FovLimitedPursuer,
        "SearchingPursuer_newteacher": SearchingPursuer,
    }
    nets = {
        "bc_clone": SAC.load(OLD_CLONE, device="cpu"),
        "bc_clone_searching": SAC.load(NEW_CLONE, device="cpu"),
        "randinit_seed4_arc": SAC(
            "MlpPolicy",
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader),
            seed=4,
            verbose=0,
            device="cpu",
        ),
        "randinit_seed3_arc": SAC(
            "MlpPolicy",
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader),
            seed=3,
            verbose=0,
            device="cpu",
        ),
        "stand_still": ConstantPolicy((-1.0, 0.0)),
    }
    matrix = {}
    for name, cls in scripted.items():
        matrix[name] = {
            e: score_scripted(cfg, cls, f, s) for e, (f, s) in EVADER_FAMILY.items()
        }
        print(f"{name:28s} worst {worst_case(matrix[name])[1]:.3f}", flush=True)
    for name, model in nets.items():
        matrix[name] = {
            e: score_network(cfg, model, f, s) for e, (f, s) in EVADER_FAMILY.items()
        }
        print(f"{name:28s} worst {worst_case(matrix[name])[1]:.3f}", flush=True)

    checks = validity_checks(matrix)
    payload = {
        "spec": "docs/pathway/family_eval_prereg_2026-09-12.md",
        "n_episodes": N_EPISODES,
        "family": {k: v[1] for k, v in EVADER_FAMILY.items()},
        "matrix": matrix,
        "worst_case": {p: list(worst_case(r)) for p, r in matrix.items()},
        "validity_checks": checks,
    }
    (OUT / "f1_family.json").write_text(json.dumps(payload, indent=2))
    write_figure(matrix, OUT / "f1_family.png")
    print()
    for p, r in matrix.items():
        cells = "  ".join(f"{e[:12]}={r[e]['rate']:.3f}" for e in EVADER_FAMILY)
        print(f"{p:28s} {cells}   WORST {worst_case(r)[1]:.3f}")
    print()
    for k in ("arc_collapses", "ordering_restored"):
        print(f"{k}: {'PASS' if checks[k]['pass'] else 'FAIL'} — {checks[k]['reason']}")
    print(f"\nfamily metric adopted: {checks['pass']}")


if __name__ == "__main__":
    main()
