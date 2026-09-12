"""T2 — does a teacher that SEARCHES fix the pipeline? (D105, D106).

Spec: docs/pathway/searching_teacher_prereg_2026-09-12.md. Nothing here may
change a threshold, a constant or a seed after a number is seen.

D105 measured the root cause of four rounds of learning failure: the BC
teacher `FovLimitedPursuer` scans by STANDING STILL, is stationary on 92.6 %
of its steps, and its demonstrations are 83.6 % 'stand still and rotate' at a
mean commanded speed of 0.164 m/s out of 1.0. Every learner started from an
imitation of a policy that barely moves.

This round changes exactly one branch of the teacher (`SearchingPursuer`
orbits instead of standing still, as a pure function of own pose) and
re-clones with the BC recipe otherwise untouched, so the TEACHER is the only
variable. Three pre-registered checks:

  T-1  the demonstrations move       stationary fraction < 0.05 (was 0.836)
  T-2  the teacher is better         rate > 0.234, Wilson lo > the old teacher
  T-3  it survives cloning           new clone Wilson lo > the old clone's rate

T-2 and T-3 fail for different reasons and point at different next steps:
T-2 failing refutes the whole idea; T-2 passing while T-3 fails isolates the
remaining problem on the hidden-state defect (D79/D105) and makes the
belief-augmented observation the next pre-registration rather than a guess.

No RL training happens here.

Usage: ./venv/bin/python scripts/derisk_t2_teacher.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.eval import wilson_ci, win_count
from pe2.scripted import (
    FovLimitedPursuer,
    GreedyEvader,
    SearchingPursuer,
    run_duel,
)

OUT = pathlib.Path("results/derisk")
OLD_CLONE = pathlib.Path("results/visual_gates/step_05/sac_bc_pursuer.zip")
NEW_CLONE = OUT / "sac_bc_searching_pursuer.zip"

# --- pre-registered (spec §3/§4). Do not tune. ---
N_EVAL = 500
DEMO_EPISODES = 300
BC_EPOCHS = 50
DEMO_SEED0 = 970_000
EVAL_SEED0 = 980_000
T1_MAX_STATIONARY = 0.05
T2_MIN_RATE = 0.234  # the old teacher's recorded rate (D105)


def demo_composition(cfg, teacher_cls, seed0: int) -> tuple[list, dict]:
    """What behaviour cloning actually trains on. This is check T-1."""
    from pe2.exploitability import collect_demonstrations

    demos = collect_demonstrations(
        cfg, "pursuer", teacher_cls, GreedyEvader, DEMO_EPISODES, seed0
    )
    acts = np.array([d[1] for d in demos])
    obs = np.array([d[0] for d in demos])
    v = (acts[:, 0] + 1.0) / 2.0 * cfg.v_max
    return demos, {
        "teacher": teacher_cls.__name__,
        "n_steps": len(acts),
        "n_episodes": DEMO_EPISODES,
        "demo_seed0": seed0,
        "stationary_fraction": float((v < 1e-6).mean()),
        "full_speed_fraction": float((v > cfg.v_max - 1e-6).mean()),
        "mean_speed": float(v.mean()),
        "visible_fraction": float((obs[:, 6] > 0.5).mean()),
    }


def score_scripted(cfg, pursuer_cls) -> dict:
    """Scripted policies go through `run_duel`, which owns per-episode policy
    lifetime (they carry state such as `last_seen`)."""
    env = PursuitEvasionEnv(cfg)
    k = sum(
        run_duel(env, pursuer_cls(), GreedyEvader(), EVAL_SEED0 + i)["outcome"]
        == "capture"
        for i in range(N_EVAL)
    )
    lo, hi = wilson_ci(k, N_EVAL)
    return {
        "k": k,
        "n": N_EVAL,
        "rate": k / N_EVAL,
        "wilson": [lo, hi],
        "path": "run_duel",
    }


def score_network(cfg, model) -> dict:
    """Network policies go through the standard paired evaluator. Same seeds
    and same env as `score_scripted`, so the cells are comparable."""
    from pe2.exploitability import evaluate_sb3_policy

    logs = evaluate_sb3_policy(
        model, cfg, "pursuer", GreedyEvader, n_episodes=N_EVAL, seed0=EVAL_SEED0
    )
    k, n = win_count(logs, "pursuer")
    lo, hi = wilson_ci(k, n)
    return {
        "k": k,
        "n": n,
        "rate": k / n,
        "wilson": [lo, hi],
        "path": "evaluate_sb3_policy",
    }


def checks(demos_new: dict, rates: dict) -> dict:
    """The three pre-registered checks, each with the numbers it compared."""
    t1_value = demos_new["stationary_fraction"]
    t1 = {
        "pass": bool(t1_value < T1_MAX_STATIONARY),
        "stationary_fraction": t1_value,
        "threshold": T1_MAX_STATIONARY,
        "reason": (
            f"the new teacher's demonstrations are {t1_value:.3f} stationary "
            f"against a threshold of {T1_MAX_STATIONARY} (the old teacher's "
            f"were 0.836)"
        ),
    }
    new_t, old_t = rates["SearchingPursuer"], rates["FovLimitedPursuer"]
    t2 = {
        "pass": bool(
            new_t["rate"] > T2_MIN_RATE and new_t["wilson"][0] > old_t["rate"]
        ),
        "new_teacher": new_t,
        "old_teacher": old_t,
        "threshold": T2_MIN_RATE,
        "reason": (
            f"SearchingPursuer {new_t['rate']:.3f} (Wilson lo "
            f"{new_t['wilson'][0]:.3f}) vs the {T2_MIN_RATE} threshold and the "
            f"old teacher's {old_t['rate']:.3f} on this same stream"
        ),
    }
    new_c, old_c = rates["clone_searching"], rates["clone_current"]
    t3 = {
        "pass": bool(new_c["wilson"][0] > old_c["rate"]),
        "new_clone": new_c,
        "old_clone": old_c,
        "reason": (
            f"the new clone's Wilson lower bound {new_c['wilson'][0]:.3f} vs "
            f"the current clone's {old_c['rate']:.3f} on the same stream"
        ),
    }
    return {"T1_demos_move": t1, "T2_teacher_better": t2, "T3_survives_cloning": t3}


def direction(c: dict) -> str:
    """Spec §5, fixed before the run."""
    if not c["T1_demos_move"]["pass"]:
        return (
            "T-1 FAILED: the implementation does not do what it says. Fix the "
            "code, not the threshold; no reading of T-2 or T-3 is warranted."
        )
    if not c["T2_teacher_better"]["pass"]:
        return (
            "T-2 FAILED: orbiting is not better than standing still. Hypothesis "
            "T is refuted, the searching-teacher idea is dropped, and nothing "
            "further is attempted without a new idea. Do NOT tune PATROL_RADIUS "
            "or PATROL_LEAD to rescue it (spec §5)."
        )
    if not c["T3_survives_cloning"]["pass"]:
        return (
            "T-2 PASSED, T-3 FAILED: the teacher is fixed but the clone cannot "
            "follow it. That localises the remaining problem squarely on the "
            "hidden-state defect (D79/D105) and makes the belief-augmented "
            "observation the next pre-registration rather than a guess."
        )
    return (
        "T-1, T-2 and T-3 all PASSED: the teacher was the binding constraint. A "
        "clone worth fine-tuning exists for the first time; the learner question "
        "is re-opened and the next pre-registration is an RL round from the new "
        "clone."
    )


def write_figure(demos: dict, rates: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = list(demos)
    ax1.bar(
        np.arange(len(labels)) - 0.2,
        [demos[k]["stationary_fraction"] for k in labels],
        width=0.4,
        label="stationary fraction",
    )
    ax1.bar(
        np.arange(len(labels)) + 0.2,
        [demos[k]["mean_speed"] for k in labels],
        width=0.4,
        label="mean speed (m/s)",
    )
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.axhline(T1_MAX_STATIONARY, color="k", ls="--", lw=0.8, label="T-1 threshold")
    ax1.set_title("what behaviour cloning trains on")
    ax1.legend(fontsize=8)

    names = list(rates)
    vals = [rates[k]["rate"] for k in names]
    err = [
        [rates[k]["rate"] - rates[k]["wilson"][0] for k in names],
        [rates[k]["wilson"][1] - rates[k]["rate"] for k in names],
    ]
    ax2.bar(np.arange(len(names)), vals, yerr=err, capsize=4)
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax2.axhline(
        T2_MIN_RATE, color="k", ls="--", lw=0.8, label=f"old teacher {T2_MIN_RATE}"
    )
    ax2.set_ylabel(f"capture rate vs GreedyEvader (N={N_EVAL}, seeds {EVAL_SEED0}+)")
    ax2.set_title("T2 — teacher and clone")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    from stable_baselines3 import SAC

    from pe2.exploitability import train_bc_actor

    OUT.mkdir(parents=True, exist_ok=True)
    if not OLD_CLONE.exists():
        raise SystemExit(f"current clone missing: {OLD_CLONE}")
    cfg = EnvConfig()

    print("collecting demonstrations from both teachers ...", flush=True)
    _demos_old_list, demos_old = demo_composition(cfg, FovLimitedPursuer, DEMO_SEED0)
    demos_new_list, demos_new = demo_composition(cfg, SearchingPursuer, DEMO_SEED0)
    for d in (demos_old, demos_new):
        print(
            f"  {d['teacher']:22s} stationary {d['stationary_fraction']:.3f} | "
            f"mean speed {d['mean_speed']:.4f} | visible {d['visible_fraction']:.3f} "
            f"| {d['n_steps']} steps",
            flush=True,
        )

    print("cloning the searching teacher (the BC recipe is unchanged) ...", flush=True)
    model, losses = train_bc_actor(
        cfg, "pursuer", GreedyEvader, demos_new_list, seed=0, epochs=BC_EPOCHS
    )
    model.save(NEW_CLONE)
    print(f"  final BC loss {losses[-1]:.5f} -> {NEW_CLONE}", flush=True)

    rates = {}
    print(f"scoring, N={N_EVAL} on seeds {EVAL_SEED0}+ ...", flush=True)
    for name, cls in (
        ("FovLimitedPursuer", FovLimitedPursuer),
        ("SearchingPursuer", SearchingPursuer),
    ):
        rates[name] = score_scripted(cfg, cls)
        print(f"  {name:22s} {rates[name]['rate']:.3f}", flush=True)
    rates["clone_current"] = score_network(cfg, SAC.load(OLD_CLONE, device="cpu"))
    print(f"  {'clone_current':22s} {rates['clone_current']['rate']:.3f}", flush=True)
    rates["clone_searching"] = score_network(cfg, SAC.load(NEW_CLONE, device="cpu"))
    print(
        f"  {'clone_searching':22s} {rates['clone_searching']['rate']:.3f}", flush=True
    )

    c = checks(demos_new, rates)
    payload = {
        "spec": "docs/pathway/searching_teacher_prereg_2026-09-12.md",
        "n_eval": N_EVAL,
        "eval_seed0": EVAL_SEED0,
        "demo_seed0": DEMO_SEED0,
        "demo_episodes": DEMO_EPISODES,
        "bc_epochs": BC_EPOCHS,
        "demos": {"FovLimitedPursuer": demos_old, "SearchingPursuer": demos_new},
        "bc_final_loss_searching": losses[-1],
        "rates": rates,
        "checks": c,
        "direction": direction(c),
        "note": (
            "scripted rows go through run_duel (it owns per-episode policy "
            "lifetime), network rows through evaluate_sb3_policy; same seeds, "
            "same env, so the cells are comparable"
        ),
    }
    (OUT / "t2_teacher.json").write_text(json.dumps(payload, indent=2))
    write_figure(payload["demos"], rates, OUT / "t2_teacher.png")

    print()
    for key, chk in c.items():
        print(f"{key}: {'PASS' if chk['pass'] else 'FAIL'} — {chk['reason']}")
    print(f"\n{payload['direction']}")
    print(f"\nwrote {OUT}/t2_teacher.{{json,png}}")


if __name__ == "__main__":
    main()
