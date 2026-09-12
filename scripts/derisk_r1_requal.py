"""R1 — does RL improve on a COMPETENT prior? (D106/D107; requalifies D100/D102/D103).

Spec: docs/pathway/requalification_prereg_2026-09-12.md. Nothing here may
change a threshold, a seed or a budget after a number is seen.

Every learner round this project ran fine-tuned the 0.212 clone, which D105
showed is an imitation of a teacher standing still on 92.6 % of its steps.
D106 produced a 0.476 clone; D107 showed its advantage holds on every
opponent that is catchable at all. So four rounds are evidence about
fine-tuning a near-stationary prior, not about whether RL works here.

Two arms, ONE variable — the prior:
  RQNEW  finetune_br from the 0.476 searching clone   <- the test
  RQOLD  finetune_br from the 0.212 original clone    <- matched control

Same learner, budget, seeds, opponent and stream. Pass = win rate >= 0.137 at
every probe from 500k. IMPROVED = final N=500 Wilson lower bound > START_NEW,
the new clone's own score measured on this stream before any training.

Usage:
  ./venv/bin/python scripts/derisk_r1_requal.py --measure-start   (first, once)
  ./venv/bin/python scripts/derisk_r1_requal.py --arms RQNEW      (one per arm)
  ./venv/bin/python scripts/derisk_r1_requal.py --summarize       (after both)
"""

from __future__ import annotations

import argparse
import json
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import (
    CLONE_CI_LO,
    ROUND_PASS_FROM,
    TrainingCaptureCounter,
    round_verdict,
)
from pe2.eval import wilson_ci, win_count
from pe2.scripted import GreedyEvader

OUT = pathlib.Path("results/derisk")
NEW_CLONE = OUT / "sac_bc_searching_pursuer.zip"
OLD_CLONE = pathlib.Path("results/visual_gates/step_05/sac_bc_pursuer.zip")
START_PATH = OUT / "r1_start.json"

# --- pre-registered (spec §2/§3). Do not tune. ---
TOTAL = 1_000_000
EVERY = 100_000
EVAL_N = 200
FINAL_EVAL_N = 500
PASS_FROM = ROUND_PASS_FROM  # 500_000, bound to the library constant
EVAL_SEED0 = 990_000  # fresh: <= 980999 consumed
CRITIC_WARMUP = 50_000
GAMMA = 0.999
SEED = 7
ARMS = {"RQNEW": NEW_CLONE, "RQOLD": OLD_CLONE}


def probe(cfg, arm):
    from pe2.exploitability import CheckpointEvalProbe

    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    return CheckpointEvalProbe(
        cfg,
        "pursuer",
        GreedyEvader,
        every=EVERY,
        eval_n=EVAL_N,
        seed0=EVAL_SEED0,
        ckpt_dir=d,
        report_path=OUT / f"r1_probe_{arm}.json",
        label=arm,
    )


def measure_start(device) -> dict:
    """Both priors scored on THIS stream before any training. START_NEW is the
    improvement bar; the old clone's rate is recorded for the comparison to the
    four earlier rounds."""
    from stable_baselines3 import SAC

    from pe2.exploitability import evaluate_sb3_policy

    cfg = EnvConfig()
    out = {}
    for name, path in (("START_NEW", NEW_CLONE), ("START_OLD", OLD_CLONE)):
        model = SAC.load(path, device=device)
        logs = evaluate_sb3_policy(
            model,
            cfg,
            "pursuer",
            GreedyEvader,
            n_episodes=FINAL_EVAL_N,
            seed0=EVAL_SEED0,
        )
        k, n = win_count(logs, "pursuer")
        lo, hi = wilson_ci(k, n)
        out[name] = {
            "k": k,
            "n": n,
            "rate": k / n,
            "wilson": [lo, hi],
            "clone": str(path),
            "eval_seed0": EVAL_SEED0,
        }
        print(
            f"{name} = {k}/{n} = {k / n:.4f}, Wilson [{lo:.4f}, {hi:.4f}]", flush=True
        )
    START_PATH.write_text(json.dumps(out, indent=2))
    return out


def load_start() -> dict:
    if not START_PATH.exists():
        raise SystemExit(
            f"start bar missing: {START_PATH}. Run --measure-start FIRST; the "
            "pre-registration requires it before any arm trains."
        )
    s = json.loads(START_PATH.read_text())
    for k in ("START_NEW", "START_OLD"):
        if k not in s:
            raise SystemExit(f"{k} missing from {START_PATH}; re-run --measure-start.")
    return s


def final_eval(cfg, arm, device) -> dict:
    from stable_baselines3 import SAC

    from pe2.exploitability import evaluate_sb3_policy

    path = OUT / f"ckpts_{arm}" / f"{arm}_{TOTAL}.zip"
    model = SAC.load(path, device=device)
    logs = evaluate_sb3_policy(
        model, cfg, "pursuer", GreedyEvader, n_episodes=FINAL_EVAL_N, seed0=EVAL_SEED0
    )
    k, n = win_count(logs, "pursuer")
    lo, hi = wilson_ci(k, n)
    return {"ckpt": str(path), "k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


def run_arm(arm: str, device: str) -> dict:
    from stable_baselines3.common.callbacks import CallbackList

    import pe2.exploitability as expl

    starts = load_start()  # FIRST: refuse before training if the bar is missing
    cfg = EnvConfig()
    pr = probe(cfg, arm)
    counter = TrainingCaptureCounter(snapshot_every=CRITIC_WARMUP)
    expl.finetune_br(
        cfg,
        "pursuer",
        GreedyEvader,
        ARMS[arm],
        TOTAL,
        seed=SEED,
        device=device,
        gamma=GAMMA,
        critic_warmup=CRITIC_WARMUP,
        callback=CallbackList([pr.callback, counter.callback]),
    )
    # Both arms are judged against the SAME bar, START_NEW: the round asks
    # whether RL reaches what a competent prior already achieves.
    bar = starts["START_NEW"]["rate"]
    result = {
        "arm": arm,
        "prior": str(ARMS[arm]),
        "seed": SEED,
        "timesteps": TOTAL,
        "gamma": GAMMA,
        "eval_n": EVAL_N,
        "eval_seed0": EVAL_SEED0,
        "bar": {"name": "START_NEW", "rate": bar},
        "start_old": starts["START_OLD"]["rate"],
        "checkpoints": pr.rows,
        "training": dict(
            counter.state,
            positives_per_1M=counter.positives_per_1M,
            segments=counter.segments,
        ),
    }
    ckpt = OUT / f"ckpts_{arm}" / f"{arm}_{TOTAL}.zip"
    if not ckpt.exists():
        verdict = "UNDETERMINED"
        reason = (
            f"final checkpoint missing: {ckpt} was never written, so the run did "
            "not reach its final step; judged with no final probe rather than "
            "losing the probe history gathered before the crash"
        )
    else:
        result["final"] = final_eval(cfg, arm, device)
        verdict, reason = round_verdict(
            pr.rows,
            result["final"]["k"],
            result["final"]["n"],
            bar,
            from_step=PASS_FROM,
        )
    result["verdict"], result["reason"] = verdict, reason
    (OUT / f"r1_arm_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] verdict={verdict}: {reason}", flush=True)
    return result


def load_results() -> dict:
    return {
        p.stem.removeprefix("r1_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob("r1_arm_*.json"))
    }


def summarize(results: dict) -> dict:
    """Spec §4, fixed before the run."""
    v = {a: r["verdict"] for a, r in results.items()}
    new, old = v.get("RQNEW"), v.get("RQOLD")
    if new is None or old is None:
        direction = "incomplete: both arms must run"
    elif new == "IMPROVED" and old == "COLLAPSED":
        direction = (
            "RQ-NEW IMPROVED and RQ-OLD collapsed: the collapse was a property of "
            "the PRIOR. D100/D102/D103 are formally requalified — re-scope them to "
            "'fine-tuning a near-stationary prior', WITHDRAW Route C as the "
            "direction, and the next pre-registration is the learner round four "
            "rounds were trying to run, now from a prior worth improving."
        )
    elif old == "IMPROVED":
        direction = (
            "RQ-OLD IMPROVED: unexpected, and it contradicts four rounds on this "
            "same configuration. Report it, change nothing else, and STOP — "
            "something about this stream or build differs and must be found "
            "before any further reading (spec §4)."
        )
    elif new == "COLLAPSED":
        direction = (
            "BOTH COLLAPSED: the negative result SURVIVES and is STRENGTHENED — the "
            "collapse holds from a prior that ranks first of eight on every "
            "catchable opponent, so it is a property of the learner, not of the "
            "demonstration data. D100/D102/D103 are re-scoped but their conclusion "
            "stands, and Route C is restated on this stronger footing."
        )
    elif new == "PRESERVED":
        direction = (
            "RQ-NEW PRESERVED: the competent prior is preserved where the weak one "
            "collapsed — a real and reportable difference — but RL still adds "
            "nothing. Route C stands, and anchoring's role (D102) is re-read: "
            "preservation may have been about the prior all along."
        )
    else:
        direction = (
            f"RQ-NEW {new}, RQ-OLD {old}: this pattern matches none the "
            "pre-registration named; report it as observed, do not force it."
        )
    return {
        "spec": "docs/pathway/requalification_prereg_2026-09-12.md",
        "arms": {
            a: {"verdict": r["verdict"], "reason": r["reason"]}
            for a, r in results.items()
        },
        "complete": new is not None and old is not None,
        "direction": direction,
    }


def write_figure(results: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for arm, r in results.items():
        rows = r["checkpoints"]
        if rows:
            ax.plot(
                [x["steps"] for x in rows],
                [x["win_rate"] for x in rows],
                marker="o",
                label=f"{arm} ({summary['arms'][arm]['verdict']})",
            )
        if "final" in r:
            lo, hi = r["final"]["wilson"]
            ax.errorbar(
                [r["timesteps"] + 20_000],
                [r["final"]["rate"]],
                yerr=[[r["final"]["rate"] - lo], [hi - r["final"]["rate"]]],
                fmt="s",
                capsize=3,
            )
    any_r = next(iter(results.values()), None)
    if any_r:
        ax.axhline(
            any_r["bar"]["rate"],
            color="tab:green",
            ls=":",
            lw=1.3,
            label=f"START_NEW = {any_r['bar']['rate']:.3f} (the bar)",
        )
        ax.axhline(
            any_r["start_old"],
            color="tab:red",
            ls=":",
            lw=1.3,
            label=f"old clone = {any_r['start_old']:.3f}",
        )
    ax.axhline(CLONE_CI_LO, color="k", ls="--", label=f"collapse floor {CLONE_CI_LO}")
    ax.axvline(
        PASS_FROM, color="grey", ls="-.", lw=0.8, label=f"pass window from {PASS_FROM}"
    )
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"win rate vs GreedyEvader (paired seeds {EVAL_SEED0}+)")
    ax.set_title(
        "R1 — requalification: RL from a competent prior"
        + ("" if summary["complete"] else " — INCOMPLETE")
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for path in ARMS.values():
        if not pathlib.Path(path).exists():
            raise SystemExit(f"prior missing: {path}")
    if args.measure_start:
        measure_start(args.device)
        return
    for arm in args.arms:
        run_arm(arm, args.device)
    if args.summarize:
        results = load_results()
        summary = summarize(results)
        (OUT / "r1_round.json").write_text(
            json.dumps({"summary": summary, "arms": results}, indent=2)
        )
        write_figure(results, summary, OUT / "r1_round.png")
        print(json.dumps(summary["arms"], indent=2))
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
