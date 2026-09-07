"""T0.5 final learner round — the last learner pre-registration (D101).

Spec: docs/pathway/t0_final_round_prereg_2026-09-07.md, reproduced here
where it binds. Nothing here may change a threshold, a seed, a budget or a
definition after a number is seen.

Arms (pursuer vs frozen GreedyEvader, probes every 100k on 200 paired seeds
840000+, final probe N=500):
  RE     the untested anchor cell: KL-anchored SAC at beta = 1.0, 1M steps
  RF     the untested AWR cell: adaptive temperature lambda_t = std(A_t), 1M
  RFNEG  RF with the advantage sign flipped, 500k: MUST collapse (control)

Pass = win rate >= 0.137 at every probe from 500k. Improved = final N=500
Wilson lower bound > START_500, the clone's OWN score measured on this
round's eval stream before any arm trains (spec 2; T0.4's bar came from a
different stream, which its final review flagged). Survivor = IMPROVED.
RF is withheld unless RFNEG collapses.

TERMINAL (spec 5): 0 IMPROVED makes Route C final and NO further learner
pre-registration follows.

This runner deliberately duplicates structure from `derisk_t0b_round.py`
rather than sharing a module with it: that script is the artefact that
produced T0.4's recorded result and is frozen. A refactor would change the
code that produced numbers already in the record.

Usage:
  ./venv/bin/python scripts/derisk_t0c_round.py --measure-start   (first, once)
  ./venv/bin/python scripts/derisk_t0c_round.py --arms RE         (one per arm)
  ./venv/bin/python scripts/derisk_t0c_round.py --summarize       (after all)
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

from pe2.awr import LAM_FLOOR
from pe2.config import EnvConfig
from pe2.derisk import (
    CLONE_CI_LO,
    ROUND_PASS_FROM,
    TrainingCaptureCounter,
    control_collapsed,
    round_verdict,
)
from pe2.eval import wilson_ci, win_count
from pe2.scripted import GreedyEvader

OUT = pathlib.Path("results/derisk")
CLONE = pathlib.Path("results/visual_gates/step_05/sac_bc_pursuer.zip")
START_PATH = OUT / "t0c_start.json"

# --- pre-registered (spec 2/3). Do not tune. ---
TOTAL = 1_000_000
EVERY = 100_000
EVAL_N = 200
FINAL_EVAL_N = 500
PASS_FROM = ROUND_PASS_FROM  # 500_000, bound to the library constant
EVAL_SEED0 = 840_000  # fresh: 700000-790999, 800000-830999 are consumed
NEG_TOTAL = 500_000  # C3-a's budget: offline ~306k + four online iterations
BETA_RE = 1.0
CRITIC_WARMUP = 50_000
RF_SEED0 = 850_000
RFNEG_SEED0 = 860_000
SEEDS = {"RE": 7, "RF": 13, "RFNEG": 13}
LEARNER_ARMS = ("RE", "RF")
DETERMINATE_VERDICTS = ("COLLAPSED", "PRESERVED", "IMPROVED")

# Spec 3.2's predicted diagnostics, stated BEFORE the run so the temperature
# specification is falsifiable separately from the learning hypothesis.
PREDICTED_RATIO_1SIGMA = math.exp(2.0)  # 7.389
SPEC_TOL = 0.05  # relative


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
        report_path=OUT / f"t0c_probe_{arm}.json",
        label=arm,
    )


def measure_start(cfg, device) -> dict:
    """The bar every arm is judged against: the gated clone's OWN score at
    N=500 on THIS round's eval stream, measured once before any training.

    T0.4 judged its arms against a bar taken from stream 720000+ while its
    probes ran on 780000+; its final review flagged the mismatch (it happened
    to disfavour the arms). Measuring it here on the same seeds the probes
    use makes the comparison paired by construction.
    """
    from stable_baselines3 import SAC

    from pe2.exploitability import evaluate_sb3_policy

    model = SAC.load(CLONE, device=device)
    logs = evaluate_sb3_policy(
        model, cfg, "pursuer", GreedyEvader, n_episodes=FINAL_EVAL_N, seed0=EVAL_SEED0
    )
    k, n = win_count(logs, "pursuer")
    lo, hi = wilson_ci(k, n)
    payload = {
        "k": k,
        "n": n,
        "rate": k / n,
        "wilson": [lo, hi],
        "eval_seed0": EVAL_SEED0,
        "clone": str(CLONE),
    }
    START_PATH.write_text(json.dumps(payload, indent=2))
    print(f"START_500 = {k}/{n} = {k / n:.4f}, Wilson [{lo:.4f}, {hi:.4f}]", flush=True)
    return payload


def load_start() -> dict:
    """Refuse to judge an arm against a bar that was never measured."""
    if not START_PATH.exists():
        raise SystemExit(
            f"start bar missing: {START_PATH}. Run --measure-start FIRST; the "
            "pre-registration requires it to exist before any arm trains."
        )
    return json.loads(START_PATH.read_text())


def anchor_pattern(verdict: str, kl_grew) -> tuple[str, str]:
    """Spec 3.1's pre-committed reading of RE's outcome pattern.

    Fixed before the run so the mechanism's interpretation is not chosen
    after seeing which pattern occurred. The VERDICT is decided by the
    protocol's rules; this only names what the round concludes about the
    mechanism.
    """
    if verdict == "IMPROVED":
        return "improved", (
            "a survivor: beta = 1.0 at 1M is the Track-1 learner and Track 2 "
            "is planned for it"
        )
    if verdict == "PRESERVED" and kl_grew is False:
        return "pinned", (
            "the strong anchor genuinely cannot learn even given 5x the budget; "
            "with RB's collapse at beta = 0.1 this closes the dichotomy D100 "
            "correctly refused to claim on the evidence then available"
        )
    if verdict == "COLLAPSED" and kl_grew is True:
        return "late_collapse", (
            "supports the beta-continuum reading: anchor strength trades "
            "time-to-collapse against learning rate and no strength tested "
            "escapes, so Route C's claim strengthens from 'these settings "
            "failed' to 'the mechanism has no operating point'"
        )
    return "other", (
        f"verdict {verdict} with kl_grew={kl_grew} matches none of the three "
        "patterns the pre-registration named; report the pattern, do not force "
        "it into one of them"
    )


def spec_check(iterations: list[dict]) -> dict:
    """Did the adaptive temperature behave as spec 3.2 predicted?

    Falsifies the TEMPERATURE SPECIFICATION independently of the learning
    hypothesis: if the observed weight ratios do not match, the
    implementation is wrong and the arm is UNDETERMINED on those grounds,
    not evidence about AWR. Iterations where the floor engaged are excluded
    (the ratio is legitimately smaller there) and counted.
    """
    active = [it for it in iterations if it["lam_used"] > LAM_FLOOR]
    floored = len(iterations) - len(active)
    if not active:
        return {
            "pass": None,
            "n_active": 0,
            "n_floored": floored,
            "reason": "no iteration ran above the temperature floor",
        }
    worst = max(
        active, key=lambda it: abs(it["weight_ratio_1sigma"] - PREDICTED_RATIO_1SIGMA)
    )
    dev = (
        abs(worst["weight_ratio_1sigma"] - PREDICTED_RATIO_1SIGMA)
        / PREDICTED_RATIO_1SIGMA
    )
    return {
        "pass": dev <= SPEC_TOL,
        "n_active": len(active),
        "n_floored": floored,
        "predicted_ratio_1sigma": PREDICTED_RATIO_1SIGMA,
        "worst_observed_ratio_1sigma": worst["weight_ratio_1sigma"],
        "worst_relative_deviation": dev,
        "reason": (
            f"worst +/-1 sigma weight ratio {worst['weight_ratio_1sigma']:.4f} vs "
            f"predicted {PREDICTED_RATIO_1SIGMA:.4f} "
            f"({dev:.1%} deviation, tolerance {SPEC_TOL:.0%})"
        ),
    }


def final_eval(cfg, arm, total, device) -> dict:
    """The N=500 improvement probe, ALWAYS from the saved final checkpoint so
    the number is tied to an artefact a reader can re-run.

    Every arm of this round is SAC-family: RE is a `KLAnchoredSAC` (a SAC
    subclass, so `SAC.load` reads it) and RF/RFNEG update the clone's own SAC
    actor in place.
    """
    from stable_baselines3 import SAC

    from pe2.exploitability import evaluate_sb3_policy

    path = OUT / f"ckpts_{arm}" / f"{arm}_{total}.zip"
    model = SAC.load(path, device=device)
    logs = evaluate_sb3_policy(
        model, cfg, "pursuer", GreedyEvader, n_episodes=FINAL_EVAL_N, seed0=EVAL_SEED0
    )
    k, n = win_count(logs, "pursuer")
    lo, hi = wilson_ci(k, n)
    return {
        "ckpt": str(path),
        "k": k,
        "n": n,
        "win_rate": k / n,
        "wilson": [lo, hi],
        "eval_seed0": EVAL_SEED0,
    }


def run_arm(arm: str, cfg: EnvConfig, device: str) -> dict:
    from stable_baselines3.common.callbacks import CallbackList

    import pe2.awr as awr_mod
    import pe2.exploitability as expl
    from pe2.anchored_sac import AnchorKLProbe, anchor_kl_grew

    # FIRST, so a missing bar refuses before any training happens.
    start = load_start()
    total = NEG_TOTAL if arm == "RFNEG" else TOTAL
    pr = probe(cfg, arm)
    # snapshot_every=CRITIC_WARMUP (50k), not EVERY (100k): 50k divides 100k
    # so the probe grid is untouched, but it also lands ON RE's phase-A/B
    # boundary, restoring the decomposition `segments` exists to give.
    counter = TrainingCaptureCounter(snapshot_every=CRITIC_WARMUP)
    extra: dict = {}
    grew = None
    if arm == "RE":
        cb = CallbackList(
            [pr.callback, counter.callback, AnchorKLProbe(pr.rows, EVERY).callback]
        )
        expl.finetune_br_anchored(
            cfg,
            "pursuer",
            GreedyEvader,
            CLONE,
            total,
            seed=SEEDS[arm],
            beta=BETA_RE,
            device=device,
            critic_warmup=CRITIC_WARMUP,
            callback=cb,
        )
        training = dict(
            counter.state,
            positives_per_1M=counter.positives_per_1M,
            segments=counter.segments,
        )
        grew, why = anchor_kl_grew(pr.rows)
        extra["anchor"] = {"beta": BETA_RE, "kl_grew": grew, "kl_reason": why}
    elif arm in ("RF", "RFNEG"):
        awr = awr_mod.AwrConfig(
            adaptive_lam=True, advantage_sign=1.0 if arm == "RF" else -1.0
        )
        _, stats = awr_mod.train_awr(
            cfg,
            "pursuer",
            GreedyEvader,
            CLONE,
            total,
            seed=SEEDS[arm],
            awr=awr,
            probe_every=EVERY,
            probe=pr.evaluate_at,
            device=device,
            rollout_seed0=RF_SEED0 if arm == "RF" else RFNEG_SEED0,
        )
        training = {
            k: stats[k]
            for k in ("total_steps", "n_episodes", "positives", "positives_per_1M")
        }
        extra["awr"] = stats
        extra["spec_check"] = spec_check(stats["iterations"])
    else:
        raise SystemExit(f"unknown arm {arm}")

    result = {
        "arm": arm,
        "seed": SEEDS[arm],
        "timesteps": total,
        "eval_n": EVAL_N,
        "eval_seed0": EVAL_SEED0,
        "start": {
            "k": start["k"],
            "n": start["n"],
            "rate": start["rate"],
            "eval_seed0": start["eval_seed0"],
            "provenance": (
                "measured on this round's eval stream before any training "
                "(spec 2); results/derisk/t0c_start.json"
            ),
        },
        "checkpoints": pr.rows,
        "training": training,
        **extra,
    }
    if arm == "RFNEG":
        verdict, reason = control_collapsed(pr.rows, from_step=EVERY)
    else:
        # A crashed arm must not get a fabricated final probe: final_eval
        # ALWAYS loads the saved final checkpoint, and if the run died
        # before writing it, that file is absent. Judge as UNDETERMINED
        # rather than raising and losing the probe history gathered so far.
        ckpt_path = OUT / f"ckpts_{arm}" / f"{arm}_{total}.zip"
        if not ckpt_path.exists():
            verdict, reason = (
                "UNDETERMINED",
                (
                    f"final checkpoint missing: {ckpt_path} was never written, so "
                    "the run did not reach its final step (crashed or was killed "
                    "before completion); judged with no final probe rather than "
                    "losing the probe history gathered before the crash"
                ),
            )
        else:
            result["final"] = final_eval(cfg, arm, total, device)
            verdict, reason = round_verdict(
                pr.rows,
                result["final"]["k"],
                result["final"]["n"],
                start["rate"],
                from_step=PASS_FROM,
            )
    result["verdict"], result["reason"] = verdict, reason
    if arm == "RE":
        # AFTER the verdict: spec 3.1's reading is a function of the outcome.
        result["anchor_pattern"], result["anchor_reading"] = anchor_pattern(
            verdict, grew
        )
    (OUT / f"t0c_arm_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] verdict={verdict}: {reason}", flush=True)
    return result


def load_results() -> dict:
    return {
        p.stem.removeprefix("t0c_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob("t0c_arm_*.json"))
    }


def summarize(results: dict) -> dict:
    """Apply the withhold rule and the spec-5 direction. Pure: no I/O."""
    arms = {
        a: {"verdict": r["verdict"], "reason": r["reason"]} for a, r in results.items()
    }
    ctrl = results.get("RFNEG")
    rf_withheld = False
    control_finding = ""
    if "RF" in arms:
        if ctrl is None:
            arms["RF"] = {
                "verdict": "UNDETERMINED",
                "reason": (
                    "withheld: the AWR control (RFNEG) has not run, so the AWR "
                    "loop and its probe path are unvalidated"
                ),
            }
            rf_withheld = True
        elif ctrl["verdict"] is not True:
            arms["RF"] = {
                "verdict": "UNDETERMINED",
                "reason": (
                    "withheld: the AWR control did not demonstrably collapse "
                    f"({ctrl['reason']})"
                ),
            }
            rf_withheld = True
            # Spec 3.3: with the weighting ENGAGED (unlike T0.4's inert
            # lambda), a control that still fails to discriminate is a
            # finding in its own right, not another null.
            control_finding = (
                "the control did not discriminate even with the weighting "
                "engaged (spec 3.3): the policy class is INSENSITIVE to the "
                "sign of the advantage on this task, i.e. self-imitation "
                "dominates the update regardless of weighting. Report this as "
                "a finding, not as another null."
            )
    survivors = [
        a for a in LEARNER_ARMS if a in arms and arms[a]["verdict"] == "IMPROVED"
    ]
    # `complete` must mean every learner arm was actually JUDGED, not merely
    # that a file exists: a crashed arm's own UNDETERMINED verdict must not
    # feed the round-ending Route-C conclusion. RF is the one pre-registered
    # exception -- a legitimately withheld RF still leaves the round decided
    # on RE alone -- so it does not block completeness.
    learners_judged = all(
        a in arms
        and (arms[a]["verdict"] in DETERMINATE_VERDICTS or (a == "RF" and rf_withheld))
        for a in LEARNER_ARMS
    )
    complete = all(a in results for a in (*LEARNER_ARMS, "RFNEG")) and learners_judged
    if survivors:
        best = max(
            survivors,
            key=lambda a: wilson_ci(results[a]["final"]["k"], results[a]["final"]["n"])[
                0
            ],
        )
        direction = (
            f"IMPROVED: {survivors}. Track-1 mechanism = {best} (highest final "
            "Wilson lower bound). Write the Track 2 (evader) plan for it; two "
            "survivors run as Track-2 arms."
        )
    elif complete:
        direction = (
            "No arm IMPROVED. Route C is FINAL: the characterisation paper is "
            "written (shield guarantee D97, exact-game shift T3, and a learning "
            "failure independent of reward shaping, algorithm family, anchoring, "
            "bootstrapping and the weighting's calibration). No further learner "
            "pre-registration follows (spec 5)."
        )
    else:
        direction = "incomplete: not every learner arm has a determinate verdict"
    summary = {
        "spec": "docs/pathway/t0_final_round_prereg_2026-09-07.md",
        "clone_ci_lo": CLONE_CI_LO,
        "pass_from": PASS_FROM,
        "final_eval_n": FINAL_EVAL_N,
        "arms": arms,
        "survivors": survivors,
        "complete": complete,
        "control_finding": control_finding,
        "direction": direction,
    }
    if "RE" in results and "anchor_pattern" in results["RE"]:
        summary["anchor_pattern"] = results["RE"]["anchor_pattern"]
        summary["anchor_reading"] = results["RE"]["anchor_reading"]
    return summary


def _verdict_label(verdict) -> str:
    """RFNEG's verdict is True/False/None (`control_collapsed`'s shape), not
    one of the VERDICTS strings the other arms use -- give it words instead
    of a literal "RFNEG (True)" on a signed visual gate."""
    if verdict is True:
        return "collapsed"
    if verdict is False:
        return "did not collapse"
    if verdict is None:
        return "undetermined"
    return verdict


def write_figure(results: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    start_rate = None
    for arm, r in results.items():
        rows = r["checkpoints"]
        if rows:
            ax.plot(
                [x["steps"] for x in rows],
                [x["win_rate"] for x in rows],
                marker="o",
                label=f"{arm} ({_verdict_label(summary['arms'][arm]['verdict'])})",
            )
        if "final" in r:
            lo, hi = r["final"]["wilson"]
            ax.errorbar(
                [r["timesteps"] + 20_000],
                [r["final"]["win_rate"]],
                yerr=[[r["final"]["win_rate"] - lo], [hi - r["final"]["win_rate"]]],
                fmt="s",
                capsize=3,
            )
        start_rate = r["start"]["rate"]
    if start_rate is not None:
        ax.axhline(
            start_rate,
            color="tab:green",
            ls=":",
            lw=1.2,
            label=f"START_500 = {start_rate:.3f} (improvement bar)",
        )
    ax.axhline(CLONE_CI_LO, color="k", ls="--", label=f"clone CI-lo {CLONE_CI_LO}")
    ax.axvline(
        PASS_FROM, color="grey", ls="-.", lw=0.8, label=f"pass window from {PASS_FROM}"
    )
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"win rate vs GreedyEvader (paired seeds {EVAL_SEED0}+)")
    title = "T0.5 final round" + (
        "" if summary["complete"] else " — INCOMPLETE, do not read as a result"
    )
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(SEEDS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig()
    if args.measure_start:
        if not CLONE.exists():
            raise SystemExit(f"gated clone missing: {CLONE}")
        measure_start(cfg, args.device)
        return
    for arm in args.arms:
        if not CLONE.exists():
            raise SystemExit(f"gated clone missing: {CLONE}")
        run_arm(arm, cfg, args.device)
    # Written ONLY under --summarize: each arm process writes its own
    # t0c_arm_{ARM}.json, but t0c_round.{json,png} aggregates ALL of them and
    # concurrent arm processes would otherwise interleave on this shared pair.
    if args.summarize:
        results = load_results()
        summary = summarize(results)
        (OUT / "t0c_round.json").write_text(
            json.dumps({"summary": summary, "arms": results}, indent=2)
        )
        write_figure(results, summary, OUT / "t0c_round.png")
        print(json.dumps(summary["arms"], indent=2))
        if summary.get("control_finding"):
            print(f"control finding: {summary['control_finding']}")
        if "anchor_pattern" in summary:
            print(
                f"RE pattern: {summary['anchor_pattern']} — {summary['anchor_reading']}"
            )
        print(f"survivors={summary['survivors']}\n{summary['direction']}")


if __name__ == "__main__":
    main()
