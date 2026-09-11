"""T1.0 horizon round — does the TASK explain the learning failure? (D103).

Spec: docs/pathway/t1_horizon_round_prereg_2026-09-08.md, reproduced here
where it binds. Nothing here may change a threshold, a seed, a budget, a
horizon or a definition after a number is seen.

Every prior round varied the LEARNER against a fixed task. This one does the
opposite: one fixed learner (`finetune_br`, the proven LayerNorm-critic /
ent_coef=1e-5 / 50k-warm-up recipe), one environment parameter moved.

The premise, measured on the clone's own 1M-step audit: the median capture
lands at step 220 and 95 % of captures are in by step 400, while the episode
runs to 1200. The last ~900 steps are both agents milling about after the
outcome is settled -- they triple the credit-assignment horizon and dilute
the per-step signal to a third.

  HA     400 steps, gamma 0.999    -- the primary test: episode length ALONE
  HC    1200 steps, gamma 0.999    -- the matched control for the horizon claim
  HB     400 steps, gamma 0.9975   -- adds the discount matched to 400 steps
  HNEG   400 steps, gamma 0.999, REWARD NEGATED -- must collapse (control)

Pass = win rate >= 0.137 at every probe from 500k. Improved = final N=500
Wilson lower bound > that arm's OWN start bar (START_400 or START_1200, both
measured from the gated clone before any arm trains). Survivor = IMPROVED.

If HNEG does not collapse, EVERY learner arm is withheld: this control
validates the harness at a NEW episode length, so its failure invalidates all
three, not one.

On 0 IMPROVED the direction is "Route C stands as D102 left it, strengthened"
-- not a new claim, the old one with one more lever tested.

This runner deliberately duplicates structure from `derisk_t0c_round.py`
rather than sharing a module with it: the earlier runners are the artefacts
that produced recorded results and are frozen.

Usage:
  ./venv/bin/python scripts/derisk_t1_round.py --measure-starts  (first, once)
  ./venv/bin/python scripts/derisk_t1_round.py --arms HA         (one per arm)
  ./venv/bin/python scripts/derisk_t1_round.py --summarize       (after all)
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
    control_collapsed,
    round_verdict,
)
from pe2.eval import wilson_ci, win_count
from pe2.rewards import NegatedRewardWrapper
from pe2.scripted import GreedyEvader

OUT = pathlib.Path("results/derisk")
CLONE = pathlib.Path("results/visual_gates/step_05/sac_bc_pursuer.zip")
START_PATH = OUT / "t1_start.json"

# --- pre-registered (spec 2/3). Do not tune. ---
TOTAL = 1_000_000
EVERY = 100_000
EVAL_N = 200
FINAL_EVAL_N = 500
PASS_FROM = ROUND_PASS_FROM  # 500_000, bound to the library constant
CRITIC_WARMUP = 50_000
SEED = 7  # A1's off-policy seed, as every prior SAC arm used
SHORT_STEPS = 400  # keeps 95 % of captures, 3x shorter credit horizon
LONG_STEPS = 1200  # the original task
GAMMA_ORIG = 0.999  # effective horizon 1000 ~ the original episode
GAMMA_MATCHED = 0.9975  # 1/(1-g) = 400, the same rule at the new length
SHORT_SEED0 = 870_000  # fresh: <= 860999 is consumed
LONG_SEED0 = 880_000
LEARNER_ARMS = ("HA", "HB", "HC")
DETERMINATE_VERDICTS = ("COLLAPSED", "PRESERVED", "IMPROVED")

ARMS = {
    # arm: (max_steps, gamma, eval seed stream, which bar it is judged against)
    "HA": (SHORT_STEPS, GAMMA_ORIG, SHORT_SEED0, "START_400"),
    "HB": (SHORT_STEPS, GAMMA_MATCHED, SHORT_SEED0, "START_400"),
    "HC": (LONG_STEPS, GAMMA_ORIG, LONG_SEED0, "START_1200"),
    "HNEG": (SHORT_STEPS, GAMMA_ORIG, SHORT_SEED0, "START_400"),
}
BARS = {
    "START_400": (SHORT_STEPS, SHORT_SEED0),
    "START_1200": (LONG_STEPS, LONG_SEED0),
}


def arm_config(arm: str):
    """The ONLY thing that differs between arms is the task (and, for the
    control, the training reward's sign). The learner is a constant."""
    steps, gamma, seed0, bar = ARMS[arm]
    return EnvConfig(max_steps=steps), gamma, seed0, bar


def probe(cfg, arm, seed0):
    from pe2.exploitability import CheckpointEvalProbe

    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    return CheckpointEvalProbe(
        cfg,
        "pursuer",
        GreedyEvader,
        every=EVERY,
        eval_n=EVAL_N,
        seed0=seed0,
        ckpt_dir=d,
        report_path=OUT / f"t1_probe_{arm}.json",
        label=arm,
    )


def measure_starts(device) -> dict:
    """Both bars, from the SAME gated clone, each on its own horizon and
    stream, measured once before any arm trains.

    Each arm is judged against the bar for ITS horizon: a 400-step episode
    gives the clone fewer chances than a 1200-step one, so a single shared
    bar would compare arms against a task they did not face.
    """
    from stable_baselines3 import SAC

    from pe2.exploitability import evaluate_sb3_policy

    model = SAC.load(CLONE, device=device)
    payload = {}
    for name, (steps, seed0) in BARS.items():
        cfg = EnvConfig(max_steps=steps)
        logs = evaluate_sb3_policy(
            model, cfg, "pursuer", GreedyEvader, n_episodes=FINAL_EVAL_N, seed0=seed0
        )
        k, n = win_count(logs, "pursuer")
        lo, hi = wilson_ci(k, n)
        payload[name] = {
            "k": k,
            "n": n,
            "rate": k / n,
            "wilson": [lo, hi],
            "eval_seed0": seed0,
            "max_steps": steps,
        }
        print(
            f"{name} = {k}/{n} = {k / n:.4f}, Wilson [{lo:.4f}, {hi:.4f}] "
            f"(max_steps {steps}, seeds {seed0}+)",
            flush=True,
        )
    START_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def load_starts() -> dict:
    """Refuse to judge an arm against a bar that was never measured."""
    if not START_PATH.exists():
        raise SystemExit(
            f"start bar missing: {START_PATH}. Run --measure-starts FIRST; the "
            "pre-registration requires both bars to exist before any arm trains."
        )
    starts = json.loads(START_PATH.read_text())
    for name in BARS:
        if name not in starts or "rate" not in starts.get(name, {}):
            raise SystemExit(
                f"start bar {name} missing from {START_PATH}: re-run "
                "--measure-starts; both bars are required before any arm trains."
            )
    return starts


def final_eval(cfg, arm, seed0, device) -> dict:
    """The N=500 improvement probe, ALWAYS from the saved final checkpoint so
    the number is tied to an artefact a reader can re-run."""
    from stable_baselines3 import SAC

    from pe2.exploitability import evaluate_sb3_policy

    path = OUT / f"ckpts_{arm}" / f"{arm}_{TOTAL}.zip"
    model = SAC.load(path, device=device)
    logs = evaluate_sb3_policy(
        model, cfg, "pursuer", GreedyEvader, n_episodes=FINAL_EVAL_N, seed0=seed0
    )
    k, n = win_count(logs, "pursuer")
    lo, hi = wilson_ci(k, n)
    return {
        "ckpt": str(path),
        "k": k,
        "n": n,
        "win_rate": k / n,
        "wilson": [lo, hi],
        "eval_seed0": seed0,
    }


def run_arm(arm: str, device: str) -> dict:
    from stable_baselines3.common.callbacks import CallbackList

    import pe2.exploitability as expl

    # FIRST, so a missing bar refuses before any training happens.
    starts = load_starts()
    cfg, gamma, seed0, bar = arm_config(arm)
    pr = probe(cfg, arm, seed0)
    # snapshot_every=CRITIC_WARMUP (50k), not EVERY (100k): 50k divides 100k
    # so the probe grid is untouched, but it also lands ON the phase-A/B
    # boundary, restoring the decomposition `segments` exists to give.
    counter = TrainingCaptureCounter(snapshot_every=CRITIC_WARMUP)
    cb = CallbackList([pr.callback, counter.callback])
    # Only the control wraps the TRAINING env; every probe and the final
    # evaluation build their own unshaped wrapper (D68's rule).
    env_wrap = NegatedRewardWrapper if arm == "HNEG" else None
    expl.finetune_br(
        cfg,
        "pursuer",
        GreedyEvader,
        CLONE,
        TOTAL,
        seed=SEED,
        device=device,
        gamma=gamma,
        critic_warmup=CRITIC_WARMUP,
        callback=cb,
        env_wrap=env_wrap,
    )
    state = counter.state
    episodes = state.get("episodes") or 0
    training = dict(
        state,
        positives_per_1M=counter.positives_per_1M,
        segments=counter.segments,
        # Spec 4: the round's own premise is that a shorter episode is a
        # DENSER reward. Record the realised fraction so a reader can check
        # it against the truncation table rather than take it on trust.
        capture_episode_fraction=(
            (state.get("captures") or 0) / episodes if episodes else None
        ),
    )
    result = {
        "arm": arm,
        "seed": SEED,
        "timesteps": TOTAL,
        "max_steps": cfg.max_steps,
        "gamma": gamma,
        "eval_n": EVAL_N,
        "eval_seed0": seed0,
        "reward_negated": arm == "HNEG",
        "start": {
            "k": starts[bar]["k"],
            "n": starts[bar]["n"],
            "rate": starts[bar]["rate"],
            "bar": bar,
            "provenance": (
                "measured from the gated clone on this arm's own horizon and "
                "stream before any training (spec 2); results/derisk/t1_start.json"
            ),
        },
        "checkpoints": pr.rows,
        "training": training,
    }
    if arm == "HNEG":
        verdict, reason = control_collapsed(pr.rows, from_step=EVERY)
    else:
        # A crashed arm must not get a fabricated final probe: final_eval
        # ALWAYS loads the saved final checkpoint, and if the run died
        # before writing it, that file is absent. Judge as UNDETERMINED
        # rather than raising and losing the probe history gathered so far.
        ckpt_path = OUT / f"ckpts_{arm}" / f"{arm}_{TOTAL}.zip"
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
            result["final"] = final_eval(cfg, arm, seed0, device)
            verdict, reason = round_verdict(
                pr.rows,
                result["final"]["k"],
                result["final"]["n"],
                starts[bar]["rate"],
                from_step=PASS_FROM,
            )
    result["verdict"], result["reason"] = verdict, reason
    (OUT / f"t1_arm_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] verdict={verdict}: {reason}", flush=True)
    return result


def load_results() -> dict:
    return {
        p.stem.removeprefix("t1_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob("t1_arm_*.json"))
    }


def summarize(results: dict) -> dict:
    """Apply the withhold rule and the spec-5 direction. Pure: no I/O."""
    arms = {
        a: {"verdict": r["verdict"], "reason": r["reason"]} for a, r in results.items()
    }
    ctrl = results.get("HNEG")
    # Unlike the earlier rounds, this control validates the harness at a NEW
    # episode length, so its failure invalidates EVERY learner arm, not one.
    withheld = False
    if ctrl is None:
        why = (
            "withheld: the negated-reward control (HNEG) has not run, so the "
            "harness is unvalidated at this episode length"
        )
        withheld = True
    elif ctrl["verdict"] is not True:
        why = (
            "withheld: the negated-reward control did not demonstrably collapse "
            f"({ctrl['reason']}), so the harness is unvalidated at this episode "
            "length"
        )
        withheld = True
    if withheld:
        for a in LEARNER_ARMS:
            if a in arms:
                arms[a] = {"verdict": "UNDETERMINED", "reason": why}
    survivors = [
        a for a in LEARNER_ARMS if a in arms and arms[a]["verdict"] == "IMPROVED"
    ]
    learners_judged = all(
        a in arms and arms[a]["verdict"] in DETERMINATE_VERDICTS for a in LEARNER_ARMS
    )
    complete = all(a in results for a in (*LEARNER_ARMS, "HNEG")) and learners_judged
    ha, hb, hc = ("HA" in survivors), ("HB" in survivors), ("HC" in survivors)
    if ha and not hc:
        direction = (
            "HA IMPROVED and HC did not: the horizon was the binding constraint. "
            "Re-plan Track 2 (belief-augmented evader) and Track 3 (the pair, "
            "shielded vs unshielded) at the short horizon, each with its own "
            "pre-registration."
        )
    elif ha and hc:
        direction = (
            "HA and HC both IMPROVED: the horizon is not what changed — this "
            "configuration was simply never run at 1200. Report as a coverage "
            "gap in the earlier rounds and re-plan Track 2/3 at the ORIGINAL "
            "horizon."
        )
    elif hb and not ha:
        direction = (
            "HB IMPROVED where HA did not: shortening alone is insufficient and "
            "the matched discount is doing the work. Carry gamma as a matched "
            "parameter into Track 2/3."
        )
    elif survivors:
        direction = (
            f"IMPROVED: {survivors}. This pattern matches none of the cases the "
            "pre-registration named; report it as observed, do not force it into "
            "one of them."
        )
    elif complete:
        direction = (
            "No arm IMPROVED. Route C stands as D102 left it, strengthened: the "
            "learning failure now survives reward shaping, algorithm family, "
            "anchoring at both strengths, bootstrapping, weighting calibration, "
            "AND a threefold reduction in the credit-assignment horizon with a "
            "2.6x denser reward."
        )
    else:
        direction = "incomplete: not every learner arm has a determinate verdict"
    return {
        "spec": "docs/pathway/t1_horizon_round_prereg_2026-09-08.md",
        "clone_ci_lo": CLONE_CI_LO,
        "pass_from": PASS_FROM,
        "final_eval_n": FINAL_EVAL_N,
        "arms": arms,
        "survivors": survivors,
        "complete": complete,
        "control_withheld_everything": withheld,
        "direction": direction,
    }


def _verdict_label(verdict) -> str:
    """HNEG's verdict is True/False/None (`control_collapsed`'s shape), not
    one of the VERDICTS strings the other arms use -- give it words instead
    of a literal "HNEG (True)" on a signed visual gate."""
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
    bars = {}
    for arm, r in results.items():
        rows = r["checkpoints"]
        if rows:
            ax.plot(
                [x["steps"] for x in rows],
                [x["win_rate"] for x in rows],
                marker="o",
                label=(
                    f"{arm} ({r.get('max_steps', '?')} steps, "
                    f"{_verdict_label(summary['arms'][arm]['verdict'])})"
                ),
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
        bars[r["start"]["bar"]] = r["start"]["rate"]
    for name, colour in (("START_400", "tab:green"), ("START_1200", "tab:purple")):
        if name in bars:
            ax.axhline(
                bars[name],
                color=colour,
                ls=":",
                lw=1.2,
                label=f"{name} = {bars[name]:.3f} (improvement bar)",
            )
    ax.axhline(CLONE_CI_LO, color="k", ls="--", label=f"clone CI-lo {CLONE_CI_LO}")
    ax.axvline(
        PASS_FROM, color="grey", ls="-.", lw=0.8, label=f"pass window from {PASS_FROM}"
    )
    ax.set_xlabel("env steps")
    ax.set_ylabel("win rate vs GreedyEvader (paired seeds, per-horizon stream)")
    title = "T1.0 horizon round" + (
        "" if summary["complete"] else " — INCOMPLETE, do not read as a result"
    )
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure-starts", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.measure_starts:
        if not CLONE.exists():
            raise SystemExit(f"gated clone missing: {CLONE}")
        measure_starts(args.device)
        return
    for arm in args.arms:
        if not CLONE.exists():
            raise SystemExit(f"gated clone missing: {CLONE}")
        run_arm(arm, args.device)
    # Written ONLY under --summarize: each arm process writes its own
    # t1_arm_{ARM}.json, but t1_round.{json,png} aggregates ALL of them and
    # concurrent arm processes would otherwise interleave on this shared pair.
    if args.summarize:
        results = load_results()
        summary = summarize(results)
        (OUT / "t1_round.json").write_text(
            json.dumps({"summary": summary, "arms": results}, indent=2)
        )
        write_figure(results, summary, OUT / "t1_round.png")
        print(json.dumps(summary["arms"], indent=2))
        print(f"survivors={summary['survivors']}\n{summary['direction']}")


if __name__ == "__main__":
    main()
