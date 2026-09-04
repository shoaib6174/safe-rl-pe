"""T0.3 pilots (derisk plan v2): three mechanism-matched 200k fine-tunes of
the gated SAC clone + the D78 naive negative control.

  P-A  change the signal   : SAC fine-tune + DenseRewardWrapper(w_vis, w_search)
  P-B  stop the drift      : KL-anchored SAC fine-tune (beta)
  P-C  change the horizon  : PPO-BC clone -> PPO+GAE fine-tune
  CTRL negative control    : naive SAC fine-tune, critic_warmup=0 (must collapse)

Pre-registered (before any run): critic_warmup=50k, total=200k, probe every
50k on N=200 paired seeds 780000+, pass = win rate >= 0.137 at every probe
from 100k (pe2.derisk.pilot_passes). Kill rule: P-A training positives/1M
< 50 -> Route C.

CTRL is the harness's own check: it is the fine-tune already measured to
collapse (D78, 0.185 -> 0.060 @100k -> 0.020 @200k). Survivors are reported
ONLY when CTRL demonstrably reproduced that collapse — `control_collapsed`,
which needs a non-empty judging window AND a final rate separated from the
clone, not merely a failed pass rule. Anything else and this script writes a
null verdict with the reason in words, stamps the figure, and exits non-zero
rather than naming survivors off an unvalidated harness.

Usage: ./venv/bin/python scripts/derisk_t0_pilots.py --arms PA PB PC CTRL [--timesteps 200000]
"""

import argparse
import json
import pathlib
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import CallbackList

from pe2.anchored_sac import AnchorKLProbe, anchor_kl_grew
from pe2.config import EnvConfig
from pe2.derisk import (
    CLONE_CI_LO,
    PASS_FROM_STEP,
    TrainingCaptureCounter,
    control_collapsed,
    pilot_passes,
    start_is_adequate,
)
from pe2.eval import win_count
from pe2.exploitability import (
    CheckpointEvalProbe,
    collect_demonstrations,
    evaluate_sb3_policy,
    finetune_br,
    finetune_br_anchored,
    finetune_ppo,
    train_bc_ppo,
)
from pe2.rewards import DenseRewardWrapper
from pe2.scripted import FovLimitedPursuer, GreedyEvader

OUT = pathlib.Path("results/derisk")
CLONE = pathlib.Path("results/visual_gates/step_05/sac_bc_pursuer.zip")
EVAL_SEED0 = 780000  # disjoint from demo (750000+) and audit (790000+) streams
DEMO_SEED0 = 750000
DEMO_EPISODES = 300  # named once: the call and the recorded row cannot disagree
# One seed for the three OFF-POLICY arms: the training seed drives critic
# initialisation, exploration noise and the training env's episode stream, so
# a seed per arm confounds the mechanism with between-seed variance — measured
# at 0.010-0.280 across weight seeds under an identical protocol in this
# project. Paired, P-A/P-B/CTRL differ only in mechanism. P-C is a different
# algorithm family (PPO: no replay, no critic warm-up, its own clone) and
# cannot be paired with them, so it keeps its own seed. This narrows the
# pre-registered seeds 7/8/9/10 and is recorded in the S121 worklog.
OFF_POLICY_SEED = 7
SEEDS = {"PA": OFF_POLICY_SEED, "PB": OFF_POLICY_SEED, "PC": 9, "CTRL": OFF_POLICY_SEED}
W_VIS, W_SEARCH, BETA = 0.1, 1e-4, 1.0
KILL_POSITIVES_PER_1M = 50


def probe(cfg, label, every, eval_n):
    d = OUT / f"ckpts_{label}"
    d.mkdir(parents=True, exist_ok=True)
    return CheckpointEvalProbe(
        cfg,
        "pursuer",
        GreedyEvader,
        every=every,
        eval_n=eval_n,
        seed0=EVAL_SEED0,
        ckpt_dir=d,
        report_path=OUT / f"t0_pilot_{label}.json",
        label=label,
    )


def run_arm(arm, cfg, timesteps, warmup, every, eval_n, device):
    pr = probe(cfg, arm, every, eval_n)
    # Snapshots on the probe grid so the kill-rule rate can be decomposed by
    # phase instead of blending them. With the pre-registered warmup=50k and
    # every=50k the phase-A/phase-B boundary IS a snapshot boundary; the row
    # records `critic_warmup` so a reader can locate it for other settings.
    counter = TrainingCaptureCounter(snapshot_every=every)
    callbacks = [pr.callback, counter.callback]
    if arm == "PB":
        # AFTER the probe and on the same schedule: it stamps the row the
        # probe has just appended. Without it the anchored arm's divergence
        # is computed at every gradient step and discarded, and "did not
        # collapse" cannot be told apart from "the anchor froze the actor".
        callbacks.append(AnchorKLProbe(pr.rows, every).callback)
    cb = CallbackList(callbacks)
    seed = SEEDS[arm]
    # One source of truth for what each arm's warm-up ACTUALLY is, so the
    # recorded row cannot disagree with the call. CTRL is the naive D78
    # fine-tune: zero warm-up by definition, never `--warmup`. P-C is PPO,
    # which has no critic warm-up analogue at all.
    arm_warmup = {"PA": warmup, "PB": warmup, "PC": None, "CTRL": 0}[arm]
    if arm == "PA":
        finetune_br(
            cfg,
            "pursuer",
            GreedyEvader,
            CLONE,
            timesteps,
            seed=seed,
            device=device,
            critic_warmup=arm_warmup,
            callback=cb,
            env_wrap=lambda e: DenseRewardWrapper(e, w_vis=W_VIS, w_search=W_SEARCH),
        )
    elif arm == "PB":
        finetune_br_anchored(
            cfg,
            "pursuer",
            GreedyEvader,
            CLONE,
            timesteps,
            seed=seed,
            beta=BETA,
            device=device,
            critic_warmup=arm_warmup,
            callback=cb,
        )
    elif arm == "PC":
        demos = collect_demonstrations(
            cfg, "pursuer", FovLimitedPursuer, GreedyEvader, DEMO_EPISODES, DEMO_SEED0
        )
        bc, losses = train_bc_ppo(
            cfg, "pursuer", GreedyEvader, demos, seed=seed, device=device, epochs=50
        )
        bc_path = OUT / "ppo_bc_pursuer.zip"
        bc.save(bc_path)
        # P-A/P-B/CTRL all start from the GATED clone, whose win rate is known
        # — that is what makes the 0.137 bound meaningful for them. P-C starts
        # from a fresh PPO clone of UNKNOWN quality and is judged against the
        # same bound, so without this measurement a P-C failure cannot be told
        # apart from "the arm never had a competent starting point", and the
        # tier would wrongly read it as refuting the horizon hypothesis.
        # Same seeds and same N as the probes use, so it is comparable.
        bk, bn = win_count(
            evaluate_sb3_policy(
                bc, cfg, "pursuer", GreedyEvader, n_episodes=eval_n, seed0=EVAL_SEED0
            ),
            "pursuer",
        )
        bc_start = {
            "win_rate": bk / bn,
            "k": bk,
            "n": bn,
            "final_bc_loss": losses[-1],
            "demo_episodes": DEMO_EPISODES,
            "demo_seed0": DEMO_SEED0,
        }
        print(
            f"[PC] BC start: {bk}/{bn} = {bk / bn:.3f} (bc loss {losses[-1]:.5f})",
            flush=True,
        )
        finetune_ppo(
            cfg,
            "pursuer",
            GreedyEvader,
            bc_path,
            timesteps,
            seed=seed,
            device=device,
            callback=cb,
        )
    elif arm == "CTRL":
        finetune_br(
            cfg,
            "pursuer",
            GreedyEvader,
            CLONE,
            timesteps,
            seed=seed,
            device=device,
            critic_warmup=arm_warmup,
            callback=cb,
        )
    else:
        raise SystemExit(f"unknown arm {arm}")
    rows = pr.rows
    rate = counter.positives_per_1M
    verdict = pilot_passes(rows)
    result = {
        "arm": arm,
        "seed": seed,
        "timesteps": timesteps,
        "critic_warmup": arm_warmup,
        "eval_n": eval_n,
        "eval_seed0": EVAL_SEED0,
        "checkpoints": rows,
        "pass": verdict,
        "training": dict(
            counter.state,
            positives_per_1M=rate,
            segments=counter.segments,
        ),
    }
    if arm == "PC":
        # The starting rate was already measured on the right seeds; here it
        # is USED. When its Wilson upper bound is below the bound the arm is
        # judged against, neither a pass nor a failure is about the horizon
        # hypothesis, so the verdict is forced to undetermined with the
        # reason in words — the pattern control_collapsed already uses.
        adequate, why = start_is_adequate(bc_start["k"], bc_start["n"])
        bc_start["adequate"] = adequate
        bc_start["reason"] = why
        result["bc_start"] = bc_start
        if adequate is None:
            result["pass"] = verdict = None
            result["pass_reason"] = why
            print(f"[PC] verdict forced to undetermined: {why}", flush=True)
    if arm == "PB":
        # The anchored arm passes the "does not collapse" rule by not moving.
        # Report whether the divergence GREW alongside the verdict, so a pass
        # that is really a preserved clone cannot be read as a survivor.
        grew, kl_why = anchor_kl_grew(rows)
        result["anchor"] = {"beta": BETA, "kl_grew": grew, "kl_reason": kl_why}
        print(f"[PB] {kl_why}", flush=True)
    # The kill rule needs a rate; with no rollouts there is none, and a
    # verdict off nothing would read as "the signal is too sparse".
    if arm == "PA" and rate is not None:
        result["kill_rule"] = {
            "threshold": KILL_POSITIVES_PER_1M,
            "fires": rate < KILL_POSITIVES_PER_1M,
        }
    shown = "undefined (no steps)" if rate is None else f"{rate:.1f}"
    print(f"[{arm}] pass={result['pass']} positives/1M={shown}", flush=True)
    return result


def summarize(results, every, eval_n, complete):
    """Build the digest. `survivors` is reported ONLY when the negative
    control demonstrably reproduced the documented collapse — that is the
    entire basis for believing any other arm's verdict."""
    if not complete:
        verdict, reason = None, "run in progress: arms still to come"
    elif "CTRL" not in results:
        verdict, reason = (
            None,
            (
                "the negative control (CTRL) was not run, so nothing shows the "
                "harness can discriminate; every arm's verdict is unvalidated"
            ),
        )
    else:
        verdict, reason = control_collapsed(results["CTRL"]["checkpoints"])
    ctrl_rows = results.get("CTRL", {}).get("checkpoints") or []
    return {
        "status": "complete" if complete else "in_progress",
        "clone_ci_lo": CLONE_CI_LO,
        # A different --every gives a materially different judging window;
        # record both so the file says which rule was actually applied.
        "pass_from_step": PASS_FROM_STEP,
        "probe_every": every,
        "eval_n": eval_n,
        "survivors": (
            [a for a, r in results.items() if a != "CTRL" and r["pass"] is True]
            if verdict is True
            else None
        ),
        "negative_control_collapsed": verdict,
        "negative_control_reason": reason,
        "negative_control_final_win_rate": (
            max(ctrl_rows, key=lambda r: r["steps"])["win_rate"] if ctrl_rows else None
        ),
        "arms": results,
    }


def write_summary(summary) -> None:
    with open(OUT / "t0_pilots.json", "w") as f:
        json.dump(summary, f, indent=2)


def write_figure(results, summary, eval_n) -> None:
    validated = summary["negative_control_collapsed"] is True
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for arm, r in results.items():
        # No pass/fail wording when the harness is unvalidated: the figure is
        # the artifact most likely to end up in a slide, and it must not say
        # an arm passed when the digest is refusing to name survivors.
        word = {True: "pass", False: "fail", None: "undetermined"}[r["pass"]]
        label = f"{arm} ({word})" if validated else arm
        ax.plot(
            [c["steps"] for c in r["checkpoints"]],
            [c["win_rate"] for c in r["checkpoints"]],
            marker="o",
            label=label,
        )
    ax.axhline(CLONE_CI_LO, color="k", ls="--", label=f"clone CI-lo {CLONE_CI_LO}")
    ax.set(xlabel="training steps", ylabel=f"win rate (N={eval_n}, paired)")
    if validated:
        ax.set_title("T0.3 pilots")
    else:
        ax.set_title(
            "T0.3 pilots — HARNESS NOT VALIDATED, DO NOT READ THE ARMS\n"
            + textwrap.fill(summary["negative_control_reason"], 100),
            fontsize=7.5,
            color="firebrick",
        )
        ax.text(
            0.5,
            0.5,
            "HARNESS NOT VALIDATED",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=24,
            color="firebrick",
            alpha=0.30,
            rotation=18,
            zorder=10,
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "t0_pilots.png", dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--arms", nargs="+", choices=list(SEEDS), default=["PA", "PB", "PC", "CTRL"]
    )
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--warmup", type=int, default=50_000)
    p.add_argument("--every", type=int, default=50_000)
    p.add_argument("--eval-n", type=int, default=200)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not CLONE.exists():
        raise SystemExit(f"gated clone missing: {CLONE}")
    cfg = EnvConfig()

    results = {}
    for arm in args.arms:
        results[arm] = run_arm(
            arm, cfg, args.timesteps, args.warmup, args.every, args.eval_n, args.device
        )
        # Flush after EVERY arm: these runs take hours, and a failure in the
        # last arm must not discard the ones that already finished. The FIGURE
        # is rewritten too — it is the artifact most likely to be read on its
        # own, and leaving a stale unstamped one beside an in-progress summary
        # is the hazard the "not validated" stamp exists to close.
        partial = summarize(results, args.every, args.eval_n, complete=False)
        write_summary(partial)
        write_figure(results, partial, args.eval_n)

    summary = summarize(results, args.every, args.eval_n, complete=True)
    write_summary(summary)
    write_figure(results, summary, args.eval_n)
    print(f"wrote {OUT}/t0_pilots.{{json,png}}")

    verdict = summary["negative_control_collapsed"]
    print(f"negative control: {summary['negative_control_reason']}")
    if verdict is True:
        print(f"survivors={summary['survivors']}")
        return
    if "CTRL" in results:
        raise SystemExit(
            "negative control did NOT demonstrably collapse — pilot harness is "
            "unvalidated, do not read the arms"
        )
    print("survivors withheld: run CTRL to validate the harness")


if __name__ == "__main__":
    main()
