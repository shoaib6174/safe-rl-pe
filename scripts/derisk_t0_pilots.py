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
import numpy as np

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
    seed_torch,
    start_is_adequate,
)
from pe2.eval import win_count
from pe2.exploitability import (
    CheckpointEvalProbe,
    SingleAgentWrapper,
    collect_demonstrations,
    evaluate_sb3_policy,
    finetune_br,
    finetune_br_anchored,
    finetune_ppo,
    train_bc_ppo,
)
from pe2.rewards import (
    DETERMINISTIC_KINDS,
    TERM_KINDS,
    DenseRewardWrapper,
    nondegeneracy_report,
)
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
# T0.2 plateau gate: rollout stream (790000+, the audit's family) offset off
# the audit's own episodes so the two diagnostics do not share episodes.
NONDEG_SEED0 = 790500
# Plateau-gate budget, in whole EPISODES.
#
# It was 6,000 steps — 5 to 7 episodes — and at that budget the gate was a
# coin flip rather than a check on the reward: run at five disjoint seeds in
# this family it aborts the whole tier on two of them (vis positive fraction
# 0.0711 / 0.0282 / 0.0612 / 0.0767 / 0.0110 against the 0.05 band edge).
# The `vis` term is strongly autocorrelated WITHIN an episode — the pursuer
# either holds the evader in view for a run of steps or does not — so the
# sampling unit is the episode, not the step (per-episode positive fraction
# 0.157, sd 0.300 over 2,500 episodes), and a 6-episode sample says almost
# nothing.
#
# 500 episodes is ~513k clone steps, ~5-11 min depending on device, against
# the multi-hour arm it gates. What it buys: the episode-level bootstrap
# interval is then ~+/-0.008 wide, so the gate DECIDES any term whose true
# positive fraction is outside roughly [0.058, 0.942] and, via
# `all_decisive`, refuses to claim a verdict for one that is not.
#
# It never decided the `vis` term, and no affordable budget could: measured
# over 2,500 episodes / 2,566,786 clone steps at five disjoint seeds (790500,
# 791500, 792500, 793500, 794500; per-seed positive fraction 0.0520 / 0.0529
# / 0.0574 / 0.0509 / 0.0509) that term's positive fraction is 0.0528 with
# bootstrap 95% CI [0.0493, 0.0563] — sitting ON the 0.05 band edge, 5.6%
# above it, with the lower bound first clearing 0.05 at ~4,000 episodes
# (4.1M steps, twenty times the training budget of the arm this gates).
# The band was NOT widened: amendment A7 (S121) replaced it, for `vis` only,
# with the criterion the spec actually names — "reward-vs-bearing curve must
# be monotone into the cone" — which is analytic, exact and free of sampling
# noise (`pe2.rewards.visibility_monotone_report`). The positive fraction is
# still measured and recorded, as an OBSERVATION (`stats_role`), because the
# number is informative; it decides nothing. This budget therefore now gates
# only the SAMPLED terms — `search` on the shipped configuration.
NONDEG_EPISODES = 500
NONDEG_BOOTSTRAP = 400
# A verdict is claimed only when the bootstrap reproduces it in at least
# this share of resamples (or contradicts it in at least that share).
DECISIVE_FRACTION = 0.975


def gate_decisiveness(per_episode, seed, n_resamples=None) -> dict[str, float]:
    """Share of episode-level bootstrap resamples in which each term passes.

    `per_episode` is one `{term: [per-step values]}` dict per episode. The
    resampling unit is the WHOLE EPISODE, not the step: within an episode
    the dense terms are strongly autocorrelated (a visibility term fires in
    runs, not independently), so a step-level interval understates the
    uncertainty by an order of magnitude, while episodes are independent by
    construction — each is reset from its own seed.

    A term whose share is neither ~0 nor ~1 has a verdict decided by the
    sample rather than by the reward, and the runner refuses to act on it
    (see `DECISIVE_FRACTION` and `main`). The per-term criteria themselves
    are untouched: each resample is scored by the same
    `nondegeneracy_report` the point estimate uses.

    Terms of a DETERMINISTIC kind (A7: `vis`, judged by the analytic
    monotone-into-cone criterion) are excluded outright and get no share.
    Their verdict is a property of the reward's geometry, identical in every
    resample by construction, so bootstrapping it would dress an exact
    result up as an estimate; `nondegeneracy_gate` marks them decisive with
    the reason instead.
    """
    rng = np.random.default_rng(seed)
    n_resamples = NONDEG_BOOTSTRAP if n_resamples is None else n_resamples
    names = sorted(
        {
            n
            for ep in per_episode
            for n in ep
            if TERM_KINDS[n] not in DETERMINISTIC_KINDS
        }
    )
    arrays = {n: [np.asarray(ep[n], dtype=float) for ep in per_episode] for n in names}
    hits = {n: 0 for n in names}
    for _ in range(n_resamples):
        idx = rng.integers(0, len(per_episode), size=len(per_episode))
        sample = {n: np.concatenate([arrays[n][i] for i in idx]) for n in names}
        scored = nondegeneracy_report(sample)
        for n in names:
            hits[n] += bool(scored["terms"][n]["passes"])
    return {n: hits[n] / n_resamples for n in names}


def nondegeneracy_gate(cfg, device) -> dict:
    """T0.2 plateau check on the EXACT dense configuration P-A trains with.

    The spec rejects a dense reward with a flat region BEFORE use, so this
    runs before P-A does and its result is written next to the pilot
    results — the record then shows the gate was applied to what actually
    trains, not to one term in a test. Rolled with the gated clone, which
    is the policy the arm starts from, in the stochastic training regime.

    Reports each term's verdict AND whether that verdict is decisive under
    an episode-level bootstrap, because this gate is a hard stop and a stop
    off sampling noise manufactures a false finding about the reward. The
    `vis` term is the exception (A7): its criterion is analytic, so it is
    decisive by construction and never bootstrapped. Its rollout statistics
    are still recorded, marked `stats_role="observation"`.
    """
    from stable_baselines3 import SAC  # lazy: optional dependency

    model = SAC.load(CLONE, device=device)
    # SEED AFTER THE LOAD, never before. `SAC.load` re-applies the seed baked
    # into the zip through `_setup_model`, so seeding first is silently
    # overwritten and the declared seed never governs a single sampled
    # action — while `rollout.torch_seed` below claims it did. That is the
    # accident `seed_torch`'s own docstring warns about, and it was real
    # here: two declared seeds gave byte-identical output.
    seed_torch(NONDEG_SEED0)
    w = DenseRewardWrapper(
        SingleAgentWrapper(cfg, "pursuer", GreedyEvader),
        w_vis=W_VIS,
        w_search=W_SEARCH,
    )
    per_episode: list[dict[str, list[float]]] = []
    steps = 0
    for ep in range(NONDEG_EPISODES):
        obs, _ = w.reset(seed=NONDEG_SEED0 + ep)
        ep_terms: dict[str, list[float]] = {}
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, _r, terminated, truncated, info = w.step(action)
            steps += 1
            for name, value in info["dense_terms"].items():
                ep_terms.setdefault(name, []).append(value)
            if terminated or truncated:
                break
        per_episode.append(ep_terms)
    terms: dict[str, list[float]] = {}
    for ep_terms in per_episode:
        for name, values in ep_terms.items():
            terms.setdefault(name, []).extend(values)
    report = nondegeneracy_report(terms, cfg=cfg, w_vis=W_VIS)
    shares = gate_decisiveness(per_episode, NONDEG_SEED0)
    for name, t in report["terms"].items():
        if name in shares:
            share = shares[name]
            t["bootstrap_pass_share"] = share
            t["decisive"] = (
                share >= DECISIVE_FRACTION or share <= 1.0 - DECISIVE_FRACTION
            )
            t["decisive_reason"] = (
                f"episode-level bootstrap reproduces the verdict in {share:.3f} "
                f"of {NONDEG_BOOTSTRAP} resamples"
            )
        else:
            # Analytic criterion (A7): exact, deterministic, no sampling
            # error. Running it through the bootstrap would report an exact
            # result as an estimate and could only invent uncertainty.
            t["bootstrap_pass_share"] = None
            t["decisive"] = True
            t["decisive_reason"] = (
                f"{t['criterion']} is an analytic property of the reward's "
                "geometry: deterministic, no sampling error, decisive by "
                "construction"
            )
    report |= {
        "all_decisive": all(t["decisive"] for t in report["terms"].values()),
        "bootstrap": {
            "unit": "episode",
            "n_resamples": NONDEG_BOOTSTRAP,
            "seed": NONDEG_SEED0,
            "decisive_share": DECISIVE_FRACTION,
            "terms": sorted(shares),
        },
        "config": {"w_vis": W_VIS, "w_search": W_SEARCH},
        "rollout": {
            "policy": str(CLONE),
            "deterministic": False,
            "steps": steps,
            "episodes": len(per_episode),
            "seed0": NONDEG_SEED0,
            "torch_seed": NONDEG_SEED0,
        },
    }
    with open(OUT / "t0_nondegeneracy.json", "w") as f:
        json.dump(report, f, indent=2)
    for name, t in report["terms"].items():
        share = t["bootstrap_pass_share"]
        print(
            f"[nondeg] {name:7s} kind={t['kind']:10s} "
            f"criterion={t['criterion']:22s} passes={t['passes']} "
            f"decisive={t['decisive']} "
            + (
                "bootstrap_pass_share=analytic (n/a)"
                if share is None
                else f"bootstrap_pass_share={share:.3f}"
            )
            + f" | {t['stats_role']}: frac_positive={t['frac_positive']:.4f} "
            f"frac_modal={t['frac_modal']:.3f} std={t['std']:.3g}",
            flush=True,
        )
        if "monotone" in t:
            m = t["monotone"]
            edge = m["transition_bearing"]
            print(
                f"[nondeg] {name:7s} sweep: monotone_into_cone="
                f"{m['monotone_into_cone']} varies={m['varies']} transition="
                + ("never positive" if edge is None else f"{edge:.4f}")
                + f" rad vs configured {m['expected_transition_bearing']:.4f} "
                f"(+/-{m['transition_tolerance']:.4f}) over {m['n_bearings']} "
                f"bearings at {m['range_m']:.2f} m",
                flush=True,
            )
    return report


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

    if "PA" in args.arms:
        # A dense reward with a flat region is rejected BEFORE use (T0.2).
        gate = nondegeneracy_gate(cfg, args.device)
        # An undecided verdict is checked FIRST: a term the bootstrap cannot
        # place is neither a pass to train on nor a finding to report, and
        # calling it a finding is how a hard stop manufactures one.
        if not gate["all_decisive"]:
            undecided = [n for n, t in gate["terms"].items() if not t["decisive"]]
            raise SystemExit(
                "T0.2 plateau gate is UNDECIDED on the shipped dense "
                f"configuration for {undecided} (see "
                f"{OUT}/t0_nondegeneracy.json): over {NONDEG_EPISODES} "
                "episodes the episode-level bootstrap does not reproduce the "
                "point verdict, so the verdict is about the sample and not "
                "about the reward. Do NOT re-roll the seed and do NOT widen "
                "the band. Only SAMPLED criteria can land here: an analytic "
                "one is decisive by construction. Precedent for what to do "
                "is amendment A7 (S121) — the `vis` term's positive fraction "
                "sat ON the 0.05 edge at 0.0528 (95% CI [0.0493, 0.0563] "
                "over 2500 episodes / 2.57M clone steps at five disjoint "
                "seeds) and no affordable budget could decide it, so the "
                "criterion was replaced with the one the spec names, and the "
                "replacement was pre-registered. Change the term, or amend "
                "the criterion for it and pre-register the amendment."
            )
        if not gate["all_pass"]:
            raise SystemExit(
                "T0.2 plateau gate FAILED on the shipped dense configuration "
                f"(see {OUT}/t0_nondegeneracy.json). That is a finding about "
                "the reward, not a threshold to relax: fix the term or drop "
                "P-A, do not widen the band."
            )

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
