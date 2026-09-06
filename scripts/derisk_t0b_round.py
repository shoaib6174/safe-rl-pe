"""T0.4 retry round — the pre-registered learner round after Tier 0 (D98).

Spec: docs/pathway/t0_retry_round_prereg_2026-09-06.md, Part C, reproduced
here verbatim where it binds. Nothing here may change a threshold, a seed, a
budget or a definition after a number is seen.

Arms (pursuer vs frozen GreedyEvader, 1M steps, probes every 100k on 200
paired seeds 780000+, final probe N=500):
  RB    KL-anchored SAC fine-tune, beta = 0.1 (the A4 pre-commitment)
  RC    PPO-BC clone -> PPO+GAE at 1M (the only Tier-0 arm rising at cutoff)
  RD    advantage-weighted regression on MC returns, offline then online
  RDNEG RD with the advantage sign flipped, 200k: MUST collapse (control)
Pass = win rate >= 0.137 at every probe from 500k. Improved = final N=500
Wilson lower bound > the arm's own start (37/200 = 0.185 SAC-family;
41/200 = 0.205 PPO). Survivor = IMPROVED. RD is withheld unless RDNEG
collapses by `control_collapsed`.

Usage:
  ./venv/bin/python scripts/derisk_t0b_round.py --arms RB        (one process per arm)
  ./venv/bin/python scripts/derisk_t0b_round.py --summarize      (after all four)
"""

from __future__ import annotations

import argparse
import json
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import (
    CLONE_CI_LO,
    TrainingCaptureCounter,
    control_collapsed,
    round_verdict,
)
from pe2.eval import wilson_ci, win_count
from pe2.scripted import GreedyEvader

OUT = pathlib.Path("results/derisk")
CLONE = pathlib.Path("results/visual_gates/step_05/sac_bc_pursuer.zip")
PPO_BC = OUT / "ppo_bc_pursuer.zip"  # Tier 0's PPO-BC clone, start 41/200

# --- pre-registered (spec C2/C3). Do not tune. ---
TOTAL = 1_000_000
EVERY = 100_000
EVAL_N = 200
FINAL_EVAL_N = 500
PASS_FROM = 500_000
EVAL_SEED0 = 780_000  # Tier 0's probe stream: every probe is paired with Tier 0's
BETA_RB = 0.1
CRITIC_WARMUP = 50_000
NEG_TOTAL = 200_000
AWR_SEED0 = 820_000
AWRNEG_SEED0 = 830_000
SEEDS = {"RB": 7, "RC": 9, "RD": 11, "RDNEG": 11}
STARTS = {"RB": (37, 200), "RC": (41, 200), "RD": (37, 200), "RDNEG": (37, 200)}
START_PROVENANCE = {
    "RB": "gated SAC clone, D76 battery (S119): 37/200 = 0.185, Wilson [0.137, 0.245]",
    "RC": "Tier 0 arms.PC.bc_start (S122): 41/200 = 0.205",
    "RD": "gated SAC clone, D76 battery (S119): 37/200 = 0.185",
    "RDNEG": "gated SAC clone, D76 battery (S119): 37/200 = 0.185",
}
TIER0_PC_EXPECTED = {100_000: 7, 200_000: 28}  # RC must reproduce these (determinism)
LEARNER_ARMS = ("RB", "RC", "RD")


def start_rate(arm: str) -> float:
    k, n = STARTS[arm]
    return k / n


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
        report_path=OUT / f"t0b_probe_{arm}.json",
        label=arm,
    )


def final_eval(cfg, arm, total, device) -> dict:
    """The N=500 improvement probe, ALWAYS from the saved final checkpoint so
    the number is tied to an artefact a reader can re-run."""
    from stable_baselines3 import PPO, SAC

    from pe2.exploitability import evaluate_sb3_policy

    path = OUT / f"ckpts_{arm}" / f"{arm}_{total}.zip"
    model = (PPO if arm == "RC" else SAC).load(path, device=device)
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

    from pe2.anchored_sac import AnchorKLProbe, anchor_kl_grew
    from pe2.awr import AwrConfig, train_awr
    from pe2.exploitability import finetune_br_anchored, finetune_ppo

    total = NEG_TOTAL if arm == "RDNEG" else TOTAL
    pr = probe(cfg, arm)
    counter = TrainingCaptureCounter(snapshot_every=EVERY)
    extra: dict = {}
    if arm == "RB":
        cb = CallbackList(
            [pr.callback, counter.callback, AnchorKLProbe(pr.rows, EVERY).callback]
        )
        finetune_br_anchored(
            cfg,
            "pursuer",
            GreedyEvader,
            CLONE,
            total,
            seed=SEEDS[arm],
            beta=BETA_RB,
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
        extra["anchor"] = {"beta": BETA_RB, "kl_grew": grew, "kl_reason": why}
    elif arm == "RC":
        cb = CallbackList([pr.callback, counter.callback])
        finetune_ppo(
            cfg,
            "pursuer",
            GreedyEvader,
            PPO_BC,
            total,
            seed=SEEDS[arm],
            device=device,
            callback=cb,
        )
        training = dict(
            counter.state,
            positives_per_1M=counter.positives_per_1M,
            segments=counter.segments,
        )
        observed = {
            r["steps"]: r["k"] for r in pr.rows if r["steps"] in TIER0_PC_EXPECTED
        }
        extra["tier0_replication"] = {
            "expected_k": TIER0_PC_EXPECTED,
            "observed_k": observed,
            "reproduced": observed == TIER0_PC_EXPECTED,
        }
    elif arm in ("RD", "RDNEG"):
        awr = AwrConfig(advantage_sign=1.0 if arm == "RD" else -1.0)
        _, stats = train_awr(
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
            rollout_seed0=AWR_SEED0 if arm == "RD" else AWRNEG_SEED0,
        )
        training = {
            k: stats[k]
            for k in ("total_steps", "n_episodes", "positives", "positives_per_1M")
        }
        extra["awr"] = stats
    else:
        raise SystemExit(f"unknown arm {arm}")

    result = {
        "arm": arm,
        "seed": SEEDS[arm],
        "timesteps": total,
        "eval_n": EVAL_N,
        "eval_seed0": EVAL_SEED0,
        "start": {
            "k": STARTS[arm][0],
            "n": STARTS[arm][1],
            "rate": start_rate(arm),
            "provenance": START_PROVENANCE[arm],
        },
        "checkpoints": pr.rows,
        "training": training,
        **extra,
    }
    if arm == "RDNEG":
        verdict, reason = control_collapsed(pr.rows, from_step=100_000)
    else:
        # A crashed arm must not get a fabricated final probe: final_eval
        # ALWAYS loads the saved final checkpoint, and if the run died
        # before writing it, that file is absent. Judge as UNDETERMINED
        # rather than raising and losing the probe history gathered so far
        # (round_verdict itself cannot detect this: it is handed final_k/n
        # rather than deriving them, so this check belongs here).
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
                start_rate(arm),
                from_step=PASS_FROM,
            )
    result["verdict"], result["reason"] = verdict, reason
    (OUT / f"t0b_arm_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] verdict={verdict}: {reason}", flush=True)
    return result


def load_results() -> dict:
    return {
        p.stem.removeprefix("t0b_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob("t0b_arm_*.json"))
    }


def summarize(results: dict) -> dict:
    """Apply the withhold rule and the C4 direction. Pure: no I/O."""
    arms = {
        a: {"verdict": r["verdict"], "reason": r["reason"]} for a, r in results.items()
    }
    ctrl = results.get("RDNEG")
    if "RD" in arms:
        if ctrl is None:
            arms["RD"] = {
                "verdict": "UNDETERMINED",
                "reason": "withheld: the AWR control (RDNEG) has not run, so the AWR loop and its probe path are unvalidated",
            }
        elif ctrl["verdict"] is not True:
            arms["RD"] = {
                "verdict": "UNDETERMINED",
                "reason": f"withheld: the AWR control did not demonstrably collapse ({ctrl['reason']})",
            }
    survivors = [
        a for a in LEARNER_ARMS if a in arms and arms[a]["verdict"] == "IMPROVED"
    ]
    complete = all(a in results for a in (*LEARNER_ARMS, "RDNEG"))
    if survivors:
        best = max(
            survivors,
            key=lambda a: wilson_ci(results[a]["final"]["k"], results[a]["final"]["n"])[
                0
            ],
        )
        direction = f"IMPROVED: {survivors}. Track-1 mechanism = {best} (highest final Wilson lower bound). Write the Track 2 (evader) plan for it; two survivors run as Track-2 arms."
    elif complete:
        direction = "No arm IMPROVED. Stabilisation results are reported as method findings; an RL-trained pair that improves on its priors is not reachable at this budget with these families -> Route C (spec C4)."
    else:
        direction = "incomplete: not every arm has run"
    return {
        "spec": "docs/pathway/t0_retry_round_prereg_2026-09-06.md",
        "clone_ci_lo": CLONE_CI_LO,
        "pass_from": PASS_FROM,
        "final_eval_n": FINAL_EVAL_N,
        "arms": arms,
        "survivors": survivors,
        "complete": complete,
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
                [r["final"]["win_rate"]],
                yerr=[[r["final"]["win_rate"] - lo], [hi - r["final"]["win_rate"]]],
                fmt="s",
                capsize=3,
            )
        ax.axhline(r["start"]["rate"], ls=":", lw=0.8)
    ax.axhline(CLONE_CI_LO, color="k", ls="--", label=f"clone CI-lo {CLONE_CI_LO}")
    ax.axvline(PASS_FROM, color="grey", ls="-.", lw=0.8, label="pass window from 500k")
    ax.set_xlabel("env steps")
    ax.set_ylabel("win rate vs GreedyEvader (paired seeds 780000+)")
    title = "T0.4 retry round" + (
        "" if summary["complete"] else " — INCOMPLETE, do not read as a result"
    )
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="*", choices=list(SEEDS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig()
    for arm in args.arms:
        if arm in ("RB", "RD", "RDNEG") and not CLONE.exists():
            raise SystemExit(f"gated clone missing: {CLONE}")
        if arm == "RC" and not PPO_BC.exists():
            raise SystemExit(f"Tier 0 PPO-BC clone missing: {PPO_BC}")
        run_arm(arm, cfg, args.device)
    if args.summarize or args.arms:
        results = load_results()
        summary = summarize(results)
        (OUT / "t0b_round.json").write_text(
            json.dumps({"summary": summary, "arms": results}, indent=2)
        )
        write_figure(results, summary, OUT / "t0b_round.png")
        print(json.dumps(summary["arms"], indent=2))
        print(f"survivors={summary['survivors']}\n{summary['direction']}")


if __name__ == "__main__":
    main()
