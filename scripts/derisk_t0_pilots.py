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
collapse (D78, 0.185 -> 0.020 @200k). If it does NOT collapse, the harness
is not measuring what it claims and no arm's verdict means anything, so this
script exits non-zero instead of reporting survivors.

Usage: ./venv/bin/python scripts/derisk_t0_pilots.py --arms PA PB PC CTRL [--timesteps 200000]
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import CallbackList

from pe2.config import EnvConfig
from pe2.derisk import CLONE_CI_LO, TrainingCaptureCounter, pilot_passes
from pe2.exploitability import (
    CheckpointEvalProbe,
    collect_demonstrations,
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
SEEDS = {"PA": 7, "PB": 8, "PC": 9, "CTRL": 10}
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
    counter = TrainingCaptureCounter()
    cb = CallbackList([pr.callback, counter.callback])
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
            cfg, "pursuer", FovLimitedPursuer, GreedyEvader, 300, DEMO_SEED0
        )
        bc, _losses = train_bc_ppo(
            cfg, "pursuer", GreedyEvader, demos, seed=seed, device=device, epochs=50
        )
        bc_path = OUT / "ppo_bc_pursuer.zip"
        bc.save(bc_path)
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
    result = {
        "arm": arm,
        "seed": seed,
        "timesteps": timesteps,
        "critic_warmup": arm_warmup,
        "eval_n": eval_n,
        "eval_seed0": EVAL_SEED0,
        "checkpoints": rows,
        "pass": pilot_passes(rows),
        "training": dict(counter.state, positives_per_1M=counter.positives_per_1M),
    }
    if arm == "PA":
        result["kill_rule"] = {
            "threshold": KILL_POSITIVES_PER_1M,
            "fires": counter.positives_per_1M < KILL_POSITIVES_PER_1M,
        }
    print(
        f"[{arm}] pass={result['pass']} positives/1M={counter.positives_per_1M:.1f}",
        flush=True,
    )
    return result


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
    results = {
        arm: run_arm(
            arm, cfg, args.timesteps, args.warmup, args.every, args.eval_n, args.device
        )
        for arm in args.arms
    }
    # None (not False) when CTRL was not run: the harness is then UNVALIDATED,
    # which must not be recorded as "the control collapsed".
    ctrl_collapsed = None if "CTRL" not in results else not results["CTRL"]["pass"]
    survivors = [a for a, r in results.items() if a != "CTRL" and r["pass"]]
    summary = {
        "clone_ci_lo": CLONE_CI_LO,
        # null when the control did NOT collapse: the harness is then not
        # measuring what it claims, so there is no survivor list to report.
        # The raw per-arm rows and `pass` flags stay under "arms" — nothing
        # is hidden, it just is not pre-digested into a verdict.
        "survivors": None if ctrl_collapsed is False else survivors,
        "negative_control_collapsed": ctrl_collapsed,
        "arms": results,
    }
    with open(OUT / "t0_pilots.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, r in results.items():
        ax.plot(
            [c["steps"] for c in r["checkpoints"]],
            [c["win_rate"] for c in r["checkpoints"]],
            marker="o",
            label=f"{arm} ({'pass' if r['pass'] else 'fail'})",
        )
    ax.axhline(CLONE_CI_LO, color="k", ls="--", label="clone CI-lo 0.137")
    ax.set(
        xlabel="training steps",
        ylabel=f"win rate (N={args.eval_n}, paired)",
        title="T0.3 pilots",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "t0_pilots.png", dpi=120)
    print(f"wrote {OUT}/t0_pilots.{{json,png}}")
    if ctrl_collapsed is False:
        raise SystemExit(
            "negative control did NOT collapse — pilot harness is broken, "
            "do not read the arms"
        )
    if ctrl_collapsed is None:
        print("negative control NOT RUN — survivors below are unvalidated")
    print(f"survivors={survivors} negative_control_collapsed={ctrl_collapsed}")


if __name__ == "__main__":
    main()
