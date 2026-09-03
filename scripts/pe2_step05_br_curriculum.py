"""Step 5 curriculum arms (D72/D73) + gate, vs the flat control arm.

Control arm (pe2_step05_br_long.py, terminal-only, true cfg) is flat:
4/100 -> 1/100 -> 1/100 over 300k steps. Two curriculum hypotheses attack
the diagnosed bootstrapping failure; both keep reward and EVALUATION
identical (selection/gate always vs the TRUE 1200-step, 1.2x-evader cfg):

- speed (D72, REFUTED): opponent speed 0.5x -> 1.2x. Stage-1 diagnostic:
  0/50 captures even vs the 0.5x evader after 100k steps — opponent
  difficulty was never the bottleneck. Kept for completeness.
- horizon (D73): episode length 150 -> 300 -> 600 -> 1200 steps. The
  bottleneck is credit assignment over the 1200-step horizon: short
  episodes give 8x more episodes, 8x less credit ambiguity, and a visible
  terminal (0.999^150 ~= 0.86), while the arena diagonal (14 m) stays
  traversable in 150 steps (max closing 2.2 m/s) so pursuit is feasible
  everywhere. The task is time-invariant: capturing within 300 steps
  implies capturing within 1200.

Protocol identical to the control arm: checkpoint + selection eval every
100k GLOBAL steps (N=100, seeds 710000+), D14 best-checkpoint selection,
strengthened gate once on the selected model (N=200, disjoint seeds
720000+, trained Wilson 95% CI entirely above random init's).

Usage: ./venv/bin/python scripts/pe2_step05_br_curriculum.py [--curriculum horizon] [--role pursuer]
"""

import argparse
import json
import pathlib
import shutil

from pe2.config import EnvConfig
from pe2.exploitability import (
    CheckpointEvalProbe,
    run_br_gate,
    train_br_curriculum,
)
from pe2.scripted import GreedyEvader, GreedyPursuer

OUT_DIR = pathlib.Path("results/visual_gates/step_05")

SELECT_SEED0 = 710000  # held-out selection eval range (shared with control)
GATE_SEED0 = 720000  # gate eval range (disjoint)
BR_TRAIN_SEED = 2  # distinct from control arm (seed 1)
OPPONENT = {"pursuer": GreedyEvader, "evader": GreedyPursuer}

# (cfg_overrides, timesteps), easiest first; final stage = true config
STAGES = {
    "speed": [
        ({"evader_speed_ratio": 0.5}, 200_000),
        ({"evader_speed_ratio": 0.8}, 200_000),
        ({"evader_speed_ratio": 1.0}, 200_000),
        ({"evader_speed_ratio": 1.2}, 400_000),
    ],
    "horizon": [
        ({"max_steps": 150}, 200_000),
        ({"max_steps": 300}, 200_000),
        ({"max_steps": 600}, 300_000),
        ({"max_steps": 1200}, 300_000),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", choices=list(STAGES), default="horizon")
    parser.add_argument("--role", choices=["pursuer", "evader"], default="pursuer")
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--gate-n", type=int, default=200)
    args = parser.parse_args()

    stages = STAGES[args.curriculum]
    cfg = EnvConfig()
    ckpt_dir = OUT_DIR / f"ckpts_{args.curriculum}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"step_05_br_{args.curriculum}_{args.role}.json"

    probe = CheckpointEvalProbe(
        cfg,  # selection eval on TRUE config
        args.role,
        OPPONENT[args.role],
        every=100_000,
        eval_n=args.eval_n,
        seed0=SELECT_SEED0,
        ckpt_dir=ckpt_dir,
        report_path=report_path,
        label=f"br{args.curriculum[0]}_{args.role}",
    )
    total = sum(s for _, s in stages)
    print(
        f"{args.curriculum} curriculum {args.role} BR: {stages} = {total} steps; "
        f"checkpoint+eval every 100k (N={args.eval_n}, true cfg, seeds {SELECT_SEED0}+)",
        flush=True,
    )
    train_br_curriculum(
        cfg,
        role=args.role,
        opponent_factory=OPPONENT[args.role],
        stages=stages,
        seed=BR_TRAIN_SEED,
        callback=probe.callback,
    )

    # --- D14: best checkpoint by paired SELECTION eval (never "last") ---
    best = max(probe.rows, key=lambda r: r["win_rate"])
    print(
        f"D14 selection: best = {best['ckpt']} "
        f"(win rate {best['win_rate']:.3f} @ {best['steps']} steps)"
    )
    selected = OUT_DIR / f"sac_br{args.curriculum[0]}_{args.role}.zip"
    shutil.copy(best["ckpt"], selected)

    # --- D76 gate on the SELECTED model, disjoint seeds ---
    report = run_br_gate(
        cfg,
        args.role,
        OPPONENT[args.role],
        selected,
        args.gate_n,
        GATE_SEED0,
        OUT_DIR / f"step_05_baselines_{args.role}.json",
    )

    with open(report_path, "w") as f:
        json.dump(
            {
                "role": args.role,
                "curriculum": args.curriculum,
                "stages": stages,
                "select_seed0": SELECT_SEED0,
                "gate_seed0": GATE_SEED0,
                "selected_ckpt": best,
                "checkpoints": probe.rows,
                "gate": report,
            },
            f,
        )

    lo_tr, _ = report["trained_br"]["win_ci"]
    p_hi = report["pooled_random_init"]["win_ci"][1]
    assert report["criteria"]["pass"], (
        f"BR sanity FAILED ({args.role}, curriculum): D76 criteria unmet — "
        f"{report['criteria']} (trained CI lo {lo_tr:.3f} vs pooled-init CI hi "
        f"{p_hi:.3f})"
    )
    print(f"exit gate: D76 criteria all pass — {report['criteria']}")


if __name__ == "__main__":
    main()
