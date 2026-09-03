"""Step 5 demonstration arm (D74): DQfD-style demo-buffer seeding + gate.

Both sparse-reward arms failed on BOOTSTRAPPING, not exploration: random
play produces ~6.5% captures but they never amplify (control 4->1->1/100),
and a 0.5x opponent curriculum didn't either (0/50 after 100k). The horizon
curriculum (D73) attacks credit-assignment length; this arm attacks the
data instead: pre-fill the SAC replay buffer with demonstrations from the
FOV-limited scripted pursuer (FovLimitedPursuer, uses_truth=False — the
obs->action mapping is learnable from the agent's own inputs; CR 0.232 in
this matchup), so the critic bootstraps from teacher captures from step 0.

Everything else identical to the control arm: terminal-only reward, true
cfg, checkpoint + selection eval every 100k (N=100, seeds 710000+), D14
best-checkpoint selection, strengthened gate once on the selected model
(N=200, disjoint seeds 720000+). Demo episodes use seeds 750000+ (disjoint
from training, selection, and gate streams).

Usage: ./venv/bin/python scripts/pe2_step05_br_demo.py [--role pursuer] [--demos 100] [--timesteps 1000000]
"""

import argparse
import json
import pathlib
import shutil

from pe2.config import EnvConfig
from pe2.exploitability import (
    CheckpointEvalProbe,
    collect_demonstrations,
    run_br_gate,
    train_br,
)
from pe2.scripted import (
    FovLimitedEvader,
    FovLimitedPursuer,
    GreedyEvader,
    GreedyPursuer,
)

OUT_DIR = pathlib.Path("results/visual_gates/step_05")

SELECT_SEED0 = 710000  # held-out selection eval range (shared across arms)
GATE_SEED0 = 720000  # gate eval range (disjoint)
DEMO_SEED0 = 750000  # demonstration episodes (disjoint from all eval streams)
BR_TRAIN_SEED = 3  # distinct from control (1) and curriculum (2) arms
OPPONENT = {"pursuer": GreedyEvader, "evader": GreedyPursuer}
TEACHER = {"pursuer": FovLimitedPursuer, "evader": FovLimitedEvader}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["pursuer", "evader"], default="pursuer")
    parser.add_argument("--demos", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--gate-n", type=int, default=200)
    args = parser.parse_args()

    cfg = EnvConfig()
    ckpt_dir = OUT_DIR / "ckpts_demo"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"step_05_br_demo_{args.role}.json"

    print(
        f"collecting {args.demos} demonstration episodes "
        f"(teacher={TEACHER[args.role].__name__}, seeds {DEMO_SEED0}+)...",
        flush=True,
    )
    demos = collect_demonstrations(
        cfg, args.role, TEACHER[args.role], OPPONENT[args.role], args.demos, DEMO_SEED0
    )
    teacher_wins = sum(
        1
        for *_, info in demos
        if info["outcome"] == ("capture" if args.role == "pursuer" else "timeout")
    )
    # win is role-relative: outcome recorded per episode-end transition
    print(
        f"  {len(demos)} transitions; teacher win episodes: "
        f"{teacher_wins}/{args.demos}",
        flush=True,
    )

    probe = CheckpointEvalProbe(
        cfg,
        args.role,
        OPPONENT[args.role],
        every=100_000,
        eval_n=args.eval_n,
        seed0=SELECT_SEED0,
        ckpt_dir=ckpt_dir,
        report_path=report_path,
        label=f"brd_{args.role}",
    )
    print(
        f"training {args.role} BR for {args.timesteps} steps atop the demo "
        f"buffer; checkpoint+eval every 100k (N={args.eval_n}, seeds {SELECT_SEED0}+)",
        flush=True,
    )
    train_br(
        cfg,
        role=args.role,
        opponent_factory=OPPONENT[args.role],
        total_timesteps=args.timesteps,
        seed=BR_TRAIN_SEED,
        callback=probe.callback,
        demonstrations=demos,
    )

    # --- D14: best checkpoint by paired SELECTION eval (never "last") ---
    best = max(probe.rows, key=lambda r: r["win_rate"])
    print(
        f"D14 selection: best = {best['ckpt']} "
        f"(win rate {best['win_rate']:.3f} @ {best['steps']} steps)"
    )
    selected = OUT_DIR / f"sac_brd_{args.role}.zip"
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
                "demos": args.demos,
                "demo_seed0": DEMO_SEED0,
                "timesteps": args.timesteps,
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
        f"BR sanity FAILED ({args.role}, demo arm): D76 criteria unmet — "
        f"{report['criteria']} (trained CI lo {lo_tr:.3f} vs pooled-init CI hi "
        f"{p_hi:.3f})"
    )
    print(f"exit gate: D76 criteria all pass — {report['criteria']}")


if __name__ == "__main__":
    main()
