"""Step 5 BC fine-tune arm (D75 second half): RL on top of the gated clone.

The BC clone (scripts/pe2_step05_bc_sanity.py) already pursues at
near-teacher level but leaves an imitation gap; RL-from-scratch failed four
ways (D68-D74). This arm fine-tunes the clone with SAC on the true
terminal-only objective (gamma=0.999), so the critics warm up on data from
a policy that already captures ~20% instead of random play's ~6.5% that
never amplifies. This is the arm that can EXCEED the scripted teacher — the
actual BR claim.

Protocol identical to every other arm: checkpoint + selection eval every
100k steps (N=100, seeds 710000+), D14 best-checkpoint selection, D76 gate
once on the selected model (N=200, disjoint seeds 720000+, vs the baseline
battery). Train seed 5 (wrapper stream 5000+, disjoint from all other arms).

Usage: ./venv/bin/python scripts/pe2_step05_bc_finetune.py [--role pursuer] [--timesteps 1000000]
"""

import argparse
import json
import pathlib
import shutil

from pe2.config import EnvConfig
from pe2.exploitability import (
    CheckpointEvalProbe,
    finetune_br,
    run_br_gate,
)
from pe2.scripted import GreedyEvader, GreedyPursuer

OUT_DIR = pathlib.Path("results/visual_gates/step_05")

SELECT_SEED0 = 710000  # held-out selection eval range (shared across arms)
GATE_SEED0 = 720000  # gate eval range (disjoint)
FT_SEED = 5  # distinct from control (1), curricula (2), demo (3), BC (4)
OPPONENT = {"pursuer": GreedyEvader, "evader": GreedyPursuer}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["pursuer", "evader"], default="pursuer")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--gate-n", type=int, default=200)
    args = parser.parse_args()

    cfg = EnvConfig()
    bc_path = OUT_DIR / f"sac_bc_{args.role}.zip"
    if not bc_path.exists():
        raise SystemExit(
            f"BC clone not found: {bc_path} — run pe2_step05_bc_sanity.py first"
        )
    ckpt_dir = OUT_DIR / "ckpts_finetune"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"step_05_br_finetune_{args.role}.json"

    probe = CheckpointEvalProbe(
        cfg,
        args.role,
        OPPONENT[args.role],
        every=100_000,
        eval_n=args.eval_n,
        seed0=SELECT_SEED0,
        ckpt_dir=ckpt_dir,
        report_path=report_path,
        label=f"brf_{args.role}",
    )
    print(
        f"fine-tuning {bc_path} for {args.timesteps} steps (gamma=0.999, "
        f"terminal-only); checkpoint+eval every 100k (N={args.eval_n}, seeds "
        f"{SELECT_SEED0}+)",
        flush=True,
    )
    finetune_br(
        cfg,
        args.role,
        OPPONENT[args.role],
        bc_path,
        args.timesteps,
        seed=FT_SEED,
        callback=probe.callback,
    )

    # --- D14: best checkpoint by paired SELECTION eval (never "last") ---
    best = max(probe.rows, key=lambda r: r["win_rate"])
    print(
        f"D14 selection: best = {best['ckpt']} "
        f"(win rate {best['win_rate']:.3f} @ {best['steps']} steps)"
    )
    selected = OUT_DIR / f"sac_brf_{args.role}.zip"
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
                "bc_init": str(bc_path),
                "timesteps": args.timesteps,
                "select_seed0": SELECT_SEED0,
                "gate_seed0": GATE_SEED0,
                "selected_ckpt": best,
                "checkpoints": probe.rows,
                "gate": report,
            },
            f,
        )

    assert report["criteria"]["pass"], (
        f"fine-tune arm FAILED ({args.role}): D76 criteria unmet — {report['criteria']}"
    )
    print(f"exit gate: D76 criteria all pass — {report['criteria']}")


if __name__ == "__main__":
    main()
