"""Step 5 LN from-scratch arm (D80 control): SAC with LayerNorm critics.

Every gamma=0.999 terminal-only arm's critics DIVERGED in this env
(measured Q magnitudes 6-70 from-scratch, +69 naive fine-tune, -3.0e9
frozen phase-A; true |Q| <= 1) — so the from-scratch refutations D68-D74
are confounded: those runs were chasing diverging Q the whole time. This
arm reruns the from-scratch protocol with the D80 LayerNorm critic to
resolve the confound: if it now learns, the old failures were value
divergence, not fundamental sample/credit limits; if it stays flat with
BOUNDED critics, the imitation-first conclusion (D75) stands clean.

Protocol identical to the fine-tune arm: checkpoint + selection eval
every 100k (N=100, seeds 710000+), D14 best-checkpoint selection, D76
gate once on the selected model (N=200, disjoint seeds 720000+, vs the
baseline battery). Train seed 6 (wrapper stream 6000+, disjoint from all
other arms: 1 control, 2 curricula, 3 demo, 4 BC, 5 fine-tune).

Usage: ./venv/bin/python scripts/pe2_step05_br_ln.py [--role pursuer] [--timesteps 1000000]
"""

import argparse
import json
import pathlib
import shutil

from pe2.config import EnvConfig
from pe2.exploitability import (
    CheckpointEvalProbe,
    ln_sac_policy,
    run_br_gate,
    train_br,
)
from pe2.scripted import GreedyEvader, GreedyPursuer

OUT_DIR = pathlib.Path("results/visual_gates/step_05")

SELECT_SEED0 = 710000  # held-out selection eval range (shared across arms)
GATE_SEED0 = 720000  # gate eval range (disjoint)
LN_SEED = 6  # distinct from control (1), curricula (2), demo (3), BC (4), FT (5)
OPPONENT = {"pursuer": GreedyEvader, "evader": GreedyPursuer}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["pursuer", "evader"], default="pursuer")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--gate-n", type=int, default=200)
    args = parser.parse_args()

    cfg = EnvConfig()
    ckpt_dir = OUT_DIR / "ckpts_ln_scratch"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"step_05_br_ln_{args.role}.json"

    probe = CheckpointEvalProbe(
        cfg,
        args.role,
        OPPONENT[args.role],
        every=100_000,
        eval_n=args.eval_n,
        seed0=SELECT_SEED0,
        ckpt_dir=ckpt_dir,
        report_path=report_path,
        label=f"brl_{args.role}",
    )
    print(
        f"training {args.role} BR from scratch with LayerNorm critics (D80) "
        f"for {args.timesteps} steps; checkpoint+eval every 100k "
        f"(N={args.eval_n}, seeds {SELECT_SEED0}+)",
        flush=True,
    )
    train_br(
        cfg,
        role=args.role,
        opponent_factory=OPPONENT[args.role],
        total_timesteps=args.timesteps,
        seed=LN_SEED,
        callback=probe.callback,
        policy=ln_sac_policy(),
    )

    # --- D14: best checkpoint by paired SELECTION eval (never "last") ---
    best = max(probe.rows, key=lambda r: r["win_rate"])
    print(
        f"D14 selection: best = {best['ckpt']} "
        f"(win rate {best['win_rate']:.3f} @ {best['steps']} steps)"
    )
    selected = OUT_DIR / f"sac_brl_{args.role}.zip"
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
                "timesteps": args.timesteps,
                "select_seed0": SELECT_SEED0,
                "gate_seed0": GATE_SEED0,
                "selected_ckpt": best,
                "checkpoints": probe.rows,
                "gate": report,
            },
            f,
        )

    # This arm is a CONTROL for the D68-D74 confound: report, don't assert.
    print(f"D76 gate result: {report['criteria']}")


if __name__ == "__main__":
    main()
