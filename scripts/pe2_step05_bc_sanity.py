"""Step 5 BC sanity (D75): behavior-clone the FLp teacher, then gate.

RL-from-scratch on the sparse 1200-step terminal-only objective failed four
ways (D68-D74). The imitation-then-RL pipeline starts here: collect
FovLimitedPursuer demonstrations through the wrapper (seeds 750000+),
behavior-clone the SAC actor (supervised regression to the squashed mean),
and run the D76 gate against the fixed baseline battery
(step_05_baselines_{role}.json from scripts/pe2_step05_baselines.py):
trained Wilson CI lo must clear every heuristic's CI hi and the POOLED
random-init CI hi, plus a paired McNemar win vs iid_random. Expected: clone
CR ~ teacher CR (0.232 reference) minus imitation gap >> pooled random
(~0.10) — a decisive harness-learning demonstration, and the starting point
for RL fine-tuning arms.

Writes step_05_bc_sanity_{role}.json + sac_bc_{role}.zip (role-suffixed —
the pursuer and evader runs must not clobber each other).

Usage: ./venv/bin/python scripts/pe2_step05_bc_sanity.py [--role pursuer] [--demos 100] [--epochs 20]
"""

import argparse
import json
import pathlib

from pe2.config import EnvConfig
from pe2.exploitability import collect_demonstrations, run_br_gate, train_bc_actor
from pe2.scripted import (
    FovLimitedEvader,
    FovLimitedPursuer,
    GreedyEvader,
    GreedyPursuer,
)

OUT_DIR = pathlib.Path("results/visual_gates/step_05")

GATE_SEED0 = 720000  # same gate range as every other arm
DEMO_SEED0 = 750000  # demonstration episodes (disjoint from eval streams)
BC_SEED = 4  # distinct from RL arms (1/2/3)
OPPONENT = {"pursuer": GreedyEvader, "evader": GreedyPursuer}
TEACHER = {"pursuer": FovLimitedPursuer, "evader": FovLimitedEvader}
WIN_OUTCOME = {"pursuer": "capture", "evader": "timeout"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["pursuer", "evader"], default="pursuer")
    parser.add_argument("--demos", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--gate-n", type=int, default=200)
    args = parser.parse_args()

    cfg = EnvConfig()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"collecting {args.demos} demos (teacher={TEACHER[args.role].__name__}, "
        f"seeds {DEMO_SEED0}+)...",
        flush=True,
    )
    demos = collect_demonstrations(
        cfg, args.role, TEACHER[args.role], OPPONENT[args.role], args.demos, DEMO_SEED0
    )
    teacher_wins = sum(
        1 for *_, info in demos if info["outcome"] == WIN_OUTCOME[args.role]
    )
    print(f"  {len(demos)} transitions; teacher wins {teacher_wins}/{args.demos}")

    print(f"behavior-cloning actor ({args.epochs} epochs)...", flush=True)
    model, losses = train_bc_actor(
        cfg,
        args.role,
        OPPONENT[args.role],
        demos,
        seed=BC_SEED,
        epochs=args.epochs,
    )
    print(f"  BC loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    model_path = OUT_DIR / f"sac_bc_{args.role}.zip"
    model.save(model_path)

    report = run_br_gate(
        cfg,
        args.role,
        OPPONENT[args.role],
        model_path,
        args.gate_n,
        GATE_SEED0,
        OUT_DIR / f"step_05_baselines_{args.role}.json",
    )
    out = {
        "role": args.role,
        "demos": args.demos,
        "epochs": args.epochs,
        "demo_seed0": DEMO_SEED0,
        "gate_seed0": GATE_SEED0,
        "teacher_win_rate": teacher_wins / args.demos,
        "bc_loss": losses,
        "gate": report,
    }
    with open(OUT_DIR / f"step_05_bc_sanity_{args.role}.json", "w") as f:
        json.dump(out, f)

    assert report["criteria"]["pass"], (
        f"BC sanity FAILED ({args.role}): gate criteria unmet — "
        f"{ {k: v for k, v in report['criteria'].items()} }; harness is broken"
    )
    print(f"exit gate: D76 criteria all pass — {report['criteria']}")


if __name__ == "__main__":
    main()
