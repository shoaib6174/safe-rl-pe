"""Step 5 long run: checkpointed BR training + D14 best-checkpoint selection.

The short-run history (8/100 flat, 3/100 annuity, 12/100 undertrained,
5/200 at 300k) shows final-checkpoint eval is a lottery. This script does
it properly:

1. Train SAC BR for --timesteps with a checkpoint + paired selection eval
   every --every steps (N=100, seeds 710000+ — HELD OUT from the gate).
2. Select the best checkpoint by selection-eval win rate (D14: never the
   last checkpoint by default).
3. Run the strengthened gate ONCE on the selected model: N=200 paired seeds
   720000+ vs an identically-seeded random init; trained Wilson 95% CI must
   lie entirely above random's. Selection seeds and gate seeds are disjoint
   — no selection leak into the reported number.

Progress is appended to step_05_br_long_{role}.json after EVERY checkpoint
(partial results survive a crash).

Usage: ./venv/bin/python scripts/pe2_step05_br_long.py [--role pursuer] [--timesteps 1000000]
"""

import argparse
import json
import pathlib
import shutil

from pe2.config import EnvConfig
from pe2.eval import binomial_pvalue, summarize, win_count
from pe2.exploitability import SingleAgentWrapper, evaluate_sb3_policy, train_br
from pe2.scripted import GreedyEvader, GreedyPursuer

OUT_DIR = pathlib.Path("results/visual_gates/step_05")

SELECT_SEED0 = 710000  # held-out selection eval range
GATE_SEED0 = 720000  # gate eval range (disjoint)
BR_TRAIN_SEED = 1
OPPONENT = {"pursuer": GreedyEvader, "evader": GreedyPursuer}


class CheckpointProbe:
    """SB3 callback: save checkpoint + paired selection eval every N steps."""

    def __init__(self, cfg, role, ckpt_dir, report_path, every, eval_n):
        from stable_baselines3.common.callbacks import BaseCallback

        class _CB(BaseCallback):
            def _on_step(inner_self) -> bool:
                if inner_self.num_timesteps % every != 0:
                    return True
                steps = inner_self.num_timesteps
                path = ckpt_dir / f"br_{role}_{steps}.zip"
                inner_self.model.save(path)
                logs = evaluate_sb3_policy(
                    inner_self.model,
                    cfg,
                    role,
                    OPPONENT[role],
                    n_episodes=eval_n,
                    seed0=SELECT_SEED0,
                )
                k, n = win_count(logs, role)
                self.rows.append(
                    {"steps": steps, "ckpt": str(path), "win_rate": k / n, "k": k, "n": n}
                )
                self._flush()
                print(f"  @{steps:>8}: selection win rate {k}/{n} = {k / n:.3f}", flush=True)
                return True

        self.rows = []
        self._report_path = report_path
        self.callback = _CB()

    def _flush(self):
        with open(self._report_path, "w") as f:
            json.dump({"checkpoints": self.rows}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["pursuer", "evader"], default="pursuer")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--every", type=int, default=100_000)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--gate-n", type=int, default=200)
    args = parser.parse_args()

    cfg = EnvConfig()
    ckpt_dir = OUT_DIR / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"step_05_br_long_{args.role}.json"

    probe = CheckpointProbe(
        cfg, args.role, ckpt_dir, report_path, args.every, args.eval_n
    )
    print(
        f"training {args.role} BR for {args.timesteps} steps; checkpoint+eval "
        f"every {args.every} (N={args.eval_n}, seeds {SELECT_SEED0}+)..."
    )
    train_br(
        cfg,
        role=args.role,
        opponent_factory=OPPONENT[args.role],
        total_timesteps=args.timesteps,
        seed=BR_TRAIN_SEED,
        callback=probe.callback,
    )

    # --- D14: best checkpoint by paired SELECTION eval (never "last") ---
    best = max(probe.rows, key=lambda r: r["win_rate"])
    print(f"D14 selection: best = {best['ckpt']} "
          f"(win rate {best['win_rate']:.3f} @ {best['steps']} steps)")
    selected = OUT_DIR / f"sac_br_{args.role}.zip"
    shutil.copy(best["ckpt"], selected)

    # --- strengthened gate on the SELECTED model, disjoint seeds ---
    from stable_baselines3 import SAC

    trained = SAC.load(selected)
    random_init = SAC(
        "MlpPolicy",
        SingleAgentWrapper(cfg, args.role, OPPONENT[args.role]),
        seed=BR_TRAIN_SEED,
        verbose=0,
    )
    report = {}
    for name, model in (("random_init", random_init), ("trained_br", trained)):
        logs = evaluate_sb3_policy(
            model, cfg, args.role, OPPONENT[args.role],
            n_episodes=args.gate_n, seed0=GATE_SEED0,
        )
        k, n = win_count(logs, args.role)
        s = summarize(logs)
        report[name] = {"summary": s, "win_rate": k / n, "logs": logs}
        print(
            f"{name:<12} gate win rate = {k / n:.3f}  "
            f"[{s['cr_ci'][0]:.3f}, {s['cr_ci'][1]:.3f}]"
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

    lo_tr, _ = report["trained_br"]["summary"]["cr_ci"]
    _, hi_ra = report["random_init"]["summary"]["cr_ci"]
    k_tr, n_tr = win_count(report["trained_br"]["logs"], args.role)
    p = binomial_pvalue(k_tr, n_tr, p0=0.5, alternative="greater")
    assert lo_tr > hi_ra, (
        f"BR sanity FAILED ({args.role}): selected-checkpoint CI [{lo_tr:.3f}, ...] "
        f"does not clear random init CI [..., {hi_ra:.3f}] — harness is broken"
    )
    print(f"exit gate: trained CI lo {lo_tr:.3f} > random CI hi {hi_ra:.3f} OK "
          f"(vs chance p={p:.2e})")


if __name__ == "__main__":
    main()
