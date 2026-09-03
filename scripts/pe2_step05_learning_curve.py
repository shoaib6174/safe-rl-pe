"""Step 5 diagnostic: BR learning curve — WHY did 300k steps do worse than
100k (2.5% vs 12% CR)?

Single training run (same setup as pe2_step05_br_sanity.py) with a paired
checkpoint eval every 50k steps (N=50, seeds 710000+i — held out from the
final gate range 720000+) plus a sliding window over TRAINING episode
outcomes. The curve shape discriminates the hypotheses:

- rise-then-collapse  -> obstacle-trap local optimum (evader shadow-seeks,
  myopic closer gets pinned) or high-gamma critic instability
- steady rise         -> undertrained; budget really is the constraint
- flat/noisy          -> learning signal still broken

Writes results/visual_gates/step_05/step_05_learning_curve.json.

Usage: ./venv/bin/python scripts/pe2_step05_learning_curve.py [--timesteps 300000]
"""

import argparse
import json
import pathlib
from collections import deque

from pe2.config import EnvConfig
from pe2.eval import summarize, win_count
from pe2.exploitability import evaluate_sb3_policy, train_br
from pe2.scripted import GreedyEvader

OUT_DIR = pathlib.Path("results/visual_gates/step_05")

CURVE_EVAL_SEED0 = 710000  # held out from the final gate range (720000+)
BR_TRAIN_SEED = 1


class CurveProbe:
    """SB3 callback: paired checkpoint evals + training-outcome window."""

    def __init__(self, cfg, eval_every=50_000, eval_n=50):
        from stable_baselines3.common.callbacks import BaseCallback

        class _CB(BaseCallback):
            def _on_step(inner_self) -> bool:
                for done, info in zip(
                    inner_self.locals["dones"], inner_self.locals["infos"], strict=True
                ):
                    if done:
                        self.outcomes.append(info.get("outcome", "?"))
                if inner_self.num_timesteps % eval_every == 0:
                    logs = evaluate_sb3_policy(
                        inner_self.model,
                        cfg,
                        "pursuer",
                        GreedyEvader,
                        n_episodes=eval_n,
                        seed0=CURVE_EVAL_SEED0,
                    )
                    k, n = win_count(logs, "pursuer")
                    caps = self.outcomes.count("capture")
                    row = {
                        "steps": inner_self.num_timesteps,
                        "eval_cr": k / n,
                        "eval_summary": summarize(logs),
                        "train_capture_frac_last100": caps / len(self.outcomes)
                        if self.outcomes
                        else None,
                    }
                    self.rows.append(row)
                    print(
                        f"  @{row['steps']:>7} steps: eval CR {k}/{n} = {k / n:.3f}, "
                        f"train captures {caps}/{len(self.outcomes)}",
                        flush=True,
                    )
                return True

        self.outcomes = deque(maxlen=100)
        self.rows = []
        self.callback = _CB()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-n", type=int, default=50)
    args = parser.parse_args()

    cfg = EnvConfig()
    probe = CurveProbe(cfg, eval_every=args.eval_every, eval_n=args.eval_n)
    print(
        f"training with checkpoint evals every {args.eval_every} steps "
        f"(N={args.eval_n}, seeds {CURVE_EVAL_SEED0}+)..."
    )
    train_br(
        cfg,
        role="pursuer",
        opponent_factory=GreedyEvader,
        total_timesteps=args.timesteps,
        seed=BR_TRAIN_SEED,
        callback=probe.callback,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step_05_learning_curve.json", "w") as f:
        json.dump(
            {
                "timesteps": args.timesteps,
                "eval_every": args.eval_every,
                "eval_n": args.eval_n,
                "eval_seed0": CURVE_EVAL_SEED0,
                "curve": probe.rows,
            },
            f,
            indent=2,
        )
    print(f"wrote {OUT_DIR / 'step_05_learning_curve.json'}")


if __name__ == "__main__":
    main()
