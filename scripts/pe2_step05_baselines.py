"""Step 5 baseline battery (D76): the fixed reference set for the BR gate.

Measures the solution-free baselines on the gate stream (seeds 720000+,
disjoint from selection 710000+ and demos 750000+):

  - heuristics: stand_still [-1,0], half_speed [0,0], full_forward [1,0],
    iid_random (seeded iid Uniform actions)
  - K random SAC inits (weight seeds 1..K): samples from the network-init
    distribution — high-variance (seed 4 is a bull-charge outlier), which is
    exactly why the gate compares against the POOLED distribution, never a
    single init (D76).

Writes step_05_baselines.json (summaries + full per-episode logs, so the
gate can run paired McNemar tests). Re-run only if the env or the gate
stream changes; the JSON is a durable harness artifact.

Usage: ./venv/bin/python scripts/pe2_step05_baselines.py [--role pursuer] [-n 200] [--inits 5]
"""

import argparse
import json
import pathlib

from pe2.config import EnvConfig
from pe2.eval import wilson_ci, win_count
from pe2.exploitability import (
    ConstantPolicy,
    IidRandomPolicy,
    SingleAgentWrapper,
    evaluate_sb3_policy,
)
from pe2.scripted import GreedyEvader, GreedyPursuer

OUT_DIR = pathlib.Path("results/visual_gates/step_05")

GATE_SEED0 = 720000  # same gate stream as every arm
OPPONENT = {"pursuer": GreedyEvader, "evader": GreedyPursuer}
HEURISTICS = {
    "stand_still": ConstantPolicy((-1.0, 0.0)),
    "half_speed": ConstantPolicy((0.0, 0.0)),
    "full_forward": ConstantPolicy((1.0, 0.0)),
    "iid_random": IidRandomPolicy(seed=0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["pursuer", "evader"], default="pursuer")
    parser.add_argument("-n", type=int, default=200)
    parser.add_argument("--inits", type=int, default=5)
    args = parser.parse_args()

    from stable_baselines3 import SAC  # lazy: optional dependency

    cfg = EnvConfig()
    opp = OPPONENT[args.role]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    policies = [(name, pol, "heuristic") for name, pol in HEURISTICS.items()]
    for seed in range(1, args.inits + 1):
        model = SAC(
            "MlpPolicy",
            SingleAgentWrapper(cfg, args.role, opp),
            seed=seed,
            verbose=0,
        )
        policies.append((f"random_init_s{seed}", model, "random_init"))

    baselines = {}
    for name, policy, kind in policies:
        logs = evaluate_sb3_policy(
            policy, cfg, args.role, opp, n_episodes=args.n, seed0=GATE_SEED0
        )
        k, n = win_count(logs, args.role)
        lo, hi = wilson_ci(k, n)
        mean_steps = sum(log["steps"] for log in logs) / n
        baselines[name] = {
            "kind": kind,
            "win_rate": k / n,
            "win_ci": (lo, hi),
            "mean_steps": mean_steps,
            "logs": logs,
        }
        print(
            f"{name:<16} {k}/{n} = {k / n:.3f}  CI [{lo:.3f}, {hi:.3f}]  "
            f"steps {mean_steps:.0f}",
            flush=True,
        )

    out = {
        "role": args.role,
        "gate_seed0": GATE_SEED0,
        "gate_n": args.n,
        "n_random_inits": args.inits,
        "baselines": baselines,
    }
    path = OUT_DIR / f"step_05_baselines_{args.role}.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
