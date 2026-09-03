"""Step 5 BR sanity: a trained approximate best response must improve over
its random init — else the harness is broken, not the agent (plan Step 5).

Scenario: SAC pursuer BR vs a FROZEN GreedyEvader (D55). The frozen target
is deliberately the truth-seeing greedy evader: a competent pursuer BR
should learn to capture it at least as readily as the scripted greedy
pursuer does (battery CR ~0.44), and a random-init policy should be much
worse. Both policies are scored on the SAME paired eval seeds (D57,
seed0=720000, disjoint from the Step 3 battery range).

Reward is terminal-only (+1 capture / -1 timeout from the pursuer's side)
at the ENV level. TRAINING runs at gamma=0.999 with potential-based
distance shaping at the SAME gamma (D68/D69, ShapedRewardWrapper —
policy-invariant per Ng et al. 1999). History: unshaped gamma=0.99 gave
trained 8/100 vs random 7/100 (terminal reward discounts to ~6e-6 — flat
objective); shaping at mismatched gamma=0.99 gave 3/100, WORSE than random
(annuity hack: idling pays (1-gamma)*d/step and capture ends the stream).
Evaluation always measures true game outcomes on the unshaped wrapper.

Exit gate (strengthened after the first run passed 8/100 vs 7/100 — a
statistically meaningless margin): the trained BR's Wilson 95% CI must lie
ENTIRELY ABOVE the random init's CI. "Improves" must be established, not
a coin flip.

SAC training on GPU is not bit-deterministic (cuDNN); the determinism
exit gate covers the EVALUATION harness (bit-identical logs, tested in
test_exploitability.py), not training.

Writes results/visual_gates/step_05/step_05_br_sanity.json.

Usage: ./venv/bin/python scripts/pe2_step05_br_sanity.py [--timesteps 100000] [--n 100]
"""

import argparse
import json
import pathlib

from pe2.config import EnvConfig
from pe2.eval import binomial_pvalue, summarize, win_count
from pe2.exploitability import SingleAgentWrapper, evaluate_sb3_policy, train_br
from pe2.scripted import GreedyEvader

OUT_DIR = pathlib.Path("results/visual_gates/step_05")

EVAL_SEED0 = 720000  # paired eval seeds (D57), disjoint from battery range
BR_TRAIN_SEED = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    cfg = EnvConfig()

    print(f"training SAC pursuer BR vs frozen GreedyEvader ({args.timesteps} steps)...")
    trained = train_br(
        cfg,
        role="pursuer",
        opponent_factory=GreedyEvader,
        total_timesteps=args.timesteps,
        seed=BR_TRAIN_SEED,
    )

    # random init: identical architecture/seed, zero learning — the honest
    # "is the harness capable of learning at all" baseline
    from stable_baselines3 import SAC

    random_init = SAC(
        "MlpPolicy",
        SingleAgentWrapper(cfg, "pursuer", GreedyEvader),
        seed=BR_TRAIN_SEED,
        verbose=0,
    )

    logs = {}
    for name, model in (("random_init", random_init), ("trained_br", trained)):
        print(
            f"evaluating {name} (N={args.n}, seeds {EVAL_SEED0}..{EVAL_SEED0 + args.n - 1})..."
        )
        logs[name] = evaluate_sb3_policy(
            model,
            cfg,
            "pursuer",
            GreedyEvader,
            n_episodes=args.n,
            seed0=EVAL_SEED0,
        )

    report = {"timesteps": args.timesteps, "n": args.n, "eval_seed0": EVAL_SEED0}
    for name, log in logs.items():
        k, n = win_count(log, "pursuer")
        lo, hi = summarize(log)["cr_ci"]
        report[name] = {
            "summary": summarize(log),
            "win_rate": k / n,
            "logs": log,
        }
        print(f"{name:<12} win rate (CR) = {k / n:.3f}  [{lo:.3f}, {hi:.3f}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step_05_br_sanity.json", "w") as f:
        json.dump(report, f)
    trained.save(OUT_DIR / "sac_br_pursuer")  # episode-grid gate renders THIS model

    # --- exit gate: trained BR's CI entirely above random init's ---
    k_tr, n_tr = win_count(logs["trained_br"], "pursuer")
    k_ra, _ = win_count(logs["random_init"], "pursuer")
    lo_tr, _ = report["trained_br"]["summary"]["cr_ci"]
    _, hi_ra = report["random_init"]["summary"]["cr_ci"]
    assert lo_tr > hi_ra, (
        f"BR sanity FAILED: trained CI [{lo_tr:.3f}, ...] does not clear "
        f"random init CI [..., {hi_ra:.3f}] ({k_tr}/{n_tr} vs {k_ra}/{n_tr}) "
        "— harness is broken, not the agent"
    )
    # context, not the gate: is trained CR consistent with a coin flip?
    p_vs_chance = binomial_pvalue(k_tr, n_tr, p0=0.5, alternative="greater")
    print(
        f"exit gate: trained CI lo {lo_tr:.3f} > random CI hi {hi_ra:.3f} OK "
        f"({k_tr}/{n_tr} vs {k_ra}/{n_tr}; trained vs chance p={p_vs_chance:.2e})"
    )


if __name__ == "__main__":
    main()
