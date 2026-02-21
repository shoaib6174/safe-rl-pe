"""Ablation study runner for Phase 2 safety integration.

Runs all 5 ablation configurations (A-E) using the safe self-play
rollout loop. Each config isolates one component of the safety system
to measure its contribution.

Configs:
  A: Full safe (CBF-Beta + w5 + N13) — baseline
  B: Unsafe (no CBF, no obstacles) — Phase 1 control
  C: CBF-QP filter only (post-hoc, no Beta policy)
  D: CBF-Beta, no w5 (safety reward ablation)
  E: CBF-Beta + w5, no N13 (feasibility ablation)

Usage:
    # Run all ablation configs (quick eval, no SB3 needed)
    python scripts/run_ablation.py

    # Run specific configs
    python scripts/run_ablation.py --configs A C E

    # Run with more episodes per config
    python scripts/run_ablation.py --n-episodes 50

    # Run with multiple seeds
    python scripts/run_ablation.py --seeds 0 1 2 3 4

    # Full Hydra-based training (requires SB3):
    python scripts/train.py --multirun +experiment=ablation_A,ablation_B,ablation_C,ablation_D,ablation_E seed=0,1,2,3,4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from training.safe_self_play import SafeSelfPlayConfig, safe_self_play_rollout


# =============================================================================
# Ablation config definitions (A-E)
# =============================================================================

def get_ablation_configs() -> dict[str, SafeSelfPlayConfig]:
    """Get all 5 ablation configs.

    Returns:
        Dict mapping config label to SafeSelfPlayConfig.
    """
    configs = {}

    # A: Full safe — CBF-Beta + obstacles + w5 + N13
    configs["A"] = SafeSelfPlayConfig(
        run_type="A_full_safe",
        use_cbf=True,
        use_beta_policy=True,
        use_safety_reward=True,
        use_feasibility_classifier=True,
        n_feasibility_iterations=3,
    )

    # B: Unsafe baseline — no CBF, no obstacles (Phase 1 equivalent)
    configs["B"] = SafeSelfPlayConfig(
        run_type="B_unsafe",
        use_cbf=False,
        use_beta_policy=False,
        use_safety_reward=False,
        n_obstacles=0,
        n_obstacle_obs=0,
    )

    # C: QP filter only — post-hoc safety, standard PPO
    configs["C"] = SafeSelfPlayConfig(
        run_type="C_cbf_qp_only",
        use_cbf=True,
        use_beta_policy=False,
        use_safety_reward=False,
    )

    # D: CBF-Beta, no w5 — safety reward ablation
    configs["D"] = SafeSelfPlayConfig(
        run_type="D_no_safety_reward",
        use_cbf=True,
        use_beta_policy=True,
        use_safety_reward=False,
        use_feasibility_classifier=True,
        n_feasibility_iterations=3,
    )

    # E: CBF-Beta + w5, no N13 — feasibility ablation
    configs["E"] = SafeSelfPlayConfig(
        run_type="E_no_feasibility",
        use_cbf=True,
        use_beta_policy=True,
        use_safety_reward=True,
        use_feasibility_classifier=False,
    )

    return configs


def run_single_config(
    label: str,
    config: SafeSelfPlayConfig,
    n_episodes: int,
    seed: int,
) -> dict:
    """Run a single ablation config and return results.

    Args:
        label: Config label (A-E).
        config: SafeSelfPlayConfig.
        n_episodes: Number of evaluation episodes.
        seed: Random seed.

    Returns:
        Dict with results and timing.
    """
    config.seed = seed
    t0 = time.time()
    results = safe_self_play_rollout(config, n_episodes=n_episodes)
    elapsed = time.time() - t0

    # Extract summary metrics
    episodes = results["episodes"]
    captures = sum(1 for ep in episodes if ep["captured"])
    capture_rate = captures / len(episodes) if episodes else 0.0
    mean_reward = float(np.mean([ep["reward"] for ep in episodes])) if episodes else 0.0
    mean_steps = float(np.mean([ep["steps"] for ep in episodes])) if episodes else 0.0

    safety = results["safety_metrics"]
    summary = results["summary"]
    targets = results["safety_targets"]
    resolver = results["resolver_metrics"]

    return {
        "config": label,
        "seed": seed,
        "n_episodes": n_episodes,
        "elapsed_s": elapsed,
        # Task performance
        "capture_rate": capture_rate,
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
        # Safety metrics
        "violations": safety.get("violations", 0),
        "violation_rate": safety.get("violation_rate", 0.0),
        "intervention_rate": safety.get("intervention_rate", 0.0),
        "backup_rate": safety.get("backup_rate", 0.0),
        "feasibility_rate": safety.get("feasibility_rate", 1.0),
        "exact_rate": safety.get("exact_rate", 1.0),
        "min_cbf_margin": safety.get("min_cbf_margin", float("nan")),
        "mean_cbf_margin": safety.get("mean_cbf_margin", float("nan")),
        # Targets
        "zero_violations": targets.get("zero_violations", True),
        "feasibility_above_99": targets.get("feasibility_above_99", True),
        "backup_below_1pct": targets.get("backup_below_1pct", True),
        "all_targets_met": targets.get("all_targets_met", True),
        # Resolver
        "resolver_exact_rate": resolver.get("exact_rate", 1.0),
        "resolver_method_counts": resolver.get("method_counts", {}),
    }


def print_comparison_table(all_results: list[dict]):
    """Print a formatted comparison table of ablation results."""
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS")
    print("=" * 100)

    header = (
        f"{'Config':<8} {'Seed':>4} {'Cap Rate':>9} {'Violations':>11} "
        f"{'Interv %':>9} {'Backup %':>9} {'Feas %':>7} {'Exact %':>8} "
        f"{'CBF Min':>8} {'Targets':>8}"
    )
    print(header)
    print("-" * 100)

    for r in all_results:
        violations_str = str(r["violations"])
        targets_str = "PASS" if r["all_targets_met"] else "FAIL"
        cbf_min = f"{r['min_cbf_margin']:.3f}" if not np.isnan(r["min_cbf_margin"]) else "N/A"
        print(
            f"{r['config']:<8} {r['seed']:>4} {r['capture_rate']:>9.3f} "
            f"{violations_str:>11} {r['intervention_rate']:>8.3f}% "
            f"{r['backup_rate']:>8.3f}% {r['feasibility_rate']:>6.3f} "
            f"{r['exact_rate']:>7.3f} {cbf_min:>8} {targets_str:>8}"
        )

    print("=" * 100)


def print_aggregate_table(all_results: list[dict]):
    """Print aggregate results across seeds."""
    from collections import defaultdict

    by_config = defaultdict(list)
    for r in all_results:
        by_config[r["config"]].append(r)

    print("\n" + "=" * 90)
    print("AGGREGATE RESULTS (mean +/- std across seeds)")
    print("=" * 90)

    header = (
        f"{'Config':<8} {'Cap Rate':>12} {'Violations':>12} "
        f"{'Interv %':>12} {'Exact %':>12} {'CBF Margin':>12} {'Targets':>8}"
    )
    print(header)
    print("-" * 90)

    for label in sorted(by_config.keys()):
        runs = by_config[label]
        cr = np.array([r["capture_rate"] for r in runs])
        viol = np.array([r["violations"] for r in runs])
        interv = np.array([r["intervention_rate"] for r in runs])
        exact = np.array([r["exact_rate"] for r in runs])
        margins = [r["mean_cbf_margin"] for r in runs if not np.isnan(r["mean_cbf_margin"])]
        margin_arr = np.array(margins) if margins else np.array([float("nan")])
        all_pass = all(r["all_targets_met"] for r in runs)

        margin_str = f"{np.mean(margin_arr):.3f}" if not np.isnan(np.mean(margin_arr)) else "N/A"
        targets_str = "ALL PASS" if all_pass else "SOME FAIL"

        print(
            f"{label:<8} {np.mean(cr):>5.3f}+/-{np.std(cr):.3f} "
            f"{np.mean(viol):>5.1f}+/-{np.std(viol):.1f} "
            f"{np.mean(interv):>5.3f}+/-{np.std(interv):.3f} "
            f"{np.mean(exact):>5.3f}+/-{np.std(exact):.3f} "
            f"{margin_str:>12} {targets_str:>8}"
        )

    print("=" * 90)


def save_results(all_results: list[dict], output_path: str):
    """Save results to JSON file."""
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = []
    for r in all_results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, dict):
                sr[k] = {kk: convert(vv) for kk, vv in v.items()}
            else:
                sr[k] = convert(v)
        serializable.append(sr)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Ablation Study")
    parser.add_argument(
        "--configs", nargs="+", default=["A", "B", "C", "D", "E"],
        help="Config labels to run (default: all)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=10,
        help="Episodes per config per seed (default: 10)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="Random seeds (default: [42])",
    )
    parser.add_argument(
        "--output", type=str, default="results/ablation_results.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    configs = get_ablation_configs()
    selected = {k: v for k, v in configs.items() if k in args.configs}

    if not selected:
        print(f"No valid configs selected from {args.configs}")
        print(f"Available: {list(configs.keys())}")
        return

    print(f"Running ablation study: configs={list(selected.keys())}, "
          f"seeds={args.seeds}, episodes={args.n_episodes}")

    all_results = []
    total = len(selected) * len(args.seeds)
    idx = 0

    for label, config in selected.items():
        for seed in args.seeds:
            idx += 1
            print(f"\n[{idx}/{total}] Running config {label} (seed={seed})...")
            result = run_single_config(label, config, args.n_episodes, seed)
            all_results.append(result)
            print(f"  Capture rate: {result['capture_rate']:.3f}, "
                  f"Violations: {result['violations']}, "
                  f"Exact rate: {result['exact_rate']:.3f}, "
                  f"Time: {result['elapsed_s']:.1f}s")

    # Print comparison tables
    print_comparison_table(all_results)
    if len(args.seeds) > 1:
        print_aggregate_table(all_results)

    # Save results
    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
