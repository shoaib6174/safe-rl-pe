#!/usr/bin/env python
"""
Evaluate all checkpoints from a training run.

Produces a CSV with per-checkpoint metrics:
- capture_rate vs random, greedy, self-play
- single-checkpoint robustness score (min of the three)
- EMA matchups (if EMA checkpoints exist)

Usage:
    python evaluate_checkpoint_sweep.py path/to/results/seed_0/config.yaml path/to/results/seed_0
"""

import sys
import csv
from pathlib import Path

import yaml
import numpy as np

from envs.pursuit_evasion_env import PursuitEvasionEnv
from evaluation.evaluate import run_evaluation


def find_checkpoints(run_dir: str):
    """Find all checkpoint directories sorted by episode number."""
    cp_dir = Path(run_dir) / "checkpoints"
    if not cp_dir.exists():
        return []
    checkpoints = []
    for p in cp_dir.glob("episode_*"):
        try:
            ep = int(p.name.split("_")[1])
            checkpoints.append((ep, p))
        except (IndexError, ValueError):
            continue
    return sorted(checkpoints, key=lambda x: x[0])


def compute_oscillation_amplitude(cr_values: list, window: int = 20) -> list:
    """Compute rolling standard deviation of capture rates."""
    amplitudes = []
    for i in range(len(cr_values)):
        start = max(0, i - window + 1)
        window_vals = cr_values[start:i + 1]
        amplitudes.append(np.std(window_vals))
    return amplitudes


def main(config_path: str, run_dir: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        print(f"No checkpoints found in {run_dir}/checkpoints")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints to evaluate")

    num_episodes = config["eval"]["eval_episodes"]
    rows = []

    for ep, cp_path in checkpoints:
        print(f"  Evaluating episode {ep}...", end=" ", flush=True)
        results = run_evaluation(str(cp_path), config, num_episodes=num_episodes)

        # Extract capture rates
        cr_random = results["pursuer_vs_random"]["capture_rate"]
        cr_greedy = results["pursuer_vs_greedy"]["capture_rate"]
        cr_selfplay = results["pursuer_vs_selfplay"]["capture_rate"]

        # Single-checkpoint robustness score
        robustness = min(cr_random, cr_greedy, cr_selfplay)

        row = {
            "episode": ep,
            "cr_random": cr_random,
            "cr_greedy": cr_greedy,
            "cr_selfplay": cr_selfplay,
            "robustness": robustness,
        }

        # Add EMA metrics if available
        if "ema_pursuer_vs_ema_evader" in results:
            row["cr_ema_selfplay"] = results["ema_pursuer_vs_ema_evader"]["capture_rate"]
        if "pursuer_vs_ema_evader" in results:
            row["cr_vs_ema_evader"] = results["pursuer_vs_ema_evader"]["capture_rate"]
        if "ema_pursuer_vs_evader" in results:
            row["cr_ema_vs_current"] = results["ema_pursuer_vs_evader"]["capture_rate"]

        rows.append(row)
        print(f"robustness={robustness:.3f}")

    # Compute oscillation amplitudes
    cr_selfplay_values = [r["cr_selfplay"] for r in rows]
    amplitudes = compute_oscillation_amplitude(cr_selfplay_values, window=20)
    for r, amp in zip(rows, amplitudes):
        r["oscillation_amp"] = amp

    # Write CSV
    output_path = Path(run_dir) / "checkpoint_metrics.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {output_path}")

    # Summary statistics
    best_robustness = max(rows, key=lambda r: r["robustness"])
    print(f"\nBest robustness: episode {best_robustness['episode']} = {best_robustness['robustness']:.3f}")
    print(f"  cr_random={best_robustness['cr_random']:.3f}")
    print(f"  cr_greedy={best_robustness['cr_greedy']:.3f}")
    print(f"  cr_selfplay={best_robustness['cr_selfplay']:.3f}")

    final_amp = rows[-1]["oscillation_amp"] if rows else 0.0
    print(f"\nFinal oscillation amplitude (last 20 checkpoints): {final_amp:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate_checkpoint_sweep.py <config.yaml> <run_dir>")
        print("  run_dir should contain checkpoints/episode_*")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
