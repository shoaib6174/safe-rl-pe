#!/usr/bin/env python3
"""Evaluate symmetric point-mass seeds 9-33 (Experiment 3) on niro-1."""
import json
import sys
from pathlib import Path

# Allow imports from repro_gonultas_isler_2024 when run from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "repro_gonultas_isler_2024"))

import yaml
from evaluation.evaluate import run_evaluation


RESULTS_DIR = Path("repro_gonultas_isler_2024/results_symmetric")
OUTPUT_FILE = RESULTS_DIR / "eval_seeds_9_33.json"
NUM_EPISODES = 50

with open("repro_gonultas_isler_2024/config_symmetric.yaml") as f:
    cfg = yaml.safe_load(f)

all_results = {}

for seed in range(9, 34):
    cp_dir = RESULTS_DIR / f"seed_{seed}" / "checkpoints" / "episode_5000"
    print(f"\n=== Seed {seed} ===")
    print(f"Checkpoint: {cp_dir}")

    if not (cp_dir / "pursuer_actor.pt").exists():
        print("  ERROR: checkpoint not found")
        all_results[seed] = {"error": "checkpoint not found"}
        continue

    try:
        results = run_evaluation(str(cp_dir), cfg, num_episodes=NUM_EPISODES)
        summary = {}
        for key, val in results.items():
            summary[key] = {
                "capture_rate": float(val["capture_rate"]),
                "mean_length": float(val["mean_length"]),
                "std_length": float(val["std_length"]),
                "mean_reward": float(val["mean_reward"]),
            }
            print(f"  {key:30s} CR={val['capture_rate']:.3f}  len={val['mean_length']:.1f}")

        # Robustness metric (same as eval_symmetric.py)
        cr_self = summary["pursuer_vs_selfplay"]["capture_rate"]
        cr_rand = summary["pursuer_vs_random"]["capture_rate"]
        cr_greedy = summary["pursuer_vs_greedy"]["capture_rate"]
        robustness = (min(cr_self, 1 - cr_self) + cr_rand + cr_greedy) / 3
        summary["robustness"] = round(robustness, 3)
        print(f"  {'ROBUSTNESS':30s} {robustness:.3f}")

        all_results[seed] = summary

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_results[seed] = {"error": str(e)}

# Save JSON results
OUTPUT_FILE.write_text(json.dumps(all_results, indent=2))
print(f"\n\nResults saved to {OUTPUT_FILE}")

# Print final tally
print("\n=== Final Tally (CR pursuer_vs_selfplay) ===")
for seed in range(9, 34):
    res = all_results.get(seed, {})
    if "error" in res:
        print(f"  seed {seed:2d}: ERROR")
    else:
        cr = res["pursuer_vs_selfplay"]["capture_rate"]
        print(f"  seed {seed:2d}: CR={cr:.3f}")
