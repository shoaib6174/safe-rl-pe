#!/usr/bin/env python3
"""Quick robustness evaluation for symmetric point-mass seeds."""
import sys
sys.path.insert(0, "/home/niro-2/Codes/safe-rl-pe/repro_gonultas_isler_2024")

from evaluation.evaluate import run_evaluation
import yaml

with open("config_symmetric.yaml") as f:
    cfg = yaml.safe_load(f)

NUM_EPISODES = 100  # Quick check

for seed in [0, 1, 2]:
    cp_dir = f"./repro_gonultas_isler_2024/results_symmetric/seed_{seed}/seed_{seed}/checkpoints/episode_5000"
    print(f"=== Seed {seed} ===")
    print(f"Checkpoint: {cp_dir}")
    try:
        results = run_evaluation(cp_dir, cfg, num_episodes=NUM_EPISODES)
        print(f"  pursuer_vs_random:   CR={results['pursuer_vs_random']['capture_rate']:.3f}  len={results['pursuer_vs_random']['mean_length']:.1f}")
        print(f"  pursuer_vs_greedy:   CR={results['pursuer_vs_greedy']['capture_rate']:.3f}  len={results['pursuer_vs_greedy']['mean_length']:.1f}")
        print(f"  pursuer_vs_selfplay: CR={results['pursuer_vs_selfplay']['capture_rate']:.3f}  len={results['pursuer_vs_selfplay']['mean_length']:.1f}")
        print(f"  random_vs_evader:    CR={results['random_vs_evader']['capture_rate']:.3f}  len={results['random_vs_evader']['mean_length']:.1f}")
        print(f"  greedy_vs_evader:    CR={results['greedy_vs_evader']['capture_rate']:.3f}  len={results['greedy_vs_evader']['mean_length']:.1f}")
        if 'ema_pursuer_vs_evader' in results:
            print(f"  ema_pursuer_vs_evader: CR={results['ema_pursuer_vs_evader']['capture_rate']:.3f}")
        if 'ema_pursuer_vs_random' in results:
            print(f"  ema_pursuer_vs_random: CR={results['ema_pursuer_vs_random']['capture_rate']:.3f}")
        if 'ema_pursuer_vs_greedy' in results:
            print(f"  ema_pursuer_vs_greedy: CR={results['ema_pursuer_vs_greedy']['capture_rate']:.3f}")
        # Compute robustness metric
        cr_self = results['pursuer_vs_selfplay']['capture_rate']
        cr_rand = results['pursuer_vs_random']['capture_rate']
        cr_greedy = results['pursuer_vs_greedy']['capture_rate']
        robustness = (min(cr_self, 1-cr_self) + cr_rand + cr_greedy) / 3
        print(f"  ROBUSTNESS: {robustness:.3f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()
