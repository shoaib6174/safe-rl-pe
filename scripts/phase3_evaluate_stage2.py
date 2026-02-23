"""Stage 2 Evaluation — evaluate trained models against Gate 2 criteria.

Evaluates the three Stage 2 configurations (2A, 2B, 2C) and compares
their performance.

Usage:
    ./venv/bin/python scripts/phase3_evaluate_stage2.py \
        --input results/stage2 --n_episodes 200
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from stable_baselines3 import PPO

from scripts.phase3_train_stage2 import make_phase3_env


def evaluate_model(
    model_path: str,
    config: dict,
    n_episodes: int = 200,
    seed: int = 99,
):
    """Evaluate a trained model and return metrics."""
    model = PPO.load(model_path)

    env = make_phase3_env(
        encoder_type=config["encoder_type"],
        use_dcbf=config["use_dcbf"],
        gamma=config.get("gamma", 0.2),
        seed=seed,
    )

    captures = 0
    timeouts = 0
    collisions = 0
    episode_rewards = []
    episode_lengths = []
    dcbf_intervention_rates = []
    inference_times = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0
        ep_len = 0

        while not done:
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            inference_times.append(time.time() - t0)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

        if info.get("is_capture", False):
            captures += 1
        else:
            timeouts += 1

        if info.get("collision", False):
            collisions += 1

        if "dcbf_intervention_rate" in info:
            dcbf_intervention_rates.append(info["dcbf_intervention_rate"])

    env.close()

    capture_rate = captures / n_episodes
    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_inference = np.mean(inference_times) * 1000  # ms

    results = {
        "capture_rate": capture_rate,
        "captures": captures,
        "timeouts": timeouts,
        "collisions": collisions,
        "mean_reward": float(mean_reward),
        "std_reward": float(np.std(episode_rewards)),
        "mean_episode_length": float(mean_length),
        "mean_inference_ms": float(mean_inference),
        "n_episodes": n_episodes,
    }

    if dcbf_intervention_rates:
        results["mean_dcbf_intervention_rate"] = float(
            np.mean(dcbf_intervention_rates)
        )

    return results


def check_gate2(results: dict, config_name: str) -> dict:
    """Check Gate 2 criteria for a single run."""
    checks = {}

    cr = results["capture_rate"]
    checks["capture_rate_pass"] = cr > 0.40
    checks["capture_rate_target"] = cr > 0.80
    checks["capture_rate"] = cr

    checks["collisions"] = results["collisions"]
    checks["zero_collisions"] = results["collisions"] == 0

    inference = results["mean_inference_ms"]
    checks["inference_ms"] = inference
    checks["inference_pass"] = inference < 20.0
    checks["inference_target"] = inference < 5.0

    if "mean_dcbf_intervention_rate" in results:
        ir = results["mean_dcbf_intervention_rate"]
        checks["dcbf_intervention_rate"] = ir
        checks["dcbf_intervention_pass"] = ir < 0.40
        checks["dcbf_intervention_target"] = ir < 0.15

    # Overall gate
    hard_fails = []
    if not checks["capture_rate_pass"]:
        hard_fails.append(f"Capture rate {cr:.1%} < 40% (HARD FAIL)")
    if not checks["zero_collisions"]:
        hard_fails.append(f"{results['collisions']} collisions (HARD FAIL)")
    if not checks["inference_pass"]:
        hard_fails.append(f"Inference {inference:.1f}ms > 20ms (HARD FAIL)")
    if "dcbf_intervention_rate" in checks and not checks["dcbf_intervention_pass"]:
        hard_fails.append(
            f"DCBF intervention {checks['dcbf_intervention_rate']:.1%} > 40%"
        )

    checks["gate_passed"] = len(hard_fails) == 0
    checks["hard_fails"] = hard_fails

    return checks


def print_comparison(all_results: dict):
    """Print comparison table across configs."""
    print(f"\n{'='*70}")
    print("STAGE 2 COMPARISON")
    print(f"{'='*70}")

    header = f"{'Metric':<30}"
    for name in sorted(all_results.keys()):
        header += f" {name:>12}"
    print(header)
    print("-" * 70)

    metrics = [
        ("Capture Rate", "capture_rate", ".1%"),
        ("Mean Reward", "mean_reward", ".2f"),
        ("Mean Ep Length", "mean_episode_length", ".0f"),
        ("Collisions", "collisions", "d"),
        ("Inference (ms)", "mean_inference_ms", ".2f"),
        ("DCBF Interv. Rate", "mean_dcbf_intervention_rate", ".1%"),
    ]

    for label, key, fmt in metrics:
        row = f"{label:<30}"
        for name in sorted(all_results.keys()):
            val = all_results[name].get(key)
            if val is not None:
                row += f" {val:>12{fmt}}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Comparison insights
    names = sorted(all_results.keys())
    if "2A" in all_results and "2B" in all_results:
        cr_a = all_results["2A"]["capture_rate"]
        cr_b = all_results["2B"]["capture_rate"]
        print(f"\n2A vs 2B (BiMDN helps?): "
              f"{'YES' if cr_a >= cr_b else 'NO'} "
              f"({cr_a:.1%} vs {cr_b:.1%})")

    if "2A" in all_results and "2C" in all_results:
        cr_a = all_results["2A"]["capture_rate"]
        cr_c = all_results["2C"]["capture_rate"]
        diff = abs(cr_a - cr_c)
        print(f"2A vs 2C (DCBF cost): "
              f"{'OK' if diff < 0.10 else 'HIGH'} "
              f"({cr_a:.1%} vs {cr_c:.1%}, diff={diff:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Evaluation")
    parser.add_argument("--input", type=str, default="results/stage2",
                        help="Stage 2 output directory")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    all_results = {}
    all_gates = {}

    for config_name in ["2A", "2B", "2C"]:
        run_dir = os.path.join(args.input, f"run_{config_name}")
        model_path = os.path.join(run_dir, "final_model.zip")
        config_path = os.path.join(run_dir, "config.json")

        if not os.path.exists(model_path):
            print(f"Skipping {config_name}: no model at {model_path}")
            continue

        with open(config_path) as f:
            config = json.load(f)

        print(f"\nEvaluating {config_name}...")
        results = evaluate_model(
            model_path, config,
            n_episodes=args.n_episodes, seed=args.seed,
        )
        all_results[config_name] = results

        # Gate 2 check
        gate = check_gate2(results, config_name)
        all_gates[config_name] = gate

        print(f"  Capture rate: {results['capture_rate']:.1%}")
        print(f"  Mean reward: {results['mean_reward']:.2f}")
        print(f"  Collisions: {results['collisions']}")
        if "mean_dcbf_intervention_rate" in results:
            print(f"  DCBF intervention: "
                  f"{results['mean_dcbf_intervention_rate']:.1%}")
        print(f"  Gate 2: {'PASSED' if gate['gate_passed'] else 'FAILED'}")
        if gate["hard_fails"]:
            for fail in gate["hard_fails"]:
                print(f"    {fail}")

    if all_results:
        print_comparison(all_results)

        # Save results
        output_path = os.path.join(args.input, "evaluation_results.json")
        with open(output_path, "w") as f:
            json.dump({"results": all_results, "gates": all_gates},
                      f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

        # Overall gate
        if "2A" in all_gates:
            if all_gates["2A"]["gate_passed"]:
                print("\nGATE 2: PASSED — proceed to Stage 3")
            else:
                print("\nGATE 2: FAILED — see failure protocols in "
                      "docs/phases/phase3_training_policy.md")


if __name__ == "__main__":
    main()
