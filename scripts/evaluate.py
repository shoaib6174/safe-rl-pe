"""Evaluation script for trained PE agents.

Usage:
    python scripts/evaluate.py --model models/local_42/final_model --n_episodes 100
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper


def evaluate_agent(
    model_path: str,
    role: str = "pursuer",
    n_episodes: int = 100,
    render: bool = False,
    seed: int = 0,
) -> dict:
    """Evaluate a trained agent against a random opponent.

    Returns dict with capture_rate, mean_capture_time, mean_episode_length,
    mean_min_distance.
    """
    model = PPO.load(model_path)

    render_mode = "human" if render else None
    base_env = PursuitEvasionEnv(render_mode=render_mode)
    env = SingleAgentPEWrapper(base_env, role=role, opponent_policy=None)

    captures = 0
    capture_times = []
    episode_lengths = []
    min_distances = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if "episode_metrics" in info:
            m = info["episode_metrics"]
            if m["captured"]:
                captures += 1
                capture_times.append(m["capture_time"])
            episode_lengths.append(m["episode_length"])
            min_distances.append(m["min_distance"])

    env.close()

    results = {
        "capture_rate": captures / n_episodes,
        "mean_capture_time": np.mean(capture_times) if capture_times else float("nan"),
        "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
        "mean_min_distance": np.mean(min_distances) if min_distances else 0,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PE agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--role", type=str, default="pursuer", choices=["pursuer", "evader"])
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = evaluate_agent(
        args.model, args.role, args.n_episodes, args.render, args.seed,
    )

    print(f"\nEvaluation Results ({args.n_episodes} episodes):")
    print(f"  Capture rate:       {results['capture_rate']:.2%}")
    print(f"  Mean capture time:  {results['mean_capture_time']:.2f}s")
    print(f"  Mean episode len:   {results['mean_episode_length']:.0f} steps")
    print(f"  Mean min distance:  {results['mean_min_distance']:.2f}m")


if __name__ == "__main__":
    main()
