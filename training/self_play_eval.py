"""Evaluation utilities for self-play: evaluate both agents head-to-head."""

import numpy as np

from envs.pursuit_evasion_env import PursuitEvasionEnv


def evaluate_both_agents(
    pursuer_model,
    evader_model,
    env: PursuitEvasionEnv,
    n_episodes: int = 100,
    seed: int = 0,
) -> dict:
    """Evaluate pursuer vs evader head-to-head.

    Both agents act deterministically. The base env is stepped with
    both actions simultaneously.

    Returns:
        Dict with capture_rate, escape_rate, mean_episode_length,
        mean_capture_time, mean_min_distance.
    """
    captures = 0
    timeouts = 0
    episode_lengths = []
    capture_times = []
    min_distances = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False

        while not done:
            p_action, _ = pursuer_model.predict(obs["pursuer"], deterministic=True)
            e_action, _ = evader_model.predict(obs["evader"], deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(p_action, e_action)
            done = terminated or truncated

        if "episode_metrics" in info:
            m = info["episode_metrics"]
            if m["captured"]:
                captures += 1
                capture_times.append(m["capture_time"])
            else:
                timeouts += 1
            episode_lengths.append(m["episode_length"])
            min_distances.append(m["min_distance"])

    return {
        "capture_rate": captures / n_episodes,
        "escape_rate": timeouts / n_episodes,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0,
        "mean_capture_time": float(np.mean(capture_times)) if capture_times else float("nan"),
        "mean_min_distance": float(np.mean(min_distances)) if min_distances else 0,
    }


def collect_trajectories(
    pursuer_model,
    evader_model,
    env: PursuitEvasionEnv,
    n_episodes: int = 20,
    seed: int = 0,
) -> list[list[tuple[float, float]]]:
    """Collect evader endpoint trajectories for diversity analysis.

    Returns list of trajectories, where each trajectory is a list of (x, y) positions.
    """
    trajectories = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        positions = []
        done = False

        while not done:
            p_action, _ = pursuer_model.predict(obs["pursuer"], deterministic=True)
            e_action, _ = evader_model.predict(obs["evader"], deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(p_action, e_action)
            done = terminated or truncated

            # Record evader position for diversity check
            positions.append(
                (float(env.evader_state[0]), float(env.evader_state[1]))
            )

        trajectories.append(positions)

    return trajectories
