import numpy as np
import torch
from pathlib import Path
from typing import Callable

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.maddpg import MADDPGAgent


def random_policy(obs: np.ndarray) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, size=2)


def greedy_pursuer_policy(obs: np.ndarray) -> np.ndarray:
    opp_x, opp_y = obs[4], obs[5]
    own_x, own_y = obs[0], obs[1]
    dx = opp_x - own_x
    dy = opp_y - own_y
    angle = np.arctan2(dy, dx)
    return np.array([angle * 0.5, 1.0])


def greedy_evader_policy(obs: np.ndarray) -> np.ndarray:
    opp_x, opp_y = obs[4], obs[5]
    own_x, own_y = obs[0], obs[1]
    dx = own_x - opp_x
    dy = own_y - opp_y
    angle = np.arctan2(dy, dx)
    return np.array([angle * 0.5, 1.0])


def evaluate_agent(
    env: PursuitEvasionEnv,
    pursuer_policy: Callable,
    evader_policy: Callable,
    num_episodes: int = 500,
    frame_stack: int = 2,
) -> dict:
    capture_count = 0
    episode_lengths = []
    pursuer_rewards = []

    for _ in range(num_episodes):
        obs_p, obs_e = env.reset()
        obs_p_buffer = [obs_p.copy() for _ in range(frame_stack)]
        obs_e_buffer = [obs_e.copy() for _ in range(frame_stack)]

        done = False
        step = 0
        ep_reward = 0.0

        while not done and step < env.max_steps:
            stacked_p = np.concatenate(obs_p_buffer[-frame_stack:])
            stacked_e = np.concatenate(obs_e_buffer[-frame_stack:])

            action_p = pursuer_policy(stacked_p)
            action_e = evader_policy(stacked_e)

            obs_p, obs_e, r_p, r_e, done = env.step(action_p, action_e)
            ep_reward += r_p
            step += 1

            obs_p_buffer.append(obs_p.copy())
            obs_e_buffer.append(obs_e.copy())

        if ep_reward > 0:
            capture_count += 1
        episode_lengths.append(step)
        pursuer_rewards.append(ep_reward)

    return {
        "capture_rate": capture_count / num_episodes,
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "mean_reward": np.mean(pursuer_rewards),
    }


def load_policy(agent: MADDPGAgent, checkpoint_path: str):
    agent.actor.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    agent.actor.eval()


def run_evaluation(
    checkpoint_dir: str,
    config: dict,
    num_episodes: int = 500,
):
    env = PursuitEvasionEnv(config)
    frame_stack = config["maddpg"]["frame_stack"]
    obs_dim = 11 * frame_stack
    action_dim = 2

    pursuer = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=obs_dim * 2,
        total_action_dim=action_dim * 2,
    )
    evader = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=obs_dim * 2,
        total_action_dim=action_dim * 2,
    )

    checkpoint_path = Path(checkpoint_dir)
    load_policy(pursuer, checkpoint_path / "pursuer_actor.pt")
    load_policy(evader, checkpoint_path / "evader_actor.pt")

    def pursuer_nn_policy(obs):
        return pursuer.select_action(obs, noise_sigma=0.0)

    def evader_nn_policy(obs):
        return evader.select_action(obs, noise_sigma=0.0)

    results = {}

    results["pursuer_vs_random"] = evaluate_agent(
        env, pursuer_nn_policy, random_policy, num_episodes, frame_stack
    )

    results["pursuer_vs_greedy"] = evaluate_agent(
        env, pursuer_nn_policy, greedy_evader_policy, num_episodes, frame_stack
    )

    results["pursuer_vs_selfplay"] = evaluate_agent(
        env, pursuer_nn_policy, evader_nn_policy, num_episodes, frame_stack
    )

    results["random_vs_evader"] = evaluate_agent(
        env, random_policy, evader_nn_policy, num_episodes, frame_stack
    )

    results["greedy_vs_evader"] = evaluate_agent(
        env, greedy_pursuer_policy, evader_nn_policy, num_episodes, frame_stack
    )

    return results


if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    checkpoint = "checkpoints/episode_1000"
    results = run_evaluation(checkpoint, config)
    for key, val in results.items():
        print(f"{key}: {val}")
