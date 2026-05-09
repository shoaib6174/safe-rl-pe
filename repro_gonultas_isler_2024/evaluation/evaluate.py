import numpy as np
import torch
from pathlib import Path
from typing import Callable

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.maddpg import MADDPGAgent
from training.belief_manager import BeliefStateManager


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
    use_belief: bool = False,
    belief_pursuer: BeliefStateManager = None,
    belief_evader: BeliefStateManager = None,
) -> dict:
    capture_count = 0
    episode_lengths = []
    pursuer_rewards = []

    belief_norm = env.arena_size / 2.0

    for _ in range(num_episodes):
        obs_p, obs_e = env.reset()

        if use_belief:
            belief_pursuer.history.clear()
            belief_evader.history.clear()
            belief_pursuer.add_observation(obs_p)
            belief_evader.add_observation(obs_e)

        if use_belief:
            b_p = belief_pursuer.get_belief() / belief_norm
            b_e = belief_evader.get_belief() / belief_norm
            aug_obs_p = np.concatenate([obs_p, b_p])
            aug_obs_e = np.concatenate([obs_e, b_e])
        else:
            aug_obs_p = obs_p
            aug_obs_e = obs_e

        obs_p_buffer = [aug_obs_p.copy() for _ in range(frame_stack)]
        obs_e_buffer = [aug_obs_e.copy() for _ in range(frame_stack)]

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

            if use_belief:
                belief_pursuer.add_observation(obs_p)
                belief_evader.add_observation(obs_e)
                b_p = belief_pursuer.get_belief() / belief_norm
                b_e = belief_evader.get_belief() / belief_norm
                aug_obs_p = np.concatenate([obs_p, b_p])
                aug_obs_e = np.concatenate([obs_e, b_e])
            else:
                aug_obs_p = obs_p
                aug_obs_e = obs_e

            obs_p_buffer.append(aug_obs_p.copy())
            obs_e_buffer.append(aug_obs_e.copy())

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
    use_belief = config.get("use_belief", False)
    use_amp = config.get("use_amp", False) and torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_obs_dim = 11
    belief_dim = 2 if use_belief else 0
    obs_dim = (raw_obs_dim + belief_dim) * frame_stack
    action_dim = 2

    # Belief managers
    belief_pursuer = None
    belief_evader = None
    if use_belief:
        bimdn_cfg = config["bimdn"]
        belief_pursuer = BeliefStateManager(
            obs_dim=raw_obs_dim,
            buffer_size=bimdn_cfg["belief_buffer_size"],
            history_length=bimdn_cfg["history_length"],
            downsample=bimdn_cfg["downsample_factor"],
            lr=bimdn_cfg["belief_lr"],
            device=device,
            use_amp=use_amp,
        )
        belief_evader = BeliefStateManager(
            obs_dim=raw_obs_dim,
            buffer_size=bimdn_cfg["belief_buffer_size"],
            history_length=bimdn_cfg["history_length"],
            downsample=bimdn_cfg["downsample_factor"],
            lr=bimdn_cfg["belief_lr"],
            device=device,
            use_amp=use_amp,
        )

    pursuer = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=obs_dim * 2,
        total_action_dim=action_dim * 2,
        use_amp=use_amp,
        device=device,
    )
    evader = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=obs_dim * 2,
        total_action_dim=action_dim * 2,
        use_amp=use_amp,
        device=device,
    )

    checkpoint_path = Path(checkpoint_dir)
    load_policy(pursuer, checkpoint_path / "pursuer_actor.pt")
    load_policy(evader, checkpoint_path / "evader_actor.pt")

    if use_belief:
        belief_pursuer.load(checkpoint_path / "bimdn_pursuer.pt")
        belief_evader.load(checkpoint_path / "bimdn_evader.pt")

    def pursuer_nn_policy(obs):
        return pursuer.select_action(obs, noise_sigma=0.0)

    def evader_nn_policy(obs):
        return evader.select_action(obs, noise_sigma=0.0)

    results = {}

    results["pursuer_vs_random"] = evaluate_agent(
        env, pursuer_nn_policy, random_policy, num_episodes, frame_stack,
        use_belief, belief_pursuer, belief_evader,
    )

    results["pursuer_vs_greedy"] = evaluate_agent(
        env, pursuer_nn_policy, greedy_evader_policy, num_episodes, frame_stack,
        use_belief, belief_pursuer, belief_evader,
    )

    results["pursuer_vs_selfplay"] = evaluate_agent(
        env, pursuer_nn_policy, evader_nn_policy, num_episodes, frame_stack,
        use_belief, belief_pursuer, belief_evader,
    )

    results["random_vs_evader"] = evaluate_agent(
        env, random_policy, evader_nn_policy, num_episodes, frame_stack,
        use_belief, belief_pursuer, belief_evader,
    )

    results["greedy_vs_evader"] = evaluate_agent(
        env, greedy_pursuer_policy, evader_nn_policy, num_episodes, frame_stack,
        use_belief, belief_pursuer, belief_evader,
    )

    return results


if __name__ == "__main__":
    import yaml
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/episode_1000"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results = run_evaluation(checkpoint, config)
    for key, val in results.items():
        print(f"{key}: {val}")
