import numpy as np
import torch
import yaml
from pathlib import Path

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.maddpg import MADDPGAgent
from agents.bimdn import BiMDN
from training.replay_buffer import ReplayBuffer
from training.curriculum import Curriculum
from utils.logger import Logger


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_episode(
    env: PursuitEvasionEnv,
    pursuer: MADDPGAgent,
    evader: MADDPGAgent,
    replay_buffer: ReplayBuffer,
    noise_sigma: float,
    frame_stack: int,
    frame_skip: int,
) -> dict:
    obs_p, obs_e = env.reset()
    # Frame stacking buffers
    obs_p_buffer = [obs_p.copy() for _ in range(frame_stack)]
    obs_e_buffer = [obs_e.copy() for _ in range(frame_stack)]

    episode_data = {
        "pursuer_reward": 0.0,
        "evader_reward": 0.0,
        "length": 0,
        "captured": False,
    }

    done = False
    step = 0
    while not done and step < env.max_steps:
        # Stack observations
        stacked_p = np.concatenate(obs_p_buffer[-frame_stack:])
        stacked_e = np.concatenate(obs_e_buffer[-frame_stack:])

        # Select actions
        action_p = pursuer.select_action(stacked_p, noise_sigma)
        action_e = evader.select_action(stacked_e, noise_sigma)

        # Frame skip: repeat action for multiple sim steps
        for _ in range(frame_skip):
            next_obs_p, next_obs_e, r_p, r_e, done = env.step(action_p, action_e)
            episode_data["pursuer_reward"] += r_p
            episode_data["evader_reward"] += r_e
            step += 1
            if done:
                break

        # Store transition (at policy decision frequency)
        next_stacked_p = np.concatenate(obs_p_buffer[-(frame_stack - 1):] + [next_obs_p])
        next_stacked_e = np.concatenate(obs_e_buffer[-(frame_stack - 1):] + [next_obs_e])

        replay_buffer.add(
            obs_p=stacked_p,
            obs_e=stacked_e,
            action_p=action_p,
            action_e=action_e,
            reward_p=r_p,
            reward_e=r_e,
            next_obs_p=next_stacked_p,
            next_obs_e=next_stacked_e,
            done=done,
        )

        obs_p_buffer.append(next_obs_p.copy())
        obs_e_buffer.append(next_obs_e.copy())

    episode_data["length"] = step
    episode_data["captured"] = (episode_data["pursuer_reward"] > 0)
    return episode_data


def train(config_path: str = "config.yaml"):
    config = load_config(config_path)

    # Environment
    env = PursuitEvasionEnv(config)
    curriculum = Curriculum(
        total_episodes=config["maddpg"]["episodes"],
        t_end_ratio=config["curriculum"]["t_end_ratio"],
        fov_start=config["curriculum"]["fov_start"],
        fov_end=config["curriculum"]["fov_end"],
        evader_vmax_start=config["curriculum"]["evader_vmax_start"],
        evader_vmax_end=config["curriculum"]["evader_vmax_end"],
    )

    # Obs dimensions with frame stacking
    raw_obs_dim = 11
    frame_stack = config["maddpg"]["frame_stack"]
    obs_dim = raw_obs_dim * frame_stack  # 22 without belief, 26 with
    action_dim = 2
    total_obs_dim = obs_dim * 2  # both agents
    total_action_dim = action_dim * 2

    # Agents
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pursuer = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=total_obs_dim,
        total_action_dim=total_action_dim,
        actor_lr=config["maddpg"]["actor_lr"],
        critic_lr=config["maddpg"]["critic_lr"],
        gamma=config["maddpg"]["gamma"],
        tau=config["maddpg"]["tau"],
        device=device,
    )
    evader = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=total_obs_dim,
        total_action_dim=total_action_dim,
        actor_lr=config["maddpg"]["actor_lr"],
        critic_lr=config["maddpg"]["critic_lr"],
        gamma=config["maddpg"]["gamma"],
        tau=config["maddpg"]["tau"],
        device=device,
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config["maddpg"]["replay_buffer_size"],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # Logger
    logger = Logger(config["log_dir"])

    # Training loop
    episodes = config["maddpg"]["episodes"]
    batch_size = config["maddpg"]["batch_size"]
    noise_sigma = config["maddpg"]["exploration_sigma"]
    frame_skip = config["maddpg"]["frame_skip"]

    total_steps = 0
    for episode in range(episodes):
        # Update curriculum
        fov = curriculum.get_fov(episode)
        evader_vmax = curriculum.get_evader_vmax(episode)
        env.pursuer_sensor.fov_angle = fov
        env.evader_v_max = evader_vmax

        # Run episode
        ep_data = run_episode(
            env, pursuer, evader, replay_buffer,
            noise_sigma, frame_stack, frame_skip,
        )

        logger.log_scalar("episode/length", ep_data["length"], episode)
        logger.log_scalar("episode/pursuer_reward", ep_data["pursuer_reward"], episode)
        logger.log_scalar("episode/evader_reward", ep_data["evader_reward"], episode)
        logger.log_scalar("episode/capture_rate", float(ep_data["captured"]), episode)

        # Update MADDPG
        if replay_buffer.size >= batch_size:
            for _ in range(ep_data["length"]):
                batch = replay_buffer.sample(batch_size)
                c_loss_p, a_loss_p = pursuer.update(batch, evader)
                c_loss_e, a_loss_e = evader.update(batch, pursuer)
                pursuer.update_targets()
                evader.update_targets()

            logger.log_scalar("loss/pursuer_critic", c_loss_p, episode)
            logger.log_scalar("loss/pursuer_actor", a_loss_p, episode)
            logger.log_scalar("loss/evader_critic", c_loss_e, episode)
            logger.log_scalar("loss/evader_actor", a_loss_e, episode)

        total_steps += ep_data["length"]

        # Checkpoint
        if episode % config["eval"]["checkpoint_freq"] == 0:
            checkpoint_dir = Path("checkpoints") / f"episode_{episode}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(pursuer.actor.state_dict(), checkpoint_dir / "pursuer_actor.pt")
            torch.save(evader.actor.state_dict(), checkpoint_dir / "evader_actor.pt")

    logger.close()


if __name__ == "__main__":
    train()
