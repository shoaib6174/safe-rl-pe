import numpy as np
import torch
import yaml
from pathlib import Path

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.maddpg import MADDPGAgent
from training.replay_buffer import ReplayBuffer
from training.curriculum import Curriculum
from training.belief_manager import BeliefStateManager
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
    use_belief: bool = False,
    belief_pursuer: BeliefStateManager = None,
    belief_evader: BeliefStateManager = None,
    bimdn_update_freq: int = 10,
    global_step: int = 0,
) -> tuple:
    obs_p, obs_e = env.reset()

    # Raw observation history buffers for BiMDN
    if use_belief:
        belief_pursuer.history.clear()
        belief_evader.history.clear()

    # Frame stacking buffers (with belief if enabled)
    obs_p_buffer = []
    obs_e_buffer = []

    episode_data = {
        "pursuer_reward": 0.0,
        "evader_reward": 0.0,
        "length": 0,
        "captured": False,
        "bimdn_loss_p": 0.0,
        "bimdn_loss_e": 0.0,
    }

    done = False
    step = 0
    bimdn_updates = 0

    while not done and step < env.max_steps:
        # Add raw observations to BiMDN history
        if use_belief:
            belief_pursuer.add_observation(obs_p)
            belief_evader.add_observation(obs_e)

        # Compute beliefs
        if use_belief:
            belief_p = belief_pursuer.get_belief()
            belief_e = belief_evader.get_belief()
            aug_obs_p = np.concatenate([obs_p, belief_p])
            aug_obs_e = np.concatenate([obs_e, belief_e])
        else:
            aug_obs_p = obs_p
            aug_obs_e = obs_e

        # Frame stack
        if len(obs_p_buffer) < frame_stack:
            obs_p_buffer.extend([aug_obs_p.copy()] * (frame_stack - len(obs_p_buffer)))
            obs_e_buffer.extend([aug_obs_e.copy()] * (frame_stack - len(obs_e_buffer)))

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
            global_step += 1

            # Add next obs to history too
            if use_belief:
                belief_pursuer.add_observation(next_obs_p)
                belief_evader.add_observation(next_obs_e)

            # Store BiMDN training samples
            if use_belief and global_step % 1 == 0:
                # Ground truth opponent positions
                evader_pos = env.evader_state[:2]
                pursuer_pos = env.pursuer_state[:2]
                belief_pursuer.store_training_sample(evader_pos)
                belief_evader.store_training_sample(pursuer_pos)

            # BiMDN update
            if use_belief and global_step % bimdn_update_freq == 0:
                if len(belief_pursuer.training_buffer) >= 256:
                    loss_p = belief_pursuer.train_step(batch_size=256)
                    loss_e = belief_evader.train_step(batch_size=256)
                    episode_data["bimdn_loss_p"] += loss_p
                    episode_data["bimdn_loss_e"] += loss_e
                    bimdn_updates += 1

            if done:
                break

        # Compute next augmented obs for replay buffer
        if use_belief:
            next_belief_p = belief_pursuer.get_belief()
            next_belief_e = belief_evader.get_belief()
            next_aug_obs_p = np.concatenate([next_obs_p, next_belief_p])
            next_aug_obs_e = np.concatenate([next_obs_e, next_belief_e])
        else:
            next_aug_obs_p = next_obs_p
            next_aug_obs_e = next_obs_e

        # Store transition
        replay_buffer.add(
            obs_p=stacked_p,
            obs_e=stacked_e,
            action_p=action_p,
            action_e=action_e,
            reward_p=r_p,
            reward_e=r_e,
            next_obs_p=np.concatenate(obs_p_buffer[-(frame_stack - 1):] + [next_aug_obs_p]),
            next_obs_e=np.concatenate(obs_e_buffer[-(frame_stack - 1):] + [next_aug_obs_e]),
            done=done,
        )

        obs_p = next_obs_p
        obs_e = next_obs_e
        obs_p_buffer.append(next_aug_obs_p.copy())
        obs_e_buffer.append(next_aug_obs_e.copy())

    episode_data["length"] = step
    episode_data["captured"] = (episode_data["pursuer_reward"] > 0)
    if bimdn_updates > 0:
        episode_data["bimdn_loss_p"] /= bimdn_updates
        episode_data["bimdn_loss_e"] /= bimdn_updates

    return episode_data, global_step


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

    # Belief toggle
    use_belief = config.get("use_belief", False)

    # Obs dimensions
    raw_obs_dim = 11
    belief_dim = 2 if use_belief else 0
    frame_stack = config["maddpg"]["frame_stack"]
    obs_dim = (raw_obs_dim + belief_dim) * frame_stack
    action_dim = 2
    total_obs_dim = obs_dim * 2
    total_action_dim = action_dim * 2

    # Belief managers
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        )
        belief_evader = BeliefStateManager(
            obs_dim=raw_obs_dim,
            buffer_size=bimdn_cfg["belief_buffer_size"],
            history_length=bimdn_cfg["history_length"],
            downsample=bimdn_cfg["downsample_factor"],
            lr=bimdn_cfg["belief_lr"],
            device=device,
        )

    # Agents
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
    bimdn_update_freq = config["bimdn"]["belief_update_freq"]

    global_step = 0
    for episode in range(episodes):
        # Update curriculum
        fov = curriculum.get_fov(episode)
        evader_vmax = curriculum.get_evader_vmax(episode)
        env.pursuer_sensor.fov_angle = fov
        env.evader_v_max = evader_vmax

        # Run episode
        ep_data, global_step = run_episode(
            env, pursuer, evader, replay_buffer,
            noise_sigma, frame_stack, frame_skip,
            use_belief, belief_pursuer, belief_evader,
            bimdn_update_freq, global_step,
        )

        logger.log_scalar("episode/length", ep_data["length"], episode)
        logger.log_scalar("episode/pursuer_reward", ep_data["pursuer_reward"], episode)
        logger.log_scalar("episode/evader_reward", ep_data["evader_reward"], episode)
        logger.log_scalar("episode/capture_rate", float(ep_data["captured"]), episode)

        if use_belief and ep_data["bimdn_loss_p"] > 0:
            logger.log_scalar("loss/bimdn_pursuer", ep_data["bimdn_loss_p"], episode)
            logger.log_scalar("loss/bimdn_evader", ep_data["bimdn_loss_e"], episode)

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

        # Checkpoint
        if episode % config["eval"]["checkpoint_freq"] == 0:
            checkpoint_dir = Path("checkpoints") / f"episode_{episode}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(pursuer.actor.state_dict(), checkpoint_dir / "pursuer_actor.pt")
            torch.save(evader.actor.state_dict(), checkpoint_dir / "evader_actor.pt")
            if use_belief:
                belief_pursuer.save(checkpoint_dir / "bimdn_pursuer.pt")
                belief_evader.save(checkpoint_dir / "bimdn_evader.pt")

    logger.close()


if __name__ == "__main__":
    train()
