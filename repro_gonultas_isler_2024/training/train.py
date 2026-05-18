import numpy as np
import torch
import torch.nn as nn
import yaml
from copy import deepcopy
from pathlib import Path

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.maddpg import MADDPGAgent
from training.replay_buffer import ReplayBuffer
from training.curriculum import Curriculum
from training.belief_manager import BeliefStateManager
from training.vectorized_env import VectorizedEnvRunner
from utils.logger import Logger


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str = "config.yaml"):
    config = load_config(config_path)

    # Curriculum
    curriculum = Curriculum(
        total_episodes=config["maddpg"]["episodes"],
        t_end_ratio=config["curriculum"]["t_end_ratio"],
        fov_start=config["curriculum"]["fov_start"],
        fov_end=config["curriculum"]["fov_end"],
        evader_vmax_start=config["curriculum"]["evader_vmax_start"],
        evader_vmax_end=config["curriculum"]["evader_vmax_end"],
    )

    # Toggles
    use_belief = config.get("use_belief", False)
    use_amp = config.get("use_amp", False) and torch.cuda.is_available()

    # Obs dimensions
    raw_obs_dim = 11
    belief_dim = 2 if use_belief else 0
    frame_stack = config["maddpg"]["frame_stack"]
    obs_dim = (raw_obs_dim + belief_dim) * frame_stack
    action_dim = 2
    total_obs_dim = obs_dim * 2
    total_action_dim = action_dim * 2

    # Parallel envs
    n_envs = config["maddpg"].get("parallel_envs", 1)
    update_freq = config["maddpg"].get("update_freq", 1)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        use_amp=use_amp,
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
        use_amp=use_amp,
    )

    # EMA (Exponential Moving Average) opponent networks for stable training
    ema_tau = config["maddpg"].get("ema_tau", 0.0)
    use_ema = ema_tau > 0.0
    if use_ema:
        ema_pursuer = deepcopy(pursuer.actor)
        ema_evader = deepcopy(evader.actor)
        ema_pursuer.eval()
        ema_evader.eval()
        for param in ema_pursuer.parameters():
            param.requires_grad = False
        for param in ema_evader.parameters():
            param.requires_grad = False
    else:
        ema_pursuer = None
        ema_evader = None

    # Replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config["maddpg"]["replay_buffer_size"],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # Output directory (include seed to prevent overwriting)
    output_dir = Path(config.get("output_dir", "."))
    seed = config.get("seed", 0)
    output_dir = output_dir / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    log_dir = output_dir / config.get("log_dir", "runs")
    logger = Logger(str(log_dir))

    # Training parameters
    episodes = config["maddpg"]["episodes"]
    batch_size = config["maddpg"]["batch_size"]
    noise_sigma = config["maddpg"]["exploration_sigma"]
    frame_skip = config["maddpg"]["frame_skip"]
    bimdn_update_freq = config["bimdn"]["belief_update_freq"]
    checkpoint_freq = config["eval"]["checkpoint_freq"]

    # Vectorized env runner
    runner = VectorizedEnvRunner(
        n_envs=n_envs,
        config=config,
        use_belief=use_belief,
        frame_stack=frame_stack,
        frame_skip=frame_skip,
        device=device,
    )
    runner.reset_all()

    # Apply initial curriculum
    fov = curriculum.get_fov(0)
    evader_vmax = curriculum.get_evader_vmax(0)
    runner.set_curriculum_params(fov, evader_vmax)

    # Training loop: step-based with continuous rollout
    completed_episodes = 0
    global_step = 0
    episodes_since_log = 0
    last_checkpoint_ep = 0

    # Episode metric aggregators
    agg_lengths = []
    agg_rewards_p = []
    agg_rewards_e = []
    agg_captures = []

    # Loss tracking for logging
    last_c_loss_p = 0.0
    last_a_loss_p = 0.0
    last_c_loss_e = 0.0
    last_a_loss_e = 0.0

    ema_status = f"EMA=ON (tau={ema_tau})" if use_ema else "EMA=OFF (vanilla SP)"
    print(f"Starting training: {episodes} episodes, {n_envs} parallel envs, device={device}, {ema_status}")
    logger.log_text("config/ema_status", ema_status, 0)
    if use_ema:
        logger.log_scalar("config/ema_tau", ema_tau, 0)

    while completed_episodes < episodes:
        # Get stacked observations for all envs
        stacked_p, stacked_e = runner.get_stacked_obs()

        # Select actions
        actions_p = [pursuer.select_action(o, noise_sigma) for o in stacked_p]
        actions_e = [evader.select_action(o, noise_sigma) for o in stacked_e]

        # Step all envs
        transitions, completed_eps, bimdn_info = runner.step(
            actions_p, actions_e, global_step, bimdn_update_freq
        )

        # Store transitions in replay buffer
        for t in transitions:
            replay_buffer.add(
                obs_p=t["obs_p"],
                obs_e=t["obs_e"],
                action_p=t["action_p"],
                action_e=t["action_e"],
                reward_p=t["reward_p"],
                reward_e=t["reward_e"],
                next_obs_p=t["next_obs_p"],
                next_obs_e=t["next_obs_e"],
                done=t["done"],
            )

        global_step += 1

        # MADDPG updates: maintain ~1 update per env step
        if replay_buffer.size >= batch_size:
            n_updates = n_envs * update_freq
            for _ in range(n_updates):
                batch = replay_buffer.sample(batch_size)
                if use_ema:
                    c_loss_p, a_loss_p = pursuer.update(batch, ema_evader, agent_id="pursuer")
                    c_loss_e, a_loss_e = evader.update(batch, ema_pursuer, agent_id="evader")
                else:
                    c_loss_p, a_loss_p = pursuer.update(batch, evader.actor, agent_id="pursuer")
                    c_loss_e, a_loss_e = evader.update(batch, pursuer.actor, agent_id="evader")
                pursuer.update_targets()
                evader.update_targets()
            # Update EMA networks after all gradient updates
            if use_ema:
                with torch.no_grad():
                    for ema_param, param in zip(ema_evader.parameters(), evader.actor.parameters()):
                        ema_param.data.mul_(1 - ema_tau).add_(param.data, alpha=ema_tau)
                    for ema_param, param in zip(ema_pursuer.parameters(), pursuer.actor.parameters()):
                        ema_param.data.mul_(1 - ema_tau).add_(param.data, alpha=ema_tau)
            last_c_loss_p = c_loss_p
            last_a_loss_p = a_loss_p
            last_c_loss_e = c_loss_e
            last_a_loss_e = a_loss_e

        # Accumulate completed episode metrics
        for ep_data in completed_eps:
            completed_episodes += 1
            episodes_since_log += 1
            agg_lengths.append(ep_data["length"])
            agg_rewards_p.append(ep_data["pursuer_reward"])
            agg_rewards_e.append(ep_data["evader_reward"])
            agg_captures.append(1.0 if ep_data["captured"] else 0.0)

        # Update curriculum when episodes complete
        if len(completed_eps) > 0:
            fov = curriculum.get_fov(completed_episodes)
            evader_vmax = curriculum.get_evader_vmax(completed_episodes)
            runner.set_curriculum_params(fov, evader_vmax)

        # Logging: aggregate and log every ~log_interval episodes
        log_interval = max(1, n_envs)
        if episodes_since_log >= log_interval and len(agg_lengths) > 0:
            logger.log_scalar("episode/length", np.mean(agg_lengths), completed_episodes)
            logger.log_scalar("episode/pursuer_reward", np.mean(agg_rewards_p), completed_episodes)
            logger.log_scalar("episode/evader_reward", np.mean(agg_rewards_e), completed_episodes)
            logger.log_scalar("episode/capture_rate", np.mean(agg_captures), completed_episodes)

            if replay_buffer.size >= batch_size:
                logger.log_scalar("loss/pursuer_critic", last_c_loss_p, completed_episodes)
                logger.log_scalar("loss/pursuer_actor", last_a_loss_p, completed_episodes)
                logger.log_scalar("loss/evader_critic", last_c_loss_e, completed_episodes)
                logger.log_scalar("loss/evader_actor", last_a_loss_e, completed_episodes)

            if use_belief and bimdn_info["updates"] > 0:
                logger.log_scalar("loss/bimdn_pursuer", bimdn_info["loss_p"], completed_episodes)
                logger.log_scalar("loss/bimdn_evader", bimdn_info["loss_e"], completed_episodes)

            # Console print for monitoring
            print(
                f"Ep {completed_episodes:5d} | "
                f"CR={np.mean(agg_captures):.3f} | "
                f"Len={np.mean(agg_lengths):.1f} | "
                f"PC={last_c_loss_p:.1f} PA={last_a_loss_p:.3f} | "
                f"EC={last_c_loss_e:.1f} EA={last_a_loss_e:.3f} | "
                f"BP={bimdn_info['loss_p']:.3f} BE={bimdn_info['loss_e']:.3f}"
            )

            # Reset aggregators
            episodes_since_log = 0
            agg_lengths = []
            agg_rewards_p = []
            agg_rewards_e = []
            agg_captures = []

        # Checkpoint
        if completed_episodes // checkpoint_freq > last_checkpoint_ep // checkpoint_freq:
            last_checkpoint_ep = (completed_episodes // checkpoint_freq) * checkpoint_freq
            checkpoint_dir = output_dir / "checkpoints" / f"episode_{last_checkpoint_ep}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(pursuer.actor.state_dict(), checkpoint_dir / "pursuer_actor.pt")
            torch.save(evader.actor.state_dict(), checkpoint_dir / "evader_actor.pt")
            if use_ema:
                torch.save(ema_pursuer.state_dict(), checkpoint_dir / "ema_pursuer.pt")
                torch.save(ema_evader.state_dict(), checkpoint_dir / "ema_evader.pt")
            if use_belief:
                for i in range(n_envs):
                    # Save only the first env's belief models (they're all the same architecture,
                    # and we only need one for evaluation)
                    runner.belief_pursuers[0].save(checkpoint_dir / "bimdn_pursuer.pt")
                    runner.belief_evaders[0].save(checkpoint_dir / "bimdn_evader.pt")
                    break

    # Final checkpoint
    final_dir = output_dir / "checkpoints" / f"episode_{completed_episodes}"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pursuer.actor.state_dict(), final_dir / "pursuer_actor.pt")
    torch.save(evader.actor.state_dict(), final_dir / "evader_actor.pt")
    if use_ema:
        torch.save(ema_pursuer.state_dict(), final_dir / "ema_pursuer.pt")
        torch.save(ema_evader.state_dict(), final_dir / "ema_evader.pt")
    if use_belief:
        runner.belief_pursuers[0].save(final_dir / "bimdn_pursuer.pt")
        runner.belief_evaders[0].save(final_dir / "bimdn_evader.pt")

    logger.close()
    print(f"Training complete: {completed_episodes} episodes, {global_step} steps")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    train(config_path)
