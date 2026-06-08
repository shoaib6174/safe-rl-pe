import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml
from copy import deepcopy

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.shared_maddpg import SharedMADDPGAgent
from training.replay_buffer import ReplayBuffer
from training.curriculum import Curriculum
from training.vectorized_env import VectorizedEnvRunner
from utils.logger import Logger


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str = "config.yaml", seed: int | None = None):
    config = load_config(config_path)
    if seed is not None:
        config["seed"] = seed

    # Curriculum
    curriculum = Curriculum(
        total_episodes=config["maddpg"]["episodes"],
        t_end_ratio=config["curriculum"]["t_end_ratio"],
        fov_start=config["curriculum"]["fov_start"],
        fov_end=config["curriculum"]["fov_end"],
        evader_vmax_start=config["curriculum"]["evader_vmax_start"],
        evader_vmax_end=config["curriculum"]["evader_vmax_end"],
    )

    use_amp = config.get("use_amp", False) and torch.cuda.is_available()

    # Obs dimensions
    raw_obs_dim = 11
    belief_dim = 0  # belief not supported with shared policy yet
    frame_stack = config["maddpg"]["frame_stack"]
    obs_dim = (raw_obs_dim + belief_dim) * frame_stack
    action_dim = 2
    total_obs_dim = obs_dim * 2
    total_action_dim = action_dim * 2

    n_envs = config["maddpg"].get("parallel_envs", 1)
    update_freq = config["maddpg"].get("update_freq", 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_compile = config.get("use_compile", False)

    # Create shared agent
    mcfg = config["maddpg"]
    agent = SharedMADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=total_obs_dim,
        total_action_dim=total_action_dim,
        actor_lr=mcfg["actor_lr"],
        critic_lr=mcfg["critic_lr"],
        gamma=mcfg["gamma"],
        tau=mcfg["tau"],
        hidden_dim=mcfg["hidden_dim"],
        device=device,
        use_amp=use_amp,
        use_compile=use_compile,
    )

    # EMA for shared actor
    ema_tau = config["maddpg"].get("ema_tau", 0.0)
    use_ema = ema_tau > 0.0
    if use_ema:
        ema_actor = deepcopy(agent.actor)
        ema_actor.eval()
        for param in ema_actor.parameters():
            param.requires_grad = False
    else:
        ema_actor = None

    # Replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config["maddpg"]["replay_buffer_size"],
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )

    # Output directory
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
    checkpoint_freq = config["eval"]["checkpoint_freq"]

    # Vectorized env runner
    runner = VectorizedEnvRunner(
        n_envs=n_envs,
        config=config,
        use_belief=False,
        frame_stack=frame_stack,
        frame_skip=frame_skip,
        device=device,
    )
    runner.reset_all()

    # Apply initial curriculum
    fov = curriculum.get_fov(0)
    evader_vmax = curriculum.get_evader_vmax(0)
    runner.set_curriculum_params(fov, evader_vmax)

    # Training loop
    completed_episodes = 0
    global_step = 0
    episodes_since_log = 0
    last_checkpoint_ep = 0

    agg_lengths = []
    agg_rewards_p = []
    agg_rewards_e = []
    agg_captures = []

    last_c_loss_p = 0.0
    last_a_loss_p = 0.0
    last_c_loss_e = 0.0
    last_a_loss_e = 0.0

    ema_status = f"SharedPolicy + EMA (tau={ema_tau})" if use_ema else "SharedPolicy (no EMA)"
    print(f"Starting training: {episodes} episodes, {n_envs} parallel envs, device={device}, {ema_status}")
    logger.log_text("config/mode", ema_status, 0)
    if use_ema:
        logger.log_scalar("config/ema_tau", ema_tau, 0)

    while completed_episodes < episodes:
        # Get stacked observations for all envs
        stacked_p, stacked_e = runner.get_stacked_obs()

        # Select actions using shared actor with role conditioning
        actions_p = agent.select_actions_batch(stacked_p, role=0, noise_sigma=noise_sigma)
        actions_e = agent.select_actions_batch(stacked_e, role=1, noise_sigma=noise_sigma)

        # Step all envs
        transitions, completed_eps, bimdn_info = runner.step(
            actions_p, actions_e, global_step, 10  # bimdn_update_freq not used
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

        # MADDPG updates
        if replay_buffer.size >= batch_size:
            n_updates = n_envs * update_freq
            for _ in range(n_updates):
                batch = replay_buffer.sample(batch_size)

                # Update critics (separate for pursuer and evader)
                c_loss_p, c_loss_e = agent.update_critics(batch, ema_actor)

                # Update shared actor (combined loss from both roles)
                a_loss_p, a_loss_e = agent.update_shared_actor(batch, ema_actor)

                # Soft update target networks
                agent.update_targets()

            # Update EMA network
            if use_ema:
                with torch.no_grad():
                    for ema_param, param in zip(ema_actor.parameters(), agent.actor.parameters()):
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

        # Update curriculum
        if len(completed_eps) > 0:
            fov = curriculum.get_fov(completed_episodes)
            evader_vmax = curriculum.get_evader_vmax(completed_episodes)
            runner.set_curriculum_params(fov, evader_vmax)

        # Logging
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

            print(
                f"Ep {completed_episodes:5d} | "
                f"CR={np.mean(agg_captures):.3f} | "
                f"Len={np.mean(agg_lengths):.1f} | "
                f"PC={last_c_loss_p:.1f} PA={last_a_loss_p:.3f} | "
                f"EC={last_c_loss_e:.1f} EA={last_a_loss_e:.3f}"
            )

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
            torch.save(agent.actor.state_dict(), checkpoint_dir / "shared_actor.pt")
            if use_ema:
                torch.save(ema_actor.state_dict(), checkpoint_dir / "ema_actor.pt")

    # Final checkpoint
    final_dir = output_dir / "checkpoints" / f"episode_{completed_episodes}"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.actor.state_dict(), final_dir / "shared_actor.pt")
    if use_ema:
        torch.save(ema_actor.state_dict(), final_dir / "ema_actor.pt")

    logger.close()
    print(f"Training complete: {completed_episodes} episodes, {global_step} steps")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    seed = None
    if len(sys.argv) > 3 and sys.argv[2] == "--seed":
        seed = int(sys.argv[3])
    train(config_path, seed)
