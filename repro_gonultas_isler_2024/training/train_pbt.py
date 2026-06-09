import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml
from copy import deepcopy
import random

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.shared_maddpg import SharedMADDPGAgent
from training.replay_buffer import ReplayBuffer
from training.vectorized_env import VectorizedEnvRunner
from utils.logger import Logger


class PBTTrainer:
    """Population-Based Training for competitive self-play.

    Maintains a population of agents. Each agent can play either role via
    role conditioning. During training, agents are randomly paired against
    each other. Periodically, the weakest agent is replaced by a mutated
    copy of the strongest, maintaining strategic diversity.
    """

    def __init__(
        self,
        population_size: int,
        config: dict,
        device: str,
        use_amp: bool = False,
        use_compile: bool = False,
    ):
        self.population_size = population_size
        self.device = device
        self.config = config

        raw_obs_dim = 11
        frame_stack = config["maddpg"]["frame_stack"]
        obs_dim = raw_obs_dim * frame_stack
        action_dim = 2
        total_obs_dim = obs_dim * 2
        total_action_dim = action_dim * 2

        mcfg = config["maddpg"]

        # Create population of agents
        self.population = []
        for _ in range(population_size):
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
            self.population.append(agent)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.frame_stack = frame_stack

        # Win tracking for tournament
        self.win_matrix = np.zeros((population_size, population_size))
        self.games_matrix = np.zeros((population_size, population_size))

    def assign_agents_to_envs(self, n_envs: int) -> list:
        """Randomly assign distinct agent pairs to each env.

        Returns list of (pursuer_idx, evader_idx) tuples.
        """
        assignments = []
        for _ in range(n_envs):
            i, j = random.sample(range(self.population_size), 2)
            assignments.append((i, j))
        return assignments

    def select_actions(
        self, stacked_p: np.ndarray, stacked_e: np.ndarray,
        assignments: list, noise_sigma: float
    ) -> tuple:
        """Select actions for each env using assigned agents."""
        n_envs = len(assignments)
        actions_p = np.zeros((n_envs, self.action_dim), dtype=np.float32)
        actions_e = np.zeros((n_envs, self.action_dim), dtype=np.float32)

        for env_idx, (p_idx, e_idx) in enumerate(assignments):
            obs_p = np.expand_dims(stacked_p[env_idx], 0)
            obs_e = np.expand_dims(stacked_e[env_idx], 0)

            a_p = self.population[p_idx].select_actions_batch(
                obs_p, role=0, noise_sigma=noise_sigma
            )
            a_e = self.population[e_idx].select_actions_batch(
                obs_e, role=1, noise_sigma=noise_sigma
            )

            actions_p[env_idx] = a_p[0]
            actions_e[env_idx] = a_e[0]

        return actions_p, actions_e

    def update_random_agents(
        self, replay_buffer: ReplayBuffer, batch_size: int, n_updates: int
    ) -> list:
        """Randomly select agents and update from shared replay buffer.

        Returns list of (agent_idx, c_loss_p, c_loss_e, a_loss_p, a_loss_e).
        """
        results = []
        for _ in range(n_updates):
            idx = random.randint(0, self.population_size - 1)
            agent = self.population[idx]

            batch = replay_buffer.sample(batch_size)
            c_loss_p, c_loss_e = agent.update_critics(batch, ema_actor=None)
            a_loss_p, a_loss_e = agent.update_shared_actor(batch, ema_actor=None)
            agent.update_targets()

            results.append((idx, c_loss_p, c_loss_e, a_loss_p, a_loss_e))
        return results

    def evaluate_tournament(
        self, env_config: dict, n_games: int = 10
    ) -> np.ndarray:
        """Round-robin: each agent plays each other as pursuer.

        Returns win rate per agent (as pursuer).
        """
        self.win_matrix.fill(0)
        self.games_matrix.fill(0)

        env = PursuitEvasionEnv(env_config)

        for i in range(self.population_size):
            for j in range(self.population_size):
                if i == j:
                    continue

                wins_i = 0
                for _ in range(n_games):
                    obs_p, obs_e = env.reset()
                    obs_p_buf = [obs_p.copy() for _ in range(self.frame_stack)]
                    obs_e_buf = [obs_e.copy() for _ in range(self.frame_stack)]
                    done = False
                    step = 0

                    while not done and step < env.max_steps:
                        sp = np.concatenate(obs_p_buf[-self.frame_stack:])
                        se = np.concatenate(obs_e_buf[-self.frame_stack:])

                        with torch.no_grad():
                            a_p = self.population[i].select_action(sp, role=0, noise_sigma=0)
                            a_e = self.population[j].select_action(se, role=1, noise_sigma=0)

                        obs_p, obs_e, _, _, done = env.step(a_p, a_e)
                        obs_p_buf.append(obs_p.copy())
                        obs_e_buf.append(obs_e.copy())
                        step += 1

                    if step < env.max_steps:
                        wins_i += 1  # pursuer i won

                self.win_matrix[i, j] = wins_i
                self.games_matrix[i, j] = n_games

        win_rates = np.zeros(self.population_size)
        for i in range(self.population_size):
            total = self.games_matrix[i, :].sum()
            if total > 0:
                win_rates[i] = self.win_matrix[i, :].sum() / total
        return win_rates

    def replace_least_balanced(
        self, win_rates: np.ndarray, mutation_std: float = 0.01
    ) -> tuple:
        """Replace least-balanced agent with mutated copy of most-balanced.

        "Best" = win_rate closest to 0.5 (most competitive/balanced).
        "Worst" = win_rate furthest from 0.5 (most dominant or weakest).

        Returns (worst_idx, best_idx, did_replace).
        """
        distance_from_balance = np.abs(win_rates - 0.5)
        worst = int(np.argmax(distance_from_balance))
        best = int(np.argmin(distance_from_balance))

        if worst == best:
            return worst, best, False

        weak_agent = self.population[worst]
        strong_agent = self.population[best]

        # Copy actor weights + mutation noise
        with torch.no_grad():
            for tp, sp in zip(weak_agent.actor.parameters(),
                              strong_agent.actor.parameters()):
                tp.copy_(sp)
                tp.add_(torch.randn_like(tp) * mutation_std)

            # Copy critic weights (no mutation — keep value estimates)
            for tp, sp in zip(weak_agent.critic_p.parameters(),
                              strong_agent.critic_p.parameters()):
                tp.copy_(sp)
            for tp, sp in zip(weak_agent.critic_p2.parameters(),
                              strong_agent.critic_p2.parameters()):
                tp.copy_(sp)
            for tp, sp in zip(weak_agent.critic_e.parameters(),
                              strong_agent.critic_e.parameters()):
                tp.copy_(sp)
            for tp, sp in zip(weak_agent.critic_e2.parameters(),
                              strong_agent.critic_e2.parameters()):
                tp.copy_(sp)

        # Reset optimizers (clear momentum)
        for opt in [
            weak_agent.actor_optimizer,
            weak_agent.critic_p_optimizer,
            weak_agent.critic_p2_optimizer,
            weak_agent.critic_e_optimizer,
            weak_agent.critic_e2_optimizer,
        ]:
            opt.state.clear()

        # Sync target networks
        weak_agent.actor_target.load_state_dict(weak_agent.actor.state_dict())
        weak_agent.critic_p_target.load_state_dict(
            weak_agent.critic_p.state_dict()
        )
        weak_agent.critic_p2_target.load_state_dict(
            weak_agent.critic_p2.state_dict()
        )
        weak_agent.critic_e_target.load_state_dict(
            weak_agent.critic_e.state_dict()
        )
        weak_agent.critic_e2_target.load_state_dict(
            weak_agent.critic_e2.state_dict()
        )

        return worst, best, True


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str = "config.yaml", seed: int | None = None):
    config = load_config(config_path)
    if seed is not None:
        config["seed"] = seed

    # Seed RNGs
    seed = config.get("seed", 0)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    use_amp = config.get("use_amp", False) and torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_compile = config.get("use_compile", False)

    raw_obs_dim = 11
    frame_stack = config["maddpg"]["frame_stack"]
    obs_dim = raw_obs_dim * frame_stack
    action_dim = 2

    n_envs = config["maddpg"].get("parallel_envs", 2)
    update_freq = config["maddpg"].get("update_freq", 1)

    # PBT-specific config
    pbt_cfg = config.get("pbt", {})
    population_size = pbt_cfg.get("population_size", 5)
    replacement_freq = pbt_cfg.get("replacement_freq", 1000)
    tournament_games = pbt_cfg.get("tournament_games", 10)
    mutation_std = pbt_cfg.get("mutation_std", 0.01)

    episodes = config["maddpg"]["episodes"]
    batch_size = config["maddpg"]["batch_size"]
    noise_sigma = config["maddpg"]["exploration_sigma"]
    frame_skip = config["maddpg"]["frame_skip"]
    checkpoint_freq = config["eval"]["checkpoint_freq"]

    # Create PBT trainer
    pbt = PBTTrainer(
        population_size=population_size,
        config=config,
        device=device,
        use_amp=use_amp,
        use_compile=use_compile,
    )

    # Shared replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config["maddpg"]["replay_buffer_size"],
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )

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

    # Output directory
    output_dir = Path(config.get("output_dir", ".")) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    log_dir = output_dir / config.get("log_dir", "runs")
    logger = Logger(str(log_dir))

    # Training loop
    completed_episodes = 0
    global_step = 0
    episodes_since_log = 0
    last_checkpoint_ep = 0
    last_replacement_ep = 0

    agg_lengths = []
    agg_rewards_p = []
    agg_rewards_e = []
    agg_captures = []

    last_losses = {}

    print(
        f"Starting PBT: pop={population_size}, {episodes} eps, "
        f"{n_envs} envs, replace_every={replacement_freq}, device={device}"
    )
    logger.log_text("config/mode", f"PBT_pop{population_size}", 0)

    while completed_episodes < episodes:
        # Randomly assign agents to envs
        assignments = pbt.assign_agents_to_envs(n_envs)

        # Get stacked observations
        stacked_p, stacked_e = runner.get_stacked_obs()

        # Select actions
        actions_p, actions_e = pbt.select_actions(
            stacked_p, stacked_e, assignments, noise_sigma
        )

        # Step all envs
        transitions, completed_eps, _ = runner.step(
            actions_p, actions_e, global_step, 10
        )

        # Store transitions
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

        # PBT updates: random agents sample from shared buffer
        if replay_buffer.size >= batch_size:
            n_updates = n_envs * update_freq
            update_results = pbt.update_random_agents(
                replay_buffer, batch_size, n_updates
            )
            # Track last losses (use first updated agent for logging)
            if update_results:
                idx, cp, ce, ap, ae = update_results[0]
                last_losses = {
                    "c_loss_p": cp, "c_loss_e": ce,
                    "a_loss_p": ap, "a_loss_e": ae,
                }

        # Accumulate episode metrics
        for ep_data in completed_eps:
            completed_episodes += 1
            episodes_since_log += 1
            agg_lengths.append(ep_data["length"])
            agg_rewards_p.append(ep_data["pursuer_reward"])
            agg_rewards_e.append(ep_data["evader_reward"])
            agg_captures.append(1.0 if ep_data["captured"] else 0.0)

        # Logging
        log_interval = max(1, n_envs)
        if episodes_since_log >= log_interval and len(agg_lengths) > 0:
            logger.log_scalar("episode/length", np.mean(agg_lengths), completed_episodes)
            logger.log_scalar("episode/pursuer_reward", np.mean(agg_rewards_p), completed_episodes)
            logger.log_scalar("episode/evader_reward", np.mean(agg_rewards_e), completed_episodes)
            logger.log_scalar("episode/capture_rate", np.mean(agg_captures), completed_episodes)

            if last_losses:
                logger.log_scalar("loss/pursuer_critic", last_losses["c_loss_p"], completed_episodes)
                logger.log_scalar("loss/pursuer_actor", last_losses["a_loss_p"], completed_episodes)
                logger.log_scalar("loss/evader_critic", last_losses["c_loss_e"], completed_episodes)
                logger.log_scalar("loss/evader_actor", last_losses["a_loss_e"], completed_episodes)

            print(
                f"Ep {completed_episodes:5d} | "
                f"CR={np.mean(agg_captures):.3f} | "
                f"Len={np.mean(agg_lengths):.1f} | "
                f"PC={last_losses.get('c_loss_p', 0):.1f} "
                f"PA={last_losses.get('a_loss_p', 0):.3f} | "
                f"EC={last_losses.get('c_loss_e', 0):.1f} "
                f"EA={last_losses.get('a_loss_e', 0):.3f}"
            )

            episodes_since_log = 0
            agg_lengths = []
            agg_rewards_p = []
            agg_rewards_e = []
            agg_captures = []

        # Checkpoint: save all agents
        if completed_episodes // checkpoint_freq > last_checkpoint_ep // checkpoint_freq:
            last_checkpoint_ep = (completed_episodes // checkpoint_freq) * checkpoint_freq
            cp_dir = output_dir / "checkpoints" / f"episode_{last_checkpoint_ep}"
            cp_dir.mkdir(parents=True, exist_ok=True)
            for i, agent in enumerate(pbt.population):
                torch.save(agent.actor.state_dict(), cp_dir / f"agent_{i}_actor.pt")

        # PBT replacement: evaluate tournament and replace weakest
        if completed_episodes // replacement_freq > last_replacement_ep // replacement_freq:
            last_replacement_ep = (completed_episodes // replacement_freq) * replacement_freq
            print(f"\n--- PBT Tournament at episode {completed_episodes} ---")
            win_rates = pbt.evaluate_tournament(config, n_games=tournament_games)

            for i, wr in enumerate(win_rates):
                print(f"  Agent {i}: win_rate={wr:.3f}")
                logger.log_scalar(f"pbt/win_rate_agent_{i}", wr, completed_episodes)

            worst, best, replaced = pbt.replace_least_balanced(win_rates, mutation_std)
            if replaced:
                print(
                    f"  REPLACED: Agent {worst} (WR={win_rates[worst]:.3f}, "
                    f"|Δ|={abs(win_rates[worst]-0.5):.3f}) "
                    f"<- Agent {best} (WR={win_rates[best]:.3f}, "
                    f"|Δ|={abs(win_rates[best]-0.5):.3f}) + noise"
                )
                logger.log_scalar("pbt/replaced_agent", worst, completed_episodes)
                logger.log_scalar("pbt/replacement_source", best, completed_episodes)
            else:
                print("  No replacement (single most-balanced agent)")
            print("---")

    # Final checkpoint
    final_dir = output_dir / "checkpoints" / f"episode_{completed_episodes}"
    final_dir.mkdir(parents=True, exist_ok=True)
    for i, agent in enumerate(pbt.population):
        torch.save(agent.actor.state_dict(), final_dir / f"agent_{i}_actor.pt")

    logger.close()
    print(f"PBT Training complete: {completed_episodes} episodes, {global_step} steps")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    seed = None
    if len(sys.argv) > 3 and sys.argv[2] == "--seed":
        seed = int(sys.argv[3])
    train(config_path, seed)
