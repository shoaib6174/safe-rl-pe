from typing import List, Optional, Tuple
import numpy as np

from envs.pursuit_evasion_env import PursuitEvasionEnv
from training.belief_manager import BeliefStateManager


class VectorizedEnvRunner:
    """Manages N independent envs, stepping them in lockstep with per-env frame stacking and beliefs."""

    def __init__(
        self,
        n_envs: int,
        config: dict,
        use_belief: bool,
        frame_stack: int,
        frame_skip: int,
        device: str,
    ):
        self.n_envs = n_envs
        self.config = config
        self.use_belief = use_belief
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.device = device

        # Independent env instances
        self.envs = [PursuitEvasionEnv(config) for _ in range(n_envs)]
        self.belief_norm = self.envs[0].arena_size / 2.0
        self.bimdn_warmup = config["bimdn"].get("belief_warmup", 10000)
        self.bimdn_samples_collected = 0

        # Per-env frame stacking buffers (hold augmented observations)
        self.obs_p_buffers: List[List[np.ndarray]] = [[] for _ in range(n_envs)]
        self.obs_e_buffers: List[List[np.ndarray]] = [[] for _ in range(n_envs)]

        # Per-env belief managers
        self.belief_pursuers: Optional[List[BeliefStateManager]] = None
        self.belief_evaders: Optional[List[BeliefStateManager]] = None
        if use_belief:
            bimdn_cfg = config["bimdn"]
            raw_obs_dim = 11
            self.belief_pursuers = [
                BeliefStateManager(
                    obs_dim=raw_obs_dim,
                    buffer_size=bimdn_cfg["belief_buffer_size"],
                    history_length=bimdn_cfg["history_length"],
                    downsample=bimdn_cfg["downsample_factor"],
                    lr=bimdn_cfg["belief_lr"],
                    device=device,
                    use_amp=False,
                )
                for _ in range(n_envs)
            ]
            self.belief_evaders = [
                BeliefStateManager(
                    obs_dim=raw_obs_dim,
                    buffer_size=bimdn_cfg["belief_buffer_size"],
                    history_length=bimdn_cfg["history_length"],
                    downsample=bimdn_cfg["downsample_factor"],
                    lr=bimdn_cfg["belief_lr"],
                    device=device,
                    use_amp=False,
                )
                for _ in range(n_envs)
            ]

        # Episode tracking
        self.episode_rewards_p = [0.0] * n_envs
        self.episode_rewards_e = [0.0] * n_envs
        self.episode_lengths = [0] * n_envs
        self.episode_captured = [False] * n_envs

        # Recent completions (list of episode data dicts)
        self.completed_episodes: List[dict] = []

    def reset_all(self):
        """Reset all envs and initialize per-env state."""
        for i in range(self.n_envs):
            self._reset_env(i)

    def _reset_env(self, env_idx: int):
        """Reset a single env and its per-env state."""
        obs_p, obs_e = self.envs[env_idx].reset()

        # Clear frame stacking buffers
        self.obs_p_buffers[env_idx] = []
        self.obs_e_buffers[env_idx] = []

        # Clear belief histories and seed with initial observation
        if self.use_belief:
            self.belief_pursuers[env_idx].history.clear()
            self.belief_evaders[env_idx].history.clear()
            self.belief_pursuers[env_idx].add_observation(obs_p)
            self.belief_evaders[env_idx].add_observation(obs_e)

        # Build augmented observations (with normalized beliefs)
        aug_obs_p, aug_obs_e = self._get_augmented_obs(env_idx, obs_p, obs_e)

        # Initialize frame stacking buffers by padding with the first observation
        self.obs_p_buffers[env_idx] = [aug_obs_p.copy() for _ in range(self.frame_stack)]
        self.obs_e_buffers[env_idx] = [aug_obs_e.copy() for _ in range(self.frame_stack)]

        # Reset episode accumulators
        self.episode_rewards_p[env_idx] = 0.0
        self.episode_rewards_e[env_idx] = 0.0
        self.episode_lengths[env_idx] = 0
        self.episode_captured[env_idx] = False

    def _get_augmented_obs(
        self, env_idx: int, obs_p: np.ndarray, obs_e: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate beliefs (normalized) with raw observations."""
        if self.use_belief:
            belief_p = self.belief_pursuers[env_idx].get_belief() / self.belief_norm
            belief_e = self.belief_evaders[env_idx].get_belief() / self.belief_norm
            aug_obs_p = np.concatenate([obs_p, belief_p])
            aug_obs_e = np.concatenate([obs_e, belief_e])
        else:
            aug_obs_p = obs_p
            aug_obs_e = obs_e
        return aug_obs_p, aug_obs_e

    def get_stacked_obs(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return frame-stacked observations for all envs."""
        stacked_p = []
        stacked_e = []
        for i in range(self.n_envs):
            sp = np.concatenate(self.obs_p_buffers[i][-self.frame_stack:])
            se = np.concatenate(self.obs_e_buffers[i][-self.frame_stack:])
            stacked_p.append(sp)
            stacked_e.append(se)
        return stacked_p, stacked_e

    def step(
        self,
        actions_p: List[np.ndarray],
        actions_e: List[np.ndarray],
        global_step: int,
        bimdn_update_freq: int,
    ) -> Tuple[List[dict], List[dict], dict]:
        """
        Step all envs for one training step (with frame skip).

        Returns:
            transitions: list of transition dicts (one per env) for replay buffer
            completed_episodes: list of completed episode data dicts
            bimdn_info: dict with 'loss_p', 'loss_e', 'updates' across all envs
        """
        transitions = []
        completed_episodes = []
        bimdn_updates = 0
        bimdn_loss_p = 0.0
        bimdn_loss_e = 0.0

        for i in range(self.n_envs):
            env = self.envs[i]
            action_p = actions_p[i]
            action_e = actions_e[i]

            # Current stacked obs (before step) for replay buffer
            stacked_p = np.concatenate(self.obs_p_buffers[i][-self.frame_stack:])
            stacked_e = np.concatenate(self.obs_e_buffers[i][-self.frame_stack:])

            # Frame skip loop
            total_r_p = 0.0
            total_r_e = 0.0
            done = False
            steps_taken = 0

            for _ in range(self.frame_skip):
                next_obs_p, next_obs_e, r_p, r_e, done = env.step(action_p, action_e)
                total_r_p += r_p
                total_r_e += r_e
                steps_taken += 1

                # Update belief histories with next observation
                if self.use_belief:
                    self.belief_pursuers[i].add_observation(next_obs_p)
                    self.belief_evaders[i].add_observation(next_obs_e)

                    # Store BiMDN training samples (normalized ground truth positions)
                    evader_pos = env.evader_state[:2] / self.belief_norm
                    pursuer_pos = env.pursuer_state[:2] / self.belief_norm
                    self.belief_pursuers[i].store_training_sample(evader_pos)
                    self.belief_evaders[i].store_training_sample(pursuer_pos)
                    self.bimdn_samples_collected += 1

                if done:
                    break

            # Build next augmented observations
            next_aug_obs_p, next_aug_obs_e = self._get_augmented_obs(i, next_obs_p, next_obs_e)

            # Build next stacked obs for replay buffer:
            # concatenate (frame_stack-1) most recent buffer entries + next_aug_obs
            next_stacked_p = np.concatenate(
                self.obs_p_buffers[i][-(self.frame_stack - 1):] + [next_aug_obs_p]
            )
            next_stacked_e = np.concatenate(
                self.obs_e_buffers[i][-(self.frame_stack - 1):] + [next_aug_obs_e]
            )

            # Accumulate episode stats
            self.episode_rewards_p[i] += total_r_p
            self.episode_rewards_e[i] += total_r_e
            self.episode_lengths[i] += steps_taken
            if total_r_p > 0:
                self.episode_captured[i] = True

            # Store transition
            transitions.append({
                "obs_p": stacked_p,
                "obs_e": stacked_e,
                "action_p": action_p,
                "action_e": action_e,
                "reward_p": total_r_p,
                "reward_e": total_r_e,
                "next_obs_p": next_stacked_p,
                "next_obs_e": next_stacked_e,
                "done": done,
            })

            # Update buffers for next step
            self.obs_p_buffers[i].append(next_aug_obs_p.copy())
            self.obs_e_buffers[i].append(next_aug_obs_e.copy())

            # Handle episode completion
            if done:
                completed_episodes.append({
                    "pursuer_reward": self.episode_rewards_p[i],
                    "evader_reward": self.episode_rewards_e[i],
                    "length": self.episode_lengths[i],
                    "captured": self.episode_captured[i],
                })
                self._reset_env(i)

        # BiMDN update (global, not per-env)
        if (self.use_belief
            and global_step > 0
            and global_step % bimdn_update_freq == 0
            and self.bimdn_samples_collected >= self.bimdn_warmup):
            for i in range(self.n_envs):
                if len(self.belief_pursuers[i].training_buffer) >= 256:
                    loss_p = self.belief_pursuers[i].train_step(batch_size=256)
                    loss_e = self.belief_evaders[i].train_step(batch_size=256)
                    bimdn_loss_p += loss_p
                    bimdn_loss_e += loss_e
                    bimdn_updates += 1

        bimdn_info = {
            "loss_p": bimdn_loss_p / bimdn_updates if bimdn_updates > 0 else 0.0,
            "loss_e": bimdn_loss_e / bimdn_updates if bimdn_updates > 0 else 0.0,
            "updates": bimdn_updates,
        }

        return transitions, completed_episodes, bimdn_info

    def set_curriculum_params(self, fov: float, evader_vmax: float):
        """Apply curriculum parameters to all envs."""
        for env in self.envs:
            env.pursuer_sensor.fov_angle = fov
            env.evader_v_max = evader_vmax
