"""Random Network Distillation (RND) for intrinsic exploration motivation.

Provides exploration bonus when the evader's extrinsic reward is uninformative
(e.g., during curriculum transitions where the evader hasn't learned the new level).

Reference: Burda et al., "Exploration by Random Network Distillation", ICLR 2019.

Architecture:
  - Target network: frozen random MLP (fixed at init)
  - Predictor network: trainable MLP that tries to match target output
  - Intrinsic reward = ||target(obs) - predictor(obs)||^2
  - High error => novel state => high intrinsic reward

The RNDRewardWrapper wraps the evader's environment and augments the reward
with the normalized intrinsic signal.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RunningMeanStd:
    """Welford's online algorithm for tracking mean and variance.

    Used to normalize the intrinsic reward signal.
    """

    def __init__(self, shape: tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # Avoid division by zero

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a batch of values.

        Args:
            x: Array of shape [batch, *shape] or [*shape].
        """
        if x.ndim == 0:
            x = x.reshape(1)
        if x.ndim == 1 and self.mean.ndim == 0:
            # Scalar tracking with batch input
            batch_mean = np.mean(x)
            batch_var = np.var(x)
            batch_count = x.shape[0]
        elif x.ndim > len(self.mean.shape):
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
        else:
            batch_mean = x
            batch_var = np.zeros_like(self.mean)
            batch_count = 1

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)


class RNDModule(nn.Module):
    """Random Network Distillation module.

    Contains a frozen target MLP and a trainable predictor MLP.
    Both map observations to an embedding space.

    Args:
        obs_dim: Observation dimension (e.g., 43 for state[7] + lidar[36]).
        embed_dim: Output embedding dimension.
        hidden_dim: Hidden layer width.
        lr: Learning rate for predictor network.
    """

    def __init__(
        self,
        obs_dim: int = 43,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim

        # Target: frozen random network
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Freeze target
        for param in self.target.parameters():
            param.requires_grad = False

        # Predictor: trainable network (same architecture)
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

        # Running stats for reward normalization
        self.reward_rms = RunningMeanStd()

        # Observation buffer for batched training
        self._obs_buffer: list[np.ndarray] = []

    def compute_intrinsic_reward(self, obs: np.ndarray) -> float:
        """Compute intrinsic reward for a single observation.

        Args:
            obs: Raw observation array of shape [obs_dim].

        Returns:
            Unnormalized intrinsic reward (prediction error).
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            target_feat = self.target(obs_t)
            pred_feat = self.predictor(obs_t)
            error = (target_feat - pred_feat).pow(2).sum(dim=-1)
        return error.item()

    def compute_normalized_reward(self, obs: np.ndarray) -> float:
        """Compute normalized intrinsic reward.

        Args:
            obs: Raw observation array of shape [obs_dim].

        Returns:
            Normalized intrinsic reward.
        """
        raw = self.compute_intrinsic_reward(obs)
        # Update running stats
        self.reward_rms.update(np.array([raw]))
        # Normalize
        normalized = raw / self.reward_rms.std
        return float(normalized)

    def train_predictor(self, obs_batch: np.ndarray) -> float:
        """Train the predictor network on a batch of observations.

        Args:
            obs_batch: Array of shape [batch, obs_dim].

        Returns:
            Mean prediction loss.
        """
        obs_t = torch.FloatTensor(obs_batch)
        with torch.no_grad():
            target_feat = self.target(obs_t)
        pred_feat = self.predictor(obs_t)

        loss = (target_feat - pred_feat).pow(2).sum(dim=-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def add_to_buffer(self, obs: np.ndarray) -> None:
        """Add observation to training buffer."""
        self._obs_buffer.append(obs.copy())

    def train_from_buffer(self, max_samples: int = 1024) -> float | None:
        """Train predictor from buffered observations, then clear buffer.

        Args:
            max_samples: Maximum samples to use from buffer.

        Returns:
            Training loss, or None if buffer was empty.
        """
        if not self._obs_buffer:
            return None
        batch = np.array(self._obs_buffer[-max_samples:])
        self._obs_buffer.clear()
        return self.train_predictor(batch)


class RNDRewardWrapper(gym.Wrapper):
    """Gymnasium wrapper that augments reward with RND intrinsic motivation.

    Wraps the evader's environment, adding `rnd_coef * normalized_intrinsic`
    to the extrinsic reward at each step.

    Args:
        env: Gymnasium environment to wrap.
        rnd_module: RNDModule instance (shared if needed).
        rnd_coef: Coefficient for intrinsic reward.
        update_freq: Train predictor every N steps.
    """

    def __init__(
        self,
        env: gym.Env,
        rnd_module: RNDModule,
        rnd_coef: float = 0.1,
        update_freq: int = 256,
    ):
        super().__init__(env)
        self.rnd = rnd_module
        self.rnd_coef = rnd_coef
        self.update_freq = update_freq
        self._step_count = 0

    @staticmethod
    def _flatten_obs(obs) -> np.ndarray:
        """Flatten observation to 1D array for RND module.

        Handles both flat Box observations and Dict observations
        (e.g., from PartialObsWrapper with state/lidar/obs_history keys).
        """
        if isinstance(obs, dict):
            # Concatenate all values, flattening each
            parts = []
            for key in sorted(obs.keys()):
                parts.append(np.asarray(obs[key]).flatten())
            return np.concatenate(parts)
        return np.asarray(obs).flatten()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Flatten obs for RND (handles Dict obs spaces)
        flat_obs = self._flatten_obs(obs)

        # Compute intrinsic reward
        intrinsic = self.rnd.compute_normalized_reward(flat_obs)
        augmented_reward = reward + self.rnd_coef * intrinsic

        # Store for predictor training
        self.rnd.add_to_buffer(flat_obs)
        self._step_count += 1

        # Periodically train predictor
        if self._step_count % self.update_freq == 0:
            self.rnd.train_from_buffer()

        # Log intrinsic reward in info
        info["rnd_intrinsic"] = intrinsic
        info["rnd_extrinsic"] = reward
        info["rnd_total"] = augmented_reward

        return obs, augmented_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
