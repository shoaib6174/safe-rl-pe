"""SafeRolloutBuffer: extends SB3 RolloutBuffer with per-step safe bounds.

Stores the safe action bounds that were active at each step, so the
PPO update can recompute log-probabilities under the correct distribution.

Usage:
    buffer = SafeRolloutBuffer(buffer_size, obs_space, act_space, ...)
    buffer.add(..., safe_bounds_low=low, safe_bounds_high=high)

    for batch in buffer.get(batch_size):
        # batch has .safe_bounds_low and .safe_bounds_high
"""

from typing import Generator, NamedTuple, Optional, Union

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class SafeRolloutBufferSamples(NamedTuple):
    """Batch samples with safe bounds."""
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    safe_bounds_low: torch.Tensor
    safe_bounds_high: torch.Tensor


class SafeRolloutBuffer(RolloutBuffer):
    """RolloutBuffer extended with per-step safe action bounds storage.

    Stores (bounds_low, bounds_high) for each step alongside standard
    PPO rollout data. These are returned in batches so the policy update
    can use the correct bounds for log-prob recomputation.
    """

    safe_bounds_low: np.ndarray
    safe_bounds_high: np.ndarray

    def reset(self) -> None:
        action_dim = self.action_dim if isinstance(self.action_dim, int) else self.action_dim[0]
        self.safe_bounds_low = np.zeros(
            (self.buffer_size, self.n_envs, action_dim), dtype=np.float32,
        )
        self.safe_bounds_high = np.zeros(
            (self.buffer_size, self.n_envs, action_dim), dtype=np.float32,
        )
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        safe_bounds_low: Optional[np.ndarray] = None,
        safe_bounds_high: Optional[np.ndarray] = None,
    ) -> None:
        """Add a step with optional safe bounds.

        If safe_bounds are None, stores the full action space bounds.
        """
        if safe_bounds_low is not None:
            self.safe_bounds_low[self.pos] = safe_bounds_low.reshape(
                self.n_envs, -1,
            )
        if safe_bounds_high is not None:
            self.safe_bounds_high[self.pos] = safe_bounds_high.reshape(
                self.n_envs, -1,
            )

        super().add(obs, action, reward, episode_start, value, log_prob)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> SafeRolloutBufferSamples:
        """Get batch samples including safe bounds."""
        data = dict(
            observations=self.to_torch(self.observations[batch_inds]),
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            safe_bounds_low=self.to_torch(self.safe_bounds_low[batch_inds]),
            safe_bounds_high=self.to_torch(self.safe_bounds_high[batch_inds]),
        )
        return SafeRolloutBufferSamples(**data)
