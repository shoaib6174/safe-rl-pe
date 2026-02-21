"""SafeBetaPolicy: BetaPolicy with dynamic CBF-derived safe bounds.

Extends BetaPolicy to accept per-step safe bounds from VCPCBFFilter,
so the Beta distribution only samples within the safe action region.

During rollout:
    policy.set_safe_bounds(v_bounds, omega_bounds)
    distribution = policy.get_distribution(obs)
    action = distribution.sample()  # guaranteed within safe bounds

During PPO update:
    policy.evaluate_actions_with_bounds(obs, actions, stored_bounds)
    # recomputes log_prob using the same bounds from rollout

Reference: Phase 2 Session 4 spec
"""

from typing import Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from safety.beta_distribution import BetaDistribution
from safety.beta_policy import BetaPolicy


class SafeBetaPolicy(BetaPolicy):
    """BetaPolicy with dynamic safe action bounds from CBF.

    Adds the ability to set per-step safe bounds that restrict the
    Beta distribution's support to the CBF-safe region. When no safe
    bounds are set, falls back to the nominal action space bounds.

    The safe bounds are stored alongside rollout data so that PPO's
    policy update can recompute log-probabilities using the correct
    (original) bounds for each step.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # Dynamic safe bounds (None = use nominal)
        self._safe_low: Optional[torch.Tensor] = None
        self._safe_high: Optional[torch.Tensor] = None

    def set_safe_bounds(
        self,
        v_bounds: Tuple[float, float],
        omega_bounds: Tuple[float, float],
    ) -> None:
        """Set dynamic safe bounds for next action sampling.

        Args:
            v_bounds: (v_min, v_max) safe velocity bounds.
            omega_bounds: (omega_min, omega_max) safe angular velocity bounds.
        """
        low = torch.tensor(
            [v_bounds[0], omega_bounds[0]],
            dtype=torch.float32,
            device=self._action_low.device,
        )
        high = torch.tensor(
            [v_bounds[1], omega_bounds[1]],
            dtype=torch.float32,
            device=self._action_high.device,
        )
        # Clamp to nominal bounds (safe bounds can't exceed action space)
        self._safe_low = torch.max(low, self._action_low)
        self._safe_high = torch.min(high, self._action_high)

        # Ensure min <= max (handle numerical edge cases)
        self._safe_high = torch.max(self._safe_high, self._safe_low + 1e-6)

    def clear_safe_bounds(self) -> None:
        """Reset to nominal action space bounds."""
        self._safe_low = None
        self._safe_high = None

    def get_current_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get currently active bounds (safe or nominal).

        Returns:
            (low, high) tensors.
        """
        if self._safe_low is not None:
            return self._safe_low, self._safe_high
        return self._action_low, self._action_high

    def get_current_bounds_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current bounds as numpy arrays (for buffer storage)."""
        low, high = self.get_current_bounds()
        return low.detach().cpu().numpy(), high.detach().cpu().numpy()

    def _get_action_dist_from_latent(
        self, latent_pi: torch.Tensor
    ) -> BetaDistribution:
        """Get Beta distribution using current safe bounds."""
        params = self.action_net(latent_pi)
        low, high = self.get_current_bounds()
        return self.action_dist.proba_distribution(params, low, high)

    def _get_action_dist_with_bounds(
        self,
        latent_pi: torch.Tensor,
        bounds_low: torch.Tensor,
        bounds_high: torch.Tensor,
    ) -> BetaDistribution:
        """Get Beta distribution with explicitly provided bounds.

        Used during PPO update to recompute log-probs with stored bounds.
        """
        params = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(params, bounds_low, bounds_high)

    def evaluate_actions_with_bounds(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        stored_bounds_low: torch.Tensor,
        stored_bounds_high: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate actions using stored safe bounds from rollout.

        This is the key method for correct PPO updates with dynamic bounds.
        Uses the bounds that were active when each action was sampled.

        Args:
            obs: Observations batch.
            actions: Actions batch.
            stored_bounds_low: Low bounds per step, shape (batch, action_dim).
            stored_bounds_high: High bounds per step, shape (batch, action_dim).

        Returns:
            (values, log_prob, entropy) tuple.
        """
        features = self.extract_features(obs, self.pi_features_extractor)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_with_bounds(
            latent_pi, stored_bounds_low, stored_bounds_high,
        )
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy
