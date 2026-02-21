"""Custom SB3 ActorCriticPolicy using Beta distribution.

Replaces SB3's default Gaussian policy with Beta distribution for
naturally bounded continuous actions. This is essential for CBF
integration where action bounds are dynamically constrained.

Usage with SB3:
    from safety.beta_policy import BetaPolicy
    model = PPO(BetaPolicy, env, ...)

Reference: Paper [16] (Suttle 2024) — CBF-constrained Beta policy
"""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from safety.beta_distribution import BetaDistribution


class BetaPolicy(ActorCriticPolicy):
    """PPO policy using Beta distribution for naturally bounded actions.

    The Beta distribution has bounded support [0, 1], which is rescaled to
    the action space bounds [low, high]. This eliminates action clipping
    and ensures all sampled actions are within bounds.

    Compatible with CBF truncation: safe bounds can be dynamically narrowed
    to restrict the sampling range (Phase 2 Session 4+).

    Args:
        observation_space: Observation space.
        action_space: Action space (must be Box).
        lr_schedule: Learning rate schedule.
        **kwargs: Additional ActorCriticPolicy arguments (net_arch, etc.).
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        **kwargs,
    ):
        # Remove use_sde if passed (not applicable to Beta)
        kwargs.pop("use_sde", None)
        kwargs.pop("log_std_init", None)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """Build network architecture with Beta distribution.

        Overrides ActorCriticPolicy._build to replace the Gaussian distribution
        with our BetaDistribution.
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        latent_dim_vf = self.mlp_extractor.latent_dim_vf

        # Create Beta distribution
        action_dim = self.action_space.shape[0]
        self.action_dist = BetaDistribution(action_dim)

        # Action network: outputs 2 * action_dim (alpha + beta params)
        self.action_net = self.action_dist.proba_distribution_net(latent_dim_pi)

        # Store action bounds as buffers (auto-moved with .to(device))
        self.register_buffer(
            "_action_low",
            torch.tensor(self.action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "_action_high",
            torch.tensor(self.action_space.high, dtype=torch.float32),
        )

        # Value function head
        self.value_net = nn.Linear(latent_dim_vf, 1)

        # Initialize weights (same as SB3 default)
        if self.ortho_init:
            module_gains = {
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if self.share_features_extractor:
                module_gains[self.features_extractor] = np.sqrt(2)
            else:
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> BetaDistribution:
        """Get Beta action distribution from latent policy features.

        Overrides the parent's isinstance-based dispatch with direct
        Beta distribution creation.
        """
        params = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(
            params, self._action_low, self._action_high
        )

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: compute actions, values, and log_probs.

        This is called during rollout collection.
        """
        features = self.extract_features(obs, self.pi_features_extractor)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, self.action_space.shape[0]))
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate actions for PPO update (compute values, log_prob, entropy).

        This is called during PPO's optimization step.
        """
        features = self.extract_features(obs, self.pi_features_extractor)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor) -> BetaDistribution:
        """Get the action distribution for given observations."""
        features = self.extract_features(obs, self.pi_features_extractor)

        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features = features[0] if isinstance(features, tuple) else features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)

        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value predictions for given observations."""
        features = self.extract_features(obs, self.vf_features_extractor)

        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
        else:
            vf_features = features[1] if isinstance(features, tuple) else features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        return self.value_net(latent_vf)

    def _predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get actions from observations (used by predict())."""
        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )
