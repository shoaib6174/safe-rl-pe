"""Beta distribution for bounded continuous actions, compatible with SB3.

The Beta distribution has support [0, 1] which is rescaled to [low, high].
This eliminates action clipping artifacts and ensures all samples are in-bounds.

Key properties:
- Naturally bounded: no probability mass outside action limits
- CBF-compatible: bounds can be dynamically narrowed for safety (Phase 2 Session 4+)
- Correct gradients: Jacobian correction ensures unbiased policy gradients
- Reparameterizable: uses rsample() for low-variance gradient estimates

Reference: Paper [16] (Suttle 2024) — CBF-constrained Beta policy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaDistribution:
    """SB3-compatible Beta distribution for bounded continuous actions.

    Follows the SB3 Distribution interface so it can be used with
    ActorCriticPolicy subclasses.

    Args:
        action_dim: Number of action dimensions.
    """

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.distribution = None
        self.low = None
        self.high = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """Create network layer that maps latent features to alpha/beta params.

        Returns nn.Linear with output dim = 2 * action_dim
        (first half: alpha params, second half: beta params).
        """
        return nn.Linear(latent_dim, 2 * self.action_dim)

    def proba_distribution(self, params: torch.Tensor,
                           low: torch.Tensor, high: torch.Tensor) -> "BetaDistribution":
        """Set distribution parameters from network output.

        Args:
            params: Raw network output, shape (..., 2 * action_dim).
            low: Lower bounds for actions, shape (action_dim,).
            high: Upper bounds for actions, shape (action_dim,).

        Returns:
            self (for chaining).
        """
        self.low = low
        self.high = high

        # Split into alpha and beta, ensure > 1 for unimodal distribution
        alpha = F.softplus(params[..., :self.action_dim]) + 1.0
        beta = F.softplus(params[..., self.action_dim:]) + 1.0

        self.distribution = torch.distributions.Beta(alpha, beta)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log probability with Jacobian correction for [low, high] rescaling.

        The change of variables from [0,1] to [low, high] requires subtracting
        log(high - low) per dimension to get the correct density.

        Args:
            actions: Actions in [low, high], shape (..., action_dim).

        Returns:
            Log probabilities, shape (...,).
        """
        scale = self.high - self.low
        # Map actions back to [0, 1]
        x = (actions - self.low) / scale
        x = torch.clamp(x, 1e-6, 1 - 1e-6)  # Numerical stability

        # Log prob = log Beta(x; alpha, beta) - log(scale) per dim
        return (self.distribution.log_prob(x) - torch.log(scale)).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Entropy of the rescaled Beta distribution.

        Rescaling shifts entropy by log(scale) per dimension.

        Returns:
            Entropy values, shape (...,).
        """
        scale = self.high - self.low
        return (self.distribution.entropy() + torch.log(scale)).sum(dim=-1)

    def sample(self) -> torch.Tensor:
        """Sample using reparameterization trick (rsample).

        Returns:
            Sampled actions in [low, high], shape (..., action_dim).
        """
        x = self.distribution.rsample()
        return self.low + (self.high - self.low) * x

    def mode(self) -> torch.Tensor:
        """Mode (deterministic action) of the Beta distribution.

        Mode of Beta(a, b) = (a - 1) / (a + b - 2) for a, b > 1.

        Returns:
            Mode actions in [low, high], shape (..., action_dim).
        """
        alpha = self.distribution.concentration1
        beta = self.distribution.concentration0
        mode_01 = (alpha - 1.0) / (alpha + beta - 2.0)
        mode_01 = torch.clamp(mode_01, 0.0, 1.0)
        return self.low + (self.high - self.low) * mode_01

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """Get actions (deterministic=mode, stochastic=sample).

        This method is called by SB3's ActorCriticPolicy.forward().
        """
        if deterministic:
            return self.mode()
        return self.sample()

    def actions_from_params(self, params: torch.Tensor,
                            low: torch.Tensor, high: torch.Tensor,
                            deterministic: bool = False):
        """Create distribution from params and return actions + log_prob."""
        self.proba_distribution(params, low, high)
        actions = self.get_actions(deterministic)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def log_prob_from_params(self, params: torch.Tensor,
                             low: torch.Tensor, high: torch.Tensor):
        """Create distribution from params and return sampled actions + log_prob."""
        self.proba_distribution(params, low, high)
        actions = self.sample()
        log_prob = self.log_prob(actions)
        return actions, log_prob
