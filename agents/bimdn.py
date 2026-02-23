"""Bidirectional Mixture Density Network (BiMDN) for belief estimation.

Maintains a probabilistic belief about the opponent's position using
observation history. Based on Paper [02] (Gonultas & Isler 2024).

Architecture:
  Observation history [batch, K, obs_dim]
      |
      +-- Forward LSTM: processes history chronologically
      |
      +-- Backward LSTM: processes history in reverse
      |
      +-- Concatenate hidden states -> MLP -> MDN parameters
              |
              +-- Mixture of M Gaussians:
                   p(x_opp | o_{t-K:t}) = SUM_m w_m * N(mu_m, sigma_m)
              |
              +-- Latent encoding [batch, latent_dim] for policy input
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiMDN(nn.Module):
    """Bidirectional Mixture Density Network for opponent belief estimation.

    Takes observation history and outputs:
    1. A latent vector for the policy network (learned belief representation)
    2. MDN parameters (pi, mu, sigma) for supervised belief training

    Args:
        obs_dim: Dimension of each observation in the history.
        hidden_dim: LSTM hidden dimension.
        n_mixtures: Number of Gaussian mixture components.
        latent_dim: Dimension of the latent belief vector for policy.
    """

    def __init__(
        self,
        obs_dim: int = 43,
        hidden_dim: int = 64,
        n_mixtures: int = 5,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_mixtures = n_mixtures
        self.latent_dim = latent_dim

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # MDN output heads (from bidirectional output: hidden_dim * 2)
        combined_dim = hidden_dim * 2
        self.pi_head = nn.Linear(combined_dim, n_mixtures)        # mixing weights
        self.mu_head = nn.Linear(combined_dim, n_mixtures * 2)    # means (x, y)
        self.sigma_head = nn.Linear(combined_dim, n_mixtures * 2) # stds (x, y)

        # Latent encoding for policy input
        self.latent_head = nn.Linear(combined_dim, latent_dim)

    def forward(
        self, obs_history: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            obs_history: [batch, K, obs_dim] observation history.

        Returns:
            latent: [batch, latent_dim] belief latent vector for policy.
            (pi, mu, sigma): MDN parameters.
                pi: [batch, M] mixing weights (sum to 1).
                mu: [batch, M, 2] mean positions (x, y).
                sigma: [batch, M, 2] std deviations (positive).
        """
        # LSTM processes the full sequence
        lstm_out, (h_n, c_n) = self.lstm(obs_history)

        # Use last timestep output (contains both forward and backward info)
        last_out = lstm_out[:, -1, :]  # [batch, hidden_dim * 2]

        # MDN parameters
        pi = F.softmax(self.pi_head(last_out), dim=-1)  # [batch, M]
        mu = self.mu_head(last_out).reshape(-1, self.n_mixtures, 2)  # [batch, M, 2]
        sigma = F.softplus(self.sigma_head(last_out)).reshape(
            -1, self.n_mixtures, 2,
        ) + 1e-6  # [batch, M, 2], ensure positive

        # Latent for policy
        latent = torch.tanh(self.latent_head(last_out))  # [batch, latent_dim]

        return latent, (pi, mu, sigma)

    def belief_loss(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood of true position under the Gaussian mixture.

        Args:
            pi: [batch, M] mixing weights.
            mu: [batch, M, 2] means.
            sigma: [batch, M, 2] stds.
            target_pos: [batch, 2] true opponent (x, y) position.

        Returns:
            Scalar loss (mean NLL across batch).
        """
        # Expand target to match mixture dimensions
        target = target_pos.unsqueeze(1).expand_as(mu)  # [batch, M, 2]

        # Log probability under each Gaussian component (independent x, y)
        # log N(x; mu, sigma) = -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5*log(2*pi)
        log_probs = (
            -0.5 * ((target - mu) / sigma).pow(2)
            - sigma.log()
            - 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
        )
        log_probs = log_probs.sum(dim=-1)  # Sum over x, y dims: [batch, M]

        # Log mixture probability: log(sum_m pi_m * N_m)
        log_mixture = torch.logsumexp(log_probs + pi.log(), dim=-1)  # [batch]

        return -log_mixture.mean()

    def get_belief_state(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> dict:
        """Convert MDN parameters to a belief state dict for visualization.

        Args:
            pi, mu, sigma: MDN parameters (single sample, no batch dim).

        Returns:
            Dict with 'means', 'stds', 'weights' as numpy arrays.
        """
        return {
            "means": mu.detach().cpu().numpy(),
            "stds": sigma.detach().cpu().numpy(),
            "weights": pi.detach().cpu().numpy(),
        }

    def effective_n_components(self, pi: torch.Tensor) -> torch.Tensor:
        """Compute effective number of mixture components (1/sum(pi^2)).

        Useful for checking multimodality:
        - n_eff ~ 1: unimodal (one component dominates)
        - n_eff ~ M: uniform (all components equally weighted)

        Args:
            pi: [batch, M] mixing weights.

        Returns:
            [batch] effective number of components.
        """
        return 1.0 / (pi.pow(2).sum(dim=-1))
