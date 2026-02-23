"""Pluggable belief encoder interface for partial observability.

Provides a common interface for different belief encoders:
- BiMDN: Full bidirectional MDN (primary, from Paper [02])
- LSTM: Simpler LSTM-only encoder (fallback if BiMDN fails)
- MLP: Simple MLP on flattened history (baseline)

All encoders take observation history [batch, K, obs_dim] and output
a latent vector [batch, latent_dim] for the policy network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from agents.bimdn import BiMDN


class BeliefEncoder(ABC):
    """Abstract base class for belief encoders."""

    @abstractmethod
    def encode(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Encode observation history to latent belief.

        Args:
            obs_history: [batch, K, obs_dim]

        Returns:
            latent: [batch, latent_dim]
        """

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimension of the latent belief vector."""


class BiMDNEncoder(BeliefEncoder, nn.Module):
    """BiMDN-based belief encoder (primary encoder).

    Wraps BiMDN to conform to the BeliefEncoder interface.
    Also provides MDN parameters for supervised belief loss.
    """

    def __init__(self, obs_dim: int = 43, hidden_dim: int = 64,
                 n_mixtures: int = 5, latent_dim: int = 32):
        nn.Module.__init__(self)
        self.bimdn = BiMDN(obs_dim, hidden_dim, n_mixtures, latent_dim)
        self._latent_dim = latent_dim
        # Cache last MDN parameters for loss computation
        self._last_mdn_params = None

    def encode(self, obs_history: torch.Tensor) -> torch.Tensor:
        latent, mdn_params = self.bimdn(obs_history)
        self._last_mdn_params = mdn_params
        return latent

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """nn.Module forward — delegates to encode."""
        return self.encode(obs_history)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def last_mdn_params(self):
        """Access last MDN parameters for belief loss computation."""
        return self._last_mdn_params

    def belief_loss(self, target_pos: torch.Tensor) -> torch.Tensor:
        """Compute belief loss using cached MDN params from last encode()."""
        if self._last_mdn_params is None:
            raise RuntimeError("Must call encode() before belief_loss()")
        pi, mu, sigma = self._last_mdn_params
        return self.bimdn.belief_loss(pi, mu, sigma, target_pos)


class LSTMEncoder(BeliefEncoder, nn.Module):
    """LSTM-only belief encoder (fallback).

    Simpler than BiMDN — just LSTM + linear projection.
    No mixture density, just mean prediction.
    """

    def __init__(self, obs_dim: int = 43, hidden_dim: int = 64,
                 latent_dim: int = 32):
        nn.Module.__init__(self)
        self._latent_dim = latent_dim
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def encode(self, obs_history: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(obs_history)
        last_out = lstm_out[:, -1, :]
        return torch.tanh(self.proj(last_out))

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.encode(obs_history)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim


class MLPEncoder(BeliefEncoder, nn.Module):
    """MLP encoder on flattened observation history (baseline).

    Simplest encoder — flattens history and passes through MLP.
    No temporal modeling. Used as a minimal baseline.
    """

    def __init__(self, obs_dim: int = 43, history_length: int = 10,
                 hidden_dim: int = 128, latent_dim: int = 32):
        nn.Module.__init__(self)
        self._latent_dim = latent_dim
        input_dim = obs_dim * history_length
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

    def encode(self, obs_history: torch.Tensor) -> torch.Tensor:
        flat = obs_history.reshape(obs_history.shape[0], -1)
        return self.mlp(flat)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.encode(obs_history)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim


def create_encoder(encoder_type: str, **kwargs) -> BeliefEncoder:
    """Factory function for belief encoders.

    Args:
        encoder_type: One of 'bimdn', 'lstm', 'mlp'.
        **kwargs: Encoder-specific arguments.

    Returns:
        BeliefEncoder instance.
    """
    encoders = {
        "bimdn": BiMDNEncoder,
        "lstm": LSTMEncoder,
        "mlp": MLPEncoder,
    }
    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder type '{encoder_type}'. "
                        f"Available: {list(encoders.keys())}")
    return encoders[encoder_type](**kwargs)
