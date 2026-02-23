"""Partial observation policy network for Phase 3.

SB3-compatible feature extractor that processes Dict observations
from PartialObsWrapper through three branches:
1. Observation history -> BiMDN -> belief latent
2. Lidar readings -> Conv1D -> spatial features
3. State features -> MLP -> state encoding

Combined into a 256-dim feature vector for SB3's actor/critic heads.
DCBF filter is applied POST-HOC at action execution, not inside the network.

Architecture (Phase 2.5 decision: standard PPO + post-hoc DCBF):
  obs_dict
    ├── obs_history [K, 43] ─> BiMDN ─> [32]    (belief latent)
    ├── lidar [1, 36] ──────> Conv1D ─> [256]    (spatial features)
    └── state [7] ──────────> MLP ────> [64]     (pose + velocity + detection)
                                        ──────
                                   concat [352] ─> MLP ─> [256] (output features)
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agents.encoders import BeliefEncoder, BiMDNEncoder, create_encoder


class PartialObsPolicyNetwork(nn.Module):
    """Multi-branch feature extractor for partial observations.

    This is the core neural network. PartialObsFeaturesExtractor below
    wraps it as an SB3-compatible BaseFeaturesExtractor.

    Args:
        raw_obs_dim: Raw observation dimension (default 43).
        lidar_dim: Number of lidar rays (default 36).
        encoder_type: Belief encoder type ('bimdn', 'lstm', 'mlp').
        encoder_kwargs: Additional kwargs for encoder creation.
        features_dim: Output feature dimension (default 256).
    """

    def __init__(
        self,
        raw_obs_dim: int = 43,
        lidar_dim: int = 36,
        state_dim: int = 7,
        encoder_type: str = "bimdn",
        encoder_kwargs: dict | None = None,
        features_dim: int = 256,
    ):
        super().__init__()

        # Create belief encoder
        kwargs = encoder_kwargs or {}
        kwargs.setdefault("obs_dim", raw_obs_dim)
        self.encoder = create_encoder(encoder_type, **kwargs)
        encoder_latent_dim = self.encoder.latent_dim

        # Lidar branch: Conv1D for spatial correlation between adjacent rays
        # Kernel size 5 captures ~50 deg sectors (5/36 * 360 deg)
        self.lidar_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
        )  # Output: 64 * 4 = 256 dims

        # State branch: [x, y, theta, v, omega, d_to_opp, bearing]
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Combined: 256 (lidar) + 64 (state) + encoder_latent_dim (belief)
        combined_dim = 256 + 64 + encoder_latent_dim
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

        self._features_dim = features_dim

    def forward(self, obs_dict: dict) -> torch.Tensor:
        """Process Dict observation to feature vector.

        Args:
            obs_dict: Dict with keys:
                'obs_history': [batch, K, raw_obs_dim]
                'lidar': [batch, 1, n_rays]
                'state': [batch, 7]

        Returns:
            features: [batch, features_dim]
        """
        # Belief encoding from observation history
        latent = self.encoder.encode(obs_dict["obs_history"])

        # Lidar spatial features
        lidar_features = self.lidar_conv(obs_dict["lidar"])

        # State features
        state_features = self.state_mlp(obs_dict["state"])

        # Combine all branches
        combined = torch.cat([lidar_features, state_features, latent], dim=-1)
        return self.combined_mlp(combined)

    @property
    def features_dim(self) -> int:
        return self._features_dim


class PartialObsFeaturesExtractor(BaseFeaturesExtractor):
    """SB3-compatible feature extractor wrapping PartialObsPolicyNetwork.

    Replaces SB3's default CombinedExtractor with our BiMDN-based architecture.
    Use with SB3's MultiInputPolicy:

        policy_kwargs = {
            'features_extractor_class': PartialObsFeaturesExtractor,
            'features_extractor_kwargs': {
                'features_dim': 256,
                'encoder_type': 'bimdn',
            },
        }
        model = PPO('MultiInputPolicy', wrapped_env, policy_kwargs=policy_kwargs)

    Args:
        observation_space: gymnasium.spaces.Dict observation space.
        features_dim: Output feature dimension.
        encoder_type: Belief encoder type ('bimdn', 'lstm', 'mlp').
        encoder_kwargs: Additional kwargs for encoder creation.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        encoder_type: str = "bimdn",
        encoder_kwargs: dict | None = None,
    ):
        super().__init__(observation_space, features_dim)

        # Infer dimensions from observation space
        obs_hist_shape = observation_space["obs_history"].shape
        lidar_shape = observation_space["lidar"].shape
        state_shape = observation_space["state"].shape
        raw_obs_dim = obs_hist_shape[-1]
        lidar_dim = lidar_shape[-1]
        state_dim = state_shape[0]

        self.net = PartialObsPolicyNetwork(
            raw_obs_dim=raw_obs_dim,
            lidar_dim=lidar_dim,
            state_dim=state_dim,
            encoder_type=encoder_type,
            encoder_kwargs=encoder_kwargs,
            features_dim=features_dim,
        )

    def forward(self, observations: dict) -> torch.Tensor:
        """Extract features from Dict observations.

        Args:
            observations: Dict with 'obs_history', 'lidar', 'state' tensors.

        Returns:
            features: [batch, features_dim]
        """
        return self.net(observations)

    @property
    def belief_encoder(self) -> BeliefEncoder:
        """Access the belief encoder for external use (e.g., pre-training)."""
        return self.net.encoder
