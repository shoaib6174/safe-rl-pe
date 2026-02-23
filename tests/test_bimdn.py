"""Tests for BiMDN belief encoder (Phase 3 Session 2).

Test Index:
  E: test_bimdn_valid_outputs
  F: test_bimdn_pretrain_loss_decreases
  G: test_bimdn_gradient_to_policy
  H: test_bimdn_multimodal_when_lost
  X: test_belief_encoder_interface (pluggable encoder)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from agents.bimdn import BiMDN
from agents.encoders import BiMDNEncoder, LSTMEncoder, MLPEncoder, create_encoder
from agents.partial_obs_policy import PartialObsPolicyNetwork


# --- Helpers ---

def _generate_synthetic_dataset(n_samples: int, obs_dim: int = 43, K: int = 10,
                                seed: int = 42):
    """Generate synthetic (obs_history, target_pos) pairs for BiMDN pre-training.

    Creates two types of scenarios:
    - "visible": target info is strongly encoded in observations (d_to_opp > 0,
      bearing set, lidar channels zeroed) — target position is tightly
      determined by the encoded distance/bearing.
    - "lost": target info is absent (d_to_opp = -1, bearing = 0, lidar channels
      zeroed) — target position is drawn from multiple spatial clusters,
      forcing the MDN to learn a multimodal distribution for this case.

    The obs layout is: [x, y, theta, v, omega, d_to_opp, bearing, lidar(36)] = 43 dims
    """
    rng = np.random.RandomState(seed)
    obs_histories = []
    target_positions = []

    # Define spatial clusters for "lost" target positions —
    # the MDN must model multiple possible positions.
    lost_clusters = np.array([
        [3.0, 3.0], [3.0, 17.0], [17.0, 3.0], [17.0, 17.0], [10.0, 10.0],
    ])

    for i in range(n_samples):
        # Zero-initialize to avoid noise on irrelevant channels
        # dominating the signal
        obs_hist = np.zeros((K, obs_dim), dtype=np.float32)
        # Own pose and velocity
        # Indices: 0=x, 1=y, 2=theta, 3=v, 4=omega, 5=d_to_opp, 6=bearing
        own_x = rng.uniform(2, 18)
        own_y = rng.uniform(2, 18)
        own_theta = rng.uniform(0, 2 * np.pi)
        for t in range(K):
            obs_hist[t, 0] = own_x + rng.normal(0, 0.05)   # x (slow drift)
            obs_hist[t, 1] = own_y + rng.normal(0, 0.05)   # y
            obs_hist[t, 2] = own_theta + rng.normal(0, 0.02)  # theta
            obs_hist[t, 3] = rng.normal(0.5, 0.05)  # v
            obs_hist[t, 4] = rng.normal(0.0, 0.02)   # omega

        if i % 2 == 0:
            # Visible: encode target distance and bearing in obs
            d = rng.uniform(1, 8)
            bearing = rng.uniform(-np.pi / 3, np.pi / 3)
            # Last few timesteps have target info
            for t in range(max(0, K - 5), K):
                obs_hist[t, 5] = d + rng.normal(0, 0.05)  # d_to_opp (low noise)
                obs_hist[t, 6] = bearing + rng.normal(0, 0.02)  # bearing (low noise)
            # Target position in world frame = own_pos + relative offset
            target_x = own_x + d * np.cos(own_theta + bearing) + rng.normal(0, 0.2)
            target_y = own_y + d * np.sin(own_theta + bearing) + rng.normal(0, 0.2)
        else:
            # Lost: target not visible (d_to_opp = -1)
            for t in range(K):
                obs_hist[t, 5] = -1.0
                obs_hist[t, 6] = 0.0
            # Target is at one of the spatial clusters (with some noise)
            cluster_idx = rng.randint(0, len(lost_clusters))
            center = lost_clusters[cluster_idx]
            target_x = center[0] + rng.normal(0, 1.0)
            target_y = center[1] + rng.normal(0, 1.0)

        obs_histories.append(obs_hist)
        target_positions.append([target_x, target_y])

    return (
        torch.tensor(np.array(obs_histories), dtype=torch.float32),
        torch.tensor(np.array(target_positions), dtype=torch.float32),
    )


def _pretrain_bimdn(bimdn, obs_histories, target_positions, epochs=10, lr=1e-3):
    """Pre-train BiMDN on synthetic data."""
    optimizer = Adam(bimdn.parameters(), lr=lr)
    dataset = TensorDataset(obs_histories, target_positions)
    loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_obs, batch_target in loader:
            _, (pi, mu, sigma) = bimdn(batch_obs)
            loss = bimdn.belief_loss(pi, mu, sigma, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / n_batches)
    return losses


# --- Test E: BiMDN outputs valid mixture parameters ---

def test_bimdn_valid_outputs():
    """Mixture weights sum to 1, sigmas are positive, shapes correct."""
    bimdn = BiMDN(obs_dim=43, n_mixtures=5, latent_dim=32)
    obs_hist = torch.randn(4, 10, 43)  # batch=4, K=10
    latent, (pi, mu, sigma) = bimdn(obs_hist)

    # Shape checks
    assert latent.shape == (4, 32), f"Latent shape: {latent.shape}"
    assert pi.shape == (4, 5), f"Pi shape: {pi.shape}"
    assert mu.shape == (4, 5, 2), f"Mu shape: {mu.shape}"
    assert sigma.shape == (4, 5, 2), f"Sigma shape: {sigma.shape}"

    # Mixing weights sum to 1
    pi_sum = pi.sum(dim=-1)
    assert torch.allclose(pi_sum, torch.ones(4), atol=1e-5), (
        f"Pi should sum to 1, got {pi_sum}"
    )

    # All pi values non-negative
    assert (pi >= 0).all(), "All mixing weights should be non-negative"

    # Sigmas are positive
    assert (sigma > 0).all(), f"All sigmas should be positive, min={sigma.min()}"

    # Latent is bounded (tanh output)
    assert latent.abs().max() <= 1.0 + 1e-6, "Latent should be in [-1, 1]"

    # No NaN
    assert torch.isfinite(latent).all(), "Latent has NaN/Inf"
    assert torch.isfinite(pi).all(), "Pi has NaN/Inf"
    assert torch.isfinite(mu).all(), "Mu has NaN/Inf"
    assert torch.isfinite(sigma).all(), "Sigma has NaN/Inf"


# --- Test F: BiMDN belief loss decreases during pre-training ---

def test_bimdn_pretrain_loss_decreases():
    """10 epochs of pre-training should reduce loss."""
    torch.manual_seed(42)
    bimdn = BiMDN(obs_dim=43, n_mixtures=5, latent_dim=32)
    obs_hist, target_pos = _generate_synthetic_dataset(200, seed=42)

    # Compute loss before training
    with torch.no_grad():
        _, (pi, mu, sigma) = bimdn(obs_hist[:64])
        loss_before = bimdn.belief_loss(pi, mu, sigma, target_pos[:64]).item()

    # Train for 10 epochs
    losses = _pretrain_bimdn(bimdn, obs_hist, target_pos, epochs=10, lr=1e-3)

    # Compute loss after training
    with torch.no_grad():
        _, (pi, mu, sigma) = bimdn(obs_hist[:64])
        loss_after = bimdn.belief_loss(pi, mu, sigma, target_pos[:64]).item()

    assert loss_after < loss_before, (
        f"Loss should decrease: before={loss_before:.4f}, after={loss_after:.4f}"
    )
    # Also check the training loss trend
    assert losses[-1] < losses[0], (
        f"Training loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


# --- Test G: BiMDN latent is informative (gradient flows to policy) ---

def test_bimdn_gradient_to_policy():
    """Backprop from policy loss through BiMDN should produce gradients."""
    torch.manual_seed(42)
    bimdn = BiMDN(obs_dim=43, n_mixtures=5, latent_dim=32)
    policy_head = nn.Linear(32, 2)  # Simple policy head

    obs_hist = torch.randn(4, 10, 43, requires_grad=True)
    latent, _ = bimdn(obs_hist)
    action = policy_head(latent)
    loss = action.sum()
    loss.backward()

    # Check that gradients flow through BiMDN to its LSTM weights
    assert bimdn.lstm.weight_ih_l0.grad is not None, (
        "LSTM weight_ih_l0 should have gradients"
    )
    assert bimdn.latent_head.weight.grad is not None, (
        "Latent head should have gradients"
    )
    # Gradients should be non-zero
    assert bimdn.lstm.weight_ih_l0.grad.abs().sum() > 0, (
        "LSTM gradients should be non-zero"
    )


# --- Test H: Pre-trained BiMDN produces higher uncertainty when target is lost ---

def test_bimdn_multimodal_when_lost():
    """After pre-training, BiMDN should show higher n_eff when target is lost vs visible.

    Uses 50 epochs and a larger dataset for robustness.
    Primary assertion: n_eff_lost > 1.3 (multiple components active).
    Secondary assertion: n_eff_lost > n_eff_visible (relative comparison).
    """
    torch.manual_seed(42)
    bimdn = BiMDN(obs_dim=43, n_mixtures=5, latent_dim=32)
    obs_hist, target_pos = _generate_synthetic_dataset(1000, seed=42)

    # Pre-train for 80 epochs — enough for the MDN to learn multimodal structure
    _pretrain_bimdn(bimdn, obs_hist, target_pos, epochs=80, lr=1e-3)

    # Create test scenarios
    rng = np.random.RandomState(123)
    n_samples = 20

    n_eff_visible_list = []
    n_eff_lost_list = []

    for _ in range(n_samples):
        # Visible scenario: target info present
        obs_vis = rng.randn(10, 43).astype(np.float32) * 0.1
        for t in range(5, 10):
            obs_vis[t, 5] = rng.uniform(2, 5)  # d_to_opp
            obs_vis[t, 6] = rng.uniform(-0.5, 0.5)  # bearing
        obs_vis_t = torch.tensor(obs_vis).unsqueeze(0)
        with torch.no_grad():
            _, (pi_vis, _, _) = bimdn(obs_vis_t)
        n_eff_vis = bimdn.effective_n_components(pi_vis).item()
        n_eff_visible_list.append(n_eff_vis)

        # Lost scenario: no target info
        obs_lost = rng.randn(10, 43).astype(np.float32) * 0.1
        for t in range(10):
            obs_lost[t, 5] = -1.0  # Not detected
            obs_lost[t, 6] = 0.0
        obs_lost_t = torch.tensor(obs_lost).unsqueeze(0)
        with torch.no_grad():
            _, (pi_lost, _, _) = bimdn(obs_lost_t)
        n_eff_lost = bimdn.effective_n_components(pi_lost).item()
        n_eff_lost_list.append(n_eff_lost)

    mean_n_eff_visible = np.mean(n_eff_visible_list)
    mean_n_eff_lost = np.mean(n_eff_lost_list)

    # Primary: lost scenarios should use multiple mixture components
    assert mean_n_eff_lost > 1.3, (
        f"n_eff_lost={mean_n_eff_lost:.2f}, expected > 1.3"
    )
    # Secondary: lost should be more uncertain than visible (on average)
    assert mean_n_eff_lost > mean_n_eff_visible, (
        f"n_eff_lost={mean_n_eff_lost:.2f} should > "
        f"n_eff_visible={mean_n_eff_visible:.2f}"
    )


# --- Test X: Pluggable belief encoder interface ---

def test_belief_encoder_interface():
    """All encoder types should implement the same interface and produce valid output."""
    obs_hist = torch.randn(4, 10, 43)

    for enc_type in ["bimdn", "lstm", "mlp"]:
        kwargs = {"obs_dim": 43, "latent_dim": 32}
        if enc_type == "mlp":
            kwargs["history_length"] = 10
        encoder = create_encoder(enc_type, **kwargs)

        # Encode should produce correct shape
        latent = encoder.encode(obs_hist)
        assert latent.shape == (4, 32), (
            f"{enc_type}: latent shape {latent.shape}, expected (4, 32)"
        )

        # Latent dim property
        assert encoder.latent_dim == 32, f"{enc_type}: latent_dim != 32"

        # Should have parameters
        params = list(encoder.parameters())
        assert len(params) > 0, f"{enc_type}: should have parameters"

        # Output should be finite
        assert torch.isfinite(latent).all(), f"{enc_type}: latent has NaN/Inf"


# --- Additional tests ---

class TestPartialObsPolicyNetwork:
    """Tests for the full policy feature extractor."""

    def test_forward_pass_shapes(self):
        """Policy network should produce correct output shape."""
        net = PartialObsPolicyNetwork(
            raw_obs_dim=43, lidar_dim=36, state_dim=7,
            encoder_type="bimdn", features_dim=256,
        )
        obs_dict = {
            "obs_history": torch.randn(4, 10, 43),
            "lidar": torch.randn(4, 1, 36),
            "state": torch.randn(4, 7),
        }
        features = net(obs_dict)
        assert features.shape == (4, 256), f"Features shape: {features.shape}"

    def test_gradient_flow_through_all_branches(self):
        """Gradients should flow through all three branches."""
        net = PartialObsPolicyNetwork(
            raw_obs_dim=43, lidar_dim=36, state_dim=7,
            encoder_type="bimdn", features_dim=256,
        )
        obs_dict = {
            "obs_history": torch.randn(4, 10, 43),
            "lidar": torch.randn(4, 1, 36),
            "state": torch.randn(4, 7),
        }
        features = net(obs_dict)
        loss = features.sum()
        loss.backward()

        # Check gradients in each branch
        assert net.encoder.bimdn.lstm.weight_ih_l0.grad is not None, (
            "BiMDN branch should have gradients"
        )
        conv_params = list(net.lidar_conv.parameters())
        assert conv_params[0].grad is not None, (
            "Lidar branch should have gradients"
        )
        state_params = list(net.state_mlp.parameters())
        assert state_params[0].grad is not None, (
            "State branch should have gradients"
        )

    def test_with_lstm_encoder(self):
        """Policy network should work with LSTM encoder fallback."""
        net = PartialObsPolicyNetwork(
            raw_obs_dim=43, lidar_dim=36, state_dim=7,
            encoder_type="lstm", features_dim=256,
        )
        obs_dict = {
            "obs_history": torch.randn(2, 10, 43),
            "lidar": torch.randn(2, 1, 36),
            "state": torch.randn(2, 7),
        }
        features = net(obs_dict)
        assert features.shape == (2, 256)
        assert torch.isfinite(features).all()

    def test_with_mlp_encoder(self):
        """Policy network should work with MLP encoder baseline."""
        net = PartialObsPolicyNetwork(
            raw_obs_dim=43, lidar_dim=36, state_dim=7,
            encoder_type="mlp",
            encoder_kwargs={"history_length": 10},
            features_dim=256,
        )
        obs_dict = {
            "obs_history": torch.randn(2, 10, 43),
            "lidar": torch.randn(2, 1, 36),
            "state": torch.randn(2, 7),
        }
        features = net(obs_dict)
        assert features.shape == (2, 256)
        assert torch.isfinite(features).all()


class TestBiMDNBeliefLoss:
    """Additional tests for BiMDN loss function."""

    def test_loss_is_scalar(self):
        """Loss should be a scalar."""
        bimdn = BiMDN(obs_dim=43, n_mixtures=5)
        obs = torch.randn(8, 10, 43)
        target = torch.randn(8, 2)
        _, (pi, mu, sigma) = bimdn(obs)
        loss = bimdn.belief_loss(pi, mu, sigma, target)
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"

    def test_loss_decreases_with_correct_prediction(self):
        """Loss should be lower when mu is close to target."""
        bimdn = BiMDN(obs_dim=43, n_mixtures=1)  # Single component
        obs = torch.randn(4, 10, 43)
        _, (pi, mu, sigma) = bimdn(obs)

        # Loss with target far from mu
        target_far = mu[:, 0, :].detach() + 100.0
        loss_far = bimdn.belief_loss(pi, mu, sigma, target_far).item()

        # Loss with target at mu
        target_close = mu[:, 0, :].detach()
        loss_close = bimdn.belief_loss(pi, mu, sigma, target_close).item()

        assert loss_close < loss_far, (
            f"Close target should have lower loss: close={loss_close}, far={loss_far}"
        )

    def test_effective_n_components(self):
        """Effective components calculation should be correct."""
        bimdn = BiMDN(obs_dim=43, n_mixtures=5)

        # Uniform weights: n_eff should be 5
        pi_uniform = torch.ones(1, 5) / 5.0
        n_eff = bimdn.effective_n_components(pi_uniform).item()
        assert abs(n_eff - 5.0) < 0.01, f"Uniform: n_eff={n_eff}, expected 5.0"

        # One-hot weights: n_eff should be 1
        pi_onehot = torch.zeros(1, 5)
        pi_onehot[0, 0] = 1.0
        n_eff = bimdn.effective_n_components(pi_onehot).item()
        assert abs(n_eff - 1.0) < 0.01, f"One-hot: n_eff={n_eff}, expected 1.0"
