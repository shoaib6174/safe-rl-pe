import numpy as np
import pytest
import torch
from agents.bimdn import BiMDN


def test_output_shapes():
    bimdn = BiMDN(obs_dim=11, hidden_dim=128, feature_dim=32, num_components=3)
    obs_history = torch.randn(5, 20, 11)
    pi, mu, sigma = bimdn(obs_history)
    assert pi.shape == (5, 3)
    assert mu.shape == (5, 3, 2)
    assert sigma.shape == (5, 3, 2)


def test_mixture_weights_sum_to_one():
    bimdn = BiMDN(obs_dim=11, hidden_dim=128, feature_dim=32, num_components=3)
    obs_history = torch.randn(10, 20, 11)
    pi, _, _ = bimdn(obs_history)
    weights_sum = pi.sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)


def test_std_positive():
    bimdn = BiMDN(obs_dim=11, hidden_dim=128, feature_dim=32, num_components=3)
    obs_history = torch.randn(10, 20, 11)
    _, _, sigma = bimdn(obs_history)
    assert torch.all(sigma > 0)


def test_nll_loss_finite():
    bimdn = BiMDN(obs_dim=11, hidden_dim=128, feature_dim=32, num_components=3)
    optimizer = torch.optim.Adam(bimdn.parameters(), lr=0.002)
    obs_history = torch.randn(32, 20, 11)
    target_pos = torch.randn(32, 2)

    loss1 = bimdn.compute_nll_loss(obs_history, target_pos)
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()
    loss2 = bimdn.compute_nll_loss(obs_history, target_pos)

    assert torch.isfinite(loss1)
    assert torch.isfinite(loss2)
    assert loss2 < loss1


def test_weighted_mean_shape():
    bimdn = BiMDN(obs_dim=11, hidden_dim=128, feature_dim=32, num_components=3)
    obs_history = torch.randn(10, 20, 11)
    mean = bimdn.get_weighted_mean(obs_history)
    assert mean.shape == (10, 2)
