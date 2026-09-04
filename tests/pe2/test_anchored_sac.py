"""Tests for pe2.anchored_sac — pilot P-B (KL anchor to the frozen clone)."""

import copy

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.exploitability import SingleAgentWrapper, ln_sac_policy
from pe2.scripted import GreedyEvader

# stable-baselines3 (and torch, which comes with it) is an optional
# dependency — CI runs `pip install -e .`, which does not install it. Every
# other test module guards it with importorskip inside the test; this one
# cannot, because pe2.anchored_sac SUBCLASSES SAC and so is unimportable
# without it. Guard at module scope: skip the module, never error collection.
pytest.importorskip("stable_baselines3")

import torch

from pe2.anchored_sac import KLAnchoredSAC, gaussian_kl


class TestGaussianKl:
    def test_identical_is_zero(self):
        m = torch.zeros(4, 2)
        s = torch.zeros(4, 2)  # log_std 0 -> std 1
        assert gaussian_kl(m, s, m, s).abs().max().item() == pytest.approx(
            0.0, abs=1e-7
        )

    def test_mean_shift_hand_computed(self):
        # KL(N(mu,1) || N(0,1)) = mu^2 / 2 per dim; two dims with mu=1 -> 1.0.
        m = torch.ones(1, 2)
        z = torch.zeros(1, 2)
        assert gaussian_kl(m, z, z, z).item() == pytest.approx(1.0, abs=1e-6)

    def test_std_shift_hand_computed(self):
        # KL(N(0,s) || N(0,1)) = -log s + s^2/2 - 1/2; s = e^-1:
        # 1 + e^-2/2 - 0.5 = 0.5677 per dim, one dim.
        z = torch.zeros(1, 1)
        ls = torch.full((1, 1), -1.0)
        expected = 1.0 + np.exp(-2.0) / 2.0 - 0.5
        assert gaussian_kl(z, ls, z, z).item() == pytest.approx(expected, abs=1e-6)

    def test_is_summed_over_action_dims_and_batched(self):
        # Per-sample rows, summed over the last axis: row 0 has two unit mean
        # shifts (0.5 each), row 1 is identical to the reference (0.0).
        mean = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        z = torch.zeros(2, 2)
        out = gaussian_kl(mean, z, z, z)
        assert out.shape == (2,)
        assert out[0].item() == pytest.approx(1.0, abs=1e-6)
        assert out[1].item() == pytest.approx(0.0, abs=1e-7)

    def test_is_asymmetric(self):
        # KL is a divergence, not a metric: swapping the arguments changes it
        # whenever the variances differ.
        z = torch.zeros(1, 1)
        ls = torch.full((1, 1), -1.0)
        assert gaussian_kl(z, ls, z, z).item() != pytest.approx(
            gaussian_kl(z, z, z, ls).item(), abs=1e-3
        )


def _model(beta, seed=0, cls=KLAnchoredSAC, **kwargs):
    cfg = EnvConfig(n_obstacles=0)
    env = SingleAgentWrapper(cfg, "pursuer", GreedyEvader, seed0=seed * 1000)
    if cls is KLAnchoredSAC:
        kwargs["beta"] = beta
    m = cls(
        ln_sac_policy(),
        env,
        seed=seed,
        verbose=0,
        device="cpu",
        gamma=0.999,
        ent_coef=1e-5,
        learning_starts=32,
        train_freq=1,
        **kwargs,
    )
    if cls is KLAnchoredSAC:
        m.set_anchor(m.actor)
    return m


class TestKLAnchoredSAC:
    def test_set_anchor_is_a_frozen_detached_copy(self):
        m = _model(beta=1.0)
        assert m.anchor is not m.actor
        for (ka, pa), (kl_, pl) in zip(
            m.anchor.state_dict().items(), m.actor.state_dict().items(), strict=True
        ):
            assert ka == kl_
            assert torch.equal(pa, pl)  # starts as an exact copy of the clone
            assert pa is not pl  # but not a shared tensor
        assert all(not p.requires_grad for p in m.anchor.parameters())

    def test_beta_zero_matches_plain_actor_update_direction(self):
        # With beta=0 the anchor term contributes nothing: a train() call must
        # leave the anchor untouched and still move the actor (SAC works).
        m = _model(beta=0.0)
        m.learn(total_timesteps=48)
        anchor_before = copy.deepcopy(m.anchor.state_dict())
        actor_before = copy.deepcopy(m.actor.state_dict())
        m.train(gradient_steps=4, batch_size=16)
        for k in anchor_before:
            assert torch.equal(anchor_before[k], m.anchor.state_dict()[k])
        assert any(
            not torch.equal(actor_before[k], m.actor.state_dict()[k])
            for k in actor_before
        )

    def test_beta_zero_is_bit_identical_to_plain_sac(self):
        # The pilot comparison is only meaningful if the beta=0 arm IS SB3's
        # SAC. Each model is constructed immediately before it trains, so
        # SB3's set_random_seed() at construction gives both the same RNG.
        from stable_baselines3 import SAC

        def trained(cls, beta):
            m = _model(beta=beta, seed=11, cls=cls)
            m.learn(total_timesteps=48)
            return {k: v.clone() for k, v in m.policy.state_dict().items()}

        plain = trained(SAC, beta=None)
        anchored = trained(KLAnchoredSAC, beta=0.0)
        assert plain.keys() == anchored.keys()
        for k in plain:
            assert torch.equal(plain[k], anchored[k]), (
                f"beta=0 diverged from SAC at {k}"
            )

    def test_large_beta_keeps_the_actor_near_the_anchor(self):
        # Same replay data, same seed: after N updates the actor with a huge
        # anchor weight must stay strictly closer to the clone than beta=0.
        def drift(beta):
            m = _model(beta=beta, seed=3)
            m.learn(total_timesteps=64)
            m.train(gradient_steps=30, batch_size=32)
            obs = torch.as_tensor(
                np.stack([m.env.envs[0].observation_space.sample() for _ in range(64)]),
                dtype=torch.float32,
            )
            with torch.no_grad():
                mean, log_std, _ = m.actor.get_action_dist_params(obs)
                mean_a, log_std_a, _ = m.anchor.get_action_dist_params(obs)
                return gaussian_kl(mean, log_std, mean_a, log_std_a).mean().item()

        assert drift(beta=1e3) < drift(beta=0.0)

    def test_logs_anchor_kl(self):
        m = _model(beta=1.0)
        m.learn(total_timesteps=48)
        m.train(gradient_steps=2, batch_size=16)
        assert "train/anchor_kl" in m.logger.name_to_value
        assert m.logger.name_to_value["train/anchor_kl"] > 0.0

    def test_rejects_gsde(self):
        # gSDE reparameterizes the actor: get_action_dist_params then returns
        # the NOISE matrix's log_std plus a latent_sde kwarg, so gaussian_kl
        # would silently compute the wrong divergence. Refuse rather than lie.
        with pytest.raises(ValueError, match="use_sde"):
            _model(beta=1.0, use_sde=True)

    def test_no_anchor_set_still_trains(self):
        # Anchor is optional: an un-anchored model must behave like plain SAC
        # rather than crash, and report a zero KL.
        m = _model(beta=1.0)
        m.anchor = None
        m.learn(total_timesteps=48)
        m.train(gradient_steps=2, batch_size=16)
        assert m.logger.name_to_value["train/anchor_kl"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
