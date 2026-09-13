"""pe2.anchored_ppo — PPO with a KL anchor to a frozen clone (S2, D109)."""

import numpy as np
import pytest

# stable-baselines3 (and torch, which comes with it) is an optional
# dependency — CI runs `pip install -e .`, which does not install it. This
# module cannot guard it inside each test, because pe2.anchored_ppo
# SUBCLASSES PPO and so is unimportable without it. Guard at module scope, as
# tests/pe2/test_anchored_sac.py does: skip the module, never error collection.
pytest.importorskip("stable_baselines3")

import torch

from pe2.anchored_ppo import KLAnchoredPPO
from pe2.anchored_sac import gaussian_kl
from pe2.config import EnvConfig
from pe2.exploitability import SingleAgentWrapper
from pe2.scripted import GreedyEvader

CFG = EnvConfig()
TINY = {"n_steps": 32, "batch_size": 16, "n_epochs": 1, "verbose": 0, "device": "cpu"}


def _env(seed):
    return SingleAgentWrapper(CFG, "pursuer", GreedyEvader, seed0=seed * 1000)


def _model(beta=0.0, seed=1, cls=KLAnchoredPPO, anchor=True, **kw):
    kwargs = dict(TINY, seed=seed, **kw)
    if cls is KLAnchoredPPO:
        kwargs["beta"] = beta
    m = cls("MlpPolicy", _env(seed), **kwargs)
    if anchor and cls is KLAnchoredPPO:
        m.set_anchor(m.policy)
    return m


def _obs_batch(m, n=64, seed=20260913):
    space = m.env.envs[0].observation_space
    space.seed(seed)
    return torch.as_tensor(
        np.stack([space.sample() for _ in range(n)]), dtype=torch.float32
    )


class TestKLAnchoredPPO:
    def test_set_anchor_is_a_frozen_detached_copy(self):
        m = _model(beta=1.0)
        assert m.anchor is not m.policy
        for (ka, pa), (kp, pp) in zip(
            m.anchor.state_dict().items(), m.policy.state_dict().items(), strict=True
        ):
            assert ka == kp
            assert torch.equal(pa, pp)
            assert pa is not pp
        assert all(not p.requires_grad for p in m.anchor.parameters())

    def test_anchor_kl_is_zero_before_any_update(self):
        m = _model(beta=1.0)
        obs = _obs_batch(m)
        assert m.anchor_kl(obs, grad=False).item() == pytest.approx(0.0, abs=1e-7)

    def test_anchor_kl_matches_gaussian_kl_on_the_policies(self):
        m = _model(beta=1.0)
        m.learn(total_timesteps=64)  # moves the policy off the anchor
        obs = _obs_batch(m)
        with torch.no_grad():
            d = m.policy.get_distribution(obs).distribution
            a = m.anchor.get_distribution(obs).distribution
            expected = (
                gaussian_kl(d.mean, torch.log(d.stddev), a.mean, torch.log(a.stddev))
                .mean()
                .item()
            )
        assert m.anchor_kl(obs, grad=False).item() == pytest.approx(expected, rel=1e-6)

    def test_beta_zero_is_bit_identical_to_plain_ppo(self):
        from stable_baselines3 import PPO

        def trained(cls, beta):
            m = _model(beta=beta, seed=11, cls=cls)
            m.learn(total_timesteps=64)
            return {k: v.clone() for k, v in m.policy.state_dict().items()}

        plain = trained(PPO, beta=None)
        anchored = trained(KLAnchoredPPO, beta=0.0)
        assert plain.keys() == anchored.keys()
        for k in plain:
            assert torch.equal(plain[k], anchored[k]), (
                f"beta=0 diverged from PPO at {k}"
            )

    def test_beta_zero_still_measures_the_kl(self):
        m = _model(beta=0.0)
        assert m.last_anchor_kl is None
        m.learn(total_timesteps=64)
        assert m.last_anchor_kl is not None and m.last_anchor_kl >= 0.0
        assert "train/anchor_kl" in m.logger.name_to_value

    def test_large_beta_keeps_the_policy_near_the_anchor(self):
        def drift(beta):
            m = _model(beta=beta, seed=3)
            m.learn(total_timesteps=32 * 6)
            return m.anchor_kl(_obs_batch(m), grad=False).item()

        assert drift(beta=1e3) < drift(beta=0.0)

    def test_priced_kl_changes_the_update(self):
        # Same seed, same data: beta=0 and beta=1e3 must end at different weights.
        def weights(beta):
            m = _model(beta=beta, seed=5)
            m.learn(total_timesteps=64)
            return {k: v.clone() for k, v in m.policy.state_dict().items()}

        w0, w1 = weights(0.0), weights(1e3)
        assert any(not torch.equal(w0[k], w1[k]) for k in w0)

    def test_no_anchor_trains_as_plain_ppo_and_reports_none(self):
        m = _model(beta=1.0, anchor=False)
        m.learn(total_timesteps=64)
        assert m.last_anchor_kl is None

    def test_rejects_gsde(self):
        with pytest.raises(ValueError, match="use_sde"):
            _model(beta=1.0, use_sde=True)

    def test_rejects_negative_beta(self):
        with pytest.raises(ValueError, match="beta"):
            _model(beta=-0.1)

    def test_checkpoint_loads_as_plain_ppo_for_evaluation(self, tmp_path):
        from stable_baselines3 import PPO

        m = _model(beta=0.1)
        m.learn(total_timesteps=64)
        m.save(tmp_path / "anc.zip")
        loaded = PPO.load(tmp_path / "anc.zip", device="cpu")
        obs = _obs_batch(m, n=4)
        a0, _ = m.predict(obs.numpy(), deterministic=True)
        a1, _ = loaded.predict(obs.numpy(), deterministic=True)
        np.testing.assert_allclose(a0, a1)
