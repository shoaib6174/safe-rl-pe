"""Tests for pe2.anchored_sac — pilot P-B (KL anchor to the frozen clone)."""

import ast
import copy
import inspect
import textwrap

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
        # SB3's seeding does NOT reach the observation space, so the probe
        # batch is seeded explicitly — otherwise the two arms are measured on
        # different draws and the strict inequality is only probabilistic.
        def drift(beta):
            m = _model(beta=beta, seed=3)
            m.learn(total_timesteps=64)
            m.train(gradient_steps=30, batch_size=32)
            space = m.env.envs[0].observation_space
            space.seed(20260904)
            obs = torch.as_tensor(
                np.stack([space.sample() for _ in range(64)]),
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

    def test_rejects_negative_beta(self):
        # A negative weight would silently disable the term (the loss is only
        # amended when beta > 0), so an arm labelled "anchored" would train as
        # plain SAC. A mis-signed ablation must fail loudly, not quietly.
        with pytest.raises(ValueError, match="beta"):
            _model(beta=-1.0)

    def test_beta_zero_logs_the_real_drift_not_a_fabricated_zero(self):
        # The control arm is the arm whose drift the diagnosis is ABOUT. If it
        # logged 0.0 the series would read as "never left the clone", the exact
        # opposite of the finding. beta=0 must still MEASURE the divergence.
        m = _model(beta=0.0)
        m.learn(total_timesteps=64)
        m.train(gradient_steps=30, batch_size=32)
        logged = m.logger.name_to_value["train/anchor_kl"]
        assert logged > 0.0, "beta=0 reported a fabricated zero drift"

        # and the logged number must be the actual actor-vs-anchor divergence,
        # not some other quantity: recompute it on a fresh probe batch and
        # require the same order of magnitude.
        space = m.env.envs[0].observation_space
        space.seed(20260904)
        obs = torch.as_tensor(
            np.stack([space.sample() for _ in range(64)]), dtype=torch.float32
        )
        with torch.no_grad():
            mean, log_std, _ = m.actor.get_action_dist_params(obs)
            mean_a, log_std_a, _ = m.anchor.get_action_dist_params(obs)
            measured = gaussian_kl(mean, log_std, mean_a, log_std_a).mean().item()
        assert 0.1 < logged / measured < 10.0, (logged, measured)

    def test_no_anchor_set_still_trains(self):
        # Anchor is optional: an un-anchored model must behave like plain SAC
        # rather than crash, and report a zero KL.
        m = _model(beta=1.0)
        m.anchor = None
        m.learn(total_timesteps=48)
        m.train(gradient_steps=2, batch_size=16)
        assert m.logger.name_to_value["train/anchor_kl"] == 0.0


def _train_method(path, cls_name):
    """The `train` FunctionDef of `cls_name` in `path`, plus its source lines."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines(keepends=True)
    tree = ast.parse("".join(lines))
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == cls_name:
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "train":
                    return fn, lines
    raise AssertionError(f"no {cls_name}.train in {path}")


class TestLibraryCopyIsFaithful:
    """The override is a copied library method; nothing else pins the version.

    pyproject declares `stable-baselines3` with NO upper bound and CI installs
    through it, so an upgrade could change SAC.train underneath this copy while
    every other pilot arm picks up the new one. The arms would then differ by
    more than the anchor term and NOTHING would fail. This is that tripwire.
    """

    def test_train_is_the_installed_sac_train_plus_only_marked_lines(self):
        import stable_baselines3
        from stable_baselines3.sac.sac import SAC as InstalledSAC

        import pe2.anchored_sac

        lib_fn, _ = _train_method(inspect.getsourcefile(InstalledSAC), "SAC")
        mine_fn, mine_lines = _train_method(
            inspect.getsourcefile(pe2.anchored_sac), "KLAnchoredSAC"
        )

        # The module docstring promises a future reader that diffing this
        # method against sac.py shows ONLY `# ANCHOR` lines. Test exactly that
        # promise: delete every marked line and what remains must BE the
        # library's method — same body, same signature, same name.
        seg = mine_lines[mine_fn.lineno - 1 : mine_fn.end_lineno]
        stripped = [ln for ln in seg if "# ANCHOR" not in ln]
        assert len(stripped) < len(seg), "no ANCHOR-marked lines found at all"
        rebuilt = ast.parse(textwrap.dedent("".join(stripped))).body[0]

        assert ast.dump(rebuilt) == ast.dump(lib_fn), (
            "KLAnchoredSAC.train no longer matches the installed "
            f"stable_baselines3 {stable_baselines3.__version__} SAC.train once "
            "the # ANCHOR lines are removed. Either an addition is unmarked, "
            "or SB3 was upgraded and the copy must be re-taken from the new "
            "source (see the pe2.anchored_sac module docstring)."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAnchorKlIsExported:
    """The anchor divergence is the one number that separates 'the anchor
    stopped harmful drift' from 'the anchor froze the policy'. Under a
    'does not collapse' pass rule, a frozen actor passes by not moving, so
    this number has to leave `train()` and reach the results file."""

    def test_last_anchor_kl_matches_the_logged_value(self):
        m = _model(beta=1.0)
        m.learn(total_timesteps=48)
        m.train(gradient_steps=2, batch_size=16)
        assert m.last_anchor_kl == pytest.approx(
            m.logger.name_to_value["train/anchor_kl"]
        )
        assert m.last_anchor_kl > 0.0

    def test_no_anchor_means_no_number_rather_than_zero(self):
        # A model with no anchor measured nothing; reporting 0.0 would read
        # as "the actor never left the clone", the opposite of no data.
        m = _model(beta=1.0)
        m.anchor = None
        assert m.last_anchor_kl is None
        m.learn(total_timesteps=48)
        m.train(gradient_steps=2, batch_size=16)
        assert m.last_anchor_kl is None


class TestAnchorKLProbe:
    def test_stamps_the_row_written_at_the_same_step(self):
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList

        from pe2.anchored_sac import AnchorKLProbe

        rows: list[dict] = []
        every = 32

        class _FakeProbe(BaseCallback):
            """Stands in for CheckpointEvalProbe: appends a row on the same
            schedule, and runs BEFORE the anchor probe in the list."""

            def _on_step(inner) -> bool:
                if inner.num_timesteps % every == 0:
                    rows.append({"steps": inner.num_timesteps})
                return True

        m = _model(beta=1.0, seed=5)  # learning_starts=32, train_freq=1
        m.learn(
            total_timesteps=64,
            callback=CallbackList([_FakeProbe(), AnchorKLProbe(rows, every).callback]),
        )
        assert [r["steps"] for r in rows] == [32, 64]
        # At 32 steps no gradient step has run yet (learning_starts=32), so
        # there is no measurement — None, not a fabricated 0.0. By 64 the
        # actor has been updated and the divergence is a real number.
        assert rows[0]["anchor_kl"] is None
        assert isinstance(rows[1]["anchor_kl"], float)
        assert rows[1]["anchor_kl"] >= 0.0

    def test_refuses_when_no_probe_row_was_written(self):
        # Wiring guard: placed BEFORE the probe (or given a different
        # `every`) it would stamp a stale row or none at all, and the arm
        # would silently be judged with no divergence readout.
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList

        from pe2.anchored_sac import AnchorKLProbe

        rows: list[dict] = []

        class _LateProbe(BaseCallback):
            def _on_step(inner) -> bool:
                if inner.num_timesteps % 32 == 0:
                    rows.append({"steps": inner.num_timesteps})
                return True

        m = _model(beta=1.0, seed=6)
        with pytest.raises(RuntimeError, match="probe row"):
            m.learn(
                total_timesteps=64,
                callback=CallbackList([AnchorKLProbe(rows, 32).callback, _LateProbe()]),
            )


class TestAnchorKlGrew:
    def test_a_moved_actor_is_reported_with_words(self):
        from pe2.anchored_sac import ANCHOR_KL_MOVED, anchor_kl_grew

        rows = [
            {"steps": 50_000, "anchor_kl": 0.0},  # phase A: actor frozen
            {"steps": 100_000, "anchor_kl": 0.4},
            {"steps": 200_000, "anchor_kl": 1.7},
        ]
        # 1.7 > ANCHOR_KL_MOVED (1e-3) by 3.2 decades, and 1.7 is 16x the
        # 0.105 the UNANCHORED control reached: this actor has plainly left
        # the clone.
        assert 1.7 > ANCHOR_KL_MOVED
        grew, why = anchor_kl_grew(rows)
        assert grew is True
        assert "1.7" in why

    def test_a_pinned_actor_is_reported_as_not_moved(self):
        # The failure this exists for: the anchor holds the actor ON the
        # clone, the arm never collapses, and "pass" would mean "preserved
        # the clone at the price of an anchored fine-tune".
        from pe2.anchored_sac import anchor_kl_grew

        rows = [
            {"steps": 100_000, "anchor_kl": 1e-6},
            {"steps": 200_000, "anchor_kl": 1e-6},
        ]
        grew, why = anchor_kl_grew(rows)
        assert grew is False
        assert "preserved the clone" in why

    def test_growth_off_the_frozen_phase_zero_is_not_enough(self):
        # THE REGRESSION THIS CLASS EXISTS FOR. Under the pre-registered
        # schedule (critic_warmup=50k, probe every 50k) the first probe
        # lands at the END of the frozen phase, where the actor still IS
        # the anchor and the divergence is zero BY CONSTRUCTION. A
        # first-vs-last comparison therefore reduces to "moved at all",
        # which these rows satisfy: 0.0 -> 4.6e-5 is strictly increasing.
        # 4.6e-5 is the value the P-B smoke run actually reported, and it
        # is 22x BELOW the 1e-3 moved threshold and 12x ABOVE the 4e-6 a
        # pinned arm holds -- a pinned policy by any reading.
        from pe2.anchored_sac import ANCHOR_KL_MOVED, anchor_kl_grew

        rows = [
            {"steps": 50_000, "anchor_kl": 0.0},
            {"steps": 100_000, "anchor_kl": 2.0e-5},
            {"steps": 200_000, "anchor_kl": 4.6e-5},
        ]
        assert rows[-1]["anchor_kl"] < ANCHOR_KL_MOVED  # 4.6e-5 < 1e-3
        grew, why = anchor_kl_grew(rows)
        assert grew is False
        assert "preserved the clone" in why

    def test_one_measurement_above_the_threshold_is_enough_to_judge(self):
        # The criterion is a MAGNITUDE, not a comparison, so a single probe
        # answers it: there is no second point to need.
        from pe2.anchored_sac import anchor_kl_grew

        grew, why = anchor_kl_grew([{"steps": 200_000, "anchor_kl": 0.5}])
        assert grew is True  # 0.5 > 1e-3
        assert "0.5" in why

    def test_no_measurement_is_undetermined(self):
        # Two distinct holes, both undetermined rather than a bare null:
        # no probe rows at all, and rows that carry no measurement (no
        # anchor set, or AnchorKLProbe not wired).
        from pe2.anchored_sac import anchor_kl_grew

        for rows in ([], [{"steps": 1}], [{"steps": 1, "anchor_kl": None}]):
            grew, why = anchor_kl_grew(rows)
            assert grew is None
            assert isinstance(why, str) and len(why) > 20

    def test_the_threshold_sits_two_decades_from_both_measured_regimes(self):
        # The threshold's whole defence is that it is derived from numbers
        # measured in this project, not invented: under an identical
        # protocol (30 gradient steps, seed 3) the UNANCHORED arm reached
        # anchor KL 0.104509 and the anchored arm at beta=1e3 held
        # 0.000004. 1e-3 is 250x the pinned value (4e-6 * 100 = 4e-4 < 1e-3)
        # and 105x below the moved one (0.105 / 100 = 1.05e-3 > 1e-3), so
        # it is wrong only if a real arm lands inside a two-decade gap in
        # which nothing has ever been observed.
        from pe2.anchored_sac import ANCHOR_KL_MOVED

        pinned, moved = 4e-6, 0.105
        assert pinned * 100 < ANCHOR_KL_MOVED < moved / 100
