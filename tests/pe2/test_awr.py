"""Tests for pe2.awr — advantage-weighted regression on MC returns (T0.4 R-D)."""

import warnings

import numpy as np
import pytest

from pe2.config import EnvConfig


class TestAdvantageWeights:
    def test_formula_clip_and_sign(self):
        from pe2.awr import advantage_weights

        r = np.array([1.0, -1.0, 0.0])
        v = np.array([0.0, 0.0, 0.0])
        w = advantage_weights(r, v, lam=0.5, w_max=20.0, sign=1.0)
        assert w[0] == pytest.approx(np.exp(2.0))  # 7.39, under the clip
        assert w[1] == pytest.approx(np.exp(-2.0))
        assert w[2] == pytest.approx(1.0)
        flipped = advantage_weights(r, v, lam=0.5, w_max=20.0, sign=-1.0)
        assert flipped[0] == pytest.approx(w[1]) and flipped[1] == pytest.approx(w[0])
        big = advantage_weights(np.array([10.0]), np.array([0.0]), 0.5, 20.0, 1.0)
        assert big[0] == 20.0

    def test_defaults_are_the_preregistered_values(self):
        from pe2.awr import LAM, W_MAX, AwrConfig

        c = AwrConfig()
        assert (LAM, W_MAX) == (0.5, 20.0)
        assert (c.lam, c.w_max, c.advantage_sign, c.gamma) == (0.5, 20.0, 1.0, 0.999)
        assert (c.initial_episodes, c.buffer_episodes, c.steps_per_iter) == (
            300,
            300,
            50_000,
        )


class TestValueFit:
    def test_regression_descends_on_a_linear_target(self):
        torch = pytest.importorskip("torch")
        from pe2.awr import fit_value, make_value_net

        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        obs = torch.as_tensor(rng.normal(size=(2000, 4)), dtype=torch.float32)
        target = obs[:, 0] * 0.5 - obs[:, 1] * 0.25
        net = make_value_net(4, 32, "cpu")
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        losses = fit_value(net, opt, obs, target, epochs=30, batch_size=256, rng=rng)
        assert losses[-1] < 0.1 * losses[0]


class TestActorUpdate:
    def _model(self, cfg):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import SAC

        from pe2.exploitability import SingleAgentWrapper
        from pe2.scripted import GreedyEvader

        return SAC(
            "MlpPolicy",
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader),
            seed=0,
            verbose=0,
            device="cpu",
        )

    def test_uniform_weights_move_tanh_mu_toward_the_actions(self):
        torch = pytest.importorskip("torch")
        from pe2.awr import awr_actor_update
        from pe2.obs import obs_dim

        cfg = EnvConfig()
        model = self._model(cfg)
        rng = np.random.default_rng(1)
        obs = torch.as_tensor(
            rng.uniform(-1, 1, size=(512, obs_dim(cfg))), dtype=torch.float32
        )
        acts = torch.full((512, 2), 0.7)
        with torch.no_grad():
            before = torch.tanh(model.actor.get_action_dist_params(obs)[0])
        opt = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
        awr_actor_update(
            model, opt, obs, acts, torch.ones(512), epochs=20, batch_size=128, rng=rng
        )
        with torch.no_grad():
            after = torch.tanh(model.actor.get_action_dist_params(obs)[0])
        assert ((after - acts) ** 2).mean() < ((before - acts) ** 2).mean()

    def test_weights_steer_which_actions_are_matched(self):
        torch = pytest.importorskip("torch")
        from pe2.awr import awr_actor_update
        from pe2.obs import obs_dim

        cfg = EnvConfig()
        model = self._model(cfg)
        rng = np.random.default_rng(2)
        obs = torch.as_tensor(
            rng.uniform(-1, 1, size=(1, obs_dim(cfg))), dtype=torch.float32
        ).repeat(400, 1)
        # Same state, two candidate actions: +0.8 (heavily weighted) vs -0.8.
        acts = torch.cat([torch.full((200, 2), 0.8), torch.full((200, 2), -0.8)])
        w = torch.cat([torch.full((200,), 20.0), torch.full((200,), 0.05)])
        opt = torch.optim.Adam(model.actor.parameters(), lr=3e-3)
        awr_actor_update(model, opt, obs, acts, w, epochs=40, batch_size=400, rng=rng)
        with torch.no_grad():
            out = torch.tanh(model.actor.get_action_dist_params(obs[:1])[0])
        assert (out > 0.4).all()  # pulled to the heavily weighted action

    def test_loss_accumulation_does_not_warn_on_a_grad_tensor(self):
        # M5: `total += float(loss)` on a requires_grad tensor emits a
        # UserWarning every run. `fit_value`'s sibling was fixed in bc6af22;
        # the actor path was missed. Numerically identical either way.
        torch = pytest.importorskip("torch")
        from pe2.awr import awr_actor_update
        from pe2.obs import obs_dim

        cfg = EnvConfig()
        model = self._model(cfg)
        rng = np.random.default_rng(3)
        obs = torch.as_tensor(
            rng.uniform(-1, 1, size=(64, obs_dim(cfg))), dtype=torch.float32
        )
        acts = torch.full((64, 2), 0.5)
        opt = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            awr_actor_update(
                model, opt, obs, acts, torch.ones(64), epochs=2, batch_size=32, rng=rng
            )
        assert not [w for w in caught if issubclass(w.category, UserWarning)]


class TestTrainAwr:
    def test_tiny_run_probes_on_the_grid_and_returns_a_usable_model(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import SAC

        from pe2.awr import AwrConfig, train_awr
        from pe2.exploitability import SingleAgentWrapper
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=40)
        clone = tmp_path / "clone.zip"
        SAC(
            "MlpPolicy",
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader),
            seed=0,
            verbose=0,
            device="cpu",
        ).save(clone)
        awr = AwrConfig(
            initial_episodes=5,
            buffer_episodes=8,
            steps_per_iter=100,
            value_epochs=2,
            actor_epochs=2,
            batch_size=64,
        )
        marks = []
        model, stats = train_awr(
            cfg,
            "pursuer",
            GreedyEvader,
            clone,
            total_timesteps=800,
            seed=3,
            awr=awr,
            probe_every=200,
            probe=lambda m, s: marks.append(s),
            device="cpu",
            rollout_seed0=990000,
        )
        assert marks == [200, 400, 600, 800]
        assert stats["total_steps"] >= 800 and stats["n_episodes"] >= 5
        assert stats["iterations"][0]["phase"] == "offline"
        assert all(it["phase"] == "online" for it in stats["iterations"][1:])
        assert [m["grid"] for m in stats["probe_marks"]] == marks
        assert stats["config"]["advantage_sign"] == 1.0
        action, _ = model.predict(
            np.zeros(
                SingleAgentWrapper(
                    cfg, "pursuer", GreedyEvader
                ).observation_space.shape,
                dtype=np.float32,
            ),
            deterministic=True,
        )
        assert action.shape == (2,)

    def test_same_seed_same_stats(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import SAC

        from pe2.awr import AwrConfig, train_awr
        from pe2.exploitability import SingleAgentWrapper
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=40)
        clone = tmp_path / "clone.zip"
        SAC(
            "MlpPolicy",
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader),
            seed=0,
            verbose=0,
            device="cpu",
        ).save(clone)
        awr = AwrConfig(
            initial_episodes=4,
            buffer_episodes=6,
            steps_per_iter=80,
            value_epochs=1,
            actor_epochs=1,
            batch_size=64,
        )
        runs = [
            train_awr(
                cfg,
                "pursuer",
                GreedyEvader,
                clone,
                300,
                seed=5,
                awr=awr,
                probe_every=150,
                probe=lambda m, s: None,
                device="cpu",
                rollout_seed0=990100,
            )[1]
            for _ in range(2)
        ]
        assert runs[0]["iterations"] == runs[1]["iterations"]
        assert runs[0]["torch_seed"] == 5
        assert runs[0]["first_action"] == runs[1]["first_action"]

    def test_different_seeds_produce_different_first_action(self, tmp_path):
        """Regression test: `seed_torch` must run AFTER `SAC.load`, or `load`
        silently re-applies the seed baked into the clone's zip and the
        declared `seed` never governs a single sampled action.

        This must compare something that only torch's global generator
        drives, NOT `stats["iterations"]` — those losses are also shaped by
        `rng = np.random.default_rng(seed)`, which is seeded straight from
        `seed` independently of the torch bug and drives minibatch
        permutation order, so `iterations` differs across seeds even when
        the torch seed is silently discarded. `first_action` is the first
        value that actually depends on torch's global generator, so it is
        identical under the bug and different once fixed."""
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import SAC

        from pe2.awr import AwrConfig, train_awr
        from pe2.exploitability import SingleAgentWrapper
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=40)
        clone = tmp_path / "clone.zip"
        SAC(
            "MlpPolicy",
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader),
            seed=0,
            verbose=0,
            device="cpu",
        ).save(clone)
        awr = AwrConfig(
            initial_episodes=4,
            buffer_episodes=6,
            steps_per_iter=80,
            value_epochs=1,
            actor_epochs=1,
            batch_size=64,
        )
        stats = [
            train_awr(
                cfg,
                "pursuer",
                GreedyEvader,
                clone,
                300,
                seed=s,
                awr=awr,
                probe_every=150,
                probe=lambda m, s: None,
                device="cpu",
                rollout_seed0=990100,
            )[1]
            for s in (5, 6)
        ]
        assert stats[0]["torch_seed"] != stats[1]["torch_seed"]
        assert stats[0]["first_action"] != stats[1]["first_action"]


class TestAdaptiveTemperature:
    """T0.5 spec §3.2: lambda_t = max(std(A_t), lam_floor) per iteration.

    The point is scale-freeness — the weight distribution must be the same
    at every iteration regardless of how far the value net has driven the
    advantage spread down. T0.4 used a fixed 0.5 against a spread that
    decayed 0.186 -> 0.095, so the +/-1 sigma weight ratio fell from 2.10
    to 1.46 and the weighting was inert.
    """

    def test_fixed_is_the_default_and_returns_the_configured_value(self):
        from pe2.awr import AwrConfig, resolve_lam

        cfg = AwrConfig()
        assert cfg.adaptive_lam is False and cfg.lam_floor == 0.01
        adv = np.random.default_rng(0).normal(0.0, 0.2, size=5000)
        assert resolve_lam(adv, cfg) == 0.5

    def test_adaptive_returns_the_measured_spread(self):
        from pe2.awr import AwrConfig, resolve_lam

        cfg = AwrConfig(adaptive_lam=True)
        adv = np.random.default_rng(1).normal(0.0, 0.2, size=200_000)
        assert resolve_lam(adv, cfg) == pytest.approx(np.std(adv))
        assert resolve_lam(adv, cfg) == pytest.approx(0.2, abs=0.01)

    def test_floor_engages_on_a_degenerate_spread(self):
        from pe2.awr import AwrConfig, resolve_lam

        cfg = AwrConfig(adaptive_lam=True)
        assert resolve_lam(np.full(100, 0.3), cfg) == 0.01  # zero spread
        assert resolve_lam(np.random.default_rng(2).normal(0, 1e-6, 100), cfg) == 0.01

    def test_the_weight_ratio_is_constant_under_adaptive_lambda(self):
        """The property the whole change exists for."""
        from pe2.awr import AwrConfig, resolve_lam, weight_ratio

        cfg = AwrConfig(adaptive_lam=True)
        rng = np.random.default_rng(3)
        # The spread T0.4 actually traversed, first iteration to last.
        for std in (0.1857, 0.1489, 0.1203, 0.0953):
            adv = rng.normal(0.0, std, size=200_000)
            lam = resolve_lam(adv, cfg)
            assert weight_ratio(1.0, float(np.std(adv)), lam) == pytest.approx(
                np.exp(2.0), rel=1e-9
            )
            assert weight_ratio(2.0, float(np.std(adv)), lam) == pytest.approx(
                np.exp(4.0), rel=1e-9
            )

    def test_the_fixed_temperature_reproduces_the_t04_numbers(self):
        """Guards the claim in the spec's own comparison table."""
        from pe2.awr import weight_ratio

        assert weight_ratio(1.0, 0.1857, 0.5) == pytest.approx(2.10, abs=0.01)
        assert weight_ratio(1.0, 0.0953, 0.5) == pytest.approx(1.46, abs=0.01)

    def test_predicted_clip_fraction_under_adaptive_lambda(self):
        """Spec §3.2 predicts ~0.0014 of weights reach the w_max=20 ceiling."""
        from pe2.awr import AwrConfig, advantage_weights, resolve_lam

        cfg = AwrConfig(adaptive_lam=True)
        adv = np.random.default_rng(4).normal(0.0, 0.15, size=400_000)
        lam = resolve_lam(adv, cfg)
        w = advantage_weights(adv, np.zeros_like(adv), lam, cfg.w_max, 1.0)
        assert float((w >= cfg.w_max).mean()) == pytest.approx(0.00137, abs=0.0006)


class TestAdaptiveTemperatureInTraining:
    def _clone(self, tmp_path, cfg):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import SAC

        from pe2.exploitability import SingleAgentWrapper
        from pe2.scripted import GreedyEvader

        path = tmp_path / "clone.zip"
        SAC(
            "MlpPolicy",
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader),
            seed=0,
            verbose=0,
            device="cpu",
        ).save(path)
        return path

    def test_iterations_record_the_temperature_and_the_ratios(self, tmp_path):
        from pe2.awr import AwrConfig, train_awr
        from pe2.config import EnvConfig
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=40)
        clone = self._clone(tmp_path, cfg)
        awr = AwrConfig(
            adaptive_lam=True,
            initial_episodes=6,
            buffer_episodes=8,
            steps_per_iter=100,
            value_epochs=2,
            actor_epochs=2,
            batch_size=64,
        )
        _, stats = train_awr(
            cfg,
            "pursuer",
            GreedyEvader,
            clone,
            400,
            seed=3,
            awr=awr,
            probe_every=200,
            probe=lambda m, s: None,
            device="cpu",
            rollout_seed0=990200,
        )
        assert stats["config"]["adaptive_lam"] is True
        for it in stats["iterations"]:
            assert it["lam_used"] > 0.0
            assert set(it) >= {"lam_used", "weight_ratio_1sigma", "weight_ratio_2sigma"}
            if it["lam_used"] > awr.lam_floor:  # floor not engaged
                assert it["weight_ratio_1sigma"] == pytest.approx(np.exp(2.0), rel=1e-6)

    def test_the_default_config_is_unchanged_by_this_feature(self, tmp_path):
        """T0.4's runs are a frozen record; the fixed path must not move."""
        from pe2.awr import AwrConfig, train_awr
        from pe2.config import EnvConfig
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=40)
        clone = self._clone(tmp_path, cfg)
        kw = {
            "initial_episodes": 5,
            "buffer_episodes": 6,
            "steps_per_iter": 80,
            "value_epochs": 1,
            "actor_epochs": 1,
            "batch_size": 64,
        }
        runs = [
            train_awr(
                cfg,
                "pursuer",
                GreedyEvader,
                clone,
                300,
                seed=5,
                awr=AwrConfig(**kw),
                probe_every=150,
                probe=lambda m, s: None,
                device="cpu",
                rollout_seed0=990300,
            )[1]
            for _ in range(2)
        ]
        assert runs[0]["first_action"] == runs[1]["first_action"]
        for it in runs[0]["iterations"]:
            assert it["lam_used"] == 0.5  # the fixed default, not the spread
