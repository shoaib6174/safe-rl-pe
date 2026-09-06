"""Tests for pe2.awr — advantage-weighted regression on MC returns (T0.4 R-D)."""

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
