"""Tests for baseline agents and discrete action wrapper."""

import numpy as np
import pytest

from envs.discrete_action_wrapper import DiscreteActionWrapper
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper
from training.baselines import (
    GreedyEvaderPolicy,
    GreedyPursuerPolicy,
    RandomPolicy,
    create_discrete_action_space,
)


class TestRandomPolicy:
    def test_predict_shape(self):
        policy = RandomPolicy()
        obs = np.zeros(14, dtype=np.float32)
        action, _ = policy.predict(obs)
        assert action.shape == (2,)

    def test_predict_bounds(self):
        policy = RandomPolicy(v_max=1.0, omega_max=2.84)
        for _ in range(100):
            action, _ = policy.predict(np.zeros(14))
            assert 0 <= action[0] <= 1.0
            assert -2.84 <= action[1] <= 2.84


class TestGreedyPursuerPolicy:
    def test_predict_shape(self):
        policy = GreedyPursuerPolicy()
        obs = np.zeros(14, dtype=np.float32)
        action, _ = policy.predict(obs)
        assert action.shape == (2,)

    def test_steers_toward_evader(self):
        """Pursuer at origin facing east, evader to the north-east.
        Should turn left (positive omega) to face evader."""
        policy = GreedyPursuerPolicy()
        # obs: x_s=0, y_s=0, theta_s=0 (facing east)
        #      x_o=0.5 (east), y_o=0.5 (north)
        obs = np.zeros(14, dtype=np.float32)
        obs[0] = 0.0   # x_self normalized
        obs[1] = 0.0   # y_self normalized
        obs[2] = 0.0   # theta_self normalized (0 = east)
        obs[5] = 0.5   # x_opp normalized (5m east)
        obs[6] = 0.5   # y_opp normalized (5m north)
        action, _ = policy.predict(obs)

        # Should have positive omega (turn left toward NE)
        assert action[1] > 0, f"Expected positive omega, got {action[1]}"

    def test_moves_forward_when_aligned(self):
        """Pursuer facing directly toward evader should have v close to v_max."""
        policy = GreedyPursuerPolicy()
        obs = np.zeros(14, dtype=np.float32)
        obs[0] = 0.0
        obs[1] = 0.0
        obs[2] = 0.0   # facing east
        obs[5] = 0.5   # evader to the east
        obs[6] = 0.0
        action, _ = policy.predict(obs)
        assert action[0] > 0.5, f"Expected high v when aligned, got {action[0]}"


class TestGreedyEvaderPolicy:
    def test_predict_shape(self):
        policy = GreedyEvaderPolicy()
        obs = np.zeros(14, dtype=np.float32)
        action, _ = policy.predict(obs)
        assert action.shape == (2,)

    def test_always_max_speed(self):
        """Evader should always run at max speed."""
        policy = GreedyEvaderPolicy(v_max=1.0)
        obs = np.zeros(14, dtype=np.float32)
        action, _ = policy.predict(obs)
        assert action[0] == pytest.approx(1.0)

    def test_steers_away_from_pursuer(self):
        """Evader at origin facing east, pursuer to the east.
        Should steer away (turn around, large omega)."""
        policy = GreedyEvaderPolicy()
        obs = np.zeros(14, dtype=np.float32)
        obs[0] = 0.0
        obs[1] = 0.0
        obs[2] = 0.0   # facing east
        obs[5] = 0.5   # pursuer to the east
        obs[6] = 0.0
        action, _ = policy.predict(obs)
        # Should have large omega (turning away from pursuer)
        assert abs(action[1]) > 1.0, f"Expected large omega to turn away, got {action[1]}"


class TestDiscreteActionSpace:
    def test_create_table_shape(self):
        table = create_discrete_action_space(n_velocities=5, n_angular=7)
        assert table.shape == (35, 2)

    def test_create_table_bounds(self):
        table = create_discrete_action_space(v_max=1.0, omega_max=2.84)
        assert table[:, 0].min() == pytest.approx(0.0)
        assert table[:, 0].max() == pytest.approx(1.0)
        assert table[:, 1].min() == pytest.approx(-2.84)
        assert table[:, 1].max() == pytest.approx(2.84)


class TestDiscreteActionWrapper:
    def test_wrapper_discrete_space(self):
        base = PursuitEvasionEnv(render_mode=None, max_steps=50)
        wrapped = SingleAgentPEWrapper(base, role="pursuer")
        env = DiscreteActionWrapper(wrapped, n_velocities=5, n_angular=7)

        assert env.action_space.n == 35
        obs, _ = env.reset(seed=42)
        assert obs.shape == (14,)

        # Step with discrete action
        obs, reward, term, trunc, info = env.step(0)
        assert obs.shape == (14,)
        env.close()

    def test_dqn_trains(self):
        """DQN should train without error on discrete wrapper."""
        from stable_baselines3 import DQN

        base = PursuitEvasionEnv(render_mode=None, max_steps=50)
        wrapped = SingleAgentPEWrapper(base, role="pursuer")
        env = DiscreteActionWrapper(wrapped, n_velocities=3, n_angular=3)

        model = DQN(
            "MlpPolicy", env, verbose=0, seed=42,
            learning_rate=1e-4, buffer_size=1000, batch_size=32,
            learning_starts=100, policy_kwargs={"net_arch": [32]},
        )
        model.learn(total_timesteps=200)

        obs, _ = env.reset(seed=0)
        action, _ = model.predict(obs, deterministic=True)
        assert 0 <= action < env.action_space.n
        env.close()


class TestGreedyVsGreedy:
    """Integration test: greedy pursuer vs greedy evader."""

    def test_greedy_runs_full_episode(self):
        env = PursuitEvasionEnv(render_mode=None, max_steps=200)
        pursuer = GreedyPursuerPolicy()
        evader = GreedyEvaderPolicy()

        obs, info = env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            p_action, _ = pursuer.predict(obs["pursuer"])
            e_action, _ = evader.predict(obs["evader"])
            obs, rewards, terminated, truncated, info = env.step(p_action, e_action)
            done = terminated or truncated
            steps += 1

        assert steps > 0
        assert "episode_metrics" in info
        env.close()
