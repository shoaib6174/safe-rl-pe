"""Tests for pe2.obs — Step 2 observation vector."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.obs import build_observation, obs_dim


def pose(x, y, theta):
    return np.array([x, y, theta], dtype=float)


class TestLayout:
    def test_dim(self):
        assert obs_dim(EnvConfig()) == 7 + 2 * EnvConfig().n_obstacles

    def test_own_pose_always_present(self):
        cfg = EnvConfig()
        own = pose(2.5, -1.0, 1.0)
        opp = pose(-100, -100, 0.0)  # far away, definitely not visible
        obs = build_observation(own, opp, [], cfg)
        half = cfg.arena_size / 2.0
        assert obs[0] == pytest.approx(2.5 / half)
        assert obs[1] == pytest.approx(-1.0 / half)
        assert obs[2] == pytest.approx(1.0 / np.pi)

    def test_invisible_opponent_zeroed(self):
        cfg = EnvConfig()
        own = pose(0, 0, 0.0)
        opp = pose(-3.0, 0.0, 0.7)  # behind -> not visible
        obs = build_observation(own, opp, [], cfg)
        assert obs[6] == 0.0
        assert obs[3] == 0.0 and obs[4] == 0.0 and obs[5] == 0.0

    def test_visible_opponent_body_frame(self):
        """Hand-computed body-frame transform for a visible opponent."""
        cfg = EnvConfig()
        own = pose(1.0, 2.0, np.pi / 2)  # facing +y
        opp = pose(1.0, 5.0, 0.3)  # 3m dead ahead
        obs = build_observation(own, opp, [], cfg)
        assert obs[6] == 1.0
        # dead ahead in body frame: dx_b = +3, dy_b = 0
        assert obs[3] == pytest.approx(3.0 / cfg.arena_size, abs=1e-6)
        assert obs[4] == pytest.approx(0.0, abs=1e-6)
        # relative heading: 0.3 - pi/2 wrapped, / pi
        expected_dtheta = (0.3 - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
        assert obs[5] == pytest.approx(expected_dtheta / np.pi, abs=1e-6)

    def test_body_frame_rotation(self):
        cfg = EnvConfig()
        own = pose(0, 0, np.pi / 2)  # facing +y
        opp = pose(
            -1.0, 3.0, 0.0
        )  # within the 90-deg cone (bearing ~18 deg off-heading)
        obs = build_observation(own, opp, [], cfg)
        assert obs[6] == 1.0
        # world delta (-1, +3); body frame (heading +y): dx_b = +3, dy_b = +1
        assert obs[3] == pytest.approx(3.0 / cfg.arena_size, abs=1e-6)
        assert obs[4] == pytest.approx(1.0 / cfg.arena_size, abs=1e-6)

    def test_occluded_opponent_zeroed(self):
        cfg = EnvConfig()
        obstacles = [{"x": 2.0, "y": 0.0, "radius": 0.8}]
        own = pose(0, 0, 0.0)
        opp = pose(4.0, 0.0, 0.0)  # in cone, in range, but occluded
        obs = build_observation(own, opp, obstacles, cfg)
        assert obs[6] == 0.0
        assert np.all(obs[3:6] == 0.0)

    def test_obstacle_features_nearest_k_sorted(self):
        cfg = EnvConfig()
        obstacles = [
            {"x": 6.0, "y": 0.0, "radius": 0.5},  # far
            {"x": 1.0, "y": 0.0, "radius": 0.5},  # nearest
        ]
        own = pose(0, 0, 0.0)
        obs = build_observation(own, pose(0, 0, 0), obstacles, cfg)
        # first obstacle slot = nearest (x=1), body-frame dx = +1
        assert obs[7] == pytest.approx(1.0 / cfg.arena_size, abs=1e-6)
        assert obs[8] == pytest.approx(0.0, abs=1e-6)
        assert obs[9] == pytest.approx(6.0 / cfg.arena_size, abs=1e-6)

    def test_obstacle_features_unmasked_by_occlusion(self):
        """Static map is known: obstacle behind another obstacle still appears."""
        cfg = EnvConfig()
        obstacles = [
            {"x": 1.0, "y": 0.0, "radius": 0.5},
            {"x": 3.0, "y": 0.0, "radius": 0.5},  # occluded by the first
        ]
        own = pose(0, 0, 0.0)
        obs = build_observation(own, pose(0, 0, 0), obstacles, cfg)
        assert obs[9] == pytest.approx(3.0 / cfg.arena_size, abs=1e-6)


class TestEnvIntegration:
    def test_env_obs_in_space(self):
        env = PursuitEvasionEnv(EnvConfig())
        obs, _ = env.reset(seed=0)
        assert env.observation_space["pursuer"].contains(obs["pursuer"])
        assert env.observation_space["evader"].contains(obs["evader"])
        for _ in range(200):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert env.observation_space["pursuer"].contains(obs["pursuer"])
            assert env.observation_space["evader"].contains(obs["evader"])
            if terminated or truncated:
                break

    def test_los_flag_matches_geometry(self):
        """Env obs los_visible flag agrees with direct sensing computation."""
        from pe2.sensing import is_visible

        env = PursuitEvasionEnv(EnvConfig())
        env.reset(seed=5)
        obs, _, _, _, _ = env.step(np.zeros(4))
        expected = is_visible(
            env.pursuer_pose,
            (env.evader_pose[0], env.evader_pose[1]),
            env.obstacles,
            env.cfg.sensing,
        )
        assert (obs["pursuer"][6] == 1.0) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
