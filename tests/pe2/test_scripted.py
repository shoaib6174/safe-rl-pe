"""Tests for pe2.scripted — Step 3 scripted agents + duel runner."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.env import OUTCOME_CAPTURE, PursuitEvasionEnv
from pe2.scripted import (
    FovLimitedEvader,
    FovLimitedPursuer,
    GreedyEvader,
    GreedyPursuer,
    run_duel,
)


class TestGreedyPursuer:
    def test_catches_stationary_evader_open_arena(self):
        """Full-obs greedy pursuer must capture a stationary evader in an
        obstacle-free arena well within the step budget."""
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = []
        pursuer = GreedyPursuer()
        obs = env._get_obs()
        for _ in range(cfg.max_steps):
            v_p, w_p = pursuer.act(obs["pursuer"], env)
            obs, _, terminated, truncated, info = env.step([v_p, w_p, 0.0, 0.0])
            if terminated or truncated:
                break
        assert info["outcome"] == OUTCOME_CAPTURE
        # distance 3+ m at ~1 m/s closing -> well under 600 steps
        assert env.current_step < 600


class TestGreedyEvader:
    def test_steers_toward_shadow_point_when_threatened(self):
        """With the pursuer close and an obstacle nearby, the evader's action
        must steer toward the shadow point (away side of the obstacle), not
        merely directly away from the pursuer."""
        cfg = EnvConfig(n_obstacles=1)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = [{"x": 0.0, "y": 0.0, "radius": 0.8}]
        # pursuer south of obstacle, evader east of it, both close
        env.pursuer_pose = np.array([0.0, -3.0, np.pi / 2])
        env.evader_pose = np.array([2.5, 0.5, 0.0])  # heading +x
        v, w = GreedyEvader().act(None, env)
        # shadow point: past the obstacle on the line away from the pursuer
        # = (0, 0) + (0.8 + 0.6) * (0,1) = (0, 1.4); from (2.5, 0.5) that is
        # up-left -> positive turn from heading 0
        assert w > 0.0
        assert v == pytest.approx(cfg.evader_v_max)

    def test_flees_directly_when_no_obstacle_near(self):
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = []
        env.pursuer_pose = np.array([0.0, -3.0, 0.0])
        env.evader_pose = np.array([0.0, 0.0, np.pi / 2])  # already facing away
        v, w = GreedyEvader().act(None, env)
        assert v == pytest.approx(cfg.evader_v_max)
        assert abs(w) < 1e-6  # dead-ahead flee, no turn


class TestFovLimitedPursuer:
    def _obs_visible(self, cfg, dx_b=2.0, dy_b=0.5):
        obs = np.zeros(7 + 2 * cfg.n_obstacles, dtype=np.float32)
        obs[3] = dx_b / cfg.arena_size
        obs[4] = dy_b / cfg.arena_size
        obs[6] = 1.0
        return obs

    def test_scans_when_never_seen(self):
        env = PursuitEvasionEnv(EnvConfig())
        policy = FovLimitedPursuer()
        obs = np.zeros(7 + 2 * env.cfg.n_obstacles, dtype=np.float32)
        v, w = policy.act(obs, env)
        assert v == 0.0
        assert w > 0.0  # scanning turn

    def test_chases_body_frame_bearing_when_visible(self):
        env = PursuitEvasionEnv(EnvConfig())
        policy = FovLimitedPursuer()
        obs = self._obs_visible(env.cfg, dx_b=2.0, dy_b=0.5)
        v, w = policy.act(obs, env)
        assert v == env.cfg.v_max
        assert w > 0.0  # opponent is to the left in body frame

    def test_goes_to_last_seen_not_true_position(self):
        """After losing sight, the policy must head for the LAST-SEEN world
        position — never the opponent's true (now different) position."""
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        policy = FovLimitedPursuer()
        # own pose at origin heading +x; opponent seen at body (2, 0) -> world (2, 0)
        obs = self._obs_visible(cfg, dx_b=2.0, dy_b=0.0)
        policy.act(obs, env)
        # now invisible; truth is somewhere else entirely
        obs[3] = obs[4] = 0.0
        obs[6] = 0.0
        env.evader_pose = np.array([-4.0, -4.0, 0.0])  # true pos behind-left
        v, w = policy.act(obs, env)
        assert v == cfg.v_max
        assert abs(w) < 1e-6  # still heading straight for last-seen (2, 0)
        assert policy.last_seen == pytest.approx([2.0, 0.0])

    def test_arrival_triggers_scan(self):
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        policy = FovLimitedPursuer()
        policy.last_seen = np.array([0.1, 0.0])  # within LAST_SEEN_TOL of origin
        obs = np.zeros(7 + 2 * cfg.n_obstacles, dtype=np.float32)
        v, w = policy.act(obs, env)
        assert v == 0.0 and w > 0.0  # scanning
        assert policy.last_seen is None


class TestFovLimitedEvader:
    def test_cruises_when_never_seen(self):
        env = PursuitEvasionEnv(EnvConfig())
        policy = FovLimitedEvader()
        obs = np.zeros(7 + 2 * env.cfg.n_obstacles, dtype=np.float32)
        v, w = policy.act(obs, env)
        assert v == env.cfg.evader_v_max
        assert w == pytest.approx(0.3 * env.cfg.omega_max)

    def test_flees_away_from_visible_pursuer(self):
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        policy = FovLimitedEvader()
        # own at origin heading +x; pursuer dead ahead at body (2, 0)
        obs = np.zeros(7 + 2 * cfg.n_obstacles, dtype=np.float32)
        obs[3] = 2.0 / cfg.arena_size
        obs[6] = 1.0
        _, w = policy.act(obs, env)
        # flee direction is -x world = behind -> |turn| large
        assert abs(w) > 0.5 * cfg.omega_max


class TestRunDuel:
    def test_deterministic_per_seed(self):
        env = PursuitEvasionEnv(EnvConfig())
        log1 = run_duel(env, GreedyPursuer(), GreedyEvader(), seed=42)
        log2 = run_duel(env, GreedyPursuer(), GreedyEvader(), seed=42)
        assert log1 == log2

    def test_log_fields_and_outcome(self):
        env = PursuitEvasionEnv(EnvConfig())
        log = run_duel(env, GreedyPursuer(), GreedyEvader(), seed=42)
        assert log["outcome"] in ("capture", "timeout")
        assert log["steps"] <= env.cfg.max_steps
        assert log["los_breaks"] >= 0
        assert log["final_distance"] >= 0.0
        assert log["pursuer"] == "GreedyPursuer"
        assert log["evader"] == "GreedyEvader"
        if log["outcome"] == "capture":
            assert log["final_distance"] <= env.cfg.capture_distance + 1e-6

    def test_occlusion_events_occur(self):
        """Across a handful of seeded duels with obstacles, at least one
        LOS break must occur (occlusion is actually in play)."""
        env = PursuitEvasionEnv(EnvConfig())
        total = sum(
            run_duel(env, GreedyPursuer(), GreedyEvader(), seed=100 + i)["los_breaks"]
            for i in range(5)
        )
        assert total > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
