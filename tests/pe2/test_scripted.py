"""Tests for pe2.scripted — Step 3 scripted agents + duel runner."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.env import OUTCOME_CAPTURE, PursuitEvasionEnv
from pe2.scripted import (
    BoxRoundTripEvader,
    FovLimitedEvader,
    FovLimitedPursuer,
    GreedyEvader,
    GreedyPursuer,
    ObstacleHuggingEvader,
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


class TestBoxRoundTripEvader:
    def test_equals_greedy_evader_after_the_scale_action_round_trip(self):
        """It must be GreedyEvader put through exactly the float32 Box that
        SingleAgentWrapper._scale_action applies to an evader-role action."""
        cfg = EnvConfig(n_obstacles=1)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = [{"x": 0.0, "y": 0.0, "radius": 0.8}]
        env.pursuer_pose = np.array([0.0, -3.0, np.pi / 2])
        env.evader_pose = np.array([2.5, 0.5, 0.0])
        v, w = GreedyEvader().act(None, env)
        a = np.array(
            [2.0 * v / cfg.evader_v_max - 1.0, w / cfg.omega_max], dtype=np.float32
        )
        a64 = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
        expected = ((a64[0] + 1.0) / 2.0 * cfg.evader_v_max, a64[1] * cfg.omega_max)
        assert BoxRoundTripEvader().act(None, env) == expected


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


class TestObstacleHuggingEvader:
    """Step 6 G3/G4 adversarial probe (see pe2.scripted.ObstacleHuggingEvader).

    Geometry constants used throughout, all from the locked defaults:
    agent_radius = 0.15, arena_size = 10 -> half = 5, x_lim = y_lim = 4.85,
    omega_max = 2.84, evader_v_max = 1.2. Probe constants: HUG_CLEARANCE
    0.05, RADIAL_GAIN 2.0, RADIAL_CLIP 1.0, WALL_HUG_BAND 0.35,
    HUG_RANGE 4.0, HUG_THREAT_RANGE 7.0.
    """

    def test_sees_truth(self):
        assert ObstacleHuggingEvader.uses_truth is True

    def test_orbits_to_the_antipode_of_the_pursuer(self):
        """Sitting exactly on the hug circle, the probe's desired direction is
        the pure tangent toward the far side of the obstacle.

        obstacle (0,0) r=0.8 -> hug_r = 0.8 + 0.15 + 0.05 = 1.00.
        evader (1.0, 0) -> r = 1.00 = hug_r, so the radial term is 0.
        pursuer (0,-3) -> bearing from the obstacle = -pi/2, antipode = +pi/2;
        evader bearing = 0, so delta = +pi/2 > 0 -> slide CCW, tangent =
        (-r_hat_y, r_hat_x) = (0, 1). Heading pi/2 is already that direction,
        so the turn command is exactly 0.
        """
        cfg = EnvConfig(n_obstacles=1)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = [{"x": 0.0, "y": 0.0, "radius": 0.8}]
        env.pursuer_pose = np.array([0.0, -3.0, 0.0])
        env.evader_pose = np.array([1.0, 0.0, np.pi / 2])
        v, w = ObstacleHuggingEvader().act(None, env)
        assert v == pytest.approx(cfg.evader_v_max)
        assert w == pytest.approx(0.0, abs=1e-9)

    def test_slides_the_other_way_when_the_pursuer_swaps_sides(self):
        """Same geometry with the pursuer at (0, +3): antipode = -pi/2,
        delta = -pi/2 -> tangent = (0, -1). From heading 0 the error is
        -pi/2, and K_OMEGA * (-pi/2) = -4.712 saturates at -omega_max.
        """
        cfg = EnvConfig(n_obstacles=1)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = [{"x": 0.0, "y": 0.0, "radius": 0.8}]
        env.pursuer_pose = np.array([0.0, 3.0, 0.0])
        env.evader_pose = np.array([1.0, 0.0, 0.0])
        v, w = ObstacleHuggingEvader().act(None, env)
        assert v == pytest.approx(cfg.evader_v_max)
        assert w == pytest.approx(-cfg.omega_max)

    def test_spirals_inward_when_off_the_hug_circle(self):
        """Off the hug circle the radial correction pulls the probe onto the
        obstacle, keeping a tangential lead so it arrives already orbiting.

        obstacle (0,0) r=0.8 -> hug_r = 1.00; evader (3, 0) -> r = 3.0.
        hug_r - r = -2.0, clipped to -1.0, times RADIAL_GAIN 2.0 -> radial
        term = -2.0 * r_hat = (-2, 0). Pursuer (0,-3) gives tangent (0, 1)
        as above, so the desired direction is (-2, 1) — inward and CCW.
        Pointing the evader exactly along atan2(1, -2) must give zero turn.
        """
        cfg = EnvConfig(n_obstacles=1)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = [{"x": 0.0, "y": 0.0, "radius": 0.8}]
        env.pursuer_pose = np.array([0.0, -3.0, 0.0])
        env.evader_pose = np.array([3.0, 0.0, float(np.arctan2(1.0, -2.0))])
        v, w = ObstacleHuggingEvader().act(None, env)
        assert v == pytest.approx(cfg.evader_v_max)
        assert w == pytest.approx(0.0, abs=1e-9)

    def test_boundary_hug_slides_along_the_wall_away_from_the_pursuer(self):
        """With no obstacle in range the probe presses against the nearest
        wall and slides along it away from the pursuer.

        evader (4.50, 0): east-wall distance 4.85 - 4.50 = 0.35 = the hug
        band, so the radial term is 0. Outward normal (1, 0) -> tangent
        (0, 1); e - p = (0, 3) has positive dot with it, so the slide is +y.
        Heading pi/2 is that direction -> zero turn.
        """
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = []
        env.pursuer_pose = np.array([4.50, -3.0, 0.0])
        env.evader_pose = np.array([4.50, 0.0, np.pi / 2])
        v, w = ObstacleHuggingEvader().act(None, env)
        assert v == pytest.approx(cfg.evader_v_max)
        assert w == pytest.approx(0.0, abs=1e-9)

    def test_boundary_hug_closes_on_the_wall_from_outside_the_band(self):
        """evader (3.85, 0): east-wall distance 1.0, so the radial term is
        clip(1.0 - 0.35, -1, 1) = 0.65 -> 2.0 * 0.65 = 1.3 along (1, 0).
        Tangent is (0, 1) as above, so the direction is (1.3, 1).
        """
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = []
        env.pursuer_pose = np.array([3.85, -3.0, 0.0])
        env.evader_pose = np.array([3.85, 0.0, float(np.arctan2(1.0, 1.3))])
        v, w = ObstacleHuggingEvader().act(None, env)
        assert v == pytest.approx(cfg.evader_v_max)
        assert w == pytest.approx(0.0, abs=1e-9)

    def test_obstacle_beyond_hug_range_is_ignored(self):
        """Obstacle (-4, 0) r=0.5 and evader (1, 0): surface distance
        5.0 - 0.5 = 4.5 > HUG_RANGE 4.0, so the probe falls through to the
        boundary hug. East-wall distance 4.85 - 1.0 = 3.85 -> radial
        clip(3.85 - 0.35) = 1.0 -> 2.0 * (1, 0); tangent (0, 1) with the
        pursuer to the south -> direction (2, 1). If the obstacle had been
        used the direction would point west instead.
        """
        cfg = EnvConfig(n_obstacles=1)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = [{"x": -4.0, "y": 0.0, "radius": 0.5}]
        env.pursuer_pose = np.array([1.0, -3.0, 0.0])
        env.evader_pose = np.array([1.0, 0.0, float(np.arctan2(1.0, 2.0))])
        v, w = ObstacleHuggingEvader().act(None, env)
        assert v == pytest.approx(cfg.evader_v_max)
        assert w == pytest.approx(0.0, abs=1e-9)

    def test_flees_directly_when_the_pursuer_is_far(self):
        """Beyond HUG_THREAT_RANGE 7.0 the probe is a plain flee-away agent,
        even with an obstacle right beside it: pursuer (0,-8) is 8.0 m away,
        so the direction is e - p = (0, 8) and heading pi/2 gives zero turn.
        """
        cfg = EnvConfig(n_obstacles=1)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = [{"x": 1.0, "y": 0.0, "radius": 0.5}]
        env.pursuer_pose = np.array([0.0, -8.0, 0.0])
        env.evader_pose = np.array([0.0, 0.0, np.pi / 2])
        v, w = ObstacleHuggingEvader().act(None, env)
        assert v == pytest.approx(cfg.evader_v_max)
        assert w == pytest.approx(0.0, abs=1e-9)

    def test_probe_drives_an_unshielded_pursuer_into_contact(self):
        """The G4 property in miniature: against this probe an unshielded
        GreedyPursuer must actually touch things. Smoke scale (20 seeds off
        the pre-registered probe stream); the gate measures the real rate.
        """
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        logs = [
            run_duel(env, GreedyPursuer(), ObstacleHuggingEvader(), 770000 + i)
            for i in range(20)
        ]
        with_contact = sum(log["pursuer_violations"] > 0 for log in logs)
        assert with_contact > 0, (
            "probe never drives the unshielded pursuer into contact — "
            "the probe is too weak (pre-registration: fix the probe, not G4)"
        )


class TestOpenSpaceEvader:
    """F1 spec §2: the evader that punishes circling.

    D104: `GreedyEvader` flees directly away with no wall awareness, corners
    itself, and is therefore collected by a slow constant arc — a randomly
    initialised net scored 0.312 against it while the hand-written pursuer
    scored 0.234. This evader refuses headings that end in a wall or an
    obstacle, so it cannot be cornered that way.
    """

    def _env(self, seed=930_000):
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=seed)
        return cfg, env

    def test_it_flees_the_pursuer_in_the_open(self):
        from pe2.scripted import OpenSpaceEvader

        cfg, env = self._env()
        env.obstacles = []
        env.pursuer_pose[:] = (0.0, 0.0, 0.0)
        env.evader_pose[:] = (1.0, 0.0, 0.0)  # due east of the pursuer, facing east
        v, omega = OpenSpaceEvader().act(env._get_obs()["evader"], env)
        assert v == pytest.approx(cfg.evader_v_max)
        assert abs(omega) < 0.5  # already pointing away: little correction

    def test_it_refuses_to_drive_into_a_wall(self):
        from pe2.scripted import OpenSpaceEvader

        cfg, env = self._env()
        env.obstacles = []
        lim = cfg.arena_size / 2.0 - cfg.agent_radius
        # Evader pinned at the +x wall, pursuer due west: fleeing "away" is +x,
        # straight into the wall. It must pick a heading along the wall instead.
        env.evader_pose[:] = (lim - 0.05, 0.0, 0.0)
        env.pursuer_pose[:] = (lim - 3.0, 0.0, 0.0)
        ev = OpenSpaceEvader()
        _v, omega = ev.act(env._get_obs()["evader"], env)
        assert abs(omega) > 1.0  # it turns hard, rather than driving on into +x

    # REMOVED: test_it_does_not_get_cornered_by_a_constant_arc.
    # It encoded D104's hypothesis — that a constant arc collects a
    # self-cornering GreedyEvader — which D105 REFUTED by measurement: a true
    # constant arc at seed 4's mean action (v = 0.5, omega = -0.078) catches
    # GreedyEvader 0 times in 60 episodes. The real mechanism is the teacher
    # standing still on 92.6 % of steps, not the evader cornering itself. The
    # wall-refusal behaviour this evader provides is still worth having and is
    # covered by test_it_refuses_to_drive_into_a_wall above; the arc-specific
    # claim is not, and is not re-asserted here under a different name.

    def test_it_is_deterministic_and_stateless(self):
        from pe2.scripted import OpenSpaceEvader

        _cfg, env = self._env()
        a = OpenSpaceEvader()
        b = OpenSpaceEvader()
        b.reset()
        obs = env._get_obs()["evader"]
        assert a.act(obs, env) == b.act(obs, env)


class TestSearchingPursuer:
    """T2 spec: the teacher that searches by MOVING.

    D105: `FovLimitedPursuer` stands still on 92.6 % of steps and its
    demonstrations are 83.6 % 'stand still and rotate', so every learner in
    four rounds cloned a policy that barely moves. This one orbits instead.
    The orbit is a pure function of own pose (spec §2) — adding a coverage
    map would reintroduce exactly the hidden-state defect that makes a
    memoryless clone unable to follow its teacher (D79).
    """

    def _env(self, seed=970_000):
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=seed)
        return cfg, env

    def test_it_never_stands_still_while_searching(self):
        """The whole point: the branch that used to return v = 0."""
        from pe2.scripted import SearchingPursuer

        cfg, env = self._env()
        p = SearchingPursuer()
        speeds = []
        for x, y, th in ((0.0, 0.0, 0.0), (2.0, 0.0, 1.5), (-3.0, 1.0, -2.0)):
            env.pursuer_pose[:] = (x, y, th)
            # Put the evader directly BEHIND the pursuer so it is outside the
            # 90 deg cone whatever the heading — otherwise this fixture tests
            # the chase branch by accident.
            env.evader_pose[:] = (x - 3.0 * np.cos(th), y - 3.0 * np.sin(th), 0.0)
            obs = env._get_obs()["pursuer"]
            assert obs[6] < 0.5, "fixture must be a not-visible state"
            v, _omega = p.act(obs, env)
            speeds.append(v)
        assert all(s == pytest.approx(cfg.v_max) for s in speeds)

    def test_the_search_heading_is_a_function_of_own_pose_alone(self):
        """Spec §2: no hidden state. A used instance must agree with a fresh
        one on the same pose."""
        from pe2.scripted import SearchingPursuer

        _cfg, env = self._env()
        fresh = SearchingPursuer()
        used = SearchingPursuer()
        for x, y in ((3.0, 3.0), (-2.0, 2.0), (0.5, -3.5)):
            env.pursuer_pose[:] = (x, y, 0.0)
            env.evader_pose[:] = (x - 3.0, y, 0.0)  # behind, out of the cone
            used.act(env._get_obs()["pursuer"], env)
        env.pursuer_pose[:] = (1.0, -1.0, 0.3)
        env.evader_pose[:] = (1.0 - 3.0 * np.cos(0.3), -1.0 - 3.0 * np.sin(0.3), 0.0)
        obs = env._get_obs()["pursuer"]
        assert obs[6] < 0.5
        assert used.act(obs, env) == fresh.act(obs, env)

    def test_it_orbits_rather_than_driving_to_the_centre(self):
        from pe2.scripted import PATROL_RADIUS, SearchingPursuer

        _cfg, env = self._env()
        env.obstacles = []
        env.evader_pose[:] = (4.5, 4.5, 0.0)
        p = SearchingPursuer()
        env.pursuer_pose[:] = (4.0, 0.0, 0.0)
        radii = []
        for _ in range(400):
            obs = env._get_obs()["pursuer"]
            v, omega = p.act(obs, env)
            env.step([v, omega, 0.0, 0.0])
            radii.append(float(np.hypot(*env.pursuer_pose[:2])))
        tail = np.array(radii[-200:])
        assert abs(tail.mean() - PATROL_RADIUS) < 1.5, tail.mean()
        assert tail.min() > 0.5, "it collapsed to the centre"

    def test_it_covers_ground(self):
        """Contrast with the old teacher, which spins in place."""
        from pe2.scripted import FovLimitedPursuer, SearchingPursuer

        _cfg, env = self._env()
        travelled = {}
        for name, cls in (("old", FovLimitedPursuer), ("new", SearchingPursuer)):
            obs, _ = env.reset(seed=970_001)
            p = cls()
            p.reset()
            dist = 0.0
            prev = env.pursuer_pose[:2].copy()
            for _ in range(600):
                v, omega = p.act(obs["pursuer"], env)
                obs, _r, term, trunc, _i = env.step([v, omega, 0.0, 0.0])
                dist += float(np.hypot(*(env.pursuer_pose[:2] - prev)))
                prev = env.pursuer_pose[:2].copy()
                if term or trunc:
                    break
            travelled[name] = dist
        assert travelled["new"] > 3.0 * travelled["old"], travelled

    def test_chase_and_last_seen_branches_are_unchanged(self):
        """Spec: exactly ONE branch differs from FovLimitedPursuer."""
        from pe2.scripted import FovLimitedPursuer, SearchingPursuer

        _cfg, env = self._env()
        env.pursuer_pose[:] = (0.0, 0.0, 0.0)
        env.evader_pose[:] = (2.0, 0.3, 0.0)
        obs = env._get_obs()["pursuer"]
        assert obs[6] > 0.5, "fixture must be a visible state"
        assert SearchingPursuer().act(obs, env) == FovLimitedPursuer().act(obs, env)

        old, new = FovLimitedPursuer(), SearchingPursuer()
        old.last_seen = np.array([4.0, 4.0])
        new.last_seen = np.array([4.0, 4.0])
        env.evader_pose[:] = (-4.5, -4.5, 0.0)
        obs = env._get_obs()["pursuer"]
        assert obs[6] < 0.5
        assert new.act(obs, env) == old.act(obs, env)
