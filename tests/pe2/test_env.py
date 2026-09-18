"""Tests for pe2.env — Step 1 automated verify.

Covers (plan Step 1):
- dynamics integrator vs hand-computed trajectories
- wall/obstacle projection never NaN, never penetrates
- capture terminates; timeout truncates; prep phase freezes pursuer
- seeded reset reproducibility
- 10K random steps, zero invariant violations (in-arena, no penetration)
"""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.env import OUTCOME_CAPTURE, OUTCOME_TIMEOUT, PursuitEvasionEnv, wrap_angle
from pe2.obs import obs_dim
from pe2.rng import derive_seed, make_rng


def open_env(**overrides) -> PursuitEvasionEnv:
    """Env with no obstacles and a big arena, for clean dynamics tests."""
    cfg = EnvConfig(arena_size=100.0, n_obstacles=0, **overrides)
    env = PursuitEvasionEnv(cfg)
    env.reset(seed=0)
    return env


def set_poses(env, pursuer, evader):
    env.pursuer_pose = np.array(pursuer, dtype=float)
    env.evader_pose = np.array(evader, dtype=float)


class TestDynamicsIntegrator:
    def test_straight_line(self):
        env = open_env()
        set_poses(env, pursuer=(0.0, 0.0, 0.5), evader=(50.0, 50.0, 0.0))
        theta0 = env.pursuer_pose[2]
        env.step([0.7, 0.0, 0.0, 0.0])
        expected_x = 0.0 + 0.7 * np.cos(theta0) * env.cfg.dt
        expected_y = 0.0 + 0.7 * np.sin(theta0) * env.cfg.dt
        np.testing.assert_allclose(
            env.pursuer_pose, [expected_x, expected_y, theta0], atol=1e-12
        )

    def test_turn_in_place(self):
        env = open_env()
        set_poses(env, pursuer=(0.0, 0.0, 0.3), evader=(50.0, 50.0, 0.0))
        env.step([0.0, 2.0, 0.0, 0.0])
        expected_theta = wrap_angle(0.3 + 2.0 * env.cfg.dt)
        np.testing.assert_allclose(
            env.pursuer_pose, [0.0, 0.0, expected_theta], atol=1e-12
        )

    def test_constant_curvature_matches_euler_recurrence(self):
        """Hand-computed Euler integration over 50 steps (independent of env code)."""
        env = open_env()
        set_poses(env, pursuer=(1.0, -2.0, 0.7), evader=(50.0, 50.0, 0.0))
        v, omega, dt = 0.9, 1.3, env.cfg.dt
        x, y, theta = 1.0, -2.0, 0.7
        for _ in range(50):
            env.step([v, omega, 0.0, 0.0])
            x = x + v * np.cos(theta) * dt
            y = y + v * np.sin(theta) * dt
            theta = wrap_angle(theta + omega * dt)
            np.testing.assert_allclose(env.pursuer_pose, [x, y, theta], atol=1e-10)

    def test_evader_uses_speed_ratio(self):
        env = open_env()
        set_poses(env, pursuer=(-50.0, -50.0, 0.0), evader=(0.0, 0.0, 0.0))
        env.step([0.0, 0.0, env.cfg.evader_v_max + 5.0, 0.0])  # clipped to evader_v_max
        np.testing.assert_allclose(
            env.evader_pose[:2], [env.cfg.evader_v_max * env.cfg.dt, 0.0], atol=1e-12
        )

    def test_action_clipping(self):
        env = open_env()
        set_poses(env, pursuer=(0.0, 0.0, 0.0), evader=(50.0, 50.0, 0.0))
        env.step([99.0, -99.0, 0.0, 0.0])
        expected_theta = wrap_angle(-env.cfg.omega_max * env.cfg.dt)
        np.testing.assert_allclose(
            env.pursuer_pose,
            [env.cfg.v_max * env.cfg.dt, 0.0, expected_theta],
            atol=1e-12,
        )


class TestContacts:
    def test_wall_clip_no_tunnel(self):
        env = open_env()  # 100x100 arena -> bound at 50 - r
        bound = 50.0 - env.cfg.agent_radius
        set_poses(env, pursuer=(bound - 0.01, 0.0, 0.0), evader=(-49.0, -49.0, 0.0))
        for _ in range(100):
            _, _, terminated, _, _ = env.step([env.cfg.v_max, 0.0, 0.0, 0.0])
            assert not terminated
            assert env.pursuer_pose[0] <= bound + 1e-12
        assert env.pursuer_pose[0] == pytest.approx(bound)

    def test_wall_contact_flag(self):
        env = open_env()
        bound = 50.0 - env.cfg.agent_radius
        set_poses(env, pursuer=(bound - 0.001, 0.0, 0.0), evader=(-49.0, -49.0, 0.0))
        _, _, _, _, info = env.step([env.cfg.v_max, 0.0, 0.0, 0.0])
        assert info["pursuer_wall_contact"] is True

    def test_obstacle_projection_no_penetration(self):
        cfg = EnvConfig(arena_size=100.0, n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = [{"x": 5.0, "y": 0.0, "radius": 1.0}]
        min_dist = 1.0 + cfg.agent_radius
        set_poses(env, pursuer=(0.0, 0.0, 0.0), evader=(-49.0, -49.0, 0.0))
        for _ in range(400):
            env.step([cfg.v_max, 0.0, 0.0, 0.0])  # drive straight at obstacle
            d = np.hypot(env.pursuer_pose[0] - 5.0, env.pursuer_pose[1] - 0.0)
            assert d >= min_dist - 1e-9
            assert np.all(np.isfinite(env.pursuer_pose))
        # Pinned against the obstacle surface by the projection
        d = np.hypot(env.pursuer_pose[0] - 5.0, env.pursuer_pose[1] - 0.0)
        assert d == pytest.approx(min_dist)


class TestTermination:
    def test_capture_terminates(self):
        env = open_env()
        d = env.cfg.capture_distance
        set_poses(env, pursuer=(0.0, 0.0, 0.0), evader=(d * 0.9, 0.0, np.pi))
        _, rewards, terminated, truncated, info = env.step([0.0, 0.0, 0.0, 0.0])
        assert terminated and not truncated
        assert env.outcome == OUTCOME_CAPTURE
        assert info["outcome"] == OUTCOME_CAPTURE
        assert rewards == {"pursuer": 1.0, "evader": -1.0}

    def test_no_capture_just_outside_radius(self):
        env = open_env()
        d = env.cfg.capture_distance
        set_poses(env, pursuer=(0.0, 0.0, 0.0), evader=(d * 1.1, 0.0, 0.0))
        _, _, terminated, truncated, _ = env.step([0.0, 0.0, 0.0, 0.0])
        assert not terminated and not truncated

    def test_timeout_truncates(self):
        env = open_env(max_steps=5)
        set_poses(env, pursuer=(-40.0, 0.0, 0.0), evader=(40.0, 0.0, np.pi))
        for _ in range(4):
            _, _, terminated, truncated, _ = env.step([0.0, 0.0, 0.0, 0.0])
            assert not terminated and not truncated
        _, rewards, terminated, truncated, info = env.step([0.0, 0.0, 0.0, 0.0])
        assert truncated and not terminated
        assert env.outcome == OUTCOME_TIMEOUT
        assert info["outcome"] == OUTCOME_TIMEOUT
        assert rewards == {"pursuer": -1.0, "evader": 1.0}

    def test_prep_phase_freezes_pursuer_translation(self):
        env = open_env(prep_steps=10)
        set_poses(env, pursuer=(0.0, 0.0, 0.0), evader=(40.0, 0.0, 0.0))
        for i in range(10):
            env.step([env.cfg.v_max, 0.0, env.cfg.evader_v_max, 0.0])
            np.testing.assert_allclose(
                env.pursuer_pose[:2],
                [0.0, 0.0],
                atol=1e-12,
                err_msg=f"pursuer moved during prep step {i}",
            )
        assert env.evader_pose[0] > 0.0  # evader was free to move
        env.step([env.cfg.v_max, 0.0, 0.0, 0.0])  # step 11: pursuer free
        assert env.pursuer_pose[0] > 0.0


class TestSeeding:
    def test_seeded_reset_reproducible(self):
        env = PursuitEvasionEnv(EnvConfig())
        env.reset(seed=42)
        state_a = (env.pursuer_pose.copy(), env.evader_pose.copy(), list(env.obstacles))
        env.reset(seed=42)
        np.testing.assert_array_equal(env.pursuer_pose, state_a[0])
        np.testing.assert_array_equal(env.evader_pose, state_a[1])
        assert env.obstacles == state_a[2]

    def test_different_seeds_differ(self):
        env = PursuitEvasionEnv(EnvConfig())
        env.reset(seed=1)
        a = np.concatenate([env.pursuer_pose, env.evader_pose])
        env.reset(seed=2)
        b = np.concatenate([env.pursuer_pose, env.evader_pose])
        assert not np.array_equal(a, b)

    def test_episode_determinism(self):
        """Same seed + same action script -> bit-identical trajectory."""
        rng = make_rng(derive_seed(0, "test_episode_determinism"))
        actions = rng.uniform(0.0, 1.0, size=(300, 4))

        def run():
            env = PursuitEvasionEnv(EnvConfig())
            env.reset(seed=7)
            for a in actions:
                # scale into bounds: v in [0, v_max], w in [-w_max, w_max]
                joint = np.array(
                    [
                        a[0] * 1.0,
                        (a[1] * 2 - 1) * 2.84,
                        a[2] * 1.2,
                        (a[3] * 2 - 1) * 2.84,
                    ]
                )
                env.step(joint)
            return np.concatenate([env.pursuer_pose, env.evader_pose])

        np.testing.assert_array_equal(run(), run())


class TestInvariantSweep:
    def test_10k_random_steps_zero_violations(self):
        """Plan Step 1 gate: 10K random steps, in-arena + no penetration + finite."""
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        rng = make_rng(derive_seed(0, "test_invariant_sweep"))
        bound = cfg.arena_size / 2.0 - cfg.agent_radius
        episodes = 0
        steps = 0
        env.reset(seed=derive_seed(0, "sweep_reset"))
        while steps < 10_000:
            joint = np.array(
                [
                    rng.uniform(0, cfg.v_max),
                    rng.uniform(-cfg.omega_max, cfg.omega_max),
                    rng.uniform(0, cfg.evader_v_max),
                    rng.uniform(-cfg.omega_max, cfg.omega_max),
                ]
            )
            _, _, terminated, truncated, _ = env.step(joint)
            steps += 1
            for pose in (env.pursuer_pose, env.evader_pose):
                assert np.all(np.isfinite(pose)), f"NaN pose at step {steps}"
                assert -bound - 1e-9 <= pose[0] <= bound + 1e-9
                assert -bound - 1e-9 <= pose[1] <= bound + 1e-9
                for obs in env.obstacles:
                    d = np.hypot(pose[0] - obs["x"], pose[1] - obs["y"])
                    assert d >= obs["radius"] + cfg.agent_radius - 1e-9, (
                        f"obstacle penetration at step {steps}"
                    )
            if terminated or truncated:
                episodes += 1
                env.reset(seed=derive_seed(episodes, "sweep_reset"))
        assert episodes >= 3  # sweep actually exercised resets


class TestEulerStepIdentity:
    """The shield predicts the env's next pose with this function; the
    zero-contact argument in the Step 6 v2 spec (B4) needs bit identity."""

    def test_pursuer_step_uses_euler_step_bitwise(self):
        from pe2.env import euler_step, pursuer_step

        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        rng = np.random.default_rng(12345)
        for _ in range(10_000):
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            # Keep away from the walls so the clip never engages.
            env.pursuer_pose[:] = (
                rng.uniform(-3.0, 3.0),
                rng.uniform(-3.0, 3.0),
                rng.uniform(-np.pi, np.pi),
            )
            env.obstacles = []
            v = float(rng.uniform(0.0, cfg.v_max))
            w = float(rng.uniform(-cfg.omega_max, cfg.omega_max))
            predicted = euler_step(env.pursuer_pose.copy(), v, w, cfg.dt)
            new_pose, wall, obstacle = pursuer_step(
                env.pursuer_pose,
                v,
                w,
                cfg,
                env.obstacles,
                (env._x_min, env._x_max),
                (env._y_min, env._y_max),
            )
            assert tuple(new_pose) == predicted  # bitwise, no approx
            assert not wall and not obstacle

    def test_zero_action_is_the_identity_on_position(self):
        from pe2.env import euler_step

        pose = np.array([1.2345678901234567, -2.3456789012345678, 0.7])
        x, y, _ = euler_step(pose, 0.0, 0.0, 0.05)
        assert x == pose[0] and y == pose[1]


class TestPhysicalObstacleCount:
    """`n_physical_obstacles` decouples the arena's obstacle COUNT from the
    observation layout (S9 attack-surface sweep): frozen policies trained
    with `n_obstacles=2` must be evaluatable in 0--4-obstacle arenas without
    their observation vector changing shape.
    """

    def test_default_is_bit_identical_to_the_explicit_value(self):
        """Unset field == n_obstacles: same seed, same obstacles AND poses."""
        a = PursuitEvasionEnv(EnvConfig())
        b = PursuitEvasionEnv(EnvConfig(n_physical_obstacles=2))
        a.reset(seed=7)
        b.reset(seed=7)
        assert a.obstacles == b.obstacles  # dict floats, compared bitwise
        assert tuple(a.pursuer_pose) == tuple(b.pursuer_pose)
        assert tuple(a.evader_pose) == tuple(b.evader_pose)

    def test_four_physical_obstacles_keep_the_observation_layout(self):
        cfg = EnvConfig(n_physical_obstacles=4)
        env = PursuitEvasionEnv(cfg)
        obs, _ = env.reset(seed=7)
        assert len(env.obstacles) == 4
        assert obs_dim(cfg) == obs_dim(EnvConfig())
        assert obs["pursuer"].shape == (obs_dim(EnvConfig()),)
        assert obs["evader"].shape == (obs_dim(EnvConfig()),)

    def test_zero_physical_obstacles_runs_with_empty_obstacle_slots(self):
        cfg = EnvConfig(n_physical_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        obs, _ = env.reset(seed=7)
        assert env.obstacles == []
        assert obs["pursuer"].shape == (obs_dim(EnvConfig()),)
        for _ in range(5):
            obs, _, terminated, truncated, _ = env.step([0.5, 0.1, 0.5, -0.1])
            if terminated or truncated:
                break
        # Slots 7..10 are the two obstacle (dx, dy) pairs of the LAYOUT.
        np.testing.assert_array_equal(obs["pursuer"][7:11], np.zeros(4))
        np.testing.assert_array_equal(obs["evader"][7:11], np.zeros(4))

    def test_generator_is_sequential_so_counts_nest(self):
        """First two obstacles of a 4-obstacle arena == the two of a
        2-obstacle arena at the same seed (rejection sampling is sequential,
        so a larger count only appends)."""
        two = PursuitEvasionEnv(EnvConfig(n_physical_obstacles=2))
        four = PursuitEvasionEnv(EnvConfig(n_physical_obstacles=4))
        two.reset(seed=7)
        four.reset(seed=7)
        assert four.obstacles[:2] == two.obstacles

    def test_counts_zero_through_four_all_reset(self):
        for n in range(5):
            env = PursuitEvasionEnv(EnvConfig(n_physical_obstacles=n))
            for seed in range(20):
                env.reset(seed=seed)
                assert len(env.obstacles) == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# S11 Task 1: the pursuer/evader dynamics live in ONE pure function
# (`pe2.env.pursuer_step`) so a lookahead safety filter can simulate the env's
# exact step instead of transcribing it. Both tests below are what makes that
# claim checkable: a pinned trace digest (the refactor changed nothing) and a
# composition equality (the env keeps going through the function).
# ---------------------------------------------------------------------------

TRACE_SEEDS = 200
TRACE_STEPS = 50

# sha256 of `build_step_trace()`'s poses+flags bytes, recorded by running the
# identical builder against the PRE-refactor env (commit ba66f8f, where
# `step` called `_advance_agent` + `_project_out_of_obstacles` directly).
HEAD_TRACE_DIGEST = "edce840ddfcc941b3df931e5bd4ad8c9e20f82afb8c4cdb70af34089afb14122"


def trace_configs():
    """Two configs: the locked default, and a tight arena that forces
    frequent wall clips, 4-obstacle projections and a prep phase."""
    return (
        EnvConfig(),
        EnvConfig(
            arena_size=6.0,
            n_physical_obstacles=4,
            prep_steps=3,
            min_init_distance=1.0,
        ),
    )


def build_step_trace(seeds=TRACE_SEEDS, steps=TRACE_STEPS):
    """Poses + contact flags over `seeds` episodes x `steps` random actions.

    Actions are drawn slightly OUTSIDE the action space so `step`'s clipping
    is exercised too. Returns (float64 (N, 6), uint8 (N, 6)).
    """
    poses: list[tuple[float, ...]] = []
    flags: list[tuple[int, ...]] = []
    for cfg_index, cfg in enumerate(trace_configs()):
        env = PursuitEvasionEnv(cfg)
        for seed in range(seeds):
            env.reset(seed=1000 * cfg_index + seed)
            rng = np.random.default_rng(90_000 + 1000 * cfg_index + seed)
            for _ in range(steps):
                action = np.array(
                    [
                        rng.uniform(-0.2, cfg.v_max + 0.2),
                        rng.uniform(-cfg.omega_max - 0.5, cfg.omega_max + 0.5),
                        rng.uniform(-0.2, cfg.evader_v_max + 0.2),
                        rng.uniform(-cfg.omega_max - 0.5, cfg.omega_max + 0.5),
                    ]
                )
                _, _, terminated, truncated, info = env.step(action)
                poses.append((*env.pursuer_pose, *env.evader_pose))
                flags.append(
                    (
                        int(info["pursuer_wall_contact"]),
                        int(info["evader_wall_contact"]),
                        int(info["pursuer_obstacle_contact"]),
                        int(info["evader_obstacle_contact"]),
                        int(terminated),
                        int(truncated),
                    )
                )
    return np.array(poses, dtype=np.float64), np.array(flags, dtype=np.uint8)


class TestPursuerStepIsTheEnvDynamics:
    def test_step_trace_is_bit_identical_to_the_pinned_pre_refactor_trace(self):
        """20 000 steps (2 configs x 200 seeds x 50 actions), BOTH roles.

        Regenerate ONLY with a logged decision: a changed digest means the
        environment's dynamics changed, which invalidates every frozen
        policy and every shielded number measured against it.
        """
        import hashlib

        poses, flags = build_step_trace()
        assert poses.shape == (2 * TRACE_SEEDS * TRACE_STEPS, 6)
        # The trace must actually exercise walls, obstacles and capture.
        for column in range(4):
            assert flags[:, column].sum() > 100, f"contact column {column} unused"
        assert flags[:, 4].sum() > 0  # captures occurred

        digest = hashlib.sha256()
        digest.update(poses.tobytes())
        digest.update(flags.tobytes())
        assert digest.hexdigest() == HEAD_TRACE_DIGEST

    def test_env_step_equals_pursuer_step_for_both_roles(self):
        """Composition equality: `step` IS `pursuer_step` per agent.

        Guards against anyone re-inlining the dynamics into the env (which
        is how the predictive filter's prediction would silently drift).
        """
        from pe2.env import pursuer_step

        for cfg in trace_configs():
            env = PursuitEvasionEnv(cfg)
            rng = np.random.default_rng(4242)
            for seed in range(60):
                env.reset(seed=seed)
                for _ in range(25):
                    action = np.array(
                        [
                            rng.uniform(-0.2, cfg.v_max + 0.2),
                            rng.uniform(-cfg.omega_max - 0.5, cfg.omega_max + 0.5),
                            rng.uniform(-0.2, cfg.evader_v_max + 0.2),
                            rng.uniform(-cfg.omega_max - 0.5, cfg.omega_max + 0.5),
                        ]
                    )
                    v_p = float(np.clip(action[0], 0.0, cfg.v_max))
                    w_p = float(np.clip(action[1], -cfg.omega_max, cfg.omega_max))
                    v_e = float(np.clip(action[2], 0.0, cfg.evader_v_max))
                    w_e = float(np.clip(action[3], -cfg.omega_max, cfg.omega_max))
                    if env.current_step < cfg.prep_steps:
                        v_p = 0.0
                    x_lim = (env._x_min, env._x_max)
                    y_lim = (env._y_min, env._y_max)
                    p_pose, p_wall, p_obs = pursuer_step(
                        env.pursuer_pose, v_p, w_p, cfg, env.obstacles, x_lim, y_lim
                    )
                    e_pose, e_wall, e_obs = pursuer_step(
                        env.evader_pose, v_e, w_e, cfg, env.obstacles, x_lim, y_lim
                    )
                    _, _, _, _, info = env.step(action)
                    assert tuple(env.pursuer_pose) == tuple(p_pose)  # bitwise
                    assert tuple(env.evader_pose) == tuple(e_pose)
                    assert info["pursuer_wall_contact"] == p_wall
                    assert info["evader_wall_contact"] == e_wall
                    assert info["pursuer_obstacle_contact"] == p_obs
                    assert info["evader_obstacle_contact"] == e_obs

    def test_pursuer_step_is_pure(self):
        """No env state, no mutation of the caller's pose — the filter will
        call it many times per step from the SAME pose."""
        from pe2.env import pursuer_step

        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=3)
        obstacles = env.obstacles
        pose = np.array([obstacles[0]["x"], obstacles[0]["y"], 0.4])  # inside a disk
        before = pose.copy()
        lim = ((env._x_min, env._x_max), (env._y_min, env._y_max))
        out_a = pursuer_step(pose, 0.8, 1.0, cfg, obstacles, *lim)
        out_b = pursuer_step(pose, 0.8, 1.0, cfg, obstacles, *lim)

        assert tuple(pose) == tuple(before)  # input untouched
        assert tuple(out_a[0]) == tuple(out_b[0])  # deterministic
        assert out_a[1:] == out_b[1:]
        assert out_a[2] is True  # the overlap was resolved
        assert out_a[0] is not out_b[0]  # a fresh array each call
        # A list pose works too (the filter simulates from tuples/lists).
        out_c = pursuer_step(list(pose), 0.8, 1.0, cfg, obstacles, *lim)
        assert tuple(out_c[0]) == tuple(out_a[0])
