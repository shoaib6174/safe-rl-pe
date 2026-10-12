"""pe2.planner — `MpcInterceptLaw` (MPC-H) and `pe2.env.pursuer_step_batch`.

Spec: docs/pathway/pursuit_strategy_plan_2026-10-12.md sections 1, 2 and 4.

Two load-bearing tests. `TestCGS1` is the one-variable invariant: with the
evader stationary the sampling never runs and the law is bit-identical to
`PlannerInterceptLaw`, logs AND per-step executed actions, on 50 obstacle-free
seeds. `test_pursuer_step_batch_equals_the_scalar_law_bitwise` is the one that
binds the vectorised dynamics to the environment's own: the law plans over a
SECOND copy of `pursuer_step`, and only that test makes "our own exact
dynamics" true rather than approximately true.
"""

import time

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv, pursuer_step, pursuer_step_batch
from pe2.obs import full_state_observation, opponent_world_from_obs, own_pose_from_obs
from pe2.planner import (
    MPC_PATH_WEIGHT,
    LeadPlannerInterceptLaw,
    MpcInterceptLaw,
    PlannerInterceptLaw,
)
from pe2.scripted import GreedyEvader, StationaryEvader
from pe2.structured import OrbitSearch, SearchMDP

CFG = EnvConfig()
OPEN_CFG = EnvConfig(n_physical_obstacles=0)  # observation layout unchanged
CONSTRUCT_SEED0 = 1_990_000
SEEDS = [CONSTRUCT_SEED0 + i for i in range(50)]
# A sampling law costs ~1.6 ms per call and an episode is up to 1200 of them, so
# the MOVING-evader episode tests run on a handful of seeds, not fifty.
FEW = SEEDS[:2]
HALF = CFG.arena_size / 2.0
X_LIM = (-HALF + CFG.agent_radius, HALF - CFG.agent_radius)
_OBSTACLE = {"x": 1.0, "y": 0.0, "radius": 0.5}


def _logs_and_trace(mdp, policy, seeds):
    out = []
    for s in seeds:
        mdp.trace = []
        out.append((mdp.play_episode(policy.predict, s), list(mdp.trace)))
    return out


def _run(cfg, law, opponent, seeds=SEEDS):
    return _logs_and_trace(SearchMDP(cfg, opponent, law=law), OrbitSearch(cfg), seeds)


def _obs(cfg, own_pose, evader_xy, visible: bool = True):
    """A hand-built observation: own pose, evader at `evader_xy`, no obstacles
    placed in the observation (the law's map comes from `reset(env=...)`)."""
    o = full_state_observation(
        np.asarray(own_pose, dtype=float),
        np.array([evader_xy[0], evader_xy[1], 0.0]),
        [],
        cfg,
    )
    if not visible:
        o[3:7] = 0.0
    return o


class _StubEnv:
    """Just enough env for `reset(env=...)`: a step index, a static map and a
    Generator to take the episode entropy from."""

    def __init__(self, step: int = 0, obstacles=(), seed: int = 0):
        self.current_step = step
        self.obstacles = list(obstacles)
        self.np_random = np.random.default_rng(seed)


def _moving(cfg, law, own_pose, a_xy, b_xy, env):
    """Two consecutive sightings of an evader moving a -> b. Returns the law's
    action at b, the decoded own pose and the decoded evader position at b."""
    law.act(_obs(cfg, own_pose, a_xy))
    env.current_step += 1
    o = _obs(cfg, own_pose, b_xy)
    return law.act(o), own_pose_from_obs(o, cfg), opponent_world_from_obs(o, cfg)


def _held_cost(law, own, pos, v_est, action):
    """The cost of ONE action held for the whole horizon, scored by the law's
    own rollout. `(cost, capture step, predicted evader path)`."""
    held = np.tile(np.asarray(action, dtype=np.float32), (1, law.n_seg, 1))
    cost, cap, e_pred = law._rollout(own, pos, v_est, held)
    return float(cost[0]), float(cap[0]), e_pred


def _chosen_and_direct(cfg, own_pose, a_xy, b_xy, obstacles=()):
    """Run MPC on two consecutive sightings a -> b, and recompute the DIRECT
    action (the parent's own) independently on the same inputs by calling the
    parent's `act` on a twin. Returns (law, chosen, direct, own, pos)."""
    law = MpcInterceptLaw(cfg)
    env = _StubEnv(obstacles=obstacles)
    law.reset(env=env)
    chosen, own, pos = _moving(cfg, law, own_pose, a_xy, b_xy, env)
    twin = MpcInterceptLaw(cfg)
    twin_env = _StubEnv(obstacles=obstacles)
    twin.reset(env=twin_env)
    twin.act(_obs(cfg, own_pose, a_xy))
    twin_env.current_step += 1
    direct = LeadPlannerInterceptLaw.act(twin, _obs(cfg, own_pose, b_xy))
    return law, chosen, direct, own, pos


class TestCGS1:
    """Plan section 4, CG-S1 — one variable: with the evader stationary the
    sampling never runs and MPC is bit-identical to `PLAN`."""

    def test_stationary_evader_is_bit_identical_over_50_episodes(self):
        got = _run(OPEN_CFG, MpcInterceptLaw(OPEN_CFG), StationaryEvader)
        want = _run(OPEN_CFG, PlannerInterceptLaw(OPEN_CFG), StationaryEvader)
        assert got == want
        # The traces are the EXECUTED actions; empty ones would be vacuous.
        assert all(len(t) > 0 for _log, t in got)

    def test_the_plan_changes_something_against_a_moving_evader(self):
        got = _run(OPEN_CFG, MpcInterceptLaw(OPEN_CFG), GreedyEvader, FEW)
        want = _run(OPEN_CFG, PlannerInterceptLaw(OPEN_CFG), GreedyEvader, FEW)
        assert got != want, "the sampled plan changed no episode at all"

    def test_an_invisible_evader_delegates_and_records_nothing(self):
        law = MpcInterceptLaw(OPEN_CFG)
        env = _StubEnv()
        law.reset(env=env)
        a = law.act(_obs(OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 0.0), visible=False))
        assert a is None  # no last-seen yet: the parent's third branch
        assert law.last_aim is None
        assert law.last_feasible_count == 0
        assert law.last_chose_direct is False


class TestDeterminism:
    def test_the_same_episode_gives_the_same_actions(self):
        a = _run(OPEN_CFG, MpcInterceptLaw(OPEN_CFG), GreedyEvader, FEW)
        b = _run(OPEN_CFG, MpcInterceptLaw(OPEN_CFG), GreedyEvader, FEW)
        assert a == b

    def test_the_episode_entropy_comes_from_the_env_and_differs_per_episode(self):
        law = MpcInterceptLaw(OPEN_CFG)
        env = PursuitEvasionEnv(OPEN_CFG)
        env.reset(seed=CONSTRUCT_SEED0)
        law.reset(env=env)
        first = law._ep_seed
        env.reset(seed=CONSTRUCT_SEED0 + 1)
        law.reset(env=env)
        assert law._ep_seed != first
        env.reset(seed=CONSTRUCT_SEED0)  # the same episode seed again
        law.reset(env=env)
        assert law._ep_seed == first
        law.reset()  # no env: the map-only unit path
        assert law._ep_seed == 0


class TestSequences:
    def test_the_direct_sequence_is_always_row_zero_held_throughout(self):
        law = MpcInterceptLaw(OPEN_CFG)
        a_direct = np.array([0.37, -0.21], dtype=np.float32)
        acts = law._sequences(a_direct, np.random.default_rng(0))
        assert acts.shape == (law.k, law.n_seg, 2)
        assert np.array_equal(acts[0], np.tile(a_direct, (law.n_seg, 1)))
        assert acts.min() >= -1.0 and acts.max() <= 1.0
        other = law._sequences(a_direct, np.random.default_rng(1))
        assert np.array_equal(other[0], acts[0])  # row 0 never depends on the rng
        assert not np.array_equal(other[1:], acts[1:])

    def test_the_horizon_must_divide_into_whole_segments(self):
        with pytest.raises(ValueError):
            MpcInterceptLaw(OPEN_CFG, horizon=0.75)


class TestSelection:
    def test_the_direct_sequence_wins_when_it_is_min_cost(self):
        """An evader 1.0 m ahead fleeing at 0.6 m/s: capture inside H needs
        essentially full speed on the bearing, which is exactly the parent's
        own action, so no sampled sequence can beat it."""
        d = 0.6 * OPEN_CFG.dt
        law, chosen, direct, own, pos = _chosen_and_direct(
            OPEN_CFG, (0.0, 0.0, 0.0), (1.0, 0.0), (1.0 + d, 0.0)
        )
        assert law.last_v_est[0] == pytest.approx(0.6, abs=1e-4)
        assert law.last_chose_direct is True
        assert np.array_equal(chosen, direct)
        cost, _cap, _e = _held_cost(law, own, pos, law.last_v_est, direct)
        assert law.last_cost == cost
        assert law.last_cost < OPEN_CFG.capture_distance  # i.e. it captures

    def test_an_earlier_capture_beats_the_direct_sequence(self):
        """A head-on closing evader: the parent's tau = 2 s lead aims BEHIND the
        pursuer, so holding its action captures only late in the horizon; a
        sampled sequence that lets the evader come captures far earlier."""
        d = 1.2 * OPEN_CFG.dt
        law, _chosen, direct, own, pos = _chosen_and_direct(
            OPEN_CFG, (0.0, 0.0, 0.0), (2.0 + d, 0.0), (2.0, 0.0)
        )
        assert law.last_v_est[0] < 0.0  # closing
        assert law.last_chose_direct is False
        assert law.last_cost < OPEN_CFG.capture_distance  # the plan captures
        cost_d, cap_d, e_pred = _held_cost(law, own, pos, law.last_v_est, direct)
        assert np.isfinite(cap_d)  # the direct sequence captures too ...
        assert law.last_cost < cost_d
        # `last_aim` is the predicted evader position at the capture step, so
        # the step it was taken from says when the chosen plan captures.
        hits = np.flatnonzero((e_pred == law.last_aim).all(axis=1))
        assert len(hits) == 1
        assert hits[0] < cap_d  # ... and later than the chosen sequence

    @pytest.mark.parametrize(
        ("a_xy", "b_xy"),
        [
            ((2.0, 1.0), (2.0, 1.06)),
            ((1.5, -0.5), (1.44, -0.5)),
            ((-2.0, 2.0), (-1.94, 2.0)),
        ],
    )
    def test_the_parents_own_cost_is_an_upper_bound_on_the_chosen_cost(
        self, a_xy, b_xy
    ):
        """Row 0 is the parent's action, so the plan can never cost more."""
        law, _chosen, direct, own, pos = _chosen_and_direct(
            OPEN_CFG, (0.0, 0.0, 0.0), a_xy, b_xy
        )
        cost, _cap, _e = _held_cost(law, own, pos, law.last_v_est, direct)
        assert law.last_cost <= cost


class TestFeasibility:
    def test_a_sequence_through_an_obstacle_costs_infinity(self):
        law = MpcInterceptLaw(CFG)
        law.reset(env=_StubEnv(obstacles=[_OBSTACLE]))
        own = np.array([0.0, 0.0, 0.0])
        pos = np.array([2.0, 0.0])
        v_est = np.array([0.0, 1.2])
        acts = np.array(
            [
                [[1.0, 0.0]] * law.n_seg,  # full speed straight into the disc
                [[-1.0, 0.0]] * law.n_seg,  # v = 0: stays where it is
            ],
            dtype=np.float32,
        )
        cost, cap, _ = law._rollout(own, pos, v_est, acts)
        assert not np.isfinite(cost[0])
        assert not np.isfinite(cap[0])
        assert np.isfinite(cost[1])

    def test_an_infeasible_sequence_is_never_chosen_when_a_feasible_one_exists(self):
        d = 1.2 * CFG.dt
        law, _chosen, _direct, _own, _pos = _chosen_and_direct(
            CFG, (0.0, 0.0, 0.0), (2.0, 0.2), (2.0, 0.2 + d), obstacles=[_OBSTACLE]
        )
        # Both kinds are in the sample, and the chosen one is a feasible one.
        assert 0 < law.last_feasible_count < law.k
        assert np.isfinite(law.last_cost)

    def test_a_wall_makes_a_sequence_infeasible(self):
        law = MpcInterceptLaw(OPEN_CFG)
        law.reset(env=_StubEnv())
        own = np.array([X_LIM[1] - 0.1, 0.0, 0.0])  # 0.1 m from the wall, facing it
        acts = np.array(
            [[[1.0, 0.0]] * law.n_seg, [[-1.0, 0.0]] * law.n_seg], dtype=np.float32
        )
        cost, _cap, _e = law._rollout(
            own, np.array([0.0, 0.0]), np.array([0.0, 1.2]), acts
        )
        assert not np.isfinite(cost[0])
        assert np.isfinite(cost[1])


class TestCost:
    def test_the_cost_is_the_min_distance_plus_the_weighted_path(self):
        """v = 0 held for the horizon: the path is zero and the cost is exactly
        the minimum predicted distance -- the cost formula, read off."""
        law = MpcInterceptLaw(OPEN_CFG)
        law.reset(env=_StubEnv())
        own = np.array([0.0, 0.0, 0.0])
        pos = np.array([2.0, 0.0])
        v_est = np.array([1.2, 0.0])  # fleeing: the minimum is the first step
        acts = np.array([[[-1.0, 0.0]] * law.n_seg], dtype=np.float32)
        cost, cap, e_pred = law._rollout(own, pos, v_est, acts)
        assert not np.isfinite(cap[0])
        assert cost[0] == pytest.approx(float(np.hypot(*e_pred[0])), abs=1e-12)
        assert len(e_pred) == law.h_steps

    def test_the_path_term_is_the_predicted_path_length(self):
        law = MpcInterceptLaw(OPEN_CFG)
        law.reset(env=_StubEnv())
        own = np.array([0.0, 0.0, 0.0])
        pos = np.array([-4.0, 0.0])  # behind, and fleeing: no capture
        v_est = np.array([-1.2, 0.0])
        acts = np.array([[[1.0, 0.0]] * law.n_seg], dtype=np.float32)
        cost, _cap, e_pred = law._rollout(own, pos, v_est, acts)
        travelled = OPEN_CFG.v_max * OPEN_CFG.dt * law.h_steps
        min_d = min(
            float(np.hypot(e[0] - OPEN_CFG.v_max * OPEN_CFG.dt * (t + 1), e[1]))
            for t, e in enumerate(e_pred)
        )
        assert cost[0] == pytest.approx(min_d + MPC_PATH_WEIGHT * travelled, abs=1e-9)


class TestBatchDynamics:
    def test_pursuer_step_batch_equals_the_scalar_law_bitwise(self):
        """The law plans over a SECOND copy of `pursuer_step`; this is what makes
        it the same dynamics. 1000 random inputs, plus the two degenerate poses
        that sit exactly on an obstacle centre."""
        rng = np.random.default_rng(20_261_012)
        obstacles = [
            {"x": 0.5, "y": -0.3, "radius": 0.6},
            {"x": -1.2, "y": 2.0, "radius": 0.35},
        ]
        n = 1000
        poses = np.column_stack(
            [
                rng.uniform(-5.0, 5.0, n),
                rng.uniform(-5.0, 5.0, n),
                rng.uniform(-4.0, 4.0, n),
            ]
        )
        poses[0] = [0.5, -0.3, 0.7]  # exactly at a centre (the 1e-10 branch)
        poses[1] = [-1.2, 2.0, -1.1]
        v = rng.uniform(0.0, CFG.v_max, n)
        w = rng.uniform(-CFG.omega_max, CFG.omega_max, n)
        got, wall, obstacle = pursuer_step_batch(
            poses, v, w, CFG, obstacles, X_LIM, X_LIM
        )
        for i in range(n):
            p, cw, co = pursuer_step(poses[i], v[i], w[i], CFG, obstacles, X_LIM, X_LIM)
            assert np.array_equal(p, got[i]), i
            assert cw == bool(wall[i]) and co == bool(obstacle[i]), i
        # Vacuous if nothing ever touched anything.
        assert wall.any() and obstacle.any()

    def test_it_matches_on_an_empty_map_too(self):
        rng = np.random.default_rng(3)
        poses = np.column_stack([rng.uniform(-1, 1, 50)] * 3)
        v = rng.uniform(0.0, CFG.v_max, 50)
        w = rng.uniform(-CFG.omega_max, CFG.omega_max, 50)
        got, _wall, obstacle = pursuer_step_batch(poses, v, w, CFG, [], X_LIM, X_LIM)
        assert not obstacle.any()
        for i in range(50):
            p, _cw, _co = pursuer_step(poses[i], v[i], w[i], CFG, [], X_LIM, X_LIM)
            assert np.array_equal(p, got[i])


def test_a_call_stays_inside_the_two_millisecond_budget(capsys):
    """100 sampling calls on the DEFAULT arena (2 obstacles). The number is
    reported; the bar is the plan's 2 ms target, and a loaded host will show it."""
    env = PursuitEvasionEnv(CFG)
    env.reset(seed=2_100_000)
    law = MpcInterceptLaw(CFG)
    law.reset(env=env)
    d = 1.2 * CFG.dt
    o1 = _obs(CFG, (0.0, 0.0, 0.0), (2.0, 0.5))
    o2 = _obs(CFG, (0.0, 0.0, 0.0), (2.0, 0.5 + d))
    law.act(o1)
    times = []
    for i in range(100):
        env.current_step = i + 1
        t0 = time.perf_counter()
        law.act(o2 if i % 2 == 0 else o1)
        times.append(time.perf_counter() - t0)
    assert law.last_aim is not None  # the sampling really ran on every call
    mean_ms = 1000.0 * float(np.mean(times))
    with capsys.disabled():
        print(f"\nMpcInterceptLaw.act: mean {mean_ms:.2f} ms over 100 calls")
    assert mean_ms <= 2.0


class TestReset:
    def test_reset_clears_the_plan_records(self):
        law = MpcInterceptLaw(OPEN_CFG)
        env = _StubEnv()
        law.reset(env=env)
        d = 1.2 * OPEN_CFG.dt
        _moving(OPEN_CFG, law, (0.0, 0.0, 0.0), (2.0, 0.0), (2.0, d), env)
        assert law.last_aim is not None
        assert law.last_feasible_count > 0
        law.reset(env=env)
        assert law.last_aim is None
        assert law.last_cost == float("inf")
        assert law.last_feasible_count == 0
        assert law.last_chose_direct is False
