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
from pe2.planner import MPC_PATH_WEIGHT, MpcInterceptLaw, PlannerInterceptLaw
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


def _scalar_roll(cfg, law, own, v, w):
    """Roll ONE constant command over the horizon with the SCALAR
    `pursuer_step`. Returns the terminal xy and the path length."""
    pose = np.asarray(own, dtype=float)
    path = 0.0
    for _ in range(law.h_steps):
        nxt, _wall, _obstacle = pursuer_step(
            pose, v, w, cfg, law.obstacles, law._x_lim, law._y_lim
        )
        path += float(np.hypot(nxt[0] - pose[0], nxt[1] - pose[1]))
        pose = nxt
    return pose[:2], path


def _held_cost(law, own, pos, v_est, action):
    """The cost of ONE action held for the whole horizon, scored by the law's
    own rollout. `(cost, capture step, predicted evader path)`."""
    held = np.tile(np.asarray(action, dtype=np.float32), (1, law.n_seg, 1))
    cost, cap, e_pred = law._rollout(own, pos, v_est, held)
    return float(cost[0]), float(cap[0]), e_pred


def _chosen_and_direct(cfg, own_pose, a_xy, b_xy, obstacles=()):
    """Run MPC on two consecutive sightings a -> b, and take the DIRECT row
    independently from a standalone `PlannerInterceptLaw` on the same
    observation (A2: the direct row is `PLAN`'s own action).
    Returns (law, chosen, direct, own, pos)."""
    law = MpcInterceptLaw(cfg)
    env = _StubEnv(obstacles=obstacles)
    law.reset(env=env)
    chosen, own, pos = _moving(cfg, law, own_pose, a_xy, b_xy, env)
    plan = PlannerInterceptLaw(cfg)
    plan.reset(env=_StubEnv(obstacles=obstacles))
    direct = plan.act(_obs(cfg, own_pose, b_xy))
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
    def test_the_direct_row_is_plans_own_action(self):
        """A2: the always-included sequence is the GRANDPARENT's action, aimed
        at the evader's CURRENT position -- not the lead aim point."""
        d = 1.2 * OPEN_CFG.dt
        law, chosen, direct, _own, _pos = _chosen_and_direct(
            OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 0.0), (2.0 + d, 0.0)
        )
        assert law.tau == 0.0  # tau plays no part in this law
        assert law.last_chose_direct is True
        assert np.array_equal(chosen, direct)
        rng = np.random.default_rng((law._ep_seed, 1))
        acts = law._sequences(direct, rng)
        assert np.array_equal(acts[0], np.tile(direct, (law.n_seg, 1)))

    def test_the_direct_sequence_wins_when_it_is_min_cost(self):
        """An evader 1.0 m ahead fleeing at 0.6 m/s: capture inside H needs
        essentially full speed on the bearing, which is exactly `PLAN`'s own
        action, so no sampled sequence can beat it."""
        d = 0.6 * OPEN_CFG.dt
        law, chosen, direct, own, pos = _chosen_and_direct(
            OPEN_CFG, (0.0, 0.0, 0.0), (1.0, 0.0), (1.0 + d, 0.0)
        )
        assert law.last_v_est[0] == pytest.approx(0.6, abs=1e-4)
        assert law.last_chose_direct is True
        assert np.array_equal(chosen, direct)
        cost, cap, _e = _held_cost(law, own, pos, law.last_v_est, direct)
        assert law.last_cost == cost
        assert np.isfinite(cap)
        assert law.last_cost < 0.0  # A2: a capture is negative

    def test_a_capturing_sample_beats_a_non_capturing_direct(self):
        """A2's ordering, end to end: an evader crossing at 0.3 m/s that `PLAN`
        never reaches inside H (its held action ends 1.66 m away), while a
        sampled sequence cuts it off -- so the sampled one is chosen."""
        law, _chosen, direct, own, pos = _chosen_and_direct(
            OPEN_CFG, (0.0, 0.0, 0.0), (1.0, 0.5 + 0.3 * OPEN_CFG.dt), (1.0, 0.5)
        )
        cost_d, cap_d, e_pred = _held_cost(law, own, pos, law.last_v_est, direct)
        assert not np.isfinite(cap_d)  # PLAN's held action never captures ...
        assert cost_d > 0.0
        assert law.last_chose_direct is False
        assert law.last_cost < 0.0  # ... the chosen one does
        assert law.last_cost < cost_d
        # `last_aim` is the predicted evader position at the capture step.
        hits = np.flatnonzero((e_pred == law.last_aim).all(axis=1))
        assert len(hits) == 1
        # cost = -(H - capture step) + 0.01 x path, and the path term is in
        # [0, 0.01 x v_max x H] = [0, 0.02].
        assert 0.0 <= law.last_cost - (hits[0] - law.h_steps) <= 0.021

    def test_an_earlier_capture_costs_strictly_less_than_a_later_one(self):
        """A2's capture bonus, read off three held sequences in a head-on
        closing geometry: slower means a later predicted capture means a
        strictly higher cost, and all three beat every non-capture."""
        law = MpcInterceptLaw(OPEN_CFG)
        law.reset(env=_StubEnv())
        own = np.array([0.0, 0.0, 0.0])
        pos = np.array([2.0, 0.0])
        v_est = np.array([-1.2, 0.0])  # closing head-on
        acts = np.array(
            [
                [[1.0, 0.0]] * law.n_seg,  # v = v_max: the earliest capture
                [[0.0, 0.0]] * law.n_seg,  # v = v_max / 2
                [[-1.0, 0.0]] * law.n_seg,  # v = 0: the evader arrives alone
            ],
            dtype=np.float32,
        )
        cost, cap, _e = law._rollout(own, pos, v_est, acts)
        assert list(cap) == sorted(cap)  # later and later captures
        assert cost[0] < cost[1] < cost[2] < 0.0
        for i in range(3):
            # cost = -(H - capture step) + 0.01 x path
            assert cost[i] == pytest.approx(cap[i] - law.h_steps, abs=0.021)

    def test_the_terminal_cost_is_not_flat_across_headings(self):
        """The A2 amendment's reason to exist: against a radially fleeing evader
        the cost must still discriminate between headings, which the original
        minimum-over-the-horizon cost did not (it was decided by the 0.01 x path
        tie-breaker, i.e. by noise)."""
        d = 1.2 * OPEN_CFG.dt
        law, chosen, direct, own, pos = _chosen_and_direct(
            OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 0.0), (2.0 + d, 0.0)
        )
        rng = np.random.default_rng((law._ep_seed, 1))
        cost, cap, e_pred = law._rollout(
            own, pos, law.last_v_est, law._sequences(direct, rng)
        )
        assert np.isfinite(cost).all()  # open arena: nothing is infeasible
        assert not np.isfinite(cap).any()  # a faster evader fleeing: no capture
        assert float(cost.max() - cost.min()) > 0.05
        # And the winner is the one that ends nearest the predicted TERMINAL
        # position -- which here is straight ahead, so `PLAN`'s own action.
        assert law.last_chose_direct is True
        assert np.array_equal(law.last_aim, e_pred[-1])
        assert float(chosen[0]) == 1.0 and float(chosen[1]) == 0.0

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
        """Row 0 is `PLAN`'s action, so the plan can never cost more (A2)."""
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
    """A2's formula, read off against the SCALAR dynamics rolled by hand."""

    def test_a_standing_sequence_costs_exactly_the_terminal_distance(self):
        law = MpcInterceptLaw(OPEN_CFG)
        law.reset(env=_StubEnv())
        own = np.array([0.0, 0.0, 0.0])
        pos = np.array([2.0, 0.0])
        v_est = np.array([1.2, 0.0])  # fleeing, and faster: no capture
        acts = np.array([[[-1.0, 0.0]] * law.n_seg], dtype=np.float32)  # v = 0
        cost, cap, e_pred = law._rollout(own, pos, v_est, acts)
        assert len(e_pred) == law.h_steps
        assert not np.isfinite(cap[0])
        assert cost[0] == pytest.approx(float(np.hypot(*e_pred[-1])), abs=1e-12)
        # NOT the minimum over the horizon, which is the first step (A2).
        assert cost[0] > float(np.hypot(*e_pred[0]))

    def test_the_cost_is_the_terminal_distance_plus_the_weighted_path(self):
        law = MpcInterceptLaw(OPEN_CFG)
        law.reset(env=_StubEnv())
        own = np.array([0.0, 0.0, 0.0])
        pos = np.array([-4.0, 0.0])  # behind, and fleeing: no capture
        v_est = np.array([-1.2, 0.0])
        acts = np.array([[[1.0, 0.0]] * law.n_seg], dtype=np.float32)
        cost, cap, e_pred = law._rollout(own, pos, v_est, acts)
        end, path = _scalar_roll(OPEN_CFG, law, own, OPEN_CFG.v_max, 0.0)
        assert not np.isfinite(cap[0])
        assert e_pred[-1][0] == pytest.approx(-law._lim)  # the prediction clips
        assert cost[0] == pytest.approx(
            float(np.hypot(*(end - e_pred[-1]))) + MPC_PATH_WEIGHT * path, abs=1e-9
        )


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
