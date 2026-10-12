"""pe2.planner — `LeadPlannerInterceptLaw` (LEAD-tau).

Spec: docs/pathway/pursuit_strategy_plan_2026-10-12.md sections 1 and 4.

The load-bearing test is `TestCGS1`: with the evader stationary the lead law
must be bit-identical to `PlannerInterceptLaw`, logs AND per-step executed
actions, on 50 obstacle-free seeds. Without it the S23 round would compare two
different laws rather than one law with one variable changed. Its companion is
`test_the_lead_changes_something_against_a_moving_evader`: an identity that
also held against a MOVING evader would mean the lead does nothing at all.
"""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.obs import full_state_observation, opponent_world_from_obs
from pe2.planner import LeadPlannerInterceptLaw, PlannerInterceptLaw
from pe2.scripted import GreedyEvader, StationaryEvader
from pe2.structured import OrbitSearch, SearchMDP

CFG = EnvConfig()
OPEN_CFG = EnvConfig(n_physical_obstacles=0)  # observation layout unchanged
CONSTRUCT_SEED0 = 1_990_000
SEEDS = [CONSTRUCT_SEED0 + i for i in range(50)]
LIM = OPEN_CFG.arena_size / 2.0 - OPEN_CFG.agent_radius
TAUS = (0.5, 1.0)


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
    placed. `full_state_observation` forces the visibility flag, so the FOV
    cone plays no part in these unit cases."""
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
    """Just enough env for `reset(env=...)`: a step index and a static map."""

    def __init__(self, step: int = 0, obstacles=()):
        self.current_step = step
        self.obstacles = list(obstacles)


class TestCGS1:
    """Plan section 4, CG-S1 — one variable: with the evader stationary every
    member of the family is bit-identical to `PLAN`."""

    @pytest.mark.parametrize("tau", TAUS)
    def test_stationary_evader_is_bit_identical_over_50_episodes(self, tau):
        got = _run(OPEN_CFG, LeadPlannerInterceptLaw(OPEN_CFG, tau), StationaryEvader)
        want = _run(OPEN_CFG, PlannerInterceptLaw(OPEN_CFG), StationaryEvader)
        assert got == want
        # The traces are the EXECUTED actions; empty ones would be vacuous.
        assert all(len(t) > 0 for _log, t in got)

    @pytest.mark.parametrize("tau", TAUS)
    def test_the_lead_changes_something_against_a_moving_evader(self, tau):
        got = _run(OPEN_CFG, LeadPlannerInterceptLaw(OPEN_CFG, tau), GreedyEvader)
        want = _run(OPEN_CFG, PlannerInterceptLaw(OPEN_CFG), GreedyEvader)
        differ = sum(1 for a, b in zip(got, want, strict=True) if a != b)
        assert differ > 0, "the lead aim point changed no episode at all"


class TestVelocityEstimate:
    def test_two_consecutive_sightings_difference_to_dt(self):
        law = LeadPlannerInterceptLaw(OPEN_CFG, tau=1.0)
        law.reset()
        o1 = _obs(OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 0.0))
        o2 = _obs(OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 0.06))
        law.act(o1)
        assert np.array_equal(law.last_v_est, np.zeros(2))  # nothing to difference
        assert law.last_aim is None
        law.act(o2)
        p1 = opponent_world_from_obs(o1, OPEN_CFG)
        p2 = opponent_world_from_obs(o2, OPEN_CFG)
        assert np.array_equal(law.last_v_est, (p2 - p1) / OPEN_CFG.dt)

    def test_a_visibility_gap_gives_zero_and_the_delegate_path(self):
        law = LeadPlannerInterceptLaw(OPEN_CFG, tau=1.0)
        law.reset()
        o1 = _obs(OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 0.0))
        law.act(o1)
        law.act(_obs(OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 0.06), visible=False))
        assert np.array_equal(law.last_v_est, np.zeros(2))
        o3 = _obs(OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 0.12))
        a = law.act(o3)  # visible again, but not the immediately preceding step
        assert np.array_equal(law.last_v_est, np.zeros(2))
        assert law.last_aim is None
        assert np.array_equal(a, PlannerInterceptLaw(OPEN_CFG).act(o3))

    def test_the_env_step_index_is_the_source_not_the_call_count(self):
        """`SearchMDP` does not call the law during a learner's hold span, so
        two consecutive CALLS can be `hold` env steps apart."""
        env = _StubEnv()
        law = LeadPlannerInterceptLaw(OPEN_CFG, tau=1.0)
        law.reset(env=env)
        law.act(_obs(OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 0.0)))  # env step 0
        env.current_step = 10  # a hold span the law was not consulted during
        law.act(_obs(OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 0.6)))
        assert np.array_equal(law.last_v_est, np.zeros(2))
        env.current_step = 11  # now immediately consecutive
        law.act(_obs(OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 0.66)))
        assert not np.array_equal(law.last_v_est, np.zeros(2))

    def test_a_stationary_evader_decodes_to_exactly_zero_velocity(self):
        """The reconstruction is own + R(theta) @ offset from a float32 obs, so
        a frozen evader's apparent position moves when OWN pose does; below the
        module's decoding precision that is read as exactly zero."""
        law = LeadPlannerInterceptLaw(OPEN_CFG, tau=1.0)
        law.reset()
        law.act(_obs(OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 1.0)))
        law.act(_obs(OPEN_CFG, (0.05, 0.0, 0.01), (2.0, 1.0)))
        assert np.array_equal(law.last_v_est, np.zeros(2))
        assert law.last_aim is None


class TestAimPoint:
    def _moving(self, cfg, law, own_pose, a_xy, b_xy, env=None):
        """Two consecutive sightings of an evader moving a -> b; returns the
        decoded position at b."""
        law.act(_obs(cfg, own_pose, a_xy))
        if env is not None:
            env.current_step += 1
        law.act(_obs(cfg, own_pose, b_xy))
        return opponent_world_from_obs(_obs(cfg, own_pose, b_xy), cfg)

    def test_aim_is_the_position_plus_tau_times_the_estimate(self):
        law = LeadPlannerInterceptLaw(OPEN_CFG, tau=1.0)
        law.reset()
        pos = self._moving(OPEN_CFG, law, (0.0, 0.0, 0.0), (2.0, 0.0), (2.0, 0.06))
        assert law.last_v_est[1] == pytest.approx(OPEN_CFG.evader_v_max, abs=1e-4)
        assert np.array_equal(law.last_aim, pos + law.tau * law.last_v_est)
        assert law.in_arena(law.last_aim)

    @pytest.mark.parametrize("tau", TAUS)
    def test_tau_scales_the_lead(self, tau):
        law = LeadPlannerInterceptLaw(OPEN_CFG, tau)
        law.reset()
        pos = self._moving(OPEN_CFG, law, (0.0, 0.0, 0.0), (2.0, 0.0), (2.0, 0.06))
        assert np.array_equal(law.last_aim, pos + tau * law.last_v_est)

    def test_an_aim_beyond_the_arena_is_clipped_to_the_inset_bound(self):
        law = LeadPlannerInterceptLaw(OPEN_CFG, tau=1.0)
        law.reset()
        pos = self._moving(OPEN_CFG, law, (0.0, 0.0, 0.0), (4.0, 0.0), (4.06, 0.0))
        raw = pos + law.tau * law.last_v_est
        assert raw[0] > LIM  # the unclipped aim really is outside
        assert law.last_aim[0] == law._lim == pytest.approx(LIM)
        assert law.in_arena(law.last_aim)

    def test_an_aim_inside_an_inflated_disc_is_snapped_by_goal_snapping(self):
        obstacle = {"x": 2.0, "y": 0.0, "radius": 0.5}
        law = LeadPlannerInterceptLaw(CFG, tau=1.0)
        env = _StubEnv(obstacles=[obstacle])
        law.reset(env=env)
        pos = self._moving(
            CFG, law, (-2.0, -1.4, 0.0), (2.0, -1.46), (2.0, -1.4), env=env
        )
        raw = np.clip(pos + law.tau * law.last_v_est, -law._lim, law._lim)
        assert law.segment_clearance(raw, raw) <= 0.0  # the raw aim is blocked
        goal = law._snap_goal(raw)
        assert goal is not None
        assert np.array_equal(law.last_aim, law._centre(*goal))
        assert law.segment_clearance(law.last_aim, law.last_aim) > 0.0


class TestReset:
    def test_reset_clears_the_velocity_memory_and_the_records(self):
        law = LeadPlannerInterceptLaw(OPEN_CFG, tau=1.0)
        law.reset()
        law.act(_obs(OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 0.0)))
        law.act(_obs(OPEN_CFG, (0.0, 0.0, 0.0), (2.0, 0.06)))
        assert law._prev is not None
        assert law.last_aim is not None
        assert not np.array_equal(law.last_v_est, np.zeros(2))
        law.reset()
        assert law._prev is None
        assert law.last_aim is None
        assert np.array_equal(law.last_v_est, np.zeros(2))
        assert law.last_seen is None  # the parent's state, cleared as before

    def test_n_unreachable_stays_cumulative_across_reset(self):
        law = LeadPlannerInterceptLaw(CFG, tau=1.0)
        law.set_map([{"x": 0.0, "y": 0.0, "radius": 0.5}])
        law._steer_planned(np.array([-2.0, 0.0, 0.0]), np.array([0.0, 0.0]))
        assert law.n_unreachable == 1
        law.reset()
        assert law.n_unreachable == 1


class TestStationaryEvader:
    def test_it_never_moves_and_matches_the_scripted_interface(self):
        e = StationaryEvader()
        assert StationaryEvader.uses_truth is GreedyEvader.uses_truth is True
        e.reset()
        assert e.act(np.zeros(11, dtype=np.float32), None) == (0.0, 0.0)
