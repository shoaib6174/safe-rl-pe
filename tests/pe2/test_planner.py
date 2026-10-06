"""pe2.planner — `PlannerInterceptLaw` (spec:
docs/pathway/planner_intercept_prereg_2026-10-06.md, section 2 and CG-A).

The load-bearing test is `TestInvariant`: on an arena with no obstacles the
planner law must be bit-identical to `InterceptLaw`, logs AND per-step
executed actions. Without it the S19 round compares two different laws rather
than one law against itself-plus-planning.
"""

import importlib.util
import pathlib
from typing import ClassVar

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.planner import CELL, LOOKAHEAD, MARGIN, PlannerInterceptLaw
from pe2.scripted import GreedyEvader
from pe2.structured import InterceptLaw, OrbitSearch, SearchMDP

CFG = EnvConfig()
OPEN_CFG = EnvConfig(n_physical_obstacles=0)  # the observation layout is
# unchanged (n_obstacles still owns it); only the PLACED obstacles go away.
CONSTRUCT_SEED0 = 1_990_000
LIM = CFG.arena_size / 2.0 - CFG.agent_radius


def _logs_and_trace(mdp, policy, seeds):
    out = []
    for s in seeds:
        mdp.trace = []
        log = mdp.play_episode(policy.predict, s)
        out.append((log, list(mdp.trace)))
    return out


class TestInvariant:
    """Prereg section 2: bit-identical on an obstacle-free arena."""

    def test_open_arena_is_bit_identical_over_50_episodes(self):
        seeds = [CONSTRUCT_SEED0 + i for i in range(50)]
        plan = SearchMDP(OPEN_CFG, GreedyEvader, law=PlannerInterceptLaw(OPEN_CFG))
        line = SearchMDP(OPEN_CFG, GreedyEvader)
        got = _logs_and_trace(plan, OrbitSearch(OPEN_CFG), seeds)
        want = _logs_and_trace(line, OrbitSearch(OPEN_CFG), seeds)
        assert got == want
        # The traces are the EXECUTED actions; an empty one would make the
        # assertion vacuous.
        assert all(len(t) > 0 for _log, t in got)

    def test_open_arena_never_needs_the_planner(self):
        law = PlannerInterceptLaw(OPEN_CFG)
        mdp = SearchMDP(OPEN_CFG, GreedyEvader, law=law)
        policy = OrbitSearch(OPEN_CFG)
        for i in range(50):
            mdp.play_episode(policy.predict, CONSTRUCT_SEED0 + i)
        assert law.n_unreachable == 0


class TestHook:
    """The `law` hook must leave the default SearchMDP untouched."""

    def test_default_law_is_the_intercept_law(self):
        assert isinstance(SearchMDP(CFG, GreedyEvader).law, InterceptLaw)
        assert type(SearchMDP(CFG, GreedyEvader).law) is InterceptLaw

    def test_passing_the_default_law_explicitly_changes_nothing(self):
        seeds = [CONSTRUCT_SEED0 + i for i in range(20)]
        implicit = SearchMDP(CFG, GreedyEvader)
        explicit = SearchMDP(CFG, GreedyEvader, law=InterceptLaw(CFG))
        assert _logs_and_trace(implicit, OrbitSearch(CFG), seeds) == _logs_and_trace(
            explicit, OrbitSearch(CFG), seeds
        )

    def test_intercept_law_reset_accepts_and_ignores_env(self):
        law = InterceptLaw(CFG)
        law.last_seen = np.array([1.0, 2.0])
        law.reset(env=object())
        assert law.last_seen is None


class TestClearLine:
    def test_a_grazing_segment_is_blocked_and_a_1e_6_miss_is_clear(self):
        law = PlannerInterceptLaw(CFG)
        law.set_map([{"x": 0.0, "y": 0.0, "radius": 0.3}])
        r_eff = 0.3 + CFG.agent_radius + MARGIN
        graze = ((-2.0, r_eff), (2.0, r_eff))
        assert law.segment_clearance(*graze) == pytest.approx(0.0, abs=1e-12)
        assert not law.is_clear(*graze)
        miss = ((-2.0, r_eff + 1e-6), (2.0, r_eff + 1e-6))
        assert law.is_clear(*miss)

    def test_endpoints_count_only_where_they_are(self):
        """A segment that ends before the disc is clear; one that reaches it
        is not (point-to-SEGMENT distance, not point-to-line)."""
        law = PlannerInterceptLaw(CFG)
        law.set_map([{"x": 0.0, "y": 0.0, "radius": 0.3}])
        assert law.is_clear((-3.0, 0.0), (-1.0, 0.0))
        assert not law.is_clear((-3.0, 0.0), (0.0, 0.0))

    def test_arena_inset_bounds_are_respected(self):
        law = PlannerInterceptLaw(CFG)
        law.set_map([])
        assert law.in_arena((LIM - 1e-3, 0.0))
        assert not law.in_arena((LIM + 1e-3, 0.0))
        assert not law.in_arena((0.0, -LIM - 1e-3))
        assert law.is_clear((0.0, 0.0), (LIM - 1e-3, 0.0))
        assert not law.is_clear((0.0, 0.0), (LIM + 1e-3, 0.0))

    def test_an_empty_map_is_always_clear(self):
        law = PlannerInterceptLaw(CFG)
        law.set_map([])
        assert law.segment_clearance((-4.0, -4.0), (4.0, 4.0)) == float("inf")


class TestAStar:
    OBS: ClassVar[dict] = {"x": 0.0, "y": 0.0, "radius": 0.5}

    def _law(self):
        law = PlannerInterceptLaw(CFG)
        law.set_map([self.OBS])
        return law

    def test_path_leaves_the_inflated_disc_at_every_vertex(self):
        law = self._law()
        own, target = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
        assert not law.is_clear(own, target)  # the straight line is blocked
        path = law.plan(own, target)
        assert path is not None and len(path) >= 3
        r_eff = self.OBS["radius"] + CFG.agent_radius + MARGIN
        for v in path:
            assert float(np.hypot(*v)) >= r_eff - 1e-9
            assert law.in_arena(v)
        assert np.allclose(path[0], own)
        assert np.allclose(path[-1], target)

    def test_the_lookahead_point_is_not_inside_the_disc(self):
        law = self._law()
        path = law.plan(np.array([-2.0, 0.0]), np.array([2.0, 0.0]))
        p = law.lookahead_point(path, LOOKAHEAD)
        r_eff = self.OBS["radius"] + CFG.agent_radius + MARGIN
        assert float(np.hypot(*p)) >= r_eff - 1e-9
        # 0.5 m along the path, as the spec says.
        assert float(np.hypot(*(p - path[0]))) == pytest.approx(LOOKAHEAD, abs=1e-9)

    def test_lookahead_is_the_goal_when_the_path_is_shorter(self):
        law = PlannerInterceptLaw(CFG)
        law.set_map([])
        path = [np.array([0.0, 0.0]), np.array([0.2, 0.0])]
        assert np.allclose(law.lookahead_point(path, LOOKAHEAD), [0.2, 0.0])

    def test_cells_are_the_spec_resolution(self):
        assert CELL == 0.1
        assert LOOKAHEAD == 0.5
        assert MARGIN == 0.0


class TestShortcut:
    def test_open_map_smooths_to_the_straight_segment(self):
        law = PlannerInterceptLaw(CFG)
        law.set_map([])
        own, target = np.array([-3.0, -1.0]), np.array([2.5, 1.5])
        path = law.plan(own, target)
        assert len(path) == 2
        assert np.allclose(path[0], own)
        assert np.allclose(path[1], target)


class TestUnreachable:
    def test_n_unreachable_counts_a_target_inside_an_inflated_obstacle(self):
        law = PlannerInterceptLaw(CFG)
        law.set_map([{"x": 0.0, "y": 0.0, "radius": 0.5}])
        own = np.array([-2.0, 0.0, 0.0])  # pose: x, y, theta
        target = np.array([0.0, 0.0])  # the obstacle's own centre
        assert law.plan(own[:2], target) is None
        assert law.n_unreachable == 0
        a = law._steer_planned(own, target)
        assert law.n_unreachable == 1
        # "steer at the target directly": the line law's action.
        assert np.array_equal(a, law._steer_at(own, target))

    def test_counter_survives_reset(self):
        """It is cumulative by design so CG-C can read a whole cell."""
        law = PlannerInterceptLaw(CFG)
        law.set_map([{"x": 0.0, "y": 0.0, "radius": 0.5}])
        law._steer_planned(np.array([-2.0, 0.0, 0.0]), np.array([0.0, 0.0]))
        law.reset()
        assert law.n_unreachable == 1


def _cg_a_module():
    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "s19_cg_a.py"
    spec = importlib.util.spec_from_file_location("s19_cg_a", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCGA:
    """Prereg section 5, CG-A — BLOCKING before any S19 capture rate is read."""

    def test_200_planned_paths_stay_clear_and_in_arena(self):
        mod = _cg_a_module()
        report = mod.check(mod.sample_cases())
        assert report["n_cases"] == 200
        assert report["violations"] == 0, report["bad_seeds"]
        # The gate would be vacuous if no case were actually obstructed.
        assert report["n_line_blocked"] > 0
