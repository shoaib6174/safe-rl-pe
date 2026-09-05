"""Tests for scripts/derisk_t3_exact_shield_shift.py -- the T3 exact twin.

The whole point of T3 is that its arithmetic is exact, so a discrepancy here is
a bug and never noise. These tests cover the three places a wrong number could
come from: the shield's transition table, the exact game solvers, and the
decision the runner makes with the numbers.

Purely tabular -- no learning library is imported, so nothing is skipped.
"""

import importlib.util
import pathlib
import sys

import numpy as np

_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "derisk_t3_exact_shield_shift.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("derisk_t3", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["derisk_t3"] = mod
    spec.loader.exec_module(mod)
    return mod


T3 = _load()


def _tab(layout, config, projection=None):
    sp, se = T3.SHIELD_CONFIGS[config]
    return T3.Tables(T3.Game(layout, sp, se, projection or T3.NEAREST_CELL, T3.HORIZON))


LAYOUTS = {ly.name: ly for ly in T3.LAYOUTS}


class TestShieldTransition:
    def test_unshielded_moves_walk_straight_into_hazards(self):
        # If the unshielded arm avoided hazards too, the experiment would be
        # comparing two shielded games.
        tab = _tab(LAYOUTS["dense"], "none")
        entered = {
            int(tab.mv_p[i, a])
            for i in tab.init_cells
            for a in range(T3.N_ACTIONS)
            if int(tab.mv_p[i, a]) in tab.hazards
        }
        assert entered, "unshielded pursuer never enters a hazard"

    def test_shielded_moves_never_land_in_a_hazard(self):
        for name, layout in LAYOUTS.items():
            tab = _tab(layout, "both")
            for i in tab.init_cells:
                for a in range(T3.N_ACTIONS):
                    assert int(tab.mv_p[i, a]) not in tab.hazards, (name, i, a)
                    assert int(tab.mv_e[i, a]) not in tab.hazards, (name, i, a)

    def test_shield_never_fails_open_on_an_occupiable_cell(self):
        for layout in LAYOUTS.values():
            assert _tab(layout, "both").fail_open_occupiable == 0

    def test_shield_leaves_safe_moves_untouched(self):
        tab = _tab(LAYOUTS["corridor"], "both")
        for i in tab.init_cells:
            for a in range(T3.N_ACTIONS):
                if int(tab.mv_raw[i, a]) not in tab.hazards:
                    assert tab.mv_safe[i, a] == tab.mv_raw[i, a]

    def test_nearest_cell_projection_is_the_l1_minimiser(self):
        # The definition under test: replace a blocked move by the reachable
        # non-hazard cell nearest in L1 to the intended target, ties to the
        # lowest action index.
        tab = _tab(LAYOUTS["corridor"], "both")
        for i in tab.init_cells:
            for a in range(T3.N_ACTIONS):
                target = int(tab.mv_raw[i, a])
                if target not in tab.hazards:
                    continue
                tr, tc = T3.rc(target)
                got = int(tab.mv_safe[i, a])
                gr, gc = T3.rc(got)
                got_d = abs(gr - tr) + abs(gc - tc)
                for b in range(T3.N_ACTIONS):
                    cand = int(tab.mv_raw[i, b])
                    if cand in tab.hazards:
                        continue
                    cr, cc = T3.rc(cand)
                    assert got_d <= abs(cr - tr) + abs(cc - tc)

    def test_pursuer_only_leaves_the_evader_unshielded(self):
        tab = _tab(LAYOUTS["dense"], "pursuer_only")
        assert np.array_equal(tab.mv_e, tab.mv_raw)
        assert not np.array_equal(tab.mv_p, tab.mv_raw)

    def test_walls_block_movement_and_are_shield_independent(self):
        for config in ("none", "both"):
            tab = _tab(T3.NEG_CONTROL, config)
            walls = tab.walls
            for i in tab.cells:
                for a in range(T3.N_ACTIONS):
                    assert int(tab.mv_p[i, a]) not in walls


class TestInitialDistributionIsIdenticalAcrossArms:
    def test_both_arms_of_a_layout_share_one_initial_set(self):
        # This is what makes the delta a shield effect rather than a hazard
        # effect: hazards are present in BOTH arms and define the same start
        # states, so they cancel and only the projection remains.
        for layout in LAYOUTS.values():
            sets = [set(_tab(layout, c).init_pairs) for c in T3.SHIELD_CONFIGS]
            assert sets[0] == sets[1] == sets[2]

    def test_start_states_exclude_hazards_walls_and_captures(self):
        tab = _tab(LAYOUTS["corridor"], "none")
        for p, e in tab.init_pairs:
            assert p not in tab.hazards and e not in tab.hazards
            assert p not in tab.walls and e not in tab.walls
            assert not tab.cap[p, e]


class TestExactSolvers:
    def test_nash_value_is_a_saddle_point(self):
        # No unilateral deviation may beat the equilibrium. Exact arithmetic,
        # so the gap must be zero to solver tolerance.
        for name, config in (("sparse", "both"), ("corridor", "pursuer_only")):
            tab = _tab(LAYOUTS[name], config)
            value, strategies, _ = T3.nash_solve(tab, collect_strategies=True)
            check = T3.verify_equilibrium(tab, strategies, value)
            assert check["passes"], (name, config, check)

    def test_nash_value_lies_between_the_two_best_response_values(self):
        tab = _tab(LAYOUTS["sparse"], "both")
        value, _, _ = T3.nash_solve(tab)
        pol_e = T3._greedy_policy(tab, "evader")
        pol_p = T3.best_response(tab, pol_e, "pursuer")
        # Best-responding to a fixed deterministic evader can only do at least
        # as well as playing the minimax strategy against a rational one.
        assert T3.evaluate_pair(tab, pol_p, pol_e) >= value - 1e-12

    def test_best_response_is_optimal_against_its_opponent(self):
        # Brute-force check on the first step: no single-state deviation from
        # the computed best response improves the pursuer's value.
        tab = _tab(LAYOUTS["sparse"], "both")
        pol_e = T3._greedy_policy(tab, "evader")
        pol_p = T3.best_response(tab, pol_e, "pursuer")
        base = T3.evaluate_pair(tab, pol_p, pol_e)
        rng = np.random.default_rng(0)
        n = tab.n
        for _ in range(25):
            t = int(rng.integers(0, tab.game.horizon))
            p = int(rng.choice(tab.cells))
            e = int(rng.choice(tab.cells))
            if tab.cap[p, e]:
                continue
            s = p * n + e
            original = int(pol_p[t, s])
            for a in range(T3.N_ACTIONS):
                pol_p[t, s] = a
                assert T3.evaluate_pair(tab, pol_p, pol_e) <= base + 1e-12
            pol_p[t, s] = original

    def test_values_are_probabilities(self):
        for layout in LAYOUTS.values():
            for config in T3.SHIELD_CONFIGS:
                value, _, _ = T3.nash_solve(_tab(layout, config))
                assert 0.0 <= value <= 1.0

    def test_br_dynamics_terminates_in_a_cycle(self):
        br = T3.br_dynamics(_tab(LAYOUTS["sparse"], "none"))
        assert br["converged"]
        assert br["cycle_length"] >= 1
        assert 0.0 <= br["br_value"] <= 1.0


class TestNegativeControl:
    def test_sealed_pocket_is_never_reachable(self):
        tab = _tab(T3.NEG_CONTROL, "none")
        hazard = T3.idx(0, 6)
        assert hazard not in tab.init_cells
        for i in tab.init_cells:
            for a in range(T3.N_ACTIONS):
                assert int(tab.mv_raw[i, a]) != hazard

    def test_shield_is_a_no_op_on_the_sealed_pocket_layout(self):
        values = {}
        for config in T3.SHIELD_CONFIGS:
            values[config], _, _ = T3.nash_solve(_tab(T3.NEG_CONTROL, config))
        assert values["both"] == values["none"]
        assert values["pursuer_only"] == values["none"]

    def test_empty_hazard_set_is_a_no_op(self):
        empty = T3.Layout("empty", frozenset(), frozenset(), "control")
        a, _, _ = T3.nash_solve(_tab(empty, "none"))
        b, _, _ = T3.nash_solve(_tab(empty, "both"))
        assert a == b


class TestProjectionRuleDoesNotChangeTheValue:
    def test_motion_preserving_matches_nearest_cell(self):
        # Both projections have the same image -- the set of safe cells one
        # action reaches -- so with exact optimisation they induce the same
        # game. The projection rule can only matter to a LEARNER, never to the
        # exact value. Guarding this keeps the reported shift from being read
        # as an artefact of how "nearest" was defined.
        for name in ("sparse", "corridor"):
            a, _, _ = T3.nash_solve(_tab(LAYOUTS[name], "both"))
            b, _, _ = T3.nash_solve(_tab(LAYOUTS[name], "both", T3.MOTION_PRESERVING))
            assert a == b

    def test_the_two_projections_really_do_differ(self):
        # Otherwise the test above would be vacuous.
        tab_a = _tab(LAYOUTS["corridor"], "both")
        tab_b = _tab(LAYOUTS["corridor"], "both", T3.MOTION_PRESERVING)
        assert not np.array_equal(tab_a.mv_safe, tab_b.mv_safe)


class TestStructuralComparability:
    def test_shielded_safe_set_stays_strongly_connected(self):
        for name, layout in LAYOUTS.items():
            assert T3.safe_set_connected(_tab(layout, "both")), name

    def test_shielding_never_adds_reachable_joint_states(self):
        for layout in LAYOUTS.values():
            unshielded = T3.reachable_joint_states(_tab(layout, "none"))
            shielded = T3.reachable_joint_states(_tab(layout, "both"))
            assert shielded <= unshielded


class TestCriterionArithmetic:
    """The pre-registered rule must not be re-interpretable after the fact."""

    def _rows(self, deltas):
        # Baseline 0.0 so that each delta is exactly the float written here and
        # the strictness of the ">" is tested rather than float addition.
        layouts = list(T3.LAYOUTS)
        results = []
        for ly, (d_both, d_po) in zip(layouts, deltas):
            for shield, value in (
                ("none", 0.0),
                ("both", d_both),
                ("pursuer_only", d_po),
            ):
                results.append(
                    {
                        "layout": ly.name,
                        "shield": shield,
                        "nash_value": value,
                        "br": {"br_value": 0.0},
                    }
                )
        return T3.evaluate_criterion(results, layouts)

    def test_threshold_is_strict_and_needs_two_layouts(self):
        v = self._rows([(0.02, 0.0), (0.02, 0.0), (0.5, 0.0)])
        # 0.02 exactly does not exceed 0.02.
        assert v["n_layouts_exceeding_both"] == 1
        assert not v["criterion_met_both"]

    def test_two_layouts_is_enough(self):
        v = self._rows([(0.03, 0.0), (-0.001, 0.0), (0.03, 0.0)])
        assert v["criterion_met_both"]
        assert v["sign_prediction_both"].startswith("positive")

    def test_opposite_signs_are_reported_as_mixed_not_averaged(self):
        v = self._rows([(0.30, 0.0), (-0.30, 0.0), (0.30, 0.0)])
        assert v["criterion_met_both"]
        assert v["sign_prediction_both"].startswith("mixed")

    def test_a_null_is_reported_as_a_null(self):
        v = self._rows([(0.001, 0.0), (-0.002, 0.0), (0.0, 0.0)])
        assert not v["criterion_met_both"]
        assert v["sign_prediction_both"] == "none (criterion not met)"

    def test_pursuer_only_arm_is_scored_separately(self):
        v = self._rows([(0.0, -0.4), (0.0, -0.4), (0.0, -0.4)])
        assert not v["criterion_met_both"]
        assert v["criterion_met_pursuer_only"]
        assert v["sign_prediction_pursuer_only"].startswith("negative")


class TestPreRegisteredConstants:
    def test_threshold_and_layout_count_match_the_spec(self):
        assert T3.SHIFT_THRESHOLD == 0.02
        assert T3.MIN_LAYOUTS == 2
        assert len(T3.LAYOUTS) == 3
        assert {ly.name for ly in T3.LAYOUTS} == {"sparse", "dense", "corridor"}

    def test_every_layout_carries_a_stated_rationale(self):
        for ly in list(T3.LAYOUTS) + [T3.NEG_CONTROL]:
            assert len(ly.rationale) > 80

    def test_hazard_density_is_ordered_sparse_then_dense(self):
        by = {ly.name: len(ly.hazards) for ly in T3.LAYOUTS}
        assert by["sparse"] < by["corridor"] < by["dense"]
