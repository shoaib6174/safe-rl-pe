"""Tests for scripts/pe2_step06_visual_gate.py — the checklist's own maths.

Rendering is not exercised here (it needs the gate JSON and 500+ frames);
what is tested is the two functions the C1 and C2 assertions rest on, plus
the `ShieldOverlay` contract that makes C4 true by construction.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.shield import CBFShield

_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "pe2_step06_visual_gate.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("pe2_step06_visual_gate", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VG = _module()
CFG = EnvConfig()


class TestInterventionClearanceBound:
    """reach = dt*(v_max + d*omega_max)/gamma
    = 0.05*(1.0 + 0.1*2.84)/0.2 = 0.0642/0.2 = 0.321 m.
    """

    def test_boundary_only_bound(self):
        # No obstacles: only the four boundary rows exist, and each can bind
        # at most `reach` metres of clearance out.
        assert VG.intervention_clearance_bound(
            CFG, [], CBFShield(CFG)
        ) == pytest.approx(0.321)

    def test_obstacle_bound(self):
        # r_eff = 0.3 + 0.15 = 0.45; K = 2*reach = 0.642;
        # dist = (0.642 + sqrt(0.642^2 + 4*0.45^2))/2
        #      = (0.642 + sqrt(0.412164 + 0.81))/2 = 0.87375763;
        # bound = 0.87375763 - 0.45 = 0.42375763 m, which exceeds the 0.321
        # boundary figure, so the obstacle row is the binding one.
        bound = VG.intervention_clearance_bound(
            CFG, [{"x": 0.0, "y": 0.0, "radius": 0.3}], CBFShield(CFG)
        )
        assert bound == pytest.approx(0.4237576322, abs=1e-9)

    def test_margin_shifts_the_bound(self):
        # Same obstacle with delta = 0.05: r_eff = 0.50, dist = 0.91517253,
        # bound = 0.41517253 m.
        bound = VG.intervention_clearance_bound(
            CFG, [{"x": 0.0, "y": 0.0, "radius": 0.3}], CBFShield(CFG, margin=0.05)
        )
        assert bound == pytest.approx(0.4151725339, abs=1e-9)

    def test_takes_the_largest_over_all_constraints(self):
        # r=1.0 -> r_eff 1.15 -> bound 0.36496; r=0.3 -> 0.42376. The larger
        # governs, since C1 must not reject an intervention any row explains.
        bound = VG.intervention_clearance_bound(
            CFG,
            [
                {"x": 0.0, "y": 0.0, "radius": 1.0},
                {"x": 3.0, "y": 3.0, "radius": 0.3},
            ],
            CBFShield(CFG),
        )
        assert bound == pytest.approx(0.4237576322, abs=1e-9)


class TestNominalViolatesARow:
    def test_head_on_wall_at_a_feasible_clearance(self):
        # x = 4.49 heading +x -> VCP 4.59, east-wall h = 4.85 - 4.59 = 0.26.
        # The exact one-step check (re-derived from CBFShield.progress) agrees
        # with the linearised estimate here: dt*a_v*v + gamma*h =
        # 0.05*(-1)*1.0 + 0.2*0.26 = +0.002 >= 0, so the row is not violated.
        shield = CBFShield(CFG)
        state = np.array([4.49, 0.0, 0.0])
        assert VG.nominal_violates_a_row(shield, state, [], 1.0, 0.0) is False

    def test_head_on_wall_inside_the_binding_clearance(self):
        # x = 4.54 -> VCP 4.64, h = 0.21: 0.05*(-1)*1.0 + 0.2*0.21 = -0.008,
        # so the linearised estimate already violates and the exact check
        # (verified numerically) agrees.
        shield = CBFShield(CFG)
        state = np.array([4.54, 0.0, 0.0])
        assert VG.nominal_violates_a_row(shield, state, [], 1.0, 0.0) is True

    def test_braking_to_zero_makes_the_same_state_feasible(self):
        # At v = 0, omega = 0 the successor state is unchanged and every row's
        # h can only decay by (1-gamma), which stays >= 0 wherever h > 0 -- so
        # the same state is feasible for a stopped robot.
        shield = CBFShield(CFG)
        state = np.array([4.54, 0.0, 0.0])
        assert VG.nominal_violates_a_row(shield, state, [], 0.0, 0.0) is False

    def test_every_flagged_intervention_has_a_violated_row(self):
        # The C1 assertion in one line, in the direction C1 asserts: on the
        # exact tier the filter never moves the action unless some row is
        # violated. The converse can legitimately fail, because a correction
        # below the shield's INTERVENTION_ATOL (1e-4) is not flagged.
        shield = CBFShield(CFG)
        obstacles = [{"x": 1.4, "y": 0.2, "radius": 0.5}]
        flagged = 0
        for x in np.linspace(-1.0, 0.6, 33):
            state = np.array([float(x), 0.0, 0.0])
            _, info = shield.filter((1.0, 0.0), state, obstacles)
            if info["method"] != "exact" or not info["intervention"]:
                continue
            flagged += 1
            assert VG.nominal_violates_a_row(shield, state, obstacles, 1.0, 0.0), (
                f"intervened at x={x} with every DCBF row satisfied"
            )
        assert flagged > 0, "sweep never intervened — the check would be vacuous"


class TestLongestRun:
    def test_empty_and_all_false(self):
        assert VG.longest_run([]) == 0
        assert VG.longest_run([False, False]) == 0

    def test_hand_counted(self):
        # T T F T T T F T -> runs of 2, 3, 1 -> longest 3.
        assert VG.longest_run([True, True, False, True, True, True, False, True]) == 3

    def test_run_at_the_end_is_counted(self):
        assert VG.longest_run([False, True, True, True]) == 3


class TestShieldOverlayContract:
    def test_draws_nothing_before_the_first_step(self):
        # `live` is empty until the caller fills it; the overlay must be a
        # no-op then rather than inventing a state.
        overlay = VG.ShieldOverlay("pursuer", CBFShield(CFG), {})
        overlay(object(), object())  # would raise if it touched either

    def test_arc_is_a_straight_line_at_zero_omega(self):
        # 1.0 m/s for 1.0 s from the origin heading +x, 20 substeps: the last
        # point is at x = 1.0 (20 * 0.05 m), y = 0.
        pts = VG.ShieldOverlay._arc((0.0, 0.0, 0.0), 1.0, 0.0, 1.0, 20)
        assert pts.shape == (21, 2)
        assert pts[-1][0] == pytest.approx(1.0)
        assert pts[-1][1] == pytest.approx(0.0)

    def test_arc_is_stationary_at_zero_speed(self):
        pts = VG.ShieldOverlay._arc((2.0, -1.0, 0.3), 0.0, 2.0, 1.0, 20)
        assert np.allclose(pts, np.array([[2.0, -1.0]] * 21))
