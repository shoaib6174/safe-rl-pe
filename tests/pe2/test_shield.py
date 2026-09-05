"""Tests for pe2.shield + pe2.metrics — Step 6 safety shield."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.env import OUTCOME_CAPTURE, PursuitEvasionEnv
from pe2.metrics import DEADLOCK_STEPS, ShieldMetrics, summarize_shield
from pe2.scripted import GreedyEvader, GreedyPursuer, run_duel, run_shielded_duel
from pe2.shield import (
    STALL_V_FRAC,
    CBFShield,
    NoOpShield,
    _boundary_constraints,
    _obstacle_constraint,
    compute_vcp,
)

CFG = EnvConfig()
X_LIM = CFG.arena_size / 2.0 - CFG.agent_radius  # 4.85


class TestVCPMath:
    def test_vcp_position(self):
        qx, qy = compute_vcp(1.0, 2.0, np.pi / 2, 0.1)
        assert qx == pytest.approx(1.0)
        assert qy == pytest.approx(2.1)

    def test_obstacle_constraint_head_on(self):
        # State at origin heading +x, d=0.1 -> q=(0.1, 0).
        # Obstacle at (1.1, 0), r_eff=0.5: dx=-1.0, h = 1 - 0.25 = 0.75.
        c = _obstacle_constraint(np.array([0.0, 0.0, 0.0]), 1.1, 0.0, 0.5, 0.1)
        assert c.h == pytest.approx(0.75)
        assert c.a_v == pytest.approx(-2.0)
        assert c.a_omega == pytest.approx(0.0, abs=1e-12)
        assert c.kind == "obstacle"

    def test_obstacle_constraint_off_axis_steering_gradient(self):
        # Obstacle at (1.1, 0.3): dy=-0.3 -> a_omega = 2*dy*d = -0.06.
        c = _obstacle_constraint(np.array([0.0, 0.0, 0.0]), 1.1, 0.3, 0.5, 0.1)
        assert c.a_v == pytest.approx(-2.0)
        assert c.a_omega == pytest.approx(-0.06)

    def test_obstacle_clearance_is_metres_not_metres_squared(self):
        # Same geometry as test_obstacle_constraint_head_on: q=(0.1, 0),
        # obstacle at (1.1, 0), r_eff=0.5 -> ||q - p_obs|| = 1.0, so the
        # signed clearance is 1.0 - 0.5 = 0.5 m while h = 0.75 m^2.
        c = _obstacle_constraint(np.array([0.0, 0.0, 0.0]), 1.1, 0.0, 0.5, 0.1)
        assert c.h == pytest.approx(0.75)
        assert c.clearance == pytest.approx(0.5)

    def test_obstacle_clearance_negative_inside_effective_radius(self):
        # q=(0.1, 0), obstacle at (0.3, 0), r_eff=0.5: dist 0.2 ->
        # h = 0.04 - 0.25 = -0.21 m^2, clearance = 0.2 - 0.5 = -0.3 m.
        c = _obstacle_constraint(np.array([0.0, 0.0, 0.0]), 0.3, 0.0, 0.5, 0.1)
        assert c.h == pytest.approx(-0.21)
        assert c.clearance == pytest.approx(-0.3)

    def test_boundary_clearance_equals_barrier(self):
        # A boundary barrier is already a distance, so it passes through.
        cons = _boundary_constraints(np.array([0.0, 0.0, 0.0]), X_LIM, X_LIM, 0.1)
        for c in cons:
            assert c.clearance == pytest.approx(c.h)

    def test_boundary_constraints_headings(self):
        cons = _boundary_constraints(np.array([0.0, 0.0, 0.0]), X_LIM, X_LIM, 0.1)
        assert len(cons) == 4
        # +x wall: h = X_LIM - qx = 4.75, a_v = -cos(0) = -1, a_omega = +d*sin(0) = 0
        assert cons[0].h == pytest.approx(X_LIM - 0.1)
        assert cons[0].a_v == pytest.approx(-1.0)
        assert cons[0].a_omega == pytest.approx(0.0, abs=1e-12)
        # +y wall: a_v = -sin(0) = 0, a_omega = -d*cos(0) = -0.1
        assert cons[2].a_v == pytest.approx(0.0, abs=1e-12)
        assert cons[2].a_omega == pytest.approx(-0.1)

    def test_body_obstacle_row_is_the_env_contact_predicate(self):
        from pe2.shield import _body_obstacle_constraint

        # Body at origin heading +x; obstacle at (1.0, 0), r_eff 0.5:
        # h = hypot(1, 0) - 0.5 = 0.5 m, a_v = cos(angle to outward normal)
        # = -1 (moving straight at it), a_omega = 0 (body row ignores heading).
        c = _body_obstacle_constraint(np.array([0.0, 0.0, 0.0]), 1.0, 0.0, 0.5)
        assert c.h == pytest.approx(0.5)
        assert c.clearance == c.h
        assert c.a_v == pytest.approx(-1.0)
        assert c.a_omega == 0.0
        assert c.kind == "obstacle" and c.hard is True and c.scale == 1.0

    def test_body_wall_rows_ignore_heading(self):
        from pe2.shield import _body_wall_constraints

        rows = _body_wall_constraints(np.array([4.0, -1.0, 1.0]), 4.85, 4.85)
        assert [r.h for r in rows] == pytest.approx([0.85, 8.85, 5.85, 3.85])
        assert all(r.a_omega == 0.0 and r.hard and r.scale == 1.0 for r in rows)
        assert rows[0].a_v == pytest.approx(-np.cos(1.0))

    def test_vcp_rows_are_soft_with_obstacle_scale(self):
        c = _obstacle_constraint(np.array([0.0, 0.0, 0.0]), 1.1, 0.0, 0.5, 0.1)
        assert c.hard is False and c.scale == pytest.approx(1.0)  # 2 * r_eff
        w = _boundary_constraints(np.array([0.0, 0.0, 0.0]), 4.85, 4.85, 0.1)
        assert all((not r.hard) and r.scale == 1.0 for r in w)

    def test_constraints_row_order_and_count(self):
        shield = CBFShield(CFG)
        obstacles = [
            {"x": 2.0, "y": 0.0, "radius": 0.4},
            {"x": -2.0, "y": 1.0, "radius": 0.3},
        ]
        cons = shield.constraints(np.array([0.0, 0.0, 0.0]), obstacles)
        assert len(cons) == 12
        assert [c.hard for c in cons] == [True] * 4 + [False] * 4 + [
            True,
            False,
            True,
            False,
        ]
        assert [c.kind for c in cons] == ["boundary"] * 8 + ["obstacle"] * 4


def _linear_optimum(v_nom, w_nom, rows, v_max, omega_max, w_v, w_omega):
    """Test helper: exact optimum of the LINEAR program over the candidates.

    The shield itself judges candidates by the exact one-step barrier
    (Task 4); this helper only checks that the enumeration still contains
    the vertex the convex 2-D QP would pick (D85).
    """
    from pe2.shield import _qp_candidates

    cands = _qp_candidates(v_nom, w_nom, rows, v_max, omega_max, w_v, w_omega)
    ok = [
        u
        for u in cands
        if all(c0 * u[0] + c1 * u[1] >= c2 - 1e-12 for c0, c1, c2 in rows)
    ]
    if not ok:
        return None
    return min(
        ok, key=lambda u: w_v * (u[0] - v_nom) ** 2 + w_omega * (u[1] - w_nom) ** 2
    )


class TestCandidates:
    def test_nominal_and_zero_are_always_present(self):
        from pe2.shield import _qp_candidates

        cands = _qp_candidates(0.5, 0.5, [], 1.0, 2.84, 150.0, 1.0)
        assert np.array_equal(cands[0], [0.5, 0.5])
        assert np.array_equal(cands[1], [0.0, 0.0])

    def test_every_candidate_is_inside_the_box(self):
        from pe2.shield import _qp_candidates

        rows = [(-0.06, -0.003, -0.04), (1.0, 1.0, 2.0), (1.0, -1.0, 2.0)]
        cands = _qp_candidates(3.0, -9.0, rows, 1.0, 2.84, 150.0, 1.0)
        assert cands.dtype == np.float64 and cands.shape[1] == 2
        assert (cands[:, 0] >= 0.0).all() and (cands[:, 0] <= 1.0).all()
        assert (np.abs(cands[:, 1]) <= 2.84).all()

    def test_single_constraint_hand_computed(self):
        # dt*a_v = -0.05, dt*a_omega = 0, c2 = -0.01 -> v <= 0.2.
        u = _linear_optimum(1.0, 0.0, [(-0.05, 0.0, -0.01)], 1.0, 2.84, 150.0, 1.0)
        assert u[0] == pytest.approx(0.2)
        assert u[1] == pytest.approx(0.0, abs=1e-9)

    def test_infeasible_rows_leave_no_linear_candidate(self):
        assert (
            _linear_optimum(
                0.0, 0.0, [(1.0, 0.0, 10.0), (-1.0, 0.0, 10.0)], 1.0, 2.84, 150.0, 1.0
            )
            is None
        )

    def test_vertex_optimum_two_constraints(self):
        u = _linear_optimum(
            0.0, 0.0, [(1.0, 1.0, 2.0), (1.0, -1.0, 2.0)], 5.0, 5.0, 1.0, 1.0
        )
        assert u[0] == pytest.approx(2.0)
        assert u[1] == pytest.approx(0.0, abs=1e-9)

    def test_anisotropic_weight_prefers_steering(self):
        u = _linear_optimum(1.0, 0.0, [(-0.06, -0.003, -0.04)], 1.0, 2.84, 150.0, 1.0)
        assert -0.06 * u[0] - 0.003 * u[1] >= -0.04 - 1e-9
        assert abs(u[1]) > 1.0 and u[0] > 0.7

    def test_enumeration_is_deterministic(self):
        from pe2.shield import _qp_candidates

        rows = [(-0.06, -0.003, -0.04), (1.0, 1.0, 2.0)]
        a = _qp_candidates(0.7, 0.1, rows, 1.0, 2.84, 150.0, 1.0)
        b = _qp_candidates(0.7, 0.1, rows, 1.0, 2.84, 150.0, 1.0)
        assert np.array_equal(a, b)


class TestCBFShieldFilter:
    def test_noop_when_safe(self):
        shield = CBFShield(CFG)
        obstacles = [{"x": 4.0, "y": 4.0, "radius": 0.3}]
        u, info = shield.filter((0.5, 0.5), np.array([0.0, 0.0, 0.0]), obstacles)
        assert np.allclose(u, [0.5, 0.5])
        assert info["method"] == "exact"
        assert not info["intervention"]

    def test_wall_approach_brakes_to_decay_rate(self):
        # At x=4.7 heading +x: qx=4.8, h=0.05 -> v <= gamma*h/(dt*|a_v|) = 0.2.
        shield = CBFShield(CFG)
        u, info = shield.filter((1.0, 0.0), np.array([4.7, 0.0, 0.0]), [])
        assert info["method"] == "exact"
        assert info["intervention"]
        assert u[0] == pytest.approx(0.2)
        assert u[1] == pytest.approx(0.0, abs=1e-9)

    def test_obstacle_approach_constraint_satisfied(self):
        shield = CBFShield(CFG)
        obstacles = [{"x": 0.7, "y": 0.3, "radius": 0.35}]  # r_eff = 0.5
        state = np.array([0.0, 0.0, 0.0])
        u, info = shield.filter((1.0, 0.0), state, obstacles)
        assert info["method"] == "exact"
        assert info["intervention"]
        # Executed action must satisfy every linearized DCBF constraint.
        for c in shield.constraints(state, obstacles):
            assert (
                CFG.dt * c.a_v * u[0] + CFG.dt * c.a_omega * u[1] + shield.gamma * c.h
                >= -1e-9
            )

    def test_relaxed_tier_when_obstacle_infeasible(self):
        # Already penetrating the effective radius dead ahead: the obstacle
        # constraint demands v < 0, impossible -> drop to boundary-only.
        shield = CBFShield(CFG)
        obstacles = [{"x": 0.5, "y": 0.0, "radius": 0.35}]  # r_eff 0.5, h < 0
        u, info = shield.filter((1.0, 0.0), np.array([0.0, 0.0, 0.0]), obstacles)
        assert info["method"] == "relaxed"
        assert u[0] >= -1e-9  # box bounds still respected

    def test_backup_tier_when_wall_infeasible(self):
        # Outside the wall limit heading +x: even boundary rows are infeasible.
        shield = CBFShield(CFG)
        u, info = shield.filter((1.0, 0.0), np.array([5.0, 0.0, 0.0]), [])
        assert info["method"] == "backup"
        assert u[0] == pytest.approx(0.1 * CFG.v_max)
        assert abs(u[1]) == pytest.approx(CFG.omega_max)

    def test_backup_turns_away_from_nearest_danger(self):
        shield = CBFShield(CFG)
        # Wall to the east, heading east -> must rotate toward west.
        u = shield.backup(np.array([4.8, 0.0, 0.0]), [])
        assert u[0] == pytest.approx(0.1 * CFG.v_max)
        assert u[1] > 0  # +omega turns toward pi

    def test_backup_head_on_north_wall_tie_breaks_ccw(self):
        # Heading +y at y=4.8: nearest danger is the north wall (angle pi/2),
        # away = -pi/2, error = wrap(-pi/2 - pi/2) = wrap(-pi) = -pi -> an exact
        # tie. Convention: ties turn CCW (+omega).
        shield = CBFShield(CFG)
        u = shield.backup(np.array([0.0, 4.8, np.pi / 2]), [])
        assert u[0] == pytest.approx(0.1 * CFG.v_max)
        assert u[1] == pytest.approx(CFG.omega_max)

    def test_backup_turns_the_short_way_when_not_a_tie(self):
        # Wall east, heading 0.5 rad north of east: away = pi, error =
        # wrap(pi - 0.5) = +2.64 -> CCW. Heading -0.5: error = wrap(pi + 0.5)
        # = -2.64 -> CW. Both saturate at omega_max (3 * 2.64 > 2.84).
        shield = CBFShield(CFG)
        u_ccw = shield.backup(np.array([4.8, 0.0, 0.5]), [])
        u_cw = shield.backup(np.array([4.8, 0.0, -0.5]), [])
        assert u_ccw[1] == pytest.approx(CFG.omega_max)
        assert u_cw[1] == pytest.approx(-CFG.omega_max)

    def test_no_inter_agent_constraint(self):
        # The shield takes obstacles only; constraints are obstacle+boundary,
        # so capture (which needs contact) can never be blocked by design.
        shield = CBFShield(CFG)
        obstacles = [{"x": 1.0, "y": 0.0, "radius": 0.3}]
        cons = shield.constraints(np.array([0.0, 0.0, 0.0]), obstacles)
        assert info_kinds(cons) == {"obstacle", "boundary"}
        assert len(cons) == 4 + len(obstacles)

    def test_deterministic(self):
        shield = CBFShield(CFG)
        obstacles = [{"x": 0.7, "y": 0.3, "radius": 0.35}]
        state = np.array([0.0, 0.0, 0.0])
        u1, _ = shield.filter((1.0, 0.2), state, obstacles)
        u2, _ = shield.filter((1.0, 0.2), state, obstacles)
        assert np.array_equal(u1, u2)

    def test_gamma_default_pinned(self):
        assert CBFShield(CFG).gamma == 0.2  # D31

    def test_filter_info_reports_feasibility_and_speeds(self):
        shield = CBFShield(CFG)
        # Safe state: exact tier, no change -> feasible, v_exec == v_nom.
        _, info = shield.filter((0.5, 0.0), np.array([0.0, 0.0, 0.0]), [])
        assert info["feasible"] is True
        assert info["v_nom"] == pytest.approx(0.5)
        assert info["v_exec"] == pytest.approx(0.5)
        assert info["stalled"] is False
        # Outside the wall: backup tier -> not feasible.
        _, info = shield.filter((1.0, 0.0), np.array([5.0, 0.0, 0.0]), [])
        assert info["feasible"] is False
        assert info["method"] == "backup"

    def test_filter_flags_stall_when_shield_stops_a_moving_nominal(self):
        # The +x wall barrier is h = x_lim - qx = 4.85 - (x + 0.1) = 4.75 - x.
        # At x = 4.749: h = 0.001, and dt*a_v*v + gamma*h >= 0 with a_v = -1
        # gives v <= gamma*h/dt = 0.2 * 0.001 / 0.05 = 0.004. That is below
        # 0.05 * v_max = 0.05 while the nominal 1.0 is above 0.5 * v_max,
        # so the step is a stall on the exact tier.
        shield = CBFShield(CFG)
        u, info = shield.filter((1.0, 0.0), np.array([4.749, 0.0, 0.0]), [])
        assert info["method"] == "exact"
        assert u[0] == pytest.approx(0.004, abs=1e-6)
        assert info["stalled"] is True

    def test_filter_not_stalled_when_nominal_is_slow(self):
        # Same wall squeeze, but the nominal itself is slow (0.2 < 0.5 * v_max):
        # braking a slow nominal is not a stall.
        shield = CBFShield(CFG)
        _, info = shield.filter((0.2, 0.0), np.array([4.749, 0.0, 0.0]), [])
        assert info["stalled"] is False

    def test_immobile_is_flagged_even_when_the_nominal_is_slow(self):
        # A policy that lowers its OWN commanded speed hides from `stalled`:
        # nominal 0.4 < 0.5 * v_max, so stalled is False by definition, but
        # the shield still holds the executed speed at 0.004 <= 0.05 * v_max.
        # `immobile` is the nominal-independent flag that catches it.
        shield = CBFShield(CFG)
        _, info = shield.filter((0.4, 0.0), np.array([4.749, 0.0, 0.0]), [])
        assert info["v_exec"] == pytest.approx(0.004, abs=1e-6)
        assert info["stalled"] is False
        assert info["immobile"] is True

    def test_immobile_and_stalled_agree_when_the_nominal_is_fast(self):
        shield = CBFShield(CFG)
        _, info = shield.filter((1.0, 0.0), np.array([4.749, 0.0, 0.0]), [])
        assert info["stalled"] is True
        assert info["immobile"] is True
        # And neither fires in open space.
        _, free = shield.filter((0.5, 0.0), np.array([0.0, 0.0, 0.0]), [])
        assert free["stalled"] is False
        assert free["immobile"] is False

    def test_backup_speed_sits_above_the_stall_band_by_construction(self):
        # The two constants are not independent: the backup controller returns
        # 0.1 * v_max, which is strictly above STALL_V_FRAC * v_max = 0.05, so
        # a backup-tier step can never be flagged stalled or immobile. That is
        # exactly why tier-3 immobility needs its own counter.
        assert 0.1 > STALL_V_FRAC
        shield = CBFShield(CFG)
        _, info = shield.filter((1.0, 0.0), np.array([5.0, 0.0, 0.0]), [])
        assert info["method"] == "backup"
        assert info["v_exec"] == pytest.approx(0.1 * CFG.v_max)
        assert info["stalled"] is False
        assert info["immobile"] is False

    def test_backup_honours_the_effective_speed_cap(self):
        # Called with a cap below 0.1 * cfg.v_max, the backup controller must
        # not return an action above the cap it was given: 0.1 * 0.5 = 0.05.
        shield = CBFShield(CFG)
        u = shield.backup(np.array([5.0, 0.0, 0.0]), [], v_max=0.5)
        assert u[0] == pytest.approx(0.05)
        # Same through filter, where the cap comes from the caller's role.
        u, info = shield.filter((1.0, 0.0), np.array([5.0, 0.0, 0.0]), [], v_max=0.5)
        assert info["method"] == "backup"
        assert u[0] == pytest.approx(0.05)
        assert u[0] <= 0.5

    def test_relaxed_tier_through_filter_reports_infeasible(self):
        # The reviewer's probe. State x = 0.05 heading +x with an obstacle at
        # the origin, radius 0.3 -> r_eff = 0.45, q = (0.15, 0), dist 0.15:
        #   h        = 0.15^2 - 0.45^2 = 0.0225 - 0.2025 = -0.18 m^2
        #   clearance = 0.15 - 0.45                      = -0.30 m
        # The obstacle row demands v < 0, so the exact program is infeasible
        # and the boundary-only program is not (walls are 4.7 m away), which
        # is the RELAXED tier. It returns the nominal unmodified.
        shield = CBFShield(CFG)
        u, info = shield.filter(
            (1.0, 0.0),
            np.array([0.05, 0.0, 0.0]),
            [{"x": 0.0, "y": 0.0, "radius": 0.3}],
        )
        assert info["method"] == "relaxed"
        assert info["feasible"] is False
        assert info["stalled"] is False
        assert info["immobile"] is False
        assert np.allclose(u, [1.0, 0.0])
        assert info["min_h_obstacle"] == pytest.approx(-0.18)
        assert info["min_h_boundary"] == pytest.approx(4.7)
        assert info["min_clearance"] == pytest.approx(-0.30)
        # An intervention-free step that is among the LEAST safe: on the
        # relaxed tier the obstacle row is simply dropped.
        assert info["intervention"] is False

    def test_relaxed_tier_info_is_classified_by_the_accumulator(self):
        # Cross-module contract for a non-exact tier: the real info dict from
        # the probe above must land as infeasible-but-not-backup.
        shield = CBFShield(CFG)
        _, info = shield.filter(
            (1.0, 0.0),
            np.array([0.05, 0.0, 0.0]),
            [{"x": 0.0, "y": 0.0, "radius": 0.3}],
        )
        m = ShieldMetrics()
        m.record(info, violation=False)
        s = m.summary()
        assert s["n_infeasible"] == 1
        assert s["n_backup"] == 0
        assert s["infeasibility_rate"] == pytest.approx(1.0)
        assert s["backup_rate"] == 0.0
        assert s["tier_histogram"] == {
            "exact": 0,
            "relaxed": 1,
            "backup": 0,
            "none": 0,
        }
        # The least-safe step of the episode logs no intervention at all.
        assert s["intervention_rate"] == 0.0
        assert s["min_clearance_floor"] == pytest.approx(-0.30)

    def test_equal_clearance_wall_and_obstacle_report_equal_clearance(self):
        # The mixed-unit trap: 0.02 m of clearance from a wall and 0.02 m from
        # a radius-1.0 obstacle must report the SAME min_clearance, while min_h
        # differs by more than 2x because the obstacle barrier is in m^2.
        shield = CBFShield(CFG)
        # Wall: x_lim = 4.85, q = x + 0.1. x = 4.73 -> q = 4.83, h = 0.02 m.
        _, wall = shield.filter((0.1, 0.0), np.array([4.73, 0.0, 0.0]), [])
        # Obstacle: q = (0.1, 0), r_eff = 1.0 + 0.15 = 1.15. Centre at
        # (0.1 + 1.17, 0) -> dist 1.17, h = 1.17^2 - 1.15^2 = 0.0464 m^2,
        # clearance = 1.17 - 1.15 = 0.02 m. Boundaries here are >= 4.75 m.
        _, obst = shield.filter(
            (0.1, 0.0),
            np.array([0.0, 0.0, 0.0]),
            [{"x": 1.27, "y": 0.0, "radius": 1.0}],
        )
        assert wall["min_h"] == pytest.approx(0.02)
        assert obst["min_h"] == pytest.approx(0.0464)
        # The reportable number agrees at identical physical clearance.
        assert wall["min_clearance"] == pytest.approx(0.02)
        assert obst["min_clearance"] == pytest.approx(0.02)
        # Per-kind barrier floors stay available for filter-internal diagnostics.
        assert wall["min_h_obstacle"] is None
        assert wall["min_h_boundary"] == pytest.approx(0.02)
        assert obst["min_h_obstacle"] == pytest.approx(0.0464)
        assert obst["min_h_boundary"] == pytest.approx(X_LIM - 0.1)


def info_kinds(cons):
    return {c.kind for c in cons}


class TestNoOpShield:
    def test_returns_the_nominal_unchanged_and_reports_the_none_tier(self):
        shield = NoOpShield(CFG)
        # A state where CBFShield would brake hard (h = 0.001 -> v <= 0.004):
        # the no-op filter must pass 1.0 straight through.
        u, info = shield.filter((1.0, 0.3), np.array([4.749, 0.0, 0.0]), [])
        assert u[0] == pytest.approx(1.0)
        assert u[1] == pytest.approx(0.3)
        assert info["method"] == "none"
        assert info["intervention"] is False
        assert info["feasible"] is True  # no QP was solved, so nothing failed
        assert info["v_nom"] == pytest.approx(1.0)
        assert info["v_exec"] == pytest.approx(1.0)
        assert info["stalled"] is False
        assert info["immobile"] is False

    def test_still_measures_the_barriers(self):
        # The barriers are a property of the STATE, not of the filter, so the
        # unshielded control reports a real clearance floor to compare against.
        # x = 4.73 -> q = 4.83, h = x_lim - q = 0.02 m.
        shield = NoOpShield(CFG)
        _, info = shield.filter((1.0, 0.0), np.array([4.73, 0.0, 0.0]), [])
        assert info["min_h"] == pytest.approx(0.02)
        assert info["min_clearance"] == pytest.approx(0.02)
        assert info["n_constraints"] == 4

    def test_info_keys_match_the_real_shield(self):
        # The two arms are compared step by step, so their info dicts must
        # carry exactly the same keys.
        state = np.array([1.0, 0.5, 0.3])
        obstacles = [{"x": 2.0, "y": 0.0, "radius": 0.4}]
        _, real = CBFShield(CFG).filter((1.0, 0.0), state, obstacles)
        _, noop = NoOpShield(CFG).filter((1.0, 0.0), state, obstacles)
        assert set(real) == set(noop)

    def test_respects_the_action_box(self):
        # The env clips too, but the filter's contract is a valid action.
        shield = NoOpShield(CFG)
        u, info = shield.filter((5.0, 99.0), np.array([0.0, 0.0, 0.0]), [])
        assert u[0] == pytest.approx(CFG.v_max)
        assert u[1] == pytest.approx(CFG.omega_max)
        assert info["v_nom"] == pytest.approx(CFG.v_max)

    def test_metrics_classify_a_none_step_distinctly(self):
        # The negative control must not look like an all-exact measurement.
        shield = NoOpShield(CFG)
        _, info = shield.filter((1.0, 0.0), np.array([0.0, 0.0, 0.0]), [])
        m = ShieldMetrics()
        m.record(info, violation=False)
        s = m.summary()
        assert s["tier_histogram"] == {
            "exact": 0,
            "relaxed": 0,
            "backup": 0,
            "none": 1,
        }
        assert s["infeasibility_rate"] == 0.0
        assert s["method_counts"] == {"none": 1}


class TestShieldedEnvIntegration:
    def test_zero_contacts_driving_straight_at_obstacle(self):
        """Shielded robot ramming an obstacle at v_max for 200 steps must
        never trigger the env's contact resolution."""
        cfg = EnvConfig(n_obstacles=1)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = [{"x": 2.0, "y": 0.0, "radius": 0.5}]
        env.pursuer_pose = np.array([0.0, 0.0, 0.0])
        env.evader_pose = np.array([-4.0, -4.0, 0.0])
        shield = CBFShield(cfg)
        contacts = 0
        for _ in range(200):
            safe, _ = shield.filter((cfg.v_max, 0.0), env.pursuer_pose, env.obstacles)
            _, _, terminated, truncated, info = env.step([safe[0], safe[1], 0.0, 0.0])
            contacts += int(
                info["pursuer_obstacle_contact"] or info["pursuer_wall_contact"]
            )
            if terminated or truncated:
                break
        assert contacts == 0

    def test_shielded_pursuer_still_captures(self):
        """Obstacle-free duel: shielded GreedyPursuer must still capture —
        the shield never blocks the inter-agent approach."""
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=0)
        env.obstacles = []
        shield = CBFShield(cfg)
        pursuer, evader = GreedyPursuer(), GreedyEvader()
        obs = env._get_obs()
        for _ in range(cfg.max_steps):
            v_p, w_p = pursuer.act(obs["pursuer"], env)
            safe_p, _ = shield.filter((v_p, w_p), env.pursuer_pose, env.obstacles)
            v_e, w_e = evader.act(obs["evader"], env)
            obs, _, terminated, truncated, info = env.step(
                [safe_p[0], safe_p[1], v_e, w_e]
            )
            if terminated or truncated:
                break
        assert info["outcome"] == OUTCOME_CAPTURE


def _info(method="exact", intervention=False, min_h=1.0, stalled=False, **over):
    """A CBFShield.filter info dict with defaults; `over` overrides any key.

    Defaults keep `min_clearance` and `min_h_boundary` equal to `min_h` so the
    hand-computed barrier assertions below stay readable; the clearance tests
    override them explicitly.
    """
    info = {
        "method": method,
        "feasible": method == "exact",
        "intervention": intervention,
        "min_h": min_h,
        "min_clearance": min_h,
        "min_h_obstacle": None,
        "min_h_boundary": min_h,
        "stalled": stalled,
        "immobile": stalled,
    }
    info.update(over)
    return info


class TestShieldMetrics:
    def test_rates_hand_computed(self):
        m = ShieldMetrics()
        m.record(_info(method="exact", intervention=True, min_h=0.5), violation=False)
        m.record(_info(method="exact", intervention=False, min_h=0.3), violation=True)
        m.record(_info(method="backup", intervention=True, min_h=0.1), violation=False)
        s = m.summary()
        assert s["n_steps"] == 3
        assert s["violation_rate"] == pytest.approx(1 / 3)
        assert s["intervention_rate"] == pytest.approx(2 / 3)
        assert s["method_counts"] == {"exact": 2, "backup": 1}
        assert s["min_h_floor"] == pytest.approx(0.1)
        assert s["min_h_mean"] == pytest.approx(0.3)

    def test_empty_summary(self):
        s = ShieldMetrics().summary()
        assert s["n_steps"] == 0
        assert s["violation_rate"] == 0.0
        assert s["min_h_floor"] is None
        assert s["infeasibility_rate"] == 0.0
        assert s["tier_histogram"] == {
            "exact": 0,
            "relaxed": 0,
            "backup": 0,
            "none": 0,
        }
        assert s["longest_stall_run"] == 0
        assert s["deadlock"] is False

    def test_infeasibility_and_tier_histogram(self):
        # 4 steps: exact, relaxed, backup, exact -> infeasible = 2/4,
        # backup = 1/4, histogram exact 2 / relaxed 1 / backup 1.
        m = ShieldMetrics()
        for method in ("exact", "relaxed", "backup", "exact"):
            m.record(_info(method=method), violation=False)
        s = m.summary()
        assert s["infeasibility_rate"] == pytest.approx(0.5)
        assert s["backup_rate"] == pytest.approx(0.25)
        assert s["tier_histogram"] == {
            "exact": 2,
            "relaxed": 1,
            "backup": 1,
            "none": 0,
        }

    def test_integer_counts_hand_computed(self):
        # 5 steps:
        #   1 exact,   intervention
        #   2 exact,   no intervention
        #   3 relaxed, intervention
        #   4 backup,  intervention
        #   5 exact,   no intervention, stalled + immobile
        # -> interventions 3, infeasible 2, backup 1, stalled 1, immobile 1.
        m = ShieldMetrics()
        m.record(_info(method="exact", intervention=True), violation=False)
        m.record(_info(method="exact"), violation=False)
        m.record(_info(method="relaxed", intervention=True), violation=False)
        m.record(_info(method="backup", intervention=True), violation=False)
        m.record(_info(method="exact", stalled=True), violation=False)
        s = m.summary()
        assert s["n_interventions"] == 3
        assert s["n_infeasible"] == 2
        assert s["n_backup"] == 1
        assert s["n_stalled"] == 1
        assert s["n_immobile"] == 1
        # The counts and the rates must agree, being the same numbers.
        assert s["n_interventions"] / s["n_steps"] == pytest.approx(
            s["intervention_rate"]
        )
        assert s["n_infeasible"] / s["n_steps"] == pytest.approx(
            s["infeasibility_rate"]
        )

    def test_unknown_method_is_refused(self):
        # A fourth tier that only half-registered would vanish from the
        # histogram while still counting toward the infeasibility rate.
        m = ShieldMetrics()
        with pytest.raises(ValueError, match="TIERS"):
            m.record(_info(method="feasible-with-margin"), violation=False)

    def test_rejected_record_leaves_the_accumulator_unchanged(self):
        # record() used to increment n_steps/n_violations/n_interventions
        # before checking the tier, so a caller that caught the ValueError and
        # carried on was left with a histogram that no longer summed to
        # n_steps. Validate first: a rejected record must be a total no-op.
        # Record one real step first so the accumulator is non-empty and any
        # stray increment would show up in the comparison.
        m = ShieldMetrics()
        m.record(_info(method="exact", intervention=True), violation=True)
        before = m.summary()
        with pytest.raises(ValueError, match="TIERS"):
            m.record(_info(method="feasible-with-margin"), violation=True)
        assert m.summary() == before

    def test_tier_histogram_sums_to_step_count_across_real_tiers(self):
        # Three real filter calls, one per QP tier, fed through the
        # accumulator: the histogram must account for every step.
        shield = CBFShield(CFG)
        states = [
            ((1.0, 0.0), np.array([0.0, 0.0, 0.0]), [], "exact"),
            (
                (1.0, 0.0),
                np.array([0.05, 0.0, 0.0]),
                [{"x": 0.0, "y": 0.0, "radius": 0.3}],
                "relaxed",
            ),
            ((1.0, 0.0), np.array([5.0, 0.0, 0.0]), [], "backup"),
        ]
        m = ShieldMetrics()
        for action, state, obstacles, expected in states:
            _, info = shield.filter(action, state, obstacles)
            assert info["method"] == expected  # the run really is multi-tier
            m.record(info, violation=False)
        s = m.summary()
        assert s["n_steps"] == 3
        assert sum(s["tier_histogram"].values()) == s["n_steps"]
        assert s["tier_histogram"] == {
            "exact": 1,
            "relaxed": 1,
            "backup": 1,
            "none": 0,
        }
        assert s["n_infeasible"] == 2
        assert s["n_backup"] == 1

    def test_clearance_and_per_kind_barrier_floors(self):
        # Three steps.
        #   step 1: min_h 0.9 (obstacle 0.9, boundary 4.0), clearance 0.30
        #   step 2: min_h 0.05 (no obstacle, boundary 0.05), clearance 0.05
        #   step 3: min_h 0.30 (obstacle 0.30, boundary 1.0), clearance 0.12
        # -> min_h_floor 0.05, min_clearance_floor 0.05,
        #    min_h_obstacle_floor 0.30, min_h_boundary_floor 0.05.
        m = ShieldMetrics()
        m.record(
            _info(
                min_h=0.9, min_clearance=0.30, min_h_obstacle=0.9, min_h_boundary=4.0
            ),
            violation=False,
        )
        m.record(
            _info(
                min_h=0.05, min_clearance=0.05, min_h_obstacle=None, min_h_boundary=0.05
            ),
            violation=False,
        )
        m.record(
            _info(
                min_h=0.30, min_clearance=0.12, min_h_obstacle=0.30, min_h_boundary=1.0
            ),
            violation=False,
        )
        s = m.summary()
        assert s["min_h_floor"] == pytest.approx(0.05)
        assert s["min_clearance_floor"] == pytest.approx(0.05)
        assert s["min_h_obstacle_floor"] == pytest.approx(0.30)
        assert s["min_h_boundary_floor"] == pytest.approx(0.05)

    def test_obstacle_floor_is_none_when_no_obstacle_was_ever_constrained(self):
        m = ShieldMetrics()
        m.record(
            _info(min_h=1.0, min_clearance=1.0, min_h_boundary=1.0), violation=False
        )
        s = m.summary()
        assert s["min_h_obstacle_floor"] is None
        assert s["min_h_boundary_floor"] == pytest.approx(1.0)

    def test_empty_summary_clearance_floors_are_none(self):
        s = ShieldMetrics().summary()
        assert s["min_clearance_floor"] is None
        assert s["min_h_obstacle_floor"] is None
        assert s["min_h_boundary_floor"] is None

    def test_stall_runs_and_deadlock_threshold(self):
        # Pattern: 3 stalled, 1 free, 5 stalled -> longest run 5, stall rate 8/9,
        # deadlock False (5 < 40).
        m = ShieldMetrics()
        for stalled in [True, True, True, False, True, True, True, True, True]:
            m.record(_info(stalled=stalled), violation=False)
        s = m.summary()
        assert s["stall_step_rate"] == pytest.approx(8 / 9)
        assert s["longest_stall_run"] == 5
        assert s["deadlock"] is False

    def test_backup_tier_pinning_has_its_own_counter(self):
        # DEADLOCK_STEPS consecutive backup-tier steps: the robot crawls at
        # 0.1 * v_max spinning at omega_max, so it is neither stalled nor
        # immobile, yet it is permanently stuck. `pinned` must fire while
        # `deadlock` stays False.
        m = ShieldMetrics()
        for _ in range(DEADLOCK_STEPS):
            m.record(
                _info(method="backup", stalled=False, immobile=False), violation=False
            )
        s = m.summary()
        assert s["longest_backup_run"] == DEADLOCK_STEPS
        assert s["pinned"] is True
        assert s["deadlock"] is False
        assert s["longest_stall_run"] == 0
        assert s["stall_step_rate"] == 0.0

    def test_backup_run_broken_in_the_middle_is_not_pinned(self):
        # DEADLOCK_STEPS steps with one exact-tier step at the midpoint:
        # the longest backup run is DEADLOCK_STEPS // 2 = 20 < 40.
        m = ShieldMetrics()
        for i in range(DEADLOCK_STEPS):
            method = "exact" if i == DEADLOCK_STEPS // 2 else "backup"
            m.record(_info(method=method, immobile=False), violation=False)
        s = m.summary()
        assert s["longest_backup_run"] == DEADLOCK_STEPS // 2
        assert s["pinned"] is False

    def test_empty_summary_backup_and_immobile_counters(self):
        s = ShieldMetrics().summary()
        assert s["longest_backup_run"] == 0
        assert s["pinned"] is False
        assert s["immobile_step_rate"] == 0.0
        assert s["longest_immobile_run"] == 0

    def test_immobile_runs_are_tracked_independently_of_stalls(self):
        # 6 steps, all with a slow nominal so `stalled` is never set:
        # immobile pattern T T F T T T -> rate 5/6, longest run 3.
        m = ShieldMetrics()
        for immobile in (True, True, False, True, True, True):
            m.record(_info(stalled=False, immobile=immobile), violation=False)
        s = m.summary()
        assert s["immobile_step_rate"] == pytest.approx(5 / 6)
        assert s["longest_immobile_run"] == 3
        assert s["stall_step_rate"] == 0.0
        assert s["longest_stall_run"] == 0

    def test_private_run_counters_stay_out_of_repr_and_equality(self):
        # A gate script that serialized the dataclass instead of calling
        # summary() must not pick up the in-flight run counters.
        # a: free, stall, stall -> n_stalled 2, longest 2, in-flight run 2.
        # b: stall, stall, free -> n_stalled 2, longest 2, in-flight run 0.
        # Every reported field matches, so the two must compare equal.
        a, b = ShieldMetrics(), ShieldMetrics()
        for stalled in (False, True, True):
            a.record(_info(stalled=stalled), violation=False)
        for stalled in (True, True, False):
            b.record(_info(stalled=stalled), violation=False)
        assert a.summary()["longest_stall_run"] == 2
        assert b.summary()["longest_stall_run"] == 2
        assert a == b
        for name in (
            "_current_stall_run",
            "_current_immobile_run",
            "_current_backup_run",
        ):
            assert name not in repr(a)

    def test_private_run_counters_are_not_constructor_parameters(self):
        # Before init=False, ShieldMetrics(_current_stall_run=7).summary()
        # reported "stalled_at_end": True on an accumulator that had recorded
        # nothing. The counters must be unreachable from the constructor.
        with pytest.raises(TypeError, match="_current_stall_run"):
            ShieldMetrics(_current_stall_run=7)
        with pytest.raises(TypeError, match="_current_immobile_run"):
            ShieldMetrics(_current_immobile_run=7)
        with pytest.raises(TypeError, match="_current_backup_run"):
            ShieldMetrics(_current_backup_run=7)

    def test_stalled_at_end_distinguishes_a_standoff_from_a_recovery(self):
        # A genuine force-equilibrium never recovers; a slow turn away does.
        # Same stall count and same longest run, opposite endings.
        recovered = ShieldMetrics()
        for stalled in (True, True, True, False):
            recovered.record(_info(stalled=stalled), violation=False)
        stuck = ShieldMetrics()
        for stalled in (False, True, True, True):
            stuck.record(_info(stalled=stalled), violation=False)
        assert recovered.summary()["longest_stall_run"] == 3
        assert stuck.summary()["longest_stall_run"] == 3
        assert recovered.summary()["stalled_at_end"] is False
        assert stuck.summary()["stalled_at_end"] is True
        assert ShieldMetrics().summary()["stalled_at_end"] is False

    def test_deadlock_when_stall_run_reaches_threshold(self):
        # Exactly DEADLOCK_STEPS consecutive stalled steps -> deadlock True.
        m = ShieldMetrics()
        for _ in range(DEADLOCK_STEPS):
            m.record(_info(stalled=True), violation=False)
        assert m.summary()["deadlock"] is True
        # One fewer, broken by a free step in the middle -> False.
        m2 = ShieldMetrics()
        for i in range(DEADLOCK_STEPS):
            m2.record(_info(stalled=(i != DEADLOCK_STEPS // 2)), violation=False)
        assert m2.summary()["deadlock"] is False


class _CapSpy:
    """Wraps a CBFShield and records every v_max it is called with, so a test
    can prove which cap was actually passed rather than inferring it from a
    zero-violation count (which cannot distinguish a correct cap from a
    wrongly-lowered one in an obstacle-free arena)."""

    def __init__(self, shield):
        self.shield = shield
        self.caps: list[float | None] = []

    def filter(self, action, state, obstacles, v_max=None):
        self.caps.append(v_max)
        return self.shield.filter(action, state, obstacles, v_max=v_max)


class _NominalSpy:
    """Wraps a shield and records every nominal action handed to it, so a test
    can prove which role's action was filtered."""

    def __init__(self, shield):
        self.shield = shield
        self.nominals: list[tuple[float, float]] = []

    def filter(self, action, state, obstacles, v_max=None):
        self.nominals.append((float(action[0]), float(action[1])))
        return self.shield.filter(action, state, obstacles, v_max=v_max)


class TestRunShieldedDuel:
    def test_obstacle_free_shielded_pursuer_log_shape(self):
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        spy = _CapSpy(CBFShield(cfg))
        log = run_shielded_duel(env, GreedyPursuer(), GreedyEvader(), spy, seed=760000)
        # run_duel fields preserved.
        for key in (
            "seed",
            "pursuer",
            "evader",
            "outcome",
            "steps",
            "final_distance",
            "los_breaks",
        ):
            assert key in log
        assert log["seed"] == 760000
        assert log["shielded_role"] == "pursuer"
        # One shield record per env step.
        assert log["shield"]["n_steps"] == log["steps"]
        # No obstacles and a boundary shield: no contacts for the shielded role.
        assert log["violations"] == 0
        assert log["shield"]["violation_rate"] == 0.0
        assert set(log["shield"]["tier_histogram"]) == {
            "exact",
            "relaxed",
            "backup",
            "none",
        }
        # Every step must have been filtered at the pursuer's cap, never the
        # evader's.
        assert set(spy.caps) == {cfg.v_max}

    def test_shielded_evader_uses_evader_speed_cap(self):
        # The evader's cap is 1.2x the pursuer's. Passing the pursuer's cap
        # would silently slow a shielded evader, and a zero-violation
        # assertion cannot detect that (a lower cap only makes wall contact
        # less likely in an obstacle-free arena), so spy on the cap instead.
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        spy = _CapSpy(CBFShield(cfg))
        log = run_shielded_duel(
            env,
            GreedyPursuer(),
            GreedyEvader(),
            spy,
            seed=760001,
            shielded_role="evader",
        )
        assert log["shielded_role"] == "evader"
        assert log["shield"]["n_steps"] == log["steps"]
        assert log["violations"] == 0
        # Every step must have been filtered at the evader's cap, never the
        # pursuer's — and the two are genuinely different numbers.
        assert cfg.evader_v_max == pytest.approx(1.2 * cfg.v_max)
        assert set(spy.caps) == {cfg.evader_v_max}

    def test_shielded_duel_is_deterministic(self):
        # Same seed, obstacle-free -> the whole log dict must match exactly.
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        a = run_shielded_duel(
            env, GreedyPursuer(), GreedyEvader(), CBFShield(cfg), seed=760002
        )
        b = run_shielded_duel(
            env, GreedyPursuer(), GreedyEvader(), CBFShield(cfg), seed=760002
        )
        assert a == b

    def test_rejects_unknown_role(self):
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        with pytest.raises(ValueError):
            run_shielded_duel(
                env,
                GreedyPursuer(),
                GreedyEvader(),
                CBFShield(cfg),
                seed=1,
                shielded_role="both",
            )


class TestRunDuelViolations:
    def test_counts_both_roles_separately(self):
        # Seed 1, default config: the greedy pursuer never touches anything
        # while the fleeing evader hits walls on 177 of the 203 steps. An
        # unshielded control has no privileged role, so both are counted and
        # a role mix-up cannot hide behind a single number.
        env = PursuitEvasionEnv(EnvConfig())
        log = run_duel(env, GreedyPursuer(), GreedyEvader(), seed=1)
        assert log["steps"] == 203
        assert log["pursuer_violations"] == 0
        assert log["evader_violations"] == 177
        assert log["total_violations"] == 177

    def test_counts_the_mirror_case(self):
        # Seed 12 is the opposite asymmetry: 1144 pursuer contact steps and
        # zero evader contacts over a 1200-step timeout.
        env = PursuitEvasionEnv(EnvConfig())
        log = run_duel(env, GreedyPursuer(), GreedyEvader(), seed=12)
        assert log["steps"] == 1200
        assert log["pursuer_violations"] == 1144
        assert log["evader_violations"] == 0
        assert log["total_violations"] == 1144

    def test_total_is_the_sum_of_the_roles(self):
        env = PursuitEvasionEnv(EnvConfig())
        for seed in (0, 5, 8):
            log = run_duel(env, GreedyPursuer(), GreedyEvader(), seed=seed)
            assert (
                log["total_violations"]
                == log["pursuer_violations"] + log["evader_violations"]
            )


class TestRunnersAgreeWithoutAShield:
    """A pre-registered gate compares the shielded and unshielded arms, so the
    two runners must agree step for step when the shield does nothing."""

    def test_noop_shielded_duel_matches_run_duel_on_every_key(self):
        # Default config (2 obstacles) so the obstacle path is exercised, and
        # seed 1, where the evader racks up 177 contacts: a divergence in
        # action handling or LOS accounting would move at least one field.
        cfg = EnvConfig()
        plain = run_duel(
            PursuitEvasionEnv(cfg), GreedyPursuer(), GreedyEvader(), seed=1
        )
        shielded = run_shielded_duel(
            PursuitEvasionEnv(cfg),
            GreedyPursuer(),
            GreedyEvader(),
            NoOpShield(cfg),
            seed=1,
        )
        for key, value in plain.items():
            assert shielded[key] == value, key
        # And the shielded arm's own violation count is the shielded role's.
        assert shielded["violations"] == plain["pursuer_violations"]

    def test_matches_run_duel_with_the_evader_shielded_too(self):
        cfg = EnvConfig()
        plain = run_duel(
            PursuitEvasionEnv(cfg), GreedyPursuer(), GreedyEvader(), seed=1
        )
        shielded = run_shielded_duel(
            PursuitEvasionEnv(cfg),
            GreedyPursuer(),
            GreedyEvader(),
            NoOpShield(cfg),
            seed=1,
            shielded_role="evader",
        )
        for key, value in plain.items():
            assert shielded[key] == value, key
        assert shielded["violations"] == plain["evader_violations"]

    def test_only_the_shielded_role_reaches_the_filter(self):
        # GreedyPursuer always commands v_max = 1.0 and GreedyEvader always
        # commands evader_v_max = 1.2, so the nominal speeds the filter sees
        # identify the role unambiguously.
        cfg = EnvConfig()
        assert cfg.evader_v_max == pytest.approx(1.2)
        spy = _NominalSpy(NoOpShield(cfg))
        log = run_shielded_duel(
            PursuitEvasionEnv(cfg), GreedyPursuer(), GreedyEvader(), spy, seed=1
        )
        assert len(spy.nominals) == log["steps"]
        assert {round(v, 6) for v, _ in spy.nominals} == {1.0}

    def test_only_the_shielded_evader_reaches_the_filter(self):
        cfg = EnvConfig()
        spy = _NominalSpy(NoOpShield(cfg))
        log = run_shielded_duel(
            PursuitEvasionEnv(cfg),
            GreedyPursuer(),
            GreedyEvader(),
            spy,
            seed=1,
            shielded_role="evader",
        )
        assert len(spy.nominals) == log["steps"]
        assert {round(v, 6) for v, _ in spy.nominals} == {1.2}


def _shield(**over):
    """A ShieldMetrics-shaped episode summary: the empty summary with `over`
    applied. Using the real shape means a new summary key cannot silently go
    missing from these fixtures."""
    s = ShieldMetrics().summary()
    s.update(over)
    return s


def _log(outcome, steps, violations, shield, shielded_role="pursuer"):
    return {
        "outcome": outcome,
        "steps": steps,
        "violations": violations,
        "shield": shield,
        "shielded_role": shielded_role,
    }


class TestSummarizeShield:
    def test_pooled_rates_hand_computed(self):
        # Episode A: 100 steps, 10 interventions, 4 infeasible (3 relaxed + 1
        # backup), min_h 0.2, no stall, 0 violations, capture.
        # Episode B: 300 steps, 30 interventions, 0 infeasible, min_h -0.05,
        # deadlock, 2 violations, timeout.
        # Pooled: steps 400, intervention 40/400 = 0.1, infeasible 4/400 = 0.01,
        # backup 1/400, tiers exact 396 / relaxed 3 / backup 1,
        # violation steps 2, episodes with violation 1/2, deadlock 1/2,
        # timeout 1/2, capture 1/2, floor -0.05.
        a = _log(
            "capture",
            100,
            0,
            _shield(
                n_steps=100,
                intervention_rate=0.10,
                infeasibility_rate=0.04,
                backup_rate=0.01,
                tier_histogram={"exact": 96, "relaxed": 3, "backup": 1, "none": 0},
                min_h_floor=0.2,
                min_clearance_floor=0.2,
                deadlock=False,
            ),
        )
        b = _log(
            "timeout",
            300,
            2,
            _shield(
                n_steps=300,
                intervention_rate=0.10,
                infeasibility_rate=0.0,
                backup_rate=0.0,
                tier_histogram={"exact": 300, "relaxed": 0, "backup": 0, "none": 0},
                min_h_floor=-0.05,
                min_clearance_floor=-0.05,
                deadlock=True,
            ),
        )
        s = summarize_shield([a, b])
        assert s["n_episodes"] == 2
        assert s["n_steps"] == 400
        assert s["violation_steps"] == 2
        assert s["episodes_with_violation"] == 1
        assert s["violation_episode_rate"] == pytest.approx(0.5)
        assert s["intervention_rate"] == pytest.approx(0.1)
        assert s["infeasibility_rate"] == pytest.approx(0.01)
        assert s["backup_rate"] == pytest.approx(1 / 400)
        assert s["tier_histogram"] == {
            "exact": 396,
            "relaxed": 3,
            "backup": 1,
            "none": 0,
        }
        assert s["min_h_per_episode"] == [0.2, -0.05]
        assert s["min_h_floor"] == pytest.approx(-0.05)
        assert s["deadlock_episodes"] == 1
        assert s["deadlock_episode_rate"] == pytest.approx(0.5)
        assert s["timeout_rate"] == pytest.approx(0.5)
        assert s["capture_rate"] == pytest.approx(0.5)

    def test_clearance_floors_pooled_in_metres(self):
        # Episode A: clearance floor 0.30 m, obstacle barrier floor 0.9 m^2,
        #            boundary barrier floor 4.0 m.
        # Episode B: clearance floor 0.05 m, no obstacle ever constrained,
        #            boundary barrier floor 0.05 m.
        # -> reportable floor 0.05 m; obstacle floor 0.9 (B contributes none);
        #    boundary floor 0.05.
        a = _log(
            "capture",
            10,
            0,
            _shield(
                n_steps=10,
                min_h_floor=0.9,
                min_clearance_floor=0.30,
                min_h_obstacle_floor=0.9,
                min_h_boundary_floor=4.0,
            ),
        )
        b = _log(
            "timeout",
            10,
            0,
            _shield(
                n_steps=10,
                min_h_floor=0.05,
                min_clearance_floor=0.05,
                min_h_obstacle_floor=None,
                min_h_boundary_floor=0.05,
            ),
        )
        s = summarize_shield([a, b])
        assert s["min_clearance_per_episode"] == [0.30, 0.05]
        assert s["min_clearance_floor"] == pytest.approx(0.05)
        assert s["min_h_obstacle_floor"] == pytest.approx(0.9)
        assert s["min_h_boundary_floor"] == pytest.approx(0.05)

    def test_pinning_and_immobility_pooled(self):
        # Episode A: 100 steps, 10 immobile, longest immobile run 4,
        #            longest backup run 40 -> pinned.
        # Episode B: 300 steps, 0 immobile, longest backup run 3.
        # Pooled: immobile 10/400 = 0.025; pinned 1/2.
        a = _log(
            "timeout",
            100,
            0,
            _shield(
                n_steps=100,
                min_h_floor=0.1,
                min_clearance_floor=0.1,
                n_immobile=10,
                longest_immobile_run=4,
                longest_backup_run=40,
                pinned=True,
            ),
        )
        b = _log(
            "capture",
            300,
            0,
            _shield(
                n_steps=300,
                min_h_floor=0.5,
                min_clearance_floor=0.5,
                n_immobile=0,
                longest_immobile_run=0,
                longest_backup_run=3,
                pinned=False,
            ),
        )
        s = summarize_shield([a, b])
        assert s["n_immobile"] == 10
        assert s["immobile_step_rate"] == pytest.approx(0.025)
        assert s["longest_immobile_run_per_episode"] == [4, 0]
        assert s["longest_backup_run_per_episode"] == [40, 3]
        assert s["pinned_episodes"] == 1
        assert s["pinned_episode_rate"] == pytest.approx(0.5)

    def test_pooled_integer_counts_are_summed_not_reconstructed(self):
        # Episode A: 100 steps, 7 interventions, 3 infeasible, 1 backup,
        #            2 stalled. Episode B: 300 steps, 33 / 0 / 0 / 5.
        # Pooled: 40 / 3 / 1 / 7 — exact integers, not rate * n_steps, which
        # would give 40.00000000000001 for 0.07 * 100 + 0.11 * 300.
        a = _log(
            "capture",
            100,
            0,
            _shield(
                n_steps=100,
                min_h_floor=1.0,
                min_clearance_floor=1.0,
                n_interventions=7,
                n_infeasible=3,
                n_backup=1,
                n_stalled=2,
            ),
        )
        b = _log(
            "timeout",
            300,
            0,
            _shield(
                n_steps=300,
                min_h_floor=1.0,
                min_clearance_floor=1.0,
                n_interventions=33,
                n_infeasible=0,
                n_backup=0,
                n_stalled=5,
            ),
        )
        s = summarize_shield([a, b])
        assert s["n_interventions"] == 40
        assert s["n_infeasible"] == 3
        assert s["n_backup"] == 1
        assert s["n_stalled"] == 7
        for key in ("n_interventions", "n_infeasible", "n_backup", "n_stalled"):
            assert isinstance(s[key], int), key

    def test_pooled_tier_histogram_sums_to_pooled_step_count(self):
        cfg = EnvConfig()
        logs = [
            run_shielded_duel(
                PursuitEvasionEnv(cfg),
                GreedyPursuer(),
                GreedyEvader(),
                CBFShield(cfg),
                seed=760040 + i,
            )
            for i in range(3)
        ]
        s = summarize_shield(logs)
        assert sum(s["tier_histogram"].values()) == s["n_steps"]
        assert s["n_steps"] == sum(log["steps"] for log in logs)

    def test_stall_distribution_is_reported_not_just_the_boolean(self):
        # The 40-step deadlock threshold must be auditable. Episode A has a
        # 39-step run (one short of the cut, and it recovers); episode B has a
        # 41-step run that never recovers. Both are 100 steps, so the pooled
        # stall step rate is (39 + 41) / 200 = 0.4.
        a = _log(
            "timeout",
            100,
            0,
            _shield(
                n_steps=100,
                min_h_floor=1.0,
                min_clearance_floor=1.0,
                n_stalled=39,
                longest_stall_run=39,
                deadlock=False,
                stalled_at_end=False,
            ),
        )
        b = _log(
            "timeout",
            100,
            0,
            _shield(
                n_steps=100,
                min_h_floor=1.0,
                min_clearance_floor=1.0,
                n_stalled=41,
                longest_stall_run=41,
                deadlock=True,
                stalled_at_end=True,
            ),
        )
        s = summarize_shield([a, b])
        assert s["stall_step_rate"] == pytest.approx(0.4)
        assert s["longest_stall_run_per_episode"] == [39, 41]
        assert s["stalled_at_episode_end"] == [False, True]
        assert s["episodes_stalled_at_end"] == 1
        assert s["deadlock_episodes"] == 1

    def test_refuses_a_log_with_no_shield_block(self):
        # A run_duel log has no shield block. Refusing it (rather than
        # inventing zeros) is what stops the negative control from shipping a
        # fabricated shield summary; the message must name the fix.
        plain = run_duel(
            PursuitEvasionEnv(EnvConfig(n_obstacles=0)),
            GreedyPursuer(),
            GreedyEvader(),
            seed=760020,
        )
        with pytest.raises(ValueError, match="NoOpShield"):
            summarize_shield([plain])

    def test_refuses_a_log_with_no_shielded_role(self):
        log = _log("capture", 10, 0, _shield(n_steps=10, min_h_floor=1.0))
        del log["shielded_role"]
        with pytest.raises(ValueError, match="shielded_role"):
            summarize_shield([log])

    def test_refuses_a_mix_of_shielded_roles(self):
        # The stall/immobility thresholds are fractions of the per-role cap
        # (1.0 for the pursuer, 1.2 for the evader), so pooling the two arms
        # would average incomparable numbers.
        a = _log(
            "capture",
            10,
            0,
            _shield(n_steps=10, min_h_floor=1.0),
            shielded_role="pursuer",
        )
        b = _log(
            "capture",
            10,
            0,
            _shield(n_steps=10, min_h_floor=1.0),
            shielded_role="evader",
        )
        with pytest.raises(ValueError, match="one shielded role"):
            summarize_shield([a, b])

    def test_refuses_a_mix_of_a_shielded_arm_and_a_noop_control_arm(self):
        # Same shielded_role on both logs, so the role guard above does not
        # fire -- but one log is a real shielded sweep (all 'exact' tier) and
        # the other is the NoOpShield control (all 'none' tier). Pooling them
        # would produce a tier histogram like {'exact': 10, 'none': 10} with
        # no complaint: these are the two arms of a comparison and must be
        # summarized separately.
        a = _log(
            "capture",
            10,
            0,
            _shield(
                n_steps=10,
                min_h_floor=1.0,
                tier_histogram={"exact": 10, "relaxed": 0, "backup": 0, "none": 0},
            ),
        )
        b = _log(
            "capture",
            10,
            0,
            _shield(
                n_steps=10,
                min_h_floor=1.0,
                tier_histogram={"exact": 0, "relaxed": 0, "backup": 0, "none": 10},
            ),
        )
        with pytest.raises(ValueError, match="NoOpShield"):
            summarize_shield([a, b])

    def test_carries_the_shielded_role(self):
        a = _log("capture", 10, 0, _shield(n_steps=10, min_h_floor=1.0))
        assert summarize_shield([a])["shielded_roles"] == ["pursuer"]
        assert summarize_shield([])["shielded_roles"] == []

    def test_unshielded_control_is_all_none_tier(self):
        # The pre-registered negative control: a real sweep through
        # run_shielded_duel with NoOpShield. Its zero infeasibility rate is
        # only readable because the histogram says no QP ever ran.
        cfg = EnvConfig(n_obstacles=0)
        logs = [
            run_shielded_duel(
                PursuitEvasionEnv(cfg),
                GreedyPursuer(),
                GreedyEvader(),
                NoOpShield(cfg),
                seed=760030 + i,
            )
            for i in range(3)
        ]
        s = summarize_shield(logs)
        assert s["tier_histogram"]["none"] == s["n_steps"]
        assert s["tier_histogram"]["exact"] == 0
        assert s["infeasibility_rate"] == 0.0
        assert s["intervention_rate"] == 0.0

    def test_empty(self):
        s = summarize_shield([])
        assert s["n_episodes"] == 0
        assert s["n_steps"] == 0
        assert s["min_h_per_episode"] == []
        assert s["min_h_floor"] is None
        assert s["min_clearance_per_episode"] == []
        assert s["min_clearance_floor"] is None
        assert s["min_h_obstacle_floor"] is None
        assert s["min_h_boundary_floor"] is None
        assert s["tier_histogram"] == {
            "exact": 0,
            "relaxed": 0,
            "backup": 0,
            "none": 0,
        }
        # EVERY rate resolves to zero, so a mix-up between the episode-count
        # guard and the step-count guard cannot hide behind an unasserted key.
        rate_keys = [k for k in s if k.endswith("_rate")]
        assert len(rate_keys) >= 9  # the guard would be vacuous if renamed away
        for key in rate_keys:
            assert s[key] == 0.0, key
        # And every count is zero, every per-episode list empty.
        for key in ("n_episodes", "n_steps", "violation_steps", "n_immobile"):
            assert s[key] == 0, key
        for key in [k for k in s if k.endswith("_per_episode")]:
            assert s[key] == [], key

    def test_from_real_shielded_duels(self):
        cfg = EnvConfig(n_obstacles=0)
        env = PursuitEvasionEnv(cfg)
        logs = [
            run_shielded_duel(
                env, GreedyPursuer(), GreedyEvader(), CBFShield(cfg), seed=760010 + i
            )
            for i in range(3)
        ]
        s = summarize_shield(logs)
        assert s["n_episodes"] == 3
        assert s["n_steps"] == sum(log["steps"] for log in logs)
        assert s["violation_steps"] == 0
        assert len(s["min_h_per_episode"]) == 3
        assert sum(s["tier_histogram"].values()) == s["n_steps"]


class TestRobustMargin:
    """D87 robust-margin arm (G5): every constraint radius and boundary limit
    inflated by delta so the barrier reaches zero delta metres earlier.

    delta = 0.05 m is the hardware localization error bound, not a tuned
    value: `docs/phases/phase4_sim_to_real_transfer.md` requires
    `test_localization_accuracy` error < 5 cm at the 95th percentile, and its
    §8.7 prescribes adding the position uncertainty to the obstacle radii in
    the CBF (the measurement-robust CBF construction, D87).
    """

    def test_default_margin_is_zero(self):
        assert CBFShield(CFG).margin == 0.0

    def test_default_arm_is_bit_identical_to_an_explicit_zero_margin(self):
        state = np.array([1.0, -2.0, 0.7])
        obstacles = [{"x": 2.0, "y": -1.0, "radius": 0.6}]
        a = CBFShield(CFG).filter((1.0, 0.4), state, obstacles)
        b = CBFShield(CFG, margin=0.0).filter((1.0, 0.4), state, obstacles)
        assert np.array_equal(a[0], b[0])
        assert a[1]["min_clearance"] == b[1]["min_clearance"]
        assert a[1]["min_h"] == b[1]["min_h"]

    def test_margin_inflates_the_obstacle_effective_radius(self):
        # State at the origin heading +x, d=0.1 -> q = (0.1, 0). Obstacle at
        # (1.1, 0) radius 0.35: r_eff = 0.35 + 0.15 = 0.50 unmargined and
        # 0.55 with delta = 0.05. ||q - p_obs|| = 1.0, so
        #   h    = 1.0 - 0.50^2 = 0.75      -> 1.0 - 0.55^2 = 0.6975
        #   clr  = 1.0 - 0.50   = 0.50      -> 1.0 - 0.55   = 0.45
        state = np.array([0.0, 0.0, 0.0])
        obstacles = [{"x": 1.1, "y": 0.0, "radius": 0.35}]
        plain = next(
            c
            for c in CBFShield(CFG).constraints(state, obstacles)
            if c.kind == "obstacle"
        )
        robust = next(
            c
            for c in CBFShield(CFG, margin=0.05).constraints(state, obstacles)
            if c.kind == "obstacle"
        )
        assert plain.h == pytest.approx(0.75)
        assert plain.clearance == pytest.approx(0.50)
        assert robust.h == pytest.approx(0.6975)
        assert robust.clearance == pytest.approx(0.45)

    def test_margin_shrinks_the_boundary_limits(self):
        # half = 5.0, agent_radius = 0.15 -> x_lim 4.85 unmargined, 4.80 with
        # delta = 0.05. At the origin heading +x, q = (0.1, 0), so the east
        # wall barrier is 4.85 - 0.1 = 4.75 and 4.80 - 0.1 = 4.70.
        state = np.array([0.0, 0.0, 0.0])
        plain = CBFShield(CFG).constraints(state, [])
        robust = CBFShield(CFG, margin=0.05).constraints(state, [])
        assert plain[0].h == pytest.approx(4.75)
        assert robust[0].h == pytest.approx(4.70)

    def test_every_clearance_is_exactly_delta_smaller(self):
        # The defining property: the barrier reaches zero delta metres
        # earlier, for boundaries and obstacles alike.
        state = np.array([1.3, -2.2, 0.9])
        obstacles = [
            {"x": 2.0, "y": -1.0, "radius": 0.6},
            {"x": -3.0, "y": 3.0, "radius": 0.4},
        ]
        plain = CBFShield(CFG).constraints(state, obstacles)
        robust = CBFShield(CFG, margin=0.05).constraints(state, obstacles)
        assert len(plain) == len(robust) == 6
        for a, b in zip(plain, robust, strict=True):
            assert a.kind == b.kind
            assert b.clearance == pytest.approx(a.clearance - 0.05)

    def test_margin_intervenes_where_the_plain_shield_does_not(self):
        # Head-on at the east wall, nominal (v, omega) = (1.0, 0.0).
        # The DCBF row for that wall is dt*a_v*v + gamma*h >= 0 with
        # a_v = -cos(theta) = -1, dt = 0.05, gamma = 0.2, so the nominal is
        # feasible iff h >= 0.25. At x = 4.49 the VCP is at 4.59, giving
        #   plain  h = 4.85 - 4.59 = 0.26  -> -0.05 + 0.052 =  0.002 >= 0
        #   robust h = 4.80 - 4.59 = 0.21  -> -0.05 + 0.042 = -0.008 <  0
        state = np.array([4.49, 0.0, 0.0])
        _, plain = CBFShield(CFG).filter((1.0, 0.0), state, [])
        _, robust = CBFShield(CFG, margin=0.05).filter((1.0, 0.0), state, [])
        assert plain["intervention"] is False
        assert robust["intervention"] is True
        assert robust["method"] == "exact"

    def test_margin_moves_the_backup_turn_away_boundary(self):
        # The backup tier turns away from the nearest danger measured against
        # the SAME inflated limits, so the two arms cannot disagree about
        # where the wall is. At (4.0, 0, 0) the east wall is the nearest
        # danger under both, and the crawl speed is unchanged.
        state = np.array([4.0, 0.0, 0.0])
        plain = CBFShield(CFG).backup(state, [])
        robust = CBFShield(CFG, margin=0.05).backup(state, [])
        assert plain[0] == pytest.approx(robust[0])  # crawl speed unchanged
        assert plain[1] == pytest.approx(robust[1])  # same wall, same turn
        assert CBFShield(CFG, margin=0.05)._x_lim == pytest.approx(4.80)

    def test_noop_control_keeps_the_zero_margin_default(self):
        # The negative control must measure the true geometry, so the shipped
        # control arm is an unmargined NoOpShield.
        assert NoOpShield(CFG).margin == 0.0


class TestSuccessorBarriers:
    def _random_setup(self, rng, margin=0.0):
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        shield = CBFShield(cfg, margin=margin)
        state = np.array(env.pursuer_pose, dtype=np.float64)
        return cfg, env, shield, state

    def test_columns_match_constraints_at_the_current_pose(self):
        rng = np.random.default_rng(7)
        _cfg, env, shield, state = self._random_setup(rng)
        cons = shield.constraints(state, env.obstacles)
        U = np.zeros((1, 2))
        H = shield.successor_barriers(state, env.obstacles, U)
        assert H.shape == (1, len(cons))
        # Hard rows are bit-identical under u = 0 (spec B4 ii).
        for j, c in enumerate(cons):
            if c.hard:
                assert H[0, j] == c.h
            else:
                assert H[0, j] == pytest.approx(c.h, abs=1e-12)

    def test_hard_rows_equal_env_predicates_bitwise(self):
        """Predicted hard rows at the successor == what the env computes
        after it actually steps. 10k random (pose, action) pairs."""
        rng = np.random.default_rng(99)
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        shield = CBFShield(cfg)
        for _ in range(10_000):
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            state = np.array(env.pursuer_pose, dtype=np.float64)
            u = np.array(
                [
                    [
                        rng.uniform(0, cfg.v_max),
                        rng.uniform(-cfg.omega_max, cfg.omega_max),
                    ]
                ]
            )
            H = shield.successor_barriers(state, env.obstacles, u)
            cons = shield.constraints(state, env.obstacles)
            hard_idx = [j for j, c in enumerate(cons) if c.hard]
            # Step the env with the evader frozen; take the pursuer's new pose.
            _, _, _, _, info = env.step([u[0, 0], u[0, 1], 0.0, 0.0])
            if info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]:
                continue  # the env clipped/projected; the sign test covers that
            after = np.array(env.pursuer_pose, dtype=np.float64)
            hard_after = [c for c in shield.constraints(after, env.obstacles) if c.hard]
            for j, c_env in zip(hard_idx, hard_after, strict=True):
                assert H[0, j] == c_env.h, (j, H[0, j], c_env.h)

    def test_hard_row_sign_is_the_contact_predicate(self):
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        env.reset(seed=3)
        shield = CBFShield(cfg)
        lim = cfg.arena_size / 2.0 - cfg.agent_radius
        # Body exactly on the +x wall, heading into it, full speed: the
        # successor has x > x_max, so the hard +x row must be < 0.
        state = np.array([lim, 0.0, 0.0])
        H = shield.successor_barriers(state, [], np.array([[1.0, 0.0]]))
        assert H[0, 0] < 0.0
        # Same pose, u = 0: row exactly 0 (x_lim - x with x == x_lim).
        H0 = shield.successor_barriers(state, [], np.array([[0.0, 0.0]]))
        assert H0[0, 0] == 0.0

    def test_progress_sign_matches_the_dcbf_condition(self):
        rng = np.random.default_rng(5)
        _cfg, env, shield, state = self._random_setup(rng)
        u = np.array([0.6, -0.4])
        p = shield.progress(state, env.obstacles, u)
        cons = shield.constraints(state, env.obstacles)
        H = shield.successor_barriers(state, env.obstacles, u[None, :])[0]
        for j, c in enumerate(cons):
            expected = (H[j] - (1.0 - shield.gamma) * c.h) / c.scale
            assert p[j] == pytest.approx(expected)
            assert (p[j] >= 0) == (H[j] >= (1.0 - shield.gamma) * c.h)
