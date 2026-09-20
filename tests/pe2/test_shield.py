"""Tests for pe2.shield + pe2.metrics — Step 6 safety shield."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.env import OUTCOME_CAPTURE, PursuitEvasionEnv, wrap_angle
from pe2.metrics import DEADLOCK_STEPS, ShieldMetrics, summarize_shield
from pe2.scripted import GreedyEvader, GreedyPursuer, run_duel, run_shielded_duel
from pe2.shield import (
    CBFShield,
    NoOpShield,
    _body_obstacle_constraint,
    _body_wall_constraints,
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
        # r_eff = 0.6 -> scale = 2 * 0.6 = 1.2, distinct from a hardcoded 1.0
        # (which a bug that always returned 1.0 would also produce at r_eff
        # 0.5, the value this test used to use).
        c = _obstacle_constraint(np.array([0.0, 0.0, 0.0]), 1.1, 0.0, 0.6, 0.1)
        assert c.hard is False and c.scale == pytest.approx(1.2)  # 2 * r_eff
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

    def test_no_inter_agent_constraint(self):
        # The shield takes obstacles only; constraints are obstacle+boundary,
        # so capture (which needs contact) can never be blocked by design.
        shield = CBFShield(CFG)
        obstacles = [{"x": 1.0, "y": 0.0, "radius": 0.3}]
        cons = shield.constraints(np.array([0.0, 0.0, 0.0]), obstacles)
        assert info_kinds(cons) == {"obstacle", "boundary"}
        assert len(cons) == 4 + 4 + 2 * len(obstacles)

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
        #
        # v = 0.004 sits EXACTLY on that barrier, and the tier test is the
        # exact one-step check with no tolerance, so in float the successor
        # barrier lands ~1 ulp below the target (-3.6e-16) and the candidate is
        # rejected. The nearest admissible candidate is then a full stop. The
        # conservative direction, and 0.2 mm/step of speed either way; a
        # tolerance here would have to be granted to the HARD rows too, which
        # is precisely the guarantee this design refuses to soften.
        shield = CBFShield(CFG)
        u, info = shield.filter((1.0, 0.0), np.array([4.749, 0.0, 0.0]), [])
        assert info["method"] == "exact"
        assert u[0] == 0.0
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
        # the shield still holds the executed speed at 0 <= 0.05 * v_max (see
        # the stall test above for why the 0.004 boundary candidate rounds out).
        # `immobile` is the nominal-independent flag that catches it.
        shield = CBFShield(CFG)
        _, info = shield.filter((0.4, 0.0), np.array([4.749, 0.0, 0.0]), [])
        assert info["v_exec"] == 0.0
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

    def test_min_clearance_hard_is_the_body_floor_not_the_vcp_floor(self):
        # Same wall geometry as test_equal_clearance_wall_and_obstacle_report_
        # equal_clearance: x = 4.73, x_lim = 4.85. The VCP leads the body by
        # d = 0.1 m, so its row (q = 4.83, h = 0.02 m) is closer to the wall
        # and sets `min_clearance`; the BODY (hard) row is x_lim - x = 0.12 m
        # -- a different value. If `min_clearance_hard` were confused with
        # `min_clearance` (e.g. aliased to it), this would not catch it.
        shield = CBFShield(CFG)
        _, wall = shield.filter((0.1, 0.0), np.array([4.73, 0.0, 0.0]), [])
        assert wall["min_clearance"] == pytest.approx(0.02)
        assert wall["min_clearance_hard"] == pytest.approx(0.12)

        # Obstacle geometry as the same test: r_eff = 1.15, body distance 1.27
        # -> hard clearance 0.12; VCP distance 1.17 -> soft clearance 0.02
        # sets `min_clearance`. Again the two must differ.
        _, obst = shield.filter(
            (0.1, 0.0),
            np.array([0.0, 0.0, 0.0]),
            [{"x": 1.27, "y": 0.0, "radius": 1.0}],
        )
        assert obst["min_clearance"] == pytest.approx(0.02)
        assert obst["min_clearance_hard"] == pytest.approx(0.12)

    def test_safe_nominal_passes_unchanged_on_the_exact_tier(self):
        shield = CBFShield(CFG)
        u, info = shield.filter((0.8, 0.3), np.array([0.0, 0.0, 0.0]), [])
        assert info["method"] == "exact" and info["feasible"]
        assert np.allclose(u, [0.8, 0.3]) and not info["intervention"]

    def test_head_on_at_the_wall_stops_and_never_contacts(self):
        shield = CBFShield(CFG)
        state = np.array([X_LIM, 0.0, 0.0])  # body on the +x wall, heading into it
        u, info = shield.filter((1.0, 0.0), state, [])
        assert u[0] < 1e-12  # sub-ulp speeds cannot move x past x_lim
        assert info["method"] in ("exact", "relaxed")  # the VCP row is already < 0
        assert shield.progress(state, [], u)[0] >= 0.0  # hard +x row kept

    def test_tangent_slide_into_obstacle_is_braked_not_penetrated(self):
        """RC2 from the spec: at the surface with the heading 96° from the
        outward normal the VCP row alone would allow ~3 mm penetration."""
        shield = CBFShield(CFG)
        r_eff = 0.3 + CFG.agent_radius  # 0.45
        obstacles = [{"x": 0.0, "y": 0.0, "radius": 0.3}]
        state = np.array([r_eff, 0.0, np.pi / 2 + np.deg2rad(6.0)])  # slightly inward
        u, info = shield.filter((1.0, 0.0), state, obstacles)
        env = PursuitEvasionEnv(CFG)
        env.reset(seed=0)
        env.obstacles = obstacles
        env.pursuer_pose[:] = state
        env.evader_pose[:] = (4.0, 4.0, 0.0)
        _, _, _, _, step_info = env.step([u[0], u[1], 0.0, 0.0])
        assert not step_info["pursuer_obstacle_contact"]
        assert info["intervention"]

    def test_reset_with_vcp_inside_uses_relaxed_tier_and_keeps_hard_rows(self):
        """RC1: body just outside the surface, VCP already inside (h_vcp < 0)."""
        shield = CBFShield(CFG)
        obstacles = [{"x": 0.0, "y": 0.0, "radius": 0.3}]
        state = np.array([0.46, 0.0, np.pi])  # heading straight at the centre
        cons = shield.constraints(state, obstacles)
        assert any(c.h < 0 for c in cons if not c.hard)
        assert all(c.h >= 0 for c in cons if c.hard)
        u, info = shield.filter((1.0, 0.0), state, obstacles)
        assert info["method"] == "relaxed" and not info["feasible"]
        p = shield.progress(state, obstacles, u)
        assert all(p[j] >= 0 for j, c in enumerate(cons) if c.hard)

    def test_backup_tier_maximises_worst_hard_progress_then_turns(self):
        """Body inside the INFLATED set (robust arm), heading at the centre:
        no candidate keeps the hard row (any v > 0 goes deeper, v = 0 holds
        h < (1-γ)h), so the tier is backup; the best hard progress is v = 0,
        and the soft-row tie-break turns the VCP out at full rate so the
        next step can drive away."""
        shield = CBFShield(CFG, margin=0.05)
        obstacles = [{"x": 0.0, "y": 0.0, "radius": 0.3}]
        state = np.array([0.47, 0.0, np.pi])  # inside 0.50 but outside 0.45
        cons = shield.constraints(state, obstacles)
        assert any(c.h < 0 for c in cons if c.hard)
        u, info = shield.filter((1.0, 0.0), state, obstacles)
        assert info["method"] == "backup" and not info["feasible"]
        assert u[0] == 0.0 and abs(u[1]) == CFG.omega_max
        p = shield.progress(state, obstacles, u)
        hard = [j for j, c in enumerate(cons) if c.hard]
        worst = min(p[hard])
        # No other candidate does better on the worst hard row.
        from pe2.shield import _qp_candidates

        rows = shield._dcbf_rows(cons)
        cands = _qp_candidates(
            1.0, 0.0, rows, CFG.v_max, CFG.omega_max, shield.w_v, shield.w_omega
        )
        for cand in cands:
            assert min(shield.progress(state, obstacles, cand)[hard]) <= worst + 1e-12

    def test_backup_tier_drives_out_when_heading_away(self):
        shield = CBFShield(CFG, margin=0.05)
        obstacles = [{"x": 0.0, "y": 0.0, "radius": 0.3}]
        state = np.array([0.47, 0.0, 0.0])  # heading away from the centre
        u, info = shield.filter((0.0, 0.0), state, obstacles)
        # Full speed away satisfies the hard row (0.52 - 0.50 >= 0.8 * -0.03),
        # so this is NOT the backup tier: the exact/relaxed tiers handle it.
        assert info["method"] in ("exact", "relaxed")
        assert u[0] > 0.0 and info["intervention"]

    def test_filter_is_deterministic(self):
        shield = CBFShield(CFG)
        obstacles = [{"x": 1.0, "y": 0.2, "radius": 0.4}]
        state = np.array([0.3, 0.0, 0.1])
        a, ia = shield.filter((1.0, 1.0), state, obstacles)
        b, ib = shield.filter((1.0, 1.0), state, obstacles)
        assert np.array_equal(a, b) and ia == ib

    def test_the_heuristic_backup_is_gone(self):
        import pe2.shield as m

        assert not hasattr(m, "BACKUP_V_FRAC") and not hasattr(m, "HEAD_ON_TIE_TOL")
        assert not hasattr(m, "FEAS_TOL") and not hasattr(CBFShield, "backup")


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
        # Body (hard) row: x_lim - x = 0.12 m, distinct from the VCP-inclusive
        # min_clearance above -- NoOpShield reports this too, like every other
        # info key (it shares CBFShield._info).
        assert info["min_clearance_hard"] == pytest.approx(0.12)
        assert info["n_constraints"] == 8  # 4 hard walls + 4 soft walls

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

    Defaults keep `min_clearance`, `min_clearance_hard` and `min_h_boundary`
    equal to `min_h` so the hand-computed barrier assertions below stay
    readable; the clearance tests override them explicitly.
    """
    info = {
        "method": method,
        "feasible": method == "exact",
        "intervention": intervention,
        "min_h": min_h,
        "min_clearance": min_h,
        "min_clearance_hard": min_h,
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
                np.array([0.46, 0.0, np.pi]),  # VCP inside, body outside
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

    def test_min_clearance_hard_floor_differs_from_vcp_inclusive_floor(self):
        # step 1: min_clearance (VCP-inclusive) 0.02, min_clearance_hard
        #   (body) 0.12 -- the VCP leads the body, so the all-rows floor is
        #   tighter than the body-only floor on this step.
        # step 2: both rows agree far from any surface, at 0.50.
        # -> min_clearance_floor 0.02 (step 1's VCP row), min_clearance_hard_
        #    floor 0.12 (step 1's body row is the worst body row seen; step 2
        #    is looser). If the two accumulators were confused (e.g. one
        #    aliased to the other), this would assert two different numbers
        #    are equal and fail.
        m = ShieldMetrics()
        m.record(_info(min_clearance=0.02, min_clearance_hard=0.12), violation=False)
        m.record(_info(min_clearance=0.50, min_clearance_hard=0.50), violation=False)
        s = m.summary()
        assert s["min_clearance_floor"] == pytest.approx(0.02)
        assert s["min_clearance_hard_floor"] == pytest.approx(0.12)
        assert s["min_clearance_floor"] != pytest.approx(s["min_clearance_hard_floor"])

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
        assert s["min_clearance_hard_floor"] is None
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
                min_clearance_hard_floor=0.2,
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
                min_clearance_hard_floor=-0.05,
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
        # Episode A: clearance floor 0.30 m (VCP-inclusive), body-only floor
        #            0.45 m, obstacle barrier floor 0.9 m^2, boundary barrier
        #            floor 4.0 m.
        # Episode B: clearance floor 0.05 m (VCP-inclusive), body-only floor
        #            0.20 m, no obstacle ever constrained, boundary barrier
        #            floor 0.05 m.
        # -> reportable floor 0.05 m; body-only floor 0.20 (pooled
        #    independently -- the VCP row, not the body row, sets the
        #    all-rows floor on both episodes, so the two floors must not
        #    collapse to the same pooled value); obstacle floor 0.9 (B
        #    contributes none); boundary floor 0.05.
        a = _log(
            "capture",
            10,
            0,
            _shield(
                n_steps=10,
                min_h_floor=0.9,
                min_clearance_floor=0.30,
                min_clearance_hard_floor=0.45,
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
                min_clearance_hard_floor=0.20,
                min_h_obstacle_floor=None,
                min_h_boundary_floor=0.05,
            ),
        )
        s = summarize_shield([a, b])
        assert s["min_clearance_per_episode"] == [0.30, 0.05]
        assert s["min_clearance_floor"] == pytest.approx(0.05)
        assert s["min_clearance_hard_per_episode"] == [0.45, 0.20]
        assert s["min_clearance_hard_floor"] == pytest.approx(0.20)
        assert s["min_clearance_hard_floor"] != pytest.approx(s["min_clearance_floor"])
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
                min_clearance_hard_floor=0.1,
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
                min_clearance_hard_floor=0.5,
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
                min_clearance_hard_floor=1.0,
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
                min_clearance_hard_floor=1.0,
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
                min_clearance_hard_floor=1.0,
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
                min_clearance_hard_floor=1.0,
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
        assert s["min_clearance_hard_per_episode"] == []
        assert s["min_clearance_hard_floor"] is None
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
            if c.kind == "obstacle" and not c.hard
        )
        robust = next(
            c
            for c in CBFShield(CFG, margin=0.05).constraints(state, obstacles)
            if c.kind == "obstacle" and not c.hard
        )
        assert plain.h == pytest.approx(0.75)
        assert plain.clearance == pytest.approx(0.50)
        assert robust.h == pytest.approx(0.6975)
        assert robust.clearance == pytest.approx(0.45)

    def test_margin_shrinks_the_boundary_limits(self):
        # half = 5.0, agent_radius = 0.15 -> x_lim 4.85 unmargined, 4.80 with
        # delta = 0.05. At the origin heading +x, q = (0.1, 0), so the soft
        # east-wall barrier (row 4, the first VCP row) is 4.85 - 0.1 = 4.75
        # and 4.80 - 0.1 = 4.70; the hard east-wall row (row 0) is on the body
        # at x = 0, so it is x_lim itself.
        state = np.array([0.0, 0.0, 0.0])
        plain = CBFShield(CFG).constraints(state, [])
        robust = CBFShield(CFG, margin=0.05).constraints(state, [])
        assert plain[4].h == pytest.approx(4.75)
        assert robust[4].h == pytest.approx(4.70)
        assert plain[0].h == pytest.approx(4.85)
        assert robust[0].h == pytest.approx(4.80)

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
        assert len(plain) == len(robust) == 12  # 4 hard + 4 soft walls, 2 per obstacle
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

    def test_constructor_h_matches_h_columns_directly(self):
        """`constraints()` overwrites `c.h` from `_h_columns`, so comparing
        the two through `constraints()` (as the test above does) compares
        `_h_columns` against itself. This instead builds the rows straight
        from the module-level constructors -- the formulas an engineer would
        actually edit -- and checks each `h` against `_h_columns`
        independently, so the two cannot silently drift apart."""
        rng = np.random.default_rng(11)
        for margin in (0.0, 0.05):
            cfg = EnvConfig()
            env = PursuitEvasionEnv(cfg)
            shield = CBFShield(cfg, margin=margin)
            for _ in range(5):
                env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                state = np.array(env.pursuer_pose, dtype=np.float64)
                direct = _body_wall_constraints(state, shield._x_lim, shield._y_lim)
                direct += _boundary_constraints(
                    state, shield._x_lim, shield._y_lim, shield.d
                )
                for obs in env.obstacles:
                    r_eff = obs["radius"] + cfg.agent_radius + margin
                    direct.append(
                        _body_obstacle_constraint(state, obs["x"], obs["y"], r_eff)
                    )
                    direct.append(
                        _obstacle_constraint(state, obs["x"], obs["y"], r_eff, shield.d)
                    )
                H = shield._h_columns(
                    np.array([state[0]]),
                    np.array([state[1]]),
                    np.array([state[2]]),
                    env.obstacles,
                )[0]
                assert len(direct) == len(H)
                for j, c in enumerate(direct):
                    assert H[j] == c.h, (margin, j, H[j], c.h)

    def test_hard_rows_equal_env_predicates_bitwise(self):
        """Predicted hard rows at the successor == what the env computes
        after it actually steps. 10k random (pose, action) pairs, each
        evaluated inside a K=137 candidate batch (Task 5's actual call shape
        for the exact tier). numpy ufunc inner loops can be SIMD-dispatched
        BY ARRAY LENGTH, so bit identity at K=1 does not logically imply bit
        identity at the batch size the gate actually uses -- this exercises
        that batch size, not just a lone candidate."""
        rng = np.random.default_rng(99)
        cfg = EnvConfig()
        env = PursuitEvasionEnv(cfg)
        shield = CBFShield(cfg)
        k_candidates = 137
        for _ in range(10_000):
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            state = np.array(env.pursuer_pose, dtype=np.float64)
            u = np.array(
                [rng.uniform(0, cfg.v_max), rng.uniform(-cfg.omega_max, cfg.omega_max)]
            )
            others = np.column_stack(
                [
                    rng.uniform(0, cfg.v_max, size=k_candidates - 1),
                    rng.uniform(-cfg.omega_max, cfg.omega_max, size=k_candidates - 1),
                ]
            )
            idx = int(rng.integers(0, k_candidates))
            batch = np.insert(others, idx, u, axis=0)
            H_batch = shield.successor_barriers(state, env.obstacles, batch)
            # The batched row must equal the K=1 evaluation of the same
            # action: a correctness lock against a numpy ufunc whose result
            # depends on array length rather than only on each element.
            H_single = shield.successor_barriers(state, env.obstacles, u[None, :])
            assert np.array_equal(H_batch[idx], H_single[0])
            H = H_batch[idx : idx + 1]
            cons = shield.constraints(state, env.obstacles)
            hard_idx = [j for j, c in enumerate(cons) if c.hard]
            # Step the env with the evader frozen; take the pursuer's new pose.
            _, _, _, _, info = env.step([u[0], u[1], 0.0, 0.0])
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


class _WallSeeker:
    """Adversarial nominal: always full speed, always steering toward the
    nearest wall or obstacle. The shield alone must keep the body off it."""

    def reset(self):
        pass

    def act(self, obs, env):
        x, y, th = env.pursuer_pose
        lim = env.cfg.arena_size / 2.0
        targets = [(lim, y), (-lim, y), (x, lim), (x, -lim)] + [
            (o["x"], o["y"]) for o in env.obstacles
        ]
        tx, ty = min(targets, key=lambda t: np.hypot(t[0] - x, t[1] - y))
        err = wrap_angle(np.arctan2(ty - y, tx - x) - th)
        return 1.0, float(np.clip(3.0 * err, -env.cfg.omega_max, env.cfg.omega_max))


class TestZeroContactGuarantee:
    """Spec B4: at δ = 0 the shielded role never produces an env contact step,
    for ANY nominal and ANY opponent. Not the gate (that is N = 1000 on the
    pre-registered streams); a local witness on unrelated seeds."""

    @pytest.mark.parametrize("seed", range(900000, 900040))
    def test_wall_seeker_never_contacts(self, seed):
        env = PursuitEvasionEnv(CFG)
        log = run_shielded_duel(
            env, _WallSeeker(), GreedyEvader(), CBFShield(CFG), seed
        )
        assert log["violations"] == 0, log["shield"]["tier_histogram"]
        assert log["shield"]["tier_histogram"]["backup"] == 0

    @pytest.mark.parametrize("seed", range(900100, 900130))
    def test_greedy_vs_hugger_never_contacts(self, seed):
        from pe2.scripted import ObstacleHuggingEvader

        env = PursuitEvasionEnv(CFG)
        log = run_shielded_duel(
            env, GreedyPursuer(), ObstacleHuggingEvader(), CBFShield(CFG), seed
        )
        assert log["violations"] == 0

    def test_wall_seeker_does_contact_without_the_shield(self):
        """Negative control for the property: the adversarial nominal is real."""
        env = PursuitEvasionEnv(CFG)
        hits = sum(
            run_shielded_duel(env, _WallSeeker(), GreedyEvader(), NoOpShield(CFG), s)[
                "violations"
            ]
            > 0
            for s in range(900000, 900010)
        )
        assert hits >= 8


# ---------------------------------------------------------------------------
# S11 (docs/pathway/second_filter_prereg_2026-09-18.md): PredictiveShield --
# the second hard filter. Its acceptance test is an exact lookahead through
# the env's own `pursuer_step`, so these tests check the IMPLEMENTATION, not a
# modelling assumption (spec section 1).
# ---------------------------------------------------------------------------

P_LIM = CFG.arena_size / 2.0 - CFG.agent_radius  # the env's inset bound, 4.85


def _bf_clearance(pose, obstacles):
    """Body-to-surface distance, re-derived through a DIFFERENT code path:
    the base class's own hard rows at margin 0. Independent of
    `PredictiveShield._body_clearance`, which the filter uses."""
    cons = CBFShield(CFG, margin=0.0).constraints(np.asarray(pose), obstacles)
    return min(c.clearance for c in cons if c.hard)


def _bf_accept(u, state, obstacles, horizon, margin=0.0):
    """Brute-force re-simulation of the acceptance test, written independently
    of the class: hold u for `horizon` steps, then one full-stop step; reject
    on any contact flag or any body clearance below `margin`."""
    from pe2.env import pursuer_step

    pose = np.asarray(state, dtype=float)
    lim = (-P_LIM, P_LIM)
    for k in range(horizon + 1):
        v, w = (float(u[0]), float(u[1])) if k < horizon else (0.0, 0.0)
        pose, wall, hit = pursuer_step(pose, v, w, CFG, obstacles, lim, lim)
        if wall or hit:
            return False
        if _bf_clearance(pose, obstacles) < margin:
            return False
    return True


def _random_obstacles(rng, n=2):
    return [
        {
            "x": float(rng.uniform(-3.5, 3.5)),
            "y": float(rng.uniform(-3.5, 3.5)),
            "radius": float(rng.uniform(0.3, 1.0)),
        }
        for _ in range(n)
    ]


def _random_free_state(rng, obstacles):
    """A contact-free pose (clearance >= 0), uniformly rejection-sampled."""
    while True:
        s = np.array(
            [
                rng.uniform(-P_LIM, P_LIM),
                rng.uniform(-P_LIM, P_LIM),
                rng.uniform(-np.pi, np.pi),
            ]
        )
        if _bf_clearance(s, obstacles) >= 0.0:
            return s


def _search_mdp_roundtrip(v, w, v_max=None):
    """The float32 action round-trip a command really makes on its way to the
    env: `SearchMDP._step_inner` normalises it into a float32 Box action and
    `SingleAgentWrapper._scale_action` clips and de-normalises it again.
    Written out here from those two functions, independent of `_snap`."""
    v_max = CFG.v_max if v_max is None else v_max
    a = np.array([2.0 * v / v_max - 1.0, w / CFG.omega_max], dtype=np.float32)
    a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
    return (a[0] + 1.0) / 2.0 * v_max, a[1] * CFG.omega_max


class TestPredictiveShieldDegenerate:
    """V-P (i): at H = 0 the filter is a clip and nothing else."""

    def test_horizon_zero_returns_the_clipped_nominal(self):
        from pe2.shield import PredictiveShield

        rng = np.random.default_rng(11000)
        shield = PredictiveShield(CFG, horizon=0)
        for _ in range(500):
            obstacles = _random_obstacles(rng)
            state = _random_free_state(rng, obstacles)
            nom = (rng.uniform(-0.5, 1.5), rng.uniform(-4.0, 4.0))
            u, info = shield.filter(nom, state, obstacles)
            clipped = (
                float(np.clip(nom[0], 0.0, CFG.v_max)),
                float(np.clip(nom[1], -CFG.omega_max, CFG.omega_max)),
            )
            # Exactly the clipped nominal snapped to the float32 action grid
            # (the grid the env is driven on -- see `_snap`); off the raw
            # float64 nominal by at most one grid step, 1.5e-8.
            assert tuple(u) == shield._snap(*clipped, CFG.v_max)
            assert u[0] == pytest.approx(clipped[0], abs=2e-8)
            assert u[1] == pytest.approx(clipped[1], abs=1e-7)
            assert info["intervention"] is False

    def test_horizon_zero_is_bit_exact_on_an_already_snapped_nominal(self):
        """V-P (i) verbatim on the path the sweep uses: a nominal that came
        through the float32 action grid (every nominal `SearchMDP` hands the
        filter) is returned bit for bit, because `_snap` is idempotent."""
        from pe2.shield import PredictiveShield

        rng = np.random.default_rng(11007)
        shield = PredictiveShield(CFG, horizon=0)
        obstacles = [{"x": 1.0, "y": 0.5, "radius": 0.6}]
        for _ in range(500):
            state = _random_free_state(rng, obstacles)
            nom = _search_mdp_roundtrip(
                rng.uniform(0.0, CFG.v_max), rng.uniform(-CFG.omega_max, CFG.omega_max)
            )
            u, _info = shield.filter(nom, state, obstacles)
            assert (u[0], u[1]) == nom


class TestPredictiveShieldExactness:
    """V-P (iii): the filter's accept decision IS the env's own rollout."""

    def test_accept_equals_brute_force_resimulation(self):
        from pe2.shield import PredictiveShield

        rng = np.random.default_rng(11001)
        shield = PredictiveShield(CFG, horizon=4)
        n_true = n_false = 0
        for _ in range(1000):
            obstacles = _random_obstacles(rng)
            # Uniform over the WHOLE arena, so overlapping/out-of-bounds
            # starts exercise the contact branches of pursuer_step too.
            state = np.array(
                [
                    rng.uniform(-P_LIM - 0.1, P_LIM + 0.1),
                    rng.uniform(-P_LIM - 0.1, P_LIM + 0.1),
                    rng.uniform(-np.pi, np.pi),
                ]
            )
            u = np.array(
                [
                    rng.uniform(0.0, CFG.v_max),
                    rng.uniform(-CFG.omega_max, CFG.omega_max),
                ]
            )
            got = shield._accept(u, state, obstacles)
            assert got == _bf_accept(u, state, obstacles, 4)
            n_true += int(got)
            n_false += int(not got)
        assert n_true > 50 and n_false > 50  # both branches really exercised

    def test_body_clearance_equals_the_base_class_hard_rows(self):
        from pe2.shield import PredictiveShield

        rng = np.random.default_rng(11002)
        shield = PredictiveShield(CFG, horizon=4)
        for _ in range(200):
            obstacles = _random_obstacles(rng)
            state = _random_free_state(rng, obstacles)
            assert shield._body_clearance(state, obstacles) == pytest.approx(
                _bf_clearance(state, obstacles)
            )

    def test_margin_is_the_clearance_threshold_not_a_second_inset(self):
        from pe2.shield import PredictiveShield

        obstacles = []
        # Facing the +x wall from 0.10 m away: at delta = 0 a full stop is
        # accepted (clearance 0.10 >= 0); at delta = 0.2 it is not.
        state = np.array([P_LIM - 0.10, 0.0, 0.0])
        assert PredictiveShield(CFG, horizon=4, margin=0.0)._accept(
            (0.0, 0.0), state, obstacles
        )
        assert not PredictiveShield(CFG, horizon=4, margin=0.2)._accept(
            (0.0, 0.0), state, obstacles
        )
        # ... and delta = 0.05 accepts, which a second inset (2 * delta) would
        # not: the body clearance is 0.10, so only a single application passes.
        assert PredictiveShield(CFG, horizon=4, margin=0.05)._accept(
            (0.0, 0.0), state, obstacles
        )


class TestPredictiveShieldExecutesWhatItSimulates:
    """FINDING 1: the returned command must survive the float32 action grid
    the env is driven on, or the executed step is not the simulated one."""

    def test_returned_action_is_a_fixed_point_of_the_env_action_roundtrip(self):
        from pe2.shield import PredictiveShield

        rng = np.random.default_rng(11008)
        shield = PredictiveShield(CFG, horizon=4)
        for _ in range(1000):
            obstacles = _random_obstacles(rng)
            state = _random_free_state(rng, obstacles)
            nom = (rng.uniform(0.0, CFG.v_max), rng.uniform(-3.0, 3.0))
            u, _info = shield.filter(nom, state, obstacles)
            back = _search_mdp_roundtrip(u[0], u[1])
            assert u[0] == back[0]  # bitwise, both components
            assert u[1] == back[1]

    def test_the_roundtrip_is_not_free_without_the_snap(self):
        """Negative control: the same round-trip really does move an
        unsnapped command (otherwise the test above is vacuous)."""
        rng = np.random.default_rng(11009)
        moved = 0
        for _ in range(1000):
            v, w = rng.uniform(0.0, CFG.v_max), rng.uniform(-3.0, 3.0)
            back = _search_mdp_roundtrip(v, w)
            moved += int(back[0] != v or back[1] != w)
        assert moved > 990

    def test_snap_is_idempotent_and_keeps_the_box(self):
        from pe2.shield import PredictiveShield

        shield = PredictiveShield(CFG, horizon=4)
        rng = np.random.default_rng(11010)
        for v_max in (CFG.v_max, CFG.v_max * CFG.evader_speed_ratio):
            for v, w in [
                (0.0, 0.0),
                (v_max, CFG.omega_max),
                (v_max, -CFG.omega_max),
                *zip(
                    rng.uniform(0.0, v_max, 200),
                    rng.uniform(-CFG.omega_max, CFG.omega_max, 200),
                ),
            ]:
                q = shield._snap(v, w, v_max)
                assert shield._snap(*q, v_max) == q
                assert 0.0 <= q[0] <= v_max and abs(q[1]) <= CFG.omega_max
        # The box endpoints are exact fixed points, so no speed cap is lost.
        assert shield._snap(CFG.v_max, 0.0, CFG.v_max)[0] == CFG.v_max
        assert shield._snap(0.0, 0.0, CFG.v_max)[0] == 0.0

    def test_snap_uses_the_callers_speed_cap_not_the_pursuers(self):
        """A shielded evader (run_shielded_duel, cap evader_v_max) must not be
        snapped onto the pursuer's grid -- that would cap it at v_max."""
        from pe2.shield import PredictiveShield

        shield = PredictiveShield(CFG, horizon=4)
        cap = CFG.v_max * CFG.evader_speed_ratio
        u, _info = shield.filter((cap, 0.0), np.array([0.0, 0.0, 0.0]), [], v_max=cap)
        assert u[0] == cap


class TestPredictiveShieldBounds:
    def test_sim_bounds_are_the_envs_bounds(self):
        """FINDING 3: the rollout must clip where the env clips, bitwise."""
        from pe2.shield import PredictiveShield

        for cfg in (CFG, EnvConfig(arena_size=7.3, agent_radius=0.22)):
            env = PursuitEvasionEnv(cfg)
            shield = PredictiveShield(cfg, horizon=4)
            assert shield._sim_x_lim == (env._x_min, env._x_max)
            assert shield._sim_y_lim == (env._y_min, env._y_max)

    def test_horizon_must_be_a_non_negative_int(self):
        from pe2.shield import PredictiveShield

        for bad in (-1, 0.2, 4.0, "4", None, True):
            with pytest.raises(ValueError, match="horizon"):
                PredictiveShield(CFG, bad)
        for good in (0, 4, np.int64(8)):
            assert PredictiveShield(CFG, good).horizon == int(good)


class TestPredictiveShieldBackupOrdering:
    """FINDING 5: a candidate whose rollout contacted must never be preferred
    over a contact-free one at equal worst clearance (both read 0, because the
    env projects a contacting body back onto the surface)."""

    def test_contact_free_beats_contacting_at_equal_clearance(self):
        from pe2.shield import _backup_choice

        contact_free = np.array([False, True])
        worst = np.array([0.0, 0.0])
        obj = np.array([0.0, 99.0])  # the contacting one is nearer the nominal
        assert _backup_choice(contact_free, worst, obj) == 1

    def test_clearance_outranks_the_nominal_and_the_nominal_breaks_ties(self):
        from pe2.shield import _backup_choice

        cf = np.array([True, True, True])
        assert (
            _backup_choice(cf, np.array([0.1, 0.3, 0.2]), np.array([0.0, 9.0, 1.0]))
            == 1
        )
        assert (
            _backup_choice(cf, np.array([0.3, 0.3, 0.3]), np.array([5.0, 9.0, 1.0]))
            == 2
        )

    def test_a_contacting_candidate_wins_only_on_clearance(self):
        from pe2.shield import _backup_choice

        # Contact-free is primary even when the contacting rollout kept more
        # clearance on average -- it is the MINIMUM that ranks, and a contact
        # pins that to 0 or below.
        assert (
            _backup_choice(
                np.array([False, True]), np.array([0.5, 0.4]), np.array([0.0, 0.0])
            )
            == 1
        )


class TestPredictiveShieldTiers:
    def test_zero_action_is_always_a_candidate(self):
        from pe2.shield import PredictiveShield, _qp_candidates

        shield = PredictiveShield(CFG, horizon=4)
        obstacles = [{"x": 1.0, "y": 0.2, "radius": 0.5}]
        state = np.array([0.0, 0.0, 0.0])
        cons = shield.constraints(state, obstacles)
        U = _qp_candidates(
            0.7, 0.3, shield._dcbf_rows(cons), CFG.v_max, CFG.omega_max, 150.0, 1.0
        )
        assert any(np.array_equal(u, [0.0, 0.0]) for u in U)
        assert any(np.allclose(u, [0.7, 0.3]) for u in U)
        # ... and they survive the snap and the dedup, which is the set the
        # rollout actually searches.
        v_nom, w_nom = shield._snap(0.7, 0.3, CFG.v_max)
        C = shield._candidates(v_nom, w_nom, cons, CFG.v_max)
        assert any(np.array_equal(u, [0.0, 0.0]) for u in C)
        assert any(np.array_equal(u, [v_nom, w_nom]) for u in C)
        assert len(C) == len(np.unique(C, axis=0)) < len(U)

    def test_no_intervention_in_open_space(self):
        from pe2.shield import PredictiveShield

        shield = PredictiveShield(CFG, horizon=4)
        u, info = shield.filter((0.5, 0.5), np.array([0.0, 0.0, 0.0]), [])
        assert np.allclose(u, [0.5, 0.5])
        assert info["method"] == "exact"
        assert info["feasible"] is True
        assert info["intervention"] is False

    def test_delta_zero_never_needs_the_backup_tier(self):
        """u = 0 holds the pose, so from a contact-free start it is always
        accepted at delta = 0 -- backup is unreachable and can therefore never
        execute into contact."""
        from pe2.shield import PredictiveShield

        rng = np.random.default_rng(11003)
        shield = PredictiveShield(CFG, horizon=4)
        for _ in range(300):
            obstacles = _random_obstacles(rng)
            state = _random_free_state(rng, obstacles)
            nom = (rng.uniform(0.0, CFG.v_max), rng.uniform(-3.0, 3.0))
            _u, info = shield.filter(nom, state, obstacles)
            assert info["method"] == "exact"

    def test_wedged_state_falls_back_to_a_full_stop(self):
        """Pressed against the +x wall inside a delta = 0.2 keep-out and
        heading into it: every candidate with v > 0 loses clearance on its
        first step (the Euler step translates along the CURRENT heading), so
        nothing is acceptable and the backup tier must hold the pose."""
        from pe2.shield import PredictiveShield

        shield = PredictiveShield(CFG, horizon=4, margin=0.2)
        state = np.array([P_LIM - 0.10, 0.0, 0.0])
        u, info = shield.filter((CFG.v_max, 0.0), state, [])
        assert info["method"] == "backup"
        assert info["feasible"] is False
        assert u[0] == 0.0
        assert info["v_exec"] == 0.0
        assert info["intervention"] is True

    def test_backup_never_executes_into_contact(self):
        """The executed action of the backup tier, applied by the env itself,
        never fires a contact flag from a contact-free start."""
        from pe2.env import pursuer_step
        from pe2.shield import PredictiveShield

        rng = np.random.default_rng(11004)
        shield = PredictiveShield(CFG, horizon=4, margin=0.3)
        seen_backup = 0
        for _ in range(200):
            obstacles = _random_obstacles(rng)
            state = _random_free_state(rng, obstacles)
            u, info = shield.filter(
                (rng.uniform(0.0, CFG.v_max), rng.uniform(-3.0, 3.0)), state, obstacles
            )
            seen_backup += int(info["method"] == "backup")
            _p, wall, hit = pursuer_step(
                state, u[0], u[1], CFG, obstacles, (-P_LIM, P_LIM), (-P_LIM, P_LIM)
            )
            assert not (wall or hit)
        assert seen_backup > 0  # the tier really was exercised


class TestPredictiveShieldInterface:
    def test_info_keys_match_the_cbf_shield(self):
        from pe2.shield import PredictiveShield

        rng = np.random.default_rng(11005)
        for _ in range(20):
            obstacles = _random_obstacles(rng)
            state = _random_free_state(rng, obstacles)
            nom = (rng.uniform(0.0, CFG.v_max), rng.uniform(-3.0, 3.0))
            _u, pinfo = PredictiveShield(CFG, horizon=4).filter(nom, state, obstacles)
            _v, cinfo = CBFShield(CFG).filter(nom, state, obstacles)
            assert pinfo.keys() == cinfo.keys()
            assert {k: type(v) for k, v in pinfo.items()} == {
                k: type(v) for k, v in cinfo.items()
            }

    def test_deterministic(self):
        from pe2.shield import PredictiveShield

        obstacles = [{"x": 1.0, "y": 0.3, "radius": 0.6}]
        state = np.array([0.0, 0.1, 0.2])
        a = PredictiveShield(CFG, horizon=4).filter((1.0, 0.5), state, obstacles)
        b = PredictiveShield(CFG, horizon=4).filter((1.0, 0.5), state, obstacles)
        assert np.array_equal(a[0], b[0]) and a[1] == b[1]


class TestPredictiveShieldGuarantee:
    """V-P (ii): the env's own contact flag never fires under the filter."""

    def test_random_pursuer_in_search_mdp_never_contacts(self):
        from pe2.scripted import GreedyEvader
        from pe2.shield import PredictiveShield
        from pe2.structured import SearchMDP

        cfg = EnvConfig(max_steps=200)
        mdp = SearchMDP(
            cfg, GreedyEvader, seed0=0, shield=PredictiveShield(cfg, horizon=4)
        )
        rng = np.random.default_rng(11006)
        for seed in range(1_200_000, 1_200_050):
            mdp.reset(seed=seed)
            done = False
            while not done:
                _o, _r, term, trunc, _i = mdp.step(rng.uniform(-1.0, 1.0, size=2))
                done = term or trunc
        assert mdp.env_steps > 2000  # the episodes really ran
        assert mdp.violations_total == 0

    def test_the_same_random_pursuer_does_contact_without_the_filter(self):
        """Negative control for the guarantee: on exactly the same 50 seeds and
        the same action stream, the unfiltered pursuer contacts constantly."""
        from pe2.scripted import GreedyEvader
        from pe2.structured import SearchMDP

        cfg = EnvConfig(max_steps=200)
        mdp = SearchMDP(cfg, GreedyEvader, seed0=0, shield=None)
        rng = np.random.default_rng(11006)
        for seed in range(1_200_000, 1_200_050):
            mdp.reset(seed=seed)
            done = False
            while not done:
                _o, _r, term, trunc, _i = mdp.step(rng.uniform(-1.0, 1.0, size=2))
                done = term or trunc
        assert mdp.violations_total > 500


class _ReferencePredictiveShield:
    """The unoptimised selection, written out separately: the FULL snapped
    candidate grid with its duplicates, every candidate rolled out to the end
    with no early exit, argmin over the accepted set. `filter` must return the
    same command as this for every state -- dedup and early exit are allowed to
    change only the work done."""

    def __init__(self, cfg, horizon, margin=0.0):
        from pe2.shield import PredictiveShield

        self.s = PredictiveShield(cfg, horizon=horizon, margin=margin)

    def filter(self, action, state, obstacles, v_max=None):
        from pe2.shield import _backup_choice, _qp_candidates

        s = self.s
        v_max = s.cfg.v_max if v_max is None else v_max
        v_nom, w_nom = s._snap(*s._clip_nominal(action, v_max), v_max)
        cons = s.constraints(state, obstacles)
        U = _qp_candidates(
            v_nom,
            w_nom,
            s._dcbf_rows(cons),
            v_max,
            s.cfg.omega_max,
            s.w_v,
            s.w_omega,
        )
        for i in range(len(U)):
            U[i] = s._snap(U[i, 0], U[i, 1], v_max)
        obj = s.w_v * (U[:, 0] - v_nom) ** 2 + s.w_omega * (U[:, 1] - w_nom) ** 2
        rolled = [s._rollout(u, state, obstacles) for u in U]
        ok = np.array([cf and worst >= s.margin for cf, worst in rolled])
        if ok.any():
            pool = np.flatnonzero(ok)
            return U[int(pool[np.argmin(obj[pool])])].copy(), "exact"
        idx = _backup_choice(
            np.array([r[0] for r in rolled]), np.array([r[1] for r in rolled]), obj
        )
        return U[idx].copy(), "backup"


class TestPredictiveShieldOptimisationsArePureSpeedups:
    """FINDING 6: dedup + two-phase rollout must not move a single command."""

    def test_returned_action_matches_the_unoptimised_reference_at_h16(self):
        from pe2.shield import PredictiveShield

        rng = np.random.default_rng(11011)
        fast = PredictiveShield(CFG, horizon=16)
        ref = _ReferencePredictiveShield(CFG, horizon=16)
        n_intervened = 0
        for i in range(500):
            obstacles = _random_obstacles(rng)
            if i % 2:
                state = _random_free_state(rng, obstacles)  # nominal usually safe
            else:
                # Pressed against the +x wall heading into it: the nominal is
                # rejected and the search really walks the candidate grid.
                state = np.array(
                    [P_LIM - rng.uniform(0.0, 0.4), rng.uniform(-3.0, 3.0), 0.0]
                )
                obstacles = []
            nom = (rng.uniform(0.0, CFG.v_max), rng.uniform(-3.0, 3.0))
            u, info = fast.filter(nom, state, obstacles)
            u_ref, method_ref = ref.filter(nom, state, obstacles)
            assert np.array_equal(u, u_ref)
            assert info["method"] == method_ref
            n_intervened += int(info["intervention"])
        assert n_intervened > 100  # the braking regime was really exercised

    def test_returned_action_matches_the_reference_on_the_backup_tier(self):
        from pe2.shield import PredictiveShield

        rng = np.random.default_rng(11012)
        fast = PredictiveShield(CFG, horizon=4, margin=0.3)
        ref = _ReferencePredictiveShield(CFG, horizon=4, margin=0.3)
        n_backup = 0
        for _ in range(200):
            obstacles = _random_obstacles(rng)
            state = _random_free_state(rng, obstacles)
            nom = (rng.uniform(0.0, CFG.v_max), rng.uniform(-3.0, 3.0))
            u, info = fast.filter(nom, state, obstacles)
            u_ref, method_ref = ref.filter(nom, state, obstacles)
            assert np.array_equal(u, u_ref)
            assert info["method"] == method_ref
            n_backup += int(info["method"] == "backup")
        assert n_backup > 0

    def test_accept_agrees_with_the_full_rollout(self):
        """The early-exiting predicate and the full ranking rollout are two
        loops over the same steps; bind them so they cannot drift."""
        from pe2.shield import PredictiveShield

        rng = np.random.default_rng(11013)
        for margin in (0.0, 0.25):
            shield = PredictiveShield(CFG, horizon=4, margin=margin)
            for _ in range(300):
                obstacles = _random_obstacles(rng)
                state = np.array(
                    [
                        rng.uniform(-P_LIM - 0.1, P_LIM + 0.1),
                        rng.uniform(-P_LIM - 0.1, P_LIM + 0.1),
                        rng.uniform(-np.pi, np.pi),
                    ]
                )
                u = (
                    rng.uniform(0.0, CFG.v_max),
                    rng.uniform(-CFG.omega_max, CFG.omega_max),
                )
                contact_free, worst = shield._rollout(u, state, obstacles)
                assert shield._accept(u, state, obstacles) == (
                    contact_free and worst >= margin
                )


# --- S16 Task 1: RecoveryShield + the escape-direction feature ------------
#
# Hand-checkable poses, all with NO obstacles so every number below follows
# from the four wall rows alone (x_lim = y_lim = 4.85).
#
# STALL_POSE = (4.805, 0, +1.2): the body row (h = 0.045, a_omega = 0) is
# satisfied at v = 0, and the soft +x wall row binds:
#     h   = x_lim - (x + d cos th) = 0.045 - 0.1 cos 1.2 = 0.0087642
#     c1  = dt * d * sin th        = 0.05 * 0.1 * sin 1.2 = 0.00466020
#     c2  = -gamma * h             = -0.2 * 0.0087642     = -0.00175284
#     c1 * omega >= c2  =>  omega >= c2 / c1 = -0.3761312
# so turning further INTO the wall is blocked and turning away is free. The
# mirror pose (theta = -1.2) blocks the other sign. A full-speed nominal is
# immobile at both.
STALL_POSE = np.array([4.805, 0.0, 1.2])
STALL_POSE_MIRROR = np.array([4.805, 0.0, -1.2])
STALL_OMEGA_LO = -0.37613120240498926  # rad/s, the bound derived above
FREE_POSE = np.array([0.0, 0.0, 0.0])  # nothing within 4.85 m: nothing binds
# Nose into the corner: every pure turn is refused as well, so the recovery
# has nothing better than the normal result to offer.
CORNER_POSE = np.array([4.9, 4.9, np.pi / 4])
FULL_SPEED = (CFG.v_max, 0.0)


class TestEscapeFeatures:
    def test_bounds_are_the_dcbf_rows_at_zero_speed(self):
        from pe2.shield import escape_omega_bounds

        shield = CBFShield(CFG)
        lo, hi = escape_omega_bounds(shield, STALL_POSE, [])
        assert lo == pytest.approx(STALL_OMEGA_LO)
        assert hi == CFG.omega_max  # the box edge: nothing binds on that side

    def test_features_are_the_bounds_normalised_by_omega_max(self):
        from pe2.shield import escape_features

        f = escape_features(CBFShield(CFG), STALL_POSE, [])
        assert f.shape == (2,) and f.dtype == np.float32
        np.testing.assert_allclose(
            f, [STALL_OMEGA_LO / CFG.omega_max, 1.0], rtol=0, atol=1e-7
        )

    def test_feature_sign_reflects_which_turn_the_filter_admits(self):
        from pe2.shield import escape_features

        shield = CBFShield(CFG)
        for pose, blocked_sign in ((STALL_POSE, -1.0), (STALL_POSE_MIRROR, +1.0)):
            f = escape_features(shield, pose, [])
            # the blocked side is pulled in off the box edge, the free side is not
            if blocked_sign < 0:
                assert -1.0 < f[0] < 0.0 and f[1] == pytest.approx(1.0)
            else:
                assert 0.0 < f[1] < 1.0 and f[0] == pytest.approx(-1.0)
            # and the filter agrees: the free sign executes the larger |omega|
            up, _ = shield.filter((0.0, CFG.omega_max), pose, [], v_max=CFG.v_max)
            um, _ = shield.filter((0.0, -CFG.omega_max), pose, [], v_max=CFG.v_max)
            free = um if blocked_sign > 0 else up
            other = up if blocked_sign > 0 else um
            assert abs(free[1]) > abs(other[1])

    def test_free_pose_gives_the_whole_action_box(self):
        from pe2.shield import escape_features

        np.testing.assert_allclose(
            escape_features(CBFShield(CFG), FREE_POSE, []), [-1.0, 1.0]
        )

    def test_infeasible_rows_give_zero_zero(self):
        from pe2.shield import escape_features, escape_omega_bounds

        # Body on the +x wall heading straight at it: the soft row has
        # a_omega = d sin(0) = 0 and h = -0.1 < 0, so c1 = 0 while c2 > 0 --
        # no omega satisfies it at v = 0.
        pose = np.array([4.85, 0.0, 0.0])
        assert escape_omega_bounds(CBFShield(CFG), pose, []) is None
        np.testing.assert_allclose(
            escape_features(CBFShield(CFG), pose, []), [0.0, 0.0]
        )

    def test_no_shield_gives_the_whole_action_box(self):
        from pe2.shield import escape_features

        np.testing.assert_allclose(escape_features(None, STALL_POSE, []), [-1.0, 1.0])

    def test_features_stay_inside_the_unit_interval_everywhere(self):
        from pe2.shield import escape_features

        rng = np.random.default_rng(16_001)
        shield = CBFShield(CFG)
        seen_narrow = 0
        for _ in range(400):
            obstacles = _random_obstacles(rng)
            state = _random_free_state(rng, obstacles)
            f = escape_features(shield, state, obstacles)
            assert f.shape == (2,) and np.all(f >= -1.0) and np.all(f <= 1.0)
            seen_narrow += int(f[0] > -1.0 or f[1] < 1.0)
        assert seen_narrow > 0  # the feature is not a constant


class _RecoveryCounter:
    """Tallies `recovery_fired` across a whole evaluation without adding any
    counter to the library class."""

    def __init__(self, shield):
        self.shield = shield
        self.fired = 0
        self.moving_recoveries = 0
        self._filter = shield.filter
        shield.filter = self._wrapped

    def _wrapped(self, action, state, obstacles, v_max=None):
        u, info = self._filter(action, state, obstacles, v_max)
        if info["recovery_fired"]:
            self.fired += 1
            self.moving_recoveries += int(float(u[0]) != 0.0)
        return u, info


class TestRecoveryShield:
    def test_fires_at_exactly_the_twentieth_immobile_step(self):
        from pe2.shield import RECOVERY_AFTER, RecoveryShield

        assert RECOVERY_AFTER == 20
        shield = RecoveryShield(CFG)
        for step in range(1, 20):
            u, info = shield.filter(FULL_SPEED, STALL_POSE, [], v_max=CFG.v_max)
            assert info["immobile"], f"step {step} was not immobile"
            assert info["recovery_fired"] is False, f"fired early at step {step}"
            np.testing.assert_allclose(u, [0.0, 0.0], atol=1e-12)
        u, info = shield.filter(FULL_SPEED, STALL_POSE, [], v_max=CFG.v_max)
        assert info["recovery_fired"] is True
        assert u[1] == pytest.approx(CFG.omega_max)

    def test_recovery_after_is_configurable(self):
        from pe2.shield import RecoveryShield

        shield = RecoveryShield(CFG, recovery_after=3)
        fired = [
            shield.filter(FULL_SPEED, STALL_POSE, [], v_max=CFG.v_max)[1][
                "recovery_fired"
            ]
            for _ in range(4)
        ]
        assert fired == [False, False, True, True]

    def test_picks_the_sign_the_filter_admits(self):
        from pe2.shield import RecoveryShield

        for pose, sign in ((STALL_POSE, +1.0), (STALL_POSE_MIRROR, -1.0)):
            shield = RecoveryShield(CFG, recovery_after=1)
            u, info = shield.filter(FULL_SPEED, pose, [], v_max=CFG.v_max)
            assert info["recovery_fired"] is True
            assert u[1] == pytest.approx(sign * CFG.omega_max)

    def test_counter_resets_on_a_step_that_moves(self):
        from pe2.shield import RecoveryShield

        shield = RecoveryShield(CFG, recovery_after=5)
        for _ in range(4):
            assert not shield.filter(FULL_SPEED, STALL_POSE, [], v_max=CFG.v_max)[1][
                "recovery_fired"
            ]
        _, info = shield.filter(FULL_SPEED, FREE_POSE, [], v_max=CFG.v_max)
        assert info["immobile"] is False and info["recovery_fired"] is False
        # the run restarted: four more immobile steps must still not fire
        for _ in range(4):
            assert not shield.filter(FULL_SPEED, STALL_POSE, [], v_max=CFG.v_max)[1][
                "recovery_fired"
            ]
        assert shield.filter(FULL_SPEED, STALL_POSE, [], v_max=CFG.v_max)[1][
            "recovery_fired"
        ]

    def test_falls_back_to_the_normal_result_when_no_turn_is_admissible(self):
        from pe2.shield import RECOVERY_OMEGA_MIN, RecoveryShield

        assert RECOVERY_OMEGA_MIN == 0.1
        shield = RecoveryShield(CFG, recovery_after=1)
        ref, _ = CBFShield(CFG).filter(FULL_SPEED, CORNER_POSE, [], v_max=CFG.v_max)
        u, info = shield.filter(FULL_SPEED, CORNER_POSE, [], v_max=CFG.v_max)
        assert abs(u[1]) <= RECOVERY_OMEGA_MIN
        assert info["recovery_fired"] is False
        assert np.array_equal(u, ref)

    def test_recovery_step_still_reports_the_policys_own_nominal(self):
        from pe2.shield import RecoveryShield

        shield = RecoveryShield(CFG, recovery_after=1)
        _, info = shield.filter(FULL_SPEED, STALL_POSE, [], v_max=CFG.v_max)
        assert info["recovery_fired"] is True
        # the recovery must not launder the stall away: the policy still asked
        # for full speed and still got none of it
        assert info["v_nom"] == pytest.approx(CFG.v_max)
        assert info["v_exec"] == pytest.approx(0.0)
        assert info["immobile"] is True and info["stalled"] is True
        assert info["intervention"] is True

    def test_info_keys_match_the_base_shield_plus_recovery_fired(self):
        from pe2.shield import RecoveryShield

        _, base = CBFShield(CFG).filter(FULL_SPEED, STALL_POSE, [], v_max=CFG.v_max)
        _, info = RecoveryShield(CFG, recovery_after=1).filter(
            FULL_SPEED, STALL_POSE, [], v_max=CFG.v_max
        )
        assert set(info) == set(base) | {"recovery_fired"}

    def test_identical_to_cbf_shield_until_the_threshold_is_reached(self):
        from pe2.shield import RecoveryShield

        rng = np.random.default_rng(16_002)
        base = CBFShield(CFG)
        shield = RecoveryShield(CFG, recovery_after=10**9)
        n_immobile = 0
        for _ in range(500):
            obstacles = _random_obstacles(rng)
            state = _random_free_state(rng, obstacles)
            nom = (
                rng.uniform(0.0, CFG.v_max),
                rng.uniform(-CFG.omega_max, CFG.omega_max),
            )
            u_ref, info_ref = base.filter(nom, state, obstacles, v_max=CFG.v_max)
            u, info = shield.filter(nom, state, obstacles, v_max=CFG.v_max)
            assert np.array_equal(u, u_ref)
            assert info.pop("recovery_fired") is False
            assert info == info_ref
            n_immobile += int(info_ref["immobile"])
        assert n_immobile > 0  # the counter really was being exercised

    def test_zero_violations_over_fifty_random_episodes(self):
        from pe2.scripted import GreedyEvader
        from pe2.shield import RecoveryShield
        from pe2.structured import SearchMDP

        class Straight:  # drives at walls, so the filter stalls it constantly
            def predict(self, obs, deterministic=True):
                return np.array([1.0, 0.0], dtype=np.float32), None

        shield = RecoveryShield(CFG)
        spy = _RecoveryCounter(shield)
        mdp = SearchMDP(CFG, GreedyEvader, hold=10, shield=shield)
        for i in range(50):
            log = mdp.play_episode(Straight().predict, 1_730_000 + i)
            assert log["violations"] == 0
        assert mdp.violations_total == 0
        assert spy.fired > 0  # the zero above is not vacuous
        assert spy.moving_recoveries == 0  # v = 0: the recovery never translates

    def test_the_counter_spans_episodes_because_the_shield_does(self):
        # Documented, deliberate: a shield carries no episode hook, so the
        # immobile run continues across an episode boundary. Pinned here so a
        # caller that needs per-episode counting builds a fresh shield.
        from pe2.shield import RecoveryShield

        shield = RecoveryShield(CFG, recovery_after=3)
        shield.filter(FULL_SPEED, STALL_POSE, [], v_max=CFG.v_max)
        shield.filter(FULL_SPEED, STALL_POSE, [], v_max=CFG.v_max)
        assert shield._immobile_run == 2
        assert shield.filter(FULL_SPEED, STALL_POSE, [], v_max=CFG.v_max)[1][
            "recovery_fired"
        ]
