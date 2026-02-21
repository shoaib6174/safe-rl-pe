"""Tests for Phase 2 safe action bound computation.

Tests cover:
- LP-based safe bounds (compute_safe_bounds_lp)
- Analytical safe bounds (compute_safe_bounds_analytical)
- VCPCBFFilter.compute_safe_bounds() integration
- Consistency between LP and analytical methods
"""

import numpy as np
import pytest

from safety.vcp_cbf import (
    VCPCBFFilter,
    compute_safe_bounds_analytical,
    compute_safe_bounds_lp,
)


class TestComputeSafeBoundsLP:
    """Tests for LP-based safe bounds."""

    def test_no_constraints_returns_nominal(self):
        """Empty constraints should return nominal bounds."""
        v, omega, feasible = compute_safe_bounds_lp([], alpha=1.0)
        assert v == (0.0, 1.0)
        assert omega == (-2.84, 2.84)
        assert feasible is True

    def test_single_v_constraint_tightens_v_max(self):
        """A constraint limiting forward speed should tighten v_max."""
        # Constraint: a_v * v + a_omega * omega >= -alpha * h
        # With a_v=1, a_omega=0, h=0.5, alpha=1:  v >= -0.5 (no tightening)
        # With a_v=-1, a_omega=0, h=0.5, alpha=1: -v >= -0.5 → v <= 0.5
        constraints = [(0.5, -1.0, 0.0)]  # h=0.5, a_v=-1, a_omega=0
        v, omega, feasible = compute_safe_bounds_lp(constraints, alpha=1.0)
        assert feasible is True
        assert v[1] <= 0.5 + 1e-6, f"v_max should be ~0.5, got {v[1]}"
        assert omega == pytest.approx((-2.84, 2.84), abs=1e-6)

    def test_single_omega_constraint_tightens_omega(self):
        """A constraint limiting angular velocity should tighten omega bounds."""
        # a_omega=-1, a_v=0, h=1.0, alpha=1: -omega >= -1 → omega <= 1
        constraints = [(1.0, 0.0, -1.0)]
        v, omega, feasible = compute_safe_bounds_lp(constraints, alpha=1.0)
        assert feasible is True
        assert omega[1] <= 1.0 + 1e-6, f"omega_max should be ~1.0, got {omega[1]}"

    def test_infeasible_constraints(self):
        """Contradictory constraints should return infeasible."""
        # v >= 2.0 (infeasible with v_max=1.0)
        constraints = [(-2.0, 1.0, 0.0)]  # h=-2, a_v=1 -> v >= 2
        v, omega, feasible = compute_safe_bounds_lp(constraints, alpha=1.0)
        assert feasible is False

    def test_bounds_within_nominal(self):
        """Safe bounds should always be within nominal bounds."""
        constraints = [(0.3, -1.0, 0.5), (0.5, 0.3, -1.0)]
        v, omega, feasible = compute_safe_bounds_lp(constraints, alpha=1.0)
        if feasible:
            assert v[0] >= 0.0 - 1e-10
            assert v[1] <= 1.0 + 1e-10
            assert omega[0] >= -2.84 - 1e-10
            assert omega[1] <= 2.84 + 1e-10

    def test_multiple_constraints_intersection(self):
        """Multiple constraints should intersect to tighten bounds."""
        constraints = [
            (0.5, -1.0, 0.0),   # v <= 0.5
            (1.0, 0.0, -1.0),   # omega <= 1.0
            (1.0, 0.0, 1.0),    # omega >= -1.0
        ]
        v, omega, feasible = compute_safe_bounds_lp(constraints, alpha=1.0)
        assert feasible is True
        assert v[1] <= 0.5 + 1e-6
        assert omega[0] >= -1.0 - 1e-6
        assert omega[1] <= 1.0 + 1e-6

    def test_custom_bounds(self):
        """Should respect custom nominal bounds."""
        constraints = []
        v, omega, feasible = compute_safe_bounds_lp(
            constraints, alpha=1.0,
            v_bounds=(0.0, 2.0), omega_bounds=(-5.0, 5.0),
        )
        assert v == (0.0, 2.0)
        assert omega == (-5.0, 5.0)

    def test_alpha_scales_constraint(self):
        """Higher alpha should allow more restrictive constraints to be satisfied."""
        # h=0.1, a_v=-1, alpha=1: v <= 0.1
        # h=0.1, a_v=-1, alpha=2: v <= 0.2
        constraints = [(0.1, -1.0, 0.0)]
        v1, _, f1 = compute_safe_bounds_lp(constraints, alpha=1.0)
        v2, _, f2 = compute_safe_bounds_lp(constraints, alpha=2.0)
        assert f1 and f2
        assert v2[1] >= v1[1] - 1e-6, "Higher alpha should give wider bounds"


class TestComputeSafeBoundsAnalytical:
    """Tests for analytical safe bounds."""

    def test_no_constraints_returns_nominal(self):
        """Empty constraints should return nominal bounds."""
        v, omega, feasible = compute_safe_bounds_analytical([], alpha=1.0)
        assert v == (0.0, 1.0)
        assert omega == (-2.84, 2.84)
        assert feasible is True

    def test_single_v_constraint(self):
        """Single v constraint should tighten v bounds."""
        constraints = [(0.5, -1.0, 0.0)]
        v, omega, feasible = compute_safe_bounds_analytical(constraints, alpha=1.0)
        assert feasible is True
        assert v[1] <= 0.5 + 1e-6

    def test_single_omega_constraint(self):
        """Single omega constraint should tighten omega bounds."""
        constraints = [(1.0, 0.0, -1.0)]
        v, omega, feasible = compute_safe_bounds_analytical(constraints, alpha=1.0)
        assert feasible is True
        assert omega[1] <= 1.0 + 1e-6

    def test_infeasible_constraints(self):
        """Contradictory constraints should return infeasible."""
        constraints = [(-2.0, 1.0, 0.0)]  # v >= 2 (infeasible)
        v, omega, feasible = compute_safe_bounds_analytical(constraints, alpha=1.0)
        assert feasible is False

    def test_analytical_outer_approximation(self):
        """Analytical bounds are an outer approximation (at least as wide as LP)."""
        constraints = [
            (0.5, -1.0, 0.3),
            (0.8, 0.2, -0.5),
            (1.0, -0.5, 0.1),
        ]
        v_lp, omega_lp, f_lp = compute_safe_bounds_lp(constraints, alpha=1.0)
        v_an, omega_an, f_an = compute_safe_bounds_analytical(constraints, alpha=1.0)
        if f_lp and f_an:
            # Analytical is outer approximation: at least as wide as LP
            assert v_an[0] <= v_lp[0] + 1e-6, \
                f"Analytical v_min={v_an[0]} should be <= LP v_min={v_lp[0]}"
            assert v_an[1] >= v_lp[1] - 1e-6, \
                f"Analytical v_max={v_an[1]} should be >= LP v_max={v_lp[1]}"

    def test_bounds_within_nominal(self):
        """Safe bounds should always be within nominal bounds."""
        constraints = [(0.3, -1.0, 0.5), (0.5, 0.3, -1.0)]
        v, omega, feasible = compute_safe_bounds_analytical(constraints, alpha=1.0)
        if feasible:
            assert v[0] >= 0.0 - 1e-10
            assert v[1] <= 1.0 + 1e-10
            assert omega[0] >= -2.84 - 1e-10
            assert omega[1] <= 2.84 + 1e-10

    def test_alpha_scales_constraint(self):
        """Higher alpha should give wider bounds."""
        constraints = [(0.1, -1.0, 0.0)]
        v1, _, f1 = compute_safe_bounds_analytical(constraints, alpha=1.0)
        v2, _, f2 = compute_safe_bounds_analytical(constraints, alpha=2.0)
        assert f1 and f2
        assert v2[1] >= v1[1] - 1e-6


class TestSafeBoundsConsistency:
    """Tests for consistency between LP and analytical methods."""

    def test_uncoupled_constraints_match(self):
        """When constraints only involve one variable, LP and analytical should match."""
        constraints = [
            (0.5, -1.0, 0.0),  # Pure v constraint
            (1.0, 0.0, -1.0),  # Pure omega constraint
        ]
        v_lp, omega_lp, _ = compute_safe_bounds_lp(constraints, alpha=1.0)
        v_an, omega_an, _ = compute_safe_bounds_analytical(constraints, alpha=1.0)
        np.testing.assert_allclose(v_lp, v_an, atol=1e-6)
        np.testing.assert_allclose(omega_lp, omega_an, atol=1e-6)

    def test_both_feasible_or_both_infeasible_basic(self):
        """For simple cases, both methods should agree on feasibility."""
        # Clearly feasible
        constraints_ok = [(1.0, -0.5, 0.0)]
        _, _, f_lp = compute_safe_bounds_lp(constraints_ok, alpha=1.0)
        _, _, f_an = compute_safe_bounds_analytical(constraints_ok, alpha=1.0)
        assert f_lp == f_an, "Both should be feasible"

        # Clearly infeasible
        constraints_bad = [(-5.0, 1.0, 0.0)]
        _, _, f_lp = compute_safe_bounds_lp(constraints_bad, alpha=1.0)
        _, _, f_an = compute_safe_bounds_analytical(constraints_bad, alpha=1.0)
        assert f_lp == f_an, "Both should be infeasible"

    def test_analytical_at_least_as_wide_as_lp(self):
        """Analytical bounds (outer approx) should be at least as wide as LP (exact)."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            n = rng.integers(1, 6)
            constraints = [
                (rng.uniform(0.1, 2.0), rng.uniform(-1, 1), rng.uniform(-1, 1))
                for _ in range(n)
            ]
            v_lp, omega_lp, f_lp = compute_safe_bounds_lp(constraints, alpha=1.0)
            v_an, omega_an, f_an = compute_safe_bounds_analytical(constraints, alpha=1.0)

            if f_lp and f_an:
                assert v_an[0] <= v_lp[0] + 1e-6, \
                    f"Analytical v_min should be <= LP: {v_an[0]} vs {v_lp[0]}"
                assert v_an[1] >= v_lp[1] - 1e-6, \
                    f"Analytical v_max should be >= LP: {v_an[1]} vs {v_lp[1]}"


class TestVCPCBFFilterSafeBounds:
    """Tests for VCPCBFFilter.compute_safe_bounds() integration."""

    def test_center_returns_full_bounds(self):
        """Robot in center with no obstacles: full bounds available."""
        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )
        state = np.array([0.0, 0.0, 0.0])
        v, omega, feasible = cbf.compute_safe_bounds(state)

        assert feasible is True
        # In center, bounds should be close to nominal
        assert v[0] <= 0.1  # v_min should be near 0
        assert v[1] >= 0.9  # v_max should be near 1
        assert omega[0] <= -2.0  # omega_min should be near -2.84
        assert omega[1] >= 2.0  # omega_max should be near 2.84

    def test_near_wall_tightens_v(self):
        """Robot near wall heading toward it: v_max should be tightened."""
        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )
        state = np.array([4.7, 0.0, 0.0])  # Near right wall, heading right
        v, omega, feasible = cbf.compute_safe_bounds(state)

        assert feasible is True
        assert v[1] < 1.0, f"v_max should be tightened near wall, got {v[1]}"

    def test_lp_method(self):
        """LP method should work and return valid bounds."""
        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )
        state = np.array([0.0, 0.0, 0.0])
        v, omega, feasible = cbf.compute_safe_bounds(state, method="lp")

        assert feasible is True
        assert v[0] <= v[1]
        assert omega[0] <= omega[1]

    def test_analytical_method(self):
        """Analytical method should work and return valid bounds."""
        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )
        state = np.array([0.0, 0.0, 0.0])
        v, omega, feasible = cbf.compute_safe_bounds(state, method="analytical")

        assert feasible is True
        assert v[0] <= v[1]
        assert omega[0] <= omega[1]

    def test_with_obstacles_and_opponent(self):
        """Safe bounds with all constraint types."""
        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
            r_min_separation=0.35,
        )
        state = np.array([0.0, 0.0, 0.0])
        opponent = np.array([2.0, 0.0, np.pi])
        obstacles = [{"x": 1.0, "y": 1.0, "radius": 0.3}]

        v, omega, feasible = cbf.compute_safe_bounds(
            state, obstacles=obstacles, opponent_state=opponent,
        )
        assert feasible is True
        assert v[0] <= v[1]
        assert omega[0] <= omega[1]

    def test_analytical_at_least_as_wide_as_lp(self):
        """Analytical (outer approx) should be at least as wide as LP (exact)."""
        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )
        state = np.array([3.0, 2.0, 0.5])  # Off-center, angled

        v_lp, omega_lp, f_lp = cbf.compute_safe_bounds(state, method="lp")
        v_an, omega_an, f_an = cbf.compute_safe_bounds(state, method="analytical")

        if f_lp and f_an:
            assert v_an[0] <= v_lp[0] + 1e-6
            assert v_an[1] >= v_lp[1] - 1e-6

    def test_safe_bounds_qp_consistency(self):
        """QP-filtered action should be within safe bounds."""
        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )
        state = np.array([3.0, 2.0, 0.5])

        v_bounds, omega_bounds, feasible = cbf.compute_safe_bounds(
            state, method="lp",
        )
        u_safe, info = cbf.filter_action(
            np.array([1.0, 2.0]), state,
        )

        if feasible and info["solver_success"]:
            assert u_safe[0] >= v_bounds[0] - 1e-4, \
                f"v={u_safe[0]} < v_min={v_bounds[0]}"
            assert u_safe[0] <= v_bounds[1] + 1e-4, \
                f"v={u_safe[0]} > v_max={v_bounds[1]}"
            assert u_safe[1] >= omega_bounds[0] - 1e-4, \
                f"omega={u_safe[1]} < omega_min={omega_bounds[0]}"
            assert u_safe[1] <= omega_bounds[1] + 1e-4, \
                f"omega={u_safe[1]} > omega_max={omega_bounds[1]}"
