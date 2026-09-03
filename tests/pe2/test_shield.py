"""Tests for pe2.shield + pe2.metrics — Step 6 safety shield."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.env import OUTCOME_CAPTURE, PursuitEvasionEnv
from pe2.metrics import ShieldMetrics
from pe2.scripted import GreedyEvader, GreedyPursuer
from pe2.shield import (
    CBFShield,
    _boundary_constraints,
    _obstacle_constraint,
    _solve_qp_2d,
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


class TestExactQP:
    def test_unconstrained_nominal_passes_through(self):
        u, feasible = _solve_qp_2d(0.5, 0.5, [], 1.0, 2.84, 150.0, 1.0)
        assert feasible
        assert np.allclose(u, [0.5, 0.5])

    def test_single_constraint_hand_computed(self):
        # dt*a_v = -0.05, dt*a_omega = 0, c2 = -0.01 -> v <= 0.2.
        u, feasible = _solve_qp_2d(1.0, 0.0, [(-0.05, 0.0, -0.01)], 1.0, 2.84, 150.0, 1.0)
        assert feasible
        assert u[0] == pytest.approx(0.2)
        assert u[1] == pytest.approx(0.0, abs=1e-9)

    def test_infeasible_rows_reported(self):
        # v >= 10 and v <= -10 simultaneously: impossible.
        u, feasible = _solve_qp_2d(0.0, 0.0, [(1.0, 0.0, 10.0), (-1.0, 0.0, 10.0)], 1.0, 2.84, 150.0, 1.0)
        assert not feasible

    def test_vertex_optimum_two_constraints(self):
        # v + w >= 2 and v - w >= 2 with nominal (0,0), equal weights:
        # optimum is the vertex (2, 0).
        u, feasible = _solve_qp_2d(0.0, 0.0, [(1.0, 1.0, 2.0), (1.0, -1.0, 2.0)], 5.0, 5.0, 1.0, 1.0)
        assert feasible
        assert u[0] == pytest.approx(2.0)
        assert u[1] == pytest.approx(0.0, abs=1e-9)

    def test_anisotropic_weight_prefers_steering(self):
        # Same constraint as the off-axis obstacle case: -1.2v - 0.06w >= -0.8
        # scaled by dt=0.05 -> (-0.06, -0.003, -0.04). With w_v=150 >> w_omega=1
        # the projection must turn far more than it brakes.
        u, feasible = _solve_qp_2d(1.0, 0.0, [(-0.06, -0.003, -0.04)], 1.0, 2.84, 150.0, 1.0)
        assert feasible
        assert -0.06 * u[0] - 0.003 * u[1] >= -0.04 - 1e-9
        assert abs(u[1]) > 1.0  # big steering correction
        assert u[0] > 0.7  # little braking


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


def info_kinds(cons):
    return {c.kind for c in cons}


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


class TestShieldMetrics:
    def test_rates_hand_computed(self):
        m = ShieldMetrics()
        m.record({"intervention": True, "method": "exact", "min_h": 0.5}, violation=False)
        m.record({"intervention": False, "method": "exact", "min_h": 0.3}, violation=True)
        m.record({"intervention": True, "method": "backup", "min_h": 0.1}, violation=False)
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
