"""VCP-CBF validation script.

Runs all Phase 1 VCP-CBF validation criteria and prints a summary report.
This is the critical gate for proceeding to Phase 2.

Usage:
    python scripts/validate_vcp_cbf.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from envs.dynamics import unicycle_step
from safety.vcp_cbf import VCPCBFFilter, vcp_cbf_obstacle


def validate_steering_over_braking():
    """Criterion: mean(|delta_omega|) / mean(|delta_v|) > 2.0 when CBF intervenes."""
    cbf = VCPCBFFilter(d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84)

    delta_v_total = 0.0
    delta_omega_total = 0.0
    n_interventions = 0

    for angle_offset in np.linspace(-0.5, 0.5, 20):
        for dist in np.linspace(0.7, 2.0, 20):
            state = np.array([0.0, 0.0, angle_offset * 0.3])
            obstacles = [{"x": dist, "y": 0.2, "radius": 0.5}]
            u_nom = np.array([1.0, 0.0])
            u_safe, info = cbf.filter_action(u_nom, state, obstacles)

            if info["intervention"]:
                delta_v_total += abs(u_safe[0] - u_nom[0])
                delta_omega_total += abs(u_safe[1] - u_nom[1])
                n_interventions += 1

    ratio = delta_omega_total / max(delta_v_total, 1e-10)
    return ratio, n_interventions


def validate_a_omega_nonzero():
    """Criterion: |a_omega| > 1e-6 for >99% of constrained steps."""
    rng = np.random.default_rng(42)
    nonzero = 0
    total = 5000

    for _ in range(total):
        x = rng.uniform(-5, 5)
        y = rng.uniform(-5, 5)
        theta = rng.uniform(-np.pi, np.pi)
        state = np.array([x, y, theta])

        ox = rng.uniform(-5, 5)
        oy = rng.uniform(-5, 5)
        obs_pos = np.array([ox, oy])

        _, _, a_omega = vcp_cbf_obstacle(state, obs_pos, 0.5)
        if abs(a_omega) > 1e-6:
            nonzero += 1

    return nonzero / total


def validate_zero_collisions():
    """Criterion: Zero obstacle/boundary collisions with CBF active over 100 episodes."""
    rng = np.random.default_rng(42)
    robot_radius = 0.15
    cbf = VCPCBFFilter(
        d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
        arena_half_w=5.0, arena_half_h=5.0, robot_radius=robot_radius,
    )

    obs_x, obs_y, obs_r = 2.0, 0.0, 0.5
    collision_dist = obs_r + robot_radius
    collisions = 0
    boundary_violations = 0
    n_episodes = 100
    dt = 0.05
    steps = 200

    for ep in range(n_episodes):
        while True:
            x = rng.uniform(-3, 3)
            y = rng.uniform(-3, 3)
            if np.sqrt((x - obs_x)**2 + (y - obs_y)**2) > collision_dist + 0.3:
                break
        theta = rng.uniform(-np.pi, np.pi)

        for _ in range(steps):
            state = np.array([x, y, theta])
            u_nom = np.array([rng.uniform(0, 1), rng.uniform(-2.84, 2.84)])
            u_safe, _ = cbf.filter_action(
                u_nom, state, [{"x": obs_x, "y": obs_y, "radius": obs_r}],
            )
            x, y, theta, _ = unicycle_step(
                x, y, theta, float(u_safe[0]), float(u_safe[1]),
                dt, 10.0, 10.0, robot_radius,
            )
            if np.sqrt((x - obs_x)**2 + (y - obs_y)**2) < collision_dist - 0.02:
                collisions += 1
            x_lim = 5.0 - robot_radius
            y_lim = 5.0 - robot_radius
            if abs(x) > x_lim + 0.02 or abs(y) > y_lim + 0.02:
                boundary_violations += 1

    return collisions, boundary_violations, n_episodes


def validate_minimal_intervention():
    """Criterion: Intervention rate < 5% when far from obstacles."""
    rng = np.random.default_rng(42)
    cbf = VCPCBFFilter(
        d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
        arena_half_w=10.0, arena_half_h=10.0,
    )
    interventions = 0
    n_trials = 500

    for _ in range(n_trials):
        x = rng.uniform(-5, 5)
        y = rng.uniform(-5, 5)
        theta = rng.uniform(-np.pi, np.pi)
        u_nom = np.array([rng.uniform(0, 1), rng.uniform(-2.84, 2.84)])
        _, info = cbf.filter_action(u_nom, np.array([x, y, theta]))
        if info["intervention"]:
            interventions += 1

    return interventions / n_trials


def main():
    print("=" * 65)
    print("  VCP-CBF VALIDATION REPORT — Phase 1 Critical Gate")
    print("=" * 65)
    print()

    all_pass = True

    # 1. Steering over braking
    ratio, n_int = validate_steering_over_braking()
    passed = ratio > 2.0
    all_pass &= passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Steering/braking ratio: {ratio:.2f} (threshold: > 2.0)")
    print(f"       Interventions tested: {n_int}")
    print()

    # 2. a_omega nonzero
    fraction = validate_a_omega_nonzero()
    passed = fraction > 0.99
    all_pass &= passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] |a_omega| > 1e-6: {fraction:.2%} (threshold: > 99%)")
    print()

    # 3. Zero collisions
    coll, bv, eps = validate_zero_collisions()
    passed = coll == 0 and bv == 0
    all_pass &= passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Obstacle collisions: {coll} (threshold: 0)")
    print(f"       Boundary violations: {bv} (threshold: 0)")
    print(f"       Episodes tested: {eps}")
    print()

    # 4. Minimal intervention far from obstacles
    rate = validate_minimal_intervention()
    passed = rate < 0.05
    all_pass &= passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Intervention rate (far): {rate:.2%} (threshold: < 5%)")
    print()

    # Summary
    print("=" * 65)
    if all_pass:
        print("  ALL VALIDATION CRITERIA PASSED — VCP-CBF ready for Phase 2")
    else:
        print("  VALIDATION FAILED — Address issues before Phase 2")
    print("=" * 65)
    print()

    # Parameters
    print("VCP-CBF Parameters:")
    print(f"  d (VCP offset):    0.1 m")
    print(f"  alpha (CBF gain):  1.0")
    print(f"  w_v (QP weight):   150.0")
    print(f"  w_omega (QP wt):   1.0")
    print(f"  robot_radius:      0.15 m")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
