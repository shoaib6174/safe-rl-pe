"""VCP-CBF for nonholonomic (unicycle) robots.

Virtual Control Point (VCP) resolves the mixed relative degree issue
for position-based CBFs on unicycle robots. By placing a virtual point
at distance d ahead of the robot, both v and omega appear in the CBF
derivative, enabling standard CBF-QP.

VCP velocity:
    q_dot = [v*cos(theta) - d*omega*sin(theta),
             v*sin(theta) + d*omega*cos(theta)]

This achieves uniform relative degree 1, making CBF-QP well-posed.

To actually prioritize steering over braking, the QP uses anisotropic
weights (w_v >> w_omega). With equal weights, the solver prefers braking
because a_v >> a_omega (by factor ~1/d). Weighted QP is standard in
VCP-CBF literature to encode the steering preference.

Reference: Zhang & Yang 2025 — VCP-CBF for nonholonomic robots.
"""

import numpy as np
from scipy.optimize import minimize


def compute_vcp(x: float, y: float, theta: float, d: float = 0.1) -> tuple[float, float]:
    """Virtual control point at distance d ahead of robot.

    Args:
        x, y: Robot position.
        theta: Robot heading.
        d: VCP offset distance (meters).

    Returns:
        (qx, qy): VCP position.
    """
    qx = x + d * np.cos(theta)
    qy = y + d * np.sin(theta)
    return qx, qy


def vcp_cbf_obstacle(
    state: np.ndarray,
    obs_pos: np.ndarray,
    obs_radius: float,
    d: float = 0.1,
    alpha: float = 1.0,
) -> tuple[float, float, float]:
    """VCP-CBF for circular obstacle avoidance.

    h(x) = ||q - p_obs||^2 - chi^2
    where q is the VCP, chi = obs_radius (should include safety margin).

    CBF condition: a_v * v + a_omega * omega + alpha * h >= 0

    Args:
        state: Robot state [x, y, theta].
        obs_pos: Obstacle center [x, y].
        obs_radius: Effective obstacle radius (including robot radius / safety margin).
        d: VCP offset distance.
        alpha: CBF class-K function parameter.

    Returns:
        (h, a_v, a_omega): CBF value and constraint coefficients.
    """
    x, y, theta = state[0], state[1], state[2]
    qx, qy = compute_vcp(x, y, theta, d)

    # CBF value
    dx = qx - obs_pos[0]
    dy = qy - obs_pos[1]
    h = dx**2 + dy**2 - obs_radius**2

    # Partial derivatives of h w.r.t. q
    dh_dqx = 2.0 * dx
    dh_dqy = 2.0 * dy

    # CBF constraint coefficients: h_dot = a_v * v + a_omega * omega
    # q_dot_x = v*cos(theta) - d*omega*sin(theta)
    # q_dot_y = v*sin(theta) + d*omega*cos(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    a_v = dh_dqx * cos_t + dh_dqy * sin_t
    a_omega = -dh_dqx * d * sin_t + dh_dqy * d * cos_t

    return float(h), float(a_v), float(a_omega)


def vcp_cbf_boundary(
    state: np.ndarray,
    arena_half_w: float,
    arena_half_h: float,
    robot_radius: float,
    d: float = 0.1,
    alpha: float = 1.0,
) -> list[tuple[float, float, float]]:
    """VCP-CBF constraints for rectangular arena boundaries.

    Uses linear barriers:
        h_right = (arena_half_w - robot_radius) - qx
        h_left  = qx - (-arena_half_w + robot_radius)
        h_top   = (arena_half_h - robot_radius) - qy
        h_bot   = qy - (-arena_half_h + robot_radius)

    Args:
        state: Robot state [x, y, theta].
        arena_half_w: Half arena width.
        arena_half_h: Half arena height.
        robot_radius: Robot radius.
        d: VCP offset distance.
        alpha: CBF class-K function parameter.

    Returns:
        List of (h, a_v, a_omega) tuples for each boundary constraint.
    """
    x, y, theta = state[0], state[1], state[2]
    qx, qy = compute_vcp(x, y, theta, d)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_lim = arena_half_w - robot_radius
    y_lim = arena_half_h - robot_radius

    constraints = []

    # Right wall: h = x_lim - qx
    # h_dot = -q_dot_x = -(v*cos(theta) - d*omega*sin(theta))
    h_right = x_lim - qx
    a_v_right = -cos_t
    a_omega_right = d * sin_t
    constraints.append((float(h_right), float(a_v_right), float(a_omega_right)))

    # Left wall: h = qx - (-x_lim) = qx + x_lim
    h_left = qx + x_lim
    a_v_left = cos_t
    a_omega_left = -d * sin_t
    constraints.append((float(h_left), float(a_v_left), float(a_omega_left)))

    # Top wall: h = y_lim - qy
    h_top = y_lim - qy
    a_v_top = -sin_t
    a_omega_top = -d * cos_t
    constraints.append((float(h_top), float(a_v_top), float(a_omega_top)))

    # Bottom wall: h = qy + y_lim
    h_bot = qy + y_lim
    a_v_bot = sin_t
    a_omega_bot = d * cos_t
    constraints.append((float(h_bot), float(a_v_bot), float(a_omega_bot)))

    return constraints


def solve_cbf_qp(
    u_nominal: np.ndarray,
    constraints: list[tuple[float, float, float]],
    alpha: float,
    v_min: float = 0.0,
    v_max: float = 1.0,
    omega_min: float = -2.84,
    omega_max: float = 2.84,
    w_v: float = 150.0,
    w_omega: float = 1.0,
) -> tuple[np.ndarray, str, dict]:
    """Solve CBF-QP: find safe action closest to nominal.

    min  w_v * (v - v_nom)^2 + w_omega * (omega - omega_nom)^2
    s.t. a_v_i * v + a_omega_i * omega + alpha * h_i >= 0  for all i
         v_min <= v <= v_max
         omega_min <= omega <= omega_max

    Anisotropic weights (w_v >> w_omega) encode preference for steering
    over braking. With d=0.1 and equal weights, a_v >> a_omega causes
    the solver to always prefer braking. Setting w_v > w_omega makes
    velocity reduction more costly, so the solver prefers angular adjustments.

    Uses scipy.optimize.minimize with SLSQP solver.

    Args:
        u_nominal: Nominal action [v, omega].
        constraints: List of (h, a_v, a_omega) from CBF functions.
        alpha: CBF class-K parameter.
        v_min, v_max: Velocity bounds.
        omega_min, omega_max: Angular velocity bounds.
        w_v: Weight on velocity deviation (higher = prefer maintaining v).
        w_omega: Weight on angular velocity deviation.

    Returns:
        (u_safe, status, info): Safe action, solver status string, and info dict.
    """
    v_nom, omega_nom = float(u_nominal[0]), float(u_nominal[1])

    # Objective: min w_v*(v-v_nom)^2 + w_omega*(omega-omega_nom)^2
    def objective(u):
        return w_v * (u[0] - v_nom) ** 2 + w_omega * (u[1] - omega_nom) ** 2

    def objective_jac(u):
        return np.array([
            2.0 * w_v * (u[0] - v_nom),
            2.0 * w_omega * (u[1] - omega_nom),
        ])

    # CBF constraints: a_v * v + a_omega * omega + alpha * h >= 0
    scipy_constraints = []
    for h, a_v, a_omega in constraints:
        scipy_constraints.append({
            "type": "ineq",
            "fun": lambda u, av=a_v, ao=a_omega, hv=h: av * u[0] + ao * u[1] + alpha * hv,
            "jac": lambda u, av=a_v, ao=a_omega: np.array([av, ao]),
        })

    bounds = [(v_min, v_max), (omega_min, omega_max)]

    # Initial guess: nominal action (clipped to bounds)
    u0 = np.array([
        np.clip(v_nom, v_min, v_max),
        np.clip(omega_nom, omega_min, omega_max),
    ])

    result = minimize(
        objective,
        u0,
        jac=objective_jac,
        method="SLSQP",
        bounds=bounds,
        constraints=scipy_constraints,
        options={"ftol": 1e-12, "maxiter": 100},
    )

    u_safe = np.array([
        np.clip(result.x[0], v_min, v_max),
        np.clip(result.x[1], omega_min, omega_max),
    ], dtype=np.float32)

    status = "optimal" if result.success else "failed"

    info = {
        "solver_success": result.success,
        "solver_message": result.message,
        "n_constraints": len(constraints),
        "min_h": min(h for h, _, _ in constraints) if constraints else float("inf"),
        "intervention": not np.allclose(u_safe, u_nominal[:2], atol=1e-4),
    }

    return u_safe, status, info


class VCPCBFFilter:
    """Safety filter using VCP-CBF for obstacle and boundary avoidance.

    Combines obstacle and boundary CBF constraints into a single QP.
    Robot radius is automatically added to obstacle radii as a safety margin.
    """

    def __init__(
        self,
        d: float = 0.1,
        alpha: float = 1.0,
        v_max: float = 1.0,
        omega_max: float = 2.84,
        arena_half_w: float = 10.0,
        arena_half_h: float = 10.0,
        robot_radius: float = 0.15,
        w_v: float = 150.0,
        w_omega: float = 1.0,
    ):
        self.d = d
        self.alpha = alpha
        self.v_max = v_max
        self.omega_max = omega_max
        self.arena_half_w = arena_half_w
        self.arena_half_h = arena_half_h
        self.robot_radius = robot_radius
        self.w_v = w_v
        self.w_omega = w_omega

        # Tracking
        self.n_interventions = 0
        self.n_total = 0

    def filter_action(
        self,
        action: np.ndarray,
        state: np.ndarray,
        obstacles: list[dict] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Filter nominal action through VCP-CBF-QP.

        Args:
            action: Nominal action [v, omega].
            state: Robot state [x, y, theta].
            obstacles: List of dicts with 'x', 'y', 'radius' keys.
                       Robot radius is added automatically as safety margin.

        Returns:
            (safe_action, info): Filtered action and safety info.
        """
        all_constraints = []

        # Obstacle constraints (add robot_radius as safety margin)
        if obstacles:
            for obs in obstacles:
                obs_pos = np.array([obs["x"], obs["y"]])
                effective_radius = obs["radius"] + self.robot_radius
                h, a_v, a_omega = vcp_cbf_obstacle(
                    state, obs_pos, effective_radius, self.d, self.alpha,
                )
                all_constraints.append((h, a_v, a_omega))

        # Boundary constraints
        boundary_constraints = vcp_cbf_boundary(
            state, self.arena_half_w, self.arena_half_h,
            self.robot_radius, self.d, self.alpha,
        )
        all_constraints.extend(boundary_constraints)

        # Solve QP
        u_safe, status, info = solve_cbf_qp(
            action, all_constraints, self.alpha,
            v_min=0.0, v_max=self.v_max,
            omega_min=-self.omega_max, omega_max=self.omega_max,
            w_v=self.w_v, w_omega=self.w_omega,
        )

        self.n_total += 1
        if info["intervention"]:
            self.n_interventions += 1

        info["a_omega_values"] = [ao for _, _, ao in all_constraints]
        return u_safe, info

    def get_metrics(self) -> dict:
        """Return safety metrics."""
        return {
            "n_interventions": self.n_interventions,
            "n_total": self.n_total,
            "intervention_rate": (
                self.n_interventions / self.n_total if self.n_total > 0 else 0
            ),
        }

    def reset_metrics(self):
        """Reset tracking counters."""
        self.n_interventions = 0
        self.n_total = 0
