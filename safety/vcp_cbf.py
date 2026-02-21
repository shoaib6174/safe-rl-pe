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

Phase 2 extensions:
- Inter-robot collision constraint (vcp_cbf_collision)
- Typed constraints for Tier 2 hierarchical relaxation
- Multi-agent support in VCPCBFFilter

Reference: Zhang & Yang 2025 — VCP-CBF for nonholonomic robots.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import minimize


class ConstraintType(Enum):
    """Constraint priority for Tier 2 relaxation (lowest priority relaxed first)."""
    COLLISION = 3    # Highest priority: inter-robot collision
    ARENA = 2        # Medium priority: arena boundaries
    OBSTACLE = 1     # Lowest priority: obstacle avoidance


@dataclass
class CBFConstraint:
    """Typed CBF constraint for hierarchical relaxation.

    Attributes:
        h: CBF value (positive = safe, negative = violated).
        a_v: Coefficient on linear velocity in CBF constraint.
        a_omega: Coefficient on angular velocity in CBF constraint.
        ctype: Constraint type for Tier 2 relaxation priority.
    """
    h: float
    a_v: float
    a_omega: float
    ctype: ConstraintType

    def as_tuple(self) -> tuple[float, float, float]:
        """Convert to (h, a_v, a_omega) tuple for backward compatibility."""
        return (self.h, self.a_v, self.a_omega)


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


def vcp_cbf_collision(
    state_ego: np.ndarray,
    state_other: np.ndarray,
    r_min: float,
    d: float = 0.1,
    alpha: float = 1.0,
) -> tuple[float, float, float]:
    """VCP-CBF for inter-robot collision avoidance.

    h(x) = ||q_ego - q_other||^2 - r_min^2

    where q_ego and q_other are VCPs for ego and other robot respectively.
    The constraint is w.r.t. ego's controls only — other's motion is treated
    as an uncontrolled disturbance (worst-case in adversarial PE).

    Note: r_min should be set to collision_radius (NOT capture_radius).
    The game ends at capture_radius > r_min, so the CBF does not prevent
    capture, only physical collision.

    Args:
        state_ego: Ego robot state [x, y, theta].
        state_other: Other robot state [x, y, theta].
        r_min: Minimum separation distance (collision_radius + margin).
        d: VCP offset distance.
        alpha: CBF class-K function parameter.

    Returns:
        (h, a_v, a_omega): CBF value and constraint coefficients for ego.
    """
    x_e, y_e, theta_e = state_ego[0], state_ego[1], state_ego[2]
    x_o, y_o, theta_o = state_other[0], state_other[1], state_other[2]

    # Compute VCPs for both robots
    qx_e, qy_e = compute_vcp(x_e, y_e, theta_e, d)
    qx_o, qy_o = compute_vcp(x_o, y_o, theta_o, d)

    # CBF value
    dx = qx_e - qx_o
    dy = qy_e - qy_o
    h = dx**2 + dy**2 - r_min**2

    # Constraint coefficients for ego's controls [v_ego, omega_ego]
    # h_dot = 2*dx*(q_dot_x_ego - q_dot_x_other) + 2*dy*(q_dot_y_ego - q_dot_y_other)
    # Ego terms (controllable):
    #   q_dot_x_ego = v_ego * cos(theta_ego) - d * omega_ego * sin(theta_ego)
    #   q_dot_y_ego = v_ego * sin(theta_ego) + d * omega_ego * cos(theta_ego)
    # Other terms (uncontrolled, treated as zero for constraint — worst case handled by alpha)
    cos_e = np.cos(theta_e)
    sin_e = np.sin(theta_e)

    a_v = 2.0 * dx * cos_e + 2.0 * dy * sin_e
    a_omega = 2.0 * dx * (-d * sin_e) + 2.0 * dy * (d * cos_e)

    return float(h), float(a_v), float(a_omega)


def get_all_constraints(
    state_ego: np.ndarray,
    state_other: np.ndarray | None,
    arena_half_w: float,
    arena_half_h: float,
    robot_radius: float,
    obstacles: list[dict] | None = None,
    r_min: float = 0.35,
    d: float = 0.1,
    alpha: float = 1.0,
) -> list[CBFConstraint]:
    """Aggregate all CBF constraints for one agent.

    Combines arena boundary, obstacle, and inter-robot collision constraints
    into a single list with type tags for Tier 2 relaxation.

    Args:
        state_ego: Ego robot state [x, y, theta].
        state_other: Other robot state [x, y, theta], or None if no opponent.
        arena_half_w: Half arena width.
        arena_half_h: Half arena height.
        robot_radius: Robot radius.
        obstacles: List of obstacle dicts with 'x', 'y', 'radius' keys.
        r_min: Minimum inter-robot separation (collision_radius + margin).
        d: VCP offset distance.
        alpha: CBF class-K parameter.

    Returns:
        List of typed CBFConstraint objects.
    """
    constraints = []

    # Arena boundary constraints (4)
    boundary = vcp_cbf_boundary(state_ego, arena_half_w, arena_half_h,
                                robot_radius, d, alpha)
    for h, a_v, a_omega in boundary:
        constraints.append(CBFConstraint(h, a_v, a_omega, ConstraintType.ARENA))

    # Obstacle constraints
    if obstacles:
        for obs in obstacles:
            obs_pos = np.array([obs["x"], obs["y"]])
            effective_radius = obs["radius"] + robot_radius
            h, a_v, a_omega = vcp_cbf_obstacle(state_ego, obs_pos, effective_radius, d, alpha)
            constraints.append(CBFConstraint(h, a_v, a_omega, ConstraintType.OBSTACLE))

    # Inter-robot collision constraint
    if state_other is not None:
        h, a_v, a_omega = vcp_cbf_collision(state_ego, state_other, r_min, d, alpha)
        constraints.append(CBFConstraint(h, a_v, a_omega, ConstraintType.COLLISION))

    return constraints


def compute_safe_bounds_lp(
    constraints: list[tuple[float, float, float]],
    alpha: float,
    v_bounds: tuple[float, float] = (0.0, 1.0),
    omega_bounds: tuple[float, float] = (-2.84, 2.84),
) -> tuple[tuple[float, float], tuple[float, float], bool]:
    """Compute tightest safe box bounds via linear programming.

    For each bound (v_min, v_max, omega_min, omega_max), solves an LP to find
    the tightest value consistent with all CBF constraints. This gives the
    smallest axis-aligned box containing the safe polytope.

    This is the accurate version — use for correctness-critical scenarios.
    For speed-critical scenarios (training), use compute_safe_bounds_analytical.

    Args:
        constraints: List of (h, a_v, a_omega) from CBF functions.
        alpha: CBF class-K parameter.
        v_bounds: Nominal velocity bounds (v_min, v_max).
        omega_bounds: Nominal angular velocity bounds (omega_min, omega_max).

    Returns:
        ((v_min_safe, v_max_safe), (omega_min_safe, omega_max_safe), feasible):
        Safe bounds and feasibility flag. If infeasible, returns nominal bounds.
    """
    from scipy.optimize import linprog

    v_lo, v_hi = v_bounds
    omega_lo, omega_hi = omega_bounds

    if not constraints:
        return v_bounds, omega_bounds, True

    # Build constraint matrix: A_ub @ x <= b_ub (linprog uses <= form)
    # Each CBF: a_v * v + a_omega * omega >= -alpha * h
    # Negate:  -a_v * v - a_omega * omega <= alpha * h
    n = len(constraints)
    A_ub = np.zeros((n, 2))
    b_ub = np.zeros(n)
    for i, (h, a_v, a_omega) in enumerate(constraints):
        A_ub[i, 0] = -a_v
        A_ub[i, 1] = -a_omega
        b_ub[i] = alpha * h

    bounds_lp = [(v_lo, v_hi), (omega_lo, omega_hi)]

    # Solve 4 LPs for tightest box bounds
    results = {}
    for name, c in [("v_min", [1, 0]), ("v_max", [-1, 0]),
                     ("omega_min", [0, 1]), ("omega_max", [0, -1])]:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method="highs")
        results[name] = res

    if not all(r.success for r in results.values()):
        return v_bounds, omega_bounds, False

    safe_v_min = results["v_min"].x[0]
    safe_v_max = results["v_max"].x[0]
    safe_omega_min = results["omega_min"].x[1]
    safe_omega_max = results["omega_max"].x[1]

    # Ensure bounds are valid (min <= max)
    if safe_v_min > safe_v_max + 1e-10 or safe_omega_min > safe_omega_max + 1e-10:
        return v_bounds, omega_bounds, False

    return (safe_v_min, safe_v_max), (safe_omega_min, safe_omega_max), True


def compute_safe_bounds_analytical(
    constraints: list[tuple[float, float, float]],
    alpha: float,
    v_bounds: tuple[float, float] = (0.0, 1.0),
    omega_bounds: tuple[float, float] = (-2.84, 2.84),
) -> tuple[tuple[float, float], tuple[float, float], bool]:
    """Compute safe box bounds analytically (no LP solver).

    For each constraint, computes the tightest bound on one variable
    assuming the other takes its most helpful value. This gives an
    outer approximation of the LP-exact bounds — potentially wider
    because each constraint picks its own best-case other variable.

    Faster than LP (~0.01ms vs ~0.1ms) but may be wider. The QP solver
    serves as backup to correct any samples outside the true safe set.

    Args:
        constraints: List of (h, a_v, a_omega) from CBF functions.
        alpha: CBF class-K parameter.
        v_bounds: Nominal velocity bounds.
        omega_bounds: Nominal angular velocity bounds.

    Returns:
        ((v_min_safe, v_max_safe), (omega_min_safe, omega_max_safe), feasible):
        Safe bounds and feasibility flag. If infeasible, returns nominal bounds.
    """
    v_lo, v_hi = v_bounds
    omega_lo, omega_hi = omega_bounds

    if not constraints:
        return v_bounds, omega_bounds, True

    # Tighten v bounds based on CBF constraints
    for h, a_v, a_omega in constraints:
        rhs = -alpha * h  # constraint: a_v * v + a_omega * omega >= rhs

        if abs(a_v) > 1e-10:
            # Best-case omega: pick omega that maximizes a_omega * omega
            best_omega = omega_hi if a_omega >= 0 else omega_lo
            slack = a_omega * best_omega
            # a_v * v >= rhs - slack
            bound = (rhs - slack) / a_v
            if a_v > 0:
                v_lo = max(v_lo, bound)
            else:
                v_hi = min(v_hi, bound)

    # Tighten omega bounds based on CBF constraints
    for h, a_v, a_omega in constraints:
        rhs = -alpha * h

        if abs(a_omega) > 1e-10:
            # Best-case v: pick v that maximizes a_v * v
            best_v = v_hi if a_v >= 0 else v_lo
            slack = a_v * best_v
            bound = (rhs - slack) / a_omega
            if a_omega > 0:
                omega_lo = max(omega_lo, bound)
            else:
                omega_hi = min(omega_hi, bound)

    if v_lo > v_hi + 1e-10 or omega_lo > omega_hi + 1e-10:
        return v_bounds, omega_bounds, False

    return (v_lo, v_hi), (omega_lo, omega_hi), True


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
    """Safety filter using VCP-CBF for obstacle, boundary, and collision avoidance.

    Combines obstacle, boundary, and inter-robot collision CBF constraints
    into a single QP. Robot radius is automatically added to obstacle radii
    as a safety margin.

    Phase 2 extensions:
    - Inter-robot collision constraint (via opponent_state parameter)
    - Typed constraints for Tier 2 hierarchical relaxation
    - CBF margin tracking for safety reward shaping
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
        r_min_separation: float = 0.35,
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
        self.r_min_separation = r_min_separation

        # Tracking
        self.n_interventions = 0
        self.n_total = 0
        self.n_infeasible = 0

    def get_constraints(
        self,
        state: np.ndarray,
        opponent_state: np.ndarray | None = None,
        obstacles: list[dict] | None = None,
    ) -> list[CBFConstraint]:
        """Get all typed CBF constraints for the given state.

        Args:
            state: Ego robot state [x, y, theta].
            opponent_state: Other robot state [x, y, theta], or None.
            obstacles: List of obstacle dicts with 'x', 'y', 'radius' keys.

        Returns:
            List of typed CBFConstraint objects.
        """
        return get_all_constraints(
            state, opponent_state,
            self.arena_half_w, self.arena_half_h,
            self.robot_radius, obstacles,
            self.r_min_separation, self.d, self.alpha,
        )

    def filter_action(
        self,
        action: np.ndarray,
        state: np.ndarray,
        obstacles: list[dict] | None = None,
        opponent_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Filter nominal action through VCP-CBF-QP.

        Args:
            action: Nominal action [v, omega].
            state: Robot state [x, y, theta].
            obstacles: List of dicts with 'x', 'y', 'radius' keys.
                       Robot radius is added automatically as safety margin.
            opponent_state: Other robot state [x, y, theta], or None.

        Returns:
            (safe_action, info): Filtered action and safety info.
        """
        # Get typed constraints
        typed_constraints = self.get_constraints(state, opponent_state, obstacles)

        # Convert to tuples for QP solver (backward compatible)
        all_constraints = [c.as_tuple() for c in typed_constraints]

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
        if not info["solver_success"]:
            self.n_infeasible += 1

        # Enrich info with typed constraint data
        info["a_omega_values"] = [c.a_omega for c in typed_constraints]
        info["constraints"] = typed_constraints
        info["h_values"] = {
            "arena": [c.h for c in typed_constraints if c.ctype == ConstraintType.ARENA],
            "obstacle": [c.h for c in typed_constraints if c.ctype == ConstraintType.OBSTACLE],
            "collision": [c.h for c in typed_constraints if c.ctype == ConstraintType.COLLISION],
        }
        info["min_h"] = min(c.h for c in typed_constraints) if typed_constraints else float("inf")

        return u_safe, info


    def compute_safe_bounds(
        self,
        state: np.ndarray,
        obstacles: list[dict] | None = None,
        opponent_state: np.ndarray | None = None,
        method: str = "analytical",
    ) -> tuple[tuple[float, float], tuple[float, float], bool]:
        """Compute safe action bounds given current constraints.

        Uses CBF constraints to compute the tightest axis-aligned box
        within the safe polytope, giving the Beta policy its support bounds.

        Args:
            state: Robot state [x, y, theta].
            obstacles: List of obstacle dicts with 'x', 'y', 'radius' keys.
            opponent_state: Other robot state [x, y, theta], or None.
            method: "lp" for exact bounds, "analytical" for fast conservative bounds.

        Returns:
            ((v_min, v_max), (omega_min, omega_max), feasible): Safe bounds.
        """
        typed_constraints = self.get_constraints(state, opponent_state, obstacles)
        constraint_tuples = [c.as_tuple() for c in typed_constraints]

        v_bounds = (0.0, self.v_max)
        omega_bounds = (-self.omega_max, self.omega_max)

        if method == "lp":
            return compute_safe_bounds_lp(
                constraint_tuples, self.alpha, v_bounds, omega_bounds,
            )
        else:
            return compute_safe_bounds_analytical(
                constraint_tuples, self.alpha, v_bounds, omega_bounds,
            )

    def get_metrics(self) -> dict:
        """Return safety metrics."""
        return {
            "n_interventions": self.n_interventions,
            "n_total": self.n_total,
            "n_infeasible": self.n_infeasible,
            "intervention_rate": (
                self.n_interventions / self.n_total if self.n_total > 0 else 0
            ),
            "feasibility_rate": (
                1.0 - self.n_infeasible / self.n_total if self.n_total > 0 else 1.0
            ),
        }

    def reset_metrics(self):
        """Reset tracking counters."""
        self.n_interventions = 0
        self.n_total = 0
        self.n_infeasible = 0
