"""3-Tier Infeasibility Handling for CBF-QP safety filtering.

When the CBF-QP becomes infeasible (contradicting constraints),
three tiers of fallback ensure the robot always has a safe action:

Tier 1 — Learned Feasibility: SVM classifier predicts whether a state
    will make CBF-QP infeasible, enabling proactive avoidance.

Tier 2 — Hierarchical Relaxation: Drops constraints by priority
    (obstacle < arena < collision) until QP becomes feasible.

Tier 3 — Backup Controller: Emergency controller that brakes and
    turns away from the nearest danger source.

SafeActionResolver chains all three tiers into a unified interface.

Reference: Paper N13 (CASRL) for iterative feasibility training.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from safety.vcp_cbf import (
    CBFConstraint,
    ConstraintType,
    VCPCBFFilter,
    solve_cbf_qp,
)


# =============================================================================
# Tier 3: Backup Controller
# =============================================================================

def backup_controller(
    state: np.ndarray,
    obstacles: list[dict] | None = None,
    arena_half_w: float = 10.0,
    arena_half_h: float = 10.0,
    v_max: float = 1.0,
    omega_max: float = 2.84,
) -> np.ndarray:
    """Emergency controller: brake + turn away from nearest danger.

    No task performance, only survival. Moves very slowly and turns
    away from the nearest danger source (obstacle edge or boundary).

    Args:
        state: Robot state [x, y, theta].
        obstacles: List of obstacle dicts with 'x', 'y', 'radius' keys.
        arena_half_w: Half arena width.
        arena_half_h: Half arena height.
        v_max: Maximum linear velocity.
        omega_max: Maximum angular velocity.

    Returns:
        Emergency action [v, omega].
    """
    x, y, theta = state[0], state[1], state[2]

    # Find nearest danger direction
    danger_dir, danger_dist = _find_nearest_danger(
        x, y, obstacles, arena_half_w, arena_half_h,
    )

    # Turn away from danger
    away_dir = danger_dir + np.pi
    heading_error = _wrap_angle(away_dir - theta)

    omega = np.clip(3.0 * heading_error, -omega_max, omega_max)
    v = 0.1 * v_max  # Very slow forward motion

    return np.array([v, omega], dtype=np.float32)


def _find_nearest_danger(
    x: float,
    y: float,
    obstacles: list[dict] | None,
    arena_half_w: float,
    arena_half_h: float,
) -> tuple[float, float]:
    """Find direction and distance to nearest danger source.

    Returns:
        (direction, distance): Direction to nearest danger in world frame,
        and the distance to it.
    """
    dangers = []

    # Arena boundaries
    # Right wall
    d_right = arena_half_w - x
    dangers.append((0.0, d_right))  # direction: +x
    # Left wall
    d_left = x + arena_half_w
    dangers.append((np.pi, d_left))  # direction: -x
    # Top wall
    d_top = arena_half_h - y
    dangers.append((np.pi / 2, d_top))  # direction: +y
    # Bottom wall
    d_bottom = y + arena_half_h
    dangers.append((-np.pi / 2, d_bottom))  # direction: -y

    # Obstacles
    if obstacles:
        for obs in obstacles:
            dx = obs["x"] - x
            dy = obs["y"] - y
            dist_center = np.sqrt(dx**2 + dy**2)
            dist_surface = max(dist_center - obs["radius"], 0.01)
            direction = np.arctan2(dy, dx)
            dangers.append((direction, dist_surface))

    # Find nearest
    nearest = min(dangers, key=lambda d: d[1])
    return nearest[0], nearest[1]


def _wrap_angle(theta: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# Tier 2: Hierarchical Relaxation
# =============================================================================

def solve_cbf_qp_with_relaxation(
    u_nominal: np.ndarray,
    constraints: list[CBFConstraint],
    alpha: float = 1.0,
    v_min: float = 0.0,
    v_max: float = 1.0,
    omega_min: float = -2.84,
    omega_max: float = 2.84,
    w_v: float = 150.0,
    w_omega: float = 1.0,
) -> tuple[np.ndarray | None, str, dict]:
    """Solve CBF-QP with hierarchical constraint relaxation.

    Tries progressively relaxed constraint sets by priority:
    1. All constraints (exact)
    2. Drop OBSTACLE constraints (lowest priority)
    3. Drop ARENA constraints (keep only COLLISION)
    4. Drop all constraints (unconstrained)

    Args:
        u_nominal: Nominal action [v, omega].
        constraints: List of typed CBFConstraint objects.
        alpha: CBF class-K parameter.
        v_min, v_max: Velocity bounds.
        omega_min, omega_max: Angular velocity bounds.
        w_v: Weight on velocity deviation.
        w_omega: Weight on angular velocity deviation.

    Returns:
        (safe_action, method, info):
        - safe_action: Filtered action, or None if all tiers fail.
        - method: 'exact', 'relaxed_obstacles', 'relaxed_arena',
                  'unconstrained', or 'infeasible'.
        - info: Dict with solver details.
    """
    qp_kwargs = dict(
        alpha=alpha, v_min=v_min, v_max=v_max,
        omega_min=omega_min, omega_max=omega_max,
        w_v=w_v, w_omega=w_omega,
    )

    # Separate constraints by type
    collision_cons = [c for c in constraints if c.ctype == ConstraintType.COLLISION]
    arena_cons = [c for c in constraints if c.ctype == ConstraintType.ARENA]
    obstacle_cons = [c for c in constraints if c.ctype == ConstraintType.OBSTACLE]

    # Tier 2a: Try all constraints (exact)
    all_tuples = [c.as_tuple() for c in constraints]
    if all_tuples:
        u_safe, status, info = solve_cbf_qp(u_nominal, all_tuples, **qp_kwargs)
        if status == "optimal":
            info["relaxation_method"] = "exact"
            info["relaxed_types"] = []
            return u_safe, "exact", info

    # Tier 2b: Relax obstacle constraints
    reduced = collision_cons + arena_cons
    reduced_tuples = [c.as_tuple() for c in reduced]
    if reduced_tuples:
        u_safe, status, info = solve_cbf_qp(u_nominal, reduced_tuples, **qp_kwargs)
        if status == "optimal":
            info["relaxation_method"] = "relaxed_obstacles"
            info["relaxed_types"] = ["obstacle"]
            return u_safe, "relaxed_obstacles", info

    # Tier 2c: Relax arena constraints (keep only collision)
    collision_tuples = [c.as_tuple() for c in collision_cons]
    if collision_tuples:
        u_safe, status, info = solve_cbf_qp(u_nominal, collision_tuples, **qp_kwargs)
        if status == "optimal":
            info["relaxation_method"] = "relaxed_arena"
            info["relaxed_types"] = ["obstacle", "arena"]
            return u_safe, "relaxed_arena", info

    # Tier 2d: Drop all constraints — return clipped nominal
    u_clipped = np.array([
        np.clip(u_nominal[0], v_min, v_max),
        np.clip(u_nominal[1], omega_min, omega_max),
    ], dtype=np.float32)
    info = {
        "relaxation_method": "unconstrained",
        "relaxed_types": ["obstacle", "arena", "collision"],
        "solver_success": True,
        "intervention": False,
        "n_constraints": 0,
        "min_h": float("inf"),
    }
    return u_clipped, "unconstrained", info


# =============================================================================
# Tier 1: Learned Feasibility Classifier
# =============================================================================

class FeasibilityClassifier:
    """SVM classifier that predicts whether a state will make CBF-QP infeasible.

    Trained iteratively alongside the RL policy. The classifier learns the
    boundary of feasible states, enabling proactive avoidance of infeasible
    regions before the CBF-QP solver encounters them.

    The decision_function provides a soft margin that can be used as an
    additional CBF-like constraint to keep the robot in feasible regions.

    Reference: Paper N13 (CASRL) — converges in ~3 iterations.
    """

    def __init__(self, min_samples: int = 50):
        """Initialize classifier.

        Args:
            min_samples: Minimum total samples before training is allowed.
        """
        self._model = None
        self._feasible_data: list[np.ndarray] = []
        self._infeasible_data: list[np.ndarray] = []
        self._min_samples = min_samples
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def n_feasible(self) -> int:
        return len(self._feasible_data)

    @property
    def n_infeasible(self) -> int:
        return len(self._infeasible_data)

    def collect_data(self, state: np.ndarray, is_feasible: bool):
        """Record a state and its feasibility outcome.

        Args:
            state: Robot state [x, y, theta].
            is_feasible: Whether the CBF-QP was feasible at this state.
        """
        if is_feasible:
            self._feasible_data.append(state.copy())
        else:
            self._infeasible_data.append(state.copy())

    def train(self) -> bool:
        """Train the SVM classifier on collected data.

        Returns:
            True if training succeeded, False if insufficient data.
        """
        n_total = len(self._feasible_data) + len(self._infeasible_data)
        if n_total < self._min_samples:
            return False

        if len(self._infeasible_data) == 0 or len(self._feasible_data) == 0:
            # Need both classes to train
            return False

        try:
            from sklearn.svm import SVC
        except ImportError:
            # scikit-learn not available — classifier won't train
            return False

        X = np.vstack(self._feasible_data + self._infeasible_data)
        y = np.array(
            [1] * len(self._feasible_data) + [0] * len(self._infeasible_data)
        )

        self._model = SVC(kernel="rbf", gamma="scale", class_weight="balanced")
        self._model.fit(X, y)
        self._trained = True
        return True

    def is_feasible(self, state: np.ndarray) -> bool:
        """Predict whether the state is in the feasible region.

        Returns True (optimistic) if the classifier is not yet trained.
        """
        if self._model is None:
            return True  # Optimistic before training
        return bool(self._model.predict(state.reshape(1, -1))[0] == 1)

    def get_feasibility_margin(self, state: np.ndarray) -> float:
        """Return distance to feasibility boundary (positive = feasible).

        Similar to a CBF value: positive means in the safe (feasible) region,
        negative means in the infeasible region.

        Returns 1.0 if the classifier is not yet trained.
        """
        if self._model is None:
            return 1.0
        return float(self._model.decision_function(state.reshape(1, -1))[0])

    def clear_data(self):
        """Clear collected data (e.g., for new training iteration)."""
        self._feasible_data.clear()
        self._infeasible_data.clear()

    def get_metrics(self) -> dict:
        """Return classifier metrics."""
        n_total = len(self._feasible_data) + len(self._infeasible_data)
        return {
            "n_feasible": len(self._feasible_data),
            "n_infeasible": len(self._infeasible_data),
            "n_total": n_total,
            "infeasibility_rate": (
                len(self._infeasible_data) / n_total if n_total > 0 else 0.0
            ),
            "trained": self._trained,
        }


# =============================================================================
# SafeActionResolver: Unified interface for all 3 tiers
# =============================================================================

class SafeActionResolver:
    """Chains Tier 1 → Tier 2 → Tier 3 into a unified safety interface.

    Usage:
        resolver = SafeActionResolver(cbf_filter)
        safe_action, method, info = resolver.resolve(
            nominal_action, state, obstacles, opponent_state,
        )

    Args:
        cbf_filter: VCPCBFFilter instance.
        obstacles: Static obstacles list.
        use_feasibility_classifier: Whether to use Tier 1.
        classifier: FeasibilityClassifier instance (or None to create one).
    """

    def __init__(
        self,
        cbf_filter: VCPCBFFilter,
        obstacles: list[dict] | None = None,
        use_feasibility_classifier: bool = False,
        classifier: Optional[FeasibilityClassifier] = None,
    ):
        self.cbf_filter = cbf_filter
        self.obstacles = obstacles or []
        self.use_feasibility_classifier = use_feasibility_classifier
        self.classifier = classifier or FeasibilityClassifier()

        # Metrics
        self.n_exact = 0
        self.n_relaxed = 0
        self.n_backup = 0
        self.n_total = 0
        self._method_counts: dict[str, int] = {}

    def resolve(
        self,
        nominal_action: np.ndarray,
        state: np.ndarray,
        obstacles: list[dict] | None = None,
        opponent_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, str, dict]:
        """Resolve a safe action using the 3-tier system.

        Args:
            nominal_action: Raw action from policy [v, omega].
            state: Robot state [x, y, theta].
            obstacles: Obstacles (overrides self.obstacles if given).
            opponent_state: Opponent state [x, y, theta], or None.

        Returns:
            (safe_action, method, info):
            - safe_action: Safe action to execute.
            - method: Resolution method used.
            - info: Dict with resolver details.
        """
        obs = obstacles if obstacles is not None else self.obstacles
        self.n_total += 1

        # Get typed constraints
        typed_constraints = self.cbf_filter.get_constraints(
            state, opponent_state, obs,
        )

        # Tier 2: Try hierarchical QP relaxation
        u_safe, method, info = solve_cbf_qp_with_relaxation(
            nominal_action,
            typed_constraints,
            alpha=self.cbf_filter.alpha,
            v_min=0.0,
            v_max=self.cbf_filter.v_max,
            omega_min=-self.cbf_filter.omega_max,
            omega_max=self.cbf_filter.omega_max,
            w_v=self.cbf_filter.w_v,
            w_omega=self.cbf_filter.w_omega,
        )

        # Track feasibility for Tier 1 classifier
        if self.use_feasibility_classifier:
            is_feasible = method == "exact"
            self.classifier.collect_data(state, is_feasible)

        # If unconstrained fallback returned (all constraints dropped),
        # use Tier 3 backup instead
        if method == "unconstrained":
            u_safe = backup_controller(
                state,
                obstacles=obs,
                arena_half_w=self.cbf_filter.arena_half_w,
                arena_half_h=self.cbf_filter.arena_half_h,
                v_max=self.cbf_filter.v_max,
                omega_max=self.cbf_filter.omega_max,
            )
            method = "backup"
            info["relaxation_method"] = "backup"
            self.n_backup += 1
        elif method == "exact":
            self.n_exact += 1
        else:
            self.n_relaxed += 1

        # Track method counts
        self._method_counts[method] = self._method_counts.get(method, 0) + 1

        # Add min_h for safety reward
        if typed_constraints:
            info["min_h"] = min(c.h for c in typed_constraints)
        else:
            info["min_h"] = float("inf")

        info["method"] = method
        return u_safe, method, info

    def get_metrics(self) -> dict:
        """Return resolver metrics."""
        return {
            "n_total": self.n_total,
            "n_exact": self.n_exact,
            "n_relaxed": self.n_relaxed,
            "n_backup": self.n_backup,
            "exact_rate": self.n_exact / self.n_total if self.n_total > 0 else 1.0,
            "relaxed_rate": self.n_relaxed / self.n_total if self.n_total > 0 else 0.0,
            "backup_rate": self.n_backup / self.n_total if self.n_total > 0 else 0.0,
            "method_counts": dict(self._method_counts),
        }

    def reset_metrics(self):
        """Reset all metrics."""
        self.n_exact = 0
        self.n_relaxed = 0
        self.n_backup = 0
        self.n_total = 0
        self._method_counts.clear()
