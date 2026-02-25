"""Reward computation for the pursuit-evasion environment.

Phase 1 reward (pursuer):
    r_P = w1 * (d_prev - d_curr) / d_max   (distance shaping)
        + w2 * I(captured)                   (capture bonus)
        + w3 * I(timeout)                    (timeout penalty)

Evader reward: r_E = -r_P  (zero-sum)

Phase 2 adds: w5 * CBF_margin term via SafetyRewardComputer subclass.

Phase 3 adds two reward modes for obstacle-based evasion:

**Mode A (w_occlusion > 0)**: Small occlusion bonus on top of zero-sum.
    Diagnosed as insufficient in Runs L/M — the bonus is drowned out by
    distance shaping and capture bonus.

**Mode B (use_visibility_reward=True)**: OpenAI Hide-and-Seek style.
    Inspired by Baker et al. (ICLR 2020). The evader's reward is decoupled
    from the pursuer's and based on visibility:
        r_evader = visibility_weight * (+1 if hidden, -1 if visible)
                 + survival_bonus per step alive
    The pursuer keeps distance-based reward. Terminal steps (capture/timeout)
    remain zero-sum. When no obstacles exist (curriculum L1/L2), falls back
    to zero-sum distance-based reward automatically.

    This works because:
    - Distance shaping actively teaches fleeing in straight lines (bad vs faster pursuer)
    - Visibility reward directly incentivizes obstacle use
    - Survival bonus aligns reward with the evader's actual objective (stay alive)
    - Early termination from capture naturally penalizes the evader (fewer reward steps)

**PBRS obstacle-seeking (w_obs_approach > 0)**: Potential-Based Reward Shaping.
    Provides a gradient toward the nearest obstacle for the evader:
        r_pbrs = w_obs_approach * (d_obs_prev - d_obs_curr) / d_max
    where d_obs = min surface distance to any obstacle.

    Theoretically justified by Ng et al. (ICML 1999): PBRS with
    Φ(s) = -d_nearest_obstacle is the ONLY additive shaping that preserves
    the optimal policy. Devlin & Kudenko (2012) extend this to multi-agent
    settings (preserves Nash equilibria).

    Designed to combine with visibility reward + prep phase to address
    the Level 3+ collapse (Runs H-O): evader had time (prep) and incentive
    (visibility) but no directional gradient toward obstacles.
"""

import numpy as np


def line_of_sight_blocked(
    p_pos: np.ndarray,
    e_pos: np.ndarray,
    obstacles: list[dict],
) -> bool:
    """Check if any obstacle blocks the line of sight between two positions.

    Uses ray-circle intersection: for each obstacle, checks if the line
    segment from p_pos to e_pos intersects the obstacle circle.

    Args:
        p_pos: Pursuer position [x, y] or [x, y, theta].
        e_pos: Evader position [x, y] or [x, y, theta].
        obstacles: List of obstacle dicts with keys 'x', 'y', 'radius'.

    Returns:
        True if at least one obstacle blocks line of sight.
    """
    if not obstacles:
        return False

    # Direction vector from pursuer to evader
    dx = e_pos[0] - p_pos[0]
    dy = e_pos[1] - p_pos[1]
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-12:
        return False  # Same position

    for obs in obstacles:
        # Vector from pursuer to obstacle center
        ox = obs["x"] - p_pos[0]
        oy = obs["y"] - p_pos[1]
        r = obs["radius"]

        # Project obstacle center onto the line segment
        # t = dot(p->obs, p->evader) / |p->evader|^2
        t = (ox * dx + oy * dy) / seg_len_sq

        # Clamp to segment [0, 1]
        t = max(0.0, min(1.0, t))

        # Closest point on segment to obstacle center
        closest_x = t * dx
        closest_y = t * dy

        # Distance from obstacle center to closest point
        dist_sq = (ox - closest_x) ** 2 + (oy - closest_y) ** 2

        if dist_sq <= r * r:
            return True

    return False


def nearest_obstacle_distance(
    pos: np.ndarray,
    obstacles: list[dict],
) -> float:
    """Compute the surface distance to the nearest obstacle.

    Surface distance = center-to-center distance minus obstacle radius,
    clamped to >= 0 (inside obstacle = 0).

    Args:
        pos: Agent position [x, y] or [x, y, theta].
        obstacles: List of obstacle dicts with keys 'x', 'y', 'radius'.

    Returns:
        Minimum surface distance to any obstacle, or inf if no obstacles.
    """
    if not obstacles:
        return float("inf")

    min_dist = float("inf")
    for obs in obstacles:
        center_dist = np.sqrt((pos[0] - obs["x"]) ** 2 + (pos[1] - obs["y"]) ** 2)
        surface_dist = max(center_dist - obs["radius"], 0.0)
        if surface_dist < min_dist:
            min_dist = surface_dist

    return min_dist


class RewardComputer:
    """Base reward computer.

    Supports three evader reward modes:

    **Default (zero-sum)**: r_evader = -r_pursuer. Standard for non-obstacle levels.

    **Mode A (w_occlusion > 0)**: Small occlusion bonus on top of zero-sum.
        Diagnosed as insufficient — bonus drowned out by distance shaping.

    **Mode B (use_visibility_reward=True)**: OpenAI Hide-and-Seek style.
        Evader gets per-step visibility reward + survival bonus, decoupled from
        pursuer. Terminal steps remain zero-sum. Falls back to zero-sum when
        no obstacles present.

    Args:
        distance_scale: Weight on distance-shaping reward.
        capture_bonus: Reward for capture (positive for pursuer).
        timeout_penalty: Penalty for timeout (negative for pursuer).
        d_max: Maximum possible distance (for normalization).
        w_occlusion: Weight on evader occlusion bonus (Mode A, default 0).
        use_visibility_reward: Enable visibility-based evader reward (Mode B).
        visibility_weight: Scale for +1/-1 visibility signal (Mode B).
        survival_bonus: Per-step bonus for evader being alive (Mode B).
        w_obs_approach: Weight on PBRS obstacle-seeking term (default 0).
    """

    def __init__(
        self,
        distance_scale: float = 1.0,
        capture_bonus: float = 100.0,
        timeout_penalty: float = -100.0,
        d_max: float = 28.28,  # diagonal of 20x20 arena
        w_occlusion: float = 0.0,
        use_visibility_reward: bool = False,
        visibility_weight: float = 1.0,
        survival_bonus: float = 0.0,
        w_obs_approach: float = 0.0,
    ):
        self.distance_scale = distance_scale
        self.capture_bonus = capture_bonus
        self.timeout_penalty = timeout_penalty
        self.d_max = d_max
        self.w_occlusion = w_occlusion
        self.use_visibility_reward = use_visibility_reward
        self.visibility_weight = visibility_weight
        self.survival_bonus = survival_bonus
        self.w_obs_approach = w_obs_approach

    def compute(
        self,
        d_curr: float,
        d_prev: float,
        captured: bool,
        timed_out: bool,
        pursuer_pos: np.ndarray | None = None,
        evader_pos: np.ndarray | None = None,
        obstacles: list[dict] | None = None,
        d_obs_curr: float | None = None,
        d_obs_prev: float | None = None,
        **kwargs,
    ) -> tuple[float, float]:
        """Compute rewards for both agents.

        Args:
            d_curr: Current distance between agents.
            d_prev: Previous distance between agents.
            captured: Whether capture occurred this step.
            timed_out: Whether episode timed out this step.
            pursuer_pos: Pursuer position [x, y, theta] (for occlusion).
            evader_pos: Evader position [x, y, theta] (for occlusion).
            obstacles: List of obstacle dicts (for occlusion).
            d_obs_curr: Current evader-to-nearest-obstacle surface distance.
            d_obs_prev: Previous evader-to-nearest-obstacle surface distance.

        Returns:
            (r_pursuer, r_evader): Reward tuple.
        """
        # ── Pursuer reward: always distance-based ──
        r_dist = self.distance_scale * (d_prev - d_curr) / self.d_max
        r_capture = self.capture_bonus if captured else 0.0
        r_timeout = self.timeout_penalty if timed_out else 0.0
        r_pursuer = r_dist + r_capture + r_timeout

        # ── Evader reward ──
        # Check if visibility mode should be active this step
        can_use_visibility = (
            self.use_visibility_reward
            and obstacles  # obstacles exist in this episode
            and pursuer_pos is not None
            and evader_pos is not None
        )

        if can_use_visibility and not captured and not timed_out:
            # Mode B: visibility-based evader reward (decoupled from pursuer)
            hidden = line_of_sight_blocked(pursuer_pos, evader_pos, obstacles)
            r_evader = self.visibility_weight * (1.0 if hidden else -1.0)
            r_evader += self.survival_bonus
        else:
            # Zero-sum fallback (no obstacles, terminal step, or mode disabled)
            r_evader = -r_pursuer

            # Mode A: small occlusion bonus on top of zero-sum
            if (
                self.w_occlusion > 0
                and pursuer_pos is not None
                and evader_pos is not None
                and obstacles
                and not captured
            ):
                if line_of_sight_blocked(pursuer_pos, evader_pos, obstacles):
                    r_evader += self.w_occlusion

        # ── PBRS obstacle-seeking (evader only, non-terminal steps) ──
        if (
            self.w_obs_approach > 0
            and d_obs_curr is not None
            and d_obs_prev is not None
            and np.isfinite(d_obs_curr)
            and np.isfinite(d_obs_prev)
            and not captured
            and not timed_out
        ):
            # F(s, s') ≈ Φ(s_prev) - Φ(s_curr) = (-d_prev) - (-d_curr)
            #           = d_obs_prev - d_obs_curr  (positive when approaching)
            r_pbrs = self.w_obs_approach * (d_obs_prev - d_obs_curr) / self.d_max
            r_evader += r_pbrs

        return float(r_pursuer), float(r_evader)


class SafetyRewardComputer(RewardComputer):
    """Reward computer with CBF safety margin shaping term.

    Supports two modes:

    **Mode 1 (default, w_safety > 0)**:
        r_safety = w_safety * clamp(h_min / h_ref, 0, 1)
        Positive reward for maintaining large safety margins.

    **Mode 2 (CBF-RL dual reward, w_cbf_penalty > 0)**:
        Inspired by CBF-RL (arXiv:2510.14959). Dual signal:
        r_cbf = -w_cbf_penalty * max(0, -hdot_min)
                -w_intervention * (1 - exp(-||u_nom - u_safe||^2 / sigma^2))

        First term: penalizes CBF condition violations (negative hdot + alpha*h).
        Second term: penalizes when safety filter modifies the action.
        Together these teach the policy to propose inherently safe actions,
        reducing filter intervention rate at deployment.

    Both modes can be combined. Set unused weights to 0 to disable.

    Reference: CBF-RL (arXiv:2510.14959) for the dual reward approach.

    Args:
        w_safety: Weight on margin-based safety reward (mode 1). Default 0.05.
        h_ref: Reference CBF value for normalization. Default 1.0.
        w_cbf_penalty: Weight on CBF condition violation penalty (mode 2). Default 0.0.
        w_intervention: Weight on filter intervention penalty (mode 2). Default 0.0.
        sigma_sq: Bandwidth for intervention penalty Gaussian. Default 0.1.
        **kwargs: Passed to RewardComputer.
    """

    def __init__(
        self,
        w_safety: float = 0.05,
        h_ref: float = 1.0,
        w_cbf_penalty: float = 0.0,
        w_intervention: float = 0.0,
        sigma_sq: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.w_safety = w_safety
        self.h_ref = h_ref
        self.w_cbf_penalty = w_cbf_penalty
        self.w_intervention = w_intervention
        self.sigma_sq = sigma_sq

    def compute(
        self,
        d_curr: float,
        d_prev: float,
        captured: bool,
        timed_out: bool,
        h_min: float | None = None,
        cbf_condition_value: float | None = None,
        intervention_magnitude: float | None = None,
        **kwargs,
    ) -> tuple[float, float]:
        """Compute rewards with safety shaping.

        Args:
            d_curr: Current distance between agents.
            d_prev: Previous distance between agents.
            captured: Whether capture occurred this step.
            timed_out: Whether episode timed out this step.
            h_min: Minimum CBF value across all constraints.
            cbf_condition_value: Minimum (a_v*v + a_omega*omega + alpha*h)
                across constraints. Negative means CBF condition is violated.
            intervention_magnitude: ||u_safe - u_nominal|| from safety filter.

        Returns:
            (r_pursuer, r_evader): Reward tuple. Always zero-sum.
        """
        # Base reward
        r_p_base, r_e_base = super().compute(
            d_curr, d_prev, captured, timed_out,
        )

        # Mode 1: Positive reward for maintaining safety margins
        r_safety = 0.0
        if h_min is not None and self.w_safety > 0:
            r_safety = self.w_safety * float(np.clip(h_min / self.h_ref, 0.0, 1.0))

        # Mode 2: CBF-RL dual penalty
        r_cbf_penalty = 0.0
        if cbf_condition_value is not None and self.w_cbf_penalty > 0:
            # Penalize negative CBF condition (constraint violation)
            r_cbf_penalty = -self.w_cbf_penalty * max(0.0, -cbf_condition_value)

        r_intervention = 0.0
        if intervention_magnitude is not None and self.w_intervention > 0:
            # Penalize filter intervention (exponential decay)
            r_intervention = -self.w_intervention * (
                1.0 - np.exp(-intervention_magnitude**2 / self.sigma_sq)
            )

        r_pursuer = r_p_base + r_safety + r_cbf_penalty + r_intervention
        r_evader = -r_pursuer  # zero-sum

        return float(r_pursuer), float(r_evader)
