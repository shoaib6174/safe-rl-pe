"""Reward computation for the pursuit-evasion environment.

Phase 1 reward (pursuer):
    r_P = w1 * (d_prev - d_curr) / d_max   (distance shaping)
        + w2 * I(captured)                   (capture bonus)
        + w3 * I(timeout)                    (timeout penalty)

Evader reward: r_E = -r_P  (zero-sum)

Phase 2 adds: w5 * CBF_margin term via SafetyRewardComputer subclass.
"""

import numpy as np


class RewardComputer:
    """Base reward computer. Phase 2 adds w5 * CBF_margin term via subclass."""

    def __init__(
        self,
        distance_scale: float = 1.0,
        capture_bonus: float = 100.0,
        timeout_penalty: float = -50.0,
        d_max: float = 28.28,  # diagonal of 20x20 arena
    ):
        self.distance_scale = distance_scale
        self.capture_bonus = capture_bonus
        self.timeout_penalty = timeout_penalty
        self.d_max = d_max

    def compute(
        self,
        d_curr: float,
        d_prev: float,
        captured: bool,
        timed_out: bool,
        **kwargs,
    ) -> tuple[float, float]:
        """Compute rewards for both agents.

        Args:
            d_curr: Current distance between agents.
            d_prev: Previous distance between agents.
            captured: Whether capture occurred this step.
            timed_out: Whether episode timed out this step.

        Returns:
            (r_pursuer, r_evader): Reward tuple. Always zero-sum.
        """
        # Distance shaping (dense signal)
        r_dist = self.distance_scale * (d_prev - d_curr) / self.d_max

        # Terminal rewards
        r_capture = self.capture_bonus if captured else 0.0
        r_timeout = self.timeout_penalty if timed_out else 0.0

        r_pursuer = r_dist + r_capture + r_timeout
        r_evader = -r_pursuer  # zero-sum

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
