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

    Adds: r_safety = w_safety * clamp(h_min / h_ref, 0, 1)

    This encourages the agent to maintain larger safety margins, even when
    the CBF filter is not actively intervening. The term is positive when
    the agent is safely away from constraints, providing a dense signal.

    Args:
        w_safety: Weight on safety reward term. Default 0.05.
        h_ref: Reference CBF value for normalization. Values of h_min >= h_ref
            map to the maximum safety reward. Default 1.0.
        **kwargs: Passed to RewardComputer.
    """

    def __init__(
        self,
        w_safety: float = 0.05,
        h_ref: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.w_safety = w_safety
        self.h_ref = h_ref

    def compute(
        self,
        d_curr: float,
        d_prev: float,
        captured: bool,
        timed_out: bool,
        h_min: float | None = None,
        **kwargs,
    ) -> tuple[float, float]:
        """Compute rewards with safety margin shaping.

        Args:
            d_curr: Current distance between agents.
            d_prev: Previous distance between agents.
            captured: Whether capture occurred this step.
            timed_out: Whether episode timed out this step.
            h_min: Minimum CBF value across all constraints (None = no safety term).

        Returns:
            (r_pursuer, r_evader): Reward tuple. Always zero-sum.
        """
        # Base reward
        r_p_base, r_e_base = super().compute(
            d_curr, d_prev, captured, timed_out,
        )

        # Safety margin shaping
        if h_min is not None:
            r_safety = self.w_safety * float(np.clip(h_min / self.h_ref, 0.0, 1.0))
        else:
            r_safety = 0.0

        r_pursuer = r_p_base + r_safety
        r_evader = -r_pursuer  # zero-sum (safety incentivizes safe play for both)

        return float(r_pursuer), float(r_evader)
