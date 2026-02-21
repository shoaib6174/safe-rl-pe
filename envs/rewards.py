"""Reward computation for the pursuit-evasion environment.

Phase 1 reward (pursuer):
    r_P = w1 * (d_prev - d_curr) / d_max   (distance shaping)
        + w2 * I(captured)                   (capture bonus)
        + w3 * I(timeout)                    (timeout penalty)

Evader reward: r_E = -r_P  (zero-sum)

Phase 2 will add: w5 * CBF_margin term via subclassing.
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
