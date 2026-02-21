"""Safety filter interface for CBF-based action filtering.

Phase 1: Pass-through (no safety filtering).
Phase 2: VCP-CBF-QP safety filter.
"""

import numpy as np


class SafetyFilter:
    """Base class for safety filtering.

    Phase 1 default: returns action unchanged.
    Phase 2 override: solves CBF-QP to produce safe action.
    """

    def filter_action(
        self,
        action: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """Return a safe action given the current state.

        Args:
            action: Nominal action [v, omega].
            state: Robot state [x, y, theta].

        Returns:
            Safe action [v, omega]. In Phase 1, returns action unchanged.
        """
        return action

    def get_metrics(self) -> dict:
        """Return safety metrics.

        Phase 1: empty dict.
        Phase 2+: CBF values, intervention count, etc.
        """
        return {}
