"""SafePPO: PPO with CBF-based safe action bounds for Beta policy.

Integrates VCPCBFFilter with PPO training by:
1. Computing safe bounds per step during rollout collection
2. Storing bounds in SafeRolloutBuffer
3. Using stored bounds during PPO update for correct log-prob recomputation

Usage:
    from training.safe_ppo import SafePPO
    from safety.safe_beta_policy import SafeBetaPolicy

    model = SafePPO(
        SafeBetaPolicy, env,
        cbf_filter=VCPCBFFilter(...),
        ...
    )
    model.learn(total_timesteps=100000)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from safety.vcp_cbf import VCPCBFFilter

if TYPE_CHECKING:
    from safety.safe_beta_policy import SafeBetaPolicy


def get_unwrapped_pe_env(env):
    """Unwrap environment to get the base PursuitEvasionEnv.

    Works with SingleAgentPEWrapper and SB3's DummyVecEnv.
    """
    # Handle VecEnv
    if hasattr(env, "envs"):
        env = env.envs[0]
    # Handle Gymnasium wrappers
    while hasattr(env, "env"):
        if hasattr(env, "pursuer_state"):
            return env
        env = env.env
    return env


class CBFBoundsComputer:
    """Computes CBF-derived safe bounds and applies QP filter.

    This class is independent of SB3 and can be used standalone or
    integrated into training via CBFSafetyCallback.

    Args:
        cbf_filter: VCPCBFFilter instance for constraint computation.
        obstacles: Static obstacles list (None if no obstacles).
        use_opponent: Whether to use opponent state for collision CBF.
        bounds_method: "analytical" (fast) or "lp" (exact).
    """

    def __init__(
        self,
        cbf_filter: VCPCBFFilter,
        obstacles: Optional[list] = None,
        use_opponent: bool = True,
        bounds_method: str = "analytical",
    ):
        self.cbf_filter = cbf_filter
        self.obstacles = obstacles
        self.use_opponent = use_opponent
        self.bounds_method = bounds_method

    def compute_bounds(
        self,
        pursuer_state: np.ndarray,
        evader_state: Optional[np.ndarray] = None,
    ) -> tuple[tuple[float, float], tuple[float, float], bool]:
        """Compute safe action bounds for the given state.

        Returns:
            (v_bounds, omega_bounds, feasible)
        """
        opponent = evader_state if self.use_opponent else None
        return self.cbf_filter.compute_safe_bounds(
            pursuer_state,
            obstacles=self.obstacles,
            opponent_state=opponent,
            method=self.bounds_method,
        )

    def compute_and_set_bounds(
        self,
        env,
        policy: "SafeBetaPolicy",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute safe bounds for current state and set on policy.

        Args:
            env: The unwrapped PE environment (has pursuer_state, evader_state).
            policy: SafeBetaPolicy to set bounds on.

        Returns:
            (bounds_low, bounds_high) numpy arrays.
        """
        state = env.pursuer_state
        opponent = env.evader_state if self.use_opponent else None

        v_bounds, omega_bounds, feasible = self.cbf_filter.compute_safe_bounds(
            state,
            obstacles=self.obstacles,
            opponent_state=opponent,
            method=self.bounds_method,
        )

        if feasible:
            policy.set_safe_bounds(v_bounds, omega_bounds)
        else:
            policy.clear_safe_bounds()

        low, high = policy.get_current_bounds_numpy()
        return low, high

    def filter_action(
        self,
        action: np.ndarray,
        env,
    ) -> tuple[np.ndarray, dict]:
        """Apply QP safety filter to sampled action.

        Args:
            action: Raw action from Beta policy.
            env: The unwrapped PE environment.

        Returns:
            (safe_action, info) from VCPCBFFilter.
        """
        state = env.pursuer_state
        opponent = env.evader_state if self.use_opponent else None
        return self.cbf_filter.filter_action(
            action, state, self.obstacles, opponent,
        )


def _get_cbf_safety_callback_class():
    """Lazy import of SB3 BaseCallback and return CBFSafetyCallback class."""
    from stable_baselines3.common.callbacks import BaseCallback

    class CBFSafetyCallback(BaseCallback):
        """SB3 callback wrapper around CBFBoundsComputer.

        Args:
            cbf_computer: CBFBoundsComputer instance.
            verbose: Verbosity level.
        """

        def __init__(
            self,
            cbf_computer: CBFBoundsComputer,
            verbose: int = 0,
        ):
            super().__init__(verbose)
            self.cbf_computer = cbf_computer

        def _on_step(self) -> bool:
            return True

    return CBFSafetyCallback
