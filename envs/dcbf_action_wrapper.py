"""DCBF action filter wrapper for safe RL training.

Wraps an environment (typically PartialObsWrapper) and applies the
VCP-CBF discrete-time filter to every action using TRUE state.
The policy sees partial observations; the safety filter sees full state.

Wrapping order: PursuitEvasionEnv → SingleAgentPEWrapper → PartialObsWrapper → DCBFActionWrapper
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from safety.vcp_cbf import VCPCBFFilter


class DCBFActionWrapper(gym.Wrapper):
    """Applies DCBF safety filter to actions during training.

    The policy outputs a nominal action from partial observations.
    This wrapper filters it through the DCBF using TRUE state,
    then passes the safe action to the underlying environment.

    Also tracks safety metrics (intervention rate, min h values).

    Args:
        env: Wrapped environment (must have a _base_env or be traversable
             to PursuitEvasionEnv for true state access).
        role: 'pursuer' or 'evader'.
        gamma: DCBF decay rate (default 0.2).
        cbf_kwargs: Additional kwargs for VCPCBFFilter constructor.
    """

    def __init__(
        self,
        env: gym.Env,
        role: str = "pursuer",
        gamma: float = 0.2,
        cbf_kwargs: dict | None = None,
    ):
        super().__init__(env)
        self.role = role
        self.gamma = gamma

        # Find base env for true state access
        base = env
        while hasattr(base, "env"):
            base = base.env
        self._base_env = base

        # Create CBF filter with base env parameters
        kwargs = cbf_kwargs or {}
        kwargs.setdefault("d", 0.1)
        kwargs.setdefault("v_max", base.pursuer_v_max)
        kwargs.setdefault("omega_max", base.pursuer_omega_max)
        kwargs.setdefault("w_v", 150.0)
        kwargs.setdefault("w_omega", 1.0)
        self.cbf_filter = VCPCBFFilter(**kwargs)

        # Episode-level metrics
        self._episode_interventions = 0
        self._episode_steps = 0
        self._episode_min_h = float("inf")

    def reset(self, **kwargs):
        self._episode_interventions = 0
        self._episode_steps = 0
        self._episode_min_h = float("inf")
        self.cbf_filter.reset_metrics()
        return self.env.reset(**kwargs)

    def step(self, action):
        # Get true state for DCBF
        base = self._base_env
        if self.role == "pursuer":
            own_state = base.pursuer_state
            opp_state = base.evader_state
        else:
            own_state = base.evader_state
            opp_state = base.pursuer_state

        # Apply DCBF filter
        safe_action, cbf_info = self.cbf_filter.dcbf_filter_action(
            action=np.asarray(action, dtype=np.float64),
            state=own_state,
            dt=base.dt,
            gamma=self.gamma,
            obstacles=base.obstacles,
            opponent_state=opp_state,
        )

        # Track metrics
        self._episode_steps += 1
        if cbf_info.get("intervention", False):
            self._episode_interventions += 1
        min_h = cbf_info.get("min_h", float("inf"))
        self._episode_min_h = min(self._episode_min_h, min_h)

        # Step with safe action
        obs, reward, terminated, truncated, info = self.env.step(safe_action)

        # Attach safety info to step info
        info["dcbf_intervention"] = cbf_info.get("intervention", False)
        info["dcbf_min_h"] = min_h

        # On episode end, add summary metrics
        if terminated or truncated:
            rate = (self._episode_interventions / max(self._episode_steps, 1))
            info["dcbf_intervention_rate"] = rate
            info["dcbf_episode_min_h"] = self._episode_min_h
            info["dcbf_episode_interventions"] = self._episode_interventions

        return obs, reward, terminated, truncated, info

    @property
    def intervention_rate(self) -> float:
        """Current episode intervention rate."""
        if self._episode_steps == 0:
            return 0.0
        return self._episode_interventions / self._episode_steps
