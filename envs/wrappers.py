"""Wrappers for adapting the dual-action PE environment to single-agent interfaces.

SingleAgentPEWrapper: Wraps PursuitEvasionEnv for SB3's single-agent step(action) API.
The opponent uses a fixed policy (frozen model or random).

FixedSpeedWrapper: Fixes v=v_max, reducing the action space to omega-only (1D).
This is critical for PPO training because Gaussian policies with clipped action
spaces struggle to learn boundary-optimal actions (v should always be v_max for pursuit).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SingleAgentPEWrapper(gym.Env):
    """Wraps the 2-agent PE env for single-agent training with SB3.

    The controlled agent takes actions; the opponent uses a fixed policy.
    Observation and action spaces match the controlled agent's perspective.

    Args:
        env: The base PursuitEvasionEnv instance.
        role: 'pursuer' or 'evader' — which agent this wrapper controls.
        opponent_policy: Callable with predict(obs) -> (action, _), or None for random.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env,
        role: str = "pursuer",
        opponent_policy=None,
    ):
        super().__init__()

        if role not in ("pursuer", "evader"):
            raise ValueError(f"role must be 'pursuer' or 'evader', got '{role}'")

        self.env = env
        self.role = role
        self.opponent_policy = opponent_policy
        self._last_obs = None

        # Set action and observation spaces
        if role == "pursuer":
            self.action_space = env.pursuer_action_space
        else:
            self.action_space = env.evader_action_space

        self.observation_space = env.observation_space
        self.render_mode = env.render_mode

    def set_opponent(self, opponent_policy):
        """Update the opponent policy (used in self-play)."""
        self.opponent_policy = opponent_policy

    def reset(self, seed=None, options=None):
        """Reset the underlying environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        # Reset stateful opponent adapters (e.g., PartialObsOpponentAdapter)
        if self.opponent_policy is not None and hasattr(self.opponent_policy, "reset"):
            self.opponent_policy.reset()
        return obs[self.role], info

    def step(self, action):
        """Step with controlled agent's action; opponent acts automatically."""
        # Get opponent observation
        opp_role = "evader" if self.role == "pursuer" else "pursuer"
        opp_obs = self._last_obs[opp_role]

        # Get opponent action
        if self.opponent_policy is None:
            if self.role == "pursuer":
                opp_action = self.env.evader_action_space.sample()
            else:
                opp_action = self.env.pursuer_action_space.sample()
        else:
            opp_action, _ = self.opponent_policy.predict(opp_obs, deterministic=False)

        # Step environment
        if self.role == "pursuer":
            obs, rewards, terminated, truncated, info = self.env.step(action, opp_action)
        else:
            obs, rewards, terminated, truncated, info = self.env.step(opp_action, action)

        self._last_obs = obs
        return obs[self.role], rewards[self.role], terminated, truncated, info

    def render(self):
        """Delegate rendering to the base environment."""
        return self.env.render()

    def close(self):
        """Close the base environment."""
        self.env.close()


class FixedSpeedWrapper(gym.ActionWrapper):
    """Fix linear velocity to v_max, only learn angular velocity (omega).

    Reduces the action space from 2D [v, omega] to 1D [omega]. This is
    critical for PPO because Gaussian policies with clipped action spaces
    struggle to learn boundary-optimal values — the pursuer should always
    go at max speed, but PPO tends to output low velocities.

    Args:
        env: Environment to wrap (must have 2D [v, omega] action space).
        v_max: Fixed linear velocity to use.
    """

    def __init__(self, env: gym.Env, v_max: float):
        super().__init__(env)
        self.v_max = v_max
        # Extract omega_max from the inner action space
        omega_max = env.action_space.high[1]
        self.action_space = spaces.Box(
            low=np.array([-omega_max], dtype=np.float32),
            high=np.array([omega_max], dtype=np.float32),
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """Convert 1D [omega] to 2D [v_max, omega]."""
        return np.array([self.v_max, action[0]], dtype=np.float32)


class FixedSpeedModelAdapter:
    """Wraps an SB3 model trained with FixedSpeedWrapper to expand 1D → 2D actions.

    When a model is trained with FixedSpeedWrapper (1D omega-only), its predict()
    returns 1D actions. This adapter expands them to 2D [v_max, omega] for use
    as an opponent in SingleAgentPEWrapper.

    Args:
        model: SB3 model trained with FixedSpeedWrapper.
        v_max: Fixed linear velocity to prepend.
    """

    def __init__(self, model, v_max: float):
        self.model = model
        self.v_max = v_max

    def predict(self, obs, deterministic: bool = False):
        action, state = self.model.predict(obs, deterministic=deterministic)
        if action.shape[-1] == 1:
            action = np.array([self.v_max, action[0]], dtype=np.float32)
        return action, state
