"""Wrappers for adapting the dual-action PE environment to single-agent interfaces.

SingleAgentPEWrapper: Wraps PursuitEvasionEnv for SB3's single-agent step(action) API.
The opponent uses a fixed policy (frozen model or random).
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
