"""Wrapper to convert continuous PE action space to discrete for DQN.

Discretizes [v, omega] into n_velocities x n_angular_velocities discrete actions.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscreteActionWrapper(gym.ActionWrapper):
    """Converts continuous action space to discrete for DQN.

    Maps integer actions to [v, omega] via a lookup table.

    Args:
        env: Environment with Box action space of shape (2,).
        n_velocities: Number of discrete velocity levels.
        n_angular: Number of discrete angular velocity levels.
    """

    def __init__(
        self,
        env: gym.Env,
        n_velocities: int = 5,
        n_angular: int = 7,
    ):
        super().__init__(env)

        v_max = env.action_space.high[0]
        omega_max = env.action_space.high[1]

        velocities = np.linspace(0, v_max, n_velocities)
        angulars = np.linspace(-omega_max, omega_max, n_angular)

        self._action_table = []
        for v in velocities:
            for omega in angulars:
                self._action_table.append(np.array([v, omega], dtype=np.float32))

        self._action_table = np.array(self._action_table)
        self.action_space = spaces.Discrete(len(self._action_table))

    def action(self, action: int) -> np.ndarray:
        """Convert discrete action index to continuous [v, omega]."""
        return self._action_table[action]
