"""Partial observability wrapper for the pursuit-evasion environment.

Wraps PursuitEvasionEnv to produce partial observations with belief state.
Owns the observation history buffer and sensor pipeline.
Exposes a gymnasium.spaces.Dict observation space for SB3 compatibility.

The DCBF safety filter operates on TRUE state (separate channel).
The policy only sees partial observations through this wrapper.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.sensors import FOVSensor, LidarSensor, ObservationHistoryBuffer


# Default sensor configuration matching Phase 3 spec
DEFAULT_SENSOR_CONFIG = {
    "fov": {"fov_angle": 60.0, "fov_range": 10.0},
    "lidar": {"n_rays": 36, "max_range": 5.0, "noise_std": 0.02},
}


class PartialObsWrapper(gym.Wrapper):
    """Wraps PursuitEvasionEnv to produce partial observations with belief state.

    Pipeline: true state -> sensors -> raw partial obs -> history buffer -> obs_dict

    The obs_dict contains:
      - 'obs_history': [K, raw_obs_dim] observation history for BiMDN
      - 'lidar': [1, n_rays] current lidar readings for Conv1D
      - 'state': [7] own pose + velocity + FOV detection [x, y, theta, v, omega, d_to_opp, bearing]

    The wrapper exposes get_true_state() for the DCBF filter to use separately.

    Args:
        env: PursuitEvasionEnv instance.
        role: 'pursuer' or 'evader' (which agent this wrapper represents).
        sensor_config: Sensor configuration dict. Uses defaults if None.
        history_length: Number of timesteps to keep in observation history (K).
    """

    # Raw observation dimensions:
    # 3 (own pose: x, y, theta) + 2 (velocity: v, omega) + 2 (FOV: d_to_opp, bearing) + 36 (lidar) = 43
    # Own pose is always known (odometry/proprioception) — partial observability
    # is about the OPPONENT, not self-state.
    RAW_OBS_DIM = 43

    def __init__(
        self,
        env: gym.Env,
        role: str = "pursuer",
        sensor_config: dict | None = None,
        history_length: int = 10,
    ):
        super().__init__(env)
        self.role = role
        config = sensor_config or DEFAULT_SENSOR_CONFIG

        # Find the base PursuitEvasionEnv by traversing the wrapper chain.
        # This allows PartialObsWrapper to work whether it wraps the base env
        # directly or wraps SingleAgentPEWrapper (or any other intermediary).
        base = env
        while hasattr(base, "env"):
            base = base.env
        self._base_env = base

        # Initialize sensors
        self.fov_sensor = FOVSensor(**config["fov"])
        self.lidar_sensor = LidarSensor(**config["lidar"])
        self.history_length = history_length
        self.n_rays = config["lidar"].get("n_rays", 36)
        self.lidar_max_range = config["lidar"].get("max_range", 5.0)

        # Observation history buffer
        self.buffer = ObservationHistoryBuffer(K=history_length, obs_dim=self.RAW_OBS_DIM)

        # Last known opponent info (for ghost marker)
        self.last_known_opp_pos = None
        self.steps_since_seen = 0

        # Define Dict observation space for SB3
        self.observation_space = spaces.Dict({
            "obs_history": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(history_length, self.RAW_OBS_DIM),
                dtype=np.float32,
            ),
            "lidar": spaces.Box(
                low=0.0, high=self.lidar_max_range,
                shape=(1, self.n_rays),
                dtype=np.float32,
            ),
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(7,),
                dtype=np.float32,
            ),
        })

        # Action space: inherit from the wrapped env if it already has one
        # (e.g., SingleAgentPEWrapper sets it), otherwise get from base env.
        if hasattr(env, "action_space") and env.action_space is not None:
            self.action_space = env.action_space
        elif role == "pursuer":
            self.action_space = self._base_env.pursuer_action_space
        else:
            self.action_space = self._base_env.evader_action_space

    def reset(self, **kwargs):
        """Reset environment and observation buffer."""
        obs, info = self.env.reset(**kwargs)
        self.buffer.reset()
        self.last_known_opp_pos = None
        self.steps_since_seen = 0
        obs_dict = self._build_obs_dict()
        return obs_dict, info

    def step(self, action):
        """Step the underlying environment and return partial observations.

        Note: This wrapper is designed for single-agent use (one role).
        For dual-agent stepping, use with SingleAgentPEWrapper or
        AMSDRLSelfPlay which handles both agents.
        """
        # The underlying env needs both actions. This wrapper expects
        # to be used inside a higher-level wrapper (e.g., SingleAgentPEWrapper)
        # that handles opponent actions. We pass through directly.
        obs, rewards, terminated, truncated, info = self.env.step(action)

        obs_dict = self._build_obs_dict()

        # Extract the relevant reward for this role
        if isinstance(rewards, dict):
            reward = rewards.get(self.role, 0.0)
        else:
            reward = rewards

        return obs_dict, reward, terminated, truncated, info

    def _build_obs_dict(self) -> dict:
        """Convert environment state to partial observation dict."""
        # Get agent states from the base PursuitEvasionEnv
        base = self._base_env
        if self.role == "pursuer":
            own_state = base.pursuer_state
            own_action = base.pursuer_action
            target_state = base.evader_state
        else:
            own_state = base.evader_state
            own_action = base.evader_action
            target_state = base.pursuer_state

        # FOV detection
        d_to_opp, bearing = self.fov_sensor.detect(own_state, target_state)

        # Update last known position
        if d_to_opp >= 0:
            self.last_known_opp_pos = target_state[:2].copy()
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1

        # Lidar scan
        arena_bounds = {
            "x_min": 0.0,
            "x_max": base.arena_width,
            "y_min": 0.0,
            "y_max": base.arena_height,
        }
        lidar = self.lidar_sensor.scan(
            own_state,
            base.obstacles,
            arena_bounds,
        )

        # State features: [x, y, theta, v, omega, d_to_opp, bearing]
        # Own pose (x, y, theta) is always known from odometry — this is NOT
        # partial observability. Without own pose, the BiMDN cannot convert
        # relative sensor readings (d_to_opp, bearing) to absolute opponent
        # position estimates.
        state_features = np.array([
            own_state[0],    # own x position
            own_state[1],    # own y position
            own_state[2],    # own heading (theta)
            own_action[0],   # current velocity
            own_action[1],   # current angular velocity
            d_to_opp,        # -1 if not detected
            bearing,          # 0 if not detected
        ], dtype=np.float32)

        # Build raw observation: [x, y, theta, v, omega, d_to_opp, bearing, lidar(36)] = 43
        raw_obs = np.concatenate([state_features, lidar]).astype(np.float32)
        self.buffer.add(raw_obs)

        return {
            "obs_history": self.buffer.get_history(),
            "lidar": lidar.reshape(1, -1).astype(np.float32),
            "state": state_features,
        }

    def get_true_state(self) -> np.ndarray:
        """Get true state for DCBF filter (bypasses partial observability).

        Returns:
            [x_P, y_P, theta_P, x_E, y_E, theta_E] full state vector.
        """
        return np.concatenate([
            self._base_env.pursuer_state,
            self._base_env.evader_state,
        ]).astype(np.float32)

    def get_sensor_info(self) -> dict:
        """Get current sensor state for rendering overlays.

        Returns:
            Dict with fov_detected, last_known_pos, steps_since_seen, lidar, etc.
        """
        return {
            "fov_detected": self.steps_since_seen == 0,
            "last_known_opp_pos": self.last_known_opp_pos,
            "steps_since_seen": self.steps_since_seen,
            "fov_half_angle": self.fov_sensor.fov_half_angle,
            "fov_range": self.fov_sensor.fov_range,
        }
