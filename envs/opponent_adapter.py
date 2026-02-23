"""Adapts partial-obs SB3 policies for use as opponents in SingleAgentPEWrapper.

SingleAgentPEWrapper provides full-state observations to the opponent policy.
But Phase 3 policies expect partial-obs Dict observations (obs_history, lidar, state).

PartialObsOpponentAdapter bridges this gap by:
1. Receiving full-state obs from SingleAgentPEWrapper
2. Processing it through its own FOV sensor, Lidar sensor, and history buffer
3. Feeding the resulting Dict obs to the partial-obs SB3 model
4. Returning the predicted action

Each adapter instance is stateful (maintains its own observation history buffer),
so it must be reset at episode boundaries.
"""

from __future__ import annotations

import numpy as np

from envs.sensors import FOVSensor, LidarSensor, ObservationHistoryBuffer
from envs.partial_obs_wrapper import DEFAULT_SENSOR_CONFIG, PartialObsWrapper


class PartialObsOpponentAdapter:
    """Wraps a partial-obs SB3 model for use as frozen opponent.

    The adapter maintains its own sensor pipeline (FOV, Lidar, history buffer)
    and converts full-state observations from SingleAgentPEWrapper into the
    Dict observation format expected by the partial-obs policy.

    Args:
        model: SB3 model with partial-obs policy (MultiInputPolicy).
        role: 'pursuer' or 'evader' — role of the opponent being adapted.
        base_env: The base PursuitEvasionEnv instance (for arena bounds, obstacles).
        sensor_config: Sensor configuration dict. Uses defaults if None.
        history_length: Number of timesteps for observation history (K).
        deterministic: Whether to use deterministic actions.
    """

    def __init__(
        self,
        model,
        role: str,
        base_env,
        sensor_config: dict | None = None,
        history_length: int = 10,
        deterministic: bool = False,
    ):
        self.model = model
        self.role = role
        self.base_env = base_env
        self.deterministic = deterministic
        self.history_length = history_length

        config = sensor_config or DEFAULT_SENSOR_CONFIG

        # Own sensor pipeline (independent from the training agent's sensors)
        self.fov_sensor = FOVSensor(**config["fov"])
        self.lidar_sensor = LidarSensor(**config["lidar"])
        self.buffer = ObservationHistoryBuffer(
            K=history_length, obs_dim=PartialObsWrapper.RAW_OBS_DIM
        )

        self.n_rays = config["lidar"].get("n_rays", 36)
        self.lidar_max_range = config["lidar"].get("max_range", 5.0)

        # Track last known opponent position
        self.last_known_opp_pos = None
        self.steps_since_seen = 0

    def predict(self, full_state_obs: np.ndarray, deterministic: bool | None = None) -> tuple:
        """Convert full-state obs to partial obs and predict action.

        This method signature matches SB3's model.predict() interface:
        returns (action, state) where state is None for non-recurrent policies.

        Args:
            full_state_obs: Full-state observation from SingleAgentPEWrapper.
                This is ignored — we read true state from base_env directly.
            deterministic: Override instance deterministic setting.

        Returns:
            (action, None) — SB3-compatible predict output.
        """
        det = deterministic if deterministic is not None else self.deterministic

        # Build partial obs from true state (accessed from base env)
        obs_dict = self._build_obs_dict()

        # Predict using the wrapped model
        action, state = self.model.predict(obs_dict, deterministic=det)
        return action, state

    def _build_obs_dict(self) -> dict:
        """Build partial obs Dict from base env's true state.

        Mirrors PartialObsWrapper._build_obs_dict() but operates independently.
        """
        base = self.base_env

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
        lidar = self.lidar_sensor.scan(own_state, base.obstacles, arena_bounds)

        # State features: [x, y, theta, v, omega, d_to_opp, bearing]
        state_features = np.array([
            own_state[0],
            own_state[1],
            own_state[2],
            own_action[0],
            own_action[1],
            d_to_opp,
            bearing,
        ], dtype=np.float32)

        # Raw obs: [x, y, theta, v, omega, d_to_opp, bearing, lidar(36)] = 43
        raw_obs = np.concatenate([state_features, lidar]).astype(np.float32)
        self.buffer.add(raw_obs)

        return {
            "obs_history": self.buffer.get_history(),
            "lidar": lidar.reshape(1, -1).astype(np.float32),
            "state": state_features,
        }

    def reset(self):
        """Reset the adapter's internal state for a new episode.

        Must be called at episode start to clear the observation history buffer.
        """
        self.buffer.reset()
        self.last_known_opp_pos = None
        self.steps_since_seen = 0
