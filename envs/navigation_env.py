"""Navigation environment for AMS-DRL cold-start (Phase S0).

Wraps the PE environment for single-agent goal-reaching and fleeing tasks.
Used to pre-train the evader with basic locomotion, obstacle avoidance,
and fleeing instincts before adversarial training begins.

The observation space is compatible with PartialObsWrapper's Dict format
(obs_history, lidar, state) so policy weights transfer directly to self-play.
The key difference: d_to_opp/bearing are replaced by d_to_goal/bearing_to_goal.

Modes:
  - goal_reaching: Navigate to random goals. Reward = distance shaping + goal bonus.
  - flee: Evade a slow scripted pursuer. Reward = survival time + distance from pursuer.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.sensors import FOVSensor, LidarSensor, ObservationHistoryBuffer
from envs.partial_obs_wrapper import DEFAULT_SENSOR_CONFIG, PartialObsWrapper


class NavigationEnv(gym.Wrapper):
    """Cold-start environment for evader pre-training.

    Produces observations compatible with PartialObsWrapper format so that
    the same SB3 model architecture (PartialObsFeaturesExtractor) can be used,
    and weights transfer directly to the self-play phase.

    In goal-reaching mode:
      - Agent navigates to random goals in the arena
      - d_to_opp/bearing become d_to_goal/bearing_to_goal
      - Reward: distance shaping + goal bonus + collision penalty

    In flee mode:
      - A slow scripted pursuer chases the agent
      - d_to_opp/bearing are actual sensor readings of the pursuer
      - Reward: survival time + distance from pursuer

    Args:
        env: Base PursuitEvasionEnv instance.
        role: Which agent is the trainee ('evader' for cold-start).
        include_flee_phase: Whether to include flee episodes (50/50 split).
        sensor_config: Sensor configuration dict.
        history_length: Observation history length (K).
        max_steps: Maximum steps per episode.
        goal_radius: Distance threshold for goal success.
        max_goals_per_episode: Number of goals to reach per episode.
        flee_pursuer_speed_ratio: Pursuer speed as fraction of max (flee mode).
    """

    def __init__(
        self,
        env,
        role: str = "evader",
        include_flee_phase: bool = True,
        sensor_config: dict | None = None,
        history_length: int = 10,
        max_steps: int = 300,
        goal_radius: float = 0.5,
        max_goals_per_episode: int = 3,
        flee_pursuer_speed_ratio: float = 0.5,
    ):
        super().__init__(env)
        self.role = role
        self.include_flee = include_flee_phase
        self.max_steps = max_steps
        self.goal_radius = goal_radius
        self.max_goals = max_goals_per_episode
        self.flee_speed_ratio = flee_pursuer_speed_ratio

        # Find base env
        base = env
        while hasattr(base, "env"):
            base = base.env
        self._base_env = base

        # Sensor pipeline (same as PartialObsWrapper)
        config = sensor_config or DEFAULT_SENSOR_CONFIG
        self.fov_sensor = FOVSensor(**config["fov"])
        self.lidar_sensor = LidarSensor(**config["lidar"])
        self.history_length = history_length
        self.n_rays = config["lidar"].get("n_rays", 36)
        self.lidar_max_range = config["lidar"].get("max_range", 5.0)
        self.buffer = ObservationHistoryBuffer(
            K=history_length, obs_dim=PartialObsWrapper.RAW_OBS_DIM
        )

        # Episode state
        self.goal = None
        self.goals_reached = 0
        self.current_step = 0
        self.is_flee_mode = False
        self._rng = np.random.default_rng()

        # Action and observation spaces (match PartialObsWrapper)
        if role == "pursuer":
            self.action_space = self._base_env.pursuer_action_space
        else:
            self.action_space = self._base_env.evader_action_space

        self.observation_space = spaces.Dict({
            "obs_history": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(history_length, PartialObsWrapper.RAW_OBS_DIM),
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

    def reset(self, seed=None, options=None):
        """Reset environment for a new episode."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        obs, info = self.env.reset(seed=seed, options=options)
        self.buffer.reset()
        self.current_step = 0
        self.goals_reached = 0

        # Decide mode for this episode
        if self.include_flee:
            self.is_flee_mode = self._rng.random() < 0.5
        else:
            self.is_flee_mode = False

        if not self.is_flee_mode:
            self.goal = self._sample_goal()
        else:
            self.goal = None

        obs_dict = self._build_obs_dict()
        info["mode"] = "flee" if self.is_flee_mode else "goal_reaching"
        return obs_dict, info

    def step(self, action):
        """Step the environment with trainee's action."""
        self.current_step += 1

        # In flee mode, generate pursuer action (simple chase)
        if self.is_flee_mode:
            pursuer_action = self._scripted_pursuer_action()
        else:
            # In goal-reaching, opponent does nothing (stands still)
            pursuer_action = np.array([0.0, 0.0], dtype=np.float32)

        # Step base env with both actions
        if self.role == "evader":
            obs, rewards, terminated, truncated, info = self.env.step(
                pursuer_action, action
            )
        else:
            obs, rewards, terminated, truncated, info = self.env.step(
                action, pursuer_action
            )

        # Compute navigation reward (overrides base env reward)
        reward, nav_terminated = self._compute_reward(terminated)

        # Check time limit
        truncated = self.current_step >= self.max_steps

        obs_dict = self._build_obs_dict()

        info["goals_reached"] = self.goals_reached
        info["mode"] = "flee" if self.is_flee_mode else "goal_reaching"

        return obs_dict, reward, nav_terminated or terminated, truncated, info

    def _compute_reward(self, base_terminated: bool) -> tuple[float, bool]:
        """Compute navigation-specific reward.

        Returns:
            (reward, terminated)
        """
        base = self._base_env
        own_state = base.evader_state if self.role == "evader" else base.pursuer_state

        if self.is_flee_mode:
            return self._flee_reward(own_state, base_terminated)
        else:
            return self._goal_reward(own_state)

    def _goal_reward(self, own_state: np.ndarray) -> tuple[float, bool]:
        """Reward for goal-reaching mode."""
        d_to_goal = np.hypot(
            own_state[0] - self.goal[0],
            own_state[1] - self.goal[1],
        )
        d_max = np.hypot(self._base_env.arena_width, self._base_env.arena_height)

        # Distance shaping: closer is better
        reward = -0.5 * d_to_goal / d_max

        # Goal reached bonus
        if d_to_goal < self.goal_radius:
            reward += 10.0
            self.goals_reached += 1
            if self.goals_reached >= self.max_goals:
                return reward, True  # Episode done after max goals
            # Sample new goal
            self.goal = self._sample_goal()

        # Boundary penalty (check if close to walls)
        margin = 0.5
        if (own_state[0] < margin or own_state[0] > self._base_env.arena_width - margin
                or own_state[1] < margin or own_state[1] > self._base_env.arena_height - margin):
            reward -= 0.1

        return reward, False

    def _flee_reward(self, own_state: np.ndarray, captured: bool) -> tuple[float, bool]:
        """Reward for flee mode."""
        base = self._base_env
        pursuer_state = base.pursuer_state if self.role == "evader" else base.evader_state

        d_to_pursuer = np.hypot(
            own_state[0] - pursuer_state[0],
            own_state[1] - pursuer_state[1],
        )
        d_max = np.hypot(base.arena_width, base.arena_height)

        # Survival + distance reward: further from pursuer is better
        reward = 0.5 * d_to_pursuer / d_max

        # Capture penalty (if pursuer catches evader)
        if captured:
            reward -= 10.0
            return reward, True

        return reward, False

    def _scripted_pursuer_action(self) -> np.ndarray:
        """Simple pursuit policy: head directly toward the evader."""
        base = self._base_env

        if self.role == "evader":
            pursuer_state = base.pursuer_state
            target_state = base.evader_state
            v_max = base.pursuer_v_max
            omega_max = base.pursuer_omega_max
        else:
            pursuer_state = base.evader_state
            target_state = base.pursuer_state
            v_max = base.evader_v_max
            omega_max = base.evader_omega_max

        dx = target_state[0] - pursuer_state[0]
        dy = target_state[1] - pursuer_state[1]
        angle_to_target = np.arctan2(dy, dx)
        angle_diff = angle_to_target - pursuer_state[2]
        # Normalize to [-pi, pi]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        v = v_max * self.flee_speed_ratio
        omega = np.clip(angle_diff * 3.0, -omega_max, omega_max)

        return np.array([v, omega], dtype=np.float32)

    def _build_obs_dict(self) -> dict:
        """Build observation dict compatible with PartialObsWrapper format.

        In goal-reaching mode: d_to_opp/bearing become d_to_goal/bearing_to_goal.
        In flee mode: d_to_opp/bearing are actual sensor readings of the pursuer.
        """
        base = self._base_env

        if self.role == "evader":
            own_state = base.evader_state
            own_action = base.evader_action
        else:
            own_state = base.pursuer_state
            own_action = base.pursuer_action

        if self.is_flee_mode:
            # Flee mode: use actual FOV sensor to detect pursuer
            if self.role == "evader":
                target_state = base.pursuer_state
            else:
                target_state = base.evader_state
            d_to_target, bearing = self.fov_sensor.detect(own_state, target_state)
        else:
            # Goal-reaching mode: compute distance/bearing to goal
            dx = self.goal[0] - own_state[0]
            dy = self.goal[1] - own_state[1]
            d_to_target = np.hypot(dx, dy)
            bearing = np.arctan2(dy, dx) - own_state[2]
            # Normalize bearing to [-pi, pi]
            bearing = (bearing + np.pi) % (2 * np.pi) - np.pi

            # Cap distance at FOV range for consistency with sensor output
            if d_to_target > self.fov_sensor.fov_range:
                d_to_target = -1.0
                bearing = 0.0

        # Lidar scan (same as PartialObsWrapper)
        arena_bounds = {
            "x_min": 0.0,
            "x_max": base.arena_width,
            "y_min": 0.0,
            "y_max": base.arena_height,
        }
        lidar = self.lidar_sensor.scan(own_state, base.obstacles, arena_bounds)

        # State features: [x, y, theta, v, omega, d_to_target, bearing]
        state_features = np.array([
            own_state[0],
            own_state[1],
            own_state[2],
            own_action[0],
            own_action[1],
            d_to_target,
            bearing,
        ], dtype=np.float32)

        # Raw obs = state_features + lidar = 43 dims
        raw_obs = np.concatenate([state_features, lidar]).astype(np.float32)
        self.buffer.add(raw_obs)

        return {
            "obs_history": self.buffer.get_history(),
            "lidar": lidar.reshape(1, -1).astype(np.float32),
            "state": state_features,
        }

    def _sample_goal(self) -> np.ndarray:
        """Sample a random collision-free goal position in the arena."""
        base = self._base_env
        margin = 1.0

        for _ in range(100):
            x = self._rng.uniform(margin, base.arena_width - margin)
            y = self._rng.uniform(margin, base.arena_height - margin)
            pos = np.array([x, y])

            # Check obstacle collisions
            collision = False
            for obs in base.obstacles:
                dist = np.hypot(pos[0] - obs["x"], pos[1] - obs["y"])
                if dist < obs["radius"] + 0.5:  # margin around obstacles
                    collision = True
                    break

            if not collision:
                return pos

        # Fallback: center of arena
        return np.array([base.arena_width / 2, base.arena_height / 2])
