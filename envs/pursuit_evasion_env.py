"""Core Pursuit-Evasion Gymnasium environment with unicycle dynamics.

Two-player zero-sum differential game:
- Pursuer: minimizes time to capture the evader
- Evader: maximizes time to capture (or escapes entirely)
- Capture: ||p_P - p_E|| <= r_capture
- Timeout: episode ends at T_max if no capture

The environment exposes a dual-action step API:
    step(pursuer_action, evader_action) -> obs, rewards, terminated, truncated, info

Use SingleAgentPEWrapper (wrappers.py) to adapt for single-agent SB3 training.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.dynamics import clip_action, unicycle_step
from envs.observations import ObservationBuilder
from envs.rewards import RewardComputer


class PursuitEvasionEnv(gym.Env):
    """1v1 Pursuit-Evasion environment with unicycle dynamics.

    Arena is centered at (0, 0), spanning [-W/2, W/2] x [-H/2, H/2].
    Both agents are unicycle robots with bounded velocity and angular velocity.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        arena_width: float = 20.0,
        arena_height: float = 20.0,
        dt: float = 0.05,
        max_steps: int = 1200,
        capture_radius: float = 0.5,
        collision_radius: float = 0.3,
        robot_radius: float = 0.15,
        pursuer_v_max: float = 1.0,
        pursuer_omega_max: float = 2.84,
        evader_v_max: float = 1.0,
        evader_omega_max: float = 2.84,
        min_init_distance: float = 3.0,
        max_init_distance: float = 15.0,
        distance_scale: float = 1.0,
        capture_bonus: float = 100.0,
        timeout_penalty: float = -50.0,
        render_mode: str | None = None,
    ):
        super().__init__()

        # Arena parameters
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.dt = dt
        self.max_steps = max_steps

        # Game parameters
        self.capture_radius = capture_radius
        self.collision_radius = collision_radius
        self.robot_radius = robot_radius
        self.min_init_distance = min_init_distance
        self.max_init_distance = max_init_distance

        # Robot parameters
        self.pursuer_v_max = pursuer_v_max
        self.pursuer_omega_max = pursuer_omega_max
        self.evader_v_max = evader_v_max
        self.evader_omega_max = evader_omega_max

        # Reward computer
        arena_diagonal = np.sqrt(arena_width**2 + arena_height**2)
        self.reward_computer = RewardComputer(
            distance_scale=distance_scale,
            capture_bonus=capture_bonus,
            timeout_penalty=timeout_penalty,
            d_max=arena_diagonal,
        )

        # Observation builder (uses pursuer's v_max/omega_max for normalization;
        # since both agents have same limits in Phase 1, this is fine)
        self.obs_builder = ObservationBuilder(
            arena_width=arena_width,
            arena_height=arena_height,
            v_max=pursuer_v_max,
            omega_max=pursuer_omega_max,
        )

        # Action spaces: [v, omega] for each agent
        # Pursuer action space (used by wrappers)
        self.pursuer_action_space = spaces.Box(
            low=np.array([0.0, -pursuer_omega_max], dtype=np.float32),
            high=np.array([pursuer_v_max, pursuer_omega_max], dtype=np.float32),
        )
        self.evader_action_space = spaces.Box(
            low=np.array([0.0, -evader_omega_max], dtype=np.float32),
            high=np.array([evader_v_max, evader_omega_max], dtype=np.float32),
        )

        # For Gymnasium compatibility, action_space is pursuer's (wrapper overrides)
        self.action_space = self.pursuer_action_space

        # Observation space: 14D per agent
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_builder.obs_dim,),
            dtype=np.float32,
        )

        # Rendering
        self.render_mode = render_mode
        self.renderer = None
        if render_mode is not None:
            from envs.rendering import PERenderer
            self.renderer = PERenderer(
                arena_w=arena_width,
                arena_h=arena_height,
            )

        # State variables (initialized in reset)
        self.pursuer_state = None  # [x, y, theta]
        self.evader_state = None   # [x, y, theta]
        self.pursuer_action = np.zeros(2, dtype=np.float32)  # [v, omega]
        self.evader_action = np.zeros(2, dtype=np.float32)
        self.current_step = 0
        self.prev_distance = 0.0
        self.min_distance_this_ep = float("inf")
        self.captured = False
        self.timed_out = False
        self.last_reward_pursuer = 0.0
        self.last_reward_evader = 0.0

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Reset environment with random initial positions.

        Returns:
            obs: Dict with 'pursuer' and 'evader' observation arrays.
            info: Dict with initial state information.
        """
        super().reset(seed=seed)

        # Generate initial positions with minimum separation constraint
        self.pursuer_state, self.evader_state = self._random_initial_states()

        # Reset actions
        self.pursuer_action = np.zeros(2, dtype=np.float32)
        self.evader_action = np.zeros(2, dtype=np.float32)

        # Reset counters
        self.current_step = 0
        self.captured = False
        self.timed_out = False
        self.prev_distance = self._compute_distance()
        self.min_distance_this_ep = self.prev_distance
        self.last_reward_pursuer = 0.0
        self.last_reward_evader = 0.0

        # Reset renderer trails
        if self.renderer is not None:
            self.renderer.reset_trails()

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(
        self,
        pursuer_action: np.ndarray,
        evader_action: np.ndarray,
    ) -> tuple[dict, dict, bool, bool, dict]:
        """Advance environment by one timestep.

        Args:
            pursuer_action: [v, omega] for the pursuer.
            evader_action: [v, omega] for the evader.

        Returns:
            obs: Dict with 'pursuer' and 'evader' observations.
            rewards: Dict with 'pursuer' and 'evader' rewards.
            terminated: True if capture occurred.
            truncated: True if episode timed out.
            info: Dict with episode metrics.
        """
        # Clip actions to physical bounds
        p_v, p_omega = clip_action(
            pursuer_action[0], pursuer_action[1],
            self.pursuer_v_max, self.pursuer_omega_max,
        )
        e_v, e_omega = clip_action(
            evader_action[0], evader_action[1],
            self.evader_v_max, self.evader_omega_max,
        )

        # Store actions for observation
        self.pursuer_action = np.array([p_v, p_omega], dtype=np.float32)
        self.evader_action = np.array([e_v, e_omega], dtype=np.float32)

        # Advance dynamics
        px, py, ptheta, p_wall = unicycle_step(
            self.pursuer_state[0], self.pursuer_state[1], self.pursuer_state[2],
            p_v, p_omega, self.dt,
            self.arena_width, self.arena_height, self.robot_radius,
        )
        ex, ey, etheta, e_wall = unicycle_step(
            self.evader_state[0], self.evader_state[1], self.evader_state[2],
            e_v, e_omega, self.dt,
            self.arena_width, self.arena_height, self.robot_radius,
        )

        self.pursuer_state = np.array([px, py, ptheta])
        self.evader_state = np.array([ex, ey, etheta])

        # Update step counter
        self.current_step += 1

        # Compute distance and check termination
        d_curr = self._compute_distance()
        self.min_distance_this_ep = min(self.min_distance_this_ep, d_curr)

        self.captured = d_curr <= self.capture_radius
        self.timed_out = (not self.captured) and (self.current_step >= self.max_steps)

        terminated = self.captured
        truncated = self.timed_out

        # Compute rewards
        r_p, r_e = self.reward_computer.compute(
            d_curr=d_curr,
            d_prev=self.prev_distance,
            captured=self.captured,
            timed_out=self.timed_out,
        )
        self.last_reward_pursuer = r_p
        self.last_reward_evader = r_e
        self.prev_distance = d_curr

        # Build observations
        obs = self._get_obs()
        rewards = {"pursuer": r_p, "evader": r_e}
        info = self._get_info()

        # Add episode metrics at termination
        if terminated or truncated:
            info["episode_metrics"] = {
                "captured": self.captured,
                "capture_time": self.current_step * self.dt,
                "min_distance": self.min_distance_this_ep,
                "episode_length": self.current_step,
                "pursuer_wall_contacts": int(p_wall),
                "evader_wall_contacts": int(e_wall),
            }

        if self.render_mode == "human":
            self._render_frame()

        return obs, rewards, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the environment (for rgb_array mode)."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()

    def _random_initial_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate random initial states with minimum separation."""
        half_w = self.arena_width / 2.0 - self.robot_radius
        half_h = self.arena_height / 2.0 - self.robot_radius

        for _ in range(1000):  # max attempts
            px = self.np_random.uniform(-half_w, half_w)
            py = self.np_random.uniform(-half_h, half_h)
            ptheta = self.np_random.uniform(-np.pi, np.pi)

            ex = self.np_random.uniform(-half_w, half_w)
            ey = self.np_random.uniform(-half_h, half_h)
            etheta = self.np_random.uniform(-np.pi, np.pi)

            dist = np.sqrt((px - ex) ** 2 + (py - ey) ** 2)
            if self.min_init_distance <= dist <= self.max_init_distance:
                return (
                    np.array([px, py, ptheta]),
                    np.array([ex, ey, etheta]),
                )

        # Fallback: place agents at fixed positions
        return (
            np.array([-5.0, 0.0, 0.0]),
            np.array([5.0, 0.0, np.pi]),
        )

    def _compute_distance(self) -> float:
        """Euclidean distance between agents."""
        dx = self.pursuer_state[0] - self.evader_state[0]
        dy = self.pursuer_state[1] - self.evader_state[1]
        return float(np.sqrt(dx**2 + dy**2))

    def _get_obs(self) -> dict:
        """Build observations for both agents."""
        obs_pursuer = self.obs_builder.build(
            self_state=self.pursuer_state,
            self_action=self.pursuer_action,
            opp_state=self.evader_state,
            opp_action=self.evader_action,
        )
        obs_evader = self.obs_builder.build(
            self_state=self.evader_state,
            self_action=self.evader_action,
            opp_state=self.pursuer_state,
            opp_action=self.pursuer_action,
        )
        return {"pursuer": obs_pursuer, "evader": obs_evader}

    def _get_info(self) -> dict:
        """Build info dict."""
        return {
            "distance": self._compute_distance(),
            "pursuer_state": self.pursuer_state.copy(),
            "evader_state": self.evader_state.copy(),
            "step": self.current_step,
        }

    def _get_render_state(self) -> dict:
        """Build state dict for the renderer."""
        return {
            "pursuer_pos": self.pursuer_state[:2].copy(),
            "pursuer_heading": self.pursuer_state[2],
            "evader_pos": self.evader_state[:2].copy(),
            "evader_heading": self.evader_state[2],
            "step": self.current_step,
            "dt": self.dt,
            "distance": self._compute_distance(),
            "reward": self.last_reward_pursuer,
            "obstacles": [],
        }

    def _render_frame(self) -> np.ndarray | None:
        """Render a single frame."""
        if self.renderer is None:
            return None
        return self.renderer.render_frame(self.render_mode, self._get_render_state())
