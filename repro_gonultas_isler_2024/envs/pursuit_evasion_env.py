import numpy as np
from typing import Tuple
from envs.sensor import WedgeSensor


class PursuitEvasionEnv:
    def __init__(self, config: dict = None, max_steps: int = None):
        # Defaults matching config.yaml
        self.arena_size = 16.0
        self.max_steps = max_steps if max_steps is not None else 400
        self.capture_radius = 0.15
        self.dt = 0.1

        # Pursuer params
        self.L = 0.30 if config is None else config.get("pursuer", {}).get("wheelbase", 0.30)
        self.pursuer_v_max = 2.5
        self.pursuer_v_min = -1.0
        self.delta_max = 0.34
        self.delta_min = -0.34
        self.accel_max = 2.0
        self.accel_min = -2.0
        self.steer_rate_max = 3.2
        self.steer_rate_min = -3.2

        # Evader params
        self.evader_v_max = 1.5
        self.evader_v_min = -1.5
        self.evader_accel_max = 9.81
        self.evader_accel_min = -9.81

        # Reward params
        self.lambda_capture = 1000
        self.lambda_time = 1
        self.lambda_distance = 1
        self.w_vis = 0.0 if config is None else config.get("maddpg", {}).get("w_vis", 0.0)

        # Sensors
        self.pursuer_sensor = WedgeSensor(2.094, 6.0)
        self.evader_sensor = WedgeSensor(6.283, 1000.0)

        self.pursuer_state = None
        self.evader_state = None
        self.step_count = 0
        self.np_random = np.random.default_rng()

    def reset(self, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        half = self.arena_size / 2
        min_dist = 3.0

        while True:
            px = self.np_random.uniform(-half, half)
            py = self.np_random.uniform(-half, half)
            ex = self.np_random.uniform(-half, half)
            ey = self.np_random.uniform(-half, half)
            if np.hypot(px - ex, py - ey) >= min_dist:
                break

        # Pursuer: [x, y, delta, v, psi]
        v_p = self.np_random.uniform(self.pursuer_v_min, self.pursuer_v_max)
        psi_p = self.np_random.uniform(-np.pi, np.pi)
        self.pursuer_state = np.array([px, py, 0.0, v_p, psi_p], dtype=np.float32)

        # Evader: [x, y, vx, vy]
        vx_e = self.np_random.uniform(self.evader_v_min, self.evader_v_max)
        vy_e = self.np_random.uniform(self.evader_v_min, self.evader_v_max)
        self.evader_state = np.array([ex, ey, vx_e, vy_e], dtype=np.float32)

        self.step_count = 0
        return self._get_observations()

    def step(
        self,
        action_p: np.ndarray,
        action_e: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        # Scale actions from [-1, 1] to actual bounds
        u_delta = np.clip(action_p[0], -1.0, 1.0) * self.steer_rate_max
        u_v = np.clip(action_p[1], -1.0, 1.0) * self.accel_max
        u_x = np.clip(action_e[0], -1.0, 1.0) * self.evader_accel_max
        u_y = np.clip(action_e[1], -1.0, 1.0) * self.evader_accel_max

        # Pursuer dynamics (Euler integration)
        x, y, delta, v, psi = self.pursuer_state
        delta = np.clip(delta + u_delta * self.dt, self.delta_min, self.delta_max)
        v = np.clip(v + u_v * self.dt, self.pursuer_v_min, self.pursuer_v_max)
        psi = psi + (v / self.L) * np.tan(delta) * self.dt
        x = x + v * np.cos(psi) * self.dt
        y = y + v * np.sin(psi) * self.dt
        self.pursuer_state = np.array([x, y, delta, v, psi], dtype=np.float32)

        # Evader dynamics
        ex, ey, evx, evy = self.evader_state
        evx = np.clip(evx + u_x * self.dt, self.evader_v_min, self.evader_v_max)
        evy = np.clip(evy + u_y * self.dt, self.evader_v_min, self.evader_v_max)
        ex = ex + evx * self.dt
        ey = ey + evy * self.dt
        self.evader_state = np.array([ex, ey, evx, evy], dtype=np.float32)

        # Clip positions to arena
        half = self.arena_size / 2
        self.pursuer_state[0] = np.clip(self.pursuer_state[0], -half, half)
        self.pursuer_state[1] = np.clip(self.pursuer_state[1], -half, half)
        self.evader_state[0] = np.clip(self.evader_state[0], -half, half)
        self.evader_state[1] = np.clip(self.evader_state[1], -half, half)

        self.step_count += 1

        # Compute distance
        dist = float(np.hypot(
            self.pursuer_state[0] - self.evader_state[0],
            self.pursuer_state[1] - self.evader_state[1],
        ))

        # Check capture
        done = False
        if dist <= 2 * self.capture_radius:
            r_p = self.lambda_capture
            done = True
        elif self.step_count >= self.max_steps:
            r_p = -self.lambda_capture
            done = True
        else:
            r_p = -(self.lambda_time + self.lambda_distance * dist)

        # Add vis_reward for pursuer if evader is in FOV (only during episode, not at terminal)
        if self.w_vis > 0.0 and not done:
            pursuer_pos = self.pursuer_state[:2]
            pursuer_yaw = self.pursuer_state[4]
            evader_pos = self.evader_state[:2]
            if self.pursuer_sensor.is_visible(pursuer_pos, pursuer_yaw, evader_pos):
                r_p += self.w_vis

        r_e = -r_p
        obs_p, obs_e = self._get_observations()
        return obs_p, obs_e, r_p, r_e, done

    def _get_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        obs_p = self._build_obs(
            is_pursuer=True,
            own_state=self.pursuer_state,
            opp_state=self.evader_state,
        )
        obs_e = self._build_obs(
            is_pursuer=False,
            own_state=self.evader_state,
            opp_state=self.pursuer_state,
        )
        return obs_p, obs_e

    def _build_obs(
        self,
        is_pursuer: bool,
        own_state: np.ndarray,
        opp_state: np.ndarray,
    ) -> np.ndarray:
        if is_pursuer:
            own_pos = own_state[:2]
            own_v = np.array([
                own_state[3] * np.cos(own_state[4]),
                own_state[3] * np.sin(own_state[4]),
            ])
            agent_yaw = own_state[4]
            sensor = self.pursuer_sensor
        else:
            own_pos = own_state[:2]
            own_v = own_state[2:4]
            agent_yaw = 0.0  # evader yaw irrelevant for omnidirectional
            sensor = self.evader_sensor

        opp_pos = opp_state[:2]
        opp_v = np.array([
            opp_state[3] * np.cos(opp_state[4]) if opp_state.shape[0] == 5 else opp_state[2],
            opp_state[3] * np.sin(opp_state[4]) if opp_state.shape[0] == 5 else opp_state[3],
        ])

        visible = sensor.is_visible(own_pos, agent_yaw, opp_pos)
        I = 1.0 if visible else -1.0

        if visible:
            opp_x, opp_y = opp_pos
            opp_vx, opp_vy = opp_v
        else:
            opp_x, opp_y = 0.0, 0.0
            opp_vx, opp_vy = 0.0, 0.0

        dist = float(np.hypot(own_pos[0] - opp_pos[0], own_pos[1] - opp_pos[1]))
        t_norm = self.step_count / self.max_steps

        # Raw observation (unnormalized)
        obs = np.array([
            own_pos[0], own_pos[1],
            own_v[0], own_v[1],
            opp_x, opp_y,
            opp_vx, opp_vy,
            I,
            t_norm,
            dist,
        ], dtype=np.float32)

        return self._normalize_obs(obs, is_pursuer)

    def _normalize_obs(self, obs: np.ndarray, is_pursuer: bool) -> np.ndarray:
        half = self.arena_size / 2
        max_dist = self.arena_size * np.sqrt(2)
        norm = obs.copy()

        # Positions: x, y
        norm[0] = norm[0] / half
        norm[1] = norm[1] / half

        # Own velocities
        if is_pursuer:
            norm[2] = (norm[2] - 0.75) / 1.75  # pursuer v in [-1, 2.5]
            norm[3] = (norm[3] - 0.75) / 1.75
        else:
            norm[2] = norm[2] / 1.5
            norm[3] = norm[3] / 1.5

        # Opponent state (already 0 if not visible)
        norm[4] = norm[4] / half
        norm[5] = norm[5] / half
        if is_pursuer:
            norm[6] = (norm[6] - 0.75) / 1.75
            norm[7] = (norm[7] - 0.75) / 1.75
        else:
            norm[6] = norm[6] / 1.5
            norm[7] = norm[7] / 1.5

        # I is already +/- 1
        # t_norm is already in [0, 1]
        norm[10] = norm[10] / max_dist

        return np.clip(norm, -1.0, 1.0)
