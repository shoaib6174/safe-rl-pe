"""Step 1: 2-agent pursuit-evasion environment (Gymnasium).

Unicycle (nonholonomic) dynamics for both agents:
    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = omega

Arena is centered at the origin: [-A/2, A/2]^2. Collisions (walls, disk
obstacles) are resolved by POSITION PROJECTION — the proven approach from the
root codebase (envs/dynamics.py), re-implemented here; pe2 imports nothing
from the old code (plan, assumption 1).

API shape: single Gymnasium env with a JOINT action [v_p, w_p, v_e, w_e]
(both agents act simultaneously each step) and a per-agent Dict observation
(partial-observability vector, pe2/obs.py — FOV+LOS masked opponent, own
pose always present). Rewards are zero-sum terminal only (+1/-1 capture,
-1/+1 timeout); reward shaping belongs to the training steps, not here.

Randomness contract (L1.1): reset(seed=...) seeds the env's own Generator
via the standard Gymnasium path (self.np_random). Nothing here touches
global RNG state.
"""

from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pe2.config import EnvConfig
from pe2.obs import build_observation, observation_space
from pe2.sensing import wrap_angle

# Outcome labels for info["outcome"] / episode logs.
OUTCOME_NONE = "running"
OUTCOME_CAPTURE = "capture"
OUTCOME_TIMEOUT = "timeout"


def euler_step(pose, v: float, omega: float, dt: float) -> tuple[float, float, float]:
    """One UNCLIPPED unicycle Euler step -- the kinematics of this env.

    `pe2.shield.CBFShield.successor_barriers` TRANSCRIBES this function in
    vectorised form rather than calling it (calling it per candidate would
    be ~150 million Python calls across a gate run), so the shield's
    one-step prediction is the env's next state bit for bit only as long as
    the two copies stay in lockstep. `test_hard_rows_equal_env_predicates_bitwise`
    (tests/pe2/test_shield.py) is what binds the two copies together --
    change either without re-running it and the identity is unverified.
    Keep the operation order exactly as written: (v * cos) * dt, then + x.
    """
    x = pose[0] + v * np.cos(pose[2]) * dt
    y = pose[1] + v * np.sin(pose[2]) * dt
    theta = wrap_angle(pose[2] + omega * dt)
    return x, y, theta


def pursuer_step(
    pose,
    v: float,
    omega: float,
    cfg: EnvConfig,
    obstacles: list[dict],
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
) -> tuple[np.ndarray, bool, bool]:
    """One full agent step: Euler unicycle, wall clip, obstacle projection.

    Pure -- no env instance, no mutation of `pose`. This is THE one-step
    dynamics of this environment: `PursuitEvasionEnv.step` advances both
    agents with it (the arithmetic is role-agnostic), and a lookahead safety
    filter can simulate a candidate command by calling it, so the filter's
    prediction is the env's successor bit for bit instead of a second
    transcription that can drift (contrast `euler_step`, which `pe2.shield`
    must transcribe for vectorisation).

    `x_lim` / `y_lim` are the arena bounds already inset by `agent_radius`
    (the env's `_x_min/_x_max`, `_y_min/_y_max`). Returns the new pose and
    the two contact flags (wall, obstacle). Heading is never modified by a
    contact. Keep the operation order exactly as written.
    """
    x_min, x_max = x_lim
    y_min, y_max = y_lim
    new_pose = np.array(pose, dtype=float)

    x, y, theta = euler_step(new_pose, v, omega, cfg.dt)
    wall_contact = False
    if x < x_min:
        x, wall_contact = x_min, True
    elif x > x_max:
        x, wall_contact = x_max, True
    if y < y_min:
        y, wall_contact = y_min, True
    elif y > y_max:
        y, wall_contact = y_max, True
    new_pose[:] = (x, y, theta)

    # Push radially out of any overlapped obstacle; iterate to handle chain
    # reactions (proven pattern, max 3 passes).
    obstacle_contact = False
    for _ in range(3):
        resolved_any = False
        for obs in obstacles:
            dx = new_pose[0] - obs["x"]
            dy = new_pose[1] - obs["y"]
            dist = np.hypot(dx, dy)
            min_dist = obs["radius"] + cfg.agent_radius
            if dist < min_dist:
                obstacle_contact = resolved_any = True
                if dist < 1e-10:
                    # Degenerate: exactly at obstacle center — push along heading
                    nx, ny = np.cos(new_pose[2]), np.sin(new_pose[2])
                else:
                    nx, ny = dx / dist, dy / dist
                new_pose[0] = obs["x"] + nx * min_dist
                new_pose[1] = obs["y"] + ny * min_dist
        if not resolved_any:
            break
    # Projection may push through a wall — re-clip (walls win).
    new_pose[0] = float(np.clip(new_pose[0], x_min, x_max))
    new_pose[1] = float(np.clip(new_pose[1], y_min, y_max))
    return new_pose, wall_contact, obstacle_contact


class PursuitEvasionEnv(gym.Env):
    """1v1 pursuit-evasion with unicycle dynamics in a bounded arena."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        c = self.cfg

        half = c.arena_size / 2.0
        self._x_min = -half + c.agent_radius
        self._x_max = half - c.agent_radius
        self._y_min = -half + c.agent_radius
        self._y_max = half - c.agent_radius

        # Joint action: [v_pursuer, w_pursuer, v_evader, w_evader]
        self.action_space = spaces.Box(
            low=np.array([0.0, -c.omega_max, 0.0, -c.omega_max], dtype=np.float32),
            high=np.array(
                [c.v_max, c.omega_max, c.evader_v_max, c.omega_max], dtype=np.float32
            ),
        )
        # PO observation per agent (pe2/obs.py, Step 2). Scripted baselines
        # read true state directly (L1.5: baselines see truth; only learned
        # policies are sensor-limited).
        self.observation_space = spaces.Dict(
            pursuer=observation_space(c),
            evader=observation_space(c),
        )

        # Episode state (set by reset)
        self.obstacles: list[dict] = []
        self.pursuer_pose = np.zeros(3)
        self.evader_pose = np.zeros(3)
        self.current_step = 0
        self.outcome = OUTCOME_NONE
        self.last_info: dict = {}

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        n_physical = self.cfg.n_physical_obstacles
        if n_physical is None:
            n_physical = self.cfg.n_obstacles
        self.obstacles = self._generate_obstacles(n_physical)
        self.pursuer_pose = self._sample_free_pose()
        while True:
            self.evader_pose = self._sample_free_pose()
            if self._distance() >= self.cfg.min_init_distance:
                break
        self.current_step = 0
        self.outcome = OUTCOME_NONE
        return self._get_obs(), self._build_info()

    def step(self, action: np.ndarray):
        c = self.cfg
        v_p, w_p, v_e, w_e = (float(a) for a in action)
        v_p = float(np.clip(v_p, 0.0, c.v_max))
        w_p = float(np.clip(w_p, -c.omega_max, c.omega_max))
        v_e = float(np.clip(v_e, 0.0, c.evader_v_max))
        w_e = float(np.clip(w_e, -c.omega_max, c.omega_max))

        # Prep phase: pursuer cannot translate (may still turn in place).
        if self.current_step < c.prep_steps:
            v_p = 0.0

        x_lim = (self._x_min, self._x_max)
        y_lim = (self._y_min, self._y_max)
        p_pose, p_wall, p_obs = pursuer_step(
            self.pursuer_pose, v_p, w_p, c, self.obstacles, x_lim, y_lim
        )
        e_pose, e_wall, e_obs = pursuer_step(
            self.evader_pose, v_e, w_e, c, self.obstacles, x_lim, y_lim
        )
        # Write in place: callers may hold a reference to the pose arrays.
        self.pursuer_pose[:] = p_pose
        self.evader_pose[:] = e_pose

        self.current_step += 1

        distance = self._distance()
        terminated = distance <= c.capture_distance
        truncated = not terminated and self.current_step >= c.max_steps

        if terminated:
            self.outcome = OUTCOME_CAPTURE
        elif truncated:
            self.outcome = OUTCOME_TIMEOUT

        # Zero-sum terminal placeholder rewards (shaping is a later step).
        r_pursuer = float(terminated) - float(truncated)
        rewards = {"pursuer": r_pursuer, "evader": -r_pursuer}

        info = self._build_info(
            distance=distance,
            pursuer_wall_contact=p_wall,
            evader_wall_contact=e_wall,
            pursuer_obstacle_contact=p_obs,
            evader_obstacle_contact=e_obs,
        )
        return self._get_obs(), rewards, terminated, truncated, info

    # ---------------------------------------------------------------- reset

    def _generate_obstacles(self, n: int) -> list[dict]:
        """Rejection-sample n non-overlapping disks (proven root-codebase rules)."""
        c = self.cfg
        rng = self.np_random
        half = c.arena_size / 2.0
        obstacles: list[dict] = []
        for _ in range(n):
            for _attempt in range(500):
                radius = rng.uniform(c.obstacle_radius_min, c.obstacle_radius_max)
                span = half - radius - c.obstacle_margin
                if span <= 0:
                    continue
                ox = rng.uniform(-span, span)
                oy = rng.uniform(-span, span)
                if all(
                    np.hypot(ox - o["x"], oy - o["y"])
                    >= radius + o["radius"] + c.obstacle_margin
                    for o in obstacles
                ):
                    obstacles.append(
                        {"x": float(ox), "y": float(oy), "radius": float(radius)}
                    )
                    break
        return obstacles

    def _sample_free_pose(self) -> np.ndarray:
        """Uniform pose inside the arena, not overlapping any obstacle."""
        c = self.cfg
        rng = self.np_random
        for _attempt in range(500):
            x = rng.uniform(self._x_min, self._x_max)
            y = rng.uniform(self._y_min, self._y_max)
            if all(
                np.hypot(x - o["x"], y - o["y"]) >= o["radius"] + c.agent_radius
                for o in self.obstacles
            ):
                theta = rng.uniform(-np.pi, np.pi)
                return np.array([x, y, theta])
        raise RuntimeError("could not place agent — arena too crowded")

    # ---------------------------------------------------------------- misc

    def _distance(self) -> float:
        return float(np.hypot(*(self.pursuer_pose[:2] - self.evader_pose[:2])))

    def _get_obs(self) -> dict:
        return {
            "pursuer": build_observation(
                self.pursuer_pose, self.evader_pose, self.obstacles, self.cfg
            ),
            "evader": build_observation(
                self.evader_pose, self.pursuer_pose, self.obstacles, self.cfg
            ),
        }

    def _build_info(self, **extra) -> dict:
        info = {
            "outcome": self.outcome,
            "step": self.current_step,
            "distance": self._distance(),
        }
        info.update(extra)
        self.last_info = info
        return info
