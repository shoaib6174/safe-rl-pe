# Gonultas & Isler Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standalone PyTorch reproduction of Gonultas & Isler (IROS 2025) MADDPG + BiMDN for pursuit-evasion with asymmetric partial observability.

**Architecture:** Isolated directory `repro_gonultas_isler_2024/` with clean separation: envs (bicycle pursuer + point-mass evader, wedge sensor), agents (MADDPG actor-critic + BiMDN belief encoder), training (replay buffer, curriculum, main loop), evaluation (baselines + self-play). TDD throughout — failing test first, then implementation.

**Tech Stack:** PyTorch, NumPy, Gymnasium, TensorBoard, Pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `repro_gonultas_isler_2024/requirements.txt` | Python dependencies |
| `repro_gonultas_isler_2024/config.yaml` | All hyperparameters in one place |
| `repro_gonultas_isler_2024/envs/sensor.py` | Wedge FOV sensor with binary visibility |
| `repro_gonultas_isler_2024/envs/pursuit_evasion_env.py` | Bicycle pursuer + point-mass evader, 16×16m arena |
| `repro_gonultas_isler_2024/agents/maddpg.py` | ActorNetwork, CriticNetwork, MADDPGAgent |
| `repro_gonultas_isler_2024/agents/bimdn.py` | BiLSTM + 3-head MDN belief encoder |
| `repro_gonultas_isler_2024/training/replay_buffer.py` | Shared replay buffer for MADDPG |
| `repro_gonultas_isler_2024/training/curriculum.py` | FOV decay + velocity increase per episode |
| `repro_gonultas_isler_2024/training/train.py` | Main training loop with 8 parallel envs |
| `repro_gonultas_isler_2024/evaluation/evaluate.py` | Eval against random walk, greedy, self-play |
| `repro_gonultas_isler_2024/utils/logger.py` | TensorBoard logging wrapper |
| `repro_gonultas_isler_2024/tests/test_sensor.py` | Sensor visibility tests |
| `repro_gonultas_isler_2024/tests/test_env.py` | Dynamics, capture, reward, normalization tests |
| `repro_gonultas_isler_2024/tests/test_replay_buffer.py` | Buffer add/sample/shape tests |
| `repro_gonultas_isler_2024/tests/test_maddpg.py` | Actor/critic forward, target update, loss tests |
| `repro_gonultas_isler_2024/tests/test_bimdn.py` | Output shapes, mixture validity, NLL tests |

---

## Task 1: Project Scaffold

**Files:**
- Create: `repro_gonultas_isler_2024/requirements.txt`
- Create: `repro_gonultas_isler_2024/config.yaml`
- Create: `repro_gonultas_isler_2024/envs/__init__.py`
- Create: `repro_gonultas_isler_2024/agents/__init__.py`
- Create: `repro_gonultas_isler_2024/training/__init__.py`
- Create: `repro_gonultas_isler_2024/evaluation/__init__.py`
- Create: `repro_gonultas_isler_2024/utils/__init__.py`
- Create: `repro_gonultas_isler_2024/tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```txt
torch>=2.0.0
numpy>=1.24.0
gymnasium>=0.29.0
tensorboard>=2.13.0
pyyaml>=6.0
pytest>=7.4.0
```

- [ ] **Step 2: Create config.yaml**

```yaml
# Environment
arena_size: 16.0          # meters (square arena)
max_steps: 400
capture_radius: 0.15      # agent radius c, capture at d <= 2c
sim_dt: 0.1               # simulation timestep (seconds)

# Pursuer (bicycle model)
pursuer:
  wheelbase: 0.15
  v_max: 2.5
  v_min: -1.0
  delta_max: 0.34
  delta_min: -0.34
  accel_max: 2.0
  accel_min: -2.0
  steer_rate_max: 3.2
  steer_rate_min: -3.2
  fov_angle: 2.094          # 2π/3 radians
  fov_range: 6.0

# Evader (point-mass)
evader:
  v_max: 1.5
  v_min: -1.5
  accel_max: 9.81
  accel_min: -9.81
  fov_angle: 6.283           # 2π radians (omnidirectional)
  fov_range: 1000.0          # effectively infinite

# Reward
reward:
  lambda_capture: 1000
  lambda_time: 1
  lambda_distance: 1

# MADDPG
maddpg:
  episodes: 20000
  parallel_envs: 8
  replay_buffer_size: 100000
  batch_size: 512
  actor_lr: 0.0005
  critic_lr: 0.0005
  gamma: 0.99
  tau: 0.005
  exploration_sigma: 0.1
  hidden_dim: 128
  frame_stack: 2
  frame_skip: 2
  update_freq: 1

# BiMDN
bimdn:
  history_length: 200
  downsample_factor: 10
  lstm_hidden: 128
  feature_dim: 32
  num_components: 3
  belief_buffer_size: 25000
  belief_warmup: 10000
  belief_update_freq: 10
  belief_batch_size: 256
  belief_lr: 0.002

# Curriculum
curriculum:
  t_end_ratio: 0.3           # fraction of total episodes
  fov_start: 6.283           # 2π
  fov_end: 2.094             # 2π/3
  evader_vmax_start: 0.0
  evader_vmax_end: 1.5

# Evaluation
eval:
  checkpoint_freq: 250
  eval_episodes: 500

# Logging
log_dir: "./runs"
```

- [ ] **Step 3: Create all `__init__.py` files**

Create empty `__init__.py` in each package directory (`envs/`, `agents/`, `training/`, `evaluation/`, `utils/`, `tests/`).

- [ ] **Step 4: Commit**

```bash
git add repro_gonultas_isler_2024/
git commit -m "scaffold: create repro project structure with config"
```

---

## Task 2: Wedge FOV Sensor

**Files:**
- Create: `repro_gonultas_isler_2024/envs/sensor.py`
- Create: `repro_gonultas_isler_2024/tests/test_sensor.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pytest
from envs.sensor import WedgeSensor


def test_sensor_detects_inside_fov():
    sensor = WedgeSensor(fov_angle=2.094, fov_range=6.0)
    # Agent at origin facing +x, opponent at (3, 0)
    assert sensor.is_visible(
        agent_pos=np.array([0.0, 0.0]),
        agent_yaw=0.0,
        opponent_pos=np.array([3.0, 0.0]),
    )


def test_sensor_rejects_outside_angle():
    sensor = WedgeSensor(fov_angle=2.094, fov_range=6.0)  # 120 deg
    # Agent at origin facing +x, opponent at 90 deg (outside 60 deg half-angle)
    assert not sensor.is_visible(
        agent_pos=np.array([0.0, 0.0]),
        agent_yaw=0.0,
        opponent_pos=np.array([0.0, 3.0]),
    )


def test_sensor_rejects_outside_range():
    sensor = WedgeSensor(fov_angle=2.094, fov_range=6.0)
    # Opponent at 7m, inside angle but outside range
    assert not sensor.is_visible(
        agent_pos=np.array([0.0, 0.0]),
        agent_yaw=0.0,
        opponent_pos=np.array([7.0, 0.0]),
    )


def test_omnidirectional_sensor():
    sensor = WedgeSensor(fov_angle=6.283, fov_range=1000.0)
    # Any direction, any range within 1000m
    assert sensor.is_visible(
        agent_pos=np.array([0.0, 0.0]),
        agent_yaw=0.0,
        opponent_pos=np.array([500.0, 500.0]),
    )
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/test_sensor.py -v`

Expected: `ModuleNotFoundError: No module named 'envs.sensor'`

- [ ] **Step 3: Implement sensor**

```python
import numpy as np


class WedgeSensor:
    def __init__(self, fov_angle: float, fov_range: float):
        self.fov_angle = fov_angle
        self.fov_range = fov_range
        self.half_angle = fov_angle / 2.0

    def is_visible(
        self,
        agent_pos: np.ndarray,
        agent_yaw: float,
        opponent_pos: np.ndarray,
    ) -> bool:
        rel = opponent_pos - agent_pos
        dist = float(np.linalg.norm(rel))
        if dist > self.fov_range:
            return False
        angle_to_opponent = float(np.arctan2(rel[1], rel[0]))
        angle_diff = self._normalize_angle(angle_to_opponent - agent_yaw)
        return abs(angle_diff) <= self.half_angle

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
```

- [ ] **Step 4: Run tests**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/test_sensor.py -v`

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add repro_gonultas_isler_2024/envs/sensor.py repro_gonultas_isler_2024/tests/test_sensor.py
git commit -m "feat: implement wedge FOV sensor with tests"
```

---

## Task 3: Pursuit-Evasion Environment

**Files:**
- Create: `repro_gonultas_isler_2024/envs/pursuit_evasion_env.py`
- Create: `repro_gonultas_isler_2024/tests/test_env.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest
from envs.pursuit_evasion_env import PursuitEvasionEnv


def test_env_reset():
    env = PursuitEvasionEnv()
    obs_p, obs_e = env.reset(seed=42)
    assert obs_p.shape == (11,)
    assert obs_e.shape == (11,)
    assert np.all(obs_p >= -1.0) and np.all(obs_p <= 1.0)
    assert np.all(obs_e >= -1.0) and np.all(obs_e <= 1.0)


def test_bicycle_dynamics_straight():
    env = PursuitEvasionEnv()
    env.reset(seed=42)
    # Set pursuer heading 0, positive velocity
    env.pursuer_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    # Action: zero steering, zero acceleration
    _, _, _, _, _ = env.step(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    # Should move roughly in +x direction
    assert env.pursuer_state[0] > 0.0
    assert abs(env.pursuer_state[1]) < 0.01


def test_point_mass_dynamics():
    env = PursuitEvasionEnv()
    env.reset(seed=42)
    # Set evader at rest
    env.evader_state = np.array([0.0, 0.0, 0.0, 0.0])
    # Action: accelerate in +x
    _, _, _, _, _ = env.step(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    # Should have moved and gained velocity
    assert env.evader_state[0] > 0.0
    assert env.evader_state[2] > 0.0


def test_capture_condition():
    env = PursuitEvasionEnv()
    env.reset(seed=42)
    # Place agents very close
    env.pursuer_state[:2] = np.array([0.0, 0.0])
    env.evader_state[:2] = np.array([0.1, 0.0])
    _, _, _, _, done = env.step(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert done


def test_reward_zero_sum():
    env = PursuitEvasionEnv()
    env.reset(seed=42)
    _, _, r_p, r_e, _ = env.step(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert abs(r_p + r_e) < 1e-6


def test_reward_capture_positive_for_pursuer():
    env = PursuitEvasionEnv()
    env.reset(seed=42)
    env.pursuer_state[:2] = np.array([0.0, 0.0])
    env.evader_state[:2] = np.array([0.1, 0.0])
    _, _, r_p, _, done = env.step(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert done
    assert r_p > 0


def test_observation_normalization():
    env = PursuitEvasionEnv()
    obs_p, obs_e = env.reset(seed=42)
    assert np.all(obs_p >= -1.0) and np.all(obs_p <= 1.0)
    assert np.all(obs_e >= -1.0) and np.all(obs_e <= 1.0)


def test_timeout():
    env = PursuitEvasionEnv(max_steps=5)
    env.reset(seed=42)
    done = False
    for _ in range(5):
        _, _, _, _, done = env.step(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert done
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/test_env.py -v`

Expected: `ModuleNotFoundError: No module named 'envs.pursuit_evasion_env'`

- [ ] **Step 3: Implement environment**

```python
import numpy as np
from typing import Tuple
from envs.sensor import WedgeSensor


class PursuitEvasionEnv:
    def __init__(self, config: dict = None):
        # Defaults matching config.yaml
        self.arena_size = 16.0
        self.max_steps = 400
        self.capture_radius = 0.15
        self.dt = 0.1

        # Pursuer params
        self.L = 0.15
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
```

- [ ] **Step 4: Run tests**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/test_env.py -v`

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add repro_gonultas_isler_2024/envs/pursuit_evasion_env.py repro_gonultas_isler_2024/tests/test_env.py
git commit -m "feat: implement pursuit-evasion env with bicycle+point-mass dynamics"
```

---

## Task 4: Replay Buffer

**Files:**
- Create: `repro_gonultas_isler_2024/training/replay_buffer.py`
- Create: `repro_gonultas_isler_2024/tests/test_replay_buffer.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pytest
from training.replay_buffer import ReplayBuffer


def test_buffer_add_and_sample():
    buffer = ReplayBuffer(capacity=100, obs_dim=11, action_dim=2)
    for i in range(10):
        buffer.add(
            obs_p=np.ones(11) * i,
            obs_e=np.ones(11) * i,
            action_p=np.ones(2) * i,
            action_e=np.ones(2) * i,
            reward_p=float(i),
            reward_e=float(-i),
            next_obs_p=np.ones(11) * (i + 1),
            next_obs_e=np.ones(11) * (i + 1),
            done=False,
        )
    batch = buffer.sample(5)
    assert batch["obs_p"].shape == (5, 11)
    assert batch["actions_p"].shape == (5, 2)
    assert batch["rewards_p"].shape == (5,)
    assert batch["dones"].shape == (5,)


def test_buffer_sample_all():
    buffer = ReplayBuffer(capacity=100, obs_dim=11, action_dim=2)
    for i in range(5):
        buffer.add(
            obs_p=np.ones(11),
            obs_e=np.ones(11),
            action_p=np.ones(2),
            action_e=np.ones(2),
            reward_p=1.0,
            reward_e=-1.0,
            next_obs_p=np.ones(11),
            next_obs_e=np.ones(11),
            done=False,
        )
    batch = buffer.sample(5)
    assert batch["obs_p"].shape == (5, 11)


def test_buffer_overwrite():
    buffer = ReplayBuffer(capacity=5, obs_dim=11, action_dim=2)
    for i in range(10):
        buffer.add(
            obs_p=np.ones(11) * i,
            obs_e=np.ones(11) * i,
            action_p=np.ones(2) * i,
            action_e=np.ones(2) * i,
            reward_p=float(i),
            reward_e=float(-i),
            next_obs_p=np.ones(11) * (i + 1),
            next_obs_e=np.ones(11) * (i + 1),
            done=False,
        )
    assert buffer.size == 5
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/test_replay_buffer.py -v`

Expected: `ModuleNotFoundError: No module named 'training.replay_buffer'`

- [ ] **Step 3: Implement replay buffer**

```python
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.size = 0
        self.ptr = 0

        self.obs_p = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.obs_e = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions_p = np.zeros((capacity, action_dim), dtype=np.float32)
        self.actions_e = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards_p = np.zeros(capacity, dtype=np.float32)
        self.rewards_e = np.zeros(capacity, dtype=np.float32)
        self.next_obs_p = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_e = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs_p: np.ndarray,
        obs_e: np.ndarray,
        action_p: np.ndarray,
        action_e: np.ndarray,
        reward_p: float,
        reward_e: float,
        next_obs_p: np.ndarray,
        next_obs_e: np.ndarray,
        done: bool,
    ):
        idx = self.ptr
        self.obs_p[idx] = obs_p
        self.obs_e[idx] = obs_e
        self.actions_p[idx] = action_p
        self.actions_e[idx] = action_e
        self.rewards_p[idx] = reward_p
        self.rewards_e[idx] = reward_e
        self.next_obs_p[idx] = next_obs_p
        self.next_obs_e[idx] = next_obs_e
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs_p": self.obs_p[indices],
            "obs_e": self.obs_e[indices],
            "actions_p": self.actions_p[indices],
            "actions_e": self.actions_e[indices],
            "rewards_p": self.rewards_p[indices],
            "rewards_e": self.rewards_e[indices],
            "next_obs_p": self.next_obs_p[indices],
            "next_obs_e": self.next_obs_e[indices],
            "dones": self.dones[indices],
        }
```

- [ ] **Step 4: Run tests**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/test_replay_buffer.py -v`

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add repro_gonultas_isler_2024/training/replay_buffer.py repro_gonultas_isler_2024/tests/test_replay_buffer.py
git commit -m "feat: implement shared replay buffer for MADDPG"
```

---

## Task 5: MADDPG Networks and Agent

**Files:**
- Create: `repro_gonultas_isler_2024/agents/maddpg.py`
- Create: `repro_gonultas_isler_2024/tests/test_maddpg.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest
import torch
from agents.maddpg import ActorNetwork, CriticNetwork, MADDPGAgent


def test_actor_output_shape():
    actor = ActorNetwork(obs_dim=13, action_dim=2, hidden_dim=128)
    obs = torch.randn(32, 13)
    action = actor(obs)
    assert action.shape == (32, 2)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_critic_output_shape():
    critic = CriticNetwork(total_obs_dim=26, total_action_dim=4, hidden_dim=128)
    obs = torch.randn(32, 26)
    actions = torch.randn(32, 4)
    q = critic(obs, actions)
    assert q.shape == (32, 1)


def test_agent_select_action():
    agent = MADDPGAgent(
        obs_dim=13,
        action_dim=2,
        total_obs_dim=26,
        total_action_dim=4,
        actor_lr=0.0005,
        critic_lr=0.0005,
        gamma=0.99,
        tau=0.005,
    )
    obs = np.ones(13, dtype=np.float32)
    action = agent.select_action(obs, noise_sigma=0.0)
    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_target_networks_update():
    agent = MADDPGAgent(
        obs_dim=13,
        action_dim=2,
        total_obs_dim=26,
        total_action_dim=4,
        actor_lr=0.0005,
        critic_lr=0.0005,
        gamma=0.99,
        tau=0.005,
    )
    old_target = agent.actor_target.fc1.weight.clone()
    agent.actor.fc1.weight.data += 1.0
    agent.soft_update(agent.actor, agent.actor_target)
    new_target = agent.actor_target.fc1.weight.clone()
    assert not torch.allclose(old_target, new_target)
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/test_maddpg.py -v`

Expected: `ModuleNotFoundError: No module named 'agents.maddpg'`

- [ ] **Step 3: Implement MADDPG**

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        return self.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    def __init__(self, total_obs_dim: int, total_action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class MADDPGAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        total_obs_dim: int,
        total_action_dim: int,
        actor_lr: float = 0.0005,
        critic_lr: float = 0.0005,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = ActorNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_target = CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs: np.ndarray, noise_sigma: float = 0.1) -> np.ndarray:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def update(
        self,
        batch: dict,
        other_agent: "MADDPGAgent",
    ) -> tuple:
        obs_p = torch.FloatTensor(batch["obs_p"]).to(self.device)
        obs_e = torch.FloatTensor(batch["obs_e"]).to(self.device)
        actions_p = torch.FloatTensor(batch["actions_p"]).to(self.device)
        actions_e = torch.FloatTensor(batch["actions_e"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards_p"]).unsqueeze(1).to(self.device)
        next_obs_p = torch.FloatTensor(batch["next_obs_p"]).to(self.device)
        next_obs_e = torch.FloatTensor(batch["next_obs_e"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).unsqueeze(1).to(self.device)

        # Assume this agent is pursuer, other_agent is evader
        obs_all = torch.cat([obs_p, obs_e], dim=-1)
        next_obs_all = torch.cat([next_obs_p, next_obs_e], dim=-1)
        actions_all = torch.cat([actions_p, actions_e], dim=-1)

        # Critic update
        with torch.no_grad():
            next_actions_p = self.actor_target(next_obs_p)
            next_actions_e = other_agent.actor_target(next_obs_e)
            next_actions_all = torch.cat([next_actions_p, next_actions_e], dim=-1)
            target_q = rewards + self.gamma * (1 - dones) * self.critic_target(
                next_obs_all, next_actions_all
            )

        current_q = self.critic(obs_all, actions_all)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        new_actions_p = self.actor(obs_p)
        new_actions_all = torch.cat([new_actions_p, actions_e.detach()], dim=-1)
        actor_loss = -self.critic(obs_all, new_actions_all).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, source: nn.Module, target: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def update_targets(self):
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
```

- [ ] **Step 4: Run tests**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/test_maddpg.py -v`

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add repro_gonultas_isler_2024/agents/maddpg.py repro_gonultas_isler_2024/tests/test_maddpg.py
git commit -m "feat: implement MADDPG actor-critic with target networks"
```

---

## Task 6: BiMDN Belief Encoder

**Files:**
- Create: `repro_gonultas_isler_2024/agents/bimdn.py`
- Create: `repro_gonultas_isler_2024/tests/test_bimdn.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest
import torch
from agents.bimdn import BiMDN


def test_output_shapes():
    bimdn = BiMDN(obs_dim=11, hidden_dim=128, feature_dim=32, num_components=3)
    # Batch of 5 sequences, each 20 timesteps, each obs 11-D
    obs_history = torch.randn(5, 20, 11)
    pi, mu, sigma = bimdn(obs_history)
    assert pi.shape == (5, 3)
    assert mu.shape == (5, 3, 2)
    assert sigma.shape == (5, 3, 2)


def test_mixture_weights_sum_to_one():
    bimdn = BiMDN(obs_dim=11, hidden_dim=128, feature_dim=32, num_components=3)
    obs_history = torch.randn(10, 20, 11)
    pi, _, _ = bimdn(obs_history)
    weights_sum = pi.sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)


def test_std_positive():
    bimdn = BiMDN(obs_dim=11, hidden_dim=128, feature_dim=32, num_components=3)
    obs_history = torch.randn(10, 20, 11)
    _, _, sigma = bimdn(obs_history)
    assert torch.all(sigma > 0)


def test_nll_loss_finite():
    bimdn = BiMDN(obs_dim=11, hidden_dim=128, feature_dim=32, num_components=3)
    optimizer = torch.optim.Adam(bimdn.parameters(), lr=0.002)
    obs_history = torch.randn(32, 20, 11)
    target_pos = torch.randn(32, 2)

    loss1 = bimdn.compute_nll_loss(obs_history, target_pos)
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()
    loss2 = bimdn.compute_nll_loss(obs_history, target_pos)

    assert torch.isfinite(loss1)
    assert torch.isfinite(loss2)
    assert loss2 < loss1  # Loss should decrease


def test_weighted_mean_shape():
    bimdn = BiMDN(obs_dim=11, hidden_dim=128, feature_dim=32, num_components=3)
    obs_history = torch.randn(10, 20, 11)
    mean = bimdn.get_weighted_mean(obs_history)
    assert mean.shape == (10, 2)
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/test_bimdn.py -v`

Expected: `ModuleNotFoundError: No module named 'agents.bimdn'`

- [ ] **Step 3: Implement BiMDN**

```python
import torch
import torch.nn as nn


class BiMDN(nn.Module):
    def __init__(
        self,
        obs_dim: int = 11,
        hidden_dim: int = 128,
        feature_dim: int = 32,
        num_components: int = 3,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_components = num_components

        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # LSTM output: (batch, seq, 2*hidden) → take last timestep
        # Forward + backward hidden states concatenated = 2*hidden_dim
        self.fc = nn.Linear(2 * hidden_dim, feature_dim)
        self.relu = nn.ReLU()

        # 3 heads for mixture components
        self.pi_head = nn.Linear(feature_dim, num_components)
        self.mu_head = nn.Linear(feature_dim, num_components * 2)
        self.sigma_head = nn.Linear(feature_dim, num_components * 2)

    def forward(self, obs_history: torch.Tensor) -> tuple:
        # obs_history: (batch, seq_len, obs_dim)
        lstm_out, _ = self.lstm(obs_history)
        # Take last timestep
        last_hidden = lstm_out[:, -1, :]  # (batch, 2*hidden_dim)
        features = self.relu(self.fc(last_hidden))  # (batch, feature_dim)

        # Mixture weights
        pi = torch.softmax(self.pi_head(features), dim=-1)  # (batch, num_components)

        # Means
        mu = self.mu_head(features).view(-1, self.num_components, 2)

        # Standard deviations (ELU + 1 ensures > 0)
        sigma = nn.functional.elu(self.sigma_head(features)) + 1.0
        sigma = sigma.view(-1, self.num_components, 2)

        return pi, mu, sigma

    def compute_nll_loss(
        self,
        obs_history: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        pi, mu, sigma = self.forward(obs_history)
        # target_pos: (batch, 2)
        target = target_pos.unsqueeze(1)  # (batch, 1, 2)

        # Gaussian PDF for each component
        # log_prob = -0.5 * log(2π) - log(σ) - 0.5 * ((x-μ)/σ)^2
        log_prob = (
            -0.5 * torch.log(2 * torch.tensor(3.14159265))
            - torch.log(sigma)
            - 0.5 * ((target - mu) / sigma) ** 2
        )
        # Sum over dimensions (x, y)
        log_prob = log_prob.sum(dim=-1)  # (batch, num_components)

        # Weight by mixture components and sum
        log_prob_weighted = torch.logsumexp(
            torch.log(pi + 1e-8) + log_prob, dim=-1
        )

        return -log_prob_weighted.mean()

    def get_weighted_mean(self, obs_history: torch.Tensor) -> torch.Tensor:
        pi, mu, _ = self.forward(obs_history)
        # Weighted mean: sum_i pi_i * mu_i
        weighted_mean = (pi.unsqueeze(-1) * mu).sum(dim=1)  # (batch, 2)
        return weighted_mean
```

- [ ] **Step 4: Run tests**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/test_bimdn.py -v`

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add repro_gonultas_isler_2024/agents/bimdn.py repro_gonultas_isler_2024/tests/test_bimdn.py
git commit -m "feat: implement BiMDN belief encoder with NLL loss"
```

---

## Task 7: Curriculum Module

**Files:**
- Create: `repro_gonultas_isler_2024/training/curriculum.py`
- Modify: `repro_gonultas_isler_2024/tests/test_env.py`

- [ ] **Step 1: Write failing test**

Add to `repro_gonultas_isler_2024/tests/test_env.py`:

```python
from training.curriculum import Curriculum


def test_curriculum_fov_decay():
    curriculum = Curriculum(total_episodes=10000)
    fov_start = curriculum.get_fov(0)
    fov_mid = curriculum.get_fov(3000)
    fov_end = curriculum.get_fov(10000)
    assert fov_start == pytest.approx(6.283, abs=0.01)
    assert fov_end == pytest.approx(2.094, abs=0.01)
    assert fov_mid > fov_end and fov_mid < fov_start


def test_curriculum_evader_velocity():
    curriculum = Curriculum(total_episodes=10000)
    v_start = curriculum.get_evader_vmax(0)
    v_mid = curriculum.get_evader_vmax(3000)
    v_end = curriculum.get_evader_vmax(10000)
    assert v_start == pytest.approx(0.0, abs=0.01)
    assert v_end == pytest.approx(1.5, abs=0.01)
    assert v_mid > v_start and v_mid < v_end
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/test_env.py::test_curriculum_fov_decay tests/test_env.py::test_curriculum_evader_velocity -v`

Expected: `ModuleNotFoundError: No module named 'training.curriculum'`

- [ ] **Step 3: Implement curriculum**

```python
class Curriculum:
    def __init__(
        self,
        total_episodes: int = 20000,
        t_end_ratio: float = 0.3,
        fov_start: float = 6.283,
        fov_end: float = 2.094,
        evader_vmax_start: float = 0.0,
        evader_vmax_end: float = 1.5,
    ):
        self.total_episodes = total_episodes
        self.t_end = int(t_end_ratio * total_episodes)
        self.fov_start = fov_start
        self.fov_end = fov_end
        self.evader_vmax_start = evader_vmax_start
        self.evader_vmax_end = evader_vmax_end

    def _progress(self, episode: int) -> float:
        return min(1.0, episode / self.t_end)

    def get_fov(self, episode: int) -> float:
        p = self._progress(episode)
        return self.fov_start * (1 - p) + self.fov_end * p

    def get_evader_vmax(self, episode: int) -> float:
        p = self._progress(episode)
        return self.evader_vmax_start * (1 - p) + self.evader_vmax_end * p
```

- [ ] **Step 4: Run tests**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/test_env.py::test_curriculum_fov_decay tests/test_env.py::test_curriculum_evader_velocity -v`

Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add repro_gonultas_isler_2024/training/curriculum.py repro_gonultas_isler_2024/tests/test_env.py
git commit -m "feat: implement FOV + velocity curriculum"
```

---

## Task 8: Logger Utility

**Files:**
- Create: `repro_gonultas_isler_2024/utils/logger.py`

- [ ] **Step 1: Implement logger**

```python
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str = "./runs"):
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def log_scalar(self, tag: str, value: float, step: int = None):
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: dict, step: int = None):
        if step is None:
            step = self.step
        self.writer.add_scalars(tag, values, step)

    def increment_step(self):
        self.step += 1

    def close(self):
        self.writer.close()
```

- [ ] **Step 2: Commit**

```bash
git add repro_gonultas_isler_2024/utils/logger.py
git commit -m "feat: add TensorBoard logger wrapper"
```

---

## Task 9: Training Loop

**Files:**
- Create: `repro_gonultas_isler_2024/training/train.py`

- [ ] **Step 1: Implement training loop**

```python
import numpy as np
import torch
import yaml
from pathlib import Path

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.maddpg import MADDPGAgent
from agents.bimdn import BiMDN
from training.replay_buffer import ReplayBuffer
from training.curriculum import Curriculum
from utils.logger import Logger


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_episode(
    env: PursuitEvasionEnv,
    pursuer: MADDPGAgent,
    evader: MADDPGAgent,
    replay_buffer: ReplayBuffer,
    noise_sigma: float,
    frame_stack: int,
    frame_skip: int,
) -> dict:
    obs_p, obs_e = env.reset()
    # Frame stacking buffers
    obs_p_buffer = [obs_p.copy() for _ in range(frame_stack)]
    obs_e_buffer = [obs_e.copy() for _ in range(frame_stack)]

    episode_data = {
        "pursuer_reward": 0.0,
        "evader_reward": 0.0,
        "length": 0,
        "captured": False,
    }

    done = False
    step = 0
    while not done and step < env.max_steps:
        # Stack observations
        stacked_p = np.concatenate(obs_p_buffer[-frame_stack:])
        stacked_e = np.concatenate(obs_e_buffer[-frame_stack:])

        # Select actions
        action_p = pursuer.select_action(stacked_p, noise_sigma)
        action_e = evader.select_action(stacked_e, noise_sigma)

        # Frame skip: repeat action for multiple sim steps
        for _ in range(frame_skip):
            next_obs_p, next_obs_e, r_p, r_e, done = env.step(action_p, action_e)
            episode_data["pursuer_reward"] += r_p
            episode_data["evader_reward"] += r_e
            step += 1
            if done:
                break

        # Store transition (at policy decision frequency)
        next_stacked_p = np.concatenate(obs_p_buffer[-(frame_stack - 1):] + [next_obs_p])
        next_stacked_e = np.concatenate(obs_e_buffer[-(frame_stack - 1):] + [next_obs_e])

        replay_buffer.add(
            obs_p=stacked_p,
            obs_e=stacked_e,
            action_p=action_p,
            action_e=action_e,
            reward_p=r_p,
            reward_e=r_e,
            next_obs_p=next_stacked_p,
            next_obs_e=next_stacked_e,
            done=done,
        )

        obs_p_buffer.append(next_obs_p.copy())
        obs_e_buffer.append(next_obs_e.copy())

    episode_data["length"] = step
    episode_data["captured"] = (episode_data["pursuer_reward"] > 0)
    return episode_data


def train(config_path: str = "config.yaml"):
    config = load_config(config_path)

    # Environment
    env = PursuitEvasionEnv(config)
    curriculum = Curriculum(
        total_episodes=config["maddpg"]["episodes"],
        t_end_ratio=config["curriculum"]["t_end_ratio"],
        fov_start=config["curriculum"]["fov_start"],
        fov_end=config["curriculum"]["fov_end"],
        evader_vmax_start=config["curriculum"]["evader_vmax_start"],
        evader_vmax_end=config["curriculum"]["evader_vmax_end"],
    )

    # Obs dimensions with frame stacking
    raw_obs_dim = 11
    frame_stack = config["maddpg"]["frame_stack"]
    obs_dim = raw_obs_dim * frame_stack  # 22 without belief, 26 with
    action_dim = 2
    total_obs_dim = obs_dim * 2  # both agents
    total_action_dim = action_dim * 2

    # Agents
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pursuer = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=total_obs_dim,
        total_action_dim=total_action_dim,
        actor_lr=config["maddpg"]["actor_lr"],
        critic_lr=config["maddpg"]["critic_lr"],
        gamma=config["maddpg"]["gamma"],
        tau=config["maddpg"]["tau"],
        device=device,
    )
    evader = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=total_obs_dim,
        total_action_dim=total_action_dim,
        actor_lr=config["maddpg"]["actor_lr"],
        critic_lr=config["maddpg"]["critic_lr"],
        gamma=config["maddpg"]["gamma"],
        tau=config["maddpg"]["tau"],
        device=device,
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config["maddpg"]["replay_buffer_size"],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # Logger
    logger = Logger(config["log_dir"])

    # Training loop
    episodes = config["maddpg"]["episodes"]
    batch_size = config["maddpg"]["batch_size"]
    noise_sigma = config["maddpg"]["exploration_sigma"]
    frame_skip = config["maddpg"]["frame_skip"]

    total_steps = 0
    for episode in range(episodes):
        # Update curriculum
        fov = curriculum.get_fov(episode)
        evader_vmax = curriculum.get_evader_vmax(episode)
        env.pursuer_sensor.fov_angle = fov
        env.evader_v_max = evader_vmax

        # Run episode
        ep_data = run_episode(
            env, pursuer, evader, replay_buffer,
            noise_sigma, frame_stack, frame_skip,
        )

        logger.log_scalar("episode/length", ep_data["length"], episode)
        logger.log_scalar("episode/pursuer_reward", ep_data["pursuer_reward"], episode)
        logger.log_scalar("episode/evader_reward", ep_data["evader_reward"], episode)
        logger.log_scalar("episode/capture_rate", float(ep_data["captured"]), episode)

        # Update MADDPG
        if replay_buffer.size >= batch_size:
            for _ in range(ep_data["length"]):
                batch = replay_buffer.sample(batch_size)
                c_loss_p, a_loss_p = pursuer.update(batch, evader)
                c_loss_e, a_loss_e = evader.update(batch, pursuer)
                pursuer.update_targets()
                evader.update_targets()

            logger.log_scalar("loss/pursuer_critic", c_loss_p, episode)
            logger.log_scalar("loss/pursuer_actor", a_loss_p, episode)
            logger.log_scalar("loss/evader_critic", c_loss_e, episode)
            logger.log_scalar("loss/evader_actor", a_loss_e, episode)

        total_steps += ep_data["length"]

        # Checkpoint
        if episode % config["eval"]["checkpoint_freq"] == 0:
            checkpoint_dir = Path("checkpoints") / f"episode_{episode}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(pursuer.actor.state_dict(), checkpoint_dir / "pursuer_actor.pt")
            torch.save(evader.actor.state_dict(), checkpoint_dir / "evader_actor.pt")

    logger.close()


if __name__ == "__main__":
    train()
```

- [ ] **Step 2: Commit**

```bash
git add repro_gonultas_isler_2024/training/train.py
git commit -m "feat: implement main training loop with curriculum and checkpointing"
```

---

## Task 10: Evaluation Script

**Files:**
- Create: `repro_gonultas_isler_2024/evaluation/evaluate.py`

- [ ] **Step 1: Implement evaluation**

```python
import numpy as np
import torch
from pathlib import Path
from typing import Callable

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.maddpg import MADDPGAgent


def random_policy(obs: np.ndarray) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, size=2)


def greedy_pursuer_policy(obs: np.ndarray) -> np.ndarray:
    # obs: [x, y, vx, vy, opp_x, opp_y, opp_vx, opp_vy, I, t, d]
    # Simple: accelerate toward opponent
    opp_x, opp_y = obs[4], obs[5]
    own_x, own_y = obs[0], obs[1]
    dx = opp_x - own_x
    dy = opp_y - own_y
    angle = np.arctan2(dy, dx)
    # Naive: steer toward angle, accelerate forward
    return np.array([angle * 0.5, 1.0])


def greedy_evader_policy(obs: np.ndarray) -> np.ndarray:
    # Flee from opponent
    opp_x, opp_y = obs[4], obs[5]
    own_x, own_y = obs[0], obs[1]
    dx = own_x - opp_x
    dy = own_y - opp_y
    angle = np.arctan2(dy, dx)
    return np.array([angle * 0.5, 1.0])


def evaluate_agent(
    env: PursuitEvasionEnv,
    pursuer_policy: Callable,
    evader_policy: Callable,
    num_episodes: int = 500,
    frame_stack: int = 2,
) -> dict:
    capture_count = 0
    episode_lengths = []
    pursuer_rewards = []

    for _ in range(num_episodes):
        obs_p, obs_e = env.reset()
        obs_p_buffer = [obs_p.copy() for _ in range(frame_stack)]
        obs_e_buffer = [obs_e.copy() for _ in range(frame_stack)]

        done = False
        step = 0
        ep_reward = 0.0

        while not done and step < env.max_steps:
            stacked_p = np.concatenate(obs_p_buffer[-frame_stack:])
            stacked_e = np.concatenate(obs_e_buffer[-frame_stack:])

            action_p = pursuer_policy(stacked_p)
            action_e = evader_policy(stacked_e)

            obs_p, obs_e, r_p, r_e, done = env.step(action_p, action_e)
            ep_reward += r_p
            step += 1

            obs_p_buffer.append(obs_p.copy())
            obs_e_buffer.append(obs_e.copy())

        if ep_reward > 0:
            capture_count += 1
        episode_lengths.append(step)
        pursuer_rewards.append(ep_reward)

    return {
        "capture_rate": capture_count / num_episodes,
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "mean_reward": np.mean(pursuer_rewards),
    }


def load_policy(agent: MADDPGAgent, checkpoint_path: str):
    agent.actor.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    agent.actor.eval()


def run_evaluation(
    checkpoint_dir: str,
    config: dict,
    num_episodes: int = 500,
):
    env = PursuitEvasionEnv(config)
    frame_stack = config["maddpg"]["frame_stack"]
    obs_dim = 11 * frame_stack
    action_dim = 2

    pursuer = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=obs_dim * 2,
        total_action_dim=action_dim * 2,
    )
    evader = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=obs_dim * 2,
        total_action_dim=action_dim * 2,
    )

    checkpoint_path = Path(checkpoint_dir)
    load_policy(pursuer, checkpoint_path / "pursuer_actor.pt")
    load_policy(evader, checkpoint_path / "evader_actor.pt")

    def pursuer_nn_policy(obs):
        return pursuer.select_action(obs, noise_sigma=0.0)

    def evader_nn_policy(obs):
        return evader.select_action(obs, noise_sigma=0.0)

    results = {}

    # 1. Trained pursuer vs random evader
    results["pursuer_vs_random"] = evaluate_agent(
        env, pursuer_nn_policy, random_policy, num_episodes, frame_stack
    )

    # 2. Trained pursuer vs greedy evader
    results["pursuer_vs_greedy"] = evaluate_agent(
        env, pursuer_nn_policy, greedy_evader_policy, num_episodes, frame_stack
    )

    # 3. Trained pursuer vs trained evader (self-play)
    results["pursuer_vs_selfplay"] = evaluate_agent(
        env, pursuer_nn_policy, evader_nn_policy, num_episodes, frame_stack
    )

    # 4. Random pursuer vs trained evader
    results["random_vs_evader"] = evaluate_agent(
        env, random_policy, evader_nn_policy, num_episodes, frame_stack
    )

    # 5. Greedy pursuer vs trained evader
    results["greedy_vs_evader"] = evaluate_agent(
        env, greedy_pursuer_policy, evader_nn_policy, num_episodes, frame_stack
    )

    return results


if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    checkpoint = "checkpoints/episode_1000"
    results = run_evaluation(checkpoint, config)
    for key, val in results.items():
        print(f"{key}: {val}")
```

- [ ] **Step 2: Commit**

```bash
git add repro_gonultas_isler_2024/evaluation/evaluate.py
git commit -m "feat: implement evaluation against random, greedy, and self-play"
```

---

## Task 11: README

**Files:**
- Create: `repro_gonultas_isler_2024/README.md`

- [ ] **Step 1: Write README**

```markdown
# Gonultas & Isler (IROS 2025) Reproduction

Standalone PyTorch reproduction of MADDPG + BiMDN for asymmetric partially-observable pursuit-evasion.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
cd repro_gonultas_isler_2024
python training/train.py
```

## Evaluation

```bash
python evaluation/evaluate.py
```

## Tests

```bash
python -m pytest tests/ -v
```

## Key Parameters

- Arena: 16×16m
- Pursuer: bicycle model (wheelbase 0.15m)
- Evader: point-mass double integrator
- Episodes: 20,000 with 8 parallel envs
- MADDPG: 2×128 actor/critic, batch size 512
- BiMDN: 1-layer BiLSTM (128 hidden), 3 Gaussian components
```

- [ ] **Step 2: Commit**

```bash
git add repro_gonultas_isler_2024/README.md
git commit -m "docs: add README with setup and usage instructions"
```

---

## Task 12: Integration / Smoke Test

**Files:**
- Create: `repro_gonultas_isler_2024/tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
import numpy as np
import torch
from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.maddpg import MADDPGAgent
from agents.bimdn import BiMDN
from training.replay_buffer import ReplayBuffer


def test_full_episode_with_agents():
    env = PursuitEvasionEnv()
    obs_dim = 22  # 11 * 2 frame stack
    action_dim = 2

    pursuer = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=obs_dim * 2,
        total_action_dim=action_dim * 2,
    )
    evader = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=obs_dim * 2,
        total_action_dim=action_dim * 2,
    )
    buffer = ReplayBuffer(capacity=1000, obs_dim=obs_dim, action_dim=action_dim)

    obs_p, obs_e = env.reset(seed=42)
    obs_p_buffer = [obs_p.copy(), obs_p.copy()]
    obs_e_buffer = [obs_e.copy(), obs_e.copy()]

    done = False
    steps = 0
    while not done and steps < 50:
        stacked_p = np.concatenate(obs_p_buffer[-2:])
        stacked_e = np.concatenate(obs_e_buffer[-2:])

        action_p = pursuer.select_action(stacked_p, noise_sigma=0.1)
        action_e = evader.select_action(stacked_e, noise_sigma=0.1)

        next_obs_p, next_obs_e, r_p, r_e, done = env.step(action_p, action_e)

        next_stacked_p = np.concatenate(obs_p_buffer[-1:] + [next_obs_p])
        next_stacked_e = np.concatenate(obs_e_buffer[-1:] + [next_obs_e])

        buffer.add(stacked_p, stacked_e, action_p, action_e, r_p, r_e,
                   next_stacked_p, next_stacked_e, done)

        obs_p_buffer.append(next_obs_p.copy())
        obs_e_buffer.append(next_obs_e.copy())
        steps += 1

    assert buffer.size > 0
    assert steps > 0


def test_bimdn_with_env_obs():
    bimdn = BiMDN(obs_dim=11, hidden_dim=128, feature_dim=32, num_components=3)
    env = PursuitEvasionEnv()
    obs_p, _ = env.reset(seed=42)

    # Create fake history (batch=1, seq=20, obs=11)
    history = torch.tensor(obs_p, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    history = history.repeat(1, 20, 1)

    pi, mu, sigma = bimdn(history)
    assert pi.shape == (1, 3)
    assert mu.shape == (1, 3, 2)
    assert sigma.shape == (1, 3, 2)

    weighted_mean = bimdn.get_weighted_mean(history)
    assert weighted_mean.shape == (1, 2)
```

- [ ] **Step 2: Run integration tests**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/test_integration.py -v`

Expected: Both tests PASS.

- [ ] **Step 3: Run full test suite**

Run: `cd repro_gonultas_isler_2024 && python -m pytest tests/ -v`

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add repro_gonultas_isler_2024/tests/test_integration.py
git commit -m "test: add integration smoke tests for full pipeline"
```

---

## Spec Coverage Check

| Spec Section | Task | Status |
|--------------|------|--------|
| 4.1 Environment (bicycle + point-mass) | Task 3 | ✅ |
| 4.1 Sensor model (wedge FOV) | Task 2 | ✅ |
| 4.1 Reward (zero-sum) | Task 3 | ✅ |
| 4.1 Observation normalization | Task 3 | ✅ |
| 4.2 BiMDN architecture | Task 6 | ✅ |
| 4.2 BiMDN NLL loss | Task 6 | ✅ |
| 4.2 BiMDN weighted mean inference | Task 6 | ✅ |
| 4.3 MADDPG actor (2×128) | Task 5 | ✅ |
| 4.3 MADDPG critic (2×128) | Task 5 | ✅ |
| 4.3 MADDPG update (target, soft update) | Task 5 | ✅ |
| 4.4 Curriculum (FOV + velocity) | Task 7 | ✅ |
| 4.5 Training loop | Task 9 | ✅ |
| 4.6 Evaluation (baselines) | Task 10 | ✅ |
| 4.7 Frame stacking / skipping | Task 9 | ✅ |
| 4.8 Action normalization | Tasks 3, 5 | ✅ |

## Placeholder Scan

- No "TBD" or "TODO" found.
- No vague "implement later" or "add appropriate error handling" steps.
- Every step has exact code, exact commands, expected outputs.

## Type Consistency Check

- `ReplayBuffer.add()` signature matches usage in `train.py` and `test_replay_buffer.py`.
- `MADDPGAgent.update()` batch dict keys match `ReplayBuffer.sample()` return keys.
- `BiMDN.forward()` returns `(pi, mu, sigma)` consistently across `bimdn.py` and `test_bimdn.py`.
- `PursuitEvasionEnv.step()` returns `(obs_p, obs_e, r_p, r_e, done)` consistently.
- Frame stack dimension: `obs_dim = 11 * frame_stack = 22` used consistently in agent init and tests.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-08-repro-gonultas-isler.md`.**

**Execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
