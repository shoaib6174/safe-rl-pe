# Model Setup Report — Evader vs Greedy Pursuer (S1)

**Date**: 2026-02-28
**Model**: `results/diag_survival_ent03/evader_final`
**Result**: 99% escape rate, 9/9 episodes survive full 600 steps

---

## 1. Game Setup

### Arena
| Parameter | Value |
|-----------|-------|
| Arena size | 10.0 x 10.0 meters |
| Coordinate system | Center-origin: [-5.0, 5.0] x [-5.0, 5.0] |
| Obstacles | 2 circular, randomly placed each episode |
| Obstacle radius range | [0.3, 1.0] meters |
| Obstacle margin | 0.5 m (from walls and each other) |

### Agents
| Parameter | Pursuer | Evader |
|-----------|---------|--------|
| Max linear velocity (v_max) | 1.0 m/s | 1.0 m/s |
| Max angular velocity (omega_max) | 2.84 rad/s | 2.84 rad/s |
| Robot radius | 0.15 m | 0.15 m |
| Control | Greedy proportional navigation (fixed) | PPO-learned policy |

Both agents have **equal speed** (1.0x). This is a deliberate design choice — at higher speeds the evader learns circular running (a fragile local optimum) instead of obstacle exploitation.

### Episode Configuration
| Parameter | Value |
|-----------|-------|
| Max steps | 600 |
| Timestep (dt) | 0.05 s |
| Real-time duration | 30 seconds |
| Capture radius | 0.5 m |
| Initial agent distance | [3.0, 15.0] m (random) |
| Termination | `terminated` = capture, `truncated` = timeout (escape) |

---

## 2. Agent Dynamics

### Unicycle Model
Both agents use the same unicycle dynamics with Euler integration:

```
x_new = x + v * cos(theta) * dt
y_new = y + v * sin(theta) * dt
theta_new = wrap_angle(theta + omega * dt)
```

- **Wall collisions**: Position clamped to arena bounds; `wall_contact` flag set
- **Obstacle collisions**: If robot body overlaps obstacle, pushed radially outward to surface (up to 3 resolution iterations for chain collisions)
- **Action clipping**: Both v and omega clipped to physical bounds before integration

### FixedSpeedWrapper
The evader's action space is reduced from 2D `[v, omega]` to 1D `[omega]`. The wrapper sets `v = v_max` (1.0 m/s) always. The evader only controls **steering** (angular velocity).

---

## 3. Observation Space

### Structure: 18-dimensional vector (14 base + 2*2 obstacle features)

Both agents receive the **same observation structure** — this is a **fully observable** environment.

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | Own x position | / half_w → [-1, 1] |
| 1 | Own y position | / half_h → [-1, 1] |
| 2 | Own heading (theta) | / pi → [-1, 1] |
| 3 | Own linear velocity | / v_max → [0, 1] |
| 4 | Own angular velocity | / omega_max → [-1, 1] |
| 5 | Opponent x position | / half_w → [-1, 1] |
| 6 | Opponent y position | / half_h → [-1, 1] |
| 7 | Opponent heading | / pi → [-1, 1] |
| 8 | Opponent linear velocity | / v_max → [0, 1] |
| 9 | Opponent angular velocity | / omega_max → [-1, 1] |
| 10 | Distance to opponent | / arena_diagonal → [0, 1] |
| 11 | Bearing to opponent (relative to heading) | / pi → [-1, 1] |
| 12 | Distance to nearest wall (x-axis) | / half_w → [0, 1] |
| 13 | Distance to nearest wall (y-axis) | / half_h → [0, 1] |
| 14 | Distance to nearest obstacle (surface) | / arena_diagonal → [0, 1] |
| 15 | Bearing to nearest obstacle (relative) | / pi → [-1, 1] |
| 16 | Distance to 2nd nearest obstacle | / arena_diagonal → [0, 1] |
| 17 | Bearing to 2nd nearest obstacle | / pi → [-1, 1] |

Key points:
- **Both agents see everything**: each other's exact position, heading, velocity
- **Obstacle features**: distance to obstacle **surface** (not center), so agent knows exactly how far the edge is
- **K=2 obstacles observed** (`n_obstacle_obs=2`), sorted by distance
- **All values normalized to [-1, 1]** for stable learning

---

## 4. Action Space

### Evader (Learned Agent)
| Parameter | Value |
|-----------|-------|
| Dimensions | 1 (omega only, via FixedSpeedWrapper) |
| Range | [-2.84, 2.84] rad/s |
| Linear velocity | Fixed at 1.0 m/s (always moves forward at max speed) |

The evader's only decision is **which direction to steer**.

### Pursuer (Greedy Baseline)
| Parameter | Value |
|-----------|-------|
| Strategy | Proportional navigation |
| Gain (K_p) | 3.0 |
| Steering | `omega = clip(K_p * heading_error, -omega_max, omega_max)` |
| Speed | `v = v_max * max(0, cos(heading_error))` — slows when turning |

The greedy pursuer:
- Always steers toward the evader's current position
- Does **not** avoid obstacles (runs straight into them and gets pushed out)
- Does **not** predict evader trajectory (pure bearing chase)

---

## 5. Reward Design

### Evader Reward (Survival-Time Maximization)

| Component | Value | When Applied |
|-----------|-------|--------------|
| **Survival bonus** | **+0.1 per step** | Every non-terminal step |
| **Visibility reward** | +0.5 if hidden, -0.5 if visible | Every non-terminal step (when obstacles exist) |
| Capture penalty | -5.0 | Terminal: capture |
| Timeout bonus | 0.0 | Terminal: timeout (no bonus) |

**Design rationale**: The primary signal is the survival bonus (+0.1/step). This creates a smooth gradient — surviving 400 steps is worth 40.0, while surviving 100 steps is only worth 10.0. The small capture penalty (-5.0) discourages risky behavior near the pursuer, but the dominant incentive is simply to stay alive as long as possible.

The **visibility reward** teaches obstacle exploitation: the evader receives +0.5 every step an obstacle blocks line-of-sight to the pursuer (ray-circle intersection test), and -0.5 when exposed. This is what drives the obstacle-hugging behavior.

### Pursuer Reward (Zero-Sum Fallback)

| Component | Value |
|-----------|-------|
| Distance-closing reward | `distance_scale * (d_prev - d_curr) / d_max` |
| Capture bonus | +5.0 |
| Timeout penalty | 0.0 |

The pursuer is **not trained** in this setup — it uses a fixed greedy policy. Its reward is computed but discarded.

### Line-of-Sight Check
The `line_of_sight_blocked()` function performs ray-circle intersection:
- Draws a line segment from pursuer position to evader position
- For each obstacle, projects the obstacle center onto this segment
- If the closest point on the segment is within the obstacle radius, line-of-sight is blocked
- Returns `True` if **any** obstacle blocks the view

---

## 6. Training Configuration (PPO)

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Policy | MlpPolicy (2 hidden layers of 64 units) |
| Learning rate | 3e-4 |
| n_steps | 512 (rollout buffer per env) |
| Batch size | 256 |
| n_epochs | 10 |
| Gamma (discount) | 0.99 |
| **Entropy coefficient** | **0.03** |
| Parallel envs | 4 |
| Total training steps | 3,000,000 |
| Evaluation frequency | Every 50,000 steps |
| Evaluation episodes | 100 |
| Seed | 47 |

**Critical finding**: `ent_coef=0.03` is the sweet spot. At `ent_coef=0.05`, the policy collapses entirely (avg survival drops to 65 steps, worse than random at 166).

---

## 7. Environment Wrapper Stack

```
PursuitEvasionEnv (base 2-agent env)
  └── SingleAgentPEWrapper (role="evader", opponent=GreedyPursuerPolicy)
        └── FixedSpeedWrapper (v_max=1.0, action: [omega] → [v_max, omega])
              └── Monitor (SB3 episode logging)
```

1. **PursuitEvasionEnv**: Full 2-agent env, takes `(pursuer_action, evader_action)`, returns dict observations
2. **SingleAgentPEWrapper**: Wraps as single-agent Gymnasium env. On each `step(action)`, queries `GreedyPursuerPolicy.predict()` for opponent action, then calls base env with both actions. Returns only the evader's observation and reward.
3. **FixedSpeedWrapper**: Reduces action space from `[v, omega]` to `[omega]` by fixing `v = v_max`
4. **Monitor**: SB3 episode statistics tracking

---

## 8. Training Results

| Phase | Steps | Escape Rate | Avg Survival |
|-------|-------|-------------|--------------|
| Random baseline | 0 | 0% | 166 steps |
| Early training | 0-1.3M | 0% | ~130 steps |
| **Phase transition** | **1.35M** | **0% → 83%** | **~500 steps** |
| Stabilization | 1.5M-3M | 92-100% | 580-600 steps |
| **Final** | **3M** | **99%** | **597 steps** |

Key observation: The learning exhibits a **phase transition** — a sudden jump from 0% to 83% escape rate at 1.35M steps, after which performance remains stably high.

---

## 9. Learned Behavior: Obstacle-Hugging Orbits

The evader's learned strategy:
1. **Navigate to nearest obstacle** from spawn
2. **Orbit tightly** around the obstacle, keeping it between itself and the pursuer
3. **Switch obstacles** if the pursuer gets too close or finds an opening

Why this works:
- The greedy pursuer uses proportional navigation → steers directly toward the evader
- This drives the pursuer **into the obstacle**, where it gets physically pushed out
- The evader exploits this by staying on the opposite side
- At equal speed, the pursuer can never close the distance while the evader orbits

Failure mode: Captured only when the evader spawns far from any obstacle and the pursuer reaches it before it can establish an orbit.

---

## 10. Architecture Diagram

```
                    ┌──────────────────────────┐
                    │   PursuitEvasionEnv       │
                    │                          │
                    │  Arena: 10x10, 2 obs     │
  GreedyPursuer ──→│  Dynamics: unicycle       │←── PPO Evader
  (K_p=3.0)        │  Rewards: RewardComputer  │    (MlpPolicy)
                    │  Obs: ObservationBuilder  │
                    └──────────┬───────────────┘
                               │
                    ┌──────────┴───────────────┐
                    │  SingleAgentPEWrapper     │
                    │  role="evader"            │
                    │  opponent=GreedyPursuer   │
                    └──────────┬───────────────┘
                               │
                    ┌──────────┴───────────────┐
                    │  FixedSpeedWrapper        │
                    │  v=1.0 always             │
                    │  action: [ω] → [1.0, ω]  │
                    └──────────┬───────────────┘
                               │
                    ┌──────────┴───────────────┐
                    │  PPO (stable-baselines3)  │
                    │  lr=3e-4, ent=0.03        │
                    │  n_steps=512, batch=256   │
                    └──────────────────────────┘
```

---

## 11. File Map

| File | Role |
|------|------|
| `envs/pursuit_evasion_env.py` | Base 2-agent environment (dynamics, termination, rewards) |
| `envs/dynamics.py` | `unicycle_step()`, `resolve_obstacle_collisions()` |
| `envs/observations.py` | `ObservationBuilder` (18D normalized observation vector) |
| `envs/rewards.py` | `RewardComputer`, `line_of_sight_blocked()` |
| `envs/wrappers.py` | `SingleAgentPEWrapper`, `FixedSpeedWrapper` |
| `training/baselines.py` | `GreedyPursuerPolicy` (proportional navigation) |
| `scripts/train_evader_vs_greedy.py` | Training script with eval loop |
| `scripts/visualize_grid_gif.py` | 3x3 animated trajectory visualization |
| `scripts/visualize_trajectory_gif.py` | Single-episode animated GIF |
| `scripts/visualize_evader_vs_greedy.py` | Static 3x3 trajectory plot |

---

## 12. Known Limitations

1. **Pursuer is fixed (not learned)**: The greedy pursuer doesn't avoid obstacles or predict evader movement. A smarter pursuer would counter the obstacle-hugging strategy.
2. **Fully observable**: Both agents see everything. Partial observability (limited FOV) would make obstacle cover even more strategically important.
3. **Fixed speed**: The evader always moves at max speed. Variable speed control could enable more nuanced tactics (slow down near obstacles for tighter turns).
4. **No safety constraints**: No DCBF or other safety layer is applied. The evader freely collides with obstacles and walls (just gets pushed out).
5. **2 obstacles only**: 4 obstacles caused the arena to be too cluttered (0% escape). The optimal obstacle count for a 10x10 arena is 2-3.
6. **Seed sensitivity**: Different random seeds for obstacle placement and spawn positions can significantly affect episode outcomes.
