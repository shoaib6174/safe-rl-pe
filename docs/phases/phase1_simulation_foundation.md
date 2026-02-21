# Phase 1: Simulation Foundation

**Timeline**: Months 1-2
**Status**: Ready for implementation
**Prerequisites**: None (this is the starting phase)
**Next Phase**: [Phase 2 — Safety Integration](./phase2_safety_integration.md)

---

## Table of Contents

1. [Phase Overview](#1-phase-overview)
2. [Background & Theory](#2-background--theory)
3. [Relevant Literature](#3-relevant-literature)
4. [Deliverables Checklist](#4-deliverables-checklist)
5. [Session-wise Implementation Breakdown](#5-session-wise-implementation-breakdown)
6. [Technical Specifications](#6-technical-specifications)
7. [Validation & Success Criteria](#7-validation--success-criteria)
8. [Risk Assessment](#8-risk-assessment)
9. [Software & Tools](#9-software--tools)
10. [Guide to Phase 2](#10-guide-to-phase-2)

---

## 1. Phase Overview

### 1.1 Goal

Build the **complete simulation foundation** for a 1v1 pursuit-evasion (PE) game on ground mobile robots. This includes:

- A custom Gymnasium environment with unicycle dynamics for both agents
- Basic PPO self-play training pipeline
- Baseline comparisons (DQN, DDPG, PPO without safety)
- Validation of the Virtual Control Point (VCP) CBF formulation on a simple unicycle obstacle avoidance task

### 1.2 Why This Phase Matters

Phase 1 establishes the **core simulation loop** that all subsequent phases build upon. Every later capability — CBF safety layers, belief encoding, self-play protocols, sim-to-real transfer — depends on a correct, efficient, and well-tested PE environment. Bugs or design flaws here propagate everywhere.

Additionally, the **VCP-CBF validation** at the end of Phase 1 is a critical gate: if the CBF formulation doesn't work for nonholonomic unicycle robots, the entire safety architecture (Phases 2-4) needs rethinking. This must be confirmed before proceeding.

### 1.3 Key Design Decisions for Phase 1

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Observation type | Full state (not partial) | Simplify first; partial obs added in Phase 3 |
| Obstacles | None (open arena) | Focus on PE dynamics; obstacles added in Phase 2 |
| Safety layer | None (unconstrained) | Learn basic PE first; CBF added in Phase 2 |
| Dynamics model | Differential-drive (unicycle) | Simpler than Ackermann; TurtleBot-compatible |
| Arena shape | Rectangular (20m x 20m) | Standard; easy boundary CBF later |
| RL algorithm | PPO (Stable Baselines 3) | Proven for continuous control; self-play compatible |
| Self-play | Vanilla alternating (freeze/train) | Simplest protocol; AMS-DRL added in Phase 3 |

---

## 2. Background & Theory

### 2.1 Pursuit-Evasion Games

A **pursuit-evasion (PE) game** is a two-player zero-sum differential game where:

- **Pursuer (P)**: Minimizes time to capture the evader
- **Evader (E)**: Maximizes time to capture (or escapes entirely)
- **Capture condition**: `||p_P - p_E|| <= r_capture`
- **Timeout**: Episode ends at `T_max` if no capture

The zero-sum structure means `r_E = -r_P` — any reward gain for the pursuer is an equal loss for the evader. This creates a natural adversarial training setup.

**Game-theoretic context**: The optimal solution to a PE game is a **Nash Equilibrium (NE)** where neither player can unilaterally improve their strategy. Self-play training aims to approximate this NE. In Phase 1, we use simple alternating training; formal NE convergence analysis comes in Phase 3.

### 2.2 Unicycle (Differential-Drive) Dynamics

The unicycle model is the standard kinematic model for differential-drive ground robots like TurtleBot:

```
x_dot = v * cos(theta)
y_dot = v * sin(theta)
theta_dot = omega

State:   s = [x, y, theta]        (position + heading)
Control: u = [v, omega]           (linear velocity, angular velocity)
Bounds:  v in [0, v_max],  omega in [-omega_max, omega_max]
```

**Key properties**:
- **Nonholonomic**: The robot cannot move sideways — it must rotate to change direction
- **3D state per robot**: 6D joint state for two robots
- **Continuous control**: Both v and omega are continuous
- **Bounded actions**: Physical limits on speed and turn rate

**Why unicycle (not Ackermann)**: Unicycle dynamics are simpler (no minimum turning radius constraint), directly match TurtleBot hardware, and are sufficient for the first implementation. Paper [34] (SHADOW) uses this exact model for PE.

### 2.3 Relative Coordinate Formulation

Following Paper [13] (Kokolakis), the PE state can be expressed in **relative coordinates** to reduce dimensionality:

```
x_rel = x_P - x_E
y_rel = y_P - y_E
theta_rel = theta_P - theta_E

Relative dynamics:
x_rel_dot = v_P*cos(theta_P) - v_E*cos(theta_E)
y_rel_dot = v_P*sin(theta_P) - v_E*sin(theta_E)
theta_rel_dot = omega_P - omega_E
```

This is the standard formulation used in DeepReach [21] and MADR [22]. In Phase 1, we use **absolute coordinates** (simpler for implementation) but design the observation space to be easily convertible to relative coordinates later.

### 2.4 Reward Design for Zero-Sum PE

The reward signal must balance multiple objectives:

```python
r_P = w1 * (d_prev - d_curr) / d_max      # distance reduction (normalized)
    + w2 * capture_bonus * I(captured)       # terminal capture reward
    + w3 * timeout_penalty * I(timeout)      # terminal timeout penalty
```

**Phase 1 weights** (simplified — no visibility or CBF terms yet):
- `w1 = 1.0` — distance shaping (dense signal for learning)
- `w2 = +100.0` — large capture bonus
- `w3 = -50.0` — timeout penalty for pursuer

**Evader reward**: `r_E = -r_P` (zero-sum)

Later phases add:
- `w4 * visibility_bonus` — maintain visual contact (Phase 3, partial obs)
- `w5 * min(h_i(x)) / h_max` — CBF margin bonus (Phase 2, safety)

### 2.5 Proximal Policy Optimization (PPO)

PPO is the chosen RL algorithm because:
1. **Stable training** — clipped objective prevents catastrophic updates
2. **On-policy** — natural fit for self-play (policies change every phase)
3. **Continuous control** — supports continuous action spaces via Gaussian/Beta distributions
4. **Proven** — used successfully in Papers [18] (AMS-DRL) and [34] (SHADOW)

**PPO clipped objective**:
```
L^CLIP(theta) = E_t[min(r_t(theta) * A_t,  clip(r_t(theta), 1-eps, 1+eps) * A_t)]
r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
```

In Phase 1, we use a **Gaussian policy** (standard for continuous control). Phase 2 switches to **Beta distribution** for CBF compatibility.

### 2.6 Self-Play (Basics)

**Vanilla alternating self-play** (Phase 1):
1. Initialize both agents randomly
2. Train Pursuer for N episodes while Evader is frozen
3. Train Evader for N episodes while Pursuer is frozen
4. Repeat until convergence

This is the simplest form of self-play. It does NOT guarantee NE convergence (that requires the AMS-DRL protocol in Phase 3), but it's sufficient to learn basic PE behaviors.

**Why not simultaneous training**: Training both agents simultaneously can be unstable due to non-stationarity (each agent's environment changes as the other learns). Alternating training stabilizes this.

### 2.7 Virtual Control Point (VCP) CBF — Preview

While CBF safety is a Phase 2 deliverable, Phase 1 includes a **VCP-CBF validation** task. This is critical because:

**The problem**: For nonholonomic (unicycle) robots, position-based CBFs have **mixed relative degree** — angular velocity omega does NOT appear in the CBF derivative. This makes standard CBF-QP ill-posed.

**The solution (VCP)**: Place a virtual control point at distance `d` ahead of the robot:
```
q = [x + d*cos(theta),  y + d*sin(theta)]    (d ~ 0.05m)
```

The VCP velocity depends on BOTH v and omega:
```
q_dot = [v*cos(theta) - d*omega*sin(theta),  v*sin(theta) + d*omega*cos(theta)]
```

This achieves **uniform relative degree 1**, making CBF-QP well-posed. The VCP formulation also naturally prioritizes steering over braking (Paper [N12]).

---

## 3. Relevant Literature

### 3.1 Core Papers for Phase 1

| Paper | Relevance to Phase 1 | Key Takeaway |
|-------|----------------------|--------------|
| **[02] Gonultas & Isler 2024** | 1v1 PE on ground robots, curriculum learning, MADDPG baseline | Environment design, reward structure, baseline comparison |
| **[18] Xiao et al. 2024 (AMS-DRL)** | Self-play protocol, PPO hyperparameters, NE convergence | Training pipeline design, hyperparameter starting points |
| **[34] La Gatta 2025 (SHADOW)** | 1v1 PE with unicycle dynamics, multi-headed architecture | Validates unicycle model choice, network architecture reference |
| **[N06] Selvam et al. 2024** | Simultaneous self-play for PE with unicycle dynamics | Alternative self-play approach for comparison |
| **[N12] Zhang & Yang 2025** | VCP-CBF for nonholonomic robots | Critical: VCP formulation for the Phase 1 validation task |
| **[12] de Souza et al. 2021** | DQN/DDPG for PE | Baseline algorithms to compare against |
| **[37] Yang 2025** | RL-PE comprehensive survey | Taxonomy reference, helps position our approach |

### 3.2 Background/Theory Papers

| Paper | Topic | Why Read It |
|-------|-------|-------------|
| **[03] Ames et al.** | CBF theory (foundational) | Understand CBF basics before Phase 2 |
| **[04] Lanctot et al.** | Self-play theory | Understand game-theoretic foundations |
| **[11]** | Self-play survey | Broader context for training protocols |
| **[13] Kokolakis 2022** | Safe PE with barrier functions | Relative coordinate formulation |
| **[25] Ganai 2024** | HJ-RL survey, shielding | Roadmap for integrating safety with RL |

### 3.3 What to Read vs. Skim

**Read carefully (implementation-critical)**:
- [02]: Environment design details, reward weights, curriculum structure
- [18]: Self-play protocol details, PPO hyperparameters
- [N12]: VCP-CBF formulation (for the validation task)

**Skim for context**:
- [34]: Architecture ideas (we use simpler version in Phase 1)
- [N06]: Alternative self-play (comparison later)
- [37]: Survey — good for introduction writing

**Reference only (no deep read needed yet)**:
- [03], [04], [11], [13], [25]: Background theory

---

## 4. Deliverables Checklist

### 4.1 Must-Have Deliverables

- [ ] **D1**: Gymnasium PE environment with unicycle dynamics (`pursuit_evasion_env.py`)
- [ ] **D2**: PPO implementation via Stable Baselines 3 (SB3)
- [ ] **D3**: Basic self-play pipeline (vanilla alternating training)
- [ ] **D4**: Baseline comparisons: DQN, DDPG, PPO without safety
- [ ] **D5**: VCP-CBF validation on simple unicycle obstacle avoidance
- [ ] **D6**: Pygame visualization system (`rendering.py`) — `PERenderer` class with real-time rendering (`human` mode), video recording (`rgb_array` mode), trail visualization, HUD overlay, and CBF overlay stub for Phase 2
- [ ] **D7**: Experiment tracking system — Weights & Biases integration with `sync_tensorboard=True`, custom `PursuitEvasionMetricsCallback`, `HParamCallback`, per-run video recording via `RecordVideo` wrapper
- [ ] **D8**: Hydra configuration management (`conf/` directory) with per-component YAML files (env, algorithm, safety, wandb, experiment), CLI overrides, and `--multirun` sweep support
- [ ] **D9**: Automated test suite for environment dynamics, reward, and training smoke tests (minimum 15 tests — see [Section 7.4](#74-minimum-test-suite))

### 4.2 Nice-to-Have Deliverables

- [ ] **D13**: Advanced experiment tracking features (wandb sweeps, artifact versioning, model registry)

### 4.3 Documentation Deliverables

- [ ] **D10**: Environment API documentation
- [ ] **D11**: Training logs and learning curves
- [ ] **D12**: Baseline comparison results table

---

## 5. Session-wise Implementation Breakdown

### Session 1: Project Scaffolding & Environment Core

**Goal**: Set up project structure and implement core environment dynamics.

**Tasks**:
1. **Project structure setup**
   ```
   claude_pursuit_evasion/
   ├── envs/
   │   ├── __init__.py
   │   ├── pursuit_evasion_env.py      # Core PE environment (Gymnasium interface)
   │   ├── dynamics.py                  # Unicycle dynamics model
   │   ├── rewards.py                   # Reward computation (separate for Phase 2 swapability)
   │   ├── observations.py             # Observation construction (extensible for Phase 3)
   │   ├── wrappers.py                 # SingleAgentPEWrapper + future wrappers
   │   └── rendering.py                # Pygame visualization (real-time + video recording)
   ├── agents/
   │   ├── __init__.py
   │   └── ppo_agent.py                # PPO wrapper for self-play
   ├── training/
   │   ├── __init__.py
   │   ├── self_play.py                # Self-play training loop
   │   ├── baselines.py               # Baseline training scripts
   │   └── health_monitor.py          # Self-play health monitoring (entropy, diversity, rollback)
   ├── safety/
   │   ├── __init__.py
   │   ├── cbf.py                      # CBF implementations (Phase 2, stub for now)
   │   ├── vcp_cbf.py                  # VCP-CBF (Phase 1 validation)
   │   └── safety_filter.py           # Safety filter interface (Phase 2 extension point)
   ├── conf/                            # Hydra config directory (MUST-HAVE)
   │   ├── config.yaml                 # Defaults list (top-level Hydra config)
   │   ├── env/
   │   │   └── pursuit_evasion.yaml    # Arena, dynamics, reward params
   │   ├── algorithm/
   │   │   └── ppo.yaml                # PPO hyperparameters
   │   ├── safety/
   │   │   └── cbf.yaml                # VCP-CBF params (Phase 2 active, Phase 1 stub)
   │   ├── wandb/
   │   │   └── default.yaml            # W&B project/entity/mode config
   │   └── experiment/
   │       └── baseline.yaml           # Experiment-specific overrides
   ├── scripts/
   │   ├── train.py                    # Main training entry point
   │   ├── evaluate.py                 # Evaluation scripts
   │   └── validate_vcp_cbf.py         # VCP-CBF validation script
   ├── tests/
   │   ├── test_dynamics.py            # Unicycle dynamics unit tests (MUST-HAVE)
   │   ├── test_env.py                 # Environment unit tests (MUST-HAVE)
   │   ├── test_rewards.py             # Reward function tests (MUST-HAVE)
   │   └── test_training_smoke.py      # Training pipeline smoke tests (MUST-HAVE)
   ├── results/                         # Training results, plots, checkpoints
   │   └── .gitkeep
   ├── requirements.txt                # Pinned package versions for reproducibility
   └── claudedocs/                     # Already exists
   ```

   **Phase 2 Integration Interfaces** (implement as stubs/pass-throughs in Phase 1):

   ```python
   # safety/safety_filter.py — Extension point for CBF safety layer
   class SafetyFilter:
       """Base class for safety filtering. Phase 1: pass-through. Phase 2: VCP-CBF-QP."""
       def filter_action(self, action, state):
           """Returns safe action. Phase 1 default: return action unchanged."""
           return action
       def get_metrics(self):
           """Returns safety metrics dict. Phase 1: empty."""
           return {}

   # envs/rewards.py — Swappable reward computation
   class RewardComputer:
       """Base reward computer. Phase 2 adds w5 * CBF_margin term."""
       def __init__(self, config):
           self.w1, self.w2, self.w3 = config['w1'], config['w2'], config['w3']
       def compute(self, state, prev_state, info):
           """Returns (r_pursuer, r_evader). Override for Phase 2 safety reward."""
           ...

   # envs/observations.py — Extensible observation builder
   class ObservationBuilder:
       """Builds observations. Phase 3 adds lidar, partial obs, belief encoding."""
       def __init__(self, config):
           self.full_state = config.get('full_state', True)
       def build(self, env_state, role):
           """Returns observation array for the given role."""
           ...
   ```

2. **Implement unicycle dynamics** (`dynamics.py`):
   ```python
   # Discrete-time unicycle step (Euler integration):
   # x_new = x + v * cos(theta) * dt
   # y_new = y + v * sin(theta) * dt
   # theta_new = theta + omega * dt
   # theta_new = wrap_angle(theta_new)  # Normalize to [-pi, pi]
   #
   # Arena wall collision model (position clipping):
   # x_new = clip(x_new, robot_radius, arena_width - robot_radius)
   # y_new = clip(y_new, robot_radius, arena_height - robot_radius)
   # If clipped (wall contact): v = 0 (velocity zeroed, robot slides along wall)
   # Rationale: Clipping with velocity zeroing is simple, deterministic,
   # and prevents robots from escaping the arena. The robot's heading (theta)
   # is NOT modified on wall contact, allowing the robot to turn away naturally.
   # Note: In Phase 2+, the VCP-CBF arena boundary constraints should prevent
   # wall contact entirely, so this clipping acts as a last-resort failsafe.
   ```
   - Test: verify that a robot driving forward at v=1.0 for 1s moves 1.0m
   - Test: verify that omega=pi/2 for 1s rotates 90 degrees
   - Test: verify boundary wrapping of theta
   - Test: verify arena wall clipping — robot at (19.9, 10.0) heading east with v=1.0, dt=0.05 should be clipped to x=(20.0 - robot_radius) and v set to 0

3. **Implement core environment** (`pursuit_evasion_env.py`):
   - `__init__`: arena size, robot params, game params
   - `reset()`: random initial positions/orientations (with minimum separation)
   - `step()`: advance both robots, check capture/timeout
   - `_get_obs()`: full state observation for both agents
   - Action space: `Box(low=[0, -omega_max], high=[v_max, omega_max])`
   - Observation space: full joint state
   - Support `render_mode` in `metadata`: `["human", "rgb_array"]`

4. **Implement pygame visualization** (`envs/rendering.py`):

   The renderer follows Gymnasium's canonical pattern: lazy pygame init, `render_mode` set at construction, `_render_frame()` called from `step()`/`reset()` in human mode.

   ```python
   class PERenderer:
       """Pygame renderer for the pursuit-evasion environment.

       Supports render_mode='human' (real-time window) and
       render_mode='rgb_array' (numpy frames for video recording).
       Lazy-initializes pygame only when first render is called.
       """
       COLORS = {
           'bg': (30, 30, 30),
           'arena': (50, 50, 60),
           'arena_border': (200, 200, 200),
           'pursuer': (0, 120, 255),
           'evader': (255, 80, 80),
           'heading': (255, 255, 255),
           'trail': (100, 100, 100),
           'obstacle': (100, 100, 100),
           'cbf_safe': (0, 200, 0),
           'cbf_warning': (255, 165, 0),
           'cbf_danger': (255, 50, 50),
           'hud_text': (200, 200, 200),
       }

       def __init__(self, arena_w, arena_h, window_size=800, render_fps=30):
           self.arena_w = arena_w
           self.arena_h = arena_h
           self.window_size = window_size
           self.render_fps = render_fps
           self.scale = window_size / max(arena_w + 2, arena_h + 2)
           self.offset_x = window_size / 2
           self.offset_y = window_size / 2
           # Lazy init
           self.window = None
           self.clock = None
           self._font = None
           # Trail buffers
           self.pursuer_trail = deque(maxlen=200)
           self.evader_trail = deque(maxlen=200)

       def _world_to_pixel(self, wx, wy):
           """Convert world coords (meters) to pixel coords."""
           px = int(self.offset_x + wx * self.scale)
           py = int(self.offset_y - wy * self.scale)  # flip Y
           return (px, py)

       def _init_pygame(self, render_mode):
           if render_mode == 'human' and self.window is None:
               pygame.init()
               pygame.display.init()
               self.window = pygame.display.set_mode(
                   (self.window_size, self.window_size))
               pygame.display.set_caption('Pursuit-Evasion')
           if self.clock is None:
               self.clock = pygame.time.Clock()
           if self._font is None:
               pygame.font.init()
               self._font = pygame.font.SysFont('monospace', 14)

       def render_frame(self, render_mode, env_state):
           """
           Main render method. Returns numpy array if rgb_array mode.

           env_state: dict with keys:
             pursuer_pos, pursuer_heading, evader_pos, evader_heading,
             step, dt, distance, captured, reward, obstacles (optional),
             cbf_values (optional), cbf_active (optional)
           """
           self._init_pygame(render_mode)
           canvas = pygame.Surface((self.window_size, self.window_size))
           canvas.fill(self.COLORS['bg'])

           # Layer 1: Arena
           self._draw_arena(canvas)
           # Layer 2: Obstacles (if any)
           self._draw_obstacles(canvas, env_state.get('obstacles', []))
           # Layer 3: CBF safety boundaries (Phase 2+)
           self._draw_cbf_overlay(canvas, env_state)
           # Layer 4: Trajectory trails
           self._draw_trails(canvas, env_state)
           # Layer 5: Agents
           self._draw_agent(canvas, env_state['pursuer_pos'],
                           env_state['pursuer_heading'], self.COLORS['pursuer'], 'P')
           self._draw_agent(canvas, env_state['evader_pos'],
                           env_state['evader_heading'], self.COLORS['evader'], 'E')
           # Layer 6: HUD
           self._draw_hud(canvas, env_state)

           if render_mode == 'human':
               self.window.blit(canvas, canvas.get_rect())
               pygame.event.pump()  # CRITICAL: prevents OS "not responding"
               pygame.display.update()
               self.clock.tick(self.render_fps)
           else:  # rgb_array
               return np.transpose(
                   np.array(pygame.surfarray.pixels3d(canvas)),
                   axes=(1, 0, 2))

       def _draw_arena(self, canvas):
           """Rectangular arena with border."""
           tl = self._world_to_pixel(-self.arena_w/2, self.arena_h/2)
           w_px = int(self.arena_w * self.scale)
           h_px = int(self.arena_h * self.scale)
           pygame.draw.rect(canvas, self.COLORS['arena'],
                           (tl[0], tl[1], w_px, h_px))
           pygame.draw.rect(canvas, self.COLORS['arena_border'],
                           (tl[0], tl[1], w_px, h_px), width=3)

       def _draw_agent(self, canvas, pos, heading, color, label):
           """Draw agent as circle with heading indicator."""
           center = self._world_to_pixel(pos[0], pos[1])
           r = max(int(0.3 * self.scale), 5)
           pygame.draw.circle(canvas, color, center, r)
           # Heading line
           nose_x = pos[0] + 0.5 * np.cos(heading)
           nose_y = pos[1] + 0.5 * np.sin(heading)
           nose = self._world_to_pixel(nose_x, nose_y)
           pygame.draw.line(canvas, self.COLORS['heading'],
                           center, nose, width=3)
           # Label
           surf = self._font.render(label, True, color)
           canvas.blit(surf, (center[0]-4, center[1]-r-16))

       def _draw_trails(self, canvas, env_state):
           """Breadcrumb trail for both agents."""
           self.pursuer_trail.append(env_state['pursuer_pos'][:2].copy())
           self.evader_trail.append(env_state['evader_pos'][:2].copy())
           for trail, color in [(self.pursuer_trail, (0, 60, 130)),
                                (self.evader_trail, (130, 40, 40))]:
               if len(trail) > 1:
                   points = [self._world_to_pixel(p[0], p[1]) for p in trail]
                   pygame.draw.lines(canvas, color, False, points, width=1)

       def _draw_obstacles(self, canvas, obstacles):
           for obs in obstacles:
               center = self._world_to_pixel(obs['x'], obs['y'])
               r = int(obs['radius'] * self.scale)
               pygame.draw.circle(canvas, self.COLORS['obstacle'], center, r)

       def _draw_cbf_overlay(self, canvas, env_state):
           """Visualize CBF safety margins. Stub for Phase 1, active Phase 2+."""
           cbf_values = env_state.get('cbf_values', None)
           if cbf_values is None:
               return
           # Color-coded ring around agents based on min CBF value
           for pos, h_min in cbf_values:
               center = self._world_to_pixel(pos[0], pos[1])
               if h_min < 0.5:
                   color = self.COLORS['cbf_danger'] + (120,)
               elif h_min < 2.0:
                   color = self.COLORS['cbf_warning'] + (80,)
               else:
                   color = self.COLORS['cbf_safe'] + (40,)
               surf = pygame.Surface((self.window_size, self.window_size),
                                    pygame.SRCALPHA)
               pygame.draw.circle(surf, color, center,
                                 int(h_min * self.scale * 0.3), width=2)
               canvas.blit(surf, (0, 0))

       def _draw_hud(self, canvas, env_state):
           """Heads-up display with key metrics."""
           lines = [
               f"t={env_state.get('step', 0):04d}  "
               f"dt={env_state.get('dt', 0.05):.3f}",
               f"dist={env_state.get('distance', 0):.2f}m",
               f"reward={env_state.get('reward', 0):.3f}",
           ]
           cbf_val = env_state.get('min_cbf_value', None)
           if cbf_val is not None:
               lines.append(f"h_cbf={cbf_val:.3f}")
           for i, line in enumerate(lines):
               surf = self._font.render(line, True, self.COLORS['hud_text'])
               canvas.blit(surf, (10, 10 + i * 18))

       def reset_trails(self):
           """Call on env.reset() to clear trail buffers."""
           self.pursuer_trail.clear()
           self.evader_trail.clear()

       def close(self):
           if self.window is not None:
               pygame.display.quit()
               pygame.quit()
               self.window = None
   ```

   **Key design decisions**:
   - **Separate renderer class**: `PERenderer` is not part of the env class — composition, not inheritance. This keeps physics clean.
   - **Lazy init**: pygame is only initialized on first render. Training with `render_mode=None` never imports pygame.
   - **`pygame.event.pump()`**: Critical on macOS/Windows to prevent "not responding" status.
   - **Trail buffers**: `deque(maxlen=200)` for lightweight trajectory visualization.
   - **CBF overlay**: Stub in Phase 1, automatically activates in Phase 2 when `cbf_values` is present in env_state.
   - **Headless mode**: For servers, set `os.environ["SDL_VIDEODRIVER"] = "dummy"` before import, or just use `render_mode=None`.

   **Integration with environment**:
   ```python
   # In pursuit_evasion_env.py
   class PursuitEvasionEnv(gym.Env):
       metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

       def __init__(self, render_mode=None, **kwargs):
           self.render_mode = render_mode
           self.renderer = None  # lazy
           if render_mode is not None:
               from envs.rendering import PERenderer
               self.renderer = PERenderer(arena_w=20, arena_h=20)

       def step(self, ...):
           # ... dynamics ...
           if self.render_mode == "human":
               self.renderer.render_frame("human", self._get_render_state())
           return obs, reward, terminated, truncated, info

       def render(self):
           if self.render_mode == "rgb_array":
               return self.renderer.render_frame("rgb_array",
                                                  self._get_render_state())

       def _get_render_state(self):
           return {
               'pursuer_pos': self.pursuer_pos,
               'pursuer_heading': self.pursuer_heading,
               'evader_pos': self.evader_pos,
               'evader_heading': self.evader_heading,
               'step': self.current_step,
               'dt': self.dt,
               'distance': np.linalg.norm(
                   self.pursuer_pos[:2] - self.evader_pos[:2]),
               'reward': self.last_reward,
               'obstacles': self.obstacles,
           }

       def close(self):
           if self.renderer:
               self.renderer.close()
   ```

   **Video recording** (for evaluation and wandb logging):
   ```python
   from gymnasium.wrappers import RecordVideo

   # Record every 100th episode to ./videos/
   eval_env = PursuitEvasionEnv(render_mode="rgb_array")
   eval_env = RecordVideo(eval_env, video_folder="./videos",
                          episode_trigger=lambda ep: ep % 100 == 0)
   ```

**Validation**:
- Environment passes `gymnasium.utils.env_checker.check_env()`
- Robots move correctly under manual control inputs
- Capture detection works at correct radius
- Episode terminates at timeout
- **Pygame window opens in human mode, shows arena + agents with correct positions**
- **Agent headings rotate correctly with omega commands**
- **Trail renders correctly over multiple steps**
- **`rgb_array` mode returns valid numpy array (H, W, 3)**
- **`render_mode=None` works without importing pygame**

**Estimated effort**: 3-4 hours (+1h buffer)

---

### Session 2: Observation & Reward Design

**Goal**: Implement the observation space and reward function.

**Tasks**:

1. **Observation space design** (full state for Phase 1):
   ```python
   # Per-agent observation (Phase 1 — full state):
   obs_P = [
       x_P, y_P, theta_P,          # own state
       v_P, omega_P,                # own velocity
       x_E, y_E, theta_E,          # opponent state (full obs)
       v_E, omega_E,                # opponent velocity
       d_to_evader,                 # distance to opponent
       bearing_to_evader,           # relative bearing
       d_to_nearest_wall_x,        # distance to nearest x-wall
       d_to_nearest_wall_y,        # distance to nearest y-wall
   ]
   # Total: ~14 dimensions per agent
   ```
   - Normalize all values to [-1, 1] or [0, 1] for neural network input
   - Design the observation to be easily extensible for Phase 3 (partial obs, lidar)

2. **Reward function implementation**:
   ```python
   def _compute_reward(self):
       d_curr = distance(self.pursuer_pos, self.evader_pos)
       d_prev = self.prev_distance

       # Distance shaping
       r_dist = (d_prev - d_curr) / self.d_max

       # Terminal rewards
       r_capture = self.capture_bonus if self.captured else 0.0
       r_timeout = self.timeout_penalty if self.timed_out else 0.0

       # Pursuer reward
       r_P = 1.0 * r_dist + 100.0 * r_capture + (-50.0) * r_timeout

       # Zero-sum
       r_E = -r_P

       return r_P, r_E
   ```

3. **Multi-agent interface decision** (**LOCKED: Option B**):
   - Use a **custom `SingleAgentPEWrapper`** that runs two SB3 agents against each other
   - The base `PursuitEvasionEnv` exposes a `step(pursuer_action, evader_action)` dual-action API
   - The wrapper adapts this to SB3's single-agent `step(action)` interface by querying the frozen opponent
   - Migrate to PettingZoo Parallel API in Phase 4 when moving to Isaac Lab
   - **Rationale**: SB3's PPO expects a standard Gymnasium `Env`; PettingZoo adds complexity without benefit until Phase 4's parallel GPU environments

**Validation**:
- Reward is positive for pursuer when closing distance
- Reward is zero-sum (r_P + r_E = 0 at every step)
- Observation normalization is correct (values in expected ranges)
- Capture bonus triggers correctly
- Timeout penalty triggers correctly

**Estimated effort**: 1-2 hours

---

### Session 3: PPO Training Pipeline

**Goal**: Get a single PPO agent training against a fixed opponent.

**Tasks**:

1. **Single-agent wrapper** for SB3:
   ```python
   class SingleAgentPEWrapper(gym.Env):
       """
       Wraps the 2-agent PE env for a single agent.
       The opponent uses a fixed policy (frozen).
       """
       def __init__(self, env, role='pursuer', opponent_policy=None):
           self.env = env
           self.role = role
           self.opponent_policy = opponent_policy  # None = random

       def step(self, action):
           # Get opponent action
           if self.opponent_policy is None:
               opp_action = self.env.action_space.sample()  # random
           else:
               opp_obs = self._get_opponent_obs()
               opp_action, _ = self.opponent_policy.predict(opp_obs)

           # Step environment with both actions
           if self.role == 'pursuer':
               obs, rewards, done, truncated, info = self.env.step(action, opp_action)
               return obs['pursuer'], rewards['pursuer'], done, truncated, info
           else:
               obs, rewards, done, truncated, info = self.env.step(opp_action, action)
               return obs['evader'], rewards['evader'], done, truncated, info
   ```

2. **PPO training script** (using SB3):
   ```python
   from stable_baselines3 import PPO

   # PPO hyperparameters (from Papers [02], [16], [18]):
   ppo_config = {
       'learning_rate': 3e-4,
       'gamma': 0.99,
       'gae_lambda': 0.95,
       'clip_range': 0.2,
       'ent_coef': 0.01,
       'n_steps': 512,
       'batch_size': 256,
       'n_epochs': 10,
       'policy_kwargs': dict(net_arch=[256, 256]),
   }

   model = PPO('MlpPolicy', env, **ppo_config, verbose=1)
   model.learn(total_timesteps=1_000_000)
   ```

3. **Training against random opponent**:
   - Train pursuer vs random evader first
   - Verify pursuer learns to approach and capture
   - Then train evader vs random pursuer
   - Verify evader learns to flee

4. **Hydra config management** (`conf/`):

   ```yaml
   # conf/config.yaml — top-level Hydra config
   defaults:
     - env: pursuit_evasion
     - algorithm: ppo
     - safety: cbf
     - wandb: default
     - _self_

   seed: 42
   total_timesteps: 1_000_000
   eval_freq: 10_000
   n_eval_episodes: 20
   save_freq: 50_000
   n_envs: 4                  # parallel envs for vectorized training
   experiment_group: "default"
   ```

   ```yaml
   # conf/env/pursuit_evasion.yaml
   arena_width: 20.0
   arena_height: 20.0
   dt: 0.05
   max_steps: 1200
   capture_radius: 0.5
   pursuer:
     v_max: 1.0
     omega_max: 2.84
   evader:
     v_max: 1.0
     omega_max: 2.84
   reward:
     capture_bonus: 100.0
     timeout_penalty: -50.0
     distance_scale: 1.0
   ```

   ```yaml
   # conf/algorithm/ppo.yaml
   learning_rate: 3.0e-4
   n_steps: 512
   batch_size: 256
   n_epochs: 10
   gamma: 0.99
   gae_lambda: 0.95
   clip_range: 0.2
   ent_coef: 0.01
   vf_coef: 0.5
   max_grad_norm: 0.5
   policy_kwargs:
     net_arch: [256, 256]
   ```

   ```yaml
   # conf/wandb/default.yaml
   entity: ""                  # your wandb team (leave empty for personal)
   project: "pursuit-evasion"
   mode: "online"              # "online", "offline", "disabled"
   ```

   **CLI usage**:
   ```bash
   # Default training
   python scripts/train.py

   # Override from CLI
   python scripts/train.py algorithm.learning_rate=1e-4 seed=123

   # Disable wandb for quick local tests
   python scripts/train.py wandb.mode=disabled

   # Multi-run hyperparameter sweep
   python scripts/train.py --multirun algorithm.learning_rate=1e-3,3e-4,1e-4 seed=0,1,2,3,4
   ```

5. **Experiment tracking — wandb + TensorBoard** (`training/tracking.py`):

   Use **both** wandb and TensorBoard: SB3 logs natively to TB, and `sync_tensorboard=True` mirrors everything to wandb automatically.

   ```python
   import wandb
   from wandb.integration.sb3 import WandbCallback
   from stable_baselines3.common.callbacks import BaseCallback
   from stable_baselines3.common.logger import HParam

   def init_tracking(cfg):
       """Initialize wandb run with Hydra config as hyperparameters."""
       run = wandb.init(
           project=cfg.wandb.project,
           entity=cfg.wandb.entity or None,
           name=f"PPO_seed{cfg.seed}",
           group=cfg.experiment_group,
           tags=["ppo", "phase1", f"seed_{cfg.seed}"],
           config=OmegaConf.to_container(cfg, resolve=True),
           sync_tensorboard=True,    # auto-sync TB → wandb
           save_code=True,
           mode=cfg.wandb.mode,
       )
       return run

   class PursuitEvasionMetricsCallback(BaseCallback):
       """Log domain-specific PE metrics to TensorBoard/wandb.

       Reads episode_metrics from the info dict at episode boundaries.
       Logs every 1024 steps to avoid overhead.
       """
       def __init__(self, verbose=0):
           super().__init__(verbose)
           self.capture_times = deque(maxlen=100)
           self.min_distances = deque(maxlen=100)
           self.capture_rates = deque(maxlen=100)
           self.episode_durations = deque(maxlen=100)

       def _on_step(self) -> bool:
           for info, done in zip(self.locals.get('infos', []),
                                  self.locals.get('dones', [])):
               if done and 'episode_metrics' in info:
                   m = info['episode_metrics']
                   self.capture_rates.append(float(m.get('captured', False)))
                   self.min_distances.append(m.get('min_distance', 0))
                   if m.get('captured'):
                       self.capture_times.append(m['capture_time'])
                   self.episode_durations.append(m.get('episode_length', 0))

           if self.n_calls % 1024 == 0 and len(self.capture_rates) > 0:
               self.logger.record('pursuit/capture_rate',
                                  np.mean(self.capture_rates))
               self.logger.record('pursuit/min_distance_mean',
                                  np.mean(self.min_distances))
               self.logger.record('pursuit/episode_duration_mean',
                                  np.mean(self.episode_durations))
               if len(self.capture_times) > 0:
                   self.logger.record('pursuit/capture_time_mean',
                                      np.mean(self.capture_times))
           return True

   class HParamCallback(BaseCallback):
       """Log hyperparameters to TensorBoard HPARAMS tab."""
       def _on_training_start(self):
           hparam_dict = {
               'algorithm': self.model.__class__.__name__,
               'learning_rate': self.model.learning_rate,
               'gamma': self.model.gamma,
               'n_steps': self.model.n_steps,
               'batch_size': self.model.batch_size,
               'ent_coef': self.model.ent_coef,
           }
           metric_dict = {
               'rollout/ep_rew_mean': 0,
               'pursuit/capture_rate': 0,
           }
           self.logger.record(
               'hparams', HParam(hparam_dict, metric_dict),
               exclude=('stdout', 'log', 'json', 'csv'))
       def _on_step(self):
           return True
   ```

   **Metrics logged automatically by SB3** (via TensorBoard → wandb sync):

   | Category | Metric | Source |
   |----------|--------|--------|
   | Rollout | `rollout/ep_rew_mean`, `rollout/ep_len_mean` | SB3 auto |
   | Train | `train/policy_loss`, `train/value_loss`, `train/entropy_loss` | SB3 auto |
   | Train | `train/approx_kl`, `train/clip_fraction` | SB3 auto |

   **Custom PE metrics** (via `PursuitEvasionMetricsCallback`):

   | Category | Metric | Logged Every |
   |----------|--------|-------------|
   | Pursuit | `pursuit/capture_rate` | 1024 steps |
   | Pursuit | `pursuit/capture_time_mean` | 1024 steps |
   | Pursuit | `pursuit/min_distance_mean` | 1024 steps |
   | Pursuit | `pursuit/episode_duration_mean` | 1024 steps |

6. **Integrate tracking into training script** (`scripts/train.py`):

   ```python
   import hydra
   from omegaconf import DictConfig, OmegaConf
   from stable_baselines3 import PPO
   from stable_baselines3.common.callbacks import CallbackList, EvalCallback
   from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
   from gymnasium.wrappers import RecordVideo

   @hydra.main(version_base=None, config_path="../conf", config_name="config")
   def main(cfg: DictConfig):
       # Init tracking
       run = init_tracking(cfg)

       # Create envs
       train_env = VecMonitor(DummyVecEnv(
           [lambda: make_pe_env(cfg, render_mode=None) for _ in range(cfg.n_envs)]
       ))
       eval_env = make_pe_env(cfg, render_mode="rgb_array")
       eval_env = RecordVideo(eval_env, video_folder=f"videos/{run.id}",
                              episode_trigger=lambda ep: ep % 50 == 0)

       # Create model
       model = PPO('MlpPolicy', train_env,
                    tensorboard_log=f"runs/{run.id}",
                    seed=cfg.seed, verbose=1,
                    **OmegaConf.to_container(cfg.algorithm, resolve=True))

       # Train
       callbacks = CallbackList([
           PursuitEvasionMetricsCallback(),
           HParamCallback(),
           WandbCallback(model_save_path=f"models/{run.id}",
                        model_save_freq=cfg.save_freq, verbose=2),
           EvalCallback(eval_env, eval_freq=cfg.eval_freq,
                       n_eval_episodes=cfg.n_eval_episodes,
                       best_model_save_path=f"models/{run.id}/best"),
       ])

       model.learn(total_timesteps=cfg.total_timesteps,
                    callback=callbacks, progress_bar=True)
       run.finish()

   if __name__ == "__main__":
       main()
   ```

   **Environment must return `episode_metrics` in info dict at episode end**:
   ```python
   # In pursuit_evasion_env.py step():
   if terminated or truncated:
       info['episode_metrics'] = {
           'captured': self.captured,
           'capture_time': self.current_step * self.dt,
           'min_distance': self.min_distance_this_ep,
           'episode_length': self.current_step,
       }
   ```

**Validation**:
- Pursuer vs random: capture rate > 90% after training
- Evader vs random: escape rate > 70% after training
- Learning curves show clear improvement
- Training completes within reasonable time (~30min for 1M steps)
- **Hydra config loads correctly; CLI overrides work**
- **wandb run appears in dashboard with all metrics**
- **TensorBoard logs are readable locally with `tensorboard --logdir runs/`**
- **Custom PE metrics (capture_rate, capture_time) appear in both TB and wandb**
- **Eval videos recorded to `videos/` folder every 50 episodes**

**Estimated effort**: 3-4 hours (+1h buffer)

---

### Session 4: Self-Play Training Loop

**Goal**: Implement vanilla alternating self-play.

**Tasks**:

1. **Self-play training loop**:
   ```python
   def alternating_self_play(
       env,
       n_phases=10,
       timesteps_per_phase=200_000,
       ppo_config=default_config,
   ):
       pursuer = PPO('MlpPolicy', pursuer_env, **ppo_config)
       evader = PPO('MlpPolicy', evader_env, **ppo_config)

       history = {'capture_rate': [], 'escape_rate': [], 'mean_episode_length': []}

       for phase in range(n_phases):
           if phase % 2 == 0:
               # Train pursuer, freeze evader
               pursuer_env.set_opponent(evader)
               pursuer.learn(total_timesteps=timesteps_per_phase)
           else:
               # Train evader, freeze pursuer
               evader_env.set_opponent(pursuer)
               evader.learn(total_timesteps=timesteps_per_phase)

           # Evaluate
           metrics = evaluate(pursuer, evader, env, n_episodes=100)
           history['capture_rate'].append(metrics['capture_rate'])
           history['escape_rate'].append(metrics['escape_rate'])

           print(f"Phase {phase}: Capture={metrics['capture_rate']:.2f}, "
                 f"Escape={metrics['escape_rate']:.2f}")

       return pursuer, evader, history
   ```

2. **Evaluation function**:
   ```python
   def evaluate(pursuer, evader, env, n_episodes=100):
       captures, escapes, episode_lengths = 0, 0, []
       for _ in range(n_episodes):
           obs = env.reset()
           done = False
           while not done:
               p_action, _ = pursuer.predict(obs['pursuer'], deterministic=True)
               e_action, _ = evader.predict(obs['evader'], deterministic=True)
               obs, rewards, done, truncated, info = env.step(p_action, e_action)
               done = done or truncated
           captures += info.get('captured', False)
           escapes += info.get('timeout', False)
           episode_lengths.append(info['episode_length'])

       return {
           'capture_rate': captures / n_episodes,
           'escape_rate': escapes / n_episodes,
           'mean_episode_length': np.mean(episode_lengths),
       }
   ```

3. **Self-play analysis**:
   - Plot capture rate vs self-play phase
   - Check for oscillation (P wins → E adapts → P adapts → ...)
   - Verify that both agents improve (not just one dominating)

4. **Self-play health monitoring** (`training/health_monitor.py`):
   ```python
   class SelfPlayHealthMonitor:
       """Monitors training health and triggers rollback on detected failures."""

       def __init__(self, config):
           self.min_entropy = 0.1           # Below this = mode collapse
           self.max_capture_rate = 0.90     # Above this for 2 phases = P dominating
           self.min_capture_rate = 0.10     # Below this for 2 phases = E dominating
           self.greedy_eval_interval = 2    # Evaluate vs greedy every N phases
           self.max_checkpoints = 5         # Rolling checkpoint window

       def check_entropy(self, model):
           """Compute policy entropy from latest rollout buffer.
           For continuous actions (Gaussian): entropy = 0.5 * log(2*pi*e*sigma^2)
           Alert if entropy < self.min_entropy for either agent."""
           ...

       def check_trajectory_diversity(self, trajectories, n_clusters=3):
           """K-means on trajectory endpoints (final x,y positions).
           Alert if all trajectories cluster into <2 groups (degenerate behavior)."""
           from sklearn.cluster import KMeans
           endpoints = np.array([[t[-1][0], t[-1][1]] for t in trajectories])
           km = KMeans(n_clusters=min(n_clusters, len(endpoints)))
           km.fit(endpoints)
           n_nonempty = len(set(km.labels_))
           return n_nonempty  # Should be >= 2

       def check_greedy_baseline(self, agent, greedy_opponent, env, role, n_eps=50):
           """Evaluate trained agent against greedy heuristic.
           Alert if agent performs WORSE than random against greedy."""
           ...

       def should_rollback(self, history):
           """Returns True if last 2 phases show capture_rate outside [0.10, 0.90]."""
           if len(history) < 2:
               return False
           last_two = history[-2:]
           return all(cr > self.max_capture_rate or cr < self.min_capture_rate
                      for cr in last_two)
   ```

   **Automated health checks per self-play phase**:
   - Policy entropy for both agents (mode collapse detection)
   - Trajectory diversity via K-means (degenerate behavior detection)
   - Every 2 phases: evaluate both agents against greedy heuristic (mutual degradation detection)
   - Checkpoint rollback: if capture rate outside [0.10, 0.90] for 2 consecutive phases, restore last balanced checkpoint

5. **Self-play experiment tracking** (wandb + video):

   Each self-play phase gets its own wandb sub-run within a grouped experiment:

   ```python
   def alternating_self_play_tracked(env, cfg):
       """Self-play loop with full wandb tracking and video recording."""
       group_id = f"selfplay_{cfg.seed}_{int(time.time())}"

       for phase in range(cfg.self_play.n_phases):
           role = 'pursuer' if phase % 2 == 0 else 'evader'

           # Each phase is a wandb run within the group
           run = wandb.init(
               project=cfg.wandb.project,
               group=group_id,
               name=f"phase{phase}_{role}",
               tags=["self-play", f"phase_{phase}", role],
               config=OmegaConf.to_container(cfg, resolve=True),
               sync_tensorboard=True,
               reinit=True,
           )

           # Train with PE metrics callback + health monitor
           callbacks = CallbackList([
               PursuitEvasionMetricsCallback(),
               WandbCallback(model_save_path=f"models/{group_id}/phase{phase}",
                            model_save_freq=cfg.save_freq),
           ])
           agent.learn(total_timesteps=cfg.self_play.timesteps_per_phase,
                       callback=callbacks)

           # Evaluate and log phase-level metrics
           metrics = evaluate(pursuer, evader, env, n_episodes=100)
           wandb.log({
               'self_play/phase': phase,
               'self_play/capture_rate': metrics['capture_rate'],
               'self_play/escape_rate': metrics['escape_rate'],
               'self_play/mean_episode_length': metrics['mean_episode_length'],
           })

           # Record evaluation video
           video_env = PursuitEvasionEnv(render_mode="rgb_array", **cfg.env)
           video_env = RecordVideo(video_env,
                                   video_folder=f"videos/{group_id}/phase{phase}",
                                   episode_trigger=lambda ep: ep < 3)
           for _ in range(3):
               obs, _ = video_env.reset()
               done = False
               while not done:
                   p_act, _ = pursuer.predict(obs['pursuer'], deterministic=True)
                   e_act, _ = evader.predict(obs['evader'], deterministic=True)
                   obs, _, term, trunc, _ = video_env.step(p_act, e_act)
                   done = term or trunc
           video_env.close()

           # Log videos to wandb
           for vf in Path(f"videos/{group_id}/phase{phase}").glob("*.mp4"):
               wandb.log({"eval_video": wandb.Video(str(vf), fps=30)})

           run.finish()
   ```

   **What gets tracked per self-play phase**:

   | Metric | Where | Logged When |
   |--------|-------|-------------|
   | PPO losses, KL, clip fraction | TensorBoard → wandb (auto) | Every rollout |
   | Capture rate, distance, duration | `PursuitEvasionMetricsCallback` | Every 1024 steps |
   | Phase-level capture/escape rates | `wandb.log()` | End of each phase |
   | Evaluation video (3 episodes) | `wandb.Video()` | End of each phase |
   | Model checkpoints | `WandbCallback` | Every `save_freq` steps |
   | Health monitor entropy/diversity | `health_monitor.py` | Per phase |

**Validation**:
- Self-play runs without crashes for 10+ phases
- Both agents show improvement over training
- Capture rate oscillates and trends toward ~50% (approximate NE)
- No catastrophic forgetting: both agents beat random opponent (>60% win rate vs random)
- Health monitor correctly triggers on synthetic degenerate scenarios
- **wandb dashboard shows grouped self-play phases with capture rate progression**
- **Evaluation videos appear in wandb for each self-play phase**
- **Phase-level metrics (capture_rate, escape_rate) are plotted against phase number in wandb**

**Estimated effort**: 4-5 hours (+1h buffer)

---

### Session 5: Baseline Implementations & Comparison

**Goal**: Implement and compare baseline algorithms.

**Tasks**:

1. **DQN baseline** (from Paper [12]):
   - Discretize action space: 5 velocities x 7 angular velocities = 35 actions
   - Use SB3's DQN with same network architecture
   - Train with alternating self-play (same protocol)

2. **DDPG baseline** (from Paper [12]):
   - Continuous action space (same as PPO)
   - Use SB3's DDPG
   - Note: DDPG may be unstable in self-play (off-policy + non-stationarity)

3. **Random baseline**:
   - Both agents sample random actions
   - Establishes lower bound

4. **Greedy heuristic baseline** (`K_p = 3.0`):

   **Proportional gain justification**: With `omega_max = 2.84 rad/s`, K_p=3.0 saturates at heading error ~0.95 rad (54°) — proportional for small corrections, bang-bang for large errors. Classical PN guidance uses N=3-5 (N>2 required for stability). Dovrat, Tripathy & Bruckstein (2022, IEEE CSL) prove that for unicycle bearing pursuit with `omega = K * bearing_angle`, sufficiently large K guarantees capture via Lyapunov methods. K_p=3.0 is used for both pursuer and evader (same justification applies to flee behavior). **Expected capture rate: ~30-50%** for greedy-vs-greedy with equal speeds in a bounded arena.

   ```python
   K_p = 3.0  # Proportional navigation gain

   # Pursuer: proportional navigation (point toward evader)
   def greedy_pursuer(obs, K_p=3.0):
       bearing = atan2(y_E - y_P, x_E - x_P)
       heading_error = wrap_angle(bearing - theta_P)
       omega = clip(K_p * heading_error, -omega_max, omega_max)
       v = v_max * max(0, cos(heading_error))  # slow down when turning
       return [v, omega]

   # Evader: run away from pursuer
   def greedy_evader(obs, K_p=3.0):
       bearing = atan2(y_P - y_E, x_P - x_E)
       away_bearing = bearing + pi  # opposite direction
       heading_error = wrap_angle(away_bearing - theta_E)
       omega = clip(K_p * heading_error, -omega_max, omega_max)
       v = v_max  # Evader always runs at max speed
       return [v, omega]
   ```

5. **Comparison table**:

   | Method | Capture Rate | Mean Capture Time | Training Time | Notes |
   |--------|-------------|-------------------|---------------|-------|
   | Random | ~50% (coin flip) | — | 0 | Lower bound |
   | Greedy | ~30-50% | — | 0 | Classical baseline (equal speed + bounded arena; see Dovrat et al. 2022) |
   | DQN + SP | ? | ? | ? | Discrete actions |
   | DDPG + SP | ? | ? | ? | Continuous, off-policy |
   | **PPO + SP** | ? | ? | ? | **Our approach** |

6. **Baseline experiment tracking**:

   Each baseline gets its own wandb run within an experiment group:
   ```python
   for method in ['dqn', 'ddpg', 'ppo']:
       for seed in [0, 1, 2]:
           run = wandb.init(
               project=cfg.wandb.project,
               group="phase1_baselines",
               name=f"{method}_seed{seed}",
               tags=["baseline", method, f"seed_{seed}"],
               config={**method_config, 'seed': seed},
               sync_tensorboard=True,
               reinit=True,
           )
           # Train baseline with same callbacks
           model.learn(callbacks=CallbackList([
               PursuitEvasionMetricsCallback(),
               WandbCallback(model_save_path=f"models/baselines/{method}_{seed}"),
           ]))
           # Record 5 evaluation videos
           record_eval_videos(model, env, f"videos/baselines/{method}_{seed}", n=5)
           run.finish()
   ```

   **wandb comparison features**:
   - Group all baselines in "phase1_baselines" → compare in wandb dashboard
   - Use wandb's "Run Comparer" to overlay learning curves across methods
   - Log final comparison table as `wandb.Table` for the paper:

   ```python
   table = wandb.Table(columns=["Method", "Capture Rate", "Capture Time",
                                 "Training Time", "Seeds"])
   for method, results in all_results.items():
       table.add_data(method, f"{np.mean(results['capture_rate']):.2f} ± "
                      f"{np.std(results['capture_rate']):.2f}", ...)
   wandb.log({"baseline_comparison": table})
   ```

**Validation**:
- All baselines train without errors
- PPO self-play outperforms random and is competitive with greedy
- Comparison results are reproducible (5 random seeds)
- **All baseline runs appear in wandb with grouped learning curves**
- **Final comparison table logged to wandb as artifact**
- **Evaluation videos for each baseline recorded and viewable in wandb**

**Estimated effort**: 3-4 hours (+1h buffer)

---

### Session 6: VCP-CBF Validation (Critical Gate)

**Goal**: Validate that the VCP-CBF formulation works correctly for unicycle robots before Phase 2.

**Tasks**:

1. **Simple obstacle avoidance test environment**:
   ```python
   class UnicycleObstacleEnv(gym.Env):
       """
       Single unicycle robot navigating to a goal while avoiding one obstacle.
       Used to validate VCP-CBF before full PE integration.
       """
       # Single robot, one circular obstacle, goal position
       # Robot must reach goal without hitting obstacle
   ```

2. **VCP-CBF implementation** (`safety/vcp_cbf.py`):
   ```python
   def compute_vcp(x, y, theta, d=0.05):
       """Virtual control point at distance d ahead of robot."""
       qx = x + d * np.cos(theta)
       qy = y + d * np.sin(theta)
       return qx, qy

   def vcp_cbf_obstacle(state, obs_pos, obs_radius, d=0.05, alpha=1.0):
       """
       VCP-CBF for circular obstacle avoidance.
       Returns: h value and CBF constraint coefficients for [v, omega].

       h(x) = ||q - p_obs||^2 - chi^2
       h_dot = dh/dq * q_dot = 2*(q - p_obs)^T * q_dot
       CBF condition: h_dot + alpha * h >= 0
       """
       x, y, theta = state
       qx, qy = compute_vcp(x, y, theta, d)

       # h value
       dx = qx - obs_pos[0]
       dy = qy - obs_pos[1]
       h = dx**2 + dy**2 - obs_radius**2

       # Partial derivatives of h w.r.t. q
       dh_dqx = 2 * dx
       dh_dqy = 2 * dy

       # q_dot = [v*cos(theta) - d*omega*sin(theta),
       #          v*sin(theta) + d*omega*cos(theta)]
       # h_dot = dh_dqx * q_dot_x + dh_dqy * q_dot_y
       # h_dot = (dh_dqx*cos(theta) + dh_dqy*sin(theta)) * v
       #       + (-dh_dqx*d*sin(theta) + dh_dqy*d*cos(theta)) * omega

       # Coefficients: h_dot = a_v * v + a_omega * omega
       a_v = dh_dqx * np.cos(theta) + dh_dqy * np.sin(theta)
       a_omega = -dh_dqx * d * np.sin(theta) + dh_dqy * d * np.cos(theta)

       # CBF constraint: a_v * v + a_omega * omega + alpha * h >= 0
       return h, a_v, a_omega

   def solve_cbf_qp(u_nominal, state, constraints, v_bounds, omega_bounds):
       """
       Solve: min ||u - u_nominal||^2
              s.t. a_v_i * v + a_omega_i * omega + alpha * h_i >= 0  for all i
                   v_min <= v <= v_max
                   omega_min <= omega <= omega_max
       """
       # Use cvxpy or scipy.optimize.minimize with linear constraints
       pass
   ```

3. **Validation tests**:
   - **Test A**: Robot heading straight toward obstacle — CBF should steer around it (NOT just brake)
   - **Test B**: Robot heading parallel to obstacle — CBF should not intervene
   - **Test C**: Robot near arena boundary — CBF should redirect inward
   - **Test D**: Verify that `a_omega != 0` in constraint — this confirms VCP resolves the relative degree issue (position-based CBF would give `a_omega = 0`)
   - **Test E**: Compare VCP-CBF vs position-based CBF — VCP should steer while position-based only brakes

4. **Numerical edge case tests** (automated, in `tests/test_vcp_cbf.py`):
   - **Test F**: `h = 0` exactly (robot VCP on obstacle boundary) — QP solver must return a valid safe action, not crash or return NaN
   - **Test G**: `h < 0` (robot already in violation) — system should recover by steering away, not enter undefined state
   - **Test H**: `theta = 0, pi/2, pi, -pi/2` — verify `cos(theta)`, `sin(theta)` near-zero cases don't produce singular coefficient matrices
   - **Test I**: `d → 0.001` (very small VCP offset) — verify system degrades gracefully, `a_omega` remains nonzero
   - **Test J**: QP solver status verification — assert solver returns `optimal` (not `optimal_inaccurate` or `infeasible`) for 100 random states away from boundaries
   - **Test K**: Action bounds respected — CBF-filtered action always satisfies `v in [0, v_max]`, `omega in [-omega_max, omega_max]`

5. **Visualization** (using `PERenderer` + matplotlib):
   - **Real-time pygame**: Run obstacle avoidance with `render_mode="human"` — shows CBF safety overlay (color-coded rings around obstacles: green=safe, orange=warning, red=danger)
   - **Matplotlib plots**: Robot trajectory with CBF active; h(x) over time; control inputs (v, omega) with vs without CBF; highlight intervention regions
   - **Video recording**: Record 5 CBF validation episodes via `RecordVideo` wrapper → log to wandb as artifacts
   - **wandb logging**: Log CBF validation plots as `wandb.Image()`, h(x) time-series as `wandb.plot.line()`

**Validation (MUST ALL PASS to proceed to Phase 2)**:
- VCP-CBF correctly constrains unicycle in obstacle avoidance
- Steering preferred over braking: `mean(|delta_omega|) / mean(|delta_v|)` > 2.0 when CBF intervenes
- `|a_omega|` > 1e-6 for >99% of constrained timesteps
- Zero obstacle/boundary collisions with CBF active over 100 episodes
- Nominal control modified minimally when far from obstacles (intervention rate < 5% when `h > 2.0`)
- All numerical edge case tests (F-K) pass

**Estimated effort**: 4-5 hours (increased from 3-4 to account for numerical edge case tests)

---

### Session 7: Integration, Testing & Cleanup

**Goal**: Polish everything, run comprehensive tests, document results.

**Tasks**:

1. **Integration testing**:
   - Full self-play pipeline end-to-end
   - Reproducibility check: run same seed twice with `DummyVecEnv`, verify identical results (see Appendix B)
   - Edge case testing (agents at boundary, agents overlapping, extreme velocities)
   - Run full `pytest tests/ -v` — all 15+ tests must pass

2. **Performance profiling**:
   - Environment step time (target: <1ms per step)
   - Training throughput (steps/second)
   - Identify bottlenecks

3. **Results compilation**:
   - Learning curves for all methods
   - Baseline comparison table
   - VCP-CBF validation plots
   - Summary of what works and what needs adjustment

4. **Code cleanup**:
   - Type hints
   - Consistent formatting
   - Remove dead code
   - Verify all config parameters are in YAML

5. **Documentation**:
   - Environment API (observation/action spaces, reward, parameters)
   - How to run training
   - How to reproduce baselines
   - VCP-CBF validation results

**Validation**:
- All tests pass
- Results are reproducible
- Code is clean and documented
- Phase 1 success criteria are met

**Estimated effort**: 2-3 hours

---

## 6. Technical Specifications

### 6.1 Environment Parameters

```python
# Arena
arena_width = 20.0          # meters
arena_height = 20.0         # meters

# Robot parameters (identical for both agents)
v_max = 1.0                 # m/s (Note: TurtleBot3 is 0.22, but using 1.0 for sim)
omega_max = 2.84            # rad/s (TurtleBot3)
r_robot = 0.15              # meters (robot radius)

# Game parameters
r_capture = 0.5             # meters (capture radius)
r_collision = 0.3           # meters (physical collision, for logging only in Phase 1)
T_max = 60.0                # seconds (episode timeout)
dt = 0.05                   # seconds (20 Hz control)
max_steps = int(T_max / dt) # 1200 steps per episode

# Initial conditions
min_init_distance = 3.0     # meters (minimum initial separation)
max_init_distance = 15.0    # meters (maximum initial separation)
```

### 6.2 PPO Hyperparameters

```python
ppo_config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,         # Anneal to 0.001 in later phases
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'n_steps': 512,            # Rollout length
    'batch_size': 256,
    'n_epochs': 10,            # Update epochs per rollout
    'policy_kwargs': {
        'net_arch': [256, 256],   # 2 hidden layers of 256
        'activation_fn': torch.nn.Tanh,
    },
    'verbose': 1,
}
```

### 6.3 Self-Play Configuration

```python
self_play_config = {
    'n_phases': 10,                    # Total self-play phases
    'timesteps_per_phase': 200_000,    # Steps per training phase
    'eval_episodes': 100,              # Episodes for evaluation
    'convergence_threshold': 0.10,     # |capture_rate - 0.5| < threshold
    'n_seeds': 5,                      # Random seeds for reproducibility
}
```

### 6.4 VCP-CBF Parameters

```python
vcp_config = {
    'd': 0.05,                 # VCP offset distance (meters)
    'alpha': 1.0,              # CBF class-K function parameter
    'safety_margin': 0.1,      # Additional safety margin (meters)
}
```

### 6.5 Visualization Parameters

```python
renderer_config = {
    # Window
    'window_size': 800,             # pixels (square window)
    'render_fps': 30,               # target FPS for human mode
    'trail_maxlen': 200,            # deque length for position trails

    # Colors (RGB tuples)
    'color_bg': (30, 30, 30),       # dark background
    'color_pursuer': (0, 120, 255), # blue
    'color_evader': (255, 80, 80),  # red
    'color_arena': (60, 60, 60),    # arena fill
    'color_border': (200, 200, 200),# arena border
    'color_obstacle': (100, 100, 100),
    'color_trail_pursuer': (0, 80, 180),
    'color_trail_evader': (180, 50, 50),
    'color_capture_zone': (255, 255, 0, 80),  # yellow, semi-transparent
    'color_cbf_safe': (0, 200, 0, 40),        # green overlay (Phase 2+)
    'color_cbf_danger': (200, 0, 0, 60),      # red overlay (Phase 2+)

    # Agent rendering
    'agent_radius_px': 12,          # pixels
    'heading_line_len': 20,         # pixels (heading indicator)

    # HUD
    'hud_font_size': 18,            # pixels
    'hud_margin': 10,               # pixels from edge

    # Video recording
    'video_folder': 'videos/',      # RecordVideo output directory
    'video_fps': 20,                # frames per second in saved video
    'record_every_n_episodes': 50,  # record 1 episode every N during training
}
```

### 6.6 Experiment Tracking Configuration

```yaml
# conf/wandb/default.yaml
wandb:
  project: "pursuit-evasion"
  entity: null                       # set to your wandb team or leave null for personal
  mode: "online"                     # "online", "offline", or "disabled"
  sync_tensorboard: true             # mirror TensorBoard logs to wandb automatically
  save_code: true                    # snapshot source code with each run
  tags: ["phase1"]
  log_frequency: 1024                # log custom metrics every N steps
  video:
    enabled: true
    record_every_n_episodes: 50      # record eval video every N episodes
    fps: 20
  model:
    save_freq: 50000                 # save model checkpoint to wandb every N steps
    save_best: true                  # always save best model by eval reward

# Key metrics logged to wandb/TensorBoard:
# - rollout/ep_rew_mean              (SB3 default via VecMonitor)
# - rollout/ep_len_mean              (SB3 default via VecMonitor)
# - pe/capture_rate                   (custom: captures / total episodes)
# - pe/mean_capture_time              (custom: mean steps to capture)
# - pe/mean_min_distance              (custom: closest approach per episode)
# - pe/evader_mean_omega              (custom: evasion aggressiveness)
# - pe/mean_distance_at_termination   (custom: how close at episode end)
# - train/policy_entropy              (from SB3 logger)
# - train/approx_kl                   (from SB3 logger)
# - eval/mean_reward                  (from EvalCallback)
```

---

## 7. Validation & Success Criteria

### 7.1 Must-Pass Criteria (Gate to Phase 2)

| Criterion | Target | How to Measure |
|-----------|--------|---------------|
| Pursuer learns to capture | Capture rate > 80% vs random evader | Evaluate over 100 episodes, 5 seeds |
| Evader learns non-trivial evasion | Mean `|omega|` > 0.5 rad/s AND trajectory curvature variance > 0.1 | Automated metric over 100 episodes (rejects straight-line flee) |
| Training converges | Mean reward improvement < 1% for 5 consecutive eval windows, within 2M timesteps | Log eval reward every 50K steps; check plateau condition |
| VCP-CBF works: steering over braking | When CBF intervenes: `mean(|delta_omega|) / mean(|delta_v|)` > 2.0 | Compute ratio over all CBF intervention timesteps in 100 test episodes |
| VCP resolves relative degree | `|a_omega|` > 1e-6 in CBF constraints for >99% of constrained steps | Numerical check logged during VCP-CBF validation runs |
| Zero collisions with VCP-CBF | 0 obstacle/boundary violations | Logged over 100 test episodes with random initial conditions |
| Self-play trends toward balance | Capture rate in [0.35, 0.65] for at least 2 of the last 3 self-play phases | Log capture rate per self-play phase |
| All automated tests pass | 15+ tests in `tests/` pass with 0 failures | `pytest tests/ -v` returns exit code 0 |
| Pygame rendering works | `render_mode="human"` displays window; `render_mode="rgb_array"` returns (H,W,3) uint8 array | Manual check + `RecordVideo` produces valid .mp4 |
| Experiment tracking operational | wandb run creates valid dashboard with ≥5 custom PE metrics | Verify `wandb.run` is not None; check logged keys include `pe/capture_rate` |
| Hydra config loads | `python scripts/train.py --cfg job` prints resolved config without errors | CLI override test: `env.arena_width=25` changes value |

**Definition of Done**: Phase 1 is COMPLETE when deliverables D1-D9 are verified AND ALL must-pass criteria above are met. Document results in `results/phase1_report.md` before proceeding to Phase 2.

### 7.2 Quality Metrics

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| Env step time | < 1ms | Must be fast for 16-64 parallel envs later |
| Training throughput | > 5000 steps/sec | With SB3 vectorized envs |
| Self-play capture rate | Oscillates within [0.3, 0.7] | Sign of balanced training |
| Mean episode length | Increases over self-play phases | Both agents getting better |
| PPO vs baselines | PPO >= DDPG > DQN > Random | Expected ordering |
| Policy entropy (PPO) | > 0.5 during training (continuous actions) | Mode collapse if < 0.1 |
| Reward variance | Decreasing over training | Sign of convergence |

### 7.3 What Failure Looks Like

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Pursuer doesn't learn | Reward too sparse | Increase distance shaping weight w1 |
| Both agents go to corners | Reward exploit | Add boundary penalty or normalize distance by arena size |
| Training diverges | Learning rate too high | Reduce LR, increase batch size |
| Self-play oscillates wildly | Phases too short | Increase timesteps_per_phase |
| VCP-CBF only brakes | VCP offset d too small | Increase d (but keep < 0.1m) |
| VCP-CBF too aggressive | alpha too high | Reduce alpha (try 0.5) |
| Mode collapse (both agents output constant actions) | Entropy coeff too low or SP phases too long | Monitor policy entropy; rollback checkpoint if entropy < 0.1 |
| Degenerate mutual degradation | Both agents forget good behavior simultaneously | Evaluate against greedy heuristic baseline every 2 SP phases |

### 7.4 Minimum Test Suite

The following 15+ automated tests are **must-have** (deliverable D9). All must pass before Phase 2.

**`tests/test_dynamics.py`** (5 tests):
```python
def test_forward_motion():
    """v=1.0, omega=0, dt=0.05 → moves 0.05m forward along heading."""
    state = [0, 0, 0]  # x, y, theta
    new_state = unicycle_step(state, v=1.0, omega=0.0, dt=0.05)
    assert abs(new_state[0] - 0.05) < 1e-10
    assert abs(new_state[1] - 0.0) < 1e-10

def test_pure_rotation():
    """v=0, omega=pi/2, dt=1.0 → rotates 90 degrees, no translation."""
    state = [5.0, 5.0, 0.0]
    new_state = unicycle_step(state, v=0.0, omega=np.pi/2, dt=1.0)
    assert abs(new_state[0] - 5.0) < 1e-10
    assert abs(new_state[2] - np.pi/2) < 1e-6

def test_angle_wrapping():
    """theta near pi wraps correctly to [-pi, pi]."""
    state = [0, 0, 3.1]
    new_state = unicycle_step(state, v=0.0, omega=0.2, dt=1.0)
    assert -np.pi <= new_state[2] <= np.pi

def test_circular_arc():
    """v=1.0, omega=1.0 for 2*pi seconds → returns near start (full circle)."""
    state = [0, 0, 0]
    for _ in range(int(2 * np.pi / 0.05)):
        state = unicycle_step(state, v=1.0, omega=1.0, dt=0.05)
    assert np.linalg.norm([state[0], state[1]]) < 0.5  # back near origin

def test_numerical_stability_long_episode():
    """1200 steps of Euler integration doesn't accumulate large error."""
    state = [0, 0, 0]
    for _ in range(1200):
        state = unicycle_step(state, v=1.0, omega=0.01, dt=0.05)
    assert all(np.isfinite(state))
    assert abs(state[2]) < 100  # theta shouldn't blow up
```

**`tests/test_env.py`** (5 tests):
```python
def test_env_checker():
    """Environment passes Gymnasium's built-in validator."""
    env = PursuitEvasionEnv()
    check_env(env.unwrapped)

def test_observation_normalization():
    """All observation values lie in expected ranges after reset."""
    env = PursuitEvasionEnv()
    obs = env.reset()
    assert np.all(obs >= -1.1) and np.all(obs <= 1.1)  # small tolerance

def test_initial_separation():
    """Initial pursuer-evader distance is in [min_init_distance, max_init_distance]."""
    env = PursuitEvasionEnv()
    for _ in range(50):
        env.reset()
        d = np.linalg.norm(env.pursuer_pos[:2] - env.evader_pos[:2])
        assert 3.0 <= d <= 15.0

def test_capture_detection():
    """Capture triggers when distance < r_capture."""
    env = PursuitEvasionEnv()
    env.pursuer_pos = np.array([10.0, 10.0, 0.0])
    env.evader_pos = np.array([10.3, 10.0, 0.0])  # d=0.3 < r_capture=0.5
    assert env._check_capture() == True

def test_episode_truncation():
    """Episode terminates at max_steps."""
    env = PursuitEvasionEnv()
    env.reset()
    for _ in range(env.max_steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    assert truncated or terminated
```

**`tests/test_rewards.py`** (3 tests):
```python
def test_zero_sum():
    """r_P + r_E == 0 for every step."""
    env = PursuitEvasionEnv()
    env.reset()
    for _ in range(100):
        _, (r_p, r_e), _, _, _ = env.step(env.action_space.sample())
        assert abs(r_p + r_e) < 1e-10

def test_capture_bonus():
    """Capture gives r_P = large positive, r_E = large negative."""
    # Setup near-capture scenario and verify bonus triggers
    ...

def test_distance_shaping_sign():
    """Closing distance gives positive r_P; increasing gives negative."""
    ...
```

**`tests/test_training_smoke.py`** (2 tests):
```python
def test_ppo_trains_1000_steps():
    """PPO runs 1000 timesteps without crashing."""
    env = SingleAgentPEWrapper(PursuitEvasionEnv(), role='pursuer')
    model = PPO('MlpPolicy', env, n_steps=64, batch_size=32, verbose=0)
    model.learn(total_timesteps=1000)

def test_self_play_one_phase():
    """One self-play phase (1000 steps each side) completes without error."""
    ...
```

### 7.5 Concrete Worked Examples

These examples serve as both documentation and regression test references.

**Example 1 — Capture Scenario**:
```
Given: Pursuer at (9.0, 10.0, 0.0), Evader at (11.0, 10.0, π)
       Both heading toward each other, v_max = 1.0 m/s, dt = 0.05s
       r_capture = 0.5m

When:  Pursuer drives forward at v=1.0, omega=0.0
       Evader drives forward at v=1.0, omega=0.0 (toward pursuer since θ_E=π)

Then:  Closing speed = 2.0 m/s (both approaching)
       Initial distance = 2.0m
       Capture at step ≈ 15:  d = 2.0 - 2.0*0.05*15 = 0.5m
       Episode terminates with info['captured'] = True
```

**Example 2 — Observation Vector**:
```
Given: Pursuer at (5.0, 5.0, π/4), v_P=0.5, omega_P=0.1
       Evader at (15.0, 10.0, π), v_E=0.8, omega_E=-0.3
       Arena: 20m x 20m

Then:  d_to_evader = sqrt((15-5)² + (10-5)²) = sqrt(125) ≈ 11.18m
       bearing = atan2(10-5, 15-5) - π/4 = atan2(5,10) - π/4 ≈ 0.4636 - 0.7854 = -0.322 rad
       d_nearest_wall_x = min(5.0, 15.0) = 5.0m
       d_nearest_wall_y = min(5.0, 15.0) = 5.0m

       obs_P (normalized) = [
           5.0/10 - 1,    # x_P norm: -0.5
           5.0/10 - 1,    # y_P norm: -0.5
           (π/4)/π,       # theta_P norm: 0.25
           0.5/1.0,       # v_P norm: 0.5
           0.1/2.84,      # omega_P norm: 0.035
           15.0/10 - 1,   # x_E norm: 0.5
           10.0/10 - 1,   # y_E norm: 0.0
           π/π,           # theta_E norm: 1.0
           0.8/1.0,       # v_E norm: 0.8
           -0.3/2.84,     # omega_E norm: -0.106
           11.18/28.28,   # d_to_evader norm: 0.395
           -0.322/π,      # bearing norm: -0.102
           5.0/10,        # d_wall_x norm: 0.5
           5.0/10,        # d_wall_y norm: 0.5
       ]
```

**Example 3 — Reward Calculation**:
```
Given: d_prev = 10.0m, d_curr = 9.5m, d_max = 28.28m (arena diagonal)
       Not captured, not timed out

Then:  r_dist = (10.0 - 9.5) / 28.28 = 0.01768
       r_capture = 0
       r_timeout = 0
       r_P = 1.0 * 0.01768 + 100.0 * 0 + (-50.0) * 0 = 0.01768
       r_E = -0.01768

Given: d_prev = 0.6m, d_curr = 0.4m, captured = True
Then:  r_P = 1.0 * (0.6-0.4)/28.28 + 100.0 * 1 + 0 = 100.00707
       r_E = -100.00707
```

**Example 4 — Healthy vs Unhealthy Self-Play**:
```
HEALTHY oscillation (converging to balance):
  Phase 0 (P trains): CR = 0.95  (P dominates random E)
  Phase 1 (E trains): CR = 0.30  (E adapts, escapes more)
  Phase 2 (P trains): CR = 0.65  (P counter-adapts)
  Phase 3 (E trains): CR = 0.45  (E counter-adapts)
  Phase 4 (P trains): CR = 0.58  (settling)
  ...
  Phase 9:            CR = 0.52  ✓ Within [0.35, 0.65]

UNHEALTHY oscillation (no convergence):
  Phase 0: CR = 0.99, Phase 1: CR = 0.01, Phase 2: CR = 0.99 ...
  → Wild swings indicate phases too short or catastrophic forgetting
  → FIX: Increase timesteps_per_phase; add checkpoint rollback

UNHEALTHY degradation (mutual collapse):
  Phase 0: CR = 0.80, Phase 3: CR = 0.55, Phase 6: CR = 0.50
  BUT: Both agents lose to greedy heuristic (capture_rate vs greedy < 0.3)
  → Both agents degraded together while maintaining balanced CR
  → FIX: Evaluate against fixed greedy baseline every 2 phases
```

---

## 8. Risk Assessment

### 8.1 Phase 1 Specific Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SB3 doesn't support multi-agent well | Medium | Medium | Custom `SingleAgentPEWrapper` (Session 3); Option B is locked in |
| Self-play is unstable | Medium | Low | Use longer phases; health monitor with checkpoint rollback (Session 4) |
| VCP-CBF validation fails | Low | HIGH | Verify math carefully; test on simpler 2D system first; fall back to PNCBF [N02] |
| Environment bugs (physics) | Medium | Medium | Automated unit test suite (D9, must-have); 15+ tests |
| Reward hacking | Low | Medium | Automated reward sanity checks: flag episodes where reward is anomalously high but no capture occurs |
| Training too slow | Low | Low | Vectorize environments; reduce network size |
| Non-reproducible results | Medium | Medium | Full seed protocol (Appendix B); use DummyVecEnv for strict reproducibility |
| Mode collapse in self-play | Medium | Medium | Policy entropy monitoring; trigger rollback if entropy < 0.1 |
| Pygame slows training | Medium | Low | Use `render_mode=None` during training; only enable for eval/recording. Lazy init avoids import overhead |
| wandb rate-limiting / network issues | Low | Low | Use `mode: "offline"` fallback; `wandb sync` after training. Set `WANDB_MODE=offline` env var |
| Hydra config conflicts | Low | Medium | Test all override combinations in CI; pin `omegaconf==2.3.0` to match `hydra-core==1.3.2` |

### 8.2 Risks That Affect Later Phases

| Risk in Phase 1 | Impact on Later Phase | How to Detect Early |
|-----------------|----------------------|-------------------|
| Poor environment design | Phase 2-5 (everything) | Check extensibility: can you easily add obstacles? partial obs? |
| Wrong observation normalization | Phase 2-3 | Monitor Q-value scales during training |
| Self-play wrapper fragile | Phase 3 (AMS-DRL) | Test with different opponent policies |
| VCP-CBF math error | Phase 2 (CBF integration) | The validation tests in Session 6 |

---

## 9. Software & Tools

### 9.1 Required Packages (Pinned Versions)

Create `requirements.txt` at project root with these pinned versions for reproducibility:

```txt
# requirements.txt — Phase 1 pinned versions (verified compatible 2026-02)
# Core RL (SB3 2.7.x requires gymnasium>=0.29.1,<1.3.0 and torch>=2.3,<3.0)
gymnasium==1.1.1
stable-baselines3[extra]==2.7.1
torch==2.6.0
numpy==2.2.0

# CBF (for VCP-CBF validation)
cvxpy==1.6.5                 # requires numpy>=1.21.6, scipy>=1.11.0
scipy==1.15.0

# Visualization (must-have)
matplotlib==3.10.0
pygame==2.6.1                # Real-time rendering + video recording

# Experiment tracking (must-have)
wandb==0.19.1
hydra-core==1.3.2            # Config management with CLI overrides + sweeps
omegaconf==2.3.0             # Required by Hydra (YAML config resolution)

# Testing & Analysis
pytest==8.3.4
scikit-learn>=1.4.0          # For K-means in health monitor

# Reproducibility & Config
pyyaml>=6.0                  # For config files
```

**Note**: Requires Python >= 3.10. Use `pip install -r requirements.txt` to install.

Install with: `pip install -r requirements.txt`

### 9.2 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores (for vectorized envs) |
| RAM | 8 GB | 16 GB |
| GPU | Not required for Phase 1 | NVIDIA GPU speeds up PPO training |
| Storage | 5 GB | 20 GB (for logs and checkpoints) |

### 9.3 Development Environment

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU training)
- IDE: VSCode with Python extension

---

## 10. Guide to Phase 2

### 10.1 What Phase 2 Needs from Phase 1

| Phase 1 Output | Phase 2 Usage |
|----------------|---------------|
| `pursuit_evasion_env.py` | Extended with obstacles, VCP-CBF safety layer |
| `dynamics.py` | Used as-is; VCP-CBF wraps around it |
| PPO training pipeline | Extended with Beta distribution policy |
| Self-play loop | Extended with AMS-DRL protocol (Phase 3, but Phase 2 uses basic SP) |
| VCP-CBF validation | Directly used for arena boundary + obstacle CBFs |
| Baseline results | Comparison: "our safe method vs unconstrained Phase 1" |

### 10.2 Key Transitions

| Phase 1 → Phase 2 Change | Details |
|--------------------------|---------|
| No safety → VCP-CBF | Add CBF constraints to environment/policy |
| Gaussian policy → Beta policy | Beta distribution for CBF-compatible truncation |
| No obstacles → 2-4 obstacles | Add static obstacles with CBF avoidance |
| Simple reward → Safety-shaped reward | Add w5 * min(h_i(x)) / h_max term |
| No infeasibility handling → 3-tier system | N13 learned feasibility + relaxation + backup |

### 10.3 What to Carry Forward

- **All trained models**: Save Phase 1 models as baselines
- **Training logs**: Needed for comparison in Phase 2
- **VCP-CBF code**: Directly used in Phase 2
- **Environment code**: Extended (not rewritten)
- **Self-play infrastructure**: Reused with safety additions

### 10.4 Potential Phase 2 Blockers from Phase 1

| Issue | Detection | Resolution |
|-------|-----------|-----------|
| VCP-CBF doesn't work | Session 6 validation fails | Switch to PNCBF [N02] or HOCBF; consult Paper [N12] more carefully |
| Environment too slow | Profiling in Session 7 | Optimize hot loops; consider C extension for dynamics |
| Self-play wrapper incompatible with CBF | Manual review | Refactor wrapper to accept safety filter as argument |
| Observation space wrong for CBF | Manual review | CBF needs full state (x, y, theta) even if obs is partial — add separate state tracking |

### 10.5 Reading List Before Starting Phase 2

Before beginning Phase 2 implementation, read these papers in detail:
1. **[16] Suttle 2024** — CBF-constrained Beta policy (core Phase 2 algorithm)
2. **[N12] Zhang & Yang 2025** — VCP-CBF (already validated in Phase 1, but read for integration details)
3. **[N13] Xiao et al. 2023** — Learned feasibility constraints
4. **[05] Yang 2025** — CBF safety filtering + reward shaping
5. **[06] Emam 2022** — RCBF-QP with GP (for deployment layer)

---

## Appendix A: Quick-Reference Equations

### A.1 Unicycle Dynamics
```
x_dot = v * cos(theta)
y_dot = v * sin(theta)
theta_dot = omega
```

### A.2 VCP Position & Velocity
```
q = [x + d*cos(theta),  y + d*sin(theta)]
q_dot = [v*cos(theta) - d*omega*sin(theta),  v*sin(theta) + d*omega*cos(theta)]
```

### A.3 VCP-CBF Obstacle Constraint
```
h(x) = ||q - p_obs||^2 - chi^2
h_dot = 2*(q - p_obs)^T * q_dot
CBF: h_dot + alpha * h >= 0
```

### A.4 PPO Objective
```
L^CLIP = E[min(r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t)]
r_t = pi_new(a|s) / pi_old(a|s)
```

### A.5 Zero-Sum Reward
```
r_P = w1*(d_prev - d_curr)/d_max + w2*I(capture) + w3*I(timeout)
r_E = -r_P
```

---

## Appendix B: Reproducibility Protocol

### B.1 Seed Management

SB3's `set_random_seed(seed)` seeds Python `random`, NumPy, and PyTorch, but does NOT cover `PYTHONHASHSEED`, `torch.use_deterministic_algorithms()`, or `CUBLAS_WORKSPACE_CONFIG`. The following protocol fills these gaps.

```python
import os
import random
import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

def setup_reproducibility(seed: int, use_cuda: bool = False):
    """Full reproducibility setup. Call BEFORE creating envs or models."""
    # 1. Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. SB3 built-in seeding (python random, numpy, torch)
    set_random_seed(seed, using_cuda=use_cuda)

    # 3. CUDA determinism (not covered by SB3)
    if use_cuda and torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def make_reproducible_env(env_class, n_envs, seed, **env_kwargs):
    """Use DummyVecEnv (not SubprocVecEnv) for strict reproducibility."""
    return make_vec_env(
        env_class, n_envs=n_envs, seed=seed,
        vec_env_cls=DummyVecEnv,  # Single-process = reproducible
        env_kwargs=env_kwargs,
    )
```

### B.2 Key Limitations

| Concern | Resolution |
|---------|-----------|
| `SubprocVecEnv` breaks reproducibility | Use `DummyVecEnv` for validation runs; `SubprocVecEnv` acceptable for exploratory training |
| PPO is more sensitive to seed than off-policy (TD3/SAC) | Run 5 seeds minimum; report mean ± std |
| Cross-platform irreproducibility | Same platform + same CUDA version required for bit-for-bit matching |
| `torch.use_deterministic_algorithms` may error | Some PyTorch ops lack deterministic impl; catch and document |

### B.3 Evaluation Seeding

```python
# Evaluation env MUST be seeded separately
eval_env = make_reproducible_env(PursuitEvasionEnv, n_envs=1, seed=seed + 1000)
# Use deterministic=True in predict() for evaluation
action, _ = model.predict(obs, deterministic=True)
```
