# Module 3: Environment Design

**Estimated reading time:** 2–3 hours  
**Prerequisites:** Module 1 (Pursuit-Evasion as a Differential Game), Module 2 (Deep RL for Continuous Multi-Agent Control)  
**Goal:** Understand every design choice in the G&I 2024 environment — the arena, the agents, the sensors, the reward, and the curriculum — and why each parameter matters. This module bridges differential game theory to the concrete simulation that all our experiments run in.

---

## 3.1 The Arena: Why 16×16 Meters?

The **arena** is the bounded domain in which the game is played. Its size is the single most consequential hyperparameter: too small and capture is trivial; too large and the episode never terminates.

### The Scaling Problem

In a differential game with capture radius $c$ and maximum speed $v_{\max}$, the natural timescale is:

$$T_{\text{capture}} \approx \frac{L_{\text{arena}}}{v_{\max}}$$

For the G&I environment: $L_{\text{arena}} = 16$ m, $v_{\max}^{(p)} = 2.5$ m/s, so $T_{\text{capture}} \approx 6.4$ s. With $\Delta t = 0.1$ s, this is $\approx 64$ steps. The episode length of 400 steps (40 s) gives roughly 6× the minimum capture time — enough for extended chases, pursuit-and-escape cycles, and boundary exploitation.

**Why not smaller?** Our Phase 1–3 10×10 m arena (S11–S84) gave $T_{\text{capture}} \approx 4$ s with 200 steps — agents reached the boundary quickly and strategies degenerated into corner-trapping. **Why not larger?** A 30×30 m arena would require 800+ steps for meaningful episodes; training time scales linearly, and the pursuer's speed advantage becomes less decisive at long range.

| Arena Size | $T_{\text{capture}}$ | Steps | Tradeoff |
|------------|----------------------|-------|----------|
| 10×10 m | 4.0 s | 40 | Too fast; strategies degenerate |
| **16×16 m** | **6.4 s** | **64** | **Balanced: chase + boundary play** |
| 30×30 m | 12.0 s | 120 | Too long; evader outruns indefinitely |

### Episode Length: 400 Steps

The episode terminates when either:
1. **Capture:** distance $\leq 2c = 0.30$ m (two agent radii touching)
2. **Timeout:** 400 steps elapsed ($40$ s at $\Delta t = 0.1$ s)

The timeout is not merely a computational convenience — it is a **game-theoretic necessity**. Without it, the evader's optimal strategy is to avoid capture forever (if possible). The timeout forces the evader to actively escape rather than passively evade, and it gives the pursuer a finite horizon within which to act.

The 400-step limit ensures a pursuer at one corner can close a diagonal gap ($\approx 22.6$ m) in $\approx 23$ s at 1.0 m/s relative speed — well within the 40 s budget — while giving the evader $\approx 30$ s of boundary exploitation before cornering.

---

## 3.2 Agent Dynamics: Bicycle vs Point Mass

The G&I environment uses **asymmetric dynamics** — the pursuer is a bicycle model, the evader is a point mass. This is a deliberate design choice that creates a nuanced tradeoff.

### Pursuer: Bicycle Model

The pursuer's state is $\mathbf{x}_p = [x, y, \delta, v, \psi]$ where:
- $(x, y)$: position
- $\delta$: steering angle
- $v$: longitudinal velocity
- $\psi$: heading angle

The dynamics (Euler-integrated at $\Delta t = 0.1$ s) are:

$$\begin{aligned}
\delta_{t+1} &= \text{clip}\bigl(\delta_t + u_{\dot{\delta}} \cdot \Delta t, \; \delta_{\min}, \; \delta_{\max}\bigr) \\
v_{t+1} &= \text{clip}\bigl(v_t + u_a \cdot \Delta t, \; v_{\min}, \; v_{\max}\bigr) \\
\psi_{t+1} &= \psi_t + \frac{v_t}{L} \tan(\delta_t) \cdot \Delta t \\
x_{t+1} &= x_t + v_t \cos(\psi_t) \cdot \Delta t \\
y_{t+1} &= y_t + v_t \sin(\psi_t) \cdot \Delta t
\end{aligned}$$

**Key parameters:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Wheelbase $L$ | 0.30 m | Distance between axles |
| $v_{\max}$ | 2.5 m/s | Top speed |
| $v_{\min}$ | −1.0 m/s | Reverse speed |
| $\delta_{\max}$ | 0.34 rad | Max steering angle |
| $a_{\max}$ | 2.0 m/s² | Max acceleration |
| $\dot{\delta}_{\max}$ | 3.2 rad/s | Max steering rate |

**The bicycle model is non-holonomic:** the pursuer cannot move sideways. To change direction, it must steer, which changes its heading $\psi$ gradually. The minimum turning radius is:

$$R_{\min} = \frac{L}{\tan(\delta_{\max})} = \frac{0.30}{\tan(0.34)} \approx 0.85 \text{ m}$$

This means the pursuer cannot turn in place — it needs space to maneuver. A tight corner or a quick direction change is physically impossible. The evader, with point-mass dynamics, can change direction instantly.

### Evader: Point-Mass Model

The evader's state is $\mathbf{x}_e = [x, y, v_x, v_y]$ with dynamics:

$$\begin{aligned}
v_{x,t+1} &= \text{clip}\bigl(v_{x,t} + u_x \cdot \Delta t, \; v_{\min}, \; v_{\max}\bigr) \\
v_{y,t+1} &= \text{clip}\bigl(v_{y,t} + u_y \cdot \Delta t, \; v_{\min}, \; v_{\max}\bigr) \\
x_{t+1} &= x_t + v_{x,t} \cdot \Delta t \\
y_{t+1} &= y_t + v_{y,t} \cdot \Delta t
\end{aligned}$$

**Key parameters:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| $v_{\max}$ | 1.5 m/s | Top speed |
| $v_{\min}$ | −1.5 m/s | Reverse speed |
| $a_{\max}$ | 9.81 m/s² | Max acceleration (≈ 1 g) |

The evader is **omnidirectional** — no heading angle, no steering constraint, no minimum turning radius. It can execute any velocity vector within the speed disk $\|\mathbf{v}\| \leq 1.5$.

### Why Asymmetric Dynamics?

| Property | Pursuer (Bicycle) | Evader (Point Mass) |
|----------|-------------------|---------------------|
| Top speed | 2.5 m/s (faster) | 1.5 m/s (slower) |
| Turning | Constrained ($R_{\min} \approx 0.85$ m) | Instantaneous |
| Reversing | Limited ($v_{\min} = -1.0$) | Full ($v_{\min} = -1.5$) |
| Acceleration | 2.0 m/s² | 9.81 m/s² |

The asymmetry creates a genuine contest: the **pursuer wins on speed** (can close gaps in open space), while the **evader wins on agility** (can dodge and exploit the pursuer's turning radius). If both were point masses with equal speed, the evader could always escape (mirroring argument, M1 §1.5). The mixed dynamics create rich strategy spaces for both agents.

### Common Misconceptions

**"Why not give both agents bicycle models?"** Equal non-holonomic dynamics would eliminate the agility asymmetry that makes the game interesting. The evader's ability to change direction instantly is its primary defense; without it, the pursuer's speed advantage would dominate in every scenario. The bicycle model also introduces heading as a state variable, which complicates the evader's strategy without adding meaningful depth.

**"Why not a larger capture radius?"** A larger capture radius (e.g., 0.50 m) would make capture trivial — the pursuer could swerve into the evader without precise alignment. The current 0.30 m effective distance is roughly 1/3 of the pursuer's minimum turning radius (0.85 m), forcing the pursuer to plan its approach trajectory rather than relying on last-second corrections. This precision requirement is what makes the game competitive rather than a simple chase.

> **Implementation note:** Our initial implementation used $L = 0.15$ m (half the paper's value), making $R_{\min} \approx 0.43$ m. The pursuer could execute hairpin turns impossible with $L = 0.30$ m — this single bug explained persistent pursuer dominance (CR = 1.0) across early runs. Fixing $L$ to 0.30 m was one of the two critical fixes in S93 (see D42, D43).

---

## 3.3 Sensing: Partial Observability via FOV

Sensing is where the G&I environment introduces **partial observability** — agents do not see the full state. This transforms the game from a fully-observable differential game into a **Partially Observable Markov Decision Process (POMDP)**.

### The Wedge Sensor

Both agents use a **WedgeSensor** class that implements a triangular field of view:

```python
visible = (dist <= fov_range) and (|angle_diff| <= fov_angle / 2)
```

**Pursuer sensor:** FOV = $120°$ ($2\pi/3$ rad), range = 6.0 m  
**Evader sensor:** FOV = $360°$ ($2\pi$ rad), range = 1000.0 m (effectively infinite)

### Why This Asymmetry?

The pursuer has **worse sensing** in two ways:
1. **Directional:** 120° wedge vs omnidirectional. The pursuer must physically turn to scan behind itself.
2. **Limited range:** 6.0 m vs 1000 m. Beyond 6 m, the pursuer has no information about the evader's position.

This compensates for the pursuer's speed advantage. In a fully observable game with the pursuer 67% faster, capture would be nearly guaranteed. With FOV-limited sensing, the evader can:
- Hide outside the pursuer's FOV
- Use the boundary to create "shadows" — regions the pursuer cannot see into
- Approach from behind the pursuer (outside the 120° wedge)

**Arena coverage:** The pursuer's 6 m range covers $\pi \cdot 6^2 \approx 113$ m² (44% of the 256 m² arena), but because the FOV is directional, effective coverage is only a 120° wedge. The fraction is:

$$\frac{\text{sector area}}{\text{arena area}} = \frac{120°}{360°} \cdot \frac{\pi \cdot 6^2}{16^2} = \frac{1}{3} \cdot \frac{36\pi}{256} \approx 14.7\%$$

So the pursuer sees roughly 15% of the arena at any orientation.

### Observation Space (11-Dimensional Raw, 22-Dimensional Stacked)

Each agent receives an 11-dimensional observation vector:

| Index | Feature | Description | Normalization |
|-------|---------|-------------|---------------|
| 0–1 | Own position $(x, y)$ | Arena coordinates | $/ (L/2)$ → $[-1, 1]$ |
| 2–3 | Own velocity $(v_x, v_y)$ | Current velocity | Pursuer: $/ 1.75$; Evader: $/ 1.5$ |
| 4–5 | Opponent position $(x_o, y_o)$ | If visible; else 0 | $/ (L/2)$ |
| 6–7 | Opponent velocity $(v_{ox}, v_{oy})$ | If visible; else 0 | Pursuer: $/ 1.75$; Evader: $/ 1.5$ |
| 8 | Visibility flag $I$ | 1.0 if visible, −1.0 otherwise | Already ±1 |
| 9 | Timestep ratio $t_{\text{norm}}$ | step / max_steps | $[0, 1]$ |
| 10 | Distance $d$ | Euclidean distance to opponent | $/ (L\sqrt{2})$ |

**Frame stacking:** Observations are stacked over `frame_stack = 2` steps, giving 22 dimensions. When the opponent is not visible, position/velocity/distance are zero and $I = -1.0$; the agent must rely on temporal context or belief (BiMDN, M7). Frame stacking provides short-term temporal context but cannot integrate information over long occlusions; see M7 for belief-state methods.

> **Note on observation dimensions.** The G&I paper (and our early reading plan) references a 43-dimensional observation space that includes Lidar rays. Our reproduction uses an 11-dimensional state-based observation (22 stacked) as described above. Lidar was part of the original environment specification but was later removed in favor of the compact state vector, which is sufficient for the MADDPG+BiMDN approach and reduces input dimensionality.

### Historical Context: Our Sensing Journey

| Sensing Mode | Sessions | Result |
|-------------|----------|--------|
| Full observability | S11–S48 | Evader collapses; no hiding possible |
| Radius-only | S50–S51 | Evader sees through obstacles; unrealistic |
| LOS-only | S59 | Too restrictive; agents lose each other |
| **Combined radius + LOS** | **S63–S64** | **Viable; evader adopts "hide and don't move"** |
| FOV wedge (G&I style) | S76–S81 | Directional sensing creates geometry-based play |

The G&I FOV design creates a **geometry-dependent information game**: the pursuer must manage orientation to maintain line of sight, while the evader can exploit blind spots by approaching from behind.

---

## 3.4 Capture Condition and Agent Radius

**Capture radius:** $c = 0.15$ m per agent. Capture occurs when:

$$\|\mathbf{x}_p - \mathbf{x}_e\| \leq 2c = 0.30 \text{ m}$$

**Why 0.15 m?** The 0.30 m effective capture distance is roughly 1/3 of the pursuer's minimum turning radius (0.85 m) — the pursuer cannot simply swerve into the evader; it must align its trajectory. Agents initialize with minimum separation 3.0 m to prevent immediate capture.

---

## 3.5 Reward Structure

The reward function is **zero-sum** and combines three terms:

$$r_p = \lambda_{\text{capture}} \cdot \mathbb{1}[\text{capture}] - \lambda_{\text{time}} - \lambda_{\text{distance}} \cdot d$$

$$r_e = -r_p$$

**Terminal rewards:**
- Capture: $r_p = +1000$, $r_e = -1000$
- Timeout (no capture): $r_p = -1000$, $r_e = +1000$

**Per-step (dense) reward:**
- Time penalty: $-\lambda_{\text{time}} = -1$ per step
- Distance penalty: $-\lambda_{\text{distance}} \cdot d = -d$ (distance in meters)

### Why This Structure?

The **terminal reward** ($\pm 1000$) provides a strong learning signal for capture/escape. The **time penalty** ($-1$ per step) discourages stalling — without it, the evader's optimal strategy is to run away indefinitely. The **distance penalty** ($-d$) provides dense gradient toward the opponent, crucial early in training when capture is rare.

**Scale balance:** The raw distance $d$ can reach $16\sqrt{2} \approx 22.6$ m, so per-step penalties range 0 to −22.6. Over 400 steps, cumulative distance penalty is at most −9040. The terminal ±1000 is roughly 10% of total episode magnitude — early training is dominated by dense penalties (approach/avoid), late training by terminal rewards (capture/escape).

### Historical Context: Reward Experiments

| Variant | Sessions | Result |
|---------|----------|--------|
| ±50 terminal | S48 | Insufficient signal; agents ignore capture |
| ±1000 terminal (G&I) | S92–S96 | Strong signal; capture clearly prioritized |
| PBRS (obstacle approach) | S44–S45 | Did not prevent collapse |
| Search staleness reward | S71 | Taught patrol instead of hunt |
| Visibility reward (w=0.1) | S83 | Only modification sustaining non-zero capture under PO (D20) |

**Key lesson (S83, D20):** Under 90° FOV, the pursuer gets no gradient without explicit reward for maintaining line of sight. The G&I environment's dense distance reward provides sufficient signal without visibility shaping.

---

## 3.6 Curriculum: Progressive Difficulty

The G&I environment uses a **two-parameter curriculum** that anneals difficulty over the first 2000 episodes (10% of training):

### FOV Annealing

$$\text{FOV}(e) = \text{FOV}_{\text{start}} \cdot (1 - p) + \text{FOV}_{\text{end}} \cdot p$$

where $p = \min(1.0, e / 2000)$, $\text{FOV}_{\text{start}} = 360°$, $\text{FOV}_{\text{end}} = 120°$.

**Rationale:** Starting with full observability lets agents learn basic behaviors before the sensing challenge. With 120° FOV from the start, the pursuer rarely sees the evader and receives almost no learning signal.

### Evader Speed Annealing

$$v_{\max}^{(e)}(e) = v_{\text{start}} \cdot (1 - p) + v_{\text{end}} \cdot p$$

where $v_{\text{start}} = 0.0$ m/s, $v_{\text{end}} = 1.5$ m/s.

**Rationale:** A stationary evader makes capture trivial, bootstrapping the pursuer with strong positive signal before the game becomes competitive. Starting from $v_{\text{start}} = 0.0$ (as opposed to, say, 0.5 m/s) maximizes this effect: the evader is completely immobile, so the pursuer can practice approach and capture with zero risk of escape. Any non-zero starting speed would introduce early evasion behavior before the pursuer has learned basic pursuit skills, degrading the bootstrapping signal. The annealing schedule ensures the pursuer has already acquired fundamental capture skills before the evader gains enough speed to require genuine evasion strategies.

### Curriculum Duration: 2000 Episodes

The paper specifies $T/10 = 2000$ episodes. Our initial implementation used $T/3 = 6000$ (S93) — a critical bug giving the pursuer 3× more full-FOV training time, contributing to dominance. Fixing to 2000 was one of the two critical fixes in S93 (see D42, D43).

### Historical Context: Our Curriculum Experiments

| Approach | Sessions | Result |
|----------|----------|--------|
| No curriculum | S48 | Both agents stall; no bootstrapping |
| Long curriculum (6000 ep) | S93 | Pursuer overtrained; dominance |
| Short curriculum (2000 ep) | S95 | Matches paper; balanced bootstrapping |
| Masking curriculum (full → partial obs) | S75 | +10–15 pp improvement before restart killed it |

---

## 3.7 Action Space and Control

Both agents output **continuous actions** in $[-1, 1]$, which are scaled to physical bounds:

### Pursuer Action: [Steering Rate, Acceleration]

$$u_{\dot{\delta}} = a_0 \cdot 3.2 \text{ rad/s}, \quad u_a = a_1 \cdot 2.0 \text{ m/s}^2$$

The pursuer controls **steering rate** and **acceleration** — heading and velocity emerge from bicycle dynamics. This indirection is realistic for car-like vehicles but makes control harder.

### Evader Action: [X Acceleration, Y Acceleration]

$$u_x = a_0 \cdot 9.81 \text{ m/s}^2, \quad u_y = a_1 \cdot 9.81 \text{ m/s}^2$$

The evader controls **acceleration in each axis independently**, directly determining velocity change. Any acceleration vector in $[-9.81, 9.81]^2$ maps to a velocity disk $\|\mathbf{v}\| \leq 1.5$ after clipping.

### Exploration Noise

During training, actions are perturbed with Gaussian noise: $a_{\text{noisy}} = a_{\text{policy}} + 0.1 \cdot \mathcal{N}(0, 1)$. This prevents premature convergence to deterministic behavior.

---

## 3.8 Vectorized Environments

With the environment fully specified, we now turn to how it is executed efficiently during training. The paper specifies **8 parallel environment instances**, and S94 implemented vectorized environments to match this. However, the committed configs (`config.yaml`, `config_vanilla.yaml`, `config_belief_noamp.yaml`) currently use `parallel_envs: 2` for local testing. On niro-2, training runs use `parallel_envs: 8` for full throughput.

With 8 envs and batch size 1024 (the paper specifies 512; we use 1024 for faster convergence), training completes in ~6–8 hours for 20K episodes on an RTX 5090. Each environment maintains independent frame stacking buffers and BeliefStateManager history to prevent leakage across episodes.

> **Note:** Batch size 1024 is a deviation from the paper's specified 512. This was chosen for faster wall-clock training and does not change the environment semantics.

---

## 3.9 Summary: The Complete Environment

| Component | Parameter | Value | Rationale |
|-----------|-----------|-------|-----------|
| **Arena** | Size | 16×16 m | Balanced: non-trivial strategies, finite episodes |
| | Episode length | 400 steps (40 s) | 6× minimum capture time; forces active play |
| | Timestep | 0.1 s | 10 Hz control |
| **Pursuer** | Dynamics | Bicycle model | Car-like constraints; $R_{\min} \approx 0.85$ m |
| | Wheelbase $L$ | 0.30 m | Critical: 0.15 m bug caused CR=1.0 (S93) |
| | $v_{\max}$ / $v_{\min}$ | 2.5 / −1.0 m/s | Faster but limited reverse |
| | Sensing | 120° FOV, 6 m | Directional; creates information game |
| **Evader** | Dynamics | Point mass | Omnidirectional; instant turns |
| | $v_{\max}$ / $v_{\min}$ | 1.5 / −1.5 m/s | Slower but full reverse |
| | Sensing | 360°, 1000 m | Omnidirectional; information advantage |
| **Capture** | Effective distance | 0.30 m ($2c$) | Small relative to arena; rewards precision |
| **Reward** | Terminal / time / dist | ±1000 / −1 / −$d$ | Strong terminal signal + dense gradient |
| **Curriculum** | FOV / speed / duration | 360°→120° / 0→1.5 / 2000 ep | Bootstraps pursuer before adding challenge (see `config_belief_noamp.yaml`; `config.yaml` and `config_vanilla.yaml` still have `t_end_ratio: 0.3` for legacy reasons) |
| **Observation** | Raw / stacked | 11 / 22 | Frame stack=2 for temporal context |
| **Action** | Pursuer / evader | [steer, accel] / [$a_x$, $a_y$] | Continuous, scaled to physical bounds |

---

## Self-Check

1. **Why does the G&I environment give the evader omnidirectional sensing (360°) while the pursuer has only a 120° wedge?** What would happen if both had 120° FOV? What if both had omnidirectional sensing?

2. **The bicycle model has a minimum turning radius of approximately 0.85 m. How does this constrain the pursuer's strategy compared to a point-mass model?** Give a concrete example of a maneuver the evader can execute that the pursuer cannot.

3. **Suppose the curriculum annealed over only 500 episodes instead of 2000. How would this affect the pursuer's ability to learn basic capture skills before the evader gains full speed? What failure mode would you expect?**

4. **The reward function combines a terminal bonus of ±1000 with per-step penalties of −1 and −$d$. If you were to increase $\lambda_{\text{capture}}$ to 10,000 while keeping the per-step penalties the same, would the pursuer become more risk-seeking or risk-averse? Would the evader learn to stall less or more? What about decreasing $\lambda_{\text{capture}}$ to 100?**

---

## Papers to Read

- **(ESSENTIAL)** Gonultas & Isler 2024, Section II: Complete environment description, including dynamics, sensing, reward, and curriculum. This is the primary reference for every parameter in this module.
- **(REFERENCE)** Bilgin & Karimpanal (2024): Car-like PE with sensor constraints. Compare their 20×20 m arena and sensing model to G&I's 16×16 m.
- **(REFERENCE)** S54 worklog (PE design choices survey): Comprehensive survey of arena sizes, speed ratios, episode lengths, and sensing modes across 15+ PE-RL papers. Shows G&I's choices are representative but not universal.
- **(OPTIONAL)** PettingZoo Simple Tag documentation: A simpler multi-agent environment with collision-based capture. Useful for understanding how capture radius and arena size interact in discrete-action settings.

---

## Deep Dive Sessions

- **S54** (PE design choices survey): Comprehensive survey of arena sizes, speed ratios, episode lengths, and sensing modes across 15+ papers. Essential for understanding why G&I chose 16×16 m, 400 steps, and asymmetric sensing.
- **S59** (variable speed + partial observability): The pivotal session where we introduced LOS-based partial observability and discovered that information asymmetry alone creates viable game dynamics.
- **S76–S81** (FOV variants): Extensive experiments with FOV angles (90°, 120°, 180°) and ranges (6 m, 8 m). FOV 90°/8 m was the best config in our Phase 1–3 framework, but G&I's 120°/6 m is well-calibrated for the 16×16 m arena.
- **S92** (full G&I implementation): The session where the environment was first implemented. Every parameter was cross-referenced against the paper.
- **S93** (NaN diagnosis + five fixes): Critical fixes including wheelbase (0.15 → 0.30) and curriculum duration (6000 → 2000). Understanding these bugs is essential for appreciating how sensitive the environment is to parameter choices.
- **S94** (vectorized environments): Implementation of 8 parallel envs with per-env frame stacking and belief managers. Essential for understanding training throughput.
- **S95** (five critical fixes): Actor observation bug, BiMDN warm-up, BiMDN architecture, tournament evaluation, and seed-based checkpoints. The first fix (actor observation) was the most critical — it explained why all prior runs showed CR = 1.0.
