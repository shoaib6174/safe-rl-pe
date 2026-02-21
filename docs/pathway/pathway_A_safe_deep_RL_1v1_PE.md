# Pathway A: Safe Deep RL for 1v1 Ground Robot Pursuit-Evasion

## Self-Contained Implementation Report
**Date**: 2026-02-21
**Status**: Research plan — ready for implementation
**Based on**: 51 papers reviewed — 36 original (see `final_literature_review.md`) + 15 new papers from validation review (see `papers/safe_rl/new_papers.md`)
**Validated**: 2026-02-21 — 3 critical fixes applied, 5 improvements integrated (see `papers/safe_rl/pathway_A_validation_report.md`)

---

## 1. Research Goal & Novelty Statement

### 1.1 Goal
Design, implement, and deploy a **safety-guaranteed deep reinforcement learning system** for 1v1 pursuit-evasion games on physical ground mobile robots with nonholonomic dynamics.

### 1.2 Novelty (Gaps Filled)
No existing work combines ALL of the following in a single system:
- **Safe DRL** with hard constraint guarantees during training AND deployment
- **1v1 pursuit-evasion** game formulation (adversarial, zero-sum)
- **Nonholonomic ground robot dynamics** (differential-drive or car-like)
- **Partial observability** (realistic sensor constraints)
- **Self-play training** with Nash equilibrium convergence
- **Real-world deployment** on physical robots

| Gap ID | Description | How We Fill It |
|--------|-------------|---------------|
| G1 | No CBF-safe DRL + 1v1 PE + ground robot + real deployment | Core contribution |
| G2 | Safety filters not designed for adversarial self-play training | CBF-constrained policies in self-play |
| G4 | Most safe PE uses simplified dynamics | Differential-drive/Ackermann kinematics |

### 1.3 Closest Existing Works
| Paper | What They Have | What They Lack |
|-------|---------------|----------------|
| [02] Gonultas & Isler | 1v1 PE, car-like robots, real deployment | No safety guarantees |
| [22] MADR | Game-optimal PE on TurtleBots | Not RL (PINN+MPC), no safety filter |
| [13] Kokolakis CDC | Safe PE with barrier functions | Simulation only, simple dynamics, not deep RL |
| [16] Suttle AISTATS | CBF-constrained safe RL with convergence | Single-agent, no PE game |
| [18] AMS-DRL | Self-play with NE convergence, real drones | No safety constraints, drones not ground robots |
| [N15] RMARL-CBF-SAM (Liu 2025) | Robust MARL + neural CBFs + safety attention + reward shaping | Navigation only (not PE), double integrator (not unicycle), no self-play, full obs, no sim-to-real |
| [N06] TD3 Self-Play PE (Selvam 2024) | 1v1 PE with unicycle dynamics, simultaneous self-play | No safety constraints, full observability only |

**Note on closest prior work**: [N15] (RMARL-CBF-SAM, Liu et al., Information Sciences 2025) is the closest existing work, combining robust MARL with neural CBFs, safety attention mechanisms, and reward shaping for multi-agent navigation. However, it differs from our approach in five critical ways: (1) navigation task, not pursuit-evasion game; (2) double integrator dynamics, not nonholonomic unicycle; (3) no self-play or game-theoretic training; (4) full state observation, not lidar-based partial observability; (5) no sim-to-real transfer. N15 validates several of our architectural choices (neural CBFs, safety reward shaping, attention mechanisms) while confirming that the PE + nonholonomic + partial obs + sim-to-real combination remains **novel**.

---

## 2. Problem Formulation

### 2.1 Game Setup
- **Players**: 1 pursuer (P), 1 evader (E)
- **Game type**: Two-player zero-sum differential game
- **Objective**: Pursuer minimizes time to capture; Evader maximizes it
- **Terminal condition**: Capture when `||p_P - p_E|| ≤ r_capture` OR timeout at `T_max`
- **Arena**: Bounded 2D workspace with static obstacles

### 2.2 Robot Dynamics

**Option A: Differential-Drive (Unicycle) — Recommended for first implementation**
```
ẋ = v · cos(θ)
ẏ = v · sin(θ)
θ̇ = ω

State: s = [x, y, θ]  (3D per robot, 6D joint state in relative coords)
Control: u = [v, ω]    (linear velocity, angular velocity)
Bounds: v ∈ [0, v_max], ω ∈ [-ω_max, ω_max]
```
- Reference: Paper [34] SHADOW uses this exact model (unicycle dynamics)
- Platforms: TurtleBot3/4, custom differential-drive

**Option B: Car-like (Ackermann) — Higher fidelity**
```
ẋ = v · cos(θ)
ẏ = v · sin(θ)
θ̇ = (v / L) · tan(δ)

State: s = [x, y, θ]
Control: u = [v, δ]    (velocity, steering angle)
Bounds: v ∈ [0, v_max], δ ∈ [-δ_max, δ_max]
```
- Reference: Paper [02] uses Ackermann kinematics on F1TENTH
- Minimum turning radius constraint: `R_min = L / tan(δ_max)`

### 2.3 Relative Coordinate Formulation
Following Paper [13] (Kokolakis), work in **relative coordinates** to reduce dimensionality:
```
x_rel = x_P - x_E
y_rel = y_P - y_E
θ_rel = θ_P - θ_E

Relative dynamics:
ẋ_rel = v_P·cos(θ_P) - v_E·cos(θ_E)
ẏ_rel = v_P·sin(θ_P) - v_E·sin(θ_E)
θ̇_rel = ω_P - ω_E
```
This is the standard "Air3D" formulation used in DeepReach [21] and MADR [22].

### 2.4 Observation Space
Following Paper [02] (Gonultas), use **partial observations with limited FOV**:

**Pursuer observation** (per timestep):
```
o_P = [
    d_to_evader,        # distance (if in FOV, else -1)
    bearing_to_evader,  # angle (if in FOV, else 0)
    own_velocity,       # [v, ω]
    lidar_scan,         # N-ray lidar (obstacles + arena boundary)
    belief_state         # BiMDN-encoded belief about evader position
]
```

**Evader observation** (symmetric):
```
o_E = [same structure with roles reversed]
```

**FOV constraint**: Agent can only detect opponent within a cone of half-angle `α_fov` (e.g., 60°-90°) and range `r_detect`.

### 2.5 Reward Design

**Pursuer reward** (per step):
```python
r_P = w1 * (d_prev - d_curr) / d_max           # distance reduction (normalized)
      + w2 * capture_bonus * I(captured)         # terminal capture reward
      + w3 * timeout_penalty * I(timeout)        # terminal timeout penalty
      + w4 * visibility_bonus * I(evader_in_fov) # maintain visual contact
      + w5 * min(h_i(x)) / h_max               # CBF margin bonus (NEW)
```

**Evader reward**: `r_E = -r_P` (zero-sum)

**Recommended weights** (from Papers [02], [05], [12], [18]):
- `w1 = 1.0` (distance shaping)
- `w2 = +100.0` (capture bonus)
- `w3 = -50.0` (timeout penalty for pursuer)
- `w4 = 0.1` (visibility incentive)
- `w5 = 0.05` (CBF margin bonus — incentivize staying away from constraint boundaries)
- `h_max` = maximum possible CBF value across all constraints (used for normalization). For circular arena: `h_max = R_arena²`; for obstacle avoidance: `h_max = max_i(||q_init - p_obs_i||² - χ_i²)` at the farthest initial position. In practice, compute once at environment init as the max h value over a grid of states.

**Safety-reward shaping rationale** (from Paper [05], Yang et al. 2025): The CBF margin bonus `w5 * min(h_i(x)) / h_max` encourages the policy to maintain distance from safety constraint boundaries. This reduces CBF filter intervention rate over training, as the agent learns inherently safe behavior rather than relying on the safety layer to correct unsafe actions. The weight `w5` is deliberately small to avoid dominating the task reward. N15 (RMARL-CBF-SAM) independently validates this approach, using a similar safety reward shaping term for multi-agent navigation.

**Zero-sum asymmetry note for w5**: Since `r_E = -r_P`, the evader receives `-w5 * min(h_i(x)) / h_max` — effectively incentivizing the evader to push the *pursuer* toward unsafe regions (low CBF margin). The evader's *own* safety is NOT incentivized by the reward signal; it is enforced entirely by CBF-Beta (training) and RCBF-QP (deployment). This is an intentional design choice: the CBF safety layer provides hard guarantees regardless of the reward, and the adversarial incentive to exploit the opponent's safety margins creates richer game-theoretic behavior (the evader learns to use boundaries strategically). If this proves problematic (evader overly aggressive near boundaries), consider making the safety bonus **non-zero-sum**: add `w5 * min(h_i_own(x)) / h_max` independently to *both* agents' rewards outside the zero-sum structure.

**Adaptive reward** (from Paper [15]): Increase `w1` in early training for exploration, shift weight to `w2` later for terminal performance.

---

## 3. System Architecture

### 3.1 High-Level Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING (Simulation)                  │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │ Pursuer  │    │  PE Env  │    │  Evader  │           │
│  │  Agent   │◄──►│ (Gym)    │◄──►│  Agent   │           │
│  └────┬─────┘    └──────────┘    └────┬─────┘           │
│       │                               │                  │
│  ┌────▼─────┐                    ┌────▼─────┐           │
│  │ Obs →    │                    │ Obs →    │           │
│  │ BiMDN →  │                    │ BiMDN →  │           │
│  │ PPO →    │                    │ PPO →    │           │
│  │ CBF-Beta │                    │ CBF-Beta │           │
│  │ → Action │                    │ → Action │           │
│  └──────────┘                    └──────────┘           │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │         AMS-DRL Self-Play Controller          │       │
│  │  S0: cold-start → S1: train P → S2: train E  │       │
│  │  → ... → Sk: converge when |SR_P - SR_E| < η │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  DEPLOYMENT (Real Robot)                  │
│                                                          │
│  Sensors → Obs Encoder → ONNX Policy → CBF-QP → Motors  │
│             (BiMDN)        (PPO)      (RCBF+GP)          │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Per-Agent Neural Network Architecture

```
Input: observation o_t
    │
    ├─── Lidar branch: Conv1D(N, 32) → Conv1D(32, 64) → Flatten
    │
    ├─── State branch: MLP(state_dim, 64, 64)
    │
    ├─── Belief branch: BiMDN(obs_history) → latent z (32-dim)
    │      [LSTM or GRU encoder over past K observations]
    │      [Mixture Density Network for position belief]
    │
    └─── Concatenate → MLP(160+32, 256, 256)
              │
              ├─── Actor head → Beta distribution parameters (α, β) per action dim
              │     [CBF-constrained: support rescaled to safe set C(x)]
              │
              └─── Critic head → V(o_t)  [state value for PPO]
```

**Key design choices**:
- **Beta distribution** (not Gaussian) for policy: naturally bounded support, compatible with CBF truncation [16]
- **BiMDN** for belief encoding: handles partial observability [02]
- **Separate lidar branch**: extract spatial features for obstacle-aware control
- **Network size**: ~256 hidden units, 2 layers (sufficient per Papers [18], [22])

**Design note — Mixture of Beta distributions**: Standard Beta is unimodal, which may be limiting for PE where bimodal actions are useful (e.g., "go left OR right around an obstacle"). Consider upgrading to a **mixture of 2-3 Beta distributions** with learned mixing weights:
```
π(u|x) = Σ_k w_k(x) · Beta(u; α_k(x), β_k(x))    k = 1..K (K=2 or 3)
```
This preserves bounded support (compatible with CBF truncation) while allowing multimodality. Start with standard Beta (simpler); upgrade to mixture if policies appear unable to represent bifurcating strategies during Phase 3 self-play. The mixing weights w_k are output by the actor head alongside α_k, β_k.

### 3.3 CBF Safety Layer

#### 3.3.1 Safety Constraints — Virtual Control Point CBF Formulation

**CRITICAL**: For nonholonomic (unicycle/car-like) robots, position-based CBFs have **mixed relative degree** — the angular velocity ω does not appear in the Lie derivative ḣ, making the CBF-QP ill-posed. This is because for h(x,y), the chain rule gives ḣ = ∂h/∂x · v·cos(θ) + ∂h/∂y · v·sin(θ), which depends on v and θ but NOT on ω. Prior methods using HOCBFs are computationally intensive and overly conservative — they prioritize braking over steering [N12, Remark 4].

**Solution**: Use **Virtual Control Point (VCP) CBFs** [N12, Zhang & Yang, Neurocomputing 2025]. Place a virtual point at distance d ahead of the robot:

```
# Virtual control point (VCP):
q = [x + d·cos(θ), y + d·sin(θ)]     # d ≈ 0.05m (small but nonzero)

# VCP time derivative (depends on BOTH v and ω — uniform relative degree 1):
q̇ = [v·cos(θ) - d·ω·sin(θ),  v·sin(θ) + d·ω·cos(θ)]

# This achieves ḣ = ∂h/∂q · q̇, which is affine in u = [v, ω]
# The M-matrix transformation prioritizes STEERING over BRAKING [N12, Remark 4]
```

Define safety via VCP-based Control Barrier Functions:

**1. Arena boundary constraint**:
```
h_arena(x) = R_arena² - ||q||²  ≥ 0     [circular arena, q = VCP position]
-- OR --
h_arena(x) = min(q_x - x_min, x_max - q_x, q_y - y_min, y_max - q_y)  ≥ 0  [rectangular]
```

**2. Obstacle avoidance** (for each obstacle i):
```
h_obs_i(x) = ||q - p_obs_i||² - χ_i²  ≥ 0
where χ_i ≥ r_robot + r_obs_i + margin + d    [safety radius increases by d, which is negligible]
```

**3. Inter-robot collision avoidance** (minimum separation):
```
h_collision(x) = ||q_P - q_E||² - r_min²  ≥ 0
where q_P, q_E are VCP positions of pursuer and evader respectively
```
This prevents physical collision even during capture (capture radius > collision radius).

**4. Velocity/acceleration limits** (implicit in action bounds).

**VCP parameter selection**: d should be small relative to the robot wheelbase (N12 uses d = 0.05m for wheelbase l = 0.3m). Larger d gives more aggressive steering but reduces the effective workspace by d. For TurtleBot3 (wheelbase ~0.16m), use d = 0.03-0.05m.

#### 3.3.2 CBF-Constrained Beta Policy (Training) — from Paper [16]
During training, the policy samples ONLY safe actions:

```python
# Safe control set from VCP-CBF (per state x):
C(x) = {u : ḣ(x,u) + α·h(x) ≥ 0, for all h_i}

# With VCP formulation (Section 3.3.1), ḣ is affine in u = [v, ω]:
# ḣ = ∂h/∂q · q̇ = ∂h/∂q · [v·cos(θ)-d·ω·sin(θ), v·sin(θ)+d·ω·cos(θ)]
# This gives linear constraints on BOTH v and ω (uniform relative degree 1)

# Compute safe bounds for each action dimension:
v_safe_min, v_safe_max = compute_cbf_bounds_v(x, h_list)
ω_safe_min, ω_safe_max = compute_cbf_bounds_ω(x, h_list)

# Rescale Beta distribution support to [safe_min, safe_max]:
v ~ Beta(α_v, β_v) rescaled to [v_safe_min, v_safe_max]
ω ~ Beta(α_ω, β_ω) rescaled to [ω_safe_min, ω_safe_max]
```

**Convergence guarantee** (Theorem 1 from Paper [16]): Under standard regularity conditions, the truncated policy gradient converges to stationary points of the constrained objective.

#### 3.3.3 RCBF-QP Safety Filter (Deployment) — from Paper [06]
At deployment, add a robust safety layer to handle model uncertainty:

```python
# Nominal dynamics + GP-estimated disturbance:
# ẋ = f(x) + g(x)·u + d(x)
# d_hat(x), σ_d(x) = GP.predict(x)

# Robust CBF constraint:
# L_f h + L_g h · u + L_d h · d_hat ≥ -α·h + κ·σ_d  (robust margin)

# QP: minimize ||u - u_rl||² s.t. RCBF constraints
u_safe = solve_qp(u_rl, x, h_list, gp_model)
```

**GP disturbance estimation** (Paper [06]):
- Online learning: observe (x_t, u_t, x_{t+1}), compute residual `d_t = x_{t+1} - f(x_t) - g(x_t)·u_t`
- GP kernel: Squared Exponential (SE) with automatic relevance determination
- Update GP every N steps (e.g., N=10)
- This handles wheel slip, uneven terrain, actuator delay

**GP Cold-Start Protocol (for real robot deployment)**:
The GP disturbance model starts empty on a new robot. Without data, the RCBF-QP has no disturbance estimate and may be either too conservative (large σ_d) or too aggressive (zero d_hat). Use this three-step initialization:
```
Step 1: Pre-fill GP with simulation residual data (transfer learning)
    - Run nominal policy in simulation with domain randomization
    - Collect (x, u, d_residual) tuples
    - Initialize GP with this dataset (~500-1000 points)
    - This gives a reasonable prior before any real-world data

Step 2: Conservative safety margin for first 100 real-world steps
    - Use κ_init = 2.0 * κ_nominal (double the robust margin)
    - This accounts for GP uncertainty being high with few real data points
    - Accept some performance degradation for safety

Step 3: Transition to normal margin as GP uncertainty decreases
    - Monitor mean GP posterior variance σ̄_d
    - When σ̄_d < threshold (e.g., 0.01), reduce κ to κ_nominal
    - Can use exponential decay: κ(t) = κ_nominal + (κ_init - κ_nominal) * exp(-t/τ)
```

#### 3.3.4 CBF-QP Infeasibility Handling (Three-Tier Strategy)

**CRITICAL**: In adversarial PE, the opponent actively creates tight situations. CBF-QP infeasibility will happen regularly, not occasionally. The previous mitigation ("reduce safety margin") is insufficient.

**Tier 1 — Learned Feasibility Constraints (during training)** [N13, Xiao et al. ECC 2023]:
```python
# Learn the feasibility boundary of the CBF-QP itself:
# Train classifiers H_j(z) ≥ 0 that approximate the feasible region
# Add to RL training so the policy proactively avoids infeasible states

# Feedback training loop (iterative):
for iteration in range(3):  # typically converges in 3 iterations
    # 1. Train RL policy with current feasibility constraints
    policy = train_ppo(env, feasibility_constraints=classifiers)

    # 2. Collect infeasible states encountered during training
    infeasible_states = collect_infeasible_states(policy, env)

    # 3. Retrain classifiers on augmented dataset
    classifiers = retrain_svm_or_dnn(infeasible_states, feasible_states)

# Result: infeasibility rate drops from 8.11% → 0.21% [N13]
# N13 handles both regular (circular) and irregular (rectangular) unsafe sets
```

**Tier 2 — Hierarchical Relaxation (at deployment)**:
```python
# Priority 1: Try exact CBF-QP
u_safe = solve_cbf_qp(u_rl, x, h_list)

if infeasible:
    # Priority 2: Relax least-important constraint first
    # Priority ordering: h_collision > h_arena > h_obs
    u_safe = solve_relaxed_cbf_qp(u_rl, x, h_list,
                                    priority=[h_collision, h_arena, h_obs])
```

**Tier 3 — Backup Controller (last resort)**:
```python
if still_infeasible:
    # Priority 3: Use backup controller (pure safety, ignore task)
    u_safe = backup_controller(x)  # brake + turn away from nearest obstacle
```

**Monitoring**: Log all infeasibility events. Report CBF feasibility rate and backup controller activation rate as evaluation metrics.

**Implementation note**: Tier 1 operates during training (Phase 2), complementary to Tiers 2-3 which operate at deployment. The N13 approach teaches the policy to proactively avoid infeasible regions rather than relying on runtime relaxation.

---

## 4. Training Pipeline

### 4.1 AMS-DRL Self-Play Protocol (from Paper [18])

```
Phase S0 (Cold-Start):
    - Train evader for basic navigation (reach random goals, avoid obstacles)
    - No adversary present
    - Purpose: learn basic locomotion and obstacle avoidance
    - Duration: ~500-1000 episodes

Phase S1 (Pursuer Training):
    - Freeze evader policy (from S0)
    - Train pursuer to capture frozen evader
    - Purpose: learn basic pursuit
    - Duration: ~1000-2000 episodes

Phase S2 (Evader Training):
    - Freeze pursuer policy (from S1)
    - Train evader to evade frozen pursuer
    - Purpose: learn basic evasion against competent pursuer
    - Duration: ~1000-2000 episodes

Phase Sk (Alternating):
    - Continue alternating S1/S2 until convergence
    - Convergence criterion: |success_rate_P - success_rate_E| < η (e.g., η = 0.10)
    - Typically converges in 4-6 phases [18]

NE Verification:
    - Record success rates at each phase
    - Nash equilibrium when neither player can unilaterally improve
    - Paper [18] proves convergence for this protocol
```

### 4.2 Curriculum Learning (from Paper [02])

Layer the curriculum on top of self-play:

```
Level 1: Close encounters, no obstacles
    - Initial distance: 2-5m
    - Open arena
    - Purpose: learn basic approach/flee

Level 2: Medium distance, no obstacles
    - Initial distance: 5-15m
    - Open arena
    - Purpose: learn long-range pursuit/evasion strategies

Level 3: Close encounters with obstacles
    - Initial distance: 2-5m
    - 2-4 static obstacles
    - Purpose: learn to use/avoid obstacles

Level 4: Full scenario
    - Random initial distance and positions
    - Random obstacles (number, size, position)
    - FOV constraints active
    - Purpose: generalize to arbitrary scenarios
```

**Advancement criterion**: Advance to next level when success rate > 70% for both players.

### 4.3 Training Hyperparameters

Based on Papers [02], [16], [18]:

| Parameter | Value | Source |
|-----------|-------|--------|
| Algorithm | PPO (Proximal Policy Optimization) | [18] |
| Learning rate | 3e-4 | Standard PPO |
| Gamma (discount) | 0.99 | [18] |
| GAE lambda | 0.95 | Standard PPO |
| Clip ratio | 0.2 | Standard PPO |
| Entropy coefficient | 0.01 → 0.001 (annealed) | [02] |
| Mini-batch size | 256 | [18] |
| Update epochs per rollout | 10 | Standard PPO |
| Rollout length | 512 steps | - |
| Parallel environments | 16-64 | [07] |
| Total timesteps per phase | 1-2M | [18] |
| Network hidden units | 256 | [22], [34] |
| Network layers | 2 | [18] |
| CBF class-K function α | 1.0 | [05] |
| CBF safety margin | 0.1m | - |
| Belief history length K | 10 timesteps | [02] |
| BiMDN mixture components | 5 | [02] |

### 4.4 Environment Specification

**Simulation environment** (Gymnasium-compatible):

```python
class PursuitEvasionEnv(gym.Env):
    """
    1v1 PE with unicycle dynamics, obstacles, partial observability.
    """
    # Arena
    arena_size = 20.0  # meters (side length or radius)

    # Robot parameters (same for both)
    v_max = 1.0        # m/s (TurtleBot3 max ~0.22, F1TENTH ~3.0)
    omega_max = 2.84    # rad/s (TurtleBot3)
    r_robot = 0.15      # meters (robot radius)

    # Game parameters
    r_capture = 0.5     # meters (capture radius)
    r_collision = 0.3   # meters (physical collision radius)
    T_max = 60.0        # seconds (episode timeout)
    dt = 0.05           # seconds (20 Hz control)

    # Sensor parameters
    fov_angle = 120     # degrees (each side from forward)
    fov_range = 10.0    # meters (detection range)
    lidar_rays = 36     # number of lidar rays
    lidar_range = 5.0   # meters

    # Obstacles
    n_obstacles = 0-6   # randomized per episode
    obstacle_radius = 0.3-1.0  # meters (randomized)
```

### 4.5 Domain Randomization (from Papers [07], [N10], [N11])

For sim-to-real transfer, randomize during training:

| Parameter | Nominal | Randomization Range |
|-----------|---------|-------------------|
| Robot mass | 1.0 kg | ±20% |
| Wheel friction | 0.5 | [0.3, 0.8] |
| Motor gain | 1.0 | [0.8, 1.2] |
| Sensor noise (lidar) | 0.0 | N(0, 0.02m) |
| Sensor noise (odom) | 0.0 | N(0, 0.01m), N(0, 0.5°) |
| Control delay | 0 ms | [0, 50] ms |
| Obstacle positions | fixed | ±0.5m per episode |
| Arena size | 20m | [18, 22]m |
| v_max | 1.0 | [0.85, 1.15] m/s |

---

## 5. Evaluation Plan

### 5.1 Metrics

**Primary metrics:**
| Metric | Definition | Target |
|--------|-----------|--------|
| Capture rate (P) | % episodes where pursuer captures evader | Track convergence |
| Escape rate (E) | % episodes where evader survives to timeout | Track convergence |
| NE gap | \|capture_rate - 0.5\| | < 0.10 at convergence |
| Safety violation rate | % episodes with CBF constraint violation | **0% (hard requirement)** |
| Mean capture time | Average time to capture (when captured) | Lower = better P policy |
| Mean min-distance (E) | Evader's minimum distance to pursuer | Higher = better E policy |
| CBF-QP feasibility rate | % timesteps where CBF-QP has a feasible solution | >99% |
| CBF margin distribution | Histogram of min h_i(x) values across timesteps | Positive skew (far from boundary) |
| Train-deploy safety gap | Performance difference CBF-Beta (training) vs RCBF-QP (deployment) | <5% |

**Secondary metrics:**
| Metric | Definition |
|--------|-----------|
| Obstacle collision rate | % episodes with any obstacle/boundary collision |
| CBF filter intervention rate | % of timesteps where CBF modifies the RL action |
| Policy entropy | Measure of exploration/exploitation balance |
| Value function loss | Critic accuracy |
| Exploitability | Reward gap vs best-response opponent (PSRO-style) |
| Strategy diversity | Number of distinct trajectory clusters (k-means on trajectory embeddings) |
| GP disturbance accuracy | RMSE of GP predictions vs actual dynamics residuals |
| QP solve time (95th pctl) | 95th percentile RCBF-QP solve time (must be <50ms for 20Hz) |
| Backup controller activation rate | % timesteps using Tier 3 backup controller (should be ~0%) |

### 5.2 Baselines

| Baseline | Source | Purpose |
|----------|--------|---------|
| DQN vs DDPG (no safety) | Paper [12] | Show safety improvement |
| MADDPG (no safety) | Paper [02] | Show safety improvement over SOTA PE |
| PPO + self-play (no CBF) | Paper [18] adapted | Ablation: value of safety layer |
| PPO + CBF filter (no self-play) | Paper [05] | Ablation: value of adversarial training |
| A-MPC + DQN | Paper [15] | Classical hybrid baseline |
| MADR (HJ-optimal) | Paper [22] | Upper bound (game-theoretic optimal) |
| Random policy | - | Lower bound |
| Greedy heuristic | Proportional navigation | Classical baseline |
| **MACPO** | **Paper [N03]** | **Standard safe MARL baseline — monotonic reward+safety improvement (Theorem 4.4); code: https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation** |
| **MAPPO-Lagrangian** | **Paper [N03]** | **Simpler safe MARL variant (soft constraints via Lagrangian relaxation)** |
| **CPO (single-agent)** | **Achiam et al. 2017** | **Foundational safe RL baseline** |
| **SB-TRPO** | **Paper [N14]** | **Safety-biased trust region (Feb 2026); 10x cheaper than CPO (Monte Carlo returns only, no learned critics); useful for quick prototyping** |

### 5.3 Ablation Studies

1. **CBF-Beta policy vs CBF-QP filter vs No safety**: Compare safety violation rates and learning curves
2. **Self-play (AMS-DRL) vs MADDPG vs Vanilla SP**: Compare NE convergence and exploitability
3. **BiMDN belief vs raw observation vs full state**: Compare performance under partial observability
4. **With/without curriculum**: Compare training stability
5. **With/without domain randomization**: Compare sim-to-real gap
6. **Unicycle vs Ackermann dynamics**: Compare strategy complexity
7. **With/without GP disturbance estimation**: Compare real-world safety
8. **AMS-DRL (alternating) vs Simultaneous self-play vs MADDPG** [N06]: Compare NE convergence speed, final performance, training stability. N06 shows simultaneous TD3 self-play (no freezing, both agents train concurrently) works for 1v1 PE with identical unicycle dynamics. Simpler to implement — if equally effective, simplifies the training pipeline.
9. **With/without safety-reward shaping (w5)**: Compare CBF filter intervention rate, policy conservativeness, and task performance
10. **CBF-Beta → RCBF-QP vs BarrierNet end-to-end**: Compare train-deploy performance gap (Phase 2.5)
11. **With/without CASRL conflict-averse gradient** [30]: If safety-reward shaping (w5) creates gradient conflicts between the pursuit objective and the CBF margin bonus, CASRL separates these into dual critics with independent gradients and uses conflict-averse optimization (max_{g} min_{i} g_i^T · g). Test in Phase 3 if w5 ablation shows pursuit-safety conflict.

---

## 6. Implementation Phases

### Phase 1: Simulation Foundation (Months 1-2)

**Deliverables:**
- [ ] D1: Gymnasium PE environment with unicycle dynamics
- [ ] D2: PPO implementation via **Stable Baselines 3** (locked decision)
- [ ] D3: Basic self-play (vanilla alternating training) with health monitoring
- [ ] D4: Baseline comparison: DQN, DDPG, PPO without safety
- [ ] D5: **Validate virtual control point (VCP) CBF on simple unicycle obstacle avoidance** (MUST DO FIRST — confirms CBF formulation before full integration)
- [ ] D8: Configuration YAML with all tunable parameters externalized
- [ ] D9: Automated test suite (15+ tests for dynamics, env, rewards, training)

**Key decisions:**
- Start with **full-state observation** (no partial observability yet)
- Start with **no obstacles** (open arena)
- Start with **no CBF** (learn basic PE first)
- Use differential-drive (unicycle) dynamics
- Use **custom `SingleAgentPEWrapper`** (Option B) for multi-agent SB3 integration (locked decision)
- Use N10's sim-to-real pipeline (Isaac Lab → ONNX → Gazebo → Real) as Phase 4 architecture reference during initial design
- Define **Phase 2 integration interfaces** as stubs: `SafetyFilter`, `RewardComputer`, `ObservationBuilder`

**Success criteria (all quantified):**
- Pursuer: capture rate > 80% vs random evader (100 episodes, 3 seeds)
- Evader: mean `|omega|` > 0.5 rad/s (non-trivial evasion, automated check)
- Training converges within 2M timesteps (reward improvement < 1% for 5 consecutive evals)
- VCP-CBF: `mean(|delta_omega|) / mean(|delta_v|)` > 2.0 when CBF intervenes
- Self-play: capture rate in [0.35, 0.65] for 2 of last 3 phases
- All 15+ automated tests pass
- See detailed Phase 1 spec: `claudedocs/phases/phase1_simulation_foundation.md`

### Phase 2: Safety Integration (Months 2-3)

**Deliverables:**
- [ ] **VCP-CBF formulation** for arena boundary + obstacles (Section 3.3.1)
- [ ] CBF-constrained Beta policy implementation (using VCP-CBFs)
- [ ] RCBF-QP safety filter implementation with **3-tier infeasibility handling** (Section 3.3.4)
- [ ] **N13 learned feasibility constraints** integrated into RL training loop
- [ ] Safety-constrained self-play training
- [ ] **Safety-reward shaping term (w5)** in reward function
- [ ] Ablation: CBF-Beta vs CBF-QP vs no safety
- [ ] **Feasibility monitoring metrics** (CBF-QP feasibility rate, backup activation rate)

**Key decisions:**
- Use **VCP-CBFs** (not position-based) from [N12] — critical for nonholonomic dynamics
- Implement CBF-Beta from Paper [16] for training safety
- Implement RCBF-QP from Paper [06] for deployment safety with 3-tier infeasibility handling [N13]
- Add static obstacles (2-4)
- Arena boundary as CBF constraint
- Add CBF margin bonus (w5 = 0.05) to reward design

**Fallback — PNCBF (Neural CBF) [N02]**: If hand-crafted VCP-CBFs prove brittle in complex environments (e.g., many obstacles, tight corridors), **PNCBF should be the first alternative attempted, not a last resort**. Key insight from N02: the value function of max-over-time cost V^{h,π} IS itself a valid CBF. This automatically captures reachable set geometry without manual CBF design.
- **Hardware validated**: quadcopters (12D state) with Boston Dynamics Spot as dynamic obstacle
- **Converges in 2-3 policy iterations** — computationally practical
- **Scales to F-16 GCAS** (16D state space)
- Implementation: train a nominal safe policy first, then compute V^{h,π} as the neural CBF
- Decision point: if VCP-CBF feasibility rate <95% after Phase 2 tuning, switch to PNCBF

**Success criteria:**
- **Zero safety violations** during training (CBF-Beta)
- Task performance within 10% of unconstrained baseline
- CBF filter intervention rate decreasing over training (agent learns to be safe)
- **CBF-QP feasibility rate >99%** after N13 learned feasibility training
- **Backup controller activation rate <1%**

### Phase 2.5: BarrierNet Experiment (Month 4) — NEW

**Rationale**: The plan uses CBF-Beta during training (Section 3.3.2) and RCBF-QP during deployment (Section 3.3.3). These are **different safety mechanisms**. BarrierNet [N04] provides a differentiable QP safety layer that can be trained end-to-end, potentially eliminating the training-deployment gap.

**Deliverables:**
- [ ] Implement differentiable QP safety layer (using cvxpylayers or qpth)
- [ ] Train policy end-to-end WITH the deployment safety layer
- [ ] Compare: CBF-Beta → RCBF-QP (Phases 1-2) vs BarrierNet end-to-end
- [ ] Measure train-deploy performance gap for both approaches

**References**: BarrierNet [N04] (T-RO 2023, MIT/Daniela Rus group), GCBF+ [N05] (T-RO 2024)
**Code**: https://github.com/Weixy21/BarrierNet

**Decision point**: Choose the best safety architecture for remaining phases based on:
- Train-deploy gap (target: <5%)
- Training wall-clock time
- Safety violation rate
- Task performance

**Success criteria:**
- Train-deploy gap measured and documented
- Clear recommendation for which safety architecture to use going forward

### Phase 3: Partial Observability + Self-Play (Months 4-6)

**Deliverables:**
- [ ] Limited FOV sensor model
- [ ] BiMDN belief encoder
- [ ] AMS-DRL self-play implementation
- [ ] **Simultaneous self-play comparison** [N06] (no freezing, both agents train concurrently)
- [ ] Curriculum learning (4 levels)
- [ ] NE convergence analysis
- [ ] **MACPO / MAPPO-Lagrangian / CPO baselines** [N03]
- [ ] **Strategy diversity metric** (k-means trajectory clustering)

**Key decisions:**
- 120° FOV, 10m range for opponent detection
- 36-ray lidar for obstacle detection
- AMS-DRL protocol from Paper [18] as primary, with simultaneous self-play [N06] comparison
- NE convergence threshold η = 0.10
- MACPO code from https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation

**Success criteria:**
- NE convergence within 6 self-play phases
- Belief encoder improves performance by >15% vs raw observation
- Policies exhibit qualitatively diverse strategies (measured via strategy diversity metric)
- **MACPO/CPO baselines complete for comparison**

### Phase 4: Sim-to-Real Transfer (Months 6-8)

**Deliverables:**
- [ ] Isaac Lab PE environment with domain randomization (native MARL via PettingZoo Parallel API [N11])
- [ ] Gazebo ROS2 intermediate validation
- [ ] ONNX model export pipeline
- [ ] Real robot deployment on TurtleBot3/4 or F1TENTH
- [ ] GP disturbance estimation on real robot
- [ ] **QP solver benchmarking** (OSQP, qpOASES — must achieve <50ms at 20Hz)
- [ ] **GP cold-start protocol**: pre-fill with simulation residuals, use conservative margin (larger κ) for first 100 steps
- [ ] **Train-deploy gap measurement**: compare CBF-Beta (training) vs RCBF-QP (deployment) performance

**Sim-to-real pipeline** (from Papers [07], [N10], [N11]):
```
Isaac Lab (training) [N11]
    │ GPU-accelerated massively parallel environments (up to 1.6M FPS)
    │ Native MARL support via PettingZoo Parallel API
    │ Built-in domain randomization + sensor simulation (lidar, cameras)
    │ PPO + CBF-Beta training
    ▼
ONNX Export [N10]
    │ PyTorch → ONNX conversion
    │ Quantization for edge deployment
    │ N10 validates this exact pipeline with 80-100% real-world success
    ▼
Gazebo/ROS2 (intermediate testing) [N10]
    │ Catches ~80% of sim-to-real issues
    │ Test with ROS2 navigation stack integration
    │ N10 confirms Isaac → ONNX → Gazebo → Real pipeline works
    ▼
Real Robot (TurtleBot4 or F1TENTH)
    │ ONNX inference (~5ms per step)
    │ RCBF-QP safety filter (real-time)
    │ GP disturbance learning (online)
    │ 20Hz control loop
```

**Hardware options:**

| Platform | Dynamics | Speed | Compute | Cost | Recommended For |
|----------|----------|-------|---------|------|----------------|
| TurtleBot3 Burger | Diff-drive | 0.22 m/s | RPi 4 | ~$600 | Initial prototyping |
| TurtleBot4 | Diff-drive | 0.3 m/s | RPi 4 + Create3 | ~$1,200 | Main platform |
| F1TENTH | Ackermann | 3+ m/s | Jetson | ~$3,000 | High-speed PE |
| Custom diff-drive | Diff-drive | 1.0 m/s | Jetson Nano | ~$500 | Budget option |

**Success criteria:**
- <10% performance gap between simulation and real-world
- Zero safety violations on real robot (RCBF-QP)
- Real-time inference at 20Hz

### Phase 5: Analysis & Publication (Months 8-10)

**Deliverables:**
- [ ] Formal safety analysis (CBF validity, constraint satisfaction proof)
- [ ] NE convergence analysis (exploitability measurement)
- [ ] Comprehensive benchmarking against all baselines
- [ ] Real-robot demonstration videos
- [ ] Conference paper (ICRA/IROS/CoRL)
- [ ] Extended journal paper (RA-L/T-RO)

---

## 7. Key Paper References (Quick Lookup)

### Must-Reference Papers for Pathway A
| Paper | What to Cite For | Key Equation/Method |
|-------|-----------------|-------------------|
| [01] Wang 2025 | RL+SMC+CBF architecture for PE | Three-term controller: u = u_RL + u_SMC + u_CBF |
| [02] Gonultas 2024 | 1v1 PE on ground robots, BiMDN, curriculum | BiMDN belief encoder, MADDPG baseline |
| [05] Yang 2025 | CBF safety filtering in RL | Closed-form CBF filter + reward shaping |
| [06] Emam 2022 | Robust CBF + GP for model uncertainty | RCBF-QP: min\|\|u-u_rl\|\|² s.t. RCBF |
| [07] Salimpour 2025 | Sim-to-real pipeline | Isaac Sim → Gazebo → ROS2, ONNX |
| [13] Kokolakis 2022 | Safe PE with barrier functions | Barrier-augmented HJI cost: L_s = L + ψB(x) |
| [16] Suttle 2024 | CBF-constrained Beta policies | Truncated policy π^C, convergence theorem |
| [18] Xiao 2024 | AMS-DRL self-play + NE convergence | Staged alternating training protocol |
| [25] Ganai 2024 | HJ-RL survey, shielding framework | Roadmap for integrating safety with RL |
| [30] Zhou 2023 | Conflict-averse gradient for safe nav | CAPO: max_{g} min_{i} g_i^T · g |
| [34] La Gatta 2025 | 1v1 PE with unicycle dynamics | SHADOW multi-headed architecture |
| [37] Yang 2025 | RL-PE comprehensive survey | Taxonomy: unilateral/bilateral strategy learning |

### New Papers (N01-N15) — from Validation Review (2026-02-21)

| Paper | What to Cite For | Key Contribution |
|-------|-----------------|-----------------|
| [N01] Dawson et al. 2023 | Learning CBFs survey | Background: taxonomy of learned CBF methods |
| [N02] So & Fan 2024 | PNCBF — neural CBF fallback | V^{h,π} IS a CBF; hardware-validated on quadcopters (12D), scales to F-16 (16D); **serious Phase 2 alternative if hand-crafted CBFs fail** |
| [N03] Gu et al. 2023 | MACPO baseline | Safe MARL with monotonic reward+safety improvement (Theorem 4.4); code available |
| [N04] Xiao et al. 2023 | BarrierNet — Phase 2.5 | Differentiable QP safety layer; end-to-end training; code: https://github.com/Weixy21/BarrierNet |
| [N05] Zhang et al. 2024 | GCBF+ | Scalable graph-based CBFs for multi-agent; T-RO 2024 |
| [N06] Selvam et al. 2024 | Simultaneous self-play PE | TD3 self-play for 1v1 PE with unicycle dynamics; no freezing needed |
| [N07] Zhang & Xu 2024 | POLICEd RL | Model-free hard constraints (limited: rel. degree 1, affine, deterministic) |
| [N08] Ma et al. 2024 | Statewise CBF projection | Alternative CBF implementation |
| [N09] Liu et al. 2024 | Safe RL/CMDP survey | Positioning reference |
| [N10] Salimpour et al. 2025 | Isaac sim-to-real pipeline | **Validates Phase 4: Isaac → ONNX → Gazebo → Real with 80-100% success** |
| [N11] Mittal et al. 2025 | Isaac Lab | **GPU-accelerated MARL via PettingZoo; up to 1.6M FPS; native sensor simulation** |
| [N12] Zhang & Yang 2025 | **CRITICAL: VCP-CBF for nonholonomic robots** | Virtual control point CBF achieves uniform relative degree 1; M-matrix prioritizes steering over braking |
| [N13] Xiao et al. 2023 | **Learned feasibility constraints** | Reduces CBF-QP infeasibility from 8.11% → 0.21% in 3 iterations |
| [N14] Gu et al. 2026 | SB-TRPO | Safety-biased trust region; 10x cheaper than CPO; alternative baseline |
| [N15] Liu et al. 2025 | **Closest prior work: RMARL-CBF-SAM** | Robust MARL + neural CBFs + safety attention + reward shaping; validates our choices but differs in 5 ways |

### Additional References by Component
| Component | Papers | Notes |
|-----------|--------|-------|
| CBF theory | [03], [10], [N01] | Surveys for background |
| CBF for nonholonomic | [N12] | **Critical**: VCP-CBF formulation |
| CBF feasibility | [N13] | Learned feasibility constraints |
| Safe MARL | [N03], [N15] | MACPO baseline + closest prior work |
| Self-play theory | [04], [11], [N06] | Surveys + simultaneous SP for PE |
| Differentiable safety | [N04], [N05] | BarrierNet + GCBF+ |
| Neural CBFs | [N02], [N15] | PNCBF + RMARL-CBF-SAM |
| Sim-to-real | [N10], [N11] | Isaac Lab + validated pipeline |
| HJ reachability (if extending to Pathway B) | [21], [22], [23], [25] | DeepReach family |
| Opponent modeling | [34], [08] | SHADOW + privileged learning |
| Hierarchical PE | [32] | Diffusion-RL (future extension) |

---

## 8. Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CBF-Beta slows training convergence | Medium | Medium | Fall back to BarrierNet [N04] or CBF-QP filter; compare both in Phase 2.5 |
| Self-play fails to converge to NE | Low | High | Use PSRO [04] as fallback; add simultaneous self-play [N06]; exploitability check |
| Sim-to-real gap too large | Medium-Low | High | Aggressive DR in Isaac Lab [N11]; Gazebo intermediate step; GP online adaptation; **N10 validates exact pipeline (Isaac → ONNX → Gazebo → Real) with 80-100% success** |
| **CBF infeasible in tight spaces** | **HIGH** | **HIGH** | **Three-tier: (1) N13 learned feasibility constraints during training (8.11%→0.21%), (2) hierarchical relaxation at deployment, (3) backup controller** |
| Real-time QP too slow | Low | Medium | Use closed-form CBF [05] or OSQP warm-starting; benchmark QP solvers in Phase 4 |
| BiMDN belief encoder fails under fast dynamics | Low | Medium | Fall back to LSTM-only belief |
| GP disturbance model inaccurate | Low | Low | Increase GP data; ensemble GP; pre-fill from simulation (cold-start protocol) |
| **Nonholonomic CBF relative degree issue** | **HIGH** | **HIGH** | **Virtual control point CBF [N12]; validate in Phase 1 before any integration** |
| **Train-deploy safety gap** | **Medium** | **Medium** | **BarrierNet end-to-end training [N04]; measure gap explicitly in Phase 2.5** |
| **CBF-safety + pursuit conflict** | **Medium** | **Medium** | **Safety-reward shaping (w5) [05]; conflict-averse gradient (CASRL) [30] as fallback** |
| Self-play + CBF creates overly conservative policies | Medium | Medium | Reduce α over training; compare with unconstrained baseline |

---

## 9. Software Stack

### Training
| Component | Tool | Version |
|-----------|------|---------|
| RL framework | Stable Baselines 3 / CleanRL | Latest |
| Environment | Gymnasium (custom) | 0.29+ |
| Physics (training) | Isaac Lab (GPU-accelerated, MARL-native via PettingZoo [N11]) or PyBullet | Latest |
| CBF computation | cvxpy (QP solver) | 1.4+ |
| GP library | GPyTorch | Latest |
| Belief encoder | PyTorch (custom BiMDN) | 2.0+ |
| Experiment tracking | Weights & Biases | - |
| Config management | Hydra | 1.3+ |

### Deployment
| Component | Tool |
|-----------|------|
| Robot middleware | ROS2 Humble |
| Inference | ONNX Runtime |
| Safety filter | C++ RCBF-QP (real-time) |
| Localization | AMCL / SLAM Toolbox |
| Simulation (validation) | Gazebo Fortress |
| Visualization | RViz2 |

---

## 10. Expected Contributions & Publications

### Contribution 1: SafePE Algorithm
- First safe DRL algorithm for 1v1 PE with provable safety guarantees
- CBF-constrained self-play with NE convergence
- Novel: CBF-Beta policy in adversarial two-player setting

### Contribution 2: Real-World Deployment
- First safety-guaranteed PE on physical ground robots
- Sim-to-real pipeline with domain randomization
- RCBF-QP + GP for real-world robustness

### Contribution 3: Comprehensive Benchmark
- First systematic comparison of safe RL methods for PE
- Ablation studies isolating each component's contribution
- Open-source environment and baselines

### Publication Targets
| Paper | Venue | Content |
|-------|-------|---------|
| Paper 1 | ICRA / IROS / CoRL | SafePE algorithm + simulation results |
| Paper 2 | RA-L / T-RO (journal) | Full system with real-robot results |
| Paper 3 | CDC / L4DC | Formal safety analysis + NE convergence proof |

### Future Extensions (Beyond Core Scope)
| Extension | Source | When to Consider | Limitations |
|-----------|--------|-----------------|-------------|
| **POLICEd RL** | [N07] (RSS 2024) | If CBF construction proves difficult for complex obstacle shapes | Only works with relative degree 1 constraints, affine constraints, deterministic dynamics — suitable for simple cases (linear boundary constraints) only, not general alternative to our CBF approach |
| **Multi-agent PE (>2 agents)** | [N05] GCBF+ | After 1v1 system validated | Requires graph-based CBF reformulation; GCBF+ scales to 1024 agents |
| **Mixture of experts policy** | - | If single policy fails to capture diverse strategies | Separate expert networks for open-field vs obstacle-rich scenarios |
| **Diffusion-based hierarchical PE** | [32] | For complex multi-stage strategies | High-level diffusion planner + low-level RL controller |

---

## 11. Quick-Start Checklist

For the next session, start with:

1. **Create the Gymnasium environment** (`pursuit_evasion_env.py`):
   - Unicycle dynamics for both agents
   - Rectangular bounded arena (20m x 20m)
   - Full-state observation (simplify first)
   - Distance-based reward (zero-sum)
   - No obstacles initially

2. **Implement basic PPO self-play**:
   - Use Stable Baselines 3 PPO
   - Vanilla alternating training (train P for N episodes, freeze, train E for N episodes)
   - Log capture rate, escape rate, episode length

3. **Verify basic learning**:
   - Pursuer should learn to approach evader
   - Evader should learn to flee
   - Neither should be dominant after convergence

4. **Validate VCP-CBF (end of Phase 1)**:
   - Implement VCP-CBF for arena boundary on simple unicycle obstacle avoidance
   - Verify steering-over-braking behavior [N12]
   - This MUST succeed before proceeding to Phase 2

5. **Then add safety (Phase 2)**:
   - Implement VCP-CBF-constrained Beta policy
   - Add 3-tier infeasibility handling [N13]
   - Add safety-reward shaping (w5)
   - Verify zero violations

---

## Appendix A: Key Equations Reference

### A.1 Unicycle Dynamics
```
ẋ = v·cos(θ),  ẏ = v·sin(θ),  θ̇ = ω
u = [v, ω],  v ∈ [0, v_max],  ω ∈ [-ω_max, ω_max]
```

### A.2 VCP-CBF Condition [N12]
```
# Virtual control point (VCP):
q = [x + d·cos(θ), y + d·sin(θ)]     (d ≈ 0.05m for TurtleBot)

# VCP time derivative (uniform relative degree 1 in both v and ω):
q̇ = [v·cos(θ) - d·ω·sin(θ),  v·sin(θ) + d·ω·cos(θ)]

# CBF condition at VCP:
ḣ(x, u) + α·h(x) ≥ 0
where ḣ = ∂h/∂q · q̇ = ∂h/∂q · [v·cos(θ)-d·ω·sin(θ), v·sin(θ)+d·ω·cos(θ)]

# This is affine in u = [v, ω], making CBF-QP well-posed
# NOTE: Position-based CBFs have mixed relative degree for unicycle —
# ω does not appear in ḣ. VCP fixes this. See Section 3.3.1.
```

### A.3 CBF-Constrained Beta Policy (Paper [16])
```
π_θ^C(u|x) = π_θ(u|x) / π_θ(C(x)|x)    for u ∈ C(x)
            = 0                              otherwise

C(x) = {u : ḣ_i(x,u) + α_i·h_i(x) ≥ 0, ∀i}
```

### A.4 RCBF-QP Safety Filter (Paper [06])
```
u* = argmin_{u} ||u - u_RL||²
s.t.  L_f h + L_g h · u ≥ -α·h(x) + κ·σ_d(x)    [robust margin]
      u ∈ U                                          [control bounds]
```

### A.5 AMS-DRL Convergence (Paper [18])
```
NE criterion: |SR_P(Sk) - SR_E(Sk)| < η
where SR = success rate, η ≈ 0.10
```

### A.6 PPO Clipped Objective
```
L^CLIP(θ) = E_t[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

### A.7 Zero-Sum Reward
```
r_P = -r_E
J_P = E[Σ γ^t · r_P_t]  (pursuer maximizes)
J_E = E[Σ γ^t · r_E_t]  (evader maximizes = pursuer minimizes)
```
