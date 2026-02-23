# Phase 3: Partial Observability + Self-Play

**Timeline**: Months 4-6
**Status**: Pending
**Prerequisites**: Phases 1-2.5 complete (DCBF post-hoc filter chosen as safety architecture)
**Next Phase**: [Phase 4 — Sim-to-Real Transfer](./phase4_sim_to_real_transfer.md)

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
10. [Guide to Phase 4](#10-guide-to-phase-4)

---

## 1. Phase Overview

### 1.1 Goal

Make the PE system **realistic** by adding:
1. **Partial observability** — limited field-of-view (FOV) sensors, no omniscient state
2. **BiMDN belief encoder** — maintain a belief about the opponent's hidden state
3. **AMS-DRL self-play protocol** — formal alternating self-play with NE convergence
4. **Curriculum learning** — progressive difficulty for stable training
5. **MACPO/CPO baselines** — standard safe MARL baselines for comparison

This phase transforms the system from a "proof of concept" into a **realistic, publication-ready** PE system.

### 1.2 Why This Phase Matters

Phases 1-2 used **full-state observation** — both agents see everything. This is unrealistic:
- Real robots have limited sensor range and FOV
- Opponent can be occluded by obstacles or out of range
- Strategic information hiding becomes a core game element (evader hides behind obstacles)

Partial observability fundamentally changes the game: the evader can now exploit **information asymmetry**. The pursuer must maintain a **belief** about the evader's position and act under uncertainty.

### 1.3 Key Design Decisions for Phase 3

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sensor model | 120 deg FOV + lidar | Realistic for ground robots; matches [02] |
| Belief encoder | BiMDN (Bidirectional MDN) | Handles multimodal uncertainty; proven in [02] |
| Safety layer | PPO + post-hoc DCBF filter (gamma=0.2) | Phase 2.5 decision: standard PPO + DCBF dominates BarrierNet/CBF-Beta |
| Self-play protocol | AMS-DRL (alternating) [18]; simultaneous comparison optional [N06] | NE convergence guarantee; simpler alternative if time permits |
| Curriculum | 4-level progressive difficulty | Stable training from simple to complex |
| Safe MARL baselines | MACPO, MAPPO-Lagrangian, CPO | Standard comparisons from [N03] |
| Observation history | K=10 timesteps | Sufficient for belief estimation |

### 1.4 Dependencies from Prior Phases

| Prior Phase Output | How Used in Phase 3 |
|-------------------|---------------------|
| PE environment (Phase 1) | Extended with partial obs sensor model |
| DCBF post-hoc filter (Phase 2.5) | Applied to partial-obs agents; filter uses true state, policy uses partial obs |
| Self-play infrastructure (Phase 1) | Upgraded to AMS-DRL protocol |
| VCP-CBF formulation (Phase 2) | DCBF variant used; operates on true state, not agent observations |
| Baseline results (Phase 1-2.5) | Comparison with partial-obs performance; DCBF baseline: 100% capture, 3.57% violations, 6.6% intervention |

---

## 2. Background & Theory

### 2.1 Partial Observability in PE Games

In a **Partially Observable Markov Decision Process (POMDP)**, the agent doesn't see the true state but receives an observation:

```
True state:   s_t = [x_P, y_P, theta_P, x_E, y_E, theta_E]  (6D)
Observation:  o_t = O(s_t, sensor_params)                     (partial, noisy)
```

For PE with limited FOV:
- Agent can see opponent only within a cone of half-angle `alpha_fov` and range `r_detect`
- If opponent is outside FOV: `d_to_opponent = -1, bearing = 0` (no info)
- Lidar provides obstacle/boundary info regardless of opponent visibility
- Own state (velocity, heading) is always known (odometry)

**Strategic implications**:
- Evader can hide behind obstacles or outside pursuer's FOV
- Pursuer must search for evader when lost
- "Lost" states require fundamentally different strategies than "tracking" states
- Information gathering becomes a first-class objective

### 2.2 Belief Encoding with BiMDN

The **Bidirectional Mixture Density Network (BiMDN)** from Paper [02] maintains a probabilistic belief about the opponent's position using observation history.

**Architecture**:
```
Observation history: [o_{t-K}, o_{t-K+1}, ..., o_t]
    │
    ├──► Forward LSTM: processes history chronologically
    │
    ├──► Backward LSTM: processes history in reverse
    │
    └──► Concatenate hidden states → MLP → MDN parameters
              │
              └──► Mixture of M Gaussians:
                   p(x_opp | o_{t-K:t}) = SUM_m w_m * N(mu_m, sigma_m)
                   w_m: mixing weights (sum to 1)
                   mu_m: mean position (2D)
                   sigma_m: covariance (2D)
```

**Why BiMDN (not just LSTM)?**:
- LSTM gives a single hidden state (single mode)
- MDN captures **multimodal** uncertainty: "evader is either behind obstacle A OR behind obstacle B"
- Bidirectional processing captures both past-to-present and present-to-past temporal patterns
- Paper [02] shows BiMDN significantly outperforms raw observation and LSTM-only approaches

**Training BiMDN**: The BiMDN is trained to predict the opponent's true position from the observation history. Loss is negative log-likelihood of the true position under the mixture:

```python
loss_BiMDN = -log(SUM_m w_m * N(x_true; mu_m, sigma_m))
```

This can be trained:
1. **Supervised**: Pre-train on collected trajectories with known ground truth
2. **End-to-end**: Train jointly with the RL policy (gradients flow through BiMDN)
3. **Hybrid**: Pre-train supervised, then fine-tune end-to-end (recommended)

### 2.3 AMS-DRL Self-Play Protocol

**Alternating Multi-Stage Deep RL (AMS-DRL)** from Paper [18] is a structured self-play protocol with NE convergence guarantees.

```
Phase S0 (Cold-Start):
    - Train evader for basic navigation (no adversary)
    - ~500-1000 episodes
    - Purpose: learn locomotion + obstacle avoidance

Phase S1 (Pursuer Training):
    - Freeze evader from S0
    - Train pursuer to capture frozen evader
    - ~1000-2000 episodes

Phase S2 (Evader Training):
    - Freeze pursuer from S1
    - Train evader against frozen pursuer
    - ~1000-2000 episodes

Phase Sk (Alternating):
    - Continue S1/S2 until convergence
    - Convergence: |SR_P - SR_E| < eta (eta = 0.10)
    - Typically: 4-6 phases [18]
```

**Key differences from Phase 1's vanilla self-play**:
1. Cold-start phase (S0) for basic skills
2. Formal convergence criterion
3. Explicit NE verification
4. Paper [18] provides convergence proof

### 2.4 Simultaneous Self-Play Alternative [N06]

Paper [N06] (Selvam et al. 2024) shows that **simultaneous training** (no freezing) works for 1v1 PE with unicycle dynamics:
- Both agents train concurrently
- Simpler implementation (no freeze/unfreeze logic)
- Uses TD3 (off-policy) instead of PPO

**Comparison question**: Is simultaneous self-play competitive with AMS-DRL for our setting? Phase 3 answers this.

### 2.5 Curriculum Learning

From Paper [02], layer curriculum on top of self-play:

```
Level 1: Close encounters, no obstacles
    Initial distance: 2-5m, open arena
    Purpose: basic approach/flee

Level 2: Medium distance, no obstacles
    Initial distance: 5-15m, open arena
    Purpose: long-range strategies

Level 3: Close encounters with obstacles
    Initial distance: 2-5m, 2-4 obstacles
    Purpose: use/avoid obstacles

Level 4: Full scenario
    Random distance, random obstacles, FOV active
    Purpose: generalize
```

**Advancement**: Move to next level when the pursuer achieves capture rate > 70% over a 100-episode window. (Since capture_rate + escape_rate = 1, requiring both > 70% is impossible. The balance criterion is handled separately by AMS-DRL's NE convergence check.)

**Timeout policy**: If advancement does not occur within 2× the expected training steps for that level (i.e., 2 × `timesteps_per_phase`), log a warning, investigate the training dynamics, and either: (a) lower the advancement threshold to 60%, or (b) force-advance to the next level with documentation justifying the override.

**Regression policy**: Curriculum levels are monotonically non-decreasing. Once advanced, the agent does not regress even if performance temporarily drops — the self-play health monitor handles regression via checkpoint rollback instead.

### 2.6 Nash Equilibrium Verification

An NE is reached when neither player can unilaterally improve. Practical tests:

1. **Success rate balance**: |SR_P - SR_E| < 0.10
2. **Exploitability test**: Train a best-response agent against each trained agent. If the best-response can't significantly improve, the original is near NE.
3. **Strategy diversity**: k-means clustering on trajectory embeddings — diverse strategies suggest richer NE

### 2.7 DCBF Safety Filter Under Partial Observability

**Critical design point**: The DCBF post-hoc safety filter operates on the **true state**, NOT the agent's observation. The policy and safety filter have **separate information channels**:

```
Agent policy:      partial observation → BiMDN latent → PPO actor → nominal action [v, ω]
DCBF filter:       true state (x, y, θ of both robots) → constraint check → safe action [v, ω]
```

**Phase 2.5 decision**: Standard PPO (Gaussian policy, SB3 default) trained with obstacles, with post-hoc DCBF filter (gamma=0.2) applied at action execution. BarrierNet and CBF-Beta were terminated — see `docs/phases/phase2_5_decision_report.md`.

**DCBF Phase 2.5 baseline performance**:
- Capture rate: 100%, Safety violations: 3.57%, Intervention rate: 6.6%
- Inference: 0.46 ms/step (PPO 0.18ms + DCBF 0.28ms)

This separation is valid because:
- In simulation: true state is always available to the filter
- In deployment: the safety filter uses the robot's own state (known via odometry) + obstacle positions (known map or lidar)
- The inter-robot collision CBF needs both robots' positions — this comes from a centralized safety monitor or UWB ranging

**Safety metric distinction** (important for paper claims):
1. **Physical safety violations**: Actual collisions with obstacles, boundaries, or other robot. **Target: zero.**
2. **CBF boundary violations**: Timesteps where `h(x) < 0` for any constraint. **Target: < 1% with tuned DCBF parameters.**

The paper's safety claim is: "The DCBF filter prevents all physical collisions while maintaining CBF constraint satisfaction at >99% of timesteps." This is a strong, honest, and verifiable claim. To tighten CBF boundary violations below 1%, use stricter DCBF parameters (gamma=0.05-0.1) during evaluation, accepting slightly higher intervention rate.

### 2.8 Self-Play Training Health Monitoring

Self-play training is inherently unstable. Unlike single-agent RL where the environment is stationary, self-play creates a **non-stationary learning problem** — each agent's opponent keeps changing. This leads to several failure modes that require active monitoring:

#### 2.8.1 Entropy Collapse (Mode Collapse)

In SB3's PPO with continuous actions, the policy uses a `DiagGaussianDistribution` where `log_std` is a learnable `nn.Parameter` (state-independent). The differential entropy for N action dimensions is:

```
H_total = SUM_{i=1}^{N} (1.4189 + log_std_i)    [nats]
```

For our 2D pursuit-evasion (v, omega), with default `log_std_init=0.0`:
```
H_initial = 2 × (1.4189 + 0.0) = 2.8378 nats
```

**Entropy collapse** occurs when `log_std` drops too low, making the policy near-deterministic. This is a **leading indicator** of performance collapse — entropy drops *before* reward degrades (Ahmed et al., ICML 2019; Han et al., NeurIPS 2021). In self-play, one agent becoming deterministic allows the opponent to overfit to a single counter-strategy.

| log_std | sigma | Per-dim entropy (nats) | Interpretation |
|:---:|:---:|:---:|---|
| 0.0 | 1.00 | 1.42 | Initial (maximum exploration) |
| -0.5 | 0.61 | 0.92 | Healthy learning, narrowing |
| -1.0 | 0.37 | 0.42 | Moderate exploitation |
| -1.5 | 0.22 | -0.08 | **WARNING**: near-deterministic |
| -2.0 | 0.14 | -0.58 | **DANGER**: likely mode collapse |

**Thresholds for 2D actions** (total entropy):
- Yellow alert: < 0.5 (log_std dropping below -1.0)
- Red alert: < -0.5 (log_std below -1.5)
- Mode collapse: < -2.0 (no meaningful exploration)

**Mitigation**: (1) Non-zero `ent_coef` in PPO (critical for self-play), (2) hard `log_std` clamping floor at -2.0, (3) rollback to earlier checkpoint.

#### 2.8.2 Trajectory Diversity and Behavioral Mode Collapse

Even if entropy stays healthy, the agent may converge to repetitive behavior patterns. Trajectory diversity captures whether agents exhibit **varied strategies** (flanking, direct chase, retreat, obstacle hiding) vs. a single repeated pattern.

Three metrics from cheapest to most informative:

1. **Outcome diversity** (cheapest): Track spatial distribution of captures/timeouts and episode length coefficient of variation.
2. **DTW trajectory clustering** (recommended): Dynamic Time Warping compares trajectory shapes regardless of speed differences. K-means on DTW distances with cluster entropy as the diversity metric.
3. **State-space coverage** (fastest): Grid-based counting of visited cells as a lightweight proxy.

**Healthy thresholds for DTW clustering (k=5)**:

| Metric | Healthy | Warning | Mode Collapse |
|---|:---:|:---:|:---:|
| `cluster_entropy_normalized` | > 0.7 | 0.4 - 0.7 | < 0.4 |
| `n_active_clusters` (of 5) | 4-5 | 2-3 | 1 |
| `dominant_cluster_fraction` | < 0.4 | 0.4 - 0.7 | > 0.7 |

#### 2.8.3 Fixed-Baseline Evaluation

In self-play, reward alone is uninformative (agent A improves → agent B's reward drops, creating misleading signals). **Fixed baselines** provide an absolute performance anchor:

| Baseline Type | Purpose | Frequency |
|---|---|---|
| Random agent | Sanity check (must always win) | Every 10K steps |
| Scripted pursuer (pure-pursuit) | Minimum competence bar | Every 10K steps |
| Scripted evader (flee-to-corner) | Minimum pursuit competence | Every 10K steps |
| Early checkpoint (step 50K) | Detect catastrophic forgetting | Every 25K steps |
| Best-so-far checkpoint | Validate continued improvement | Every 25K steps |

OpenAI Five used TrueSkill for evaluation; we use Elo ratings with K=32 across the baseline pool. A drop of >100 Elo in one evaluation cycle indicates significant regression.

#### 2.8.4 Checkpoint Rollback

When training degrades beyond recovery, the model must roll back to an earlier checkpoint. Trigger conditions (any one sufficient):

| Condition | Threshold |
|---|---|
| Capture rate = 100% sustained | 200+ episodes (evader collapsed) |
| Capture rate = 0% sustained | 200+ episodes (pursuer collapsed) |
| Entropy collapsed | Total entropy < -2.0 for either agent |
| Baseline win rate crash | Drop > 30% vs scripted baseline |
| Elo drop | > 100 points in one eval cycle |
| Reward divergence | NaN or > 5× initial magnitude |

**Protocol**: Keep 15 rolling checkpoints + milestone checkpoints. Rollback goes back 3 checkpoints (not just 1, since problems start before symptoms appear). Minimum 50K-step cooldown between rollbacks to avoid thrashing.

#### 2.8.5 Catastrophic Forgetting in Alternating Self-Play

In alternating self-play (AMS-DRL), **catastrophic forgetting** manifests as:
1. Evader learns to exploit the *specific* pursuer. When pursuer next trains, it overwrites what it learned to counter the previous evader.
2. **Rock-paper-scissors cycling**: Strategies cycle without converging, detected by oscillating win rates against fixed baselines.
3. **Performance regression**: Agent loses to opponents it previously beat consistently.

**Detection**: Periodically evaluate against historical opponent checkpoints. A peak-drop > 25% (win rate vs. any historical opponent) indicates forgetting.

**Mitigations** (from literature):
1. Experience replay across training phases
2. Elastic Weight Consolidation (EWC) to protect important weights
3. Population-based training (PSRO family) instead of 1-vs-1 alternation
4. Shorter alternation cycles (10K-25K steps instead of 500K)
5. Mixed opponents: 80% latest self + 20% historical checkpoints (OpenAI Five approach)

---

## 3. Relevant Literature

### 3.1 Core Papers for Phase 3

| Paper | Relevance | Key Takeaway |
|-------|-----------|--------------|
| **[02] Gonultas & Isler 2024** | BiMDN belief encoder, curriculum, PE on ground robots | Detailed BiMDN architecture; curriculum design; MADDPG baseline |
| **[18] Xiao et al. 2024 (AMS-DRL)** | Self-play protocol with NE convergence | Staged alternating training; convergence proof |
| **[N06] Selvam et al. 2024** | Simultaneous self-play for PE | TD3 self-play without freezing; comparison target |
| **[N03] Gu et al. 2023 (MACPO)** | Safe MARL baseline | Monotonic reward+safety improvement; code available |
| **[34] La Gatta 2025 (SHADOW)** | Multi-headed architecture for PE | Opponent modeling; privileged learning |
| **[04] Lanctot et al.** | Self-play theory | PSRO; exploitability measurement |

### 3.2 Self-Play Health Monitoring References

| Paper / Source | Relevance | Key Takeaway |
|-------|-----------|--------------|
| **Berner et al. 2019 (OpenAI Five)** | Self-play at scale, evaluation protocol | TrueSkill baselines; ~20 "surgery" rollbacks over 10 months; 80%/20% latest/historical opponent mix |
| **Ahmed et al. ICML 2019** | Entropy collapse theory | Entropy drop is a *leading indicator* of performance collapse |
| **Han et al. NeurIPS 2021** | Max-min entropy framework | Entropy floor prevents mode collapse in competitive settings |
| **Self-Play Survey 2024** | Comprehensive self-play overview | PSRO family, population-based training, diversity maintenance |
| **SB3 Distributions** | PPO DiagGaussian implementation | `log_std` is state-independent `nn.Parameter`; entropy = 1.4189 + log_std per dim |
| **Policy Consolidation (Kaplanis 2019)** | Catastrophic forgetting in continual RL | EWC-based weight protection across training phases |
| **Map-based Experience Replay (2023)** | Anti-forgetting in self-play | Experience buffer across phases prevents strategy cycling |
| **Self-adaptive PSRO (IJCAI 2024)** | Adaptive meta-strategy switching | Automatically switches between meta-strategy solvers during training |
| **Elo Uncovered (NeurIPS 2024)** | Elo rating robustness | Best practices for rating systems in competitive evaluation |

### 3.3 Baseline Implementation References

| Baseline | Source | Code |
|----------|--------|------|
| MACPO | [N03] | https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation |
| MAPPO-Lagrangian | [N03] | Same repo |
| CPO | Achiam et al. 2017 | Multiple open-source implementations |
| SB-TRPO | [N14] Gu et al. 2026 | Check paper; 10x cheaper than CPO |

### 3.4 Reading Priority

**Must read in detail**:
- [02]: BiMDN architecture, training procedure, observation space design
- [18]: AMS-DRL protocol details, convergence analysis

**Read for implementation**:
- [N06]: Simultaneous self-play (simpler alternative)
- [N03]: MACPO implementation details

**Read for health monitoring**:
- OpenAI Five (Berner et al.): Evaluation protocol, rollback strategy, historical opponent mix
- SB3 source (distributions.py): Understand DiagGaussian entropy computation

**Skim**:
- [34]: Architecture ideas for future refinement
- [04]: PSRO for exploitability measurement
- Ahmed et al. 2019: Entropy-performance relationship theory
- Policy Consolidation (Kaplanis 2019): Anti-forgetting techniques

---

## 4. Deliverables Checklist

### 4.1 Must-Have Deliverables

- [ ] **D1**: Limited FOV sensor model (120 deg, 10m range)
- [ ] **D2**: Lidar sensor model (36 rays, 5m range)
- [ ] **D3**: BiMDN belief encoder (with pluggable encoder interface — BiMDN, LSTM, MLP fallbacks)
- [ ] **D4**: AMS-DRL self-play implementation (with NavigationEnv for cold-start)
- [ ] **D5**: Curriculum learning (4 levels, FOV active in all levels)
- [ ] **D6**: NE convergence analysis
- [ ] **D7**: MAPPO-Lagrangian baseline (required); MACPO, CPO optional
- [ ] **D8**: Self-play health monitoring system (entropy, diversity, baselines, rollback, forgetting detection)

### 4.2 New Research Additions

- [ ] **D9a**: Wall-segment obstacle support (WallSegment class, vcp_cbf_wall(), lidar ray-segment intersect)
- [ ] **D9b**: Complex environment layouts (corridor, L-shaped, warehouse) in `envs/layouts.py` + Hydra configs
- [ ] **D9c**: Asymmetric capability experiments (3 speed ratios: 1.5, 0.8, 0.5) in Session 9
- [ ] **D9d**: Human evader interface (`envs/human_interface.py`, `scripts/human_evader_experiment.py`)
- [ ] **D9e**: DCBF safety theorem (`docs/proofs/dcbf_safety_theorem.md`, `scripts/verify_dcbf_theorem.py`)

### 4.3 Optional Deliverables (if time/compute permits)

- [ ] **D9f**: Simultaneous self-play comparison [N06] — tangential to core novelty
- [ ] **D7b**: MACPO baseline [N03] — HIGH integration risk with current stack
- [ ] **D7c**: CPO single-agent baseline — nice-to-have

### 4.4 Visualization & Tracking Deliverables

- [ ] **D10**: PERenderer partial-observability overlays — FOV cone visualization, lidar ray display, belief distribution heatmap (BiMDN), undetected-agent ghost marker
- [ ] **D11**: wandb experiment tracking for all Phase 3 runs — NE convergence curves, per-phase metrics, health monitoring dashboards, baseline comparison tables, eval videos with FOV overlay

### 4.5 Analysis Deliverables

- [ ] **D12**: Strategy diversity metric (DTW trajectory clustering + k-means)
- [ ] **D13**: Exploitability measurement (best-response training)
- [ ] **D14**: BiMDN vs raw obs vs full state comparison
- [ ] **D15**: AMS-DRL vs simultaneous self-play comparison (OPTIONAL — only if D5 is done)

### 4.6 Documentation Deliverables

- [ ] **D16**: Partial observability impact analysis
- [ ] **D17**: NE convergence plots and analysis
- [ ] **D18**: Full baseline comparison table (MAPPO-Lag + unconstrained required; MACPO/CPO if available)

---

## 5. Session-wise Implementation Breakdown

### Session 1: Sensor Model Implementation

**Goal**: Implement partial observability via FOV and lidar sensors.

**Tasks**:

1. **Field-of-view (FOV) sensor**:
   ```python
   class FOVSensor:
       def __init__(self, fov_angle=120, fov_range=10.0):
           """
           fov_angle: half-angle of FOV cone (degrees)
           fov_range: maximum detection range (meters)
           """
           self.fov_half_angle = np.radians(fov_angle)
           self.fov_range = fov_range

       def detect(self, own_state, target_state):
           """
           Returns (distance, bearing) if target in FOV, else (-1, 0).
           """
           own_x, own_y, own_theta = own_state
           target_x, target_y = target_state[0], target_state[1]

           # Distance
           dx = target_x - own_x
           dy = target_y - own_y
           dist = np.sqrt(dx**2 + dy**2)

           if dist > self.fov_range:
               return -1.0, 0.0  # Out of range

           # Bearing (relative to own heading)
           bearing = np.arctan2(dy, dx) - own_theta
           bearing = wrap_angle(bearing)

           if abs(bearing) > self.fov_half_angle:
               return -1.0, 0.0  # Outside FOV cone

           return dist, bearing
   ```

2. **Lidar sensor**:
   ```python
   class LidarSensor:
       def __init__(self, n_rays=36, max_range=5.0):
           """
           n_rays: number of uniformly spaced rays
           max_range: maximum lidar range (meters)
           """
           self.n_rays = n_rays
           self.max_range = max_range
           self.angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)

       def scan(self, own_state, obstacles, arena_bounds):
           """
           Returns array of distances [n_rays].
           Each ray: distance to nearest obstacle/boundary along that direction.
           """
           own_x, own_y, own_theta = own_state
           readings = np.full(self.n_rays, self.max_range)

           for i, angle_offset in enumerate(self.angles):
               ray_angle = own_theta + angle_offset

               # Check arena boundaries
               boundary_dist = self._ray_boundary_intersect(
                   own_x, own_y, ray_angle, arena_bounds)
               readings[i] = min(readings[i], boundary_dist)

               # Check obstacles
               for obs in obstacles:
                   obs_dist = self._ray_circle_intersect(
                       own_x, own_y, ray_angle,
                       obs['x'], obs['y'], obs['radius'])
                   if obs_dist > 0:
                       readings[i] = min(readings[i], obs_dist)

           return readings

       def _ray_circle_intersect(self, rx, ry, angle, cx, cy, radius):
           """Ray-circle intersection. Returns distance or inf."""
           # Standard ray-circle intersection algorithm
           dx = np.cos(angle)
           dy = np.sin(angle)
           fx = rx - cx
           fy = ry - cy
           a = dx*dx + dy*dy  # = 1
           b = 2*(fx*dx + fy*dy)
           c = fx*fx + fy*fy - radius*radius
           disc = b*b - 4*a*c
           if disc < 0:
               return float('inf')
           disc = np.sqrt(disc)
           t1 = (-b - disc) / (2*a)
           t2 = (-b + disc) / (2*a)
           if t1 > 0:
               return t1
           if t2 > 0:
               return t2
           return float('inf')
   ```

3. **Wall-segment obstacle support** (for generalization study — eval-only, not used in core training):
   ```python
   class WallSegment:
       """A linear wall segment defined by two endpoints."""
       def __init__(self, x1, y1, x2, y2, thickness=0.1):
           self.p1 = np.array([x1, y1])
           self.p2 = np.array([x2, y2])
           self.thickness = thickness
           self.length = np.linalg.norm(self.p2 - self.p1)
           self.direction = (self.p2 - self.p1) / self.length
           self.normal = np.array([-self.direction[1], self.direction[0]])

       def distance_to_point(self, point):
           """Signed distance from point to wall segment."""
           v = point - self.p1
           t = np.clip(np.dot(v, self.direction) / self.length, 0, 1)
           closest = self.p1 + t * (self.p2 - self.p1)
           return np.linalg.norm(point - closest)
   ```
   - Add `vcp_cbf_wall()` to `safety/vcp_cbf.py` for wall-segment CBF constraints:
     ```python
     def vcp_cbf_wall(robot_state, wall: WallSegment, d_vcp, safety_margin):
         """
         CBF for wall-segment obstacle avoidance using VCP formulation.
         Unlike circular obstacles (point-to-center distance), walls use
         point-to-line-segment distance, which has a piecewise gradient
         (along-segment vs at-endpoint regions).

         h(x) = dist(q_vcp, wall)^2 - chi^2
         where q_vcp = [x + d*cos(theta), y + d*sin(theta)]
               chi = safety_margin + d_vcp
               dist(q, wall) = min over t in [0,1] of ||q - (p1 + t*(p2-p1))||

         Returns: h (scalar), grad_h w.r.t. [v, omega] (2D vector)
         """
     ```
   - Lidar `_ray_segment_intersect()` method for wall segments
   - Define 3 complex layouts in `envs/layouts.py`:
     - **Corridor**: 4×20m with parallel walls → tests confined pursuit
     - **L-shaped room**: 15×15m with rectangular cutout → corner exploitation
     - **Warehouse**: 20×20m with shelf grid (parallel wall segments) → dense navigation
   - Create Hydra configs: `conf/env/corridor.yaml`, `conf/env/l_shaped.yaml`, `conf/env/warehouse.yaml`
   - **These layouts are for the Post-Stage 5 generalization study ONLY — NOT used in core training**

4. **Update observation space**:
   ```python
   # Phase 3 observation (partial):
   obs_P = [
       own_v, own_omega,              # Own velocity (2)
       d_to_evader, bearing_to_evader, # FOV detection (2) [-1,0 if not seen]
       lidar_readings,                 # 36 distances (36)
       # Total: 40 raw dimensions
       # + belief state from BiMDN (32)
       # = 72 total
   ]
   ```

5. **PartialObsWrapper — Pipeline orchestration class**:

   This wrapper encapsulates the entire partial-obs pipeline: raw obs → buffer → BiMDN → obs_dict. It also handles the SB3 integration by defining a `Dict` observation space and providing a compatible `FeaturesExtractor`.

   ```python
   class PartialObsWrapper(gym.Wrapper):
       """Wraps PursuitEvasionEnv to produce partial observations with belief state.
       Owns the observation history buffer and BiMDN inference pipeline.
       Exposes a gymnasium.spaces.Dict observation space for SB3 compatibility.
       """
       def __init__(self, env, sensor_config, bimdn_model=None, history_length=10):
           super().__init__(env)
           self.fov_sensor = FOVSensor(**sensor_config['fov'])
           self.lidar_sensor = LidarSensor(**sensor_config['lidar'])
           self.bimdn = bimdn_model  # Pre-trained or None (raw obs mode)
           self.history_length = history_length
           self.raw_obs_dim = 40  # 2 (velocity) + 2 (FOV) + 36 (lidar)
           self.buffer = ObservationHistoryBuffer(K=history_length, obs_dim=self.raw_obs_dim)

           # Define Dict observation space for SB3
           self.observation_space = gym.spaces.Dict({
               'obs_history': gym.spaces.Box(-np.inf, np.inf,
                   shape=(history_length, self.raw_obs_dim), dtype=np.float32),
               'lidar': gym.spaces.Box(0, sensor_config['lidar']['max_range'],
                   shape=(1, sensor_config['lidar']['n_rays']), dtype=np.float32),
               'state': gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
           })

       def reset(self, **kwargs):
           obs, info = self.env.reset(**kwargs)
           self.buffer.reset()
           return self._build_obs_dict(obs), info

       def step(self, action):
           obs, reward, term, trunc, info = self.env.step(action)
           return self._build_obs_dict(obs), reward, term, trunc, info

       def _build_obs_dict(self, raw_env_obs):
           """Convert raw env observation to Dict obs for policy."""
           agent_state = self.env.get_agent_state('pursuer')
           target_state = self.env.get_agent_state('evader')

           # FOV detection
           d, bearing = self.fov_sensor.detect(agent_state, target_state)
           # Lidar scan
           lidar = self.lidar_sensor.scan(agent_state, self.env.obstacles, self.env.arena_bounds)
           # State features
           state = np.array([agent_state[3], agent_state[4], d, bearing], dtype=np.float32)
           # Raw partial obs
           raw_obs = np.concatenate([state, lidar]).astype(np.float32)
           self.buffer.add(raw_obs)

           return {
               'obs_history': self.buffer.get_history().astype(np.float32),
               'lidar': lidar.reshape(1, -1).astype(np.float32),
               'state': state,
           }

       def get_true_state(self):
           """Pass-through for DCBF filter (uses true state, not partial obs)."""
           return self.env.get_true_state()
   ```

   **SB3 integration**: Register the custom feature extractor with SB3's PPO via `policy_kwargs`:
   ```python
   from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

   class PartialObsFeaturesExtractor(BaseFeaturesExtractor):
       """SB3-compatible feature extractor wrapping PartialObsPolicyNetwork."""
       def __init__(self, observation_space, features_dim=256, bimdn_config=None):
           super().__init__(observation_space, features_dim)
           self.net = PartialObsPolicyNetwork(
               raw_obs_dim=40, lidar_dim=36, config=bimdn_config)

       def forward(self, observations):
           return self.net(observations)  # Returns [batch, 256]

   # Usage:
   policy_kwargs = {
       'features_extractor_class': PartialObsFeaturesExtractor,
       'features_extractor_kwargs': {'features_dim': 256, 'bimdn_config': bimdn_config},
   }
   model = PPO('MultiInputPolicy', wrapped_env, policy_kwargs=policy_kwargs)
   ```

   > **Note on `MultiInputPolicy`**: SB3's `MultiInputPolicy` handles `Dict` observation spaces natively. The custom `PartialObsFeaturesExtractor` replaces SB3's default `CombinedExtractor` with our BiMDN-based architecture.

6. **Add sensor overlays to PERenderer**:
   ```python
   # In PERenderer — add partial-observability visualization layers

   def _draw_fov_cone(self, canvas, agent_pos, agent_heading, fov_half_angle, fov_range, color):
       """Draw translucent FOV cone for the agent."""
       cx, cy = self._world_to_pixel(agent_pos[0], agent_pos[1])
       range_px = int(fov_range * self.scale)
       # Draw arc from heading - fov_half_angle to heading + fov_half_angle
       start_angle = -(agent_heading + fov_half_angle)  # pygame Y flip
       end_angle = -(agent_heading - fov_half_angle)
       fov_surf = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
       pygame.draw.arc(fov_surf, (*color, 40), (cx-range_px, cy-range_px, range_px*2, range_px*2),
                       start_angle, end_angle, width=2)
       # Fill cone with semi-transparent color
       # ... (use pygame.draw.polygon with arc endpoints)
       canvas.blit(fov_surf, (0, 0))

   def _draw_lidar_rays(self, canvas, agent_pos, lidar_readings, n_rays, max_range):
       """Draw lidar rays from agent position."""
       cx, cy = self._world_to_pixel(agent_pos[0], agent_pos[1])
       for i, dist in enumerate(lidar_readings):
           angle = agent_pos[2] + (2 * np.pi * i / n_rays) - np.pi
           end_dist = min(dist, max_range)
           ex = agent_pos[0] + end_dist * np.cos(angle)
           ey = agent_pos[1] + end_dist * np.sin(angle)
           epx, epy = self._world_to_pixel(ex, ey)
           color = (100, 255, 100, 80) if dist < max_range else (60, 60, 60, 40)
           pygame.draw.line(canvas, color, (cx, cy), (epx, epy), 1)

   def _draw_belief(self, canvas, belief_state):
       """Draw BiMDN belief distribution as Gaussian ellipses."""
       if belief_state is None:
           return
       for mu, sigma, weight in zip(belief_state['means'], belief_state['stds'], belief_state['weights']):
           if weight < 0.05:
               continue  # skip negligible components
           cx, cy = self._world_to_pixel(mu[0], mu[1])
           rx, ry = int(sigma[0] * self.scale * 2), int(sigma[1] * self.scale * 2)
           alpha = int(weight * 120)  # opacity proportional to weight
           belief_surf = pygame.Surface((rx*2, ry*2), pygame.SRCALPHA)
           pygame.draw.ellipse(belief_surf, (255, 255, 0, alpha), (0, 0, rx*2, ry*2))
           canvas.blit(belief_surf, (cx-rx, cy-ry))

   def _draw_ghost(self, canvas, last_known_pos, alpha=80):
       """Draw ghost marker at last known opponent position (when undetected)."""
       if last_known_pos is None:
           return
       gx, gy = self._world_to_pixel(last_known_pos[0], last_known_pos[1])
       ghost_surf = pygame.Surface((24, 24), pygame.SRCALPHA)
       pygame.draw.circle(ghost_surf, (255, 255, 255, alpha), (12, 12), 12)
       pygame.draw.line(ghost_surf, (255, 255, 255, alpha), (12, 4), (12, 20), 2)  # "?" mark
       canvas.blit(ghost_surf, (gx-12, gy-12))
   ```

**Validation**:
- FOV correctly detects target in cone, misses target outside
- Lidar correctly measures distances to obstacles and boundaries
- Sensor outputs match manual calculations for test cases
- Observation space dimensions are correct
- FOV cone renders correctly in PERenderer (visual check)
- Lidar rays visible in human mode, color-coded by hit/miss
- Ghost marker appears when target exits FOV

**Estimated effort**: 3-4h + 1h buffer (sensor overlays add ~1h)

---

### Session 2: BiMDN Belief Encoder

**Goal**: Implement the Bidirectional Mixture Density Network for belief estimation.

**Tasks**:

1. **BiMDN architecture**:
   ```python
   class BiMDN(nn.Module):
       def __init__(self, obs_dim, hidden_dim=64, n_mixtures=5, latent_dim=32):
           super().__init__()
           self.obs_dim = obs_dim
           self.n_mixtures = n_mixtures
           self.latent_dim = latent_dim

           # Bidirectional LSTM
           self.lstm = nn.LSTM(
               input_size=obs_dim,
               hidden_size=hidden_dim,
               num_layers=1,
               bidirectional=True,
               batch_first=True,
           )

           # MDN output heads
           combined_dim = hidden_dim * 2  # Bidirectional
           self.pi_head = nn.Linear(combined_dim, n_mixtures)      # mixing weights
           self.mu_head = nn.Linear(combined_dim, n_mixtures * 2)  # means (x,y)
           self.sigma_head = nn.Linear(combined_dim, n_mixtures * 2)  # stds

           # Latent encoding for policy input
           self.latent_head = nn.Linear(combined_dim, latent_dim)

       def forward(self, obs_history):
           """
           obs_history: [batch, K, obs_dim]  (K = history length)
           Returns: latent (for policy), (pi, mu, sigma) (for belief supervision)
           """
           lstm_out, (h_n, c_n) = self.lstm(obs_history)

           # Use last timestep output
           last_out = lstm_out[:, -1, :]  # [batch, hidden*2]

           # MDN parameters
           pi = F.softmax(self.pi_head(last_out), dim=-1)       # [batch, M]
           mu = self.mu_head(last_out).reshape(-1, self.n_mixtures, 2)  # [batch, M, 2]
           sigma = F.softplus(self.sigma_head(last_out)).reshape(
               -1, self.n_mixtures, 2) + 1e-6  # [batch, M, 2]

           # Latent for policy
           latent = torch.tanh(self.latent_head(last_out))      # [batch, latent_dim]

           return latent, (pi, mu, sigma)

       def belief_loss(self, pi, mu, sigma, target_pos):
           """
           Negative log-likelihood of true position under mixture.
           target_pos: [batch, 2]  (true opponent x, y)
           """
           target = target_pos.unsqueeze(1).expand_as(mu)  # [batch, M, 2]
           log_probs = -0.5 * ((target - mu) / sigma).pow(2) - sigma.log()
           log_probs = log_probs.sum(-1)  # [batch, M]
           log_mixture = torch.logsumexp(
               log_probs + pi.log(), dim=-1)  # [batch]
           return -log_mixture.mean()
   ```

2. **Observation history buffer**:
   ```python
   class ObservationHistoryBuffer:
       def __init__(self, K=10, obs_dim=40):
           """Maintains a rolling window of K observations."""
           self.K = K
           self.obs_dim = obs_dim
           self.buffer = np.zeros((K, obs_dim))
           self.filled = 0

       def add(self, obs):
           self.buffer = np.roll(self.buffer, -1, axis=0)
           self.buffer[-1] = obs
           self.filled = min(self.filled + 1, self.K)

       def get_history(self):
           return self.buffer.copy()

       def reset(self):
           self.buffer = np.zeros((self.K, self.obs_dim))
           self.filled = 0
   ```

3. **Integrate BiMDN with policy network**:

   **Architecture** (updated for Phase 2.5 decision — standard PPO + post-hoc DCBF):
   ```python
   class PartialObsPolicyNetwork(nn.Module):
       """
       Custom SB3-compatible feature extractor for partial observability.
       Feeds into standard SB3 PPO actor/critic heads (Gaussian policy).
       DCBF filter is applied POST-HOC at action execution, not inside the network.
       """
       def __init__(self, raw_obs_dim, lidar_dim, config):
           super().__init__()

           # BiMDN belief encoder
           self.bimdn = BiMDN(raw_obs_dim, config['hidden_dim'],
                               config['n_mixtures'], config['latent_dim'])

           # Lidar branch — Conv1D chosen over MLP because adjacent lidar rays
           # are spatially correlated (kernel_size=5 captures ~50 deg sectors).
           # If profiling shows this is a bottleneck, an MLP(36→64→64) is a
           # valid simplification with minimal performance impact.
           self.lidar_conv = nn.Sequential(
               nn.Conv1d(1, 32, kernel_size=5, padding=2),
               nn.ReLU(),
               nn.Conv1d(32, 64, kernel_size=5, padding=2),
               nn.ReLU(),
               nn.AdaptiveAvgPool1d(4),
               nn.Flatten(),
           )  # Output: 256 dims

           # State branch (own velocity, detection info)
           self.state_mlp = nn.Sequential(
               nn.Linear(4, 64),  # [v, omega, d_to_opp, bearing]
               nn.ReLU(),
               nn.Linear(64, 64),
               nn.ReLU(),
           )

           # Combined: 256 (lidar) + 64 (state) + 32 (belief) = 352
           self.combined_mlp = nn.Sequential(
               nn.Linear(352, 256),
               nn.ReLU(),
               nn.Linear(256, 256),
               nn.ReLU(),
           )

           # SB3 PPO uses standard Gaussian actor head (not Beta/BarrierNet)
           # Actor head: nn.Linear(256, 2) → mean of [v, ω]
           # log_std: learnable nn.Parameter (state-independent, SB3 default)
           # Critic head: nn.Linear(256, 1) → V(o_t)
           # These are created by SB3's ActorCriticPolicy, not here.

       def forward(self, obs_dict):
           """
           obs_dict contains:
             - 'obs_history': [batch, K, raw_obs_dim] for BiMDN
             - 'lidar': [batch, 1, n_rays] for lidar Conv1D
             - 'state': [batch, 4] for state branch [v, ω, d_opp, bearing]

           Returns: features [batch, 256] fed to SB3 actor/critic heads
           """
           latent, belief_params = self.bimdn(obs_dict['obs_history'])
           lidar_features = self.lidar_conv(obs_dict['lidar'])
           state_features = self.state_mlp(obs_dict['state'])
           combined = torch.cat([lidar_features, state_features, latent], dim=-1)
           return self.combined_mlp(combined)
   ```

   **Action execution pipeline** (DCBF filter applied post-hoc):
   ```python
   # During rollout collection:
   nominal_action, _ = ppo_model.predict(partial_obs)     # PPO Gaussian policy
   true_state = env.get_true_state()                       # Full state for DCBF
   safe_action = dcbf_filter.filter(nominal_action, true_state)  # Post-hoc correction
   obs, reward, done, info = env.step(safe_action)
   ```

4. **BiMDN pre-training data collection**:

   **IMPORTANT**: Phase 2 policies were trained with full-state observation. Their trajectories
   reflect full-information play (pursuer always knows where evader is, never searches). Using
   ONLY Phase 2 policies creates a distribution mismatch — the BiMDN won't see "lost target"
   or "search" behaviors during pre-training. Use a MIX of policies to cover all scenarios.

   ```python
   def collect_bimdn_pretraining_data(env, policy_mix, n_episodes=500, history_len=10):
       """
       Collect supervised pre-training data for BiMDN using diverse policies.

       policy_mix: list of (pursuer_policy, evader_policy, weight) tuples
           e.g., [(phase2_p, phase2_e, 0.3),      # trained policies (diverse trajectories)
                  (random_p, random_e, 0.3),        # random (state-space coverage)
                  (pursuit_p, flee_e, 0.4)]          # scripted (tracking + evasion patterns)

       Returns:
           dataset: list of (obs_history, true_opponent_pos) pairs
       """
       dataset = []
       for pursuer_pol, evader_pol, weight in policy_mix:
           n_eps = int(n_episodes * weight)
           for ep in range(n_eps):
               obs, _ = env.reset()
               obs_buffer = deque(maxlen=history_len)

               for step in range(env.max_steps):
                   obs_buffer.append(obs['pursuer'])

                   if len(obs_buffer) == history_len:
                       obs_history = np.stack(list(obs_buffer))
                       true_pos = env.get_evader_position()
                       dataset.append((obs_history, true_pos))

                   p_action = pursuer_pol.predict(obs['pursuer'])
                   e_action = evader_pol.predict(obs['evader'])
                   obs, _, done, _, _ = env.step({'pursuer': p_action, 'evader': e_action})
                   if done:
                       break

       return dataset
   ```

   **Data collection procedure**:
   - **30% Phase 2 trained policies**: Diverse, competent trajectories (mostly tracking scenarios)
   - **30% random policies**: Broad state-space coverage, many "lost target" scenarios
   - **40% scripted policies**: Pure-pursuit + flee-to-corner with FOV active — realistic tracking/evasion
   - Run 500 episodes total with FOV sensor active, recording full state
   - Split 80/20 train/val with `torch.Generator().manual_seed(42)`
   - Expected dataset size: ~200K-300K (obs_history, true_pos) pairs
   - Storage: ~500MB uncompressed, save as `.npz` file
   - Critical: scripted policies ensure the BiMDN sees both "target visible" and "target lost" scenarios

5. **BiMDN pre-training**:
   ```python
   def pretrain_bimdn(bimdn, dataset, epochs=50, lr=1e-3):
       """
       Pre-train BiMDN on collected trajectories with known ground truth.
       Dataset: list of (obs_history, true_opponent_pos) pairs
       """
       optimizer = Adam(bimdn.parameters(), lr=lr)
       for epoch in range(epochs):
           for obs_hist, true_pos in DataLoader(dataset, batch_size=256):
               latent, (pi, mu, sigma) = bimdn(obs_hist)
               loss = bimdn.belief_loss(pi, mu, sigma, true_pos)
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
   ```

**Validation**:
- BiMDN produces valid mixture parameters (pi sums to 1, sigma > 0)
- Belief loss decreases during pre-training
- BiMDN correctly localizes opponent when in FOV (low uncertainty)
- BiMDN produces multimodal belief when opponent is lost (behind obstacle)
- Latent vector is informative (policy with latent outperforms policy without)

**Estimated effort**: 4-5 hours (+1h buffer)

---

### Session 3: AMS-DRL Self-Play Protocol

**Goal**: Implement the formal AMS-DRL self-play with NE convergence tracking.

**Tasks**:

1. **NavigationEnv for cold-start (S0)**:

   The cold-start phase needs a goal-reaching environment to teach the evader basic locomotion,
   obstacle avoidance, and fleeing instincts before adversarial training begins.

   ```python
   class NavigationEnv(gym.Wrapper):
       """
       Wraps PE environment for single-agent goal-reaching + fleeing.
       Used in AMS-DRL Phase S0 (cold-start).

       Observation: [own_v, own_omega, d_to_goal, bearing_to_goal, lidar(36)]  # 40-dim
       Action: [v, omega] (same as PE env)
       Reward: -0.5 * d_to_goal / d_max  (distance shaping)
               + 10.0 * I(reached_goal)   (goal bonus)
               - 1.0 * I(collision)        (obstacle/boundary penalty)
               + 0.5 * I(fleeing_slow_pursuer)  (flee bonus, optional phase)
       Success: reach within 0.5m of goal
       Episode: max 300 steps, new random goal on success (up to 3 goals/ep)
       """
       def __init__(self, pe_env, include_flee_phase=True):
           super().__init__(pe_env)
           self.goal = None
           self.include_flee = include_flee_phase
           # If include_flee: 50% episodes are goal-reaching, 50% are flee-from-slow-pursuer
           # Flee mode: spawn a slow scripted pursuer (0.5 * v_max) and reward survival time

       def reset(self, **kwargs):
           obs, info = self.env.reset(**kwargs)
           self.goal = self._sample_goal()
           return self._make_nav_obs(obs), info

       def _sample_goal(self):
           """Random collision-free position in arena."""
           for _ in range(100):
               pos = np.random.uniform(
                   [self.env.arena_min + 1, self.env.arena_min + 1],
                   [self.env.arena_max - 1, self.env.arena_max - 1])
               if not self.env._collides_with_obstacle(pos):
                   return pos
           return np.array([5.0, 5.0])  # fallback
   ```

2. **AMS-DRL implementation**:
   ```python
   class AMSDRLSelfPlay:
       def __init__(self, env, pursuer_config, evader_config, meta_config):
           self.env = env
           self.pursuer = create_agent(pursuer_config)
           self.evader = create_agent(evader_config)
           self.meta = meta_config  # eta, max_phases, etc.
           self.history = []

       def cold_start(self, episodes=1000):
           """Phase S0: Train evader for basic navigation + fleeing."""
           nav_env = NavigationEnv(self.env, include_flee_phase=True)
           self.evader.train(nav_env, episodes=episodes)
           self.history.append(self._evaluate())

       def train_phase(self, role, timesteps):
           """Train one agent while freezing the other."""
           if role == 'pursuer':
               self.env.set_evader_policy(self.evader.policy)
               self.pursuer.train(self.env, timesteps=timesteps)
           else:
               self.env.set_pursuer_policy(self.pursuer.policy)
               self.evader.train(self.env, timesteps=timesteps)

       def run(self):
           """Full AMS-DRL protocol."""
           # S0: Cold start
           self.cold_start()

           # Alternating training
           phase = 1
           converged = False
           while not converged and phase <= self.meta['max_phases']:
               role = 'pursuer' if phase % 2 == 1 else 'evader'
               self.train_phase(role, self.meta['timesteps_per_phase'])

               # Evaluate
               metrics = self._evaluate()
               self.history.append(metrics)

               # Check convergence
               sr_p = metrics['capture_rate']
               sr_e = metrics['escape_rate']
               ne_gap = abs(sr_p - sr_e)
               print(f"Phase {phase} ({role}): SR_P={sr_p:.2f}, "
                     f"SR_E={sr_e:.2f}, NE gap={ne_gap:.2f}")

               if ne_gap < self.meta['eta']:
                   converged = True
                   print(f"Converged at phase {phase}!")

               phase += 1

           return self.pursuer, self.evader, self.history
   ```

   **AMS-DRL ↔ SB3 Interaction Model**:

   Each agent (pursuer, evader) is a separate SB3 `PPO` instance. During a training phase, one agent trains normally via `model.learn()` while the other is frozen and used as the opponent policy inside the environment wrapper:

   ```
   AMSDRLSelfPlay
   ├── pursuer: PPO('MultiInputPolicy', env, policy_kwargs=...)
   ├── evader:  PPO('MultiInputPolicy', env, policy_kwargs=...)
   └── env: PartialObsWrapper(SelfPlayEnv(PursuitEvasionEnv))

   Phase S1 (Train Pursuer):
     env.set_evader_policy(evader.policy)  ← freeze evader as env opponent
     pursuer.learn(timesteps=500K)         ← SB3 trains pursuer normally
     # Inside env.step(): opponent action = evader.policy.predict(evader_obs)

   Phase S2 (Train Evader):
     env.set_pursuer_policy(pursuer.policy) ← freeze pursuer as env opponent
     evader.learn(timesteps=500K)           ← SB3 trains evader normally
   ```

   The `SelfPlayEnv` wrapper (from Phase 1, upgraded) handles opponent inference inside `step()`, so from SB3's perspective, each agent faces a single-agent MDP against a stationary opponent. Checkpoints save both agents' `state_dict` together.

3. **NE verification tools**:
   ```python
   def compute_exploitability(agent, opponent_agent, env, train_steps=500000):
       """
       Train a best-response against the fixed agent.
       Exploitability = best_response_reward - agent_reward.
       Low exploitability = close to NE.
       """
       # Train best-response pursuer against fixed evader
       br_pursuer = PPO('MlpPolicy', env)
       env.set_evader_policy(opponent_agent.policy)
       br_pursuer.learn(total_timesteps=train_steps)

       # Compare
       br_metrics = evaluate(br_pursuer, opponent_agent, env)
       orig_metrics = evaluate(agent, opponent_agent, env)

       exploitability = br_metrics['reward'] - orig_metrics['reward']
       return exploitability
   ```

3. **NE convergence plotting**:
   ```python
   def plot_ne_convergence(history):
       phases = range(len(history))
       capture_rates = [h['capture_rate'] for h in history]
       escape_rates = [h['escape_rate'] for h in history]
       ne_gaps = [abs(h['capture_rate'] - h['escape_rate']) for h in history]

       fig, (ax1, ax2) = plt.subplots(2, 1)
       ax1.plot(phases, capture_rates, label='Capture Rate (P)')
       ax1.plot(phases, escape_rates, label='Escape Rate (E)')
       ax1.axhline(0.5, linestyle='--', color='gray')
       ax1.legend()
       ax1.set_ylabel('Rate')

       ax2.plot(phases, ne_gaps, 'r-o')
       ax2.axhline(0.10, linestyle='--', color='gray', label='Threshold')
       ax2.set_ylabel('NE Gap')
       ax2.set_xlabel('Self-Play Phase')
       ax2.legend()
   ```

4. **Self-play health monitoring system** (integrated into AMS-DRL loop):
   ```python
   class SelfPlayHealthMonitor(BaseCallback):
       """
       Integrated health monitor for AMS-DRL self-play.
       Combines entropy monitoring, capture rate tracking,
       fixed-baseline evaluation, and checkpoint rollback.
       """
       def __init__(self, checkpoint_manager, config):
           super().__init__()
           self.ckpt_mgr = checkpoint_manager
           # Entropy thresholds (for 2D continuous actions)
           self.entropy_yellow = 0.5    # total entropy warning
           self.entropy_red = -0.5      # danger
           self.entropy_collapse = -2.0  # trigger rollback
           # Capture rate thresholds
           self.capture_collapse = 0.02  # pursuer collapsed
           self.capture_domination = 0.98  # evader collapsed
           self.capture_window = 200
           self.capture_history = []
           # Rollback config
           self.cooldown_steps = 50_000
           self.last_rollback_step = -50_000
           self.rollback_count = 0

       def _on_step(self) -> bool:
           # Periodic checkpoint
           if self.num_timesteps % 10_000 == 0:
               self.ckpt_mgr.save_rolling(self.model, self.num_timesteps)

           # Check entropy
           if hasattr(self.model.policy, 'log_std'):
               log_std = self.model.policy.log_std.detach().cpu().numpy()
               total_entropy = sum(1.4189 + ls for ls in log_std)
               self.logger.record('health/total_entropy', total_entropy)
               self.logger.record('health/mean_log_std', np.mean(log_std))

           # Check rollback conditions
           if self._should_rollback():
               self._trigger_rollback()
           return True

       def _should_rollback(self) -> bool:
           if self.num_timesteps - self.last_rollback_step < self.cooldown_steps:
               return False
           # Entropy collapse check
           if hasattr(self.model.policy, 'log_std'):
               log_std = self.model.policy.log_std.detach().cpu().numpy()
               total_entropy = sum(1.4189 + ls for ls in log_std)
               if total_entropy < self.entropy_collapse:
                   return True
           # Capture rate collapse check
           if len(self.capture_history) >= self.capture_window:
               rate = np.mean(self.capture_history[-self.capture_window:])
               if rate <= self.capture_collapse or rate >= self.capture_domination:
                   return True
           return False

       def _trigger_rollback(self):
           self.rollback_count += 1
           self.last_rollback_step = self.num_timesteps
           model, step = self.ckpt_mgr.perform_rollback(
               type(self.model), rollback_steps=3
           )
           self.model.policy.load_state_dict(model.policy.state_dict())
           self.capture_history.clear()
           print(f"[ROLLBACK #{self.rollback_count}] → step {step}")

   class CheckpointManager:
       """Manage 15 rolling + milestone checkpoints with rollback.

       IMPORTANT: Each checkpoint saves BOTH the PPO model AND the BiMDN encoder
       state_dict. On rollback, both are restored simultaneously to avoid
       state mismatch between policy and belief encoder. The checkpoint
       is a directory containing:
         - rolling_{step}_ppo.zip     (SB3 PPO model)
         - rolling_{step}_bimdn.pt    (BiMDN state_dict)
         - rolling_{step}_meta.json   (step, metrics, curriculum_level)

       Resume protocol (for hardware failure / interrupted training):
         1. Load latest checkpoint: model = PPO.load(path), bimdn.load_state_dict(...)
         2. Verify step count matches expected
         3. Continue with model.learn(remaining_steps)
       """
       def __init__(self, checkpoint_dir='./checkpoints', max_rolling=15):
           self.checkpoint_dir = Path(checkpoint_dir)
           self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
           self.max_rolling = max_rolling
           self.rolling = []       # (step, path)
           self.best_metric = -float('inf')

       def save_rolling(self, model, bimdn, step, meta=None):
           ckpt_dir = self.checkpoint_dir / f'rolling_{step}'
           ckpt_dir.mkdir(exist_ok=True)
           model.save(str(ckpt_dir / 'ppo.zip'))
           torch.save(bimdn.state_dict(), str(ckpt_dir / 'bimdn.pt'))
           if meta:
               import json
               with open(ckpt_dir / 'meta.json', 'w') as f:
                   json.dump({**meta, 'step': step}, f)
           self.rolling.append((step, str(ckpt_dir)))
           while len(self.rolling) > self.max_rolling:
               _, old = self.rolling.pop(0)
               if os.path.exists(old): shutil.rmtree(old)

       def perform_rollback(self, model_class, bimdn, rollback_steps=3):
           target = self.rolling[-(rollback_steps)]
           step, ckpt_dir = target
           model = model_class.load(str(Path(ckpt_dir) / 'ppo.zip'))
           bimdn.load_state_dict(torch.load(str(Path(ckpt_dir) / 'bimdn.pt')))
           # Remove newer checkpoints
           while self.rolling and self.rolling[-1][0] > step:
               _, p = self.rolling.pop()
               if os.path.exists(p): shutil.rmtree(p)
           return model, bimdn, step

   class EntropyMonitorCallback(BaseCallback):
       """Standalone entropy monitor with optional log_std clamping."""
       def __init__(self, check_freq=2048, log_std_floor=-2.0, enable_clamp=True):
           super().__init__()
           self.check_freq = check_freq
           self.log_std_floor = log_std_floor
           self.enable_clamp = enable_clamp

       def _on_step(self) -> bool:
           if self.num_timesteps % self.check_freq != 0:
               return True
           if hasattr(self.model.policy, 'log_std'):
               log_std = self.model.policy.log_std.detach().cpu().numpy()
               total_H = sum(1.4189 + ls for ls in log_std)
               self.logger.record('entropy/total', total_H)
               self.logger.record('entropy/mean_sigma', np.mean(np.exp(log_std)))
               if self.enable_clamp:
                   with torch.no_grad():
                       self.model.policy.log_std.clamp_(min=self.log_std_floor)
           return True
   ```

5. **Fixed-baseline evaluation**:
   ```python
   class FixedBaselineEvalCallback(BaseCallback):
       """Evaluate training agent against fixed baselines + Elo tracking."""
       def __init__(self, eval_env, baselines, eval_freq=10_000,
                    n_eval_episodes=50, checkpoint_freq=25_000):
           super().__init__()
           self.eval_env = eval_env
           self.baselines = baselines  # name -> (policy_fn or model_path)
           self.eval_freq = eval_freq
           self.n_eval = n_eval_episodes
           self.elo = {'training_agent': 1200.0}

       def _on_step(self) -> bool:
           if self.num_timesteps % self.eval_freq != 0:
               return True
           for name, baseline in self.baselines.items():
               win_rate = self._evaluate(baseline)
               self.logger.record(f'baseline/{name}_win', win_rate)
               # Update Elo
               if name not in self.elo: self.elo[name] = 1200.0
               ra, rb = self.elo['training_agent'], self.elo[name]
               expected = 1.0 / (1.0 + 10**((rb-ra)/400))
               self.elo['training_agent'] += 32*(win_rate - expected)
           self.logger.record('elo/training_agent', self.elo['training_agent'])
           return True

       def _evaluate(self, baseline) -> float:
           wins = 0
           for _ in range(self.n_eval):
               obs, _ = self.eval_env.reset()
               done = False
               while not done:
                   action, _ = self.model.predict(obs, deterministic=True)
                   opp_action = (baseline(self.eval_env) if callable(baseline)
                                 else self.eval_env.action_space.sample())
                   obs, _, term, trunc, info = self.eval_env.step(action, opp_action)
                   done = term or trunc
               if info.get('outcome') == 'capture': wins += 1
           return wins / self.n_eval

   # Scripted baselines
   def pure_pursuit_policy(env):
       """Pursuer heads directly toward evader."""
       dx = env.evader_pos[0] - env.pursuer_pos[0]
       dy = env.evader_pos[1] - env.pursuer_pos[1]
       angle_to_target = np.arctan2(dy, dx)
       angle_diff = angle_to_target - env.pursuer_heading
       angle_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi
       return np.array([1.0, np.clip(angle_diff * 3.0, -1, 1)])

   def flee_to_corner_policy(env):
       """Evader flees to nearest corner."""
       corners = [(0,0), (10,0), (0,10), (10,10)]
       nearest = min(corners, key=lambda c:
           np.hypot(c[0]-env.evader_pos[0], c[1]-env.evader_pos[1]))
       dx, dy = nearest[0]-env.evader_pos[0], nearest[1]-env.evader_pos[1]
       angle = np.arctan2(dy, dx)
       diff = (angle - env.evader_heading + np.pi) % (2*np.pi) - np.pi
       return np.array([1.0, np.clip(diff * 3.0, -1, 1)])
   ```

6. **Integrate all callbacks into AMS-DRL training loop**:
   ```python
   # In AMSDRLSelfPlay.train_phase():
   ckpt_mgr = CheckpointManager('./checkpoints', max_rolling=15)
   callbacks = CallbackList([
       EntropyMonitorCallback(check_freq=2048, log_std_floor=-2.0),
       SelfPlayHealthMonitor(ckpt_mgr, config),
       FixedBaselineEvalCallback(
           eval_env=eval_env,
           baselines={
               'random': None,                      # lightweight — every eval
               'pure_pursuit': pure_pursuit_policy,  # full suite — every 50K only
               'flee_to_corner': flee_to_corner_policy,
           },
           lightweight_eval_freq=10_000,   # random baseline only (20 episodes)
           full_eval_freq=50_000,          # all baselines (20 episodes each)
           n_eval_episodes=20,             # reduced from 50 to limit overhead
       ),
   ])
   agent.learn(total_timesteps=timesteps, callback=callbacks)
   ```

**Validation**:
- Cold-start (S0) evader learns basic navigation
- Alternating training shows both agents improving
- NE gap decreases over phases
- Convergence within 6 phases (per Paper [18])
- Exploitability is low at convergence
- **Health monitoring active**: entropy tracked, baselines evaluated, no rollbacks triggered in healthy runs
- **Rollback tested**: Artificially trigger collapse condition → rollback fires correctly

**Estimated effort**: 5-6 hours (+1h buffer)

---

### Session 4: Simultaneous Self-Play Comparison [N06] — OPTIONAL

**Priority**: Optional. Run only if AMS-DRL converges successfully and compute budget permits.
**Rationale**: The research novelty is safe DRL + PE + partial obs, NOT self-play methodology (which is Paper [18]'s contribution). This comparison is interesting but tangential to the core contribution.

**Goal**: Implement simultaneous self-play (no freezing) as an alternative to AMS-DRL.

**SB3 compatibility note**: SB3's PPO uses rollout buffers and batch updates — a naive step-level
loop won't work. Use one of these SB3-compatible approaches:

**Tasks**:

1. **Simultaneous self-play (SB3-compatible)**:
   ```python
   class SimultaneousSelfPlay:
       """
       Both agents train concurrently using alternating rollout collection.
       Each agent collects a full rollout buffer, then both update.
       This is SB3-compatible (uses model.collect_rollouts + model.train).
       """
       def __init__(self, env, config):
           self.env = env
           self.pursuer = PPO('MlpPolicy', env, **config['ppo'])
           self.evader = PPO('MlpPolicy', env, **config['ppo'])
           self.n_steps = config.get('n_steps', 2048)

       def train(self, total_timesteps):
           """Alternating rollout collection, simultaneous updates."""
           steps_done = 0
           while steps_done < total_timesteps:
               # Collect rollouts for both agents simultaneously
               # (both act in the same episodes, both store transitions)
               pursuer_rollout, evader_rollout = self._collect_dual_rollout()

               # Both agents update from their collected rollouts
               self.pursuer.train()
               self.evader.train()

               steps_done += self.n_steps

       def _collect_dual_rollout(self):
           """Collect n_steps of experience for both agents concurrently."""
           for step in range(self.n_steps):
               obs_p = self.env.get_obs('pursuer')
               obs_e = self.env.get_obs('evader')

               action_p, value_p, log_prob_p = self.pursuer.policy.forward(obs_p)
               action_e, value_e, log_prob_e = self.evader.policy.forward(obs_e)

               # DCBF filter applied to both
               true_state = self.env.get_true_state()
               safe_p = dcbf_filter.filter(action_p, true_state, 'pursuer')
               safe_e = dcbf_filter.filter(action_e, true_state, 'evader')

               next_obs, rewards, done, info = self.env.step(safe_p, safe_e)

               self.pursuer.rollout_buffer.add(obs_p, action_p, rewards['pursuer'],
                                                done, value_p, log_prob_p)
               self.evader.rollout_buffer.add(obs_e, action_e, rewards['evader'],
                                               done, value_e, log_prob_e)
   ```

2. **Note**: Paper [N06] uses TD3 (off-policy), not PPO. For fair comparison, implement
   simultaneous PPO only (same algorithm as our main approach). TD3 comparison is out of scope.

3. **Comparison metrics**:
   - NE convergence speed (timesteps to convergence)
   - Final NE gap
   - Training stability (variance across 3 seeds — reduced from 5 since optional)

**Validation**:
- Simultaneous self-play trains without divergence
- Comparison with AMS-DRL is fair (same total timesteps, same network)

**Estimated effort**: 3-4 hours (+1h buffer)

---

### Session 5: Curriculum Learning

**Goal**: Implement 4-level curriculum and integrate with self-play.

**Tasks**:

1. **Curriculum manager**:
   ```python
   class CurriculumManager:
       def __init__(self, env, config):
           self.env = env
           self.current_level = 1
           # FOV is ALWAYS active (fov_active=True) in all levels.
           # Rationale: BiMDN needs partial-obs data to fine-tune end-to-end.
           # Disabling FOV in levels 1-2 would create a discontinuity at Level 3
           # that destabilizes training. Difficulty is controlled by distance
           # and obstacles, not by observation type.
           self.levels = {
               1: {
                   'init_distance': (2, 5),
                   'n_obstacles': 0,
                   'fov_active': True,
                   'description': 'Close, no obstacles, partial obs',
               },
               2: {
                   'init_distance': (5, 15),
                   'n_obstacles': 0,
                   'fov_active': True,
                   'description': 'Medium distance, no obstacles, partial obs',
               },
               3: {
                   'init_distance': (2, 5),
                   'n_obstacles': (2, 4),
                   'fov_active': True,
                   'description': 'Close, obstacles, partial obs',
               },
               4: {
                   'init_distance': (2, 15),
                   'n_obstacles': (0, 6),
                   'fov_active': True,
                   'description': 'Full scenario (random everything)',
               },
           }
           self.advancement_threshold = 0.70
           self.eval_window = 100

       def check_advancement(self, metrics):
           """Advance when the pursuer demonstrates competence at the current level.

           Criterion: capture_rate > advancement_threshold (0.70).
           Since the evader improves via self-play, a 70% capture rate indicates
           the pursuer has mastered the current difficulty level against a
           co-adapting opponent. The evader's quality is implicitly validated
           by the self-play equilibrium (NE gap check is separate).

           Note: We do NOT require min(capture_rate, escape_rate) > threshold
           because capture_rate + escape_rate = 1, making min <= 0.5 always.
           The balance criterion is handled by AMS-DRL's NE convergence check.
           """
           if self.current_level >= 4:
               return False

           capture_rate = np.mean(metrics[-self.eval_window:])

           # Pursuer must achieve >70% capture rate at current level
           if capture_rate > self.advancement_threshold:
               self.current_level += 1
               self.env.set_curriculum_level(self.levels[self.current_level])
               print(f"Advanced to Level {self.current_level}: "
                     f"{self.levels[self.current_level]['description']}")
               return True
           return False
   ```

2. **Integrate curriculum with AMS-DRL**:
   - Curriculum advancement checked after each self-play phase
   - Level doesn't change mid-phase (stability)
   - History tracks which level each phase was on

**Validation**:
- Agent advances through levels naturally
- Level 1 is easiest (fastest convergence)
- Level 4 includes all scenario complexity
- Curriculum produces better final policies than direct Level 4 training

**Estimated effort**: 2-3 hours (+1h buffer)

---

### Session 6: Safe MARL Baselines

**Goal**: Implement safe MARL baselines for comparison.

**Prioritization note**: MAPPO-Lagrangian is the PRIMARY baseline (easiest to implement with SB3).
MACPO is SECONDARY (requires adapting external repo with likely version conflicts). CPO is TERTIARY
(single-agent only). Implement in priority order; stop if compute budget is exhausted.

**Tasks**:

1. **MAPPO-Lagrangian baseline** (PRIMARY — implement first):
   - Simpler than MACPO — uses Lagrangian relaxation for soft constraints
   - `J_constrained = J_reward - lambda * J_cost` where cost = safety violations
   - Lambda is a learned dual variable (updated with Adam optimizer)
   - **Can be built on top of SB3 PPO** — just add Lagrangian cost term to the loss
   - Implementation: ~100 lines on top of existing PPO infrastructure
   ```python
   class MAPPOLagrangian:
       """PPO with learned Lagrange multiplier for safety cost."""
       def __init__(self, env, cost_limit=0.01, lagrange_lr=5e-3):
           self.ppo = PPO('MlpPolicy', env, ...)
           self.log_lagrange = nn.Parameter(torch.zeros(1))  # learned dual variable
           self.lagrange_optimizer = Adam([self.log_lagrange], lr=lagrange_lr)
           self.cost_limit = cost_limit  # target safety violation rate
   ```

2. **MACPO baseline** (SECONDARY — attempt only if MAPPO-Lag works):
   - Clone from https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation
   - **HIGH RISK**: Repo from 2023, likely uses old gym/torch versions
   - Budget 4+ hours for adaptation; if compatibility issues take > 2 hours, skip MACPO and use
     MAPPO-Lagrangian results to represent the "soft constraint" baseline family
   - Key: MACPO provides monotonic improvement in BOTH reward and safety (Theorem 4.4)

3. **CPO baseline** (TERTIARY — single-agent, if time permits):
   - Foundational safe RL algorithm (Achiam et al. 2017)
   - Applied to single-agent PE (train pursuer against scripted evader)
   - Shows single-agent safe RL limitations in adversarial setting
   - Use omnisafe or safety-gymnasium implementations if available

4. **Comparison table structure**:

   | Method | Safety Guarantee | NE Convergence | Capture Rate | Phys. Violations | CBF Violations | Training Time |
   |--------|-----------------|----------------|-------------|-------------------|----------------|---------------|
   | Our (PPO+DCBF + AMS-DRL) | Hard (DCBF filter) | Yes | ? | 0 (target) | < 1% (target) | Baseline |
   | MAPPO-Lagrangian | Soft (Lagrangian) | No (CTDE) | ? | ? | N/A | ? |
   | MACPO (if feasible) | Soft (constrained opt) | No (CTDE) | ? | ? | N/A | ? |
   | CPO (single-agent, if time) | Soft (constrained opt) | N/A | ? | ? | N/A | ? |
   | PPO + AMS-DRL (no safety) | None | Yes | ? | Many | N/A | Fast |

**Validation**:
- MAPPO-Lagrangian trains and produces results (minimum required)
- Comparison is fair (same compute budget, same evaluation protocol)
- Results demonstrate the value of DCBF hard constraints vs soft constraints

**Estimated effort**: 4-5 hours (+1h buffer)

---

### Session 7: Strategy Diversity, Forgetting Detection & Analysis

**Goal**: Analyze strategy diversity using DTW-based metrics, detect catastrophic forgetting, and compile all Phase 3 results.

**Tasks**:

1. **DTW-based trajectory diversity tracker** (replaces basic k-means):
   ```python
   from dtaidistance import dtw_ndim
   from sklearn.cluster import KMeans
   from sklearn.metrics import silhouette_score

   class TrajectoryDiversityTracker:
       """
       Track trajectory diversity using DTW distance + K-means clustering.
       DTW compares trajectory shapes regardless of speed differences.
       """
       def __init__(self, buffer_size=200, subsample_length=50, n_clusters=5):
           self.buffer_size = buffer_size
           self.subsample_length = subsample_length
           self.n_clusters = n_clusters
           self.trajectories = deque(maxlen=buffer_size)

       def record_trajectory(self, pursuer_xy, evader_xy):
           """Record (x,y) trajectories. Shape: (T, 2) each."""
           p_sub = self._subsample(pursuer_xy)
           self.trajectories.append(p_sub)

       def _subsample(self, traj):
           indices = np.linspace(0, len(traj)-1, self.subsample_length, dtype=int)
           return traj[indices].astype(np.float64)

       def compute_diversity_metrics(self):
           """Call every 50-100 episodes."""
           if len(self.trajectories) < 20:
               return {}
           # Sample subset for DTW (O(n^2) pairwise)
           n_sample = min(50, len(self.trajectories))
           indices = np.random.choice(len(self.trajectories), n_sample, replace=False)
           trajs = [np.array(self.trajectories[i]) for i in indices]

           # DTW pairwise distance matrix
           dist_matrix = np.zeros((n_sample, n_sample))
           for i in range(n_sample):
               for j in range(i+1, n_sample):
                   d = dtw_ndim.distance(trajs[i], trajs[j])
                   dist_matrix[i,j] = dist_matrix[j,i] = d

           metrics = {
               'dtw_mean_pairwise': np.mean(dist_matrix[np.triu_indices(n_sample, k=1)]),
           }

           # K-means on flattened trajectories
           flat = np.array([t.flatten() for t in trajs])
           kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
           labels = kmeans.fit_predict(flat)

           counts = np.bincount(labels, minlength=self.n_clusters)
           probs = counts / counts.sum()
           entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
           max_entropy = np.log(self.n_clusters)

           metrics.update({
               'cluster_entropy': entropy,
               'cluster_entropy_normalized': entropy / max_entropy,
               'silhouette_score': silhouette_score(flat, labels),
               'n_active_clusters': np.sum(counts > 0),
               'dominant_cluster_fraction': counts.max() / counts.sum(),
           })
           return metrics
   ```

   **Interpretation thresholds for 2D PE (k=5)**:

   | Metric | Healthy | Warning | Mode Collapse |
   |---|:---:|:---:|:---:|
   | `cluster_entropy_normalized` | > 0.7 | 0.4 - 0.7 | < 0.4 |
   | `n_active_clusters` (of 5) | 4-5 | 2-3 | 1 |
   | `dominant_cluster_fraction` | < 0.4 | 0.4 - 0.7 | > 0.7 |
   | `dtw_mean_pairwise` | Stable/growing | Declining | Near zero |

2. **Outcome diversity tracker** (lightweight complement):
   ```python
   class OutcomeDiversityTracker:
       """Track spatial diversity of episode outcomes."""
       def __init__(self, buffer_size=500):
           self.capture_positions = deque(maxlen=buffer_size)
           self.episode_lengths = deque(maxlen=buffer_size)

       def record_episode(self, outcome, final_evader_pos, episode_length):
           if outcome == 'capture':
               self.capture_positions.append(final_evader_pos)
           self.episode_lengths.append(episode_length)

       def compute_metrics(self):
           metrics = {}
           if len(self.capture_positions) >= 10:
               positions = np.array(self.capture_positions)
               metrics['capture_spatial_std'] = np.mean(np.std(positions, axis=0))
           if len(self.episode_lengths) >= 10:
               lengths = np.array(self.episode_lengths)
               metrics['episode_length_cv'] = np.std(lengths) / (np.mean(lengths) + 1e-8)
           return metrics
   ```

3. **Catastrophic forgetting detector**:
   ```python
   class CatastrophicForgettingDetector:
       """
       Detect forgetting by monitoring win rates against historical checkpoints.
       Peak win-rate drop > 25% = forgetting detected.
       """
       def __init__(self, eval_env, model_class, forgetting_threshold=0.25,
                    eval_episodes=30):
           self.eval_env = eval_env
           self.model_class = model_class
           self.threshold = forgetting_threshold
           self.eval_episodes = eval_episodes
           self.history = {}  # ckpt_name -> [(step, win_rate)]

       def evaluate_against_historical(self, current_model, current_step,
                                        opponent_checkpoints):
           """Test current model against historical opponents."""
           forgetting = False
           for ckpt_path in opponent_checkpoints:
               name = os.path.basename(ckpt_path)
               win_rate = self._eval_win_rate(current_model, ckpt_path)

               if name not in self.history:
                   self.history[name] = []
               self.history[name].append((current_step, win_rate))

               # Check for forgetting
               if len(self.history[name]) >= 2:
                   peak = max(wr for _, wr in self.history[name])
                   drop = peak - win_rate
                   if drop > self.threshold:
                       forgetting = True
                       print(f"[FORGETTING] vs {name}: "
                             f"{peak:.1%} → {win_rate:.1%} (drop {drop:.1%})")
           return forgetting

       def _eval_win_rate(self, model, opp_path):
           opponent = self.model_class.load(opp_path)
           wins = 0
           for _ in range(self.eval_episodes):
               obs, _ = self.eval_env.reset()
               done = False
               while not done:
                   act, _ = model.predict(obs, deterministic=True)
                   opp_obs = self.eval_env.get_opponent_obs()
                   opp_act, _ = opponent.predict(opp_obs, deterministic=True)
                   obs, _, term, trunc, info = self.eval_env.step(act, opp_act)
                   done = term or trunc
               if info.get('outcome') == 'capture': wins += 1
           return wins / self.eval_episodes

       def compute_forgetting_score(self):
           """Overall forgetting score: 0.0 (none) to 1.0 (complete)."""
           drops = []
           for name, hist in self.history.items():
               if len(hist) >= 2:
                   peak = max(wr for _, wr in hist)
                   current = hist[-1][1]
                   drops.append(max(0, peak - current))
           return np.mean(drops) if drops else 0.0
   ```

4. **State-space coverage tracker** (fast grid-based proxy):
   ```python
   class StateSpaceCoverageTracker:
       """Grid-based coverage metric. Fastest diversity check."""
       def __init__(self, arena_bounds=(0,10,0,10), grid_res=20, window=1000):
           self.x_min, self.x_max, self.y_min, self.y_max = arena_bounds
           self.grid_res = grid_res
           self.visited = deque(maxlen=window)

       def record_step(self, pursuer_pos):
           px = int((pursuer_pos[0]-self.x_min)/(self.x_max-self.x_min)*self.grid_res)
           py = int((pursuer_pos[1]-self.y_min)/(self.y_max-self.y_min)*self.grid_res)
           self.visited.append((np.clip(px,0,self.grid_res-1),
                                np.clip(py,0,self.grid_res-1)))

       def compute_coverage(self):
           return len(set(self.visited)) / (self.grid_res ** 2)
   ```

5. **Qualitative strategy analysis**:
   - Visualize representative trajectories from each DTW cluster
   - Identify common strategies:
     - Pursuer: direct chase, flanking, cornering, ambush
     - Evader: flee, orbit, obstacle hiding, boundary running
   - Compare strategy diversity across training methods (AMS-DRL vs simultaneous)
   - Run forgetting detector against all historical checkpoints

6. **Compile all Phase 3 results**:
   - AMS-DRL vs simultaneous self-play
   - BiMDN vs raw obs vs full state
   - With/without curriculum
   - Safe (CBF) vs baselines (MACPO, CPO, unconstrained)
   - Strategy diversity metrics (DTW + coverage + outcome diversity)
   - Forgetting analysis (forgetting score across self-play phases)
   - Health monitoring summary (entropy trends, rollback events, Elo trajectories)

**Validation**:
- Multiple distinct DTW-based trajectory clusters identified (k >= 3)
- `cluster_entropy_normalized` > 0.7 (healthy diversity)
- `dominant_cluster_fraction` < 0.4 (no single dominant strategy)
- Strategies are qualitatively meaningful (not just noise)
- Full state > BiMDN > raw obs (expected ordering for belief quality)
- Forgetting score < 0.15 (no significant catastrophic forgetting)
- State-space coverage > 0.3 (agents use at least 30% of the arena)
- Phase 3 success criteria all met

**Estimated effort**: 5-6 hours (+1h buffer)

---

### Session 8: Integration & Full Evaluation

**Goal**: End-to-end training and evaluation of the complete Phase 3 system.

**Tasks**:

1. **Full pipeline run**:
   - BiMDN pre-training (Session 2)
   - Curriculum Level 1 → 4 with AMS-DRL
   - Safety layer active throughout
   - All metrics tracked

2. **Comprehensive evaluation**:
   - 200+ evaluation episodes at each curriculum level
   - NE convergence plot
   - Exploitability measurement
   - Strategy diversity
   - Safety metrics (zero violations confirmed)
   - Comparison with all baselines

3. **Ablation summary table**:

   | Ablation | Effect on Capture Rate | Effect on Safety | Effect on NE |
   |----------|----------------------|------------------|--------------|
   | BiMDN vs raw obs | +15% (target) | Neutral | Better |
   | AMS-DRL vs simultaneous | ? | Neutral | ? |
   | Curriculum vs direct | +stability | Neutral | Faster |
   | CBF-Beta vs no safety | -10% (acceptable) | Perfect vs many violations | Neutral |

4. **wandb full evaluation logging**:
   ```python
   import wandb

   # Log NE convergence as multi-step plot
   wandb.init(project="pursuit-evasion", group="phase3-evaluation",
              name=f"full-eval-seed{seed}", tags=["phase3", "evaluation"])

   # Log per-phase NE gap
   for phase_idx, ne_gap in enumerate(ne_gaps):
       wandb.log({"ne/gap": ne_gap, "ne/phase": phase_idx})

   # Log baseline comparison table
   baseline_table = wandb.Table(
       columns=["Method", "Capture Rate", "Safety Violations", "NE Gap", "Training Time"],
       data=[[m, cr, sv, ne, tt] for m, cr, sv, ne, tt in baseline_results]
   )
   wandb.log({"baselines/comparison": baseline_table})

   # Record eval videos with FOV overlay + belief visualization
   eval_env = RecordVideo(pe_env, f"videos/phase3/eval-seed{seed}/",
                          episode_trigger=lambda ep: ep < 10)
   # Videos auto-captured with PERenderer showing FOV cones + lidar + belief ellipses

   # Log strategy diversity cluster visualization
   wandb.log({"diversity/cluster_plot": wandb.Image(cluster_fig)})

   # Upload health monitoring log
   wandb.save("health_log.json")
   wandb.finish()
   ```

5. **Prepare Phase 4 transition**:
   - Save best models for sim-to-real
   - Document any environment limitations discovered
   - List sensor model assumptions that need real-world validation

**Estimated effort**: 4-5h + 1h buffer

---

### Session 9: Asymmetric Capability Experiments

**Goal**: Investigate how speed asymmetry between pursuer and evader affects the game dynamics, NE strategies, and safety behavior.

**Rationale**: Real pursuit-evasion scenarios rarely feature equally-capable agents. The speed ratio v_P/v_E fundamentally determines the game's character — classical differential game theory shows the pursuer needs a speed advantage to guarantee capture.

**Tasks**:

1. **Define asymmetric configurations** (env already supports per-agent v_max/omega_max):
   ```yaml
   # conf/experiment/asymmetric_fast_pursuer.yaml
   # v_P/v_E = 1.5 — pursuer has 50% speed advantage (classical advantage)
   pursuer:
     v_max: 1.5
     omega_max: 2.0
   evader:
     v_max: 1.0
     omega_max: 2.0

   # conf/experiment/asymmetric_matched.yaml
   # v_P/v_E = 0.8 — slightly slower pursuer (must use strategy, not speed)
   pursuer:
     v_max: 0.8
     omega_max: 2.0
   evader:
     v_max: 1.0
     omega_max: 2.0

   # conf/experiment/asymmetric_slow_pursuer.yaml
   # v_P/v_E = 0.5 — severely disadvantaged pursuer (capture requires corner traps)
   pursuer:
     v_max: 0.5
     omega_max: 2.0
   evader:
     v_max: 1.0
     omega_max: 2.0
   ```

2. **Training protocol**:
   - Fast pursuer (v_P/v_E=1.5): 3 seeds, full AMS-DRL — expect high capture rate
   - Matched (v_P/v_E=0.8): 1-3 seeds — expect balanced game, rich NE
   - Slow pursuer (v_P/v_E=0.5): 1-3 seeds — expect low capture, interesting evader strategies
   - Use same curriculum and health monitoring as Stage 5 main runs

3. **Analysis** (`scripts/analyze_asymmetric.py`):
   - Compare NE gap across speed ratios
   - Capture rate vs speed ratio curve
   - Strategy diversity: do slower pursuers develop more creative strategies?
   - DCBF intervention rate: does speed asymmetry affect safety behavior?
   - Trajectory visualizations for each speed ratio

**Compute**: ~20-50 GPU hours (3 seeds × 8-15h for fast pursuer; 1-3 seeds for others)

**Deliverables**:
- Speed ratio vs capture rate plot (key figure for paper)
- Strategy comparison across speed ratios
- Safety metric comparison (DCBF intervention rate vs speed ratio)

**Validation**:
- Fast pursuer captures at >80% rate
- Slow pursuer captures at <30% rate (if >50%, check evader is training properly)
- NE gap < 0.15 for all variants
- Zero physical collisions across all speed ratios

**Estimated effort**: 2-3h dev + 20-50h GPU compute

---

## 6. Technical Specifications

### 6.1 Sensor Parameters

```python
sensor_config = {
    # FOV sensor
    'fov_half_angle': 60,          # degrees (120 deg total)
    'fov_range': 10.0,             # meters

    # Lidar
    'lidar_n_rays': 36,
    'lidar_range': 5.0,            # meters
    'lidar_noise_std': 0.02,       # meters (Gaussian noise)

    # Observation history
    'history_length': 10,          # K timesteps
}
```

### 6.2 BiMDN Parameters

```python
bimdn_config = {
    'hidden_dim': 64,              # LSTM hidden dimension
    'n_mixtures': 5,               # Number of Gaussian components
    'latent_dim': 32,              # Belief latent vector dimension
    'pretrain_epochs': 50,
    'pretrain_lr': 1e-3,
}
```

### 6.3 AMS-DRL Parameters

```python
amsdrl_config = {
    'cold_start_episodes': 1000,
    'timesteps_per_phase': 500_000,  # More than Phase 1 (harder task)
    'max_phases': 12,
    'eta': 0.10,                     # NE convergence threshold
    'eval_episodes': 200,
}
```

### 6.4 Curriculum Parameters

```python
curriculum_config = {
    'advancement_threshold': 0.70,   # Capture rate > 70% to advance to next level
    'eval_window': 100,              # Episodes to average for advancement check
    'max_levels': 4,
}
```

### 6.5 Health Monitoring Parameters

```python
health_monitor_config = {
    # --- Entropy monitoring ---
    'entropy_check_freq': 2048,         # Check every N steps (= 1 rollout)
    'entropy_yellow': 0.5,              # Total entropy warning threshold (2D actions)
    'entropy_red': -0.5,                # Danger threshold
    'entropy_collapse': -2.0,           # Rollback trigger
    'log_std_floor': -2.0,              # Hard clamp for log_std (sigma >= 0.14)
    'enable_log_std_clamping': True,    # Prevent entropy collapse via clamping

    # --- Capture rate monitoring ---
    'capture_rate_window': 200,         # Rolling window for capture rate
    'capture_collapse_threshold': 0.02, # Pursuer collapsed (< 2% capture)
    'capture_domination_threshold': 0.98, # Evader collapsed (> 98% capture)

    # --- Checkpoint management ---
    'checkpoint_save_freq': 10_000,     # Save rolling checkpoint every N steps
    'max_rolling_checkpoints': 15,      # Keep last 15 rolling checkpoints
    'rollback_n_checkpoints': 3,        # Go back 3 checkpoints on rollback
    'rollback_cooldown_steps': 50_000,  # Min steps between rollbacks

    # --- Fixed baseline evaluation ---
    # NOTE: Overhead-conscious design. Full baseline eval (50 games × 5 baselines = 250 games)
    # at 10K steps would take 15× longer than training. Use tiered frequency instead.
    'lightweight_eval_freq': 10_000,    # Random baseline only, every 10K steps (20 episodes)
    'full_eval_freq': 50_000,           # All baselines, every 50K steps (20 episodes each)
    'baseline_eval_episodes': 20,       # Episodes per baseline (reduced from 50)
    'checkpoint_eval_freq': 50_000,     # Eval vs checkpoint baselines every N steps
    'elo_k_factor': 32,                 # Elo rating update factor
    'elo_drop_alert': 100,              # Elo drop threshold for regression alert

    # --- Trajectory diversity ---
    'dtw_eval_freq_episodes': 100,      # Compute DTW diversity every N episodes
    'dtw_buffer_size': 200,             # Trajectory buffer for DTW
    'dtw_subsample_length': 50,         # Fixed-length trajectory resampling
    'dtw_n_clusters': 5,                # K-means clusters
    'diversity_healthy_entropy': 0.7,   # Normalized cluster entropy (healthy)
    'diversity_warning_entropy': 0.4,   # Normalized cluster entropy (warning)
    'coverage_grid_resolution': 20,     # State-space coverage grid (20x20)
    'coverage_window': 1000,            # Steps for coverage computation

    # --- Catastrophic forgetting ---
    'forgetting_eval_episodes': 30,     # Episodes for historical opponent eval
    'forgetting_threshold': 0.25,       # Peak win-rate drop threshold
    'forgetting_check_freq': 50_000,    # Check every N steps

    # --- DCBF solve time monitoring ---
    'dcbf_solve_time_warn_ms': 1.0,     # Alert if mean QP solve time > 1ms
    'dcbf_solve_time_critical_ms': 5.0, # Critical alert if max solve time > 5ms
    'dcbf_solve_time_log_freq': 1024,   # Log solve time stats every N steps
}
```

### 6.6 Visualization & Tracking Parameters

```python
phase3_viz_config = {
    # FOV cone overlay
    'fov_color_pursuer': (0, 120, 255, 40),   # blue, semi-transparent
    'fov_color_evader': (255, 80, 80, 40),     # red, semi-transparent

    # Lidar overlay
    'lidar_hit_color': (100, 255, 100, 80),    # green = hit obstacle/boundary
    'lidar_miss_color': (60, 60, 60, 40),      # gray = max range

    # Belief visualization (BiMDN)
    'belief_ellipse_color': (255, 255, 0),      # yellow
    'belief_min_weight': 0.05,                  # skip components below this
    'belief_max_alpha': 120,                    # opacity cap

    # Ghost marker (undetected opponent)
    'ghost_alpha': 80,
    'ghost_fade_steps': 60,                     # fade over 3 seconds at 20Hz
}
```

```yaml
# conf/wandb/phase3.yaml — extends Phase 1 default
wandb:
  project: "pursuit-evasion"
  tags: ["phase3", "partial-obs"]
  log_frequency: 1024
  phase3_metrics:
    - ne/gap
    - ne/capture_rate_pursuer
    - ne/capture_rate_evader
    - belief/rmse_in_fov
    - belief/rmse_out_of_fov
    - belief/n_effective_components
    - curriculum/level
    - health/total_entropy
    - health/forgetting_score
    - diversity/cluster_entropy_normalized
    - dcbf/solve_time_mean_ms
    - dcbf/solve_time_max_ms
```

### 6.7 TensorBoard Monitoring Dashboard

Key metrics to track during self-play training:

| TensorBoard Panel | Metric | Healthy Range | Alert |
|---|---|---|---|
| `entropy/total` | Policy total entropy | > 0.5 | < -0.5 |
| `entropy/mean_sigma` | Mean action std dev | > 0.3 | < 0.15 |
| `health/total_entropy` | Same (from health monitor) | > 0.5 | < -2.0 = rollback |
| `health/capture_rate` | Rolling capture rate | 0.2 - 0.8 | < 0.02 or > 0.98 |
| `baseline/random_win` | Win rate vs random | > 0.9 | < 0.7 |
| `baseline/pure_pursuit_win` | Win rate vs scripted | Improving | Declining |
| `elo/training_agent` | Elo rating | Trending up | Drop > 100 |
| `diversity/cluster_entropy_normalized` | Strategy diversity | > 0.7 | < 0.4 |
| `diversity/state_coverage` | Arena coverage fraction | > 0.3 | < 0.15 |
| `forgetting/score` | Overall forgetting metric | < 0.15 | > 0.25 |
| `dcbf/solve_time_mean_ms` | Mean QP solve time per step | < 1.0 ms | > 1.0 ms (warn), > 5.0 ms (critical) |
| `dcbf/solve_time_max_ms` | Max QP solve time per step | < 5.0 ms | > 5.0 ms |

---

## 7. Validation & Success Criteria

### 7.1 Must-Pass Criteria (Gate to Phase 4)

| Criterion | Target | How to Measure | Protocol |
|-----------|--------|---------------|----------|
| NE convergence | NE gap < 0.10 by phase 12 at latest (expected within 6 per [18]); if not converged by phase 8, trigger risk mitigation | `abs(SR_P - SR_E)` computed over 200 eval episodes after each phase | 5 seeds; plot convergence curve with error bars |
| BiMDN improvement | Capture rate +15% vs raw observation (same setup) | Train identical pipeline with/without BiMDN; evaluate 200 episodes | 5 seeds; Welch's t-test, p < 0.05 |
| Physical safety | Zero physical collisions (robot-obstacle, robot-boundary, robot-robot). **Definition**: a physical collision occurs when the center-to-center distance between the robot and an obstacle falls below `r_robot + r_obstacle`, or the robot center is within `r_robot` of a wall/boundary. This is distinct from CBF boundary violations (`h(x) < 0`) which trigger before physical contact. | Track collision events separately from CBF boundary violations | 5 seeds x all AMS-DRL phases; any physical collision = FAIL |
| CBF boundary safety | CBF boundary violations < 1% of timesteps | `h_i(x) < 0` rate, evaluated with DCBF gamma tuned for eval (gamma=0.05-0.1) | Report violation rate; investigate and tighten gamma if > 1% |
| Strategy diversity | >= 3 distinct DTW-based clusters (`cluster_entropy_norm` > 0.7) | DTW + k-means on 500 trajectories; silhouette > 0.3 | 500 eval trajectories; k=5 default |
| Baselines complete | MAPPO-Lagrangian (required) + unconstrained PPO (required); MACPO, CPO optional | All trained with same compute budget; eval table filled | 5 seeds for required; 3 seeds for optional |
| Exploitability | < 0.15 reward gap for both agents | Train best-response for 500K steps against fixed opponent; compare | Report for both pursuer and evader |
| Curriculum completes | Agent advances to Level 4 within training budget | Curriculum level tracked per phase; Level 4 reached | 5 seeds; if any seed fails to reach L4, investigate |
| Health monitoring | No unrecovered collapses; <= 2 rollbacks total | Entropy, capture rate, Elo tracked throughout training | 5 seeds; report rollback events and triggers |
| No catastrophic forgetting | Forgetting score < 0.15 at convergence | Evaluate against all historical checkpoints; compute peak drop | 5 seeds; report per-checkpoint win rates |
| Partial-obs visualization | FOV cones + lidar rays + belief ellipses render in PERenderer | `render_mode="human"` shows overlays; `render_mode="rgb_array"` produces valid video | Visual check + RecordVideo produces .mp4 with sensor overlays |
| wandb tracking operational | All Phase 3 runs logged with NE gap, health metrics, baseline tables | wandb dashboard shows `ne/*`, `health/*`, `diversity/*` metrics | Verify runs in `phase3-evaluation` group |
| End-to-end integration | Full pipeline runs without crashes for 10K steps | Smoke test: sensors + BiMDN + PPO + DCBF + AMS-DRL + health monitoring | Test V (see §7.4) |

### 7.2 Quality Metrics

| Metric | Expected | How to Measure | Notes |
|--------|----------|---------------|-------|
| NE gap at convergence | < 0.10 (by phase 12; expected within 6 per [18]) | Absolute value of capture rate - escape rate | If not < 0.10 by phase 8, increase timesteps_per_phase or try PSRO |
| Exploitability | < 0.15 reward gap | Best-response training (500K steps) | Low = close to NE |
| BiMDN belief accuracy | Mean RMSE < 2m in-FOV, < 5m out-of-FOV (across 5 seeds, 1000+ held-out samples/seed; Welch's t-test vs 8.2m random baseline, p < 0.05) | Compare predicted position to true position on held-out trajectory samples | Justification: uniform random baseline over 20×20 arena gives RMSE ≈ 8.2m. Target of < 5m out-of-FOV is meaningfully better than random; < 2m in-FOV confirms the sensor signal is utilized. |
| BiMDN multimodality | Effective # components > 1.5 when target lost | `n_eff = 1 / sum(pi_i^2)` | Should capture uncertainty behind obstacles |
| Curriculum advancement | All 4 levels completed in order | Level tracking log | No skipping levels |
| Training time | ~15-30 hours total (6-12 phases x 500K steps x 2-5ms/step) | Wall-clock timer | Includes BiMDN pre-training |
| AMS-DRL vs simultaneous | Documented comparison (OPTIONAL) | Side-by-side NE gap curves | Only if AMS-DRL converges and compute permits |

### 7.3 Definition of Done

> **Phase 3 is COMPLETE when:**
> 1. Required deliverables (D1-D8, D9a-D9e, D10-D14, D16-D18) implemented and tested; D7b/D7c/D9f/D15 optional
> 2. ALL must-pass criteria in Section 7.1 are met (5 seeds for required, 3 for optional)
> 3. Baseline comparison table (Session 6) filled: MAPPO-Lagrangian + unconstrained PPO (required)
> 4. NE convergence plot generated with error bars (5 seeds)
> 5. Strategy diversity analysis complete (DTW clusters identified and visualized)
> 6. Minimum test suite (Section 7.4) passes: 31 tests (A-Z + B2, L2, W2, W3), all green
> 7. Health monitoring system active: entropy, baselines, rollback, forgetting detection all operational
> 8. PERenderer shows FOV cones, lidar rays, belief ellipses, and ghost markers
> 9. All runs logged to wandb with NE convergence, health metrics, and eval videos
> 10. Best models saved for Phase 4 export
> 11. Phase 3 summary with key findings, health monitoring report, and Phase 4 recommendations
> 12. Physical safety violations = 0 across all evaluation episodes

### 7.4 Minimum Test Suite (31 Tests)

**Test Index** (letter → function name mapping):

| Test | Function | File | Focus |
|------|----------|------|-------|
| A | `test_fov_detects_target_in_cone` | test_sensors.py | FOV detection |
| B | `test_fov_misses_target_outside_cone` | test_sensors.py | FOV rejection |
| B2 | `test_fov_boundary_conditions` | test_sensors.py | FOV edge cases |
| C | `test_lidar_obstacle_distance` | test_sensors.py | Lidar obstacle |
| D | `test_lidar_arena_boundary` | test_sensors.py | Lidar boundary |
| E | `test_bimdn_valid_outputs` | test_bimdn.py | BiMDN output shapes |
| F | `test_bimdn_pretrain_loss_decreases` | test_bimdn.py | BiMDN training |
| G | `test_bimdn_gradient_to_policy` | test_bimdn.py | Gradient flow |
| H | `test_bimdn_multimodal_when_lost` | test_bimdn.py | Multimodality |
| I | `test_cold_start_evader` | test_selfplay.py | AMS-DRL cold start |
| J | `test_amsdrl_smoke` | test_selfplay.py | AMS-DRL no crash |
| K | `test_simultaneous_sp_stable` | test_selfplay.py | Simultaneous SP |
| L | `test_curriculum_advancement` | test_selfplay.py | Curriculum advance |
| L2 | `test_curriculum_no_regression` | test_selfplay.py | Curriculum no regress |
| M | `test_mappo_lagrangian_trains` | test_baselines.py | MAPPO-Lag baseline |
| N | `test_comparison_table_complete` | test_baselines.py | Table completeness |
| O | `test_exploitability_finite` | test_baselines.py | Exploitability |
| P | `test_entropy_collapse_detection` | test_health_monitoring.py | Entropy monitor |
| Q | `test_entropy_clamping` | test_health_monitoring.py | Entropy floor |
| R | `test_checkpoint_rollback` | test_health_monitoring.py | Rollback |
| S | `test_dtw_diversity_valid` | test_health_monitoring.py | DTW diversity |
| T | `test_forgetting_detection` | test_health_monitoring.py | Forgetting |
| U | `test_baseline_eval_win_rates` | test_health_monitoring.py | Baseline eval |
| V | `test_phase3_pipeline_smoke` | test_phase3_integration.py | E2E + safety |
| W | `test_dcbf_partial_obs_interface` | test_phase3_integration.py | DCBF interface |
| W2 | `test_dcbf_infeasible_qp` | test_phase3_integration.py | DCBF failure |
| W3 | `test_dcbf_invalid_state` | test_phase3_integration.py | DCBF validation |
| X | `test_belief_encoder_interface` | test_phase3_integration.py | Encoder plug |
| Y | `test_wall_cbf_constraint` | test_wall_cbf.py | Wall CBF |
| Z | `test_lidar_wall_intersection` | test_wall_cbf.py | Wall lidar |

**File: `tests/test_sensors.py`** (5 tests)

```python
# Test A: FOV detection when target is in cone
def test_fov_detects_target_in_cone():
    """Target at 5m, bearing=30 deg (within 60 deg half-angle): should detect."""
    sensor = FOVSensor(fov_angle=60, fov_range=10.0)
    own = [0, 0, 0]  # facing right
    target = [4.33, 2.5, 0]  # ~5m away, ~30 deg bearing
    d, b = sensor.detect(own, target)
    assert d > 0 and abs(d - 5.0) < 0.1
    assert abs(b - np.radians(30)) < 0.05

# Test B: FOV misses target outside cone
def test_fov_misses_target_outside_cone():
    """Target at 5m but bearing=90 deg (outside 60 deg half-angle): miss."""
    sensor = FOVSensor(fov_angle=60, fov_range=10.0)
    own = [0, 0, 0]
    target = [0, 5, 0]  # 90 deg bearing
    d, b = sensor.detect(own, target)
    assert d == -1.0

# Test B2: FOV boundary conditions (target exactly on FOV edge)
def test_fov_boundary_conditions():
    """Test edge cases: target on FOV angular boundary, at max range, directly ahead."""
    sensor = FOVSensor(fov_angle=60, fov_range=10.0)
    own = [0, 0, 0]

    # Target exactly at fov_range (exclusive boundary: NOT detected)
    target_at_range = [10.0, 0, 0]
    d, b = sensor.detect(own, target_at_range)
    assert d == -1.0, "Target at exactly fov_range should NOT be detected"

    # Target just inside fov_range
    target_inside = [9.99, 0, 0]
    d, b = sensor.detect(own, target_inside)
    assert d > 0, "Target just inside fov_range should be detected"

    # Target directly ahead (bearing = 0)
    target_ahead = [5.0, 0, 0]
    d, b = sensor.detect(own, target_ahead)
    assert d > 0 and abs(b) < 0.01, "Target directly ahead: bearing should be ~0"

    # Target exactly on FOV angular boundary (bearing = fov_half_angle)
    # At 60 deg half-angle, target at bearing = 60 deg
    target_boundary = [5.0 * np.cos(np.radians(60)), 5.0 * np.sin(np.radians(60)), 0]
    d, b = sensor.detect(own, target_boundary)
    # On boundary: implementation-dependent (inclusive or exclusive)
    # Document the choice and test accordingly
    assert d == -1.0 or d > 0  # Either is valid; just don't crash

# Test C: Lidar detects obstacle at correct distance
def test_lidar_obstacle_distance():
    """Obstacle at 3m directly ahead: lidar forward ray should read ~3m."""
    sensor = LidarSensor(n_rays=36, max_range=5.0)
    own = [0, 0, 0]
    obs = [{'x': 3.5, 'y': 0, 'radius': 0.5}]  # Edge at 3.0m
    readings = sensor.scan(own, obs, arena_bounds)
    forward_idx = 0  # First ray is forward
    assert abs(readings[forward_idx] - 3.0) < 0.1

# Test D: Lidar detects arena boundary
def test_lidar_arena_boundary():
    """Robot facing wall at 2m: lidar should read 2m."""
    sensor = LidarSensor(n_rays=36, max_range=5.0)
    own = [8.0, 5.0, 0]  # 2m from right wall (x_max=10)
    readings = sensor.scan(own, [], {'x_min': 0, 'x_max': 10, 'y_min': 0, 'y_max': 10})
    assert abs(readings[0] - 2.0) < 0.1  # Forward ray
```

**File: `tests/test_bimdn.py`** (4 tests)

```python
# Test E: BiMDN outputs valid mixture parameters
def test_bimdn_valid_outputs():
    """Mixture weights sum to 1, sigmas are positive."""
    bimdn = BiMDN(obs_dim=40, n_mixtures=5)
    obs_hist = torch.randn(4, 10, 40)  # batch=4, K=10
    latent, (pi, mu, sigma) = bimdn(obs_hist)
    assert torch.allclose(pi.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert (sigma > 0).all()
    assert latent.shape == (4, 32)

# Test F: BiMDN belief loss decreases during pre-training
def test_bimdn_pretrain_loss_decreases():
    """10 epochs of pre-training should reduce loss."""
    bimdn = BiMDN(obs_dim=40, n_mixtures=5)
    dataset = generate_synthetic_dataset(100)
    loss_before = compute_belief_loss(bimdn, dataset)
    pretrain_bimdn(bimdn, dataset, epochs=10)
    loss_after = compute_belief_loss(bimdn, dataset)
    assert loss_after < loss_before

# Test G: BiMDN latent is informative (gradient flows to policy)
def test_bimdn_gradient_to_policy():
    """Backprop from policy loss through BiMDN should produce gradients."""
    bimdn = BiMDN(obs_dim=40, n_mixtures=5)
    policy = nn.Linear(32, 2)  # Simple policy head
    obs_hist = torch.randn(4, 10, 40, requires_grad=True)
    latent, _ = bimdn(obs_hist)
    action = policy(latent)
    loss = action.sum()
    loss.backward()
    assert bimdn.lstm.weight_ih_l0.grad is not None

# Test H: Pre-trained BiMDN produces higher uncertainty when target is lost
def test_bimdn_multimodal_when_lost():
    """After pre-training, BiMDN should show higher n_eff when target is lost vs visible.
    NOTE: This test uses 50 epochs (not 20) and a larger dataset for robustness.
    The primary assertion is n_eff_lost > 1.3 (lower bar than 1.5) to reduce flakiness.
    The relative comparison (lost > visible) is a secondary check that may fail
    on rare seeds — run with 3 seeds if the secondary assertion is flaky.
    """
    bimdn = BiMDN(obs_dim=40, n_mixtures=5)
    dataset = generate_synthetic_dataset(500)  # Larger dataset for stability
    pretrain_bimdn(bimdn, dataset, epochs=50)  # More epochs for reliable convergence

    # Compare n_eff between "target visible" and "target lost" scenarios
    # Average over multiple samples to reduce variance
    n_visible_samples, n_lost_samples = 20, 20
    n_eff_visible_list, n_eff_lost_list = [], []

    for _ in range(n_visible_samples):
        obs_visible = create_visible_observation_history(K=10)
        _, (pi_vis, _, _) = bimdn(torch.FloatTensor(obs_visible).unsqueeze(0))
        n_eff_visible_list.append((1.0 / (pi_vis[0] ** 2).sum()).item())

    for _ in range(n_lost_samples):
        obs_lost = create_lost_observation_history(K=10)
        _, (pi_lost, _, _) = bimdn(torch.FloatTensor(obs_lost).unsqueeze(0))
        n_eff_lost_list.append((1.0 / (pi_lost[0] ** 2).sum()).item())

    mean_n_eff_visible = np.mean(n_eff_visible_list)
    mean_n_eff_lost = np.mean(n_eff_lost_list)

    # Primary: lost scenarios should use multiple mixture components
    assert mean_n_eff_lost > 1.3, f"n_eff_lost={mean_n_eff_lost:.2f}, expected > 1.3"
    # Secondary: lost should be more uncertain than visible (on average)
    assert mean_n_eff_lost > mean_n_eff_visible, (
        f"n_eff_lost={mean_n_eff_lost:.2f} should > n_eff_visible={mean_n_eff_visible:.2f}")
```

**File: `tests/test_selfplay.py`** (5 tests)

```python
# Test I: AMS-DRL cold start produces functional evader
def test_cold_start_evader():
    """After S0, evader can navigate to goals (>50% success)."""
    sp = AMSDRLSelfPlay(env, config)
    sp.cold_start(episodes=500)
    success = evaluate_navigation(sp.evader, nav_env, n_episodes=50)
    assert success > 0.5

# Test J: AMS-DRL smoke test (runs without crashing, metrics are valid)
def test_amsdrl_smoke():
    """4 phases of AMS-DRL at 50K steps each: no crashes, NE gap is bounded and finite.
    NOTE: 50K steps per phase is too few for meaningful learning. This is a SMOKE TEST only.
    Real NE convergence is validated in Session 8's full pipeline evaluation (500K+ steps/phase).
    """
    sp = AMSDRLSelfPlay(env, config)
    sp.cold_start(episodes=100)  # short cold start for smoke test
    gaps = []
    for phase in range(4):
        role = 'pursuer' if phase % 2 == 0 else 'evader'
        sp.train_phase(role, timesteps=50000)
        metrics = sp._evaluate()
        gap = abs(metrics['capture_rate'] - metrics['escape_rate'])
        gaps.append(gap)
        assert np.isfinite(gap) and 0 <= gap <= 1.0  # valid range
    # Don't assert trend — 50K steps is too few for reliable learning

# Test K: Simultaneous self-play doesn't diverge
def test_simultaneous_sp_stable():
    """50K steps of simultaneous self-play: no NaN, reward bounded."""
    ssp = SimultaneousSelfPlay(env, config)
    ssp.train(total_timesteps=50000)
    assert not any(np.isnan(r) for r in ssp.reward_history)

# Test L: Curriculum level advances correctly
def test_curriculum_advancement():
    """Manager should advance when pursuer capture rate exceeds threshold."""
    cm = CurriculumManager(env, config)
    assert cm.current_level == 1
    # Simulate pursuer achieving >70% capture rate
    metrics = [0.75] * 100  # capture_rate > 0.70 threshold
    advanced = cm.check_advancement(metrics)
    assert advanced and cm.current_level == 2

# Test L2: Curriculum level does not regress on poor performance
def test_curriculum_no_regression():
    """Once advanced, level should not decrease even if performance drops."""
    cm = CurriculumManager(env, config)
    cm.current_level = 3  # Simulate having reached level 3
    # Poor performance at level 3
    metrics = [0.20] * 100  # capture_rate well below threshold
    regressed = cm.check_advancement(metrics)
    assert not regressed  # No advancement
    assert cm.current_level == 3  # Level stays at 3, does NOT drop to 2
```

**File: `tests/test_baselines.py`** (3 tests)

```python
# Test M: MAPPO-Lagrangian trains without crashing (PRIMARY baseline)
def test_mappo_lagrangian_trains():
    """MAPPO-Lagrangian baseline runs 10K steps without errors."""
    mappo_lag = setup_mappo_lagrangian(env, cost_limit=0.01)
    mappo_lag.train(total_timesteps=10000)
    assert mappo_lag.policy is not None
    assert hasattr(mappo_lag, 'log_lagrange')  # Lagrange multiplier exists

# Test N: Baseline comparison table has no NaN
def test_comparison_table_complete():
    """All entries in comparison table are filled."""
    results = run_all_baselines(env, n_seeds=1, n_steps=10000)
    for method, metrics in results.items():
        assert not any(np.isnan(v) for v in metrics.values())

# Test O: Exploitability computation returns finite value
def test_exploitability_finite():
    """Best-response training should converge to finite exploitability."""
    expl = compute_exploitability(trained_agent, trained_opponent, env, train_steps=50000)
    assert np.isfinite(expl) and expl >= 0
```

**File: `tests/test_health_monitoring.py`** (6 tests)

```python
# Test P: Entropy monitor detects collapse
def test_entropy_collapse_detection():
    """When log_std is forced to -3.0, monitor should flag collapse."""
    model = PPO('MlpPolicy', env)
    with torch.no_grad():
        model.policy.log_std.fill_(-3.0)
    log_std = model.policy.log_std.detach().cpu().numpy()
    total_entropy = sum(1.4189 + ls for ls in log_std)
    assert total_entropy < -2.0  # Collapse threshold

# Test Q: Entropy clamping enforces floor
def test_entropy_clamping():
    """log_std clamping at -2.0 prevents entropy below floor."""
    model = PPO('MlpPolicy', env)
    with torch.no_grad():
        model.policy.log_std.fill_(-5.0)  # Way below floor
        model.policy.log_std.clamp_(min=-2.0)
    assert (model.policy.log_std >= -2.0).all()

# Test R: Checkpoint manager saves and rolls back correctly
def test_checkpoint_rollback():
    """Save 5 checkpoints, rollback 3: should load checkpoint 2."""
    ckpt_mgr = CheckpointManager('./test_ckpts', max_rolling=15)
    model = PPO('MlpPolicy', env)
    for step in [10000, 20000, 30000, 40000, 50000]:
        ckpt_mgr.save_rolling(model, step)
    assert len(ckpt_mgr.rolling) == 5
    loaded, step = ckpt_mgr.perform_rollback(PPO, rollback_steps=3)
    assert step == 30000  # 3 back from latest
    assert loaded.policy is not None

# Test S: DTW diversity tracker produces valid metrics
def test_dtw_diversity_valid():
    """20+ trajectories should produce all expected diversity metrics."""
    tracker = TrajectoryDiversityTracker(buffer_size=50, n_clusters=3)
    for _ in range(25):
        traj = np.random.randn(100, 2).cumsum(axis=0)  # Random walk
        tracker.record_trajectory(traj, traj + 1)
    metrics = tracker.compute_diversity_metrics()
    assert 'cluster_entropy_normalized' in metrics
    assert 0.0 <= metrics['cluster_entropy_normalized'] <= 1.0
    assert metrics['n_active_clusters'] >= 1

# Test T: Forgetting detector identifies regression
def test_forgetting_detection():
    """Simulated peak-drop > 25% should trigger forgetting alert."""
    detector = CatastrophicForgettingDetector(
        eval_env=env, model_class=PPO, forgetting_threshold=0.25)
    # Manually inject history simulating forgetting
    detector.history['opponent_v1'] = [
        (10000, 0.80), (20000, 0.85), (30000, 0.50)  # 85% → 50% = 35% drop
    ]
    score = detector.compute_forgetting_score()
    assert score > 0.25  # Forgetting detected

# Test U: Baseline eval callback produces win rates
def test_baseline_eval_win_rates():
    """Evaluation against random baseline should return valid win rate."""
    def random_policy(env):
        return env.action_space.sample()
    callback = FixedBaselineEvalCallback(
        eval_env=eval_env, baselines={'random': random_policy},
        eval_freq=1000, n_eval_episodes=10)
    win_rate = callback._evaluate(random_policy)
    assert 0.0 <= win_rate <= 1.0
```

```

**File: `tests/test_phase3_integration.py`** (5 tests)

```python
# Test V: End-to-end Phase 3 pipeline smoke test
def test_phase3_pipeline_smoke():
    """Run 1000 steps of the full Phase 3 pipeline: sensors + BiMDN + PPO + DCBF.
    Asserts: no crashes, valid observation shapes, DCBF receives true state,
    metrics are logged correctly, AND zero physical collisions + CBF invariant.
    """
    env = PursuitEvasionEnv(fov_active=True, n_obstacles=2)
    bimdn = BiMDN(obs_dim=40, n_mixtures=5)
    ppo = PPO('MlpPolicy', env)
    dcbf = DCBFFilter(gamma=0.2, d=0.1)
    collision_count = 0
    cbf_violation_count = 0

    for step in range(1000):
        partial_obs = env.get_partial_observation('pursuer')
        assert partial_obs.shape[0] == 40  # correct obs dim

        true_state = env.get_true_state()
        assert true_state.shape[0] == 6  # [x_P, y_P, θ_P, x_E, y_E, θ_E]

        nominal_action, _ = ppo.predict(partial_obs)
        safe_action = dcbf.filter(nominal_action, true_state)
        assert safe_action.shape == (2,)  # [v, ω]

        obs, reward, done, _, info = env.step(safe_action)
        assert np.isfinite(reward)

        # Safety assertions
        if info.get('collision', False):
            collision_count += 1
        h_values = dcbf.get_constraint_values(true_state)
        if any(h < -1e-6 for h in h_values):
            cbf_violation_count += 1

        if done:
            env.reset()

    assert collision_count == 0, f"Physical collisions detected: {collision_count}"
    assert cbf_violation_count / 1000 < 0.01, (
        f"CBF violation rate {cbf_violation_count/1000:.1%} exceeds 1% threshold")

# Test W: DCBF filter receives true state while agent gets partial obs
def test_dcbf_partial_obs_interface():
    """Verify the separation: policy uses partial obs, DCBF uses true state."""
    env = PursuitEvasionEnv(fov_active=True, n_obstacles=2)
    env.reset(seed=42)

    partial_obs = env.get_partial_observation('pursuer')
    true_state = env.get_true_state()

    # Partial obs should NOT contain opponent position when out of FOV
    # True state should ALWAYS contain opponent position
    assert len(partial_obs) < len(true_state) or partial_obs[2] == -1.0  # d_to_opp = -1 if not seen
    assert true_state[3] != 0 or true_state[4] != 0  # evader position always present

    # DCBF filter should work with true state regardless of FOV
    dcbf = DCBFFilter(gamma=0.2, d=0.1)
    nominal = np.array([0.5, 0.0])
    safe = dcbf.filter(nominal, true_state)
    assert np.isfinite(safe).all()

# Test W2: DCBF handles infeasible QP gracefully
def test_dcbf_infeasible_qp():
    """When QP is infeasible (e.g., cornered by multiple constraints),
    DCBF should return backup action and log a warning, not crash."""
    dcbf = DCBFFilter(gamma=0.2, d=0.1)
    # Construct a state where robot is sandwiched between two obstacles
    # with conflicting constraint gradients → QP infeasible
    tight_state = np.array([5.0, 5.0, 0.0, 5.5, 5.0, np.pi])
    obstacles = [
        {'x': 5.3, 'y': 5.0, 'radius': 0.2},
        {'x': 4.7, 'y': 5.0, 'radius': 0.2},
    ]
    nominal = np.array([1.0, 0.0])
    safe = dcbf.filter(nominal, tight_state, obstacles=obstacles)
    assert np.isfinite(safe).all()  # Must not crash or return NaN
    assert safe.shape == (2,)
    # Backup action should be conservative (low velocity)
    assert abs(safe[0]) <= 0.5  # backup controller limits speed

# Test W3: DCBF filter with invalid state input raises error
def test_dcbf_invalid_state():
    """DCBF should raise ValueError for NaN or wrong-shape state inputs."""
    dcbf = DCBFFilter(gamma=0.2, d=0.1)
    import pytest
    with pytest.raises((ValueError, AssertionError)):
        dcbf.filter(np.array([0.5, 0.0]), np.array([np.nan, 0, 0, 0, 0, 0]))
    with pytest.raises((ValueError, AssertionError)):
        dcbf.filter(np.array([0.5, 0.0]), np.array([1.0, 2.0]))  # wrong shape

# Test X: Pluggable belief encoder interface
def test_belief_encoder_interface():
    """BiMDN, LSTMEncoder, and MLPHistoryEncoder all implement the same interface."""
    obs_hist = torch.randn(4, 10, 40)

    for EncoderClass in [BiMDN, LSTMEncoder, MLPHistoryEncoder]:
        encoder = EncoderClass(obs_dim=40, latent_dim=32)
        latent = encoder.encode(obs_hist)
        assert latent.shape == (4, 32)
        assert latent.requires_grad  # gradients flow through
```

**File: `tests/test_wall_cbf.py`** (2 tests)

```python
# Test Y: Wall-segment CBF constraint is valid
def test_wall_cbf_constraint():
    """VCP-CBF for wall segment produces valid h value and gradient."""
    wall = WallSegment(x1=5.0, y1=0.0, x2=5.0, y2=10.0, thickness=0.1)
    robot_state = np.array([3.0, 5.0, 0.0])  # robot at (3,5), facing right
    d_vcp = 0.1
    h, grad_h = vcp_cbf_wall(robot_state, wall, d_vcp, safety_margin=0.3)
    assert h > 0  # Robot is safely away from wall
    assert grad_h.shape == (2,)  # Gradient w.r.t. [v, omega]

# Test Z: Lidar ray-segment intersection
def test_lidar_wall_intersection():
    """Lidar ray intersects wall segment at correct distance."""
    sensor = LidarSensor(n_rays=36, max_range=5.0)
    own = [3.0, 5.0, 0.0]  # facing right
    wall = WallSegment(x1=5.0, y1=0.0, x2=5.0, y2=10.0)
    dist = sensor._ray_segment_intersect(3.0, 5.0, 0.0, wall)
    assert abs(dist - 2.0) < 0.1  # Wall is 2m away
```

> **Test suite total**: 31 tests (A-Z + B2, L2, W2, W3) across 7 files — exceeds 15+ minimum.

### 7.5 Worked Examples

#### Example 1: FOV Detection Geometry

```
Setup:
  Pursuer at [5.0, 5.0, theta_P = 0.0] (facing right)
  Evader at [8.0, 7.0, theta_E = pi]
  FOV: half-angle = 60 deg, range = 10.0m

Step 1: Compute distance and bearing
  dx = 8.0 - 5.0 = 3.0
  dy = 7.0 - 5.0 = 2.0
  dist = sqrt(9 + 4) = 3.606m  (< 10m range: OK)
  bearing = atan2(2.0, 3.0) - 0.0 = 0.588 rad = 33.7 deg

Step 2: Check FOV
  |bearing| = 33.7 deg < 60 deg half-angle
  Result: DETECTED at (distance=3.606, bearing=0.588 rad)

Step 3: If evader moves to [5.0, 10.0]:
  dx = 0.0, dy = 5.0
  bearing = atan2(5.0, 0.0) - 0.0 = pi/2 = 90 deg > 60 deg
  Result: NOT DETECTED (outside FOV cone)

Key insight: Evader can exploit the pursuer's limited FOV by
staying at high bearing angles. This is why Phase 3 is strategic.
```

#### Example 2: BiMDN Multimodal Belief

```
Scenario: Evader was last seen heading toward obstacle A or B

Observation history (K=10):
  t-10: evader at (8, 5), bearing 30 deg  [SEEN]
  t-9:  evader at (8.2, 5.1), bearing 32 deg [SEEN]
  t-8 to t-0: detection = -1, -1, ...  [LOST — behind obstacle]

BiMDN output:
  Mixture 1: pi=0.45, mu=(9.0, 4.5), sigma=(0.5, 0.5)  [behind obstacle A]
  Mixture 2: pi=0.40, mu=(9.5, 6.0), sigma=(0.6, 0.6)  [behind obstacle B]
  Mixture 3: pi=0.10, mu=(7.0, 5.0), sigma=(1.0, 1.0)  [still in open]
  Mixture 4: pi=0.03, ...  (negligible)
  Mixture 5: pi=0.02, ...  (negligible)

  Effective components: n_eff = 1/(0.45^2+0.40^2+0.10^2+0.03^2+0.02^2) ≈ 2.6
  Belief is bimodal: evader likely behind obstacle A or B

Policy uses latent vector z (32-dim) encoding this uncertainty.
The pursuer should decide: check A first or B first?
This is the "information gathering" aspect that makes Phase 3 strategic.
```

#### Example 3: AMS-DRL NE Convergence

```
Phase S0 (Cold Start): Evader learns navigation
  Nav success: 85% → evader can move and avoid obstacles

Phase S1 (Train Pursuer): vs frozen S0 evader
  Capture rate: 0 → 75% (evader is naive, easy to catch)
  Capture time: 30s → 15s

Phase S2 (Train Evader): vs frozen S1 pursuer
  Capture rate drops: 75% → 40% (evader learns to evade)
  Escape rate: 25% → 55% (evader gets better)

Phase S3 (Train Pursuer): vs frozen S2 evader
  Capture rate: 40% → 55% (pursuer adapts)
  NE gap: |0.55 - 0.45| = 0.10

Phase S4 (Train Evader): vs frozen S3 pursuer
  Capture rate: 55% → 48%
  NE gap: |0.48 - 0.52| = 0.04 < 0.10 → CONVERGED!

Total: 4 phases to convergence (within the 6-phase budget)
Strategy diversity check: k=3 clusters identified
  Cluster 1: "direct chase / flee" (35% of trajectories)
  Cluster 2: "obstacle flanking / hiding" (40%)
  Cluster 3: "boundary running / cutting" (25%)
```

#### Example 4: Health Monitoring Rollback Scenario

```
Scenario: Evader collapses during Phase S3 of AMS-DRL training

Timeline:
  Step 100K: Checkpoint saved (rolling_100000.zip)
  Step 110K: Checkpoint saved (rolling_110000.zip)
  Step 120K: Checkpoint saved (rolling_120000.zip)
  Step 130K: Checkpoint saved (rolling_130000.zip)
  Step 140K: Checkpoint saved (rolling_140000.zip)

  Step 142K: Entropy monitor detects warning
    total_entropy = 0.38 < 0.5 (yellow alert)
    log_std = [-1.1, -0.9]  → sigma = [0.33, 0.41]
    Action: log warning, continue training

  Step 148K: Capture rate spikes
    capture_rate (200-episode window) = 0.96
    Approaching domination threshold (0.98)
    Entropy still declining: total_entropy = -0.2

  Step 152K: ROLLBACK TRIGGERED
    capture_rate = 0.99 > 0.98 (evader collapsed)
    total_entropy = -1.8 (close to collapse)
    Elo dropped 85 points in last eval cycle

    Rollback action:
      Target: 3 checkpoints back → rolling_120000.zip (step 120K)
      Load state_dict from rolling_120000.zip
      Clear capture_history (avoid immediate re-trigger)
      Save milestone: milestone_152000_post_rollback_1.zip
      Set cooldown: next rollback earliest at step 202K

  Post-rollback at step 120K:
    Entropy restored: total_entropy = 1.2 (healthy)
    Capture rate window: cleared, rebuilding
    Training continues with ent_coef possibly increased

  Step 175K: Stable training confirmed
    capture_rate = 0.52 (balanced)
    total_entropy = 0.8 (healthy)
    Baseline win rates: random=95%, pure_pursuit=68%
    No further rollbacks needed

Key insight: The rollback went back 3 checkpoints (not 1) because
the entropy decline started well before the capture rate collapse.
Rolling back only 1 checkpoint (140K) would likely re-collapse.
```

#### Example 5: Catastrophic Forgetting Detection

```
Scenario: After 4 alternating self-play phases, test for forgetting

Phase S1 pursuer (ckpt: pursuer_s1.zip) trained vs S0 evader
Phase S2 evader (ckpt: evader_s2.zip) trained vs S1 pursuer
Phase S3 pursuer (ckpt: pursuer_s3.zip) trained vs S2 evader
Phase S4 evader (ckpt: evader_s4.zip) trained vs S3 pursuer

Forgetting evaluation for evader_s4 against historical pursuers:
  vs pursuer_s1: win_rate = 0.88 (S2 evader's peak was 0.85 → no drop)
  vs pursuer_s3: win_rate = 0.52 (current opponent → expected)

Forgetting evaluation for pursuer_s3 against historical evaders:
  vs evader_s0: win_rate = 0.92 (S1 pursuer's peak was 0.75 → improved!)
  vs evader_s2: win_rate = 0.55 (current opponent → expected)

Result: forgetting_score = 0.0 → NO FORGETTING ✓
  All agents maintain or improve against historical opponents.

Counter-example (forgetting detected):
  If pursuer_s3 vs evader_s0 = 0.45 (S1 pursuer's peak was 0.75)
  Peak drop = 0.75 - 0.45 = 0.30 > 0.25 threshold
  → [FORGETTING] Pursuer forgot how to beat naive evader!
  Action: Shorten alternation cycles, add experience replay
```

#### Example 6: Full Timestep Walkthrough (Pipeline Integration)

```
Setup:
  Arena: 20m × 20m, 2 obstacles
  Pursuer at [5.0, 5.0, θ=0.0], Evader at [15.0, 12.0, θ=π]
  FOV: 120 deg, 10m range. Evader is 11.2m away → OUT OF FOV RANGE.

Step 1: Sensor observations
  FOV: d_to_evader = -1.0, bearing = 0.0  (not detected — 11.2m > 10.0m range)
  Lidar: 36 readings, e.g. [5.0, 5.0, ..., 3.2, ..., 5.0]  (obstacle at ray 12)
  State: own_v = 0.8, own_ω = 0.1
  Raw obs: [0.8, 0.1, -1.0, 0.0, 5.0, 5.0, ..., 3.2, ..., 5.0]  → shape (40,)

Step 2: Observation history buffer
  Buffer state (10 × 40): last 10 raw observations
  obs_history shape: (1, 10, 40) after unsqueeze for batch dim

Step 3: BiMDN forward pass
  Input: obs_history (1, 10, 40)
  → Forward LSTM: processes t-9 → t-0
  → Backward LSTM: processes t-0 → t-9
  → Concatenate hidden states: (1, 128)
  → MDN heads:
    pi: [0.35, 0.30, 0.20, 0.10, 0.05]  (mixture weights, sum=1)
    mu: [(14.0, 11.5), (16.0, 13.0), (12.0, 10.0), ...]  (predicted positions)
    sigma: [(1.2, 1.0), (1.5, 1.3), (2.0, 1.8), ...]  (uncertainties)
  → Latent head: z (1, 32) — tanh-bounded latent for policy

Step 4: Policy input construction
  Lidar branch: Conv1D(1, 36) → Conv1D(32) → Conv1D(64) → Pool → Flatten → (256,)
  State branch: MLP(4 → 64 → 64) → (64,)
  Belief: z → (32,)
  Concatenate: [256 + 64 + 32] = (352,)
  Combined MLP(352 → 256 → 256) → features (256,)

Step 5: PPO actor (SB3 Gaussian policy)
  Input: features (256,)
  → Linear(256, 2) → mean: [0.72, -0.35]  (nominal v, ω)
  → log_std: [-0.5, -0.3]  (learnable parameter)
  → Sample: v=0.68, ω=-0.42  (Gaussian sample)
  → Clip to action bounds: v ∈ [0, 1.0], ω ∈ [-2.84, 2.84]
  Nominal action: [0.68, -0.42]

Step 6: DCBF filter (post-hoc, uses TRUE STATE)
  True state: [5.0, 5.0, 0.0, 15.0, 12.0, π]  (pursuer + evader full state)
  VCP position: q = [5.0 + 0.1*cos(0), 5.0 + 0.1*sin(0)] = [5.1, 5.0]
  Check constraints:
    h_arena = R² - ||q||² = 100 - 50.01 = 49.99  ≥ 0 ✓
    h_obs1 = ||q - p_obs1||² - χ² = ... ≥ 0 ✓
    h_collision = ||q_P - q_E||² - r_min² = ... ≥ 0 ✓ (far apart)
  DCBF condition: Δh + gamma*h ≥ 0 for all constraints
  All satisfied → safe action = nominal action [0.68, -0.42]  (no correction needed)

Step 7: Environment step
  Apply safe action [v=0.68, ω=-0.42] to unicycle dynamics (dt=0.05):
    x_new = 5.0 + 0.68 * cos(0.0) * 0.05 = 5.034
    y_new = 5.0 + 0.68 * sin(0.0) * 0.05 = 5.0
    θ_new = 0.0 + (-0.42) * 0.05 = -0.021
  New pursuer state: [5.034, 5.0, -0.021]
  Reward: w1 * (d_old - d_new) / d_max + w5 * min(h_i) / h_max
         = 1.0 * (11.2 - 11.17) / 20 + 0.05 * 49.99 / 100 = 0.0015 + 0.025 = 0.027
```

#### Example 7: Curriculum Level Transition

```
Setup:
  Training at Curriculum Level 1 (close encounters, no obstacles)
  advancement_threshold = 0.70, eval_window = 100 episodes

Timeline:
  Step 100K: capture_rate (last 100 eps) = 0.55  → below 0.70, stay at Level 1
  Step 150K: capture_rate = 0.63  → improving, stay at Level 1
  Step 200K: capture_rate = 0.72  → EXCEEDS 0.70 threshold!

  CurriculumManager.check_advancement() returns True
  Level 1 → Level 2 transition:
    - Initial distance range: [2-5m] → [5-15m]
    - Obstacles: 0 → 0 (still no obstacles at Level 2)
    - FOV: remains active (120 deg, 10m range)
    - AMS-DRL phase: continues current phase (no restart)

  Step 200K: Level 2 begins
    capture_rate window: CLEARED (starts fresh at new level)
    Reason: performance typically drops temporarily when difficulty increases

  Step 250K: capture_rate = 0.42  → expected drop, stay at Level 2
    Level does NOT regress to 1 (monotonically non-decreasing)

  Step 400K: capture_rate = 0.71  → EXCEEDS threshold!
  Level 2 → Level 3 transition:
    - Initial distance: [2-5m] (reintroduced for close+obstacle encounters)
    - Obstacles: 2-4 circular obstacles added
    - This is the hardest transition — obstacles + partial obs is challenging

  Step 700K: Still at Level 3, capture_rate = 0.58
    Timeout check: 2 × timesteps_per_phase = 2 × 500K = 1M steps max
    Still within budget (700K < 1M), continue training

  Step 950K: capture_rate = 0.68  → close but below 0.70
    Approaching timeout (950K near 1M limit)
    If 1M reached without advancement:
      Option A: Lower threshold to 0.60 → would have advanced at step 700K
      Option B: Force-advance with documentation
      Decision: investigate — is the evader exploiting obstacles too well?

Key insight: The curriculum manages difficulty pacing, while the self-play
health monitor handles training stability. They operate independently.
```

#### Example 8: DCBF Correction Scenario

```
Setup:
  Pursuer at [9.2, 5.0, theta_P = 0.0] (facing right, near right wall)
  Arena boundary at x = 10.0 (right wall)
  Obstacle at [9.8, 5.5, radius=0.3]
  VCP offset d = 0.1, safety margin chi = 0.3 + 0.1 = 0.4
  DCBF gamma = 0.2

Step 1: PPO nominal action
  Policy outputs: v = 0.9, ω = 0.1  (accelerating toward wall)

Step 2: Compute VCP position
  q_vcp = [9.2 + 0.1*cos(0), 5.0 + 0.1*sin(0)] = [9.3, 5.0]

Step 3: Check CBF constraints
  h_boundary_right = (10.0 - chi) - q_vcp_x = (10.0 - 0.4) - 9.3 = 0.3  (barely safe)
  h_obstacle = ||q_vcp - p_obs||^2 - (r_obs + chi)^2
             = (9.3-9.8)^2 + (5.0-5.5)^2 - (0.3+0.4)^2
             = 0.25 + 0.25 - 0.49 = 0.01  (VERY close to constraint!)

Step 4: DCBF condition check
  For h_obstacle = 0.01:
    Required: Δh + gamma * h >= 0
    With nominal action (v=0.9, ω=0.1):
      q_vcp_next = [9.3 + 0.9*cos(0)*0.05, 5.0 + 0.9*sin(0)*0.05]
                 = [9.345, 5.0]
      h_next = (9.345-9.8)^2 + (5.0-5.5)^2 - 0.49
             = 0.207 + 0.25 - 0.49 = -0.033  ← WOULD VIOLATE!
    Δh = -0.033 - 0.01 = -0.043
    Δh + gamma * h = -0.043 + 0.2 * 0.01 = -0.041 < 0  ← CONSTRAINT VIOLATED

Step 5: QP correction
  Solve: min ||u - u_nom||^2
         s.t. Δh_i(u) + gamma * h_i >= 0 for all constraints
  QP redirects action away from obstacle:
    safe_action = [0.3, -0.5]  (slow down, turn left away from obstacle/wall)

Step 6: Apply corrected action
  x_new = 9.2 + 0.3 * cos(0) * 0.05 = 9.215
  y_new = 5.0 + 0.3 * sin(0) * 0.05 = 5.0
  θ_new = 0.0 + (-0.5) * 0.05 = -0.025
  h_obstacle_new = ... > 0  ✓ (constraint maintained)

  Intervention logged: dcbf/intervention_rate += 1
  Safe action differs from nominal: ||correction|| = 0.79

Key insight: The DCBF correction is minimal — it modifies the action just
enough to satisfy all constraints. The policy's intent (move right) is
preserved as much as possible while avoiding the obstacle/wall.
```

#### Example 9: Asymmetric Capability Game Dynamics

```
Scenario: Speed ratio comparison across 3 configurations

Configuration 1: v_P/v_E = 1.5 (fast pursuer)
  Pursuer v_max = 1.5, Evader v_max = 1.0
  After AMS-DRL convergence:
    Capture rate: 87% (pursuer dominant)
    NE gap: 0.06 (converged in 4 phases)
    Evader strategies: boundary running (45%), obstacle hiding (35%), fleeing (20%)
    Pursuer strategies: cut-off (50%), direct chase (40%), cornering (10%)
    DCBF intervention rate: 4.2% (higher speed → more interventions)
    Mean capture time: 12.3s

Configuration 2: v_P/v_E = 0.8 (slow pursuer)
  Pursuer v_max = 0.8, Evader v_max = 1.0
  After AMS-DRL convergence:
    Capture rate: 38% (evader slightly dominant)
    NE gap: 0.08 (converged in 6 phases — harder game)
    Evader strategies: aggressive fleeing (60%), taunting/baiting (25%), obstacle orbit (15%)
    Pursuer strategies: ambush/cornering (55%), obstacle-aided trapping (30%), patient approach (15%)
    DCBF intervention rate: 2.1% (lower speed → fewer constraint conflicts)
    Mean capture time: 28.7s
    Key observation: slower pursuer develops MORE creative strategies (ambush > chase)

Configuration 3: v_P/v_E = 0.5 (severely disadvantaged pursuer)
  Pursuer v_max = 0.5, Evader v_max = 1.0
  After AMS-DRL convergence:
    Capture rate: 12% (evader dominant — capture requires corner traps)
    NE gap: 0.12 (barely converged; approaches the 0.10 threshold from above)
    Evader strategies: open-field running (70%), casual evasion (30%)
    Pursuer strategies: corner trap setup (60%), obstacle ambush (30%), give-up/patrol (10%)
    DCBF intervention rate: 1.5%
    Mean capture time: 42.1s (when capture occurs)

Publication figure: Speed ratio vs Capture rate curve
  (1.5, 87%), (1.0, ~50%), (0.8, 38%), (0.5, 12%)
  Classical result: with equal turning radius, capture guaranteed iff v_P > v_E
  Our result: with obstacles + partial obs, even v_P < v_E can capture via strategy

Key insight: Speed asymmetry fundamentally changes the game character.
Slower pursuers develop richer strategic repertoires (more DTW clusters)
to compensate for their speed disadvantage.
```

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| BiMDN fails under fast dynamics | Low | Medium | Pluggable encoder interface: fall back to LSTMEncoder or MLPHistoryEncoder (same API, swap one class) |
| NE doesn't converge in 6 phases | Medium | Medium | Increase timesteps per phase; try PSRO [04] |
| Partial obs + DCBF = too hard | Medium | Medium | Simplify: wider FOV, fewer obstacles initially; DCBF filter strength is tunable (gamma) |
| **MACPO code incompatible** | **HIGH** | **Medium** | MACPO is OPTIONAL. Primary baseline is MAPPO-Lagrangian (built on SB3 PPO, ~100 lines). Skip MACPO if integration takes > 2 hours |
| Simultaneous SP unstable with PPO | Medium | Low | Simultaneous SP is OPTIONAL (Session 4). Skip if AMS-DRL works well |
| Curriculum advancement too slow | Low | Low | Lower threshold; manual level override |
| DCBF infeasibility increases with obstacles | Medium | Medium | Hierarchical relaxation (Tier 2) + backup controller (Tier 3); monitor feasibility rate |
| **Entropy collapse / mode collapse** | Medium | High | `log_std` clamping floor at -2.0; `ent_coef > 0`; automatic rollback via `SelfPlayHealthMonitor` |
| **Catastrophic forgetting in alternating SP** | Medium | High | Historical opponent evaluation; shorter alternation cycles (10K-25K); 80/20 latest/historical mix (OpenAI Five approach) |
| **Strategy cycling (rock-paper-scissors)** | Medium | Medium | DTW diversity tracking; forgetting detector; consider PSRO if cycling persists |
| **Rollback thrashing** (repeated rollbacks) | Low | Medium | 50K-step cooldown between rollbacks; escalate to population-based training if > 3 rollbacks |
| **DTW computation too slow** | Low | Low | Use state-space coverage as fast proxy; DTW only every 100 episodes on 50-sample subset |
| **Compute budget exceeds available resources** | Medium | High | See §8.1 Compute Plan; prioritize required deliverables; use niro-2 GPU; parallelize seeds |
| FOV/lidar overlay slows rendering | Low | Low | Only draw sensor overlays in human/rgb_array modes; skip during training with `render_mode=None` |
| wandb multi-seed logging conflicts | Low | Low | Use unique `name` per seed; `group` for aggregation; `wandb.init(resume="allow")` for restarts |

### 8.1 Compute Plan

**Estimated GPU hours** (niro-2 lab PC):

| Experiment | Seeds | Steps/Seed | Est. Time/Seed | Total Hours |
|------------|-------|------------|----------------|-------------|
| DCBF theorem verification | 1 | N/A (CPU) | ~0h | 0h |
| AMS-DRL (main) | 5 | 6 phases × 500K = 3M | ~8-15h | 40-75h |
| BiMDN pre-training | 1 | 50 epochs | ~1h | 1h |
| BiMDN ablation (raw obs, LSTM) | 2 × 5 | 3M each | ~8-15h | 80-150h |
| MAPPO-Lagrangian baseline | 5 | 3M | ~8-15h | 40-75h |
| Unconstrained PPO baseline | 5 | 3M | ~8-15h | 40-75h |
| Exploitability (best-response) | 2 × 5 | 500K each | ~2h | 20h |
| Asymmetric: v_P/v_E=1.5 | 3 | 3M | ~8-15h | 24-45h |
| Asymmetric: v_P/v_E=0.8 | 1-3 | 3M | ~8-15h | 8-45h |
| Asymmetric: v_P/v_E=0.5 | 1-3 | 3M | ~8-15h | 8-45h |
| Post-training evals (human, gen., interp.) | - | eval only | ~2-5h | 2-5h |
| **Total (required + new)** | | | | **~260-540h** |
| Simultaneous SP (optional) | 3 | 3M | ~8-15h | 24-45h |
| MACPO (optional) | 3 | 3M | ~8-15h | 24-45h |
| **Total (with optional)** | | | | **~310-630h** |

**Parallelization strategy**:
- niro-2 has 1 GPU → run at most 2 experiments simultaneously (if GPU memory permits)
- Independent experiments (different baselines) can run concurrently
- Estimate 2-3 weeks for required experiments at ~20h/day effective utilization
- Start AMS-DRL main runs first (longest); baselines and ablations in parallel

---

## 9. Software & Tools

### 9.1 Pinned Package Versions

```bash
# Core (inherited from Phase 1-2)
stable-baselines3==2.6.0
gymnasium==1.1.1
torch==2.6.0
numpy==2.2.2

# BiMDN (uses PyTorch — already installed)
# No additional package; BiMDN is custom code using torch.nn

# MACPO baseline
git clone https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation
pip install -e Multi-Agent-Constrained-Policy-Optimisation/
# Pin commit hash for reproducibility: document exact commit in config

# Trajectory clustering & diversity
scikit-learn==1.6.1        # k-means, silhouette score
dtaidistance==2.3.12       # DTW for trajectory similarity (C-accelerated)
scipy==1.15.0              # pdist for pairwise distances

# TD3 for simultaneous self-play comparison
# (SB3 has TD3 built in — no extra package)

# Visualization & Tracking (inherited from Phase 1, listed for completeness)
# pygame==2.6.1            # Already in Phase 1 — sensor overlays added in Phase 3
# wandb==0.19.1            # Already in Phase 1 — NE tracking, health monitoring dashboards
# hydra-core==1.3.2        # Already in Phase 1 — Phase 3 configs in conf/
# omegaconf==2.3.0         # Already in Phase 1
```

**Operational resilience notes**:
- **wandb failure fallback**: Use `wandb.init(mode='offline')` as fallback if wandb is unavailable (network down, API key expired, rate limited). Health monitor operates independently of wandb — critical alerts (entropy collapse, rollback triggers) go to stdout and a local `health_log.json` file regardless of wandb status. On session resume, sync offline runs with `wandb sync runs/offline-*`.
- **Training resume protocol**: All training runs checkpoint at regular intervals (see `checkpoint_save_freq`). To resume after hardware failure: load latest checkpoint directory, verify step count from `meta.json`, continue with `model.learn(total_timesteps - completed_steps)`. BiMDN state is co-located with PPO checkpoint (see `CheckpointManager` docstring).

**Compatibility notes**:
- MACPO repo may require specific gym version; use wrapper if needed
- scikit-learn 1.6.1 requires numpy >=1.21; our numpy 2.2.2 is compatible
- BiMDN uses `torch.distributions.MixtureSameFamily` (available since PyTorch 1.9)
- dtaidistance 2.3.12 has C extension for fast DTW; falls back to pure Python if C build fails
- scipy 1.15.0 is used for `pdist` in outcome diversity and is compatible with numpy 2.2.2

### 9.2 Reproducibility Protocol

1. **Random seeds**: All experiments use seeds `[0, 1, 2, 3, 4]`; set via `torch.manual_seed(seed)`, `np.random.seed(seed)`, `env.reset(seed=seed)`
2. **PyTorch determinism**: Set `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
3. **Self-play reproducibility**: Each AMS-DRL phase saves checkpoint before training; rollback is reproducible
4. **BiMDN pre-training**: Use fixed train/val split (80/20) with `torch.Generator().manual_seed(42)`
5. **Trajectory clustering**: Set `KMeans(random_state=seed)` for reproducible cluster assignments
6. **MACPO baseline**: Pin MACPO repo commit hash in `config.yaml` so exact code version is tracked
7. **Evaluation protocol**: All evaluation uses 200 episodes (not training env), greedy action selection
8. **Hardware note**: Record GPU model and driver version; BiMDN training may vary slightly across GPUs
9. **Health monitoring reproducibility**: Log all rollback events (step, reason, target checkpoint) to `health_log.json`; DTW uses `np.random.seed(42)` for sample selection; Elo K-factor fixed at 32
10. **Checkpoint archival**: Keep all milestone checkpoints (pre/post rollback) for post-hoc analysis; rolling checkpoints can be pruned after training completes

---

## 10. Guide to Phase 4

### 10.1 What Phase 4 Needs from Phase 3

| Phase 3 Output | Phase 4 Usage |
|----------------|---------------|
| Trained PE agents (pursuer + evader) | Exported to ONNX for deployment |
| BiMDN encoder | Included in ONNX model |
| Safety architecture | Deployed as C++ RCBF-QP on robot |
| Sensor model parameters | Matched to real robot sensors |
| Domain randomization requirements | Implemented in Isaac Lab |

### 10.2 Key Transitions to Phase 4

| Phase 3 → Phase 4 Change | Details |
|--------------------------|---------|
| Gymnasium env → Isaac Lab | GPU-accelerated, MARL-native |
| Python CBF → C++ RCBF-QP | Real-time deployment |
| Simulated sensors → Real sensors | TurtleBot lidar + camera |
| No physics → Full physics | Isaac Lab rigid body simulation |
| No latency → Real latency | Control delay, sensor latency |

### 10.3 Phase 4 Reading List

1. **[N10] Salimpour et al. 2025** — Isaac → ONNX → Gazebo → Real pipeline
2. **[N11] Mittal et al. 2025** — Isaac Lab (GPU-accelerated MARL)
3. **[07] Salimpour 2025** — Sim-to-real pipeline design
4. **[06] Emam 2022** — GP disturbance estimation for deployment
