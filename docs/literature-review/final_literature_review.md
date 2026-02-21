# Comprehensive Literature Review: 1v1 Pursuit-Evasion Games Using Mobile Ground Robots with Deep Reinforcement Learning

## Date: 2026-02-21
## Papers Reviewed: 36 (20 original collection + 16 newly downloaded)

---

## 1. Executive Summary

This review synthesizes 36 papers spanning pursuit-evasion (PE) game theory, deep reinforcement learning (DRL), safety-constrained learning, Hamilton-Jacobi (HJ) reachability, self-play training, sim-to-real transfer, and real-robot deployment. The focus is on **1v1 pursuit-evasion games using mobile ground robots with deep RL**.

**Key findings:**
- The field is rapidly maturing, with multiple papers demonstrating **real-world PE on physical robots** (Papers 02, 08, 18, 22)
- **Safety during training and deployment** is the critical enabling technology, with CBFs and HJ reachability as the two dominant paradigms
- **Self-play** is the standard training paradigm for adversarial PE, with theoretical convergence guarantees to Nash equilibrium
- **HJ reachability + neural networks** (DeepReach family) provides game-theoretic optimal solutions for PE, now demonstrated on TurtleBots
- A clear gap exists: **no paper combines all of** safe DRL + 1v1 ground robot PE + nonholonomic dynamics + real-world deployment in a single work

---

## 2. Background and Problem Definition

### 2.1 Pursuit-Evasion Games
A pursuit-evasion game is a two-player zero-sum differential game where the pursuer seeks to minimize the time or distance to capture, while the evader maximizes it. Mathematically:

- **Dynamics**: ẋ = f(x, u_P, u_E) where x is the joint state, u_P is pursuer control, u_E is evader control
- **Objective**: min_{u_P} max_{u_E} J(x, u_P, u_E) (saddle-point / Nash equilibrium)
- **Terminal condition**: Capture when ||x_P - x_E|| ≤ r_capture

### 2.2 Ground Robot Dynamics
For mobile ground robots, the relevant kinematic models are:
- **Differential-drive (unicycle)**: ẋ = v·cos(θ), ẏ = v·sin(θ), θ̇ = ω — control inputs (v, ω)
- **Car-like (Ackermann)**: ẋ = v·cos(θ), ẏ = v·sin(θ), θ̇ = (v/L)·tan(δ) — control inputs (v, δ)
- **Dubins car**: Constant speed, turn-rate control — frequently used in HJ reachability analysis

### 2.3 Scope of Review
| Aspect | In Scope | Out of Scope |
|--------|----------|-------------|
| Agents | 1 pursuer, 1 evader | Multi-agent teams (reviewed for transferable techniques) |
| Platform | Ground mobile robots | Fixed-wing aircraft, underwater vehicles |
| Method | Deep RL (primary), HJ reachability (complementary) | Classical optimal control only |
| Environment | 2D with obstacles | 3D environments (reviewed for methodology) |
| Safety | CBF/CLF/HJR safety constraints | Unconstrained approaches |

---

## 3. Thematic Analysis

### 3.1 Direct 1v1 PE with Deep RL

**Most relevant papers**: [01], [02], [08], [12], [13], [14], [15], [22], [34]

#### 3.1.1 State of the Art
Paper [02] (Gonultas & Isler, 2024) represents the current state-of-the-art for **1v1 PE on ground robots with DRL**:
- Car-like (Ackermann) kinematics with realistic sensor constraints (limited FOV)
- BiMDN belief encoder for partial observability
- MADDPG with centralized training, decentralized execution
- Curriculum learning for training stability
- **Real deployment on F1TENTH and JetRacer platforms**

Paper [08] (Bajcsy et al., ICRA 2024) demonstrates the leading **vision-based PE approach**:
- Privileged teacher-student framework with latent intent modeling
- DAGGER distillation from omniscient teacher to camera-only student
- **Real deployment on Unitree A1 quadruped**

Paper [22] (MADR, Teoh et al., 2025) achieves **game-theoretically optimal PE on TurtleBots**:
- DeepReach + adversarial MPC for computing optimal PE strategies
- MADR-FOLLOW for long-horizon PE (500+ second engagements)
- **Real deployment on TurtleBots** (ground robots)

Paper [34] (SHADOW, La Gatta et al., 2025) introduces **information-aware PE**:
- 1v1 PE with unicycle dynamics (ground-robot-compatible)
- Multi-headed RL: TD3 for navigation + PPO for query decisions
- Opponent modeling with uncertainty estimation
- Information-exposure trade-off formulation

#### 3.1.2 Algorithm Comparison for 1v1 PE

| Paper | Algorithm | Dynamics | Obs. Type | Self-Play | Hardware |
|-------|-----------|----------|-----------|-----------|----------|
| [01] | RL+SMC+CBF | General | Full state | Stackelberg | No |
| [02] | MADDPG | Car-like | Partial (limited FOV) | Yes | F1TENTH, JetRacer |
| [08] | PPO+DAGGER | General | Vision (camera) | Teacher-student | Unitree A1 |
| [12] | DQN, DDPG | Point mass | Full state | Separate training | No |
| [13] | Critic-only RL | General | Full state | HJI-based | No |
| [15] | A-MPC+DQN | Double integrator | Full state | Nash analysis | No |
| [22] | PINN+MPC | Dubins car | Full state | Adversarial MPC | TurtleBot |
| [34] | TD3+PPO | Unicycle | Partial (query-based) | Both trained | No |

### 3.2 Safety-Constrained RL

**Key papers**: [01], [03], [05], [06], [10], [13], [14], [16], [27], [28], [30]

Safety is the most critical enabling technology for deploying PE robots. Two main paradigms emerge:

#### 3.2.1 Control Barrier Functions (CBFs)
CBFs define a safe set S = {x : h(x) ≥ 0} and enforce forward invariance via the constraint:
ḣ(x,u) + α(h(x)) ≥ 0

**Integration approaches** (from survey [03]):

| Approach | Papers | Pros | Cons |
|----------|--------|------|------|
| CBF safety filter (QP) | [05], [06] | Hard safety guarantee, modular | QP overhead, may be infeasible |
| CBF reward shaping | [05] | Soft, differentiable | No hard guarantee |
| CBF-constrained policy | [16] | Convergence + safety | Complex implementation |
| Robust CBF + GP | [06], [14] | Handles model uncertainty | GP scaling challenges |
| Closed-form CBF filter | [05] | Fast, differentiable | Limited to simple constraints |

**Paper [16]** (Suttle et al., AISTATS 2024) introduces **CBF-constrained Beta policies** — the first approach to achieve BOTH convergence guarantees AND hard safety constraints simultaneously. This is a breakthrough: truncated stochastic policies sample directly from the safe control set.

**Paper [05]** (Yang et al., Caltech 2025) proposes a **dual approach**: closed-form CBF filter + reward shaping. The RL agent learns to naturally satisfy safety constraints over time, reducing reliance on the filter.

**Paper [06]** (Emam et al., 2022) adds **GP-based disturbance estimation** to handle model uncertainty, crucial for ground robots with wheel slip and terrain variation.

#### 3.2.2 Hamilton-Jacobi Reachability
HJ reachability computes the exact safe set by solving the HJI PDE, providing the strongest safety guarantees.

**Papers [21], [22], [23], [25], [27], [35]** form a coherent HJ-RL pipeline:

| Paper | Contribution | Dimensionality |
|-------|-------------|----------------|
| [21] DeepReach | Neural HJI solver with sine activations | Up to 10D |
| [22] MADR | Adversarial MPC + DeepReach for games | Up to 20D |
| [23] Convergence | Formal proof DeepReach → true solution | Theory |
| [24] DeepReach Activations | Mixed Sine/ReLU layers for better BRT | Up to 9D |
| [25] HJ-RL Survey | Complete roadmap for HJ + RL integration | Survey |
| [27] RESPO | Reachability estimation for safe RL (stochastic) | Up to 76D |
| [31] Safe HJ MADDPG | HJ reachability + CQL for safe MARL | 3-12 agents |
| [35] NeHMO | Scalable neural HJR for multi-agent | Up to 12-DoF |

**Paper [24]** (Wang & Wu, 2023) demonstrates that **mixed Sine/ReLU activation layers** can improve DeepReach performance on high-dimensional systems — best violation rate 18.43% (down from 19.0%). First and last layers have dominant effect. Practical guidance for implementing DeepReach-based PE solvers.

**Paper [25]** (Ganai et al., 2024) is the definitive survey bridging HJ reachability and RL. Key insight: the **shielding framework** (task RL policy + HJ-based safe backup) is the most practical architecture for deployment.

**Paper [31]** (Zhu et al., JIRS 2024) integrates **HJ reachability with model-free MADDPG** via conservative Q-learning. Achieves near-zero collisions (0.06/episode) in cooperative settings and dramatically outperforms baselines under disturbance/non-cooperation (0.07 vs 1.57 for soft MADDPG). The Discounted Reach-Avoid Bellman Equation is directly applicable to PE value functions. Policy inference: 0.1ms per agent.

#### 3.2.3 Conflict-Averse Safe RL
**Paper [30]** (CASRL, Zhou et al., CAAI 2023) introduces **conflict-averse gradient manipulation** for safe robot navigation. Instead of combining goal-reaching and collision-avoidance into a single reward, CASRL separates them into dual critics with independent policy gradients, then optimizes for worst-case improvement across both. Achieves 84.9% success rate (+8.2% over baseline) with 100% real-world success on a Fetch robot. The conflict-averse technique is directly transferable to PE where pursuit objectives conflict with safety constraints.

#### 3.2.4 Combined CLF-CBF Approaches
**Paper [28]** (LBAC, Du et al., ICRA 2023) unifies CLF (stability/reachability) and CBF (safety) into a single CLBF certificate, avoiding the well-known CLF-CBF conflict. Model-free, validated on real CrazyFlie.

**Paper [13]** (Kokolakis & Vamvoudakis, CDC 2022) integrates barrier functions directly into the HJI cost function for safe PE, achieving finite-time convergence with safety guarantees.

### 3.3 Self-Play and Training Paradigms

**Key papers**: [04], [11], [18], [19], [20], [37]

#### 3.3.1 Self-Play Methods for PE
PE is inherently adversarial — both agents must be trained. The standard approaches:

| Method | Papers | Description | Convergence |
|--------|--------|-------------|-------------|
| Vanilla Self-Play | [04] | Train against copy of self | May cycle |
| Fictitious SP | [04] | Train against average of past | Converges to NE (normal-form) |
| PSRO | [04] | Meta-game Nash mixture | Converges to NE |
| AMS-DRL | [18] | Staged asynchronous alternation | Proven NE convergence |
| Self-play Iteration | [19] | Alternating optimal control | Proven NE convergence |
| MADDPG | [02], [20] | Centralized critic + decentralized actors | Standard for 1v1 |

**Paper [37]** (Yang et al., CJA 2025) provides a comprehensive **235-reference survey of RL approaches for PE games**, organizing methods by strategy learning paradigm: unilateral (fixed opponent), bilateral alternating (self-play/PSRO), and bilateral simultaneous (MADDPG/MAPPO). Key insight: **PSRO** is highlighted as the "groundbreaking" framework for strategy-rich games, while MADDPG/MAPPO show fastest convergence. The survey identifies sim-to-real transfer and safe RL as critical open challenges.

**Paper [18]** (AMS-DRL, Xiao & Feroskhan, 2024) provides the most practical self-play recipe for PE:
1. Cold-start: train evader for navigation without adversary
2. Asynchronous stages: alternately train pursuer and evader
3. Convergence criterion: |success_rate_P - success_rate_E| < η
- Proven NE convergence, validated with real drones (39.2% matches sim 37.1%)

#### 3.3.2 Opponent Modeling
- **Paper [34]** (SHADOW): TD3-based opponent position predictor with uncertainty
- **Paper [08]**: Latent intent estimation — predicts evader's future trajectory
- **Paper [11]**: LOLA — higher-order gradients to anticipate opponent learning

### 3.4 Sim-to-Real Transfer

**Key papers**: [07], [02], [08], [18], [22]

| Paper | Simulator | Real Platform | Transfer Method | Gap |
|-------|-----------|---------------|-----------------|-----|
| [07] | Isaac Sim | TurtleBot4 | Domain randomization + ONNX | Small |
| [02] | Custom | F1TENTH, JetRacer | Curriculum | ~5% |
| [08] | Custom | Unitree A1 | DAGGER distillation | ~10% |
| [18] | Unity ML-Agents | Tello Edu drones | ONNX export | ~4% |
| [22] | N/A (PINN) | TurtleBot, Crazyflie | Direct deployment | ~2% |

**Key insights for ground robot PE sim-to-real:**
1. **Domain randomization** (Paper [07]): Randomize friction, mass, sensor noise during training
2. **ONNX export** (Papers [07], [18]): Cross-platform model deployment, ~1.6-5ms inference
3. **Privileged teacher → student distillation** (Paper [08]): Train omniscient teacher, distill to sensor-limited student
4. **Isaac Sim → Gazebo → Real** pipeline (Paper [07]): Gazebo catches 80% of sim-to-real issues early

### 3.5 Hierarchical and Novel Architectures

**Key papers**: [17], [32], [34]

- **Paper [32]** (Diffusion-RL, Wu et al., 2024): Hierarchical diffusion + SAC for PE — diffusion model generates global paths, RL handles local evasion. 51.4% performance improvement.
- **Paper [17]** (ViPER, Wang et al., CoRL 2024): Graph attention network for visibility-based PE — 6-layer masked self-attention, scales to large environments.
- **Paper [34]** (SHADOW): Modular multi-headed architecture — separate navigation, communication, and opponent modeling modules.

---

## 4. Gap Analysis

### 4.1 What Exists
- 1v1 PE with DRL on ground robots: [02] (car-like), [22] (Dubins/TurtleBot)
- Safe RL with CBFs: [05], [06], [16] (single-agent)
- Safe PE with barrier functions: [01], [13], [14] (simulation only)
- Self-play for PE: [18] (proven convergence, real drones)
- HJ reachability for PE: [21], [22] (optimal, real TurtleBots)
- Vision-based PE: [08] (real quadruped)

### 4.2 What's Missing (Research Gaps)

| Gap | Description | Opportunity |
|-----|-------------|-------------|
| **G1** | No work combines CBF-safe DRL + 1v1 ground robot PE + real deployment | Directly fill this gap |
| **G2** | CBF-constrained self-play for PE — safety filters designed for single agent, not adversarial training | Extend [16] to adversarial setting |
| **G3** | HJ reachability + RL hybrid for PE — HJ provides safety, RL provides adaptability | Combine [22] and [02] approaches |
| **G4** | Nonholonomic dynamics in safe PE — most safe PE papers use simplified dynamics | Use differential-drive/Ackermann models |
| **G5** | Partial observability + safety — belief-state CBFs for PE | Extend [02] BiMDN with [06] CBF |
| **G6** | Online adaptation of PE strategies — current methods train offline | Real-time strategy adjustment |
| **G7** | Bounded rationality in DRL PE — Level-k thinking ([14]) not yet with deep RL | Combine [14] with modern DRL |

---

## 5. Recommended Research Pathways

### Pathway A: Safe Deep RL for 1v1 Ground Robot PE (HIGHEST PRIORITY)
**Fills gaps**: G1, G2, G4

**Architecture**:
```
Observation → BiMDN Belief Encoder → PPO/SAC Policy → CBF Safety Filter → Robot Actions
                                          ↕ (self-play)
Observation → BiMDN Belief Encoder → PPO/SAC Policy → CBF Safety Filter → Robot Actions
```

**Key components** (with paper references):
1. **Dynamics**: Differential-drive (unicycle) or Ackermann kinematics [02]
2. **Observation**: Limited FOV lidar/camera with BiMDN belief encoding [02]
3. **Policy**: PPO with CBF-constrained Beta policies [16] — safety during training
4. **Safety layer**: RCBF-QP with GP disturbance estimation [06] — robustness to model error
5. **Training**: AMS-DRL self-play [18] with curriculum [02]
6. **Sim-to-real**: Isaac Sim → Gazebo → ROS2 [07], ONNX deployment
7. **Hardware**: TurtleBot4 or F1TENTH

**Novelty**: First safe DRL system for 1v1 PE on ground robots with:
- Hard safety guarantees during training AND deployment
- Nonholonomic dynamics
- Partial observability
- Self-play with NE convergence

**Risk level**: MODERATE — all individual components demonstrated, integration is the challenge

**Estimated papers to produce**: 2-3 (method paper, real-robot paper, safety analysis)

---

### Pathway B: HJ Reachability + RL Hybrid for Ground Robot PE
**Fills gaps**: G3, G4

**Architecture**:
```
State → DeepReach Value Function V(x) → {
    If V(x) ≤ safety_threshold: Use HJ-optimal safe controller (from ∇V)
    Else: Use RL pursuit/evasion policy (PPO/SAC)
}
```

**Key components**:
1. **Offline**: Train DeepReach [21] / MADR [22] on robot-specific dynamics
2. **Online**: RL policy for task-level behavior (aggressive pursuit/creative evasion)
3. **Shielding**: HJ value function defines when to override RL with safe controller [25]
4. **Adaptation**: RL handles situations outside HJ training distribution

**Novelty**: Combine the formal safety of HJ reachability with the adaptability of RL for PE
- HJ provides worst-case guarantees (no one has done this for ground robot PE with RL)
- RL provides performance beyond the conservative HJ policy

**Key reference**: Survey [25] explicitly identifies this as the most promising direction

**Risk level**: MODERATE-HIGH — MADR [22] demonstrated on TurtleBots, but HJ-RL integration for PE is novel

**Estimated papers**: 2-3

---

### Pathway C: Vision-Based 1v1 PE for Ground Robots
**Fills gaps**: G5

**Architecture**:
```
Camera → CNN Encoder → Latent Intent Estimator → Actor-Critic Policy → Safety Filter → Actions
(privileged teacher with full state during training, distilled to camera-only student)
```

**Key components**:
1. **Privileged teacher**: Trained with full state info (position, velocity) [08]
2. **Latent intent**: Predict evader's future trajectory from visual history [08]
3. **Student distillation**: DAGGER from omniscient teacher to camera student [08]
4. **Safety**: CBF on estimated state from visual odometry [05]
5. **Platform**: TurtleBot with camera or F1TENTH

**Novelty**: Vision-based PE for ground robots (Paper [08] used a quadruped)
- Different perception challenges: 2D plane vs legged locomotion
- Camera + safety filter integration for ground robots

**Risk level**: MODERATE — [08] demonstrated the paradigm, adaptation to ground robot is engineering challenge

**Estimated papers**: 1-2

---

### Pathway D: Information-Aware PE with Opponent Modeling
**Fills gaps**: G5, G7

**Architecture**:
```
Observation → LSTM Memory → {
    Navigation Module (TD3): continuous control
    Sensing Module (PPO): when to actively sense opponent
    Opponent Model (TD3): predict opponent position + uncertainty
}
```

**Key components**:
1. **Multi-headed architecture** from SHADOW [34]
2. **Unicycle dynamics** (ground-robot-native) [34]
3. **Active sensing**: Trade-off between information gain and detection risk
4. **Opponent modeling**: Predict adversary with uncertainty quantification [34]
5. **Safety**: CBF constraints on navigation module [05]

**Novelty**: Extend SHADOW's PEEC formulation to physical ground robots with safety
- SHADOW [34] is simulation-only → opportunity for real deployment
- Add CBF safety layer not present in SHADOW

**Risk level**: MODERATE — strong theoretical foundation from [34], needs real-robot validation

**Estimated papers**: 1-2

---

### Pathway E: Hierarchical Diffusion-RL for PE in Complex Environments
**Fills gaps**: G6

**Architecture**:
```
Environment Map → Diffusion Model (global path planner) → {
    SAC Policy (local evasion/pursuit)
    Cost Map (detection risk)
} → Actions
```

**Key components**:
1. **Diffusion-based global planner**: Generate diverse paths [32]
2. **RL local controller**: Handle adversarial interactions [32]
3. **Risk-aware path selection**: Cost map from training experience [32]
4. **Ground robot adaptation**: Nonholonomic path constraints
5. **Safety**: CBF for collision avoidance during local control

**Novelty**: Adapt hierarchical diffusion-RL from [32] to ground robots
- [32] used 2D point agents; ground robots have kinematic constraints
- Add pursuer co-training (not present in [32])

**Risk level**: HIGH — diffusion models for robotics still maturing, complex system

**Estimated papers**: 1-2

---

### Pathway F: Bounded Rationality DRL for Practical PE (NEW)
**Fills gaps**: G7

**Architecture**:
```
State → Level-k Policy Hierarchy → {
    Level-0: Random/heuristic baseline
    Level-1: Best response to Level-0 (pre-trained)
    Level-k: Best response to Level-(k-1) (DRL-trained)
} → Actions
```

**Key components**:
1. **Level-k thinking model** from [14] (Kokolakis dissertation)
2. **Deep RL** implementation (vs [14]'s critic-only RL)
3. **Practical opponent modeling**: Assume bounded rationality rather than NE
4. **Curriculum**: Start from Level-1, progressively increase sophistication
5. **Ground robot dynamics**: Differential-drive or Dubins car

**Novelty**: First DRL implementation of Level-k PE for ground robots
- Paper [14] used control-theoretic RL; extend to modern deep RL
- More realistic than Nash assumption for physical robots with computational limits

**Risk level**: MODERATE — well-founded theory from [14], clear implementation path

**Estimated papers**: 1-2

---

## 6. Pathway Ranking and Recommendation

| Rank | Pathway | Impact | Feasibility | Novelty | Papers | Recommendation |
|------|---------|--------|-------------|---------|--------|----------------|
| 1 | **A: Safe DRL for 1v1 PE** | Very High | High | High | 2-3 | **START HERE** |
| 2 | **B: HJ-RL Hybrid** | Very High | Moderate | Very High | 2-3 | Strong alternative |
| 3 | **D: Info-Aware PE** | High | High | High | 1-2 | Good complement to A |
| 4 | **C: Vision-Based PE** | High | High | Moderate | 1-2 | Engineering-focused |
| 5 | **F: Bounded Rationality** | Moderate | High | High | 1-2 | Theoretical contribution |
| 6 | **E: Hierarchical Diffusion** | Moderate | Low-Moderate | High | 1-2 | Higher risk |

### Primary Recommendation: Pathway A
**Safe Deep RL for 1v1 Ground Robot PE with Nonholonomic Dynamics**

**Justification**:
1. **Fills the clearest gap**: No existing work combines safe DRL + 1v1 PE + ground robot + real deployment
2. **All building blocks demonstrated**: Each component has a working reference paper
3. **High impact**: Safety-guaranteed PE is critical for real-world deployment
4. **Clear publication path**: Method paper (algorithm) → robot paper (deployment) → analysis paper (safety proofs)
5. **Strong baselines available**: [02] for PE performance, [22] for game-theoretic optimality

### Suggested Progression
1. **Phase 1** (Months 1-3): Implement Pathway A in simulation
   - Gymnasium-based PE environment with unicycle dynamics
   - PPO + CBF safety filter + self-play training
   - Baseline comparison vs [12] (DQN/DDPG), [15] (A-MPC+DQN)

2. **Phase 2** (Months 3-6): Sim-to-real transfer
   - Isaac Sim environment with domain randomization
   - Gazebo ROS2 intermediate validation
   - TurtleBot4 or F1TENTH deployment

3. **Phase 3** (Months 6-9): Extend with Pathway B or D
   - Add HJ reachability shielding (Pathway B) OR
   - Add information-aware sensing (Pathway D)
   - Formal safety analysis and proofs

4. **Phase 4** (Months 9-12): Publications and analysis
   - Method paper (conference: ICRA, IROS, CoRL)
   - Extended journal paper (RA-L, T-RO)
   - Safety analysis paper (CDC, L4DC)

---

## 7. Key Technical Decisions

### 7.1 RL Algorithm Selection
| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **PPO** | Stable, well-tested for PE [18] | Sample inefficient | Default choice |
| **SAC** | Sample efficient, continuous | Less tested for PE | Complex environments |
| **TD3** | Good for continuous control [34] | Deterministic | Navigation modules |
| **MADDPG** | Standard for 2-player games [02] | Off-policy complexity | Centralized training |

**Recommendation**: Start with PPO (proven in PE [18]), consider SAC for sample efficiency.

### 7.2 Safety Method Selection
| Method | Papers | Guarantee | Model Need | Overhead |
|--------|--------|-----------|------------|----------|
| **CBF-QP filter** | [05], [06] | Hard | Partial (CBF) | QP solve |
| **CBF-Beta policy** | [16] | Hard + convergence | CBF | Minimal |
| **Robust CBF + GP** | [06] | Hard (robust) | GP model | GP + QP |
| **HJ shielding** | [22], [25], [31] | Strongest | Full dynamics | Offline + lookup |
| **CLBF** | [28] | Hard | Model-free | Training time |

**Recommendation**: CBF-constrained Beta policies [16] for Pathway A (best convergence-safety balance). Add Robust CBF + GP [06] for real deployment.

### 7.3 Training Paradigm
| Paradigm | Papers | Convergence | Complexity |
|----------|--------|-------------|------------|
| **AMS-DRL** | [18] | Proven NE | Multi-stage |
| **MADDPG** | [02], [20] | Empirical | Standard |
| **PSRO** | [04] | Proven NE | High |
| **Vanilla self-play** | [04] | May cycle | Simple |

**Recommendation**: AMS-DRL [18] — staged alternating training with proven NE convergence.

### 7.4 Observation Space
| Type | Papers | Realism | Complexity |
|------|--------|---------|------------|
| **Full state** | [01], [12], [13] | Low | Low |
| **Limited FOV lidar** | [02] | High | Medium |
| **Camera (vision)** | [08] | Highest | High |
| **Query-based** | [34] | Novel | Medium |

**Recommendation**: Start with limited FOV lidar [02] (realistic but manageable). Extend to vision [08] later.

---

## 8. Papers Still Needed

### Remaining Inaccessible (2 papers)
1. **"Safe Robust MARL with Neural CBFs + Safety Attention"** — Information Sciences 2024 (Elsevier paywall, DOI: 10.1016/j.ins.2024.121577)
2. **"Transfer RL for Multi-agent PE with Obstacles"** — Asian J. Control 2024 (Wiley Cloudflare, DOI: 10.1002/asjc.3328)

### Potentially Missing (identified from Paper [37]'s 235 references)
3. **AD-VAT+** (Zhong et al., TPAMI 2021) — Asymmetric dueling for visual active tracking
4. **End-to-end Active Object Tracking** (Luo et al., TPAMI 2020) — ConvNet-LSTM+A3C, sim-to-real
5. **C-VAT** (Devo et al., RAS 2021) — Continuous control for mobile robot visual tracking
6. **NFSP** (Heinrich & Silver, 2016) — Neural Fictitious Self-Play
7. **PSRO** (Lanctot et al., 2017) — Policy Space Response Oracle framework

### Previously Needed, Now Downloaded and Read
- ~~"Safe MARL via Approximate HJ Reachability"~~ — JIRS 2024 → **Paper [31]**
- ~~"Enhancing DeepReach via Activation Functions"~~ — arXiv 2023 → **Paper [24]**
- ~~"Hot Starts in PE Games (GNN)"~~ — arXiv 2025 → **Paper [33]**
- ~~"Neural Networks in Optimizing PE Tactics"~~ — Dynamic Games & Apps 2024 → **Paper [26]**
- ~~"CASRL: Conflict-Averse Safe RL"~~ — CAAI Trans. 2023 → **Paper [30]**
- ~~"Bio-Inspired NN for Real-Time Evasion"~~ — Biomimetics 2024 → **Paper [36]**
- ~~"RL Approaches for PE Games: A Review"~~ — CJA 2025 → **Paper [37]**

---

## 9. Paper Index by Relevance to 1v1 Ground Robot PE

### Tier 1: Directly Applicable (MUST READ)
| # | Short Title | Why Essential |
|---|------------|---------------|
| 01 | RL+SMC+CBF for PE | Three-term safe PE controller architecture |
| 02 | Car-like PE with sensors | SOTA 1v1 ground robot PE with DRL + real deployment |
| 08 | Vision-based PE | Privileged learning for PE + real robot |
| 13 | Safe finite-time PE | Barrier functions in HJI for PE |
| 16 | CBF-Beta PPO | Convergence + safety breakthrough |
| 18 | AMS-DRL | Self-play with NE convergence + real drones |
| 22 | MADR | Game-optimal PE on TurtleBots |
| 25 | HJ-RL Survey | Complete roadmap |
| 34 | SHADOW | 1v1 PE with unicycle dynamics + info trade-offs |

### Tier 2: Important Supporting Methods
| # | Short Title | Why Important |
|---|------------|---------------|
| 03 | CBF-RL Survey | Taxonomy of safety methods |
| 04 | Self-Play Survey | Training paradigm selection |
| 05 | CBF-RL filtering | Practical CBF + reward shaping |
| 06 | Robust CBF + GP | Handles model uncertainty |
| 07 | Sim-to-Real pipeline | Isaac Sim → ROS2 recipe |
| 10 | Safe RL Survey | Comprehensive safety method reference |
| 14 | Kokolakis dissertation | Safety + GP + bounded rationality for PE |
| 21 | DeepReach | Foundational neural HJI solver |
| 24 | DeepReach Activations | Practical guidance for sine/ReLU mixing |
| 27 | RESPO | Safe RL with reachability estimation |
| 30 | CASRL | Conflict-averse gradient for safe navigation + real robot |
| 31 | Safe HJ MADDPG | HJ reachability + CQL for safe MARL, 0.1ms inference |
| 37 | RL-PE Review | Comprehensive 235-ref taxonomy of RL for PE games |

### Tier 3: Useful Background and Techniques
| # | Short Title | Transferable Insight |
|---|------------|---------------------|
| 09 | Multi-UAV PE | Evader prediction + adaptive curriculum |
| 11 | Game Theory MARL | Nash equilibrium + opponent modeling theory |
| 12 | Dog-sheep PE | DQN vs DDPG baseline comparison |
| 15 | Mobile robot PE | A-MPC + DRL hybrid |
| 17 | ViPER | Graph attention for PE environments |
| 20 | Emergent behaviors | Behavior taxonomy + clustering analysis |
| 23 | DeepReach convergence | Theoretical validation |
| 28 | LBAC | CLBF for model-free reach-avoid |
| 32 | Diffusion-RL PE | Hierarchical planning for PE |
| 35 | NeHMO | Scalable neural HJR |

### Tier 4: Supplementary
| # | Short Title | Niche Value |
|---|------------|-------------|
| 19 | Self-play iteration | NE convergence proof for PE |
| 26 | HNSN | Hybrid reactive/strategic NN architecture (not RL, 3D) |
| 33 | Hot Starts PE | GCN initial placement for multi-pursuer (not 1v1) |
| 36 | BINN Evasion | Bio-inspired reactive evasion baseline, ROS robots |

---

## 10. Conclusion

The field of 1v1 PE with DRL for ground robots is at an inflection point. Individual building blocks — safe RL (CBFs, HJ reachability), self-play training, partial observability handling, sim-to-real transfer — have each been demonstrated successfully. The clear research opportunity is **integration**: combining these components into a complete, safety-guaranteed PE system on physical ground robots.

**Pathway A (Safe Deep RL)** is the recommended starting point because it fills the most impactful gap with the lowest risk, building directly on demonstrated components from Papers [02], [05], [06], [16], [18], and [07]. The HJ-RL hybrid approach (Pathway B) offers the strongest theoretical guarantees and should be pursued as a complementary or follow-on direction.

The 36 papers reviewed here provide a complete toolkit. No fundamental scientific barriers remain — the challenge is engineering integration, careful system design, and rigorous real-world validation. Two papers behind paywalls (Elsevier, Wiley) remain inaccessible but are unlikely to alter the core recommendations. Paper [37]'s 235-reference survey confirms our coverage is comprehensive and identifies a few additional visual tracking papers (AD-VAT, C-VAT) that could supplement Pathway C.
