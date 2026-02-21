# Deep Reinforcement Learning for 1v1 Pursuit-Evasion Games with Ground Robots
## Comprehensive Research Report

**Date**: 2026-02-21
**Scope**: One-pursuer, one-evader pursuit-evasion games using deep RL on ground mobile robots
**Confidence Level**: HIGH (based on 30+ recent sources, 2022–2025)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Classical Foundations](#2-classical-foundations)
3. [Deep RL Approaches — Taxonomy](#3-deep-rl-approaches--taxonomy)
4. [Key Algorithms Used in PE Games](#4-key-algorithms-used-in-pe-games)
5. [Environment & Simulation Landscape](#5-environment--simulation-landscape)
6. [Ground Robot–Specific Considerations](#6-ground-robot-specific-considerations)
7. [Observation Spaces & Partial Observability](#7-observation-spaces--partial-observability)
8. [Reward Design Strategies](#8-reward-design-strategies)
9. [Training Paradigms](#9-training-paradigms)
10. [Sim-to-Real Transfer](#10-sim-to-real-transfer)
11. [Safety-Aware PE Games](#11-safety-aware-pe-games)
12. [Open Problems & Research Gaps](#12-open-problems--research-gaps)
13. [Suggested Research Pathways](#13-suggested-research-pathways)
14. [Key Reference Papers](#14-key-reference-papers)

---

## 1. Executive Summary

Pursuit-Evasion Games (PEGs) are a fundamental class of adversarial multi-agent problems where a pursuer aims to capture an evader that is actively trying to escape. Deep reinforcement learning (deep RL) has emerged as the dominant paradigm for solving PEGs, replacing classical differential-game-theoretic solutions (which require analytical solvability) with data-driven, end-to-end learned policies.

**Current state of the field (2024–2025):**
- Most work focuses on **multi-pursuer** scenarios (UAVs, USVs); pure **1v1 ground robot** PE with deep RL remains **relatively underexplored** — presenting a clear opportunity.
- Dominant algorithms: **PPO, SAC, TD3, MADDPG** for continuous action spaces; **DQN** variants for discrete.
- Self-play is the standard training paradigm for competitive 1v1 games.
- Sim-to-real transfer for ground robots has been demonstrated (F1TENTH, TurtleBot) but not extensively for adversarial PE.
- Major gaps exist in: **partial observability with vision**, **nonholonomic dynamics integration**, **safety-constrained learning**, and **real-world validation**.

---

## 2. Classical Foundations

### 2.1 Differential Game Theory
PEGs originate from Rufus Isaacs' differential game theory (1965). The 1v1 PE game is formalized as a **zero-sum differential game** where:
- The pursuer minimizes and the evader maximizes a payoff (typically time-to-capture or terminal distance)
- Optimal strategies satisfy the **Hamilton-Jacobi-Isaacs (HJI) equation**
- Analytical solutions exist only for simple dynamics (e.g., simple motion, linear kinematics)

### 2.2 Why Deep RL?
Classical methods break down when:
- Dynamics are nonlinear/nonholonomic (differential-drive, car-like robots)
- Environments contain obstacles
- Observation is partial (limited sensors, field-of-view constraints)
- Opponent models are unknown

Deep RL replaces analytical computation of Nash equilibrium strategies with **learned policies** that can handle these complexities through experience.

### 2.3 HJI + Neural Networks (Emerging Hybrid)
Recent work bridges classical and learning approaches:
- **DeepReach** uses neural networks to approximate HJI value functions with convergence guarantees ([Convergence Guarantees for NN-Based HJ Reachability, ResearchGate](https://www.researchgate.net/publication/384680365_Convergence_Guarantees_for_Neural_Network-Based_Hamilton-Jacobi_Reachability))
- **PINN-based policy iteration** solves nonconvex HJI equations using physics-informed neural networks ([arxiv.org/html/2507.15455v2](https://arxiv.org/html/2507.15455v2))
- **Hamilton-Jacobi Deep Q-Learning** combines HJI structure with DQN for deterministic continuous control ([JMLR paper](https://jmlr.csail.mit.edu/papers/volume22/20-1235/20-1235.pdf))

---

## 3. Deep RL Approaches — Taxonomy

### 3.1 By Action Space

| Type | Algorithms | PE Application |
|------|-----------|----------------|
| **Discrete** | DQN, Double DQN, Dueling DQN | Grid-world PE, simplified environments |
| **Continuous** | DDPG, TD3, SAC, PPO | Realistic robot PE with velocity/heading control |
| **Hybrid** | Parameterized Action Space | Discrete maneuver selection + continuous parameters |

### 3.2 By Number of Learning Agents

| Configuration | Method | Notes |
|---------------|--------|-------|
| **Single-agent** (evader is scripted) | Standard RL (PPO, SAC) | Simpler but doesn't produce robust policies |
| **Two-agent competitive** | Self-play, MARL | Both agents learn simultaneously; produces Nash-approaching strategies |
| **Asymmetric training** | Alternating optimization | Train pursuer against fixed evader, then swap |

### 3.3 By Modeling Approach

| Approach | Description | Key Papers |
|----------|-------------|------------|
| **Model-free** | Direct policy/value learning | Most PE-DRL work |
| **Model-based** | Learn dynamics model, plan with it | DRL + MPPI ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231225007179)) |
| **Hybrid** | Classical control + RL-based high-level planning | A-MPC + DRL ([Springer](https://link.springer.com/article/10.1007/s13235-025-00647-1)) |

---

## 4. Key Algorithms Used in PE Games

### 4.1 PPO (Proximal Policy Optimization)
- **Most popular** for PE games due to stability and ease of tuning
- Used in multi-agent variants (MAPPO) for pursuit-evasion
- Effective for both discrete and continuous action spaces
- Applied in F1TENTH sim-to-real demonstrations

### 4.2 SAC (Soft Actor-Critic)
- Maximum entropy framework encourages exploration — important in adversarial settings
- Well-suited for continuous control of ground robots
- Combined with attention mechanisms (GAP_SAC) for state-feature prioritization
- **Recommended for 1v1 PE** due to robustness and sample efficiency

### 4.3 TD3 (Twin Delayed DDPG)
- Addresses overestimation bias in DDPG
- Used in multi-robot cooperative PE (MATD3)
- Strong performance in continuous control benchmarks ([Atlantis Press, DAI-23](https://www.atlantis-press.com/proceedings/dai-23/125998066))

### 4.4 DDPG (Deep Deterministic Policy Gradient)
- Original off-policy actor-critic for continuous control
- Extended to multi-agent settings (MADDPG)
- GE-DDPG variant proposed for 3D pursuit-evasion ([Springer, 2025](https://link.springer.com/article/10.1007/s12555-025-0071-0))

### 4.5 MADDPG (Multi-Agent DDPG)
- Centralized training, decentralized execution (CTDE)
- Used with trajectory prediction networks for PE
- Combined with curriculum learning for sensor-constrained PE ([arxiv.org/html/2405.05372v1](https://arxiv.org/html/2405.05372v1))

### 4.6 DQN Variants
- Applied in grid-world and simplified PE environments
- Used in F1TENTH with LiDAR input ([GitHub: MichaelBosello/f1tenth-RL](https://github.com/MichaelBosello/f1tenth-RL))
- Less suitable for continuous ground robot control

### Algorithm Recommendation for 1v1 Ground Robot PE

| Criterion | Best Choice |
|-----------|------------|
| Sample efficiency | SAC |
| Stability & ease of use | PPO |
| Continuous control quality | TD3 / SAC |
| Self-play training | PPO (MAPPO) or SAC with population-based training |
| Vision-based observation | PPO with CNN encoder |

---

## 5. Environment & Simulation Landscape

### 5.1 Available Frameworks

| Platform | PE Support | Robot Support | Sim-to-Real |
|----------|-----------|---------------|-------------|
| **PettingZoo** | Built-in pursuit env | Agent-based (abstract) | No |
| **OpenAI Gym / Gymnasium** | Custom envs needed | TurtleBot, generic | Via ROS bridge |
| **Gazebo + ROS2** | Custom | Full robot models | Direct |
| **NVIDIA Isaac Sim** | Custom | High-fidelity physics | Demonstrated (2025) |
| **MuJoCo** | Custom | Excellent for contact | Via domain randomization |
| **Unity ML-Agents** | Custom | Visual environments | Limited |
| **F1TENTH Gym** | Racing (adversarial) | Car-like robots | F1TENTH hardware |

### 5.2 Custom Environment Considerations for 1v1 Ground Robot PE
- **Arena**: Bounded 2D space with optional obstacles
- **Dynamics**: Differential-drive or car-like (Ackermann) kinematics
- **Observations**: LiDAR scans, relative position/velocity, or camera images
- **Episode termination**: Capture (distance threshold), timeout, or boundary violation
- **Recommend**: Custom Gymnasium environment → Gazebo bridge → real robot

---

## 6. Ground Robot–Specific Considerations

### 6.1 Dynamics Models

**Differential Drive (e.g., TurtleBot)**:
```
ẋ = v·cos(θ)
ẏ = v·sin(θ)
θ̇ = ω
Action: (v, ω) — linear and angular velocity
```

**Car-like / Ackermann (e.g., F1TENTH)**:
```
ẋ = v·cos(θ)
ẏ = v·sin(θ)
θ̇ = (v/L)·tan(δ)
Action: (v, δ) — speed and steering angle
Constraint: minimum turning radius
```

**Double Integrator**:
```
ẍ = u_x,  ÿ = u_y
Action: (u_x, u_y) — force/acceleration
More expressive but less realistic for wheeled robots
```

### 6.2 Nonholonomic Constraints
- Wheeled robots **cannot move laterally** — this fundamentally changes the game dynamics
- A pursuer with a minimum turning radius can be exploited by an agile evader
- RL policies must learn to account for these constraints implicitly
- Research shows nonholonomic constraints make the PE game significantly harder than the point-mass case ([Springer, 2025](https://link.springer.com/article/10.1007/s13235-025-00647-1))

### 6.3 Speed Asymmetry
- Classical game theory result: if evader speed ≥ pursuer speed, capture in open space is impossible
- In practice, speed ratios between 0.6–0.9 (evader/pursuer) produce interesting games
- RL can discover strategies that exploit environmental structure even at disadvantageous speed ratios

### 6.4 Hardware Platforms for Ground Robot PE

| Platform | Type | Cost | Sensors | RL Suitability |
|----------|------|------|---------|----------------|
| **TurtleBot3/4** | Differential drive | ~$600-1500 | LiDAR, camera | Excellent (ROS2 native) |
| **F1TENTH** | Ackermann | ~$3000 | LiDAR, IMU | Excellent (active community) |
| **JetRacer** | Ackermann | ~$250 | Camera | Good (affordable) |
| **Khepera IV** | Differential drive | ~$3000 | IR, ultrasonic | Legacy, limited |
| **Custom (ROS2-based)** | Any | Variable | Any | Most flexible |

---

## 7. Observation Spaces & Partial Observability

### 7.1 Full Observability
- Both agents know exact positions and velocities of all entities
- Standard in simulation studies
- Unrealistic for real deployment

### 7.2 Partial Observability (POSG/Dec-POMDP)
This is the **critical challenge** for real ground robot PE:

| Observation Type | Pros | Cons | Key Papers |
|-----------------|------|------|------------|
| **Relative state** (position, velocity) | Simple, fast | Requires global localization | Most PE-DRL papers |
| **LiDAR scans** | Range-based, works indoors | No identity of opponent | [Isaac Sim to Real, 2025](https://arxiv.org/abs/2501.02902) |
| **RGB / RGB-D camera** | Rich information | High-dimensional, requires CNNs | [Bajcsy et al., ICRA 2024](https://abajcsy.github.io/vision-based-pursuit/) |
| **Wedge-shaped sensor** | Models real FoV | Binary visibility flags | [Car-like PE with sensor constraints, 2024](https://arxiv.org/abs/2405.05372) |

### 7.3 Handling Partial Observability in RL
- **Recurrent policies** (LSTM/GRU) to maintain belief state from observation history
- **Frame stacking** — concatenating last N observations
- **Attention mechanisms** — learning to focus on relevant features
- **Belief encoding** — learned latent representation of hidden state
- The 2024 paper by Coon et al. encodes history into belief states and achieves **16% improvement** in capture rate over baselines ([arxiv.org/abs/2405.05372](https://arxiv.org/abs/2405.05372))

---

## 8. Reward Design Strategies

### 8.1 Common Reward Structures

**Sparse Reward (simplest)**:
```
R = +1 if capture, -1 if timeout (pursuer)
R = +1 if escape, -1 if captured (evader)
```
Problem: extremely difficult to learn from — agents rarely encounter reward signals early in training.

**Dense Distance-Based Reward**:
```
R_pursuer = -α·distance(P, E) + β·Δdistance  (reward for closing distance)
R_evader  = +α·distance(P, E) - β·Δdistance   (reward for increasing distance)
```

**Multi-Objective Reward** (recent best practice):
```
R = w1·capture_reward + w2·distance_shaping + w3·energy_penalty + w4·collision_penalty + w5·smoothness_reward
```
The 2025 A-MPC + DRL paper uses an **adaptive reward structure** balancing evasion, energy efficiency, and collision avoidance.

### 8.2 Two-Stage Reward Refinement
From the OPEN paper (2024):
- **Stage 1**: Exclude smoothness rewards, focus on capture/collision/distance
- **Stage 2**: Gradually introduce smoothness rewards for deployable policies

### 8.3 Potential-Based Reward Shaping
- Guarantees policy invariance (same optimal policy as without shaping)
- PVDN uses potential-based shaping to ensure consistent performance regardless of goal distance ([IJCAI-25](https://www.ijcai.org/proceedings/2025/0036.pdf))

---

## 9. Training Paradigms

### 9.1 Self-Play
The **gold standard** for 1v1 competitive games:
- Agent plays against copies of itself (current or historical)
- Naturally produces increasingly sophisticated strategies
- Key in AlphaStar, OpenAI Five successes
- For PE: pursuer and evader co-evolve strategies
- **Self-play iteration** converges to Nash equilibrium under certain conditions ([Springer, Machine Intelligence Research, 2024](https://link.springer.com/article/10.1007/s11633-022-1413-5))

### 9.2 Asynchronous Multi-Stage Training (AMS-DRL)
- Pursuers and evaders trained **asynchronously** in a bipartite graph structure
- Convergence guaranteed via Nash equilibrium analysis from game theory
- Avoids non-stationarity issues of simultaneous training

### 9.3 Curriculum Learning
- Start with **easier configurations** (pursuer has better sensors/speed) and progressively increase difficulty
- The MADDPG + curriculum approach for car-like PE starts with wider sensor coverage and linearly decays ([arxiv.org/abs/2405.05372](https://arxiv.org/abs/2405.05372))
- Automatic curriculum generation based on success rate thresholds (σ_min=0.5, σ_max=0.9)

### 9.4 Transfer Learning
- Train in simple environments → transfer to complex ones
- Multi-agent PE differential game with obstacles uses RL + transfer learning ([Asian Journal of Control, 2024](https://onlinelibrary.wiley.com/doi/abs/10.1002/asjc.3328))
- Domain randomization for sim-to-real transfer

### 9.5 Population-Based Training
- Maintain a population of diverse opponent policies
- Train against randomly sampled opponents → more robust strategies
- Prevents strategy collapse from self-play overfitting

---

## 10. Sim-to-Real Transfer

### 10.1 Current State for Ground Robots
- **Zero-shot sim-to-real** has been demonstrated for ground robot navigation ([Isaac Sim → Gazebo → TurtleBot, 2025](https://arxiv.org/abs/2501.02902))
- For PE specifically: demonstrated on **F1TENTH and JetRacer** at 2 m/s with simulation-trained MADDPG policies ([arxiv.org/abs/2405.05372](https://arxiv.org/abs/2405.05372))
- Vision-based PE deployed on **quadruped robots** (Unitree Go1) in forests and fields ([Bajcsy et al., ICRA 2024](https://abajcsy.github.io/vision-based-pursuit/))

### 10.2 Key Techniques
| Technique | Description |
|-----------|-------------|
| **Domain Randomization** | Vary physics parameters (friction, mass, sensor noise) during training |
| **System Identification** | Fit simulation to measured real-robot dynamics |
| **ONNX Export** | Export trained policies for ROS2 deployment |
| **Sim-to-Sim Validation** | Train in Isaac Sim → test in Gazebo → deploy on real robot |
| **Action Smoothing** | Add smoothness rewards to prevent jerky real-world behavior |

### 10.3 Gap Analysis
| Challenge | Severity for 1v1 Ground PE |
|-----------|---------------------------|
| Actuator dynamics mismatch | Medium — addressable with system ID |
| Sensor noise (LiDAR/camera) | Medium — domain randomization helps |
| Latency / communication delay | High — adversarial games are latency-sensitive |
| Opponent behavior mismatch | **High** — real opponents behave differently than trained models |
| Floor friction variation | Low–Medium |

---

## 11. Safety-Aware PE Games

### 11.1 Why Safety Matters
Real robots must avoid collisions with walls, obstacles, and each other — even during aggressive pursuit/evasion maneuvers.

### 11.2 Approaches

**Control Barrier Functions (CBFs) + RL** ([arxiv.org/abs/2507.19516](https://arxiv.org/abs/2507.19516)):
- Safeguard policy enforces collision avoidance **independently** of RL learning
- Stackelberg game structure: safety controller is "leader", RL policy is "follower"
- Handles high-order constraints and moving obstacles
- **Most promising approach for real ground robot PE**

**Constrained MDPs (CMDPs)**:
- Add safety cost constraints to the RL objective
- Lagrangian relaxation or projection methods
- Comprehensive survey published in 2025 ([arxiv.org/html/2505.17342v1](https://arxiv.org/html/2505.17342v1))

**Gaussian Process Safety** ([IEEE TNNLS, 2022](https://ieeexplore.ieee.org/iel7/5962385/10454107/09913917.pdf)):
- Learn unknown environment boundaries using GPs
- Integrate learned safety constraints into RL

---

## 12. Open Problems & Research Gaps

### 12.1 Critical Gaps (High Novelty Potential)

| Gap | Current State | Opportunity |
|-----|--------------|-------------|
| **1v1 ground robot PE with realistic dynamics** | Most work uses point-mass or simplified dynamics | Integrate differential-drive/Ackermann constraints directly into RL |
| **Vision-only PE for ground robots** | Only Bajcsy et al. (2024) on quadrupeds | No work on wheeled ground robots with camera-only PE |
| **Safety-constrained 1v1 PE with deep RL** | Theoretical (2025), no real robot validation | CBF + RL on real ground robots in PE scenarios |
| **Heterogeneous dynamics PE** | Rarely studied | Pursuer and evader with different dynamics models |
| **Long-horizon PE in structured environments** | Most work is open-field or simple obstacles | Indoor PE with rooms, corridors, doors |
| **Opponent modeling without communication** | Limited work | Real-time inference of opponent intent from observations |

### 12.2 Methodological Gaps

| Gap | Description |
|-----|-------------|
| **Model-based RL for PE** | Almost no work uses learned world models (MuZero-style) for PE |
| **Hierarchical RL for PE** | Option framework proposed but not validated on real robots |
| **Foundation model integration** | No work combining LLM/VLM reasoning with PE control |
| **Multi-objective optimization** | Balancing capture time, energy, safety, smoothness remains ad-hoc |
| **Formal verification of PE policies** | No guarantees on learned policy behavior |

### 12.3 Experimental Gaps

| Gap | Description |
|-----|-------------|
| **Standardized benchmarks** | No agreed-upon benchmark for 1v1 ground robot PE |
| **Real-world PE experiments** | Very few; most are simulation-only |
| **Human-vs-robot PE** | Almost completely unexplored for ground robots |
| **Outdoor unstructured PE** | Only vision-based quadruped work; nothing for wheeled robots |

---

## 13. Suggested Research Pathways

### Pathway A: **Safe Deep RL for 1v1 Ground Robot PE with Nonholonomic Dynamics** (HIGH NOVELTY)

**Core Idea**: Combine safety-constrained RL (CBF + RL) with self-play training for a 1v1 PE game on differential-drive robots, with real-world validation.

**Why it matters**: No existing work integrates safety guarantees with deep RL self-play for ground robot PE.

**Methodology**:
1. Custom Gymnasium environment with differential-drive dynamics + obstacles
2. SAC or PPO with self-play for policy learning
3. CBF safety layer for collision avoidance (walls + opponent)
4. Curriculum learning: open field → sparse obstacles → dense obstacles
5. Sim-to-real: Train in Isaac Sim/MuJoCo → validate in Gazebo → deploy on TurtleBot3/4
6. Compare against: classical PE strategies, unconstrained RL, heuristic methods

**Expected Contributions**:
- First safe RL framework for 1v1 ground robot PE with real-world validation
- Analysis of how safety constraints affect Nash equilibrium convergence
- Open-source benchmark environment

**Estimated Timeline**: 8–12 months

---

### Pathway B: **Vision-Based 1v1 PE for Wheeled Ground Robots** (HIGH NOVELTY)

**Core Idea**: Learn pursuit and evasion policies from camera images (RGB or RGB-D) on wheeled ground robots using privileged learning + self-play.

**Why it matters**: Bajcsy et al. (ICRA 2024) showed this works for quadrupeds; no work exists for wheeled robots. Camera-only PE eliminates need for expensive LiDAR.

**Methodology**:
1. Simulation environment with realistic rendering (Isaac Sim or Unity)
2. Privileged learning: train teacher policy with full state → distill to student with camera input
3. CNN/ViT encoder for image observations
4. Self-play with population-based training for diverse strategies
5. Sim-to-real: domain randomization on visual appearance
6. Hardware: JetRacer (affordable, camera-equipped) or TurtleBot with camera

**Expected Contributions**:
- First vision-only PE system for wheeled ground robots
- Analysis of privileged learning effectiveness for ground robot PE
- Comparison of CNN vs. Transformer visual encoders for PE

**Estimated Timeline**: 10–14 months

---

### Pathway C: **Model-Based Deep RL with Learned World Models for PE** (MEDIUM-HIGH NOVELTY)

**Core Idea**: Use a learned world model (Dreamer-style) to plan pursuit/evasion strategies, improving sample efficiency and enabling online adaptation.

**Why it matters**: All PE-DRL work is model-free. World models could dramatically improve sample efficiency and enable real-time replanning.

**Methodology**:
1. Train a world model (RSSM or Transformer-based) from PE interactions
2. Use imagination-based planning (DreamerV3) for policy optimization
3. Self-play in the learned world model for rapid strategy iteration
4. Compare sample efficiency vs. model-free baselines (SAC, PPO)
5. Online adaptation: fine-tune world model during deployment to handle novel opponents

**Expected Contributions**:
- First model-based deep RL approach for adversarial PE games
- Sample efficiency analysis (potentially 10-100x improvement)
- Online opponent adaptation framework

**Estimated Timeline**: 10–14 months

---

### Pathway D: **Hierarchical RL for PE in Structured Indoor Environments** (MEDIUM NOVELTY)

**Core Idea**: Use hierarchical RL (options framework) where a high-level policy selects tactical maneuvers (chase, ambush, flank, retreat) and a low-level policy executes them.

**Why it matters**: Current PE-RL treats the problem as flat decision-making. Structured environments (offices, warehouses) require tactical planning.

**Methodology**:
1. Define tactical primitives: direct pursuit, ambush (hide and intercept), patrol, flee
2. High-level policy (PPO) selects primitives based on belief state
3. Low-level policies trained for each primitive via RL or classical control
4. Self-play at the hierarchical level
5. Indoor environments with rooms, corridors, partial walls

**Expected Contributions**:
- First hierarchical RL framework for 1v1 ground robot PE
- Emergent tactical behaviors in structured environments
- Interpretable strategy decomposition

**Estimated Timeline**: 8–12 months

---

### Pathway E: **Pursuit-Evasion with Asymmetric Information and Opponent Modeling** (MEDIUM-HIGH NOVELTY)

**Core Idea**: Develop an explicit opponent modeling module that infers the evader's policy type and intent from observation history, and adapts the pursuit strategy accordingly.

**Why it matters**: Real opponents don't follow a single policy. Adaptive pursuit against unknown evader strategies is essential for deployment.

**Methodology**:
1. Train a diverse population of evader policies (heuristic, RL-trained, random)
2. Train an opponent-type classifier from observation trajectories
3. Condition pursuit policy on inferred opponent type (meta-learning / context-conditional)
4. Evaluate adaptation speed to novel (unseen) evader strategies
5. Compare: LOLA, ToM-based approaches, Bayesian opponent modeling

**Expected Contributions**:
- Adaptive pursuit system robust to unknown evader behavior
- Quantitative analysis of opponent modeling impact on capture rate
- Framework for real-world deployment against human evaders

**Estimated Timeline**: 10–14 months

---

### Recommended Research Pathway Ranking

| Rank | Pathway | Novelty | Feasibility | Impact | Publishability |
|------|---------|---------|-------------|--------|----------------|
| 1 | **A: Safe RL + Ground Robot PE** | HIGH | HIGH | HIGH | Top robotics venue (ICRA/IROS/RA-L) |
| 2 | **B: Vision-Based Wheeled PE** | HIGH | MEDIUM | HIGH | Top venue (ICRA/CoRL) |
| 3 | **C: Model-Based RL for PE** | HIGH | MEDIUM | MEDIUM-HIGH | Top ML/RL venue (NeurIPS/ICML) |
| 4 | **E: Opponent Modeling** | MEDIUM-HIGH | MEDIUM | MEDIUM-HIGH | AAMAS/AAAI |
| 5 | **D: Hierarchical Indoor PE** | MEDIUM | HIGH | MEDIUM | IROS/RA-L |

**My top recommendation**: Start with **Pathway A** as it is the most feasible, offers high novelty, and provides a strong foundation. You can extend it with elements from B (vision) or E (opponent modeling) for follow-up work.

---

## 14. Key Reference Papers

### Surveys & Reviews
1. "A review of reinforcement learning approaches for pursuit-evasion games" — ScienceDirect, Nov 2025 ([link](https://www.sciencedirect.com/science/article/pii/S1000936125005461))
2. "An Overview of Recent Advances in Pursuit–Evasion Games with Unmanned Surface Vehicles" — JMSE, 2025 ([link](https://www.mdpi.com/2077-1312/13/3/458))
3. "Game Theory and Multi-Agent Reinforcement Learning: From Nash Equilibria to Evolutionary Dynamics" — arXiv, Dec 2024 ([link](https://arxiv.org/html/2412.20523v1))

### Foundational PE + DRL
4. "Pursuit and Evasion Strategy of a Differential Game Based on Deep Reinforcement Learning" — Frontiers, 2022 ([link](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2022.827408/full))
5. "Optimal Strategy for Aircraft PE Games via Self-play Iteration" — Machine Intelligence Research, 2024 ([link](https://link.springer.com/article/10.1007/s11633-022-1413-5))

### Ground Robot PE
6. "Mobile Robot Control Using PE Differential Game Strategy with DRL" — Dynamic Games and Applications, 2025 ([link](https://link.springer.com/article/10.1007/s13235-025-00647-1))
7. "Pursuit-Evasion for Car-like Robots with Sensor Constraints" — arXiv, May 2024 ([link](https://arxiv.org/abs/2405.05372))
8. "Learning Vision-based Pursuit-Evasion Robot Policies" — ICRA 2024 ([link](https://abajcsy.github.io/vision-based-pursuit/))

### Safety in PE
9. "Reinforcement learning in pursuit-evasion differential game: safety, stability and robustness" — arXiv, Jul 2025 ([link](https://arxiv.org/abs/2507.19516))
10. "Safety-Aware PE Games in Unknown Environments Using GPs and Finite-Time Convergent RL" — IEEE TNNLS, 2022 ([link](https://ieeexplore.ieee.org/iel7/5962385/10454107/09913917.pdf))

### Training Paradigms
11. "Emergent behaviors in multiagent PE games within a bounded 2D grid world" — Scientific Reports, 2025 ([link](https://www.nature.com/articles/s41598-025-15057-x))
12. "Transfer RL for multi-agent PE differential game with obstacles" — Asian Journal of Control, 2024 ([link](https://onlinelibrary.wiley.com/doi/abs/10.1002/asjc.3328))
13. "Multi-Robot Cooperative PE Control: A DRL Approach based on PER" — CCEAI 2024 ([link](https://dl.acm.org/doi/10.1145/3640824.3640843))

### Environments & Sim-to-Real
14. "Sim-to-Real Transfer for Mobile Robots with RL: from NVIDIA Isaac Sim to Gazebo and Real ROS 2 Robots" — arXiv, Jan 2025 ([link](https://arxiv.org/abs/2501.02902))
15. "PettingZoo: Gym for Multi-Agent Reinforcement Learning" — arXiv, 2020 ([link](https://arxiv.org/abs/2009.14471))
16. "Advancing Autonomous Racing: A Comprehensive Survey of the F1TENTH Platform" — arXiv, Jun 2025 ([link](https://arxiv.org/html/2506.15899v1))

### Advanced Methods
17. "PINN-based Policy Iteration for Solving Nonconvex HJI Equations" — arXiv, Jul 2025 ([link](https://arxiv.org/html/2507.15455v2))
18. "Distributed PE Game Decision-Making Based on Multi-Agent DRL" (Transformer-based) — Electronics, 2025 ([link](https://www.mdpi.com/2079-9292/14/11/2141))
19. "MSMAR-RL: Multi-Step Masked-Attention Recovery RL" (safe maneuver in PE) — IJCAI-25 ([link](https://www.ijcai.org/proceedings/2025/0036.pdf))
20. "Gaussian-enhanced RL for scalable evasion strategies in multi-agent PE games" — Neurocomputing, 2025 ([link](https://www.sciencedirect.com/science/article/abs/pii/S0925231225017527))

---

## Appendix: Quick-Start Technical Stack Recommendation

For a researcher starting a 1v1 ground robot PE project:

| Component | Recommendation |
|-----------|---------------|
| **Simulation** | Custom Gymnasium env → Gazebo (ROS2) for validation |
| **Physics** | MuJoCo (best for contact) or Isaac Sim (GPU-parallel) |
| **RL Library** | Stable-Baselines3 (single-agent) or CleanRL (transparency) |
| **Multi-Agent** | PettingZoo API + custom self-play wrapper |
| **Algorithm** | SAC for continuous 1v1; PPO for vision-based |
| **Robot** | TurtleBot3 Burger ($550) or F1TENTH ($3000) |
| **Sensors** | LiDAR (LDS-01) + optional camera |
| **Deployment** | ONNX export → ROS2 node |
| **Tracking** | Weights & Biases for experiment tracking |

---

*Report generated on 2026-02-21 by Claude Code research agent.*
*Based on analysis of 30+ sources from 2022–2025.*
