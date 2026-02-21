# Paper Summaries: Papers 01-10

## Paper 01: RL in Pursuit-Evasion Differential Game: Safety, Stability and Robustness
- **Authors**: Wang et al.
- **Venue**: arXiv, 2025
- **Type**: Research article

### Problem Formulation
- 1v1 pursuit-evasion as zero-sum differential game
- Addresses three critical properties simultaneously: safety, stability, robustness
- Continuous-time dynamics with bounded controls

### Key Innovation: Three-Term Controller
- **RL component**: Learns optimal pursuit/evasion strategies via policy optimization
- **Sliding Mode Control (SMC)**: Provides robustness against model uncertainties and disturbances
- **Control Barrier Functions (CBF)**: Ensures safety constraints (obstacle avoidance, boundary compliance)
- **Stackelberg decoupling**: Hierarchical game formulation where leader commits to strategy first

### Methodology
- Combined RL + SMC + CBF architecture
- Stackelberg game decomposition: pursuer leads, evader responds (or vice versa)
- CBF safety filter acts as hard constraint layer on top of RL policy
- SMC handles unmodeled dynamics and external disturbances

### Key Results
- Agents maintain safety throughout learning and deployment
- Finite-time stability guarantees
- Robust to bounded disturbances and model mismatch
- Stackelberg formulation more tractable than simultaneous Nash computation

### Relevance to 1v1 Ground Robot PE
- **HIGHEST RELEVANCE**: Directly addresses 1v1 PE with safety, stability, and robustness
- Three-term controller architecture (RL + SMC + CBF) is a practical blueprint for ground robots
- Stackelberg decoupling simplifies the two-player game training
- **Key paper for Pathway A (Safe Deep RL for 1v1 Ground Robot PE)**

---

## Paper 02: Pursuit-Evasion for Car-like Robots with Sensor Constraints
- **Authors**: Gonultas & Isler
- **Venue**: arXiv, 2024
- **Type**: Research article

### Problem Formulation
- 1v1 PE with car-like (Ackermann) robot kinematics
- Limited field-of-view sensors (not omniscient)
- Partial observability due to sensor constraints
- Belief state formulation for handling uncertainty

### Key Innovation
- **BiMDN (Bidirectional Mixture Density Network)**: Encoder for belief states under partial observability
- Belief state encodes probability distribution over opponent's position given sensor history
- MADDPG (Multi-Agent DDPG) for training both pursuer and evader
- Curriculum learning: progressive difficulty increase

### Methodology
- Car-like (Ackermann) kinematics with minimum turning radius
- BiMDN processes sensor history → compressed belief state representation
- MADDPG with centralized critic, decentralized actors
- Curriculum: start with close encounters, gradually increase distance/complexity
- F1TENTH and JetRacer platforms for real-world deployment

### Key Results
- Belief state encoding significantly outperforms raw observation approaches
- Curriculum learning critical for training stability
- **Real-world deployment** on F1TENTH and JetRacer car-like robots
- Handles realistic sensor constraints (limited FOV, noisy measurements)

### Relevance to 1v1 Ground Robot PE
- **HIGHEST RELEVANCE**: Exactly 1v1 PE with car-like ground robots
- Real hardware deployment on actual ground robot platforms
- Handles realistic sensor limitations
- BiMDN belief encoder addresses key partial observability challenge
- Curriculum learning provides practical training recipe
- **Key paper for Pathway A and Pathway B**

---

## Paper 03: Learning Control Barrier Functions and their Application in RL: A Survey
- **Authors**: Guerrier et al.
- **Venue**: arXiv, 2024
- **Type**: Survey

### Key Content
Comprehensive taxonomy of CBF-RL integration approaches:

1. **CBF as Safety Filter**: Post-hoc projection of RL actions onto safe set (QP-based)
2. **CBF in Reward Shaping**: Penalize CBF violations in reward function (soft constraint)
3. **CBF-Constrained Policy Optimization**: Constrained MDP with CBF constraints
4. **Learned CBFs**: Neural network approximation of CBF from data
5. **CBF + Lyapunov**: Combined safety (CBF) and stability (CLF) guarantees
6. **Robust CBFs**: Handle model uncertainty via GP or ensemble methods

### Relevance to 1v1 Ground Robot PE
- **HIGHLY RELEVANT**: Provides complete menu of CBF-RL integration options
- Essential reference for designing safety layer in PE agents
- Comparison of approaches helps select best CBF method for ground robot constraints
- **Foundation paper for all safety-aware PE approaches**

---

## Paper 04: A Survey on Self-play Methods in Reinforcement Learning
- **Authors**: Zhang et al.
- **Venue**: arXiv, 2024
- **Type**: Survey

### Key Content
Unified framework covering all self-play variants:

1. **Vanilla Self-play (SP)**: Train against copy of self
2. **Fictitious Self-play (FSP)**: Train against average of past policies
3. **Policy-Space Response Oracle (PSRO)**: Train against meta-game Nash mixture
4. **Counterfactual Regret Minimization (CFR)**: Information-set based regret minimization
5. **Population-based Training (PBT)**: Evolve population of diverse agents
6. **League Training**: AlphaStar-style multi-tier league

### Relevance to 1v1 Ground Robot PE
- **HIGHLY RELEVANT**: Self-play is the standard paradigm for training adversarial PE agents
- PE is a two-player zero-sum game → self-play methods directly applicable
- Helps choose between SP variants for training pursuer/evader
- PSRO provides principled approach to avoid strategy cycling
- **Foundation paper for training paradigm selection**

---

## Paper 05: CBF-RL: Safety Filtering Reinforcement Learning in Training with CBFs
- **Authors**: Yang et al. (Caltech, Ames group)
- **Venue**: arXiv, 2025
- **Type**: Research article

### Problem Formulation
- Safe RL during both training and deployment
- CBF-based safety filtering that preserves learning performance

### Key Innovation: Dual Safety Approach
1. **Closed-form CBF filter**: Analytically projects unsafe RL actions to safe set boundary
   - Avoids QP solver overhead (faster, differentiable)
2. **CBF reward shaping**: Augments reward with CBF-based penalty
   - Helps RL policy learn to naturally satisfy safety constraints
   - Reduces reliance on safety filter over time

### Methodology
- CBF filter: h(x) >= 0 defines safe set, filter minimally modifies unsafe actions
- Reward augmentation: r_safe = r_task + lambda * max(0, -h_dot - alpha*h)
- Tested with PPO and SAC on various control tasks
- Comparison with QP-based safety filters

### Key Results
- Closed-form filter faster than QP-based approaches
- Combined filter + reward shaping outperforms either alone
- RL agents learn safer policies that need less filtering over time
- Maintains task performance while guaranteeing safety

### Relevance to 1v1 Ground Robot PE
- **HIGHLY RELEVANT**: Practical method for safe RL training on ground robots
- Closed-form CBF filter is fast enough for real-time robot control
- Dual approach (filter + shaping) is directly applicable to PE agent training
- From the Ames group at Caltech — leading CBF authority
- **Key enabling technology for Pathway A**

---

## Paper 06: Safe Reinforcement Learning using Robust Control Barrier Functions
- **Authors**: Emam et al.
- **Venue**: arXiv, 2021/2022
- **Type**: Research article

### Problem Formulation
- RL agent must satisfy safety constraints despite model uncertainty
- Uncertainty in dynamics modeled and compensated

### Key Innovation: RCBF-QP + GP Disturbance Estimation
- **Robust CBF (RCBF)**: Extends CBF to handle bounded model uncertainty
- **Differentiable QP layer**: Safety filter as differentiable optimization layer in neural network
- **Gaussian Process (GP)**: Learns residual dynamics (model error) from data
- **Modular architecture**: Task RL + Safety layer are separate, composable modules

### Methodology
- Nominal dynamics + GP-estimated disturbance d_hat(x)
- RCBF condition: h_dot(x,u) + alpha*h(x) >= -|L_g h * d_hat| (robust margin)
- QP projects RL action: min ||u - u_rl||^2 s.t. RCBF constraint
- End-to-end differentiable: gradients flow through QP to RL policy
- GP trained online from observed state transitions

### Key Results
- Maintains safety even with significant model mismatch
- GP disturbance learning improves over time
- Differentiable QP enables end-to-end training
- Modular: can swap RL algorithm without changing safety layer

### Relevance to 1v1 Ground Robot PE
- **HIGHLY RELEVANT**: Ground robots always have model uncertainty
- GP disturbance estimation handles wheel slip, uneven terrain, etc.
- Modular safety layer can be added to any PE RL algorithm
- Differentiable QP preserves end-to-end learning
- **Key enabling technology for Pathway A**

---

## Paper 07: Sim-to-Real Transfer for Mobile Robots with RL: Isaac Sim to Gazebo to ROS2
- **Authors**: Salimpour et al.
- **Venue**: arXiv, 2025
- **Type**: Research article

### Problem Formulation
- End-to-end sim-to-real pipeline for mobile robot RL policies
- Bridge the reality gap for ground robot navigation

### Key Innovation: Three-Stage Transfer Pipeline
1. **NVIDIA Isaac Sim**: High-fidelity physics simulation for RL training
   - Domain randomization (friction, mass, sensor noise)
   - Parallel environments for fast training
2. **Gazebo (ROS2)**: Intermediate validation in ROS-compatible simulator
   - ONNX model export for cross-platform deployment
3. **Real ROS2 Robot**: Final deployment on TurtleBot4
   - ROS2 integration with standard navigation stack

### Methodology
- PPO training in Isaac Sim with domain randomization
- Policy exported as ONNX model
- Gazebo intermediate testing catches sim-to-real issues early
- ROS2 node wraps ONNX model for real-time inference
- TurtleBot4 platform (differential-drive)

### Key Results
- Successful zero-shot transfer from Isaac Sim to real robot
- Domain randomization critical for robustness
- ONNX export enables fast inference (~5ms per step)
- Gazebo intermediate step catches 80% of sim-to-real issues
- TurtleBot4 navigation in real indoor environments

### Relevance to 1v1 Ground Robot PE
- **HIGHLY RELEVANT**: Provides exact sim-to-real pipeline needed for PE
- Isaac Sim → Gazebo → Real Robot workflow directly applicable
- Domain randomization recipe for ground robot dynamics
- ONNX export for real-time RL policy deployment
- TurtleBot4 is a common ground robot platform
- **Key paper for sim-to-real deployment stage**

---

## Paper 08: Learning Vision-based Pursuit-Evasion Robot Policies
- **Authors**: Bajcsy et al.
- **Venue**: ICRA 2024 (IEEE International Conference on Robotics and Automation)
- **Type**: Research article

### Problem Formulation
- 1v1 PE where pursuer uses only onboard vision (camera) to track and capture evader
- No external tracking system (GPS, motion capture) available
- Must handle partial observability from limited camera FOV

### Key Innovation: Privileged Teacher + Latent Intent
1. **Privileged teacher**: Trained with full state information (omniscient)
2. **Latent intent estimation**: Teacher policy conditioned on evader's future intent
3. **DAGGER distillation**: Student policy learns from teacher using only camera observations
4. Student learns implicit opponent modeling through distillation

### Methodology
- Teacher: full-state PPO with access to evader position, velocity, intent
- Intent encoder: predicts evader's next N actions from trajectory history
- Student: CNN processes camera images, MLP processes proprioception
- DAGGER: student queries teacher for expert actions on student's own observations
- Unitree A1 quadruped robot for real-world experiments

### Key Results
- Student policy closely matches teacher performance despite vision-only input
- Latent intent enables anticipatory pursuit (not just reactive chasing)
- **Real-world deployment** on Unitree A1 quadruped robot
- Handles diverse evader strategies including adversarial evasion
- Outperforms baselines without intent modeling

### Relevance to 1v1 Ground Robot PE
- **HIGHEST RELEVANCE**: Exactly 1v1 PE with real robot deployment
- Privileged learning + distillation is state-of-art for sim-to-real PE
- Latent intent modeling enables anticipatory strategies
- Vision-based approach relevant for ground robots with cameras
- **Key paper for Pathway B (Vision-Based PE)**

---

## Paper 09: Multi-UAV PE with Online Planning in Unknown Environments by DRL
- **Authors**: Chen et al.
- **Venue**: IEEE RA-L, 2024/2025
- **Type**: Research article

### Problem Formulation
- Multiple UAV pursuers vs single evader in unknown environments
- Online planning (no pre-computed map)
- Evader has its own intelligent policy

### Key Innovation: OPEN Algorithm
1. **Online Planning with Evader Network (OPEN)**: Multi-stage approach
2. **MAPPO** (Multi-Agent PPO): Cooperative pursuit policy
3. **Evader prediction network**: Predicts evader's future trajectory
4. **Adaptive environment generator**: Auto-curricula for training difficulty

### Methodology
- MAPPO for cooperative multi-pursuer coordination
- Separate evader prediction module forecasts future evader positions
- Environment generator adjusts obstacle density, arena size, evader skill
- Unknown environment: pursuers explore while pursuing
- Communication between pursuers for shared map

### Key Results
- OPEN outperforms baselines in capture rate and time-to-capture
- Evader prediction significantly improves pursuit efficiency
- Adaptive curriculum essential for training convergence
- Generalizes to varying number of obstacles and arena sizes

### Relevance to 1v1 Ground Robot PE
- **MODERATELY RELEVANT**: Multi-UAV, but training techniques transferable
- MAPPO can be simplified to single-agent PPO for 1v1
- Evader prediction module useful even for 1v1 (opponent modeling)
- Adaptive environment generator provides curriculum learning recipe
- **Limitation**: UAV dynamics, multi-agent coordination

---

## Paper 10: A Review on Safe Reinforcement Learning Using Lyapunov and Barrier Functions
- **Authors**: Kushwaha & Biron
- **Venue**: arXiv, 2025
- **Type**: Comprehensive survey (~110 papers reviewed)

### Key Content: Taxonomy of Safe RL Approaches

1. **Lyapunov-based Safe RL**:
   - Control Lyapunov Functions (CLF) for stability
   - Lyapunov constraints in policy optimization
   - Neural Lyapunov certificates

2. **Barrier-based Safe RL**:
   - CBF safety filters (QP-based)
   - CBF reward augmentation
   - Learned/neural CBFs
   - Zeroing CBFs (ZBF) vs reciprocal CBFs

3. **Combined CLF-CBF**:
   - CLF-CBF-QP for simultaneous safety + stability
   - Trade-offs between safety and performance

4. **Model-based vs Model-free**:
   - Model-based: exploit dynamics knowledge for tighter guarantees
   - Model-free: learn safety constraints from data

5. **Constrained MDP approaches**:
   - Lagrangian methods
   - Interior point methods
   - Primal-dual optimization

### Relevance to 1v1 Ground Robot PE
- **HIGHLY RELEVANT**: Comprehensive reference for safe RL methodology selection
- Maps the entire landscape of Lyapunov + Barrier function methods for safe RL
- Helps choose between CBF variants for ground robot safety
- Combined CLF-CBF approaches particularly relevant for PE (safety + pursuit/evasion stability)
- **Essential survey for designing safety guarantees in PE systems**

---

## Cross-Paper Summary Table (Papers 1-10)

| Paper | Topic | Method | Key Contribution | HW Validation | 1v1 Relevance |
|-------|-------|--------|-----------------|---------------|---------------|
| 01 | Safe PE | RL + SMC + CBF | Three-term controller, Stackelberg | No | HIGHEST |
| 02 | Car-like PE | BiMDN + MADDPG | Belief encoding, curriculum | Yes (F1TENTH) | HIGHEST |
| 03 | CBF-RL Survey | Survey | CBF-RL integration taxonomy | N/A | HIGH |
| 04 | Self-Play Survey | Survey | Unified SP framework | N/A | HIGH |
| 05 | Safe RL | CBF filter + reward | Closed-form CBF + dual approach | No | HIGH |
| 06 | Safe RL | RCBF-QP + GP | Robust CBF + disturbance learning | No | HIGH |
| 07 | Sim-to-Real | Isaac→Gazebo→ROS2 | End-to-end transfer pipeline | Yes (TurtleBot4) | HIGH |
| 08 | Vision PE | Teacher-student + intent | Privileged learning + DAGGER | Yes (Unitree A1) | HIGHEST |
| 09 | Multi-UAV PE | MAPPO + OPEN | Evader prediction + curriculum | No | MODERATE |
| 10 | Safe RL Survey | Survey | Lyapunov + Barrier taxonomy (~110 papers) | N/A | HIGH |
