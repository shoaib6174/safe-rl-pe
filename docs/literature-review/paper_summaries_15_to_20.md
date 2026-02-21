# Paper Summaries: Papers 15-20

## Paper 15: Mobile Robot Control Using PE Differential Game Strategy with DRL
- **Authors**: Li Zhenxiang et al.
- **Venue**: Dynamic Games & Applications (Springer), 2025
- **Type**: Research article

### Problem Formulation
- 1v1 PE with double integrator dynamics (position + velocity states)
- Differential game formulation seeking Nash equilibrium
- Mobile robot setting with continuous control

### Key Innovation
- **A-MPC + DRL hybrid**: Combines Adaptive Model Predictive Control with Deep RL (DQN)
- Adaptive reward mechanism that changes with game stage
- Nash equilibrium analysis for pursuit-evasion strategies

### Methodology
- DQN (Deep Q-Network) for discrete action selection
- A-MPC provides model-based trajectory optimization
- Double integrator dynamics: x_dot = v, v_dot = u
- Adaptive reward: early game rewards exploration, late game rewards capture/evasion

### Results
- Pursuer successfully captures evader in various initial configurations
- Nash equilibrium strategies emerge from training
- A-MPC component improves sample efficiency over pure DRL

### Relevance to 1v1 Ground Robot PE
- **DIRECTLY RELEVANT**: 1v1 mobile robot PE with DRL
- Double integrator dynamics are simplified but relatable to ground robots
- Nash equilibrium analysis provides theoretical grounding
- Hybrid A-MPC + DRL approach could be extended to more complex dynamics
- **Limitation**: Simple dynamics model, simulation only

---

## Paper 16: Sampling-Based Safe Reinforcement Learning for Nonlinear Dynamical Systems
- **Authors**: Wesley A. Suttle, Vipul K. Sharma, Krishna C. Kosaraju, S. Sivaranjani, Ji Liu, Vijay Gupta, Brian M. Sadler
- **Venue**: AISTATS 2024 (International Conference on Artificial Intelligence and Statistics)
- **Year**: 2024
- **arXiv**: 2403.04007

### Problem Formulation
- Addresses the fundamental tension: safety-filter methods destroy RL convergence guarantees, while standard safe RL methods can't provide hard constraint satisfaction
- Discounted MDP with deterministic dynamics and state-dependent safe control set C(x) defined via CBFs
- Goal: maximize reward while **only ever sampling actions from C(x)** -- safety during both training AND deployment

### Key Innovation: Truncated Stochastic Policies
- **Single-stage approach** avoiding the "learn then project" two-stage paradigm
- Truncated policy: pi_theta^C(u|x) = pi_theta(u|x) / pi_theta(C(x)|x) for u in C(x), else 0
- Policy distribution renormalized over the safe set rather than clipped/projected
- **Convergence theorem** (Theorem 1): guarantees asymptotic convergence to stationary points under truncated policies
- **CBF-constrained Beta policies**: Beta distributions whose support is rescaled to CBF-derived bounds

### Methodology
- Safe-RPG Algorithm with random-horizon policy gradient scheme
- Actions sampled via rejection sampling from truncated policy
- Monte Carlo estimation of safe set probability pi_theta(C(x)|x)
- PPO adapted from Stable Baselines 3
- Experiments: quadcopter navigation with obstacle avoidance, safety-constrained inverted pendulum

### Results
- CBF-constrained Beta policies achieve ~100% safety throughout training
- Safety-filter Gaussian policies FAIL when obstacles interfere with direct path
- Truncated policies maintain both convergence and safety guarantees simultaneously

### Relevance to 1v1 Ground Robot PE
- **HIGHLY RELEVANT as enabling technology**: For PE robots that must avoid obstacles/boundaries during RL training
- Solves critical problem of guaranteeing safety constraints during physical training
- CBF-constrained Beta policy could be directly applied to pursuer/evader policy parameterization
- Framework is general -- applicable to ground robot dynamics (unicycle, differential drive)
- **Limitation**: Single-agent, no adversarial/game-theoretic formulation

---

## Paper 17: ViPER: Visibility-based Pursuit-Evasion via Reinforcement Learning
- **Authors**: Yizhuo Wang, Yuhong Cao, Jimmy Chiun, Subhadeep Koley, Mandy Pham, Guillaume Sartoretti (NUS/IIEST/UC Berkeley)
- **Venue**: CoRL 2024 (8th Conference on Robot Learning), Munich
- **Year**: 2024
- **Code**: https://github.com/marmotlab/ViPER

### Problem Formulation
- **Visibility-based pursuit-evasion** (area clearing/adversarial search)
- Team of n pursuers with limited omnidirectional sensing (range r_fov) must guarantee detection of ALL evaders
- Worst-case evaders: omniscient, potentially infinitely fast
- Environment classified as cleared vs contaminated; recontamination possible
- Objective: expand cleared area to entire free space, minimize trajectory length

### Key Innovation
1. **Graph Attention Network** for multi-agent coordination: encoder-decoder with 6 masked self-attention layers
2. **Privileged learning for critic**: critic sees ground-truth map, policy sees partial observations
3. **Attentive critic** incorporating other agents' actions via attention (from MAAC)
4. 7 carefully designed node features (relative position, occupancy, utility, cleared signal, guidepost)

### Methodology
- Environment represented as graph G_t = (V_t, E_t) with candidate viewpoints
- SAC (Soft Actor-Critic) for training
- Policy decoder: attention scores over neighboring nodes = action probabilities
- 4000 random multiply-connected maps (100m x 100m), 40 hours training on RTX 3090
- Reward: area-cleared increment - trajectory cost + completion bonus

### Results
- **Significantly outperforms classical planners**: 86-100% success rate vs 24-89% for Durham et al.
- Generalizes to unseen maps and larger environments (4x) without retraining
- Robust to agent failure (adapts dynamically)
- **Hardware validation**: 3 Crazyflie drones + 1 TurtleBot3, inference < 0.01s per agent
- First neural solution for visibility-based PE with worst-case guarantees

### Relevance to 1v1 Ground Robot PE
- **MODERATELY RELEVANT**: Different PE formulation (area clearing vs capture)
- Graph attention architecture transferable to obstacle-rich 1v1 settings
- Privileged learning approach useful for stabilizing adversarial training
- Observation space design (frontier/visibility encoding) informative
- Hardware deployment validates RL for PE on physical robots
- **Limitation**: Multi-agent team formulation, not 1v1 capture game

---

## Paper 18: Learning Multi-Pursuit Evasion for Safe Targeted Navigation of Drones (AMS-DRL)
- **Authors**: Jiaping Xiao, Mir Feroskhan
- **Venue**: arXiv (submitted to IEEE), 2024
- **Year**: 2024

### Problem Formulation
- Multi-Pursuit Evasion with Targeted Navigation (MPETN): 1 runner drone vs 2+ chaser drones
- 3D indoor environment, runner must reach target while evading capture
- Not zero-sum: runner has additional navigation objective
- High-level velocity command model for 6-DOF quadrotor

### Key Innovation: AMS-DRL (Asynchronous Multi-Stage Deep RL)
1. **Cold-Start Learning (S0)**: Train runner for navigation without adversaries
2. **Asynchronous Learning (S1-Sk)**: Alternately train chaser and runner teams, freezing the other
3. **Nash Equilibrium convergence guarantee**: Proven unique NE existence + convergence when success rate difference < threshold eta

### Methodology
- PPO for policy updates at each stage
- 3 FC hidden layers (512, 512, 4), Sigmoid activation
- Runner observes relative positions to target and chasers
- Chasers observe relative positions to each other and runner
- 4 continuous velocity commands (v_x, v_y, v_z + yaw rate)
- Unity-based 3D simulation, ML-Agents Toolkit
- Sim2Real: Tello Edu drones with OptiTrack, ROS Melodic, ONNX inference

### Results
- Convergence at phase S5, eta = 10%
- Runner success: 45.31% vs AMS-DRL chasers (best adversary)
- Runner success: 95.1% vs random chasers
- **Physical experiments**: 189 real-world flights, 39.2% success (matches sim 37.1%)
- Inference: ~1.6ms on Raspberry Pi 4B

### Relevance to 1v1 Ground Robot PE
- **HIGHLY RELEVANT**: Self-play training framework directly transferable to 1v1
- AMS-DRL staged training provides practical curriculum for pursuer/evader
- NE convergence guarantee applicable (even simpler for 1v1)
- Successful sim-to-real validates pipeline for PE games
- PPO confirmed as strong choice for continuous PE
- Speed ratio analysis useful for hardware selection
- **Limitation**: 3D drone dynamics; needs adaptation for ground robots

---

## Paper 19: Optimal Strategy for Aircraft PE Games via Self-play Iteration
- **Authors**: Xin Wang, Qing-Lai Wei, Tao Li, Jie Zhang
- **Venue**: Machine Intelligence Research, vol. 21, no. 3, pp. 585-596
- **Year**: 2024

### Problem Formulation
- Two-player zero-sum differential game: pursuer vs evader aircraft in 3D
- Open-loop: players cannot observe each other's state
- 3-DOF point-mass aircraft dynamics with state vector [v_P, psi_P, gamma_P, v_E, psi_E, gamma_E, dx, dy, dz]
- Pursuer minimizes, evader maximizes cost J (terminal distance + control energy)

### Key Innovation: Self-play Iteration via NLP
1. **Decomposition**: Bilateral game → two alternating one-sided optimal control problems
2. **Direct collocation**: Continuous-time OCP → NLP via trapezoidal quadrature
3. **Convergence proof**: Alternating optimization converges to Nash Equilibrium under uniqueness assumption

### Methodology
- Classical optimal control (Pontryagin's Maximum Principle) + NLP solvers
- Self-play Algorithm 1: Initialize evader → optimize pursuer → optimize evader → repeat
- Control bounds: N_xa in [-2,3], N_za in [-9,9], phi in [-45,45] degrees
- Time horizon = 1 second, initial velocity = 150 m/s

### Results
- Symmetric case: identical strategies, constant relative distance (confirms NE)
- Asymmetric case: distinct strategies reflecting different capabilities
- Convergence within few iterations

### Relevance to 1v1 Ground Robot PE
- **LOW-MODERATE**: Classical optimal control, not deep RL
- Theoretical foundation: proves self-play iteration converges to NE -- validates self-play RL
- Aircraft dynamics not applicable to ground robots
- Open-loop assumption unrealistic for most robot PE
- **Takeaway**: Mathematical justification for why self-play training works in PE games

---

## Paper 20: Emergent Behaviors in Multiagent PE Games within a Bounded 2D Grid World
- **Authors**: Sihan Xu, Zhaohui Dang
- **Venue**: Scientific Reports (Nature), vol. 15, article 29376
- **Year**: 2025

### Problem Formulation
- 2 pursuers vs 1 evader on bounded 2D grid
- Discrete time/space, complete information
- Equal speeds (u_P^max = u_E^max = 1 cell/step)
- 4 cardinal directions + stay, capture when Manhattan distance <= r_s
- Max 50 rounds per game

### Key Innovation
1. **Systematic cooperative action taxonomy**: 6 single-pursuer actions (flank, engage, ambush, drive, chase, intercept) → 21 two-pursuer cooperative actions
2. **K-means clustering for behavior recognition**: Trajectories clustered to identify 4 emergent strategies
3. **Four emergent strategies identified**:
   - Serpentine corner encirclement
   - Stepwise corner approach
   - Same-side boundary suppression
   - Two-sided pincer movement
4. **"Lazy pursuit" discovery**: Social loafing emerges from shared rewards

### Methodology
- MADDPG (centralized training, decentralized execution)
- Actor/Critic: 2 hidden layers, 128 neurons each
- Observation: own position + relative differences to all agents
- Reward: time penalty + outcome reward (no distance shaping)
- 200,000 episodes training
- Analysis: action classification → K-means clustering → t-SNE visualization

### Results
- 99.9% pursuer success rate (1000 trials)
- 675 boundary captures, 100 corner captures
- Lazy pursuit: 82 cases in Strategy 1 (eliminated by individual rewards)
- With obstacles: ~96.8% success rate
- Without boundaries: need 2x speed advantage for capture

### Relevance to 1v1 Ground Robot PE
- **MODERATE**: 2v1 grid world, but analysis tools are transferable
- Action taxonomy (flank, drive, chase, intercept) applicable to analyzing 1v1 behaviors
- K-means trajectory clustering methodology for interpreting learned strategies
- Boundary effects analysis relevant for bounded arena PE
- Speed ratio findings useful for hardware design
- MADDPG baseline for PE training
- **Limitation**: Discrete grid, simple dynamics, 2v1 not 1v1

---

## Cross-Paper Summary Table

| Paper | Domain | Agents | Method | Key Contribution | HW Validation | 1v1 Relevance |
|-------|--------|--------|--------|-----------------|---------------|---------------|
| 15 | Ground robot | 1v1 | A-MPC + DQN | Hybrid control + DRL | No | HIGH |
| 16 | General control | Single | CBF-Beta PPO | Safe policy parameterization | No | HIGH (enabling) |
| 17 | Indoor environments | Multi-team | SAC + GAT | Visibility-based PE with RL | Yes (drones+TurtleBot) | MODERATE |
| 18 | 3D indoor | 1 vs 2+ | PPO + AMS-DRL | Staged self-play with NE proof | Yes (Tello drones) | HIGH |
| 19 | 3D aircraft | 1v1 | NLP optimal control | Self-play NE convergence proof | No | LOW-MODERATE |
| 20 | 2D grid | 2v1 | MADDPG | Emergent behavior taxonomy | No | MODERATE |
