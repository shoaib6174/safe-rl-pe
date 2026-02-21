# Paper Summaries: Papers 11-14 (Progress Report)

## Paper 11: Game Theory and Multi-Agent RL: From Nash Equilibria to Evolutionary Dynamics
- **Authors**: De La Fuente, Noguer i Alonso, Casadella (UAB/AIFI/AllRead)
- **Venue**: arXiv, Dec 2024
- **Pages**: 20
- **Type**: Tutorial/Survey

### Key Content
Comprehensive mathematical treatment of four MARL challenges:
1. **Non-stationarity**: Moving target problem — each agent's optimal policy shifts as others learn. Bellman equation becomes time-dependent. Implications: convergence failure, sample inefficiency, credit assignment difficulty.
2. **Partial Observability**: POSG formulation with observation function O_i. Requires belief states b_i(s), introduces non-Markovian dependencies. LSTM/RNN needed for history.
3. **Scalability**: Joint action space |A| = product of |A_i| grows exponentially. Memory constraints, inference latency.
4. **Decentralized Learning**: No global coordinator → coordination failures, policy oscillations.

### Game-Theoretic Concepts Covered
- **Nash Equilibrium in stochastic games**: Minimax-DQN for two-player zero-sum (maximin profile)
- **Evolutionary Game Theory (EGT)**: Replicator dynamics, Boltzmann policy selection, MERL algorithm (evolutionary + policy gradient dual optimization)
- **Correlated Equilibrium**: Extends Nash via coordination device; Regret Minimization, Correlated Q-Learning (LP-based CE computation per state)
- **Adversarial MARL**: Zero-sum formulation r_1 = -r_2; opponent modeling via estimator; IDDPG, MADDPG (centralized critic), FACMAC (fully centralized), LOLA (opponent-learning awareness with higher-order gradients), GAIL (adversarial imitation)

### Relevance to 1v1 PE
- PE is a two-player zero-sum game → minimax formulations directly applicable
- Nash equilibrium policies = saddle point solutions of PE differential game
- MADDPG with centralized training, decentralized execution is standard for 1v1 PE
- LOLA could enable pursuer to anticipate evader's learning trajectory
- Opponent modeling critical for partial observability PE

---

## Paper 12: Pursuit and Evasion Strategy of a Differential Game Based on DRL (Frontiers 2022)
- **Authors**: Xu et al.
- **Venue**: Frontiers in Bioengineering and Biotechnology, 2022
- **Status**: RE-DOWNLOADED (previously corrupted, now correct)

### Problem Formulation
- "Dog-Sheep Game" pursuit-evasion scenario in circular constrained environment
- 1v1 PE with simplified dynamics (point-mass agents)
- Bounded circular arena as constraint

### Key Innovation
- Comparison of **DQN** (Deep Q-Network) vs **DDPG** (Deep Deterministic Policy Gradient) for PE
- Dog (pursuer) and sheep (evader) trained separately
- Discrete actions (DQN) vs continuous actions (DDPG) for PE comparison

### Methodology
- DQN: discrete action space (8 cardinal directions)
- DDPG: continuous action space (angle and speed)
- Circular environment constraint: agents reflected at boundary
- Simple point-mass dynamics with maximum speed limits
- Reward: distance-based for pursuer (negative distance), positive distance for evader

### Results
- DDPG outperforms DQN for both pursuer and evader (continuous control advantage)
- Pursuer learns to cut off evader's escape routes
- Evader learns to maximize distance and use boundary effectively
- Capture time decreases with training for pursuer, increases for evader

### Relevance to 1v1 PE
- **MODERATELY RELEVANT**: 1v1 PE with DRL but simplified dynamics
- DQN vs DDPG comparison useful for algorithm selection
- Simple circular environment limits transferability
- No safety considerations, no realistic robot dynamics
- Baseline reference for DRL-based PE approaches

---

## Paper 13: Safe Finite-Time Reinforcement Learning for Pursuit-Evasion Games
- **Authors**: Nick-Marios T. Kokolakis & Kyriakos G. Vamvoudakis (Georgia Tech)
- **Venue**: IEEE CDC 2022 (61st Conference on Decision and Control)
- **Pages**: 6

### Problem Formulation
- PE as zero-sum differential game in relative coordinates: x = x_p - x_e
- Dynamics: ẋ(t) = f(x) + G(x)u(t) + K(x)d(t) where u=pursuer, d=evader
- Cost functional: min_u max_d ∫L(x,u,d)dt (saddle-point problem)
- Safe set S defined via obstacle function h(x) = O(x-p_c) - c

### Key Innovation: Safe Finite-Time Stable Zero-Sum Game
- **Barrier function B(x)** integrated into running cost: L_s = L_1(x) + ψB(x) + u^T R_u u - d^T R_d d
- ψ > 0 trades off safety vs. optimality
- B(x) = g(x)/h(x) → grows unbounded near obstacle boundary → ensures safety
- Hamilton-Jacobi-Isaacs (HJI) equation derived with barrier-augmented cost
- **Safe Nash equilibrium**: saddle-point that is both finite-time stable AND keeps safe set positively invariant

### Learning Architecture
- **Critic-only** RL: single neural network approximates value function V(x) ≈ W^T φ(x)
- Approximate Nash strategies derived directly from critic: û = -½R_u^{-1}G^T φ'^T Ŵ, d̂ = ½R_d^{-1}K^T φ'^T Ŵ
- **Non-Lipschitz learning law** with experience replay for finite-time convergence (not just asymptotic)
- Finite-time settling: T(x_0) ≤ (V(x_0))^{1-η} / (c(1-η))

### Simulation Results
- Pursuer captures evader within t=1.5s while avoiding circular obstacle
- Without barrier function: collision occurs
- Critic weights converge in finite time
- Nash strategies converge to approximate Nash equilibrium

### Relevance to 1v1 PE
- **Directly applicable**: 1v1 PE with obstacle avoidance via barrier functions
- Finite-time guarantees important for real-time robotics
- Critic-only architecture is lightweight (no separate actor network)
- Same authors as Paper 14 — this is the conference version, dissertation extends it
- Limitation: continuous-state formulation, simulation-only, simple dynamics

---

## Paper 14: Fixed-Time RL-Based Control for Safe Autonomy (PhD Dissertation)
- **Author**: Nikolaos Marios Kokolakis (Georgia Tech, Advisor: Vamvoudakis)
- **Venue**: PhD Dissertation, August 2024
- **Pages**: 183 (full dissertation)

### Dissertation Structure
1. **Chapter 2**: Safe Finite-Time RL for PE in Unknown Environments using GPs (= expanded Paper 13 + Paper TNNLS)
2. **Chapter 3**: Bounded Rational PE Games using RL (Dubins aircraft kinematics, Level-k thinking, multi-evader assignment)
3. **Chapter 4**: Online Fixed-Time RL for Optimal Feedback Control
4. **Chapter 5**: Online Fixed-Time RL for Safety Verification using Reachability Analysis

### Key Contributions (PE-relevant chapters)
**Chapter 2 — Safe PE with GPs**:
- Same framework as Paper 13 but extended with:
  - Gaussian Processes to learn **unknown obstacle shapes/positions** online
  - GP posterior used to construct safe set S in real-time
  - Finite-time convergent critic learning
  - Safety via barrier functions integrated into cost
- Contribution: agent can safely play PE game in completely unknown cluttered environment

**Chapter 3 — Bounded Rational PE**:
- Moves beyond Nash equilibrium assumption
- **Level-k thinking model**: agents have bounded rationality, predict opponents at k-1 level
  - Level-0: random/naive
  - Level-1: best response to Level-0
  - Level-k: best response to Level-(k-1)
- **Dubins aircraft kinematics** for pursuer/evader (constant speed, turn rate control)
- Learning-based coordination for multi-pursuer assignment to multiple evaders
- Data-driven finite-time allocation algorithm

### Relevance to 1v1 PE
- **Highly relevant**: Combines safety (CBF-like barrier functions), game theory (HJI), and learning (critic-only RL)
- GP-based obstacle learning applicable to ground robots with unknown environments
- Level-k bounded rationality more realistic than Nash for real robots
- Dubins kinematics relatable to differential-drive/car-like ground robots
- Limitation: all simulation-based, control-theoretic (not deep RL), continuous-time

---

## Summary Statistics (Papers 11-14)
- Papers read: 3 valid + 1 corrupted
- Key themes: Game-theoretic MARL (Nash, CE, EGT), Safe PE with barrier functions, Finite-time convergence, GP-based environment learning, Bounded rationality
- Papers remaining to read: 15, 16, 17, 18, 19, 20 (6 more)
