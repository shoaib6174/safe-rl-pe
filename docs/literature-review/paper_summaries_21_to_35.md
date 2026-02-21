# Paper Summaries: Papers 21-35 (Newly Downloaded)

## Paper 21: DeepReach — A Deep Learning Approach to High-Dimensional Reachability
- **Authors**: Somil Bansal, Claire J. Tomlin (UC Berkeley)
- **Venue**: ICRA 2021
- **Year**: 2021

### Problem Formulation
- Hamilton-Jacobi (HJ) reachability analysis for high-dimensional systems
- Computing Backward Reachable Tubes (BRT) — set of states from which system inevitably reaches target despite worst-case disturbances
- Grid-based HJ methods scale exponentially (curse of dimensionality), limited to ~5-6D
- Formulated as zero-sum differential game: control (maximizer) vs disturbance (minimizer)

### Key Innovation: Neural Network Value Function
1. **Sinusoidal activations**: Sine functions accurately represent both value function AND its gradients (critical for safety controllers). ReLU/tanh/sigmoid are an order of magnitude worse.
2. **Self-supervised training**: Loss function derived from HJI VI itself — no ground-truth supervision needed
3. **Curriculum learning**: Training proceeds backward from terminal time T

### Methodology
- 3-layer DNN, 512 hidden units, sine nonlinearities
- 65k randomly sampled state-time pairs per iteration, Adam optimizer (lr=1e-4)
- Pre-training on terminal conditions (10k iterations) + 100k curriculum iterations
- Runtime: 16-25 hours on single GPU

### Results
- **3D Air3D (PE)**: MSE 1.9e-4, BRT volume error 0.43% vs analytical solution
- **6D collision avoidance**: MSE 2.1e-4 (complexity scales with signal, not dimensionality)
- **9D three-vehicle**: Captures multi-agent interactions missed by pairwise approximations
- **10D narrow passage**: Produces cooperative collision-avoidance behavior

### Relevance to 1v1 Ground Robot PE
- **VERY HIGH**: Air3D example IS a 1v1 PE game with Dubins car dynamics (closely related to ground robots)
- Computes exact game-theoretic solution (BRT) for PE
- Value function gradient = optimal safety/pursuit controller
- Can serve as expert policy baseline or warm-start for RL
- **Limitation**: Offline method (16-25 hrs), cannot adapt in real-time

---

## Paper 22: MADR — MPC-guided Adversarial DeepReach
- **Authors**: Ryan Teoh, Sander Tonkens, William Sharpless, Aijia Yang, Zeyuan Feng, Somil Bansal, Sylvia Herbert (UCLA/UCSD/Stanford)
- **Venue**: arXiv, October 2025
- **Year**: 2025
- **Link**: https://land-dev.github.io/madr/

### Problem Formulation
- Two-player zero-sum differential games via HJ reachability
- Control-and-disturbance affine dynamics
- Computing value function V(x,t) for safe set characterization against worst-case adversary
- HJI-VI intractable on grids in >6D

### Key Innovation: PINN + Adversarial MPC
1. **Value-only DeepReach (PINN)**: Self-supervised HJI-PDE learning, avoiding actor-critic co-learning instabilities
2. **Adversarial MPC supervision**: Sampling-based MPC generates datasets from BOTH players' perspectives
3. **MADR-FOLLOW filter**: Special pursuit-evasion filter for long-horizon games — switches between "follow" and "capture" policies
4. Opponent policy defined through current value gradient approximation

### Methodology
- Two complementary MPC datasets: ego-MPC + policy-driven disturbance, and vice versa
- Combined loss: DeepReach PDE loss + supervised MPC losses from both perspectives
- Curriculum training with iterative MPC data collection
- 3-layer sinusoidal NN, 512 neurons, RTX 4090

### Results
- **13D Quadrotor**: 98.9% safe rate (vs 86.6% ISAACS)
- **6D Dubins PE**: IOU 0.997 with ground truth (vs 0.969 Vanilla DeepReach)
- **20D Drone PE**: MADR evader avoids capture from all competing policies
- **HARDWARE**: TurtleBots (6D Dubins PE, 500s rollouts), Crazyflie 2.1, humanoid-vs-drone

### Relevance to 1v1 Ground Robot PE
- **VERY HIGH**: Explicitly solves 6D Dubins PE on TurtleBots (ground mobile robots!)
- Provides optimal strategies for BOTH pursuer AND evader from single value function
- MADR-FOLLOW addresses long-horizon PE (critical for real deployment)
- Hardware validated on TurtleBots with capture radius 0.36
- **Limitation**: Requires known dynamics model for MPC rollouts (not model-free RL)

---

## Paper 23: Convergence Guarantees for NN-Based Hamilton-Jacobi Reachability
- **Authors**: William Hofgard (Stanford)
- **Venue**: arXiv, October 2024
- **Year**: 2024

### Problem Formulation
- Does DeepReach converge to the true HJI solution?
- Given dynamics with control u and disturbance d, does the neural network value function V_theta approach the true V?

### Key Innovation: First Formal Convergence Proof
1. **Theorem 4.1 (Existence)**: For every ε > 0, there exists θ such that DeepReach loss L(θ) ≤ C·ε
2. **Theorem 4.2 (Uniform Convergence)**: If L(θ^(k)) → 0, then V_{θ^(k)} converges uniformly to true V on compact sets
3. **Sup-norm loss**: Switching from L^1/L^2 to L-∞ loss improves approximation quality

### Methodology
- Uses viscosity solution theory for HJI equations
- Constructs viscosity sub/supersolutions from NN sequence
- Applies comparison principles
- Verified on 3D two-vehicle collision avoidance (Dubins cars)

### Results
- Sup-norm fine-tuning for 1K epochs (~5 min) significantly improves models pre-trained for 100K epochs with L^1 loss
- BRT closely matches analytically-computed ground truth

### Relevance to 1v1 Ground Robot PE
- **MODERATE-HIGH (Theoretical)**: Validates that DeepReach-based PE solutions are theoretically sound
- Baseline experiment IS a PE scenario with Dubins car dynamics
- Sup-norm loss recommendation improves PE value function quality
- **Limitation**: Purely theoretical, no new algorithms or hardware experiments

---

## Paper 25: Hamilton-Jacobi Reachability in Reinforcement Learning: A Survey
- **Authors**: Milan Ganai, Sicun Gao, Sylvia Herbert (UC San Diego)
- **Venue**: IEEE Control Systems Society journal, 2024
- **Year**: 2024

### Key Content: Unified HJ-RL Framework
The ROADMAP paper bridging HJ reachability and RL.

**Taxonomy (Table 1)**: Classifies 14 primary works across:
- Problem type (optimal control, safe RL, reach-avoid, robust deployment)
- Model access (model-based vs model-free)
- Noise type (adversarial, stochastic, deterministic)
- Max dimensionality (2D to 112D)

**Key Sections**:
1. **Traditional HJ for learned controls**: Model-based, verify learned policies
2. **Learning reachability model-free**: Two Bellman formulations:
   - Non-discounted: exact but no γ-contraction, hard to learn
   - Discounted (Fisac et al.): guaranteed TD convergence, but approximation error
3. **Reach-avoid**: Combining reaching target + avoiding failure (directly PE-relevant)
4. **Model-free safe RL**: RCRL (deterministic), RESPO (stochastic)
5. **Shielding**: Task policy + safe backup policy for real-world deployment
6. **Limitations**: Discounting errors, sample efficiency, catastrophic forgetting

### Relevance to 1v1 Ground Robot PE
- **VERY HIGH**: This IS the roadmap for the PE project
- HJ reachability for differential games = PE games
- Reach-avoid = pursuer must reach evader while avoiding failure states
- Bellman formulations (Eq 8-10, 16-17, 22) = exact RL training objectives for PE value functions
- Shielding = aggressive RL pursuit + safe backup near dangerous states
- Covers Dubins car dynamics explicitly (Hsu et al. attack-defense)
- Model-free approaches for unknown ground robot dynamics

---

## Paper 27: RESPO — Iterative Reachability Estimation for Safe RL
- **Authors**: Milan Ganai, Zheng Gong, Chenning Yu, Sylvia Herbert, Sicun Gao (UC San Diego)
- **Venue**: NeurIPS 2023
- **Year**: 2023

### Problem Formulation
- Safe RL with persistent state-wise safety (not just cumulative constraints)
- Stochastic environments with probabilistic feasible set membership
- Avoid overly conservative behavior

### Key Innovation: Reachability Estimation Function (REF)
1. **REF φ^π(s) ∈ [0,1]**: Probability of EVER reaching violation state from s under policy π
   - Extends HJ reachability from binary (deterministic) to probabilistic (stochastic)
   - Bellman recursion: φ^π(s) = max{1_{s∈S_v}, E_{s'}[φ^π(s')]}
2. **Dual optimization**: In feasible set → maximize reward; in infeasible set → minimize violations
3. **Four-timescale actor-critic**: Provable convergence with critics (fastest) → policy → REF → Lagrange multiplier (slowest)

### Results
- Safety Gym (76D obs): ~3x fewer violations than RCRL with highest reward
- Safety MuJoCo: HalfCheetah with 0 violations, highest reward among safe methods
- Multi-drone: correctly prioritizes hard constraints over soft constraints
- Outperforms PPOLag, CRPO, P3O, PCPO, RCRL, CBF, FAC

### Relevance to 1v1 Ground Robot PE
- **MODERATE-HIGH**: Safety layer for PE robots
- REF reinterpretable for PE: probability of capture from any state
- Handles stochastic dynamics (wheel slip, sensor noise)
- Scalable to rich sensor inputs (76D)
- **Limitation**: Single-agent, no adversarial opponent modeling

---

## Paper 28: LBAC — RL for Safe Robot Control using Control Lyapunov Barrier Functions
- **Authors**: Desong Du, Shaohang Han, Naiming Qi, Haitham Bou Ammar, Jun Wang, Wei Pan (HIT/TU Delft/Huawei/UCL/Manchester)
- **Venue**: ICRA 2023
- **Year**: 2023

### Problem Formulation
- Reach-avoid in CMDPs: reach S_goal while NEVER entering S_unsafe
- Model-free setting (unknown dynamics)

### Key Innovation: Unified CLBF + Model-Free RL
1. **Control Lyapunov Barrier Function (CLBF)**: Single certificate for BOTH reachability (CLF) and safety (CBF)
   - Avoids CLF-CBF conflict where gradients oppose each other → infeasible QP
2. **LBAC (Lyapunov Barrier Actor-Critic)**: Critic approximates CLBF, actor learns safe policy
3. **Data-based CLBF Theorem**: Sufficient conditions for safety without explicit dynamics model

### Methodology
- Critic Q_LB trained as valid CLBF via Lagrangian relaxation
- Maximum entropy for exploration
- 2-layer, 256 neurons, ReLU
- PyBullet gym-pybullet-drones simulator

### Results
- Converges within ~2300 episodes with fewest safety violations
- **Sim-to-real**: CrazyFlie 2.0 quadrotor, 2D navigation with obstacles
- Outperforms RSPO, SQRL, RCPO in both reachability and safety
- CLF-CBF-QP approach gets stuck due to gradient conflicts; LBAC does not

### Relevance to 1v1 Ground Robot PE
- **MODERATE**: Reach-avoid structurally related to PE
- Model-free with sim-to-real validation
- CLBF could encode pursuit objectives + safety
- **Limitation**: Single-agent, static obstacles, no adversarial game formulation

---

## Paper 32: Diffusion-RL Hierarchical Motion Planning in Adversarial Games
- **Authors**: Zixuan Wu, Sean Ye, Manisha Natarajan, Matthew C. Gombolay (Georgia Tech)
- **Venue**: arXiv, 2024 (updated May 2025)
- **Year**: 2024

### Problem Formulation
- Evasive target in partially observable multi-agent adversarial PE game (POMDP)
- Evader navigates to hideout while avoiding heterogeneous pursuit team
- Evader: limited FOV, no prior terrain knowledge, slower than pursuers

### Key Innovation: Hierarchical Diffusion + RL
1. **High-level diffusion model**: Generates diverse global paths from RRT* training data, conditioned on start/goal/obstacles
2. **Low-level SAC policy**: Follows waypoints while adapting evasive behavior
3. **Implicit cost map**: Records detection-risk positions from training episodes, selects safest diffusion path at inference

### Results
- **+77.18%** detection avoidance, **+47.38%** goal-reaching vs baselines
- **+51.4%** overall performance improvement
- Diffusion paths **85.7% faster** than RRT*
- Saves **25.2 hours** of training time
- Validated on **Robotarium** physical robot testbed

### Relevance to 1v1 Ground Robot PE
- **MODERATE-HIGH**: Hierarchical framework transferable to 1v1
- POMDP + partial observability handling relevant
- Risk-aware path selection applicable
- Robotarium validation demonstrates physical feasibility
- **Limitation**: Evader-only, fixed heuristic pursuers, 2D point dynamics

---

## Paper 34: SHADOW — Learning Information Trade-offs in Pursuit-Evasion Games
- **Authors**: Valerio La Gatta, Dolev Mutzari, Sarit Kraus, VS Subrahmanian (Northwestern/Bar-Ilan)
- **Venue**: arXiv, October 2025
- **Year**: 2025

### Problem Formulation
- **PEEC Game**: 1v1 PE with information-exposure trade-off
- Pursuer can query evader position but REVEALS its own position
- Non-holonomic unicycle dynamics: x_dot = v·cos(ψ), y_dot = v·sin(ψ), ψ_dot = u/v
- Partial observability with strategic information acquisition
- Asymmetric speeds (v_e/v_p varied)

### Key Innovation: Multi-Headed Sequential RL (SHADOW)
1. **Navigation Module**: TD3 + LSTM for continuous control
2. **Query Decision Module**: PPO + LSTM for discrete communication decisions
3. **Opponent Modeling Module**: TD3-based predictor for evader position + uncertainty
4. **Cost of Information Acquisition (CIAC)**: First formal quantification

### Results
- P_win = 62% (vs 18.4% no-communication, 57.6% best periodic, 39.6% P-DQN)
- LSTM memory critical: without it, P_win drops to 11.2%
- Three-phase learning: avoid shots → experiment with communication → strategic timing
- Adapts communication frequency based on threat level

### Relevance to 1v1 Ground Robot PE
- **VERY HIGH**: 1v1 PE with non-holonomic dynamics!
- Unicycle model = ground robot model
- Deep RL (TD3 + PPO) for both agents
- Modular architecture transferable
- Opponent modeling with uncertainty
- **Limitation**: Simulation only, "shooting" mechanic specific to PEEC

---

## Paper 35: NeHMO — Neural HJR for Decentralized Safe Multi-Agent Motion Planning
- **Authors**: Qingyi Chen, Ahmed H. Qureshi (Purdue)
- **Venue**: arXiv, July 2025
- **Year**: 2025

### Problem Formulation
- Safe multi-agent motion planning via HJ reachability
- BRT computation via HJI equation as zero-sum game
- Homogeneous agents, perfect state perception

### Key Innovation
1. **Symmetry exploitation**: Training on half the state space, bijective mapping for full inference
2. **Explicit boundary condition**: Learns residual V_res = V - l(x), decoupling boundary from NN
3. **MPC-based trajectory optimization** using learned HJR

### Results
- 85% success rate with 0% collisions (8 agents)
- **12-DoF Dual-UR5**: 92% SR, 2% CR, 36ms planning (real-time 25Hz)
- Outperforms vanilla DeepReach significantly on high-dimensional systems

### Relevance to 1v1 Ground Robot PE
- **LOW-MODERATE**: Cooperative collision avoidance, not adversarial PE
- HJI framework inherently models differential games (theoretically PE-relevant)
- Air3D case study is classic PE benchmark
- Could serve as safety layer for RL-based PE
- **Limitation**: Not RL, cooperative, full observability

---

## Paper 24: Enhancing the Performance of DeepReach through Optimizing Activation Functions
- **Authors**: Qian Wang, Tianhao Wu (University of Southern California)
- **Venue**: arXiv (arXiv:2312.17583)
- **Year**: 2023

### Problem Formulation
- DeepReach uses sinusoidal activations for accurate HJI value function gradients, but optimization is harder vs ReLU
- Investigates whether mixed Sine/ReLU activation layers can improve DeepReach performance on high-dimensional systems

### Key Innovation
- **Intertwined Sine/ReLU activations**: Systematically tests combinations across layers
- First and last layers have dominant effect on performance
- Sine on both first and last layers keeps violation rates below 20%

### Results
- Best config (6-layer all-Sine "sssssl"): **18.43% violation** (vs 19.0% baseline)
- Air3D (3D): Sine layers critical for BRT accuracy (MSE ~10^-4)
- 9D multi-vehicle: All-ReLU ~23.5% violation, mixed configs 18-27%

### Relevance to 1v1 Ground Robot PE
- **MODERATE**: Air3D benchmark IS a PE game with Dubins dynamics
- Activation function insights applicable to any NN-based PE value function
- Narrow contribution (empirical study), modest improvements
- No RL, no hardware experiments

---

## Paper 26: Neural Networks in Optimizing Pursuit-Evasion Game Tactics (HNSN)
- **Authors**: Zhenhua Guo, Wei Wang, Lili Sun, Biqiang Han, Ying Zhang (Nanjing U. Aero & Astro, et al.)
- **Venue**: Dynamic Games and Applications (Springer)
- **Year**: 2024/2025

### Problem Formulation
- Multi-pursuer PE in continuous 3D space with environmental disturbances
- Cost minimization over time horizon with velocity, pitch, yaw controls

### Key Innovation: Hybrid Neural Strategy Network (HNSN)
1. **ANN component**: Reactive short-term control with custom "s-parabola" activation
2. **DNN component**: Long-term strategic trajectory planning via approximate HJB
3. **Blended control**: u_HNSN = α·u_ann + (1-α)·u_dnn
4. Apollonius sphere geometry for interception

### Results
- HNSN: MAE 0.20, success rate 94.12% (vs DNN 90.41%, ANN 85.82%)
- Evader escapes in 9/10 scenarios with HNSN strategy

### Relevance to 1v1 Ground Robot PE
- **LOW**: Not RL-based (supervised learning), 3D aerial dynamics, multi-pursuer focus
- Hybrid reactive/strategic concept could inform RL architecture design
- No real-robot experiments, limited evaluation

---

## Paper 31: Safe Multi-Agent RL via Approximate Hamilton-Jacobi Reachability
- **Authors**: Kai Zhu, Fengbo Lan, Wenbo Zhao, Tao Zhang (Tsinghua University)
- **Venue**: Journal of Intelligent & Robotic Systems (JIRS), Vol 111
- **Year**: 2024

### Problem Formulation
- Safe MARL with state-wise safety constraints (collision avoidance)
- Decentralized CMDP with HJ reach-avoid formulation
- V(s) ≥ 0 indicates membership in reach-avoid safe set

### Key Innovation: Safe HJ MADDPG
1. **Centralized Discounted Reach-Avoid Critic**: Learns HJ value function via DRABE (contraction mapping)
2. **Conservative Value Iteration**: CQL-style regularization → under-estimates safe set → robust to non-cooperative agents
3. **Fully Decentralized Execution**: 0.1ms inference, no centralized shielding needed at runtime

### Results
- Cooperative: 0.06 collisions/episode (vs MADDPG 11.30, Hard MADDPG 2.92)
- Disturbed (shield disabled): 0.07 collisions (vs Soft MADDPG 1.57)
- Non-cooperative: 1.80 collisions (vs Soft MADDPG 5.31)
- Scales to N=12 agents without retraining
- Predator-Prey experiment demonstrates PE-like competitive task

### Relevance to 1v1 Ground Robot PE
- **MODERATE**: HJ reachability + model-free deep RL integration is directly relevant
- Discounted Reach-Avoid Bellman Equation applicable to PE value functions
- Conservative Q-learning for safety robustness transferable
- **Limitation**: Multi-agent cooperative focus, point-mass dynamics, no real hardware

---

## Paper 33: Fast and the Furious — Hot Starts in Pursuit-Evasion Games
- **Authors**: Gabriel Smithline (U. Michigan), Scott Nivison (Air Force Research Lab)
- **Venue**: arXiv (arXiv:2510.10830)
- **Year**: 2025

### Problem Formulation
- Strategic initial pursuer placement before evaders appear in multi-agent PE
- Optimizing "hot starts" that maximize capture probability against unknown evader positions

### Key Innovation
1. **Graph Feature Space (GFS)**: Encodes configs via capture potential, distance, heading features
2. **Pareto Frontier** via NSGA-II multi-objective optimization
3. **GCN** trained on Pareto-optimal configs with custom Pareto loss
4. Inspired by DeepMind's TacticAI (sports positioning)

### Results
- Hot starts reduce evader survival across all game types (p < 0.001)
- Strategic spatial patterns emerge (diagonal lines, crescents, S-curves)
- Diminishing returns at high pursuer-to-evader ratios
- GCN Pareto loss: 0.0234 at convergence

### Relevance to 1v1 Ground Robot PE
- **LOW**: Multi-pursuer placement problem, not 1v1 or RL-based
- GFS features (capture potential, distance, heading) could inspire reward design
- GNN for spatial relationship encoding transferable to multi-agent RL
- **Limitation**: Initial config only (not control policy), no learning-based control

---

## Paper 30: CASRL — Conflict-Averse Safe RL for Autonomous Navigation
- **Authors**: Zhiqian Zhou, Junkai Ren, Zhiwen Zeng, Junhao Xiao, Xinglong Zhang, Xian Guo, Zongtan Zhou, Huimin Lu (NUDT / NanKai)
- **Venue**: CAAI Transactions on Intelligence Technology (IET/Wiley)
- **Year**: 2023

### Problem Formulation
- Autonomous mobile robot navigation in dynamic environments (pedestrians via ORCA)
- Safe MDP with separate reward (goal-reaching) and cost (collision avoidance) functions
- Differential-drive robot with continuous acceleration actions

### Key Innovation: Conflict-Averse Gradient Manipulation
1. **Task decomposition**: Separate policy gradients g_r (goal) and g_c (safety) from dual critics
2. **CAPO optimization**: Finds g* = ω_r·g_r + ω_c·g_c that maximizes worst local improvement across both sub-tasks
3. **HGAT state representation**: Heterogeneous Graph Attention Network encoding robot-obstacle interactions
4. Built on TD3 backbone

### Results
- 84.9% success rate (vs 76.7% vanilla HGAT-DRL), +8.2% average, +13.4% in hardest scenario
- Collision rate: 14.8% (vs 23.3% baseline)
- Training overhead: only ~6.7% additional time
- **Real-world**: 100% success on Fetch robot (40 navigation tasks), 0.643 m/s avg speed

### Relevance to 1v1 Ground Robot PE
- **MODERATE**: Differential-drive robot + safe RL + real deployment
- Conflict-averse gradient technique transferable to PE (pursuit vs safety trade-off)
- HGAT for agent interaction encoding relevant
- **Limitation**: Non-adversarial (ORCA pedestrians), no game theory, navigation not PE

---

## Paper 36: Bio-Inspired Neural Network for Real-Time Evasion
- **Authors**: Junfei Li, Simon X. Yang (University of Guelph)
- **Venue**: Biomimetics (MDPI), Vol 9
- **Year**: 2024

### Problem Formulation
- Multi-evader (3) vs single faster pursuer (speed ratio 3:1) on 2D grid
- Dynamic environments with appearing/disappearing obstacles
- Pursuer uses greedy nearest-target strategy

### Key Innovation: Bio-Inspired Neural Network (BINN)
1. **Shunting neurodynamics** (Hodgkin-Huxley derived): neural activity landscape for evasion
2. **Global vs local asymmetry**: pursuer signal propagates globally, obstacles remain local
3. **No training required**: purely reactive, real-time response to environmental changes
4. **Proven collision-free** (Theorem 2): obstacle neurons always negative, evasion selects only non-negative

### Results
- Outperforms DQN self-play in dynamic environments: 24.70 vs 23.57 survival time (moving obstacles)
- DQN slightly better in static: 23.10 vs 21.97
- Validated on 3 ROS ground robots (Raspberry Pi 4, RPLIDAR A1)

### Relevance to 1v1 Ground Robot PE
- **MODERATE**: PE on ground robots with real hardware
- Useful as non-learning baseline for comparison
- Collision-free guarantee is valuable
- **Limitation**: Not RL, grid-based point robots, simple greedy pursuer, multi-evader focus

---

## Paper 37: A Review of RL Approaches for Pursuit-Evasion Games
- **Authors**: Kun Yang, Ao Shen, Nengwei Xu, Fang Deng, Maobin Lu, Chen Chen (Beijing Institute of Technology)
- **Venue**: Chinese Journal of Aeronautics (Elsevier)
- **Year**: 2025 (pre-proof)
- **References**: 235

### Key Taxonomy: Strategy Learning Paradigm
**A. Unilateral** (one agent learns, opponent fixed):
- Direct interaction: DQN, PPO, DDPG, SAC
- Opponent assumption: Minimax-Q, CFR, Fictitious Play
- Opponent modeling: DPIQN, PR2 (explicit/implicit)

**B. Bilateral Alternating** (agents take turns training):
- Adversarial learning: GAN-inspired, Asymmetric Dueling (AD) with curriculum
- Self-play: FSP, NFSP, **PSRO** (highlighted as "groundbreaking")

**C. Bilateral Simultaneous** (both update together):
- Centralized: Nash-Q, MADDPG, MAPPO, VDN, QMIX
- Distributed: IPPO, Mean Field RL

### Application Domains
- **Tactical combat** (18+ papers): air combat majority, MADDPG/MAPPO fastest convergence
- **Unmanned systems** (15+ papers): active tracking, sim-to-real, camera-based
- **Spacecraft interception** (12+ papers): orbital PE, meta-RL

### Key Findings
- PSRO best for strategy-rich games but highest complexity
- MADDPG/MAPPO fastest convergence for multi-agent
- Sim-to-real remains critical bottleneck
- Hierarchical RL prevalent for sparse-reward PE decomposition

### Papers We May Be Missing (from 235 refs)
- **AD-VAT** (Zhong 2021, TPAMI): Asymmetric dueling for visual active tracking
- **Luo 2020** (TPAMI): End-to-end active object tracking, sim-to-real
- **Devo 2021**: C-VAT continuous control mobile robot tracking
- **NFSP** (Heinrich & Silver 2016): Neural Fictitious Self-Play
- **PSRO** (Lanctot 2017): Policy Space Response Oracle framework

### Relevance to 1v1 Ground Robot PE
- **HIGH**: First comprehensive RL-PE taxonomy, directly applicable to algorithm selection
- Strategy learning paradigm classification guides training design
- Sim-to-real discussion relevant to deployment
- **Limitation**: Aerospace bias, no HJ-RL coverage, limited safety-constrained RL discussion, no quantitative benchmarking

---

## Summary Table: All New Papers

| # | Paper | Year | Method | PE Setting | HW | Relevance |
|---|-------|------|--------|-----------|------|-----------|
| 21 | DeepReach | 2021 | Neural HJI | 1v1 (Air3D) | No | VERY HIGH |
| 22 | MADR | 2025 | PINN+MPC | 1v1+Multi | TurtleBot,Crazyflie | VERY HIGH |
| 23 | Convergence | 2024 | Theory | PE analysis | No | MOD-HIGH |
| 24 | DeepReach Activations | 2023 | Sine/ReLU mix | BRT computation | No | MODERATE |
| 25 | HJ-RL Survey | 2024 | Survey | All | N/A | VERY HIGH |
| 26 | HNSN | 2024 | ANN+DNN hybrid | Multi-pursuer 3D | No | LOW |
| 27 | RESPO | 2023 | Safe RL | Safety layer | No | MOD-HIGH |
| 28 | LBAC | 2023 | CLBF+RL | Reach-avoid | CrazyFlie | MODERATE |
| 30 | CASRL | 2023 | TD3+CAPO | Safe navigation | Fetch robot | MODERATE |
| 31 | Safe HJ MADDPG | 2024 | HJ+CQL+MADDPG | Safe MARL | No | MODERATE |
| 32 | Diffusion-RL | 2024 | Diffusion+SAC | Evader only | Robotarium | MOD-HIGH |
| 33 | Hot Starts PE | 2025 | GCN+NSGA-II | Multi-pursuer init | No | LOW |
| 34 | SHADOW | 2025 | TD3+PPO | 1v1 PEEC | No | VERY HIGH |
| 35 | NeHMO | 2025 | Neural HJR+MPC | Cooperative | No | LOW-MOD |
| 36 | BINN Evasion | 2024 | Bio-inspired NN | Multi-evader | ROS robots | MODERATE |
| 37 | RL-PE Review | 2025 | Survey (235 refs) | All PE | N/A | HIGH |
