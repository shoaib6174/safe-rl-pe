# Newly Identified Papers for Safe RL + Pursuit-Evasion

## Papers Found During Plan Review (Feb 2026)
### Summaries: PDF-Verified (updated 2026-02-21)

---

### N01. Learning CBFs and their Application in RL: A Survey
- **Authors**: Maeva Guerrier, Hassan Fouad, Giovanni Beltrame
- **Date**: April 2024
- **Venue**: arXiv (2404.16879)
- **URL**: https://arxiv.org/abs/2404.16879
- **Status**: PDF-verified
- **Summary**: Comprehensive survey on data-driven approaches to learning CBFs for safe RL in robotics. Categorizes approaches into three main classes: (1) learning the CBF itself (neural CBFs, SOS-based, etc.), (2) learning the dynamics model to be used within a CBF framework, and (3) using CBFs within RL training as safety constraints or reward shaping.
- **Relevance to Pathway A**: **HIGH** -- Provides systematic overview of learned CBF techniques. If hand-crafted CBFs prove limiting (e.g., for complex obstacle shapes), learned CBFs offer an alternative. The survey bridges safe RL with control theory for faster sim-to-real transfer.
- **Key Insight**: Learned CBFs can adapt to environments where analytical CBF construction is intractable. The three-way categorization helps position our CBF-Beta approach within the broader landscape.

---

### N02. How to Train Your Neural Control Barrier Function (PNCBF)
- **Authors**: Oswin So, Zachary Serlin, Makai Mann, Jake Gonzales, Kwesi Rutledge, Nicholas Roy, Chuchu Fan
- **Institution**: MIT
- **Date**: October 2023 (ICRA 2024)
- **Venue**: arXiv (2310.15478)
- **URL**: https://arxiv.org/abs/2310.15478
- **Status**: PDF-verified (major update from web-search version)
- **Summary**: Proposes Policy Neural CBF (PNCBF), built on a fundamental theoretical insight connecting value functions to CBFs. The policy value function V^{h,pi}(x_0) = sup_{t>=0} h(x_t^pi) is proven to be a valid CBF (Theorem 1). Since V^{h,pi} >= h and the Lie derivative nabla V^{h,pi}^T (f + g pi) <= 0, it satisfies CBF conditions for any policy pi and any class-K function alpha > 0. A policy iteration scheme is then introduced: using V^{h,pi} as a CBF-QP filter yields a new policy pi_tilde whose forward-invariant set is guaranteed to be a SUPERSET of the original (Theorem 2), ensuring monotonic safety improvement. In practice, the method uses a discounted cost V_lambda^{h,pi} to avoid undesirable trivial solutions, starting with small lambda and decreasing to 0. The full algorithm is: (1) collect rollouts with nominal policy, (2) learn V^{h,pi,theta} via supervised loss, (3) use as CBF-QP safety filter. Only 2-3 iterations of policy iteration are needed for convergence.
- **Experiments**: Double integrator, Segway, F-16 fighter jet (16D state space), two-agent quadcopter (12D state). Hardware validated on two custom drones with a Boston Dynamics Spot as dynamic obstacle. Compared against NCBF, NSCBF, HOCBF, MPC, and SOS baselines.
- **Relevance to Pathway A**: **VERY HIGH** -- The value-function-is-a-CBF insight is fundamental and could replace hand-crafted CBFs in Phase 2 if they prove insufficient. The policy iteration scheme provides a principled way to iteratively improve both safety and performance. Demonstrated on systems of comparable complexity to our PE setup.
- **Key Insight**: The max-over-time cost value function IS a valid CBF. This elegant theoretical result means any learned value function can be repurposed as a safety certificate, bridging RL value learning with control-theoretic safety.

---

### N03. Multi-Agent Constrained Policy Optimisation (MACPO)
- **Authors**: Shangding Gu, Jakub Grudzien Kuba, Munning Wen, Ruiqing Chen, Ziyan Wang, Zheng Tian, Jun Wang, Alois Knoll, Yaodong Yang
- **Institutions**: TU Munich, University of Oxford, Shanghai Jiao Tong University, UCL, Peking University
- **Date**: October 2021 (revised Feb 2022)
- **Venue**: arXiv (2110.02793)
- **URL**: https://arxiv.org/abs/2110.02793
- **GitHub**: https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation.git
- **Status**: PDF-verified (updated with theoretical details)
- **Summary**: Formulates safe MARL as a constrained Markov game <N, S, A, p, rho^0, gamma, R, C, c>. The key theoretical contribution is the Multi-Agent Advantage Decomposition (Lemma 3.1), which decomposes the joint advantage function as a sum of individual agent advantages, enabling decentralized policy updates with global safety guarantees. Two algorithms are proposed: MACPO (trust region optimization with backtracking line search for hard constraint satisfaction) and MAPPO-Lagrangian (Lagrange multiplier approach for soft constraints). Theorem 4.4 provides monotonic improvement guarantees for both reward AND safety constraint satisfaction at every policy iteration.
- **Benchmarks created**: Safe Multi-Agent MuJoCo (SMAMuJoCo) and Safe Multi-Agent Robosuite (SMARobosuite). Both methods achieve zero constraint costs while maintaining reward performance comparable to unconstrained IPPO/MAPPO/HAPPO baselines.
- **Relevance to Pathway A**: **HIGH** -- MACPO provides a principled alternative to the CBF-Beta approach for the two-player PE setting. The constrained Markov game formulation directly maps to the PE problem. The multi-agent advantage decomposition lemma could inform how to handle safety constraints in the adversarial self-play setting.
- **Key Insight**: Trust-region + constraint satisfaction guarantees in multi-agent settings. MAPPO-Lagrangian is simpler to implement and nearly as effective as the full MACPO, making it a practical baseline.

---

### N04. BarrierNet: Differentiable CBFs for Safe Robot Control
- **Authors**: Wei Xiao, Tsun-Hsuan Wang, Ramin Hasani, Makram Chahine, Alexander Amini, Xiao Li, Daniela Rus
- **Institution**: MIT
- **Date**: 2023
- **Venue**: IEEE Transactions on Robotics (T-RO), Vol. 39, No. 3, pp. 2289-2307
- **URL**: https://ieeexplore.ieee.org/document/10077790/
- **GitHub**: https://github.com/Weixy21/BarrierNet
- **Status**: PDF-verified (minor update)
- **Summary**: End-to-end trainable differentiable CBFs (dCBFs) that adapt to changing environments. dCBFs are embedded into differentiable QPs that serve as safety layers within neural network architectures. Unlike static CBFs, dCBFs can adapt their parameters based on the current environment state. Validated on traffic merging, 2D/3D navigation, and autonomous driving tasks.
- **Relevance to Pathway A**: **VERY HIGH** -- BarrierNet's differentiable QP safety layer is directly applicable to the deployment architecture. Unlike static CBFs, dCBFs can adapt to changing obstacle configurations in the PE arena. The end-to-end training could improve upon the separate CBF-Beta training + QP deployment pipeline.
- **Key Insight**: Making the CBF-QP differentiable enables joint training of policy + safety filter, potentially resolving the performance gap between CBF-constrained training and CBF-QP deployment.

---

### N05. GCBF+: Neural Graph CBF for Distributed Safe Multi-Agent Control
- **Authors**: Songyuan Zhang, Oswin So, Kunal Garg, Chuchu Fan
- **Institution**: MIT
- **Date**: January 2024 (revised Feb 2025)
- **Venue**: IEEE Transactions on Robotics (T-RO), arXiv (2401.14554)
- **URL**: https://arxiv.org/abs/2401.14554
- **GitHub**: https://github.com/MIT-REALM/gcbfplus
- **Status**: PDF-verified (updated with formal definitions and architecture details)
- **Summary**: Introduces the Graph CBF (GCBF) framework. Definition 1 defines h: X^M -> R as a GCBF that certifies safety for a multi-agent system of ANY size N (Theorem 1 proves this generalization). The architecture uses a GNN with graph attention for neighbor aggregation, enabling scalable distributed control. GCBF+ improves upon GCBF by using the QP-filtered policy pi_QP (instead of pi_nom) in the control loss, which eliminates the fundamental safety-vs-liveness tradeoff present in prior work. It also considers actuation limits via a look-ahead mechanism. Training minimizes L(theta, phi) = L_CBF(theta, phi) + L_ctrl(phi).
- **Experiments**: 5 environments (SingleIntegrator, DoubleIntegrator, DubinsCar, LinearDrone, CrazyflieDrone). Trained with 8 agents, tested with up to 1024. Outperforms hand-crafted CBF-QP by up to 20% (linear systems) and 30%+ (nonlinear systems). Hardware validated on a Crazyflie drone swarm.
- **Relevance to Pathway A**: **MEDIUM** -- More relevant for multi-agent extensions beyond 2 agents. However, the GNN architecture for processing observations and the learned CBF framework are transferable. The GCBF+ insight of using pi_QP in the control loss is directly applicable to our CBF-Beta training.
- **Key Insight**: Graph-structured safety certificates scale to arbitrary agent counts. Using the QP-filtered policy in the training loss (instead of the nominal policy) eliminates the safety-vs-liveness tradeoff.

---

### N06. Robotic PE in Constrained Game Area with Self-Play + DRL
- **Authors**: Chiraz Ben Jabeur, Hassene Seddik, Khaled Khnissi (Tunisia), Ahmad Hably (Grenoble INP)
- **Date**: April 2025
- **Venue**: Research Square (preprint, rs-6279213/v1)
- **URL**: https://www.researchsquare.com/article/rs-6279213/v1
- **Status**: PDF-verified (major update from web-search version)
- **Summary**: Uses TD3 (Twin Delayed DDPG, not DDPG or PPO) for both pursuer and evader in a 1v1 pursuit-evasion game with simultaneous training from scratch (no freezing, no alternating phases). Unicycle dynamics: x_dot = v cos(theta), y_dot = v sin(theta), theta_dot = omega. The action space is angular velocity omega only (linear velocity is held constant). Full observability with O = (x_p, y_p, theta_p, x_e, y_e, theta_e). Both agents have equal speed and agility.
- **Training configurations**: (1) Normal self-play, (2) Buffer zone (soft safety margin around boundaries), (3) Noisy actions (for robustness).
- **Reward design**: Multi-faceted reward combining step reward (distance gradient multiplied by gain g=1000), outcome reward, duration penalty, and boundary penalty.
- **Results**: Outperforms NMPC baseline. No safety constraints are enforced -- purely task-focused optimization.
- **Relevance to Pathway A**: **HIGH** -- Very close to our proposed work but critically lacks safety guarantees (no CBF). Confirms the viability of simultaneous self-play TD3 for 1v1 PE. The "buffer zone" training variant is similar to a soft safety constraint and could inform curriculum design. Serves as a direct baseline comparison to demonstrate the value of our CBF-based safety layer.
- **Key Insight**: Simultaneous self-play (no freezing) with TD3 works for 1v1 PE -- simpler than the AMS-DRL alternating protocol. Buffer zones act as soft safety margins but provide no formal guarantees.

---

### N07. POLICEd RL: Provable Hard Constraint Satisfaction
- **Authors**: Jean-Baptiste Bouvier, Kartik Nagpal, Negar Mehr
- **Institution**: UC Berkeley, ICON Lab
- **Date**: March 2024 (revised Nov 2024)
- **Venue**: RSS 2024, arXiv (2403.13297)
- **URL**: https://arxiv.org/abs/2403.13297
- **GitHub**: https://github.com/labicon/POLICEd-RL
- **Status**: PDF-verified (major update with theoretical details)
- **Summary**: Forces the learned policy to be AFFINE in a buffer region B = {s in S : Cs in [d-r, d]} near the unsafe boundary Cs <= d. Uses the POLICE algorithm to constrain DNN outputs to be affine within B. The core theoretical result (Theorem 1) states: if the repulsion condition C f(v, mu_theta(v)) <= -2 epsilon holds at all vertices v in V(B), then all trajectories starting in the safe set remain safe. The approximation measure epsilon quantifies how well the nonlinear dynamics can be approximated as affine within B. The method is model-free (no dynamics model needed at training time) and incurs zero computational overhead at deployment since the affine policy in B is stored as a bias term.
- **Experiments**: Inverted pendulum and 7DOF robotic manipulator in MuJoCo.
- **Limitations**: Restricted to relative degree 1 constraints only, affine constraints only (Cs <= d), and deterministic dynamics.
- **Relevance to Pathway A**: **MEDIUM-HIGH** -- Alternative to CBF-Beta for hard constraint enforcement during training. RL-algorithm-agnostic. The affine-near-boundary approach is complementary to CBF truncation. However, the restriction to affine constraints and relative degree 1 limits direct applicability to the PE setting where obstacle constraints may be nonlinear.
- **Key Insight**: Model-free hard constraint satisfaction via policy architecture design rather than CBF. Zero deployment overhead is attractive, but the affine constraint restriction is a significant limitation for general PE environments.

---

### N08. Constrained RL with Statewise Projection via CBF
- **Authors**: Jin, X., Li, K. & Jia, Q.
- **Institution**: Tsinghua University
- **Date**: 2024
- **Venue**: Science China Information Sciences
- **URL**: https://link.springer.com/article/10.1007/s11432-023-3872-9
- **Status**: PDF-verified (updated with algorithmic details)
- **Summary**: Proposes a two-step approach for safe RL: (1) policy optimization within a trust region, (2) CBF-based projection of the resulting policy onto the safe action set. Provides a convergence theorem for the Q-learning variant and a performance bound for the policy optimization variant. The CBF inherently learns a safe policy through a set certificate, projecting unsafe actions to the nearest safe action at each state.
- **Relevance to Pathway A**: **MEDIUM** -- Offers an alternative implementation of CBF-constrained RL (projection-based rather than truncation-based as in the CBF-Beta approach). The convergence theorem and performance bound provide theoretical grounding.
- **Key Insight**: Statewise projection of actions onto the CBF-safe set with formal convergence guarantees. Simpler than sampling-based CBF-Beta but may lose gradient information during the projection step.

---

### N09. Safe RL and Constrained MDPs: Technical Survey (Single-Agent & Multi-Agent)
- **Authors**: Kushwaha et al.
- **Institution**: IIIT Hyderabad
- **Date**: May 2025
- **Venue**: arXiv (2505.17342)
- **URL**: https://arxiv.org/abs/2505.17342
- **Status**: PDF-verified (updated with formulation details and open problems)
- **Summary**: Comprehensive technical survey covering the CMDP framework: M_C = (S, A, P, r, {c^(i)}, gamma) with cost functions and constraint thresholds. Covers constraint types including instantaneous constraints, cumulative cost constraints, probability of failure, and risk measures (CVaR). Details the Lagrangian formulation L(pi, lambda) = J(pi) + sum lambda_i (d_i - J_{c^(i)}(pi)) and the linear programming solution for finite-state CMDPs via occupancy measures. Proposes five open research problems, with three specifically focusing on SafeMARL challenges. Critically, the survey identifies extending safe RL to competitive/adversarial settings as an open challenge.
- **Relevance to Pathway A**: **HIGH** -- Reference for positioning our work in the broader safe RL landscape. The explicit identification of competitive/adversarial safe RL as an open problem directly validates the novelty of our pursuit-evasion work. Covers CPO, PCPO, and other methods we should compare against.
- **Key Insight**: The survey explicitly identifies extending safe RL to competitive/adversarial settings as an open challenge -- this directly validates the novelty and importance of Pathway A.

---

### N10. Sim-to-Real Transfer for Mobile Robots: Isaac Sim to Gazebo to Real ROS2 Robots
- **Authors**: Sahar Salimpour, Jorge Pena-Queralta, Diego Paez-Granados
- **Institutions**: University of Turku, ETH Zurich
- **Date**: January 2025
- **Venue**: arXiv (2501.02902)
- **URL**: https://arxiv.org/abs/2501.02902
- **GitHub**: https://github.com/sahars93/RL-Navigation
- **Status**: PDF-verified (major update with full pipeline details)
- **Summary**: Demonstrates a complete 4-step sim-to-real pipeline: (1) Robot URDF/SDF model creation, (2) Isaac Sim RL training, (3) ONNX export + ROS2 node + Gazebo intermediate testing + Nav2 benchmark comparison, (4) Zero-shot sim-to-real deployment. Uses Isaac Gym (JetbotTask inheriting from BaseTask in omni.isaac.core), NOT Isaac Lab. Training: PPO Actor-Critic with MLP [256, 128, 64], 64 parallel environments, 1500 iterations, 1200 max episode steps. Also tested an LSTM layer (128 hidden units) for dynamic obstacle environments.
- **Observation space**: 2D lidar (120 scans, 6 degree resolution, 0.15-3m range), relative goal (distance d_t, angle theta_t), previous velocities.
- **Action space**: Linear velocity [0.1, 0.5] m/s, angular velocity [-0.5, 0.5] rad/s.
- **Reward**: R_t = r_distance + r_collision (exponential penalty) + r_time + terminal rewards (Goal/Collision/MaxLength).
- **Key training detail**: Curriculum learning is critical for dynamic obstacles (static obstacles first, then dynamic after 300 episodes).
- **Gazebo intermediate testing**: Comparable to Nav2 in static environments; LSTM-RL outperforms Nav2 in dynamic environments.
- **Real-world deployment**: TurtleBot 4 Lite with RPLIDAR A1M8, Raspberry Pi 4B, ROS2 Galactic. Results: 80-100% success rate across 4 experiments, minimum lidar clearance 0.25-0.43m.
- **Relevance to Pathway A**: **VERY HIGH** -- Validates the exact sim-to-real pipeline proposed in Phase 4. Provides concrete implementation details (URDF creation, ONNX export, ROS2 integration, curriculum learning schedule). The observation and action space design directly informs our PE robot setup. The curriculum learning insight (static then dynamic) is critical.
- **Key Insight**: The full Isaac Sim -> Gazebo -> Real pipeline works for zero-shot transfer with 80-100% success. Curriculum learning (static then dynamic obstacles) is essential. Gazebo intermediate testing catches the majority of sim-to-real issues before costly real-world experiments.

---

### N11. Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning
- **Authors**: NVIDIA (multiple contributors)
- **Date**: November 2025
- **Venue**: arXiv (2511.04831)
- **URL**: https://arxiv.org/abs/2511.04831
- **GitHub**: https://github.com/isaac-sim/IsaacLab
- **Status**: PDF-verified (major update with architecture and feature details)
- **Summary**: Isaac Lab is the natural successor to Isaac Gym, open-sourced by NVIDIA. Built on OpenUSD (scene description), PhysX (GPU-accelerated physics), and RTX rendering. Key architectural features:
  - **Two workflow paradigms**: Manager-based (modular, reusable managers for observations, actions, rewards, terminations, curricula, events, commands) and Direct (minimal overhead, tight GPU coupling, ideal for performance-sensitive training).
  - **Custom actuator models**: Implicit PD, Explicit PD, DC Motor, Delayed PD, Remotized PD, and Neural Network actuators.
  - **Three sensor classes**: Physics-based (IMU, contact, frame transformer), rendering-based (Pinhole/Fisheye cameras with RTX), warp-based (RayCaster for LiDAR and heightscan simulation).
  - **Domain randomization**: Geometric + visual randomization across assets and environments.
  - **Terrain generation**: Procedural generation, mesh import, and interactive editing.
  - **Multi-agent support**: Via PettingZoo Parallel API, including support for competitive environments where agents play against each other.
  - **Curriculum support**: Built into the manager workflow.
  - **Environment step function** (Algorithm 2): Pre-physics, physics simulation, post-physics, reset, observation computation phases.
- **Performance**: 1.6M FPS on 8x RTX Pro 6000 GPUs for manipulation tasks.
- **Environment families**: Classic, Locomotion, Manipulation, Navigation.
- **Multi-Agent quote from paper**: "includes and supports the creation of custom environments for solving general physical-based MARL tasks, including competition (agents play against each other)" -- directly relevant to PE.
- **Relevance to Pathway A**: **VERY HIGH** -- Isaac Lab is the recommended training framework for Phase 1-3. The Direct workflow is ideal for our performance-sensitive self-play training. Multi-agent competitive environment support via PettingZoo API directly maps to PE. LiDAR simulation via RayCaster enables realistic sensor observation. Domain randomization for sim-to-real transfer is built in. Curriculum support aligns with our proposed training progression.
- **Key Insight**: Isaac Lab natively supports competitive multi-agent environments, massively parallel GPU training (1.6M FPS), and has all the sensor/actuator/domain randomization tooling needed for the PE sim-to-real pipeline. The Direct workflow provides minimal overhead for tight training loops.

---

### N12. Dynamic Obstacle Avoidance for Car-like Mobile Robots based on Neurodynamic Optimization with CBFs
- **Authors**: Zheng Zhang, Guang-Hong Yang
- **Institutions**: Northeastern University, Shenyang, China (College of Information Science and Engineering; State Key Lab of Synthetical Automation of Process Industries)
- **Date**: August 2025 (received April 2025, accepted August 2025)
- **Venue**: Neurocomputing 654 (2025) 131252 (Elsevier)
- **DOI**: https://doi.org/10.1016/j.neucom.2025.131252
- **Status**: PDF-verified
- **Summary**: Solves the mixed relative degree problem for CBFs on car-like mobile robots (CLMRs) with physical constraints (input saturation + bounded steering angle). Three key contributions:
  1. **Virtual control point CBF**: Instead of using the center point p_c, defines a virtual control point q = p_f + Δz̄ that integrates position, orientation ψ, AND steering angle δ. Here p_f = [x + l cos ψ, y + l sin ψ] is the front axle, z̄ = [cos(ψ+δ), sin(ψ+δ)] is the heading vector, and Δ ≠ 0 is a small scalar offset. The CBF is then h_o = ||q - p_o||² - χ²_o where χ_o ≥ d_o + l/2 + Δ.
  2. **Auxiliary input transformation**: Defines τ = Mu where M is a full-rank matrix (det(M) = Δ cos δ / l ≠ 0), transforming the nonlinear CLMR dynamics into the linear form q̇ = τ. This achieves **uniform relative degree 1** for both linear velocity v and angular velocity w.
  3. **Collision-free module via neurodynamic optimization**: QP formulation: min ||τ̃ - τ||² subject to CBF constraint -(q-p_o)ᵀ τ̃ ≤ (μ₁₀/2)h_o - ||q-p_o||v̄_o, steering CBFs (h_max = δ_max - δ, h_min = δ_max + δ), and input constraints. Solved in real-time by a one-layer recurrent neural network (RNN).
- **CLMR kinematics** (Eq. 5): ẋ = v cos ψ, ẏ = v sin ψ, ψ̇ = (v/l) tan δ, δ̇ = w. State: [p, ψ, δ] = [x, y, ψ, δ]. Input: u = [v, w] (linear velocity and steering angular velocity).
- **Physical constraints** (Eq. 6): ||v|| ≤ v_max, ||w|| ≤ w_max, ||δ|| ≤ δ_max < π/2.
- **Key problem identified**: Traditional position-based CBF h(p_c) = ||p_c - p_o||² - d²_o has mixed relative degree — the Lie derivative L_g h_o involves only v (relative degree 1) but the steering w appears at relative degree 2. This means w cannot directly enforce the CBF constraint. Prior works using HOCBFs are computationally intensive and conservative.
- **Lemma 1**: The designed CBF h_o (Eq. 18) on the virtual control point q has **relative degree 1 uniformly** in both v and w on the state space Ξ.
- **Lemma 2**: Forward invariance of the safe set S¹ = {q ∈ ℝ² | h_o ≥ 0} under the CBF condition: -(q-p_o)ᵀ τ ≤ (μ₁₀/2)h_o - ||q-p_o||v̄_o.
- **Lemma 3**: Forward invariance of steering angle constraints S² ∩ S³ under CBF conditions on h_max and h_min.
- **Theorem 1** (Main result): Under the collision-free module, if χ_o ≥ d_o + l/2 + Δ, then ||p_c - p_o|| ≥ d_o for all time, guaranteeing collision avoidance while respecting ALL physical constraints.
- **Remark 2** (Engineering significance): The virtual control point acts as a "look-ahead" point, decoupling position control from heading control and avoiding the singularity at v=0.
- **Remark 4** (Key advantage over prior work): Previous methods [29-31] using the auxiliary input on actual u prioritize braking (reducing v) over steering. This paper's method prioritizes STEERING via the M matrix transformation, improving obstacle avoidance efficiency.
- **Comparison with degraded scheme** (Eq. 31, Fig. 5): Without the M-matrix transformation, the robot stops (v→0) when encountering obstacles because the CBF condition is primarily governed by v. With the full method, the robot steers around obstacles while maintaining velocity.
- **Simulation**: CLMR with l=0.3m, radius 0.25m, v_max=0.5 m/s, w_max=π/4 rad/s, δ_max=π/4 rad. Successfully avoids 2 static obstacles (Fig. 2) and 3 dynamic obstacles (Fig. 7) with safety distance maintained at all times. RNN solver parameters: ε=0.2, σ=0.01.
- **Relevance to Pathway A**: **CRITICAL** — This paper solves the exact problem identified in our validation report (Section 2.1). Key takeaways for our implementation:
  1. The virtual control point q = p_f + Δz̄ with the M-matrix auxiliary input is the correct approach for nonholonomic robots
  2. The offset parameter Δ should be small but nonzero (they use Δ=0.05m)
  3. The approach handles BOTH linear velocity AND steering angle constraints simultaneously
  4. For our differential-drive robots (unicycle model, which is a simplified CLMR), the same principle applies but is simpler since there's no steering angle state variable — the virtual point is just q = [x + d cos θ, y + d sin θ]
  5. The neurodynamic QP solver enables real-time computation (1 kHz or higher)
- **Key Insight**: The virtual control point + auxiliary input transformation achieves uniform relative degree 1 for ALL control inputs, making the CBF-QP well-posed. The critical innovation is prioritizing STEERING over BRAKING via the M-matrix, which previous methods miss. For our unicycle robots, this simplifies to placing q ahead of the robot at distance d along the heading direction.

---

### N13. Learning Feasibility Constraints for Multi-Agent CBF Controllers (HOCBF Feasibility)
- **Authors**: Wei Xiao, Christos G. Cassandras, Calin A. Belta
- **Institutions**: MIT, Boston University
- **Date**: 2023
- **Venue**: ECC 2023 (European Control Conference)
- **Status**: PDF-verified (major update -- was previously described generically)
- **Summary**: Directly addresses CBF-QP INFEASIBILITY, a critical practical failure mode. Uses High Order CBFs (HOCBFs) for constraints with high relative degree. Identifies two classes of unsafe sets: Regular (feasibility independent of initial condition, e.g., circular obstacles) and Irregular (feasibility depends on initial condition, e.g., rectangular obstacles with corners). The approach learns a feasibility constraint H_j(z) >= 0 using ML classification (SVM or DNN). H_j partitions the state space into feasible (+1) and infeasible (-1) regions. The learned feasibility constraint is ADDED to the QP as an additional HOCBF, effectively steering the system away from states where the original QP would become infeasible.
- **Training procedure**: Iterative feedback training -- sample states, classify feasibility, add learned constraint to QP, resample. After 3 iterations, infeasibility rate drops from 0.0811 to 0.0021 (Table I).
- **Applications**: Robot navigation with irregular obstacles, autonomous driving (vehicle overtaking scenario).
- **Robot dynamics used**: x_dot = v cos(theta), y_dot = v sin(theta), theta_dot = u1, v_dot = u2 -- the same unicycle dynamics as our PE agents.
- **Relevance to Pathway A**: **VERY HIGH** -- Directly addresses the CBF infeasibility risk identified in the plan. More practical than hierarchical relaxation for our setting because it proactively avoids infeasible states rather than reactively relaxing constraints. Uses the exact same unicycle dynamics as our PE robots. The learned feasibility constraint approach should be the primary mitigation strategy for CBF-QP infeasibility.
- **Key Insight**: CBF-QP infeasibility can be systematically eliminated by learning a feasibility constraint and adding it to the QP. The feedback training procedure (sample, classify, add, resample) converges in just 3 iterations with dramatic infeasibility reduction (0.0811 -> 0.0021).

---

### N14. Safety-Biased Trust Region Policy Optimisation (SB-TRPO)
- **Authors**: Ankit Kanwar (Sony), Dominik Wagner, Luke Ong (Nanyang Technological University)
- **Date**: February 2026
- **Venue**: arXiv (2512.23770)
- **URL**: https://arxiv.org/abs/2512.23770
- **Status**: PDF-verified (major update with algorithm details)
- **Summary**: Proposes a Safety-Biased Trust Region approach using a dynamic convex combination of reward and cost trust regions: Delta = (1-mu) Delta_r + mu Delta_c. The safety bias parameter beta in (0,1) controls how aggressively safety is prioritized: epsilon = beta (J_c(pi_old) - c*_{pi_old}). When beta=1, the method recovers CPO; when beta<1, slack is allowed for reward improvement. No separate recovery phase is needed (unlike CPO and C-TRPO).
- **Theoretical results** (Lemma 4.1): (1) Cost is monotonically decreasing, (2) Reward improves when reward and cost gradients are aligned, (3) Convergence to a trust-region local optimum.
- **Key design choice**: Eschews critic networks entirely (Monte Carlo returns only), resulting in approximately 10x less computational cost than critic-based methods.
- **Experiments**: Safety Gymnasium benchmarks -- Point, Car x {Push, Button, Circle, Goal} + Hopper, Swimmer. Safety bias beta=0.75 works well across all tasks.
- **Baselines beaten**: CPO, C-TRPO, TRPO-Lagrangian, PPO-Lagrangian, CUP, FOCOPS, RCPO, PCPO, P3O.
- **Relevance to Pathway A**: **MEDIUM-HIGH** -- Alternative to CBF-Beta for hard-constrained policy optimization. The no-critic design makes it computationally cheap. The dynamic trust region convex combination could be adapted for the self-play setting. Beats a comprehensive set of constrained RL baselines.
- **Key Insight**: Trust-region approach with a tunable safety bias (beta=0.75 is robust) eliminates the need for separate recovery phases while maintaining monotonic cost decrease. The critic-free design offers 10x computational savings.

---

### N15. Safe Robust Multi-Agent RL with Neural CBFs and Safety Attention Mechanism (RMARL-CBF-SAM)
- **Authors**: Shihan Liu, Lijun Liu, Zhen Yu (Xiamen University, China)
- **Date**: October 2024 (received March 2023)
- **Venue**: Information Sciences 690 (2025) 121567 (Elsevier)
- **DOI**: https://doi.org/10.1016/j.ins.2024.121567
- **Status**: PDF-verified
- **Summary**: Proposes RMARL-CBF-SAM — a safe robust MARL method combining three key innovations: (1) H∞-inspired robust MARL that treats modeling errors and external disturbances as adversaries, (2) decentralized robust neural CBFs learned from trajectory data, and (3) a Safety Attention Mechanism (SAM) that weights neighboring agents by danger level.
- **Problem formulation**: Partially observable Markov game ⟨S, {A_i}, P, O, {R_i}, γ⟩ with discrete-time dynamics s^{t+1} = f_i(s^t, u^t) + d^t_i where d^t_i is unknown bounded disturbance. Models errors Δ_i and disturbances d^t_i as adversary v_i.
- **Robust MARL** (Section 3.1): Q-function includes adversary actions: Q*_i(o^t_i, u^t_i, u^t_{-i}, w^t_i, w^t_{-i}) with max-min optimization. Theorem 1: convergence to optimal Q* if 0 < 1 - α + αγ < 1.
- **Neural CBF** (Section 3.2): Decentralized robust neural CBFs with three conditions: (C1) h_i(o^t_i) < 0 ∀s ∈ C_d, (C2) h_i(o^t_i) ≥ 0 ∀s ∈ C_s, (C3) Δh_i + εh_i(o^t_i) ≥ 0 ∀s ∈ C_s. Loss function L_{h,i} = L_{hs,i} + L_{hd,i} + L_{hn,i} covering safe set, dangerous set, and descent condition.
- **Safety reward shaping**: r^t_{s,i} = γh_i(o^{t+1}_i) - h_i(o^t_i) added to task reward. Proposition 1 proves augmented reward doesn't change optimal solution.
- **Safety Attention Mechanism** (Section 3.3): Uses attention weights a_{ij} = softmax(LeakyReLU(e_{ij})) where e_{ij} = f_s(e^T_i W^T_k W_q e_j). Aggregates neighbor info: e_{-i} = Σ_{m∈B_i} a_{im} W_v e_m. The SAM handles variable-size neighborhoods and is shared across Q-function, actor, adversary, and CBF networks.
- **Online fine-tuning** (Sections S.1-S.3): Safety regulator adjusts control and adversary policies post-training with stop criterion based on loss convergence.
- **Algorithm**: RMARL-CBF-SAM (Algorithm 1) extends MADDPG with adversary networks, safe controllers, and neural CBF training interleaved every N_{cbf}=10 steps.
- **Experiments**: Multi-agent navigation (double integrator dynamics: ṗ = v, v̇ = u/m + d). Random and structured environments, 10-50 agents, 10-50 obstacles.
- **Metrics**: Safety rate, risk margin rate (Δd < 0.2), performance reward, navigation steps.
- **Results** (Table 3): RMARL-CBF-SAM achieves best safety rate (0.9992-0.9998) and risk margin rate (0.0261-0.0388) across all scenarios. Maintains >99.9% safety even with 50 agents + 50 obstacles.
- **Baselines compared**: MADDPG-P, MAAC-P, M3DDPG-P, MADDPG-P, MARL-RCPO, MACBF, MARL-BLAC, MARL-Barrier, RMARL-CBF-MAX (ablation without SAM).
- **Robustness** (Fig. 7): Maintains >99.2% safety even with 50% modeling error and max disturbance d_M = 10.
- **SAM vs max-pooling**: SAM outperforms max-pooling (RMARL-CBF-MAX) significantly, especially in risk margin rate, confirming attention > uniform aggregation.
- **Relevance to Pathway A**: **VERY HIGH** — This is the closest paper to our proposed approach. It combines:
  - Robust MARL with adversarial disturbances (similar to our H∞ robustness need)
  - Neural CBFs learned from data (alternative to hand-crafted CBFs)
  - Safety reward shaping (same concept as our proposed w5 term)
  - Attention mechanism for varying agent interactions (could enhance our BiMDN belief encoder)
  - Decentralized execution (relevant for multi-robot deployment)
  However, key differences from our work: (1) Navigation task, NOT pursuit-evasion, (2) No self-play or game-theoretic training, (3) Double integrator dynamics (not unicycle/nonholonomic), (4) Full state-based (no lidar/partial obs), (5) No sim-to-real transfer. Our novelty claim still holds as we address the PE game setting with nonholonomic dynamics and partial observability.
- **Key Insight**: The combination of H∞-robust MARL + neural CBFs + safety attention mechanism achieves >99.9% safety in cluttered multi-agent scenarios. The SAM's danger-weighted attention is more effective than uniform or max-pooling aggregation of neighbor information. The safety reward shaping r_{s,i} = γh(s') - h(s) preserves optimal policy (Proposition 1) while guiding learning toward safe behavior.
