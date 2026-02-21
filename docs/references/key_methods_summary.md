# Key Methods Quick Reference

## 1. CBF Safety Layer Approaches (Training)

### 1a. CBF-Constrained Beta Policy [Paper 16]
```
pi_C(u|x) = pi(u|x) / pi(C(x)|x)   for u in C(x)
           = 0                         otherwise

C(x) = {u : dh_i(x,u) + alpha_i * h_i(x) >= 0, for all i}
```
- **Pros**: Hard safety during training, convergence guarantee
- **Cons**: Requires analytical CBF, Beta distribution only
- **Implementation**: Rescale Beta distribution support to [safe_min, safe_max]

### 1b. BarrierNet / Differentiable CBF-QP [Paper N04]
```
u_safe = argmin_u ||u - u_nn||^2
s.t.  dh(x,u) + alpha * h(x) >= 0   (CBF constraint)

# Differentiable QP: gradients flow through the QP solver
# Enables end-to-end training of policy + safety layer
# Uses dCBFs (discretized CBFs) that adapt to changing environments
# dCBF condition: Delta_h(x_k, u_k) + gamma * h(x_k) >= 0
# where Delta_h = h(x_{k+1}) - h(x_k)
```
- **Pros**: End-to-end trainable, dCBFs adapt to changing environments, works with imitation learning or RL
- **Cons**: QP solver overhead, requires differentiable QP library
- **Implementation**: Use cvxpylayers or qpth for differentiable QP
- **Code**: https://github.com/Weixy21/BarrierNet

### 1c. POLICEd RL [Paper N07]
```
# Force policy to be affine near unsafe set boundary:
# Unsafe region: Cs <= d (affine constraints on state)
# Buffer zone B = {s in S : Cs in [d-r, d]} near unsafe boundary
#   where r > 0 is the buffer width

# Policy is piecewise affine (CPWL) in the buffer:
# pi(x) = A * x + b    for x in buffer B

# Theorem 1 (Safety Guarantee):
# If repulsion condition Cf(v, mu_theta(v)) <= -2*epsilon
# holds at all vertices v in V(B) of the buffer polytope,
# then the policy is safe (trajectories cannot enter unsafe set)

# Key insight: checking vertices of buffer polytope is sufficient
# because policy is affine (linear interpolation property)
```
- **Pros**: Model-free (no dynamics needed), works with any RL algo (PPO, TD3, SAC), hard safety guarantees via vertex checking
- **Cons**: Relative degree 1 constraints only, affine constraints only, deterministic dynamics required, may reduce expressiveness near boundaries
- **Implementation**: Policy architecture constraint (not post-hoc filter); CPWL network enforced via MaxAffine layers
- **Institution**: UC Berkeley ICON Lab

### 1d. MACPO / MAPPO-Lagrangian [Paper N03]
```
# Multi-Agent Constrained Policy Optimisation (MACPO)
# Key: Multi-Agent Advantage Decomposition (Lemma 3.1)
#   Decomposes joint advantage into per-agent advantages

# MACPO uses second-order approximation + primal-dual optimization:
# For each agent i:
#   max_{pi_i} A_i(pi_i)                   (reward advantage)
#   s.t. KL(pi_i || pi_i_old) <= delta      (trust region)
#        D_c(pi_i) <= d - J_c(pi_old)       (safety margin)
# where D_c uses second-order Taylor expansion of cost advantage

# Theorem 4.4: Guarantees monotonic improvement in BOTH:
#   - Cumulative reward (non-decreasing)
#   - Safety constraint satisfaction (non-increasing cost)

# MAPPO-Lagrangian (simpler variant):
# max_{pi} J(pi) - lambda * (C(pi) - d)
# Lambda updated via dual gradient ascent
```
- **Pros**: Multi-agent native, theoretical guarantees (monotonic reward AND safety improvement), handles coupled safety constraints
- **Cons**: Second-order approximation adds computational cost, Lagrangian variant has softer constraints
- **Benchmarks**: Created SMAMuJoCo and SMARobosuite benchmarks for multi-agent safe RL
- **Code**: https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation.git

### 1e. Policy Neural CBF (PNCBF) [Paper N02]
```
# Key insight: The worst-case value function IS a valid CBF
# V^{h,pi}(x0) = sup_{t>=0} h(x_t^pi)
# where h(x) is the constraint function (h(x) <= 0 means safe)

# Why it works (Theorem 1):
#   1. V^{h,pi} >= h          (by definition, sup >= any element)
#   2. nabla V^{h,pi}^T (f + g*pi) <= 0  (non-increasing along trajectories)
#   => V^{h,pi} satisfies CBF conditions

# Policy Iteration Algorithm (Theorem 2 - enlarges safe set):
# 1. Collect rollouts with nominal policy pi
# 2. Learn V^{h,pi,theta} via regression loss:
#    L = ||V_theta(x) - max{max_t h(x_t), V_theta(x_T)}||^2
#    (bootstrapped target for long horizons)
# 3. Use V^{h,pi,theta} as CBF in QP safety filter:
#    u* = argmin ||u - pi(x)||^2
#    s.t. dV/dx * (f + g*u) + alpha * V(x) <= 0
# 4. Repeat with filtered policy (converges in 2-3 iterations)

# Scales to F-16 aircraft (16D state), hardware-validated on quadcopters
```
- **Pros**: Learns valid CBFs from any nominal policy; no hand-crafting needed; scales to high dimensions (16D); policy iteration provably enlarges forward-invariant set
- **Cons**: Requires nominal policy; neural network approximation may have errors near boundaries; relies on rollout data quality
- **Implementation**: PyTorch value network + OSQP for CBF-QP solving

### 1f. Safety-Biased Trust Region Policy Optimization (SB-TRPO) [Paper N14]
```
# Dynamic convex combination of reward and cost gradients:
# Delta = (1 - mu) * Delta_r + mu * Delta_c
# where Delta_r = reward improvement direction
#       Delta_c = cost reduction direction

# mu determined by safety bias parameter beta:
#   epsilon = beta * (J_c(pi_old) - c*_{pi_old})
#   where c*_{pi_old} is the cost limit
#   mu adapts: when cost is high, mu -> 1 (prioritize safety)
#              when cost is low,  mu -> 0 (prioritize reward)

# beta controls aggressiveness of cost reduction:
#   beta = 1   recovers CPO behavior
#   beta = 0.75 recommended (good balance)
#   beta = 0   ignores cost entirely

# No separate recovery phase needed (unlike CPO)
# No critic networks needed (Monte Carlo returns only)
# Trust region via conjugate gradient + line search (like TRPO)
```
- **Pros**: Hard constraints via adaptive blending, ~10x cheaper than CPO (no critics, no recovery phase), monotonic cost decrease guaranteed, simple to implement on top of TRPO
- **Cons**: Monte Carlo only (high variance in estimates), model-free (no CBF integration possible), requires on-policy data

## 2. CBF Safety Layer Approaches (Deployment)

### 2a. RCBF-QP + GP [Paper 06]
```
u* = argmin_u ||u - u_RL||^2
s.t. L_f h + L_g h * u + L_d h * d_hat >= -alpha * h(x) + kappa * sigma_d
     u in U

# GP disturbance model:
d_hat(x), sigma_d(x) = GP.predict(x)
# Residual: d_t = x_{t+1} - f(x_t) - g(x_t) * u_t
```
- **Pros**: Handles model uncertainty, robust safety
- **Cons**: GP computational cost, needs residual data
- **Implementation**: GPyTorch with SE-ARD kernel, online updates

### 2b. Virtual Control Point CBF for Nonholonomic Robots [Paper N12]
```
# PROBLEM: Standard position-based CBF h(p_c) = ||p_c - p_o||^2 - d_o^2
# has MIXED relative degree for nonholonomic robots:
#   - Relative degree 1 in linear velocity v
#   - Relative degree 2 in steering/angular velocity w
# => w cannot directly enforce CBF constraint => QP may be infeasible

# SOLUTION (Zhang & Yang, Neurocomputing 2025):
# For car-like robot (CLMR): state = [x, y, ψ, δ], input u = [v, w]
# Kinematics: ẋ = v cos ψ, ẏ = v sin ψ, ψ̇ = (v/l) tan δ, δ̇ = w

# Step 1: Virtual control point ahead of robot
p_f = [x + l*cos(ψ), y + l*sin(ψ)]       # front axle
z̄ = [cos(ψ+δ), sin(ψ+δ)]                 # heading vector
q = p_f + Δ*z̄                              # virtual point (Δ ≈ 0.05m)

# Step 2: Auxiliary input transformation via M matrix
# q̇ = M * u  where det(M) = Δ cos(δ) / l ≠ 0
# Define auxiliary input: τ = M * u  =>  q̇ = τ  (LINEAR!)
# Actual input recovered: u = M^{-1} * τ

# Step 3: CBF on virtual point (NOW has uniform relative degree 1)
h_o = ||q - p_o||^2 - χ_o^2     # where χ_o ≥ d_o + l/2 + Δ

# Step 4: Collision-free QP module (solved by neurodynamic RNN)
# τ* = argmin ||τ̃ - τ_nom||^2
# s.t. -(q-p_o)^T τ̃ ≤ (μ₁₀/2)h_o - ||q-p_o||v̄_o   (obstacle CBF)
#      η^T τ̃ ≤ μ₂ h_max                               (steering upper)
#      -η^T τ̃ ≤ μ₃ h_min                              (steering lower)
#      -u_max ≤ M^{-1} τ̃ ≤ u_max                      (input saturation)
# Then: u* = M^{-1} τ*

# FOR UNICYCLE (simplified CLMR, no steering state δ):
# Virtual point: q = [x + d*cos(θ), y + d*sin(θ)]
# q̇ depends on BOTH v and ω:
#   q̇ = [v cos(θ) - d*ω sin(θ),  v sin(θ) + d*ω cos(θ)]
# => Relative degree 1 in both v and ω ✓

# Now dh is directly controlled by [v, omega]
```
- **Pros**: Achieves uniform relative degree 1 for ALL control inputs; prioritizes STEERING over BRAKING (key insight from N12); handles input saturation + bounded steering; real-time via neurodynamic RNN solver (1 kHz+); proven collision avoidance (Theorem 1)
- **Cons**: Offset parameter Δ must be nonzero but small (Δ=0.05m works well); safety margin increases by l/2 + Δ; requires obstacle velocity upper bound v̄_o; paper validates on CLMR only (not unicycle directly, but the simplification is straightforward)
- **Implementation**: For unicycle: q = [x + d cos θ, y + d sin θ], then CBF h = ||q - p_o||² - χ². For CLMR: full M-matrix transformation. QP solved by OSQP or neurodynamic RNN.

### 2c. Learned Feasibility Constraints for CBF-QP [Paper N13]
```
# Problem: CBF-QP can become infeasible when tight state + control constraints
# conflict (especially with high-order CBFs / HOCBFs)
# Two unsafe set classes studied:
#   - Regular obstacles (circular) -- typically feasible
#   - Irregular obstacles (rectangular corners) -- often cause infeasibility

# Solution: Learn a feasibility constraint H_j(z) >= 0 via SVM or DNN:
# 1. Sample states uniformly, solve QP at each state
#    Classify as feasible (+1) or infeasible (-1)
# 2. Train binary classifier H_j(z) (SVM with RBF kernel or small DNN)
# 3. Add H_j(z) >= 0 as additional HOCBF constraint to the QP
#    (keeps the system away from states where QP would fail)
# 4. Feedback loop: resample near decision boundary, retrain
#    (3 iterations of refinement typically sufficient)

# Result: infeasibility rate reduced from 0.0811 to 0.0021

# Robot dynamics (unicycle model):
# x_dot = v * cos(theta)
# y_dot = v * sin(theta)
# theta_dot = u1    (angular velocity)
# v_dot = u2        (linear acceleration)
```
- **Pros**: Dramatically reduces QP infeasibility, works with any HOCBF setup, iterative refinement improves boundary accuracy
- **Cons**: Requires offline sampling phase, learned constraint is approximate (may miss edge cases), adds complexity to QP
- **Implementation**: SVM (scikit-learn) or small DNN for feasibility classifier; integrate as additional HOCBF in existing QP formulation

### 2d. RMARL-CBF-SAM: Robust MARL + Neural CBF + Safety Attention [Paper N15]
```
# H∞-inspired Robust MARL: treat modeling errors + disturbances as adversary
# Dynamics: s^{t+1} = f_hat(s^t, u^t) + Δ^t + d^t   (Δ = model error, d = disturbance)
# Adversary: v_i = {Δ_i + d_i} tries to maximize damage
# Q-function: Q*_i(o, u, u_{-i}, w, w_{-i}) with max_u min_w optimization

# Decentralized Robust Neural CBFs:
# (C1) h_i(o) < 0  for dangerous states
# (C2) h_i(o) >= 0  for safe states
# (C3) Δh_i(o, u_s, u_{s,-i}, w_s, w_{s,-i}) + ε*h_i(o) >= 0  (descent condition)
# Loss: L_{h,i} = L_{hs,i} + L_{hd,i} + L_{hn,i}

# Safety Attention Mechanism (SAM):
# e_i = E_s(s_i)                           # state encoding
# e_{ij} = f_s(e_i^T W_k^T W_q e_j)       # pairwise similarity
# a_{ij} = softmax(LeakyReLU(e_{ij}))      # attention weights (danger-weighted)
# e_{-i} = Σ_{m∈B_i} a_{im} W_v e_m       # aggregated neighbor info

# Safety reward shaping (Proposition 1: preserves optimal policy):
# r_{s,i} = γ*h_i(o^{t+1}) - h_i(o^t)
# r_{augmented} = r_task + r_safety
```
- **Pros**: Robust to model errors + disturbances, decentralized, scales to 50+ agents, >99.9% safety, SAM handles variable neighborhoods
- **Cons**: Double integrator dynamics only (not nonholonomic), no game-theoretic PE, no sim-to-real
- **Implementation**: Extends MADDPG with adversary + safe controller networks; CBF trained every 10 RL steps; Adam optimizer + RMSprop for fine-tuning

## 3. Self-Play Protocols

### 3a. AMS-DRL [Paper 18]
```
S0: Cold-start evader (500-1000 episodes)
S1: Train pursuer vs frozen evader (1000-2000 episodes)
S2: Train evader vs frozen pursuer (1000-2000 episodes)
Sk: Alternate until |SR_P - SR_E| < eta (eta ~ 0.10)
```

### 3b. Simultaneous TD3 Self-Play [Paper N06]
```
# Both agents trained simultaneously (no freezing)
# Action space: angular velocity omega only (linear velocity is constant)
# Observation: O = (x_p, y_p, theta_p, x_e, y_e, theta_e)
#   Full state of both pursuer and evader

# Multi-faceted reward structure:
#   R_step = distance_gradient * g    (g = 1000 scaling factor)
#     distance_gradient = (d_{t-1} - d_t) for pursuer (reward closing)
#     distance_gradient = (d_t - d_{t-1}) for evader (reward opening)
#   R_outcome = +/- large bonus for capture/escape
#   R_duration = penalty for long episodes
#   R_boundary = penalty for leaving arena

# Three self-play variants:
1. Standard self-play
2. Buffer zone self-play (soft safety margin around arena)
3. Noisy action self-play (exploration noise on opponent)

# Result: Outperforms NMPC (nonlinear MPC) baseline
# Agents develop emergent pursuit-evasion strategies
```

## 4. Neural Network Architecture

### From Paper [02] (Gonultas):
```
Lidar --> Conv1D(N,32) --> Conv1D(32,64) --> Flatten
State --> MLP(dim,64,64)
Belief --> BiMDN(LSTM/GRU + MDN) --> z(32-dim)
Concat --> MLP(160+32, 256, 256)
  |--> Actor: Beta(alpha, beta) per action dim
  |--> Critic: V(o_t)
```

### From Paper N04 (BarrierNet):
```
Observation --> state_net (CNN/MLP) --> predicted state
Predicted state --> barrier_net --> differentiable QP --> safe action
```

## 5. Sim-to-Real Pipeline [Papers 07, N10, N11]
```
1. Isaac Lab (GPU-accelerated training)  [Paper N11]
   - Natural successor to Isaac Gym, open-source
   - Code: github.com/isaac-sim/IsaacLab
   - Core architecture: OpenUSD scenes + PhysX GPU physics + RTX rendering
   - Two workflow styles:
     * Manager-based: modular, configurable via task/reward/observation managers
     * Direct: maximum performance, manual environment stepping
   - Multi-agent support via PettingZoo Parallel API
   - Built-in domain randomization:
     * Geometric properties: mass, friction, joint stiffness
     * Visual properties: textures, lighting (via RTX)
   - Sensor suite: RayCaster (LiDAR), cameras (Pinhole/Fisheye), IMU, contact sensors
   - Custom actuator models: Implicit PD, DC Motor, Delayed PD, Neural Net actuators
   - Performance: up to 1.6M FPS on multi-GPU setups

2. ONNX Export  [Paper N10]
   - PyTorch --> ONNX conversion for deployment
   - ROS2 node: subscribes to LiDAR + Odometry topics, publishes cmd_vel
   - Inference latency: ~1.6-5ms on Raspberry Pi 4B

3. Gazebo/ROS2 (intermediate testing)  [Paper N10]
   - LSTM-RL agent outperforms Nav2 stack in dynamic environments
   - Comparable to Nav2 in static environments
   - Uses same ROS2 inference node as real robot deployment
   - Catches sim-to-real issues before hardware testing

4. Real Robot  [Paper N10]
   - Hardware: TurtleBot 4 Lite + RPLIDAR A1M8 + Raspberry Pi 4B
   - Software: ROS2 Galactic
   - Performance: 80-100% navigation success rate
   - Minimum safety distance maintained: 0.25m
   - Curriculum learning critical: static obstacles first, then dynamic after ~300 episodes
   - RCBF-QP safety filter (C++ implementation for real-time)
   - GP disturbance learning (online updates)
   - 20Hz control loop
```
