# Phase 2.5: BarrierNet Experiment (Month 4)

## Safe Deep RL for 1v1 Ground Robot Pursuit-Evasion

**Phase**: 2.5 of 5
**Timeline**: Month 4 (after Phase 2: Safety Integration)
**Duration**: 3-4 weeks (7 sessions)
**Status**: Planning
**Last Updated**: 2026-02-21
**Parent Document**: `claudedocs/pathway_A_safe_deep_RL_1v1_PE.md`

---

## Table of Contents

1. [Phase Overview](#1-phase-overview)
2. [Background & Theoretical Foundations](#2-background--theoretical-foundations)
3. [Relevant Literature](#3-relevant-literature)
4. [Session-wise Implementation Breakdown](#4-session-wise-implementation-breakdown)
5. [Testing Plan (Automated)](#5-testing-plan-automated)
6. [Manual Validation Checklist](#6-manual-validation-checklist)
7. [Success Criteria & Phase Gates](#7-success-criteria--phase-gates)
8. [Troubleshooting Guide](#8-troubleshooting-guide)
9. [Guide to Next Phase](#9-guide-to-next-phase)

---

## 1. Phase Overview

### 1.1 Title

**BarrierNet Experiment: Differentiable QP Safety Layer vs CBF-Beta for Pursuit-Evasion**

### 1.2 Timeline

- **Start**: After Phase 2 completion (all Phase 2 artifacts validated)
- **End**: Decision document produced, architecture chosen for Phase 3
- **Estimated duration**: 3-4 weeks, 7 implementation sessions

### 1.3 Objectives

1. Implement a differentiable QP safety layer (BarrierNet architecture) using cvxpylayers or qpth for VCP-CBF constraints on unicycle dynamics
2. Integrate the differentiable QP into the PPO actor network so that the policy outputs a nominal action that passes through the QP layer to produce a safe action
3. Train a BarrierNet-augmented policy end-to-end on the PE environment from Phase 2
4. Quantitatively compare the CBF-Beta (training) to RCBF-QP (deployment) pipeline against the BarrierNet end-to-end pipeline
5. Produce a decision document with a clear, data-backed recommendation for which safety architecture to use in Phase 3 and beyond

### 1.4 Prerequisites (Phase 2 Artifacts)

The following must exist and be validated before starting Phase 2.5:

| Artifact | Description | Location (expected) |
|----------|-------------|---------------------|
| PE Gymnasium environment | 1v1 pursuit-evasion env with unicycle dynamics, obstacles, arena boundary | `src/envs/pursuit_evasion_env.py` |
| VCP-CBF implementation | Virtual Control Point CBF for arena boundary + obstacles (Section 3.3.1 of pathway doc) | `src/safety/vcp_cbf.py` |
| CBF-Beta policy | PPO with Beta distribution, truncated sampling to safe set C(x) | `src/agents/cbf_beta_ppo.py` |
| RCBF-QP safety filter | Deployment-time QP safety filter with robust margins | `src/safety/rcbf_qp.py` |
| Trained CBF-Beta policy checkpoint | Converged pursuer and evader policies from Phase 2 | `checkpoints/phase2/` |
| Phase 2 evaluation metrics | Safety violation rate, capture rate, CBF margin distribution, training curves | `results/phase2/` |
| 3-tier infeasibility handler | Learned feasibility (N13) + hierarchical relaxation + backup controller | `src/safety/infeasibility_handler.py` |
| Safety-reward shaping | CBF margin bonus (w5 = 0.05) integrated in reward function | Part of PE env |

### 1.5 Why This Phase Exists: The Training-Deployment Safety Gap Problem

**The core problem**: Phases 1-2 use two fundamentally different safety mechanisms at two different stages:

1. **Training (CBF-Beta, Paper [16])**: The policy's action distribution (Beta) is *truncated* so that it only samples from the safe control set C(x). The policy *never sees* unsafe actions during training. Safety is enforced by restricting the distribution's support.

2. **Deployment (RCBF-QP, Paper [06])**: A quadratic program *filters* the policy's output action, projecting it to the nearest safe action if it violates CBF constraints. The policy outputs whatever it wants; the QP corrects it.

**Why this is a problem**: These are fundamentally different mechanisms:

- CBF-Beta changes *what actions the policy can explore* during training. The policy is optimized knowing it can only sample safe actions. It learns within a constrained action space.
- RCBF-QP changes *the action after the policy decides*. The policy was never trained with this post-hoc correction. It may have learned behaviors that rely on the truncated sampling boundaries (e.g., "pushing against" the safe set boundary), which behave differently under QP projection.

**Concrete failure modes**:
- The policy may learn to use the CBF-Beta truncation boundary as a "wall" to lean against. Under RCBF-QP, the QP projection may push the action in a different direction than the truncation would have.
- The policy's value function was trained under CBF-Beta dynamics. Under RCBF-QP, the actual state transitions differ, causing value estimation errors.
- The exploration distribution during training (truncated Beta) does not match the effective distribution during deployment (Beta + QP projection), violating the on-policy assumption of PPO.

**The BarrierNet solution**: BarrierNet [N04] embeds the deployment QP *directly inside* the neural network as a differentiable layer. During training, the policy learns with the *exact same* safety mechanism it will use at deployment. Gradients flow through the QP via implicit differentiation of the KKT conditions. There is zero training-deployment gap by construction.

**This phase answers one question**: Is the training-deployment gap between CBF-Beta and RCBF-QP significant enough to justify the additional complexity and computational cost of BarrierNet?

---

## 2. Background & Theoretical Foundations

### 2.1 The Training-Deployment Gap Problem in Detail

#### 2.1.1 CBF-Beta: Truncated Sampling (Training)

From Paper [16] (Suttle et al., AISTATS 2024), the CBF-constrained Beta policy is:

```
pi_theta^C(u|x) = pi_theta(u|x) / pi_theta(C(x)|x)    for u in C(x)
                 = 0                                      otherwise

where C(x) = {u : dh_i(x,u) + alpha_i * h_i(x) >= 0, for all i}
```

**How it works for unicycle with VCP-CBF**:

The safe control set C(x) is defined by the VCP-CBF constraints (from N12). For each CBF h_i, the constraint `dh_i(x,u) + alpha_i * h_i(x) >= 0` is affine in `u = [v, omega]` (thanks to the virtual control point achieving uniform relative degree 1). This defines a polytope in (v, omega) space.

The Beta distribution's support `[v_min, v_max] x [omega_min, omega_max]` is intersected with this polytope. In practice, for each action dimension, the safe bounds are computed:
```python
# For each state x:
v_safe_min, v_safe_max = compute_cbf_bounds_v(x, h_list, alpha_list)
omega_safe_min, omega_safe_max = compute_cbf_bounds_omega(x, h_list, alpha_list)

# Beta distribution rescaled to safe interval:
v ~ Beta(alpha_v, beta_v) * (v_safe_max - v_safe_min) + v_safe_min
omega ~ Beta(alpha_omega, beta_omega) * (omega_safe_max - omega_safe_min) + omega_safe_min
```

**Key property**: The policy gradient is computed with respect to the *truncated* distribution. The normalizing constant `pi_theta(C(x)|x)` appears in the gradient, and Theorem 1 of [16] guarantees convergence to stationary points of the constrained objective.

**Limitation**: The safe bounds are computed per-dimension (axis-aligned box approximation of the polytope C(x)). This is conservative -- the actual safe set is a polytope, but we approximate it as a box. The policy may learn to exploit the box boundaries, which do not correspond to the actual polytope boundaries.

#### 2.1.2 RCBF-QP: Optimization-Based Filtering (Deployment)

From Paper [06] (Emam et al., 2022), the deployment safety filter is:

```
u* = argmin_{u} ||u - u_RL||^2
s.t.  L_f h_i + L_g h_i * u >= -alpha_i * h_i(x) + kappa * sigma_d(x)   for all i
      u in U   (control bounds)
```

**How it works**: The policy outputs `u_RL` without any safety consideration. The QP finds the closest safe action `u*` to `u_RL`. If `u_RL` is already safe, then `u* = u_RL` (the QP is a no-op). If `u_RL` violates any CBF constraint, the QP projects it onto the constraint boundary.

**Key difference from CBF-Beta**:
- The QP projection is onto the *full polytope* C(x), not an axis-aligned box approximation
- The QP can project in any direction, while CBF-Beta only clips per-dimension
- The robust margin `kappa * sigma_d(x)` from the GP disturbance model further changes the effective safe set
- The projection is *not* the same as truncated sampling -- truncation changes the probability density, projection changes a single point

#### 2.1.3 Quantifying the Gap

The training-deployment gap manifests in multiple measurable ways:

| Metric | How CBF-Beta Differs from RCBF-QP |
|--------|-----------------------------------|
| Action distribution | Truncated Beta vs. QP-projected deterministic |
| Safe set shape | Axis-aligned box vs. full polytope |
| Gradient signal | Through truncation normalization vs. none (QP is non-differentiable) |
| Robustness margin | None (nominal CBF) vs. GP-based robust margin |
| Infeasibility handling | Graceful (distribution narrows) vs. hard (QP infeasible) |

**Expected gap magnitude**: Without measurement, we hypothesize 5-15% performance degradation when switching from CBF-Beta (training) to RCBF-QP (deployment), based on:
- The axis-aligned box approximation of C(x) is conservative, so the policy was trained in a smaller action space than RCBF-QP allows
- The robust margin in RCBF-QP further shrinks the safe set
- The QP projection direction may differ from the truncation direction, causing unexpected state transitions

### 2.2 BarrierNet: Differentiable QP Safety Layer [N04]

#### 2.2.1 Core Architecture

BarrierNet (Xiao et al., T-RO 2023, MIT) embeds a differentiable QP layer inside the neural network:

```
Input: state x
  |
  v
Policy Network: MLP(x) --> nominal action u_nom (unconstrained)
  |
  v
Differentiable QP Layer:
  u* = argmin_{u} ||u - u_nom||^2
  s.t. dh_i(x,u) + alpha_i * h_i(x) >= 0   for all i   (VCP-CBF constraints)
       u_min <= u <= u_max                                (control bounds)
  |
  v
Output: safe action u*
```

**The key innovation**: The QP layer is differentiable. Gradients of the loss with respect to the network parameters theta flow *through* the QP solution via implicit differentiation of the KKT conditions:

```
Forward pass:  u* = QP(u_nom(theta), x)
Backward pass: d(Loss)/d(theta) = d(Loss)/d(u*) * d(u*)/d(u_nom) * d(u_nom)/d(theta)
                                                    ^^^^^^^^^^^^^^^^^
                                                    This is the key: gradient through QP
```

#### 2.2.2 Differentiable QP: How Gradients Flow Through KKT Conditions

The QP has the form:
```
minimize    (1/2) u^T Q u + p^T u
subject to  G u <= h_vec
            A u = b
```

For our VCP-CBF problem:
- `Q = I` (identity, minimizing ||u - u_nom||^2)
- `p = -u_nom`
- `G u <= h_vec` encodes the CBF constraints (negated because CBF gives >= 0) and control bounds
- No equality constraints

The KKT conditions for the optimal solution (u*, lambda*, nu*) are:
```
Q u* + p + G^T lambda* + A^T nu* = 0        (stationarity)
G u* - h_vec <= 0                             (primal feasibility)
lambda* >= 0                                  (dual feasibility)
lambda_i * (G_i u* - h_i) = 0                (complementary slackness)
```

**Implicit differentiation**: Differentiating the KKT conditions with respect to any parameter (e.g., u_nom, which affects p = -u_nom) yields a system of linear equations that can be solved for the Jacobian d(u*)/d(u_nom).

Specifically, let the active constraints be indexed by A = {i : lambda_i > 0}. Then:
```
d(u*)/d(u_nom) = (Q + G_A^T diag(lambda_A / s_A) G_A)^{-1}
```
where `s_A = h_A - G_A u*` are the slacks of the active constraints. This is computed efficiently by the differentiable QP solver (cvxpylayers or qpth).

#### 2.2.3 Discretized CBFs (dCBFs) in BarrierNet

BarrierNet uses discretized CBFs (dCBFs) rather than continuous-time CBFs, since training operates in discrete time:

```
Continuous CBF condition: dh/dt(x, u) + alpha * h(x) >= 0

Discretized CBF condition: h(x_{k+1}) - h(x_k) + gamma * h(x_k) >= 0
                          = Delta_h(x_k, u_k) + gamma * h(x_k) >= 0

where:
  x_{k+1} = f_discrete(x_k, u_k)   (one-step dynamics)
  gamma in (0, 1]                    (discrete decay rate, analogous to alpha)
```

For the unicycle with VCP-CBF at virtual point q_k = [x_k + d*cos(theta_k), y_k + d*sin(theta_k)]:

```
q_{k+1} = q_k + dt * q_dot_k
         = q_k + dt * [v_k*cos(theta_k) - d*omega_k*sin(theta_k),
                        v_k*sin(theta_k) + d*omega_k*cos(theta_k)]

h_obs(x_{k+1}) = ||q_{k+1} - p_obs||^2 - chi^2

dCBF constraint:
  ||q_{k+1} - p_obs||^2 - chi^2 - ||q_k - p_obs||^2 + chi^2 + gamma * (||q_k - p_obs||^2 - chi^2) >= 0

Simplifying:
  ||q_{k+1} - p_obs||^2 - (1 - gamma) * ||q_k - p_obs||^2 - gamma * chi^2 >= 0
```

This is still affine in u_k = [v_k, omega_k] (since q_{k+1} is affine in u_k), so it fits the QP framework.

#### 2.2.4 Adaptive dCBFs

A key feature of BarrierNet is that the CBF parameters can be made state-dependent (adaptive):

```
gamma_i(x) = nn_gamma(x)   (a small network outputs the decay rate)
alpha_i(x) = nn_alpha(x)   (or the class-K function parameter)
```

This allows the safety layer to be less conservative when far from obstacles and more aggressive when close. The network `nn_gamma` is trained end-to-end alongside the policy. For our implementation, we start with fixed gamma and add adaptive gamma as an enhancement if needed.

### 2.3 Differentiable Optimization Layers: Software Options

Three libraries implement differentiable QP solving:

#### 2.3.1 cvxpylayers (Recommended)

- **Repository**: https://github.com/cvxgrp/cvxpylayers
- **Paper**: Agrawal et al., NeurIPS 2019
- **Interface**: Define QP in cvxpy, convert to differentiable PyTorch layer
- **Pros**: Natural cvxpy syntax, supports many problem types (LP, QP, SOCP, SDP), well-maintained
- **Cons**: Overhead from cvxpy compilation, may be slower than specialized QP solvers for small problems
- **Installation**: `pip install cvxpylayers`

```python
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# Define QP parameters (will be filled at runtime)
u = cp.Variable(2)           # [v, omega]
u_nom = cp.Parameter(2)      # nominal action from policy
G = cp.Parameter((n_constraints, 2))   # CBF constraint matrix
h_vec = cp.Parameter(n_constraints)     # CBF constraint vector

# QP: min ||u - u_nom||^2  s.t.  G @ u <= h_vec
objective = cp.Minimize(cp.sum_squares(u - u_nom))
constraints = [G @ u <= h_vec]
problem = cp.Problem(objective, constraints)

# Convert to differentiable layer
qp_layer = CvxpyLayer(problem, parameters=[u_nom, G, h_vec], variables=[u])

# Forward pass (in training loop):
u_safe, = qp_layer(u_nom_tensor, G_tensor, h_vec_tensor)
# u_safe is differentiable w.r.t. u_nom_tensor, G_tensor, h_vec_tensor
```

#### 2.3.2 qpth

- **Repository**: https://github.com/locuslab/qpth
- **Paper**: Amos & Kolter, ICML 2017 (OptNet)
- **Interface**: Direct QP interface (Q, p, G, h, A, b)
- **Pros**: Fast for small QPs, GPU-native, batch support
- **Cons**: Less flexible than cvxpylayers (QP only), can have numerical issues
- **Installation**: `pip install qpth`

```python
from qpth.qp import QPFunction

# QP: min (1/2) u^T Q u + p^T u  s.t.  G u <= h,  A u = b
# For our problem: Q = I, p = -u_nom
Q = torch.eye(2).unsqueeze(0).expand(batch_size, -1, -1)  # (B, 2, 2)
p = -u_nom  # (B, 2)

u_safe = QPFunction(verbose=False)(Q, p, G, h_vec, A_eq, b_eq)
# u_safe is differentiable w.r.t. Q, p, G, h_vec
```

#### 2.3.3 OptNet (Legacy)

- **Paper**: Amos & Kolter, ICML 2017
- **Note**: qpth is the maintained successor. Use qpth instead.

#### 2.3.4 Recommendation

**Use cvxpylayers for prototyping** (cleaner interface, easier debugging) and **switch to qpth if performance is insufficient** (faster for small batch QPs). Both support GPU and batched operations.

### 2.4 GCBF+ [N05]: Background Context

GCBF+ (Zhang et al., T-RO 2024, MIT) is a graph-based CBF framework for multi-agent systems. While not directly used in Phase 2.5 (we have only 2 agents), two insights from GCBF+ are relevant:

1. **QP-filtered policy in training loss**: GCBF+ trains with `L_ctrl(phi) = ||pi_QP(x) - pi_nom(x)||^2` where `pi_QP` is the QP-filtered policy. This is conceptually similar to BarrierNet -- the training signal includes the effect of the safety filter.

2. **Look-ahead for actuation limits**: GCBF+ uses a look-ahead mechanism to ensure the QP remains feasible given actuation constraints. This is relevant for our VCP-CBF QP, which must respect `v in [0, v_max]` and `omega in [-omega_max, omega_max]`.

### 2.5 Comparison: CBF-Beta vs BarrierNet vs RCBF-QP

| Property | CBF-Beta (Training) | BarrierNet (End-to-End) | RCBF-QP (Deployment) |
|----------|---------------------|------------------------|---------------------|
| **When applied** | Training only | Training AND deployment | Deployment only |
| **Mechanism** | Truncate distribution support | Differentiable QP layer in network | Non-differentiable QP filter |
| **Safe set shape** | Axis-aligned box approximation | Full polytope (QP) | Full polytope (QP) |
| **Gradient flow** | Through truncation normalization | Through KKT implicit differentiation | None (not used in training) |
| **Computational cost** | Low (bound computation) | Medium-High (QP solve per forward pass) | Low (QP solve at inference only) |
| **Training-deployment gap** | Yes (different mechanism at deployment) | Zero (same mechanism) | N/A (not used in training) |
| **Robustness margin** | None | Can include GP margin | GP-based robust margin |
| **Infeasibility behavior** | Distribution narrows gracefully | QP may fail (needs handling) | QP may fail (needs handling) |
| **Policy type** | Stochastic (Beta) | Deterministic (QP output) | Deterministic (QP output) |
| **Convergence guarantee** | Theorem 1 of [16] | Empirical (no formal guarantee) | N/A |

---

## 3. Relevant Literature

### 3.1 Primary References

| Paper ID | Citation | What to Use |
|----------|----------|-------------|
| [N04] | Xiao et al., "BarrierNet: Differentiable Control Barrier Functions for Learning of Safe Robot Control," T-RO 2023 | Core BarrierNet architecture, dCBF formulation, differentiable QP layer design, training procedure. Code: https://github.com/Weixy21/BarrierNet |
| [16] | Suttle et al., "Sampling-Based Safe RL with CBF-Constrained PPO," AISTATS 2024 | CBF-Beta policy (the baseline being compared against), convergence guarantee (Theorem 1), truncated Beta sampling |
| [06] | Emam et al., "Safe RL with Robust CBF," 2022 | RCBF-QP deployment filter (the deployment mechanism being compared against), GP disturbance model |
| [N12] | Zhang & Yang, "Dynamic Obstacle Avoidance via CBF for Car-like Robots," Neurocomputing 2025 | VCP-CBF formulation for nonholonomic robots -- the CBF constraints that go INTO the differentiable QP |

### 3.2 Supporting References

| Paper ID | Citation | What to Use |
|----------|----------|-------------|
| [N05] | Zhang et al., "GCBF+: Scalable Graph CBFs," T-RO 2024 | Insight: use QP-filtered policy in training loss; look-ahead for actuation feasibility |
| [N02] | So & Fan, "PNCBF: How to Train Your Neural CBF," ICRA 2024 | Fallback: if BarrierNet with hand-crafted VCP-CBFs fails, use learned neural CBFs inside the differentiable QP |
| [N13] | Xiao et al., "Learning Feasibility Constraints for CBF," ECC 2023 | Infeasibility handling for the differentiable QP layer (same challenge as RCBF-QP) |

### 3.3 Software References

| Software | Reference | What to Use |
|----------|-----------|-------------|
| cvxpylayers | Agrawal et al., NeurIPS 2019, https://github.com/cvxgrp/cvxpylayers | Differentiable QP implementation (primary) |
| qpth | Amos & Kolter, ICML 2017, https://github.com/locuslab/qpth | Differentiable QP implementation (alternative, faster for small QPs) |
| BarrierNet code | https://github.com/Weixy21/BarrierNet | Reference implementation to study |

---

## 4. Session-wise Implementation Breakdown

**Time buffer note**: All session duration estimates include a 1h buffer for debugging, unexpected issues, and documentation. Total estimated: 30-38 hours + 7h buffer = 37-45 hours.

### Session 1: Study BarrierNet Codebase

**Duration**: 1 session (3-4 hours + 1h buffer)
**Objective**: Understand the BarrierNet reference implementation, identify how to adapt it for our VCP-CBF constraints and unicycle dynamics.

**Files to read**:
- BarrierNet repository: https://github.com/Weixy21/BarrierNet
- Focus on: network architecture, QP layer setup, training loop, dCBF formulation
- Our existing VCP-CBF: `src/safety/vcp_cbf.py` (from Phase 2)
- Our existing RCBF-QP: `src/safety/rcbf_qp.py` (from Phase 2)

**Step-by-step**:

1. Clone the BarrierNet repository:
   ```bash
   cd /Users/mohammadshoaib/Codes/robotics/claude_pursuit_evasion
   git clone https://github.com/Weixy21/BarrierNet.git external/BarrierNet
   ```

2. Study the repository structure. Key files to examine:
   - `BarrierNet/barriernet.py` or equivalent -- the differentiable QP layer class
   - Training scripts -- how the loss is computed and backpropagated through the QP
   - CBF definition files -- how CBF constraints are parameterized
   - Any examples with vehicle/robot dynamics

3. Document the BarrierNet architecture by answering these questions:
   - How are CBF constraints passed to the QP layer? (as matrices G, h or symbolically?)
   - Which differentiable QP backend is used? (cvxpylayers, qpth, custom?)
   - How are dCBFs (discretized CBFs) formulated vs continuous CBFs?
   - How does the training loop handle QP infeasibility?
   - What is the computational overhead of the QP layer per forward pass?
   - Does the code support batched QP solving (multiple states in parallel)?

4. Map BarrierNet components to our problem:

   | BarrierNet Component | Our Equivalent |
   |---------------------|----------------|
   | State input | PE observation (relative coords, own state, lidar) |
   | Policy network (MLP) | PPO actor head (outputs nominal [v, omega]) |
   | CBF constraints | VCP-CBF: arena boundary, obstacles, inter-robot collision |
   | QP layer | To be implemented using cvxpylayers or qpth |
   | Training loss | PPO clipped objective (gradients through QP) |

5. Identify adaptations needed:
   - BarrierNet examples likely use simpler dynamics (point mass, double integrator). We need unicycle + VCP.
   - BarrierNet may use continuous-time CBFs. We need discrete-time dCBFs matching our simulation dt = 0.05s.
   - BarrierNet may not handle the adversarial (self-play) setting. We need to integrate with AMS-DRL.

**Verification**:
- [ ] BarrierNet repository cloned and explored
- [ ] Architecture mapping document created (saved to `docs/barriernet_study_notes.md`)
- [ ] Key design decisions for adaptation documented
- [ ] Identified which differentiable QP library to use (cvxpylayers recommended)

---

### Session 2: Implement Differentiable QP Layer for VCP-CBF

**Duration**: 1 session (4-5 hours)
**Objective**: Create a standalone differentiable QP layer that takes a nominal action and state, applies VCP-CBF constraints, and outputs a safe action with correct gradients.

**Files to create**:
- `src/safety/differentiable_qp.py` -- the differentiable QP layer
- `tests/test_differentiable_qp.py` -- unit tests

**Step-by-step**:

1. **Define the QP problem structure for VCP-CBF constraints**:

   For a unicycle robot at state `x = [x_pos, y_pos, theta]` with VCP offset `d`:
   ```python
   # Virtual control point
   q = [x_pos + d * cos(theta), y_pos + d * sin(theta)]

   # VCP time derivative (affine in u = [v, omega]):
   # q_dot = J(theta) @ u
   # where J(theta) = [[cos(theta), -d*sin(theta)],
   #                    [sin(theta),  d*cos(theta)]]

   # For obstacle i at position p_obs_i with safety radius chi_i:
   # h_i(x) = ||q - p_obs_i||^2 - chi_i^2
   #
   # dCBF constraint (discrete time):
   # h_i(x_{k+1}) >= (1 - gamma) * h_i(x_k)
   #
   # Expanding x_{k+1} = x_k + dt * f(x_k, u_k):
   # q_{k+1} = q_k + dt * J(theta_k) @ u_k
   #
   # ||q_{k+1} - p_obs_i||^2 - chi_i^2 >= (1 - gamma) * (||q_k - p_obs_i||^2 - chi_i^2)
   #
   # Let delta_q = q_k - p_obs_i, then q_{k+1} - p_obs_i = delta_q + dt * J @ u
   # ||delta_q + dt * J @ u||^2 >= (1-gamma) * ||delta_q||^2 + gamma * chi_i^2
   #
   # Expanding left side:
   # ||delta_q||^2 + 2*dt*(delta_q^T @ J @ u) + dt^2*||J @ u||^2
   #    >= (1-gamma)*||delta_q||^2 + gamma*chi_i^2
   #
   # For small dt, drop the dt^2 term (or keep it for accuracy):
   # 2*dt*(delta_q^T @ J @ u) + gamma*(||delta_q||^2 - chi_i^2) >= 0
   # 2*dt*(delta_q^T @ J) @ u + gamma * h_i(x_k) >= 0
   #
   # This is LINEAR in u = [v, omega]:
   # a_i^T @ u >= -gamma * h_i(x_k) / (2*dt)
   # where a_i = delta_q^T @ J (a 1x2 vector)
   ```

2. **Implement the QP layer class using cvxpylayers**:

   ```python
   # src/safety/differentiable_qp.py

   import torch
   import torch.nn as nn
   import cvxpy as cp
   import numpy as np
   from cvxpylayers.torch import CvxpyLayer


   class DifferentiableVCPCBFQP(nn.Module):
       """
       Differentiable QP layer for VCP-CBF safety constraints on unicycle dynamics.

       Given a nominal action u_nom and the current state x, solves:
           u* = argmin_{u} ||u - u_nom||^2
           s.t. A_cbf @ u <= b_cbf    (VCP-CBF constraints, negated for <= form)
                u_min <= u <= u_max    (control bounds)

       Gradients flow through u* via implicit differentiation of KKT conditions.
       """

       def __init__(self, n_constraints_max, v_min=0.0, v_max=1.0,
                    omega_min=-2.84, omega_max=2.84):
           super().__init__()
           self.n_constraints_max = n_constraints_max
           self.v_min = v_min
           self.v_max = v_max
           self.omega_min = omega_min
           self.omega_max = omega_max

           # Total constraints: n_cbf + 4 (control bounds: v_min, v_max, omega_min, omega_max)
           n_total = n_constraints_max + 4

           # Define cvxpy problem
           u = cp.Variable(2)                          # [v, omega]
           u_nom_param = cp.Parameter(2)               # nominal action
           G_param = cp.Parameter((n_total, 2))        # constraint matrix
           h_param = cp.Parameter(n_total)              # constraint vector

           objective = cp.Minimize(cp.sum_squares(u - u_nom_param))
           constraints = [G_param @ u <= h_param]
           problem = cp.Problem(objective, constraints)

           assert problem.is_dpp(), "Problem must be DPP-compliant for cvxpylayers"

           self.qp_layer = CvxpyLayer(
               problem,
               parameters=[u_nom_param, G_param, h_param],
               variables=[u]
           )

       def build_constraints(self, states, obstacles, arena_params, d_vcp, dt, gamma):
           """
           Build the constraint matrices G and h for batched states.

           Args:
               states: (B, 3) tensor [x, y, theta] for each robot
               obstacles: list of (pos, radius) tuples
               arena_params: dict with arena boundary info
               d_vcp: VCP offset distance (scalar)
               dt: simulation timestep
               gamma: dCBF decay rate

           Returns:
               G: (B, n_total, 2) constraint matrix
               h: (B, n_total) constraint vector
           """
           B = states.shape[0]
           device = states.device

           x, y, theta = states[:, 0], states[:, 1], states[:, 2]

           # VCP position
           q_x = x + d_vcp * torch.cos(theta)
           q_y = y + d_vcp * torch.sin(theta)

           # Jacobian J(theta): q_dot = J @ u
           cos_th = torch.cos(theta)
           sin_th = torch.sin(theta)
           # J = [[cos(theta), -d*sin(theta)],
           #      [sin(theta),  d*cos(theta)]]

           G_list = []
           h_list = []

           # CBF constraints for each obstacle
           for obs_pos, obs_chi in obstacles:
               delta_x = q_x - obs_pos[0]
               delta_y = q_y - obs_pos[1]
               h_val = delta_x**2 + delta_y**2 - obs_chi**2  # h_i(x_k)

               # a_i = 2 * delta_q^T @ J  (gradient of h w.r.t. q times J)
               a_v = 2 * (delta_x * cos_th + delta_y * sin_th)
               a_omega = 2 * (-delta_x * d_vcp * sin_th + delta_y * d_vcp * cos_th)

               # CBF constraint: a_i^T @ u >= -gamma * h_i / (2*dt)
               # In <= form: -a_i^T @ u <= gamma * h_i / (2*dt)
               # But we want: 2*dt*(a^T @ u) + gamma * h_i >= 0
               # => -(2*dt*a^T) @ u <= gamma * h_i

               G_row = torch.stack([-2*dt*a_v, -2*dt*a_omega], dim=-1)  # (B, 2)
               h_row = gamma * h_val  # (B,)

               G_list.append(G_row)
               h_list.append(h_row)

           # Arena boundary constraints (similar structure)
           # ... (arena-specific CBF constraints added here)

           # Pad to n_constraints_max if fewer constraints
           while len(G_list) < self.n_constraints_max:
               G_list.append(torch.zeros(B, 2, device=device))
               h_list.append(torch.ones(B, device=device) * 1e6)  # inactive

           # Control bound constraints
           # -v <= -v_min  =>  [-1, 0] @ u <= -v_min
           G_list.append(torch.tensor([[-1.0, 0.0]], device=device).expand(B, -1))
           h_list.append(torch.full((B,), -self.v_min, device=device))
           # v <= v_max  =>  [1, 0] @ u <= v_max
           G_list.append(torch.tensor([[1.0, 0.0]], device=device).expand(B, -1))
           h_list.append(torch.full((B,), self.v_max, device=device))
           # -omega <= -omega_min  =>  [0, -1] @ u <= -omega_min
           G_list.append(torch.tensor([[0.0, -1.0]], device=device).expand(B, -1))
           h_list.append(torch.full((B,), -self.omega_min, device=device))
           # omega <= omega_max  =>  [0, 1] @ u <= omega_max
           G_list.append(torch.tensor([[0.0, 1.0]], device=device).expand(B, -1))
           h_list.append(torch.full((B,), self.omega_max, device=device))

           G = torch.stack(G_list, dim=1)  # (B, n_total, 2)
           h = torch.stack(h_list, dim=1)  # (B, n_total)

           return G, h

       def forward(self, u_nom, states, obstacles, arena_params, d_vcp, dt, gamma):
           """
           Forward pass: solve QP for each state in batch.

           Args:
               u_nom: (B, 2) nominal actions [v, omega]
               states: (B, 3) robot states [x, y, theta]
               ... (other args for constraint building)

           Returns:
               u_safe: (B, 2) safe actions
           """
           G, h = self.build_constraints(states, obstacles, arena_params, d_vcp, dt, gamma)

           # Solve batched QP
           # cvxpylayers handles batching automatically
           u_safe_list = []
           for i in range(u_nom.shape[0]):
               try:
                   u_safe_i, = self.qp_layer(
                       u_nom[i], G[i], h[i],
                       solver_args={'solve_method': 'ECOS'}
                   )
                   u_safe_list.append(u_safe_i)
               except Exception:
                   # Infeasibility fallback: return clamped nominal action
                   u_clamped = torch.clamp(
                       u_nom[i],
                       min=torch.tensor([self.v_min, self.omega_min]),
                       max=torch.tensor([self.v_max, self.omega_max])
                   )
                   u_safe_list.append(u_clamped)

           u_safe = torch.stack(u_safe_list, dim=0)
           return u_safe
   ```

3. **Optimize with batched qpth** (if cvxpylayers is too slow for per-sample solving):

   ```python
   # Alternative using qpth for true batched solving
   from qpth.qp import QPFunction

   class DifferentiableVCPCBFQP_qpth(nn.Module):
       def forward(self, u_nom, G, h):
           B = u_nom.shape[0]
           Q = torch.eye(2, device=u_nom.device).unsqueeze(0).expand(B, -1, -1)
           p = -u_nom  # (B, 2)
           A_eq = torch.zeros(B, 0, 2, device=u_nom.device)  # no equality constraints
           b_eq = torch.zeros(B, 0, device=u_nom.device)

           u_safe = QPFunction(verbose=False, maxIter=20)(
               Q.double(), p.double(), G.double(), h.double(),
               A_eq.double(), b_eq.double()
           )
           return u_safe.float()
   ```

4. **Write unit tests** (`tests/test_differentiable_qp.py`):

   ```python
   import torch
   import pytest
   from src.safety.differentiable_qp import DifferentiableVCPCBFQP

   class TestDifferentiableQP:

       def setup_method(self):
           self.qp_layer = DifferentiableVCPCBFQP(
               n_constraints_max=6, v_min=0.0, v_max=1.0,
               omega_min=-2.84, omega_max=2.84
           )
           self.d_vcp = 0.05
           self.dt = 0.05
           self.gamma = 0.5

       def test_forward_pass_safe_action(self):
           """QP should return nominal action when it is already safe."""
           # State far from all obstacles
           states = torch.tensor([[0.0, 0.0, 0.0]])
           u_nom = torch.tensor([[0.5, 0.0]])  # go straight, moderate speed
           obstacles = [
               (torch.tensor([5.0, 5.0]), 1.0),  # far obstacle
           ]
           arena_params = {'radius': 10.0}

           u_safe = self.qp_layer(u_nom, states, obstacles, arena_params,
                                   self.d_vcp, self.dt, self.gamma)

           # Nominal action should be returned (it is safe)
           assert torch.allclose(u_safe, u_nom, atol=1e-3)

       def test_forward_pass_unsafe_action_corrected(self):
           """QP should modify unsafe nominal action to be safe."""
           # State near obstacle, heading toward it
           states = torch.tensor([[0.0, 0.0, 0.0]])  # facing +x
           u_nom = torch.tensor([[1.0, 0.0]])  # full speed toward obstacle
           obstacles = [
               (torch.tensor([0.5, 0.0]), 0.3),  # obstacle directly ahead, close
           ]
           arena_params = {'radius': 10.0}

           u_safe = self.qp_layer(u_nom, states, obstacles, arena_params,
                                   self.d_vcp, self.dt, self.gamma)

           # Safe action should differ from nominal (reduced speed or added steering)
           assert not torch.allclose(u_safe, u_nom, atol=1e-2)
           # Safe action should satisfy control bounds
           assert u_safe[0, 0] >= 0.0 and u_safe[0, 0] <= 1.0
           assert u_safe[0, 1] >= -2.84 and u_safe[0, 1] <= 2.84

       def test_gradient_flows_through_qp(self):
           """Gradients should flow from loss through QP to u_nom."""
           states = torch.tensor([[0.0, 0.0, 0.0]])
           u_nom = torch.tensor([[0.5, 0.0]], requires_grad=True)
           obstacles = [
               (torch.tensor([2.0, 0.0]), 0.5),
           ]
           arena_params = {'radius': 10.0}

           u_safe = self.qp_layer(u_nom, states, obstacles, arena_params,
                                   self.d_vcp, self.dt, self.gamma)

           # Compute a dummy loss and backprop
           loss = u_safe.sum()
           loss.backward()

           assert u_nom.grad is not None
           assert not torch.all(u_nom.grad == 0)

       def test_batch_consistency(self):
           """Batched solving should give same results as individual solving."""
           B = 4
           states = torch.randn(B, 3)
           states[:, 2] = torch.rand(B) * 2 * 3.14159 - 3.14159  # theta in [-pi, pi]
           u_nom = torch.rand(B, 2) * torch.tensor([1.0, 5.68]) + torch.tensor([0.0, -2.84])
           obstacles = [
               (torch.tensor([3.0, 3.0]), 0.8),
           ]
           arena_params = {'radius': 10.0}

           u_safe = self.qp_layer(u_nom, states, obstacles, arena_params,
                                   self.d_vcp, self.dt, self.gamma)

           assert u_safe.shape == (B, 2)

       def test_qp_produces_feasible_solutions(self):
           """All output actions should satisfy CBF constraints."""
           states = torch.tensor([[1.0, 1.0, 0.5]])
           u_nom = torch.tensor([[0.8, 1.5]])
           obstacles = [
               (torch.tensor([2.0, 1.5]), 0.5),
               (torch.tensor([1.5, 2.0]), 0.4),
           ]
           arena_params = {'radius': 5.0}

           u_safe = self.qp_layer(u_nom, states, obstacles, arena_params,
                                   self.d_vcp, self.dt, self.gamma)

           # Verify CBF constraints are satisfied
           G, h_vec = self.qp_layer.build_constraints(
               states, obstacles, arena_params, self.d_vcp, self.dt, self.gamma
           )
           constraint_values = (G[0] @ u_safe[0]) - h_vec[0]
           assert torch.all(constraint_values <= 1e-4), \
               f"Constraint violation: {constraint_values}"
   ```

**Verification**:
- [ ] `DifferentiableVCPCBFQP` class implemented and documented
- [ ] All 5 unit tests pass
- [ ] Gradient flow verified (test_gradient_flows_through_qp passes)
- [ ] QP solve time measured: target <5ms per state on CPU, <1ms on GPU

---

### Session 3: Integrate Differentiable QP into Actor Network

**Duration**: 1 session (4-5 hours)
**Objective**: Create the BarrierNet actor network that chains the PPO policy MLP with the differentiable QP layer, replacing the CBF-Beta truncation.

**Files to create/modify**:
- `src/agents/barriernet_actor.py` -- BarrierNet actor network
- `src/agents/barriernet_ppo.py` -- PPO agent using BarrierNet actor
- `tests/test_barriernet_actor.py` -- integration tests

**Step-by-step**:

1. **Design the BarrierNet actor network**:

   ```python
   # src/agents/barriernet_actor.py

   import torch
   import torch.nn as nn
   from src.safety.differentiable_qp import DifferentiableVCPCBFQP


   class BarrierNetActor(nn.Module):
       """
       BarrierNet actor: MLP policy -> differentiable QP safety layer.

       Architecture:
           observation -> MLP -> nominal action u_nom (unconstrained, in [v_min, v_max] x [omega_min, omega_max])
           (state, u_nom) -> DifferentiableQP -> safe action u*

       The QP layer is differentiable, so gradients from the PPO loss
       flow through the QP back to the MLP parameters.

       Unlike CBF-Beta (which truncates the Beta distribution support),
       this produces a DETERMINISTIC safe action. For PPO exploration,
       we add learned noise BEFORE the QP layer.
       """

       def __init__(self, obs_dim, hidden_dim=256, n_layers=2,
                    n_constraints_max=6, v_max=1.0, omega_max=2.84,
                    d_vcp=0.05, dt=0.05, gamma_cbf=0.5):
           super().__init__()

           # MLP backbone (same architecture as CBF-Beta actor, minus Beta head)
           layers = []
           in_dim = obs_dim
           for _ in range(n_layers):
               layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
               in_dim = hidden_dim
           self.backbone = nn.Sequential(*layers)

           # Nominal action head: outputs u_nom in [action_min, action_max]
           self.action_mean = nn.Linear(hidden_dim, 2)  # [v, omega]
           self.action_log_std = nn.Parameter(torch.zeros(2))  # learnable exploration noise

           # Differentiable QP safety layer
           self.qp_layer = DifferentiableVCPCBFQP(
               n_constraints_max=n_constraints_max,
               v_min=0.0, v_max=v_max,
               omega_min=-omega_max, omega_max=omega_max
           )

           # CBF parameters
           self.d_vcp = d_vcp
           self.dt = dt
           self.gamma_cbf = gamma_cbf
           self.v_max = v_max
           self.omega_max = omega_max

       def get_nominal_action(self, obs):
           """Get the unconstrained nominal action from the MLP."""
           features = self.backbone(obs)
           mean = self.action_mean(features)

           # Squash to action bounds using tanh
           v_nom = (torch.tanh(mean[:, 0]) + 1) / 2 * self.v_max  # [0, v_max]
           omega_nom = torch.tanh(mean[:, 1]) * self.omega_max     # [-omega_max, omega_max]

           return torch.stack([v_nom, omega_nom], dim=-1)

       def forward(self, obs, states, obstacles, arena_params, deterministic=False):
           """
           Forward pass: obs -> MLP -> nominal action -> QP -> safe action.

           Args:
               obs: (B, obs_dim) observations
               states: (B, 3) robot states [x, y, theta] (needed for CBF computation)
               obstacles: list of (pos, radius) tuples
               arena_params: dict with arena boundary info
               deterministic: if True, use mean action (no exploration noise)

           Returns:
               u_safe: (B, 2) safe actions
               u_nom: (B, 2) nominal actions (for logging)
               log_prob: (B,) log probability (for PPO)
               info: dict with diagnostic info
           """
           # Step 1: Get nominal action from MLP
           u_nom_mean = self.get_nominal_action(obs)

           if deterministic:
               u_nom = u_nom_mean
               log_prob = torch.zeros(obs.shape[0], device=obs.device)
           else:
               # Add exploration noise (Gaussian in unbounded space, then squash)
               std = torch.exp(self.action_log_std)
               noise = torch.randn_like(u_nom_mean) * std
               u_nom = u_nom_mean + noise

               # Clamp to action bounds (pre-QP)
               u_nom[:, 0] = torch.clamp(u_nom[:, 0], 0.0, self.v_max)
               u_nom[:, 1] = torch.clamp(u_nom[:, 1], -self.omega_max, self.omega_max)

               # Log probability of the nominal action (Gaussian)
               log_prob = -0.5 * ((u_nom - u_nom_mean) / std).pow(2).sum(dim=-1) \
                          - 0.5 * 2 * torch.log(std).sum() \
                          - 0.5 * 2 * torch.log(torch.tensor(2 * 3.14159))

           # Step 2: Pass through differentiable QP safety layer
           u_safe = self.qp_layer(
               u_nom, states, obstacles, arena_params,
               self.d_vcp, self.dt, self.gamma_cbf
           )

           # Compute QP correction magnitude (for monitoring)
           correction = (u_safe - u_nom).norm(dim=-1)

           info = {
               'u_nom': u_nom.detach(),
               'u_safe': u_safe.detach(),
               'qp_correction': correction.detach(),
               'action_std': std.detach() if not deterministic else torch.zeros(2),
           }

           return u_safe, u_nom, log_prob, info
   ```

2. **Handle the log-probability correction**:

   A subtle issue: PPO needs `log pi(a|s)` where `a = u_safe` is the action actually taken. But `u_safe` is a deterministic function of `u_nom`, which is sampled from a Gaussian. The log-probability of `u_safe` under the policy is:

   ```
   log pi(u_safe | s) = log pi(u_nom | s) - log |det(d(u_safe)/d(u_nom))|
   ```

   The Jacobian `d(u_safe)/d(u_nom)` is available from the differentiable QP. However, computing the log-determinant is expensive. Two practical approaches:

   **Approach A (Recommended for simplicity)**: Use `log pi(u_nom | s)` as an approximation. When the QP is not active (u_safe == u_nom), this is exact. When the QP is active, this introduces bias but is commonly used in practice (BarrierNet paper uses this).

   **Approach B (More correct)**: Compute the correction using the Jacobian:
   ```python
   # After solving QP, compute Jacobian via autograd
   J = torch.autograd.functional.jacobian(
       lambda u: qp_layer(u, states, ...), u_nom
   )
   log_det = torch.log(torch.abs(torch.det(J)))
   log_prob_safe = log_prob_nom - log_det
   ```

   Start with Approach A; switch to Approach B if training is unstable.

3. **Create the BarrierNet PPO agent**:

   ```python
   # src/agents/barriernet_ppo.py

   import torch
   import torch.nn as nn
   from src.agents.barriernet_actor import BarrierNetActor


   class BarrierNetCritic(nn.Module):
       """Standard PPO critic (no QP layer needed -- critic estimates value only)."""

       def __init__(self, obs_dim, hidden_dim=256, n_layers=2):
           super().__init__()
           layers = []
           in_dim = obs_dim
           for _ in range(n_layers):
               layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
               in_dim = hidden_dim
           layers.append(nn.Linear(hidden_dim, 1))
           self.net = nn.Sequential(*layers)

       def forward(self, obs):
           return self.net(obs).squeeze(-1)


   class BarrierNetPPO:
       """
       PPO agent using BarrierNet actor (differentiable QP safety layer).

       Key difference from standard PPO:
       - Actor outputs go through a differentiable QP layer
       - Gradients from PPO loss flow through the QP to the actor parameters
       - Actions are deterministically safe (no separate safety filter at deployment)
       """

       def __init__(self, obs_dim, hidden_dim=256, n_constraints_max=6,
                    lr_actor=3e-4, lr_critic=3e-4, gamma=0.99,
                    gae_lambda=0.95, clip_ratio=0.2, entropy_coeff=0.01,
                    v_max=1.0, omega_max=2.84, d_vcp=0.05, dt=0.05,
                    gamma_cbf=0.5):
           self.actor = BarrierNetActor(
               obs_dim=obs_dim, hidden_dim=hidden_dim,
               n_constraints_max=n_constraints_max,
               v_max=v_max, omega_max=omega_max,
               d_vcp=d_vcp, dt=dt, gamma_cbf=gamma_cbf
           )
           self.critic = BarrierNetCritic(obs_dim=obs_dim, hidden_dim=hidden_dim)

           self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
           self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

           self.gamma = gamma
           self.gae_lambda = gae_lambda
           self.clip_ratio = clip_ratio
           self.entropy_coeff = entropy_coeff

       def get_action(self, obs, states, obstacles, arena_params, deterministic=False):
           """Get action for environment interaction."""
           with torch.no_grad() if deterministic else torch.enable_grad():
               u_safe, u_nom, log_prob, info = self.actor(
                   obs, states, obstacles, arena_params, deterministic
               )
           return u_safe, log_prob, info

       def update(self, rollout_buffer):
           """
           PPO update with gradients flowing through the QP layer.

           The key difference: when computing the policy loss, the action
           u_safe = QP(u_nom(theta), x) depends on theta through u_nom.
           The PPO clipped objective gradient flows through the QP.
           """
           for epoch in range(10):  # PPO epochs
               for batch in rollout_buffer.get_batches(batch_size=256):
                   obs = batch['obs']
                   states = batch['states']
                   actions_old = batch['actions']
                   log_probs_old = batch['log_probs']
                   advantages = batch['advantages']
                   returns = batch['returns']
                   obstacles = batch['obstacles']
                   arena_params = batch['arena_params']

                   # Forward pass through BarrierNet actor
                   u_safe, u_nom, log_probs_new, info = self.actor(
                       obs, states, obstacles, arena_params
                   )

                   # PPO clipped objective
                   ratio = torch.exp(log_probs_new - log_probs_old)
                   surr1 = ratio * advantages
                   surr2 = torch.clamp(ratio, 1 - self.clip_ratio,
                                       1 + self.clip_ratio) * advantages
                   policy_loss = -torch.min(surr1, surr2).mean()

                   # Entropy bonus (from the Gaussian before QP, not after)
                   std = torch.exp(self.actor.action_log_std)
                   entropy = 0.5 * (1 + torch.log(2 * 3.14159 * std**2)).sum()
                   entropy_loss = -self.entropy_coeff * entropy

                   # Actor loss
                   actor_loss = policy_loss + entropy_loss

                   self.optimizer_actor.zero_grad()
                   actor_loss.backward()  # Gradients flow through QP layer
                   torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                   self.optimizer_actor.step()

                   # Critic loss (standard, no QP involved)
                   values = self.critic(obs)
                   critic_loss = 0.5 * (values - returns).pow(2).mean()

                   self.optimizer_critic.zero_grad()
                   critic_loss.backward()
                   torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                   self.optimizer_critic.step()

           return {
               'policy_loss': policy_loss.item(),
               'critic_loss': critic_loss.item(),
               'entropy': entropy.item(),
               'mean_qp_correction': info['qp_correction'].mean().item(),
           }
   ```

**Verification**:
- [ ] `BarrierNetActor` forward pass produces actions within control bounds
- [ ] `BarrierNetActor` forward pass produces actions satisfying CBF constraints
- [ ] Gradient flows from PPO loss through QP to MLP parameters (verified via `.grad` attributes)
- [ ] `BarrierNetPPO.update()` runs without errors on synthetic data
- [ ] QP correction magnitude is non-zero when near obstacles and zero when far

---

### Session 4: Modify PPO Training to Backprop Through QP Layer

**Duration**: 1 session (4-5 hours)
**Objective**: Integrate the BarrierNet PPO agent with the PE environment from Phase 2, handling the rollout collection with QP-in-the-loop and ensuring training stability.

**Files to create/modify**:
- `src/training/barriernet_trainer.py` -- training loop with QP-aware rollout collection
- `src/training/rollout_buffer.py` -- rollout buffer that stores states for QP reconstruction
- `tests/test_barriernet_training.py` -- training integration tests

**Step-by-step**:

1. **Create QP-aware rollout buffer**:

   The standard PPO rollout buffer stores (obs, action, log_prob, reward, done). For BarrierNet, we also need to store the robot state (for QP constraint reconstruction during the update phase):

   ```python
   # src/training/rollout_buffer.py

   class BarrierNetRolloutBuffer:
       """
       Rollout buffer that additionally stores robot states and environment
       info needed to reconstruct QP constraints during PPO updates.
       """

       def __init__(self, buffer_size, obs_dim, action_dim=2, state_dim=3):
           self.obs = torch.zeros(buffer_size, obs_dim)
           self.states = torch.zeros(buffer_size, state_dim)  # [x, y, theta]
           self.actions = torch.zeros(buffer_size, action_dim)
           self.log_probs = torch.zeros(buffer_size)
           self.rewards = torch.zeros(buffer_size)
           self.dones = torch.zeros(buffer_size)
           self.values = torch.zeros(buffer_size)
           self.advantages = torch.zeros(buffer_size)
           self.returns = torch.zeros(buffer_size)

           # Environment info for QP reconstruction
           self.obstacle_positions = []  # list of obstacle configs per step
           self.arena_params_list = []

           self.ptr = 0
           self.buffer_size = buffer_size

       def add(self, obs, state, action, log_prob, reward, done, value,
               obstacles, arena_params):
           idx = self.ptr
           self.obs[idx] = obs
           self.states[idx] = state
           self.actions[idx] = action
           self.log_probs[idx] = log_prob
           self.rewards[idx] = reward
           self.dones[idx] = done
           self.values[idx] = value
           self.obstacle_positions.append(obstacles)
           self.arena_params_list.append(arena_params)
           self.ptr += 1

       def compute_gae(self, last_value, gamma=0.99, gae_lambda=0.95):
           """Compute GAE advantages and returns."""
           last_gae = 0
           for t in reversed(range(self.ptr)):
               if t == self.ptr - 1:
                   next_value = last_value
               else:
                   next_value = self.values[t + 1]
               next_non_terminal = 1.0 - self.dones[t]
               delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
               last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
               self.advantages[t] = last_gae
           self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

       def get_batches(self, batch_size):
           """Yield random mini-batches for PPO update."""
           indices = torch.randperm(self.ptr)
           for start in range(0, self.ptr, batch_size):
               end = min(start + batch_size, self.ptr)
               batch_indices = indices[start:end]
               yield {
                   'obs': self.obs[batch_indices],
                   'states': self.states[batch_indices],
                   'actions': self.actions[batch_indices],
                   'log_probs': self.log_probs[batch_indices],
                   'advantages': self.advantages[batch_indices],
                   'returns': self.returns[batch_indices],
                   'obstacles': [self.obstacle_positions[i] for i in batch_indices],
                   'arena_params': self.arena_params_list[0],  # same for all steps
               }

       def reset(self):
           self.ptr = 0
           self.obstacle_positions = []
           self.arena_params_list = []
   ```

2. **Create the training loop**:

   ```python
   # src/training/barriernet_trainer.py

   import torch
   import time
   from src.agents.barriernet_ppo import BarrierNetPPO
   from src.training.rollout_buffer import BarrierNetRolloutBuffer


   class BarrierNetTrainer:
       """
       Training loop for BarrierNet PPO on the PE environment.

       Key differences from standard PPO trainer:
       1. Rollout collection passes states to actor for QP constraint computation
       2. PPO update backprops through the QP layer
       3. Additional logging for QP-specific metrics (correction magnitude, solve time, infeasibility rate)
       """

       def __init__(self, env, agent, config):
           self.env = env
           self.agent = agent
           self.config = config
           self.rollout_buffer = BarrierNetRolloutBuffer(
               buffer_size=config['rollout_length'],
               obs_dim=env.observation_space.shape[0],
           )

       def collect_rollout(self):
           """Collect a rollout with QP-in-the-loop."""
           obs, info = self.env.reset()
           total_qp_time = 0
           n_infeasible = 0
           n_steps = 0

           for step in range(self.config['rollout_length']):
               obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
               state_tensor = torch.FloatTensor(info['robot_state']).unsqueeze(0)
               obstacles = info['obstacles']
               arena_params = info['arena_params']

               # Get action from BarrierNet actor
               t0 = time.time()
               with torch.no_grad():
                   u_safe, log_prob, act_info = self.agent.get_action(
                       obs_tensor, state_tensor, obstacles, arena_params
                   )
                   value = self.agent.critic(obs_tensor)
               qp_time = time.time() - t0
               total_qp_time += qp_time

               # Track infeasibility
               if act_info.get('infeasible', False):
                   n_infeasible += 1

               # Step environment
               action = u_safe.squeeze(0).numpy()
               next_obs, reward, terminated, truncated, next_info = self.env.step(action)
               done = terminated or truncated

               # Store in buffer
               self.rollout_buffer.add(
                   obs=obs_tensor.squeeze(0),
                   state=state_tensor.squeeze(0),
                   action=u_safe.squeeze(0),
                   log_prob=log_prob.squeeze(0),
                   reward=reward,
                   done=float(done),
                   value=value.squeeze(0),
                   obstacles=obstacles,
                   arena_params=arena_params,
               )

               obs = next_obs
               info = next_info
               n_steps += 1

               if done:
                   obs, info = self.env.reset()

           # Compute GAE
           with torch.no_grad():
               last_value = self.agent.critic(
                   torch.FloatTensor(obs).unsqueeze(0)
               ).squeeze(0)
           self.rollout_buffer.compute_gae(last_value, self.agent.gamma, self.agent.gae_lambda)

           return {
               'mean_qp_time': total_qp_time / n_steps,
               'infeasibility_rate': n_infeasible / n_steps,
               'n_steps': n_steps,
           }

       def train(self, total_timesteps):
           """Main training loop."""
           timesteps = 0
           iteration = 0

           while timesteps < total_timesteps:
               # Collect rollout
               rollout_info = self.collect_rollout()
               timesteps += rollout_info['n_steps']
               iteration += 1

               # PPO update (gradients flow through QP)
               update_info = self.agent.update(self.rollout_buffer)

               # Logging
               if iteration % self.config['log_interval'] == 0:
                   self.log_metrics(iteration, timesteps, rollout_info, update_info)

               # Save checkpoint
               if iteration % self.config['save_interval'] == 0:
                   self.save_checkpoint(iteration)

               self.rollout_buffer.reset()

       def log_metrics(self, iteration, timesteps, rollout_info, update_info):
           """Log training metrics including QP-specific ones."""
           metrics = {
               'iteration': iteration,
               'timesteps': timesteps,
               'policy_loss': update_info['policy_loss'],
               'critic_loss': update_info['critic_loss'],
               'entropy': update_info['entropy'],
               'mean_qp_correction': update_info['mean_qp_correction'],
               'mean_qp_solve_time': rollout_info['mean_qp_time'],
               'qp_infeasibility_rate': rollout_info['infeasibility_rate'],
           }
           # Log to wandb or console
           print(f"[Iter {iteration}] steps={timesteps}, "
                 f"policy_loss={metrics['policy_loss']:.4f}, "
                 f"qp_correction={metrics['mean_qp_correction']:.4f}, "
                 f"qp_time={metrics['mean_qp_solve_time']*1000:.1f}ms, "
                 f"infeasibility={metrics['qp_infeasibility_rate']:.4f}")

       def save_checkpoint(self, iteration):
           """Save model checkpoint."""
           path = f"checkpoints/phase2_5/barriernet_iter_{iteration}.pt"
           torch.save({
               'actor': self.agent.actor.state_dict(),
               'critic': self.agent.critic.state_dict(),
               'optimizer_actor': self.agent.optimizer_actor.state_dict(),
               'optimizer_critic': self.agent.optimizer_critic.state_dict(),
               'iteration': iteration,
           }, path)
   ```

3. **Address training stability concerns**:

   - **Gradient clipping**: Already included (clip_grad_norm_ = 0.5). Monitor gradient norms; increase if QP gradients are large.
   - **Learning rate warmup**: Start with lr = 1e-4 for first 10 iterations, then increase to 3e-4.
   - **QP solver warm-starting**: Use the previous solution as initial point for the next QP solve (speeds up convergence).
   - **Infeasibility handling**: If QP is infeasible, fall back to clamped nominal action and flag the step. Do not backprop through infeasible steps.

**Verification**:
- [ ] Training loop runs for 100 iterations without crashing
- [ ] Gradient norms are stable (no explosion or vanishing)
- [ ] QP solve time is acceptable (<10ms per step)
- [ ] QP infeasibility rate is <5%
- [ ] Policy loss decreases over iterations

---

### Session 5: Train BarrierNet Policy on PE Environment

**Duration**: 1-2 sessions (8-10 hours including training time)
**Objective**: Train the BarrierNet policy to convergence on the same PE environment used in Phase 2, with the same hyperparameters.

**Files to create/modify**:
- `scripts/train_barriernet.py` -- training script
- `configs/barriernet_config.yaml` -- hyperparameters
- `results/phase2_5/barriernet/` -- results directory

**Step-by-step**:

1. **Create configuration file**:

   ```yaml
   # configs/barriernet_config.yaml

   # Environment (must match Phase 2 exactly)
   env:
     arena_size: 20.0
     v_max: 1.0
     omega_max: 2.84
     r_robot: 0.15
     r_capture: 0.5
     T_max: 60.0
     dt: 0.05
     n_obstacles: 2  # start with 2 (same as Phase 2)

   # BarrierNet actor
   actor:
     hidden_dim: 256
     n_layers: 2
     n_constraints_max: 8  # 2 obstacles + arena boundary (4 walls) + inter-robot + slack
     d_vcp: 0.05
     gamma_cbf: 0.5

   # PPO hyperparameters (match Phase 2)
   ppo:
     lr_actor: 3e-4
     lr_critic: 3e-4
     gamma: 0.99
     gae_lambda: 0.95
     clip_ratio: 0.2
     entropy_coeff: 0.01
     n_epochs: 10
     batch_size: 256
     rollout_length: 512

   # Training
   training:
     total_timesteps: 2000000  # 2M (same as Phase 2)
     log_interval: 10
     save_interval: 100
     eval_interval: 50
     n_eval_episodes: 20

   # QP solver
   qp:
     backend: 'cvxpylayers'  # or 'qpth'
     max_iter: 50
     warm_start: true
   ```

2. **Create training script**:

   ```python
   # scripts/train_barriernet.py

   import yaml
   import torch
   from src.envs.pursuit_evasion_env import PursuitEvasionEnv
   from src.agents.barriernet_ppo import BarrierNetPPO
   from src.training.barriernet_trainer import BarrierNetTrainer

   def main():
       with open('configs/barriernet_config.yaml') as f:
           config = yaml.safe_load(f)

       env = PursuitEvasionEnv(**config['env'])
       obs_dim = env.observation_space.shape[0]

       agent = BarrierNetPPO(
           obs_dim=obs_dim,
           hidden_dim=config['actor']['hidden_dim'],
           n_constraints_max=config['actor']['n_constraints_max'],
           lr_actor=config['ppo']['lr_actor'],
           lr_critic=config['ppo']['lr_critic'],
           gamma=config['ppo']['gamma'],
           gae_lambda=config['ppo']['gae_lambda'],
           clip_ratio=config['ppo']['clip_ratio'],
           entropy_coeff=config['ppo']['entropy_coeff'],
           v_max=config['env']['v_max'],
           omega_max=config['env']['omega_max'],
           d_vcp=config['actor']['d_vcp'],
           dt=config['env']['dt'],
           gamma_cbf=config['actor']['gamma_cbf'],
       )

       trainer = BarrierNetTrainer(env, agent, config['training'])
       trainer.train(config['training']['total_timesteps'])

   if __name__ == '__main__':
       main()
   ```

3. **Run training and monitor**:
   - Monitor QP solve time per step (target: <10ms)
   - Monitor gradient norms through QP layer (watch for vanishing or exploding)
   - Monitor QP correction magnitude (should decrease as policy learns to be inherently safe)
   - Monitor QP infeasibility rate (should be <5%, ideally <1%)
   - Compare wall-clock training time to Phase 2 CBF-Beta training

4. **Expected training behavior**:
   - **Early training**: High QP correction (policy outputs many unsafe actions, QP corrects them). Gradients through QP teach the policy to avoid unsafe regions.
   - **Mid training**: QP correction decreases as policy learns inherently safe behavior. Training speed may be slower than CBF-Beta due to QP overhead.
   - **Late training**: QP is mostly a no-op (policy is inherently safe). Convergence to similar performance as CBF-Beta.

**Verification**:
- [ ] Training completes 2M timesteps
- [ ] Training curves show convergence (reward plateaus)
- [ ] QP correction magnitude decreases over training
- [ ] QP infeasibility rate < 5% throughout training
- [ ] Wall-clock time measured and compared to Phase 2

---

### Session 6: Comparative Evaluation

**Duration**: 1 session (5-6 hours)
**Objective**: Systematically compare CBF-Beta -> RCBF-QP (Phase 2 approach) vs BarrierNet end-to-end (Phase 2.5 approach) on all relevant metrics.

**Files to create**:
- `scripts/evaluate_comparison.py` -- comparative evaluation script
- `src/evaluation/comparison_framework.py` -- metrics computation
- `results/phase2_5/comparison/` -- all comparison results

**Step-by-step**:

1. **Define the comparison framework**:

   ```python
   # src/evaluation/comparison_framework.py

   import torch
   import numpy as np
   from dataclasses import dataclass
   from typing import Dict, List


   @dataclass
   class EvaluationResult:
       """Results for one approach across N evaluation episodes."""
       approach_name: str
       n_episodes: int

       # Safety metrics
       safety_violation_rate: float        # % episodes with CBF violation
       mean_min_cbf_margin: float          # mean of min h_i(x) across episodes
       cbf_margin_distribution: np.ndarray # histogram of min h_i(x) values

       # Task performance metrics
       capture_rate: float                 # % episodes where pursuer captures evader
       mean_capture_time: float            # mean time to capture (when captured)
       mean_episode_reward: float          # mean total reward per episode

       # Train-deploy gap metrics (only for CBF-Beta approach)
       training_capture_rate: float        # capture rate with training safety (CBF-Beta)
       deployment_capture_rate: float      # capture rate with deployment safety (RCBF-QP)
       train_deploy_gap: float             # |training - deployment| / training

       # Computational metrics
       mean_inference_time: float          # mean time per action (ms)
       training_wall_clock: float          # total training time (hours)
       qp_infeasibility_rate: float        # % timesteps with infeasible QP

       # Behavioral metrics
       mean_qp_correction: float           # mean ||u_safe - u_nom|| per step
       cbf_filter_intervention_rate: float  # % timesteps where QP modifies action


   def evaluate_approach(agent, env, n_episodes, safety_filter=None,
                         approach_name='unknown', record_video=True):
       """
       Evaluate a trained agent over n_episodes.

       Args:
           agent: trained policy (CBF-Beta or BarrierNet)
           env: PE environment
           n_episodes: number of evaluation episodes
           safety_filter: optional RCBF-QP filter (for CBF-Beta deployment evaluation)
           approach_name: name for logging

       Returns:
           EvaluationResult with all metrics
       """
       results = {
           'captures': 0,
           'violations': 0,
           'capture_times': [],
           'episode_rewards': [],
           'min_cbf_margins': [],
           'inference_times': [],
           'qp_corrections': [],
           'interventions': 0,
           'total_steps': 0,
           'infeasible_steps': 0,
       }

       for ep in range(n_episodes):
           obs, info = env.reset()
           episode_reward = 0
           min_margin = float('inf')
           done = False

           while not done:
               t0 = time.time()

               # Get action from agent
               action, _, act_info = agent.get_action(
                   torch.FloatTensor(obs).unsqueeze(0),
                   torch.FloatTensor(info['robot_state']).unsqueeze(0),
                   info['obstacles'],
                   info['arena_params'],
                   deterministic=True,
               )
               action = action.squeeze(0).numpy()

               # Apply deployment safety filter if provided (for CBF-Beta approach)
               if safety_filter is not None:
                   action_pre = action.copy()
                   action, feasible = safety_filter.filter(
                       action, info['robot_state'], info['obstacles'], info['arena_params']
                   )
                   correction = np.linalg.norm(action - action_pre)
                   results['qp_corrections'].append(correction)
                   if correction > 1e-4:
                       results['interventions'] += 1
                   if not feasible:
                       results['infeasible_steps'] += 1
               else:
                   # BarrierNet: correction is already computed in act_info
                   results['qp_corrections'].append(
                       act_info.get('qp_correction', torch.tensor(0.0)).item()
                   )

               inference_time = (time.time() - t0) * 1000  # ms
               results['inference_times'].append(inference_time)

               # Step environment
               obs, reward, terminated, truncated, info = env.step(action)
               done = terminated or truncated
               episode_reward += reward
               results['total_steps'] += 1

               # Check CBF margins
               cbf_margins = info.get('cbf_margins', [])
               if cbf_margins:
                   min_margin = min(min_margin, min(cbf_margins))
                   if min(cbf_margins) < 0:
                       results['violations'] += 1

           results['episode_rewards'].append(episode_reward)
           results['min_cbf_margins'].append(min_margin)
           if terminated and info.get('captured', False):
               results['captures'] += 1
               results['capture_times'].append(info.get('episode_time', 0))

       # Compute final metrics
       return EvaluationResult(
           approach_name=approach_name,
           n_episodes=n_episodes,
           safety_violation_rate=results['violations'] / results['total_steps'],
           mean_min_cbf_margin=np.mean(results['min_cbf_margins']),
           cbf_margin_distribution=np.array(results['min_cbf_margins']),
           capture_rate=results['captures'] / n_episodes,
           mean_capture_time=np.mean(results['capture_times']) if results['capture_times'] else float('inf'),
           mean_episode_reward=np.mean(results['episode_rewards']),
           training_capture_rate=0.0,  # filled in separately
           deployment_capture_rate=0.0,
           train_deploy_gap=0.0,
           mean_inference_time=np.mean(results['inference_times']),
           training_wall_clock=0.0,  # filled in separately
           qp_infeasibility_rate=results['infeasible_steps'] / results['total_steps'],
           mean_qp_correction=np.mean(results['qp_corrections']),
           cbf_filter_intervention_rate=results['interventions'] / results['total_steps'],
       )
   ```

2. **Run evaluation for all configurations**:

   ```python
   # scripts/evaluate_comparison.py

   # Configuration A: CBF-Beta policy with CBF-Beta safety (training conditions)
   result_A_train = evaluate_approach(
       agent=cbf_beta_agent,
       env=pe_env,
       n_episodes=200,
       safety_filter=None,  # CBF-Beta is built into the policy
       approach_name='CBF-Beta (training)'
   )

   # Configuration B: CBF-Beta policy with RCBF-QP safety (deployment conditions)
   result_A_deploy = evaluate_approach(
       agent=cbf_beta_agent_no_truncation,  # Beta policy WITHOUT truncation
       env=pe_env,
       n_episodes=200,
       safety_filter=rcbf_qp_filter,  # external RCBF-QP filter
       approach_name='CBF-Beta -> RCBF-QP (deployment)'
   )

   # Configuration C: BarrierNet policy (same in training and deployment)
   result_B = evaluate_approach(
       agent=barriernet_agent,
       env=pe_env,
       n_episodes=200,
       safety_filter=None,  # QP is built into the policy
       approach_name='BarrierNet (end-to-end)'
   )

   # Compute train-deploy gaps
   gap_A = abs(result_A_train.capture_rate - result_A_deploy.capture_rate) / max(result_A_train.capture_rate, 1e-6)
   gap_B = 0.0  # BarrierNet has zero gap by construction

   # Statistical significance test
   from scipy import stats
   t_stat, p_value = stats.ttest_ind(
       result_A_deploy.cbf_margin_distribution,
       result_B.cbf_margin_distribution
   )
   ```

3. **Produce the comparison table**:

   | Metric | CBF-Beta (Training) | CBF-Beta -> RCBF-QP (Deploy) | BarrierNet E2E | Winner |
   |--------|--------------------|-----------------------------|----------------|--------|
   | Safety violations (%) | Measure | Measure | Measure | Lower is better |
   | Capture rate (%) | Measure | Measure | Measure | Higher is better |
   | Mean capture time (s) | Measure | Measure | Measure | Lower is better |
   | Train-deploy gap (%) | N/A | gap_A | 0% (by construction) | Lower is better |
   | Mean inference time (ms) | Measure | Measure | Measure | Lower is better |
   | Training wall-clock (hours) | Phase 2 value | Same | Measure | Lower is better |
   | QP infeasibility rate (%) | N/A | Measure | Measure | Lower is better |
   | Mean QP correction | N/A | Measure | Measure | Lower is better |
   | CBF filter intervention (%) | N/A | Measure | Measure | Lower is better |

4. **wandb logging and video recording for comparison**:
   ```python
   import wandb
   from gymnasium.wrappers import RecordVideo

   for config_name, agent, safety_filter in comparisons:
       wandb.init(
           project="pursuit-evasion",
           group="phase2.5-comparison",
           name=f"eval-{config_name}-seed{seed}",
           tags=["phase2.5", "comparison", config_name],
       )

       # Record side-by-side eval videos with PERenderer CBF overlay
       eval_env = RecordVideo(pe_env, f"videos/phase2.5/{config_name}/",
                              episode_trigger=lambda ep: ep < 5)  # first 5 episodes

       result = evaluate_approach(agent, eval_env, n_episodes=200,
                                  safety_filter=safety_filter, approach_name=config_name)

       # Log summary metrics
       wandb.log({
           "eval/safety_violations": result.safety_violation_rate,
           "eval/capture_rate": result.capture_rate,
           "eval/train_deploy_gap": result.train_deploy_gap,
           "eval/mean_inference_ms": result.mean_inference_time,
           "eval/qp_infeasibility": result.qp_infeasibility_rate,
       })

       # Upload eval videos as wandb artifacts
       for video_path in glob.glob(f"videos/phase2.5/{config_name}/*.mp4"):
           wandb.log({"eval/video": wandb.Video(video_path, fps=20)})

       wandb.finish()
   ```

**Verification**:
- [ ] All three configurations evaluated on 200 episodes each
- [ ] Train-deploy gap computed for both approaches
- [ ] Statistical significance tests run (p-value < 0.05)
- [ ] Comparison table produced with all metrics filled in
- [ ] Results saved to `results/phase2_5/comparison/`
- [ ] All eval runs logged to wandb with `phase2.5-comparison` group
- [ ] Eval videos recorded with CBF overlay visible (PERenderer active)

---

### Session 7: Decision Analysis and Documentation

**Duration**: 1 session (3-4 hours)
**Objective**: Analyze comparison results, fill in decision matrix, produce a clear recommendation, and document everything for Phase 3.

**Files to create**:
- `results/phase2_5/decision_report.md` -- final decision document
- `results/phase2_5/comparison/figures/` -- visualization directory

**Step-by-step**:

1. **Fill in the decision matrix**:

   | Criterion | Weight | CBF-Beta -> RCBF-QP | BarrierNet E2E | Notes |
   |-----------|--------|---------------------|----------------|-------|
   | Train-deploy gap | 30% | Score (1-5) | Score (1-5) | Primary criterion |
   | Safety violation rate | 25% | Score (1-5) | Score (1-5) | Must be 0% for either to pass |
   | Task performance | 20% | Score (1-5) | Score (1-5) | Capture rate, episode reward |
   | Training speed | 15% | Score (1-5) | Score (1-5) | Wall-clock time |
   | Implementation complexity | 10% | Score (1-5) | Score (1-5) | Maintainability, debuggability |
   | **Weighted Total** | **100%** | **Compute** | **Compute** | **Higher wins** |

   Scoring guide:
   - 5: Excellent (clearly superior)
   - 4: Good (meaningfully better)
   - 3: Acceptable (no significant difference)
   - 2: Concerning (meaningfully worse)
   - 1: Unacceptable (fails criterion)

2. **Decision rules**:

   - **If train-deploy gap for CBF-Beta -> RCBF-QP is <5%**: The gap is small enough that the simpler CBF-Beta approach is preferred (faster training, proven convergence guarantee from Theorem 1 of [16]).
   - **If train-deploy gap for CBF-Beta -> RCBF-QP is 5-15%**: BarrierNet is likely preferred, as the gap is significant but not catastrophic. Weight the training speed cost against the gap reduction.
   - **If train-deploy gap for CBF-Beta -> RCBF-QP is >15%**: BarrierNet is strongly preferred. The gap is too large for the CBF-Beta approach to be practical.
   - **If BarrierNet training fails to converge**: Use CBF-Beta -> RCBF-QP regardless of gap, and document the failure mode for future investigation.
   - **If both have >0% safety violations**: Neither is acceptable. Investigate root cause before proceeding to Phase 3.

3. **Generate visualizations** (using PERenderer + matplotlib + wandb):

   - Side-by-side trajectory plots (CBF-Beta vs BarrierNet) in 3-4 representative scenarios — use `PERenderer` with `render_mode="rgb_array"` to capture frames with CBF overlays, then compose side-by-side with matplotlib
   - Training curves comparison (reward vs timesteps for both approaches) — pull from wandb API: `wandb.Api().runs("pursuit-evasion", filters={"group": "phase2.5-comparison"})`
   - QP correction magnitude over training (should decrease for BarrierNet)
   - CBF margin distribution histograms (compare shapes)
   - Safety violation heatmap (state-space visualization of where violations occur)
   - Gradient norm through QP layer over training (watch for vanishing/exploding)
   - Upload all figures to wandb: `wandb.log({"figures/trajectory_comparison": wandb.Image(fig)})`

4. **Write the decision report** (`results/phase2_5/decision_report.md`):

   ```markdown
   # Phase 2.5 Decision Report: BarrierNet vs CBF-Beta

   ## Summary
   [1-paragraph summary of findings and recommendation]

   ## Quantitative Results
   [Comparison table with all metrics]

   ## Decision Matrix
   [Weighted scoring matrix]

   ## Recommendation
   [Clear statement: use approach X for Phase 3 because Y]

   ## Caveats and Limitations
   [What we could not measure, where results may not generalize]

   ## Impact on Phase 3
   [Specific changes to Phase 3 plan based on this decision]
   ```

**Verification**:
- [ ] Decision matrix filled in with quantitative scores
- [ ] All visualizations generated and saved
- [ ] Decision report written with clear recommendation
- [ ] Phase 3 impact analysis documented
- [ ] All results archived to `results/phase2_5/`

---

## 5. Testing Plan (Automated)

### 5.1 Unit Tests

| Test ID | Name | Description | Inputs | Expected Output | Pass Criteria |
|---------|------|-------------|--------|-----------------|---------------|
| UT-01 | `test_qp_forward_pass_safe` | QP returns nominal action when already safe | State far from obstacles, safe nominal action | u_safe == u_nom (within tolerance) | `torch.allclose(u_safe, u_nom, atol=1e-3)` |
| UT-02 | `test_qp_forward_pass_unsafe` | QP corrects unsafe nominal action | State near obstacle, heading toward it | u_safe != u_nom, u_safe satisfies CBF | `not allclose(u_safe, u_nom)` AND `G @ u_safe <= h + 1e-4` |
| UT-03 | `test_qp_gradient_flow` | Gradients propagate through QP | u_nom with requires_grad=True | u_nom.grad is not None and non-zero | `u_nom.grad is not None` AND `not torch.all(u_nom.grad == 0)` |
| UT-04 | `test_qp_gradient_magnitude` | QP gradients are not vanishing or exploding | Random batch of states and nominal actions | Gradient norms in [1e-6, 1e3] | `1e-6 < grad_norm < 1e3` |
| UT-05 | `test_qp_feasibility` | QP produces feasible solutions for valid states | 100 random valid states with obstacles | All solutions satisfy constraints | `all(G[i] @ u_safe[i] <= h[i] + 1e-4)` for all i |
| UT-06 | `test_qp_control_bounds` | Output respects control bounds | Random nominal actions | v in [0, v_max], omega in [-omega_max, omega_max] | Bounds satisfied for all outputs |
| UT-07 | `test_qp_batch_consistency` | Batched solving matches individual solving | Same inputs, batched vs. loop | Outputs match within tolerance | `allclose(u_batch, u_loop, atol=1e-4)` |
| UT-08 | `test_constraint_matrix_construction` | Constraint matrices are correct | Known state, known obstacle | Hand-computed G, h match code output | `allclose(G_code, G_hand, atol=1e-6)` |
| UT-09 | `test_dcbf_discrete_time` | Discrete CBF condition is correctly formulated | State, action, next_state | h(x_{k+1}) >= (1-gamma)*h(x_k) | CBF condition satisfied after QP |
| UT-10 | `test_vcp_relative_degree` | VCP achieves uniform relative degree 1 | CBF Lie derivative depends on both v and omega | dh/du has non-zero entries for both v and omega | `abs(dh_dv) > 1e-6` AND `abs(dh_domega) > 1e-6` |

### 5.2 Integration Tests

| Test ID | Name | Description | Inputs | Expected Output | Pass Criteria |
|---------|------|-------------|--------|-----------------|---------------|
| IT-01 | `test_barriernet_actor_forward` | Actor network produces valid actions | Random observations, states | Actions within bounds AND satisfying CBF | Bounds and CBF constraints verified |
| IT-02 | `test_barriernet_actor_gradient` | PPO loss gradients reach MLP parameters | Full forward-backward pass | All MLP parameters have non-zero grad | `all(p.grad is not None and p.grad.norm() > 0 for p in mlp_params)` |
| IT-03 | `test_barriernet_ppo_update` | PPO update step runs without errors | Synthetic rollout buffer with 512 steps | No exceptions, losses are finite | `not torch.isnan(actor_loss)` AND `not torch.isnan(critic_loss)` |
| IT-04 | `test_barriernet_training_convergence` | Short training run shows learning | 50K timesteps on simple env | Reward increases over training | `mean_reward[-10:] > mean_reward[:10]` |
| IT-05 | `test_barriernet_safe_actions_all_states` | Safety holds across diverse states | 1000 random states from PE env | All actions satisfy CBF constraints | `violation_count == 0` |
| IT-06 | `test_barriernet_env_integration` | BarrierNet agent interacts with PE env | 10 episodes of env interaction | No crashes, valid observations | No exceptions raised |
| IT-07 | `test_rollout_buffer_qp_reconstruction` | Buffer stores enough info to reconstruct QP | Collect rollout, then reconstruct QP from buffer | Reconstructed QP matches original | `allclose(u_safe_original, u_safe_reconstructed, atol=1e-4)` |

### 5.3 System Tests

| Test ID | Name | Description | Inputs | Expected Output | Pass Criteria |
|---------|------|-------------|--------|-----------------|---------------|
| ST-01 | `test_train_deploy_gap_barriernet` | BarrierNet has zero train-deploy gap | Evaluate same policy in "training" and "deployment" mode | Identical behavior | `gap < 0.01` (1%) |
| ST-02 | `test_train_deploy_gap_cbf_beta` | Measure CBF-Beta train-deploy gap | Evaluate CBF-Beta with CBF-Beta vs with RCBF-QP | Gap measured | Value recorded (expected 5-15%) |
| ST-03 | `test_safety_violation_rate_barriernet` | BarrierNet achieves 0% violations | 200 evaluation episodes | Zero CBF violations | `violation_rate == 0.0` |
| ST-04 | `test_safety_violation_rate_comparison` | Both approaches achieve 0% violations | 200 episodes each | Zero violations for both | `violation_rate_A == 0.0` AND `violation_rate_B == 0.0` |
| ST-05 | `test_task_performance_comparison` | Task performance is comparable | 200 episodes each | Capture rates within 10% | `abs(capture_A - capture_B) / max(capture_A, 0.01) < 0.10` |
| ST-06 | `test_statistical_significance` | Performance differences are significant | 200 episodes each, Welch's t-test | p-value < 0.05 for significant differences | `p_value < 0.05` OR difference is not meaningful |
| ST-07 | `test_training_wallclock_comparison` | Training time overhead is acceptable | Train both for 2M steps | BarrierNet < 5x slower than CBF-Beta | `time_barriernet / time_cbf_beta < 5.0` |

### 5.4 Comparison Framework Tests

| Test ID | Name | Description | Pass Criteria |
|---------|------|-------------|---------------|
| CF-01 | `test_comparison_table_complete` | All cells in comparison table filled | No NaN or None values |
| CF-02 | `test_decision_matrix_consistent` | Weights sum to 100%, scores in [1,5] | `sum(weights) == 1.0` AND `all(1 <= score <= 5)` |
| CF-03 | `test_reproducibility` | Same evaluation gives same results (within tolerance) | Run twice, results match within 5% |

---

## 6. Manual Validation Checklist

### 6.1 Side-by-Side Trajectory Visualization

For each of the following scenarios, generate trajectory plots for both approaches and visually inspect:

- [ ] **Scenario A: Open field pursuit** -- Pursuer starts 10m from evader, no obstacles, open arena. Compare trajectories for smoothness, directness, and capture efficiency.
- [ ] **Scenario B: Obstacle avoidance during pursuit** -- Pursuer must navigate around a central obstacle to reach the evader. Compare how each approach handles the obstacle (steering vs braking).
- [ ] **Scenario C: Near-boundary chase** -- Both robots near the arena boundary. Compare how each approach respects the boundary while maintaining pursuit.
- [ ] **Scenario D: Corner scenario** -- Evader cornered. Compare how each approach handles the tight geometry (QP feasibility, trajectory smoothness).
- [ ] **Scenario E: High-speed approach** -- Pursuer at maximum speed heading toward obstacle. Compare emergency maneuvers (steering response, minimum clearance).

For each scenario, annotate:
- CBF margin values along the trajectory (color-coded: green=safe, yellow=marginal, red=violation)
- QP correction vectors (arrows showing the difference between nominal and safe actions)
- Time-to-capture comparison

### 6.2 QP Layer Gradient Monitoring

- [ ] **Gradient magnitude histogram**: Plot the distribution of `||d(Loss)/d(u_nom)||` across a training run. Should be unimodal, centered away from zero.
- [ ] **Gradient norm over training**: Plot gradient norm vs. training iteration. Should be stable (no trend toward zero or infinity).
- [ ] **Per-parameter gradient analysis**: For each MLP layer, plot gradient norm over training. The QP layer should not cause gradients to vanish in earlier layers.
- [ ] **Active constraint analysis**: Plot the fraction of CBF constraints that are active (tight) during training. If too many constraints are active, gradients may be poorly conditioned.

**Red flags to watch for**:
- Gradient norm drops below 1e-6 (vanishing through QP)
- Gradient norm exceeds 1e3 (exploding through QP)
- Gradient norm oscillates wildly (QP active set changes frequently)
- Gradients are zero for the omega (angular velocity) component (VCP offset d may be too small)

### 6.3 Training Curves Comparison

- [ ] **Reward vs. timesteps**: CBF-Beta and BarrierNet on the same plot. BarrierNet may converge slower but should reach similar final performance.
- [ ] **Policy loss vs. timesteps**: Compare learning stability. BarrierNet may show higher variance due to QP gradient conditioning.
- [ ] **Entropy vs. timesteps**: Compare exploration. BarrierNet's deterministic QP output may reduce effective exploration.
- [ ] **QP correction magnitude vs. timesteps** (BarrierNet only): Should decrease over training as policy learns inherently safe behavior.
- [ ] **CBF filter intervention rate vs. timesteps** (CBF-Beta with RCBF-QP): Should decrease over training.

### 6.4 Safety Violation Heatmaps

For both approaches, create a 2D heatmap of the arena showing:
- [ ] **CBF margin heatmap**: Average minimum CBF margin at each (x, y) position. Both approaches should show high margins (green) everywhere except near obstacles/boundaries.
- [ ] **Violation heatmap**: Number of violations at each (x, y) position. Should be all zeros for both approaches. If not, identify the problematic regions.
- [ ] **QP correction heatmap**: Average QP correction magnitude at each (x, y) position. High correction near obstacles is expected; high correction in open space is concerning.

### 6.5 Behavioral Differences

- [ ] **Trajectory smoothness**: Compute jerk (third derivative of position) for both approaches. BarrierNet should produce smoother trajectories (QP smooths the output).
- [ ] **Action distribution**: Plot the 2D histogram of (v, omega) actions for both approaches. CBF-Beta may show clustering at truncation boundaries; BarrierNet should show smoother distribution.
- [ ] **Steering vs. braking preference**: Near obstacles, measure the ratio of angular velocity change to linear velocity change. BarrierNet should prefer steering (per VCP-CBF design).

### 6.6 Decision Matrix Template

Fill in after running all evaluations:

```
| Criterion              | Weight | CBF-Beta->RCBF-QP | BarrierNet E2E |
|------------------------|--------|--------------------| ---------------|
| Train-deploy gap       |   30%  |    ____ / 5        |    ____ / 5    |
| Safety violation rate  |   25%  |    ____ / 5        |    ____ / 5    |
| Task performance       |   20%  |    ____ / 5        |    ____ / 5    |
| Training speed         |   15%  |    ____ / 5        |    ____ / 5    |
| Impl. complexity       |   10%  |    ____ / 5        |    ____ / 5    |
| WEIGHTED TOTAL         |  100%  |    ____ / 5        |    ____ / 5    |

RECOMMENDATION: ____________________________________________
CONFIDENCE: High / Medium / Low
RATIONALE: _________________________________________________
```

### 6.7 Specific Test Scenarios

Run both approaches on these challenging scenarios and record observations:

| Scenario | Description | What to Observe |
|----------|-------------|-----------------|
| Head-on approach | Pursuer and evader facing each other, 2m apart | Does the pursuer smoothly steer around? Or brake hard? |
| Tight corridor | Two obstacles creating a narrow gap (1.5x robot width) | Can the policy navigate the gap? QP feasibility? |
| Boundary wrap | Evader runs along the arena boundary | Does the pursuer cut corners or follow the boundary? |
| Obstacle shadow | Evader hides behind an obstacle | How does the pursuer navigate around? |
| Multi-obstacle | 4+ obstacles scattered in arena | QP infeasibility rate? Performance degradation? |
| Speed mismatch | Evader is 20% faster than pursuer | Can the pursuer still achieve captures via strategy? |

---

## 7. Success Criteria & Phase Gates

### 7.0 Definition of Done

> **Phase 2.5 is COMPLETE when:**
> 1. BarrierNet implementation passes all unit tests (UT-01 through UT-10)
> 2. BarrierNet training either converges (reward plateau) OR failure is documented with analysis
> 3. All 5 Phase Gate criteria (PG-01 through PG-05) are met
> 4. Decision report exists at `results/phase2_5/decision_report.md` with quantitative backing
> 5. Decision matrix is filled with actual numerical scores (no blanks)
> 6. Phase 3 plan is updated to reflect the chosen safety architecture
> 7. All comparison runs logged to wandb with eval videos showing CBF overlay
> 8. Visualization figures generated and saved to `results/phase2_5/comparison/figures/`

### 7.1 Phase Gate: Must-Pass Criteria

These criteria MUST be met before proceeding to Phase 3:

| Gate ID | Criterion | Threshold | Measurement Method |
|---------|-----------|-----------|-------------------|
| PG-01 | Train-deploy gap measured for BOTH approaches | Values documented | Evaluation script output |
| PG-02 | Safety violation rate for chosen approach | 0% over 200 eval episodes | System test ST-03/ST-04 |
| PG-03 | Clear recommendation produced | Decision report exists | Document review |
| PG-04 | Quantitative backing for recommendation | Decision matrix filled, p-values computed | Statistical tests |
| PG-05 | BarrierNet training converges OR failure documented | Reward plateaus OR failure analysis written | Training curves |

### 7.2 Target Metrics

| Metric | Target | Stretch Goal |
|--------|--------|-------------|
| Train-deploy gap (chosen approach) | <5% | <2% |
| Safety violations (deployment) | 0% | 0% |
| Task performance vs. unconstrained baseline | Within 10% | Within 5% |
| Training wall-clock overhead (BarrierNet vs CBF-Beta) | <5x | <3x |
| QP infeasibility rate (BarrierNet) | <5% | <1% |

### 7.3 Decision Outcomes

**Outcome A: CBF-Beta -> RCBF-QP is chosen**
- Train-deploy gap < 5% (acceptable)
- BarrierNet training is significantly slower (>5x)
- Phase 3 uses CBF-Beta for training, RCBF-QP for deployment
- Document the gap and note it as a limitation in the paper

**Outcome B: BarrierNet is chosen**
- Train-deploy gap for CBF-Beta > 5% (problematic)
- BarrierNet training overhead is acceptable (<5x)
- Phase 3 uses BarrierNet end-to-end for both training and deployment
- Removes RCBF-QP from the deployment pipeline (simpler architecture)

**Outcome C: Hybrid approach**
- BarrierNet for training (policy learns with QP), RCBF-QP with GP for deployment (adds robustness)
- This gives the best of both worlds: zero training gap + robust deployment
- Phase 3 uses BarrierNet training + RCBF-QP+GP deployment

**Outcome D: Both approaches fail**
- Safety violations > 0% for both approaches
- Investigate root cause (VCP-CBF formulation issue? QP numerical instability?)
- Consider PNCBF (N02) as fallback: learn the CBF from the value function
- Do NOT proceed to Phase 3 until safety is resolved

---

## 8. Troubleshooting Guide

### 8.1 QP Layer Gradient Issues

**Problem: Vanishing gradients through QP layer**
- **Symptom**: MLP parameter gradients are near zero; training does not progress
- **Diagnosis**: Check `actor.backbone[0].weight.grad.norm()` -- if < 1e-6 across many updates, gradients are vanishing
- **Cause 1**: All CBF constraints are inactive (QP is trivially u* = u_nom). Gradient is identity (no issue), so look elsewhere.
- **Cause 2**: Many constraints are active with nearly degenerate constraint matrix. The Jacobian `d(u*)/d(u_nom)` becomes ill-conditioned.
- **Fix for Cause 2**:
  - Add a small regularization to the QP: `Q = I + epsilon * I` with epsilon = 1e-4
  - Reduce the number of constraints (consolidate similar obstacles)
  - Increase the VCP offset `d` slightly (0.05 -> 0.08)

**Problem: Exploding gradients through QP layer**
- **Symptom**: Gradient norms > 1e3; loss spikes; NaN in parameters
- **Diagnosis**: Check gradient norms after backward pass
- **Cause**: QP active set changes rapidly (constraint switching), causing discontinuous gradients
- **Fix**:
  - Increase gradient clipping (reduce from 0.5 to 0.1)
  - Add gradient norm monitoring and skip updates where grad_norm > threshold
  - Use a learning rate warmup (start at 1e-5, increase to 3e-4 over 50 iterations)

**Problem: Gradients are zero for omega (angular velocity) but not for v**
- **Symptom**: Policy learns to control v but not omega
- **Cause**: VCP offset `d` is too small, making the omega-dependent terms in the CBF constraint negligible
- **Fix**: Increase d from 0.05 to 0.10. Verify that `dh/d(omega)` is non-negligible for the increased d.

### 8.2 cvxpylayers Numerical Instability

**Problem: cvxpylayers raises SolverError**
- **Symptom**: `SolverError: Solver 'ECOS' failed` during forward pass
- **Cause 1**: Infeasible QP (conflicting constraints)
- **Fix for Cause 1**: Implement fallback to clamped nominal action; log the infeasible state for analysis
- **Cause 2**: Numerical conditioning of constraint matrix
- **Fix for Cause 2**: Scale constraints to similar magnitudes; use `solver_args={'solve_method': 'SCS'}` (more robust, slightly slower)

**Problem: cvxpylayers gives NaN gradients**
- **Symptom**: `u_safe` is valid but `u_nom.grad` contains NaN
- **Cause**: Degenerate KKT system (constraints are nearly parallel or nearly identical)
- **Fix**:
  - Remove redundant constraints (e.g., if two obstacles have nearly identical constraint rows)
  - Add constraint matrix regularization: `G = G + 1e-6 * torch.randn_like(G)`
  - Switch to qpth, which uses a different differentiation method

**Problem: cvxpylayers is too slow**
- **Symptom**: QP solve takes >50ms per state, training is impractically slow
- **Fix**:
  - Switch to qpth (native PyTorch, batched GPU solving)
  - Reduce n_constraints_max (consolidate constraints)
  - Use QP warm-starting (initialize from previous solution)
  - For training only: solve QP on GPU in parallel across the batch

### 8.3 Slow Training with Differentiable QP

**Problem: Training is >10x slower than CBF-Beta**
- **Symptom**: Wall-clock time per iteration is unacceptable
- **Cause 1**: QP solver overhead per forward pass
- **Fix for Cause 1**:
  - Use qpth with batched GPU solving
  - Profile the training loop to identify bottleneck (QP solve vs. gradient computation)
  - Reduce rollout length (512 -> 256) and compensate with more parallel environments
- **Cause 2**: Gradient computation through QP is expensive
- **Fix for Cause 2**:
  - Only backprop through QP every K updates (e.g., K=3): use detached QP for K-1 updates, full backprop for 1
  - This is an approximation but significantly speeds up training
- **Cause 3**: cvxpylayers recompiles the problem each time
- **Fix for Cause 3**: Pre-compile the problem once in __init__; ensure parameter shapes do not change

### 8.4 Infeasibility in Differentiable QP

**Problem: High infeasibility rate (>5%)**
- **Symptom**: Many QP solves fail; frequent fallback to clamped actions
- **Cause 1**: Conflicting CBF constraints (e.g., obstacle very close to arena boundary)
- **Fix for Cause 1**: Reduce obstacle sizes or increase arena size for initial experiments
- **Cause 2**: Robot state is already in an unsafe region (h_i(x) < 0 for some i)
- **Fix for Cause 2**: Add slack variables to the QP:
  ```
  u* = argmin_{u, s} ||u - u_nom||^2 + M * ||s||^2
  s.t. dh_i + alpha_i * h_i + s_i >= 0   (s_i is slack, penalized heavily)
       u in U
       s >= 0
  ```
  The large penalty M (e.g., M = 1000) keeps slack near zero when feasible but allows constraint relaxation when necessary.
- **Cause 3**: Discrete-time CBF condition is too aggressive (gamma too large)
- **Fix for Cause 3**: Reduce gamma_cbf from 0.5 to 0.3 or 0.2 (less aggressive safety requirement)
- **Integrated approach**: Use the N13 learned feasibility constraints from Phase 2 to train the BarrierNet policy to avoid infeasible regions. The same SVM/DNN classifier from Phase 2's Tier 1 infeasibility handling can be used.

### 8.5 BarrierNet Policy Produces Overly Conservative Behavior

**Problem: BarrierNet agent is much more conservative than CBF-Beta agent**
- **Symptom**: Lower capture rate, longer capture times, larger CBF margins (too far from obstacles)
- **Cause**: The QP layer's gradient signal may push the policy toward actions that are "easy" for the QP (far from all constraints), rather than task-optimal actions that are close to constraints.
- **Fix**:
  - Reduce gamma_cbf (less aggressive safety, more freedom for task performance)
  - Add a reward term that penalizes excessive QP correction: `r_correction = -w6 * ||u_safe - u_nom||`
  - Increase exploration noise (increase `action_log_std` initial value)
  - Use adaptive gamma (Section 2.2.4): let the network learn when to be conservative vs. aggressive

---

## 9. Guide to Next Phase

### 9.1 How This Phase Feeds Into Phase 3

Phase 3 (Partial Observability + Self-Play, Months 4-6) adds:
- Limited FOV sensor model
- BiMDN belief encoder
- AMS-DRL self-play protocol
- Curriculum learning

The Phase 2.5 decision directly affects Phase 3 architecture:

**If CBF-Beta -> RCBF-QP is chosen**:
- Phase 3 uses CBF-Beta for training (truncated Beta sampling from Phase 2)
- Deployment pipeline: Policy -> RCBF-QP -> Robot
- Self-play: both agents use CBF-Beta during training
- BiMDN belief encoder feeds into the Beta policy network
- No additional QP overhead during training

**If BarrierNet is chosen**:
- Phase 3 uses BarrierNet for training (differentiable QP in actor)
- Deployment pipeline: Policy with built-in QP -> Robot (simpler, no separate filter)
- Self-play: both agents have QP layers in their actors
- BiMDN belief encoder feeds into the BarrierNet backbone (before QP)
- QP overhead during training (budget for slower training)
- May need to use qpth for batched GPU solving to handle self-play compute requirements

**If Hybrid is chosen**:
- Phase 3 trains with BarrierNet (end-to-end)
- At deployment, replace the differentiable QP with RCBF-QP+GP for robustness
- The train-deploy gap should be small (both are QP-based, just different margins)
- This is the most complex but potentially best-performing option

### 9.2 Artifacts Produced by This Phase

| Artifact | Location | Used By |
|----------|----------|---------|
| Differentiable QP layer implementation | `src/safety/differentiable_qp.py` | Phase 3 (if BarrierNet chosen) |
| BarrierNet actor network | `src/agents/barriernet_actor.py` | Phase 3 (if BarrierNet chosen) |
| BarrierNet PPO agent | `src/agents/barriernet_ppo.py` | Phase 3 (if BarrierNet chosen) |
| BarrierNet trainer | `src/training/barriernet_trainer.py` | Phase 3 (if BarrierNet chosen) |
| Comparison evaluation framework | `src/evaluation/comparison_framework.py` | Phase 3 (for ablation studies) |
| Trained BarrierNet checkpoint | `checkpoints/phase2_5/` | Phase 3 (warm-starting if BarrierNet chosen) |
| Decision report | `results/phase2_5/decision_report.md` | Phase 3 planning, paper writing |
| Comparison metrics and figures | `results/phase2_5/comparison/` | Paper (ablation study 10 from pathway doc) |

### 9.3 Specific Phase 3 Changes Based on Decision

Regardless of the decision, the following ablation study from the pathway document is now complete:
- **Ablation 10**: "CBF-Beta -> RCBF-QP vs BarrierNet end-to-end: Compare train-deploy performance gap (Phase 2.5)"

The results feed directly into:
- Section 5.1 of the pathway doc (Primary metrics: train-deploy safety gap)
- Section 5.3 (Ablation studies)
- Section 8 (Risk assessment: "Train-deploy safety gap" risk)

---

## 10. Software & Reproducibility

### 10.1 Additional Packages (Beyond Phase 2)

```txt
# Phase 2.5 additions (append to Phase 2 requirements.txt)
cvxpylayers==0.1.6         # Differentiable QP layer (primary)
qpth==0.0.16               # Alternative differentiable QP (faster for small QPs)
# ecos==2.0.14             # Already a cvxpy dependency; QP solver backend

# Visualization & Tracking (inherited from Phase 1, listed for completeness)
# pygame==2.6.1            # Already in Phase 1 — PERenderer used for comparison videos
# wandb==0.19.1            # Already in Phase 1 — comparison eval logging
# hydra-core==1.3.2        # Already in Phase 1
```

**Version compatibility**: cvxpylayers 0.1.6 requires cvxpy>=1.1 (Phase 2 pins 1.6.5, compatible) and torch>=1.0 (Phase 1 pins 2.6.0, compatible). qpth 0.0.16 requires torch>=1.0.

### 10.2 Reproducibility Protocol

```python
# Same seed protocol as Phase 1 (Appendix B) for all Phase 2.5 runs.
# Additional notes for differentiable QP:

# cvxpylayers is deterministic for a given input (QP has unique solution).
# qpth is also deterministic.
# No additional seeding needed for QP solver.

# IMPORTANT: BarrierNet training uses exploration noise (Gaussian before QP).
# Seed the noise RNG:
torch.manual_seed(seed)

# Run all evaluation with 3 seeds: [0, 1, 2]
# Statistical significance: Welch's t-test with p < 0.05
```

---

## Appendix A: Mathematical Details

### A.1 VCP-CBF Constraint Linearization for QP

For obstacle avoidance with VCP at `q = [x + d*cos(theta), y + d*sin(theta)]`:

```
h(x) = ||q - p_obs||^2 - chi^2

Gradient w.r.t. q:
  grad_q h = 2 * (q - p_obs)

VCP Jacobian (q_dot = J @ u):
  J = [[cos(theta), -d*sin(theta)],
       [sin(theta),  d*cos(theta)]]

dCBF constraint (continuous time):
  dh/dt + alpha * h >= 0
  grad_q h^T @ J @ u + alpha * h >= 0

  Let a = J^T @ grad_q h = J^T @ 2*(q - p_obs)
  a^T @ u >= -alpha * h

  This is LINEAR in u = [v, omega].

dCBF constraint (discrete time, Euler integration):
  h(x_{k+1}) - (1 - gamma) * h(x_k) >= 0

  where x_{k+1} uses q_{k+1} = q_k + dt * J @ u_k

  Expanding (ignoring dt^2 terms):
  2*dt * (q_k - p_obs)^T @ J @ u_k + gamma * h(x_k) >= 0

  a_k^T @ u_k >= -gamma * h(x_k) / (2 * dt)
  where a_k = 2 * (q_k - p_obs)^T @ J
```

### A.2 KKT Differentiation for Differentiable QP

The QP:
```
min  (1/2) ||u - u_nom||^2
s.t. A_cbf @ u <= b_cbf    (m inequality constraints)
```

KKT conditions at optimum (u*, lambda*):
```
u* - u_nom + A_cbf^T @ lambda* = 0           (stationarity)
lambda* >= 0                                    (dual feasibility)
lambda*_i * (A_cbf_i @ u* - b_cbf_i) = 0      (complementary slackness)
A_cbf @ u* <= b_cbf                            (primal feasibility)
```

Let I_active = {i : lambda*_i > 0} be the active set.

Differentiating w.r.t. u_nom:
```
d(u*)/d(u_nom) - I + A_active^T @ d(lambda_active)/d(u_nom) = 0
A_active @ d(u*)/d(u_nom) = 0

Solving:
d(u*)/d(u_nom) = I - A_active^T @ (A_active @ A_active^T)^{-1} @ A_active

This is the projection onto the null space of active constraints.
```

When no constraints are active: `d(u*)/d(u_nom) = I` (identity, gradients pass through unchanged).
When all constraints are active: `d(u*)/d(u_nom)` projects gradients onto the feasible manifold.

### A.3 Comparison of Gradient Signals

**CBF-Beta gradient** (from [16]):
```
grad_theta J = E_{u ~ pi_theta^C} [grad_theta log pi_theta^C(u|x) * A(x, u)]

where log pi_theta^C = log pi_theta(u|x) - log pi_theta(C(x)|x)

The second term (normalization) provides a gradient signal that depends on the
safe set C(x), pushing the policy toward actions well within the safe set.
```

**BarrierNet gradient**:
```
grad_theta J = E [grad_{u*} J * d(u*)/d(u_nom) * d(u_nom)/d(theta)]

The QP Jacobian d(u*)/d(u_nom) acts as a "safety-aware" gradient modulator:
- When u_nom is safe (no active constraints): gradient passes through unchanged
- When u_nom is unsafe (active constraints): gradient is projected onto feasible manifold
```

The key difference: CBF-Beta's gradient includes the normalization constant (how much of the Beta distribution falls within the safe set), while BarrierNet's gradient includes the QP Jacobian (how the QP solution changes with the nominal action). Both provide safety-aware gradient information, but through different mathematical mechanisms.
