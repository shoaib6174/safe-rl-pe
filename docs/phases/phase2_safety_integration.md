# Phase 2: Safety Integration

**Timeline**: Months 2-3
**Status**: Pending (requires Phase 1 completion)
**Prerequisites**: Phase 1 — all success criteria met, VCP-CBF validated
**Next Phase**: [Phase 2.5 — BarrierNet Experiment](./phase2_5_barriernet_experiment.md)

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
10. [Guide to Phase 2.5](#10-guide-to-phase-25)

---

## 1. Phase Overview

### 1.1 Goal

Integrate **safety guarantees** into the PE training pipeline. This is the core novelty of the project — making deep RL for pursuit-evasion **provably safe** during both training and deployment.

Specifically:
- Implement VCP-CBF constraints for arena boundary, obstacles, and inter-robot collision
- Implement the CBF-constrained Beta policy for safe training (Paper [16])
- Implement the RCBF-QP safety filter for deployment (Paper [06])
- Implement 3-tier infeasibility handling (Paper [N13])
- Add safety-reward shaping (w5 term)
- Ablation: demonstrate the value of each safety component

### 1.2 Why This Phase Matters

This phase delivers the **primary research contribution**: the first safe DRL system for PE with hard constraint guarantees. Without safety, the Phase 1 system is just another RL-for-PE paper. The safety layer is what makes the work novel and publishable.

The safety architecture has two components working together:
1. **Training safety (CBF-Beta)**: Ensures the policy ONLY samples safe actions during training
2. **Deployment safety (RCBF-QP)**: Provides robust safety under real-world model uncertainty

### 1.3 Key Design Decisions for Phase 2

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CBF formulation | VCP-CBF [N12] | Resolves relative degree issue for unicycle |
| Training safety | CBF-constrained Beta policy [16] | Samples only safe actions; convergence guarantee |
| Deployment safety | RCBF-QP filter [06] | Robust to model uncertainty via GP |
| Infeasibility handling | 3-tier: learned constraints + relaxation + backup [N13] | CBF infeasibility is HIGH risk in adversarial PE |
| Policy distribution | Beta (not Gaussian) | Naturally bounded; compatible with CBF truncation |
| Obstacles | 2-4 static circular | Enough to test safety; more complex in Phase 3 |
| Safety reward | w5 = 0.05 * min(h_i)/h_max | Incentivize policy to stay away from constraint boundaries |

### 1.4 Dependencies from Phase 1

| Phase 1 Output | How Used in Phase 2 |
|----------------|---------------------|
| PE environment | Extended with obstacles and CBF layer |
| VCP-CBF validation code | Directly reused for multi-constraint scenarios |
| PPO training pipeline | Modified for Beta distribution policy |
| Self-play loop | Extended with safety-constrained training |
| Baseline models | Compared against safe models for ablation |

---

## 2. Background & Theory

### 2.1 Control Barrier Functions (CBFs)

A **Control Barrier Function** h(x) defines a safe set C = {x : h(x) >= 0}. The system remains safe if:

```
h_dot(x, u) + alpha * h(x) >= 0    for all time
```

where alpha > 0 is a class-K function parameter controlling how quickly the system can approach the boundary.

**Intuition**: The CBF constraint says "the rate at which you approach the boundary must be limited by how far you are from it." Far from the boundary (h large), you have freedom. Near the boundary (h small), you must slow down or turn away.

### 2.2 The Relative Degree Problem for Unicycle Robots

For a standard position-based CBF like `h(x) = ||p - p_obs||^2 - r^2`:

```
h_dot = 2*(p - p_obs)^T * p_dot
      = 2*(x - x_obs) * v*cos(theta) + 2*(y - y_obs) * v*sin(theta)
```

**Problem**: h_dot depends on `v` and `theta` but NOT on `omega`. This means the angular velocity control cannot directly affect the CBF constraint — only linear velocity can. The result: **the CBF-QP can only brake, not steer**, which is overly conservative and physically wrong.

### 2.3 VCP-CBF Solution

The **Virtual Control Point (VCP)** approach [N12] places a point at distance `d` ahead of the robot:

```
q = [x + d*cos(theta),  y + d*sin(theta)]

q_dot = [v*cos(theta) - d*omega*sin(theta),
         v*sin(theta) + d*omega*cos(theta)]
```

Now `q_dot` depends on BOTH `v` and `omega`. Using `q` instead of the robot center `p` in the CBF:

```
h(x) = ||q - p_obs||^2 - chi^2

h_dot = 2*(q - p_obs)^T * q_dot
      = a_v * v + a_omega * omega
```

Both `a_v` and `a_omega` are nonzero (generically), giving **uniform relative degree 1** and a well-posed CBF-QP.

**Key insight from [N12]**: The M-matrix transformation in VCP-CBF naturally prioritizes steering over braking, producing more agile and less conservative safety behavior.

### 2.4 CBF-Constrained Beta Policy (Paper [16])

During **training**, we don't use a QP filter. Instead, we modify the policy distribution itself to only output safe actions.

**Standard approach**: Policy outputs Gaussian → sample action → clip to CBF-safe set (but this distorts gradients)

**Paper [16] approach**: Policy outputs Beta distribution → rescale support to CBF-safe set → sample

```
pi_theta^C(u|x) = pi_theta(u|x) / pi_theta(C(x)|x)     for u in C(x)
                 = 0                                       otherwise

C(x) = {u : h_dot_i(x,u) + alpha_i * h_i(x) >= 0, for all i}
```

**Why Beta, not Gaussian**:
- Beta distribution has bounded support [0, 1] (rescalable to [a, b])
- Truncation to safe set is natural (just change bounds)
- No probability mass wasted on out-of-bounds actions
- Convergence guarantee (Theorem 1, Paper [16])

**Implementation**: For each state x, compute the safe action bounds [v_min_safe, v_max_safe] and [omega_min_safe, omega_max_safe] from the CBF constraints. The Beta distribution samples within these bounds.

### 2.5 RCBF-QP Safety Filter (Paper [06])

During **deployment**, model uncertainty exists (wheel slip, terrain, delays). The Robust CBF-QP adds a GP-estimated disturbance:

```
Nominal: x_dot = f(x) + g(x)*u
Actual:  x_dot = f(x) + g(x)*u + d(x)     (d = disturbance)

GP model: d_hat(x), sigma_d(x) = GP.predict(x)

RCBF-QP:
  minimize  ||u - u_RL||^2
  subject to  L_f h + L_g h * u + L_d h * d_hat >= -alpha*h + kappa*sigma_d
              u in U (control bounds)
```

The `kappa * sigma_d` term adds a **robust margin** proportional to the GP uncertainty. More uncertain → more conservative.

### 2.6 Three-Tier Infeasibility Handling

In adversarial PE, the opponent actively pushes you into tight situations. CBF-QP infeasibility is expected.

**Tier 1 — Learned Feasibility Constraints (training, Paper [N13])**:
- Train SVM/DNN classifiers that predict whether a state will make the CBF-QP infeasible
- Add these classifiers as additional constraints during RL training
- Result: the policy learns to proactively avoid infeasible regions
- Paper [N13] reports reduction from 8.11% to 0.21% infeasibility

**Tier 2 — Hierarchical Relaxation (deployment)**:
- If CBF-QP is infeasible, relax the least-important constraint
- Priority: collision > arena > obstacles

**Tier 3 — Backup Controller (last resort)**:
- Pure safety: brake + turn away from nearest danger
- No task performance, only survival

### 2.7 Safety-Reward Shaping

Add a reward term that incentivizes staying away from constraint boundaries:

```
r_safety = w5 * min_i(h_i(x)) / h_max
```

where `w5 = 0.05` (small, to not dominate task reward).

**Effect**: Over training, the CBF filter intervenes less often because the policy learns inherently safe behavior. Paper [05] validates this approach; [N15] independently confirms it.

**Zero-sum subtlety**: Since r_E = -r_P, the evader receives `-w5 * min(h_i)` — actually incentivizing it to push the pursuer toward unsafe regions. This is intentional (creates richer game-theoretic behavior). The evader's own safety is enforced by CBF, not reward.

---

## 3. Relevant Literature

### 3.1 Core Papers for Phase 2

| Paper | Relevance | Key Takeaway |
|-------|-----------|--------------|
| **[16] Suttle 2024** | CBF-constrained Beta policy | Core training safety algorithm; Theorem 1 convergence |
| **[06] Emam 2022** | RCBF-QP + GP | Deployment safety filter; GP disturbance estimation |
| **[N12] Zhang & Yang 2025** | VCP-CBF for nonholonomic | Already validated in Phase 1; now integrated fully |
| **[N13] Xiao et al. 2023** | Learned feasibility constraints | 3-tier infeasibility handling; reduces infeasibility 40x |
| **[05] Yang 2025** | CBF safety filtering + reward shaping | Closed-form CBF filter; safety-reward shaping rationale |
| **[N02] So & Fan 2024** | PNCBF (neural CBF) | **Fallback if VCP-CBF is brittle** |
| **[03] Ames et al.** | CBF theory (foundational) | Theoretical foundation for all CBF work |

### 3.2 Reference Papers

| Paper | Topic | Why Relevant |
|-------|-------|-------------|
| **[N01] Dawson et al. 2023** | Learning CBFs survey | Background for neural CBF alternatives |
| **[N04] Xiao et al. 2023** | BarrierNet | Phase 2.5 preview; differentiable QP layer |
| **[N07] Zhang & Xu 2024** | POLICEd RL | Alternative: model-free hard constraints (limited applicability) |
| **[N08] Ma et al. 2024** | Statewise CBF | Alternative CBF implementation |
| **[30] Zhou 2023** | CASRL conflict-averse gradient | If safety-reward creates gradient conflict |

### 3.3 What to Read vs. Skim

**Read carefully (implementation-critical)**:
- [16]: Full paper — understand Beta policy truncation, convergence theorem, implementation details
- [N12]: VCP-CBF details — M-matrix transformation, VCP parameter selection
- [N13]: Learned feasibility — SVM training, feedback loop with RL
- [05]: CBF filter + reward shaping — closed-form filter, w5 rationale

**Read for deployment layer**:
- [06]: RCBF-QP formulation, GP kernel selection, update frequency

**Skim for alternatives**:
- [N02]: PNCBF — only needed if VCP-CBF fails
- [N04]: BarrierNet — Phase 2.5 reading

---

## 4. Deliverables Checklist

### 4.1 Must-Have Deliverables

- [ ] **D1**: VCP-CBF formulation for arena boundary + circular obstacles + inter-robot collision
- [ ] **D2**: CBF-constrained Beta policy implementation
- [ ] **D3**: RCBF-QP safety filter implementation
- [ ] **D4**: 3-tier infeasibility handling (learned feasibility + relaxation + backup)
- [ ] **D5**: Safety-constrained self-play training
- [ ] **D6**: Safety-reward shaping term (w5) in reward function
- [ ] **D7**: Ablation study: CBF-Beta vs CBF-QP vs no safety

### 4.2 Visualization & Tracking Deliverables

- [ ] **D8**: CBF visualization overlay in `PERenderer` — real-time safe region (green) / danger zone (red) display, obstacle CBF contours, intervention flash indicator
- [ ] **D9**: wandb experiment tracking for all training runs — per-run dashboards with safety metrics, ablation group tags, eval video recording via `RecordVideo`
- [ ] **D10**: Hydra config files for ablation matrix — `conf/experiment/ablation_{A-E}.yaml` with per-config overrides

### 4.3 Monitoring Deliverables

- [ ] **D11**: CBF feasibility rate tracking
- [ ] **D12**: Backup controller activation rate tracking
- [ ] **D13**: CBF filter intervention rate over training
- [ ] **D14**: CBF margin distribution histograms

### 4.4 Documentation Deliverables

- [ ] **D15**: Safety architecture documentation
- [ ] **D16**: Ablation results table
- [ ] **D17**: Comparison vs Phase 1 baselines

---

## 5. Session-wise Implementation Breakdown

### Session 1: Beta Distribution Policy (3-4h + 1h buffer)

**Goal**: Replace Gaussian policy with Beta distribution for CBF compatibility.

**Tasks**:

1. **Understand the Beta distribution for RL**:
   - Beta(alpha, beta) is defined on [0, 1]
   - Rescale to [a, b]: `x = a + (b-a) * Beta_sample`
   - The actor network outputs `alpha, beta > 0` for each action dimension
   - Use softplus to ensure positivity: `alpha = softplus(raw_alpha) + 1`

2. **Custom Beta policy for SB3**:
   ```python
   class BetaDistribution:
       """
       Beta distribution for bounded continuous actions.
       Compatible with CBF truncation (dynamic bounds).
       """
       def __init__(self, action_dim):
           self.action_dim = action_dim

       def proba_distribution(self, alpha, beta, low, high):
           """Create Beta distribution with bounds [low, high]."""
           self.distribution = torch.distributions.Beta(alpha, beta)
           self.low = low
           self.high = high
           return self

       def sample(self):
           x = self.distribution.sample()
           return self.low + (self.high - self.low) * x

       def log_prob(self, actions):
           # Rescale action back to [0, 1]
           x = (actions - self.low) / (self.high - self.low)
           x = torch.clamp(x, 1e-6, 1 - 1e-6)  # Numerical stability
           # Jacobian correction for change of variables: [0,1] -> [low, high]
           # Without this, policy gradients are incorrect when bounds != [0,1]
           return self.distribution.log_prob(x).sum(dim=-1) \
               - torch.log(self.high - self.low).sum(dim=-1)
   ```

3. **Integrate with SB3 or custom PPO**:
   - Option A: Custom SB3 policy class that uses BetaDistribution
   - Option B: Use CleanRL and implement Beta distribution directly
   - **Recommended**: CleanRL for more control over the distribution

4. **Test without CBF first**:
   - Train with Beta policy on Phase 1 environment
   - Verify similar performance to Gaussian policy
   - Bounds: v in [0, v_max], omega in [-omega_max, omega_max]

**Validation**:
- Beta policy trains successfully on Phase 1 PE task
- Performance within 10% of Gaussian policy
- Action samples are within bounds [0, v_max] and [-omega_max, omega_max]
- Log probabilities are computed correctly

**Estimated effort**: 3-4 hours

---

### Session 2: Multi-Constraint VCP-CBF (3-4h + 1h buffer)

**Goal**: Extend Phase 1's VCP-CBF to handle all safety constraints simultaneously.

**Tasks**:

1. **Arena boundary constraint (rectangular)**:
   ```python
   def vcp_cbf_arena(state, arena_bounds, d=0.05, alpha=1.0):
       """
       Four constraints for rectangular arena:
       h1 = q_x - x_min >= 0    (left wall)
       h2 = x_max - q_x >= 0    (right wall)
       h3 = q_y - y_min >= 0    (bottom wall)
       h4 = y_max - q_y >= 0    (top wall)
       """
       x, y, theta = state
       qx = x + d * np.cos(theta)
       qy = y + d * np.sin(theta)

       constraints = []

       # Left wall: h1 = qx - x_min
       h1 = qx - arena_bounds['x_min']
       a_v_1 = np.cos(theta)
       a_omega_1 = -d * np.sin(theta)
       constraints.append((h1, a_v_1, a_omega_1))

       # Right wall: h2 = x_max - qx
       h2 = arena_bounds['x_max'] - qx
       a_v_2 = -np.cos(theta)
       a_omega_2 = d * np.sin(theta)
       constraints.append((h2, a_v_2, a_omega_2))

       # Bottom wall: h3 = qy - y_min
       h3 = qy - arena_bounds['y_min']
       a_v_3 = np.sin(theta)
       a_omega_3 = d * np.cos(theta)
       constraints.append((h3, a_v_3, a_omega_3))

       # Top wall: h4 = y_max - qy
       h4 = arena_bounds['y_max'] - qy
       a_v_4 = -np.sin(theta)
       a_omega_4 = -d * np.cos(theta)
       constraints.append((h4, a_v_4, a_omega_4))

       return constraints
   ```

2. **Obstacle avoidance constraints**:
   ```python
   def vcp_cbf_obstacles(state, obstacles, d=0.05, alpha=1.0):
       """
       For each obstacle i with position p_i and radius chi_i:
       h_i = ||q - p_i||^2 - chi_i^2 >= 0
       """
       constraints = []
       x, y, theta = state
       qx = x + d * np.cos(theta)
       qy = y + d * np.sin(theta)

       for obs in obstacles:
           dx = qx - obs['x']
           dy = qy - obs['y']
           chi = obs['radius'] + robot_radius + safety_margin + d

           h = dx**2 + dy**2 - chi**2
           a_v = 2*dx*np.cos(theta) + 2*dy*np.sin(theta)
           a_omega = 2*dx*(-d*np.sin(theta)) + 2*dy*(d*np.cos(theta))
           constraints.append((h, a_v, a_omega))

       return constraints
   ```

3. **Inter-robot collision constraint**:
   ```python
   def vcp_cbf_collision(state_P, state_E, r_min, d=0.05, alpha=1.0):
       """
       h = ||q_P - q_E||^2 - r_min^2 >= 0
       Note: This is a coupled constraint (depends on both agents).
       For the pursuer, we treat the evader's VCP as moving.
       """
       # Compute VCPs for both agents
       qPx, qPy = compute_vcp(*state_P, d)
       qEx, qEy = compute_vcp(*state_E, d)

       dx = qPx - qEx
       dy = qPy - qEy
       h = dx**2 + dy**2 - r_min**2

       # For pursuer's CBF-QP (treating evader motion as disturbance):
       # h_dot = 2*dx*(qP_dot_x - qE_dot_x) + 2*dy*(qP_dot_y - qE_dot_y)
       # Pursuer controls: [v_P, omega_P]
       # a_v_P = 2*dx*cos(theta_P) + 2*dy*sin(theta_P)
       # a_omega_P = 2*dx*(-d*sin(theta_P)) + 2*dy*(d*cos(theta_P))
       theta_P = state_P[2]
       a_v = 2*dx*np.cos(theta_P) + 2*dy*np.sin(theta_P)
       a_omega = 2*dx*(-d*np.sin(theta_P)) + 2*dy*(d*np.cos(theta_P))

       return h, a_v, a_omega
   ```

4. **Aggregate all constraints**:
   ```python
   def get_all_cbf_constraints(state_P, state_E, arena, obstacles, d, alpha):
       constraints = []
       constraints.extend(vcp_cbf_arena(state_P, arena, d, alpha))
       constraints.extend(vcp_cbf_obstacles(state_P, obstacles, d, alpha))
       h_col, av_col, aw_col = vcp_cbf_collision(state_P, state_E, r_min, d, alpha)
       constraints.append((h_col, av_col, aw_col))
       return constraints
   ```

**Validation**:
- Arena boundary constraints keep robot inside arena
- Obstacle constraints prevent collision (with margin)
- Inter-robot constraint prevents physical collision (r_min < r_capture)
- All constraints are linear in [v, omega] (QP-compatible)
- Constraints combine correctly (no conflicting math)

**Estimated effort**: 3-4 hours

---

### Session 3: CBF-QP Solver & Safe Action Bounds (3-4h + 1h buffer)

**Goal**: Implement the CBF-QP solver and compute safe action bounds for the Beta policy.

**Tasks**:

1. **CBF-QP solver using cvxpy**:
   ```python
   import cvxpy as cp

   def solve_cbf_qp(u_nominal, constraints, v_bounds, omega_bounds):
       """
       minimize  ||u - u_nominal||^2
       s.t.      a_v_i * v + a_omega_i * omega + alpha * h_i >= 0  for all i
                 v_min <= v <= v_max
                 omega_min <= omega <= omega_max
       """
       v = cp.Variable()
       omega = cp.Variable()
       u = cp.vstack([v, omega])
       u_nom = np.array(u_nominal)

       objective = cp.Minimize(cp.sum_squares(u - u_nom))

       cons = []
       for h, a_v, a_omega in constraints:
           cons.append(a_v * v + a_omega * omega + alpha * h >= 0)
       cons.append(v >= v_bounds[0])
       cons.append(v <= v_bounds[1])
       cons.append(omega >= omega_bounds[0])
       cons.append(omega <= omega_bounds[1])

       prob = cp.Problem(objective, cons)
       prob.solve(solver=cp.OSQP, warm_start=True)

       if prob.status == 'infeasible':
           return None, False
       return np.array([v.value, omega.value]), True
   ```

2. **Safe bound computation for Beta policy**:
   ```python
   def compute_safe_bounds(constraints, v_bounds, omega_bounds, alpha):
       """
       Compute the tightest safe box bounds for [v, omega] given CBF constraints.
       For Beta policy: sample within these bounds instead of using QP.

       For each constraint: a_v * v + a_omega * omega >= -alpha * h
       This defines a half-plane in (v, omega) space.

       We compute conservative box bounds by solving 4 LPs (one per bound):
       - v_min:   minimize v   s.t. all CBF constraints, v in v_bounds, omega in omega_bounds
       - v_max:   maximize v   s.t. (same)
       - omega_min: minimize omega s.t. (same)
       - omega_max: maximize omega s.t. (same)

       This is conservative because the true safe set is a polytope, not a box,
       but it is tight along each axis (the box is the smallest axis-aligned box
       containing the safe polytope intersected with the control bounds).
       """
       from scipy.optimize import linprog

       v_lo, v_hi = v_bounds
       omega_lo, omega_hi = omega_bounds

       # Build constraint matrix: A_ub @ x <= b_ub  (linprog uses <= form)
       # Each CBF constraint: a_v * v + a_omega * omega >= -alpha * h
       # Negate for <= form:  -a_v * v - a_omega * omega <= alpha * h
       n_cbf = len(constraints)
       A_ub = np.zeros((n_cbf, 2))
       b_ub = np.zeros(n_cbf)
       for i, (h, a_v, a_omega) in enumerate(constraints):
           A_ub[i, 0] = -a_v
           A_ub[i, 1] = -a_omega
           b_ub[i] = alpha * h

       bounds_lp = [(v_lo, v_hi), (omega_lo, omega_hi)]

       # Solve 4 LPs for tightest box bounds
       # v_min: minimize v  =>  c = [1, 0]
       res = linprog([1, 0], A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
       safe_v_min = res.x[0] if res.success else v_lo

       # v_max: maximize v  =>  minimize -v  =>  c = [-1, 0]
       res = linprog([-1, 0], A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
       safe_v_max = res.x[0] if res.success else v_hi

       # omega_min: minimize omega  =>  c = [0, 1]
       res = linprog([0, 1], A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
       safe_omega_min = res.x[1] if res.success else omega_lo

       # omega_max: maximize omega  =>  minimize -omega  =>  c = [0, -1]
       res = linprog([0, -1], A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
       safe_omega_max = res.x[1] if res.success else omega_hi

       # Ensure bounds are valid (min <= max)
       if safe_v_min > safe_v_max or safe_omega_min > safe_omega_max:
           # Infeasible: no safe box exists. Return nominal bounds
           # and let the 3-tier infeasibility handler deal with it.
           return v_bounds, omega_bounds

       return (safe_v_min, safe_v_max), (safe_omega_min, safe_omega_max)
   ```

   **Note**: Computing exact safe bounds for a Beta policy from linear constraints requires solving linear programs. For each bound (v_min, v_max, omega_min, omega_max), solve an LP to find the tightest bound consistent with all constraints. This is fast for small numbers of constraints.

3. **Analytical alternative (faster, no LP solver needed)**:
   ```python
   def compute_safe_bounds_analytical(constraints, v_bounds, omega_bounds, alpha):
       """
       Fast analytical bound computation for the common case of few constraints.
       For each constraint: a_v * v + a_omega * omega >= -alpha * h

       For v_min: for each constraint, compute the tightest lower bound on v
       by assuming omega takes its most helpful value (the one that maximizes
       the LHS of the constraint, leaving the most room for v to be small).
       """
       v_lo, v_hi = v_bounds
       omega_lo, omega_hi = omega_bounds

       for h, a_v, a_omega in constraints:
           rhs = -alpha * h  # constraint: a_v * v + a_omega * omega >= rhs

           if abs(a_v) < 1e-10:
               # Constraint does not involve v; check omega feasibility only
               continue

           # Best-case omega for minimizing v: pick omega that maximizes a_omega*omega
           best_omega = omega_hi if a_omega >= 0 else omega_lo
           slack = a_omega * best_omega

           # a_v * v >= rhs - slack  =>  v >= (rhs - slack) / a_v  (if a_v > 0)
           #                          =>  v <= (rhs - slack) / a_v  (if a_v < 0)
           bound = (rhs - slack) / a_v
           if a_v > 0:
               v_lo = max(v_lo, bound)
           else:
               v_hi = min(v_hi, bound)

       # Repeat for omega bounds (swap roles)
       for h, a_v, a_omega in constraints:
           rhs = -alpha * h
           if abs(a_omega) < 1e-10:
               continue
           best_v = v_hi if a_v >= 0 else v_lo
           slack = a_v * best_v
           bound = (rhs - slack) / a_omega
           if a_omega > 0:
               omega_lo = max(omega_lo, bound)
           else:
               omega_hi = min(omega_hi, bound)

       if v_lo > v_hi or omega_lo > omega_hi:
           return v_bounds, omega_bounds  # Infeasible: fallback to nominal

       return (v_lo, v_hi), (omega_lo, omega_hi)
   ```
   **Note**: The analytical version is more conservative than the LP version because
   it uses the best-case omega for EACH constraint independently, rather than finding
   a single omega that satisfies ALL constraints simultaneously. In practice, with
   ~5-8 constraints, the difference is small. Use the LP version for accuracy, the
   analytical version for speed (<0.01ms vs ~0.1ms).

**Validation**:
- CBF-QP solver returns feasible solutions for normal states
- CBF-QP correctly identifies infeasible states
- Safe bounds are tighter than original bounds when near obstacles
- Safe bounds equal original bounds when far from all obstacles
- QP solve time < 5ms (sufficient for training; deployment needs <50ms)

**Estimated effort**: 3-4 hours

---

### Session 4: CBF-Beta Policy Integration (4-5h + 1h buffer)

**Goal**: Combine Beta policy with CBF safe bounds for safe training.

**Tasks**:

1. **Modify PPO actor to use dynamic bounds**:
   ```python
   class SafeBetaActor(nn.Module):
       def __init__(self, obs_dim, action_dim, hidden=[256, 256]):
           super().__init__()
           self.net = build_mlp(obs_dim, hidden)
           self.alpha_head = nn.Linear(hidden[-1], action_dim)
           self.beta_head = nn.Linear(hidden[-1], action_dim)

       def forward(self, obs, safe_bounds):
           """
           safe_bounds: [(v_min, v_max), (omega_min, omega_max)]
           """
           features = self.net(obs)
           alpha = F.softplus(self.alpha_head(features)) + 1.0
           beta = F.softplus(self.beta_head(features)) + 1.0

           # Create Beta distribution with safe bounds
           dist = BetaDistribution(alpha, beta, safe_bounds)
           return dist
   ```

2. **Modified PPO rollout with CBF**:
   ```python
   def collect_rollout_safe(env, actor, critic, cbf, n_steps):
       buffer = RolloutBuffer()

       for step in range(n_steps):
           # Get CBF safe bounds for current state
           state = env.get_state()
           constraints = cbf.get_constraints(state)
           safe_bounds = cbf.compute_safe_bounds(constraints)

           # Sample safe action from Beta policy
           obs = env.get_obs()
           dist = actor(obs, safe_bounds)
           action = dist.sample()
           log_prob = dist.log_prob(action)
           value = critic(obs)

           # Step environment
           next_obs, reward, done, truncated, info = env.step(action)

           # Store in buffer
           buffer.add(obs, action, reward, done, log_prob, value, safe_bounds)

       return buffer
   ```

3. **Modified PPO update with safe log-probs**:
   ```python
   def ppo_update_safe(actor, critic, optimizer, buffer):
       for epoch in range(n_epochs):
           for batch in buffer.get_batches(batch_size):
               # Recompute log-probs with stored safe bounds
               dist = actor(batch.obs, batch.safe_bounds)
               new_log_prob = dist.log_prob(batch.actions)

               # Standard PPO loss (with safe distribution)
               ratio = (new_log_prob - batch.old_log_prob).exp()
               clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
               policy_loss = -torch.min(ratio * batch.advantages,
                                         clipped * batch.advantages).mean()

               # Critic loss
               value_loss = F.mse_loss(critic(batch.obs), batch.returns)

               # Entropy (of truncated distribution)
               entropy = dist.entropy().mean()

               loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
   ```

**Validation**:
- Safe training produces **zero safety violations** (h_i >= 0 for all i at all steps)
- Training converges (reward improves over time)
- Task performance within 10% of unconstrained Phase 1 baseline
- CBF filter intervention rate decreases over training epochs

**Estimated effort**: 4-5 hours

---

### Session 5: Obstacles & Environment Extension (2-3h + 1h buffer)

**Goal**: Add static obstacles to the PE environment.

**Tasks**:

1. **Add obstacles to environment**:
   ```python
   class PursuitEvasionEnv:
       def __init__(self, ..., n_obstacles=4):
           self.obstacles = self._generate_obstacles(n_obstacles)

       def _generate_obstacles(self, n):
           """Generate n non-overlapping circular obstacles."""
           obstacles = []
           for _ in range(n):
               while True:
                   obs = {
                       'x': np.random.uniform(x_min + margin, x_max - margin),
                       'y': np.random.uniform(y_min + margin, y_max - margin),
                       'radius': np.random.uniform(0.3, 1.0),
                   }
                   if not self._overlaps(obs, obstacles):
                       obstacles.append(obs)
                       break
           return obstacles

       def _check_obstacle_collision(self, state):
           """Check if robot collides with any obstacle."""
           x, y = state[0], state[1]
           for obs in self.obstacles:
               dist = np.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
               if dist < obs['radius'] + self.r_robot:
                   return True
           return False
   ```

2. **Integrate CBF constraints with environment**:
   ```python
   def step(self, action_P, action_E):
       # Compute CBF-safe actions for both agents
       constraints_P = self.cbf.get_constraints(self.state_P, self.state_E)
       constraints_E = self.cbf.get_constraints(self.state_E, self.state_P)

       safe_action_P = self.cbf.filter(action_P, constraints_P)
       safe_action_E = self.cbf.filter(action_E, constraints_E)

       # Step dynamics with safe actions
       self.state_P = self.dynamics.step(self.state_P, safe_action_P)
       self.state_E = self.dynamics.step(self.state_E, safe_action_E)

       # Check safety (should be guaranteed by CBF)
       assert not self._check_obstacle_collision(self.state_P)
       assert not self._check_obstacle_collision(self.state_E)
   ```

3. **Update observation space** (add obstacle information):
   ```python
   # Phase 2 observation (still full state):
   obs_P = [
       # ... existing state info ...
       obstacle_distances,    # distances to nearest K obstacles
       obstacle_bearings,     # bearings to nearest K obstacles
   ]
   ```

4. **Update reward with safety term**:
   ```python
   # Compute minimum CBF margin
   h_values = [h for h, _, _ in constraints]
   h_min = min(h_values) if h_values else h_max
   r_safety = 0.05 * h_min / h_max

   r_P = r_distance + r_capture + r_timeout + r_safety
   r_E = -r_P  # Note: zero-sum makes evader get -r_safety
   ```

5. **Activate CBF overlay in PERenderer** (from Phase 1 stub):
   ```python
   # In PERenderer._draw_cbf_overlay() — replace Phase 1 stub with active rendering
   def _draw_cbf_overlay(self, canvas, env_state):
       """Draw CBF safe/danger regions around obstacles and arena boundaries."""
       if env_state.get('cbf_margins') is None:
           return  # Phase 1 mode: no CBF active

       # Draw obstacle CBF contours (h_i = 0 boundary)
       for obs in env_state.get('obstacles', []):
           center = self._world_to_pixel(obs['x'], obs['y'])
           # Danger zone: obstacle radius + safety_margin (red, semi-transparent)
           danger_r = int((obs['radius'] + self.config['safety_margin']) * self.scale)
           danger_surf = pygame.Surface((danger_r*2, danger_r*2), pygame.SRCALPHA)
           pygame.draw.circle(danger_surf, (200, 0, 0, 60), (danger_r, danger_r), danger_r)
           canvas.blit(danger_surf, (center[0]-danger_r, center[1]-danger_r))

       # Flash intervention indicator when CBF modifies action
       if env_state.get('cbf_intervened', False):
           flash_surf = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
           flash_surf.fill((255, 100, 0, 30))  # orange flash
           canvas.blit(flash_surf, (0, 0))

   # Update HUD to show CBF metrics
   def _draw_hud(self, canvas, env_state):
       # ... existing HUD items ...
       cbf_margin = env_state.get('min_cbf_margin', None)
       if cbf_margin is not None:
           color = (0, 200, 0) if cbf_margin > 0.3 else (255, 165, 0) if cbf_margin > 0.1 else (255, 0, 0)
           self._draw_text(canvas, f"CBF: {cbf_margin:.2f}", (self.hud_margin, 70), color)
       feasibility = env_state.get('cbf_feasible', True)
       if not feasibility:
           self._draw_text(canvas, "INFEASIBLE", (self.hud_margin, 90), (255, 0, 0))
   ```

   **env_state contract** (env must populate in `step()`):
   ```python
   env_state = {
       'pursuer_pos': self.state_P,
       'evader_pos': self.state_E,
       'obstacles': self.obstacles,
       'cbf_margins': h_values,           # list of CBF margin values
       'min_cbf_margin': min(h_values),
       'cbf_intervened': (action != safe_action).any(),
       'cbf_feasible': feasible,
       'step': self.current_step,
       'episode_reward': self.cumulative_reward,
   }
   ```

**Validation**:
- Obstacles spawn correctly (no overlaps, within arena)
- Observation space includes obstacle info
- CBF prevents all obstacle collisions
- Reward includes safety shaping term
- CBF overlay renders: danger zones visible around obstacles (visual check)
- Intervention flash triggers when CBF modifies action (visual check)
- HUD shows real-time CBF margin value with color coding

**Estimated effort**: 2-3h + 1h buffer (CBF overlay activation)

---

### Session 6: 3-Tier Infeasibility Handling (4-5h + 1h buffer)

**Goal**: Implement the complete infeasibility handling system.

**Tasks**:

1. **Tier 1 — Learned Feasibility Constraints [N13]**:
   ```python
   class FeasibilityClassifier:
       """
       SVM/DNN that predicts whether a state will make CBF-QP infeasible.
       Trained iteratively alongside the RL policy.
       """
       def __init__(self):
           self.model = None
           self.feasible_data = []
           self.infeasible_data = []

       def collect_data(self, state, is_feasible):
           if is_feasible:
               self.feasible_data.append(state)
           else:
               self.infeasible_data.append(state)

       def train(self):
           """Retrain classifier on collected data."""
           X = np.vstack([self.feasible_data, self.infeasible_data])
           y = np.array([1]*len(self.feasible_data) + [0]*len(self.infeasible_data))
           self.model = SVM(kernel='rbf').fit(X, y)

       def is_feasible(self, state):
           if self.model is None:
               return True  # Optimistic before any data
           return self.model.predict([state])[0] == 1

       def get_feasibility_margin(self, state):
           """Return distance to feasibility boundary (CBF-like)."""
           if self.model is None:
               return 1.0
           return self.model.decision_function([state])[0]
   ```

2. **Tier 2 — Hierarchical Relaxation**:
   ```python
   def solve_cbf_qp_with_relaxation(u_nominal, constraints, bounds):
       """
       Try exact QP first, then relax constraints by priority.
       Priority: collision > arena > obstacles
       """
       # Tier 1: Try exact QP
       result, feasible = solve_cbf_qp(u_nominal, constraints, *bounds)
       if feasible:
           return result, 'exact'

       # Tier 2: Relax obstacle constraints (lowest priority)
       collision_cons = [c for c in constraints if c.type == 'collision']
       arena_cons = [c for c in constraints if c.type == 'arena']
       obs_cons = [c for c in constraints if c.type == 'obstacle']

       # Try without obstacle constraints
       reduced = collision_cons + arena_cons
       result, feasible = solve_cbf_qp(u_nominal, reduced, *bounds)
       if feasible:
           return result, 'relaxed_obstacles'

       # Try with only collision constraints
       result, feasible = solve_cbf_qp(u_nominal, collision_cons, *bounds)
       if feasible:
           return result, 'relaxed_arena'

       return None, 'infeasible'
   ```

3. **Tier 3 — Backup Controller**:
   ```python
   def backup_controller(state, obstacles, arena):
       """
       Emergency controller: brake + turn away from nearest danger.
       No task performance, only survival.
       """
       # Find nearest danger (obstacle or boundary)
       nearest_danger_dir = find_nearest_danger_direction(state, obstacles, arena)

       # Turn away from danger
       away_dir = nearest_danger_dir + np.pi
       heading_error = wrap_angle(away_dir - state[2])

       omega = np.clip(3.0 * heading_error, -omega_max, omega_max)
       v = 0.1 * v_max  # Very slow forward motion

       return np.array([v, omega])
   ```

4. **Integrate into training loop**:
   ```python
   def safe_step(self, action, state, constraints):
       # Try CBF-QP
       safe_action, method = solve_cbf_qp_with_relaxation(
           action, constraints, self.bounds)

       if safe_action is None:
           # Tier 3: Backup controller
           safe_action = backup_controller(state, self.obstacles, self.arena)
           method = 'backup'

       # Log
       self.metrics['cbf_method'].append(method)
       return safe_action
   ```

5. **Iterative feasibility training (N13 protocol)**:
   ```python
   def train_with_feasibility(env, config):
       classifier = FeasibilityClassifier()

       for iteration in range(3):  # N13: converges in 3 iterations
           # Train RL policy with current feasibility constraints
           policy = train_ppo_safe(env, feasibility_classifier=classifier)

           # Collect infeasible states from rollouts
           for episode in range(1000):
               run_episode(policy, env, classifier)

           # Retrain classifier
           classifier.train()

           # Report infeasibility rate
           rate = len(classifier.infeasible_data) / total_steps
           print(f"Iteration {iteration}: Infeasibility rate = {rate:.4f}")
   ```

**Validation**:
- Tier 1 classifier reduces infeasibility rate over iterations
- Tier 2 relaxation handles cases Tier 1 misses
- Tier 3 backup controller never violates safety
- Combined system: CBF-QP feasibility rate > 99%
- Backup controller activation rate < 1%

**Estimated effort**: 4-5 hours

---

### Session 7: Safety-Constrained Self-Play Training (4-5h + 1h buffer)

**Goal**: Train pursuer and evader with safety in alternating self-play.

**Tasks**:

1. **Full safe self-play training loop**:
   - Both agents use CBF-Beta policy
   - Both agents have independent CBF constraints
   - Safety metrics tracked per phase

2. **Training configuration**:
   ```python
   safe_training_config = {
       # Self-play
       'n_phases': 10,
       'timesteps_per_phase': 300_000,  # Slightly more than Phase 1 (safety overhead)

       # CBF
       'cbf_alpha': 1.0,
       'vcp_d': 0.05,
       'safety_margin': 0.1,

       # Safety reward
       'w5': 0.05,

       # Feasibility
       'n_feasibility_iterations': 3,
       'feasibility_data_episodes': 1000,

       # Obstacles
       'n_obstacles': 4,
       'obstacle_radius_range': [0.3, 1.0],
   }
   ```

3. **Monitor safety metrics during training**:
   ```python
   safety_metrics = {
       'safety_violations': 0,           # Must be 0
       'cbf_feasibility_rate': [],        # Per phase, target >99%
       'cbf_intervention_rate': [],       # Should decrease over training
       'backup_activation_rate': [],      # Target <1%
       'min_cbf_margin': [],              # Distribution of min h_i values
       'mean_cbf_margin': [],             # Should increase over training
   }
   ```

4. **Comparison runs** (each logged to wandb with group tags):
   - **Run A**: Safe self-play (CBF-Beta + obstacles + w5)
   - **Run B**: Unsafe self-play (no CBF, from Phase 1)
   - **Run C**: CBF-QP filter only (no Beta policy — QP clips actions post-hoc)
   - **Run D**: Safe self-play without w5 (safety reward ablation)

5. **wandb experiment tracking for safety training**:
   ```python
   import wandb

   for run_label, config in [("A_full_safe", config_A), ("B_unsafe", config_B), ...]:
       wandb.init(
           project="pursuit-evasion",
           group="phase2-safety-selfplay",
           name=f"safety-{run_label}-seed{seed}",
           tags=["phase2", "safety", run_label],
           config={**config, "seed": seed},
           sync_tensorboard=True,
       )

       # Custom SafetyMetricsCallback logs per-step CBF data
       class SafetyMetricsCallback(BaseCallback):
           def _on_step(self):
               info = self.locals.get("infos", [{}])[0]
               if self.n_calls % 1024 == 0:
                   wandb.log({
                       "safety/cbf_intervention_rate": info.get("cbf_intervention_rate", 0),
                       "safety/min_cbf_margin": info.get("min_cbf_margin", 0),
                       "safety/feasibility_rate": info.get("cbf_feasibility_rate", 1.0),
                       "safety/backup_activations": info.get("backup_activations", 0),
                       "safety/violations": info.get("safety_violations", 0),
                   }, step=self.num_timesteps)
               return True

       # Record eval videos every 50 episodes
       eval_env = RecordVideo(eval_env, f"videos/phase2/{run_label}/",
                              episode_trigger=lambda ep: ep % 50 == 0)

       wandb.finish()
   ```

**Validation**:
- **Zero safety violations in Run A** (critical)
- Run A task performance within 10% of Run B
- CBF intervention rate decreases over training (agent learns safe behavior)
- Run A vs Run C: Beta policy better than QP filtering
- Run A vs Run D: w5 reduces CBF intervention rate
- All 4 runs logged to wandb with correct group/tags
- Eval videos recorded and visible in wandb dashboard

**Estimated effort**: 4-5h + 1h buffer

---

### Session 8: Ablation Study & Results (3-4h + 1h buffer)

**Goal**: Complete ablation study, compile results, prepare for Phase 2.5.

**Tasks**:

1. **Run all ablation combinations** (5 seeds each):

   | Config | CBF-Beta | CBF-QP | w5 | N13 Feasibility | Description |
   |--------|----------|--------|-----|-----------------|-------------|
   | A | Yes | No | Yes | Yes | Full safe training |
   | B | No | No | No | No | Unsafe baseline (Phase 1) |
   | C | No | Yes | No | No | QP filter only |
   | D | Yes | No | No | Yes | No safety reward |
   | E | Yes | No | Yes | No | No learned feasibility |

2. **Compile results table**:

   | Config | Safety Violations | Capture Rate | CBF Feasibility | CBF Intervention | Notes |
   |--------|------------------|--------------|-----------------|------------------|-------|
   | A | 0 | ? | >99% | Low | Full system |
   | B | Many | ? | N/A | N/A | Unsafe |
   | C | ? | ? | ? | High | QP clipping |
   | D | 0 | ? | >99% | Higher than A | No w5 |
   | E | 0 | ? | <99% | Similar to A | No N13 |

3. **Hydra ablation configs** (one YAML per config):
   ```yaml
   # conf/experiment/ablation_A.yaml — Full safe training
   # @package _global_
   defaults:
     - /env: pursuit_evasion
     - /algorithm: ppo
     - /safety: cbf
     - /wandb: default

   experiment:
     name: "ablation_A_full_safe"
     tags: ["phase2", "ablation", "config_A"]
   safety:
     cbf_beta: true
     cbf_qp: false
     safety_reward_w5: 0.05
     n13_feasibility: true
   ```

   Create `ablation_{A-E}.yaml` for each config. Run all with:
   ```bash
   # Run full ablation matrix: 5 configs x 5 seeds
   python scripts/train.py --multirun \
     experiment=ablation_A,ablation_B,ablation_C,ablation_D,ablation_E \
     seed=0,1,2
   ```

4. **wandb ablation tracking**:
   ```python
   # Each ablation run auto-logged via Hydra + wandb integration
   # wandb group = "phase2-ablation", name = f"{config_name}-seed{seed}"
   # After all runs, generate comparison table:
   import wandb
   api = wandb.Api()
   runs = api.runs("pursuit-evasion", filters={"group": "phase2-ablation"})
   comparison = wandb.Table(
       columns=["Config", "Seed", "Safety Violations", "Capture Rate",
                "CBF Feasibility", "CBF Intervention Rate"],
       data=[[r.config["experiment"]["name"], r.config["seed"],
              r.summary.get("safety/violations", 0),
              r.summary.get("pe/capture_rate", 0),
              r.summary.get("safety/feasibility_rate", 0),
              r.summary.get("safety/cbf_intervention_rate", 0)]
             for r in runs]
   )
   wandb.log({"ablation_comparison": comparison})
   ```

5. **Generate plots** (from wandb data or TensorBoard):
   - Learning curves (reward vs training steps) for all configs
   - Safety violation rate over training
   - CBF intervention rate over training (should decrease for A, constant for C)
   - CBF margin distribution histograms
   - Capture rate vs self-play phase

6. **Write summary**:
   - Key finding: "CBF-Beta with safety reward achieves zero violations with <10% performance loss"
   - Ablation findings for each component
   - Recommendation for Phase 2.5 comparison

**Validation**:
- All ablation runs complete without errors (25 runs in wandb)
- Results are reproducible (5 seeds)
- wandb comparison table populated with actual numbers
- Hydra configs produce correct overrides (verify with `--cfg job`)
- Clear story: safety comes at modest performance cost
- Phase 2 success criteria all met

**Estimated effort**: 3-4h + 1h buffer

---

## 6. Technical Specifications

### 6.1 CBF Parameters

```python
cbf_config = {
    # VCP
    'vcp_d': 0.05,                    # meters (virtual control point offset)

    # CBF class-K function
    'alpha': 1.0,                      # CBF decay rate

    # Safety margins
    'arena_margin': 0.2,              # meters (additional buffer from arena walls)
    'obstacle_margin': 0.1,           # meters (additional buffer from obstacles)
    'collision_margin': 0.05,         # meters (additional buffer between robots)

    # Constraint radii
    'r_robot': 0.15,                  # meters
    'r_capture': 0.5,                 # meters (capture radius > collision radius)
    'r_collision': 0.3,              # meters (physical collision radius)
    'r_min_separation': 0.35,        # r_collision + collision_margin

    # QP solver
    'qp_solver': 'OSQP',
    'qp_warm_start': True,
    'qp_max_iter': 200,
}
```

### 6.2 Safety Reward Parameters

```python
safety_reward_config = {
    'w5': 0.05,                       # Safety reward weight (small)
    'h_max': None,                    # Computed at env init (max h over grid)
}
```

### 6.3 Feasibility Classifier Parameters

```python
feasibility_config = {
    'classifier_type': 'svm',         # SVM or DNN
    'svm_kernel': 'rbf',
    'svm_C': 1.0,
    'n_iterations': 3,                # N13: converges in 3
    'data_collection_episodes': 1000,
}
```

### 6.4 CBF Visualization Parameters

```python
cbf_viz_config = {
    # CBF overlay colors (RGBA, A=alpha for transparency)
    'color_cbf_safe': (0, 200, 0, 40),       # green: h_i > safety_margin
    'color_cbf_danger': (200, 0, 0, 60),      # red: h_i < safety_margin
    'color_intervention_flash': (255, 100, 0, 30),  # orange: CBF modified action

    # HUD CBF display
    'cbf_margin_green': 0.3,   # h_min > 0.3 → green text
    'cbf_margin_yellow': 0.1,  # h_min > 0.1 → yellow text, else red

    # Video recording for safety evaluation
    'record_safety_episodes': True,  # always record when CBF intervenes
    'safety_video_folder': 'videos/phase2/safety/',
}
```

### 6.5 Experiment Tracking Configuration

```yaml
# conf/wandb/phase2.yaml — extends Phase 1 default
wandb:
  project: "pursuit-evasion"
  tags: ["phase2", "safety"]
  log_frequency: 1024
  safety_metrics:                   # Phase 2 additions
    - safety/cbf_intervention_rate
    - safety/min_cbf_margin
    - safety/feasibility_rate
    - safety/backup_activations
    - safety/violations
  ablation:
    group: "phase2-ablation"
    configs: [A, B, C, D, E]
    seeds: [0, 1, 2, 3, 4]
```

---

## 7. Validation & Success Criteria

### 7.1 Must-Pass Criteria (Gate to Phase 2.5)

| Criterion | Target | How to Measure | Protocol |
|-----------|--------|---------------|----------|
| Zero safety violations | 0 violations across ALL CBF-Beta training runs | `assert h_i(x) >= -1e-6` every timestep; log violations to W&B | Run 5 seeds x 300K steps each; any single violation = FAIL |
| Task performance preserved | Capture rate within 10% of Phase 1 unconstrained baseline (e.g., if Phase 1 = 82%, Phase 2 >= 72%) | Compare mean capture rate over 100 eval episodes, 5 seeds | Use identical eval config (same initial positions via fixed eval seed) |
| CBF-QP feasibility rate | > 99% after N13 iterative training (3 iterations) | Count infeasible QP solves / total QP solves per training run | Report per-iteration improvement: expect ~8% -> ~2% -> <1% |
| Backup controller activation | < 1% of total timesteps | Count Tier 3 activations / total timesteps | If >1%, increase N13 data collection episodes or tune SVM C |
| CBF intervention decreasing | Intervention rate in last 50K steps < 50% of first 50K steps | `intervention = (action != safe_action).any()` per step; moving average | Plot intervention rate vs training step; should show clear downtrend |
| Ablation complete | 5 configs x 5 seeds = 25 runs | All runs converge (reward plateau) or reach 300K step budget | Log to W&B with group tags for easy comparison |
| Beta policy matches Gaussian | Beta policy (no CBF) within 5% of Gaussian policy on Phase 1 task | Train both on identical Phase 1 env; compare capture rate at convergence | Establishes Beta distribution is not the bottleneck |
| CBF visualization works | CBF overlay renders danger zones + intervention flash in `PERenderer` | `render_mode="human"` shows CBF margins; `render_mode="rgb_array"` produces valid video | Visual check + `RecordVideo` produces .mp4 with CBF overlay |
| wandb tracking operational | All training runs logged with safety metrics and group tags | wandb dashboard shows `safety/*` metrics for all 15 ablation runs | Verify `wandb.run` is not None; check logged keys |

### 7.2 Quality Metrics

| Metric | Expected Range | How to Measure | Notes |
|--------|---------------|---------------|-------|
| Training time overhead | 2-5x vs unconstrained | Wall-clock time per 1000 steps (Phase 2 / Phase 1) | Profile with `line_profiler` if >5x |
| QP solve time (mean) | < 5ms | `time.perf_counter()` around `prob.solve()` | OSQP with warm start; log 95th percentile too |
| QP solve time (95th pctl) | < 10ms | Same profiling | If >10ms, consider closed-form filter [05] for training |
| Safety margin distribution | 90th percentile of min(h_i) > 0.5m | Histogram of per-step min(h_i) values | Healthy = most mass far from zero |
| N13 infeasibility reduction | 8%+ -> <1% over 3 iterations | Per-iteration infeasibility rate | Track classifier accuracy too |
| w5 effect on intervention | >20% reduction in CBF interventions | Compare Config A (with w5) vs Config D (without w5) | Key ablation finding |

### 7.3 Definition of Done

> **Phase 2 is COMPLETE when:**
> 1. Deliverables D1-D10 are implemented and tested
> 2. ALL must-pass criteria in Section 7.1 are met (5 seeds each)
> 3. Ablation results table (Section 5, Session 8) is filled with actual numbers
> 4. Minimum test suite (Section 7.4) passes: 18+ tests, all green
> 5. Safety metrics are logged to wandb with correct group tags; CBF visualization active in PERenderer
> 6. Hydra ablation configs (`conf/experiment/ablation_{A-E}.yaml`) produce correct runs
> 7. Phase 2 summary document written with key findings and recommendation for Phase 2.5

### 7.4 Minimum Test Suite (18+ Tests)

All tests are **must-pass** gate criteria for Phase 2 completion.

**File: `tests/test_cbf.py`** (8 tests)

```python
# Test A: VCP-CBF arena constraint produces correct h and gradient signs
def test_vcp_cbf_arena_values():
    """Robot near right wall (x=9.5 in 10m arena) should have small positive h2."""
    state = [9.5, 5.0, 0.0]  # Near right wall, facing right
    constraints = vcp_cbf_arena(state, {'x_min': 0, 'x_max': 10, 'y_min': 0, 'y_max': 10})
    h_right_wall = constraints[1][0]  # h2 = x_max - qx
    assert 0 < h_right_wall < 1.0  # Small positive (near wall)
    assert constraints[1][1] < 0  # a_v < 0 (moving right increases violation)

# Test B: VCP-CBF obstacle constraint is positive when far, negative when inside
def test_vcp_cbf_obstacle_far_and_near():
    """Far from obstacle: h > 0. Inside obstacle safety radius: h < 0."""
    obs = {'x': 5.0, 'y': 5.0, 'radius': 1.0}
    far_state = [0.0, 0.0, 0.0]   # 7.07m away
    near_state = [4.5, 5.0, 0.0]  # 0.5m away (inside chi)
    h_far = vcp_cbf_obstacles(far_state, [obs])[0][0]
    h_near = vcp_cbf_obstacles(near_state, [obs])[0][0]
    assert h_far > 0
    assert h_near < 0

# Test C: CBF-QP returns feasible solution for normal state
def test_cbf_qp_feasible_normal():
    """Robot in center of arena with no obstacles: QP should be trivially feasible."""
    state = [5.0, 5.0, 0.0]
    constraints = vcp_cbf_arena(state, arena_bounds)
    u_safe, feasible = solve_cbf_qp([0.5, 0.0], constraints, (0, 1.0), (-2.84, 2.84))
    assert feasible
    assert 0 <= u_safe[0] <= 1.0
    assert -2.84 <= u_safe[1] <= 2.84

# Test D: CBF-QP modifies action near boundary
def test_cbf_qp_modifies_unsafe_action():
    """Robot near wall heading toward it: QP should modify the action."""
    state = [9.8, 5.0, 0.0]  # Near right wall, heading right
    u_nominal = [1.0, 0.0]   # Full speed toward wall
    constraints = vcp_cbf_arena(state, arena_bounds)
    u_safe, feasible = solve_cbf_qp(u_nominal, constraints, (0, 1.0), (-2.84, 2.84))
    assert feasible
    assert u_safe[0] < u_nominal[0] or abs(u_safe[1]) > 0.1  # Slowed or steered

# Test E: Inter-robot collision CBF is symmetric-aware
def test_inter_robot_cbf():
    """Two robots approaching: h should decrease; CBF should prevent collision."""
    state_P = [4.0, 5.0, 0.0]   # Pursuer heading right
    state_E = [6.0, 5.0, np.pi]  # Evader heading left (toward pursuer)
    h, a_v, a_omega = vcp_cbf_collision(state_P, state_E, r_min=0.35)
    assert h > 0  # Currently separated (2m apart, r_min=0.35)
    assert a_v < 0  # Moving toward evader increases violation

# Test F: Multi-constraint QP handles arena + obstacle + collision simultaneously
def test_multi_constraint_qp():
    """All three constraint types active: QP should find safe action."""
    state_P = [8.0, 5.0, 0.0]
    state_E = [9.0, 5.0, np.pi]
    obs = [{'x': 8.5, 'y': 6.0, 'radius': 0.5}]
    all_cons = get_all_cbf_constraints(state_P, state_E, arena, obs, d=0.05, alpha=1.0)
    u_safe, feasible = solve_cbf_qp([0.5, 0.0], all_cons, v_bounds, omega_bounds)
    assert feasible  # Should find a solution even in tight space

# Test G: QP infeasibility triggers Tier 2 relaxation
def test_tier2_relaxation():
    """Create intentionally infeasible scenario; verify relaxation activates."""
    # Robot boxed in: near wall, near obstacle, near opponent
    state = [9.9, 5.0, 0.0]  # Very near wall
    constraints = create_tight_constraints(state)  # All constraints tight
    u_safe, method = solve_cbf_qp_with_relaxation([0.5, 0.0], constraints, bounds)
    assert method in ['relaxed_obstacles', 'relaxed_arena', 'backup']

# Test H: Backup controller produces valid bounded action
def test_backup_controller_bounds():
    """Backup controller should produce action within control bounds."""
    state = [9.9, 5.0, 0.0]
    u_backup = backup_controller(state, obstacles, arena)
    assert 0 <= u_backup[0] <= v_max
    assert -omega_max <= u_backup[1] <= omega_max
```

**File: `tests/test_beta_policy.py`** (5 tests)

```python
# Test I: Beta distribution samples within specified bounds
def test_beta_samples_in_bounds():
    """All samples from Beta policy must be within [low, high]."""
    dist = BetaDistribution(action_dim=2)
    dist.proba_distribution(alpha=torch.tensor([2.0, 2.0]),
                            beta=torch.tensor([2.0, 2.0]),
                            low=torch.tensor([0.0, -2.84]),
                            high=torch.tensor([1.0, 2.84]))
    samples = torch.stack([dist.sample() for _ in range(1000)])
    assert (samples[:, 0] >= 0.0).all() and (samples[:, 0] <= 1.0).all()
    assert (samples[:, 1] >= -2.84).all() and (samples[:, 1] <= 2.84).all()

# Test J: Log probability is finite for valid actions
def test_beta_log_prob_finite():
    """Log prob should be finite for actions within bounds."""
    # ... setup distribution ...
    actions = torch.tensor([[0.5, 0.0], [0.1, -1.0], [0.9, 2.0]])
    log_probs = dist.log_prob(actions)
    assert torch.isfinite(log_probs).all()

# Test K: Dynamic bounds change with CBF constraints
def test_beta_dynamic_bounds():
    """Safe bounds should be tighter than nominal when near obstacle."""
    far_bounds = compute_safe_bounds(far_constraints, v_bounds, omega_bounds, alpha=1.0)
    near_bounds = compute_safe_bounds(near_constraints, v_bounds, omega_bounds, alpha=1.0)
    # Near obstacle: tighter bounds
    assert (near_bounds[0][1] - near_bounds[0][0]) <= (far_bounds[0][1] - far_bounds[0][0])

# Test L: Beta policy gradient flows correctly (no NaN)
def test_beta_gradient_flow():
    """Backprop through Beta log_prob should produce finite gradients."""
    actor = SafeBetaActor(obs_dim=6, action_dim=2)
    obs = torch.randn(4, 6)
    safe_bounds = [(0.0, 1.0), (-2.84, 2.84)]
    dist = actor(obs, safe_bounds)
    action = dist.sample()
    loss = -dist.log_prob(action).mean()
    loss.backward()
    for p in actor.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()

# Test M: Beta policy with CBF bounds produces zero violations
def test_beta_cbf_zero_violations():
    """100-step rollout with CBF-Beta policy: no safety violations."""
    env = PursuitEvasionEnv(n_obstacles=4)
    actor = SafeBetaActor(obs_dim=env.obs_dim, action_dim=2)
    cbf = VCPCBFSystem(env)
    violations = 0
    obs, _ = env.reset(seed=42)
    for _ in range(100):
        constraints = cbf.get_constraints(env.get_state())
        bounds = cbf.compute_safe_bounds(constraints)
        dist = actor(torch.tensor(obs), bounds)
        action = dist.sample().numpy()
        obs, _, done, _, info = env.step(action)
        if info.get('safety_violation', False):
            violations += 1
        if done:
            obs, _ = env.reset()
    assert violations == 0
```

**File: `tests/test_safety_reward.py`** (3 tests)

```python
# Test N: Safety reward is positive when far from constraints
def test_safety_reward_positive_far():
    """In center of open arena, safety reward should be near maximum."""
    h_values = [50.0, 45.0, 48.0, 47.0]  # All walls far
    h_max = 100.0
    r_safety = 0.05 * min(h_values) / h_max
    assert r_safety > 0
    assert abs(r_safety - 0.05 * 45.0 / 100.0) < 1e-6

# Test O: Safety reward decreases near boundary
def test_safety_reward_decreases_near_boundary():
    """Robot near wall should get smaller safety reward."""
    h_values_far = [50.0, 45.0, 48.0, 47.0]
    h_values_near = [0.5, 45.0, 48.0, 47.0]  # Near one wall
    r_far = 0.05 * min(h_values_far) / 100.0
    r_near = 0.05 * min(h_values_near) / 100.0
    assert r_near < r_far

# Test P: Zero-sum structure: evader gets negative safety reward
def test_zero_sum_safety_reward():
    """Evader reward should be negative of pursuer reward (including safety)."""
    r_P = compute_pursuer_reward(state, prev_state, info)
    r_E = compute_evader_reward(state, prev_state, info)
    assert abs(r_P + r_E) < 1e-6  # Zero-sum: r_P + r_E = 0
```

**File: `tests/test_feasibility.py`** (2 tests)

```python
# Test Q: N13 classifier improves over iterations
def test_n13_iterative_improvement():
    """Infeasibility rate should decrease over 3 N13 iterations."""
    rates = []
    classifier = FeasibilityClassifier()
    for iteration in range(3):
        rate = run_iteration(classifier, env, policy)
        rates.append(rate)
    assert rates[-1] < rates[0]  # Final < initial
    assert rates[-1] < 0.02     # < 2% after 3 iterations

# Test R: Feasibility classifier correctly flags known infeasible state
def test_feasibility_classifier_accuracy():
    """After training, classifier should flag corner states as infeasible."""
    classifier = train_classifier_on_data(feasible_data, infeasible_data)
    corner_state = [9.9, 9.9, 0.0]  # Known tight corner
    assert not classifier.is_feasible(corner_state)  # Should flag as infeasible
```

### 7.5 Worked Examples

#### Example 1: CBF-QP Solving a Near-Wall Scenario

```
Setup:
  Robot state: [x=9.5, y=5.0, theta=0.0] (near right wall, heading right)
  Arena: 10m x 10m
  VCP offset: d = 0.05m
  Alpha: 1.0
  Nominal action: [v=0.8, omega=0.0] (heading straight toward wall)

Step 1: Compute VCP position
  qx = 9.5 + 0.05*cos(0) = 9.55
  qy = 5.0 + 0.05*sin(0) = 5.0

Step 2: Compute right-wall constraint (h2 = x_max - qx)
  h2 = 10.0 - 9.55 = 0.45
  a_v = -cos(0) = -1.0
  a_omega = d*sin(0) = 0.0

  CBF condition: a_v*v + a_omega*omega + alpha*h >= 0
                 -1.0*v + 0.0*omega + 1.0*0.45 >= 0
                 v <= 0.45

Step 3: Solve QP
  minimize ||[v, omega] - [0.8, 0.0]||^2
  s.t.  v <= 0.45   (from CBF)
        0 <= v <= 1.0, -2.84 <= omega <= 2.84

  Solution: v=0.45, omega=0.0  (slowed from 0.8 to 0.45)

Result: CBF reduced speed by 44% to maintain safety margin.
Note: VCP's omega coefficient is zero at theta=0 (facing wall orthogonally),
so steering doesn't help here. At theta=pi/4, omega would be nonzero.
```

#### Example 2: Beta Policy Truncation

```
Setup:
  Actor outputs: alpha_v=3.0, beta_v=2.0, alpha_omega=2.0, beta_omega=2.0
  Nominal bounds: v in [0, 1.0], omega in [-2.84, 2.84]
  CBF safe bounds (from constraint): v in [0, 0.45], omega in [-2.84, 2.84]

Step 1: Create Beta distribution with safe bounds
  v ~ 0 + 0.45 * Beta(3.0, 2.0)     # Rescaled to [0, 0.45] instead of [0, 1.0]
  omega ~ -2.84 + 5.68 * Beta(2.0, 2.0)  # Full range (unconstrained)

Step 2: Sample
  Beta(3.0, 2.0) has mode at (3-1)/(3+2-2) = 0.67
  v_sample = 0.45 * 0.67 = 0.30 m/s
  Beta(2.0, 2.0) has mode at 0.5
  omega_sample = -2.84 + 5.68 * 0.5 = 0.0 rad/s

Step 3: Compute log probability (with Jacobian correction for change of variables)
  x_v = (0.30 - 0) / (0.45 - 0) = 0.667
  log_prob_v = Beta(3.0, 2.0).log_prob(0.667) - log(0.45 - 0)
             = Beta(3.0, 2.0).log_prob(0.667) - log(0.45)
  x_omega = (0.0 - (-2.84)) / (2.84 - (-2.84)) = 0.5
  log_prob_omega = Beta(2.0, 2.0).log_prob(0.5) - log(2.84 - (-2.84))
                 = Beta(2.0, 2.0).log_prob(0.5) - log(5.68)
  total_log_prob = log_prob_v + log_prob_omega

  NOTE: The Jacobian correction (subtracting log(high - low) per dimension) is
  essential for correct policy gradients. Without it, the rescaling from [0,1]
  to [low, high] distorts the probability density, causing biased gradients.

Key insight: No probability mass is wasted outside the safe set.
Compare with Gaussian: would need to clip/reject samples, distorting gradients.
```

#### Example 3: Infeasibility Tier Escalation

```
Scenario: Pursuer boxed into corner during PE

State: Pursuer at [9.8, 9.7, pi/4], Evader at [9.0, 9.0, -3pi/4]
       Obstacle at [9.5, 9.2, r=0.4]

Tier 1 (Exact QP):
  Arena constraints: h_right=0.15, h_top=0.25 (both tight)
  Obstacle constraint: h_obs=0.08 (very tight)
  Collision constraint: h_col=0.65 (comfortable)
  QP: All 7 constraints active -> INFEASIBLE (no action satisfies all)

Tier 2 (Relax obstacles first):
  Remove obstacle constraint (lowest priority)
  Remaining: 4 arena + 1 collision = 5 constraints
  QP: FEASIBLE -> u_safe = [0.1, -2.0] (slow down, turn left)
  Method: 'relaxed_obstacles'

  Note: Robot may briefly enter obstacle safety margin, but r_robot < chi
  guarantees no physical collision. The margin is the buffer.

If Tier 2 also failed (all constraints conflict):
  Tier 3 (Backup):
  Nearest danger: right wall at bearing 0 rad
  Turn away: omega = 3.0 * wrap_angle(pi - pi/4) = 3.0 * 2.36 = clamped to 2.84
  Brake: v = 0.1 * 1.0 = 0.1
  u_backup = [0.1, 2.84] (slow crawl, hard left turn)

Monitoring log:
  step=42301, method='relaxed_obstacles', h_obs_violation=0.02m
  -> This event is counted toward backup_activation_rate metric
```

---

## 8. Risk Assessment

### 8.1 Phase 2 Specific Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Beta policy hard to train | Medium | Medium | Start with Gaussian baseline; compare; tune alpha/beta init |
| CBF-QP infeasibility > 1% | Medium | High | Tune N13 classifier; increase safety margins; reduce alpha |
| CBF slows training too much | Medium | Medium | Profile; optimize QP solve; batch constraints |
| Safety-task conflict (w5) | Medium | Medium | Tune w5 down; try CASRL [30] if persistent |
| VCP-CBF too conservative | Low | Medium | Reduce alpha; increase d slightly |
| Multi-constraint QP infeasible | High | High | 3-tier handling; increase arena size; reduce obstacle density |
| CBF overlay slows rendering | Low | Low | Only render CBF overlay in `human`/`rgb_array` modes; skip during training with `render_mode=None` |
| wandb ablation runs fail mid-sweep | Low | Medium | Use `wandb.init(resume="allow")`; Hydra `--multirun` auto-retries; check `wandb sync` for offline runs |

### 8.2 Fallback: PNCBF [N02]

If VCP-CBF proves brittle (feasibility rate < 95% after tuning):
1. Implement PNCBF: train value function V^{h,pi} as neural CBF
2. PNCBF automatically captures reachable set geometry
3. Hardware-validated on 12D quadcopter state space
4. Converges in 2-3 policy iterations
5. **Decision point**: If VCP-CBF feasibility < 95% after Phase 2 tuning, switch to PNCBF

---

## 9. Software & Tools

### 9.1 Additional Packages (Beyond Phase 1)

```txt
# Phase 2 additions (append to Phase 1 requirements.txt)
# CBF / Safety
cvxpy==1.6.5               # CBF-QP formulation
osqp==0.6.7.post3          # Fast QP solver backend (OSQP is cvxpy's default)
scipy==1.15.0              # LP for safe bound computation (already in Phase 1)
gpytorch==1.14             # GP disturbance model (prepare for deployment layer)
scikit-learn==1.6.1        # SVM for N13 feasibility classifier
line-profiler==4.2.0       # Profile CBF computation bottlenecks

# Visualization & Tracking (inherited from Phase 1, listed for completeness)
# pygame==2.6.1            # Already in Phase 1 — CBF overlay activated in Phase 2
# wandb==0.19.1            # Already in Phase 1 — safety metrics added
# hydra-core==1.3.2        # Already in Phase 1 — ablation configs added
# omegaconf==2.3.0         # Already in Phase 1
```

**Version compatibility note**: cvxpy 1.6.5 requires numpy>=1.20 (Phase 1 pins numpy==2.2.0, compatible). OSQP 0.6.7 is the solver backend used by cvxpy by default. GPyTorch 1.14 requires PyTorch 2.0+ (Phase 1 pins torch==2.6.0, compatible).

### 9.2 Compute Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| QP solver | CPU (OSQP) | ~1-5ms per solve |
| Beta policy | Same as PPO | Minimal overhead |
| N13 classifier | CPU (SVM) | Train periodically (~30s per retrain) |
| Training | GPU recommended | 2-5x slower than Phase 1 due to CBF |
| Full ablation (25 runs) | ~50-125 GPU-hours | Based on Phase 1 estimates x 2-5x overhead |

### 9.3 Reproducibility Protocol

Follow Phase 1's reproducibility protocol (Appendix B) for all Phase 2 runs. Additional Phase 2 notes:

```python
# CBF-QP solver is deterministic (OSQP with warm_start=True)
# No additional seeding needed for QP solver

# N13 feasibility classifier:
# SVM is deterministic for fixed data; DNN alternative needs seed
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', C=1.0, random_state=seed)

# Ablation runs: use seeds [0, 1, 2, 3, 4] consistently across all configs
# Log seed in W&B config for every run
```

---

## 10. Guide to Phase 2.5

### 10.1 What Phase 2.5 Needs from Phase 2

| Phase 2 Output | Phase 2.5 Usage |
|----------------|-----------------|
| CBF-Beta training performance | Baseline for BarrierNet comparison |
| RCBF-QP deployment filter | The deployment-side of the train-deploy gap |
| Safety metrics | Comparison metrics (violations, intervention rate) |
| VCP-CBF code | Reused inside BarrierNet's differentiable QP |
| Ablation results | Context for whether BarrierNet is needed |

### 10.2 Key Question Phase 2.5 Answers

> **Is the train-deploy safety gap significant enough to warrant BarrierNet?**

If Phase 2 shows < 5% performance gap between CBF-Beta (training) and RCBF-QP (deployment), BarrierNet may not be worth the added complexity. Phase 2.5 measures this explicitly.

### 10.3 Reading List for Phase 2.5

1. **[N04] Xiao et al. 2023** — BarrierNet (core Phase 2.5 algorithm)
2. **cvxpylayers documentation** — Differentiable QP layer
3. **[N05] Zhang et al. 2024** — GCBF+ (alternative scalable approach)

### 10.4 Potential Phase 2.5 Blockers from Phase 2

| Issue | Impact on Phase 2.5 | Resolution |
|-------|---------------------|-----------|
| VCP-CBF switched to PNCBF | BarrierNet formulation changes | PNCBF is already differentiable — may not need BarrierNet |
| Train-deploy gap is tiny (<3%) | Phase 2.5 may be unnecessary | Skip Phase 2.5, proceed to Phase 3 |
| CBF-QP too slow for training | BarrierNet (also QP-based) will be slow too | Optimize solver; consider closed-form filter [05] |
