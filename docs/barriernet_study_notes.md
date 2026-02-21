# BarrierNet Architecture Study Notes

## 1. Repository Structure

```
BarrierNet/
├── 2D_Robot/          # Unicycle robot (closest to our problem)
│   ├── models.py      # BarrierNet + FCNet classes
│   ├── my_classes.py  # Dataset + QP solver wrapper
│   └── robot-barriernet.py  # Training/testing
├── 3D_Robot/          # Quadrotor (double integrator)
├── Merging/           # Vehicle merging (1D control)
├── Driving/           # Vision-based driving (most complex)
│   ├── models/
│   │   ├── barrier_net.py  # Full architecture
│   │   ├── base.py         # Training loop
│   │   └── utils.py        # cvx_solver fallback
│   └── task_cbf.py
└── README.md
```

## 2. Core Architecture Pattern

```
Input: state x
  │
  ▼
FC Layers (shared backbone)
  │
  ├─► Branch 1: Nominal control q = FC(x)     [nControls outputs]
  │
  └─► Branch 2: CBF penalties p = 4*σ(FC(x))  [2 outputs per constraint]
  │
  ▼
Differentiable QP Layer (qpth):
  min_u  ‖u − q‖²
  s.t.   G(x,p)·u ≤ h(x,p)     (CBF constraints)
         u_min ≤ u ≤ u_max       (control bounds)
  │
  ▼
Output: safe action u*
```

**Key insight**: Both the nominal control AND the CBF penalty parameters are learned. The penalties `p₁, p₂ ∈ [0, 4]` (via sigmoid) control how aggressively the CBF enforces safety. The network learns to balance task performance with safety.

## 3. QP Backend: qpth

**Library**: `qpth.qp.QPFunction`
**Solver**: Primal-Dual Interior Point Method (PDIPM)
**Batching**: `QPSolvers.PDIPM_BATCHED` for training

```python
# Standard form: min 0.5 x^T Q x + p^T x  s.t. Gx ≤ h, Ax = b
x = QPFunction(verbose=0)(Q, q, G, h, e, e)  # e = empty (no equality)
```

**Backward pass**: Differentiates KKT conditions; reuses LU factorization from forward pass → nearly free gradient computation.

**Fallback**: `cvxopt.solvers.qp()` for single-sample evaluation.

## 4. 2D Robot (Unicycle) — Our Template

### Dynamics
```
State: [px, py, θ, v]  (4D — note: includes velocity as state)
Control: [u₁, u₂] = [ω, a] = [angular_vel, acceleration]

ẋ = v·cos(θ)
ẏ = v·sin(θ)
θ̇ = u₁  (angular velocity)
v̇ = u₂  (acceleration)
```

### CBF Constraint (2nd-order, circular obstacle)
```
Barrier: b(x) = (px - ox)² + (py - oy)² - R²

Lie derivatives:
  Lf b  = 2(px-ox)·v·cos(θ) + 2(py-oy)·v·sin(θ)
  Lf²b  = 2v²
  LgLfb = [-2(px-ox)·v·sin(θ) + 2(py-oy)·v·cos(θ),     ← for u₁ (ω)
            2(px-ox)·cos(θ) + 2(py-oy)·sin(θ)]            ← for u₂ (a)

QP constraint: -LgLfb · u ≤ Lf²b + (p₁+p₂)·Lfb + p₁·p₂·b
```

### Our Adaptation Differences
| BarrierNet 2D Robot | Our VCP-CBF PE |
|---------------------|----------------|
| State: [px,py,θ,v] (4D) | State: [x,y,θ] (3D, v is control) |
| Control: [ω, a] (2D) | Control: [v, ω] (2D) |
| 2nd-order CBF (relative degree 2) | 1st-order VCP-CBF (relative degree 1 via VCP) |
| Single obstacle | 4+ obstacles + arena + collision |
| Supervised learning (MSE) | RL (PPO clipped objective) |
| Static obstacle | Dynamic opponent |

**Critical difference**: Our VCP trick reduces relative degree from 2 to 1, which SIMPLIFIES the QP constraints. Each constraint is just `a_v·v + a_ω·ω + α·h ≥ 0`, making our QP layer simpler than BarrierNet's.

## 5. Infeasibility Handling

BarrierNet does NOT explicitly handle infeasibility. It relies on:
1. **Learned penalties** (p₁, p₂) adapt to avoid infeasible regions
2. **Nominal control** from same network tends to be near-safe
3. **MSE loss** implicitly regularizes toward feasible solutions

**For our RL setting**, we need explicit handling → keep our 3-tier system (Tier 1: SVM classifier, Tier 2: hierarchical relaxation, Tier 3: backup controller) as a fallback when the differentiable QP is infeasible.

## 6. Training Strategy Differences

| Aspect | BarrierNet (original) | Our BarrierNet-PPO |
|--------|----------------------|-------------------|
| Loss | MSE (supervised) | PPO clipped objective |
| Labels | Optimal control from MPC | Reward signal |
| Exploration | None (supervised) | On-policy (PPO) |
| Batch | Standard batching | Rollout buffer |
| Gradient | Through QP + MSE | Through QP + PPO loss |

**Key challenge**: PPO expects stochastic policies. The QP layer is deterministic (u* = argmin QP). We need to either:
- (a) Output nominal action distribution, sample, then filter through QP (stochastic pre-QP, deterministic post-QP)
- (b) Parameterize noise in the QP cost function
- Phase 2.5 spec recommends approach (a)

## 7. Mapping to Our Problem

| BarrierNet Component | Our Implementation |
|---------------------|-------------------|
| State input | PE observation (14D + 2K obstacle features) |
| Policy network (MLP) | PPO actor: outputs mean/std of [v, ω] |
| Branch 2 (CBF params) | Not needed — our VCP-CBF has fixed α=1.0 |
| CBF constraints | VCP-CBF: 4 arena + N obstacle + 1 collision |
| QP layer | qpth: min ‖u-u_nom‖² s.t. a_v·v + a_ω·ω ≥ -α·h |
| Training loss | PPO clipped surrogate objective |
| Infeasibility | Our 3-tier system (kept as fallback) |

## 8. Implementation Plan

### Differentiable QP Layer (`safety/differentiable_qp.py`)
```python
class DifferentiableVCPCBFLayer(torch.nn.Module):
    def forward(self, u_nominal, states, obstacles, opponent_states):
        # 1. Compute VCP-CBF constraints (vectorized, batched)
        G, h = self._build_constraint_matrices(states, obstacles, opponent_states)
        # 2. Build QP: min ‖u - u_nom‖²  s.t. Gu ≤ h
        Q = torch.eye(2).expand(batch, 2, 2)
        p = -u_nominal
        # 3. Solve via qpth (differentiable)
        u_safe = QPFunction(verbose=0)(Q, p, G, h, e, e)
        return u_safe
```

### Integration with PPO Actor
```python
class BarrierNetActor(torch.nn.Module):
    def __init__(self, obs_dim):
        self.backbone = MLP(obs_dim, [256, 256])
        self.mean_head = nn.Linear(256, 2)   # [v, ω] mean
        self.log_std = nn.Parameter(...)      # learnable std
        self.qp_layer = DifferentiableVCPCBFLayer(...)

    def forward(self, obs, state_info):
        features = self.backbone(obs)
        mean = self.mean_head(features)
        # Sample from Gaussian (for PPO exploration)
        u_nominal = mean + std * noise
        # Filter through differentiable QP
        u_safe = self.qp_layer(u_nominal, state_info)
        return u_safe, mean, std
```

## 9. Dependencies

```
pip install qpth cvxopt
```
- `qpth >= 0.0.18` — differentiable QP solver
- `cvxopt` — fallback QP solver for evaluation
- `torch >= 2.0` — already have
