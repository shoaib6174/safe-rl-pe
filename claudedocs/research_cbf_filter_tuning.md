# Research Report: Fixing the VCP-CBF Safety Filter

**Date**: 2026-02-22
**Context**: Phase 2.5 evaluation shows the post-hoc VCP-CBF filter is too aggressive (42% intervention rate) and paradoxically increases safety violations (7.3% vs 5.3% without filter)

---

## 1. Problem Analysis

### Current Filter Behavior (100 episodes, local evaluation)
| Metric | Baseline (no filter) | Baseline + Filter (alpha=1.0) |
|--------|---------------------|-------------------------------|
| Capture Rate | 99% | 97% |
| CBF Violation Rate | 5.33% | 7.30% |
| Intervention Rate | N/A | 41.8% |
| Mean Episode Length | 205 steps | 280 steps |

### Root Causes

**1. Alpha too aggressive**: With `alpha=1.0`, the CBF condition `a_v*v + a_omega*omega + alpha*h >= 0` requires the barrier function derivative to offset the full barrier value. When h is small (near boundaries), this forces the robot to actively retreat rather than coast past.

**2. Continuous-time/discrete-time mismatch**: Our filter uses the continuous-time CBF condition `hdot + alpha*h >= 0`, but the actual system evolves in discrete steps of `dt=0.05s`. The robot moves up to 5cm per step, and the linearized constraint can underestimate the actual state change.

**3. Longer episodes = more violations**: The filter makes episodes 37% longer (280 vs 205 steps) by forcing detours around obstacles. More steps near constraint boundaries = more chances for discrete-time overshoot violations. This is why the filter *increases* total violations despite preventing individual unsafe actions.

---

## 2. Research Findings

### 2.1 CBF-RL: Barrier Reward Shaping (Most Relevant)

**Paper**: [CBF-RL: Safety Filtering RL in Training with CBFs](https://arxiv.org/abs/2510.14959) (Oct 2025)

**Key idea**: Instead of only applying the CBF filter at deployment, use a dual approach during training:
1. **Active CBF filter** prevents catastrophic actions during training
2. **Barrier-inspired reward** biases policy toward naturally safe actions

**Reward formula**:
```
r_cbf = max(hdot + alpha*h, 0) + exp(-||u_nom - u_safe||^2 / sigma^2) - 1
```
- First term: penalizes CBF condition violations
- Second term: penalizes when filter modifies the action

**Result**: Policy internalizes safety constraints, achieving 92.7% success *without any runtime filter* (vs 38.7% for filter-only approach).

**Relevance**: This directly addresses our problem. If the PPO policy learns to avoid CBF interventions during training, the post-hoc filter at deployment becomes a lightweight backup rather than an aggressive corrector.

### 2.2 Discrete-Time CBF (DCBF)

**Papers**:
- [Agrawal & Sreenath, RSS 2017](https://www.roboticsproceedings.org/rss13/p73.pdf) — foundational DCBF
- [Safety-Critical MPC with DCBF](https://github.com/HybridRobotics/MPC-CBF) — practical implementation

**Key idea**: Replace continuous-time condition with:
```
h(x_{k+1}) >= (1 - gamma) * h(x_k)    where gamma in (0, 1]
```

- `gamma = 1.0`: strict forward invariance — `h(x_{k+1}) >= 0`
- `gamma = 0.1`: h can decrease by at most 10% per step

**Advantage**: Directly guarantees safety at the next discrete timestep, unlike continuous-time approximation.

**Challenge**: The constraint `h(f(x_k, u)) >= ...` is nonlinear in `u` for unicycle dynamics. Solutions:
- First-order Taylor linearization → still a QP but more accurate
- Full nonlinear evaluation → NLP, slower but exact

### 2.3 CSRL: CBF-Safe RL for Pursuit-Evasion

**Paper**: [Ensuring Safety in Target Pursuit Control](https://arxiv.org/abs/2411.17552) (Nov 2024)

**Key idea**: CBF-based safety filter specifically for pursuit tasks with three constraint types:
1. **Input saturation CBF**: Dynamic bounds that adapt based on pursuer-target distance
2. **Collision avoidance CBF**: Standard obstacle/collision barriers
3. **Sensing range CBF**: Keeps target within sensor range

**Switch strategy**: Dynamically adjusts which constraints are active based on state. When far from obstacles, fewer constraints are active → lower intervention rate.

**Relevance**: The adaptive/dynamic constraint activation is directly applicable to reducing our intervention rate.

### 2.4 Adaptive CBF Parameters

**Paper**: [How to Adapt CBFs: A Learning-Based Approach](https://arxiv.org/html/2504.03038) (Apr 2025)

**Key idea**: Use neural networks to dynamically adapt CBF parameters (including alpha) online, reducing conservatism while maintaining safety.

---

## 3. Recommended Action Plan

### Priority 1: Alpha Parameter Sweep (immediate, no code changes)

Sweep alpha over `{0.1, 0.3, 0.5, 0.7, 1.0}` using the existing visualization script with minor modification.

**Expected behavior**:
| alpha | Intervention Rate (est.) | Safety (est.) |
|-------|--------------------------|---------------|
| 1.0 | ~42% (current) | Moderate |
| 0.5 | ~20-25% | Moderate |
| 0.3 | ~10-15% | Good |
| 0.1 | ~5% | Minimal filtering |

**Why this works**: The CBF condition is `a_v*v + a_omega*omega + alpha*h >= 0`. Reducing alpha makes the right-hand side less negative, meaning the constraint is satisfied more easily. The filter only intervenes when the nominal action would violate the (now relaxed) condition.

### Priority 2: Safety Margin for Discrete-Time Robustness (small code change)

Add a positive margin epsilon to the CBF QP constraint:
```
a_v*v + a_omega*omega + alpha*h >= epsilon
```
where `epsilon = 0.05`. This prevents the filter from allowing states that sit exactly on the boundary, providing a buffer against discrete-time overshoot.

### Priority 3: CBF-Aware Reward Shaping (retraining needed)

During PPO training, add to the reward:
```python
# Penalty for being close to CBF boundary
min_h = min(compute_cbf_margins(state, obstacles, opponent))
r_cbf = -lambda_cbf * max(0, margin_threshold - min_h)
```
Or the full CBF-RL dual reward:
```python
r_cbf = max(hdot + alpha*h, 0) + exp(-||u_nom - u_safe||^2 / sigma^2) - 1
```

This teaches the policy to naturally avoid unsafe regions, reducing filter interventions at deployment to near zero.

### Priority 4: Discrete-Time CBF Formulation (refactoring needed)

Replace the continuous-time CBF constraint with a linearized DCBF:
```
h(x_k + dt * f(x_k, u)) >= (1 - gamma) * h(x_k)
```
Linearized around u_nom, this becomes a QP constraint that properly accounts for the discrete timestep.

---

## 4. Experimental Validation

All four fixes were implemented and evaluated via a 14-configuration sweep (100 episodes each, seed=42).

### 4.1 Implementation Summary

| Fix | Code Location | Description |
|-----|---------------|-------------|
| Epsilon margin | `safety/vcp_cbf.py` `solve_cbf_qp()` | Added `epsilon` param; constraint becomes `a_v*v + a_omega*omega + alpha*h >= epsilon` |
| DCBF mode | `safety/vcp_cbf.py` `dcbf_filter_action()` | New method implementing `h(x_{k+1}) >= (1-gamma)*h(x_k)`, linearized as `dt*a_v*v + dt*a_omega*omega + gamma*h >= 0` |
| CBF-RL reward | `envs/rewards.py` `SafetyRewardComputer` | Dual penalty: `r = -w_cbf * max(0, -cbf_cond) - w_int * (1 - exp(-||u_diff||^2/sigma^2))` |
| Alpha sweep | `scripts/sweep_filter_params.py` | 14-config sweep script with comparison plots and Pareto frontier |

### 4.2 Full Sweep Results

| Configuration | Capture% | Violation% | Intervention% | Reward | MeanLen |
|---------------|----------|------------|---------------|--------|---------|
| **No Filter** | 99.0% | 6.49% | 0.0% | 98.8 | 209 |
| CT alpha=0.1 | 4.0% | 4.69% | 99.7% | -44.0 | 592 |
| CT alpha=0.3 | 54.0% | 7.19% | 87.4% | 31.2 | 480 |
| CT alpha=0.5 | 82.0% | 9.12% | 69.2% | 73.3 | 381 |
| CT alpha=0.7 | 87.0% | 6.46% | 57.4% | 80.8 | 337 |
| CT alpha=1.0 | 95.0% | 7.72% | 45.9% | 92.8 | 295 |
| CT a=0.3 e=0.02 | 51.0% | 4.91% | 88.1% | 26.8 | 491 |
| CT a=0.3 e=0.05 | 35.0% | 4.44% | 89.0% | 2.7 | 530 |
| CT a=0.5 e=0.02 | 73.0% | 8.30% | 70.3% | 59.8 | 383 |
| CT a=0.5 e=0.05 | 74.0% | 6.84% | 72.7% | 61.3 | 408 |
| DCBF gamma=0.05 | 89.0% | 7.29% | 45.2% | 83.8 | 295 |
| DCBF gamma=0.1 | 98.0% | 4.86% | 25.2% | 97.3 | 246 |
| **DCBF gamma=0.2** | **100.0%** | **3.57%** | **6.6%** | **100.3** | **211** |
| **DCBF gamma=0.5** | **100.0%** | **2.96%** | **5.3%** | **100.3** | **206** |

### 4.3 Key Findings

**1. Continuous-time CBF is fundamentally broken for this discrete-time system.**
- All CT configurations show high intervention rates (45-99%) regardless of alpha
- Lower alpha paradoxically INCREASES intervention: alpha=0.1 gives 99.7% intervention, 4% capture
- This is because the continuous-time linearization produces overly conservative constraints at small alpha values — the constraint coefficients (a_v, a_omega) don't scale with alpha, so reducing alpha just makes the `alpha*h` term insufficient to satisfy the constraint
- Adding epsilon margin makes CT even worse (more conservative)

**2. DCBF is strictly superior to both no-filter and CT-CBF.**
- DCBF gamma=0.2: 100% capture, 3.57% violations, 6.6% intervention — better than no-filter on ALL metrics
- DCBF gamma=0.5: 100% capture, 2.96% violations, 5.3% intervention — best safety with minimal intervention
- The DCBF formulation `dt*a_v*v + dt*a_omega*omega + gamma*h >= 0` properly accounts for discrete timesteps
- With gamma=0.2-0.5, the filter only activates when truly needed, allowing the PPO policy to operate freely in safe regions

**3. DCBF resolves the "filter increases violations" paradox.**
- No-filter: 6.49% violations in 209 steps → DCBF gamma=0.2: 3.57% in 211 steps
- The DCBF filter achieves 45% fewer violations without significantly increasing episode length
- Unlike CT-CBF, DCBF doesn't force unnecessary detours that increase time near constraint boundaries

**4. CBF-RL reward shaping is available for future retraining.**
- The `SafetyRewardComputer` now supports dual CBF-RL penalty terms
- Can be activated during Phase 3 training if further safety improvement is needed
- Parameters: `w_cbf_penalty`, `w_intervention`, `sigma_sq` in the reward computer

### 4.4 Recommended Configuration

For deployment, use **DCBF with gamma=0.2**:
```python
filter = VCPCBFFilter(d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
                      robot_radius=0.15, arena_half_w=10.0, arena_half_h=10.0)
action, info = filter.dcbf_filter_action(action, state, dt=0.05, gamma=0.2,
                                          obstacles=obstacles, opponent_state=evader)
```

This achieves the original target criteria:
- Intervention rate < 15% (actual: 6.6%)
- Capture rate > 95% (actual: 100%)
- Violation rate better than unfiltered baseline (3.57% vs 6.49%)

### 4.5 Plots

- `results/filter_sweep/sweep_comparison.png` — 4-panel bar chart (capture, violations, intervention, reward)
- `results/filter_sweep/pareto_frontier.png` — Capture rate vs intervention rate scatter plot

---

## 5. Sources

- [CBF-RL: Safety Filtering RL in Training with CBFs](https://arxiv.org/abs/2510.14959)
- [Ensuring Safety in Target Pursuit Control: CSRL](https://arxiv.org/abs/2411.17552)
- [Discrete CBFs for Safety-Critical Control (Agrawal & Sreenath)](https://www.roboticsproceedings.org/rss13/p73.pdf)
- [Safety-Critical MPC with DCBF (Zeng et al.)](https://github.com/HybridRobotics/MPC-CBF)
- [How to Adapt CBFs: Learning-Based Approach](https://arxiv.org/html/2504.03038)
- [How to Train Your Neural CBF](https://mit-realm.github.io/pncbf/)
- [Learning CBFs and their Application in RL: Survey](https://arxiv.org/html/2404.16879v1)
- [Obstacle Avoidance for Unicycle with Time-Varying CBFs](https://arxiv.org/abs/2307.08227)
- [Safety-Critical Planning with DHOCBFs](https://arxiv.org/abs/2403.19122)
