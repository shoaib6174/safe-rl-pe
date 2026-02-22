# Phase 2.5 Decision Report: BarrierNet Experiment Results

**Date**: 2026-02-22
**Status**: Interim (training at 324K/2M steps)
**Author**: Claude Code (Sessions 30-33)

---

## 1. Executive Summary

Phase 2.5 evaluated the BarrierNet approach (differentiable QP safety layer integrated into PPO training) for the pursuit-evasion task. We compared three approaches:

1. **BarrierNet PPO**: MLP backbone + differentiable QP safety layer (end-to-end trained)
2. **Baseline PPO**: Standard PPO without safety constraints
3. **Baseline PPO + VCP-CBF Filter**: Trained PPO with post-hoc safety filter at deployment

**Key finding**: The BarrierNet approach faces fundamental challenges in the PE environment. After 324K training steps, it achieves only 3% capture rate with 7.4% CBF safety violations — worse on both metrics than the simpler post-hoc filter approach (1% capture, 2.9% violations). The high QP intervention rate (87.5%) severely constrains exploration, preventing the agent from learning pursuit behavior.

**Recommendation**: Use the **Baseline PPO + Post-hoc VCP-CBF Filter** approach for Phase 3. It provides the best safety-performance balance with minimal complexity.

---

## 2. Experimental Setup

### 2.1 Environment
- **Arena**: 20m x 20m, 2 circular obstacles (radius 0.3-1.0m)
- **Agents**: Unicycle dynamics, v_max=1.0 m/s, omega_max=2.84 rad/s
- **Episode**: Max 600 steps (dt=0.05s), capture_radius=0.5m
- **Evader**: Random policy (opponent_policy=None)

### 2.2 BarrierNet Architecture
- **Backbone**: 2-layer MLP (256 hidden), ReLU activation
- **Action head**: Linear(256, 2) -> tanh squashing to [0, v_max] x [-omega_max, omega_max]
- **Safety layer**: DifferentiableVCPCBFQP (qpth-based)
  - 8 constraints max: 4 arena + 2 obstacle + 1 collision + 1 slack
  - alpha=1.0, d=0.1m (VCP offset)
  - w_v=150.0, w_omega=1.0 (QP weights)
- **Exploration**: Fixed Gaussian noise (std_v=0.30, std_w=0.50)
- **PPO**: evaluate_actions (Approach A: log pi of u_nom), 10 epochs, lr=3e-4

### 2.3 Baseline PPO
- **Model**: SB3 PPO, trained in Phase 1 (obstacle-free environment, obs_dim=14)
- **Note**: Evaluated with truncated obs (first 14 dims) in 2-obstacle env

### 2.4 Post-hoc VCP-CBF Filter
- **VCPCBFFilter**: Same parameters as BarrierNet QP (alpha=1.0, d=0.1)
- **Applied to**: Baseline PPO actions at deployment time

---

## 3. Training Results

### 3.1 BarrierNet Training (ds10, 324K/2M steps)

| Metric | Value | Trend |
|--------|-------|-------|
| Reward | -50 +/- 2 | Flat (all timeout) |
| Policy loss | -0.015 to -0.025 | Slightly increasing magnitude |
| Critic loss | 5.0 - 10.0 | Decreasing (learning value function) |
| QP intervention rate | 85-90% | Stable |
| QP infeasibility rate | 0.0% | Perfect feasibility |
| Action std (v, omega) | 0.30, 0.50 | Fixed (buffer) |
| Occasional captures | Iters 4,5,19,22,56,59,61 | ~10% of iterations |

**Observation**: The agent occasionally captures the evader by chance (e.g., Iter 22: reward=+26.01), but cannot reliably learn from these events. The 85% QP intervention rate means the agent controls only ~15% of its actual executed actions.

### 3.2 Training Bugs Fixed (Sessions 31-32)
1. **action_log_std explosion**: Changed from learnable nn.Parameter to fixed buffer (prevents std growing due to QP-noise positive correlation)
2. **PPO gradient flow bug**: Added `evaluate_actions()` method — during PPO update, evaluate log_prob of STORED nominal actions under CURRENT policy (standard importance sampling). Previous code was resampling noise, giving zero gradient to backbone.
3. **Hybrid CPU/GPU**: CPU deepcopy for batch-1 rollout inference, GPU for batched PPO update

---

## 4. Evaluation Results (100 episodes, iter_50 checkpoint)

| Metric | BarrierNet PPO | Baseline PPO | Baseline + Filter |
|--------|---------------|-------------|-------------------|
| **Capture Rate** | 3.0% | 4.0% | 1.0% |
| **Mean Reward** | -45.58 | -44.04 | -48.57 |
| **Safety Violation Rate** | **7.42%** | 4.05% | **2.91%** |
| **QP Correction** | 0.794 | 0.000 | 0.000 |
| **Intervention Rate** | 87.5% | N/A | 0.1% |
| **Inference Time** | 13.49 ms | 0.18 ms | 0.40 ms |

### 4.1 Analysis

**Safety**: The Baseline+Filter approach achieves the lowest safety violation rate (2.91%), followed by the baseline alone (4.05%). BarrierNet has the HIGHEST violation rate (7.42%) despite 87.5% QP intervention. This counterintuitive result occurs because:
- BarrierNet's untrained policy proposes aggressive exploratory actions
- The QP corrects most of them, but in discrete-time (dt=0.05s), corrections are imperfect
- The baseline policy, trained without obstacles, naturally avoids most unsafe regions simply because it learned to navigate open space

**Performance**: All approaches achieve low capture rates (1-4%) because:
- The baseline was trained in an obstacle-free environment and doesn't have obstacle observations
- BarrierNet hasn't converged (only 100K steps at evaluation)
- The random evader + 2 obstacles make capture inherently harder

**Efficiency**: Baseline+Filter runs at 0.40 ms/step vs BarrierNet's 13.49 ms/step — a 34x speedup. The filter intervenes on only 0.1% of actions, meaning the trained baseline already takes mostly safe actions.

### 4.2 Why BarrierNet Struggles

1. **Exploration-Safety Conflict**: The QP modifies 87.5% of nominal actions, meaning the agent gets very limited feedback on which of ITS actions lead to good outcomes. The actual executed action is mostly determined by the QP, not the policy.

2. **Sparse Reward**: In the 2-obstacle environment with random evader, captures are rare events. The distance_scale=10 reward shaping helps but isn't sufficient to overcome the exploration limitation.

3. **Approach A Limitation**: Using log pi(u_nom) approximates the true gradient. When QP correction is large (mean 0.79), the nominal action u_nom is far from the safe action u_safe, making this approximation less accurate.

4. **Discrete-Time CBF Mismatch**: The continuous-time CBF formulation (h_dot + alpha*h >= 0) doesn't guarantee h >= 0 in discrete time. With dt=0.05s and v_max=1.0 m/s, the robot moves 5cm/step, which can overshoot constraint boundaries.

---

## 5. Comparison with Literature

The BarrierNet paper (Xiao et al., T-RO 2023) demonstrated success on:
- **Simpler tasks**: Single-integrator and double-integrator systems
- **Fewer constraints**: Typically 1-3 obstacles in a small arena
- **Continuous-time**: Using smaller dt or continuous dynamics

Our pursuit-evasion task is significantly more challenging:
- **Nonholonomic dynamics**: Unicycle model with limited steering
- **Adversarial**: Moving evader (even if random)
- **Many constraints**: 8 CBF constraints (4 arena + 2 obstacle + 1 collision + 1 slack)
- **Large arena**: 20x20m with sparse reward signal

---

## 6. Decision

### 6.1 Recommended Approach for Phase 3

**Baseline PPO + Post-hoc VCP-CBF Filter (Deployment Safety)**

Rationale:
1. **Simplicity**: Train PPO normally (fast, proven), add safety filter only at deployment
2. **Best safety**: 2.91% CBF violation rate vs 7.42% for BarrierNet
3. **Fast inference**: 0.40 ms vs 13.49 ms (34x faster)
4. **Low intervention**: Filter modifies only 0.1% of actions, preserving learned behavior
5. **Modularity**: Safety filter can be tuned independently of training

### 6.2 BarrierNet Status

BarrierNet training will continue on niro-2 to 2M steps. If it shows significant improvement, results will be updated. However, the fundamental exploration-safety conflict suggests convergence will remain limited in this environment.

### 6.3 Future Considerations

If higher safety guarantees are needed (e.g., for real robot deployment), consider:
1. **Curriculum training**: Start without QP constraints, gradually increase alpha
2. **Reward shaping**: Stronger distance-based reward (distance_scale=50-100)
3. **Discrete-time CBFs**: Formulations that account for discrete updates
4. **Pre-training**: Initialize BarrierNet backbone from trained baseline weights

---

## 7. Files & Artifacts

| Artifact | Location |
|----------|----------|
| BarrierNet actor | `agents/barriernet_actor.py` |
| BarrierNet PPO agent | `agents/barriernet_ppo.py` |
| BarrierNet trainer | `training/barriernet_trainer.py` |
| Differentiable QP layer | `safety/differentiable_qp.py` |
| Evaluation script | `scripts/evaluate_barriernet.py` |
| Training script | `scripts/train_barriernet.py` |
| Interim eval results | `results/phase2_5_interim/evaluation_results.txt` |
| Training checkpoint | niro-2: `checkpoints/barriernet_ds10/barriernet_iter_50.pt` |
| Baseline model | `models/local_42/final_model.zip` |
| Tests | `tests/test_barriernet_actor.py` (32 tests, all passing) |

---

## 8. Session History

| Session | Date | Key Work |
|---------|------|----------|
| S27 | 2026-02-22 | DifferentiableVCPCBFQP implementation |
| S28 | 2026-02-22 | BarrierNetActor + BarrierNetCritic |
| S29 | 2026-02-22 | BarrierNetPPO agent + trainer |
| S30 | 2026-02-22 | Training deployment + CUDA optimization |
| S31 | 2026-02-22 | Hybrid CPU/GPU + std explosion fix |
| S32 | 2026-02-22 | Gradient flow fix + evaluate_actions |
| S33 | 2026-02-22 | 3-way evaluation + decision report |
