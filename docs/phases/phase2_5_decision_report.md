# Phase 2.5 Decision Report: BarrierNet Experiment Results

**Date**: 2026-02-22
**Status**: Final
**Author**: Claude Code (Sessions 30-34)

---

## 1. Executive Summary

Phase 2.5 evaluated the BarrierNet approach (differentiable QP safety layer integrated into PPO training) for the pursuit-evasion task. We compared three approaches:

1. **BarrierNet PPO**: MLP backbone + differentiable QP safety layer (end-to-end trained)
2. **Baseline PPO**: Standard PPO without safety constraints
3. **Baseline PPO + VCP-CBF Filter**: Trained PPO with post-hoc safety filter at deployment

**Key finding**: The BarrierNet approach faces fundamental challenges in the PE environment. After 324K training steps, it achieves only 2% capture rate with 7.56% CBF safety violations. In contrast, a standard PPO baseline trained WITH obstacles achieves 100% capture with 6.07% violations. Adding a post-hoc VCP-CBF safety filter to this baseline reduces capture to 91.5% due to high intervention (44.5%), while violations remain similar at 6.84%.

**Recommendation**: For Phase 3, **train PPO normally with obstacles**, then optionally add a **tuned post-hoc VCP-CBF filter** at deployment. The filter's current aggressiveness (44.5% intervention, reducing capture from 100% to 91.5%) suggests the CBF parameters need tuning before deployment use.

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

### 2.3 Baseline PPO (Retrained with Obstacles)
- **Model**: SB3 PPO, retrained with 2 obstacles (1M steps, seed=42)
- **Obs dim**: 18 (14 base + 4 obstacle features for 2 nearest obstacles)
- **Training**: `env.n_obstacles=2 env.n_obstacle_obs=2 total_timesteps=1000000`
- **Result**: ep_rew_mean=100 at convergence (~100% capture rate vs random evader)

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

## 4. Evaluation Results

### 4.0 Interim Results (100 episodes, old baseline without obstacle training)

| Metric | BarrierNet PPO | Baseline PPO (no obs) | Baseline + Filter |
|--------|---------------|----------------------|-------------------|
| **Capture Rate** | 3.0% | 4.0% | 1.0% |
| **Mean Reward** | -45.58 | -44.04 | -48.57 |
| **Safety Violation Rate** | 7.42% | 4.05% | 2.91% |
| **Intervention Rate** | 87.5% | N/A | 0.1% |

*Note: Low capture rates across all approaches because the baseline was trained without obstacles.*

### 4.1 Final Results (200 episodes, baseline retrained WITH obstacles)

| Metric | BarrierNet PPO | Baseline PPO | Baseline + Filter |
|--------|---------------|-------------|-------------------|
| **Capture Rate** | 2.0% | **100.0%** | 91.5% |
| **Mean Reward** | -47.08 | **100.30** | 87.54 |
| **Safety Violation Rate** | 7.56% | **6.07%** | 6.84% |
| **QP Correction** | 0.77 | 0.00 | 0.90 |
| **Intervention Rate** | 88.2% | N/A | 44.5% |
| **Inference Time** | 5.42 ms | 0.18 ms | 0.46 ms |

### 4.2 Analysis

**Performance**: The retrained baseline achieves **100% capture** against the random evader, confirming that standard PPO can solve this task when given proper observations (obstacle features). The post-hoc filter reduces this to 91.5% because the filter intervenes on 44.5% of actions, blocking some pursuit maneuvers near obstacles.

**Safety**: All three approaches show similar violation rates (6-7.6%). The baseline without filter has the *lowest* violations (6.07%), suggesting it learned implicit obstacle avoidance during training. The filter's high intervention rate (44.5%) indicates the CBF parameters (alpha=1.0, d=0.1) are too conservative — the filter treats many safe pursuit actions near obstacles as unsafe.

**BarrierNet**: Still fundamentally limited at 2% capture with 88% QP intervention. The end-to-end approach cannot overcome the exploration-safety conflict in this environment.

**Efficiency**: Baseline runs at 0.18 ms/step, Baseline+Filter at 0.46 ms/step (2.6x overhead), BarrierNet at 5.42 ms/step (30x overhead).

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

**Standard PPO trained with obstacles + optional post-hoc VCP-CBF filter at deployment**

Rationale:
1. **Performance**: Standard PPO achieves 100% capture when trained with obstacle observations
2. **Implicit safety**: PPO learns obstacle avoidance during training (6.07% violations without any filter)
3. **Speed**: 14 minutes to train 1M steps; 0.18 ms/step inference
4. **Simplicity**: No differentiable QP, no special training loop needed

### 6.2 Post-hoc Filter Status

The VCP-CBF filter with current parameters (alpha=1.0, d=0.1) is **too aggressive** — it intervenes on 44.5% of actions and reduces capture from 100% to 91.5%. Before deployment:
1. **Tune alpha**: Reduce from 1.0 to 0.3-0.5 for less conservative filtering
2. **Reduce d**: Decrease VCP offset from 0.1m to 0.05m
3. **Re-evaluate**: Target <10% intervention rate while maintaining safety

### 6.3 BarrierNet Status

BarrierNet training has been terminated. The approach is not viable for this environment due to the fundamental exploration-safety conflict (88% QP intervention prevents learning).

### 6.4 Future Considerations

If higher safety guarantees are needed (e.g., for real robot deployment):
1. **Tuned post-hoc filter**: Adjust CBF parameters for lower intervention rate
2. **Safety reward shaping**: Add CBF-based penalty during training to learn CBF-aware policies
3. **Discrete-time CBFs**: Formulations that account for discrete updates
4. **Curriculum training**: Start without filter, gradually introduce CBF constraints

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
| Final eval results | niro-2: `results/phase2_5_final/evaluation_results.txt` |
| Training checkpoint | niro-2: `checkpoints/barriernet_ds10/barriernet_iter_50.pt` |
| Baseline model (with obstacles) | niro-2: `models/local_42/final_model.zip` (1M steps, 2 obs) |
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
| S34 | 2026-02-22 | Retrain baseline with obstacles, final 200-ep evaluation |
