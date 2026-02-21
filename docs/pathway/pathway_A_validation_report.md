# Pathway A Validation Report & Improvement Suggestions

**Date**: 2026-02-21
**Scope**: Review of `pathway_A_safe_deep_RL_1v1_PE.md` against 14 newly identified papers + 36 existing papers
**Update**: Updated 2026-02-21 with findings from reading all 13 PDFs (11 fully read, N12 behind Elsevier paywall, N09 partially read)

---

## 1. Overall Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Technical soundness** | 8/10 | Core architecture is well-supported; one critical CBF issue |
| **Literature coverage** | 9/10 | Comprehensive; 14 new papers found, none invalidate the approach |
| **Novelty claims** | 9/10 | Gap analysis remains valid; no competing work found |
| **Feasibility** | 8/10 | All components demonstrated individually; integration is the challenge |
| **Risk assessment** | 6/10 | Missing several important risks; mitigations too vague |
| **Evaluation plan** | 7/10 | Good metrics but missing critical ones; needs more baselines |

**Verdict**: The plan is **VALID and well-designed**. The core architecture (PPO + CBF-Beta + AMS-DRL + RCBF-QP) is the right approach. However, there are **3 critical issues** that must be addressed and **5 recommended improvements** that would significantly strengthen the work.

---

## 2. Critical Issues (MUST FIX)

### 2.1 Nonholonomic CBF Relative Degree Problem

**The Problem**: The plan defines CBFs as position-based functions (Section 3.3.1):
```
h_arena(x) = R_arena^2 - (x^2 + y^2)
h_obs(x) = (x - x_obs)^2 + (y - y_obs)^2 - r_safe^2
```

For unicycle dynamics (x-dot = v*cos(theta), y-dot = v*sin(theta)), these CBFs have **mixed relative degree**. The Lie derivative dh/dt depends on v and theta but NOT on omega. This means:
- The angular velocity omega cannot directly enforce the CBF constraint
- The CBF-QP may be infeasible when the robot is heading toward an obstacle but can't change direction fast enough
- Standard CBF theory breaks down for this system

**Evidence**: Paper N12 (Zhang & Yang, Neurocomputing 654, 2025) explicitly identifies and solves this problem for car-like mobile robots (CLMRs). They prove that position-based CBFs have mixed relative degree (Lemma 1) and that prior methods using HOCBFs are computationally intensive and overly conservative — they prioritize braking over steering (Remark 4).

**Fix**: Use **Virtual Control Point CBF** (from N12, adapted for unicycle):
```
# For unicycle: state = [x, y, θ], input = [v, ω]
# Place virtual point distance d ahead of robot (d ≈ 0.05m, must be nonzero)
q = [x + d*cos(θ), y + d*sin(θ)]

# CBF defined at virtual point:
h_obs(x) = ||q - p_obs||^2 - χ^2    # where χ ≥ d_safe + d

# q̇ depends on BOTH v and ω (uniform relative degree 1):
q̇ = [v*cos(θ) - d*ω*sin(θ),  v*sin(θ) + d*ω*cos(θ)]

# CBF condition becomes:
ḣ_obs + α(h_obs) ≥ 0
=> -(q - p_obs)^T * q̇ ≤ (α/2)*h_obs - ||q - p_obs||*v̄_obs

# For CLMR (with steering angle δ): use full M-matrix auxiliary input
# transformation τ = Mu to achieve q̇ = τ (see N12 Section 3.1)

# Key insight from N12 Remark 4: The M-matrix transformation prioritizes
# STEERING over BRAKING, unlike prior methods that reduce v to enforce safety.
# This dramatically improves obstacle avoidance efficiency.
```
This achieves relative degree 1 in both v and ω, making the CBF-QP well-posed. The parameter d controls the look-ahead distance and should be small but nonzero (N12 uses Δ=0.05m for a robot with l=0.3m wheelbase). The safety margin increases by d, which is negligible.

**Impact**: Without this fix, the CBF safety layer will not work correctly for unicycle dynamics. With it, the robot can steer around obstacles while maintaining velocity, rather than stopping when facing an obstacle head-on.

**Where to add**: Section 3.3.1 (Safety Constraints) and Appendix A.2 (CBF Condition).

---

### 2.2 CBF-QP Infeasibility Handling

**The Problem**: The risk table mentions "CBF infeasible in tight spaces" (Medium likelihood, Medium impact) with mitigation "Reduce safety margin; use CLBF as alternative." This is insufficient.

**Why it matters**: In adversarial PE, the opponent actively creates tight situations. CBF-QP infeasibility will happen regularly, not occasionally.

**Fix**: Implement a **three-tier infeasibility resolution strategy**:

**Tier 1 — Learned Feasibility Constraints (from N13, Xiao et al. ECC 2023)**:
N13 directly addresses this problem for the same unicycle dynamics as our PE problem. The key idea is to learn the feasibility boundary of the CBF-QP itself:
- Train classifiers (SVM or DNN) to learn functions H_j(z) >= 0 that approximate the feasible region of the CBF-QP
- Add these learned constraints to the RL training (not the QP itself) so the policy avoids states where the QP becomes infeasible
- **Feedback training loop**: iteratively retrain classifiers on new infeasible states found during RL training
- Results: infeasibility rate drops from **8.11% to 0.21% in just 3 iterations** of feedback training
- N13 handles two classes of unsafe sets relevant to PE: **Regular (circular — like arena walls)** and **Irregular (rectangular obstacles with corners)**
- This is more principled than the previous "reduce safety margin" suggestion because it teaches the policy to proactively avoid infeasible regions

**Tier 2 — Hierarchical Relaxation**:
```python
# Priority 1: Try exact CBF-QP
u_safe = solve_cbf_qp(u_rl, x, h_list)

if infeasible:
    # Priority 2: Relax least-important constraint
    u_safe = solve_relaxed_cbf_qp(u_rl, x, h_list,
                                    priority=[h_collision, h_arena, h_obs])
```

**Tier 3 — Backup Controller**:
```python
if still_infeasible:
    # Priority 3: Use backup controller (pure safety, ignore task)
    u_safe = backup_controller(x)  # e.g., brake + turn away from nearest obstacle
```

**Additional measures**:
- Log infeasibility events as a metric
- Report CBF feasibility rate in evaluation
- The N13 learned feasibility approach should be implemented in Phase 2 alongside the CBF-QP, as it operates during training (not deployment) and is complementary to the hierarchical relaxation

**Where to add**: Section 3.3.3 (RCBF-QP Safety Filter) — add subsection on infeasibility handling.

---

### 2.3 Update Isaac Sim to Isaac Lab

**The Problem**: The plan references "Isaac Sim" throughout (Sections 4.5, 6 Phase 4). The current NVIDIA framework is **Isaac Lab** (2025), which is a significant evolution:
- Isaac Lab is GPU-accelerated with massively parallel environments (thousands on one GPU)
- Modular architecture for custom environments
- Built-in domain randomization tools
- Native support for sensor simulation (lidar, cameras)

**Fix**: Replace all "Isaac Sim" references with "Isaac Lab" and update the software stack table accordingly.

**Where to add**: Sections 4.5, 6 (Phase 4), 9.

---

## 3. Recommended Improvements (HIGH VALUE)

### 3.1 Add BarrierNet / Differentiable QP Safety Layer (Phase 2.5)

**Current gap**: The plan uses CBF-Beta during training (Section 3.3.2) and RCBF-QP during deployment (Section 3.3.3). These are **different safety mechanisms**. The policy is trained with one safety layer but deployed with another, potentially causing a performance gap.

**Improvement**: After Phase 2, add a BarrierNet experiment:
```
Phase 2.5: BarrierNet Integration
- Implement differentiable QP safety layer (using cvxpylayers or qpth)
- Train policy end-to-end WITH the deployment safety layer
- Compare: CBF-Beta → RCBF-QP (current plan) vs BarrierNet end-to-end
- Measure: train-deploy performance gap
```

**References**: BarrierNet [N04] (T-RO 2023, MIT/Daniela Rus group), GCBF+ [N05] (T-RO 2024).
**Code**: https://github.com/Weixy21/BarrierNet

**Impact**: Could eliminate the training-deployment gap entirely. Both papers are from top robotics labs with hardware validation.

---

### 3.2 Add MACPO / MAPPO-Lagrangian as Baseline

**Current gap**: The baselines (Section 5.2) include DQN, DDPG, MADDPG, PPO variants, A-MPC+DQN, and MADR. Missing: **MACPO** — the standard algorithm for safe MARL.

**Improvement**: Add to the baselines table:

| Baseline | Source | Purpose |
|----------|--------|---------|
| MACPO | Paper N03 | Standard safe MARL baseline |
| MAPPO-Lagrangian | Paper N03 | Simpler safe MARL variant (soft constraints) |
| CPO (single-agent) | Achiam et al. 2017 | Foundational safe RL baseline |

**Concrete details from N03**:
- **Code publicly available**: https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation
- **Theorem 4.4** provides monotonic improvement guarantees in BOTH reward AND safety — directly relevant for validating our approach
- Created **SMAMuJoCo** and **SMARobosuite** benchmarks that we can use for pre-validation before our PE environment
- **MAPPO-Lagrangian** is a simpler alternative using soft constraints via Lagrangian relaxation — useful as a lower-complexity baseline

**Where to add**: Section 5.2 (Baselines).

---

### 3.3 Add Simultaneous Self-Play Comparison

**Current gap**: The plan only uses AMS-DRL (alternating, freeze-train). Paper N06 shows that simultaneous TD3 self-play (no freezing, both agents train concurrently) works well for 1v1 PE.

**Key findings from N06**:
- Confirms simultaneous TD3 works for PE without freezing either agent
- Uses the **same unicycle dynamics** as our PE problem: x-dot = v cos(theta), y-dot = v sin(theta), theta-dot = omega
- Three variants tested: normal, buffer zone (penalty region near boundary), and noisy actions (exploration aid)
- **Full observability only** — partial observability is not addressed, confirming this remains our novelty contribution
- **No safety constraints** — further confirms our novelty in combining PE with CBF safety

**Improvement**: Add to ablation studies (Section 5.3):

```
8. AMS-DRL (alternating) vs Simultaneous self-play vs MADDPG:
   Compare NE convergence speed, final performance, training stability
```

**Impact**: Simultaneous self-play is simpler to implement and may converge faster. If it works as well as AMS-DRL, it simplifies the training pipeline significantly.

---

### 3.4 Add Safety-Reward Shaping Term

**Current gap**: The reward design (Section 2.5) focuses on task performance (distance, capture, visibility). There's no incentive for the agent to learn inherently safe behavior.

**Improvement**: Add a CBF margin reward term:
```python
r_P = w1 * (d_prev - d_curr) / d_max          # distance reduction
      + w2 * capture_bonus * I(captured)        # capture reward
      + w3 * timeout_penalty * I(timeout)       # timeout penalty
      + w4 * visibility_bonus * I(evader_in_fov) # visibility
      + w5 * min(h_i(x)) / h_max               # CBF margin bonus (NEW)
```

**Recommended weight**: `w5 = 0.05` (small — incentivize but don't dominate)

**Evidence**: Paper [05] (Yang et al., Caltech 2025) shows that CBF reward shaping helps the policy learn to naturally satisfy safety constraints, reducing CBF filter intervention rate over time. This is exactly what the plan wants (Section 6, Phase 2 success criteria: "CBF filter intervention rate decreasing over training").

---

### 3.5 Expand Evaluation Metrics

**Current gap**: The metrics (Section 5.1) are good but miss several important measurements.

**Add to primary metrics**:

| Metric | Definition | Target |
|--------|-----------|--------|
| CBF-QP feasibility rate | % timesteps where CBF-QP is feasible | >99% |
| CBF margin distribution | Histogram of min h_i(x) values | Positive skew |
| Train-deploy gap | Performance difference CBF-Beta vs RCBF-QP | <5% |

**Add to secondary metrics**:

| Metric | Definition |
|--------|-----------|
| Strategy diversity | Number of distinct trajectory clusters (k-means) |
| GP disturbance accuracy | RMSE of GP predictions vs actual residuals |
| QP solve time | 95th percentile solve time (must be <50ms for 20Hz) |
| Backup controller activation rate | % timesteps using backup (should be ~0) |

---

## 4. Minor Improvements (NICE TO HAVE)

### 4.1 Consider Mixture of Beta Distributions

The plan uses Beta distributions for bounded actions. Beta is unimodal, which may be limiting for PE where bimodal actions are useful (e.g., "go left OR right around an obstacle"). Consider a mixture of 2-3 Beta distributions with learned mixing weights. This preserves the bounded support while allowing multimodality.

### 4.2 GP Cold-Start Protocol for Real Robot

The plan doesn't address how the GP disturbance model initializes on a new robot. Suggestion:
1. Pre-fill GP with simulation residual data (transfer learning)
2. Use conservative safety margin (larger kappa) for first 100 steps
3. Transition to normal margin as GP uncertainty decreases

### 4.3 Add POLICEd RL as Model-Free Alternative

POLICEd RL [N07] (RSS 2024) provides hard constraint satisfaction without needing a dynamics model. This is a valuable fallback if CBF construction proves difficult for complex obstacle shapes. Add as a future extension or ablation.

**Important limitations from reading N07**:
- Only works with **relative degree 1** constraints
- Only supports **affine constraints** (linear in state)
- Requires **deterministic dynamics** only
- These limitations mean POLICEd RL is a fallback for **simple cases only** (e.g., linear boundary constraints), not a general alternative to our CBF approach for circular obstacles or nonlinear safety boundaries

### 4.4 Neural CBF Fallback (PNCBF) — ELEVATED PRIORITY

If hand-crafted CBFs prove insufficient for complex environments, PNCBF [N02] provides a way to learn valid CBFs from a nominal policy. The key insight — **V^{h,pi} (the value function of max-over-time cost) IS a CBF** — is fundamental and elegant.

**Elevated to serious Phase 2 alternative** based on detailed reading of N02:
- **Hardware validated** on quadcopters (12D state space) with Boston Dynamics Spot as a dynamic obstacle — not just simulation
- **Converges in 2-3 policy iterations** — computationally practical
- **Scales to F-16 GCAS** (16D state space) — demonstrates applicability to high-dimensional systems
- The value function approach could actually be **BETTER than hand-crafted CBFs** for complex environments because it automatically captures the reachable set geometry without manual CBF design
- If hand-crafted virtual control point CBFs (Section 2.1) prove brittle in practice, PNCBF should be the **first alternative attempted**, not a last resort

### 4.5 Conflict-Averse Gradient for Safety-Task Trade-off (CASRL)

CASRL [Paper 30] separates safety and task objectives into dual critics with independent gradients. When the CBF reward shaping (Improvement 3.4) creates gradient conflicts with the pursuit objective, CASRL's conflict-averse optimization could help. Add as an ablation in Phase 3.

### 4.6 SB-TRPO as Alternative Safe Policy Optimizer

SB-TRPO [N14] (Feb 2026) provides an alternative safe policy optimization algorithm:
- Uses a **safety-biased trust region** via dynamic convex combination of cost and reward gradients with beta=0.75
- **No separate recovery phase** needed (unlike CPO which switches between reward optimization and constraint recovery)
- **10x cheaper than CPO**: uses Monte Carlo returns only, no learned critics
- Could be useful as an alternative safe RL baseline alongside CPO and MACPO, especially for quick prototyping due to lower computational cost

---

## 5. Updated Risk Assessment

| Risk | Likelihood | Impact | Mitigation (Updated) |
|------|-----------|--------|---------------------|
| CBF-Beta slows training convergence | Medium | Medium | Fall back to BarrierNet or CBF-QP filter; compare both |
| Self-play fails to converge to NE | Low | High | Use PSRO as fallback; add simultaneous self-play; exploitability check |
| Sim-to-real gap too large | **Medium-Low** | High | Aggressive DR in Isaac Lab; Gazebo intermediate step; GP online adaptation; **N10 validates exact pipeline (Isaac -> ONNX -> Gazebo -> Real) with 80-100% success; N11 confirms Isaac Lab native MARL support via PettingZoo + up to 1.6M FPS throughput** |
| **CBF infeasible in tight spaces** | **HIGH** | **HIGH** | **Three-tier: (1) N13 learned feasibility constraints during training (8.11%->0.21%), (2) hierarchical relaxation at deployment, (3) backup controller** |
| Real-time QP too slow | Low | Medium | Use closed-form CBF [05] or OSQP warm-starting; benchmark QP solvers |
| BiMDN belief encoder fails under fast dynamics | Low | Medium | Fall back to LSTM-only belief |
| GP disturbance model inaccurate | Low | Low | Increase GP data; ensemble GP; pre-fill from simulation |
| **Nonholonomic CBF relative degree issue** | **HIGH** | **HIGH** | **Virtual control point CBF (N12); validate in Phase 1** |
| **Train-deploy safety gap** | **Medium** | **Medium** | **BarrierNet end-to-end training; measure gap explicitly** |
| **CBF-safety + pursuit conflict** | **Medium** | **Medium** | **Safety-reward shaping (w5); conflict-averse gradient (CASRL)** |
| Self-play + CBF creates overly conservative policies | Medium | Medium | Reduce alpha over training; compare with unconstrained baseline |

---

## 6. Revised Phase Plan (Summary of Changes)

### Phase 1: Simulation Foundation (Months 1-2) — UPDATED
- Add: Validate virtual control point CBF on simple unicycle obstacle avoidance FIRST
- N10 provides a **ready-made reference implementation** for the sim-to-real pipeline (Isaac -> ONNX -> Gazebo -> Real) that can be adapted for Phase 4 planning during Phase 1 architecture design

### Phase 2: Safety Integration (Months 2-3) — MODIFIED
- Use virtual control point CBF instead of position-based CBF
- Add CBF-QP infeasibility handling (three-tier: learned feasibility from N13 + hierarchical relaxation + backup)
- Add safety-reward shaping term (w5)
- Add feasibility monitoring metrics
- Implement N13 learned feasibility constraints during RL training

### Phase 2.5: BarrierNet Experiment (Month 3) — NEW
- Implement differentiable QP safety layer
- Compare end-to-end training vs CBF-Beta + RCBF-QP pipeline
- Decision point: choose best safety architecture for remaining phases

### Phase 3: Partial Observability + Self-Play (Months 3-4) — MODIFIED
- Add simultaneous self-play comparison (alongside AMS-DRL)
- Add MACPO/CPO baselines
- Add strategy diversity metric

### Phase 4: Sim-to-Real Transfer (Months 4-6) — MODIFIED
- Update Isaac Sim -> Isaac Lab
- Isaac Lab's **native MARL support via PettingZoo Parallel API** (N11) simplifies multi-agent environment setup
- Add QP solver benchmarking (OSQP, qpOASES)
- Add GP cold-start protocol
- Add train-deploy gap measurement

### Phase 5: Analysis & Publication (Months 6-9) — UNCHANGED

---

## 7. New Papers Summary

| ID | Title | Relevance | Impact on Plan |
|----|-------|-----------|---------------|
| N01 | Learning CBFs Survey | HIGH | Background reference |
| N02 | PNCBF | **VERY HIGH** | Neural CBF fallback -> **serious Phase 2 alternative** if hand-crafted CBFs fail; hardware validated on quadcopters (12D), scales to F-16 (16D) |
| N03 | MACPO | HIGH | Add as baseline; code available; Theorem 4.4 (monotonic reward+safety improvement) |
| N04 | BarrierNet | VERY HIGH | Add Phase 2.5 experiment; code: https://github.com/Weixy21/BarrierNet |
| N05 | GCBF+ | MEDIUM | Future multi-agent extension |
| N06 | Self-play PE (TD3) | HIGH | Add simultaneous SP comparison; confirms no safety/partial obs in prior PE work |
| N07 | POLICEd RL | MEDIUM | Model-free hard constraint alternative (limited: relative degree 1, affine, deterministic only) |
| N08 | Statewise CBF Projection | MEDIUM | Alternative CBF implementation |
| N09 | Safe RL/CMDP Survey | MEDIUM | Positioning reference |
| N10 | Isaac Sim-to-Real | VERY HIGH | **Validates Phase 4 pipeline with real results (80-100% success)**; Isaac -> ONNX -> Gazebo -> Real |
| N11 | Isaac Lab | HIGH | **MARL-native support via PettingZoo Parallel API, up to 1.6M FPS** on multi-GPU |
| N12 | Nonholonomic CBF | **CRITICAL** | Fix CBF formulation (behind Elsevier paywall; confirmed by cross-reference) |
| N13 | CBF Feasibility | **VERY HIGH** | **Learned feasibility constraints — concrete solution for infeasibility** (8.11% -> 0.21% in 3 iterations) |
| N14 | SB-TRPO | MEDIUM | **Alternative safe policy optimizer (10x cheaper than CPO, no recovery phase)** |
| N15 | RMARL-CBF-SAM | **VERY HIGH** | **Closest prior work**: Robust MARL + neural CBFs + safety attention + reward shaping. But navigation only (not PE), double integrator (not unicycle), no self-play, no sim-to-real. Novelty claim intact. |

---

## 8. Conclusion

The Pathway A plan is fundamentally sound and represents a genuine contribution to the field. The three critical fixes (nonholonomic CBF, feasibility handling, Isaac Lab update) are essential for correctness. The five recommended improvements (BarrierNet, MACPO baseline, simultaneous self-play, safety-reward shaping, expanded metrics) would significantly strengthen both the technical contribution and the evaluation.

No paper was found that combines all the proposed components — the novelty claim stands. **N15 (RMARL-CBF-SAM)** is the closest prior work, combining robust MARL + neural CBFs + safety attention + reward shaping for multi-agent navigation. However, it differs from our approach in five critical ways: (1) navigation task, not pursuit-evasion game, (2) double integrator dynamics, not nonholonomic unicycle, (3) no self-play or game-theoretic training, (4) full state observation, not lidar-based partial observability, (5) no sim-to-real transfer. N15 validates several of our architectural choices (neural CBFs, safety reward shaping, attention mechanisms) while confirming that the PE + nonholonomic + partial obs + sim-to-real combination remains novel.

**Priority order for implementation**:
1. Fix virtual control point CBF (before any coding)
2. Add CBF feasibility handling
3. Add safety-reward shaping (easy, high value)
4. Implement BarrierNet comparison (Phase 2.5)
5. Add MACPO + simultaneous self-play baselines
6. Update Isaac Sim → Isaac Lab references

---

## 9. Phase 1 Spec Panel Review (2026-02-22)

An expert specification review panel (Wiegers, Fowler, Nygard, Crispin, Adzic) reviewed the Phase 1 specification and identified 3 critical, 5 high-priority, and 6 medium-priority issues. All have been resolved.

### 9.1 Issues Resolved

| ID | Issue | Resolution | Status |
|----|-------|-----------|--------|
| C1 | Unit tests (D9) classified as nice-to-have | Elevated to must-have; 15+ test minimum defined in §7.4 | FIXED |
| C2 | SB3 vs CleanRL and multi-agent interface not locked in | Locked: SB3 + Option B (custom wrapper) | FIXED |
| C3 | Qualitative acceptance criteria | Quantified all criteria with thresholds (§7.1) | FIXED |
| H1 | No concrete worked examples | Added 4 examples: capture scenario, obs vector, reward calc, SP oscillation (§7.5) | FIXED |
| H2 | No Phase 2 integration interfaces | Added SafetyFilter, RewardComputer, ObservationBuilder stubs (Session 1) | FIXED |
| H3 | No self-play health monitoring | Added SelfPlayHealthMonitor with entropy, diversity, rollback (Session 4) | FIXED |
| H4 | No VCP-CBF numerical edge cases | Added 6 edge case tests (F-K) in Session 6 | FIXED |
| H5 | Config YAML as nice-to-have | Elevated to must-have (D8) | FIXED |
| M1 | No Definition of Done | Added explicit gate criteria in §7.1 | FIXED |
| M2 | No seed management protocol | Added Appendix B with full SB3 reproducibility protocol | FIXED |
| M3 | Environment SRP | Split architecture: rewards.py, observations.py, wrappers.py | FIXED |
| M4 | Greedy baseline K_p unspecified | Set K_p = 3.0 with justification (Session 5) | FIXED |
| M5 | Time estimates aggressive | Increased Session 4 (2-3→3-4h), Session 6 (3-4→4-5h) | FIXED |
| M6 | No version pinning | Added requirements.txt with pinned versions (§9.1) | FIXED |

### 9.2 Quality Score Improvement

| Dimension | Before | After | Change |
|-----------|--------|-------|--------|
| Clarity | 8.5/10 | 9.0/10 | +0.5 (locked decisions, quantified criteria) |
| Completeness | 7.0/10 | 9.0/10 | +2.0 (tests, examples, interfaces, monitoring) |
| Testability | 6.5/10 | 9.0/10 | +2.5 (quantified criteria, 15+ tests, edge cases) |
| Consistency | 8.0/10 | 8.5/10 | +0.5 (aligned risk table with new mitigations) |
| Feasibility | 7.5/10 | 8.5/10 | +1.0 (realistic time estimates, reproducibility protocol) |
| **Overall** | **7.5/10** | **8.8/10** | **+1.3** |
