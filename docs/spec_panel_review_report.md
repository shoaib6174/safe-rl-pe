# Expert Specification Panel Review Report

**Project**: Safe Deep RL for 1v1 Pursuit-Evasion
**Review Date**: 2026-02-22
**Documents Reviewed**: 6 phase specifications + pathway document
**Review Mode**: Critique
**Expert Panel**: Karl Wiegers (Requirements), Martin Fowler (Architecture), Michael Nygard (Operations/Reliability), Lisa Crispin (Testing), Gojko Adzic (Specification by Example)

---

## Executive Summary

| Phase | Lines | Quality Score | Critical Issues | Major Issues | Minor Issues |
|-------|-------|--------------|----------------|-------------|-------------|
| Phase 1: Simulation Foundation | 2,077 | **8.8/10** | 0 | 4 | 7 |
| Phase 2: Safety Integration | 1,664 | **8.5/10** | 1 | 5 | 5 |
| Phase 2.5: BarrierNet Experiment | ~2,300 | **8.3/10** | 1 | 4 | 6 |
| Phase 3: Partial Observability | 2,246 | **8.4/10** | 0 | 5 | 6 |
| Phase 4: Sim-to-Real Transfer | 2,373 | **8.2/10** | 1 | 5 | 7 |
| Phase 5: Analysis & Publication | 2,791 | **8.0/10** | 0 | 4 | 7 |
| **Cross-Phase Consistency** | -- | **7.5/10** | 2 | 3 | 2 |

**Overall Assessment**: **8.2/10** -- Exceptionally detailed specification suite (13,451+ total lines) with implementation-ready code, formal acceptance criteria, and worked examples. The documentation quality significantly exceeds typical academic research project specifications. However, several cross-phase consistency issues, a critical compute planning gap, and unresolved technical decisions weaken the overall coherence.

**Verdict**: The specifications are **ready for implementation** with targeted fixes for critical and major issues identified below.

---

## 1. Per-Phase Expert Review

---

### Phase 1: Simulation Foundation (Score: 8.8/10)

**KARL WIEGERS -- Requirements Quality**:

> "This is one of the most thorough phase specifications I've encountered in a research context. The 11 must-pass criteria with quantified thresholds (capture rate > 80%, convergence < 1% over 5 windows, self-play balance in [0.35, 0.65]) are exemplary. The VCP-CBF validation as a critical gate is a particularly mature design decision -- identifying this as a kill-switch before investing in Phases 2-5 shows strong risk management."

Issues identified:

| Severity | Issue | Recommendation |
|----------|-------|----------------|
| MAJOR | **Boundary handling undefined**: Arena wall collision model (clipping, bouncing, penalty) is never specified despite being fundamental to the dynamics. | Define explicit wall collision behavior in Section 2.2. Recommend position clipping with velocity zeroing at boundaries. |
| MAJOR | **Timeout penalty sign confusion**: `w3 = -50.0` in formula + `self.timeout_penalty` in code risks double-negation errors. | Standardize: define `TIMEOUT_PENALTY = -50.0` as a single constant. Add a worked example showing the exact computation. |
| MAJOR | **v_max mismatch**: Training at 1.0 m/s vs TurtleBot3 at 0.22 m/s (~4.5x gap) with no bridging plan. | Add a note documenting the deliberate choice and the Phase 4 re-training strategy. |
| MAJOR | **Observation design inconsistency**: Uses both absolute positions AND relative features but transition to pure relative coordinates is unspecified. | Clarify: Phase 1 uses absolute+relative hybrid. Phase 3 transitions to relative-only. Document this explicitly. |
| MINOR | Session effort (20-27h) vs calendar timeline (Months 1-2) relationship undefined. | Add mapping: "7 sessions across ~6 weeks, allowing for debugging and iteration between sessions." |
| MINOR | Deliverable numbering jumps D12 to D13. | Renumber or add D12.5 explanation. |
| MINOR | No curriculum learning despite citing Paper [02] for it. | Add explicit note: "Curriculum learning deferred to Phase 3 by design." |

**GOJKO ADZIC -- Specification by Example**:

> "The 4 worked examples (capture scenario, observation vector, reward calculation, healthy vs unhealthy self-play) are excellent. They transform abstract requirements into concrete, verifiable scenarios. The self-play health monitoring example with 'healthy oscillation' vs 'unhealthy patterns' is particularly valuable for debugging."

Issues:

| Severity | Issue | Recommendation |
|----------|-------|----------------|
| MINOR | No worked example for boundary collision. | Add Example 5: robot at (19.8, 10.0) heading east with v=1.0, showing what happens at arena edge. |
| MINOR | Greedy baseline expected capture rate (30-50%) overlaps with random (50%). Could confuse comparison. | Add worked example showing why greedy underperforms in certain arena geometries. |

**LISA CRISPIN -- Testing Quality**:

> "The 15+ test suite with actual Python code is impressive. However, several tests directly access internal state (`env.pursuer_pos`) which may not be exposed through the wrapper API. This creates a fragile test dependency."

Issues:

| Severity | Issue | Recommendation |
|----------|-------|----------------|
| MINOR | Tests assume unwrapped env state access. | Add `env.unwrapped.pursuer_pos` pattern or provide a `get_state()` debug API. |
| MINOR | No performance regression test. | Add test: `assert step_time < 1e-3` for the <1ms per step requirement. |
| MINOR | wandb `reinit=True` loop pattern has no error handling for connection failures. | Add try/except with graceful degradation to TensorBoard-only mode. |

---

### Phase 2: Safety Integration (Score: 8.5/10)

**MARTIN FOWLER -- Architecture Quality**:

> "The safety architecture with training safety (CBF-Beta) and deployment safety (RCBF-QP) is well-separated. The 3-tier infeasibility handling from Paper [N13] is a mature pattern. However, the RCBF-QP with GP implementation is referenced but never detailed in any session -- this is a significant gap for the 'core novelty' phase."

Issues:

| Severity | Issue | Recommendation |
|----------|-------|----------------|
| CRITICAL | **Beta distribution log-prob missing Jacobian correction**: When rescaling from [0,1] to [low, high], the change-of-variables formula requires subtracting `log(high - low)`. Without this, policy gradients are incorrect, potentially causing training instability or convergence to wrong policies. | Add Jacobian correction: `log_prob -= torch.log(high - low).sum(dim=-1)`. This is a mathematical correctness issue, not a design choice. |
| MAJOR | **GP disturbance estimation never implemented**: GPyTorch listed as dependency but no session covers GP training, kernel selection, data collection, or integration with RCBF-QP. | Either add Session 8.5 for GP implementation, or explicitly defer to Phase 4 Session 12 and remove GPyTorch from Phase 2 dependencies. |
| MAJOR | **Discretization gap unaddressed**: CBF guarantees are continuous-time but implementation uses dt=0.05s discrete steps. No discrete-time CBF formulation is provided. | Add note on discrete-time CBF validity: for dt=0.05s and v_max=1.0, max displacement per step is 0.05m, which is within the 0.1m safety margin. Alternatively, use dCBFs as in Phase 2.5. |
| MAJOR | **Inter-robot collision CBF is one-sided**: Each agent's QP treats the opponent as a disturbance, but neither accounts for the other's constrained action. In tight scenarios, both agents may simultaneously try to avoid each other, creating oscillations. | Add note acknowledging this limitation. For Phase 1-2, the 0.3m collision radius with 0.5m capture radius provides sufficient margin. |
| MAJOR | **`compute_safe_bounds` left as `pass` stub**: This is the critical function bridging CBF-QP to Beta policy and it's not implemented. | Provide complete LP-based implementation or clearly mark as "Session 3, Step 3" with pseudocode. |
| MINOR | Zero-sum safety reward game-theoretic implications (evader incentivized to push pursuer into unsafe regions) not analyzed for degenerate dynamics. | Add note: "w5=0.05 is small enough that safety shaping does not dominate task reward. Monitor for degenerate behavior in ablation." |
| MINOR | Obstacle randomization per-episode vs per-training unclear. | Specify: "Obstacles randomized per episode reset." |
| MINOR | 27-34h effort for "Months 2-3" -- generous timeline includes compute time for 15 ablation runs. | Clarify the split between coding time and compute time explicitly. |
| MINOR | Infeasibility example mislabels Tier 2 relaxation as backup_activation_rate. | Fix metric categorization: Tier 2 events counted under `relaxation_rate`, not `backup_activation_rate`. |
| MINOR | 11 must-pass criteria but the table format differs from Phase 1's numbered table. | Standardize criterion formatting across phases. |

---

### Phase 2.5: BarrierNet Experiment (Score: 8.3/10)

**MARTIN FOWLER -- Architecture Quality**:

> "The comparison framework between CBF-Beta and BarrierNet is methodologically sound. The 4 decision rules with threshold-based outcomes (A through D) provide clear, pre-committed decision criteria -- this eliminates post-hoc rationalization. The weighted decision matrix is a best practice."

Issues:

| Severity | Issue | Recommendation |
|----------|-------|----------------|
| CRITICAL | **Arena boundary constraints are stubs**: The `build_constraints` method has a comment placeholder for arena boundary CBF constraints but no implementation. Arena boundary safety is critical -- without it, the BarrierNet QP only handles obstacles, not the arena walls. | Implement the 4 arena boundary VCP-CBF constraints in `build_constraints`, matching Phase 2 Session 2's implementation. |
| MAJOR | **Per-sample QP loop (Python for-loop over batch)**: cvxpylayers forward pass iterates over batch dimension in Python. For batch_size=2048, this will be extremely slow (~seconds), likely making the <10ms target impossible. | Implement batched solving using qpth as the primary path, with cvxpylayers as fallback for debugging. |
| MAJOR | **Log-prob Approach A (approximate) is default during high QP activity**: Early training has maximum QP intervention, exactly when the approximation is most biased. | Default to Approach B (Jacobian-based) or at minimum, add monitoring for the approximation error and a switch criterion. |
| MAJOR | **Evader safety architecture unspecified**: Does the evader use BarrierNet too, or CBF-Beta? Self-play fairness requires both agents to use the same safety mechanism. | Add explicit statement: "Both agents use the same safety architecture during self-play comparison." |
| MAJOR | **Statistical power with 3 seeds**: Welch's t-test with n=3 per group has very low statistical power (~25% for medium effects). The comparison may not detect real differences. | Increase to 5 seeds minimum. If compute-limited, acknowledge the power limitation and report effect sizes alongside p-values. |
| MINOR | Path reference `claudedocs/pathway_A_...` should be `docs/pathway/pathway_A_...`. | Fix path reference. |
| MINOR | `dt^2` term dropped in dCBF linearization. With dt=0.05 and v_max=1.0, the term is ~0.0025m which may matter near obstacles. | Add quantitative justification: "dt^2 * v_max = 0.0025m << safety_margin (0.1m), so dropping is safe." |
| MINOR | GPU/CPU contradiction: cvxpylayers ECOS does not support GPU but document mentions GPU targets. | Clarify: "QP solving is CPU-only; GPU used for policy forward pass and training." |
| MINOR | Adaptive gamma mentioned but no implementation provided. | Remove from scope or add stub with "Future work" label. |
| MINOR | Obstacle list stored per-timestep in buffer wastes memory. | Store obstacle list per-episode, index into it per-timestep. |
| MINOR | Inter-robot collision CBF mentioned in config (`n_constraints_max=8`) but not in `build_constraints`. | Either implement or reduce `n_constraints_max` to match actual constraints. |

---

### Phase 3: Partial Observability + Self-Play (Score: 8.4/10)

**KARL WIEGERS -- Requirements Quality**:

> "The 11 must-pass criteria are well-defined with quantitative thresholds. The NE gap < 0.10, exploitability < 0.15, and BiMDN +15% improvement targets are all measurable and testable. The health monitoring system (entropy collapse, DTW diversity, checkpoint rollback, catastrophic forgetting) demonstrates exceptional operational maturity."

Issues:

| Severity | Issue | Recommendation |
|----------|-------|----------------|
| MAJOR | **Curriculum threshold mismatch**: Code checks `min(capture_rate, escape_rate) > 0.35` but text says "capture rate > 70% for both agents." These are fundamentally different criteria. | Fix code to match text: `min(capture_rate, escape_rate) > self.advancement_threshold` (0.70). |
| MAJOR | **BiMDN pre-training data collection unspecified**: Pre-training requires trajectories with known ground truth, but no session covers data collection. | Add to Session 2: "Collect 50K transitions using Phase 2 trained agents with full-state recording for BiMDN supervised pre-training." |
| MAJOR | **Simultaneous self-play uses TD3 (off-policy) per Paper [N06] but integration with SB3 PPO/TD3 APIs is vague**: The `SimultaneousSelfPlay` class shows a step loop but actual SB3 integration is not detailed. | Add implementation specifics for both PPO-simultaneous and TD3-simultaneous variants with SB3 API calls. |
| MAJOR | **NE gap definition unclear on timeouts**: |SR_P - SR_E| uses success rates, but timeout episodes are neither capture nor escape. | Define: "Timeout counts as evader success (survival). SR_P = capture_rate, SR_E = 1 - capture_rate." |
| MAJOR | **MACPO integration under-specified**: "Clone GitHub repo and adapt to PE environment" omits the significant engineering work of adapting custom obs/action spaces, reward/cost signals, and multi-agent wrapper. | Add detailed MACPO integration steps: observation adapter, cost function definition (CBF violations), action space mapping. |
| MINOR | Phase 2.5 dependency creates architecture uncertainty: safety actor head is `...  # From chosen safety architecture`. | Add conditional implementation: "If CBF-Beta: use SafeBetaActor. If BarrierNet: use BarrierNetActor." |
| MINOR | SB3 `policy.log_std` access is an implementation detail that may break if policy architecture changes. | Access via `policy.action_dist.log_std` or add version check. |
| MINOR | DTW O(n^2) cost on 50 trajectories not profiled. | Add estimate: "50 trajectories x 50 timesteps x 2D: ~50ms per pairwise DTW, ~62.5s total for 1,225 pairs. Run every 50K steps." |
| MINOR | `_ray_boundary_intersect` method referenced but not implemented in sensor code. | Provide implementation (ray-line intersection with arena boundary). |
| MINOR | No compute resource specification (GPU model, RAM for training). | Add: "Requires 1x GPU with >= 8GB VRAM. Lab PC niro-2 or equivalent." |
| MINOR | Observation space 352D (256 lidar + 64 state + 32 belief) validated against SB3 integration? | Add note: "SB3 PPO supports arbitrary Box observation spaces. 352D is within typical range." |

---

### Phase 4: Sim-to-Real Transfer (Score: 8.2/10)

**MICHAEL NYGARD -- Operations & Reliability**:

> "The staged pipeline (Isaac Lab -> ONNX -> Gazebo -> Real) is the correct approach, validated by Paper [N10]. The ROS2 node architecture with explicit timing budgets (21ms used of 50ms budget) shows strong operational thinking. The GP cold-start protocol addresses a real deployment concern that most papers ignore. However, the specification is optimistic about hardware readiness and iteration time."

Issues:

| Severity | Issue | Recommendation |
|----------|-------|----------------|
| CRITICAL | **No iteration buffer in 18-session plan**: Real sim-to-real transfer requires multiple debugging cycles. The linear session plan has no time for "Session 9 failed, need to revisit Session 7." | Add 3-4 buffer sessions explicitly. Restructure as: "Sessions 1-6 (sim), Sessions 7-9 (Gazebo), Sessions 10-13 (safety), Sessions 14-16 (real robot), Sessions 17-18 (eval), Sessions 19-21 (iteration buffer)." |
| MAJOR | **Hardware availability assumed**: Two TurtleBot4s (~$2,400 total) with procurement timeline not addressed. | Add prerequisite: "Hardware procurement must begin in Month 4 (during Phase 3) to ensure availability by Month 6." |
| MAJOR | **v_max not resolved**: DR table centers on 1.0 m/s but TurtleBot4 max is 0.3 m/s. "Re-train with matched v_max" is mentioned but no session covers this re-training. | Add explicit session: "Session 4b: Re-train with v_max=0.3 m/s, arena scaled to 6x6m. Compare with 1.0 m/s model." |
| MAJOR | **Isaac Lab GPU requirement**: Requires NVIDIA GPU with CUDA 11.8+ and Isaac Sim 4.2+, but lab PC (niro-2) GPU capabilities are not verified. | Add prerequisite check: verify niro-2 GPU is compatible (RTX 2060+ minimum). |
| MAJOR | **BiMDN ONNX export**: LSTM/GRU export to ONNX is acknowledged as complex with "export separately if needed" -- this handwave could become a blocker. | Add dedicated sub-step in Session 5: "Export BiMDN with explicit hidden state management. Test with sequential inference loop." |
| MINOR | Two-robot communication (direct odom sharing vs lidar detection vs external tracking) not committed. | Commit: "Phase 4 uses shared odometry via ROS2 topics. Real-world lidar detection is future work." |
| MINOR | Gazebo Fortress version compatibility not verified with ROS2 Humble. | Add version check step in Session 7. |
| MINOR | GP multi-output handling: single GP or separate per-dimension? | Specify: "3 independent single-output GPs for (dx, dy, dtheta) using shared kernel hyperparameters." |
| MINOR | GP ARD input dimensions: Session 12 uses 6D but Session 13 data generation produces 5D. | Standardize to 5D: `[x, y, theta, v, omega]`. Remove time dimension or add it to data generation. |
| MINOR | ONNX Runtime on ARM (RPi4) may need source compilation. | Add time estimate: "ONNX RT ARM build: allow 2-4h for compilation if prebuilt unavailable." |
| MINOR | Arena scaling from 20x20m to 6x6m not validated against obstacle placement and dynamics. | Add validation: "6x6m arena with 0.3 m/s v_max preserves time-scale ratio of 20x20m at 1.0 m/s." |
| MINOR | Session 9 evaluates without RCBF-QP safety filter, which could mask integration issues. | Add sub-step: "Session 9b: repeat evaluation WITH RCBF-QP active in Gazebo." |

---

### Phase 5: Analysis & Publication (Score: 8.0/10)

**GOJKO ADZIC -- Specification by Example**:

> "The 4 formal theorems with numerical verification scripts, 3 worked examples for statistical analysis, and pre-committed reviewer response strategies are outstanding. The metrics collection reference in Appendix C with expected ranges and red flags transforms abstract statistical concepts into concrete, verifiable procedures."

**KARL WIEGERS -- Requirements Quality**:

> "The 4-gate progressive validation (analysis -> figures -> papers -> submission) is a well-structured milestone system. However, the tension between the hard target of p < 0.05 with Holm-Bonferroni correction and the practical constraint of 5 seeds is a specification defect -- the worked example itself demonstrates this target may be unachievable."

Issues:

| Severity | Issue | Recommendation |
|----------|-------|----------------|
| MAJOR | **Statistical power insufficient**: Worked Example 1 shows 5 seeds + 12-comparison Bonferroni correction yields p_adj = 0.276 for Cohen's d = 1.62 (a very large effect). The hard target of p < 0.05 corrected is likely unachievable. | Increase to 10 seeds for key comparisons, OR switch to uncorrected p-values with explicit multiple-testing caveat, OR use FDR (Benjamini-Hochberg) instead of FWER (Bonferroni). |
| MAJOR | **Compute budget never aggregated**: ~150 ablation runs + 60+ baseline runs + 5-10 seed replication. At ~2-5 GPU-hours per run, this is 420-1,050+ GPU-hours -- potentially weeks of compute. | Add compute budget table: "Total estimated: X GPU-hours on [specific GPU]. Timeline: Y weeks of continuous training. Prioritized if budget exceeded." |
| MAJOR | **Theorem 4 weakness**: Capture convergence bound holds only in expectation conditioned on NE strategy profile. CDC/L4DC reviewers may find this insufficient. | Add discussion: "Theorem 4 provides a bounded expected capture time rather than almost-sure convergence, which is the strongest result achievable under adversarial evasion." Add fallback: "If Theorem 4 is deemed insufficient, Paper 3 scope reduces to Theorems 1-3." |
| MAJOR | **Real-robot trial placeholders**: Reviewer response strategy uses "N", "M", "K" for trial counts. | Define: "N = 50 episodes (minimum from Phase 4), M = 5 seed-equivalent runs, K = 3 challenge scenarios." |
| MINOR | Duplicate "Step 3" in Session 6 (lines 922-931). | Renumber Step 3b. |
| MINOR | MADR baseline requires "DeepReach + adversarial MPC (use official code if available)" -- uncertain availability. | Add fallback: "If MADR implementation unavailable, replace with additional game-theoretic baseline (e.g., HJ-reachability via helperOC)." |
| MINOR | BibTeX entries use "and others" instead of full author lists. | Complete all author lists before submission. |
| MINOR | Phase 2.5 conditionality: Ablation 10 (CBF-Beta vs BarrierNet) "if implemented." | Make unconditional: "Report Phase 2.5 comparison results regardless of outcome." |
| MINOR | Security: CLAUDE.md contains plaintext password -- code release Section 6.6 warns about this but project itself violates the rule. | Remove password from CLAUDE.md immediately. Use SSH key authentication or environment variables. |
| MINOR | No contingency for poor Phase 4 data quality. | Add: "If real-robot trials < 30 episodes, Paper 1 focuses on simulation results; Paper 2 (journal) includes real-robot with extended data collection." |
| MINOR | 14 sessions in 3 months is ambitious alongside potential paper revisions. | Add buffer: "Months 10-11 are revision buffer for rejected/revision-requested papers." |

---

## 2. Cross-Phase Consistency Analysis

### Dependency Chain Validation

```
Phase 1 ──> Phase 2 ──> Phase 2.5 ──> Phase 3 ──> Phase 4 ──> Phase 5
 (Mo 1-2)   (Mo 2-3)    (Mo 4)       (Mo 4-6)    (Mo 6-8)    (Mo 8-10)
```

| Dependency | Status | Issues |
|------------|--------|--------|
| Phase 1 -> Phase 2 | **Well-defined** | Phase 1 Section 10 provides explicit transition guide. Interface stubs (SafetyFilter, RewardComputer, ObservationBuilder) designed for extension. |
| Phase 2 -> Phase 2.5 | **Well-defined** | 8 specific artifacts listed. Clear skip condition: "If train-deploy gap < 5%, proceed directly to Phase 3." |
| Phase 2.5 -> Phase 3 | **Conditional** | Phase 3 safety architecture depends on Phase 2.5 decision. Code uses `...  # From chosen safety architecture` placeholder. Creates implementation uncertainty. |
| Phase 3 -> Phase 4 | **Well-defined** | ONNX export, C++ RCBF-QP, real sensors clearly specified. Isaac Lab transition documented. |
| Phase 4 -> Phase 5 | **Well-defined** | Section 9 of Phase 4 lists all Phase 5 inputs. Data collection requirements match Phase 5's analysis needs. |

### Cross-Phase Issues

| Severity | Issue | Phases Affected | Recommendation |
|----------|-------|----------------|----------------|
| CRITICAL | **v_max inconsistency propagates across all phases**: Phase 1 trains at 1.0 m/s, Phase 4 deploys at 0.3 m/s, but no phase explicitly handles the re-training with matched dynamics. DR in Phase 4 centers on 1.0 m/s. | 1, 4, 5 | Add explicit re-training session in Phase 4. Update DR table to center on robot's actual v_max. Back-propagate normalized dynamics into Phase 1 design. |
| CRITICAL | **Total compute budget never aggregated**: Phase 2 (30-75 GPU-h), Phase 2.5 (~40 GPU-h), Phase 3 (~30 GPU-h training), Phase 4 (Isaac Lab training, unknown), Phase 5 (420-1,050+ GPU-h). Total: potentially **600-1,200+ GPU-hours**. | All | Create a compute budget appendix in the pathway document. Identify GPU resources available (niro-2 specs). Establish priority ordering if budget is exceeded. |
| MAJOR | **Phase 2.5 conditionality creates cascading uncertainty**: If BarrierNet is chosen, Phases 3-5 need different actor architectures, different log-prob computations, different ablation designs. The specifications handle this with placeholders but not full conditional implementations. | 2.5, 3, 4, 5 | Pre-commit to CBF-Beta as the default path. Document BarrierNet as the "upgrade" path with explicit diff patches for each subsequent phase. |
| MAJOR | **Test framework inconsistency**: Phase 1 uses inline pytest code, Phase 2.5 has numbered test IDs (UT-01), Phase 5 has lettered tests. No shared test infrastructure design. | All | Standardize test naming: `test_{phase}_{category}_{number}`. Create shared fixtures in `tests/conftest.py`. |
| MAJOR | **Seed count varies**: Phase 1 uses 3 seeds, Phase 2 uses 3 seeds, Phase 2.5 uses 3 seeds, Phase 5 requires 5-10 seeds. Earlier phases should use 5 seeds to match Phase 5's requirements. | 1, 2, 2.5, 5 | Standardize on 5 seeds minimum across all phases. |
| MINOR | Timeline overlap: Phase 2 ends Month 3, Phase 2.5 starts Month 4, Phase 3 starts Month 4. Parallel execution of 2.5 and 3 not addressed. | 2.5, 3 | Clarify: "Phase 2.5 and Phase 3 Session 1 (sensors) can proceed in parallel." |
| MINOR | wandb project naming not standardized across phases. | All | Define: `project="safe-pe"` with `group=f"phase{N}"` and `tags=[phase, session, config]`. |

---

## 3. Expert Consensus Findings

### Strengths (Unanimous Agreement)

1. **Exceptional detail level**: 13,451+ total lines of specification with implementation-ready code snippets, making this more of an "implementation guide" than a specification. Most research projects lack even 10% of this detail.

2. **Quantified acceptance criteria**: Every phase has measurable, testable success criteria with specific numerical thresholds. The VCP-CBF validation as a Phase 1 gate is a standout risk mitigation strategy.

3. **Worked examples**: Concrete numerical examples (Phase 1: 4 examples, Phase 2: 3 examples, Phase 3: 5 examples, Phase 5: 3 examples) transform abstract requirements into verifiable scenarios.

4. **Risk-aware design**: VCP-CBF validation gate, 3-tier infeasibility handling, BarrierNet comparison as a controlled experiment, GP cold-start protocol, staged sim-to-real pipeline -- all demonstrate mature engineering judgment.

5. **Traceability**: Papers are cited with specific contributions mapped to implementation decisions. The 51-paper literature base is well-integrated.

6. **Reproducibility infrastructure**: Pinned dependency versions, seed management protocols, Hydra configuration, wandb experiment tracking, Docker release plan.

### Weaknesses (Unanimous Agreement)

1. **No code exists yet**: All 13,000+ lines are specification, not implementation. The gap between specification and reality is untested. Session time estimates may be optimistic.

2. **Compute planning absent**: No aggregated compute budget across phases. For a project requiring potentially 1,000+ GPU-hours, this is a significant planning oversight.

3. **Statistical methodology tension**: The Holm-Bonferroni correction with 5 seeds is demonstrated (in the spec's own worked example) to be insufficient for detecting even large effects. This undermines the publication viability.

4. **Hardware assumptions**: TurtleBot4 procurement, lab PC GPU capabilities, and compute infrastructure are assumed but not verified.

5. **Over-specification risk**: Some code snippets in the spec may be premature -- they encode design decisions that should emerge from implementation. If the actual architecture diverges (e.g., SB3 API changes), significant spec rewrites would be needed.

---

## 4. Prioritized Recommendations

### Priority 1: Fix Before Starting Implementation -- ALL RESOLVED

| # | Action | Status | What Was Done |
|---|--------|--------|---------------|
| 1 | **Fix Beta log-prob Jacobian correction** (Phase 2) | FIXED | Added `- torch.log(self.high - self.low).sum(dim=-1)` to `BetaDistribution.log_prob()` + updated worked Example 2 with Jacobian math |
| 2 | **Define wall collision model** (Phase 1) | FIXED | Added position clipping + velocity zeroing model to `dynamics.py` spec in Session 1, with rationale and test case |
| 3 | **Implement arena boundary constraints** (Phase 2.5) | FIXED | Ported 4 VCP-CBF wall constraints (left/right/bottom/top) from Phase 2 into `build_constraints`, updated `n_constraints_max` default from 6 to 8 |
| 4 | **Standardize seed count to 5** (All phases) | FIXED | Updated 25+ references across Phases 1-4: success criteria, ablation configs, reproducibility protocols, seed lists, run counts (15->25 in Phase 2) |
| 5 | **Create compute budget appendix** (Pathway doc) | FIXED | Added Appendix B with per-phase GPU-hour estimates (465-1,130 total), hardware requirements, timeline mapping, prioritization strategy, multi-GPU guidance |

### Priority 2: Fix During Implementation -- ALL RESOLVED

| # | Action | Status | What Was Done |
|---|--------|--------|---------------|
| 6 | **Add iteration buffer sessions to Phase 4** | FIXED | Added Sessions 19-21 (Sim-to-Gazebo iteration, Gazebo-to-Real iteration, End-to-End polish) with detailed objectives and fallback usage |
| 7 | **Implement `compute_safe_bounds` in Phase 2** | FIXED | Replaced `pass` stub with full LP-based implementation using scipy.optimize.linprog (4 LPs for tight box bounds) + analytical alternative for speed-critical paths |
| 8 | **Fix curriculum threshold mismatch in Phase 3** | FIXED | Corrected `check_advancement()` to use `capture_rate > threshold` (not `min(capture, escape) > threshold * 0.5`). Added docstring explaining why `min` criterion is impossible (capture + escape = 1). Updated config comment. |
| 9 | **Specify BiMDN pre-training data collection** | FIXED | Added `collect_bimdn_pretraining_data()` function in Phase 3 Session 2 with full implementation: 500 episodes using Phase 2 policies, ~200-300K samples, 80/20 split, .npz storage |
| 10 | **Add v_max re-training to Phase 4** | FIXED | Expanded Session 4 with matched dynamics: TurtleBot4 v_max=0.3 m/s, 6x6m arena, scaled VCP-CBF params, DR centered on real v_max. Updated Session 14 arena setup for consistency. |

### Priority 3: Fix Before Publication

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 11 | Switch from Holm-Bonferroni to Benjamini-Hochberg FDR control OR increase to 10 seeds. | 1 hour (spec) | Enables statistically significant claims |
| 12 | Complete BibTeX entries (replace "and others"). | 30 min | Publication requirement |
| 13 | Remove plaintext password from CLAUDE.md. | 5 min | Security fix before code release |
| 14 | Pre-commit to default safety architecture path. | 2 hours (spec) | Resolves Phase 2.5 cascading uncertainty |

---

## 5. Phase Gate Readiness Assessment

| Phase | Ready to Implement? | Blocking Issues | Recommended Pre-Work |
|-------|---------------------|-----------------|---------------------|
| Phase 1 | **YES** | None critical | Fix wall collision model definition |
| Phase 2 | **YES** (with fix) | Beta log-prob Jacobian | Fix Jacobian, clarify GP deferral |
| Phase 2.5 | **YES** (with fix) | Arena boundary stubs | Implement constraints before starting |
| Phase 3 | **YES** (with fixes) | Curriculum threshold, BiMDN data | Fix code/text mismatch, add data collection plan |
| Phase 4 | **CONDITIONAL** | Hardware availability, GPU verification | Verify niro-2 GPU, begin procurement |
| Phase 5 | **CONDITIONAL** | Statistical methodology, compute budget | Resolve seed count and correction method |

---

## 6. Summary Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Requirements Clarity** | 9.0/10 | Quantified thresholds, measurable criteria, worked examples |
| **Architecture Quality** | 8.5/10 | Clean separation of concerns, well-defined interfaces; GP and BarrierNet gaps |
| **Testability** | 8.5/10 | 100+ tests defined with code; inconsistent naming conventions |
| **Completeness** | 8.0/10 | Compute budget missing, hardware unverified, some stubs |
| **Consistency** | 7.5/10 | v_max, seed counts, test naming, and Phase 2.5 conditionality |
| **Feasibility** | 7.5/10 | Ambitious scope (10 months, 3 papers, real robots); no implementation started |
| **Specification Quality** | 8.5/10 | Exceptional detail; some over-specification risk |
| **Overall** | **8.2/10** | Publication-grade specification suite ready for implementation with targeted fixes |

---

*Report generated by Expert Specification Panel (Wiegers, Fowler, Nygard, Crispin, Adzic)*
*Review methodology: Multi-expert critique mode with cross-phase consistency analysis*
