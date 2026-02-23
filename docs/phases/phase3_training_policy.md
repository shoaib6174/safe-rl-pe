# Phase 3 Training Policy — Staged Validation Protocol

**Date**: 2026-02-22
**Purpose**: Prevent wasting 220-400 GPU hours on a broken pipeline. Validate each component before scaling.

**Philosophy**: Never run a 5-seed experiment on something that hasn't passed a 1-seed smoke test. Never run a 1-seed full run on components that haven't passed unit validation.

---

## Pipeline Overview

```
Stage -1: DCBF Theorem       (~0h compute, ~2h dev)    ← formal safety proof verified
Stage 0: Unit Tests           (~0h compute, ~2h dev)    ← components work in isolation
Stage 1: BiMDN Pre-training   (~1h compute)             ← belief encoder works
Stage 2: Single-Agent Valid.  (~3-4h compute)            ← partial obs + DCBF pipeline works end-to-end
Stage 3: Self-Play Smoke      (~3-4h compute)            ← AMS-DRL doesn't collapse
Stage 4: Single Seed Full     (~8-15h compute)           ← full curriculum, NE convergence, health monitoring
Stage 5: Full Experiment      (~200-350h compute)        ← 5 seeds, baselines, ablations, asymmetric runs
Post-5:  Evaluations          (~2-5h compute)            ← human evader, generalization, interpretability
                               ─────────────────
                     Validation budget: ~15-23h (5-8% of total)
```

Each stage is a **gate**. Do NOT proceed to the next stage until the current gate passes.

---

## Stage -1: DCBF Safety Theorem Verification

**Compute**: 0 GPU hours (math + CPU verification only)
**Dev time**: ~2h
**What**: Prove that the DCBF post-hoc filter with gamma=0.2 guarantees zero physical collisions under stated assumptions, then verify numerically.

### Protocol

1. **Write formal theorem** (`docs/proofs/dcbf_safety_theorem.md`):
   - **Theorem (DCBF Safety Guarantee)**: Given the unicycle dynamics with VCP offset d > 0, and the DCBF filter with decay rate gamma=0.2, if the QP is feasible at all states along the trajectory and the initial state satisfies h_i(x_0) >= 0 for all constraints i, then h_i(x(t)) >= 0 for all t >= 0 (forward invariance of the safe set).
   - State all assumptions explicitly:
     - A1: Unicycle dynamics with bounded inputs (v_max, omega_max)
     - A2: VCP offset d > 0 (ensures relative degree 1)
     - A3: DCBF class-K function alpha(h) = gamma * h with gamma = 0.2
     - A4: QP feasible at all reachable states (verified numerically)
     - A5: Continuous-time CBF condition discretized at dt=0.05s with sufficient margin
   - **Proof**: Use Nagumo's theorem + comparison principle. At the boundary h_i = 0, the DCBF constraint h_i_dot + gamma * h_i >= 0 reduces to h_i_dot >= 0, ensuring the vector field points inward. By the comparison principle, h_i(t) >= h_i(0) * exp(-gamma * t) >= 0.
   - **Corollary (Zero Physical Collisions)**: If the CBF safety margin chi includes the robot radius r_robot, obstacle radius r_obs, and VCP offset d (i.e., chi >= r_robot + r_obs + d + margin), then forward invariance of h >= 0 implies no physical collisions.

2. **Numerical verification** (`scripts/verify_dcbf_theorem.py`):
   ```bash
   ./venv/bin/python scripts/verify_dcbf_theorem.py \
     --n_states 10000 \
     --gamma 0.2 \
     --d_vcp 0.1 \
     --output results/dcbf_verification/
   ```
   - Sample 10,000 random states in the arena
   - For each state, verify the DCBF-QP is feasible
   - For 1,000 random trajectories (100 steps each), verify h(t) >= 0 throughout
   - Report feasibility rate and violation count

### Gate -1 Criteria

| Check | Pass | Fail Response |
|-------|------|---------------|
| Theorem statement complete with all assumptions | All 5 assumptions stated | Identify missing assumptions |
| Proof follows from Nagumo + comparison principle | Proof is logically sound | Consult [N12], [06] for reference proofs |
| Numerical feasibility rate | > 99.9% (10,000 states) | Identify infeasible states; adjust safety margins |
| Trajectory violation count | 0 violations in 1,000 trajectories | Debug DCBF implementation or tighten gamma |
| Discretization error bounded | |h_continuous - h_discrete| < epsilon | Reduce dt or add discretization margin |

### If Gate -1 Fails

- **QP infeasible at some states**: These are typically corner states where multiple constraints conflict. Add constraint prioritization or enlarge arena slightly.
- **Trajectory violations found**: The discretization error is too large. Either reduce dt from 0.05s to 0.025s, or add a discretization safety margin term to the DCBF constraint.
- **Proof assumptions too restrictive**: Document the gap between theory and practice. The paper can state "zero violations observed empirically" even if the formal guarantee requires stronger assumptions.

---

## Stage 0: Component Unit Tests

**Compute**: 0 GPU hours (CPU only)
**Dev time**: ~2h
**What**: Run the 31-test suite from Phase 3 spec §7.4

### Gate 0 Criteria

| Check | Pass | Fail Response |
|-------|------|---------------|
| All 31 tests (A-Z + B2, L2, W2, W3) green | All pass | Fix failing tests before proceeding |
| FOV sensor: detect/miss correct | Tests A, B | Debug sensor geometry |
| Lidar: obstacle + boundary detection | Tests C, D | Debug ray casting |
| BiMDN: valid mixture params, gradient flows | Tests E, F, G | Fix architecture |
| BiMDN: multimodality on synthetic data | Test H | Fix mixture loss or increase n_mixtures |
| Partial-obs policy: forward pass, shapes | Tests I, K | Fix feature extractor dimensions |
| AMS-DRL: cold-start + phases don't crash | Test J | Fix training loop |
| DCBF: receives true state, policy gets partial obs | Test W | Fix state routing |
| Pluggable encoder interface | Test X | Fix encoder abstraction |
| End-to-end: 1000 steps no crash | Test V | Fix integration |
| Curriculum manager: level advancement | Test L | Fix advancement logic |
| MAPPO-Lagrangian: 10K steps no crash | Test M | Fix Lagrange multiplier |
| Health monitoring: entropy + rollback | Tests N, O | Fix callbacks |

### How to Run
```bash
./venv/bin/python -m pytest tests/test_sensors.py tests/test_bimdn.py tests/test_selfplay.py tests/test_phase3_integration.py -v
```

---

## Stage 1: BiMDN Pre-training

**Compute**: ~1 GPU hour
**What**: Collect dataset + pre-train BiMDN + validate belief quality

### Protocol

1. **Collect data** (30 min):
   ```bash
   ./venv/bin/python scripts/collect_bimdn_data.py \
     --n_episodes 500 \
     --policy_mix phase2:0.3,random:0.3,scripted:0.4 \
     --output data/bimdn_pretrain.npz
   ```
   - Verify dataset size: expect ~200-300K samples
   - Verify both "target visible" and "target lost" scenarios present

2. **Pre-train** (20 min):
   ```bash
   ./venv/bin/python scripts/pretrain_bimdn.py \
     --data data/bimdn_pretrain.npz \
     --epochs 50 --lr 1e-3 \
     --output models/bimdn_pretrained.pt
   ```

3. **Validate** (10 min):
   ```bash
   ./venv/bin/python scripts/validate_bimdn.py \
     --model models/bimdn_pretrained.pt \
     --data data/bimdn_pretrain.npz
   ```

### Gate 1 Criteria

| Metric | Target | Hard Fail | Fail Response |
|--------|--------|-----------|---------------|
| Training loss | Final < 50% of epoch-1 loss | Loss doesn't decrease for 10 epochs | Check lr, data quality, loss function |
| In-FOV RMSE | < 2.0m | > 4.0m | BiMDN isn't using sensor input — debug forward pass |
| Out-of-FOV RMSE | < 5.0m | > 7.0m (barely better than random 8.2m) | Increase LSTM hidden dim, check obs history encoding |
| n_eff (target visible) | ~1.0 (unimodal) | > 3.0 | Mixture not collapsing to single mode when certain |
| n_eff (target lost) | > 1.5 (multimodal) | < 1.1 (always unimodal) | Loss not encouraging multimodality — check NLL vs MSE |
| Val loss | Within 20% of train loss | > 2× train loss | Overfitting — add dropout, reduce epochs |

### If Gate 1 Fails
- **Loss doesn't converge**: Try lr=3e-4, increase hidden_dim to 128, check data normalization
- **RMSE too high**: Inspect data — are ground truth positions correct? Is obs history properly stacked?
- **No multimodality**: Switch from NLL loss to explicit multimodal loss. Or accept unimodal and use LSTM fallback (Test X encoder interface)
- **After 2h of debugging with no fix**: Fall back to **LSTMEncoder** (simpler, no mixture, just mean prediction). Continue to Stage 2 with LSTM. BiMDN becomes an ablation study, not a dependency.

---

## Stage 2: Single-Agent Validation

**Compute**: ~3-4 GPU hours (3 short training runs)
**What**: Train pursuer with partial obs against FIXED random evader. Validates the full observation-to-action pipeline without self-play complexity.

### Protocol

Run 3 experiments (can run in parallel if GPU memory permits):

| Run | Config | Purpose | Steps |
|-----|--------|---------|-------|
| **2A** | PPO + BiMDN + DCBF + partial obs | Full pipeline | 500K |
| **2B** | PPO + raw obs history (no BiMDN) + DCBF + partial obs | BiMDN ablation | 500K |
| **2C** | PPO + BiMDN + NO DCBF + partial obs | DCBF ablation | 500K |

```bash
# Run 2A
./venv/bin/python scripts/train_single_agent.py \
  --pursuer_obs partial --belief bimdn --dcbf gamma=0.2 \
  --evader random --steps 500000 --seed 42 \
  --output results/stage2/run_2A

# Run 2B
./venv/bin/python scripts/train_single_agent.py \
  --pursuer_obs partial --belief raw_history --dcbf gamma=0.2 \
  --evader random --steps 500000 --seed 42 \
  --output results/stage2/run_2B

# Run 2C
./venv/bin/python scripts/train_single_agent.py \
  --pursuer_obs partial --belief bimdn --dcbf none \
  --evader random --steps 500000 --seed 42 \
  --output results/stage2/run_2C
```

### Gate 2 Criteria

| Metric | Run 2A Target | Hard Fail | Fail Response |
|--------|---------------|-----------|---------------|
| Capture rate vs random | > 80% | < 40% | Partial obs pipeline is broken |
| Reward curve | Increasing trend by 200K steps | Flat after 300K steps | Learning signal not reaching policy |
| DCBF intervention rate | < 15% | > 40% | DCBF too aggressive under uncertainty — tune gamma |
| Physical collisions | 0 | Any | DCBF or collision detection broken |
| Inference time | < 5ms/step | > 20ms/step | Optimize BiMDN or reduce history length |

**Comparison checks** (not hard gates, but informative):

| Comparison | Expected | Action if Violated |
|------------|----------|--------------------|
| 2A vs 2B: BiMDN helps? | 2A capture >= 2B | If 2A < 2B, BiMDN is hurting. Debug or fall back to LSTM. |
| 2A vs 2C: DCBF cost? | 2A capture within 10% of 2C | If 2A << 2C, DCBF too aggressive. Lower gamma to 0.1. |
| 2A vs Phase 2 baseline | 2A < Phase 2 (expected, harder task) | If 2A < 50% of Phase 2, partial obs is too hard. Check FOV params. |

### If Gate 2 Fails

- **< 40% capture with full pipeline**: The partial obs representation is too hard. Try:
  1. Increase FOV to 180° (less partial)
  2. Increase lidar range to 10m
  3. Use full obs history (K=20 instead of K=10)
  4. If still failing, train with full obs first, then fine-tune with partial obs (curriculum on observability)

- **DCBF intervention > 40%**: The policy under partial obs outputs more "unsafe-looking" actions. Try:
  1. gamma=0.1 (more permissive)
  2. Train without DCBF, add DCBF only at final evaluation
  3. Add CBF-RL reward shaping (w_intervention penalty) during training

- **BiMDN hurts performance**: The latent vector is confusing the policy. Try:
  1. Freeze BiMDN weights during PPO training (don't fine-tune end-to-end)
  2. Fall back to LSTM encoder
  3. Fall back to raw obs history (concatenated frames)

---

## Stage 3: Self-Play Smoke Test

**Compute**: ~3-4 GPU hours
**What**: Run AMS-DRL for 3 phases (cold-start + 2 alternating) at curriculum Level 1 only. Validates that self-play dynamics work.

### Protocol

```bash
./venv/bin/python scripts/train_amsdrl.py \
  --max_phases 3 \
  --timesteps_per_phase 200000 \
  --curriculum_max_level 1 \
  --seed 42 \
  --output results/stage3/smoke_test
```

### Gate 3 Criteria

| Check | Target | Hard Fail | Fail Response |
|-------|--------|-----------|---------------|
| Cold-start (S0) | Evader reaches >50% goals in NavigationEnv | < 20% | NavigationEnv reward too sparse — add shaping |
| Phase 1 (pursuer trains) | Capture rate > 20% | < 5% | Pursuer not learning — check reward, obs pipeline |
| Phase 2 (evader trains) | Capture rate decreases from Phase 1 | Capture rate increases | Evader not training or reward sign is wrong |
| Entropy (both agents) | Stays above -0.5 | Drops below -2.0 (rollback trigger) | Entropy collapse — check log_std_floor clamping |
| No crashes | Full 3 phases complete | Any exception | Debug the crash |
| Health monitor | No rollbacks triggered | Rollback triggers on phase 1 | Health monitor thresholds too tight for early training |
| Training speed | Measure ms/step | > 20ms/step | Profile and optimize bottleneck |

### Key Observations to Record

Even if gate passes, record these for tuning Stage 4:
- Actual ms/step (for budgeting Stage 5)
- Actual DCBF intervention rate during self-play (may differ from single-agent)
- Entropy trajectory for both agents
- GPU memory usage (can we run 2 seeds in parallel?)

### If Gate 3 Fails

- **Cold-start evader doesn't learn navigation**: Simplify NavigationEnv — remove flee phase, just do goal-reaching. Or skip cold-start and initialize evader with Phase 2 random policy.

- **Pursuer doesn't improve in Phase 1**: Check that evader is actually frozen. Check that reward signal is correct with partial obs. Try reducing problem difficulty (init_distance=(1,3)).

- **Both improve but entropy collapses**: The self-play pressure is too high. Try:
  1. Lower learning rate (3e-4 → 1e-4)
  2. Increase entropy coefficient (0.01 → 0.05)
  3. Shorter training phases (200K → 100K steps)
  4. Add exploration bonus (intrinsic curiosity)

- **Evader improves but capture rate goes UP**: Reward sign error. Evader's reward should be negative of pursuer's (or separate escape reward).

---

## Stage 4: Single Seed Full Run

**Compute**: ~8-15 GPU hours
**What**: One complete AMS-DRL run through all 4 curriculum levels, with full health monitoring.

### Protocol

```bash
./venv/bin/python scripts/train_amsdrl.py \
  --max_phases 12 \
  --timesteps_per_phase 500000 \
  --curriculum_max_level 4 \
  --eta 0.10 \
  --seed 42 \
  --health_monitoring full \
  --wandb_project pursuit-evasion \
  --wandb_tags phase3,stage4,validation \
  --output results/stage4/full_run_seed42
```

### Gate 4 Criteria

| Check | Target | Hard Fail | Fail Response |
|-------|--------|-----------|---------------|
| Curriculum reaches Level 4 | Level 4 reached | Stuck at Level 1-2 after 8 phases | Lower advancement threshold (0.7 → 0.5) or extend phases |
| NE gap trend | Decreasing trend | Oscillating or increasing after 6 phases | Adjust alternation schedule or phase length |
| NE gap at end | < 0.15 | > 0.30 | Self-play not converging — see failure protocols |
| Physical collisions | 0 | Any | DCBF broken — fix before proceeding |
| CBF violations | < 3% | > 10% | Tune DCBF gamma (try 0.1) |
| Rollbacks | <= 2 total | > 4 | Training is unstable — reduce lr, increase entropy coeff |
| Forgetting score | < 0.20 | > 0.35 | Catastrophic forgetting — add experience replay or checkpoint ensemble |
| Wall-clock time | Measured | > 20h | Optimize before running 5 seeds |
| Strategy diversity | cluster_entropy_norm > 0.5 | < 0.3 (mode collapse) | Add diversity bonus or trajectory-based reward shaping |

### Monitoring Checklist (check wandb during run)

Check these every ~2h while the run is active:

- [ ] `health/total_entropy` — both agents above -0.5?
- [ ] `health/capture_rate` — not stuck at 0 or 1?
- [ ] `curriculum/level` — advancing?
- [ ] `ne/gap` — trending down?
- [ ] `baseline/random_win` — pursuer still beats random? (> 0.7)
- [ ] `elo/training_agent` — not dropping?
- [ ] GPU utilization — is training actually running?

### If Gate 4 Fails

- **Stuck at Level 1**: The 70% capture threshold may be too aggressive for partial obs with FOV. Lower to 50%. Or make Level 1 even easier (init_distance=(1,3)).

- **NE gap doesn't decrease**: Try:
  1. Longer phases (500K → 1M steps)
  2. More phases (max_phases=16)
  3. Reduce eta threshold (0.10 → 0.15)
  4. Check if one agent is fundamentally stronger (asymmetric action space?)

- **Catastrophic forgetting**: The agent forgets how to beat old strategies when training against new ones.
  1. Add historical opponent pool (sample 30% from past checkpoints)
  2. Lower learning rate for later phases
  3. Use larger replay buffer

- **Mode collapse (low diversity)**: Both agents converge to a single strategy.
  1. Add population-based training (maintain 3 policies per agent)
  2. Add diversity reward bonus
  3. Increase exploration noise in later phases

---

## Stage 5: Full Experiment Suite

**Compute**: ~180-300 GPU hours
**What**: 5 seeds for main experiment, baselines, ablations. Only run after Stage 4 passes.

### Execution Plan

**Priority order** (run in this order; stop if compute budget runs out):

| Priority | Experiment | Seeds | Est. Hours | Cumulative |
|----------|-----------|-------|------------|------------|
| 1 | AMS-DRL main (PPO+BiMDN+DCBF) | 5 | 40-75h | 40-75h |
| 2 | Unconstrained PPO baseline | 5 | 40-75h | 80-150h |
| 3 | MAPPO-Lagrangian baseline | 5 | 40-75h | 120-225h |
| 4 | BiMDN ablation (raw obs) | 5 | 40-75h | 160-300h |
| 5 | Exploitability (best-response) | 10 | 20h | 180-320h |
| 6 | LSTM encoder ablation | 5 | 40-75h | 220-395h |
| 7 | Asymmetric: v_P/v_E = 1.5 (fast pursuer) | 3 | 8-15h | 228-410h |
| 8 | Asymmetric: v_P/v_E = 0.8 (matched) | 1-3 | 3-15h | 231-425h |
| 9 | Asymmetric: v_P/v_E = 0.5 (slow pursuer) | 1-3 | 3-15h | 234-440h |
| *10* | *Simultaneous SP (optional)* | *3* | *24-45h* | *258-485h* |
| *11* | *MACPO (optional)* | *3* | *24-45h* | *282-530h* |

### Parallelization on niro-2

```
Week 1: AMS-DRL seeds 1-2  +  Unconstrained PPO seeds 1-2   (parallel if GPU fits)
Week 2: AMS-DRL seeds 3-5  +  MAPPO-Lagrangian seeds 1-3
Week 3: MAPPO-Lag seeds 4-5 + BiMDN ablation seeds 1-5 + Exploitability
```

### Early-Stop Rules for Stage 5

- If a seed crashes: restart from last checkpoint. If crashes again, debug before running more seeds.
- If 3/5 seeds show same failure pattern: stop remaining seeds, diagnose, fix, then restart all 5.
- If mean NE gap across first 3 seeds is > 0.25 at convergence: stop, investigate, re-tune.

### Final Evaluation Protocol

After all training completes:
```bash
./venv/bin/python scripts/evaluate_phase3.py \
  --models results/stage5/amsdrl_seed*/best_model.zip \
  --baselines results/stage5/mappo_lag_*/best_model.zip \
               results/stage5/unconstrained_*/best_model.zip \
  --n_episodes 200 \
  --output results/phase3_final/
```

Produces:
- Comparison table (Table 1 for the paper)
- NE convergence curves with error bars (Figure X)
- Strategy diversity visualization (Figure Y)
- Safety violation breakdown (Table 2)
- Exploitability results (Table 3)

---

## Post-Stage 5: Zero-Cost Evaluations

**Compute**: ~2-5 GPU hours total (inference only — no training)
**What**: Three evaluation studies using the trained Stage 5 models. These require no additional training and produce high-impact paper results.

### 5A. Human Evader Experiment

**Purpose**: Test the trained pursuer against a human-controlled evader to evaluate real-world robustness and provide a qualitative comparison beyond scripted/learned opponents.

**Protocol**:
1. Create a keyboard/joystick evader controller (`scripts/human_evader_experiment.py`, `envs/human_interface.py`)
2. Use the best Stage 5 pursuer model (PPO+BiMDN+DCBF)
3. Run 20 episodes per participant (target: 2-3 participants)
4. Record: capture rate, capture time, safety violations, trajectory data
5. Optional: ask participants to rate pursuit "intelligence" (1-5 Likert)

**Deliverables**:
- Human evader vs AI pursuer capture rate comparison
- Trajectory visualizations of human strategies vs learned evader strategies
- Qualitative analysis: do humans find novel strategies the learned evader didn't?

**Compute**: 0 (real-time inference only, ~1h wall-clock per participant)

### 5B. Generalization Study

**Purpose**: Test how well policies trained on the simple arena (Phase 3 training env) generalize to unseen complex environments. Zero training cost — evaluation only.

**Protocol**:
1. Take best Stage 5 models (trained on 20×20m open arena with circular obstacles)
2. Evaluate on 3 complex layouts defined in `envs/layouts.py`:
   - Corridor (4×20m): narrow passage, tests pursuit in confined space
   - L-shaped room (15×15m with cutout): corner exploitation, line-of-sight breaks
   - Warehouse (20×20m with shelf grid): dense obstacles, multiple hiding spots
3. Also test: trained with 2 obstacles → evaluated with 5; trained on 20×20m → evaluated on 15×15m
4. Run 100 eval episodes per layout × 5 seeds
5. Report capture rate degradation, safety violation rate, DCBF intervention rate

**Script**: `scripts/run_generalization_study.py`
**Compute**: 0 training, ~1h CPU for evaluation

### 5C. Interpretability Analysis

**Purpose**: Generate interpretability visualizations that explain what the learned policies are doing. Strengthens the paper's analysis section significantly.

**Methods**:
1. **Input saliency maps**: Compute gradient of V(s) w.r.t. observation dimensions → which inputs matter most
2. **BiMDN belief evolution**: Animate belief distribution over time for representative episodes → how does uncertainty evolve
3. **Policy phase portraits**: Plot action vector field across (x, y) positions → global pursuit/evasion strategy visualization
4. **Attention to safety**: Correlate DCBF intervention regions with saliency hotspots

**Scripts**: `evaluation/interpretability/` (saliency.py, belief_animation.py, phase_portraits.py, safety_attention.py)
**Compute**: 2-5 GPU hours (gradient computation over many states)

### Gate: Post-Stage 5

These evaluations are NOT gates — they cannot block paper writing. However, all three should be completed before the Phase 5 paper writing sessions. Results feed directly into the paper:
- Human evader → Paper 1 Section VI-F or Paper 2 Section VII
- Generalization → Paper 2 Section VIII (Generalization Analysis)
- Interpretability → Paper 1 Figure 10 (saliency), Paper 2 Section IX (Analysis)

---

## Quick Reference: Failure Decision Tree

```
Training not learning at all?
├── Check reward signal → is it reaching the agent?
├── Check observation shapes → any NaN or inf?
├── Check gradient norms → exploding or vanishing?
└── Reduce problem complexity → fewer obstacles, closer starts

Entropy collapsing?
├── Increase entropy coefficient (0.01 → 0.05)
├── Verify log_std_floor clamping is active
├── Reduce learning rate
└── Shorter self-play phases

DCBF intervening too much (>30%)?
├── Lower gamma (0.2 → 0.1 → 0.05)
├── Train without filter, add at eval only
└── Add intervention penalty to reward

BiMDN not helping?
├── Try freezing BiMDN weights (no fine-tuning)
├── Fall back to LSTM encoder
└── Fall back to raw obs history

Self-play not converging?
├── Longer phases (500K → 1M)
├── More phases (12 → 16)
├── Add historical opponent sampling (30%)
└── Try simultaneous self-play instead

Stuck at curriculum level?
├── Lower advancement threshold (0.7 → 0.5)
├── Extend training at current level
└── Make current level easier (closer starts, fewer obstacles)
```

---

## Compute Budget Summary

| Stage | Hours | Cumulative | Decision Point |
|-------|-------|------------|----------------|
| -1: DCBF theorem | 0h | 0h | Formal proof + numerical verification pass? |
| 0: Unit tests | 0h | 0h | All tests pass? |
| 1: BiMDN pre-train | 1h | 1h | Belief quality good? |
| 2: Single-agent | 3-4h | 4-5h | Pipeline works end-to-end? |
| 3: Self-play smoke | 3-4h | 7-9h | AMS-DRL dynamics healthy? |
| 4: Single seed full | 8-15h | 15-24h | Full pipeline validated? |
| 5: Full experiment | 200-350h | 215-374h | Paper results (incl. asymmetric runs) |
| Post-5: Evaluations | 2-5h | 217-379h | Human evader, generalization, interpretability |
| **Validation total** | **15-24h** | | **5-7% of total budget** |
| **New additions** | **22-55h** | | **Asymmetric (20-50h) + interpretability (2-5h)** |

Spending 15-24h on validation to derisk 200-350h of compute is a 10-15× return on investment.
