# ICRA 2027 Reading Plan: First-Principles Understanding

**Purpose:** Before writing the paper, understand every concept, decision, and result from foundational principles — as if you originated the research yourself.

**Format:** 14 modules progressing from fundamentals to specific contributions. Each module includes: concept foundation → mathematical derivation → project-specific application → historical context (what was tried, what failed, why) → paper references.

**Cross-references:** `S##` = session/worklog number, `D##` = decision in `docs/decisions.md`.

---

## Prerequisite Map

```
M1 (PE Fundamentals) ──┬──→ M3 (Environment) ──→ M4 (Rewards)
M2 (RL Foundations) ───┘                        │
                       └────────────────────────┘
                                ↓
M5 (Self-Play Theory) ←─────────────────────────┘
       ↓
M6 (Curriculum/Infra) ←→ M7 (Partial Obs) ←→ M8 (Safety/CBF)
       ↓
M9 (EMA) ←→ M10 (Pop-FSP)
       ↓         ↓
M11 (Evaluation) M12 (G&I Repro)
       ↓
M13 (Failure Archive) ←── references ALL prior modules
       ↓
M14 (Paper Narrative) ←── capstone
```

**Fast paths:**
- **Know RL basics?** Skip M2, reference as needed.
- **Know PE theory?** Skip M1.
- **Want the story arc first?** Read M1→M5→M9→M10→M14 (12 hours), then fill in details.
- **Writing the Intro?** Focus on M1, M5, M14.
- **Writing the Methods?** Focus on M3, M4, M6, M9, M10.
- **Writing the Results?** Focus on M9, M10, M11, M12.

**Suggested reading paths by background:**

| Your Background | Start With | Then | Skip (reference as needed) |
|-----------------|------------|------|---------------------------|
| Strong RL, new to PE | M1, M3 | M5→M9→M14 | M2 |
| Strong PE, new to RL | M2, M5 | M9→M10→M14 | M1 |
| Know both, want project history | M14 (narrative) | M9, M10, M12 | M1, M2, M3 |
| Writing specific section | See module-to-section table in M14 | — | Everything else |

**Priority markers:**
- **(ESSENTIAL)** = Must read to understand the project
- **(REFERENCE)** = Read if you need detail
- **(OPTIONAL)** = For deep dives only

---

## Module 1: Pursuit-Evasion as a Differential Game
**Estimated time:** 3-4 hours

**Foundation:** Two-player zero-sum differential games. Value function V(x). Isaacs' equations. Capture sets vs escape sets. Saddle-point equilibrium.

**Why equal-speed PE in bounded domains is hard:** No trivial capture strategy exists. The evader can always mirror pursuer movements. Most PE literature gives evader 10-30% speed advantage.

**Project specifics:** G&I reproduction environment (16×16m arena, bicycle model, wheelbase 0.30m). Equal speeds (v_max 2.5 vs 1.5 is asymmetric in acceleration but not top speed). Our earlier framework (S48) used 10×10m with equal top speeds.

**Historical context:** Initially tried evader advantage (1.05x, 1.1x, 1.2x in S59-S64). Discovered equal-speed is actually viable with partial observability.

**Self-check:**
1. Why is equal-speed PE in a bounded domain fundamentally harder than unequal-speed?
2. What does CR=0.5 mean in terms of game-theoretic equilibrium?
3. Why did most papers in the survey (S54) give evader a speed advantage?

**Papers:** G&I 2024 Section II **(ESSENTIAL)**, Paper 25 (HJ reachability survey) **(REFERENCE)**, Paper 19 (aircraft PE self-play) **(OPTIONAL)**

**Deep dive:** S54 **(REFERENCE)** (PE design choices survey), S64 **(OPTIONAL)** (speed diagnostics)

---

## Module 2: Deep RL for Continuous Multi-Agent Control
**Estimated time:** 4-5 hours

**Foundation:** Actor-critic architecture. Policy gradient theorem. Advantage estimation (GAE).

**Algorithms:**
- PPO: Clipped surrogate objective, KL penalty, why it's stable
- SAC: Maximum entropy, twin critics, automatic temperature tuning
- MADDPG: Centralized critic with decentralized actors, experience replay for multi-agent

**Project specifics:** Phase 1-3 used PPO (sb3). G&I reproduction used MADDPG. RecurrentPPO (LSTM) attempted (S68) and abandoned.

**When to use which:** PPO for stable single-agent, SAC for sample efficiency, MADDPG for competitive multi-agent with discrete/continuous hybrid.

**Self-check:**
1. Why does MADDPG use a centralized critic but decentralized actors?
2. When would you choose PPO over SAC for a competitive game?
3. What is the policy gradient theorem, and why does it fail in competitive settings?

**Papers:** MADDPG (Lowe et al.) **(ESSENTIAL)**, PPO (Schulman et al.) **(REFERENCE)**, SAC (Haarnoja et al.) **(REFERENCE)**

**Deep dive:** S13 **(OPTIONAL)** (PPO training), S41 **(OPTIONAL)** (AMS-DRL), S68 **(OPTIONAL)** (RecurrentPPO)

---

## Module 3: Environment Design
**Estimated time:** 2-3 hours

**Arena:** 16×16m bounded square. Episode length: 400 steps. sim_dt = 0.1s.

**Pursuer:** Bicycle model with wheelbase 0.30m. v_max = 2.5 m/s, v_min = -1.0 m/s. FOV sensing: 120° (2π/3 rad), range 6m.

**Evader:** Point-mass model. v_max = 1.5 m/s, v_min = -1.5 m/s. Omnidirectional sensing (2π rad, 1000m range).

**Capture condition:** Agent radius c = 0.15m. Capture when distance ≤ 2c = 0.30m.

**Observation space (43 dims):** Own pose, opponent relative pose, opponent velocity, Lidar rays, timestep, own velocity, frame stack buffer (frame_stack=2).

**Action space:** Continuous [linear acceleration, steering rate] per agent.

**Reward structure:** λ_capture = 1000, λ_time = 1, λ_distance = 1. Dense reward combining capture bonus, time penalty, and distance-to-opponent.

**Curriculum:** Progressive difficulty. t_end_ratio = 0.1 (2000 episodes). FOV annealing: 360° → 120°. Evader v_max annealing: 0.0 → 1.5 m/s.

**Note:** Our earlier Phase 1-3 framework (S11-S84) used a different environment: 10×10m arena, 200 steps, 0-3 obstacles, radius+LOS masking. That work informed our understanding but is not part of the G&I reproduction paper. The experiments in Modules 9-12 all use the G&I environment described above.

---

## Module 4: Reward Engineering
**Estimated time:** 3-4 hours

**Terminal rewards:** Capture (+10 pursuer, -10 evader), timeout (-10 both), escape (+10 evader). Variants tried: ±50 (S48), ±10 (S55).

**PBRS (Potential-Based Reward Shaping):** Obstacle-approach shaping (S44-S45). w_obs_approach parameter. Did not prevent collapse.

**Wall penalties:** w_collision for obstacle/arena-boundary contact (S50-S51).

**Survival bonus:** Small positive reward for evader staying alive. Attempted to prevent evader collapse.

**Exploration rewards:** SearchStalenessTracker (S71) — 10×10 grid, rewards visiting stale cells. w_search parameter. Visibility-gated variant.

**What worked:** No single reward modification prevented collapse. The fundamental issue was structural (self-play dynamics), not reward design.

---

## Module 5: Self-Play Theory and Competitive Balance
**Estimated time:** 5-6 hours

**SimGD vs AltGD:** Simultaneous gradient descent diverges in competitive games (S53). Alternating updates more stable but still fragile.

**Nash equilibrium in policy space:** CR=0.5 represents balanced Nash. CR→0 or CR→1 indicates collapse.

**Collapse modes:**
- Pursuer-dominant (CR→1): Evader cannot learn, receives only negative rewards
- Evader-dominant (CR→0): Pursuer cannot bootstrap, receives no capture rewards

**Cycling and non-transitivity:** Policies can cycle A→B→C→A. Population methods (PSRO, FSP) designed to break cycles.

**Phase length:** Discovered long frozen phases (500K steps) were fundamentally wrong (S53-S55). Rapid micro-phases (2048 steps) attempted.

**Self-check:**
1. Why does SimGD diverge in zero-sum games? Draw the gradient dynamics.
2. What is the difference between CR→0 collapse and CR→1 collapse? Which is harder to recover from?
3. Why do long frozen phases (500K steps) fail in competitive self-play?

**Papers:** Self-play survey (Paper 04) **(ESSENTIAL)**, PSRO **(REFERENCE)**, OpenAI Five **(OPTIONAL)**

**Deep dive:** S53 **(ESSENTIAL)** (self-play stabilization research), S55 **(REFERENCE)** (rapid alternation redesign), S57 **(REFERENCE)** (adaptive ratio + LR dampening), S58 **(OPTIONAL)** (collapse rollback + PFSP-lite)

**Decisions:** D8 **(ESSENTIAL)** (rapid micro-phases, S55), D5 **(REFERENCE)** (evader speed advantage, S64)

---

## Module 6: Curriculum and Training Infrastructure
**Estimated time:** 2-3 hours

**Curriculum levels:** L1-L4 progressive difficulty. Gates based on competitive ratio threshold.

**Checkpointing:** Save best model by eval metric. Checkpoint sweep: evaluate 20 checkpoints × 100 episodes each.

**Vectorized environments:** 8 parallel envs (S94). Speedup via batched inference.

**AMP:** Automatic Mixed Precision for GPU training. Initially caused NaN; fixed with twin critics (S93).

**Health monitoring:** Track competitive ratio, loss magnitudes, gradient norms. Detect collapse early.

---

## Module 7: Partial Observability
**Estimated time:** 4-5 hours

**POMDPs:** Belief state b(s), particle filters, sufficient statistics.

**Masking modes:**
- Radius sensing: visible within r meters
- LOS (Line-of-Sight): blocked by obstacles
- FOV (Field of View): triangular wedge (S76)
- Combined: radius + LOS

**BiMDN:** Bidirectional Mixture Density Network from G&I 2024. Outputs belief over opponent state.

**Project attempts:**
- Belief-state self-play (S90): BeliefTracker with particle filter
- BELIEF_s200: Peaked at CR=0.48, then collapsed (S91)
- Key finding: drift is structural, not curriculum-induced

**Papers:** G&I 2024 Section III **(ESSENTIAL)**, PO-GRL literature **(REFERENCE)**

**Deep dive:** S90 **(REFERENCE)** (belief-state implementation), S91 **(ESSENTIAL)** (BELIEF_s200 failure), S75 **(OPTIONAL)** (masking curriculum)

**Decisions:** None in decision log; see S90-S91 for belief-state rationale

---

## Module 8: Safety via Control Barrier Functions
**Estimated time:** 4-5 hours

**CBF definition:** h(x) such that C = {x | h(x) ≥ 0} is forward invariant.

**QP formulation:** min ||u - u_nom||² s.t. Lfh + Lgh·u + α(h) ≥ 0.

**VCP-CBF:** Velocity-Constraint-Preserving CBF. Handles car-like dynamics.

**Safety filtering vs end-to-end:** Post-hoc filter (apply CBF to RL output) vs training with safety constraint.

**3-tier infeasibility:** Backup control, hierarchical relaxation, SVM classifier (S23).

**Project decision:** Post-hoc filter chosen over end-to-end (S32). BarrierNet attempted (S28) but abandoned.

**Papers:** CBF survey (Paper 03), Paper 05, Paper 28 (LBAC)

**Deep dive:** S16 (VCP-CBF validation), S21 (CBF-Beta integration), S32 (post-hoc filter decision)

**Decisions:** None in decision log; see S32 (post-hoc filter decision report)

---

## Module 9: EMA Opponent Stabilization
**Estimated time:** 4-5 hours

**Mathematical foundation:** Exponential moving average. θ_EMA ← (1-τ)·θ_EMA + τ·θ_current. Half-life = ln(2)/τ.

**Why tau=0.001:** Half-life ≈ 693 update steps ≈ 3.5 episodes. Fast enough to track improvement, slow enough to smooth oscillations (S97).

**Algorithm:** Maintain EMA copy of opponent actor. During training: current agent vs EMA opponent. During eval: compare current vs EMA.

**Evaluation modes:**
- CR_selfplay: current vs current
- CR_ema_selfplay: current vs EMA
- CR_vs_ema_evader: current pursuer vs EMA evader
- CR_ema_vs_current: EMA pursuer vs current evader

**Robustness metric:** min(CR_random, CR_greedy, CR_selfplay). Worst-case over opponent types.

**Results (7 seeds):**
- Seed 0: robustness=0.698 (SUCCESS)
- Seeds 1-6: all <0.30 (various collapse modes)

**Motivation arc:** EMA was tried after 20+ sessions of freeze/unfreeze, adaptive ratio, and exploration rewards all failed. The insight: instead of preventing collapse through complex phase switching, smooth the opponent's policy so oscillations dampen naturally.

**Self-check:**
1. Derive the EMA half-life formula. Why is tau=0.001 ≈ 3.5 episodes?
2. Why does smoothing the opponent policy reduce oscillation? What assumption does this make about the loss landscape?
3. If EMA had 7/7 success, what would that imply about the nature of self-play collapse?

**Analysis:** 1/7 reproducibility. Seed 0 is an outlier. EMA helps but does not guarantee convergence.

**Deep dive:** S97 **(ESSENTIAL)** (EMA design), S98 **(ESSENTIAL)** (seeds 0-2 analysis), S99 **(REFERENCE)** (seed 3 monitoring), S104 **(ESSENTIAL)** (seeds 4-6 + Gate 1 NO-GO)

**Decisions:** D42 **(ESSENTIAL)** (Gate 1 NO-GO, S104), D43 **(REFERENCE)** (NaN as intrinsic finding, S104). Note: LR=0.0005 identified in S98; EMA introduced in S97 (not in decision log).

---

## Module 10: Pop-FSP and Structural ∀/∃ Asymmetry
**Estimated time:** 5-6 hours

**FSP (Fictitious Self-Play):** Best response to historical mixture of opponent policies.

**PopulationManager design:**
- FIFO eviction, uniform/recency sampling
- Asymmetric populations (16 pursuer vs 8 evader attempted)
- Gradual mixing, adaptive ratio

**Config history:**
- v3: uniform sampling → CR=1.0 (pursuer-dominant)
- v4: recency-biased + asymmetric → evader-dominant
- v5-v7: various corrections → all failed

**The ∀/∃ Asymmetry Finding:**
- Pursuer objective: ∀ opponent policies, maximize capture → structurally easier (existence of one good strategy)
- Evader objective: ∃ opponent policy that avoids capture → structurally harder (must handle all)
- In symmetric games with identical architecture, this asymmetry causes systematic collapse

**Self-check:**
1. Formalize the pursuer's objective with ∀ and the evader's with ∃. Why is ∀ structurally easier in symmetric games?
2. Why does uniform sampling (v3) produce pursuer-dominant collapse? Why does recency bias (v4) overcorrect?
3. If you could redesign the architecture asymmetrically, what would you change?

**Implication:** Population methods cannot fix this without explicit architectural asymmetry.

**Deep dive:** S100 **(ESSENTIAL)** (Pop-FSP implementation), S101 **(REFERENCE)** (v2 balance fix), S102 **(REFERENCE)** (v3-v5 postmortem), S103 **(ESSENTIAL)** (v6/v7 failure + asymmetry finding)

**Decisions:** D37 **(ESSENTIAL)** (Pop-FSP ∀/∃ asymmetry is fundamental, S103)

---

## Module 11: Evaluation and Experimental Rigor
**Estimated time:** 3-4 hours

**Metrics:**
- Capture Rate (CR): fraction of episodes with capture
- Competitive Ratio: CR_pursuer / (CR_pursuer + CR_evader)
- Robustness: min(CR_random, CR_greedy, CR_selfplay)
- Oscillation amplitude: std of CR over recent checkpoints

**Checkpoint sweep:** Evaluate 20 checkpoints per run × 100 episodes × 3 opponent types.

**BR SAC exploitability analysis:**
- Train best-response SAC against frozen checkpoint
- H1: both non-exploitable (Nash)
- H2: one exploitable
- H3: both exploitable

**Tournament evaluation:** Round-robin among checkpoints/population members.

**Deep dive:** S97 **(REFERENCE)** (evaluation framework), S88 **(ESSENTIAL)** (BR exploitability analysis)

---

## Module 12: Gonultas & Isler 2024 Reproduction
**Estimated time:** 3-4 hours

**Paper claims:** Self-play CR=0.55, BiMDN improves over vanilla, curriculum necessary.

**Our reproduction:**
- Self-play CR=0.550 ✓ (matches paper)
- vs-random=0.936 (near paper)
- 5 critical fixes needed (S95):
  1. Actor observation bug
  2. BiMDN warm-up missing
  3. BiMDN architecture mismatch
  4. Tournament eval incorrect
  5. Seed-specific paths

**NaN diagnosis:** Scale mismatch + AMP + single critic → twin critics fix (S93).

**NaN debugging story (S92-S94):**
1. Symptom: BiMDN losses → NaN within 100-500 episodes
2. Diagnosis: Three root causes:
   - Scale mismatch: BiMDN outputs unbounded scale parameters
   - AMP interaction: fp16 underflow with small gradients
   - Single critic: overestimation → Q-value explosion
3. Fix: Twin critics (Ackermann 2019), gradient clipping, disable AMP for BiMDN, wheelbase correction (0.15→0.30)
4. Verification: Seeds 4/5/6 trained 19 episodes with no NaN

**What we learned:** Paper result is reproducible but fragile. Oscillatory dynamics are intrinsic to the problem, not implementation bugs.

**Deep dive:** S92 **(REFERENCE)** (full implementation), S93 **(ESSENTIAL)** (NaN diagnosis), S94 **(OPTIONAL)** (vectorized envs), S95 **(ESSENTIAL)** (5 fixes), S96 **(ESSENTIAL)** (verification)

---

## Module 13: The Complete Failure Archive
**Estimated time:** 4-5 hours (reference as needed)

**Summary table:**

| Category | Attempts | Sessions | Root Cause | Key Finding |
|----------|----------|----------|------------|-------------|
| Curriculum collapse | Run P, Q, R, S, T, U, Z1, Z2 | S40-S52 | Gates too low, no recovery | Long frozen phases fundamentally wrong |
| Speed/ratio tuning | S1v2-S1v6b | S59-S64 | Pure speed strategy | Equal speed + partial obs is viable |
| Freeze/unfreeze | SP7-SP14, SP15-SP18 | S66, S82-S96 | Collapse resumes after unfreeze | vis_reward=0.2 only sustained help |
| Masking curriculum | SP12_mc, SP12b, SP12c | S75 | Annealing disrupted by restart | +10-15pp possible |
| FOV variants | SP13a-SP14b | S76-S81 | No single config worked | FOV 90°/8m + asymmetric best |
| Adaptive ratio | RA7b, RA8 | S57 | Vanishing reward gradient | Insufficient alone |
| LR dampening | RA7b, RA8 | S57 | Scales LR but not gradients | Did not prevent collapse |
| EWC + RND | S52 | S52 | L2 collapse persists | Not sufficient |
| Exploration rewards | w_search | S67, S71 | Agent lacks capacity | Insufficient representation |
| Belief-state | BELIEF_s200 | S90-S91 | Structural drift | Drift is intrinsic, not curriculum |
| RecurrentPPO | S68 | S68 | Never launched | LSTM doesn't solve the core issue |
| Opponent Pool | FIFO, reservoir | S43, S55 | Diversity without balance | Doesn't address asymmetry |
| Pop-FSP | v1-v7 | S100-S103 | ∀/∃ structural asymmetry | **Core theoretical contribution** |

**Detailed entries:**

**Curriculum collapse:** Run P (L3+ collapse), Q, R, S, T, U, Z1, Z2. Root causes: gates too low, phase length too long, no recovery mechanism.

**Speed/ratio tuning:** S1v2-S1v6b. 1.05x failed, 1.2x held 35-50% briefly. Pure speed strategy, no obstacle use.

**Freeze/unfreeze strategies:** SP7-SP14. Freeze evader → pursuer climbs. Unfreeze → collapse resumes. Threshold tuning (0.6 oscillates, 0.8 stable but regresses).

**Masking curriculum:** SP12. PO-GRL annealing showed +10-15pp improvement before restart killed it.

**FOV variants:** SP13a-SP14b. FOV 90°/8m best. Asymmetric obs partial success.

**Adaptive training ratio:** RA7b/RA8. Loser gets extra phases. Insufficient — vanishing reward gradient.

**LR dampening:** Scale LR near equilibrium. Did not prevent collapse.

**EWC + RND:** S52. Elastic weight consolidation + random network distillation. Did not prevent L2 collapse.

**Exploration rewards:** w_search (S71). Agent lacked capacity to exploit.

**Belief-state self-play:** BELIEF_s200 (S90-S91). Particle filter belief. Peaked 0.48 then collapsed.

**RecurrentPPO:** S68. LSTM policy. Implemented but never successfully launched.

**SP15-SP18 (final freeze/unfreeze series):** S82-S96. SP15: γ=0.9/0.99, EWC, vis reward, stepped schedule. SP16: vis reward + forced switch. SP17: 20K ep pure self-play. SP18: 30K ep extension. Key finding: vis_reward=0.2 was the only modification that produced sustained non-zero capture (0.57 vs 0.10 baseline).

**Opponent Pool / Reservoir Sampling:** S43 (FIFO pool), S55 (reservoir sampling for diversity). Did not prevent collapse.

**Pop-FSP v1-v7:** S100-S103. All configurations failed. Led to ∀/∃ asymmetry finding.

---

## Module 14: The Paper Narrative
**Estimated time:** 2-3 hours

**Framing:** "A Systematic Study of Self-Play Stabilization Methods for Competitive Deep RL in PO Pursuit-Evasion"

**Why systematic study:** 103 sessions of scope. Both positive (EMA seed 0, G&I repro) and negative (Pop-FSP, EMA seeds 1-6) results. Negative results are central contributions.

**Why "systematic study" framing:**
- 103 sessions of scope — too broad for a single method paper
- EMA 1/7 reproducibility means it's a partial positive result, not a guaranteed method
- Pop-FSP failure is a genuine structural insight, not just "didn't work"
- G&I reproduction validates the environment and shows the problem is hard
- Negative results are central: they define the boundary of what's possible

**Module-to-section mapping:**
| Paper Section | Modules | Purpose |
|---------------|---------|---------|
| Introduction | 1, 5, 14 | Problem motivation, why PE is hard, what we study |
| Related Work | 2, 5, 7, 8 | RL for PE, self-play methods, PO, safety |
| Methods | 3, 4, 6, 9, 10 | Env, EMA, Pop-FSP, eval metrics |
| Results | 9, 10, 11, 12 | EMA seeds, Pop-FSP configs, G&I repro, robustness |
| Discussion | 13, 14 | What failed and why, future directions |

**Key contributions:**
1. **EMA characterization:** First systematic evaluation of EMA opponent smoothing in competitive PE. Shows promise (seed 0: 0.698) but poor reproducibility (1/7).
2. **∀/∃ structural asymmetry:** Novel theoretical insight explaining why population methods systematically fail in symmetric competitive games.
3. **G&I 2024 verification:** Independent reproduction with 5 critical fixes and fragility analysis. Validates the problem difficulty.

**Self-check:**
1. Why is a 1/7 positive result still a contribution in a systematic study framing?
2. What would you say to a reviewer who asks "if EMA doesn't work reliably, why include it?"
3. Map each module to a specific paragraph in the paper. Which modules have no corresponding paragraph?

**What to emphasize:** The journey from "method paper" (S104 initial plan) to "systematic study" (Gate 1 NO-GO, S104). Intellectual honesty as a feature, not a bug.

**Decisions:** D40 **(ESSENTIAL)** (systematic study framing, S104), D42 **(ESSENTIAL)** (Gate 1 NO-GO, S104), D43 **(REFERENCE)** (NaN as intrinsic finding, S104)

---

## Appendix: Key Hyperparameters

**Base environment (all experiments in Modules 9-12):**
- Arena: 16×16m, max_steps: 400, capture_radius: 0.15 (effective 0.30)
- Pursuer: bicycle model, wheelbase 0.30m, v_max 2.5, FOV 120°/6m
- Evader: point mass, v_max 1.5, omnidirectional sensing

| Parameter | EMA (S97-S104) | Pop-FSP (S100-S103) | G&I Repro (S92-S96) |
|-----------|---------------|---------------------|---------------------|
| Algorithm | MADDPG | MADDPG | MADDPG |
| Actor LR | 0.0001 (s0-s2), 0.0005 (s3-s6) | 0.0005 | 0.0005 |
| Critic LR | 0.0005 | 0.0005 | 0.0005 |
| Batch size | 1024 | 1024 | 1024 (×2 envs) |
| Tau (target) | 0.005 | 0.005 | 0.005 |
| EMA tau | 0.001 | 0.0 (disabled) | 0.0 (vanilla) |
| Hidden dim | 128 | 128 | 128 |
| Episodes | 5000 | 20000 | 20000 |
| Frame stack | 2 | 2 | 2 |
| Parallel envs | 2 | 2 | 2 |
| AMP | Yes | Yes | Yes |
| Twin critics | Yes | Yes | Yes |
| Curriculum t_end | 0.1 (2000 ep) | 0.3 | 0.3 |
| Pop size | — | 8 pursuer, 16 evader | — |
| Pop recency λ | — | 0.15 | — |
| Pop mixing max | — | 0.80 | — |

**Note:** Our Phase 1-3 framework (S11-S84) used different settings: PPO/SAC, 10×10m arena, 200 steps, 0.3m capture radius, 0-3 obstacles, radius+LOS masking. See `docs/worklogs/` for those experiments.
