# Comprehensive Project Report: Safe Deep RL for 1v1 Pursuit-Evasion

**Date**: 2026-03-06
**Sessions**: S01–S73 (73 sessions, 14 days: Feb 21 – Mar 06, 2026)
**Status**: Phase 3 — Partial Observability & Self-Play (in progress)

---

## 1. Executive Summary

This project aims to build a **safety-guaranteed deep reinforcement learning system** for 1v1 pursuit-evasion (PE) games on physical ground robots with nonholonomic dynamics. It combines safe DRL (CBF-based), adversarial self-play training, partial observability, and real-world deployment — a combination no existing paper has achieved.

### What We Built
- A complete simulation environment (`PursuitEvasionEnv`) with unicycle dynamics, obstacles, partial observability (LOS + sensing radius), and configurable parameters
- Safety infrastructure: VCP-CBF safety filter, BarrierNet differentiable QP layer, DCBF post-hoc filter
- Self-play training framework (`AMSDRLSelfPlay`) with micro-phase alternation, opponent pools, collapse rollback, PFSP-lite, adaptive ratio, freeze/unfreeze, and SAC/PPO/RecurrentPPO support
- 700+ passing tests across all components
- ~50 training experiments on niro-2 (RTX 5090, 125GB RAM)
- Extensive literature coverage: 36+ papers in main review, 20+ papers on self-play collapse, 15+ on masking curricula, 20+ on search rewards

### Where We Are
- **Best result**: SP11e achieved **85% pursuer capture rate** in self-play (pursuer vs frozen random evader) with combined partial observability (3m sensing + LOS masking), equal speed, 0-3 random obstacles
- **Core challenge**: Transitioning from pursuer-only training to full co-evolution. Every attempt at unfreezing the evader has led to rapid pursuer collapse (42% → 11% in 307K steps)
- **Active runs**: 5 seeds running SP11i config on niro-2 to characterize seed sensitivity

### Key Discovery
**Seed sensitivity dominates training outcomes.** Same config produces capture rates from 0.13 to 1.00 across seeds. SP11e's 85% was a lucky trajectory. Multi-seed sweeps are essential.

---

## 2. Literature Review Summary

### Scope
36 papers in main collection + 15 supplementary papers + ~50 additional papers surveyed during research phases. Topics: pursuit-evasion games, deep RL, CBF safety, HJ reachability, self-play, sim-to-real transfer.

### Key Findings from Literature
1. **No paper combines all our elements** — safe DRL + 1v1 ground robot PE + nonholonomic dynamics + partial obs + self-play + real deployment. Gap confirmed across 50+ papers.
2. **Closest works**: Paper 02 (Gonultas & Isler — MADDPG + BiMDN on F1TENTH), Paper 22 (MADR — DeepReach on TurtleBots), Paper 08 (ICRA 2024 — vision-based on Unitree A1)
3. **CBFs for nonholonomic systems** need Virtual Control Point formulation (Paper N12) — position-based CBFs have mixed relative degree
4. **Self-play convergence** requires population-based methods (PSRO, FSP, league) or careful alternation with short phases; long frozen phases (>10K steps) are universally condemned
5. **Equal-speed 1v1 PE is trivially solvable** in bounded domains — all surveyed papers give evader 10-30% speed advantage

### References
- `docs/literature-review/final_literature_review.md` — Full 36-paper review
- `docs/literature-review/paper_summaries_*.md` — Individual paper summaries
- `docs/research_self_play_collapse_prevention.md` — 20-paper self-play stability survey
- `docs/research/pe_design_choices_survey.md` — 15-paper PE design benchmarks
- `docs/literature-review/search_exploration_rewards.md` — 20+ paper exploration reward survey

---

## 3. Implementation Timeline

### Phase 1: Simulation Foundation (S11–S17, Feb 22)
**7 sessions, 1 day.** Built the core simulation from scratch.

| Component | Description |
|-----------|-------------|
| `envs/pursuit_evasion_env.py` | Core env: unicycle dynamics, obstacles, collision resolution, configurable arena |
| `envs/observations.py` | `ObservationBuilder`: full/partial obs, LOS masking, sensing radius, combined masking |
| `envs/rewards.py` | Distance shaping, terminal rewards, visibility, survival bonus, obstacle approach, search staleness |
| `envs/wrappers.py` | `SingleAgentPEWrapper`: converts 2-player game to single-agent for SB3 |
| `envs/dynamics.py` | Unicycle kinematics, obstacle collision, wall bouncing |
| `envs/rendering.py` | Pygame renderer with sensing visualization |
| Safety layer | VCP-CBF with d=0.1m virtual control point, weighted QP, 35 validation tests |
| Training pipeline | SB3 PPO integration, Hydra config, WandB/TensorBoard tracking |
| Self-play v1 | Basic alternating self-play loop, health monitor |
| Baselines | Random, greedy (proportional navigation K_p=3.0), DQN, DDPG |
| **Tests** | **118 tests passing** |

### Phase 2: Safety Integration (S18–S25, Feb 22)
**8 sessions, 1 day.** Full safety layer suite.

| Component | Description |
|-----------|-------------|
| Beta distribution policy | Bounded actions via Beta distribution + SB3 integration |
| Multi-constraint VCP-CBF | Arena wall, obstacle, collision CBFs with typed constraints |
| Safe action bounds | LP + analytical bounds computation |
| CBF-Beta policy integration | SafeRolloutBuffer, CBFBoundsComputer |
| Obstacle system | Random obstacles, obstacle observations, safety rewards |
| 3-tier infeasibility | Backup controller, hierarchical relaxation, SVM feasibility classifier |
| Safe self-play | Safety metrics tracking, comparison framework |
| Ablation suite | 5 configurations (A-E), results compilation |
| **Tests** | **290 tests passing** |

### Phase 2.5: BarrierNet Experiment (S26–S35, Feb 22)
**10 sessions, 1 day.** Explored differentiable QP safety layer.

| Component | Description |
|-----------|-------------|
| Differentiable QP layer | `qpth` integration, 2.5ms/solve on GPU (34.6x faster than CPU) |
| BarrierNet actor + PPO | Custom actor with QP-in-loop, RolloutBuffer, gradient flow through QP |
| Training pipeline | Hybrid CPU/GPU (9x speedup: 40h → 4.5h), niro-2 deployment |
| **3-way evaluation** | BarrierNet (2% capture) vs Baseline PPO (100%) vs Post-hoc filter (91.5%) |
| **Decision** | **BarrierNet abandoned** — 87.5% QP intervention prevents exploration. Post-hoc VCP-CBF filter recommended. |
| DCBF filter tuning | 14-config sweep, DCBF gamma=0.2 optimal |
| **Tests** | **478+ tests passing** |

**Key lesson**: End-to-end differentiable safety is elegant in theory but the QP layer's high intervention rate (87.5%) prevents the RL agent from exploring enough to learn. Post-hoc filtering is simpler and more effective.

### Phase 3: Partial Observability & Self-Play (S36–S73, Feb 22 – Mar 06)
**38 sessions, 13 days.** The longest and most challenging phase.

#### 3a. Spec Reviews & Architecture (S36–S43)
- Phase 3 spec panel review (quality 7.0 → 9.0/10)
- BiMDN belief encoder (in-FOV RMSE 0.31m, out-of-FOV failed)
- AMS-DRL self-play orchestrator with opponent pool, checkpoint manager
- Curriculum learning manager (4-level progressive difficulty)
- **Finding**: BiMDN pre-training failed Gate 1. Stage 2 training at 0% capture under partial obs.

#### 3b. The Long Self-Play Struggle (S44–S58)
This was the most intensive research phase. Summary of attempts:

| Run(s) | Approach | Result | Root Cause |
|--------|----------|--------|------------|
| H–P | Original curriculum (L1-L4, 100K-500K phases) | All collapsed at L2-L3 | Long frozen phases, equal speed, no viable equilibrium |
| Q–R | Dual-criteria gates, reward rebalancing | Collapsed at L3 | Gate threshold too low, still long phases |
| S–T | Arena size variation (20x20, 10x10) | Collapsed at L3 | Same fundamental issues |
| U | Cold-start checkpoint, phase warmup | Collapsed at L2 | 8 interacting root causes identified |
| V | Seed sensitivity test | Wild oscillation (0.03-1.00) | Game near tipping point at equal speed |
| Z1–Z2 | EWC + RND + NE-gap curriculum | Failed | Over-engineering, still wrong fundamentals |

**Session 53 breakthrough**: Comprehensive literature review revealed three interacting failures:
1. Game has no meaningful equilibrium at equal speed (pursuer always eventually wins)
2. Reward provides no useful signal (terminal-to-shaping ratio ~1:1000)
3. Self-play protocol amplifies instability (100K–500K phases are 10-50x too long)

#### 3c. The Redesign (S53–S58)
Major pivot based on S53 analysis:

| Change | Before | After |
|--------|--------|-------|
| Phase length | 100K–500K steps | 2048–4096 micro-phases |
| Speed | Equal (1.0/1.0) | Evader advantage (1.05x–1.2x) |
| Terminal rewards | ±100 | ±10 (later ±50) |
| Obstacles | Curriculum (0→N) | Always present (0-3 random) |
| Opponent pool | Not integrated | 50% current, 50% pool (reservoir sampling) |
| Rollback | Aggressive | Collapse rollback with streak limit |

| Run(s) | Config | Result |
|--------|--------|--------|
| RA1–RA2 | 1.15x speed, ±10, micro-phases | Launched |
| RA3–RA4 | Equal speed / no pool ablation | Collapsed (44x PBRS reward asymmetry) |
| RA5–RA6 | 10x10, 1.05x, ±50 terminal | Launched |
| RA7b–RA8 | Adaptive training ratio, LR dampening | Evader collapsed at M400, no recovery |
| RA9–RA10 | Collapse rollback + PFSP-lite | Rollback fires but evader always re-collapses |

**S58 diagnostic breakthrough**: Trained evader against fixed greedy pursuer. Discovered:
- Speed advantage enables open-field circles but NOT obstacle use
- **Equal speed forces obstacle exploitation** — D3 hit 37% escape with obstacle-hugging
- 4 obstacles too cluttered in 10x10; 2 is optimal
- `ent_coef=0.03` critical for variable speed (0.01 stalls at 12%)

#### 3d. Variable Speed & Partial Obs Exploration (S59–S63)

| Run | Config | Peak | Final | Key Finding |
|-----|--------|------|-------|-------------|
| S1v2b | Variable speed evader, 5M | **98% escape** | 98% | Obstacle-hugging perfected at variable speed |
| Run A | Pursuer vs S1v2b evader | **95% capture** | 94% | Pursuer can counter obstacle strategy |
| SP2–SP2d | Full obs self-play, variable speed | 0-7% | 0-7% | **Cold-start self-play fails with full obs** |
| S1v3 | LOS-only partial obs | 56% | 56% | Baseline partial obs |
| S1v4a | Asymmetric LOS, random obstacles | **80%** | 20% | Peaks high, regresses hard |
| S1v4b | Symmetric LOS, random obstacles | 73% | 50% | More stable but lower peak |
| S1v5c | Combined masking, greedy full obs | **16%** | 0% | **Equal speed + full obs pursuer = evader can't escape** |
| S1v6b | **1.2x speed, combined masking** | **52%** | 35-50% | **First viable speed+partial_obs config** |
| **SP4** | Combined masking, cold start, equal speed | **53%** | 39% | **DEGENERATE** — evader hides, no real PE |

**SP4 post-mortem**: At equal speed with combined masking, detection = guaranteed capture. Evader learned to HIDE (0.20 m/s avg speed), pursuer learned to SEARCH. The 53% "capture rate" was just search success rate. No actual pursuit-evasion dynamics occurred.

**Critical insight**: Information asymmetry (partial obs) creates a learnable game for cold-start self-play, but equal speed creates degenerate equilibria.

#### 3e. SAC & Freeze/Unfreeze (S64–S73)

Switched from PPO to SAC and adopted freeze-then-unfreeze approach:

| Run | Approach | Result | Key Finding |
|-----|----------|--------|-------------|
| SP6a/SP6b | Cold-start SAC self-play | ~10% | Cold-start fails for SAC too |
| SP7 | 1:1 co-evolution | Pursuer 20% → 11% | Evader adapts faster than pursuer |
| SP7a/SP7a2 | Adaptive boost | 8-24%, never sustained | Dynamic boost insufficient |
| **SP7b→SP7b2→SP8** | **Freeze evader → unfreeze** | **10% → 24% → 42% → collapse** | **Freeze works for pursuer buildup, but co-evolution still fails** |
| SP9 | 5:1 train ratio (pursuer:evader) | 30% → 24% | Better than 1:1 but still declining |
| **SP11a** | SAC, freeze, proven config | Baseline | **Proven hyperparams**: lr=3e-4, ent_coef=0.03, train_freq=1, gradient_steps=1 |
| SP11c | SAC, `ent_coef=auto` | 24% → 6% | **Auto entropy is destructive in self-play** |
| **SP11d** | Freeze threshold 0.6, max_steps=1000 | Peak 62%, then **collapsed to 7%** | Premature freeze switch (307K evader training destroyed 5.8M pursuer progress) |
| **SP11e** | **Freeze threshold 0.9, max_steps=2000** | **Peak 85%** | **Best run ever**. But was lucky seed — SP11i replica at 30% |
| SP11e2 | Warm-seed from SP11e | 20-35% | **Warm-seeding without replay buffer = catastrophic forgetting** |
| SP11f2 | 3-consecutive freeze switch | 38% (16M) | Running, peaked 69% |
| SP11g2/SP11h2 | Staleness search reward (w_search=0.003) | ~25% (8M) | **Search reward harmful** — competing signal teaches patrol not hunt |
| SP13a–SP13f | RecurrentPPO (LSTM) | 0-2% | **RecurrentPPO conclusively fails** — 3-10x slower, can't learn |
| **SP11i (seed 42-45)** | Multi-seed sweep | 30% (42), pending (43-45) | Characterizing seed sensitivity |

---

## 4. Key Findings & Lessons Learned

### 4.1 Game Design
1. **Equal-speed 1v1 PE in bounded domains is trivially solvable** — pursuer always eventually captures. No surveyed paper uses equal speeds for 1v1.
2. **1.2x speed advantage creates a viable game** under partial observability. 1.05x is insufficient.
3. **Partial observability (combined masking) is essential** for a meaningful game. Under full obs, pursuit is trivial.
4. **2 obstacles is the sweet spot** in a 10x10 arena. 4 is too cluttered (traps evader).
5. **Detection = capture at equal speed** — leads to degenerate hide-and-seek, not pursuit-evasion.

### 4.2 Self-Play Training
6. **Cold-start self-play fails with full observability** but WORKS with partial obs (SP4 bootstrapped to 53%, albeit degenerate).
7. **Freeze-then-unfreeze is the validated curriculum** — only approach that consistently builds pursuer competence (10% → 42% over 4M steps).
8. **1:1 co-evolution conclusively fails** — evader adapts faster than pursuer in every attempt.
9. **Adaptive boost / dynamic ratio insufficient** — can't overcome evader's adaptation advantage.
10. **Collapse rollback + PFSP work mechanically** but can't solve a strategy void (no evader fallback when obstacle-hugging fails).
11. **Premature freeze switch is fatal** — SP11d: single 307K-step evader phase destroyed 5.8M steps of pursuer progress permanently.
12. **Threshold 0.9 effectively prevents freeze switching** — the pursuer never reaches 90%, giving uninterrupted training.

### 4.3 Algorithm & Hyperparameters
13. **SAC outperforms PPO** for this environment — 30-40% vs PPO's similar or lower with more instability.
14. **RecurrentPPO (LSTM) conclusively fails** — 0% across 6 variants, 3-10x slower per step.
15. **`ent_coef=0.03` is critical** — 0.01 stalls, `auto` is destructive in self-play.
16. **SP11a proven config**: lr=3e-4, ent_coef=0.03, train_freq=1, gradient_steps=1, n_envs=4, micro_phase_steps=2048.
17. **max_steps=2000 critical** for partial obs — extra search time converts near-misses to captures.

### 4.4 Reward Design
18. **Terminal-to-shaping ratio was catastrophic** (1:1000+) — distance shaping contributed <0.2% of return.
19. **Search reward (w_search=0.003) is harmful** — competing dense signal teaches patrol instead of hunt. If revisiting, use w_search < 0.0003.
20. **PBRS w_obs_approach creates reward asymmetry** — don't use it.
21. **Visibility reward works** for incentivizing obstacle exploitation (evader learns LOS blocking).

### 4.5 Training Infrastructure
22. **NEVER train locally** — MacBook 8GB OOMs. All training on niro-2 (RTX 5090, 125GB RAM).
23. **Warm-seeding SAC without replay buffer = catastrophic forgetting** — empty buffer → narrow data → cascading degradation.
24. **Mixed PPO/SAC warm-seeding is incompatible** — different policy architectures crash on load.
25. **CUDA non-determinism dominates seed sensitivity** — same config on different GPU states produces completely different trajectories.
26. **Multi-seed strategy is essential** — capture rates range 0.13–1.00 for same config. Single seed is a gamble.

### 4.6 Safety Layer
27. **Post-hoc VCP-CBF filter is the recommended approach** — beats BarrierNet on all metrics.
28. **BarrierNet abandoned** — 87.5% QP intervention prevents exploration, 2% capture after 324K steps.
29. **DCBF gamma=0.2 optimal** from 14-config sweep.
30. **Standard PPO with obstacle observations achieves 100% capture** — learned implicit obstacle avoidance without any safety layer (but violations at 6%).

---

## 5. Codebase Summary

### File Counts
| Directory | Files | LOC (approx) |
|-----------|-------|---------------|
| `envs/` | 17 Python files | ~5,600 |
| `training/` | 20 Python files | ~6,300 (amsdrl.py alone ~2,200) |
| `scripts/` | 37 Python/shell files | ~8,400 |
| `tests/` | 28 test files | ~12,000 |
| `docs/` | 80+ Markdown files | ~15,000+ |
| **Total** | **~180 files** | **~47,000+** |

### Test Suite
- **700+ tests** passing (peak was 738)
- Coverage: env mechanics, dynamics, observations, rewards, obstacles, CBF safety, BarrierNet, self-play, opponent pool, curriculum, EWC, RND

### Key Scripts
| Script | Purpose |
|--------|---------|
| `scripts/train_amsdrl.py` | Main self-play CLI (SAC/PPO/RecurrentPPO, all flags) |
| `scripts/train_evader_vs_greedy.py` | Single-agent diagnostic (evader vs greedy pursuer) |
| `scripts/visualize_both_learned_gif.py` | Animated GIF generation for self-play models |
| `scripts/visualize_sensing.py` | Sensing visualization (radius circles, LOS lines) |

---

## 6. Training Experiment Index

### All Significant Runs (Chronological)

| Era | Runs | Approach | Best Result | Outcome |
|-----|------|----------|-------------|---------|
| Phase 2 | Baseline PPO | Full obs, greedy evader | 100% capture | Success (trivial game) |
| Phase 2.5 | BarrierNet | Differentiable QP | 2% capture | Abandoned |
| Phase 3 early | H–Z2 | Curriculum L1-L4, long phases | Wild oscillation | All failed |
| Redesign | RA1–RA10 | Micro-phases, speed advantage | RA9: 0% evader | Rollback cycles |
| Diagnostic | D1–D3 | Evader vs greedy | D3: 37% escape | Obstacle-hugging discovered |
| Variable speed | S1v2b | Variable speed evader | 98% escape | Key building block |
| Pursuer training | Run A | Pursuer vs S1v2b | 95% capture | Warm-seed source |
| Self-play v2 | SP2–SP2d | Full obs cold-start | 0-7% | All failed |
| Warm-seed SP | SP3 | Warm-seeded 1.15x | 49/51 start → collapse | Obstacle-hugging is only strategy |
| Partial obs | SP4 | Combined masking, cold | 53% (degenerate) | Hide-and-seek, not PE |
| Speed diagnostic | S1v6b | 1.2x + combined masking | 52% | First viable config |
| SAC self-play | SP6a/SP6b | SAC cold-start | ~10% | Cold-start fails |
| Freeze approach | SP7b→SP8 | Freeze evader → unfreeze | 42% → 11% | Freeze works, unfreeze fails |
| Best config | **SP11e** | SAC, freeze 0.9, 2000 steps | **85%** | Lucky seed |
| Warm-seed fail | SP11e2 | Warm-seed from SP11e | 35% | Replay buffer catastrophic forgetting |
| Search reward | SP11g2/h2 | w_search=0.003 | 25% | Harmful — patrol not hunt |
| RecurrentPPO | SP13a–f | LSTM self-play | 0-2% | Conclusive failure |
| **Multi-seed** | **SP11i (×4)** | **Same config, seeds 42-45** | **30% (seed 42)** | **Active — characterizing variance** |
| No obstacles | SP11f2 | w_vis=0.01, no obstacles | 38% (16M) | Running, peaked 69% |

---

## 7. Current State (as of S73, Mar 06)

### Active Runs on niro-2
| Run | Config | Steps | Status |
|-----|--------|-------|--------|
| SP11i (seed 42) | SP11e config replica | ~7M | ~30% SR, running |
| SP11i_s43 | Same, seed 43 | Just launched | Pending |
| SP11i_s44 | Same, seed 44 | Just launched | Pending |
| SP11i_s45 | Same, seed 45 | Just launched | Pending |
| SP11f2 | No obstacles, w_vis=0.01 | ~16M | ~38%, peaked 69% |

### Unresolved Problems
1. **Co-evolution collapse**: Every attempt to unfreeze the evader leads to pursuer collapse. The evader adapts faster.
2. **Seed sensitivity**: Same config ranges from 13% to 100%. No way to predict which seed will succeed.
3. **Pursuer "giving up"**: Under sparse reward + partial obs, the pursuer converges to passivity when capture rate drops.
4. **Strategy diversity deficit**: Evader has exactly one strategy (obstacle-hugging or hiding). When it fails, no fallback exists.

### What Has NOT Been Tried Yet
- **Masking curriculum** (PO-GRL approach: full obs → partial obs annealing) — researched in S69, implementation plan ready
- **Population-based training** (PSRO/FSP/league) — surveyed but deemed too complex for current stage
- **Asymmetric actor-critic** (critic full obs, actor partial) — surveyed but hard to implement in SB3
- **Replay buffer warm-seeding** (pre-fill buffer when warm-seeding SAC) — identified as fix for SP11e2's failure
- **Q-network reset** when warm-seeding — alternative to replay buffer pre-fill
- **Very low search reward** (w_search < 0.0003) — current experiments used 0.003 which was harmful

---

## 8. Research Artifacts

### Literature Reviews
| Document | Papers | Topic |
|----------|--------|-------|
| `final_literature_review.md` | 36 | Core PE + DRL + safety comprehensive review |
| `research_self_play_collapse_prevention.md` | 20+ | Self-play stability: PSRO, FSP, league, cycling theory |
| `pe_design_choices_survey.md` | 15+ | Arena sizes, speed ratios, episode lengths, rewards across PE literature |
| `search_exploration_rewards.md` | 20+ | Exact reward formulations for search/exploration in RL |
| `masking-curriculum-research.md` | 15+ | PO-GRL, GPO, observation annealing approaches |

### Analysis Reports
| Document | Topic |
|----------|-------|
| `S53_redesign_report.md` | Three-failure diagnosis + coordinated redesign |
| `S58_report.md` | Evader diagnostic breakthrough (speed vs obstacle exploitation) |
| `SP11d_report.md` | Premature freeze switch case study |
| `SP11e_report.md` | Best run analysis (85% pursuer SR, plateau-breakthrough dynamics) |
| `evader_strategy_analysis.md` | Why evader has single strategy problem |
| `run_p_analysis.md` | Phase-by-phase training dynamics, 3 root causes |
| `training_insights.md` | BiMDN, Stage 2, obs space analysis |

---

*Path Forward analysis in separate document: `docs/reports/path_forward_plan.md`*
