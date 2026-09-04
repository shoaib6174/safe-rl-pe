# Pursuit-Evasion Research Workflow Tracker
## Last Updated: 2026-09-04 (S121 — Strong-paper odds reassessed at 40–50 % (no learned agent yet). **Four pre-registered de-risk tests gate Steps 8–12 (D91, `docs/pathway/derisk_plan_2026-09-04.md`)**: T1 PPO+GAE pursuer, T2 belief-conditioned evader, T3 exact-game shield-shift twin, T4 Step 6 gates G1–G5; decision table pre-committed. Next: build PPO-BC path, launch T1/T2 on GPU, run T3/T4 on CPU. Paper 1 RA-L **held**.)

## Research Topic
**1v1 Pursuit-Evasion Games using Mobile Ground Robots with Deep RL**
- One pursuer, one evader
- Focus on deep reinforcement learning approaches
- Ground mobile robots (differential-drive / car-like)

## Current Blockers
<!-- Unresolved blockers from any session. Check at session start, update at session end. -->

| Blocker | Since Session | Details |
|---------|--------------|---------|
| ABL-3/ABL-4 divergence check | S88+ | ✅ **RESOLVED** |
| BR-5/BR-6 completion | S88+ | ✅ **COMPLETE** |
| SP17b_s49/s50 completion | S89 | ✅ **COMPLETE** |
| BR-ws-s50 warm-start | S89 | ❌ **FAILED** |
| RecurrentPPO partial obs | S89 | ❌ **FAILED** |
| BELIEF_s200 convergence | S90-S91 | ❌ **FAILED** |
| GI reproduction | S96 | ✅ **REPRODUCED** — Self-play CR=0.550 matches paper. |
| EMA seed 3 completion | S99 | ❌ **FAILED** — Training completed (5000 ep) but best robustness=0.292 (below 0.40 target). Evader-dominant collapse. Seed-dependent; seed 0 outlier at 0.698. |
| niro-2 stability | S98-S99 | ✅ **RESOLVED** — Rebooted, CUDA driver fixed, Tailscale re-authed. |
| **Pop-FSP competitive balance** | **S103** | **❌ FAILED** — 0/5 configurations (v3-v7) achieved CR≈0.5. Structural ∀/∃ asymmetry is fundamental. **Pivoted to EMA seeds 4-6 + GI vanilla 20K baseline.** |
| **A-EMA probe** | **S105** | **❌ NO-GO** — Seed 0 best robustness=0.060 (mean=0.026). Seeds 1, 2 collapsed. 0/3 with robustness > 0.40. Asymmetric LR alone fails. |
| **Transfer Learning v1** | **S106** | **❌ FAILED** — Immediate collapse at ep 2000. Root cause: EMA evader random at transition (deepcopy of never-trained actor). |
| **Transfer Learning v2** | **S106** | **❌ COMPLETE** — 0/3 seeds balanced. Seed 0: CR=1.0 (pursuer-dominant). Seeds 1,2: CR=0.0 (evader-dominant). Mixing + EMA re-init insufficient. |
| **Symmetric Point-Mass seeds 0-2** | **S106** | **✅ COMPLETE** — Seed 2 shows balanced cycling (37% CR=0, 36% CR=1, 26% CR=0.5). First genuine competitive cycling in project. Seeds 0/1 collapsed. |
| **Symmetric Point-Mass seeds 3-8** | **S107-S108** | **✅ COMPLETE** — All 9 seeds done. **2/9 balanced (22%): seeds 2 + 3.** 5/9 pursuer lock-in, 1/9 evader lock-in, 1/9 pursuer-biased. |
| **Symmetric evaluation** | **S108** | **✅ COMPLETE** — All 9 seeds evaluated. Seeds 2,3: CR≈0.5 all modes. **2/9 balanced (22%).** |
| **Shared Policy** | **S108-S109** | **❌ FAILED** — 0/3 seeds balanced. Root cause: separate critics have different value scales; shared actor follows well-trained dominant critic. |
| **PBT experiment** | **S109/S110/S112** | **❌ STOPPED on niro-1** — Final verdict: complete pursuer lock-in across all 3 seeds. Results archived to `repro_gonultas_isler_2024/archive/pbt_v3_final_20260714/`; findings in `docs/pbt_v3_findings_2026-07-14.md`. GPU freed for Experiment 3. |
| **Experiment 3: symmetric cohort 9-33** | **S113/S115/S117** | **✅ COMPLETE** — 25 seeds finished ~2026-07-15. Evaluated 2026-07-17; **re-evaluated 2026-08-03 (S117) under seeded paired protocol** (N=500; N=1000 band-edge). **Full cohort: 4/34 balanced (11.8%)**: seeds 2, 3, 20, 21 (21 in, 30 out vs S115). 26/34 pursuer lock-in/biased, 4/34 evader. Canonical data: `repro_gonultas_isler_2024/results_symmetric/eval_seeds_9_33_final.json`. |

## Session Index

| # | Date | Summary | Worklog |
|---|------|---------|---------|
| 1 | 2026-02-21 | Initial research report, read papers 11-14, paper collection README | [S01](worklogs/2026-02-21_S01.md) |
| 2 | 2026-02-21 | Workflow tracker, read papers 1-10 & 15-20, found corrupted papers | [S02](worklogs/2026-02-21_S02.md) |
| 3 | 2026-02-21 | Fixed corrupted papers 12/16/17, downloaded 9 new papers (21-28,32,34,35) | [S03](worklogs/2026-02-21_S03.md) |
| 4 | 2026-02-21 | SCP'd papers from remote, read 24/26/31/33, paywalled papers failed | [S04](worklogs/2026-02-21_S04.md) |
| 5 | 2026-02-21 | Read papers 30/36/37, finalized lit review — **36 papers total** | [S05](worklogs/2026-02-21_S05.md) |
| 6 | 2026-02-22 | Expert Spec Panel Review of Phase 1, resolved 14 issues (7.5 -> 8.8/10) | [S06](worklogs/2026-02-22_S06.md) |
| 7 | 2026-02-22 | Applied Phase 1 quality improvements to Phases 2-5 | [S07](worklogs/2026-02-22_S07.md) |
| 8 | 2026-02-22 | Self-play health monitoring research & Phase 3 integration | [S08](worklogs/2026-02-22_S08.md) |
| 9 | 2026-02-22 | Phase 1 pygame visualization & experiment tracking enhancement | [S09](worklogs/2026-02-22_S09.md) |
| 10 | 2026-02-22 | Propagated visualization & tracking to Phases 2-5 | [S10](worklogs/2026-02-22_S10.md) |
| 11 | 2026-02-22 | **Phase 1 Session 1**: Project scaffolding, env core, dynamics, rewards, wrappers, renderer, 41 tests | [S11](worklogs/2026-02-22_S11.md) |
| 12 | 2026-02-22 | **Phase 1 Session 2**: Observation & reward validation — already complete from S11 | [S12](worklogs/2026-02-22_S12.md) |
| 13 | 2026-02-22 | **Phase 1 Session 3**: PPO training pipeline, Hydra config, wandb/TB tracking, 47 tests pass | [S13](worklogs/2026-02-22_S13.md) |
| 14 | 2026-02-22 | **Phase 1 Session 4**: Self-play loop, health monitor, eval, 59 tests pass | [S14](worklogs/2026-02-22_S14.md) |
| 15 | 2026-02-22 | **Phase 1 Session 5**: Baselines (random, greedy, DQN, DDPG), comparison script, 72 tests | [S15](worklogs/2026-02-22_S15.md) |
| 16 | 2026-02-22 | **Phase 1 Session 6**: VCP-CBF validation — d=0.1, weighted QP, 35 tests, all 4 criteria pass | [S16](worklogs/2026-02-22_S16.md) |
| 17 | 2026-02-22 | **Phase 1 Session 7**: Integration testing, performance benchmarks, 118 tests — Phase 1 complete! | [S17](worklogs/2026-02-22_S17.md) |
| 18 | 2026-02-22 | **Phase 2 Session 1**: Beta distribution policy, SB3 integration, 20 tests, 138 total | [S18](worklogs/2026-02-22_S18.md) |
| 19 | 2026-02-22 | **Phase 2 Session 2**: Multi-constraint VCP-CBF, collision CBF, typed constraints, 19 tests, 157 total | [S19](worklogs/2026-02-22_S19.md) |
| 20 | 2026-02-22 | **Phase 2 Session 3**: Safe action bounds (LP + analytical), VCPCBFFilter integration, 25 tests, 182 total | [S20](worklogs/2026-02-22_S20.md) |
| 21 | 2026-02-22 | **Phase 2 Session 4**: CBF-Beta policy integration, SafeRolloutBuffer, CBFBoundsComputer, 22 tests | [S21](worklogs/2026-02-22_S21.md) |
| 22 | 2026-02-22 | **Phase 2 Session 5**: Obstacles, obstacle obs, safety reward, CBF overlay, 40 tests, 195 total | [S22](worklogs/2026-02-22_S22.md) |
| 23 | 2026-02-22 | **Phase 2 Session 6**: 3-Tier infeasibility (backup ctrl, hierarchical relaxation, SVM classifier), 33 tests | [S23](worklogs/2026-02-22_S23.md) |
| 24 | 2026-02-22 | **Phase 2 Session 7**: Safe self-play training, safety metrics, 4 comparison runs, 34 tests, 262 total | [S24](worklogs/2026-02-22_S24.md) |
| 25 | 2026-02-22 | **Phase 2 Session 8**: Ablation configs A-E, runner script, results compilation, 43 tests, 290 total — Phase 2 complete! | [S25](worklogs/2026-02-22_S25.md) |
| 26 | 2026-02-22 | **Phase 2.5**: Differentiable QP solver research — cvxpylayers, qpth, dQP, SCQPTH, BPQP, BarrierNet | [S26](worklogs/2026-02-22_S26.md) |
| 27 | 2026-02-22 | **Phase 2.5 Session 2**: Differentiable QP layer (qpth), 50 tests, 404 total, 2.5ms/solve | [S27](worklogs/2026-02-22_S27.md) |
| 28 | 2026-02-22 | **Phase 2.5 Session 3**: BarrierNet actor + PPO agent, rollout buffer, 32 tests, 436 total | [S28](worklogs/2026-02-22_S28.md) |
| 29 | 2026-02-22 | **Phase 2.5 Session 4**: Training integration, trainer script, 21 tests, 457 total | [S29](worklogs/2026-02-22_S29.md) |
| 30 | 2026-02-22 | **Phase 2.5 Session 5**: niro-2 deployment, comparison framework, 2 training runs launched, 21 tests, 478 total | [S30](worklogs/2026-02-22_S30.md) |
| 31 | 2026-02-22 | **Phase 2.5 Session 6**: CUDA optimization, hybrid CPU/GPU training (9x speedup: 40h→4.5h) | [S31](worklogs/2026-02-22_S31.md) |
| 32 | 2026-02-22 | **Phase 2.5 Session 7**: Gradient flow fix, 3-way evaluation, decision report — recommend post-hoc VCP-CBF filter | [S32](worklogs/2026-02-22_S32.md) |
| 33 | 2026-02-22 | **Phase 2.5 Final**: Retrained baseline with obstacles (100% capture), final 200-ep evaluation, decision report finalized | [S33](worklogs/2026-02-22_S33.md) |
| 34 | 2026-02-22 | **Phase 2.5**: Baseline comparison visualizations (23 plots), 3-way eval monitoring | [S34](worklogs/2026-02-22_S34.md) |
| 35 | 2026-02-22 | **Phase 2.5 Filter Tuning**: CBF filter fixes (epsilon, DCBF, CBF-RL reward), 14-config sweep — DCBF gamma=0.2 optimal | [S35](worklogs/2026-02-22_S35.md) |
| 36 | 2026-02-22 | **Phase 3 Spec Panel Review**: 21 issues fixed (3 critical, 10 high, 8 medium), quality 7.0→8.5+/10 | [S36](worklogs/2026-02-22_S36.md) |
| 37 | 2026-02-22 | **Spec Enhancement**: 9 research additions integrated into Phases 3-5 (DCBF theorem, complex envs, asymmetric, HJ baseline, TB3, generalization, interpretability, opponent modeling, human evader) | [S37](worklogs/2026-02-22_S37.md) |
| 38 | 2026-02-22 | **Phase 3 Spec Panel Review #2**: 24 issues fixed (1 critical, 7 high, 11 medium, 5 low), quality 8.3→9.0+/10, tests 26→31, examples 6→9 | [S38](worklogs/2026-02-22_S38.md) |
| 39 | 2026-02-22 | **Phase 3 Sessions 1-2**: Sensors (FOV, Lidar, WallSegment, buffer, wrapper) + BiMDN (encoder, policy network), 29 tests, 123 core passing | [S39](worklogs/2026-02-22_S39.md) |
| 40 | 2026-02-22 | **Phase 3 Training**: Fixed obs space (added own pose, 40→43 dims), 3 Stage 1 runs (In-FOV 0.31m), Stage 2 all 0% capture — Gates 1&2 failed | [S40](worklogs/2026-02-22_S40.md) |
| 41 | 2026-02-22 | **Phase 3 Session 3**: AMS-DRL self-play — opponent adapter, NavigationEnv, CheckpointManager, health callbacks, AMSDRLSelfPlay orchestrator, NE tools, 38 tests (474 total) | [S41](worklogs/2026-02-22_S41.md) |
| 42 | 2026-02-23 | **Phase 3 Session 5**: Curriculum learning — CurriculumManager (4-level progressive difficulty), AMS-DRL integration, --curriculum CLI flag, 29 tests (503 total) | [S42](worklogs/2026-02-23_S42.md) |
| 43 | 2026-02-23 | **Phase 3 Session 7**: Opponent pool for self-play diversity — OpponentPool class, per-sub-env sampling, FIFO eviction, 16 tests | [S43](worklogs/2026-02-23_S43.md) |
| 44 | 2026-02-24 | **Phase 3 Research**: PBRS obstacle-seeking — literature review (5 papers), sensory feasibility analysis, arena size impact, research report | [S44](worklogs/2026-02-24_S44.md) |
| 45 | 2026-02-24 | **Phase 3 Implementation**: PBRS obstacle-seeking — `nearest_obstacle_distance()`, `w_obs_approach` param, env tracking, 17 new tests (all pass) | [S45](worklogs/2026-02-24_S45.md) |
| 46 | 2026-02-24 | **Phase 3 Research**: Self-play collapse prevention — 6 approaches (asymmetric training, dual-criteria curriculum, PBT, FSP/PSRO, PAIRED, exploitability), 20 papers, action plan | [S46](worklogs/2026-02-24_S46.md) |
| 47 | 2026-02-24 | **Phase 3 Analysis**: Run P failed (L3+ collapse persists) — 3 root causes identified, 3-pronged fix proposed, 21 BibTeX refs added | [S47](worklogs/2026-02-24_S47.md) |
| 48 | 2026-02-24 | **Phase 3 Fix**: Three-pronged fix (dual-criteria gate, regression, asymmetric training), reward rebalancing (timeout=-100, survival_bonus=0), equal speeds. Run Q+R launched. 14 new tests (590 pass) | [S48](worklogs/2026-02-24_S48.md) |
| 49 | 2026-02-25 | **Phase 3 Monitoring**: Run Q killed (L4 collapse). Run R collapsing at L3 (gate=0.05 too low). Launched Runs S (20x20, gate=0.15) + T (10x10, gate=0.15). 3-run experimental matrix. | [S49](worklogs/2026-02-25_S49.md) |
| 50 | 2026-02-25 | **Obstacle Physics Fix**: `resolve_obstacle_collisions()` — hard projection with tangential sliding, `w_collision` penalty, 13 new tests (603 total passing) | [S50](worklogs/2026-02-25_S50.md) |
| 51 | 2026-02-25 | **Wall Penalty + L2 Collapse Investigation**: `w_wall` impl, penalty lit review (N16–N20), killed R/S/T, Run U collapsed at L2, root cause analysis (8 causes, 9 fixes) | [S51](worklogs/2026-02-25_S51.md) |
| 52 | 2026-02-26 | **L2 Collapse Countermeasures + L1 Oscillation Fixes**: Tier 1-3 (EWC+RND), 2 bugs fixed, seed sensitivity discovery (massive variance). 3 structural fixes: cold-start checkpoint, phase warmup (100K→500K), NE-gap curriculum advancement. Runs Z1/Z2 launched. 57 new tests (687 pass) | [S52](worklogs/2026-02-26_S52.md) |
| 53 | 2026-02-26 | **Self-Play Stabilization Research**: 30+ papers reviewed — simultaneous vs alternating (SimGD diverges), PSRO/FSP/league (population prevents cycling), phase length (10K-50K not 500K), PE-specific methods, simple baselines, cycling theory. Key finding: our long frozen phases are fundamentally wrong. | [S53](worklogs/2026-02-26_S53.md) |
| 54 | 2026-02-26 | **PE Design Choices Survey**: 15+ papers benchmarked — arena sizes (0.9m-20km), speed ratios (0.8-1.3x), episode lengths (25-3000 steps), reward designs (dense/sparse/zero-sum), curricula, action spaces. Key finding: equal-speed 1v1 in bounded domain is trivially solvable; most papers give evader 10-30% speed advantage. | [S54](worklogs/2026-02-26_S54.md) |
| 55 | 2026-02-26 | **RA Redesign Implementation**: Implemented micro-phase rapid alternation (2048-step phases), evader speed advantage (1.15x), rebalanced rewards (±10), obstacle observability fix, opponent pool integration. 8 new CLI args, 5 new methods, 3 files modified. Smoke-tested both RA1 and RA2. All 687 existing tests pass. | [S55](worklogs/2026-02-26_S55.md) |
| 56 | 2026-02-27 | **Pool Reservoir + Convergence Fix + RA3-RA6**: Reservoir sampling, convergence_consecutive=5, 9 new tests (696 total). RA3/RA4 collapsed (44x PBRS reward asymmetry). Root cause analysis → launched RA5/RA6 with reward fix (10x10, 1.05x speed, ±50 terminal, PBRS 5.0 vs 0.0). | [S56](worklogs/2026-02-27_S56.md) |
| 57 | 2026-02-27 | **Adaptive Training Ratio + LR Dampening**: Anti-cycling mechanisms — adaptive ratio gives loser extra training phases when gap > threshold, LR dampening scales LR down near equilibrium. RA7b/RA8 completed 10M steps — both show evader collapse at M400, adaptive ratio insufficient. Root cause: vanishing reward gradient, no recovery mechanism. | [S57](worklogs/2026-02-27_S57.md) |
| 58 | 2026-02-28 | **Collapse Rollback + PFSP-lite + Survival Bonus**: 3-pronged fix for evader collapse — collapse rollback (restore best checkpoint when SR < threshold for N evals), PFSP-lite (bias pool toward weaker opponents when losing), configurable survival_bonus. 7 new tests (706 total). Launching RA9/RA10. | [S58](worklogs/2026-02-28_S58.md) |
| 59 | 2026-03-01 | **Variable Speed Self-Play**: S1v2b evader 98% escape. SP2+B+C+D all failed (pursuer can't bootstrap in self-play). Run A pursuer training hit 94% capture. Launched SP3: warm-seeded self-play starting at 49/51 balance. | [S59](worklogs/2026-03-01_S59.md) |
| 60 | 2026-03-01 | **Asymmetric Obs + Randomized Obstacles**: S1v4a (asymmetric LOS) peak 80%→20% final, S1v4b (symmetric) peak 73%→50% final. Late regression dominant problem. 5 new tests (727 pass). | [S60](worklogs/2026-03-01_S60.md) |
| 61 | 2026-03-01 | **S1v5: Radius Sensing + Combined Masking**: Radius-based sensing (3.0m), combined masking mode (radius+LOS), sensing visualization script, best-model checkpointing. 9 new tests (736 pass). S1v5 (radius-only) + S1v5b (combined) launched on niro-2. | [S61](worklogs/2026-03-01_S61.md) | [S60](worklogs/2026-03-01_S60.md) |
| 62 | 2026-03-01 | **S1v5c: Greedy Full Obs Fix**: `greedy_full_obs` flag gives greedy pursuer unmasked obs while evader stays under partial obs. Fixes inflated S1v5b escape rate (pursuer was also blinded). 2 new tests (738 pass). | [S62](worklogs/2026-03-01_S62.md) |
| 63 | 2026-03-01 | **SP4: First Successful Cold-Start Self-Play**: Combined partial obs (3.0m+LOS) + variable obstacles (0-3). Best-model checkpointing + eval fix in amsdrl.py. Pursuer bootstrapped to 53% (M2550), min gap=0.02. 10M steps, 1.7h. | [S63](worklogs/2026-03-01_S63.md) |
| 64 | 2026-03-02 | **SP4 Degenerate + Speed Diagnostics**: SP4 trajectory analysis — evader hides (0.20 m/s), no PE dynamics. S1v6 (1.05x) failed. **S1v6b (1.2x) holding 35-50%** — first viable speed+partial_obs config. | [S64](worklogs/2026-03-02_S64.md) |
| 65 | 2026-03-02 | **SP5: Partial Warm-Seed + Launch Prep**: Implemented partial warm-seeding in `_warm_seed()` — load only evader (S1v6b), create pursuer from scratch. SP5 config: 1.2x speed, combined masking, 0-3 obstacles. | [S65](worklogs/2026-03-02_S65.md) |
| 66 | 2026-03-02 | **SAC Self-Play + Freeze→Unfreeze**: SAC in self-play, `--freeze_role`, eval_interval tuning. SP7b2 freeze evader climbed to **42%** pursuer SR. Unfrozen as **SP8** (co-evolution). RunD (1.1x evader) hit 81%. GIF viz confirms no obstacle use — pure speed strategy. SP7a2 adaptive boost consistently failing. | [S66](worklogs/2026-03-02_S66.md) |
| 67 | 2026-03-03 | **Exploration reward attempted then reverted**: Implemented `--w_exploration` but removed — agent lacks observational capacity to exploit it. Relaunched SP11a/SP11b to train longer. | [S67](worklogs/2026-03-03_S67.md) |
| 68 | 2026-03-03 | **RecurrentPPO (LSTM) support**: Added `sb3-contrib` + `RecurrentPPO` with `MlpLstmPolicy` to self-play framework. Builder, adapter, dispatch, recurrent eval. Ready for SP13a/SP13b launch. | [S68](worklogs/2026-03-03_S68.md) |
| 69 | 2026-03-03 | **Masking Curriculum Research**: 15+ papers surveyed — PO-GRL (full→partial annealing with SAC), GPO, asymmetric actor-critic, observation dropout, PE-specific curricula. Concrete implementation strategies identified. | [S69](worklogs/2026-03-03_S69.md) |
| 70 | 2026-03-04 | **Search/Exploration Reward Literature Review**: 20+ papers surveyed across PE/POMDP, multi-agent search, hide-and-seek, coverage/patrol, intrinsic motivation. Exact reward formulations documented. Top candidates: entropy reduction, 1/sqrt(visit_count), scan staleness. | [S70](worklogs/2026-03-04_S70.md) |
| 71 | 2026-03-04 | **Staleness-Based Search Reward**: Implemented `SearchStalenessTracker` — 10x10 grid, tracks cell observation times, rewards pursuer for visiting stale cells. New `--w_search`/`--t_stale` CLI flags. 733 tests pass. | [S71](worklogs/2026-03-04_S71.md) |
| 72 | 2026-03-05 | **Disk cleanup + 4 new runs**: All runs died from disk full. Freed 45GB. Added `--freeze_switch_consecutive` + visibility-gated search reward. Launched SP11e2 (warm-seed), SP11f2 (3-consec freeze), SP11g2/h2 (gated search). | [S72](worklogs/2026-03-05_S72.md) |
| 73 | 2026-03-06 | **Diagnosis + multi-seed sweep**: SP11e2/g2/h2 all underperforming. Code bisect confirmed zero regression. Root cause: CUDA non-determinism + seed sensitivity. Killed weak runs + ComfyUI. Launched 4-seed sweep (SP11i seeds 42-45). | [S73](worklogs/2026-03-06_S73.md) |
| 74 | 2026-03-06 | **Comprehensive project report**: Read all 73 worklogs + research docs. Created full project report (findings, lessons, experiment index) and path forward plan (masking curriculum → graduated co-evolution). | [S74](worklogs/2026-03-06_S74.md) |
| 75 | 2026-03-06 | **Masking curriculum + launch**: Implemented PO-GRL p_full_obs annealing (wrapper+loop+CLI, 2 tests). Launched SP12_mc (4 seeds, 5M anneal), SP12b (2 seeds, 10M slow anneal), SP12c (2 seeds, w_search=0.0001). Moved results to /mnt/data2 (symlink). SP12_mc showed +10-15pp over baseline before restart. | [S75](worklogs/2026-03-06_S75.md) |
| 76 | 2026-03-08 | **FOV partial obs + tooling**: Implemented FOV triangle sensing (`in_field_of_view`, `_is_visible`). Built remote results viewer (:8080) + dashboard. Set up TensorBoard (:6006) with `history_to_tb.py` backfill. Launched SP13a (FOV 90°/8m, no obs) + SP13b (FOV 90°/8m, obs+LOS). 12 runs total. | [S76](worklogs/2026-03-08_S76.md) |
| 77 | 2026-03-09 | **SP12 killed + SP13 expansion**: SP12 post-mortem (3m omni failed, peak 0.48). Fixed FOV wedge viz. Launched SP13c/d (5m range), SP13e/f (8m alternate freeze), SP13g/h (asymmetric obs). 16 runs on niro-2. | [S77](worklogs/2026-03-09_S77.md) |
| 78 | 2026-03-09 | **Novelty assessment + threshold ablation**: Assessed paper novelty (integration + FOV sensing + masking curriculum). Launched SP13i/j (alternate_freeze 0.8) for threshold comparison vs SP13e/f (0.6). 20 runs on niro-2. |
| 79 | 2026-03-09 | **Run status + alt-freeze report**: Full metrics for all 20 runs. Alt-freeze analysis: thr=0.6 oscillates, thr=0.8 more stable, asymmetric s42 failing (seed). Launched SP13k/l (asymmetric+0.8, s43). TB backfilled. 22 runs. | [S79](worklogs/2026-03-09_S79.md) |
| 80 | 2026-03-10 | **Group-by-group SP13 report + SP14 launch**: Detailed analysis of all 22 runs. SP13a_s42 best (CR=0.82 at full PO), SP13j_s43 best self-play (CR=0.73). Launched SP14a/b: pure self-play (no freeze), FOV 90°, 8m, asymmetric obs. 23 runs. | [S80](worklogs/2026-03-10_S80.md) |
| 81 | 2026-03-10 | **Status report**: SP13k_s43 new top run (CR=0.82, rising), SP14b_s43 surging (+0.42). SP13j_s43 regressed (0.87→0.57). Kill recommendations: c-d + g-h (8 cores). | [S81](worklogs/2026-03-10_S81.md) |
| 82 | 2026-03-15 | **SP13/14 final analysis + SP15 launch**: SP13j_s43 champion (CR=0.69). Diagnosed γ=0.2 lazy pursuit. Implemented freeze_thr_schedule + self_play_start_steps. Launched 10 SP15 runs: γ=0.9/0.99, EWC, vis reward, stepped schedule→pure SP. | [S82](worklogs/2026-03-15_S82.md) |
| 83 | 2026-03-16 | **SP15 analysis + SP16 launch**: SP15 killed (no switches, thr=0.8 unreachable). Vis reward only thing that worked (0.57 vs 0.10). Added force_first_switch_steps. Launched 8 SP16 runs: vis reward + thr=0.65/0.55, forced switch, γ=0.99, vis=0.2, EWC. | [S83](worklogs/2026-03-16_S83.md) |
| 84 | 2026-04-28 | **SP16 status check + analysis + visualization**: 4 converged runs analyzed on niro-2, GPU-rendered GIFs for all 4, frame-by-frame skill analysis, 2 reports written. s42/s44 crash bug identified. | [S84](worklogs/2026-04-28_S84.md) |
| 85 | 2026-04-29 | **SP17 design, implementation, and launch**: Brainstormed path to paper. Fixed forced switch placement bug (inside eval block). Crash bug could not reproduce. Launched 4-seed SP17 on niro-2. | [S85](worklogs/2026-04-29_S85.md) |
| 86 | 2026-04-30 | **SP17 s43 behavioral analysis + SP17b monitoring**: Trajectory visualizations generated (3×3 grid, 7/9 captures). Discovered CPU viz is 60× slower than GPU; created GPU script. SP17b seeds 46-49 running, logs buffered. s43 stable at gap=0.08, s45 gap widened to 0.28. | [S86](worklogs/2026-04-30_S86.md) |
| 87 | 2026-05-01 | **CoRL 2026 Sprint — Plan & Launch**: CoRL feasibility assessment, hybrid experimental plan (niro-1 + niro-2), niro-2 cleanup (ComfyUI killed, SP12 deleted), PyTorch upgrade on niro-1 (2.10.0+cu128), launched s50/s51 on niro-2 + ABL-1/ABL-2 on niro-1, LaTeX skeleton created. | [S87](worklogs/2026-05-01_S87.md) |
| 88 | 2026-05-02 | **BR exploitability test for SP17b cohort**: trajectory analysis revealed curriculum-end drift; implemented `scripts/train_br_sac.py` + `scripts/analyze_exploitability.py` (TDD); ran BR-1..BR-4 (4× 1.5M steps) on s48/s49 frozen at M2550; **verdict: H3** — both seeds L3 (evader exploitable, pursuer not), gap +0.08 to +0.10 > ε=0.05. Framing C adopted: paper reframes to characterize curriculum-induced exploitability hole. Abstract updated in `paper/main.tex`. | [S88](worklogs/2026-05-02_S88.md) |
| 89 | 2026-05-03 | **S88 completion — visualizations + paper intro + git**: Generated self-play GIFs on niro-2 (obstacle + open arena), added visualization section to README, updated paper Introduction for Framing C, committed to git. Push pending (no GitHub auth on niro-1). | [S89](worklogs/2026-05-03_S89.md) |
| 90 | 2026-05-07 | **Belief-state self-play implementation**: Implemented BeliefTracker (particle filter), integrated into PartialObsWrapper + policy network + opponent adapter. Two critical runtime fixes (KeyError 'belief', obs space mismatch). Launched BELIEF_s200 on niro-2. Design spec + impl plan written. All 13 tests pass. | [S90](worklogs/2026-05-07_S90.md) |
| 91 | 2026-05-08 | **Status check + BELIEF_s200 failure discovery**: Checked all running experiments. ABL-1/ABL-2 completed. BELIEF_s200 CR peaked at 0.48 then declined to 0.10 — **validation FAILED**. Key finding: drift is structural, not curriculum-induced. Updated training-runs.md and docs. | [S91](worklogs/2026-05-08_S91.md) |
| 92 | 2026-05-08 | **Gonultas & Isler reproduction — full implementation + AMP + parallel launch**: Implemented all 12 components (env, MADDPG, BiMDN, curriculum, training loop). Added mixed precision (AMP) support. Evaluated vanilla at ep 2500/4000 (CR=0.81). Launched GI-v1-amp (vanilla) and GI-v2-amp (BiMDN) in parallel on niro-2. Early BiMDN result: CR=1.00 at ep 311 vs vanilla CR=0.00 at ep 757. | [S92](worklogs/2026-05-08_S92.md) |
| 92 | 2026-05-08 | **Gonultas & Isler reproduction — full implementation**: 12-task inline execution. MADDPG + BiMDN standalone repro in `repro_gonultas_isler_2024/`. 35 tests passing. Training smoke test confirms end-to-end execution. `use_belief: false` default — run vanilla first, then toggle BiMDN. | [S92](worklogs/2026-05-08_S92.md) |
| 93 | 2026-05-12 | **Gonultas & Isler reproduction — Debugging NaN & Q-value explosion**: Diagnosed BiMDN NaN losses (scale mismatch + AMP trigger). Implemented twin critics (Ackermann 2019), gradient clipping, fixed wheelbase (0.15→0.30) and curriculum duration (0.3→0.1). Synced corrected code to niro-2, relaunched seeds 4/5/6. Early metrics: 19ep, no NaNs. | [S93](worklogs/2026-05-12_S93.md) |
| 94 | 2026-05-13 | **GI reproduction — Parallel vectorized envs + actor fix**: Fixed actor update to use `min(q1_actor, q2_actor)` (root cause of s5/s6 NaN). Implemented `VectorizedEnvRunner` for 8 parallel envs matching paper. Rewrote `train.py` with step-based continuous rollout. Smoke tests pass. Launched GI-v4-vectorized seeds 7/8/9 on niro-2. | [S94](worklogs/2026-05-13_S94.md) |
| 95 | 2026-05-15 | **GI reproduction — 5 critical fixes + SP17 launch**: Implemented all fixes from paper audit (actor obs bug, BiMDN warm-up, BiMDN arch, tournament eval, seed paths). Smoke test confirms evader now learns (non-zero EA). Launched SP17 seeds 0/1/2 on niro-2 (20K ep, 8 envs, CUDA). | [S95](worklogs/2026-05-15_S95.md) |
| 96 | 2026-05-16 | **GI reproduction verification + SP18 launch**: Fixed capture counting bug. Confirmed oscillatory competitive dynamics. Self-play CR=0.550 matches paper. vs-random=0.936 near paper. SP18 (seeds 3-5, 30K ep) launched on niro-2. | [S96](worklogs/2026-05-16_S96.md) |
| 97 | 2026-05-18 | **EMA opponent stabilization — Phase 1**: Refactored MADDPG.update() for external opponent actors. Implemented EMA with tau=0.001 (3.5-episode half-life). Added checkpoint persistence, EMA evaluation modes, sweep script with robustness + oscillation metrics. Critical analysis fixed 10 issues. 5K comparison (vanilla vs EMA) launched on niro-2. | [S97](worklogs/2026-05-18_S97.md) |
| 98 | 2026-05-20 | **EMA 5K analysis — seeds 0-2**: Seed 0 EMA robustness=0.698 (exceeds target). Seed 1 vanilla NaN collapse, EMA evader-dominant. Seed 2 low robustness (0.148) due to actor_lr=0.0001. Key finding: LR=0.0005 needed. Analysis doc written. | [S98](worklogs/2026-05-20_S98.md) |
| 99 | 2026-05-22 | **EMA seed 3 monitoring + departure**: Relaunched seed 3 with LR=0.0005 after niro-2 reboot/CUDA fix. Training at ep 1573 (29.6%), CR oscillating, losses stable. Persistent monitor active (30-min intervals). User leaving for 2 weeks. | [S99](worklogs/2026-05-22_S99.md) |
| 100 | 2026-05-23 | **Pop-FSP implementation + first training**: Brainstormed, designed, and implemented actor-only PopulationManager. Integrated into training loop. Fixed 3 bugs (frame_stack=1 slice, hidden_dim missing, duplicate adds). 500-ep baseline completed (CR=0.000, evader-dominant as expected). Pop-FSP 5K run launched and monitoring. | [S100](worklogs/2026-05-23_S100.md) |
| 101 | 2026-05-24 | **Pop-FSP v2 balance fix**: Diagnosed pursuer-dominant root cause (structural ∀/∃ asymmetry). Implemented recency-biased sampling + asymmetric populations (16 vs 8) + gradual mixing. 12 tests pass. 3-seed training launched on niro-2. | [S101](worklogs/2026-05-24_S101.md) |
| 102 | 2026-05-24 | **Pop-FSP v3/v4/v5 postmortem + v6 launch prep**: v3 all seeds CR=1.0 (pursuer-dominant). v4 overcorrected to evader-dominant. v5 still pursuer-dominant. v6 (arena=19, evader=1.65, ratio=1.52x) prepared as intermediate sweet spot. | [S102](worklogs/2026-05-24_S102.md) |
| 103 | 2026-05-25 | **Pop-FSP v6/v7 complete failure + pivot**: 0/5 configs achieved balance. Structural ∀/∃ asymmetry identified. Launched EMA seeds 4-6 + GI vanilla 20K baseline. Wrote asymmetry analysis doc. | [S103](worklogs/2026-05-25_S103.md) |
| 104 | 2026-05-26 | **Full project reanalysis + ICRA 2027 master plan**: Comprehensive achievement assessment, publication strategy, phased plan with decision gates and contingencies. | [S104](worklogs/2026-05-26_S104.md) |
| 105 | 2026-06-03 | **A-EMA Probe implementation + results**: Fixed config, launched 3 seeds on niro-2, all complete. Seed 0 best robustness=0.060. **NO-GO** — 0/3 seeds pass. Checkpoint sweeps running for seeds 1, 2. | [S105](worklogs/2026-06-03_S105.md) |
| 106 | 2026-06-04 | **Transfer Learning diagnosis + Symmetric experiment**: Found root cause (random EMA at transition), implemented EMA re-init + mixing. v2 complete: 0/3 balanced. Theoretical analysis of why self-play fails. Launched symmetric point-mass experiment based on Nash equilibrium theory. | [S106](worklogs/2026-06-04_S106.md) |
| 107 | 2026-06-05 | **Symmetric results + new seeds**: Seeds 0-2 complete. Seed 2 shows balanced cycling! Eval running. Launched seeds 3-5 for reproducibility. | [S107](worklogs/2026-06-05_S107.md) |
| 108 | 2026-06-08 | **niro-2 recovery + full eval** — All 9 seeds complete. **2/9 balanced (seeds 2, 3).** niro-2 rebooted, results recovered. Evals copied locally. | [S108](worklogs/2026-06-08_S108.md) |
| 109 | 2026-06-08 | **PBT implementation + launch** — Diagnosed shared policy failure (critic scale imbalance). Implemented PBT with population of 5, tournament + weakest replacement. Launched 3 seeds on niro-2. | [S109](worklogs/2026-06-08_S109.md) |
| 110 | 2026-07-14 | **niro-2 offline + PBT relocation** — Could not reach niro-2 (Tailscale offline, SSH timeout). Verified niro-1 has RTX 5090/125 GB RAM/CUDA 13.0 and can run training. Preparing to launch PBT v3 seeds on niro-1. | [S110](worklogs/2026-07-14_S110.md) |
| 111 | 2026-07-14 | **PBT v3 stopped/collapsed** — Processes no longer running; seeds 1/2 pursuer lock-in, seed 0 unstable. Documentation updated. Pivot to paper writing. | [S111](worklogs/2026-07-14_S111.md) |
| 112 | 2026-07-14 | **PBT v3 relaunched** — User confirmed PC was stopped. Archived partial run, relaunched seeds 0-2 on niro-1 (PIDs 10679, 10744, 10810), verified GPU/utilization. | [S112](worklogs/2026-07-14_S112.md) |
| 113 | 2026-07-14 | **Publishability assessment + Experiments 1–4 plan + Experiment 1 complete + Experiment 3 launch** — ARS-3W confirms ICRA 2027 publishable. Route B decided (RA-L → ICRA). RA-L draft created. PBT v3 stopped and archived (pursuer lock-in). Tabular ∀/∃ analysis complete (Nash balanced; pure-strategy commitment favors pursuer). Symmetric seeds 9–33 launched. | [S113](worklogs/2026-07-14_S113.md) |
| 115 | 2026-07-17 | **Experiment 3 evaluated + Figure 3 updated** — Evaluated symmetric seeds 9–33. Full cohort: 4/34 balanced. Updated `paper_data_summary.md`, `training-runs.md`, figure spec, and RA-L draft. | [S115](worklogs/2026-07-17_S115.md) |
| 116 | 2026-08-03 | **RA-L Priority 1 integration** — Experiment 1 (tabular ∀/∃) added as Results §IV-C with figures; Wilson CIs + binomial tests for all balance rates; PBT v3 negative-evidence subsection; frozen-opponent Evaluation Protocol (Methods §III-H); fixed broken/truncated references.bib and added real citations. Draft compiles clean at 4 pages. Experiments 2 & 4 descoped from RA-L. | [S116](worklogs/2026-08-03_S116.md) |
| 117 | 2026-08-03 | **Seeds 9/16 eval discrepancy root-caused + seeded re-eval** — Training-log "CR=0.500" was a 2-episode window (true log means 0.468/0.893); opponent non-stationarity dominates over measurement protocol (frozen nets under training conditions: 0.250/0.640). Found env resets unseeded → N=50 evals irreproducible. Seeded paired re-eval of cohort 9–33 (N=500; N=1000 band-edge): balanced set {2, 3, 20, 21} (21 in, 30 out), tally stays 4/34. §III-H rewritten with mechanism attribution; Figure 4 regenerated; draft at 5 pages, compiles clean. | [S117](worklogs/2026-08-03_S117.md) |
| 118 | 2026-08-23 | **Pathway B kickoff — reference docs + verifiable plan** — Scoped six-element system paper (spine: safety throughout adversarial co-adaptation; convergence = bounded measured exploitability). Full codebase inventory + test verification (root 750/818 pass, all failures = abandoned BarrierNet `qpth`; repro 48/48). Created `docs/reference/lessons_learned.md` (rules L1–L5b, all cited) + `docs/reference/codebase_reference.md` (reuse map: elements 2/3/4/6 engineering-complete, 1 built-untested-under-coadaptation, 5 = the research). Wrote six-phase plan `docs/pathway/pathway_B_safe_exploitability_bounded_PE.md`: Phase 0 reproducibility+harness before training → env viability → shield integration study → NFSP convergence (kill gate) → cohort+ablations → writing. Decisions D60–D63. | [S118](worklogs/2026-08-23_S118.md) |
| 119 | 2026-08-24/25 | **Pathway B Steps 0–5 — BR harness gate PASSED, imitation-first established** — pe2 foundations (D65 CI, D66 seed contract) → env+renderer → PO sensing/occlusion → scripted agents + seeded battery (D55–D57 conventions) → belief baseline (D67 BiMDN deferred) → Step 5 exploitability harness: **D76 battery gate PASSED by BC clone (0.185 [0.137, 0.245], McNemar p=1.26e-3)** + 3×3 episode grid + scenario suite (D70; occluded/shadow 0.00 = Step 7 target). Complete RL-failure arc root-caused: 5 from-scratch arms (D68–D74) + 4 fine-tune configs (naive/D78/D80/D81) — entropy-term swamps ±1 reward ~40:1 at γ=0.999, auto-α explodes under frozen actor (D81: fixed ent_coef=1e-5 + LN critics); phase-B collapse under residual TD drift (D82: RL fine-tune refuted at 1M, deferred to Step 7 belief policy / n-step·TD(λ) library scope). Evader BC refuted — non-Markov teacher (D79). Unconfounded LN-scratch control COMPLETE: 1M flat (max 3/100, gate 0.005) — D68–D74 confound closed. 145/145 tests. **Step 5 SIGNED OFF by user 2026-08-25 (all three asks accepted) + D83 gate refinements for Step 6+ + D84 (Step 7 RL-half retry = PPO-GAE, no library rewrite).** | [S119](worklogs/2026-08-24_S119.md) |
| 120 | 2026-09-03 | **Paper 2 direction: "Shielded Games" + real robots (D86)** — hostile review of plan v2 (headline tautological for a projection shield; novelty unverified; no external baselines) → novelty audit (CEP May-2026 predicts "conservative safe equilibria" under projection shields; CBF-RL Oct-2025 shield-in-training single-agent; six-element framing crowded) → user chose study core + safety-first system + 2× TB3 → plan v3 (R1–R4, H1–H5, Steps 3b/7–12, 8-alt fallback, similarity-transform hardware scaling), reconciled with S119 execution state; D85 reconstructed from `pe2/shield.py`; RA-L held. **Verification pass + reading list (D87/D88)**: CBF-RL confirmed ICRA-2026 single-agent w/ humanoid hardware (Step 7 = replication of it), Wang confirmed TNNLS-2026 sim-only, CEP/Neurocomputing paywalled 403; new plan-changing finds — Timbers IJCAI-2022 (exploitability = lower bound), Gleave ICLR-2020 + Wang ICML-2023 (adversarial-policy template; adversarial training did not remove the Go exploit → R3 must not overclaim), Grover deadlock theory (shield standoffs, now a Step 6 diagnostic), Agrawal & Sreenath RSS-2017 (DCBF citation), measurement-robust CBFs (principled hardware δ). Hardware confirmed → Step 12 committed. | [S120](worklogs/2026-09-03_S120.md) **Step 6 parts 1+2 executed** as a reviewed plan (`docs/superpowers/plans/2026-09-03-step6-backup-fix-and-gate-metrics.md`): backup-tier tie-break fix + infeasibility/clearance/tier-histogram/stall/immobility/pinning metrics + `run_shielded_duel` + `summarize_shield` + `NoOpShield` control. 6 tasks each reviewed, then a whole-branch review (1 Critical + 7 Important), one fix wave, one residual correction, each re-reviewed. Critical was a mixed-unit reportable margin (D89). 182→229 tests. |
| 121 | 2026-09-04 | **De-risk plan (D91)** — odds: strong 40–50 %, good 70–80 %, publishable >85 %; single risk = Step 7 learning (D68–D82, D79). Pre-registered T1 (PPO+GAE pursuer beats battery + SAC clone + own BC start, ≥2/3 seeds, 2M steps, random-init negative control), T2 (belief-conditioned evader clone ≥0.15 then fine-tune; RecurrentPPO fallback; D79 clone as negative control), T3 (exact gridworld twin with hazards + projection shield, |ΔV|>0.02 on ≥2/3 layouts = R1 direction), T4 (G1–G5). Decision table maps T1×T2 outcomes to four paper variants; no post-hoc budget or criterion changes. | [S121](worklogs/2026-09-04_S121.md) |

## Paper Reading Status

### Original Collection (Papers 01-20)
| # | File | Status | Summary Location |
|---|------|--------|-----------------|
| 01 | 01_RL_in_PE_safety_stability_robustness_2025.pdf | DONE | paper_summaries_01_to_10.md |
| 02 | 02_PE_car_like_robots_sensor_constraints_2024.pdf | DONE | paper_summaries_01_to_10.md |
| 03 | 03_CBF_in_RL_survey_2024.pdf | DONE | paper_summaries_01_to_10.md |
| 04 | 04_self_play_survey_2024.pdf | DONE | paper_summaries_01_to_10.md |
| 05 | 05_CBF_RL_safety_filtering_2025.pdf | DONE | paper_summaries_01_to_10.md |
| 06 | 06_safe_RL_robust_CBF_2021.pdf | DONE | paper_summaries_01_to_10.md |
| 07 | 07_sim_to_real_isaac_sim_gazebo_ros2_2025.pdf | DONE | paper_summaries_01_to_10.md |
| 08 | 08_vision_based_PE_robot_policies_ICRA2024.pdf | DONE | paper_summaries_01_to_10.md |
| 09 | 09_multi_UAV_PE_online_planning_DRL_2024.pdf | DONE | paper_summaries_01_to_10.md |
| 10 | 10_safe_RL_survey_lyapunov_barrier_2025.pdf | DONE | paper_summaries_01_to_10.md |
| 11 | 11_game_theory_MARL_nash_2024.pdf | DONE | paper_summaries_11_to_14.md |
| 12 | 12_PE_differential_game_DRL_frontiers_2022.pdf | DONE (re-downloaded) | paper_summaries_11_to_14.md |
| 13 | 13_safe_finite_time_RL_PE_CDC2022_preprint.pdf | DONE | paper_summaries_11_to_14.md |
| 14 | 14_safety_aware_PE_GP_TNNLS2022.pdf | DONE | paper_summaries_11_to_14.md |
| 15 | 15_mobile_robot_PE_differential_game_DRL_2025.pdf | DONE | paper_summaries_15_to_20.md |
| 16 | 16_sampling_based_safe_RL_CBF_PPO_AISTATS2024.pdf | DONE (re-downloaded) | paper_summaries_15_to_20.md |
| 17 | 17_ViPER_visibility_PE_CoRL2024.pdf | DONE (re-downloaded) | paper_summaries_15_to_20.md |
| 18 | 18_AMS_DRL_multi_pursuit_evasion_drones.pdf | DONE | paper_summaries_15_to_20.md |
| 19 | 19_selfplay_PE_aircraft_games_2024.pdf | DONE | paper_summaries_15_to_20.md |
| 20 | 20_emergent_behaviors_PE_2025.pdf | DONE | paper_summaries_15_to_20.md |

### Newly Downloaded Papers (Papers 21-37)
| # | File | Status | Summary Location |
|---|------|--------|-----------------|
| 21 | 21_DeepReach_HJI_neural_ICRA2021.pdf | DONE | paper_summaries_21_to_35.md |
| 22 | 22_MADR_adversarial_DeepReach_2025.pdf | DONE | paper_summaries_21_to_35.md |
| 23 | 23_convergence_guarantees_DeepReach_2024.pdf | DONE | paper_summaries_21_to_35.md |
| 24 | 24_enhancing_DeepReach_activations_2023.pdf | DONE | paper_summaries_21_to_35.md |
| 25 | 25_HJ_reachability_RL_survey_2024.pdf | DONE | paper_summaries_21_to_35.md |
| 26 | 26_neural_networks_PE_tactics_DGA_2024.pdf | DONE | paper_summaries_21_to_35.md |
| 27 | 27_RESPO_safe_RL_reachability_NeurIPS2023.pdf | DONE | paper_summaries_21_to_35.md |
| 28 | 28_LBAC_safe_RL_CLBF_ICRA2023.pdf | DONE | paper_summaries_21_to_35.md |
| 30 | 30_CASRL_safe_RL_navigation_CAAI2023.pdf | DONE | paper_summaries_21_to_35.md |
| 31 | 31_safe_MARL_HJ_reachability_JIRS_2024.pdf | DONE | paper_summaries_21_to_35.md |
| 32 | 32_diffusion_RL_adversarial_PE_2024.pdf | DONE | paper_summaries_21_to_35.md |
| 33 | 33_hot_starts_PE_GNN_2025.pdf | DONE | paper_summaries_21_to_35.md |
| 34 | 34_SHADOW_info_tradeoffs_PE_2025.pdf | DONE | paper_summaries_21_to_35.md |
| 35 | 35_NeHMO_neural_HJR_multiagent_2025.pdf | DONE | paper_summaries_21_to_35.md |
| 36 | 36_bio_inspired_NN_evasion_biomimetics_2024.pdf | DONE | paper_summaries_21_to_35.md |
| 37 | 37_RL_PE_games_review_CJA_2025.pdf | DONE | paper_summaries_21_to_35.md |

### Papers Not Downloaded (Paywall/Access Issues)
| # | Title | Venue | Year | Access Issue |
|---|-------|-------|------|-------------|
| 29 | Safe Robust MARL with Neural CBFs + Safety Attention | Information Sciences | 2024 | Elsevier paywall (403) |
| 38 | Transfer RL for Multi-agent PE with Obstacles | Asian J. Control | 2024 | Wiley Cloudflare block |

## Document Inventory
| File | Description | Status |
|------|-------------|--------|
| claudedocs/research_1v1_pursuit_evasion_deep_rl_ground_robots.md | Main research report (web-based) | COMPLETE |
| claudedocs/paper_summaries_01_to_10.md | Summaries of papers 1-10 | COMPLETE |
| claudedocs/paper_summaries_11_to_14.md | Summaries of papers 11-14 | COMPLETE |
| claudedocs/paper_summaries_15_to_20.md | Summaries of papers 15-20 | COMPLETE |
| claudedocs/paper_summaries_21_to_35.md | Summaries of papers 21-37 | COMPLETE |
| claudedocs/final_literature_review.md | Comprehensive final review | COMPLETE |
| docs/workflow_tracker.md | This file | ACTIVE |
| docs/worklogs/ | Per-session detailed worklogs | ACTIVE |
| docs/research_self_play_collapse_prevention.md | Self-play collapse analysis: 6 approaches, 20 papers, ranked action plan | COMPLETE |
| docs/run_p_analysis.md | Run P training dynamics: phase-by-phase, 3 root causes, PBRS effectiveness | COMPLETE |
| papers/README.md | Paper collection index | COMPLETE |

## Research Pathways (from initial report, to be refined in final review)
1. **Pathway A**: Safe Deep RL for 1v1 Ground Robot PE with Nonholonomic Dynamics (HIGHEST PRIORITY)
2. **Pathway B**: Vision-Based 1v1 PE for Wheeled Ground Robots
3. **Pathway C**: Model-Based Deep RL with Learned World Models for PE
4. **Pathway D**: Hierarchical RL for PE in Structured Indoor Environments
5. **Pathway E**: PE with Asymmetric Information and Opponent Modeling
6. **Pathway F** (NEW): HJ Reachability + Deep RL Hybrid for PE (from new papers)

