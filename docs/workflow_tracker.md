# Pursuit-Evasion Research Workflow Tracker
## Last Updated: 2026-02-28 (S58)

## Research Topic
**1v1 Pursuit-Evasion Games using Mobile Ground Robots with Deep RL**
- One pursuer, one evader
- Focus on deep reinforcement learning approaches
- Ground mobile robots (differential-drive / car-like)

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
| 59 | 2026-03-01 | **Variable Speed Support**: Removed FixedSpeedWrapper default — agents now control 2D [v, omega] actions. Updated diagnostic scripts (train_evader_vs_greedy, train_pursuer_vs_evader) and visualization. Auto-detect action dim in PPOOpponentPolicy. 120 tests pass. | [S59](worklogs/2026-03-01_S59.md) |

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
