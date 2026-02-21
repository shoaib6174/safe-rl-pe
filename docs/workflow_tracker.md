# Pursuit-Evasion Research Workflow Tracker
## Last Updated: 2026-02-22

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
| papers/README.md | Paper collection index | COMPLETE |

## Research Pathways (from initial report, to be refined in final review)
1. **Pathway A**: Safe Deep RL for 1v1 Ground Robot PE with Nonholonomic Dynamics (HIGHEST PRIORITY)
2. **Pathway B**: Vision-Based 1v1 PE for Wheeled Ground Robots
3. **Pathway C**: Model-Based Deep RL with Learned World Models for PE
4. **Pathway D**: Hierarchical RL for PE in Structured Indoor Environments
5. **Pathway E**: PE with Asymmetric Information and Opponent Modeling
6. **Pathway F** (NEW): HJ Reachability + Deep RL Hybrid for PE (from new papers)
