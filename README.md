# Safe Deep RL for 1v1 Pursuit-Evasion

Research project on **safe deep reinforcement learning** for **1v1 ground robot pursuit-evasion games** with nonholonomic dynamics and control barrier function (CBF) safety guarantees.

## Repository Structure

```
safe-rl-pe/
├── docs/
│   ├── literature-review/       # Paper summaries and comprehensive review
│   ├── pathway/                 # Pathway A: research plan and validation
│   ├── phases/                  # Phase 1-5 implementation specifications
│   ├── references/              # BibTeX, method indices, paper lists
│   └── workflow_tracker.md      # Session log and progress tracking
├── papers/
│   ├── original/                # Papers 01-37 (PDFs gitignored)
│   └── supplementary/           # Papers N01-N15 (PDFs gitignored)
```

## Documentation Index

### Literature Review
| Document | Description |
|----------|-------------|
| [Final Literature Review](docs/literature-review/final_literature_review.md) | Comprehensive review of 36 papers across safe RL, CBFs, pursuit-evasion, self-play, and sim-to-real |
| [Initial Research Report](docs/literature-review/research_1v1_pursuit_evasion_deep_rl_ground_robots.md) | Web-based research report on 1v1 PE with deep RL for ground robots |
| [Paper Summaries 01-10](docs/literature-review/paper_summaries_01_to_10.md) | Core PE, CBF, self-play, and sim-to-real papers |
| [Paper Summaries 11-14](docs/literature-review/paper_summaries_11_to_14.md) | Game theory, differential games, safe finite-time RL |
| [Paper Summaries 15-20](docs/literature-review/paper_summaries_15_to_20.md) | Mobile robot PE, sampling-based safe RL, ViPER, emergent behaviors |
| [Paper Summaries 21-35](docs/literature-review/paper_summaries_21_to_35.md) | DeepReach, HJ reachability, RESPO, LBAC, diffusion RL, and more |

### Research Pathway
| Document | Description |
|----------|-------------|
| [Pathway A: Safe Deep RL for 1v1 PE](docs/pathway/pathway_A_safe_deep_RL_1v1_PE.md) | Full research plan: VCP-CBF safety + PPO + self-play for nonholonomic robots |
| [Pathway A Validation Report](docs/pathway/pathway_A_validation_report.md) | Expert validation of Pathway A against 51 papers |

### Implementation Phases
| Phase | Document | Description |
|-------|----------|-------------|
| 1 | [Simulation Foundation](docs/phases/phase1_simulation_foundation.md) | Gymnasium env, VCP-CBF, PPO baseline, pygame viz, wandb tracking |
| 2 | [Safety Integration](docs/phases/phase2_safety_integration.md) | CBF-PPO training, safety filtering, Lagrangian baselines |
| 2.5 | [BarrierNet Experiment](docs/phases/phase2_5_barriernet_experiment.md) | End-to-end differentiable safety via BarrierNet comparison |
| 3 | [Partial Observability + Self-Play](docs/phases/phase3_partial_observability_selfplay.md) | FOV/LiDAR observations, AMS-DRL self-play, health monitoring |
| 4 | [Sim-to-Real Transfer](docs/phases/phase4_sim_to_real_transfer.md) | Domain randomization, ONNX export, ROS 2 deployment |
| 5 | [Analysis & Publication](docs/phases/phase5_analysis_publication.md) | Statistical analysis, figures, paper writing |

### References
| Document | Description |
|----------|-------------|
| [BibTeX References](docs/references/bibtex_references.bib) | Citation database for all papers |
| [Key Methods Summary](docs/references/key_methods_summary.md) | Quick reference for CBF, BarrierNet, RESPO, POLICEd RL, etc. |
| [Paper Collection Index](papers/original/README.md) | Full index of 37 original papers by category |
| [Supplementary Papers](docs/references/safe_rl_papers_readme.md) | Index of 15 supplementary safe RL papers |
| [New Papers List](docs/references/new_papers.md) | Papers identified during extended search |

### Project Tracking
| Document | Description |
|----------|-------------|
| [Workflow Tracker](docs/workflow_tracker.md) | Session log, paper reading status, research pathways |

## Research Summary

**Goal**: Train a pursuit agent and an evasion agent for 1v1 ground robot PE using deep RL, with formal safety guarantees via control barrier functions.

**Key Approach (Pathway A)**:
- **Dynamics**: Unicycle/differential-drive with velocity-constrained polar (VCP) coordinate CBF
- **Algorithm**: PPO (Stable-Baselines3) with CBF safety filter
- **Training**: Asymmetric self-play (AMS-DRL) with health monitoring
- **Safety**: CBF-QP layer ensuring collision avoidance + boundary constraints
- **Transfer**: Domain randomization + ONNX export for ROS 2 deployment

**Papers Read**: 36 (+ 15 supplementary) across safe RL, CBFs, HJ reachability, pursuit-evasion, self-play, and sim-to-real transfer.
