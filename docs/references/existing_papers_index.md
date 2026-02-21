# Existing Papers Index (from Literature Review)

## 36 Papers Reviewed — Organized by Relevance to Safe DRL for 1v1 PE

### Tier 1: Directly Applicable (MUST READ/IMPLEMENT)

| ID | Title (Short) | Authors | Year | Venue | Key Contribution |
|----|--------------|---------|------|-------|-----------------|
| [01] | RL+SMC+CBF for PE | Wang et al. | 2025 | arXiv | Three-term controller: u_RL + u_SMC + u_CBF; Stackelberg hierarchy |
| [02] | Car-like PE with sensors | Gonultas & Isler | 2024 | - | SOTA 1v1 ground robot PE; BiMDN belief; F1TENTH deployment |
| [05] | CBF-RL filtering | Yang et al. (Caltech) | 2025 | - | Closed-form CBF filter + reward shaping; dual approach |
| [06] | Robust CBF + GP | Emam et al. | 2022 | - | RCBF-QP + GP disturbance estimation for model uncertainty |
| [08] | Vision-based PE | Bajcsy et al. | 2024 | ICRA | Privileged teacher-student; DAGGER; Unitree A1 deployment |
| [13] | Safe finite-time PE | Kokolakis & Vamvoudakis | 2022 | CDC | Barrier in HJI cost; finite-time convergence + safety |
| [16] | CBF-Beta PPO | Suttle et al. | 2024 | AISTATS | CBF-constrained Beta policy; convergence + hard safety |
| [18] | AMS-DRL self-play | Xiao & Feroskhan | 2024 | - | Staged alternating training; proven NE; real drones |
| [22] | MADR | Teoh et al. | 2025 | - | DeepReach + adversarial MPC; TurtleBot deployment |
| [25] | HJ-RL Survey | Ganai et al. | 2024 | - | Complete roadmap for HJ + RL integration; shielding framework |
| [34] | SHADOW | La Gatta et al. | 2025 | - | 1v1 PE unicycle; multi-headed TD3+PPO; info trade-offs |

### Tier 2: Important Supporting Methods

| ID | Title (Short) | Authors | Year | Venue | Key Contribution |
|----|--------------|---------|------|-------|-----------------|
| [03] | CBF-RL Survey | - | - | - | Taxonomy of CBF integration approaches |
| [04] | Self-Play Survey | - | - | - | Vanilla SP, FSP, PSRO comparison |
| [07] | Sim-to-Real pipeline | Salimpour et al. | 2025 | - | Isaac Sim → Gazebo → ROS2; ONNX; TurtleBot4 |
| [10] | Safe RL Survey | - | - | - | Comprehensive safety method reference |
| [14] | Kokolakis dissertation | Kokolakis | 2022 | PhD | Safety + GP + bounded rationality for PE |
| [21] | DeepReach | Bansal & Tomlin | - | - | Neural HJI solver; sine activations |
| [24] | DeepReach Activations | Wang & Wu | 2023 | - | Mixed Sine/ReLU for better BRT |
| [27] | RESPO | - | - | - | Reachability estimation for safe RL (stochastic) |
| [28] | LBAC | Du et al. | 2023 | ICRA | Unified CLF-CBF certificate; model-free; CrazyFlie |
| [30] | CASRL | Zhou et al. | 2023 | CAAI | Conflict-averse gradient for safe navigation |
| [31] | Safe HJ MADDPG | Zhu et al. | 2024 | JIRS | HJ + CQL for safe MARL; 0.1ms inference |
| [37] | RL-PE Review | Yang et al. | 2025 | CJA | 235-ref taxonomy of RL for PE games |

### Tier 3: Useful Background

| ID | Title (Short) | Authors | Year | Venue | Key Contribution |
|----|--------------|---------|------|-------|-----------------|
| [09] | Multi-UAV PE | - | - | - | Evader prediction + adaptive curriculum |
| [11] | Game Theory MARL | - | - | - | NE + opponent modeling theory |
| [12] | Dog-sheep PE | - | - | - | DQN vs DDPG baseline comparison |
| [15] | Mobile robot PE | - | - | - | A-MPC + DRL hybrid |
| [17] | ViPER | Wang et al. | 2024 | CoRL | Graph attention for PE |
| [19] | Self-play iteration | - | - | - | NE convergence proof for PE |
| [20] | Emergent behaviors | - | - | - | Behavior taxonomy |
| [23] | DeepReach convergence | - | - | - | Theory validation |
| [26] | HNSN | - | 2024 | - | Hybrid NN architecture (not RL) |
| [32] | Diffusion-RL PE | Wu et al. | 2024 | - | Hierarchical diffusion + SAC |
| [33] | Hot Starts PE | - | 2025 | - | GCN placement (multi-pursuer) |
| [35] | NeHMO | - | - | - | Scalable neural HJR |
| [36] | BINN Evasion | - | 2024 | Biomimetics | Bio-inspired reactive evasion; ROS |
