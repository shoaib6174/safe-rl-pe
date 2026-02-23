# Phase 5: Analysis & Publication (Months 8-10)

## Self-Contained Implementation Document for Claude Code AI Agent
**Project**: Safe Deep RL for 1v1 Ground Robot Pursuit-Evasion
**Phase**: 5 of 5
**Timeline**: Months 8-10
**Date Created**: 2026-02-21
**Status**: Ready for implementation

---

## Table of Contents

1. [Phase Overview](#1-phase-overview)
2. [Background & Theoretical Foundations](#2-background--theoretical-foundations)
3. [Relevant Literature](#3-relevant-literature)
4. [Session-wise Implementation Breakdown](#4-session-wise-implementation-breakdown)
5. [Testing Plan (Automated)](#5-testing-plan-automated)
6. [Manual Validation Checklist](#6-manual-validation-checklist)
7. [Success Criteria & Phase Gates](#7-success-criteria--phase-gates)
8. [Troubleshooting Guide](#8-troubleshooting-guide)
9. [Publication Roadmap](#9-publication-roadmap)

---

## 1. Phase Overview

### 1.1 Title
**Phase 5: Formal Analysis, Comprehensive Benchmarking, and Multi-Paper Publication**

### 1.2 Timeline
- **Start**: Month 8 (after Phase 4: Sim-to-Real Transfer is complete)
- **End**: Month 10
- **Duration**: 3 months (approximately 18 working sessions)

### 1.3 Objectives
1. **Formal Safety Analysis**: Prove CBF validity for the VCP formulation, prove constraint satisfaction under GP uncertainty (RCBF probabilistic guarantee), and establish Lyapunov-based stability
2. **Nash Equilibrium Analysis**: Compute exploitability metrics, verify NE convergence via best-response opponents and PSRO-style verification
3. **Comprehensive Benchmarking**: Run all 13 baselines (including HJ/DeepReach optimal) and all 11 ablation studies with statistical rigor (5-10 seeds, confidence intervals, significance tests)
4. **Publication Artifacts**: Generate all figures, tables, and demonstration videos for 3 papers
5. **Paper Writing**: Draft and submit conference paper (ICRA/IROS/CoRL), journal paper (RA-L/T-RO), and theory paper (CDC/L4DC)
6. **Open-Source Release**: Clean, documented, reproducible codebase with Docker support
7. **Generalization Study**: Zero-shot evaluation on unseen complex environments (corridor, L-shaped, warehouse)
8. **Interpretability Analysis**: Saliency maps, BiMDN belief evolution, policy phase portraits, emergent strategy classification
9. **Opponent Modeling**: Post-hoc trajectory analysis of opponent adaptation and strategy classification
10. **Human Evader Experiment**: Evaluate pursuer policy against human-controlled evader

### 1.4 Prerequisites (All Must Be Complete)
- **Phase 1 (Simulation Foundation)**: Gymnasium PE environment, PPO self-play, VCP-CBF validated
- **Phase 2 (Safety Integration)**: CBF-Beta policy, RCBF-QP filter, 3-tier infeasibility handling, zero safety violations in training
- **Phase 2.5 (BarrierNet Experiment)**: Train-deploy gap measured, safety architecture decision made
- **Phase 3 (Partial Observability + Self-Play)**: BiMDN belief encoder, AMS-DRL self-play, curriculum learning, NE convergence achieved, MACPO/CPO baselines complete
- **Phase 4 (Sim-to-Real Transfer)**: Isaac Lab environment, ONNX export, Gazebo validation, real robot deployment on TurtleBot3 Burger, GP disturbance estimation, real-world data collected

### 1.5 Required Inputs
| Input | Source | Format |
|-------|--------|--------|
| Trained SafePE policies (pursuer + evader) | Phase 3 | PyTorch .pt / ONNX .onnx |
| Training logs (all seeds) | Phases 1-3 | WandB / TensorBoard logs |
| Baseline policies (all 12) | Phases 1-3 | PyTorch .pt |
| Ablation experiment logs (all 11) | Phases 2-3 | WandB logs |
| Real-robot deployment data | Phase 4 | ROS2 bag files |
| Real-robot video recordings | Phase 4 | MP4/MKV files |
| GP disturbance model | Phase 4 | GPyTorch model |
| Environment configs (all scenarios) | Phases 1-4 | Hydra YAML |
| Simulation checkpoints | Phases 1-3 | .pt checkpoint files |

### 1.6 Relationship to Project
Phase 5 is the culmination of the entire research project. It transforms experimental results from Phases 1-4 into publishable research contributions. This phase produces no new algorithms or implementations; instead, it rigorously analyzes existing results, proves theoretical properties, and communicates findings to the research community.

### 1.7 Key Output Artifacts
| Artifact | Description | Target |
|----------|-------------|--------|
| Paper 1 (conference) | SafePE algorithm + simulation results | ICRA/IROS/CoRL |
| Paper 2 (journal) | Full system + real-robot results | RA-L/T-RO |
| Paper 3 (theory) | Formal safety + NE convergence proofs | CDC/L4DC |
| Demo videos | Real-robot PE demonstrations | YouTube / project page |
| Open-source repo | Code, docs, Docker, examples | GitHub |
| Supplementary material | Extended proofs, additional experiments | Paper appendices |

---

## 2. Background & Theoretical Foundations

### 2.1 Formal Safety Analysis

#### 2.1.1 CBF Validity for VCP Formulation

**Definition (Control Barrier Function)**: Given a dynamical system x_dot = f(x) + g(x)u, a continuously differentiable function h: R^n -> R is a CBF for the safe set S = {x in R^n : h(x) >= 0} if there exists a class-K_inf function alpha such that:

```
sup_{u in U} [L_f h(x) + L_g h(x) u + alpha(h(x))] >= 0,  for all x in S
```

where L_f h = (nabla h)^T f(x) and L_g h = (nabla h)^T g(x) are the Lie derivatives.

**Theorem 1 (VCP-CBF Validity for Unicycle)**: For the unicycle dynamics:
```
x_dot = v cos(theta)
y_dot = v sin(theta)
theta_dot = omega
```
with virtual control point q = [x + d cos(theta), y + d sin(theta)] (d > 0), and CBF h(q) defined over q, the constraint:
```
dh/dq * q_dot + alpha * h(q) >= 0
```
is affine in u = [v, omega] with:
```
q_dot = [v cos(theta) - d omega sin(theta),  v sin(theta) + d omega cos(theta)]
```

**Proof sketch**:
1. Compute q_dot by differentiating q with respect to time:
   - q_x_dot = x_dot - d theta_dot sin(theta) = v cos(theta) - d omega sin(theta)
   - q_y_dot = y_dot + d theta_dot cos(theta) = v sin(theta) + d omega cos(theta)
2. The CBF condition becomes dh/dq * q_dot = [dh/dq_x, dh/dq_y] * [v cos(theta) - d omega sin(theta), v sin(theta) + d omega cos(theta)]
3. This is linear in v and omega (both appear at relative degree 1), so the CBF-QP is well-posed
4. For h_arena(q) = R^2 - ||q||^2, we get: -2q^T q_dot + alpha(R^2 - ||q||^2) >= 0, which is affine in [v, omega]
5. The safe set S_q = {q : h(q) >= 0} maps to a safe set S_x in the original state space. Since ||p_c - p_obs|| >= ||q - p_obs|| - d (triangle inequality), safety of q guarantees safety of the robot center p_c if chi >= r_robot + r_obs + margin + d

**Source**: [N12] Zhang & Yang 2025, Lemma 1, adapted for unicycle (simplified from car-like).

#### 2.1.2 Forward Invariance Theorem

**Theorem 2 (Forward Invariance under VCP-CBF)**: If the CBF-QP:
```
u* = argmin_{u in U} ||u - u_nom||^2
s.t. dh_i/dq * q_dot(u) + alpha_i * h_i(q) >= 0, for all i
```
is feasible at every state x(t) along the trajectory, then the safe set S = intersection_i {x : h_i(q(x)) >= 0} is forward invariant under the closed-loop dynamics.

**Proof**: By Nagumo's theorem, forward invariance of S requires that at every boundary point (h_i = 0), the vector field points inward (h_i_dot >= 0). The CBF constraint h_i_dot + alpha_i * h_i >= 0 ensures h_i_dot >= -alpha_i * h_i. At the boundary h_i = 0, this gives h_i_dot >= 0. By comparison principle, h_i(t) >= h_i(0) * exp(-alpha_i * t) >= 0 for all t >= 0 if h_i(0) >= 0.

**Source**: Standard CBF theory [03], [10], [N01], adapted for VCP formulation [N12].

#### 2.1.3 Constraint Satisfaction Under GP Uncertainty (RCBF)

**Theorem 3 (Probabilistic Safety Guarantee under RCBF-QP)**: Given the uncertain dynamics:
```
x_dot = f(x) + g(x) u + d(x)
```
where d(x) is estimated by a GP with posterior mean mu_d(x) and variance sigma_d^2(x), the RCBF-QP:
```
u* = argmin_{u in U} ||u - u_rl||^2
s.t. L_f h + L_g h u + (nabla h)^T mu_d + alpha h >= kappa * ||(nabla h)^T|| * sigma_d
```
guarantees P(h(x(t)) >= 0 for all t) >= 1 - delta, where:
- kappa = Phi^{-1}(1 - delta) (inverse normal CDF)
- delta is the desired failure probability (e.g., delta = 0.05 for 95% confidence)

**Proof sketch**:
1. The true disturbance d(x) is a sample from the GP posterior: d(x) ~ N(mu_d(x), sigma_d^2(x))
2. The robust margin kappa * sigma_d ensures that the CBF condition holds with probability >= 1 - delta for each state
3. By union bound over discretized time steps (or continuous-time argument via Borel-Cantelli), the trajectory-level safety probability is bounded
4. The GP posterior contracts as more data is collected, so kappa * sigma_d -> 0, recovering the nominal CBF guarantee

**Source**: [06] Emam et al. 2022, Theorem 2, adapted for VCP formulation.

**GP Cold-Start Protocol effect on guarantees**:
- Phase 1 (pre-filled GP from simulation): sigma_d is moderate, kappa_init = 2 * kappa_nominal provides extra conservatism
- Phase 2 (first 100 real steps): sigma_d decreases as real data is incorporated
- Phase 3 (normal operation): sigma_d is small, kappa = kappa_nominal, near-nominal performance

#### 2.1.4 Lyapunov-Based Stability Analysis

**Theorem 4 (Convergence to Capture under Safe Control)**: For the pursuer policy pi_P with CBF safety filter, define the Lyapunov function V(x) = ||p_P - p_E||^2 (squared distance between agents). If the pursuer's expected reward is positive (E[r_P] > 0) and the CBF filter intervention rate is bounded (intervention_rate < rho_max), then the expected time to capture is bounded:
```
E[T_capture] <= V(x_0) / (eta - rho_max * eta_cbf)
```
where eta is the nominal approach rate and eta_cbf is the maximum approach rate reduction due to CBF intervention.

**Note**: This is a weaker result than classical Lyapunov stability because the adversarial evader actively increases V(x). The bound holds only in expectation over the stochastic policy, conditioned on the NE strategy profile.

### 2.2 Nash Equilibrium Analysis

#### 2.2.1 Exploitability Computation

**Definition (Exploitability)**: For a strategy profile (pi_P, pi_E), the exploitability is:
```
exploit(pi_P, pi_E) = max_{pi_P'} J_P(pi_P', pi_E) - J_P(pi_P, pi_E)
                     + max_{pi_E'} J_E(pi_P, pi_E') - J_E(pi_P, pi_E)
```
where J_P, J_E are the expected returns for pursuer and evader respectively.

At a Nash equilibrium, exploit(pi_P*, pi_E*) = 0. In practice, we target exploit < 0.10.

**Computation method**:
1. Fix pi_E, train a best-response pi_P' for N_br episodes (e.g., 500K timesteps)
2. Fix pi_P, train a best-response pi_E' for N_br episodes
3. Compute exploit = [J_P(pi_P', pi_E) - J_P(pi_P, pi_E)] + [J_E(pi_P, pi_E') - J_E(pi_P, pi_E)]
4. Repeat with multiple random seeds for the best-response training

**Source**: [04] Self-play survey, [18] AMS-DRL convergence criterion.

#### 2.2.2 Best-Response Calculation

For each agent, compute the best response by:
1. Freeze the opponent's policy
2. Train a fresh policy from scratch against the frozen opponent using PPO with the same hyperparameters as the original training
3. Use 5 random seeds and take the best-performing seed (to avoid local optima)
4. Train for the same compute budget as one phase of AMS-DRL (1-2M timesteps)

The best-response value provides an upper bound on what any policy could achieve against the frozen opponent.

#### 2.2.3 PSRO-Style Verification

**Policy Space Response Oracle (PSRO)** verification procedure:
1. Collect the population of policies from all self-play phases: {pi_P^1, pi_P^2, ..., pi_P^K, pi_E^1, ..., pi_E^K}
2. Compute the payoff matrix M where M[i,j] = E[r_P | pi_P^i vs pi_E^j]
3. Find the Nash equilibrium of the matrix game (linear program)
4. The NE mixture weights indicate which policies are essential
5. If the final policy pair (pi_P^K, pi_E^K) receives high NE weight, self-play has converged
6. If weight is spread across many policies, the game has rich strategic structure requiring mixed strategies

**Source**: [04] Lanctot et al. 2017 (PSRO), adapted for PE verification.

### 2.3 Statistical Analysis for RL Experiments

#### 2.3.1 Multiple Seed Protocol

All experiments must be run with **at minimum 5 seeds** (target 10 seeds for key results):
- Seeds: [0, 1, 2, 3, 4] (minimum), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (full)
- Each seed controls: environment randomization, network initialization, action sampling
- Report: mean +/- standard error, or median with IQR

#### 2.3.2 Confidence Intervals

Use **bootstrapped confidence intervals** (non-parametric):
```python
def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lower, upper
```

Report 95% CIs for all primary metrics.

#### 2.3.3 Significance Testing

- **Paired comparisons** (SafePE vs each baseline): Welch's t-test or Mann-Whitney U test (if non-normal)
- **Multiple comparisons correction**: Holm-Bonferroni (13 baselines = 13 comparisons, including HJ/DeepReach optimal)
- **Effect size**: Cohen's d for continuous metrics, odds ratio for binary metrics (safety violations)
- **Significance threshold**: p < 0.05 after correction
- **Power analysis**: Verify n >= 5 seeds provides power > 0.80 for expected effect sizes

#### 2.3.4 Training Curve Analysis

- Use **exponential moving average** (EMA) smoothing with alpha = 0.99 for visualization
- Report **area under the training curve** (AUC) for learning efficiency comparisons
- Show **individual seed traces** (thin lines) behind the mean (thick line) in training curve plots

### 2.4 Benchmarking Methodology

#### 2.4.1 Fair Comparison Principles

1. **Same compute budget**: All baselines trained for the same number of environment steps (total_timesteps)
2. **Hyperparameter tuning**: Each baseline gets a small grid search (learning rate, network size) with the same budget
3. **Same evaluation protocol**: 100 episodes x 5 seeds for all methods, same initial conditions
4. **Same environment version**: All baselines evaluated on identical environment configurations
5. **Code verification**: Use official implementations where available; validate reproduction on simple benchmarks first

#### 2.4.2 Compute Normalization

Report all results with:
- Wall-clock training time (hours on specified hardware)
- Total environment steps
- Total gradient updates
- FLOPs estimate (if possible)
- GPU memory usage

#### 2.4.3 Hyperparameter Sensitivity

For the SafePE algorithm, report sensitivity to:
- CBF class-K function alpha (0.5, 1.0, 2.0)
- Safety margin (0.05m, 0.10m, 0.20m)
- Self-play convergence threshold eta (0.05, 0.10, 0.15)
- Reward weight w5 (0.0, 0.025, 0.05, 0.10)
- VCP offset d (0.02m, 0.05m, 0.10m)

### 2.5 Publication Standards

#### 2.5.1 ICRA / IROS (IEEE Conference)

- **Page limit**: 6 pages + references (ICRA), 6 pages + references (IROS)
- **Format**: IEEE two-column
- **Expected content**: Problem formulation, method, simulation results, (optional) real-robot demo
- **Figures**: 5-7 figures typical, must be readable at column width (3.5 inches)
- **Review criteria**: Technical correctness, novelty, clarity, experimental validation
- **Supplementary**: Video strongly encouraged (1-3 minutes)
- **LaTeX template**: `IEEEtran.cls`

#### 2.5.2 CoRL (Conference on Robot Learning)

- **Page limit**: 8 pages + references (main paper), unlimited appendix
- **Format**: CoRL style (modified NeurIPS)
- **Expected content**: Method + experiments with strong learning component
- **Figures**: Higher quality expected than IEEE; include training curves
- **Review criteria**: Learning methodology, reproducibility, real-world relevance
- **Supplementary**: Appendix with proofs, additional experiments; video encouraged

#### 2.5.3 RA-L / T-RO (IEEE Journal)

- **Page limit**: 8 pages (RA-L), 12-20 pages (T-RO)
- **Format**: IEEE two-column
- **Expected content**: Complete system description, extensive experiments, real-robot validation
- **Figures**: 8-15 figures, high quality
- **Review criteria**: Completeness, reproducibility, real-world impact, thoroughness
- **Supplementary**: Multimedia attachment (video), code repository
- **RA-L option**: Can be presented at ICRA/IROS if accepted

#### 2.5.4 CDC / L4DC (Control / Learning Conference)

- **Page limit**: 6 pages + references (CDC), 8 pages (L4DC)
- **Format**: IEEE (CDC), PMLR (L4DC)
- **Expected content**: Theoretical contribution with formal proofs, simulation validation
- **Figures**: 3-5 figures (theory-focused)
- **Review criteria**: Mathematical rigor, novelty of theoretical results

### 2.6 Open-Source Release Best Practices

#### 2.6.1 Reproducibility Requirements

- **Fixed random seeds**: All experiments reproducible with specified seeds
- **Version pinning**: requirements.txt with exact versions (pip freeze)
- **Docker image**: Complete environment in a container
- **Configuration files**: All experiment configs in Hydra YAML
- **Checkpoints**: Pre-trained model checkpoints downloadable

#### 2.6.2 Documentation Standards

- **README.md**: Installation, quick start, project overview
- **CONTRIBUTING.md**: How to contribute, code style
- **LICENSE**: MIT or Apache 2.0 (permissive for academic use)
- **API documentation**: Docstrings for all public functions
- **Tutorials**: Jupyter notebooks for key workflows
- **Paper link**: Reference to published paper(s)

#### 2.6.3 Repository Structure

```
safe-pe/
  README.md
  LICENSE
  setup.py / pyproject.toml
  requirements.txt
  Dockerfile
  configs/               # Hydra YAML configs
    env/
    training/
    evaluation/
    baselines/
  safe_pe/               # Main package
    envs/                # Gymnasium environments
    agents/              # PPO, baselines
    safety/              # CBF, RCBF-QP, VCP
    belief/              # BiMDN encoder
    self_play/           # AMS-DRL controller
    evaluation/          # Metrics, exploitability
    utils/
  scripts/
    train.py
    evaluate.py
    run_baselines.py
    run_ablations.py
    generate_figures.py
    generate_tables.py
  experiments/           # Experiment logs and results
  figures/               # Generated publication figures
  tests/                 # Unit and integration tests
  notebooks/             # Tutorial notebooks
  ros2_ws/               # ROS2 workspace for deployment
  docker/
    Dockerfile.train
    Dockerfile.deploy
```

---

## 3. Relevant Literature

### 3.1 Papers for Positioning and Framing (Introduction/Related Work)

| Ref | Paper | Use in Paper |
|-----|-------|-------------|
| [37] | Yang 2025 — RL-PE comprehensive survey (235 refs) | Primary positioning reference: taxonomy of PE methods, identify our gap |
| [25] | Ganai 2024 — HJ-RL survey | Safety framing: connect CBF approach to broader safe RL landscape |
| [N09] | Liu 2024 — Safe RL/CMDP survey | CMDP framing: position among constrained RL methods |
| [N01] | Dawson 2023 — Learning CBFs survey | CBF background: taxonomy of learned vs hand-crafted CBFs |
| [03] | Ames 2024 — CBF-RL survey | CBF-RL integration taxonomy |
| [04] | Czarnecki 2020 — Self-play survey | Self-play background and convergence theory |
| [10] | Brunke 2025 — Safe RL survey | Broad safe RL landscape |

### 3.2 Papers for Technical Foundations (Method Section)

| Ref | Paper | Use in Paper |
|-----|-------|-------------|
| [16] | Suttle 2024 — CBF-Beta PPO (AISTATS) | Core method: CBF-constrained Beta policy, convergence theorem |
| [N12] | Zhang & Yang 2025 — VCP-CBF | Critical: VCP formulation for nonholonomic robots |
| [06] | Emam 2022 — Robust CBF + GP | Deployment safety: RCBF-QP with GP uncertainty |
| [18] | Xiao 2024 — AMS-DRL | Self-play protocol: alternating training with NE convergence |
| [02] | Gonultas 2024 — PE with sensors | BiMDN belief encoder, curriculum learning |
| [05] | Yang 2025 — CBF-RL filtering | Safety reward shaping (w5 term), closed-form CBF filter |
| [N13] | Xiao 2023 — Learned feasibility | CBF-QP infeasibility handling via learned constraints |
| [N04] | Xiao 2023 — BarrierNet | Differentiable QP safety layer (if used) |
| [30] | Zhou 2023 — CASRL | Conflict-averse gradient (if used) |

### 3.3 Papers for Baseline Comparisons

| Ref | Paper | Baseline |
|-----|-------|----------|
| [12] | Li 2022 — Dog-sheep PE | DQN, DDPG baselines |
| [02] | Gonultas 2024 | MADDPG baseline |
| [18] | Xiao 2024 (adapted) | PPO + self-play (no CBF) |
| [05] | Yang 2025 | PPO + CBF (no self-play) |
| [15] | Zhang 2025 — Mobile robot PE | A-MPC + DQN |
| [22] | Teoh 2025 — MADR | HJ-optimal upper bound |
| [N03] | Gu 2023 — MACPO | MACPO, MAPPO-Lagrangian |
| [N14] | Gu 2026 — SB-TRPO | SB-TRPO baseline |
| Achiam 2017 | CPO | CPO baseline |

### 3.4 Papers for Ablation Study Context

| Ref | Paper | Ablation |
|-----|-------|----------|
| [N06] | Selvam 2024 — TD3 self-play PE | Ablation 2 (self-play variants), Ablation 8 (AMS vs simultaneous) |
| [N04] | Xiao 2023 — BarrierNet | Ablation 10 (CBF-Beta vs BarrierNet) |
| [30] | Zhou 2023 — CASRL | Ablation 11 (conflict-averse gradient) |
| [N15] | Liu 2025 — RMARL-CBF-SAM | Context for Ablation 9 (safety reward shaping) |

### 3.5 Papers for Sim-to-Real Discussion

| Ref | Paper | Use |
|-----|-------|-----|
| [N10] | Salimpour 2025 — Isaac sim-to-real | Validate pipeline: Isaac -> ONNX -> Gazebo -> Real |
| [N11] | Mittal 2025 — Isaac Lab | GPU-accelerated training framework |
| [07] | Salimpour 2025 — Sim-to-real | Domain randomization methodology |

### 3.6 Papers for Theory Paper (Paper 3)

| Ref | Paper | Use |
|-----|-------|-----|
| [16] | Suttle 2024 — CBF-Beta PPO | Convergence theorem for constrained policies |
| [N12] | Zhang & Yang 2025 — VCP-CBF | VCP forward invariance (Lemma 1, Theorem 1) |
| [06] | Emam 2022 — RCBF + GP | Probabilistic safety guarantee |
| [N02] | So 2024 — PNCBF | Value-function-is-CBF insight (if extending theory) |
| [13] | Kokolakis 2022 — Safe PE CDC | Barrier-augmented HJI for safe PE |
| [19] | Rasmussen 2024 — Self-play iteration | NE convergence proof for PE |
| [11] | Dafoe 2020 — Game theory MARL | Nash equilibrium theory |

### 3.7 Complete Paper Index (All 51 Papers)

**Original 36 papers [01]-[37]**:
[01] Wang 2025 (RL+SMC+CBF), [02] Gonultas 2024 (car-like PE), [03] Ames 2024 (CBF-RL survey), [04] Czarnecki 2020 (self-play survey), [05] Yang 2025 (CBF-RL filtering), [06] Emam 2022 (RCBF+GP), [07] Salimpour 2025 (sim-to-real), [08] Bajcsy 2024 (vision PE), [09] Multi-UAV PE, [10] Brunke 2025 (safe RL survey), [11] Dafoe 2020 (game theory), [12] Li 2022 (dog-sheep PE), [13] Kokolakis 2022 (safe PE CDC), [14] Kokolakis dissertation, [15] Zhang 2025 (mobile robot PE), [16] Suttle 2024 (CBF-Beta PPO), [17] Wang 2024 (ViPER), [18] Xiao 2024 (AMS-DRL), [19] Rasmussen 2024 (self-play iteration), [20] Emergent behaviors, [21] Bansal 2021 (DeepReach), [22] Teoh 2025 (MADR), [23] DeepReach convergence, [24] Wang 2023 (DeepReach activations), [25] Ganai 2024 (HJ-RL survey), [26] HNSN, [27] RESPO, [28] LBAC, [30] Zhou 2023 (CASRL), [31] Zhu 2024 (safe HJ MADDPG), [32] Wu 2024 (diffusion-RL PE), [33] Hot starts PE, [34] La Gatta 2025 (SHADOW), [35] NeHMO, [36] BINN evasion, [37] Yang 2025 (RL-PE survey)

**New 15 papers [N01]-[N15]**:
[N01] Dawson 2023 (learning CBFs survey), [N02] So 2024 (PNCBF), [N03] Gu 2023 (MACPO), [N04] Xiao 2023 (BarrierNet), [N05] Zhang 2024 (GCBF+), [N06] Selvam 2024 (TD3 self-play PE), [N07] Bouvier 2024 (POLICEd RL), [N08] Jin 2024 (statewise CBF projection), [N09] Liu 2024 (safe RL survey), [N10] Salimpour 2025 (Isaac sim-to-real), [N11] Mittal 2025 (Isaac Lab), [N12] Zhang & Yang 2025 (VCP-CBF), [N13] Xiao 2023 (learned feasibility), [N14] Gu 2026 (SB-TRPO), [N15] Liu 2025 (RMARL-CBF-SAM)

---

## 4. Session-wise Implementation Breakdown

### Session 1: Formal Safety Analysis — VCP-CBF Validity Proof

**Objectives**:
- Write formal theorem statements and proof sketches for VCP-CBF validity
- Prove that VCP achieves uniform relative degree 1 for unicycle dynamics
- Prove forward invariance of the safe set under VCP-CBF-QP control

**Files to create/modify**:
- `paper3_theory/proofs/vcp_cbf_validity.tex` — Main proof document
- `paper3_theory/proofs/forward_invariance.tex` — Forward invariance theorem
- `scripts/verify_cbf_properties.py` — Numerical verification of proof conditions

**Instructions**:

Step 1: State the unicycle dynamics formally in the notation of [N12]:
```latex
\dot{x} = f(x) + g(x)u, \quad
f(x) = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}, \quad
g(x) = \begin{bmatrix} \cos\theta & 0 \\ \sin\theta & 0 \\ 0 & 1 \end{bmatrix}, \quad
u = \begin{bmatrix} v \\ \omega \end{bmatrix}
```

Step 2: Define the VCP and compute its time derivative:
```latex
q = \begin{bmatrix} x + d\cos\theta \\ y + d\sin\theta \end{bmatrix}, \quad
\dot{q} = \begin{bmatrix} v\cos\theta - d\omega\sin\theta \\ v\sin\theta + d\omega\cos\theta \end{bmatrix}
```

Step 3: Prove Lemma (Uniform Relative Degree): Show that for any CBF h(q), the Lie derivative L_g(L_f^0 h) is non-degenerate in both v and omega. Specifically:
```
dh/dq * dq/du = [dh/dq_x, dh/dq_y] * [[cos(theta), -d sin(theta)], [sin(theta), d cos(theta)]]
```
This 1x2 matrix is nonzero for any nonzero gradient dh/dq (since the 2x2 matrix has determinant d, which is nonzero).

Step 4: Prove Theorem 1 (VCP-CBF Validity) using Lemma + standard CBF theory.

Step 5: Prove Theorem 2 (Forward Invariance) using Nagumo's theorem.

Step 6: Write numerical verification script that:
- Samples 10,000 random states in the arena
- For each state, verifies the CBF-QP is feasible
- Checks that the Lie derivative condition is satisfied
- Reports feasibility rate (should be >99.9%)

**Note — DCBF Safety Theorem (from Phase 3 Stage -1)**:
The discrete-time CBF (DCBF) safety theorem with gamma=0.2 was formally proved and numerically verified in Phase 3 (Stage -1) before training. That theorem proves zero physical collisions under 5 stated assumptions (A1-A5) via Nagumo's theorem + comparison principle. Session 1 here extends that work to the continuous-time VCP-CBF formulation and adds the RCBF probabilistic guarantee. Reference the Phase 3 proof (`docs/proofs/dcbf_safety_theorem.md`) and build upon it:
- Phase 3 DCBF theorem → discrete-time safety guarantee during training
- Phase 5 Theorem 1 → continuous-time VCP-CBF validity
- Phase 5 Theorem 2 → forward invariance of safe set
- Together: complete safety chain from training through deployment

**Verification**:
- [ ] All theorems stated with complete conditions
- [ ] Proof sketches are mathematically sound
- [ ] Numerical verification script runs and reports >99.9% feasibility
- [ ] Proofs reference [N12] and adapt correctly for unicycle (vs car-like)
- [ ] Cross-reference with Phase 3 DCBF theorem is explicit and consistent

---

### Session 2: Formal Safety Analysis — RCBF Probabilistic Guarantee

**Objectives**:
- Prove probabilistic safety guarantee under GP uncertainty
- Formalize the GP cold-start protocol's safety properties
- Derive bounds on the train-deploy safety gap

**Files to create/modify**:
- `paper3_theory/proofs/rcbf_probabilistic.tex` — RCBF probabilistic guarantee
- `paper3_theory/proofs/gp_cold_start.tex` — GP cold-start safety properties
- `paper3_theory/proofs/train_deploy_gap.tex` — CBF-Beta vs RCBF-QP gap bound
- `scripts/verify_rcbf_guarantee.py` — Monte Carlo verification

**Instructions**:

Step 1: Formalize the uncertain dynamics model:
```latex
\dot{x} = f(x) + g(x)u + d(x), \quad d(x) \sim \mathcal{GP}(\mu_d(x), k_d(x,x'))
```

Step 2: State and prove Theorem 3 (Probabilistic Safety):
- Start from the GP posterior: P(|d(x) - mu_d(x)| <= kappa * sigma_d(x)) >= 1 - delta
- Show that the RCBF constraint implies h_dot + alpha * h >= 0 with probability >= 1 - delta
- Apply forward invariance argument conditioned on the disturbance bound holding
- Discuss the union bound over trajectory time steps

Step 3: Formalize the GP cold-start protocol:
- Define the three phases (pre-fill, conservative, normal)
- Prove that kappa_init = 2 * kappa_nominal provides additional safety margin
- Show that sigma_d decreases as O(1/sqrt(n)) with n data points (standard GP convergence)

Step 4: Derive the train-deploy gap bound:
- CBF-Beta (training): actions sampled from truncated Beta distribution on C(x)
- RCBF-QP (deployment): actions projected onto robust safe set C_r(x) subset C(x)
- The gap arises because C_r(x) is smaller (more conservative) than C(x)
- Bound: |J_train - J_deploy| <= max_x |C(x) \ C_r(x)| * max_u |r(x,u)|

Step 5: Write Monte Carlo verification:
```python
# For 1000 random states and GP models:
# 1. Sample true disturbance from GP
# 2. Apply RCBF-QP with kappa * sigma margin
# 3. Verify safety constraint is satisfied
# 4. Count empirical violation rate
# Target: empirical violation rate < delta = 0.05
```

**Verification**:
- [ ] Theorem 3 stated with complete probability bounds
- [ ] GP cold-start safety properties formalized
- [ ] Train-deploy gap bound derived
- [ ] Monte Carlo verification matches theoretical prediction (within CI)

---

### Session 3: NE Convergence Analysis

**Objectives**:
- Compute exploitability of final policy pair
- Train best-response opponents
- Run PSRO-style verification on policy population
- Measure NE gap convergence across self-play phases

**Files to create/modify**:
- `scripts/compute_exploitability.py` — Exploitability computation
- `scripts/train_best_response.py` — Best-response training
- `scripts/psro_verification.py` — PSRO payoff matrix analysis
- `experiments/ne_analysis/` — Results directory

**Instructions**:

Step 1: Implement exploitability computation:
```python
def compute_exploitability(env, pi_P, pi_E, n_episodes=100, n_seeds=5):
    """
    Compute exploitability of strategy profile (pi_P, pi_E).
    Returns: exploit_value, br_P_value, br_E_value
    """
    # 1. Evaluate current profile
    baseline_return_P = evaluate(env, pi_P, pi_E, n_episodes)

    # 2. Train best-response pursuer against frozen pi_E
    br_P = train_best_response(env, role='pursuer', opponent=pi_E,
                                total_timesteps=500_000, n_seeds=n_seeds)
    br_return_P = evaluate(env, br_P, pi_E, n_episodes)

    # 3. Train best-response evader against frozen pi_P
    br_E = train_best_response(env, role='evader', opponent=pi_P,
                                total_timesteps=500_000, n_seeds=n_seeds)
    br_return_E = evaluate(env, pi_P, br_E, n_episodes)

    # 4. Compute exploitability
    exploit = (br_return_P - baseline_return_P) + (br_return_E - (-baseline_return_P))
    return exploit
```

Step 2: Run exploitability analysis on the final policy pair and on intermediate checkpoints from each AMS-DRL phase (S0, S1, S2, ..., SK). Plot exploitability over self-play phases.

Step 3: Implement PSRO payoff matrix:
```python
def compute_payoff_matrix(env, pursuer_policies, evader_policies, n_episodes=50):
    """
    Compute M[i,j] = E[r_P | pi_P^i vs pi_E^j]
    """
    n_P = len(pursuer_policies)
    n_E = len(evader_policies)
    M = np.zeros((n_P, n_E))
    for i in range(n_P):
        for j in range(n_E):
            M[i, j] = evaluate(env, pursuer_policies[i], evader_policies[j], n_episodes)
    return M
```

Step 4: Solve for NE of the matrix game using linear programming (scipy.optimize.linprog or nashpy library).

Step 5: Report:
- Exploitability of final pair: target < 0.10
- Exploitability convergence curve over self-play phases
- NE mixture weights from PSRO analysis
- Payoff matrix heatmap

**Verification**:
- [ ] Exploitability computed for final policy pair and all intermediate phases
- [ ] Best-response opponents trained with 5 seeds each
- [ ] PSRO payoff matrix computed and NE solved
- [ ] Exploitability < 0.10 at convergence
- [ ] All results saved to `experiments/ne_analysis/`

---

### Session 4: Comprehensive Benchmarking — Run All Baselines

**Objectives**:
- Run all 13 baselines (including HJ/DeepReach optimal) with 5+ seeds each
- Ensure fair comparison (same compute, same evaluation)
- Collect all primary and secondary metrics

**Files to create/modify**:
- `scripts/run_baselines.py` — Master baseline runner
- `configs/baselines/` — Config files for each baseline
- `experiments/baselines/` — Results directory

**Instructions**:

Step 1: Create configuration files for each baseline:

```yaml
# configs/baselines/dqn.yaml
baseline: dqn
algorithm: DQN
safety: none
self_play: none
total_timesteps: 2_000_000
network:
  hidden_sizes: [256, 256]
  activation: relu
hyperparams:
  learning_rate: 1e-3
  buffer_size: 100_000
  batch_size: 64
  exploration_fraction: 0.1
  target_update_interval: 1000
seeds: [0, 1, 2, 3, 4]
```

Create similar configs for all 13 baselines (12 learning-based + HJ/DeepReach optimal):
1. `dqn.yaml` — DQN (no safety) [12]
2. `ddpg.yaml` — DDPG (no safety) [12]
3. `maddpg.yaml` — MADDPG (no safety) [02]
4. `ppo_selfplay_no_cbf.yaml` — PPO + self-play, no CBF [18 adapted]
5. `ppo_cbf_no_selfplay.yaml` — PPO + CBF, no self-play [05]
6. `ampc_dqn.yaml` — A-MPC + DQN [15]
7. `madr.yaml` — MADR HJ-optimal [22]
8. `random.yaml` — Random policy
9. `greedy.yaml` — Greedy proportional navigation heuristic
10. `macpo.yaml` — MACPO [N03]
11. `mappo_lagrangian.yaml` — MAPPO-Lagrangian [N03]
12. `cpo.yaml` — CPO (Achiam 2017)

Note: SB-TRPO [N14] is an additional baseline if compute allows.

Step 2: Implement the master baseline runner:
```python
# scripts/run_baselines.py
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="baseline_runner")
def run_baseline(cfg: DictConfig):
    for seed in cfg.seeds:
        # 1. Initialize environment with seed
        env = make_pe_env(cfg.env, seed=seed)

        # 2. Initialize baseline agent
        agent = make_baseline_agent(cfg.baseline, env)

        # 3. Train
        agent.learn(total_timesteps=cfg.total_timesteps)

        # 4. Evaluate
        metrics = evaluate_agent(agent, env, n_episodes=100)

        # 5. Save results
        save_results(metrics, cfg.baseline.name, seed)
```

Step 3: Run all baselines. For baselines requiring opponent policies:
- DQN/DDPG: Train pursuer against a fixed evader (best available from ablations)
- MADDPG: Centralized training, decentralized execution
- PPO + self-play: Use AMS-DRL protocol without CBF
- PPO + CBF: Single-agent training with CBF, fixed scripted opponent
- A-MPC + DQN: Implement A-MPC for pursuer, DQN for evader
- MADR: Run DeepReach + adversarial MPC (use official code if available)
- Random/Greedy: No training needed
- MACPO/MAPPO-Lagrangian/CPO: Use official implementations from [N03]

Step 4: Collect metrics for each baseline:
```python
metrics_to_collect = {
    'primary': ['capture_rate', 'escape_rate', 'ne_gap', 'safety_violation_rate',
                'mean_capture_time', 'mean_min_distance_evader',
                'cbf_feasibility_rate', 'train_deploy_gap'],
    'secondary': ['obstacle_collision_rate', 'cbf_intervention_rate',
                   'policy_entropy', 'exploitability', 'qp_solve_time_p95']
}
```

**Verification**:
- [ ] All 13 baselines produce valid metrics (no NaN, no crashes)
- [ ] Each baseline run with minimum 5 seeds
- [ ] Same total_timesteps for all baselines
- [ ] Same evaluation protocol (100 episodes, same initial conditions)
- [ ] Results saved in structured format to `experiments/baselines/`
- [ ] Safety-relevant baselines report safety violation rate

#### Session 4b: HJ/DeepReach Classical Baseline (Addition 6)

**Objectives**:
- Compute the HJ-optimal value function for the PE game using DeepReach
- Extract the optimal pursuer/evader policies from the learned value function
- Compare RL capture rate vs HJ-optimal capture region
- Provide a classical game-theoretic optimality benchmark

**Files to create/modify**:
- `external/deepreach/` — Clone of DeepReach repository
- `scripts/train_deepreach_baseline.py` — DeepReach training for our PE formulation
- `scripts/eval_deepreach_vs_rl.py` — Head-to-head comparison
- `configs/baselines/deepreach.yaml` — DeepReach hyperparameters

**Instructions**:

Step 1: Clone DeepReach and set up the PE game formulation:
```python
# scripts/train_deepreach_baseline.py
# 6D state: [x_P, y_P, theta_P, x_E, y_E, theta_E]
# Relative formulation: [dx, dy, dtheta, x_E, y_E, theta_E]
# Value function V(x,t): min time-to-capture (pursuer minimizes, evader maximizes)

import deepreach
from deepreach.dataio import ReachabilityDataset

class PursuitEvasionGame(deepreach.Module):
    def __init__(self, v_max=1.0, omega_max=1.0, capture_radius=0.5):
        super().__init__()
        self.v_max = v_max
        self.omega_max = omega_max
        self.r_capture = capture_radius

    def hamiltonian(self, x, dvdx):
        """HJI Hamiltonian: H = min_u_P max_u_E (dvdx^T f(x,u_P,u_E))"""
        # Decompose into pursuer-minimizing and evader-maximizing terms
        # Pursuer chooses u_P to minimize H, evader chooses u_E to maximize H
        pass

    def boundary_fn(self, x):
        """Target set: ||pos_P - pos_E|| <= r_capture"""
        return torch.norm(x[..., :2] - x[..., 3:5], dim=-1) - self.r_capture
```

Step 2: Train the 6D value function:
- Network: 512-unit, 3-layer MLP with sine activations (per DeepReach [21])
- Training: 100K iterations, batch size 65536
- Time horizon: T = 55s (matching episode length)
- Compute: ~10-30 GPU hours on single A100

Step 3: Extract optimal policies and evaluate:
```python
# scripts/eval_deepreach_vs_rl.py
def extract_hj_policy(value_net, state, dt=0.05):
    """Extract optimal action via gradient of value function."""
    state.requires_grad_(True)
    V = value_net(state)
    dVdx = torch.autograd.grad(V, state)[0]
    # Pursuer: argmin_u (dVdx^T f(x,u))
    # Evader: argmax_u (dVdx^T f(x,u))
    return optimal_pursuer_action, optimal_evader_action

def compare_capture_regions(value_net, rl_policy, arena_size=20.0, grid_res=100):
    """Compare HJ capture region vs RL capture rate on a state grid."""
    # HJ capture region: {x : V(x,0) <= 0}
    # RL capture rate: empirical over 100 episodes per grid cell
    pass
```

Step 4: Visualization:
- Overlay HJ capture region boundary on RL capture heatmap
- Plot value function slices at fixed relative orientations
- Compare trajectory optimality: HJ vs RL path efficiency

**Compute**: 10-30 GPU hours (DeepReach training only)

**Verification**:
- [ ] DeepReach value function converges (loss < 1e-3)
- [ ] HJ-optimal policy achieves near-100% capture in HJ capture region
- [ ] RL capture rate compared quantitatively with HJ capture region
- [ ] Visualization showing HJ boundary overlaid on RL heatmap
- [ ] Results saved to `experiments/baselines/deepreach/`

---

### Session 5: Run All 11 Ablation Studies

**Objectives**:
- Execute all 11 ablation experiments with 5+ seeds
- Each ablation isolates exactly one component
- Collect all metrics for ablation analysis

**Files to create/modify**:
- `scripts/run_ablations.py` — Master ablation runner
- `configs/ablations/` — Config files for each ablation
- `experiments/ablations/` — Results directory

**Instructions**:

Step 1: Define ablation configurations. Each ablation has a control (full SafePE) and variant(s):

**Ablation 1: Safety mechanism**
- Control: SafePE (CBF-Beta training + RCBF-QP deploy)
- Variant A: CBF-QP filter only (no Beta truncation, project after sampling)
- Variant B: No safety (unconstrained)

**Ablation 2: Self-play protocol**
- Control: AMS-DRL (alternating, from [18])
- Variant A: MADDPG (centralized critic) [02]
- Variant B: Vanilla self-play (simple alternating, no cold-start)
- Variant C: Simultaneous self-play (no freezing) [N06]

**Ablation 3: Observation space**
- Control: BiMDN belief encoding
- Variant A: Raw observation (no belief encoder)
- Variant B: Full state (oracle, no partial observability)

**Ablation 4: Curriculum learning**
- Control: With curriculum (4 levels)
- Variant: Without curriculum (random scenarios from start)

**Ablation 5: Domain randomization**
- Control: With DR (Phase 4 parameters)
- Variant: Without DR (fixed parameters)

**Ablation 6: Dynamics model**
- Control: Unicycle (differential-drive)
- Variant: Ackermann (car-like)

**Ablation 7: GP disturbance estimation**
- Control: With GP (RCBF-QP)
- Variant: Without GP (nominal CBF-QP)

**Ablation 8: AMS-DRL vs Simultaneous [N06]**
- Control: AMS-DRL (alternating phases)
- Variant: Simultaneous training (both agents update every step)

**Ablation 9: Safety reward shaping (w5)**
- Control: With w5 = 0.05
- Variant: Without w5 = 0.0

**Ablation 10: Safety architecture**
- Control: CBF-Beta (training) -> RCBF-QP (deploy)
- Variant: BarrierNet end-to-end (if implemented in Phase 2.5)

**Ablation 11: Conflict-averse gradient**
- Control: Standard single-critic PPO
- Variant: CASRL dual-critic with conflict-averse optimization [30]

Step 2: Implement ablation runner that systematically varies one component while keeping others fixed:
```python
# scripts/run_ablations.py
ABLATIONS = {
    'safety_mechanism': ['cbf_beta', 'cbf_qp', 'none'],
    'self_play': ['ams_drl', 'maddpg', 'vanilla_sp', 'simultaneous_sp'],
    'observation': ['bimdn', 'raw', 'full_state'],
    'curriculum': [True, False],
    'domain_randomization': [True, False],
    'dynamics': ['unicycle', 'ackermann'],
    'gp_disturbance': [True, False],
    'self_play_timing': ['alternating', 'simultaneous'],
    'safety_reward_w5': [0.05, 0.0],
    'safety_arch': ['cbf_beta_rcbf', 'barriernet'],
    'casrl': [True, False],
}
```

Step 3: Run all ablations (total: ~30 configurations x 5 seeds = 150 runs). Parallelize across GPUs where possible.

Step 4: For each ablation, compute effect size (Cohen's d) of the variant vs control for each primary metric.

**Verification**:
- [ ] All 11 ablations produce valid metrics
- [ ] Each variant run with minimum 5 seeds
- [ ] Control condition (full SafePE) is identical across all ablations
- [ ] Effect sizes computed for each ablation
- [ ] Results saved to `experiments/ablations/`

---

### Session 6: Statistical Analysis

**Objectives**:
- Compute confidence intervals for all results
- Run significance tests comparing SafePE to each baseline
- Generate statistical summary tables
- Verify all claims are statistically supported

**Files to create/modify**:
- `scripts/statistical_analysis.py` — Main analysis script
- `scripts/utils/stats.py` — Statistical utility functions
- `experiments/statistics/` — Analysis results

**Instructions**:

Step 1: Implement statistical utilities:
```python
# scripts/utils/stats.py
import numpy as np
from scipy import stats

def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    boot_means = [np.mean(np.random.choice(data, len(data), replace=True))
                  for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, [alpha * 100, (1 - alpha) * 100])

def welch_t_test(data1, data2):
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
    cohens_d = (np.mean(data1) - np.mean(data2)) / np.sqrt(
        (np.var(data1) + np.var(data2)) / 2)
    return t_stat, p_value, cohens_d

def holm_bonferroni(p_values, alpha=0.05):
    """Apply Holm-Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    corrected = np.zeros(n, dtype=bool)
    for rank, idx in enumerate(sorted_idx):
        adjusted_alpha = alpha / (n - rank)
        if p_values[idx] <= adjusted_alpha:
            corrected[idx] = True
        else:
            break
    return corrected

def compute_all_statistics(safpe_results, baseline_results_dict):
    """Compute all pairwise comparisons with correction."""
    p_values = {}
    effect_sizes = {}
    for name, results in baseline_results_dict.items():
        _, p, d = welch_t_test(safpe_results, results)
        p_values[name] = p
        effect_sizes[name] = d

    # Apply Holm-Bonferroni correction
    names = list(p_values.keys())
    pvals = [p_values[n] for n in names]
    significant = holm_bonferroni(pvals)

    return {name: {'p': p_values[name], 'd': effect_sizes[name],
                    'significant': sig}
            for name, sig in zip(names, significant)}
```

Step 2: Pull all experiment data from wandb (or TensorBoard logs):
```python
# Pull all runs from Phases 1-4 via wandb API
import wandb
api = wandb.Api()
all_runs = api.runs("pursuit-evasion")
# Filter by group/tags to get phase-specific data
phase2_ablation = [r for r in all_runs if "phase2-ablation" in (r.group or "")]
phase3_eval = [r for r in all_runs if "phase3-evaluation" in (r.group or "")]
```

Step 3: Run statistical analysis for all primary metrics:
- For each metric, compare SafePE against all 13 baselines
- Apply Holm-Bonferroni correction for the 13 comparisons
- Report: mean, std, 95% CI, p-value (corrected), Cohen's d, significance

Step 3: Run ablation statistical analysis:
- For each of the 11 ablations, compare control vs variant(s)
- Report effect sizes and significance

Step 4: Generate summary:
```python
# Check: are all key claims statistically significant?
claims = {
    'SafePE achieves 0% safety violations': check_zero_violations(results),
    'SafePE outperforms DQN/DDPG': check_significant('dqn', 'capture_rate'),
    'NE gap < 0.10': check_ne_gap(results),
    'CBF-QP feasibility > 99%': check_feasibility(results),
    'Sim-to-real gap < 10%': check_sim_real_gap(results),
    'Self-play improves over no self-play': check_significant('ppo_cbf_no_sp', 'ne_gap'),
    'CBF improves safety over no CBF': check_significant('ppo_sp_no_cbf', 'safety'),
    'BiMDN improves over raw obs': check_ablation_significant('observation'),
}
for claim, result in claims.items():
    print(f"{'PASS' if result else 'FAIL'}: {claim}")
```

**Verification**:
- [ ] 95% CIs computed for all metrics across all experiments
- [ ] Significance tests run for all 12 baseline comparisons (with correction)
- [ ] Ablation effect sizes computed
- [ ] All paper claims verified as statistically supported
- [ ] Summary report generated at `experiments/statistics/summary.json`

---

### Session 7: Generate Publication Figures

**Objectives**:
- Generate all figures for the three papers
- Ensure figures are publication-quality (300 DPI, readable at column width, colorblind-friendly)
- Use consistent style across all papers

**Files to create/modify**:
- `scripts/generate_figures.py` — Master figure generation script
- `scripts/utils/plot_style.py` — Shared plotting configuration
- `figures/` — Output directory

**Instructions**:

Step 1: Set up consistent plotting style:
```python
# scripts/utils/plot_style.py
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    'safpe': '#0072B2',       # Blue
    'baseline_1': '#E69F00',  # Orange
    'baseline_2': '#009E73',  # Green
    'baseline_3': '#CC79A7',  # Pink
    'baseline_4': '#D55E00',  # Red-orange
    'baseline_5': '#56B4E9',  # Light blue
    'baseline_6': '#F0E442',  # Yellow
    'unsafe': '#D55E00',      # Red-orange
    'safe': '#009E73',        # Green
}

def setup_style():
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'legend.fontsize': 7,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'text.usetex': True,
    })

COLUMN_WIDTH = 3.5   # inches (IEEE single column)
DOUBLE_COL = 7.16    # inches (IEEE double column)
```

Step 2: Generate specific figures:

**Figure 1 — System Architecture** (`figures/architecture.pdf`):
- Block diagram of SafePE: Observation -> BiMDN -> PPO -> CBF-Beta -> Action
- Training loop with AMS-DRL self-play
- Deployment pipeline with RCBF-QP
- Use TikZ or matplotlib patches

**Figure 2 — Training Curves** (`figures/training_curves.pdf`):
- 2x2 subplot: (a) Capture rate over episodes, (b) Safety violation rate, (c) NE gap, (d) CBF intervention rate
- Show mean (thick) and individual seeds (thin, transparent)
- EMA smoothing with alpha=0.99
- X-axis: environment steps (millions)

**Figure 3 — Trajectory Plots** (`figures/trajectories.pdf`):
- 2x3 grid showing representative PE episodes
- Show pursuer path (blue), evader path (red), obstacles (gray), arena boundary (black)
- Mark start positions (circles), end positions (squares), capture point (star)
- Include safety margin visualization (dashed circles around obstacles)
- Select episodes that show: (a) direct pursuit, (b) obstacle use by evader, (c) corner trapping, (d) near-miss with CBF intervention, (e) timeout/evasion success, (f) real-robot trajectory
- **Implementation**: Use `PERenderer` with `render_mode="rgb_array"` to capture frames with CBF/FOV overlays, then compose grid with matplotlib. For publication quality, override PERenderer colors with the colorblind-friendly COLORS palette. Alternative: pure matplotlib for cleaner vector output (PDF)

**Figure 4 — Safety Heatmap** (`figures/safety_heatmap.pdf`):
- Heatmap of minimum CBF value h_min(x,y) across the arena
- Overlay representative trajectories
- Show that trajectories avoid low-CBF regions
- Colorbar: red (dangerous, h near 0) to green (safe, h >> 0)

**Figure 5 — NE Convergence** (`figures/ne_convergence.pdf`):
- (a) Exploitability vs self-play phase
- (b) PSRO payoff matrix heatmap
- (c) NE mixture weights (bar chart)

**Figure 6 — Ablation Bar Charts** (`figures/ablation_results.pdf`):
- Grouped bar chart for each ablation
- Show key metrics: capture rate, safety violation rate, NE gap
- Error bars: 95% CI
- Significance stars: * p<0.05, ** p<0.01, *** p<0.001

**Figure 7 — Baseline Comparison** (`figures/baseline_comparison.pdf`):
- Parallel coordinates or radar chart showing all metrics for all baselines
- OR: Multi-panel bar chart grouped by metric

**Figure 8 — Real-Robot Results** (`figures/real_robot.pdf`):
- (a) Photo of experimental setup
- (b) Real-robot trajectory overlaid on arena map
- (c) Sim vs real trajectory comparison
- (d) GP disturbance estimate over time

**Figure 9 — Sim-to-Real Gap** (`figures/sim_to_real_gap.pdf`):
- Bar chart comparing sim vs real metrics
- Show: capture rate, safety violations, capture time, min distance

**Verification**:
- [ ] All figures generated at 300 DPI
- [ ] All figures readable at 3.5 inch width (IEEE column)
- [ ] Colorblind-friendly palette used consistently
- [ ] All axes labeled with units
- [ ] All legends readable
- [ ] Figures saved as both PDF and PNG

---

### Session 8: Generate Publication Tables

**Objectives**:
- Generate all comparison tables for the three papers
- Ensure every number is traceable to raw experiment data
- Format for LaTeX with proper precision

**Files to create/modify**:
- `scripts/generate_tables.py` — Master table generation script
- `tables/` — Output LaTeX table files

**Instructions**:

Step 1: Generate Table 1 — Main Results Comparison:
```latex
% tables/main_comparison.tex
\begin{table*}[t]
\centering
\caption{Comprehensive comparison of SafePE against baselines across all primary metrics.
Bold: best result. Underline: second best. $\dagger$: statistically significant improvement
over baseline ($p < 0.05$, Holm-Bonferroni corrected).}
\label{tab:main_results}
\begin{tabular}{lcccccccc}
\toprule
Method & Cap. Rate & NE Gap & Safety Viol. & CBF Feas. & Cap. Time & Min Dist. & Train-Deploy \\
       & (\%) $\uparrow$ & $\downarrow$ & (\%) $\downarrow$ & (\%) $\uparrow$ & (s) $\downarrow$ & (m) $\uparrow$ & Gap (\%) $\downarrow$ \\
\midrule
Random & -- & -- & -- & -- & -- & -- & -- \\
Greedy (Prop. Nav) & -- & -- & -- & -- & -- & -- & -- \\
DQN [12] & -- & -- & -- & -- & -- & -- & -- \\
DDPG [12] & -- & -- & -- & -- & -- & -- & -- \\
MADDPG [02] & -- & -- & -- & -- & -- & -- & -- \\
PPO+SP (no CBF) [18] & -- & -- & -- & -- & -- & -- & -- \\
PPO+CBF (no SP) [05] & -- & -- & -- & -- & -- & -- & -- \\
A-MPC+DQN [15] & -- & -- & -- & -- & -- & -- & -- \\
MADR [22] & -- & -- & -- & -- & -- & -- & -- \\
CPO & -- & -- & -- & -- & -- & -- & -- \\
MACPO [N03] & -- & -- & -- & -- & -- & -- & -- \\
MAPPO-Lag. [N03] & -- & -- & -- & -- & -- & -- & -- \\
HJ-Optimal (DeepReach) [21] & -- & -- & N/A & N/A & -- & N/A & N/A \\
\midrule
\textbf{SafePE (Ours)} & \textbf{--} & \textbf{--} & \textbf{0.0} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} \\
\bottomrule
\end{tabular}
\end{table*}
```

Step 2: Generate Table 2 — Ablation Results:
```latex
% tables/ablation_results.tex
\begin{table}[t]
\centering
\caption{Ablation study results. Each row removes or replaces one component.
$\Delta$: change vs full SafePE.}
\label{tab:ablations}
\begin{tabular}{lccc}
\toprule
Variant & Cap. Rate & Safety & NE Gap \\
\midrule
Full SafePE & -- & 0.0\% & -- \\
\midrule
- CBF-Beta $\to$ CBF-QP & -- ($\Delta$) & -- & -- \\
- No safety & -- & -- & -- \\
- MADDPG (no AMS-DRL) & -- & -- & -- \\
- Vanilla SP & -- & -- & -- \\
- Simultaneous SP [N06] & -- & -- & -- \\
- Raw obs (no BiMDN) & -- & -- & -- \\
- Full state (oracle) & -- & -- & -- \\
- No curriculum & -- & -- & -- \\
- No DR & -- & -- & -- \\
- Ackermann dynamics & -- & -- & -- \\
- No GP disturbance & -- & -- & -- \\
- No w5 reward shaping & -- & -- & -- \\
- BarrierNet [N04] & -- & -- & -- \\
- CASRL [30] & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

Step 3: Generate Table 3 — Sim-to-Real Gap:
```latex
% tables/sim_to_real.tex
\begin{table}[t]
\centering
\caption{Simulation vs real-robot performance comparison.}
\label{tab:sim_to_real}
\begin{tabular}{lccc}
\toprule
Metric & Simulation & Real Robot & Gap (\%) \\
\midrule
Capture rate (\%) & -- & -- & -- \\
Safety violations (\%) & 0.0 & 0.0 & 0.0 \\
Mean capture time (s) & -- & -- & -- \\
CBF feasibility (\%) & -- & -- & -- \\
GP disturbance RMSE & N/A & -- & N/A \\
QP solve time (ms, p95) & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

Step 4: Generate Table 4 — Hyperparameter Sensitivity:
```latex
% tables/hyperparameter_sensitivity.tex
\begin{table}[t]
\centering
\caption{Sensitivity of SafePE to key hyperparameters.}
\label{tab:sensitivity}
\begin{tabular}{lcccc}
\toprule
Parameter & Value & Cap. Rate & Safety & NE Gap \\
\midrule
\multirow{3}{*}{$\alpha$ (CBF)} & 0.5 & -- & -- & -- \\
& \textbf{1.0} & -- & -- & -- \\
& 2.0 & -- & -- & -- \\
\midrule
\multirow{3}{*}{Safety margin} & 0.05m & -- & -- & -- \\
& \textbf{0.10m} & -- & -- & -- \\
& 0.20m & -- & -- & -- \\
\midrule
\multirow{3}{*}{$w_5$ (safety reward)} & 0.0 & -- & -- & -- \\
& 0.025 & -- & -- & -- \\
& \textbf{0.05} & -- & -- & -- \\
& 0.10 & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

Step 5: Write script to auto-populate all `--` placeholders from experiment data:
```python
# scripts/generate_tables.py
def fill_table(template_path, results_dict, output_path):
    """Replace -- placeholders with actual values from experiment results."""
    with open(template_path, 'r') as f:
        content = f.read()
    for key, value in results_dict.items():
        content = content.replace(f'{{{{ {key} }}}}', f'{value:.2f}')
    with open(output_path, 'w') as f:
        f.write(content)
```

**Verification**:
- [ ] All table templates created
- [ ] Auto-population script works correctly
- [ ] Every number in every table traceable to raw data
- [ ] Proper formatting (significant digits, units, bold/underline)
- [ ] Tables compile in LaTeX without errors

---

### Session 9: Write Paper 1 (Conference) — SafePE Algorithm + Simulation

**Objectives**:
- Write complete conference paper (6-8 pages) for ICRA/IROS/CoRL
- Focus on the SafePE algorithm and simulation results
- Include system architecture, training pipeline, and benchmarking

**Files to create/modify**:
- `papers/paper1_conference/main.tex`
- `papers/paper1_conference/sections/` — Section files
- `papers/paper1_conference/figures/` — Symlinks to generated figures
- `papers/paper1_conference/tables/` — Symlinks to generated tables

**Instructions**:

**Paper 1 LaTeX Outline**:

```latex
% papers/paper1_conference/main.tex
\documentclass[conference]{IEEEtran}  % or CoRL style
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{algorithm,algorithmic}
\usepackage{hyperref}

\title{SafePE: Safety-Guaranteed Deep Reinforcement Learning\\
for 1v1 Pursuit-Evasion on Ground Robots}

\begin{document}
\maketitle

\begin{abstract}
% ~150 words
% Problem: 1v1 PE on ground robots lacks safety guarantees
% Method: CBF-constrained self-play with NE convergence
% Key results: 0% safety violations, NE gap < 0.10, outperforms 13 baselines (incl. HJ-optimal)
% Significance: First safe DRL for 1v1 PE with provable safety
\end{abstract}

% Section I: Introduction (~1 page)
% - PE games are fundamental to robotics (cite [37])
% - Safety is critical for real deployment (cite [25], [N09])
% - No existing work combines safe DRL + 1v1 PE + ground robots (gap statement)
% - Our contributions (3 bullet points)
% - Paper organization

% Section II: Related Work (~0.75 pages)
% - PE with DRL: [02], [12], [18], [22], [34], [N06] (cite [37] for survey)
% - Safe RL: CBFs [03], [16], constrained optimization [N03], [N14] (cite [N09])
% - Safe PE: [01], [13], [14] (simulation only)
% - Self-play: [04], [18], [N06]
% - Position our work: fills gap G1 (Table comparing closest works)

% Section III: Problem Formulation (~0.75 pages)
% - Game setup (2-player zero-sum, unicycle dynamics)
% - Safety constraints (arena, obstacles, inter-robot)
% - Observation space (partial, FOV-limited)
% - Objective: find NE strategy profile with safety guarantees

% Section IV: SafePE Algorithm (~1.5 pages)
% A. VCP-CBF for Nonholonomic Safety (cite [N12])
% B. CBF-Constrained Beta Policy (cite [16])
%    - Algorithm 1: CBF-Beta PPO update
% C. AMS-DRL Self-Play with Safety (cite [18])
%    - Algorithm 2: Safe self-play protocol
% D. BiMDN Belief Encoder (cite [02])
% E. Safety-Reward Shaping (cite [05])

% Section V: Theoretical Analysis (~0.5 pages)
% - Theorem 1: VCP-CBF validity (sketch, full proof in supplementary)
% - Theorem 2: Forward invariance guarantee
% - Proposition: NE convergence under CBF constraints

% Section VI: Experiments (~2 pages)
% A. Setup (environment, hyperparameters, baselines)
% B. Main Results (Table 1: comparison with 12 baselines + HJ/DeepReach optimal)
% C. Safety Analysis (0% violations, CBF feasibility, DCBF theorem verification)
% D. NE Convergence (exploitability curve, PSRO verification)
% E. Ablation Studies (Table 2: 11 ablations, key findings)
% F. Asymmetric Capabilities (v_P/v_E = 1.5, 0.8, 0.5 — speed advantage analysis)
% G. Generalization Study (zero-shot transfer to corridor, L-shaped, warehouse)
% H. Interpretability (saliency maps, phase portraits, emergent strategies)
% I. Human Evader Experiment (pursuer vs human-controlled evader)

% Section VII: Conclusion (~0.25 pages)
% - Summary of contributions
% - Limitations (arena size, obstacle complexity, 2D only)
% - Future work (multi-agent, vision-based, sim-to-real in Paper 2)

\end{document}
```

Step 1: Write each section following the outline above.

Step 2: Key writing guidelines:
- Every claim must reference a table, figure, or equation
- Every baseline comparison must include statistical significance
- State limitations honestly (do not overstate safety claims)
- Use active voice: "We propose..." not "It is proposed..."
- Keep the paper self-contained (a reader should not need to read other papers to understand)

Step 3: Run LaTeX compilation and fix errors.

Step 4: Check page count (target: 6 pages for ICRA/IROS, 8 for CoRL, + references).

**Verification**:
- [ ] Paper compiles without LaTeX errors
- [ ] All figures and tables referenced in text
- [ ] All citations present in bibliography
- [ ] Page count within limit
- [ ] Abstract accurately summarizes the paper
- [ ] No orphan citations (every citation in bib is referenced)

---

### Session 10: Prepare Real-Robot Demonstration Videos

**Objectives**:
- Edit and annotate demonstration videos from Phase 4
- Create multiple scenario demonstrations
- Produce supplementary video for paper submission

**Files to create/modify**:
- `videos/raw/` — Raw recordings from Phase 4
- `videos/edited/` — Edited demonstration videos
- `videos/supplementary_video.mp4` — Final supplementary video
- `scripts/annotate_video.py` — Video annotation overlay script

**Instructions**:

Step 1: Organize raw footage into scenarios:
- Scenario A: Open arena, basic pursuit-evasion
- Scenario B: With obstacles, evader uses obstacles strategically
- Scenario C: Near-boundary encounter, CBF intervention visible
- Scenario D: Close encounter, capture event
- Scenario E: Timeout/successful evasion
- Scenario F: CBF-QP intervention (action correction visible)

Step 2: Create video annotation overlay:
```python
# scripts/annotate_video.py
import cv2
import numpy as np

def annotate_frame(frame, metadata):
    """Add annotations to video frame."""
    # Add text overlays:
    # - Current time
    # - Distance between agents
    # - CBF margin value (color-coded: green=safe, yellow=warning, red=near-boundary)
    # - "CBF ACTIVE" indicator when filter modifies action
    # - Agent labels (Pursuer, Evader)
    # - Safety margin circles around obstacles
    pass
```

Step 3: Create the supplementary video (target: 2-3 minutes):
```
00:00-00:15  Title card: "SafePE: Safety-Guaranteed Deep RL for Pursuit-Evasion"
00:15-00:30  Problem statement (text + animation of unsafe PE)
00:30-01:00  System architecture diagram (animated)
01:00-01:30  Simulation results: 3 representative episodes with different strategies
01:30-02:00  Real-robot demonstration: 2-3 scenarios (side-by-side sim vs real)
02:00-02:20  Safety guarantee visualization: CBF margin heatmap with trajectory
02:20-02:40  Key results summary (metrics on screen)
02:40-03:00  Closing: contributions, future work, QR code to code repo
```

Step 4: Export at 1080p, 30fps, MP4 (H.264).

**Verification**:
- [ ] All scenarios recorded and edited
- [ ] Annotations are clear and readable
- [ ] Supplementary video is 2-3 minutes
- [ ] Video resolution: 1080p
- [ ] No audio issues (if narrated)
- [ ] CBF intervention clearly visible in relevant scenarios

---

### Session 11: Write Paper 2 (Journal) — Full System + Real-Robot

**Objectives**:
- Write journal paper (8-12 pages) for RA-L or T-RO
- Include complete system description, extensive experiments, and real-robot validation
- Extend Paper 1 with real-robot results, sim-to-real analysis, GP disturbance results

**Files to create/modify**:
- `papers/paper2_journal/main.tex`
- `papers/paper2_journal/sections/` — Section files

**Instructions**:

**Paper 2 LaTeX Outline** (extends Paper 1):

```latex
\title{Safe Pursuit-Evasion on Ground Robots: From CBF-Constrained
Self-Play to Real-World Deployment}

% Sections (8-12 pages for RA-L, 12-20 for T-RO):

% I. Introduction (1 page)
% - Extended motivation with real-robot focus
% - Contribution 1: SafePE algorithm (refer to Paper 1 if published)
% - Contribution 2: Sim-to-real pipeline with safety guarantees
% - Contribution 3: Comprehensive benchmark + open-source release

% II. Related Work (1.5 pages)
% - Extended coverage: sim-to-real [07, N10, N11], real PE [02, 08, 22]
% - Position vs [N15] RMARL-CBF-SAM (closest prior work)

% III. Problem Formulation (1 page)
% - Same as Paper 1 but with additional detail on robot dynamics
% - Include Ackermann model alongside unicycle

% IV. SafePE Algorithm (2 pages)
% - Full algorithm description (more detail than Paper 1)
% - VCP-CBF formulation with complete derivation
% - CBF-Beta policy with convergence guarantee
% - 3-tier infeasibility handling (N13 learned feasibility)
% - AMS-DRL self-play protocol
% - BiMDN belief encoder architecture

% V. Sim-to-Real Transfer Pipeline (1.5 pages)
% A. Isaac Lab Training Environment [N11]
% B. Domain Randomization Strategy [07, N10]
% C. ONNX Export and Gazebo Validation [N10]
% D. RCBF-QP Deployment Safety Filter [06]
% E. GP Disturbance Estimation and Cold-Start Protocol
% F. Real-Time QP Solver Benchmarking

% VI. Theoretical Analysis (1 page)
% - Theorem: VCP-CBF validity (full proof)
% - Theorem: Probabilistic safety under GP uncertainty
% - Proposition: NE convergence

% VII. Experiments (4 pages)
% A. Simulation Results (summary from Paper 1, including HJ/DeepReach baseline)
% B. Sim-to-Real Gap Analysis (Table 3)
% C. Real-Robot Results
%    - Capture rate, safety, NE gap on physical robots (TurtleBot3 Burger)
%    - GP disturbance estimation accuracy
%    - QP solve time distribution
% D. Baseline Comparison (full Table 1, including HJ-optimal)
% E. Ablation Studies (full Table 2)
% F. Asymmetric Capability Analysis (speed ratio experiments)
% G. Generalization Study (complex environments: corridor, L-shaped, warehouse)
% H. Human Evader Experiment (pursuer vs human opponents)
% I. Interpretability Analysis (saliency maps, belief evolution, phase portraits)
% J. Opponent Modeling Analysis (strategy classification, adaptation metrics)
% K. Hyperparameter Sensitivity (Table 4)

% VIII. Discussion (0.75 pages)
% - Limitations: arena size, obstacle complexity, 2D planar
% - Safety claims: what is proven (DCBF theorem) vs empirical
% - Compute requirements
% - When SafePE should/shouldn't be used
% - Generalization limitations and failure modes

% IX. Conclusion (0.5 pages)
```

Step 1: Extend Paper 1 content with real-robot sections.
Step 2: Add comprehensive sim-to-real analysis.
Step 3: Include hyperparameter sensitivity study.
Step 4: Add extended discussion section.

**Verification**:
- [ ] Paper compiles without errors
- [ ] Real-robot results prominently featured
- [ ] Sim-to-real gap quantified for all metrics
- [ ] GP disturbance results included
- [ ] Page count: 8-12 pages (RA-L) or 12-20 (T-RO)
- [ ] Supplementary video referenced

---

### Session 12: Write Paper 3 (Theory) — Formal Safety + NE Convergence

**Objectives**:
- Write theory paper (6-8 pages) for CDC or L4DC
- Focus on formal proofs: VCP-CBF validity, RCBF probabilistic guarantee, NE convergence under safety constraints
- Simulation validation only (no real-robot required)

**Files to create/modify**:
- `papers/paper3_theory/main.tex`
- `papers/paper3_theory/sections/`

**Instructions**:

**Paper 3 LaTeX Outline**:

```latex
\title{Provably Safe Nash Equilibrium Learning for\\
Pursuit-Evasion with Nonholonomic Dynamics}

% Sections (6-8 pages):

% I. Introduction (0.75 pages)
% - Formal safety in adversarial games
% - Challenge: CBFs + nonholonomic dynamics + game-theoretic training
% - Contribution: theoretical framework + proofs

% II. Preliminaries (0.75 pages)
% - CBF definition and forward invariance
% - Zero-sum differential games and NE
% - Unicycle dynamics and nonholonomic constraints

% III. VCP-CBF for Nonholonomic Systems (1.5 pages)
% - Definition of virtual control point
% - Lemma 1: Uniform relative degree 1
% - Theorem 1: VCP-CBF validity
% - Theorem 2: Forward invariance under VCP-CBF-QP
% - Theorem 2b: DCBF safety guarantee (gamma=0.2, from Phase 3 Stage -1)
%   * Discrete-time formulation via Nagumo + comparison principle
%   * Bridges training-time (discrete) and deployment (continuous) safety
% - Remark: Comparison with HOCBF approach

% IV. Probabilistic Safety under Model Uncertainty (1 page)
% - GP disturbance model
% - Theorem 3: RCBF probabilistic safety guarantee
% - Corollary: Safety probability improves with more data
% - GP cold-start protocol and its guarantees

% V. Nash Equilibrium under Safety Constraints (1.5 pages)
% - Definition: Constrained Nash equilibrium for PE
% - Theorem 4: CBF-Beta preserves NE convergence of AMS-DRL
%   (key insight: truncation preserves gradient direction)
% - Proposition: Exploitability bound for CBF-constrained self-play
% - Connection to CMDP formulation [N09]

% VI. Numerical Validation (1.5 pages)
% - Verify Theorem 1: CBF-QP feasibility rate >99.9%
% - Verify Theorem 2b: DCBF 10K-state + 1K-trajectory verification (Phase 3)
% - Verify Theorem 3: Monte Carlo violation rate matches prediction
% - Verify Theorem 4: Exploitability convergence with/without CBF
% - Comparison with HJ/DeepReach optimal value function (capture region overlap)

% VII. Conclusion (0.25 pages)
```

Step 1: Write formal proofs with full mathematical rigor.
Step 2: Include all lemmas, theorems, and corollaries with complete proofs.
Step 3: Add numerical validation experiments.
Step 4: Ensure proof notation is consistent throughout.

**Verification**:
- [ ] All theorems have complete proofs (not just sketches)
- [ ] Notation consistent throughout
- [ ] Numerical results match theoretical predictions
- [ ] Paper is self-contained (reader needs only basic CBF and game theory background)
- [ ] Page count within limit (6 pages + references for CDC, 8 for L4DC)

---

### Session 13: Open-Source Release

**Objectives**:
- Clean and organize codebase for public release
- Write comprehensive documentation
- Create Docker images for reproducibility
- Write tutorial notebooks
- Set up GitHub repository

**Files to create/modify**:
- `README.md` — Project overview and quick start
- `CONTRIBUTING.md` — Contribution guidelines
- `LICENSE` — MIT or Apache 2.0
- `setup.py` / `pyproject.toml` — Package configuration
- `Dockerfile` — Training/evaluation Docker image
- `notebooks/` — Tutorial Jupyter notebooks
- `.github/workflows/ci.yml` — CI pipeline

**Instructions**:

Step 1: Clean the codebase:
- Remove debug code, commented-out experiments
- Ensure all functions have docstrings
- Run linter (flake8/ruff) and formatter (black)
- Remove any hardcoded paths; use config files
- Remove any credentials or private information

Step 2: Write README.md:
```markdown
# SafePE: Safety-Guaranteed Deep RL for 1v1 Pursuit-Evasion

[Paper 1](link) | [Paper 2](link) | [Paper 3](link) | [Video](link)

## Overview
SafePE is the first safety-guaranteed deep reinforcement learning system
for 1v1 pursuit-evasion on ground robots with nonholonomic dynamics.

## Key Features
- CBF-constrained Beta policies for safe training
- RCBF-QP safety filter for robust deployment
- AMS-DRL self-play with NE convergence
- BiMDN belief encoder for partial observability
- Sim-to-real pipeline (Isaac Lab -> ONNX -> Gazebo -> Real)

## Quick Start
[Installation, training, evaluation instructions]

## Results
[Key metrics table, figures]

## Citation
[BibTeX]
```

Step 3: Write tutorial notebooks:
- `notebooks/01_environment.ipynb` — PE environment walkthrough
- `notebooks/02_cbf_safety.ipynb` — CBF safety layer demonstration
- `notebooks/03_training.ipynb` — Training with self-play
- `notebooks/04_evaluation.ipynb` — Evaluation and metrics

Step 4: Create Docker images:
```dockerfile
# docker/Dockerfile.train
FROM nvidia/cuda:12.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /workspace/safe-pe
WORKDIR /workspace/safe-pe
CMD ["python", "scripts/train.py"]
```

Step 5: Set up CI:
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[test]"
      - run: pytest tests/ -v
```

**Verification**:
- [ ] README is comprehensive and accurate
- [ ] `pip install -e .` works in a fresh environment
- [ ] `pytest tests/` passes all tests
- [ ] Docker image builds and runs
- [ ] Tutorial notebooks execute without errors
- [ ] No hardcoded paths or credentials in codebase
- [ ] LICENSE file present
- [ ] CI pipeline passes

---

### Session 14: Supplementary Material and Appendices

**Objectives**:
- Write extended appendices for all three papers
- Include complete proofs, additional experiments, hyperparameter details
- Prepare camera-ready supplementary material

**Files to create/modify**:
- `papers/paper1_conference/supplementary.tex`
- `papers/paper2_journal/appendix.tex`
- `papers/paper3_theory/supplementary.tex`

**Instructions**:

Step 1: Paper 1 Supplementary:
- Complete proofs of all theorems (moved from main paper for space)
- Extended ablation results (all 11 ablations with all metrics)
- Hyperparameter sensitivity analysis (full tables)
- Training details (all hyperparameters, network architectures)
- Additional trajectory visualizations (generate using `PERenderer` with `render_mode="rgb_array"`)
- Supplementary video: generate annotated PE episodes using `PERenderer` + `RecordVideo` wrapper, showing CBF overlays, FOV cones, belief distributions. Stitch with ffmpeg. Upload to YouTube and link from paper

Step 2: Paper 2 Appendix:
- Full VCP-CBF derivation
- GP kernel selection and hyperparameter tuning
- Isaac Lab environment implementation details
- ROS2 deployment architecture
- Real-robot hardware specifications
- Extended sim-to-real analysis

Step 3: Paper 3 Supplementary:
- Complete proofs with all intermediate steps
- Extended numerical validation
- Sensitivity of theoretical bounds to assumptions
- Connection to prior CBF results [N12], [16], [06]

**Verification**:
- [ ] All supplementary material compiles
- [ ] Every proof is mathematically complete
- [ ] All hyperparameters documented
- [ ] Additional experiment results included
- [ ] References consistent with main papers

---

### Session 15: Generalization Study (Addition 7)

**Objectives**:
- Evaluate Phase 3 trained policies on unseen complex environments
- Measure zero-shot transfer to corridor, L-shaped room, and warehouse layouts
- Test distribution shift: train 2 obstacles → test 5; train 20×20m → test 15×15m
- Quantify generalization gaps to identify failure modes

**Files to create/modify**:
- `scripts/run_generalization_study.py` — Generalization evaluation runner
- `experiments/generalization/` — Results directory

**Instructions**:

Step 1: Define the generalization test matrix:
```python
# scripts/run_generalization_study.py
GENERALIZATION_TESTS = {
    # Layout generalization (train: simple arena with circular obstacles)
    'corridor': {'layout': 'corridor', 'size': (4, 20), 'walls': 4},
    'l_shaped': {'layout': 'l_shaped', 'size': (15, 15), 'cutout': (7, 7)},
    'warehouse': {'layout': 'warehouse', 'size': (20, 20), 'shelves': 8},

    # Obstacle count generalization (train: 2 obstacles)
    'obs_3': {'layout': 'simple', 'n_obstacles': 3},
    'obs_5': {'layout': 'simple', 'n_obstacles': 5},
    'obs_8': {'layout': 'simple', 'n_obstacles': 8},

    # Arena size generalization (train: 20×20m)
    'arena_15': {'layout': 'simple', 'arena_size': 15.0},
    'arena_10': {'layout': 'simple', 'arena_size': 10.0},
    'arena_25': {'layout': 'simple', 'arena_size': 25.0},
}
```

Step 2: Run each test: load Phase 3 trained models (best seed), evaluate 100 episodes per test, record all primary metrics.

Step 3: Compute generalization gap = |metric_test - metric_train| / metric_train for each test condition.

Step 4: Generate heatmap: rows = test conditions, columns = metrics, cells = generalization gap (%).

**Compute**: 0 GPU hours (evaluation only, ~1h CPU)

**Verification**:
- [ ] All 9 test conditions evaluated (100 episodes each)
- [ ] Generalization gap table complete for all primary metrics
- [ ] Heatmap figure generated and saved to `figures/generalization_heatmap.pdf`
- [ ] Failure mode analysis for worst-performing conditions
- [ ] Results saved to `experiments/generalization/`

---

### Session 16: Interpretability Analysis (Addition 9)

**Objectives**:
- Generate input saliency maps showing which observation dimensions drive policy decisions
- Visualize BiMDN belief evolution over episode trajectories
- Create policy phase portraits (action vector fields across the arena)
- Provide intuitive understanding of learned strategies

**Files to create/modify**:
- `evaluation/interpretability/saliency_maps.py` — Gradient-based input saliency
- `evaluation/interpretability/belief_evolution.py` — BiMDN belief animation
- `evaluation/interpretability/phase_portraits.py` — Policy vector fields
- `evaluation/interpretability/strategy_analysis.py` — Emergent strategy classification
- `figures/interpretability/` — Output directory

**Instructions**:

Step 1: Input saliency maps:
```python
# evaluation/interpretability/saliency_maps.py
def compute_saliency(policy, obs, method='gradient'):
    """Compute gradient of V(s) w.r.t. observation dimensions."""
    obs_tensor = torch.tensor(obs, requires_grad=True)
    value = policy.predict_values(obs_tensor)
    value.backward()
    saliency = obs_tensor.grad.abs()  # |dV/d(obs_i)|
    return saliency

def generate_saliency_heatmap(policy, env, n_episodes=20):
    """Average saliency across episodes, grouped by observation type."""
    # Group by: ego state, relative position, lidar, belief
    # Normalize and visualize as bar chart + heatmap over time
    pass
```

Step 2: BiMDN belief evolution:
```python
# evaluation/interpretability/belief_evolution.py
def animate_belief_evolution(belief_encoder, episode_trajectory):
    """Create animated visualization of belief distribution over time."""
    # For each timestep: extract GMM parameters (mu, sigma, pi)
    # Plot 2D Gaussian mixture contours over arena
    # Show ground truth opponent position + belief distribution
    # Save as animated GIF / MP4
    pass
```

Step 3: Policy phase portraits:
```python
# evaluation/interpretability/phase_portraits.py
def generate_phase_portrait(policy, arena_size=20.0, grid_res=40):
    """Generate vector field of policy actions across arena."""
    # Fix opponent at center, vary agent position on grid
    # At each position: query policy for action (v, omega)
    # Plot quiver plot with arrows showing velocity direction
    # Color by speed magnitude
    pass
```

Step 4: Strategy classification:
- Cluster episode trajectories by shape (DTW distance + k-means)
- Identify emergent strategies: intercept, cut-off, wall-trap (pursuer); dodge, spiral, wall-hug (evader)
- Report strategy distribution and transition frequencies

**Compute**: 2-5 GPU hours (gradient computation on trained models)

**Verification**:
- [ ] Saliency maps generated for both pursuer and evader policies
- [ ] BiMDN belief animation shows convergence to true opponent state
- [ ] Phase portraits show qualitatively different strategies in different arena regions
- [ ] Strategy classification identifies at least 3 distinct strategy types per role
- [ ] All visualizations saved to `figures/interpretability/`

---

### Session 17: Opponent Modeling Analysis (Addition 8)

**Objectives**:
- Post-hoc analysis of trajectory data to understand opponent modeling behavior
- Classify evader strategies from trajectories
- Measure pursuer adaptation to different evader strategies
- (Optional) Integrate BiMDN intent prediction head for explicit opponent modeling

**Files to create/modify**:
- `evaluation/opponent_modeling_analysis.py` — Trajectory-based opponent modeling analysis
- `experiments/opponent_modeling/` — Results directory

**Instructions**:

Step 1: Strategy classification from trajectory data:
```python
# evaluation/opponent_modeling_analysis.py
from sklearn.cluster import KMeans
from dtaidistance import dtw

def classify_evader_strategies(trajectories, n_clusters=5):
    """Cluster evader trajectories by shape similarity."""
    # Compute DTW distance matrix between all trajectory pairs
    # Apply k-means / spectral clustering
    # Label clusters: straight-line, spiral, wall-hugging, random, evasive
    return cluster_labels, cluster_centers

def measure_pursuer_adaptation(pursuer_trajectories, evader_strategy_labels):
    """Measure whether pursuer changes behavior based on evader strategy."""
    # For each evader strategy cluster:
    #   - Compute average pursuer interception angle
    #   - Compute average time-to-capture
    #   - Compute pursuit trajectory curvature
    # Test if differences are statistically significant (ANOVA)
    pass
```

Step 2: Belief-action correlation analysis:
- Extract BiMDN belief states from evaluation episodes
- Correlate belief uncertainty (entropy) with action conservatism (CBF margin)
- Plot: x = belief entropy, y = CBF safety margin, color = capture outcome

Step 3: (Optional, +5-15 GPU hours) Explicit intent prediction:
- Add a lightweight MLP head to BiMDN that predicts evader's next-step velocity
- Train on Phase 3 trajectory data
- Evaluate prediction accuracy vs episode timestep (does prediction improve over time?)

**Compute**: 0 GPU hours (trajectory analysis only; optional intent head: 5-15h)

**Verification**:
- [ ] Evader strategy clusters identified with clear separation (silhouette score > 0.3)
- [ ] Pursuer adaptation analysis shows statistically significant behavior differences
- [ ] Belief-action correlation plot generated
- [ ] Results saved to `experiments/opponent_modeling/`

---

### Session 18: Human Evader Experiment (Addition 4)

**Objectives**:
- Evaluate trained pursuer policy against human-controlled evader
- Provide keyboard/joystick interface for human control
- Run structured experiment: 20 episodes per participant
- Collect and analyze human vs AI performance data

**Files to create/modify**:
- `scripts/human_evader_experiment.py` — Human evader experiment runner
- `envs/human_interface.py` — Keyboard/joystick controller
- `experiments/human_evader/` — Results directory

**Instructions**:

Step 1: Implement human control interface:
```python
# envs/human_interface.py
import pygame

class HumanEvaderController:
    """Replace evader policy with human keyboard/joystick input."""

    def __init__(self, control_mode='keyboard'):
        self.control_mode = control_mode
        pygame.init()
        if control_mode == 'joystick' and pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

    def get_action(self):
        """Map human input to (v, omega) action."""
        if self.control_mode == 'keyboard':
            keys = pygame.key.get_pressed()
            v = 0.0
            omega = 0.0
            if keys[pygame.K_UP]: v = 1.0
            if keys[pygame.K_DOWN]: v = -0.5
            if keys[pygame.K_LEFT]: omega = 1.0
            if keys[pygame.K_RIGHT]: omega = -1.0
            return np.array([v, omega])
        elif self.control_mode == 'joystick':
            v = -self.joystick.get_axis(1)  # Forward/back
            omega = -self.joystick.get_axis(0)  # Left/right
            return np.array([v, omega])
```

Step 2: Experiment protocol:
- 3-5 participants (lab members)
- Each participant: 5 practice episodes (not recorded) + 20 test episodes
- Record: capture outcome, time-to-capture/escape, trajectory, human reaction time
- Pursuer uses best Phase 3 policy with CBF safety filter active
- Render in real-time with `PERenderer` at 30 FPS

Step 3: Analysis:
- Compare: AI evader capture rate vs human evader capture rate
- Does the pursuer policy generalize to human opponents?
- Qualitative: what strategies do humans use that AI doesn't?
- Report mean ± std across participants

**Compute**: 0 GPU hours (real-time inference only)

**Verification**:
- [ ] Human interface works with both keyboard and joystick
- [ ] 20 episodes per participant recorded with full trajectory data
- [ ] Capture rate comparison table: AI evader vs human evader
- [ ] Qualitative strategy analysis documented
- [ ] Results saved to `experiments/human_evader/`

---

## 5. Testing Plan (Automated)

### 5.1 Unit Tests

**Test U1: Exploitability Computation Correctness**
- **Name**: `test_exploitability_known_game`
- **Description**: Verify exploitability computation on a known game (Rock-Paper-Scissors) with known NE
- **Inputs**: Payoff matrix for RPS, strategy profile at NE (1/3, 1/3, 1/3)
- **Expected**: exploit = 0.0 (within tolerance 1e-6)
- **Pass criteria**: |computed_exploit - 0.0| < 1e-6

**Test U2: Exploitability Non-Zero for Non-NE**
- **Name**: `test_exploitability_non_ne`
- **Description**: Verify exploitability > 0 for a known non-NE strategy
- **Inputs**: Payoff matrix for RPS, strategy (1, 0, 0) (always Rock)
- **Expected**: exploit > 0
- **Pass criteria**: computed_exploit > 0.1

**Test U3: Statistical Test Implementations**
- **Name**: `test_bootstrap_ci`
- **Description**: Verify bootstrap CI contains true mean for known distribution
- **Inputs**: 1000 samples from N(5, 1)
- **Expected**: 95% CI contains 5.0
- **Pass criteria**: lower < 5.0 < upper

**Test U4: Holm-Bonferroni Correction**
- **Name**: `test_holm_bonferroni`
- **Description**: Verify correction produces expected results on known p-values
- **Inputs**: p_values = [0.001, 0.01, 0.04, 0.06], alpha = 0.05
- **Expected**: [True, True, True, False] (first 3 significant after correction)
- **Pass criteria**: Output matches expected

**Test U5: Figure Generation Scripts Run Without Errors**
- **Name**: `test_figure_generation`
- **Description**: Verify all figure generation functions execute without exceptions
- **Inputs**: Synthetic data (random arrays with correct shapes)
- **Expected**: No exceptions, output files created
- **Pass criteria**: All figure files exist and are non-empty

**Test U6: Table Generation Scripts Run Without Errors**
- **Name**: `test_table_generation`
- **Description**: Verify all table generation functions produce valid LaTeX
- **Inputs**: Synthetic results dictionary
- **Expected**: Valid LaTeX output
- **Pass criteria**: LaTeX compiles without errors

**Test U7: Bootstrap CI Coverage**
- **Name**: `test_bootstrap_coverage`
- **Description**: Run 1000 CI computations and verify coverage
- **Inputs**: Samples from known distribution
- **Expected**: ~95% of CIs contain true mean
- **Pass criteria**: Coverage in [0.93, 0.97]

**Test U8: Cohen's d Computation**
- **Name**: `test_cohens_d`
- **Description**: Verify effect size on known distributions
- **Inputs**: N(0,1) vs N(0.5,1) — known d = 0.5
- **Expected**: |computed_d - 0.5| < 0.1 (with n=1000)
- **Pass criteria**: Within tolerance

### 5.2 Integration Tests

**Test I1: All Baselines Produce Valid Metrics**
- **Name**: `test_baseline_metrics_valid`
- **Description**: Run each baseline for 1000 steps and verify metrics are valid (no NaN, correct range)
- **Inputs**: Each baseline config with reduced timesteps
- **Expected**: All metrics in valid range (e.g., capture_rate in [0, 1])
- **Pass criteria**: No NaN values, all metrics in expected ranges

**Test I2: Ablation Pipeline Runs End-to-End**
- **Name**: `test_ablation_pipeline`
- **Description**: Run one ablation (e.g., curriculum on/off) with reduced timesteps
- **Inputs**: Ablation config with total_timesteps=10000
- **Expected**: Two sets of results (with/without curriculum)
- **Pass criteria**: Both variants complete, results saved, comparison script runs

**Test I3: Figure Generation from Raw Data**
- **Name**: `test_figures_from_raw`
- **Description**: Generate all figures from actual (or mock) experiment data
- **Inputs**: Experiment result files
- **Expected**: All figure PDFs generated
- **Pass criteria**: All expected figure files exist, non-zero size, render correctly

**Test I4: Table Generation from Raw Data**
- **Name**: `test_tables_from_raw`
- **Description**: Generate all tables from actual experiment data
- **Inputs**: Experiment result files
- **Expected**: All table .tex files generated
- **Pass criteria**: All table files exist, compile in LaTeX

**Test I5: Statistical Pipeline End-to-End**
- **Name**: `test_stats_pipeline`
- **Description**: Run full statistical analysis from experiment results to summary report
- **Inputs**: Experiment result files (all baselines and ablations)
- **Expected**: Summary JSON with all comparisons, corrected p-values, effect sizes
- **Pass criteria**: Summary file exists, all comparisons present, no missing values

### 5.3 System Tests

**Test S1: Claims Supported by Data**
- **Name**: `test_claims_supported`
- **Description**: For each claim in the paper, verify the supporting data
- **Inputs**: Paper claims list + experiment results
- **Expected**: Each claim has statistical support (p < 0.05 or CI excludes baseline)
- **Pass criteria**: All claims pass verification

**Test S2: Reproducibility from Checkpoint**
- **Name**: `test_reproducibility`
- **Description**: Re-run evaluation from saved checkpoint and verify results match
- **Inputs**: Checkpoint file + original evaluation results
- **Expected**: Evaluation metrics match within 2% (due to stochastic evaluation)
- **Pass criteria**: |new_metric - original_metric| / original_metric < 0.02 for all metrics

**Test S3: Docker Reproducibility**
- **Name**: `test_docker_reproducibility`
- **Description**: Build Docker image, run evaluation, verify results
- **Inputs**: Dockerfile + checkpoint
- **Expected**: Same results as host evaluation
- **Pass criteria**: Results match within tolerance

### 5.4 Validation Tests

**Test V1: LaTeX Compilation**
- **Name**: `test_latex_compiles`
- **Description**: Compile all three papers and supplementary material
- **Inputs**: All .tex files
- **Expected**: No errors (warnings acceptable)
- **Pass criteria**: PDF files generated without LaTeX errors

**Test V2: Figure Rendering**
- **Name**: `test_figure_rendering`
- **Description**: Verify all figures render correctly at publication size
- **Inputs**: Generated figure PDFs
- **Expected**: Readable text, no clipping, correct aspect ratio
- **Pass criteria**: Visual inspection passes (automated: check DPI, dimensions)

**Test V3: Reference Consistency**
- **Name**: `test_references`
- **Description**: Check that all citations in text have bib entries and vice versa
- **Inputs**: .tex files + .bib file
- **Expected**: No missing references, no orphan bib entries
- **Pass criteria**: Zero missing citations, zero orphan entries

**Test V4: No Orphan Figures/Tables**
- **Name**: `test_no_orphan_floats`
- **Description**: Check that all figures/tables are referenced in text
- **Inputs**: .tex files
- **Expected**: Every \label has a corresponding \ref
- **Pass criteria**: All floats referenced

---

## 6. Manual Validation Checklist

### 6.1 Paper Content Review

- [ ] **Introduction**: Does it clearly state the problem, gap, and contributions?
- [ ] **Related work**: Is our positioning accurate? Are we fair to prior work?
- [ ] **Method**: Can a reader reproduce the algorithm from the paper alone?
- [ ] **Experiments**: Do results support all claims? Are baselines fairly compared?
- [ ] **Discussion**: Are limitations honestly stated? Safety claims not overstated?
- [ ] **Abstract**: Does it accurately summarize contributions and key results?

### 6.2 Data Integrity

- [ ] **Every number in every table**: Verify against raw experiment logs
- [ ] **Training curves**: Match logged data (check WandB timestamps)
- [ ] **Statistical tests**: Verify p-values using independent implementation
- [ ] **Confidence intervals**: Verify using independent bootstrap implementation
- [ ] **Sim-to-real gap**: Numbers match between simulation logs and ROS bag analysis

### 6.3 Figure Quality

- [ ] **Readable at conference print size**: Print at actual column width (3.5") and verify
- [ ] **Colorblind-friendly**: Check with colorblind simulator (e.g., Coblis)
- [ ] **Proper axis labels**: All axes have labels with units
- [ ] **Legend readable**: No overlapping text, sufficient font size
- [ ] **Consistent style**: Same fonts, colors, line styles across all figures
- [ ] **No aliasing artifacts**: Vector graphics (PDF) for plots, high-DPI for photos

### 6.4 Demonstration Videos

- [ ] **Clearly show claimed capabilities**: Safety (near-miss with CBF), NE behavior, pursuit/evasion
- [ ] **Annotations visible and accurate**: CBF margin, distance, time overlays
- [ ] **Audio quality** (if narrated): Clear, no background noise
- [ ] **Resolution**: 1080p minimum

### 6.5 External Review

- [ ] **Colleague 1**: Reads Paper 1 draft, provides feedback on clarity and claims
- [ ] **Colleague 2**: Reads Paper 2 draft, checks real-robot claims
- [ ] **Colleague 3** (optional): Reads Paper 3, checks proof correctness
- [ ] **Address all reviewer feedback** before submission

### 6.6 Open-Source Repository

- [ ] **Clone test**: Fresh clone + pip install + basic example works
- [ ] **Reproduce basic results**: Can an outsider reproduce Table 1 (at least approximately)?
- [ ] **Documentation complete**: README, API docs, tutorials all present and accurate
- [ ] **No sensitive data**: No credentials, API keys, server passwords in repo
- [ ] **License appropriate**: MIT or Apache 2.0

### 6.7 Ethical Review

- [ ] **Safety claims not overstated**: "0% violations in N episodes" not "perfectly safe forever"
- [ ] **Limitations clearly stated**: Arena size, obstacle count, 2D planar, specific hardware
- [ ] **Dual-use considerations**: PE technology could be misused; include responsible use statement
- [ ] **Reproducibility commitment**: Code and data will be released
- [ ] **Fair baseline comparison**: Each baseline given fair compute and hyperparameter tuning

### 6.8 Sign-Off Criteria Per Paper

**Paper 1 (Conference)**:
- [ ] All co-authors reviewed and approved
- [ ] All figures and tables finalized
- [ ] Supplementary video ready
- [ ] LaTeX compiles cleanly
- [ ] Page count within limit
- [ ] Submitted to venue portal

**Paper 2 (Journal)**:
- [ ] Real-robot results complete and verified
- [ ] Extended experiments beyond Paper 1
- [ ] Cover letter drafted
- [ ] Multimedia attachment prepared

**Paper 3 (Theory)**:
- [ ] All proofs verified by co-author with control theory expertise
- [ ] Numerical validation matches theoretical predictions
- [ ] Notation consistent with standard CBF literature

---

## 7. Success Criteria & Phase Gates

### 7.1 Phase Gate: Analysis Complete (End of Month 8)

| Criterion | Target | Measurement | Protocol |
|-----------|--------|-------------|----------|
| Formal proofs written | 5 theorems complete (4 + DCBF from Phase 3) | Count of complete proofs with numerical verification | Each theorem: statement + proof + verification script with >99.9% pass rate |
| Exploitability computed | < 0.10 | Final policy pair metric | 5 seeds for best-response training; 100 eval episodes; report mean ± 95% CI |
| All baselines run | 13/13 complete (incl. HJ/DeepReach) | Count of successful runs | Each baseline: 5 seeds × 100 eval episodes; same total_timesteps; no NaN metrics |
| All ablations run | 11/11 complete | Count of successful runs | Each variant: 5 seeds; control identical across all ablations |
| Statistical analysis | All comparisons done | Summary report complete | Welch's t-test + Holm-Bonferroni correction; Cohen's d for all pairs; 95% bootstrap CIs |

### 7.2 Phase Gate: Figures and Tables (End of Month 8.5)

| Criterion | Target | Measurement | Protocol |
|-----------|--------|-------------|----------|
| All figures generated | 9+ figures | Count of PDF files in `figures/` | Each at 300 DPI, readable at 3.5" width, colorblind-friendly palette |
| All tables generated | 4+ tables | Count of `.tex` files in `tables/` | Auto-populated from experiment data; every number traceable to raw logs |
| Demo videos edited | 5+ scenarios | Count of edited MP4s in `videos/edited/` | Each annotated with CBF margin, distance, time overlay |
| Publication quality | Pass review | Manual checklist (Section 6.3) | Print-test at column width; check with Coblis colorblind simulator |

### 7.3 Phase Gate: Papers Written (End of Month 9)

| Criterion | Target | Measurement | Protocol |
|-----------|--------|-------------|----------|
| Paper 1 draft | Complete | LaTeX compiles, all sections written | Zero LaTeX errors; all figures/tables referenced; page count ≤ 6+refs (ICRA) or ≤ 8+refs (CoRL) |
| Paper 2 draft | Complete | LaTeX compiles, real-robot sections done | Extends Paper 1 with sim-to-real + GP + real robot; 8-12 pages (RA-L) |
| Paper 3 draft | Complete | All proofs included | All 4 theorems with complete proofs (not just sketches); notation consistent |
| Internal review | Complete | 2+ colleagues reviewed | Collect written feedback; address all major concerns before submission |

### 7.4 Phase Gate: Submission Ready (End of Month 10)

| Criterion | Target | Measurement | Protocol |
|-----------|--------|-------------|----------|
| Conference paper submitted | ICRA/IROS/CoRL | Submission confirmation email/receipt | Verify supplementary video attached; author list confirmed |
| Journal paper drafted | RA-L/T-RO ready | Complete draft | All real-robot results included; cover letter drafted |
| Theory paper drafted | CDC/L4DC ready | Complete draft | Proofs verified by co-author with control theory background |
| All ablations with significance | p < 0.05 for key claims | Statistical report | Holm-Bonferroni corrected; claims that fail = report honestly |
| Demo videos produced | YouTube-ready | Published link | 1080p, 2-3 min supplementary; annotated with metrics |
| Open-source code released | GitHub public | Repository URL | Fresh clone + pip install works; pytest passes; Docker builds |

### 7.5 Key Metrics Summary

| Metric | Target | Hard/Soft | How to Measure |
|--------|--------|-----------|---------------|
| NE gap | < 0.10 | Hard | `abs(capture_rate - escape_rate)` over 100 eval episodes × 5 seeds |
| Safety violation rate | 0% | Hard | `any(h_i(x) < -1e-6)` per timestep across all experiments |
| CBF-QP feasibility rate | > 99% | Hard | `count(tier_used == 1) / total_steps` across 10K test states |
| Train-deploy gap | < 5% | Soft | `abs(capture_rate_train - capture_rate_deploy)` on same initial conditions |
| Sim-to-real gap | < 10% | Soft | `abs(metric_sim - metric_real) / metric_sim` for capture rate |
| Number of seeds (key results) | >= 5 | Hard | Count of distinct random seeds per experiment |
| Statistical significance (vs baselines) | p < 0.05 (corrected) | Hard | Welch's t-test + Holm-Bonferroni; Cohen's d reported |
| Paper 1 submitted | By month 10 | Hard | Submission portal receipt |
| Open-source release | By month 10 | Soft | Public GitHub URL with passing CI |

### 7.6 Definition of Done

> **Phase 5 is COMPLETE when:**
> 1. ALL four phase gates (7.1-7.4) are met with documented evidence
> 2. Paper 1 submitted to ICRA/IROS/CoRL and posted to arXiv
> 3. Paper 2 (journal) draft complete and ready for submission
> 4. Paper 3 (theory) draft complete with verified proofs
> 5. All 5 formal theorems (4 + DCBF from Phase 3) have complete proofs + numerical verification
> 6. All 13 baselines (incl. HJ/DeepReach) and 11 ablations run with >= 5 seeds, statistical analysis complete
> 7. Minimum test suite (Section 7.8) passes: 15+ tests, all green
> 8. Open-source repo: README, LICENSE, Docker, tutorials, CI pipeline passing
> 9. All demo videos (7+) edited, annotated, and published
> 10. Reproducibility validated: Docker image builds, fresh clone + pytest passes
> 11. Generalization study complete (9 test conditions, heatmap generated)
> 12. Interpretability analysis complete (saliency maps, phase portraits, belief evolution)
> 13. Human evader experiment complete (20 episodes per participant, comparison table)
> 14. Opponent modeling analysis complete (strategy classification, adaptation metrics)

### 7.7 Phase Gate Checklist

Before declaring Phase 5 complete, verify:
- [ ] Formal proofs: 5 theorems written (4 + DCBF), reviewed, verified numerically
- [ ] Exploitability < 0.10 for final policy pair (5 seeds, 95% CI)
- [ ] All 13 baselines run (incl. HJ/DeepReach, 5+ seeds each, same compute, same evaluation)
- [ ] All 11 ablations run (5+ seeds each, effect sizes computed)
- [ ] Statistical analysis: Holm-Bonferroni corrected comparisons, summary report
- [ ] 9+ publication figures generated (300 DPI, colorblind-friendly)
- [ ] 4+ LaTeX tables auto-populated from data
- [ ] Paper 1 submitted (confirmation received)
- [ ] Paper 2 and Paper 3 drafts complete
- [ ] Internal review by 2+ colleagues completed
- [ ] Demo videos edited and published
- [ ] Open-source repo: clean code, README, Docker, CI, tutorials
- [ ] All claims in papers verified against statistical analysis
- [ ] Minimum test suite passes (Section 7.8)
- [ ] Generalization study: 9 test conditions evaluated, heatmap figure generated
- [ ] Interpretability: saliency maps, phase portraits, belief evolution visualizations
- [ ] Human evader experiment: 20 episodes per participant, comparison table
- [ ] Opponent modeling: strategy classification, adaptation metrics

### 7.8 Minimum Test Suite (15+ Tests)

**File: `tests/test_statistics.py`** (5 tests)

```python
# Test A: Bootstrap CI contains known mean
def test_bootstrap_ci_coverage():
    """95% CI should contain true mean of N(5.0, 1.0)."""
    np.random.seed(42)
    data = np.random.normal(5.0, 1.0, size=100)
    lower, upper = bootstrap_ci(data, n_bootstrap=10000, ci=0.95)
    assert lower < 5.0 < upper

# Test B: Holm-Bonferroni correction
def test_holm_bonferroni_known_pvalues():
    """Known p-values: [0.001, 0.01, 0.04, 0.06] at alpha=0.05."""
    p_values = np.array([0.001, 0.01, 0.04, 0.06])
    result = holm_bonferroni(p_values, alpha=0.05)
    assert list(result) == [True, True, True, False]

# Test C: Cohen's d on known distributions
def test_cohens_d_known():
    """N(0,1) vs N(0.8,1) should give d ≈ 0.8."""
    np.random.seed(42)
    d1 = np.random.normal(0.0, 1.0, 1000)
    d2 = np.random.normal(0.8, 1.0, 1000)
    _, _, d = welch_t_test(d1, d2)
    assert abs(d - 0.8) < 0.15  # Within tolerance

# Test D: Welch's t-test significant for large effect
def test_welch_significant_large_effect():
    """Two clearly different distributions should be significant."""
    np.random.seed(42)
    d1 = np.random.normal(0.0, 1.0, 50)
    d2 = np.random.normal(2.0, 1.0, 50)
    _, p, _ = welch_t_test(d1, d2)
    assert p < 0.001

# Test E: Welch's t-test NOT significant for same distribution
def test_welch_not_significant_same():
    """Two samples from same distribution should usually not be significant."""
    np.random.seed(42)
    d1 = np.random.normal(0.0, 1.0, 50)
    d2 = np.random.normal(0.0, 1.0, 50)
    _, p, _ = welch_t_test(d1, d2)
    assert p > 0.05  # Not significant (same distribution)
```

**File: `tests/test_exploitability.py`** (4 tests)

```python
# Test F: Exploitability zero at known NE (RPS)
def test_exploitability_rps_ne():
    """Rock-Paper-Scissors at NE (1/3, 1/3, 1/3): exploit ≈ 0."""
    M = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])  # RPS payoff
    strategy = np.array([1/3, 1/3, 1/3])
    exploit = compute_matrix_exploitability(M, strategy, strategy)
    assert abs(exploit) < 1e-6

# Test G: Exploitability positive for non-NE
def test_exploitability_rps_non_ne():
    """Always-Rock in RPS: exploit > 0."""
    M = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    strategy = np.array([1.0, 0.0, 0.0])  # Always Rock
    exploit = compute_matrix_exploitability(M, strategy, strategy)
    assert exploit > 0.5

# Test H: PSRO payoff matrix is correct shape
def test_psro_payoff_matrix():
    """Payoff matrix M has shape (n_pursuers, n_evaders)."""
    policies_P = [RandomPolicy(), RandomPolicy(), RandomPolicy()]
    policies_E = [RandomPolicy(), RandomPolicy()]
    M = compute_payoff_matrix(env, policies_P, policies_E, n_episodes=10)
    assert M.shape == (3, 2)
    assert np.all(np.isfinite(M))

# Test I: Best-response improves over random
def test_best_response_improves():
    """Best-response against random should outperform random."""
    random_P = RandomPolicy()
    br_P = train_best_response(env, role='pursuer', opponent=random_P,
                                total_timesteps=10000)
    random_return = evaluate(env, random_P, random_P, n_episodes=20)
    br_return = evaluate(env, br_P, random_P, n_episodes=20)
    assert br_return > random_return
```

**File: `tests/test_figure_generation.py`** (3 tests)

```python
# Test J: Figure generation scripts run without errors
def test_all_figures_generate():
    """All figure generation functions execute without exception."""
    # Use synthetic data
    synthetic_results = generate_synthetic_results()
    for fig_name, fig_func in FIGURE_FUNCTIONS.items():
        fig = fig_func(synthetic_results)
        assert fig is not None
        fig.savefig(f"/tmp/test_{fig_name}.pdf")
        assert os.path.exists(f"/tmp/test_{fig_name}.pdf")
        plt.close(fig)

# Test K: LaTeX tables compile
def test_latex_tables_compile():
    """Generated LaTeX tables compile without errors."""
    synthetic_results = generate_synthetic_results()
    for table_name, table_func in TABLE_FUNCTIONS.items():
        tex_content = table_func(synthetic_results)
        assert '\\begin{table' in tex_content
        assert '\\end{table' in tex_content
        assert 'NaN' not in tex_content

# Test L: Reference consistency check
def test_reference_consistency():
    """All citations in .tex files have matching .bib entries."""
    tex_citations = extract_citations("papers/paper1_conference/main.tex")
    bib_entries = extract_bib_keys("papers/references.bib")
    missing = tex_citations - bib_entries
    assert len(missing) == 0, f"Missing bib entries: {missing}"
```

**File: `tests/test_claims_verification.py`** (3 tests)

```python
# Test M: All paper claims have statistical support
def test_claims_supported():
    """Every claim in the paper is supported by data."""
    results = load_experiment_results()
    claims = {
        'zero_safety_violations': results['safety_violation_rate'] == 0.0,
        'ne_gap_below_threshold': results['ne_gap'] < 0.10,
        'cbf_feasibility_high': results['cbf_feasibility_rate'] > 0.99,
    }
    for claim, supported in claims.items():
        assert supported, f"Claim '{claim}' not supported by data"

# Test N: Reproducibility from checkpoint
def test_reproducibility_from_checkpoint():
    """Re-evaluation from checkpoint matches original within 2%."""
    original = load_experiment_results()
    reproduced = run_evaluation_from_checkpoint("checkpoints/final.pt", seed=0)
    for metric in ['capture_rate', 'ne_gap']:
        rel_diff = abs(original[metric] - reproduced[metric]) / max(original[metric], 1e-8)
        assert rel_diff < 0.02, f"{metric} diff: {rel_diff:.4f}"

# Test O: Docker builds and pytest passes
def test_docker_builds():
    """Docker image builds successfully (smoke test)."""
    result = subprocess.run(
        ["docker", "build", "-t", "safpe-test", "-f", "docker/Dockerfile.train", "."],
        capture_output=True, timeout=600
    )
    assert result.returncode == 0, f"Docker build failed: {result.stderr.decode()}"
```

### 7.9 Worked Examples

#### Example 1: Bootstrap CI Computation

```
Scenario: Compute 95% CI for SafePE capture rate across 5 seeds

Raw data (capture rates per seed):
  Seed 0: 0.52
  Seed 1: 0.48
  Seed 2: 0.55
  Seed 3: 0.50
  Seed 4: 0.51

Step 1: Compute sample statistics
  Mean = (0.52 + 0.48 + 0.55 + 0.50 + 0.51) / 5 = 0.512
  Std  = 0.0259

Step 2: Bootstrap (n_bootstrap = 10000)
  For each iteration:
    Sample 5 values with replacement from [0.52, 0.48, 0.55, 0.50, 0.51]
    Compute mean of sample
  Result: 10000 bootstrap means

Step 3: Extract CI
  2.5th percentile: 0.486
  97.5th percentile: 0.536
  95% CI: [0.486, 0.536]

Step 4: Report
  "SafePE achieves a capture rate of 51.2% (95% CI: [48.6%, 53.6%])"

Step 5: Compare with baseline (e.g., PPO+SP no CBF)
  Baseline mean: 0.47, 95% CI: [0.44, 0.50]
  Welch's t-test: t=2.86, p=0.023
  After Holm-Bonferroni (13 baselines): p_adj = 0.023 * 13 = 0.299
  → NOT significant after correction
  Cohen's d = (0.512 - 0.47) / pooled_std = 1.62 (large effect)
  Report: "large effect size (d=1.62) but not significant after
  correction for 12 comparisons (p_adj = 0.276, n=5 seeds)"
```

#### Example 2: Exploitability Computation

```
Scenario: Compute exploitability for final SafePE policy pair

Step 1: Evaluate current profile (pi_P, pi_E)
  100 episodes, seed 0:
    Capture rate (pursuer reward proxy): J_P = 0.52
    Escape rate (evader reward proxy):   J_E = 0.48

Step 2: Train best-response pursuer against frozen pi_E
  500K timesteps, 5 seeds
  Best seed: J_P(BR_P, pi_E) = 0.58
  Improvement: 0.58 - 0.52 = 0.06

Step 3: Train best-response evader against frozen pi_P
  500K timesteps, 5 seeds
  Best seed: J_E(pi_P, BR_E) = 0.55
  Improvement: 0.55 - 0.48 = 0.07

Step 4: Compute exploitability
  exploit = (J_P(BR_P, pi_E) - J_P(pi_P, pi_E)) +
            (J_E(pi_P, BR_E) - J_E(pi_P, pi_E))
         = 0.06 + 0.07 = 0.13

  But wait — this is > 0.10 threshold!

Step 5: Investigation
  The exploitability is borderline. Options:
  a) Run more self-play phases (increase from 6 to 8)
  b) Increase timesteps per phase (1M → 2M)
  c) Report honestly: "exploit = 0.13 ± 0.03, near but above 0.10"

After 2 more self-play phases:
  exploit = 0.08 < 0.10 ✓ (CONVERGED)
```

#### Example 3: Figure Generation Workflow

```
Scenario: Generate Figure 2 (Training Curves) for Paper 1

Step 1: Load raw data from W&B
  wandb_api = wandb.Api()
  runs = wandb_api.runs("project/safpe-training")
  # 5 seeds × 2M timesteps each

Step 2: Extract per-step metrics
  For each seed:
    capture_rate[seed] = array of shape (2000,)  # logged every 1K steps
    safety_violations[seed] = array of shape (2000,)
    ne_gap[seed] = array of shape (2000,)
    cbf_intervention[seed] = array of shape (2000,)

Step 3: Apply EMA smoothing
  For each metric, each seed:
    smoothed = ema(raw, alpha=0.99)

Step 4: Plot
  fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.0))

  For each subplot:
    - Plot individual seeds: thin lines, alpha=0.2, color='#0072B2'
    - Plot mean: thick line, alpha=1.0, color='#0072B2'
    - X-axis: "Environment Steps (M)" with ticks [0, 0.5, 1.0, 1.5, 2.0]
    - Y-axis: metric name with unit
    - Title: "(a) Capture Rate", etc.

Step 5: Save
  fig.savefig("figures/training_curves.pdf", dpi=300, bbox_inches='tight')
  fig.savefig("figures/training_curves.png", dpi=300, bbox_inches='tight')
  File size: ~150KB (PDF), ~800KB (PNG)

Step 6: Verify
  - Open PDF, zoom to 3.5" width: all text readable ✓
  - Run through Coblis colorblind simulator: distinguishable ✓
  - All axes labeled with units ✓
  - Legend in (a), not repeated in (b)-(d) ✓
```

---

## 8. Troubleshooting Guide

### 8.1 Baselines Underperforming (Unfair Comparison Risk)

**Symptom**: A baseline performs much worse than reported in its original paper.

**Diagnosis**:
1. Check hyperparameters match the original paper
2. Verify environment setup is compatible (observation space, reward scale)
3. Run baseline on its original benchmark to verify your implementation works

**Solutions**:
- Use official code repositories where available (MACPO: github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation)
- Perform hyperparameter search for the baseline (at least learning rate and network size)
- Contact original paper authors if reproduction fails
- If baseline still underperforms, report the best result you can achieve and note the discrepancy in the paper
- **Never report a strawman baseline** — reviewers will catch this

### 8.2 Statistical Significance Not Achieved

**Symptom**: Key comparisons have p > 0.05 after Holm-Bonferroni correction.

**Diagnosis**:
1. Check the number of seeds — may need more
2. Check the effect size — if small, may need many more seeds
3. Check variance — high variance reduces power

**Solutions**:
- **Add more seeds**: Increase from 5 to 10 (or even 20 for borderline cases)
- **Power analysis**: Compute required n for observed effect size at power=0.80
- **Use paired tests**: If same initial conditions across methods, use paired t-test (more powerful)
- **Report effect sizes**: Even if not significant, a large Cohen's d is informative
- **Be honest**: If a difference is not significant, say so. "SafePE achieves X +/- Y, which is higher than baseline Z +/- W, though the difference is not statistically significant (p=0.08)"
- **Consider practical significance**: 0% safety violations vs 5% may be practically important even if n is small

### 8.3 Paper Too Long

**Symptom**: Conference paper exceeds 6-page limit.

**Cutting priority** (what to move to supplementary):
1. Extended proof details (move to supplementary, keep sketch in main)
2. Additional ablation results (keep top 3-4 most impactful, rest in supplementary)
3. Hyperparameter sensitivity (move to supplementary)
4. Extended related work (trim to essential comparisons)
5. Implementation details (move to supplementary or appendix)
6. **Never cut**: Main results table, key figures, problem formulation, algorithm description

### 8.4 Reviewer Response Strategies

**Common reviewer concerns and responses**:

**"The safety guarantee is only for the CBF, not end-to-end"**:
- Response: Correct. We guarantee forward invariance of the safe set under the CBF-QP filter. End-to-end safety depends on correct state estimation and model fidelity. We address model uncertainty via GP-RCBF (Theorem 3). We clearly state this limitation in Section VIII.

**"The NE analysis is only approximate"**:
- Response: True NE computation is intractable for continuous PE games. We follow the standard practice of computing exploitability against best-response opponents, consistent with [18] and the self-play literature [04]. Our exploitability < 0.10 indicates near-NE behavior.

**"More baselines needed"**:
- Response: We compare against 13 baselines spanning DRL-only, safe RL, self-play, classical, and HJ game-theoretic optimal methods. This is the most comprehensive comparison in the PE literature. We are happy to add specific baselines the reviewer suggests.

**"Real-robot experiments are limited (few trials)"**:
- Response: We run N real-robot trials (more than [02] with M trials and [22] with K trials). Real-robot experiments are inherently expensive. We supplement with extensive simulation results (5-10 seeds x 100 episodes).

**"The sim-to-real gap is larger than claimed"**:
- Response: We report the gap honestly in Table 3. The gap is within 10% for all primary metrics, consistent with state-of-the-art sim-to-real results [N10] (80-100% success). We use domain randomization [07] and GP online adaptation [06] to minimize the gap.

### 8.5 Computational Budget Issues

**Symptom**: Not enough compute to run all baselines x all seeds.

**Solutions**:
- **Prioritize**: Run key baselines (MADDPG, PPO+SP no CBF, MADR, MACPO) with full seeds; run secondary baselines (Random, Greedy, DQN) with fewer seeds
- **Reduce timesteps**: For baselines that clearly converge early, use early stopping
- **Parallelize**: Use multiple GPUs or cloud instances
- **Use checkpoints**: Resume from intermediate checkpoints if runs fail

### 8.6 Ablation Shows Unexpected Results

**Symptom**: Removing a component improves performance (e.g., no CBF is better than CBF).

**Diagnosis**:
1. CBF may be too conservative (alpha too large, safety margin too wide)
2. CBF intervention may interfere with learning signal
3. Check if the improvement is statistically significant

**Solutions**:
- Report honestly — negative results are valuable
- Investigate the cause (e.g., plot CBF intervention rate)
- Check if the improvement holds across all metrics (e.g., better capture rate but more violations)
- Tune the CBF parameters and re-run
- Discuss in the paper: "Removing CBF improves capture rate by X% but introduces Y% safety violations, demonstrating the safety-performance trade-off"

### 8.7 LaTeX Issues

**Common problems**:
- **Overfull hbox**: Reduce figure width, use `\resizebox`, or rewrite text
- **Missing citations**: Run `bibtex` before `pdflatex`
- **Float placement**: Use `[t]` or `[h!]` placement, or `\FloatBarrier`
- **Table too wide**: Use `\resizebox{\columnwidth}{!}{...}` or switch to `table*`

---

## 9. Publication Roadmap

### 9.1 Submission Timeline

```
Month 8:
  Week 1-2: Sessions 1-3 (formal analysis, NE analysis)
  Week 3-4: Sessions 4-4b, 5-6 (benchmarking incl. HJ/DeepReach, ablations, statistics)

Month 9:
  Week 1: Sessions 7, 15 (figures, generalization study)
  Week 2: Sessions 8, 16-17 (tables, interpretability, opponent modeling)
  Week 3-4: Sessions 9-10, 18 (Paper 1 writing, videos, human evader)

Month 10:
  Week 1: Session 11 (Paper 2 writing — includes new results sections)
  Week 2: Session 12 (Paper 3 writing — includes DCBF theorem)
  Week 3: Sessions 13-14 (open-source, supplementary)
  Week 4: Final review, submission
```

### 9.2 Target Venues and Deadlines

| Paper | Primary Venue | Typical Deadline | Backup Venue |
|-------|--------------|-----------------|-------------|
| Paper 1 | ICRA 2027 | Sep 2026 | IROS 2027 (Mar 2027) |
| Paper 1 (alt) | CoRL 2026 | Jun 2026 | ICRA 2027 |
| Paper 2 | RA-L | Rolling (anytime) | T-RO (rolling) |
| Paper 3 | CDC 2027 | Mar 2027 | L4DC 2027 (Feb 2027) |

**Note**: RA-L papers accepted before ICRA/IROS deadlines can be presented at those conferences.

### 9.3 Revision Strategy

**If Paper 1 is rejected**:
1. Read all reviews carefully; identify major vs minor concerns
2. Address major concerns first (additional experiments, clarifications)
3. Revise and submit to backup venue within 2-4 weeks
4. Incorporate reviewer feedback into Paper 2 (journal) as well

**If Paper 1 receives "revise and resubmit"** (RA-L):
1. Address all reviewer comments systematically
2. Write a detailed response letter (point-by-point)
3. Highlight changes in the manuscript (use color or change tracking)
4. Resubmit within the allowed revision period (typically 3 months)

### 9.4 Backup Venues

| Tier | Venues | Notes |
|------|--------|-------|
| Tier 1 (top) | ICRA, IROS, CoRL, RA-L, T-RO, CDC | Primary targets |
| Tier 2 (strong) | L4DC, RSS, WAFR, ACC | Theory or methods focus |
| Tier 3 (specialized) | MRS, AAMAS, AAAI (safety track) | Multi-agent or AI focus |
| Workshop | ICRA/IROS/NeurIPS workshops | For early-stage results |

### 9.5 Extended Publication Plan

```
Month 10:  Paper 1 submitted to ICRA or CoRL
Month 11:  Paper 2 submitted to RA-L
Month 12:  Paper 3 submitted to CDC or L4DC
Month 13:  Address Paper 1 reviews (if received)
Month 14:  Address Paper 2 reviews
Month 15:  Camera-ready versions
Month 16+: Present at conferences, follow-up work
```

### 9.6 ArXiv Strategy

- **Paper 1**: Post to arXiv simultaneously with conference submission (establishes priority)
- **Paper 2**: Post to arXiv after journal submission (check journal policy on preprints)
- **Paper 3**: Post to arXiv before conference deadline (common in controls community)
- **Use consistent naming**: `SafePE: ...` in all titles for brand recognition

---

## 10. Software Versions & Reproducibility

### 10.1 Pinned Package Versions

```bash
# Core (inherited from Phases 1-4)
stable-baselines3==2.6.0
gymnasium==1.1.1
torch==2.6.0
numpy==2.2.2

# Statistical analysis
scipy==1.15.0              # Welch's t-test, linprog for NE
scikit-learn==1.6.1        # Trajectory clustering (k-means)

# Plotting
matplotlib==3.10.0
seaborn==0.13.2

# Game theory
nashpy==0.0.41             # Nash equilibrium solver for PSRO

# LaTeX table generation
pandas==2.2.3              # DataFrames for table generation
jinja2==3.1.5              # LaTeX template filling

# Experiment management
wandb==0.19.6              # Experiment logging and data retrieval
hydra-core==1.3.2          # Configuration management

# Video processing & demo generation
pygame==2.6.1              # PERenderer for supplementary video generation (CBF/FOV overlays)
opencv-python==4.11.0.86   # Video annotation
ffmpeg-python==0.2.0       # Video editing / stitching

# Open-source release
black==25.1.0              # Code formatting
ruff==0.9.6                # Linting
pytest==8.3.4              # Testing
docker (system)            # Containerization
```

**Compatibility notes**:
- nashpy requires scipy; our scipy 1.15.0 is compatible
- matplotlib 3.10+ required for `text.usetex` with modern LaTeX
- wandb 0.19+ required for API v2 access to historical runs
- opencv-python may conflict with headless systems; use `opencv-python-headless` if needed

### 10.2 Reproducibility Protocol

1. **Random seeds**: All experiments use seeds `[0, 1, 2, 3, 4]` (minimum 5); key results extended to `[0..9]`
2. **Bootstrap seeds**: Fix bootstrap RNG with `np.random.seed(42)` for CI computation reproducibility
3. **Baseline fairness**: Each baseline gets identical `total_timesteps`, identical evaluation protocol (100 episodes × 5 seeds), identical initial conditions
4. **Ablation control**: The full SafePE control condition is identical across all 11 ablation studies (same checkpoint, same evaluation seed)
5. **Figure determinism**: All figures generated from saved experiment data (not live runs); `generate_figures.py` produces identical output on re-run
6. **Table traceability**: Every number in every table has a corresponding entry in `experiments/` JSON files; `generate_tables.py` auto-populates from these
7. **Docker reproducibility**: `docker/Dockerfile.train` and `docker/Dockerfile.deploy` pin all system dependencies
8. **Git tagging**: Each paper submission tagged with git tag (e.g., `paper1-icra-v1`); W&B runs tagged with commit hash
9. **Data archival**: All experiment results archived to persistent storage (not just W&B); raw CSV + JSON backups in `experiments/`
10. **LaTeX reproducibility**: BibTeX files committed; `latexmk` build script ensures reproducible compilation

---

## Appendix A: LaTeX Templates

### A.1 IEEE Conference Template (Paper 1)

```latex
\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage{subcaption}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}
\title{SafePE: Safety-Guaranteed Deep Reinforcement Learning for 1v1 Pursuit-Evasion on Ground Robots}
\author{
\IEEEauthorblockN{Author Names}
\IEEEauthorblockA{Affiliations}
}
\maketitle
\begin{abstract}
Your abstract here.
\end{abstract}
\begin{IEEEkeywords}
pursuit-evasion, safe reinforcement learning, control barrier functions, self-play, ground robots
\end{IEEEkeywords}
% Sections here
\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
```

### A.2 Figure Specification Templates

**Training Curve Figure**:
```python
fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.0))
# (a) Capture rate: ax[0,0], x=env_steps(M), y=capture_rate(%)
# (b) Safety violations: ax[0,1], x=env_steps(M), y=violation_rate(%)
# (c) NE gap: ax[1,0], x=env_steps(M), y=|capture_rate - 0.5|
# (d) CBF intervention: ax[1,1], x=env_steps(M), y=intervention_rate(%)
# Style: mean=thick line, seeds=thin transparent, EMA smoothing
```

**Trajectory Plot Figure**:
```python
fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 4.5))
# Each subplot: arena boundary (black), obstacles (gray fill),
# pursuer path (blue, arrow), evader path (red, arrow),
# start (circle), end (square), capture (star),
# safety margins (dashed gray circles)
# Title: scenario description
```

### A.3 Algorithm Pseudocode Template

```latex
\begin{algorithm}[t]
\caption{SafePE: CBF-Constrained Self-Play}
\label{alg:safepe}
\begin{algorithmic}[1]
\REQUIRE Environment $\mathcal{E}$, CBFs $\{h_i\}$, AMS-DRL phases $K$
\STATE Initialize policies $\pi_P$, $\pi_E$ randomly
\STATE $\pi_E \gets$ \textsc{ColdStart}($\mathcal{E}$) \COMMENT{Phase S0}
\FOR{$k = 1, 2, \ldots, K$}
    \IF{$k$ is odd}
        \STATE Freeze $\pi_E$
        \FOR{episode $= 1, \ldots, N$}
            \STATE Collect trajectory with CBF-Beta: $a_P \sim \pi_P^C(\cdot|o_P)$
            \STATE Update $\pi_P$ via PPO with truncated gradient
        \ENDFOR
    \ELSE
        \STATE Freeze $\pi_P$, train $\pi_E$ similarly
    \ENDIF
    \IF{$|\text{SR}_P - \text{SR}_E| < \eta$} \textbf{break} \ENDIF
\ENDFOR
\RETURN $\pi_P$, $\pi_E$
\end{algorithmic}
\end{algorithm}
```

---

## Appendix B: BibTeX Entries for Key References

```bibtex
@inproceedings{suttle2024cbf_beta,
  title={Sampling-Based Safe Reinforcement Learning with Control Barrier Functions},
  author={Suttle, Wesley and others},
  booktitle={AISTATS},
  year={2024},
  note={[16]}
}

@article{zhang2025vcp_cbf,
  title={Dynamic Obstacle Avoidance for Car-like Mobile Robots based on Neurodynamic Optimization with Control Barrier Functions},
  author={Zhang, Zheng and Yang, Guang-Hong},
  journal={Neurocomputing},
  volume={654},
  year={2025},
  note={[N12]}
}

@article{emam2022rcbf_gp,
  title={Safe Reinforcement Learning Using Robust Control Barrier Functions},
  author={Emam, Yousef and others},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  note={[06]}
}

@article{xiao2024ams_drl,
  title={AMS-DRL: Multi-Agent Self-Play with Asynchronous Staged Training},
  author={Xiao, Yue and Feroskhan, Mir},
  year={2024},
  note={[18]}
}

@article{gonultas2024pe_sensors,
  title={Pursuit-Evasion on Car-Like Robots with Sensor Constraints},
  author={Gonultas, Burak and Isler, Volkan},
  year={2024},
  note={[02]}
}

@article{gu2023macpo,
  title={Multi-Agent Constrained Policy Optimisation},
  author={Gu, Shangding and others},
  year={2023},
  note={[N03]}
}

@article{yang2025rl_pe_survey,
  title={Reinforcement Learning Approaches for Pursuit-Evasion Games: A Review},
  author={Yang, J. and others},
  journal={Chinese Journal of Aeronautics},
  year={2025},
  note={[37]}
}

@article{ganai2024hj_rl_survey,
  title={Hamilton-Jacobi Reachability and Reinforcement Learning: A Survey},
  author={Ganai, Milan and others},
  year={2024},
  note={[25]}
}

@article{liu2024safe_rl_survey,
  title={Safe Reinforcement Learning and Constrained MDPs: A Technical Survey},
  author={Kushwaha and others},
  year={2024},
  note={[N09]}
}

@article{liu2025rmarl_cbf_sam,
  title={Safe Robust Multi-Agent RL with Neural CBFs and Safety Attention},
  author={Liu, Shihan and Liu, Lijun and Yu, Zhen},
  journal={Information Sciences},
  year={2025},
  note={[N15]}
}
```

---

## Appendix C: Metrics Collection Reference

### C.1 Primary Metrics Computation

```python
def compute_all_metrics(env, pi_P, pi_E, n_episodes=100, seeds=range(5)):
    """Compute all primary and secondary metrics."""
    all_results = []
    for seed in seeds:
        env.seed(seed)
        episode_results = []
        for ep in range(n_episodes):
            obs_P, obs_E = env.reset()
            done = False
            ep_data = {'min_cbf': float('inf'), 'cbf_interventions': 0,
                       'total_steps': 0, 'captured': False, 'violated': False}
            while not done:
                # Get RL action
                a_P_rl = pi_P.predict(obs_P)
                a_E_rl = pi_E.predict(obs_E)

                # Apply CBF filter
                a_P_safe, intervened_P = cbf_filter(a_P_rl, env.state)
                a_E_safe, intervened_E = cbf_filter(a_E_rl, env.state)

                # Track metrics
                ep_data['total_steps'] += 1
                ep_data['cbf_interventions'] += int(intervened_P or intervened_E)
                min_h = min(h_i(env.state) for h_i in cbf_list)
                ep_data['min_cbf'] = min(ep_data['min_cbf'], min_h)
                ep_data['violated'] = ep_data['violated'] or (min_h < 0)

                # Step environment
                obs_P, obs_E, reward, done, info = env.step(a_P_safe, a_E_safe)
                ep_data['captured'] = info.get('captured', False)

            episode_results.append(ep_data)
        all_results.append(episode_results)

    # Aggregate
    metrics = {
        'capture_rate': mean([ep['captured'] for seed_eps in all_results for ep in seed_eps]),
        'safety_violation_rate': mean([ep['violated'] for seed_eps in all_results for ep in seed_eps]),
        'cbf_intervention_rate': mean([ep['cbf_interventions']/ep['total_steps']
                                       for seed_eps in all_results for ep in seed_eps]),
        # ... etc
    }
    return metrics
```

### C.2 Expected Metric Ranges

| Metric | Expected Range | Red Flag |
|--------|---------------|----------|
| Capture rate | 40-60% (at NE) | < 20% or > 80% (not NE) |
| Safety violations | 0% | Any violation |
| NE gap | 0.00 - 0.10 | > 0.15 (not converged) |
| CBF feasibility | 99-100% | < 98% (infeasibility problem) |
| CBF intervention rate | 5-20% | > 40% (too conservative) or 0% (CBF not active) |
| Capture time | 10-40s | > 55s (near timeout) |
| Sim-to-real gap | 0-10% | > 15% (transfer problem) |

---

*End of Phase 5 Document*
