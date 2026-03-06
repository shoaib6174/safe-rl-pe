# Search & Exploration Rewards in RL Literature

## Compiled: 2026-03-04 (Session 69)

This document surveys exact reward formulations used to incentivize search and exploration behavior in reinforcement learning, with focus on settings where an agent cannot always see its target.

---

## 1. Pursuit-Evasion Games Under Partial Observability

### 1.1 R2PS: Worst-Case Robust Real-Time Pursuit Strategies (2025)
- **Paper**: "R2PS: Worst-Case Robust Real-Time Pursuit Strategies under Partial Observability"
- **Source**: arXiv:2511.17367
- **Approach**: Binary sparse reward + belief state representation
- **Reward**:
  ```
  r(s, a, b) = +1  if capture (f(s) = 1)
  r(s, a, b) =  0  otherwise
  ```
- **Key insight**: NO exploration reward shaping. Instead, the belief state is embedded into the RL state representation. The agent tracks a belief distribution over possible evader positions:
  ```
  belief_new(s_e) = Sum_v nu(v, s_e) * belief_old(v)   if s_e in Pos
                  = 0                                     otherwise
  ```
  When evader not observed, the position set shrinks: `Pos_new = Remove(Neighbor(Pos))`. The pursuer learns to search implicitly through the belief dynamics — visiting locations eliminates them from the belief.
- **Algorithm**: SAC with GNN-based policy

### 1.2 Multi-UAV Pursuit-Evasion with Online Planning (2024)
- **Paper**: "Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep Reinforcement Learning"
- **Authors**: Jiayu Chen et al.
- **Source**: arXiv:2409.15866
- **Reward** (4 components):
  ```
  r_capture   = +2              (bonus when evader enters capture radius)
  r_distance  = -0.1 * d(i,e)  (proportional to distance to evader)
  r_collision = -10             (penalty for hitting obstacles)
  r_smooth    = exp(-||a_t - a_{t-1}||)   (smoothness bonus)
  ```
- **Key insight**: Distance reward provides a continuous gradient toward the evader. No explicit search reward — uses an evader prediction-enhanced network with attention to handle partial observability.

### 1.3 Emergent Behaviors in Multi-Agent Pursuit-Evasion on 2D Grid (2025)
- **Paper**: "Emergent behaviors in multiagent pursuit evasion games within a bounded 2D grid world"
- **Source**: Scientific Reports, 2025
- **Reward**:
  ```
  R_total = R_process + R_outcome

  Pursuer process:  R_process^P(t) = -c_t    (time penalty per step)
  Evader process:   R_process^E(t) = +c_t    (time bonus per step)

  On capture:  R_outcome^P = +c_r,  R_outcome^E = -c_r
  On escape:   R_outcome^P = -c_r,  R_outcome^E = +c_r
  ```
- **Key insight**: "Distance is treated purely as an outcome-based reward term and is NOT included in the guiding rewards." The time penalty alone incentivizes the pursuer to search efficiently — every wasted timestep costs reward.

### 1.4 Deep Decentralized Multi-task Multi-Agent RL (Omidshafiei et al., 2017)
- **Paper**: "Deep Decentralized Multi-task Multi-Agent Reinforcement Learning under Partial Observability"
- **Source**: ICML 2017 (PMLR 70:2681-2690)
- **Reward**:
  ```
  r = +1  (terminal, only on simultaneous capture of all targets)
  r =  0  (all other timesteps)
  ```
- **Key insight**: Pure sparse reward in a partially observable grid world. Agents observe their own position always but targets' positions are randomly obscured with probability P_f (observation flickering). No exploration bonus — agents learn search through the DRQN (Deep Recurrent Q-Network) architecture with experience replay.

### 1.5 Learning Information Trade-offs in Pursuit-Evasion Games (2025)
- **Paper**: "Strategic Communication under Threat: Learning Information Trade-offs in Pursuit-Evasion Games"
- **Authors**: La Gatta et al.
- **Source**: arXiv:2510.07813
- **Approach**: Information-theoretic framework where the evader must trade off between revealing information (to communicate) and hiding. The pursuer's reward is implicitly tied to information gained about the evader's position.

---

## 2. Multi-Agent Target Search & Rescue

### 2.1 Multi-UAV Escape Target Search (2024)
- **Paper**: "Multi-UAV Escape Target Search: A Multi-Agent Reinforcement Learning Method"
- **Source**: Sensors, 2024, 24(21), 6859
- **Reward** (3-component weighted sum):
  ```
  R(t) = w1 * J_T(t) + w2 * J_E(t) + w3 * J_C(t)
  ```
  **Target Search Reward**:
  ```
  J_T(t) = Sum_{(x,y) in T1} beta1(x,y,t) * r_t1  +  Sum_{(x,y) in T2} beta2(x,y,t) * r_t2
  ```
  where T1 = initial discovery locations, T2 = rediscovery locations, beta = 1 if target probability exceeds threshold epsilon, r_t1/r_t2 are reward factors for first vs subsequent discovery.

  **Environment Exploration Reward (ENTROPY REDUCTION)**:
  ```
  J_E(t) = Sum_x Sum_y [u(x,y,t) - u(x,y,t+1)]
  ```
  where u(x,y,t) is the Shannon entropy (uncertainty) of grid cell (x,y) at time t. **Rewards the REDUCTION in uncertainty** across the mission area. Positive reward when entropy decreases (i.e., when cells are visited/observed).

  **Collision Prevention**:
  ```
  J_C(t) = r_c * Sum_i Sum_k 1[||B_k - u_i,t|| <= d_safe]
  ```
- **Key insight**: The exploration reward J_E directly rewards information gain (entropy reduction) per cell. This is the most directly applicable formulation to our problem.

### 2.2 Cooperative Search Method for Multiple UAVs (2022)
- **Paper**: "Cooperative Search Method for Multiple UAVs Based on Deep Reinforcement Learning"
- **Source**: PMC (Sensors journal)
- **Reward** (3 components):
  ```
  R_total = Sum_{i=1}^{3} R_i * k_i

  R1 (Shortest path): 40 if reached destination; else 100 * d_t * 10^(-y)
  R2 (Collision):     -exp(-100*d_u)  if d_u <= 2*sqrt(2)*d_g;  -1 if d_u=0
  R3 (Exploration):   R_OR  if cell NOT previously searched (rho=0);  0 if already searched (rho=1)
  ```
- **Key insight**: R3 gives a bonus for visiting unsearched cells. Simple binary: searched vs unsearched. The k_i weights allow different UAVs to prioritize exploration vs other objectives.

### 2.3 Multi-Target Radar Search and Track (2025)
- **Paper**: "Multi-Target Radar Search and Track Using Sequence-Capable Deep Reinforcement Learning"
- **Source**: arXiv:2502.13584
- **Reward** (2 components combined):
  ```
  r_t = r_{SV,t} + r_{TL,t}

  Search reward:   r_{SV,t} = -SSV([psi, theta], t)
  Track reward:    r_{TL,t} = Sum { ||P_t||_fro - ||P_{t-1}||_fro | D_t U D_{t-1} }
  ```
  where SSV = Scaled Scan Value (cumulative scan history normalized by max), P = covariance matrix (Frobenius norm measures uncertainty), D_t = detections.
- **Key insight**: Search reward is CONTINUOUS — negating the cumulative scan value means the agent is incentivized to scan areas with LOWER historical coverage. Naturally encourages exploring unscanned regions. Tracking reward activates only when detections occur. This asymmetry is highly relevant to our problem.

---

## 3. Hide-and-Seek RL Environments

### 3.1 OpenAI Hide-and-Seek (Baker et al., 2020)
- **Paper**: "Emergent Tool Use From Multi-Agent Autocurricula"
- **Authors**: Bowen Baker et al., OpenAI
- **Source**: ICLR 2020 (arXiv:1909.07528)
- **Reward**:
  ```
  Hiders:   r = +1  if ALL hiders are hidden (not visible to any seeker)
            r = -1  if ANY hider is visible to a seeker

  Seekers:  r = +1  if ANY hider is visible
            r = -1  if ALL hiders are hidden
  ```
- **Key insight**: Pure visibility-based reward, applied EVERY timestep. No distance reward, no exploration bonus. The seeker is incentivized to FIND hiders (make them visible), but the reward gives no gradient for HOW to search. The competitive self-play autocurriculum drives emergent search strategies through 6 phases of adaptation/counter-adaptation. This is the purest form of "let competition drive exploration."

### 3.2 Visual Hide and Seek (Chen & Song, 2019)
- **Paper**: "Visual Hide and Seek"
- **Authors**: Boyuan Chen, Shuran Song (Columbia University)
- **Source**: arXiv:1910.07882
- **Reward** (for the HIDER only — seeker is scripted):
  ```
  Hider sparse:  +0.001 per timestep survived
                 -1     on capture (collision with seeker)

  Hider dense (variant):  +0.001 when seeker CANNOT see hider
                          -0.001 when seeker CAN see hider
  ```
  **Seeker behavior** (hand-coded, not learned):
  ```
  if seeker sees hider -> move toward hider
  if seeker lost hider -> move toward LAST KNOWN POSITION
  if still searching   -> explore via WAYPOINTS in round-robin
  ```
- **Key insight**: The scripted seeker uses a 3-phase strategy: pursue if visible, go to last-known position if lost, then round-robin waypoints. This is a simple but effective search heuristic that could inspire a reward-based equivalent.

### 3.3 Replication of OpenAI Hide-and-Seek (2023)
- **Paper**: "Replication of Multi-Agent Reinforcement Learning for the Hide and Seek Problem"
- **Source**: arXiv:2310.05430
- **Reward** (modified from OpenAI):
  ```
  Hiders:   +0.001 per frame hidden
  Seekers:  +0.001 if hiders in field of vision; +1 if they tag (crash into) hiders
  ```
- **Key insight**: "Constant negative reward damages Agent's navigational phase" — they deliberately avoided penalizing seekers for NOT finding hiders, as it degraded pathfinding behavior. Only positive rewards for successful search.

---

## 4. Coverage / Patrol Rewards in Robotics

### 4.1 Cooperative Patrol Routing for Crime Surveillance (2025)
- **Paper**: "Cooperative Patrol Routing: Optimizing Urban Crime Surveillance through Multi-Agent Reinforcement Learning"
- **Source**: arXiv:2501.08020
- **Reward** (multi-component with exploration bonus):
  ```
  R_t(a_i) = R'_t(a_i) + Sum_{j=1}^{N} R'_t(a_j)   (cooperative: includes peers' rewards)

  For nodes within area:
    R'_t(a_i) = sigma(rho_t(a_i)) / eta[upsilon(rho_t(a_i))] + tau_t(rho_t(a_i))

  Exploration bonus tau_t:
    tau_t = alpha+  if visit_count == 1 AND target_value >= threshold  (first visit to high-value)
    tau_t = alpha-  if visit_count == 1 AND target_value < threshold   (first visit to low-value)
    tau_t = 0       if visit_count > 1                                 (already visited)

  Parameters: alpha+ = 50-100, alpha- = 5-10, threshold = 10
  Outside area: R'_t = nu (penalty, -25 or -10)
  ```
- **Key insight**: Visit count normalization (eta[upsilon]) reduces reward for repeatedly visiting the same node. First-visit exploration bonus (tau) explicitly rewards discovering new areas. The cooperative reward structure prevents "lazy agents."

### 4.2 Multi-Robot Patrolling Idleness (Santana et al., 2004; Portugal & Rocha survey)
- **Paper**: "Multi-agent patrolling with reinforcement learning" / "A Survey on Multi-robot Patrolling Algorithms"
- **Source**: AAMAS 2004 / Autonomous Agents and Multi-Agent Systems
- **Idleness definition**:
  ```
  Idleness(v, t) = t - t_last_visit(v)
  ```
  where t_last_visit(v) is the last time vertex v was visited by any agent.
- **Performance metrics**:
  ```
  Average idleness:     I_avg = (1/|V|) * Sum_v Idleness(v, t)
  Worst-case idleness:  I_max = max_v Idleness(v, t)
  Weighted idleness:    I_w(v, t) = w(v) * Idleness(v, t)
  ```
- **RL reward**: Agents receive rewards proportional to reducing idleness at visited nodes. The Q-learning formulation uses idleness as the reward signal.
- **Key insight**: Idleness is essentially "time since last visit" — directly applicable as an exploration reward for grid cells in our PE problem.

### 4.3 Occupancy Reward-Driven Exploration (2022)
- **Paper**: "Occupancy Reward-Driven Exploration with Deep Reinforcement Learning for Mobile Robot System"
- **Source**: Applied Sciences, 2022, 12(18), 9249
- **Approach**: Grid-based exploration where:
  ```
  r_explore = +r_new  when robot visits a NEW (previously unoccupied) cell
  r_step    = -c      small negative per step (encourages efficiency)
  r_total_variation = penalty for leaving small holes of non-covered space
  ```
- **Key insight**: The total variation term is novel — it penalizes leaving gaps in coverage, encouraging systematic sweeps rather than random exploration.

### 4.4 RL for Multi-Robot Field Coverage (2019)
- **Source**: NSF report (par.nsf.gov/servlets/purl/10185705)
- **Approach**: DDQN agents in grid world where:
  ```
  r = +1   when robot visits a reachable grid cell (coverage reward)
  r = -c   small negative per step (efficiency incentive)
  r = -p   penalty for visiting already-covered cells
  ```
- **Key insight**: Simple binary coverage reward — positive for new cells, penalty for revisiting.

---

## 5. Intrinsic Motivation / Curiosity for Exploration

### 5.1 Count-Based Exploration Bonus (Bellemare et al., 2016)
- **Paper**: "Unifying Count-Based Exploration and Intrinsic Motivation"
- **Authors**: Marc G. Bellemare, Sriram Srinivasan, Georg Ostrovski, et al.
- **Source**: NeurIPS 2016
- **Reward**:
  ```
  r_total = r_extrinsic + beta * r_intrinsic

  r_intrinsic = (N_hat(s_t, a_t) + 0.01)^{-1/2}
  ```
  where N_hat is the pseudo-count derived from a density model. For hash-based variant:
  ```
  r_intrinsic(s) = N(phi(s))^{-1/2}
  ```
  where N(phi(s)) tracks empirical occurrences of discretized hash codes.

  **Pseudo-count derivation**:
  ```
  N_hat(x) = rho_n(x) * (1 - rho'_n(x)) / (rho'_n(x) - rho_n(x))
  ```
  where rho_n = N_hat(x) / n, rho'_n = (N_hat(x)+1) / (n+1), and rho is estimated by a CTS density model.
- **Key insight**: Reward is proportional to 1/sqrt(visit_count). High reward for rarely-visited states, diminishing as states are revisited. This is the classic count-based exploration bonus.

### 5.2 ICM: Intrinsic Curiosity Module (Pathak et al., 2017)
- **Paper**: "Curiosity-driven Exploration by Self-supervised Prediction"
- **Authors**: Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, Trevor Darrell
- **Source**: ICML 2017
- **Reward**:
  ```
  r_total = r_extrinsic + beta * r_intrinsic

  r_intrinsic_t = || phi_hat(s_{t+1}) - phi(s_{t+1}) ||_2^2
  ```
  where phi(s) is a learned feature embedding of state s, and phi_hat(s_{t+1}) is the predicted next-state embedding from a forward dynamics model given (phi(s_t), a_t).
- **Key insight**: Curiosity = forward model prediction error in a learned feature space. States that are surprising (hard to predict) get high intrinsic reward. The inverse dynamics model ensures the feature space only captures aspects affected by the agent's actions (filtering out noise/distractors).

### 5.3 RND: Random Network Distillation (Burda et al., 2019)
- **Paper**: "Exploration by Random Network Distillation"
- **Authors**: Yuri Burda, Harrison Edwards, Ari Storkey, Oleg Klimov
- **Source**: ICLR 2019 (arXiv:1810.12894)
- **Reward**:
  ```
  r_intrinsic(s_t) = || f_hat(s_t; theta) - f(s_t) ||_2^2
  ```
  where f(s) is a FIXED randomly initialized target network and f_hat(s; theta) is a trainable predictor network, both mapping states to k-dimensional embeddings.
- **Key insight**: No forward dynamics model needed. The predictor learns to match the random network's output. For frequently-seen states, the predictor becomes accurate (low bonus). For novel states, prediction error is high (high bonus). Simpler than ICM, very effective in practice (Montezuma's Revenge).

### 5.4 VIME: Variational Information Maximizing Exploration (Houthooft et al., 2016)
- **Paper**: "VIME: Variational Information Maximizing Exploration"
- **Authors**: Rein Houthooft, Xi Chen, Yan Duan, John Schulman, Filip De Turck, Pieter Abbeel
- **Source**: NeurIPS 2016
- **Reward**:
  ```
  r'(s_t, a_t, s_{t+1}) = r(s_t, a_t) + eta * D_KL[q(theta; phi_{t+1}) || q(theta; phi_t)]
  ```
  where q(theta; phi) is the approximate posterior over dynamics model parameters, and the KL divergence measures information gain from the transition.
- **Key insight**: Intrinsic reward = information gain about the environment dynamics model. Actions that change the agent's belief about the world (i.e., provide surprising information) get high reward. eta controls exploration-exploitation tradeoff.

### 5.5 NGU: Never Give Up (Badia et al., 2020)
- **Paper**: "Never Give Up: Learning Directed Exploration Strategies"
- **Authors**: Adria Puigdomenech Badia, Pablo Sprechmann, et al. (DeepMind)
- **Source**: ICLR 2020
- **Reward** (two-level curiosity):
  ```
  r_t = r_t^e + beta * r_t^i

  Intrinsic reward combines episodic + lifelong:
  r_t^i = r_t^episodic * min(max(alpha_t, 1), L)

  Episodic reward (k-NN pseudo-counts):
  r_t^episodic = 1 / sqrt(Sum_{f_i in N_k} K(f(x_t), f_i) + c)

  where K is a kernel function, f(x_t) is the controllable state embedding,
  N_k is the set of k-nearest neighbors in episodic memory M,
  c = 10^{-3} is a small constant for numerical stability.

  Lifelong modulation (RND-based):
  alpha_t proportional to || g_hat(x_t; theta) - g(x_t) ||^2
  ```
- **Key insight**: Two timescales of novelty: (1) episodic — resets each episode, encourages visiting new states WITHIN an episode (prevents revisiting); (2) lifelong — decays slowly across episodes, focuses on globally novel states. The episodic component is most relevant for within-episode search behavior.

### 5.6 Maximum State Entropy in POMDPs (Zamboni et al., 2024)
- **Paper**: "How to Explore with Belief: State Entropy Maximization in POMDPs"
- **Source**: arXiv:2406.02295
- **Objective** (not a reward bonus, but an objective function):
  ```
  Maximum State Entropy (MSE):
  max_pi { J^S(pi) = E[H(d(tau_S))] }

  Maximum Observation Entropy (MOE):
  max_pi { J^O(pi) = E[H(d(tau_O))] }

  Maximum Believed Entropy (MBE):
  max_pi { J_tilde(pi) = E_{tau_B} E_{tau_S_tilde | tau_B} [H(d(tau_S_tilde))] }

  Regularized MBE (to prevent hallucination):
  J_tilde_rho(pi) = J_tilde(pi) - rho * E[Sum_t H(b_t)]
  ```
  where H is Shannon entropy, d(tau) is the empirical state distribution over a trajectory, b_t is the belief state.
- **Key insight**: Rather than adding an exploration bonus, treats entropy maximization as the SOLE objective. The regularization term penalizes high-entropy beliefs (prevents agent from creating uncertainty to inflate perceived entropy). Applicable to POMDPs where the agent can't observe the full state.

---

## 6. Summary Table: Applicability to Our Problem

Our problem: 1v1 pursuit-evasion, partial observability (combined LOS + sensing radius), pursuer needs to search for evader when not visible.

| Approach | Complexity | Requires | Pros | Cons | Applicability |
|----------|-----------|----------|------|------|--------------|
| **Entropy reduction per cell** (Sec 2.1) | Medium | Grid discretization | Direct reward for reducing uncertainty; theoretically grounded | Needs grid overlay; may add computational cost | **HIGH** — most natural fit |
| **Scan value / coverage** (Sec 2.3) | Medium | Visit history per cell | Continuous reward; encourages unvisited areas | Needs grid overlay | **HIGH** |
| **1/sqrt(visit_count)** (Sec 5.1) | Low | Visit counter per cell | Simple; well-studied; diminishing returns built in | Grid needed; may not scale to continuous | **HIGH** — easiest to implement |
| **Idleness-based** (Sec 4.2) | Low | Last-visit timestamp | Simple; encourages revisiting stale areas | More suited to persistent patrol than one-shot search | **MEDIUM** |
| **Time penalty** (Sec 1.3) | Very Low | Nothing extra | Zero implementation cost; implicit search incentive | Weak gradient; no spatial guidance | **Already have this** |
| **Visibility reward** (Sec 3.1) | Low | Visibility check | Per-step signal; directly rewards finding target | No spatial guidance for WHERE to search | **MEDIUM** |
| **First-visit bonus** (Sec 4.1, 4.4) | Low | Visited set | Very simple; rewards discovery | Binary; no reward for revisiting | **MEDIUM** |
| **RND** (Sec 5.3) | High | Extra neural networks | State-space agnostic; handles continuous states | Heavyweight; may not be needed for our grid-like problem | **LOW** — overkill |
| **ICM** (Sec 5.2) | High | Forward + inverse models | Handles continuous states | Complex; prediction error may not align with search | **LOW** — overkill |
| **Belief-based** (Sec 1.1) | High | Belief tracker | Theoretically optimal | Complex implementation; not standard in SB3 | **LOW** for now |

---

## 7. Recommended Formulations for Our Problem

### Option A: Grid-Cell Visitation Count Bonus (Simplest)
```
r_explore(t) = w_explore * (1 / sqrt(N(cell(x_t, y_t)) + 1))
```
where N(cell) is the number of times the pursuer has visited that grid cell in the current episode. Inspired by Bellemare et al. (2016) count-based exploration.

**Advantages**: Simple, well-understood, diminishing returns for revisiting.
**Tuning**: w_explore weight, grid resolution.

### Option B: Entropy Reduction Reward (Most Principled)
```
r_explore(t) = w_explore * Sum_{cells in FOV} [H(cell, t) - H(cell, t+1)]
```
where H(cell, t) is the uncertainty about whether the evader is in that cell. Inspired by the Multi-UAV Search paper (Sec 2.1).

**Advantages**: Information-theoretically grounded; rewards observing new areas.
**Tuning**: w_explore weight, uncertainty model.

### Option C: Scan Staleness Reward (Best Balance)
```
r_explore(t) = w_explore * (1/|cells_in_FOV|) * Sum_{c in FOV} min(staleness(c), S_max) / S_max
```
where staleness(c) = t - t_last_observed(c), capped at S_max.

**Advantages**: Continuous signal; naturally guides pursuer to areas not recently observed; time-aware (unlike visit count).
**Tuning**: w_explore weight, S_max cap, grid resolution.

### Option D: Visibility-Based Reward (Zero Implementation Cost)
```
r_vis(t) = w_vis * 1[evader_visible_t]
```
Small positive reward every step the evader is in the pursuer's field of view. Inspired by OpenAI Hide-and-Seek (Sec 3.1).

**Advantages**: Trivial to implement; directly rewards what we want (finding the evader).
**Disadvantages**: No spatial guidance for WHERE to search; may cause pursuer to "stare" at evader rather than approach.
