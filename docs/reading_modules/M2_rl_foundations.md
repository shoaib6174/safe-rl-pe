# Module 2: Deep RL for Continuous Multi-Agent Control

**Estimated reading time:** 4–5 hours  
**Prerequisites:** Module 1 (Pursuit-Evasion as a Differential Game)  
**Goal:** Understand the deep RL algorithms that power our pursuit-evasion experiments, why each was chosen, and why some failed catastrophically. This module bridges single-agent RL theory to the competitive multi-agent setting where the "environment" is another learning agent.

---

## 2.1 The Single-Agent RL Foundation

Before tackling two agents that learn simultaneously, we need the single-agent framework. An agent interacts with an environment modeled as a **Markov Decision Process (MDP)**:

- State space $\mathcal{S}$, action space $\mathcal{A}$
- Transition dynamics $P(s' \mid s, a)$
- Reward function $R(s, a)$
- Discount factor $\gamma \in [0, 1)$

The agent's goal is to learn a **policy** $\pi(a \mid s)$ that maximizes the expected **discounted return**:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

where $\tau = (s_0, a_0, r_0, s_1, \dots)$ is a trajectory sampled by following $\pi$.

### The Policy Gradient Theorem

**Policy gradient methods** directly optimize the policy parameters $\theta$ by estimating the gradient $\nabla_\theta J(\pi_\theta)$. The **policy gradient theorem** (Sutton et al., 2000) states:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot Q^\pi(s, a) \right]$$

where $\rho^\pi$ is the state visitation distribution under $\pi$ and $Q^\pi(s, a)$ is the action-value function. In practice, we replace $Q^\pi(s, a)$ with an **advantage estimate** $A(s, a) = Q(s, a) - V(s)$ to reduce variance.

**Why this matters for PE:** The policy gradient theorem assumes a stationary environment. In competitive self-play, the opponent's policy changes continuously, so the state visitation distribution $\rho^\pi$ is non-stationary. This is the root cause of oscillation and collapse.

---

## 2.2 Actor-Critic Architecture

Most modern deep RL algorithms for continuous control use the **actor-critic** architecture, which maintains two neural networks:

| Component | Network | Purpose |
|-----------|---------|---------|
| **Actor** | $\pi_\theta(s) \rightarrow a$ | Maps states to actions (the policy) |
| **Critic** | $Q_\phi(s, a) \rightarrow \mathbb{R}$ | Estimates the value of state-action pairs |

The actor is updated using the critic's estimates (policy gradient), while the critic is updated via temporal-difference (TD) learning to approximate the true $Q$-function. This separation is crucial: the critic provides a learning signal for the actor, while the actor generates the data distribution the critic learns from.

### Target Networks and Soft Updates

To stabilize training, the critic typically uses a **target network** — a slowly-updated copy of the main network. The target $Q$-values are computed using the target network, preventing the "moving target" problem where the critic chases its own tail. In our MADDPG implementation, target networks are updated via **soft updates**:

$$\phi_{\text{target}} \leftarrow (1 - \tau) \cdot \phi_{\text{target}} + \tau \cdot \phi$$

with $\tau = 0.005$ (in G&I 2024). This is much slower than the main network's gradient updates, keeping the target stable.

---

## 2.3 PPO: Stable Policy Optimization

**Proximal Policy Optimization (PPO)** (Schulman et al., 2017) is the workhorse of stable RL. It addresses the instability of vanilla policy gradients by constraining how far the policy can change in a single update.

### The Clipped Surrogate Objective

PPO replaces the vanilla policy gradient with a **clipped surrogate objective** built in three stages:

**Stage 1 — Probability ratio.** Define the ratio of new to old policy probabilities:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

If $r_t > 1$, the new policy is more likely to take action $a_t$; if $r_t < 1$, it is less likely.

**Stage 2 — The clip operator.** PPO clips the ratio to a small interval around 1:

$$\text{clip}\bigl(r_t(\theta), 1 - \epsilon, 1 + \epsilon\bigr)$$

This forces the ratio to stay within $[1 - \epsilon, 1 + \epsilon]$, typically $\epsilon = 0.2$. Even if the advantage estimate is strongly positive, the clipped term cannot grow beyond $1 + \epsilon$.

**Stage 3 — The full objective.** PPO takes the minimum of the unclipped and clipped objectives, so the gradient only flows through the smaller (more conservative) term:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\Bigl( r_t(\theta) \hat{A}_t,\; \text{clip}\bigl(r_t(\theta), 1 - \epsilon, 1 + \epsilon\bigr) \hat{A}_t \Bigr) \right]$$

**Why PPO is stable:** The clip acts as a trust region. Even if the advantage estimate is wrong, the policy cannot change too aggressively. This makes PPO robust to noisy gradients and hyperparameter choices.

### The KL-Penalty Variant (2.3.2)

An alternative trust-region mechanism is an **adaptive KL penalty**. Instead of clipping, PPO can penalize the KL divergence between the old and new policies:

$$L^{\text{KL}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \hat{A}_t - \beta \cdot \text{KL}\bigl(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta\bigr) \right]$$

The coefficient $\beta$ is adapted dynamically: if the KL divergence exceeds a target threshold, $\beta$ is increased; if it falls below, $\beta$ is decreased. This serves the same purpose as clipping — preventing overly large policy updates — but uses a soft penalty rather than a hard bound. Stable-Baselines3 uses the **clipped objective by default**; we never experimented with the KL variant.

### Generalized Advantage Estimation (GAE)

PPO uses **GAE** to compute advantage estimates with a bias-variance tradeoff controlled by $\lambda \in [0, 1]$:

$$\hat{A}_t^{\text{GAE}(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD residual. With $\lambda = 0$, GAE reduces to one-step TD (high bias, low variance). With $\lambda = 1$, it becomes Monte Carlo (unbiased but high variance). In practice, $\lambda = 0.95$ is common.

### Project Application: Phase 1–3

Our early work (S11–S84) used **PPO from Stable-Baselines3 (sb3)** for self-play in a 10×10m arena with obstacles. PPO was chosen for its stability and ease of use. However, we discovered a critical limitation:

- PPO is **on-policy**: it can only learn from data collected by the current policy.
- In competitive self-play, both agents change simultaneously, making on-policy data stale almost immediately.
- **RecurrentPPO (LSTM)** was attempted (S68) but abandoned — LSTM memory does not solve the core non-stationarity problem.

**Decision D1 (S66):** Switched from PPO to **SAC** after SAC reached 30–40% capture rate at step counts where PPO stalled. RecurrentPPO conclusively failed at 0–2% across 6 variants.

**Why SAC comes next.** PPO's stability comes at a cost: it is **on-policy**, which is fatal when the opponent changes every episode. The data PPO collected under yesterday's opponent is already stale today. Off-policy algorithms can reuse old experience, making them better suited to competitive self-play.

---

## 2.4 SAC: Maximum Entropy RL

**Soft Actor-Critic (SAC)** (Haarnoja et al., 2018) is an off-policy actor-critic algorithm that maximizes both expected return and policy entropy:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t \left( r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t)) \right) \right]$$

The **entropy bonus** $\alpha \mathcal{H}(\pi)$ encourages exploration by rewarding stochastic policies. SAC uses a **twin-critic** architecture (two $Q$-networks) to combat overestimation bias, taking the minimum $Q$-value for target computation.

### Automatic Temperature Tuning

In the original SAC formulation, the entropy coefficient $\alpha$ is not fixed but learned via **dual gradient descent** to maintain a target entropy $\bar{\mathcal{H}}$:

$$\min_{\alpha} \; \mathbb{E}_{a \sim \pi} \left[ -\alpha \log \pi(a \mid s) - \alpha \bar{\mathcal{H}} \right]$$

The gradient $\partial L / \partial \alpha = -\mathbb{E}[\log \pi(a \mid s)] - \bar{\mathcal{H}}$ pushes $\alpha$ up when the policy is too deterministic and down when it is too stochastic. This auto-tuning removes the need to hand-tune $\alpha$.

**Why we disabled it:** Auto-tuning caused collapse in our experiments (SP11c dropped from 24% to 6% capture rate). We fixed $\alpha$ (ent_coef = 0.03) instead. See **Decision D2 (S67)**.

### Why SAC Over PPO for Self-Play?

| Property | PPO | SAC |
|----------|-----|-----|
| On/off-policy | On-policy | Off-policy |
| Sample efficiency | Lower | Higher |
| Exploration | Noise injection | Entropy maximization |
| Replay buffer | No | Yes |
| Continuous actions | Yes | Yes |

**Off-policy learning is essential for self-play.** SAC's replay buffer stores transitions from past policies, so even when the opponent changes, the buffer contains diverse experience. This partially mitigates non-stationarity.

### SAC in Our Early Work

In Phase 1–3 (S66), SAC outperformed PPO for competitive self-play. However, SAC alone was insufficient to prevent collapse. The entropy coefficient $\alpha$ (or `ent_coef`) became a critical hyperparameter:

- **Decision D2 (S67):** `ent_coef = 0.03` fixed, never auto-tuned. Auto entropy caused collapse (SP11c dropped from 24% to 6%).
- **Decision D3 (S59):** `ent_coef = 0.03` for variable-speed evader. `0.01` stalled at 12% escape; `0.03` exploded to 98%.

The lesson: exploration is critical, but too much or too little entropy destroys learning dynamics.

### Why We Abandoned SAC for MADDPG

SAC was our best single-agent algorithm for self-play, but the Gonultas & Isler 2024 reproduction required **MADDPG** — the paper's stated algorithm. Once we switched to the G&I codebase, all subsequent experiments (EMA, Pop-FSP, belief-state extensions) were built on that MADDPG foundation. SAC was never tested in the G&I environment. This transition is recorded in **Decision D4 (S92)**.

---

## 2.5 MADDPG: Multi-Agent Deep Deterministic Policy Gradient

**MADDPG** (Lowe et al., 2017) is the algorithm at the heart of our G&I 2024 reproduction and all subsequent experiments (EMA, Pop-FSP). It extends DDPG (Lillicrap et al., 2015) to multi-agent settings with a key insight: **centralized training with decentralized execution (CTDE)**.

### Centralized Training with Decentralized Execution

In MADDPG, each agent $i$ has:
- A **decentralized actor** $\pi_{\theta_i}(o_i)$ that observes only its own observation $o_i$
- A **centralized critic** $Q_{\phi_i}(o_1, \dots, o_N, a_1, \dots, a_N)$ that sees all observations and actions during training

During **training**, the critic has full state information (all observations, all actions), enabling accurate $Q$-value estimation. During **execution**, each agent uses only its actor and its own observation — no central coordinator needed.

### Why Centralized Critic but Decentralized Actor?

Consider two agents learning simultaneously. If each agent's critic only saw its own observation, it would face the **non-stationarity problem**: the opponent's policy changes, so the transition dynamics $P(s' \mid s, a)$ appear non-stationary from any single agent's perspective. The centralized critic sidesteps this by conditioning on all agents' policies explicitly:

$$\nabla_{\theta_i} J(\theta_i) = \mathbb{E}_{\mathbf{o}, \mathbf{a} \sim \mathcal{D}} \Bigl[ \underbrace{\nabla_{\theta_i} \pi_i(o_i)}_{\text{actor gradient}} \cdot \underbrace{\nabla_{a_i} Q_i^{\pi}(\mathbf{o}, \mathbf{a})}_{\text{critic gradient}} \Big|_{a_i = \pi_i(o_i)} \Bigr]$$

where $\mathbf{o} = (o_1, \dots, o_N)$ and $\mathbf{a} = (a_1, \dots, a_N)$. The outer gradient $\nabla_{\theta_i} \pi_i(o_i)$ is the policy gradient (how the actor's parameters affect its actions). The inner gradient $\nabla_{a_i} Q_i$ tells the actor which direction to move its action to increase the $Q$-value. The evaluation bar $\big|_{a_i = \pi_i(o_i)}$ means the critic gradient is computed at the action the actor currently outputs.

**Why this works despite non-stationarity:** The critic is *centralized* — it sees all observations and all actions — so it can learn $Q_i(\mathbf{o}, \mathbf{a})$ directly without needing to model the opponent's policy as part of the environment dynamics. The actor remains *decentralized*, using only its own observation at execution time.

### Experience Replay for Multi-Agent

MADDPG uses a **shared replay buffer** $\mathcal{D}$ that stores transitions $(o_i, a_i, r_i, o_i')$ for all agents. Because the critic is centralized, it can learn from off-policy data even when the opponent has changed. This is why MADDPG works for competitive self-play where PPO struggles.

**Critical implementation detail:** In our code (`agents/maddpg.py`), the critic takes concatenated observations and actions: `obs_all = torch.cat([obs_p, obs_e], dim=-1)` and `actions_all = torch.cat([actions_p, actions_e], dim=-1)`. The critic outputs a single scalar $Q$-value for the joint state-action pair.

---

## 2.6 Twin Critics and Overestimation Bias

A subtle but devastating issue in actor-critic methods is **overestimation bias**. When the actor uses the same critic it is training to select actions, it naturally gravitates toward actions where the critic overestimates the $Q$-value. This creates a positive feedback loop:

1. Critic overestimates action $a$
2. Actor selects $a$ more often
3. Critic trains more on $a$, reinforcing the overestimate
4. $Q$-values explode → NaN

### The Fix: Twin Critics

Following **Ackermann et al. (2019)** — cited by G&I 2024 — we implemented **twin (double) critics**:

- Two independent critic networks: $Q_1$ and $Q_2$
- Target $Q$-value: $y = r + \gamma \cdot \min(Q_{1,\text{target}}, Q_{2,\text{target}})$
- Actor loss: $-\min(Q_1(s, \pi(s)), Q_2(s, \pi(s)))$

Taking the minimum of two independent estimates is a conservative bias correction. If one critic overestimates, the other likely does not, and the minimum prevents the actor from exploiting overestimation.

### Additional Stability Fixes (S93)

Beyond twin critics, we applied several numerical-stability measures:

1. **Gradient clipping** (`max_norm=1.0`) to prevent exploding gradients
2. **Q-clipping** (`Q_MIN = -1500.0`, `Q_MAX = 1500.0`) to bound target $Q$-values and prevent the actor loss from exploding
3. **Disable AMP for BiMDN** — Automatic Mixed Precision (fp16) caused NaN cascades when combined with BiMDN's unbounded scale outputs; vanilla MADDPG tolerates AMP

These fixes are discussed in detail in the NaN story below.

### The NaN Story (S92–S94)

Our G&I reproduction initially used a **single critic** and hit catastrophic NaN losses:

| Episode | Evader Critic Loss |
|---------|-------------------|
| 15790 | 1,956,107,958,353,920 (1.9 quadrillion) |
| 15798 | 2,205,154,589,802,496 |
| 15802 | **NaN** |

**Root cause:** Single-critic MADDPG overestimation bias → positive feedback loop in target $Q$ update.

**Fix (S93):** Implemented twin critics per Ackermann et al. (2019). Result: 20K episodes completed with **zero NaNs**. Max critic loss bounded at 25,819.

**Additional fix (S93):** The actor must also use $\min(Q_1, Q_2)$, not just $Q_1$:
```python
# OLD — actor exploits critic1 overestimation
actor_loss = -q1_actor.mean()

# NEW — conservative actor update
actor_loss = -torch.min(q1_actor, q2_actor).mean()
```

This was a subtle bug discovered in S93: even though target computation used `min(target_q1, target_q2)`, the actor gradient still exploited whichever critic happened to overestimate more.

---

## 2.7 EMA Opponent Stabilization

> **Moved to Module 9.** EMA opponent stabilization is covered in depth there, including the half-life derivation, the full sweep across $\tau \in \{0.0001, 0.0005, 0.001, 0.005, 0.01\}$, and the seed-dependence results (1/5 balanced with LR = 0.0005). See **M9 §9.2**.

Briefly: EMA replaces the rapidly-changing current opponent with a slowly-evolving exponential moving average copy. With $\tau_{\text{EMA}} = 0.001$, the opponent policy changes by roughly 18% per episode — slow enough to dampen oscillations, fast enough to track genuine improvement. It was seed-dependent: only 1 of 5 seeds achieved balanced play.

---

## 2.8 Pop-FSP: Population-Based Fictitious Self-Play

> **Moved to Module 10.** The full Pop-FSP training loop, the ∀/∃ structural asymmetry finding, and the v3–v7 configuration sweep (0/5 balanced) are covered in **M10 §10.3**.

Briefly: Pop-FSP maintains a population of past policies and samples opponents uniformly. Our experiments (v3–v7, 5 configs tested) revealed a fundamental ∀/∃ asymmetry: the pursuer only needs to catch *some* evader ($\exists$), while the evader must escape *all* pursuers ($\forall$). In a symmetric game this structurally favors the pursuer.

---

## 2.9 When to Use Which Algorithm

| Algorithm | Best For | Our Experience |
|-----------|----------|----------------|
| **PPO** | Stable single-agent, discrete action spaces | Phase 1–3; failed in competitive self-play due to on-policy limitation |
| **SAC** | Sample-efficient continuous control, single-agent | Better than PPO for self-play, but still collapsed |
| **MADDPG** | Competitive multi-agent with continuous actions | G&I reproduction; works but oscillates without stabilization |
| **MADDPG + EMA** | Competitive multi-agent with oscillation damping | Seed-dependent (1/5 success with LR=0.0005); promising but not reliable. See M9. |
| **MADDPG + Pop-FSP** | Game-theoretic training with population diversity | Eliminated oscillation but caused pursuer-dominant collapse due to ∀/∃ asymmetry. See M10. |

### Historical Context: What Was Tried, What Failed

| Approach | Sessions | Result | Root Cause |
|----------|----------|--------|------------|
| PPO (sb3) | S11–S66 | Stalled at low capture rates | On-policy, cannot handle non-stationary opponent |
| RecurrentPPO (LSTM) | S68 | Never launched successfully | LSTM memory does not solve core non-stationarity |
| SAC | S66 | 30–40% capture, then collapse | Better than PPO but insufficient alone |
| MADDPG (vanilla) | S92–S96 | CR=0.550 matches paper, then oscillates | Intrinsic to competitive dynamics |
| MADDPG + EMA | S97–S99, S104 | 1/5 seeds balanced (LR=0.0005) | Seed-dependent; LR and initialization matter |
| MADDPG + Pop-FSP | S100–S103 | 0/5 configs balanced (v3–v7) | ∀/∃ structural asymmetry |

---

---

## 2.10 Summary

| Concept | Meaning for PE |
|---------|---------------|
| Policy gradient theorem | Foundation for actor-critic; fails in competitive settings due to non-stationarity |
| Actor-critic | Actor maps obs→actions; critic estimates Q-values; separation enables stable learning |
| PPO | Clipped surrogate objective; stable but on-policy — ill-suited for competitive self-play |
| SAC | Off-policy, maximum entropy; better sample efficiency but still collapses |
| MADDPG | Centralized critic + decentralized actors; CTDE sidesteps non-stationarity in critic |
| Twin critics | $\min(Q_1, Q_2)$ prevents overestimation bias; essential for stable MADDPG |
| EMA opponent | See M9 §9.2 — low-pass filter on opponent policy; dampens oscillation but seed-dependent |
| Pop-FSP | See M10 §10.3 — population diversity prevents overfitting; ∀/∃ asymmetry causes systematic collapse |
| AMP/fp16 | See M6 §6.4 — ~1.4× speedup but triggers NaN cascade with BiMDN; disable for belief networks |

---

## Self-Check

1. **Why does MADDPG use a centralized critic but decentralized actors?** What problem does this solve, and what limitation does it not address?

2. **When would you choose PPO over SAC for a competitive game, and why did we ultimately use neither for our G&I reproduction?** Consider sample efficiency, on-policy vs. off-policy, and the specific requirements of reproducing a published paper.

3. **Why did twin critics fix the NaN problem in our G&I reproduction?** Explain the positive feedback loop of overestimation bias and how $\min(Q_1, Q_2)$ breaks it.

---

## Papers to Read

- **(ESSENTIAL)** Lowe et al. (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." (MADDPG — the algorithm at the core of all our experiments.)
- **(ESSENTIAL)** Ackermann et al. (2019). "Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics." (Twin critics — cited by G&I 2024, implemented in S93.)
- **(REFERENCE)** Schulman et al. (2017). "Proximal Policy Optimization Algorithms." (PPO — understand clipped surrogate objective and why it's stable.)
- **(REFERENCE)** Haarnoja et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." (SAC — maximum entropy, twin critics, automatic temperature tuning.)
- **(REFERENCE)** Heinrich & Silver (2016). "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games." (FSP — the theoretical basis for Pop-FSP.)
- **(OPTIONAL)** Lillicrap et al. (2015). "Continuous Control with Deep Reinforcement Learning." (DDPG — the precursor to MADDPG.)

---

## Deep Dive Sessions

- **S13** (PPO training specifics, Phase 1): Detailed hyperparameter sweeps and environment configurations for early PPO self-play. See worklog for the full parameter grid.
- **S41** (AMS-DRL survey): Review of multi-agent deep RL methods that informed our algorithm selection before committing to MADDPG.
- **S66** (SAC vs PPO decision): The pivotal session where we switched from PPO to SAC after conclusive evidence that PPO stalls in competitive settings.
- **S68** (RecurrentPPO): LSTM policy implementation that was completed but never successfully launched — a cautionary tale about adding complexity without diagnosing the root cause. Six variants tested; all stalled at 0–2% capture rate.
- **S92–S94** (G&I reproduction + NaN debugging): The complete story of implementing MADDPG, diagnosing NaN root causes, and fixing them with twin critics. Essential reading for understanding numerical stability in competitive RL.
- **S97** (EMA design): Mathematical analysis of EMA half-life, implementation decisions, and the critical fixes that made EMA evaluation possible. See M9 for full results.
- **S100–S103** (Pop-FSP and ∀/∃ asymmetry): The full arc from Pop-FSP implementation through v3–v7 configuration sweeps to the structural asymmetry finding. See M10 for full analysis.
