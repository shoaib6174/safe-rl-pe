# ICRA 2027 Research Plan: Stabilized Belief-State Self-Play for Pursuit-Evasion

**Working Title:** *Population-Based Fictitious Self-Play Stabilizes Competitive Training in Partially Observable Pursuit-Evasion Games*

**Target Venue:** ICRA 2027 (deadline: ~September 2026)
**Estimated Duration:** 3-4 months
**Builds on:** Gonultas & Isler 2024 reproduction (completed)

---

## 1. Problem Statement

### 1.1 The Oscillation Problem

Both our independent SAC self-play and the reproduced G&I MADDPG+BiMDN exhibit **oscillatory competitive dynamics**:

- Phase 1 (ep 0-5K): Pursuer loses almost every episode
- Phase 2 (ep 5K-10K): Pursuer catches up and dominates
- Phase 3 (ep 10K-15K): Evader adapts and catches up
- Phase 4 (ep 15K-20K): Evader dominates
- ...cycle repeats...

**Consequence:** No single checkpoint achieves all metrics simultaneously. G&I reports 0.98 vs random, 0.98 vs greedy, and 0.54 self-play — but these come from *different* checkpoints. This is never acknowledged in the paper.

### 1.2 Root Cause Analysis

The oscillation arises from **training against a non-stationary opponent**:

1. When pursuer trains, the evader is simultaneously changing
2. The pursuer overfits to the current evader strategy
3. By the time the pursuer converges, the evader has already adapted
4. The evader's new strategy exploits the pursuer's learned behavior
5. Result: perpetual strategy cycling (rock-paper-scissors dynamics)

This is the **fundamental limitation of vanilla self-play** in competitive games.

### 1.3 Why This Matters

- **Real robot deployment:** You cannot deploy a policy that only works against a specific training checkpoint
- **Sample efficiency:** Oscillation wastes training budget cycling through dominated strategies
- **Generalization:** Policies overfit to the training distribution and fail against simple baselines (our 0.25 vs greedy result)
- **Theoretical:** No convergence guarantees to Nash equilibrium

### 1.4 Gap in Literature

| Work | Method | Handles PO | Handles Oscillation | Real Robots |
|------|--------|-----------|---------------------|-------------|
| G&I 2024 | MADDPG+BiMDN | Yes | No | Yes |
| Wang et al. 2025 | RL+SMC+CBF | No | No | No |
| Zhang et al. 2024 (Survey) | PSRO/FSP theory | N/A | Yes | No |
| AlphaStar (Vinyals et al.) | League training | Yes | Yes | No |
| **Ours (proposed)** | **Pop-FSP + BiMDN** | **Yes** | **Yes** | **No** |

**No existing work combines:** learned belief states + stabilized competitive training + rigorous convergence analysis for ground robot PE.

---

## 2. Proposed Method: Population-Based Fictitious Self-Play (Pop-FSP)

### 2.1 Core Idea

Instead of training each agent against the **current** opponent, train against a **diverse population** of past opponents. This forces each agent to learn robust strategies that work against multiple opponent types simultaneously, preventing overfitting and strategy cycling.

### 2.2 Algorithm

```
Initialize: pursuer policy π_p, evader policy π_e
Initialize: populations Π_p = {π_p}, Π_e = {π_e}

For episode = 1 to N:
    # Collect experience with current policies
    trajectories = run_episode(π_p, π_e)
    
    # MADDPG update with population-sampled opponent
    For each update step:
        π_e_sample ~ Uniform(Π_e)  # sample past evader
        π_p_sample ~ Uniform(Π_p)  # sample past pursuer
        
        # Pursuer update: uses π_e_sample for target Q and actor gradient
        update_pursuer(trajectories, π_e_sample)
        
        # Evader update: uses π_p_sample for target Q and actor gradient
        update_evader(trajectories, π_p_sample)
    
    # Population management (every M episodes)
    If episode % M == 0:
        Π_p ← Π_p ∪ {checkpoint(π_p)}
        Π_e ← Π_e ∪ {checkpoint(π_e)}
        If |Π_p| > K: remove oldest from Π_p
        If |Π_e| > K: remove oldest from Π_e
```

### 2.3 Key Design Decisions

**Population size K:** K=8 matches the paper's 8 parallel environments. Small enough for memory, large enough for diversity.

**Update frequency M:** Every 250 episodes (same as checkpoint frequency). This balances population diversity against computational cost.

**Sampling strategy:** Uniform random sampling from population. Alternatives:
- Weighted by opponent strength (prioritize stronger opponents)
- Weighted by recency (prioritize recent opponents)
- Nash mixture (compute meta-game Nash, sample accordingly)

**Population pruning:** Remove oldest (FIFO). Alternatives:
- Remove weakest (keep strongest opponents)
- Remove most similar (maximize population diversity)

### 2.4 Integration with BiMDN

Critical: Each opponent in the population must have its **own BiMDN** (belief network). When sampling opponent `i`:
- Use opponent `i`'s actor network for action selection
- Use opponent `i`'s BiMDN for belief computation

This means storing K actor networks + K BiMDN networks per agent.

Memory optimization: Only store network weights, not full training state. With K=8 and 128-hidden networks, this is ~10MB per population — negligible.

### 2.5 Convergence Analysis (Theoretical Contribution)

**Hypothesis:** Pop-FSP converges to an approximate Nash equilibrium with bounded exploitability.

**Sketch proof:**
1. Define exploitability: ε(π_p, π_e) = max_π' V_p(π', π_e) - V_p(π_p, π_e)
2. In vanilla SP, ε oscillates because each agent overfits to the current opponent
3. In Pop-FSP, each agent must be robust to K opponents, so exploitability against any single opponent is bounded by the population diversity
4. As K → ∞, Pop-FSP approaches true FSP which converges to Nash (Heinrich & Silver 2016)
5. For finite K, we get an ε-approximate Nash where ε decreases with K

**Deliverable:** Formal lemma/theorem + empirical validation.

---

## 3. Implementation Plan

### 3.1 File Structure

```
repro_gonultas_isler_2024/
├── agents/
│   └── maddpg.py              # Modify: accept external opponent for update
├── training/
│   ├── train.py               # Modify: add population management
│   ├── population_manager.py  # NEW: population sampling + storage
│   └── vectorized_env.py      # Modify: support population opponents
├── evaluation/
│   └── evaluate.py            # Modify: evaluate against all population members
└── config_pop_fsp.yaml        # NEW: hyperparameters for Pop-FSP
```

### 3.2 Components

#### Component 1: PopulationManager

```python
class PopulationManager:
    def __init__(self, max_size: int, device: str):
        self.max_size = max_size
        self.policies = []      # List of actor state dicts
        self.bimdns = []        # List of BiMDN state dicts
        self.device = device
    
    def add(self, actor_state: dict, bimdn_state: dict):
        """Add new checkpoint to population."""
        self.policies.append(copy.deepcopy(actor_state))
        self.bimdns.append(copy.deepcopy(bimdn_state))
        if len(self.policies) > self.max_size:
            self.policies.pop(0)  # FIFO
            self.bimdns.pop(0)
    
    def sample(self) -> tuple:
        """Sample random opponent from population."""
        idx = random.randint(0, len(self.policies) - 1)
        return self.policies[idx], self.bimdns[idx]
```

#### Component 2: Modified MADDPG Update

Current MADDPG update takes `other_agent` as argument. Modify to accept `opponent_actor` and `opponent_bimdn` directly:

```python
def update(
    self,
    batch: dict,
    opponent_actor: nn.Module,      # NEW: external opponent actor
    opponent_bimdn: Optional[nn.Module],  # NEW: external opponent BiMDN
    agent_id: str = "pursuer",
) -> tuple:
    # ... existing code ...
    # Use opponent_actor instead of other_agent.actor_target
    # Use opponent_bimdn for belief computation if needed
```

#### Component 3: Modified Training Loop

```python
# Initialize populations
pursuer_pop = PopulationManager(max_size=K, device=device)
evader_pop = PopulationManager(max_size=K, device=device)

# Training loop
while completed_episodes < episodes:
    # ... collect experience ...
    
    # Sample opponents from populations
    opp_p_actor, opp_p_bimdn = pursuer_pop.sample()
    opp_e_actor, opp_e_bimdn = evader_pop.sample()
    
    # MADDPG updates against sampled opponents
    for _ in range(n_updates):
        batch = replay_buffer.sample(batch_size)
        c_loss_p, a_loss_p = pursuer.update(
            batch, opp_e_actor, opp_e_bimdn, agent_id="pursuer"
        )
        c_loss_e, a_loss_e = evader.update(
            batch, opp_p_actor, opp_p_bimdn, agent_id="evader"
        )
    
    # Update populations every M episodes
    if completed_episodes % M == 0:
        pursuer_pop.add(
            copy.deepcopy(pursuer.actor.state_dict()),
            copy.deepcopy(belief_pursuer.bimdn.state_dict()),
        )
        evader_pop.add(
            copy.deepcopy(evader.actor.state_dict()),
            copy.deepcopy(belief_evader.bimdn.state_dict()),
        )
```

### 3.3 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Population size K | 8 | Matches paper's 8 parallel envs |
| Population update freq M | 250 episodes | Same as checkpoint frequency |
| Population sampling | Uniform | Baseline; try weighted later |
| Population pruning | FIFO | Simplest; try diversity-based later |
| Episodes | 20,000 | Same as G&I for fair comparison |
| Seeds | 5 | Statistical robustness |

---

## 4. Experiments

### 4.1 Baselines

| Baseline | Description | Purpose |
|----------|-------------|---------|
| G&I Vanilla | MADDPG+BiMDN, vanilla self-play | Reproduction baseline |
| EMA Opponent | Train against EMA of opponent weights | Ablation: smooth opponent |
| Pop-Only | Population without EMA | Ablation: discrete opponents |
| Pop-FSP (ours) | Full population method | Proposed method |

### 4.2 Evaluation Metrics

**Primary metrics (from G&I paper):**
1. Self-play capture rate (target: ~0.50, balanced)
2. vs-random capture rate (target: >0.90)
3. vs-greedy capture rate (target: >0.90)

**Novel metrics (our contribution):**
4. **Oscillation amplitude:** Standard deviation of self-play CR over last 5K episodes
5. **Single-checkpoint robustness:** Best checkpoint's minimum CR across all three metrics
6. **Exploitability:** Tournament win rate against population
7. **Population diversity:** Average behavioral distance between population members

### 4.3 Experiment 1: Oscillation Reduction

**Question:** Does Pop-FSP reduce oscillation amplitude?

**Method:**
- Train all 4 baselines for 20K episodes, 5 seeds each
- Record self-play CR every 250 episodes
- Compute oscillation amplitude: std(CR) over sliding window of 20 checkpoints

**Expected result:** Pop-FSP has significantly lower oscillation amplitude than vanilla.

**Figure:** Time series plot showing CR vs episodes for all baselines. Pop-FSP should show damped oscillation converging to steady-state.

### 4.4 Experiment 2: Single-Checkpoint Robustness

**Question:** Can a single Pop-FSP checkpoint dominate all baselines?

**Method:**
- For each baseline, find the checkpoint with best "single-checkpoint robustness score":
  ```
  score = min(CR_selfplay, CR_vs_random, CR_vs_greedy)
  ```
- Report the max score across all checkpoints

**Expected result:** Pop-FSP achieves the highest single-checkpoint robustness score. Vanilla SP has score ≈ 0.25 (limited by greedy). Pop-FSP should achieve >0.50.

**Figure:** Bar chart comparing single-checkpoint robustness across baselines.

### 4.5 Experiment 3: Population Size Ablation

**Question:** How does population size K affect performance?

**Method:**
- Train Pop-FSP with K ∈ {1, 2, 4, 8, 16}
- K=1 is equivalent to vanilla SP
- Measure: oscillation amplitude, single-checkpoint robustness, training time

**Expected result:** Performance improves with K up to a point (diminishing returns). K=8 is near-optimal.

**Figure:** Plot of metrics vs K.

### 4.6 Experiment 4: Population Diversity Analysis

**Question:** Does the population actually contain diverse strategies?

**Method:**
- Compute behavioral distance between population members:
  ```
  d(π_i, π_j) = E_s[||π_i(s) - π_j(s)||^2]
  ```
- Measure how diversity evolves during training
- Correlate diversity with oscillation amplitude

**Expected result:** Higher population diversity correlates with lower oscillation.

**Figure:** Heatmap of pairwise behavioral distances in population.

### 4.7 Experiment 5: Generalization to Unseen Opponents

**Question:** Does Pop-FSP improve generalization?

**Method:**
- Train all baselines
- Evaluate best checkpoint against:
  1. Random walk (seen during training indirectly)
  2. Greedy evader (not seen during training)
  3. Rash evader (from G&I paper, not seen)
  4. Pure pursuit heuristic (not seen)

**Expected result:** Pop-FSP generalizes better to unseen opponents because it was trained against diverse strategies.

---

## 5. Theoretical Analysis

### 5.1 Formal Problem Setup

**Definition 1 (Partially Observable Stochastic Game):**
G = (S, {A_i}, {O_i}, P, {R_i}, γ) where players have limited observations.

**Definition 2 (Exploitability):**
For joint policy π = (π_p, π_e), the exploitability is:
```
ε(π) = max_π_p' V_p(π_p', π_e) - V_p(π_p, π_e) + max_π_e' V_e(π_p, π_e') - V_e(π_p, π_e)
```

**Definition 3 (ε-Nash Equilibrium):**
π is an ε-Nash if ε(π) ≤ ε.

### 5.2 Propositions

**Proposition 1 (Vanilla SP Oscillation):**
In vanilla self-play with simultaneous updates, the joint policy sequence {π^(t)} does not converge to Nash in general. The exploitability ε(π^(t)) oscillates with amplitude bounded by the learning rate and policy capacity.

**Proposition 2 (Pop-FSP Convergence):**
Under Pop-FSP with population size K and uniform sampling, the expected exploitability satisfies:
```
E[ε(π^(t))] ≤ C / sqrt(K) + δ
```
where C depends on the game structure and δ is the approximation error from finite training.

**Proposition 3 (Population Diversity):**
If the population diversity D(Π) = E_{i,j}[d(π_i, π_j)] is above a threshold τ, then no single strategy can dominate all population members, preventing strategy cycling.

### 5.3 Deliverables

- Formal definitions and propositions (1-2 pages)
- Proof sketches for Propositions 1-2
- Empirical validation of all propositions

---

## 6. Paper Structure (Target: 8 pages ICRA format)

```
I. Introduction (1 page)
   - PE problem, partial observability, safety
   - Oscillation problem in competitive self-play
   - Our contribution: Pop-FSP stabilizes training
   - Results preview: single checkpoint hits all metrics

II. Related Work (0.75 page)
   - Self-play methods (SP, FSP, PSRO)
   - Belief states in PE (G&I, others)
   - Multi-agent RL for robotics

III. Preliminaries (0.5 page)
   - POSG formulation
   - MADDPG background
   - BiMDN belief states

IV. Method: Pop-FSP (2 pages)
   - Algorithm description
   - Population management
   - Integration with BiMDN
   - Theoretical analysis

V. Experiments (2.5 pages)
   - Setup and baselines
   - Results: oscillation reduction
   - Results: single-checkpoint robustness
   - Ablations: K, sampling, pruning
   - Diversity analysis

VI. Discussion & Limitations (0.5 page)
   - Computational cost of population
   - Scaling to >2 agents
   - Real robot deployment

VII. Conclusion (0.25 page)
```

---

## 7. Timeline

| Phase | Duration | Deliverable | Deadline |
|-------|----------|-------------|----------|
| **Phase 1: Implementation** | 3 weeks | Pop-FSP code, unit tests | June 8 |
| **Phase 2: Baseline Runs** | 2 weeks | Vanilla + EMA + Pop-Only results | June 22 |
| **Phase 3: Main Experiments** | 3 weeks | Pop-FSP full results, ablations | July 13 |
| **Phase 4: Analysis & Theory** | 2 weeks | Diversity analysis, theorem proofs | July 27 |
| **Phase 5: Writing** | 3 weeks | Full draft, figures, tables | Aug 17 |
| **Phase 6: Revision** | 2 weeks | Internal review, polish | Aug 31 |
| **Buffer** | 2 weeks | Unexpected issues | Sep 14 |
| **ICRA Deadline** | — | Submission | ~Sep 15 |

**Total: ~17 weeks (4 months)**

### 7.1 Phase 1 Breakdown (Implementation)

**Week 1:**
- [ ] Modify MADDPG update to accept external opponent
- [ ] Implement PopulationManager class
- [ ] Unit tests for population sampling

**Week 2:**
- [ ] Modify training loop for population updates
- [ ] Implement EMA opponent baseline
- [ ] Unit tests for full training loop

**Week 3:**
- [ ] Smoke tests: 500-episode runs for all baselines
- [ ] Debug and fix issues
- [ ] Launch full baseline runs on niro-2

---

## 8. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Pop-FSP doesn't reduce oscillation | Medium | High | Have EMA-only as fallback; if both fail, pivot to analysis paper explaining why oscillation is fundamental |
| Computational cost too high (K× memory) | Low | Medium | K=8 is small; networks are tiny (~1MB each); can use CPU for sampled opponents |
| BiMDN integration bugs | Medium | High | Extensive unit tests; compare with vanilla SP first |
| Training takes longer than expected | Medium | Medium | Run 5 seeds in parallel on niro-2; reduce to 3 seeds if needed |
| Theory proofs are too hard | Medium | Medium | Focus on empirical convergence; include proof sketches rather than full proofs |
| ICRA deadline moves earlier | Low | High | Front-load implementation; have draft ready by Aug 1 |

---

## 9. Success Criteria

**Minimum viable result (must achieve):**
- Pop-FSP reduces oscillation amplitude by >30% vs vanilla
- Single-checkpoint robustness score >0.40 (vanilla is ~0.25)

**Strong result (target):**
- Oscillation amplitude reduced by >50%
- Single checkpoint achieves: self-play ≥0.50, vs-random ≥0.90, vs-greedy ≥0.90
- Formal convergence analysis with proofs

**Home run (stretch):**
- Single checkpoint matches ALL G&I metrics simultaneously
- Novel theoretical result (convergence rate bound)
- Best Paper nomination

---

## 10. Next Steps (This Week)

1. **Review and approve this plan** — Does the scope and direction look right?
2. **Implement PopulationManager** — Start with simple FIFO, uniform sampling
3. **Modify MADDPG update** — Accept external opponent actor
4. **Smoke test** — 500-episode run with K=2 to verify mechanics
5. **Launch baseline runs** — Vanilla SP for 1 seed, 20K episodes (for comparison)

---

## 11. Open Questions

1. Should we use **FSP** (average policy weights) or **PSRO** (Nash mixture) instead of uniform population sampling?
2. Should the population include **checkpoints from multiple seeds** for extra diversity?
3. Should we add a **diversity bonus** to the actor loss (maximize distance from population members)?
4. Should we **freeze** population members or allow them to continue training?
5. How do we handle **BiMDN state** when sampling old opponents? (The BiMDN history is episode-specific, so we only need the network weights, not the history.)

---

*Plan created: 2026-05-18*
*Status: Draft — awaiting review*
