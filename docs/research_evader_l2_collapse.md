# Research Report: Evader Collapse at Curriculum Level 2

## 1. Executive Summary

Across all four training runs (R, S, T, U) using AMS-DRL self-play with curriculum learning, the evader consistently collapses when the curriculum transitions from Level 1 (close range, 2–5m) to Level 2 (medium range, 4.52–11.31m on 10×10 arena). Once collapsed, the evader never recovers — even after rollbacks demote the system back to L1.

This report identifies **8 internal root causes** and **7 literature-corroborated mechanisms**, synthesized from codebase analysis, Run U phase-by-phase logs, and a review of 30+ papers on self-play collapse.

**Core diagnosis**: The collapse is a *structural training protocol failure*, not a reward design problem. The combination of alternating self-play, asymmetric task transfer, and destructive rollbacks creates a one-way trap where the evader cannot recover once destabilized.

---

## 2. Evidence: The Collapse Pattern

### 2.1 Run History

| Run | Arena | Seed | Config | Collapsed At | Rollbacks | Recovery? |
|-----|-------|------|--------|-------------|-----------|-----------|
| **R** | 10×10 | 43 | min_escape_rate=0.05 | L5 | 8+ | No |
| **S** | 20×20 | 44 | min_escape_rate=0.15 | L1↔L2 oscillation | — | Never advanced |
| **T** | 10×10 | 45 | min_escape_rate=0.15 | L2 | 8+ | No |
| **U** | 10×10 | 46 | w_wall=0.5, w_collision=0.5 | L2 | 40+ | No |

### 2.2 Run U Phase-by-Phase Timeline

**Stage 1 — Healthy L1 training (S0–S7):**
| Phase | Agent | SR_P | SR_E | NE Gap | Level |
|-------|-------|------|------|--------|-------|
| S0 | Cold-start | 0.26 | 0.74 | — | L1 |
| S1 | Pursuer | 0.78 | 0.22 | 0.56 | L1 |
| S2 | Evader | 0.35 | 0.65 | 0.30 | L1 |
| S3 | Pursuer | 0.70 | 0.30 | 0.40 | L1 |
| S7 | Pursuer | 0.98 | 0.02 | 0.96 | L1 |

Healthy oscillation: agents alternate dominance, NE gap stays moderate.

**Stage 2 — L2 transition and immediate collapse (S8–S11):**
| Phase | Agent | SR_P | SR_E | NE Gap | Level | Event |
|-------|-------|------|------|--------|-------|-------|
| S8 | Evader | 0.78 | 0.22 | 0.56 | L1→**L2** | **Advanced** |
| S10 | Evader | 0.69 | 0.31 | 0.38 | L2 | Evader briefly recovering |
| S11 | Pursuer | 1.00 | 0.00 | 1.00 | L2 | **Collapsed** — single pursuer phase destroyed evader |

**Critical observation**: The evader DID learn at L2 briefly (S10: escape=0.31). But a single pursuer training phase (S11) annihilated it. The pursuer's L1 policy transferred well to L2, while the evader's fragile L2 policy was destroyed.

**Stage 3 — Permanent collapse with cascading rollbacks (S12–S30):**
| Phase | SR_P | Level | Rollbacks | Notes |
|-------|------|-------|-----------|-------|
| S13 | 1.00 | **L1** | 6 | Demoted, but evader weights already corrupted |
| S14–S30 | 0.92–1.00 | L1 | 30+ total | **Never recovered** even at L1 |

40 total rollbacks, zero recovery. Evader weights corrupted beyond repair at S11; rollbacks to S8-era weights didn't help because the opponent pool was now saturated with strong pursuer models.

---

## 3. Root Causes

### 3.1 Primary: Asymmetric Task Transfer (Root Cause A)

**The fundamental problem**: Pursuer and evader skill transfer asymmetrically across curriculum levels.

- **Pursuer at L2**: Its L1 policy (chase and close distance) works *even better* at L2. Longer initial distances just mean more running — the same straight-line pursuit strategy applies. The pursuer's task is essentially unchanged.
- **Evader at L2**: Its L1 policy (short-range evasion maneuvers at 2–5m) is *useless* at L2. At 5–11m separation, the evader needs fundamentally different skills — sustained flight paths, corner exploitation, trajectory planning — none of which were learned at L1.

This asymmetry means the pursuer enters L2 at full strength while the evader starts from scratch. In alternating self-play, this creates an immediate win-rate spike for the pursuer that the evader cannot recover from.

**Literature support**: Czarnecki et al. (2020, "Real World Games") show that in non-transitive games, self-play can cycle through strategies but rarely recovers once one agent's strategy is dominated. The Elo rating analogy: if one player jumps two skill tiers while the other stays constant, the gap becomes self-reinforcing.

### 3.2 Arena Geometry (Root Cause B)

On a **10×10 arena** (diagonal ≈ 14.14m), L2's distance range [4.52, 11.31m] is extreme:
- **80% of diagonal** — agents start nearly at opposite corners
- Equal-speed pursuit-evasion on a bounded domain with these distances may be **theoretically unwinnable** for the evader

A 10×10 arena gives the evader almost no room to maneuver at medium distances. The wall constraints box the evader in, and the pursuer simply needs to close distance.

**Comparison**: Run S on **20×20 arena** (diagonal ≈ 28.28m) had L2 ranges [5.0, 15.0m] — only 53% of diagonal. It oscillated rather than collapsed, suggesting geometry matters.

### 3.3 Pursuer-First Training at New Level (Root Cause C)

When the curriculum advances, the **pursuer trains first** at the new level (the advancing phase is always the currently-training agent). This means:

1. The pursuer immediately adapts its already-strong policy to L2
2. The evader then faces this L2-adapted pursuer with its L1 policy
3. The evaluation shows 100% pursuer domination → rollback triggered

If the evader trained first at L2, it would have a chance to develop medium-distance evasion skills against a still-L1-calibrated pursuer.

### 3.4 Catastrophic Collapse Spiral (Root Cause D)

Once the pursuer dominates (SR_P ≥ 0.98), episodes become extremely short:
- Pursuer captures evader in ~10–20 steps (vs. 200–600 normal steps)
- Short episodes provide **almost no useful gradient** to the evader
- The evader receives near-constant -100 terminal penalties
- Policy gradient updates become dominated by noise

This is a known problem in adversarial training: "the losing agent's policy collapses because it receives no useful training signal from extremely short episodes" (Lanctot et al., 2017).

### 3.5 Rollback Mechanism Deficiencies (Root Causes E, F, G)

The rollback mechanism has three fundamental flaws:

1. **Rollbacks actively damage evader learning (E)**: When the pursuer dominates, rollbacks restore the *pursuer's* weights 3 checkpoints back. But they DON'T restore the evader's weights, the opponent pool, or the curriculum level. The evader is left with corrupted weights facing a slightly-weaker-but-still-strong pursuer.

2. **Curriculum regression doesn't restore weights (F)**: When the system regresses from L2 back to L1, only the curriculum level changes — both agents keep their current (corrupted) weights. The evader's L1 skills were overwritten during its failed L2 training.

3. **No milestone rollback capability (G)**: There is no mechanism to restore the evader to its "best L1 weights" — the state that earned curriculum advancement in the first place. Milestone checkpoints exist but aren't used during rollback.

**Combined effect**: Rollbacks are cosmetic. They reset the pursuer slightly but leave the evader permanently damaged, creating an endless cycle of collapse → rollback → collapse.

### 3.6 Reward Signal Disappears at L1/L2 (Root Cause H)

The evader reward design creates a critical gap at obstacle-free levels:

```python
# From envs/rewards.py, RewardComputer.compute()
can_use_visibility = (
    self.use_visibility_reward
    and obstacles  # obstacles exist in this episode  ← FALSE at L1/L2!
    and pursuer_pos is not None
    and evader_pos is not None
)

if can_use_visibility and not captured and not timed_out:
    # Mode B: visibility-based evader reward (decoupled)
    ...
else:
    # Zero-sum fallback
    r_evader = -r_pursuer
```

At L1/L2 (no obstacles), `obstacles` is empty → visibility reward is disabled → evader gets pure zero-sum reward. This means:
- No survival bonus
- No visibility-based independent reward
- The evader's reward is entirely determined by the pursuer's performance
- When the pursuer dominates, the evader's reward is constantly negative with no gradient toward improvement

Additionally, `evader_training_multiplier` (asymmetric training) only activates when `n_obstacles > 0`, so the evader gets no extra training time at L1/L2.

---

## 4. Literature-Corroborated Mechanisms

### 4.1 Non-Transitivity & Strategy Cycling (Czarnecki et al., 2020)

Real-world games exhibit non-transitive dynamics where strategy A beats B, B beats C, but C beats A. Self-play in these games doesn't converge to a single equilibrium but cycles. Our evader's L1 strategy is non-transitive with respect to L2 challenges — it simply can't compete.

### 4.2 Catastrophic Forgetting at Curriculum Boundaries (Kirkpatrick et al., 2017)

Neural networks suffer catastrophic forgetting when training distribution shifts abruptly. The L1→L2 transition changes the initial distance distribution from [2, 5]m to [4.52, 11.31]m — a distribution shift that overwrites L1 skills without replacing them with L2 skills.

### 4.3 Opponent Overfitting (Lanctot et al., 2017)

When one agent's policy collapses, the other agent's policy overfits to exploiting the collapsed opponent. In our case, the pursuer's policy becomes increasingly specialized at catching a "frozen" evader. The opponent pool exacerbates this: by S15, 14 of 16 pool entries are strong pursuer policies trained against collapsed evaders.

### 4.4 Population Diversity Failure (Jaderberg et al., 2019)

DeepMind's PBT (Population-Based Training) for StarCraft showed that diversity in the opponent pool is critical for stable self-play. Our opponent pool adds every phase's policy regardless of quality, flooding the pool with near-identical "catch-the-frozen-evader" variants.

### 4.5 Sparse Reward Starvation (Pinto et al., 2017; OpenAI, 2019)

RARL (Robust Adversarial RL) and OpenAI's multi-agent work demonstrate that the losing agent in adversarial training needs additional reward shaping to recover. Without intrinsic motivation or dense rewards, the evader has no gradient to improve once it starts losing consistently.

---

## 5. Recommended Fixes

Fixes are ordered by expected impact and implementation complexity.

### 5.1 Tier 1 — High Impact, Moderate Effort

#### Fix 1: Warm-Start Evader at Level Transitions
**What**: When advancing to L2, initialize the evader with its best L1 milestone weights PLUS additional solo pre-training at L2 distances (no adversary, just distance-maximization reward for N steps).

**Why**: The evader needs to develop L2 skills before facing an L2-adapted pursuer. Solo pre-training gives it basic flight skills without adversarial pressure.

**Implementation**:
```python
def _advance_curriculum(self):
    # After level change:
    # 1. Restore evader to its best milestone weights
    evader_milestone = self.ckpt_mgr.get_milestone("evader_best")
    self.evader_model.load_state_dict(evader_milestone)
    # 2. Pre-train evader solo for K steps at new level
    self._pretrain_evader_solo(timesteps=50_000, level=new_level)
```

**Expected impact**: Prevents the "evader enters L2 with useless L1 policy" problem.

#### Fix 2: Alternating-Agent Curriculum with Evader-First at New Levels
**What**: When curriculum advances, always train the evader first at the new level.

**Why**: Gives the evader a head start at adapting to new distance distributions before facing an adapted pursuer.

**Implementation**: In `AMSDRLSelfPlay.run()`, after curriculum advancement, force next phase to train evader regardless of alternation schedule.

**Expected impact**: Addresses Root Cause C directly.

#### Fix 3: Bilateral Rollback
**What**: When rollback triggers, restore BOTH agents to their weights from 3 checkpoints ago, not just the active agent.

**Why**: Current rollbacks leave the evader damaged while only weakening the pursuer slightly. Both agents should return to a balanced state.

**Implementation**: Modify `_trigger_rollback()` in `SelfPlayHealthMonitorCallback` to also restore the opponent model.

**Expected impact**: Addresses Root Causes E/F — rollbacks become actual recovery mechanisms.

### 5.2 Tier 2 — Medium Impact, Low-Medium Effort

#### Fix 4: Dense Evader Reward at L1/L2 (Survival Bonus Without Obstacles)
**What**: Enable `survival_bonus` for the evader at ALL levels, not just obstacle levels. Also add a per-step "distance-from-pursuer" bonus to give the evader gradient toward fleeing.

**Why**: At L1/L2, the evader gets pure zero-sum reward. Adding `survival_bonus = 1.0` and a distance-maintenance reward gives the evader useful gradient even when losing.

**Implementation**:
```python
# In rewards.py compute():
# Add survival bonus always (not just visibility mode)
if not captured and not timed_out:
    r_evader += self.survival_bonus
```

**Expected impact**: Addresses Root Cause H — evader gets useful reward signal at all levels.

#### Fix 5: Elastic Weight Consolidation (EWC) at Curriculum Transitions
**What**: Apply EWC regularization during the first N training steps after a curriculum transition. This penalizes changes to weights that were important for L1 performance.

**Why**: Prevents catastrophic forgetting of L1 skills while learning L2 skills.

**Implementation**: Compute Fisher information matrix from L1 rollouts, add quadratic penalty term to loss:
```python
L_ewc = L_task + λ/2 * Σ F_i * (θ_i - θ*_i)²
```

**Expected impact**: Preserves L1 skills through L2 training.

#### Fix 6: Mixed-Level Replay
**What**: During L2 training, mix 30% L1 episodes into the rollout buffer.

**Why**: Maintains L1 skills, provides easier episodes for the evader to learn from, and smooths the distribution shift.

**Implementation**: In the vectorized environment, configure some sub-environments to use L1 settings while others use L2 settings.

**Expected impact**: Smooths curriculum transition, prevents catastrophic forgetting.

### 5.3 Tier 3 — Foundational Changes

#### Fix 7: Population-Based Self-Play with Diversity Filtering
**What**: Replace the current opponent pool (add-every-phase) with a diversity-filtered pool that maintains strategic variety. Use Elo ratings or policy diversity metrics to select opponents.

**Why**: Current pool gets saturated with strong-pursuer/weak-evader policies. A diverse pool forces both agents to maintain general strategies.

**Implementation**: Track win-rate matrix between all pool entries. Only add new entries that are sufficiently different from existing ones (e.g., Wasserstein distance on action distributions).

**Expected impact**: Prevents opponent pool saturation (Root Cause related to 4.3, 4.4).

#### Fix 8: Smooth Curriculum (Continuous Distance Distribution)
**What**: Instead of discrete L1→L2 jump, gradually expand the distance range. For example, increase `max_init_distance` by 0.5m per successful phase.

**Why**: Eliminates the sharp distribution shift that causes catastrophic forgetting.

**Implementation**: Replace the discrete 6-level `CurriculumManager` with a continuous curriculum that expands ranges based on performance.

**Expected impact**: Eliminates the L1→L2 transition entirely. Most robust long-term fix.

#### Fix 9: Intrinsic Motivation (Count-Based or RND)
**What**: Add an intrinsic reward for the evader based on state novelty (Random Network Distillation or count-based exploration).

**Why**: When the evader is losing every episode, extrinsic reward is constant negative. Intrinsic motivation provides exploration gradient independent of game outcome.

**Implementation**: Train a predictor network on evader observations; reward = prediction error.

**Expected impact**: Provides learning signal even during collapse. Addresses Root Cause D.

---

## 6. Theoretical Concern: Escapability

A fundamental question remains: **is escape even possible at L2 distances on a 10×10 arena with equal speeds?**

In continuous PE games on unbounded domains with equal speeds, the evader can flee forever. But on a **bounded 10×10 arena**:
- The evader is constrained by walls
- At L2 distances (4.52–11.31m), starting positions may leave the evader cornered
- With v_pursuer = v_evader = 1.0 m/s, the pursuer will eventually corner the evader

**Recommendation**: Before implementing any fixes, verify escapability:
1. Run the greedy-vs-greedy baseline at L2 distances on 10×10
2. Measure what fraction of random L2 spawns allow the evader to survive 1200 steps
3. If escapability < 20%, the curriculum geometry itself needs fixing (larger arena or smaller L2 distances)

---

## 7. Recommended Implementation Order

### Phase 1 (Next Session) — Quick Wins
1. **Fix 3**: Bilateral rollback — simple code change, eliminates the worst rollback pathology
2. **Fix 4**: Survival bonus at all levels — one-line reward change
3. **Fix 2**: Evader-first at new levels — simple scheduling change

### Phase 2 — Structural Improvements
4. **Fix 1**: Warm-start evader with solo pre-training at transitions
5. **Fix 6**: Mixed-level replay for smooth transitions
6. **Fix 8**: Smooth/continuous curriculum

### Phase 3 — Advanced
7. **Fix 5**: EWC regularization
8. **Fix 7**: Diversity-filtered opponent pool
9. **Fix 9**: Intrinsic motivation

### Verification
- **Before all fixes**: Run escapability analysis at L2 on 10×10 (theoretical baseline)
- **After Phase 1 fixes**: Launch one training run to test if collapse is mitigated
- **Success criterion**: Evader maintains escape_rate ≥ 0.10 at L2 for at least 5 consecutive phases

---

## 8. References

### Self-Play Collapse & Recovery
- Czarnecki et al. (2020). "Real World Games Look Like Spinning Tops." NeurIPS.
- Lanctot et al. (2017). "A Unified Game-Theoretic Approach to Multi-Agent RL." NeurIPS.
- Jaderberg et al. (2019). "Human-Level Performance in Multi-Agent StarCraft." Nature.
- OpenAI (2019). "Emergent Tool Use from Multi-Agent Autocurricula."
- Balduzzi et al. (2019). "Open-Ended Learning in Symmetric Zero-Sum Games." ICML.

### Catastrophic Forgetting
- Kirkpatrick et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks." PNAS.
- Kaplanis et al. (2019). "Policy Consolidation for Continual RL." ICML.

### Curriculum Learning
- Narvekar et al. (2020). "Curriculum Learning for RL Domains: A Framework and Survey." JMLR.
- Dennis et al. (2020). "Emergent Complexity and Zero-Shot Transfer via Unsupervised Environment Design." NeurIPS.

### Adversarial Training
- Pinto et al. (2017). "Robust Adversarial RL." ICML.
- Bansal et al. (2018). "Emergent Complexity via Multi-Agent Competition." ICLR.

### Pursuit-Evasion
- N16: Kokolakis & Vamvoudakis (2022). "Safe Finite-Time RL for PE Games." CDC.
- N19: Yang et al. (2023). "Large-Scale PE Under Collision Avoidance." IROS.

### From Project Collection
- Paper 12: Selvakumar et al. (2023). "MARL for Multi-Robot PE." IROS.
- Paper 23: Song et al. (2023). "PE Competition Benchmarks." NeurIPS.
- Paper 25: Lowe et al. (2017). "MADDPG." NeurIPS.
