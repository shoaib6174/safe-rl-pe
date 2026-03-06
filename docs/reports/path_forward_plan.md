# Path Forward Plan: Safe Deep RL for 1v1 Pursuit-Evasion

**Date**: 2026-03-06
**Current Phase**: Phase 3 — Partial Observability & Self-Play
**Companion**: See `comprehensive_project_report.md` for full project history

---

## 1. Situation Assessment

### What Works
- **Pursuer can learn search-and-capture** from scratch under partial obs (SP11e: 85% vs random evader)
- **SAC with proven hyperparams** (lr=3e-4, ent_coef=0.03) is stable for single-agent training
- **Freeze-then-unfreeze** is the only validated self-play curriculum (10% → 42%)
- **Combined masking** (3m sensing + LOS) creates a meaningful game with obstacles
- **Collapse rollback + PFSP** work mechanically (catch collapse, bias toward weaker opponents)

### What Doesn't Work
- **Co-evolution** at any ratio (1:1, 5:1, adaptive) — evader adapts too fast, pursuer collapses
- **Cold-start self-play** with full obs — bootstrapping fails (0-7%)
- **RecurrentPPO** — 0% across 6 variants, 3-10x slower
- **Search reward at w=0.003** — competing signal, teaches patrol not hunt
- **Warm-seeding without replay buffer** — catastrophic Q-value degradation
- **Equal speed** in any self-play config — degenerate hide-and-seek

### The Core Bottleneck
The pursuer is competent at search-and-capture against a random/frozen evader, but **cannot maintain performance when the evader starts learning**. This is because:
1. The pursuer's search strategy is fragile — any minor evader intelligence breaks it
2. Sparse reward under partial obs means the pursuer gets zero gradient when capture rate drops
3. The evader only needs to learn ONE thing (hide/evade) while the pursuer must learn TWO things (search + capture)

---

## 2. Proposed Strategy: Masking Curriculum + Graduated Co-Evolution

The fundamental insight is that the pursuer's bottleneck is trying to learn search and pursuit simultaneously. We should separate these challenges.

### Phase A: Masking Curriculum for Strong Pursuer (Priority 1)

**Goal**: Train a pursuer that has mastered PURSUIT (catching a visible evader) and only needs to improve at SEARCH.

**Approach**: PO-GRL-style smooth annealing (researched in S69, implementation plan ready)
- Start with p_full = 1.0 (full observability override — pursuer always sees evader)
- Linearly anneal p_full → 0.0 over first 5M agent training steps
- Per-timestep probability: each step, with probability p_full, override masking and give full obs
- Observation format stays constant (15 + 2K dims) — just populate evader features when overriding
- Evals ALWAYS run at p_full = 0 (true masking)

**Expected outcome**: Pursuer learns interception/capture during full-obs phase, then gradually adapts to searching. This is strictly easier than learning both simultaneously (which SP11e managed with lucky seeds).

**Config**:
```bash
python scripts/train_amsdrl.py \
  --algorithm sac \
  --freeze_role evader \
  --freeze_switch_threshold 0.9 \
  --max_steps 2000 \
  --full_obs --partial_obs_los --sensing_radius 3.0 --combined_masking \
  --n_obstacles_min 0 --n_obstacles_max 3 --n_obstacle_obs 3 \
  --lr 3e-4 --ent_coef 0.03 \
  --collapse_threshold 0.1 --collapse_streak_limit 5 --pfsp \
  --masking_curriculum --p_full_start 1.0 --p_full_end 0.0 --p_full_anneal_steps 5000000 \
  --seed 42 --seed 43 --seed 44 --seed 45 \
  --max_total_steps 30000000
```

**Implementation effort**: ~2-3 hours (observation override in SingleAgentPEWrapper, CLI flags, per-agent step tracking)

**Validation**: If pursuer reaches 85%+ eval SR under TRUE masking (p_full=0), the curriculum worked. Compare with SP11e baseline (85% lucky seed, 30% typical seed).

### Phase B: Graduated Co-Evolution (Priority 2)

**Goal**: Safely introduce evader training without destroying the pursuer.

Once the pursuer is strong (>80% against random evader under true masking), begin co-evolution with extreme caution:

**Approach**: Ultra-conservative thawed evader training
1. **Start**: Evader gets 1 micro-phase per 20 pursuer micro-phases (train_ratio=20)
2. **Monitor**: If pursuer SR drops below 60%, freeze evader immediately and let pursuer recover
3. **Advance**: Every 2M steps, if pursuer SR > 70%, reduce ratio: 20:1 → 10:1 → 5:1 → 3:1
4. **Save best**: Best-model checkpointing for both agents, rollback if necessary

**Key difference from previous attempts**: The masking curriculum pursuer should be MUCH more robust because it learned pursuit fundamentals under full obs. Previous pursuers (SP11e) learned search-and-capture simultaneously, making their strategies brittle.

**Alternative**: If graduated co-evolution still fails:
- **Option B1**: Replay buffer warm-seeding — pre-fill SAC replay buffer with 100K transitions from current strong pursuer before resuming training
- **Option B2**: Q-network reset — load only the actor network, reinitialize the Q-networks (avoids stale value estimates)
- **Option B3**: Population-based (PSRO-lite) — maintain a pool of 5-10 best models per role, train against weighted mixture

### Phase C: Speed Advantage Introduction (Priority 3)

If equal-speed co-evolution produces degenerate equilibria (hide-and-seek), introduce 1.2x evader speed:
- S1v6b already showed this creates a viable game (35-50% escape against full-obs greedy)
- Speed advantage forces genuine pursuit-evasion dynamics instead of hide-and-seek
- BUT: this changes the game fundamentally — obstacle exploitation becomes less critical

**Decision point**: Only if equal-speed co-evolution fails after masking curriculum. Equal speed is scientifically more interesting (forces obstacle use).

---

## 3. Secondary Priorities

### 3a. Replay Buffer Pre-Fill for Warm-Seeding
**Problem**: SP11e2 showed warm-seeding SAC without replay buffer causes catastrophic forgetting.
**Solution**: When loading a trained SAC model, also pre-fill the replay buffer by running 100K inference steps with the loaded model.
**Effort**: ~1 hour (simple loop: run model, store transitions)

### 3b. Very Low Search Reward (w_search < 0.0003)
**Problem**: w_search=0.003 was harmful (50% of distance reward magnitude).
**Solution**: Try w_search=0.0001 or w_search=0.0003 — 10-30x lower.
**Effort**: Zero (just a CLI flag change)

### 3c. Eval Metrics Enhancement
Add diagnostic eval metrics to better understand pursuer behavior:
- **Search efficiency**: fraction of arena cells observed within first 500 steps
- **Detection-to-capture time**: steps from first evader detection to capture
- **Search pattern entropy**: how uniformly the pursuer covers the arena

### 3d. Visualization & Analysis Pipeline
- Automated trajectory analysis for each eval checkpoint
- Heat maps of pursuer coverage patterns
- Speed/angular velocity profiles over training

---

## 4. What NOT to Try

Based on extensive experimentation, these approaches are conclusively ruled out:

| Approach | Why | Evidence |
|----------|-----|----------|
| RecurrentPPO / LSTM | 0% across 6 variants, 3-10x slower | SP13a-f |
| ent_coef=auto for SAC | Destructive in self-play | SP11c (24% → 6%) |
| 1:1 co-evolution | Evader always wins the arms race | SP7, SP8 |
| Long frozen phases (>50K) | Over-exploitation, catastrophic forgetting | Runs H-Z2 |
| BarrierNet end-to-end | 87.5% QP intervention, 2% capture | Phase 2.5 |
| w_search=0.003 | Patrol behavior, not hunt | SP11g2/h2 |
| Cold-start PPO self-play | Bootstrapping failure | SP2-SP2d, RA1-RA10 |
| Equal speed without partial obs | Trivially solvable, no game | S53 analysis |
| 4+ obstacles in 10x10 arena | Too cluttered, evader trapped | D1, D2 |

---

## 5. Timeline Estimate

| Phase | Task | Sessions | Effort |
|-------|------|----------|--------|
| A1 | Implement masking curriculum | 1 session | 2-3 hours code |
| A2 | Launch 4-seed masking curriculum sweep | 1 session | 1 hour (4 runs on niro-2) |
| A3 | Monitor & analyze masking curriculum results | 2-3 sessions | Wait for 10-15M steps (~1 day) |
| B1 | Implement graduated co-evolution (if A succeeds) | 1 session | 2 hours code |
| B2 | Launch co-evolution experiments | 1 session | 1 hour |
| B3 | Monitor & iterate | 3-5 sessions | Depends on results |
| C | Speed advantage if needed | 1 session | Config change only |

**Realistic estimate**: 8-12 sessions over 4-7 days to determine if masking curriculum + graduated co-evolution works. If it does, we have a publishable result. If not, we pivot to population-based methods (PSRO) or accept the partial result.

---

## 6. Success Criteria

### Minimum Viable Result (publishable)
- Pursuer achieves **>70% capture rate** against a **trained evader** in co-evolution under combined partial observability with obstacles
- Training is stable (no collapse over 5M+ steps)
- Repeatable across 2+ seeds

### Stretch Goal
- Balanced co-evolution: pursuer capture rate stabilizes at 40-60% (genuine Nash equilibrium)
- Both agents exhibit diverse strategies (not single-strategy dominance)
- Can deploy with post-hoc VCP-CBF safety filter without significant performance degradation

### Ultimate Goal (Phase 4-5)
- Sim-to-real transfer to physical TurtleBot3 robots
- Safety guarantees maintained during real-world operation
- Publication-quality results

---

## 7. Immediate Next Steps (Next Session)

1. **Check multi-seed sweep results** — SP11i seeds 42-45 should have 1-3M steps by now
2. **Implement masking curriculum** — p_full annealing in SingleAgentPEWrapper
3. **Launch masking curriculum runs** — 4 seeds, 30M steps each
4. **If SP11i shows a high-performing seed**: use it for warm-seeded co-evolution experiments in parallel

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Masking curriculum doesn't help | Medium | High | Fall back to high-seed-count strategy (run 10+ seeds, keep best) |
| Co-evolution still collapses | High | High | Try replay buffer pre-fill, Q-network reset, or PSRO-lite |
| CUDA non-determinism makes results unreliable | High | Medium | Always run 4+ seeds, report distributions not point estimates |
| Disk space on niro-2 | Medium | Medium | Regular cleanup, rotate old checkpoints |
| Time pressure | Low | Medium | Focus on masking curriculum only if time is short — it's the most promising single intervention |
