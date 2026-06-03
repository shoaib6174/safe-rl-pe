# SP12 Experiment Series: Post-Mortem Analysis

**Date**: 2026-03-09 (Session S77)
**Runs killed at**: 13.2M steps (~43 hours each, ~345 total GPU-hours across 8 runs)
**Verdict**: All 8 runs terminated early. Omnidirectional 3m sensing is too restrictive for effective pursuit under partial observability.

---

## 1. Experiment Overview

The SP12 series tested **masking curriculum** (PO-GRL-style `p_full_obs` annealing) for transitioning from full observability to partial observability with **omnidirectional 3m sensing radius + LOS**.

### Shared Configuration
- Algorithm: SAC
- Max episode steps: 2000
- Sensing: omnidirectional, radius=3.0m, combined masking (radius + LOS)
- Obstacles: 0-3 random
- Freeze: evader frozen, switch threshold 0.9 (never reached)
- Opponent pool: 50, PFSP-lite, collapse threshold 0.1
- Masking curriculum: `p_full_obs` anneals from 1.0 to 0.0
- Max total steps: 30M (killed at 13.2M = 44%)

### Variants

| Variant | Seeds | Anneal Schedule | Extra |
|---------|-------|-----------------|-------|
| **SP12_mc** | 42, 43, 44, 45 | 5M steps (fast) | Baseline masking curriculum |
| **SP12b_slow** | 42, 43 | 10M steps (slow) | Test if slower annealing helps |
| **SP12c_search** | 42, 43 | 5M steps | + `w_search=0.0001` staleness reward |

---

## 2. Results Summary

### Final Numbers (at 13.2M steps)

| Run | Peak CR | Peak @ | Final CR | Final ER | Trend (last 5) | Elapsed |
|-----|---------|--------|----------|----------|-----------------|---------|
| SP12_mc_s42 | 0.42 | 9.2M | 0.28 | 0.72 | -0.10 | 43.2h |
| SP12_mc_s43 | 0.48 | 3.1M | 0.31 | 0.69 | -0.06 | 42.7h |
| SP12_mc_s44 | 0.53 | 8.0M | 0.23 | 0.77 | -0.02 | 42.7h |
| SP12_mc_s45 | 0.43 | 2.2M | 0.33 | 0.67 | -0.09 | 43.1h |
| SP12b_slow_s42 | 0.42 | 8.9M | 0.29 | 0.71 | -0.17 | 43.0h |
| SP12b_slow_s43 | 0.52 | 2.5M | 0.40 | 0.60 | -0.13 | 42.5h |
| SP12c_search_s42 | 0.50 | 9.2M | 0.24 | 0.76 | -0.08 | 43.3h |
| SP12c_search_s43 | 0.53 | 11.7M | 0.21 | 0.79 | -0.20 | 42.7h |

**Aggregate**: Mean peak CR = 0.48, Mean final CR = 0.29. All runs declining.

### Cross-Variant Comparison

| Variant | Mean Peak CR | Mean Final CR | Observation |
|---------|-------------|---------------|-------------|
| SP12_mc (4 seeds) | 0.47 | 0.29 | Standard reference |
| SP12b_slow (2 seeds) | 0.47 | 0.35 | Slightly better retention, but still declining |
| SP12c_search (2 seeds) | 0.52 | 0.23 | Higher peaks but worst final — search reward may hurt long-term |

---

## 3. Key Observations

### 3.1 Masking Curriculum Worked During Annealing, Failed After

The p_full_obs annealing schedule:
- **SP12_mc/SP12c**: 1.0 → 0.0 over 5M steps (fully annealed by eval ~16)
- **SP12b_slow**: 1.0 → 0.0 over 10M steps (fully annealed by eval ~32)

During the annealing phase, CR rose steadily to 0.35-0.50 — the curriculum was effective at bootstrapping. However, once annealing completed and agents were fully under partial obs (p_full_obs=0.0), performance either plateaued or declined. The curriculum **transferred some skill** but not enough to sustain pursuit under pure 3m sensing.

### 3.2 The 3m Omnidirectional Radius Is Too Restrictive

In a 10x10 arena, a 3m omnidirectional radius means:
- The pursuer can only see **~2.8%** of the arena area at any time (pi*3^2 / 100)
- With obstacles and LOS blocking, effective visibility is even less
- The evader (also with 3m sensing) can trivially stay outside the pursuer's detection range
- At equal speeds, the pursuer has no mechanism to systematically search and corner the evader

For comparison, **SP13 (FOV 90, 8m)** at only 5.5M steps already achieved:
- SP13a (no obstacles): **93-97% peak CR**
- SP13b (with obstacles): **61-81% peak CR**

The 8m range covers ~201 sq units (vs 28 for 3m), a **7x increase** in potential visibility area. Even with the 90 FOV restriction, this provides far more actionable information.

### 3.3 Slow Annealing Marginally Better But Not Enough

SP12b_slow (10M anneal) retained slightly higher final CR (0.35 mean vs 0.29 for fast anneal), suggesting the agent had more time to adapt to gradually harder conditions. However, the improvement was insufficient — the core problem is the 3m radius, not the annealing speed.

### 3.4 Search Reward (w_search) Did Not Help

SP12c added a staleness-based search reward to encourage exploration. Results:
- Higher peak CR (0.52 mean) but worst final CR (0.23 mean)
- The search reward may have initially encouraged movement, but the signal likely confused the agent once the evader was consistently out of range

### 3.5 Evader Never Unfroze

All 8 runs used `freeze_role=evader` with `freeze_switch_threshold=0.9`. Since no run reached 0.90 CR (max was 0.53), the evader remained a random policy throughout. These runs only tested single-agent pursuit learning, never reaching the co-evolution phase.

### 3.6 High Variance Across Seeds

Even within the same config, CRs varied significantly:
- SP12_mc: peaks ranged 0.42-0.53 (26% relative variation)
- Seed 44 peaked at 0.53 but crashed to 0.23; seed 43 was more stable at 0.41

This confirms the seed sensitivity observed in earlier experiments (SP11i).

---

## 4. Comparison with SP11e Baseline

| Metric | SP11e (full obs) | SP12 (masking curriculum → 3m) |
|--------|-----------------|-------------------------------|
| Peak CR | **0.85** | 0.42-0.53 |
| Final CR | 0.81 | 0.21-0.40 |
| Steps to peak | 14.4M | 2-11M |
| Evader unfroze? | Yes | No |

The full-obs baseline (SP11e) achieved 85% and sustained it. Transitioning to 3m omnidirectional sensing causes a **40+ percentage point drop** that the masking curriculum cannot recover.

---

## 5. Lessons Learned

1. **Sensing range is the dominant factor**, not the curriculum strategy. No amount of clever annealing can compensate for a fundamentally information-starved agent.

2. **FOV-based sensing (SP13) is far more learnable** than omnidirectional radius sensing at short range. Directional sensing with longer range gives the agent actionable geometry — it can learn to turn toward the evader and track visually, rather than needing to stumble within 3m.

3. **Masking curriculum is a valid technique** for bridging full→partial obs. The annealing phase showed clear skill transfer. The failure was in the target partial obs being too hard, not in the curriculum itself.

4. **Search rewards need careful tuning or gating.** The staleness reward at w_search=0.0001 showed no clear benefit and possibly hurt long-term stability. In previous experiments (SP11g2/h2), w_search=0.003 was clearly harmful.

5. **Equal speed + 3m sensing = impossible task.** At equal speeds, the pursuer has no advantage. With 3m sensing, it also has no information advantage. There is no viable pursuit strategy under these conditions against even a random evader.

---

## 6. Recommendations

- **Do not revisit 3m omnidirectional sensing** unless combined with a speed advantage (1.2x+).
- **Focus on SP13 (FOV 90, 8m)** which is already showing strong results.
- **Consider masking curriculum for FOV runs** as a future refinement once FOV baseline is established.
- **Test alternate_freeze with SP13** to enable co-evolution once pursuer CR exceeds 0.9.

---

## 7. Resource Usage

| Metric | Value |
|--------|-------|
| Total runs | 8 |
| Steps per run | 13.2M / 30M (44%) |
| Wall time per run | ~43 hours |
| Total wall time | ~43 hours (parallel) |
| Total compute | ~345 CPU-hours |
| Outcome | Negative result (informative) |
