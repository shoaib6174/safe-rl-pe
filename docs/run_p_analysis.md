# Run P Training Dynamics Analysis

**Date**: 2026-02-24 | **Sessions**: S45 (implementation), S47 (analysis)
**Server**: niro-2 (RTX 5090) | **Runtime**: 2.3 hours | **Result**: Did NOT converge

## Configuration

```
--max_phases 18 --timesteps_per_phase 300000 --no_dcbf
--distance_scale 10.0 --pursuer_v_max 1.1 --fixed_speed
--arena_width 10.0 --arena_height 10.0 --max_steps 600
--n_envs 16 --n_steps 2048 --batch_size 512
--curriculum --opponent_pool_size 5
--use_visibility_reward --survival_bonus 1.0 --prep_steps 100
--w_obs_approach 10.0 --seed 42
--output results/stage3/run_p
```

New vs Run O: `--w_obs_approach 10.0` (PBRS obstacle-seeking)

## Convergence Summary

| Metric | Value |
|--------|-------|
| Converged | **No** |
| Final NE gap | 0.94 (target: < 0.10) |
| Min NE gap | 0.18 (Phase S1) |
| Mean NE gap | 0.80 |
| NE gap trend | Increasing/flat |
| Final capture rate | 0.97 |
| Final escape rate | 0.03 |
| Total phases | 19 (S0-S18) |
| Health rollbacks | 25 (all in pursuer phases S7-S17) |

## Phase-by-Phase Results

| Phase | Role | SR_P | SR_E | NE gap | Level | Curriculum Event |
|-------|------|------|------|--------|-------|-----------------|
| S0 | Cold-start | 0.41 | 0.59 | — | 1 | — |
| S1 | Pursuer | 0.40 | 0.60 | 0.20 | 1 | — |
| S2 | Evader | 0.18 | 0.82 | 0.64 | 1 | — |
| S3 | Pursuer | 0.77 | 0.23 | 0.54 | 1 | **L1→L2** (0.77 > 0.70) |
| S4 | Evader | 0.80 | 0.20 | 0.60 | 2 | **L2→L3** (0.80 > 0.70) |
| S5 | Pursuer | 0.92 | 0.08 | 0.84 | 3 | **L3→L4** (0.92 > 0.70) |
| S6 | Evader | 0.97 | 0.03 | 0.94 | 4 | **L4→L5** (0.97 > 0.70) |
| S7 | Pursuer | 0.99 | 0.01 | 0.98 | 5 | **L5→L6** (0.99 > 0.70) |
| S8 | Evader | 0.98 | 0.02 | 0.96 | 6 | — |
| S9 | Pursuer | 1.00 | 0.00 | 1.00 | 6 | 4 rollbacks |
| S10 | Evader | 0.89 | 0.11 | 0.78 | 6 | Best L6 evader |
| S11 | Pursuer | 0.94 | 0.06 | 0.88 | 6 | 4 rollbacks |
| S12 | Evader | 0.93 | 0.07 | 0.86 | 6 | — |
| S13 | Pursuer | 1.00 | 0.00 | 1.00 | 6 | 5 rollbacks |
| S14 | Evader | 0.97 | 0.03 | 0.94 | 6 | — |
| S15 | Pursuer | 0.99 | 0.01 | 0.98 | 6 | 6 rollbacks |
| S16 | Evader | 1.00 | 0.00 | 1.00 | 6 | Worst evader phase |
| S17 | Pursuer | 1.00 | 0.00 | 1.00 | 6 | 5 rollbacks |
| S18 | Evader | 0.97 | 0.03 | 0.94 | 6 | — |

## Root Cause Analysis

### 1. Curriculum Raced Past Obstacle Levels

The curriculum advanced from Level 1 to Level 6 in 5 consecutive phases (S3-S7). The advancement gate (`curriculum.py:check_advancement()`) only checks pursuer capture_rate > 0.70 and is evaluated after EVERY phase (both pursuer and evader).

**Critical finding**: The evader NEVER trained at Level 3 (close + obstacles). L3 existed for exactly one phase — S5, which was a **pursuer** training phase. By S6 (evader's turn), the level was already L4.

### 2. PBRS Had Zero Effective Training Time

PBRS activates only when obstacles exist (returns 0 when `nearest_obstacle_distance = inf`).

| Evader Phase | Level | Obstacles | PBRS Active | Steps | Escape Rate After |
|-------------|-------|-----------|-------------|-------|-------------------|
| S0 (cold) | 1 | 0 | **No** | 200k | 0.59 |
| S2 | 1 | 0 | **No** | 300k | 0.82 |
| S4 | 2 | 0 | **No** | 300k | 0.20 |
| S6 | 4 | 3 | Yes | 300k | 0.03 |
| S8-S18 | 6 | 3 | Yes | 1.8M | 0.00-0.11 |

The evader trained for 800k steps with PBRS inactive (learning "flee in straight lines"). By the time PBRS activated (S6), the pursuer already had 92% capture rate.

### 3. PBRS Magnitude Insufficient

| Signal | Magnitude per Step |
|--------|-------------------|
| PBRS (w=10.0) | ~0.07 (for 0.1m approach, d_max=14.14) |
| Visibility reward | ±1.0 |
| Survival bonus | +1.0 |

PBRS gradient is **14x weaker** than visibility signal.

### 4. Evader Policy Never Converged

- Policy sigma increased from 1.03 to 1.30 (more exploration, not less — never found a coherent strategy)
- Episode lengths declined at L6 (384→377 steps) — pursuer catching evader faster over time
- Best evader moment: S10 (11% escape rate, reward spike to +134), but immediately regressed

### 5. Health Rollbacks Ineffective

25 rollbacks across S7-S17 (capture domination ≥ 0.98). Rollbacks restore weights within a phase but do NOT:
- Prevent curriculum advancement
- Give the evader extra training time
- Reduce pursuer capability

## Comparison with Previous Runs

| Run | Key Change | Level 3+ Behavior | Final NE gap |
|-----|-----------|-------------------|-------------|
| H | Curriculum (4-level) | Collapse at L4 (100% pursuer) | ~1.0 |
| I-J | Opponent pool | Same collapse pattern | ~1.0 |
| K-L | Visibility reward | Collapse at L3+ | ~1.0 |
| M-N | Prep phase | Collapse at L3+ | ~1.0 |
| O | 6-level curriculum | Collapse at L3+ | ~1.0 |
| **P** | **PBRS w=10.0** | **Collapse at L3+ (same)** | **0.94** |

All runs share the same failure mode: curriculum advances before the evader learns obstacle-seeking behavior.

## Proposed Fix

Three compounding root causes require a three-pronged fix:

1. **Dual-criteria curriculum gate**: Advance only when BOTH capture_rate > 0.70 AND escape_rate > 0.10, with minimum 4 phases per level and regression allowed
2. **Asymmetric training ratio**: Evader gets 2-3x training steps at obstacle levels
3. **Increased PBRS weight**: w_obs_approach 10→50 (0.35/step, competitive with ±1.0 visibility)

See `docs/research_self_play_collapse_prevention.md` for the full research report (20 papers, 6 approaches).
