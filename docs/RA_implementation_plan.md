# Implementation Plan: Runs RA1 and RA2

## Status: IMPLEMENTED (S54)

All code changes below have been implemented and smoke-tested. Ready for launch on niro-2.

## Overview

Both runs share the same code changes (the "RA redesign"). They differ only in CLI arguments:
- **RA1**: No curriculum — fixed spawn 2–10m, 2 obstacles
- **RA2**: Simple distance curriculum — start at 2–5m, increment 1m, 2 obstacles always

---

## Code Changes Implemented

### 1. Parameter Plumbing: `evader_v_max`, `capture_bonus`, `n_obstacle_obs`

**`training/amsdrl.py` → `_make_partial_obs_env()`**
- Added parameters: `evader_v_max`, `capture_bonus`, `n_obstacle_obs`
- Passed through to `PursuitEvasionEnv()` and `RewardComputer()`
- CRITICAL FIX: `n_obstacle_obs` was missing — agents were blind to obstacles in `--full_obs` mode

**`training/amsdrl.py` → `AMSDRLSelfPlay.__init__()`**
- Added: `evader_v_max`, `capture_bonus`, `n_obstacle_obs`, `min_obstacles`, `micro_phase_steps`, `eval_interval_micro`, `snapshot_freq_micro`, `max_total_steps`
- All stored in `self.env_kwargs` for propagation

**`scripts/train_amsdrl.py`**
- 8 new CLI args added (see table below)
- All passed through to `AMSDRLSelfPlay`

### 2. Evaluation Functions Fixed

**`_evaluate_head_to_head_full_obs()` and `_evaluate_head_to_head()`**
- Added `evader_v_max` and `n_obstacle_obs` parameters
- Now eval envs match training envs exactly

### 3. Curriculum: `min_obstacles` Floor

**`training/curriculum.py` → `SmoothCurriculumManager`**
- Added `min_obstacles: int = 0` parameter
- `n_obstacles` property now returns `max(self.min_obstacles, ...)` instead of 0

### 4. Micro-Phase Rapid Alternation (`_run_micro_phases()`)

Core new method (~200 lines) implementing:
- **Persistent environments**: Created once, reused across all micro-phases
- **n_envs mismatch handling**: Save/reload pattern when cold-start uses 1 env
- **Opponent weight sync**: `_sync_opponent_weights_vec()` copies `state_dict` in-place
- **Opponent pool integration**: `_resample_pool_opponents_vec()` with 50/50 current/historical
- **Periodic evaluation**: Every `eval_interval_micro` micro-phases
- **Progress logging**: Compact status line every 10 micro-phases, full eval summary at intervals
- **Checkpointing**: Models saved at every eval interval for crash recovery
- **Incremental history**: `history.json` saved at each eval for monitoring
- **NE-gap curriculum**: Integrated with micro-phase eval loop
- **Convergence detection**: Stops when NE gap < eta at max curriculum level
- **No rollback**: `SelfPlayHealthMonitorCallback` not attached in micro-phase mode

Key helper methods:
- `_set_opponents_in_vec_env()`: Set opponents in all sub-envs of a vec_env
- `_sync_opponent_weights_vec()`: Copy policy weights without recreating envs/adapters
- `_resample_pool_opponents_vec()`: 50% current, 50% pool sampling
- `_save_micro_snapshot()`: Save to opponent pool
- `_save_history_incremental()`: Crash recovery

### 5. Dispatch in `run()`

After cold-start + eval, if `self.micro_phase_steps > 0`, calls `_run_micro_phases()` instead of legacy phase loop. Legacy mode fully preserved for backwards compatibility.

---

## Summary of All New CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--evader_v_max` | float | 1.0 | Evader max speed (1.15 for speed advantage) |
| `--capture_bonus` | float | 100.0 | Terminal capture reward magnitude |
| `--n_obstacle_obs` | int | 0 | Nearest obstacles in obs vector (set to 2 for RA runs) |
| `--min_obstacles` | int | 0 | Minimum obstacle count floor |
| `--micro_phase_steps` | int | 0 | Steps per micro-phase (0=disabled) |
| `--eval_interval_micro` | int | 50 | Evaluate every N micro-phases |
| `--snapshot_freq_micro` | int | 5 | Save opponent snapshot every N micro-phases |
| `--max_total_steps` | int | 0 | Max total steps (0=unlimited) |

## Summary of All Modified Files

| File | Changes |
|------|---------|
| `training/amsdrl.py` | `evader_v_max`/`capture_bonus`/`n_obstacle_obs` plumbing, `_run_micro_phases()` + 5 helper methods, dispatch in `run()`, eval functions fixed |
| `training/curriculum.py` | `min_obstacles` floor in `SmoothCurriculumManager` |
| `scripts/train_amsdrl.py` | 8 new CLI args, pass-through to `AMSDRLSelfPlay` |
| `envs/rewards.py` | No changes (already parameterized) |
| `envs/pursuit_evasion_env.py` | No changes (already has `evader_v_max`, `n_obstacle_obs`) |

---

## Launch Commands

### RA1: No Curriculum

```bash
cd ~/claude_pursuit_evasion
mkdir -p results/run_RA1_no_curriculum

PYTHONUNBUFFERED=1 nohup ./venv/bin/python scripts/train_amsdrl.py \
    --micro_phase_steps 2048 \
    --eval_interval_micro 50 \
    --snapshot_freq_micro 5 \
    --max_total_steps 10000000 \
    --evader_v_max 1.15 \
    --capture_bonus 10.0 \
    --timeout_penalty -10.0 \
    --survival_bonus 0.02 \
    --use_visibility_reward \
    --visibility_weight 0.1 \
    --n_obstacles 2 \
    --min_obstacles 2 \
    --n_obstacle_obs 2 \
    --max_steps 600 \
    --max_phases 1000 \
    --opponent_pool_size 20 \
    --w_obs_approach 50.0 \
    --prep_steps 50 \
    --pursuer_v_max 1.0 \
    --fixed_speed \
    --arena_width 20 \
    --arena_height 20 \
    --no_dcbf \
    --full_obs \
    --seed 47 \
    --output results/run_RA1_no_curriculum \
    > results/run_RA1_no_curriculum/train.log 2>&1 &
```

**Key parameters explained:**
- `--micro_phase_steps 2048`: 1 PPO rollout per micro-phase (4 envs × 512 n_steps = 2048)
- `--eval_interval_micro 50`: Evaluate every 50 micro-phases = every ~100K total steps
- `--max_total_steps 10000000`: 10M step budget (~2-4 hours on GPU)
- `--evader_v_max 1.15`: 15% evader speed advantage
- `--capture_bonus 10.0 --timeout_penalty -10.0`: Rebalanced terminal rewards
- `--survival_bonus 0.02`: Evader gets +0.02 per step alive
- `--visibility_weight 0.1`: Toned-down visibility reward
- `--n_obstacles 2 --min_obstacles 2 --n_obstacle_obs 2`: Always 2 obstacles, visible in obs
- `--max_steps 600`: 30-second episodes

### RA2: Distance Curriculum

```bash
cd ~/claude_pursuit_evasion
mkdir -p results/run_RA2_distance_curriculum

PYTHONUNBUFFERED=1 nohup ./venv/bin/python scripts/train_amsdrl.py \
    --micro_phase_steps 2048 \
    --eval_interval_micro 50 \
    --snapshot_freq_micro 5 \
    --max_total_steps 10000000 \
    --evader_v_max 1.15 \
    --capture_bonus 10.0 \
    --timeout_penalty -10.0 \
    --survival_bonus 0.02 \
    --use_visibility_reward \
    --visibility_weight 0.1 \
    --n_obstacles 2 \
    --min_obstacles 2 \
    --n_obstacle_obs 2 \
    --max_steps 600 \
    --smooth_curriculum \
    --smooth_curriculum_increment 1.0 \
    --ne_gap_advancement \
    --ne_gap_threshold 0.15 \
    --ne_gap_consecutive 3 \
    --min_phases_per_level 2 \
    --opponent_pool_size 20 \
    --w_obs_approach 50.0 \
    --prep_steps 50 \
    --pursuer_v_max 1.0 \
    --fixed_speed \
    --arena_width 20 \
    --arena_height 20 \
    --no_dcbf \
    --full_obs \
    --seed 47 \
    --output results/run_RA2_distance_curriculum \
    > results/run_RA2_distance_curriculum/train.log 2>&1 &
```

**Differences from RA1:**
- `--smooth_curriculum`: Enables distance-only curriculum (starts at 2–5m)
- `--smooth_curriculum_increment 1.0`: +1m per advancement
- `--ne_gap_advancement --ne_gap_threshold 0.15 --ne_gap_consecutive 3`: Advance when balanced for 3 consecutive eval windows
- `--min_phases_per_level 2`: At least 2 eval windows at each distance level

---

## Success Criteria

After ~2M total steps (~30–60 min wall time on niro-2):

| Metric | Target | Failure |
|--------|--------|---------|
| NE gap (\|SR_P - SR_E\|) | < 0.30 sustained | > 0.80 for 500K+ steps |
| Rollbacks triggered | 0 (disabled) | N/A |
| SR_P oscillation amplitude | < 0.30 between evals | > 0.70 swings |
| Curriculum advancement (RA2) | At least 1 level | Stuck at initial level after 2M steps |
| Training stability | No crashes, no NaN | Crash or divergence |

## Monitoring Commands

```bash
# Quick status check
cat ~/claude_pursuit_evasion/results/run_RA1_no_curriculum/train.log | tr '\r' '\n' | grep -E "(SR_P|M\s*[0-9]|Eval|gap=|pool=|Converged)"

# Tail live
tail -f ~/claude_pursuit_evasion/results/run_RA1_no_curriculum/train.log | tr '\r' '\n'

# Check history.json
cat ~/claude_pursuit_evasion/results/run_RA1_no_curriculum/history.json | python3 -m json.tool | tail -20
```

## Spec Panel Fixes Applied

1. **CRITICAL**: Added `--n_obstacle_obs` CLI arg — agents can now see obstacles in `--full_obs` mode
2. **HIGH**: n_envs mismatch after cold-start handled via save/reload pattern
3. **HIGH**: Corrected micro_phase_steps math: 2048 = 1 rollout (n_envs=4 × n_steps=512)
4. **HIGH**: Added progress logging (every 10 micro-phases) and full eval summaries
5. **HIGH**: Added checkpointing at eval intervals + incremental history saves
6. **HIGH**: Opponent adapter traversal through full wrapper stack (Monitor → FixedSpeed → SingleAgent)
7. **MEDIUM**: Added `--max_total_steps` for convergence detection
8. **MEDIUM**: Pool resampling at snapshot intervals (not every micro-phase)
