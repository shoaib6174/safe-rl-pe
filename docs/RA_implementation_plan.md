# Implementation Plan: Runs RA1 and RA2

## Overview

Both runs share the same code changes (the "RA redesign"). They differ only in CLI arguments:
- **RA1**: No curriculum — fixed spawn 2–10m, 2 obstacles
- **RA2**: Simple distance curriculum — start at 2–5m, increment 1m, 2 obstacles always

---

## Code Changes (7 files, ~15 edits)

### 1. Environment: Add `evader_v_max` plumbing

**File: `envs/pursuit_evasion_env.py`** — Already has `evader_v_max` parameter (line 50, default 1.0). No change needed here.

**File: `training/amsdrl.py` → `_make_partial_obs_env()`** (line 54)
- **Add parameter**: `evader_v_max: float = 1.0`
- **Pass through** to `PursuitEvasionEnv(evader_v_max=evader_v_max, ...)`
- Currently the function creates `PursuitEvasionEnv` but never passes `evader_v_max`, so it always defaults to 1.0.

**File: `training/amsdrl.py` → `_make_vec_env()`** (line 145)
- **Add parameter**: `evader_v_max: float = 1.0`
- **Pass through** to `_make_partial_obs_env(evader_v_max=evader_v_max, ...)`

**File: `training/amsdrl.py` → `AMSDRLSelfPlay.__init__()`** (line 498)
- **Add parameter**: `evader_v_max: float = 1.0`
- **Store in `self.env_kwargs`**: `"evader_v_max": evader_v_max`

**File: `scripts/train_amsdrl.py` → `parse_args()`**
- **Add CLI arg**: `--evader_v_max` (float, default 1.0)
- **Pass to AMSDRLSelfPlay**: `evader_v_max=args.evader_v_max`

### 2. Rewards: Add `capture_bonus` plumbing + new defaults

**File: `training/amsdrl.py` → `_make_partial_obs_env()`**
- **Add parameter**: `capture_bonus: float = 100.0`
- **Pass through** to `RewardComputer(capture_bonus=capture_bonus, ...)`

**File: `training/amsdrl.py` → `AMSDRLSelfPlay.__init__()`**
- **Add parameter**: `capture_bonus: float = 100.0`
- **Store in `self.env_kwargs`**: `"capture_bonus": capture_bonus`

**File: `scripts/train_amsdrl.py` → `parse_args()`**
- **Add CLI arg**: `--capture_bonus` (float, default 100.0)

### 3. Self-Play: Micro-Phase Alternation

This is the most significant change. The current `run()` method calls `_train_phase()` once per phase, and `_train_phase()` creates+destroys environments each time.

**File: `training/amsdrl.py` → New method `_run_micro_phases()`**

Replace the alternating training loop in `run()` with a new mode when `micro_phase_steps > 0`:

```
def _run_micro_phases(self):
    """Rapid alternation self-play with micro-phases."""

    # 1. Create PERSISTENT environment sets (one for pursuer training, one for evader training)
    #    These are created ONCE and reused across all micro-phases.
    pursuer_vec_env, pursuer_base_envs = _make_vec_env(role="pursuer", ...)
    evader_vec_env, evader_base_envs = _make_vec_env(role="evader", ...)

    # 2. Set initial opponents
    _set_opponents(pursuer_base_envs, self.evader_model, "evader")
    _set_opponents(evader_base_envs, self.pursuer_model, "pursuer")

    # 3. Attach models to their environments
    self.pursuer_model.set_env(pursuer_vec_env)
    self.evader_model.set_env(evader_vec_env)

    # 4. Main loop: alternate micro-phases
    total_steps = 0
    role = "pursuer"
    eval_interval = self.eval_interval_micro  # e.g. 50 micro-phases
    snapshot_interval = ...  # e.g. every 5 micro-phases = 20K steps

    for micro in range(max_micro_phases):
        # Train active agent for micro_phase_steps
        if role == "pursuer":
            model = self.pursuer_model
        else:
            model = self.evader_model

        model.learn(
            total_timesteps=self.micro_phase_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        total_steps += self.micro_phase_steps

        # Sync opponent: copy just-trained agent's weights to the other env's opponent
        if role == "pursuer":
            _sync_opponent(evader_base_envs, self.pursuer_model, "pursuer")
        else:
            _sync_opponent(pursuer_base_envs, self.evader_model, "evader")

        # Snapshot to opponent pool
        if micro % snapshot_interval == 0:
            _save_snapshot(...)

        # Periodic evaluation
        if micro % eval_interval == 0:
            metrics = self._evaluate()
            # Check NE-gap advancement (if curriculum enabled)
            # Log to history

        # Alternate role
        role = "evader" if role == "pursuer" else "pursuer"

    # Cleanup
    pursuer_vec_env.close()
    evader_vec_env.close()
```

**Key helper functions needed:**

```python
def _set_opponents(base_envs, opponent_model, opponent_role):
    """Set the frozen opponent in each sub-env's SingleAgentPEWrapper."""
    for base_env in base_envs:
        # Traverse wrapper stack to find SingleAgentPEWrapper
        # Set opponent adapter with the model
        ...

def _sync_opponent(base_envs, updated_model, role):
    """Copy updated model weights into the opponent adapters in base_envs."""
    # For each sub-env, update the opponent's policy weights
    # WITHOUT recreating the environment or adapter
    for base_env in base_envs:
        wrapper = _find_wrapper(base_env, SingleAgentPEWrapper)
        if wrapper.opponent is not None:
            # Update opponent model reference weights
            wrapper.opponent.model.policy.load_state_dict(
                updated_model.policy.state_dict()
            )
```

**File: `training/amsdrl.py` → `AMSDRLSelfPlay.__init__()`**
- **Add parameters**:
  - `micro_phase_steps: int = 0` (0 = disabled, use legacy alternating)
  - `eval_interval_micro: int = 50` (evaluate every N micro-phases)
  - `snapshot_freq_micro: int = 5` (save snapshot every N micro-phases)

**File: `training/amsdrl.py` → `AMSDRLSelfPlay.run()`**
- **Add dispatch**: If `self.micro_phase_steps > 0`, call `_run_micro_phases()` instead of the legacy phase loop.

**File: `scripts/train_amsdrl.py` → `parse_args()`**
- **Add CLI args**:
  - `--micro_phase_steps` (int, default 0)
  - `--eval_interval_micro` (int, default 50)
  - `--snapshot_freq_micro` (int, default 5)

### 4. Opponent Pool Integration with Micro-Phases

**File: `training/amsdrl.py` → `_run_micro_phases()`**

Within the micro-phase loop, when sampling opponents for sub-envs:
- 50% of sub-envs get the current opponent (latest weights)
- 50% sample from the opponent pool (historical snapshots)

This reuses the existing `OpponentPool` class from S43. The pool is populated by saving snapshots every `snapshot_freq_micro` micro-phases.

### 5. Curriculum Changes: `min_obstacles` floor

**File: `training/curriculum.py` → `SmoothCurriculumManager`**
- **Add parameter**: `min_obstacles: int = 0`
- **Modify `n_obstacles` property** (line 342): Return `max(self.min_obstacles, ...)` instead of 0 when below threshold.

**File: `training/amsdrl.py` → `AMSDRLSelfPlay.__init__()`**
- **Add parameter**: `min_obstacles: int = 0`
- **Pass through** to `SmoothCurriculumManager(min_obstacles=min_obstacles, ...)`

**File: `scripts/train_amsdrl.py`**
- **Add CLI arg**: `--min_obstacles` (int, default 0)

### 6. Remove Rollback for Micro-Phase Mode

**File: `training/amsdrl.py` → `_run_micro_phases()`**
- Do NOT attach `SelfPlayHealthMonitorCallback` in micro-phase mode.
- Only attach `EntropyMonitorCallback` for logging.

### 7. Episode Length Default

**No code change needed** — `max_steps` is already a CLI parameter (default 1200). We'll pass `--max_steps 600` at launch time.

---

## Summary of All New CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--evader_v_max` | float | 1.0 | Evader max speed (1.15 for speed advantage) |
| `--capture_bonus` | float | 100.0 | Terminal capture reward magnitude |
| `--micro_phase_steps` | int | 0 | Steps per micro-phase (0=disabled, use legacy) |
| `--eval_interval_micro` | int | 50 | Evaluate every N micro-phases |
| `--snapshot_freq_micro` | int | 5 | Save opponent snapshot every N micro-phases |
| `--min_obstacles` | int | 0 | Minimum obstacle count (floor, never go below) |

## Summary of All Modified Files

| File | Changes |
|------|---------|
| `training/amsdrl.py` | Add `evader_v_max`/`capture_bonus` plumbing, `_run_micro_phases()` method, `_sync_opponent()` helper, dispatch in `run()` |
| `training/curriculum.py` | Add `min_obstacles` floor to `SmoothCurriculumManager` |
| `scripts/train_amsdrl.py` | 6 new CLI args, pass-through to `AMSDRLSelfPlay` |
| `envs/rewards.py` | No changes (already parameterized) |
| `envs/pursuit_evasion_env.py` | No changes (already has `evader_v_max`) |

---

## Launch Commands

### RA1: No Curriculum

```bash
cd ~/claude_pursuit_evasion

PYTHONUNBUFFERED=1 nohup ./venv/bin/python scripts/train_amsdrl.py \
    --micro_phase_steps 4096 \
    --eval_interval_micro 50 \
    --snapshot_freq_micro 5 \
    --evader_v_max 1.15 \
    --capture_bonus 10.0 \
    --timeout_penalty -10.0 \
    --survival_bonus 0.02 \
    --use_visibility_reward \
    --visibility_weight 0.1 \
    --n_obstacles 2 \
    --min_obstacles 2 \
    --min_init_distance 2.0 \
    --max_init_distance 10.0 \
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
- `--micro_phase_steps 4096`: 1 PPO rollout per micro-phase (4 envs × 512 n_steps × 2 = 4096)
- `--eval_interval_micro 50`: Evaluate every 50 micro-phases = every ~200K total steps
- `--evader_v_max 1.15`: 15% evader speed advantage
- `--capture_bonus 10.0 --timeout_penalty -10.0`: Rebalanced terminal rewards
- `--survival_bonus 0.02`: Evader gets +0.02 per step alive
- `--visibility_weight 0.1`: Toned-down visibility reward
- `--n_obstacles 2 --min_obstacles 2`: Always 2 obstacles
- `--max_steps 600`: 30-second episodes
- `--opponent_pool_size 20`: Historical opponent pool with 20 slots
- `--max_phases 1000`: High limit since each "phase" is one micro-phase now (ignored in micro-phase mode — runs until convergence or wall time)

### RA2: Distance Curriculum

```bash
cd ~/claude_pursuit_evasion

PYTHONUNBUFFERED=1 nohup ./venv/bin/python scripts/train_amsdrl.py \
    --micro_phase_steps 4096 \
    --eval_interval_micro 50 \
    --snapshot_freq_micro 5 \
    --evader_v_max 1.15 \
    --capture_bonus 10.0 \
    --timeout_penalty -10.0 \
    --survival_bonus 0.02 \
    --use_visibility_reward \
    --visibility_weight 0.1 \
    --n_obstacles 2 \
    --min_obstacles 2 \
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
- `--smooth_curriculum_increment 1.0`: +1m per advancement (coarser than 0.5m, faster progression)
- `--ne_gap_advancement --ne_gap_threshold 0.15 --ne_gap_consecutive 3`: Advance when balanced for 3 consecutive eval windows
- `--min_phases_per_level 2`: At least 2 eval windows at each distance level
- No `--min_init_distance` / `--max_init_distance` (curriculum manager sets these)

---

## Implementation Order

1. **Plumbing** (~30 min): `evader_v_max` and `capture_bonus` through `_make_partial_obs_env` → `_make_vec_env` → `AMSDRLSelfPlay.__init__` → CLI
2. **`min_obstacles` floor** (~15 min): `SmoothCurriculumManager` property change + CLI arg
3. **`_run_micro_phases()` method** (~2–3 hours): The core new method with env caching, opponent sync, periodic eval, pool integration
4. **`_sync_opponent()` helper** (~30 min): Weight copy without env recreation
5. **Dispatch in `run()`** (~10 min): If `micro_phase_steps > 0`, call `_run_micro_phases()`
6. **Tests** (~1 hour): Test micro-phase training runs for 2–3 micro-phases without crashing
7. **Launch on niro-2** (~30 min): Push, pull, kill old runs, start RA1 + RA2

**Total estimated: ~5–6 hours**

---

## Success Criteria

After ~2M total steps (~30–60 min wall time on niro-2):

| Metric | Target | Failure |
|--------|--------|---------|
| NE gap (|SR_P - SR_E|) | < 0.30 sustained | > 0.80 for 500K+ steps |
| Rollbacks triggered | 0 (disabled) | N/A |
| SR_P oscillation amplitude | < 0.30 between evals | > 0.70 swings |
| Curriculum advancement (RA2) | At least 1 level | Stuck at initial level after 2M steps |
| Training stability | No crashes, no NaN | Crash or divergence |

## Monitoring Commands

```bash
# Quick status check
cat ~/claude_pursuit_evasion/results/run_RA1_no_curriculum/train.log | tr '\r' '\n' | grep -E "(SR_P|Phase|Eval|NE.gap|micro|Pool)"

# Tail live
tail -f ~/claude_pursuit_evasion/results/run_RA1_no_curriculum/train.log | tr '\r' '\n'
```
