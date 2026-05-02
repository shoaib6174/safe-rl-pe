# SP17 Co-Evolution Refinement Design

**Date:** 2026-04-29
**Approach:** B — Moderate Refinement of SP16
**Goal:** Fix crash bug, enable search reward, add forced periodic switching, validate across 4+ seeds

---

## 1. Scope and Objectives

Produce reproducible, multi-seed-validated co-evolution results for a conference paper on self-play training for pursuit-evasion under partial observability.

**In scope:**
- Fix SP16a_s42/s44 reproducible crash
- Enable search staleness reward at `w_search=0.0001`
- Add forced periodic switching every 3M steps
- Run 4+ seeds for statistical validation
- Generate trajectory visualizations and convergence analysis

**Out of scope:**
- Lower freeze threshold (stays at 0.65)
- Evader obstacle-seeking reward
- CBF safety layer integration during training
- Sim-to-real (Phase 4)

**Timeline:** 2-3 weeks
**Compute:** 4 seeds x 30M steps ~ 8 GPU-days on niro-2

---

## 2. Configuration Design

**Base:** SP16g (vis=0.2, thr=0.65, best converged run: CR=0.52)

| Parameter | SP16 Value | SP17 Value | Rationale |
|-----------|-----------|------------|-----------|
| `w_search` | 0.0 | **0.0001** | 30x lower than harmful 0.003; nudges coverage without patrol |
| `force_switch_steps` | None | **3M** | SP16e's forced switch sustained 23M steps; breaks freeze-sticking |
| `seed` | 43 only | **42, 43, 44, 45** | Multi-seed validation |

**Full SP17 config:**

```bash
python scripts/train_amsdrl.py \
  --algorithm sac \
  --sensing_mode fov \
  --fov_angle 90 \
  --fov_range 8.0 \
  --combined_masking \
  --arena_size 20 \
  --max_steps 1200 \
  --n_obstacles_min 0 --n_obstacles_max 3 --n_obstacle_obs 3 \
  --speed_pursuer 1.0 --speed_evader 1.0 \
  --lr 3e-4 --ent_coef 0.03 \
  --micro_phase_steps 2048 \
  --eval_interval 150 \
  --freeze_switch_threshold 0.65 \
  --force_switch_steps 3000000 \
  --w_visibility 0.2 \
  --w_search 0.0001 \
  --t_stale 100 \
  --collapse_threshold 0.1 \
  --collapse_streak_limit 5 \
  --pfsp \
  --masking_curriculum \
  --p_full_start 1.0 --p_full_end 0.0 --p_full_anneal_steps 5000000 \
  --max_total_steps 30000000 \
  --seed <42|43|44|45>
```

**Ablations (2 seeds each, if compute allows):**

| Run | What changes | Purpose |
|-----|-------------|---------|
| SP17a_s4x | Full config | Primary result |
| SP17b_s4x | `w_search=0.0` | Isolate search reward effect |
| SP17c_s4x | `force_switch_steps=0` | Isolate forced switch effect |

**Why these values:**
- `w_search=0.0001`: Max episode search reward ~0.12 vs visibility ~240 vs terminal +-20. A nudge, not a teacher.
- `t_stale=100`: Cell stale after 100 steps (~8% of episode)
- `force_switch_steps=3M`: Matches SP16e; ~6 switches per 30M run

---

## 3. Implementation Changes

### Change 1: Crash Bug Fix

**Problem:** Reproducible crash after S0 (200K steps) with seeds 42, 44. Logs end at VecMonitor warning.

**Process:**
1. Reproduce locally: `python scripts/train_amsdrl.py ... --seed 42 --max_total_steps 500000`
2. Capture traceback
3. Likely causes: opponent pool init, VecMonitor env reset, checkpoint loading edge case
4. Fix + regression test

**Effort:** 2-4 hours

### Change 2: Search Reward Integration

**Current state:** `SearchStalenessTracker` exists (S71), disabled via `w_search=0.0`.

**Changes:**
- Verify `--w_search` and `--t_stale` CLI args propagate in `scripts/train_amsdrl.py`
- Verify `envs/rewards.py` instantiates tracker when `w_search > 0`
- Verify `envs/wrappers.py` resets tracker state per episode

**No new code needed** — enable existing infrastructure.

### Change 3: Forced Periodic Switching

**Verify existing code:**
- `scripts/train_amsdrl.py`: `--force_switch_steps` flag exists
- `training/amsdrl.py`: enforces switch at configured step count
- Interaction with `freeze_switch_threshold`: whichever triggers first

**Effort:** 1 hour (verification + test)

### Change 4: Multi-Seed Launch Script

**Create:** `scripts/launch_sp17.sh`

```bash
#!/bin/bash
SEEDS=(42 43 44 45)
for seed in "${SEEDS[@]}"; do
    nohup python scripts/train_amsdrl.py \
        --algorithm sac \
        --sensing_mode fov \
        --fov_angle 90 --fov_range 8.0 \
        --combined_masking \
        --arena_size 20 --max_steps 1200 \
        --n_obstacles_min 0 --n_obstacles_max 3 \
        --lr 3e-4 --ent_coef 0.03 \
        --micro_phase_steps 2048 \
        --freeze_switch_threshold 0.65 \
        --force_switch_steps 3000000 \
        --w_visibility 0.2 \
        --w_search 0.0001 \
        --t_stale 100 \
        --masking_curriculum \
        --p_full_anneal_steps 5000000 \
        --max_total_steps 30000000 \
        --seed $seed \
        > results/SP17_s${seed}.log 2>&1 &
done
```

**Total effort:** 4-6 hours

---

## 4. Experiment Protocol

### Phase 1: Local Verification (Day 1)

1. Fix crash bug via interactive reproduction
2. Run single short test (--max_total_steps 500000) with seed 42
3. Verify search reward activates (check logs for `w_search`)
4. Verify forced switch triggers at 3M steps
5. All existing tests must still pass

### Phase 2: niro-2 Launch (Day 1-2)

1. Push code to GitHub
2. SSH to niro-2, pull
3. Run `scripts/launch_sp17.sh`
4. Verify all 4 seeds started (check `ps aux | grep train_amsdrl`)

### Phase 3: Monitoring (Days 2-10)

**Daily checks:**
- Process status (all 4 running?)
- Early convergence signs (CR trajectory)
- Disk space

**Kill criteria:**
- Crash (obviously)
- CR < 0.10 for 5+ consecutive evals (collapse)
- No improvement after 10M steps

### Phase 4: Analysis (Days 10-14)

1. Extract `history.json` from all completed runs
2. Compute: final CR, NE gap, switch count, convergence steps
3. Cross-seed statistics: mean, std, min, max
4. Generate trajectory GIFs for best seed
5. Write results report

---

## 5. Success Criteria

### Minimum (publishable workshop/late-breaking)
- 2+ of 4 seeds achieve functional co-evolution (>=1 role switch)
- Mean final CR across seeds: 0.45-0.55 (balanced play)
- No crashes
- Search reward shows measurable effect (SP17a vs SP17b)

### Target (full conference paper)
- 3+ of 4 seeds converge (NE gap < 0.10)
- Mean final CR: 0.48-0.52 with std < 0.08
- Forced switching shows measurable stability improvement (SP17a vs SP17c)
- Trajectory analysis shows search behavior improvement over SP16

### Fallbacks

| If this happens | Then do this |
|----------------|--------------|
| Crash bug is unfixable quickly | Skip seeds 42/44; run 43/45/46/47 |
| Search reward has no effect | Increase to 0.0003; or drop and focus on forced switch |
| Forced switch disrupts learning | Increase interval to 5M; or make conditional (only if no natural switch) |
| All seeds collapse | Lower threshold to 0.55; or revert to freeze-evader-only for paper |
| Only 1-2 seeds converge | Report distributions; emphasize seed sensitivity as a finding |

---

## 6. Publication Story

**Title candidate:** "Stabilizing Adversarial Self-Play for Pursuit-Evasion Under Partial Observability"

**Contributions:**
1. Visibility reward (w=0.2) stabilizes pursuer learning under FOV partial observability
2. Forced periodic switching prevents freeze-sticking and sustains co-evolution to 23M+ steps
3. Search staleness reward fills the critical search skill gap
4. Multi-seed validation showing reproducible convergence

**Comparison baselines:**
- SP16a (no search, no forced switch)
- SP13a_s42 (freeze evader, upper bound)
- SP16e (forced switch only)

**Venue:** ICRA 2027, IROS 2027, or CoRL 2026 (depending on timeline)
