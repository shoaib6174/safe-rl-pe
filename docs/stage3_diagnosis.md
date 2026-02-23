# Stage 3 Diagnosis — Pursuer Can't Learn Pursuit

## Date: 2026-02-22 (updated 2026-02-23)

## Complete Experimental Evidence

| Run | Config | S1 | S2 | S3 | Final | Key Finding |
|-----|--------|----|----|----|-------|-------------|
| Original | DCBF on, partial-obs, 200K | 3% | 3% | 3% | 3% | Baseline |
| A (no DCBF) | No DCBF, partial-obs, 200K | 3% | 7% | 0% | 0% | DCBF NOT the bottleneck |
| B (longer) | DCBF on, partial-obs, 500K | 0% | 0% | 0% | 0% | More steps made it WORSE |
| C (full-obs) | Full-obs, no DCBF, 200K | 0% | 0% | 7% | 7% | Partial-obs NOT the sole issue |
| D (reward fix) | Full-obs, dist_scale=10, v_max=1.5 | 2% | ? | ? | 5% | Reward + speed helps slightly |
| E (fixed speed) | Full-obs, dist_scale=10, v_max=1.5, fixed_speed | **100%** | 90% | **100%** | 100% | **GATE 3 PASSED** — pursuer dominates |
| F (partial-obs) | Partial-obs, dist_scale=10, v_max=1.5, fixed_speed | **76%** | 75% | **77%** | 77% | **GATE 3 PASSED** — partial-obs works! NE gap 0.50-0.54 |
| G (more phases) | Partial-obs, dist_scale=10, v_max=1.2, fixed_speed, 6×300K, 10×10 arena, 16 envs | 43% | 21%→73% | 98%→71% | 71% | Arms race happening! Min NE gap 0.14, oscillating |
| H (curriculum) | Partial-obs, dist_scale=10, v_max=1.2, fixed_speed, 12×300K, 10×10, curriculum | 93% | 26%→71% | 100%→100% | 100% | Curriculum L1→L4 in 7 phases. Best NE gap 0.26 (L3). L4 collapse: pursuer 100% |

## Root Cause Chain (Diagnosed Layer by Layer)

### Layer 1: NOT partial observability alone
- Run C proved full-obs still fails (7% capture)
- Partial-obs makes it harder, but isn't the primary blocker

### Layer 2: NOT DCBF
- Run A proved removing DCBF doesn't help

### Layer 3: NOT insufficient training steps
- Run B proved 500K steps is worse than 200K (overfits to degenerate policy)

### Layer 4: Anemic reward signal (CONFIRMED)
- `distance_scale=1.0` gives reward of 0.035 per meter closed (= 1.0/28.28)
- Over 1200 steps of perfect closure: ~42 total reward
- Timeout penalty is -50, which dominates
- PPO gets almost no gradient from this weak signal

### Layer 5: 2D action space kills PPO learning (PRIMARY ROOT CAUSE)
- Action space: [v, omega] where v in [0, v_max], omega in [-omega_max, omega_max]
- PPO's Gaussian policy starts with mean ≈ 0 for both dims
- The model outputs v ≈ 0.4 (measured!) — pursuer barely moves
- A random evader at v=1.0 actually moves FASTER than the trained pursuer
- After 50K steps with 2D actions: **0% capture rate**
- After 50K steps with 1D actions (fixed v=v_max): **33% capture rate**

### Key Diagnostic Result
- Pure pursuit heuristic (v=v_max, steer toward evader) achieves **100% capture** vs both random and fleeing evaders when pursuer_v_max=1.5
- So the env IS solvable — PPO just can't discover "always go full speed"

## Solution: FixedSpeedWrapper (IMPLEMENTED)

Created `FixedSpeedWrapper` (envs/wrappers.py) that:
- Fixes v = v_max, only learns omega (1D action space)
- Eliminates the "learn to go fast" problem entirely
- Reduces action space from 2D to 1D
- Also created `FixedSpeedModelAdapter` to expand 1D→2D for opponent models

### Implementation Details
- `FixedSpeedWrapper` in `envs/wrappers.py` — gym.ActionWrapper that converts [omega] → [v_max, omega]
- `FixedSpeedModelAdapter` in `envs/wrappers.py` — wraps SB3 model predict() to expand 1D actions
- `--fixed_speed` CLI flag in `scripts/train_amsdrl.py`
- `fixed_speed` parameter threaded through AMSDRLSelfPlay → _make_partial_obs_env → _make_vec_env → _train_phase → _build_callbacks → _evaluate
- `FixedBaselineEvalCallback` updated with `fixed_speed_v_max` param
- Both eval functions handle 1D→2D expansion for fixed_speed models

### Other Changes Made This Session
- Added `--distance_scale` CLI arg (controls dense reward strength)
- Added `--pursuer_v_max` CLI arg (pursuer speed advantage)
- Both params threaded through entire pipeline (env creation, evaluation, etc.)
- All 38 tests pass

## Run E Results & Analysis (COMPLETED 2026-02-23)

**Config**: Full-obs, no DCBF, distance_scale=10, pursuer_v_max=1.5, fixed_speed, 3 phases × 200K steps

| Phase | SR_P | SR_E | NE Gap | Notes |
|-------|------|------|--------|-------|
| S0 (cold-start) | 25% | 75% | — | Already above 20% Gate 3 before training |
| S1 (train pursuer) | **100%** | 0% | 1.0 | Pursuer learned perfect pursuit |
| S2 (train evader) | 90% | 10% | 0.8 | Evader barely improved |
| S3 (train pursuer) | **100%** | 0% | 1.0 | Pursuer dominates completely |

### Key Findings from Run E
1. **Gate 3 PASSED**: Pursuer went from 0-7% (Runs A-D) to 100% capture. FixedSpeedWrapper is the validated fix.
2. **Capture domination rollbacks**: Health monitor triggered 3 rollbacks in S1 and 3 in S3 for >98% capture rate. These rollbacks are counter-productive in this config.
3. **NE convergence failed** (gap=1.0): Expected — with 1.5× speed advantage + full observability, the pursuer physically cannot lose. The evader (v_max=1.0) can never escape a faster pursuer in an open arena.
4. **Total training time**: ~7 minutes (CPU mode, very fast with MlpPolicy).

### Decision: Proceed to Partial-Obs (Run F)
- Full-obs with speed advantage creates an unbalanced game (pursuer always wins)
- **Partial observability is the mechanism that gives the evader a chance** — the pursuer can't always see the evader
- The speed advantage (1.5×) compensates for the sensing disadvantage
- This is the intended game design: pursuer is faster but partially blind

## Current Status (2026-02-23 05:40)

**Run H (curriculum) COMPLETED on niro-2** (GPU, 1.7h).
All runs through H complete. Curriculum validated — smooth L1→L4 progression.
Next: opponent pool / diversity (Session 7) to fix Level 4 collapse.

> **DECISION: Use 10×10 arena for ALL development and iteration.**
> 10×10 is ~4× faster and gives quick feedback for tuning.
> Run F already validated 20×20 works (77% capture).
> **Switch to 20×20 + max_steps=1200 ONLY for final validation runs**
> once all implementation is complete (opponent pools, curriculum, etc.).
> Strategies will need retuning at 20×20 but that's expected.

## Run F Results & Analysis (COMPLETED 2026-02-23)

**Config**: Partial-obs, no DCBF, distance_scale=10, pursuer_v_max=1.5, fixed_speed, 3 phases × 200K steps, GPU (RTX 5090)

| Phase | SR_P | SR_E | NE Gap | Notes |
|-------|------|------|--------|-------|
| S0 (cold-start) | 3% | 97% | — | Partial-obs makes cold-start much harder |
| S1 (train pursuer) | **76%** | 24% | 0.52 | Pursuer learned pursuit with limited sensing! |
| S2 (train evader) | **75%** | 25% | 0.50 | Evader barely improved (1% reduction) |
| S3 (train pursuer) | **77%** | 23% | 0.54 | Pursuer slightly improved again |

**Convergence**: NE gap 0.54, trend decreasing, min 0.50. NOT converged (need <0.10).
**Total time**: 30 min on GPU.

### Key Findings from Run F
1. **Gate 3 PASSED in partial-obs**: 77% capture rate — massive improvement from 0-7% in Runs A-D
2. **FixedSpeedWrapper works in partial-obs**: The 1D action space fix is validated across both observation modes
3. **Evader isn't learning effectively**: Only reduced capture by 1% (76% → 75%). The evader's policy isn't improving enough
4. **NE gap stuck at ~0.5**: Pursuer dominates, evader can't close the gap
5. **No capture domination rollbacks**: Unlike Run E (full-obs), partial-obs keeps capture below 98%

### Bug Fixed During Run F
- **FixedSpeedWrapper not applied to evader cold-start**: `NavigationEnv` didn't wrap with `FixedSpeedWrapper`, causing action space mismatch when S2 tried to load the evader model. Fixed by wrapping `NavigationEnv` output with `FixedSpeedWrapper` when `fixed_speed=True`.
- **Opponent adapter expansion in partial-obs**: `PartialObsOpponentAdapter` needed to expand 1D→2D actions for fixed-speed opponents. Fixed by wrapping opponent model with `FixedSpeedModelAdapter` before passing to adapter.

### Analysis: Why Evader Isn't Learning
1. The evader's cold-start was only goal-reaching + flee (NavigationEnv) — not adversarial
2. 200K steps per phase may be insufficient for the evader to learn evasion strategies
3. The pursuer's 1.5× speed advantage makes evasion fundamentally hard
4. The evader may need more training steps or a stronger evasion reward

## Run G Results & Analysis (COMPLETED 2026-02-23)

**Config**: Partial-obs, no DCBF, distance_scale=10, pursuer_v_max=1.2, fixed_speed, 6 phases × 300K steps, 10×10 arena, 16 envs, n_steps=2048, batch_size=512, GPU

| Phase | SR_P | SR_E | NE Gap | Notes |
|-------|------|------|--------|-------|
| S0 | 31% | 69% | — | Higher cold-start (10×10 arena) |
| S1 | 43% | 57% | **0.14** | Near-convergence! |
| S2 | 21% | 79% | 0.58 | Evader learns well |
| S3 | 76% | 24% | 0.52 | Pursuer counter-adapts |
| S4 | 73% | 27% | 0.46 | Evader barely budges |
| S5 | 98% | 2% | 0.96 | Pursuer dominates |
| S6 | 71% | 29% | 0.42 | Evader recovers |

**Convergence**: min NE gap 0.14, final 0.42, trend oscillating (not converging).
**Total time**: 54 min on GPU. ~4× faster than Run F would have been at this scale.

### Key Findings from Run G
1. **Real arms race**: Both agents learn and counter-adapt. The evader IS learning (unlike Run F).
2. **Oscillation problem**: NE gap swings wildly (0.14 → 0.58 → 0.96 → 0.42). Classic unstable self-play.
3. **Near-convergence achieved**: S1 NE gap 0.14 is very close to η=0.10 threshold.
4. **Speed: 10×10 arena + 16 envs + larger batch** = ~4× faster iteration. Good for prototyping.

### Root Cause of Oscillation
Each agent overfits to the frozen opponent during its training phase. When roles switch, the new policy exploits weaknesses the old policy didn't train against. Standard self-play stabilization techniques needed:
1. **Opponent sampling from history** (not just latest frozen model)
2. **Shorter phases** (less overfitting per phase)
3. **More phases** (more alternation)
4. **Population-based training** (maintain pool of diverse opponents)

### Recommended Next Steps (from Run G)
1. **Try shorter phases**: 12 phases × 100K steps (same total compute, more alternation)
2. **Opponent pool**: Sample opponent from recent N checkpoints instead of just the latest
3. **Entropy bonus increase**: Prevent policy collapse to single strategy

## Run H Results & Analysis (COMPLETED 2026-02-23)

**Config**: Partial-obs, no DCBF, distance_scale=10, pursuer_v_max=1.2, fixed_speed, **curriculum enabled**, 12 phases × 300K steps, 10×10 arena, 16 envs, n_steps=2048, batch_size=512, GPU

```bash
./venv/bin/python -u scripts/train_amsdrl.py \
    --max_phases 12 --timesteps_per_phase 300000 \
    --no_dcbf \
    --distance_scale 10.0 --pursuer_v_max 1.2 --fixed_speed \
    --arena_width 10.0 --arena_height 10.0 --max_steps 600 \
    --n_envs 16 --n_steps 2048 --batch_size 512 \
    --curriculum \
    --seed 42 --output results/stage3/run_h_curriculum
```

| Phase | SR_P | SR_E | NE Gap | Level | Notes |
|-------|------|------|--------|-------|-------|
| S0 | 33% | 67% | — | 1 | Close range cold-start |
| S1 | **93%** | 7% | 0.86 | **1→2** | Mastered close range |
| S2 | 26% | 74% | 0.48 | 2 | Evader counter-adapts |
| S3 | 32% | 68% | 0.36 | 2 | Pursuer improving |
| S4 | 30% | 70% | 0.40 | 2 | Slight oscillation |
| S5 | **71%** | 29% | 0.42 | **2→3** | Mastered medium range |
| S6 | 37% | 63% | **0.26** | 3 | Best NE gap! Obstacles introduced |
| S7 | **86%** | 14% | 0.72 | **3→4** | Mastered obstacles quickly |
| S8 | 83% | 17% | 0.66 | 4 | Evader barely adapts |
| S9 | **100%** | 0% | 1.00 | 4 | Pursuer dominates |
| S10 | **100%** | 0% | 1.00 | 4 | Evader can't recover |
| S11 | **100%** | 0% | 1.00 | 4 | Pursuer locked in |
| S12 | **100%** | 0% | 1.00 | 4 | Final: pursuer dominates |

**Convergence**: Not converged (NE gap 1.0 at Level 4). Best NE gap 0.26 at Level 3 (S6).
**Total time**: 1.7h on GPU (RTX 5090).
**Curriculum progression**: L1→L2 (S1), L2→L3 (S5), L3→L4 (S7) — all 4 levels reached by phase 7 of 12.

### Key Findings from Run H
1. **Curriculum progression works**: All 4 levels reached naturally via 70% capture threshold. Smooth, predictable advancement.
2. **Best NE gap 0.26** at Level 3 (S6) — comparable to Run G's best (0.14), but achieved in a structured way rather than by oscillation.
3. **Level 4 collapse**: Once at full scenario (variable distance + obstacles), the pursuer dominates 100%. The evader can't learn to use obstacles for cover within 300K steps.
4. **Obstacles didn't destabilize training**: Levels 3-4 introduced obstacles without crashes. PursuitEvasionEnv handles obstacle generation correctly.
5. **Bug fix during run**: `_make_partial_obs_env()` and eval functions didn't accept `min_init_distance`/`max_init_distance` from curriculum overrides. Fixed by adding these params to function signatures.

### Why Level 4 Collapses
The Level 4 collapse is the same root cause as Run G's oscillation — each agent overfits to the frozen opponent:
- The pursuer learns to exploit the frozen evader's policy perfectly
- The evader can't adapt in 300K steps against a 100%-winning pursuer
- Once at 100%, the evader's reward signal becomes pure penalty (never escapes), making learning harder

### Curriculum vs No-Curriculum (Run G vs H)
| Metric | Run G (no curriculum) | Run H (curriculum) |
|--------|----------------------|-------------------|
| Phases | 6 | 12 |
| Best NE gap | 0.14 (S1) | 0.26 (S6) |
| Final NE gap | 0.42 | 1.00 |
| Oscillation | Wild (0.14→0.96→0.42) | Smooth progression then collapse |
| Obstacles | Never used | Used in L3-L4 |
| Level progression | N/A | L1→L2→L3→L4 |
| Key advantage | Lower min NE gap | Structured difficulty progression |

### Recommended Next Steps
1. **Opponent pool (Session 7)**: Sample opponent from recent N checkpoints — prevents overfitting to single frozen opponent
2. **More evader training at L4**: Give evader 2-3× more steps than pursuer at max level
3. **Entropy boost at level transitions**: Increase ent_coef when advancing to prevent immediate exploitation of new scenario

## Remaining Fixes for Partial-Obs (saved for later)

### Fix 2: Add "Last Seen" Features to Observation
- Add `last_known_x, last_known_y, steps_since_seen` to state features
- Gives pursuer persistent memory beyond K=10 history window
- Modify PartialObsWrapper._build_obs_dict() and RAW_OBS_DIM accordingly

### Fix 3: Increase History Length K
- Change K from 10 to 30 or 50
- More history = more detection events preserved in buffer

### Fix 4: Add Exploration Reward
- When blind (d_to_opp == -1), reward pursuer for rotating/scanning
- Small bonus for angular velocity when not detecting evader

### Fix 5: Curriculum from Full-Obs to Partial-Obs
- Start self-play with full observability (100% detection range)
- Gradually reduce FOV range: 28m → 20m → 15m → 10m

### Fix 6: Pre-train BiMDN Properly Before Self-Play
- Fix Stage 1 data collection (use intentional pursuit, not random)
- Achieve Gate 1 (out-of-FOV RMSE < 5m) before attempting self-play

## Files Modified This Session
| File | Changes |
|------|---------|
| `envs/wrappers.py` | Added FixedSpeedWrapper, FixedSpeedModelAdapter |
| `envs/rewards.py` | No changes (distance_scale was already a param) |
| `training/amsdrl.py` | Added distance_scale, pursuer_v_max, fixed_speed params throughout pipeline |
| `training/selfplay_callbacks.py` | Added fixed_speed_v_max to FixedBaselineEvalCallback |
| `scripts/train_amsdrl.py` | Added --distance_scale, --pursuer_v_max, --fixed_speed CLI args |
| `docs/stage3_diagnosis.md` | This file — complete diagnosis and status |
