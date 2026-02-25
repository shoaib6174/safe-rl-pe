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
| I (opponent pool) | Same as H + opponent_pool_size=5 | 84% | 26% | **49%** | **49%** | **CONVERGED** NE gap 0.02! But at Level 2 only (premature) |
| J (curriculum gate) | Same as I + convergence requires max level | 80% | 21%→86% | 100%→100% | 100% | Gate works. Same L4 collapse — pool alone insufficient |
| K (6-level) | Same as J + 6-level curriculum, 18 phases | 86% | 20% | 99% | 99% | L4 collapse persists — 6-level curriculum alone insufficient |
| L (occlusion) | Same as K + w_occlusion=0.05 | 86% | 15% | 99% | 99% | L4 collapse — occlusion bonus too weak (0.05) |
| M (speed+occlusion) | Same as K + w_occlusion=0.2, v_max=1.1 | 70% | 36% | 96% | 96% | L4 collapse persists — occlusion bonus approach fundamentally flawed |
| N (visibility) | Same as M + visibility reward + survival bonus | 52% | 15% | 97% | 97% | Visibility reward alone insufficient — evader can't reach obstacles to benefit |
| O (prep phase) | Same as N + prep_steps=100 | 52% | 45% | 90% | 90% | Prep phase slows L1-L2 but same L3+ collapse — evader doesn't know to move toward obstacles |

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

## Current Status (2026-02-24)

**Runs N and O both collapsed at obstacle levels (L3+).** The visibility reward helps slightly
(SR_E=0.10-0.12 vs ~0.01-0.04 in Runs H-M), but the evader can't learn to USE obstacles
even with 100 steps of head start (prep phase). The issue: during prep phase, the evader
has no gradient toward obstacles — the visibility reward only activates once BEHIND an obstacle.
The evader needs either (a) to spawn near obstacles, or (b) an obstacle-seeking reward during prep.

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

## Run I Results & Analysis (COMPLETED 2026-02-23)

**Config**: Same as Run H + `--opponent_pool_size 5` (curriculum + opponent pool)

```bash
./venv/bin/python -u scripts/train_amsdrl.py \
    --max_phases 12 --timesteps_per_phase 300000 \
    --no_dcbf \
    --distance_scale 10.0 --pursuer_v_max 1.2 --fixed_speed \
    --arena_width 10.0 --arena_height 10.0 --max_steps 600 \
    --n_envs 16 --n_steps 2048 --batch_size 512 \
    --curriculum --opponent_pool_size 5 \
    --seed 42 --output results/stage3/run_i_opponent_pool
```

| Phase | SR_P | SR_E | NE Gap | Level | Pool Status |
|-------|------|------|--------|-------|-------------|
| S0 | 33% | 67% | — | 1 | Cold-start |
| S1 | 84% | 16% | 0.68 | 1→2 | No checkpoints yet |
| S2 | 26% | 74% | 0.48 | 2 | 14 model, 2 random |
| S3 | **49%** | **51%** | **0.02** | 2 | 7 model, 9 random |

**Converged**: Yes — NE gap 0.02 at phase S3.
**Total time**: 0.5h (3.4× faster than Run H).

### Run H vs Run I Comparison

| Metric | Run H (Curriculum) | Run I (Curriculum + Pool) |
|--------|-------------------|--------------------------|
| Converged | No | **Yes** |
| Phases | 12 | **4** |
| Time | 1.7h | **0.5h** |
| Final NE Gap | 1.00 (collapse) | **0.02** |
| Min NE Gap | 0.26 | **0.02** |
| Final SR_P/SR_E | 100%/0% | **49%/51%** |
| Curriculum Level | 4 (full) | 2 (no obstacles) |

### Key Findings from Run I
1. **Opponent pool completely prevents oscillation**: NE gap decreased monotonically (0.68 → 0.48 → 0.02)
2. **Near-perfect Nash Equilibrium**: SR_P=0.49, SR_E=0.51 — the closest to balanced play achieved so far
3. **Premature convergence**: Agents converged at Level 2 (no obstacles). Never tested with obstacles.
4. **Root cause of premature convergence**: The convergence criterion (NE gap < 0.10) had no curriculum gate — it could trigger at any level

### Fix Applied
Added `curriculum_at_max` gate to convergence check in `amsdrl.py`:
```python
curriculum_ready = (self.curriculum is None) or self.curriculum.at_max_level
if ne_gap < self.eta and curriculum_ready:
    converged = True
```
Convergence now requires being at the max curriculum level before declaring convergence.

## Run J Results & Analysis (COMPLETED 2026-02-23, curriculum gate fix)

**Config**: Same as Run I but with curriculum convergence gate (must reach Level 4 before converging).

| Phase | SR_P | SR_E | NE Gap | Level | Notes |
|-------|------|------|--------|-------|-------|
| S0 | 33% | 67% | — | 1 | Cold-start |
| S1 | 80% | 20% | 0.60 | 1→2 | |
| S2 | 21% | 79% | 0.58 | 2 | |
| S3 | 47% | 53% | **0.06** | 2 | Gate held — "curriculum not at max level (2/4)" |
| S4 | 47% | 53% | **0.06** | 2 | Gate held again |
| S5 | 86% | 14% | 0.72 | 2→3 | |
| S6 | 86% | 14% | 0.72 | 3→4 | Rapid advancement |
| S7 | 98% | 2% | 0.96 | 4 | **Collapse begins** |
| S8 | 97% | 3% | 0.94 | 4 | |
| S9 | 100% | 0% | 1.00 | 4 | Full collapse |
| S10 | 100% | 0% | 1.00 | 4 | |
| S11 | — | — | — | 4 | Health monitor rollbacks (5×) |

### Key Findings from Run J
1. **Curriculum gate works correctly**: S3 and S4 properly held convergence (NE gap 0.06 < 0.10 but only at Level 2)
2. **Opponent pool keeps balance at Level 2**: NE gap 0.06 at Level 2 — excellent
3. **Same Level 4 collapse**: Once at Level 4, the pursuer dominates 100% — identical to Run H
4. **Opponent pool does NOT prevent Level 4 collapse**: The pool prevents oscillation but not the fundamental asymmetry at Level 4
5. **Rapid L3→L4 advancement**: Level 3 was mastered in one phase (S6), suggesting it's too easy or too similar to Level 2

### Conclusion from Run J
The Level 4 collapse is NOT caused by opponent overfitting (which the pool fixes). It's a **curriculum design problem** — the jump from Level 3 (close, 2-5m, obstacles) to Level 4 (variable, 2-15m, obstacles) is too large.

## Level 4 Collapse: Deep Analysis

### Three Compounding Causes

**1. Speed asymmetry guarantees capture in open space**

With `v_pursuer=1.2` and `v_evader=1.0`, the pursuer is 20% faster. In open space without obstacles blocking the path, the evader **cannot escape** — the distance will always close. The evader's only hope is to use obstacles as barriers to force the pursuer into longer paths. But:

**2. The reward function gives zero incentive to use obstacles**

```python
r_evader = -(distance_shaping + capture_bonus + timeout_penalty)
```

The evader is rewarded for increasing distance from the pursuer. Hiding behind an obstacle **doesn't increase distance** — it maintains it. There is no reward for:
- Obstacle proximity or hiding
- Line-of-sight breaking
- Using obstacles as barriers

So the policy never learns to seek obstacles as a defensive strategy.

**3. The Level 3→4 curriculum jump is too large**

| Level | Distance | Obstacles | What the evader learned |
|-------|----------|-----------|------------------------|
| 3 | 2–5m | 3 | Close-range obstacle evasion (pursuer always in FOV) |
| 4 | 2–15m | 3 | ??? (pursuer often starts outside FOV at >10m) |

At Level 3, the evader always starts close to the pursuer (2-5m). The FOV range is 10m, so the evader **always sees the pursuer** and can react. The evader learns reactive evasion: "when I see the pursuer, turn away."

At Level 4, the evader can start 15m from the pursuer — **outside FOV range**. The evader receives `d_to_opp=-1` (not detected) and has no escape behavior. Meanwhile the pursuer (also partially blind) explores and eventually finds the evader. By the time FOV detection occurs, the faster pursuer is already closing rapidly.

The evader's Level 3 policy (reactive close-range evasion) doesn't generalize to Level 4's variable distances.

### Why the Opponent Pool Doesn't Help at Level 4

The opponent pool prevents **policy forgetting** (oscillation between phases). But the Level 4 collapse isn't about forgetting — it's about the evader being **physically unable to survive** with:
- A 20% speed disadvantage
- A distance-only reward that doesn't incentivize obstacle use
- No prior training at medium-to-far distances with obstacles

### Fix Applied: 6-Level Curriculum

Expanded curriculum from 4 to 6 levels with intermediate distance+obstacle stages:

| Level | Distance | Obstacles | Description | Rationale |
|-------|----------|-----------|-------------|-----------|
| 1 | 2–5m | 0 | Close, no obstacles | Baseline — learn basic pursuit/evasion |
| 2 | 5–15m | 0 | Medium distance, no obstacles | Learn long-range tracking/evasion |
| 3 | 2–5m | 3 | Close, with obstacles | Learn obstacle awareness at close range |
| **4** | **5–8m** | **3** | **Medium distance, with obstacles** | **NEW — bridge close to medium with obstacles** |
| **5** | **5–12m** | **3** | **Far distance, with obstacles** | **NEW — extend to larger distances with obstacles** |
| 6 | 2–15m | 3 | Full scenario | Final difficulty — variable distance + obstacles |

**Key design decisions:**
- Levels 4 and 5 bridge the gap between L3 (close+obstacles) and L6 (full scenario)
- The evader gets to practice obstacle-based evasion at progressively larger distances
- L4 (5-8m) stays within FOV range (10m) so the evader always detects the pursuer
- L5 (5-12m) introduces occasional FOV non-detection (>10m spawns)
- L6 is the same as old L4 — full variable distance + obstacles

### Potential Further Fixes (if 6-level curriculum insufficient)

**Fix B — Obstacle-aware reward shaping**: Add bonus for evader when line-of-sight to pursuer is blocked by obstacle. Directly incentivizes hiding behavior.

**Fix C — Reduce speed asymmetry**: Change `pursuer_v_max` from 1.2 to 1.1 or 1.05. Smaller advantage gives evader more time to learn spatial strategies.

**Fix D — Asymmetric training budgets**: Give evader 2× more training steps than pursuer at higher levels, since evasion with obstacles is harder to learn than pursuit.

## Run K Results & Analysis (COMPLETED 2026-02-23, killed — L4 collapse)

**Config**: Same as Run J but with 6-level curriculum and 18 max phases.

| Phase | SR_P | SR_E | NE Gap | Level | Notes |
|-------|------|------|--------|-------|-------|
| S0 | 33% | 67% | — | 1 | Cold-start |
| S1 | 86% | 14% | 0.72 | 1→2 | |
| S2 | 20% | 80% | 0.60 | 2 | Evader adapts |
| S3 | 83% | 17% | 0.66 | 2→3 | |
| S4 | 72% | 28% | 0.44 | 3→4 | Advanced to L4 (medium+obstacles) |
| S5 | **99%** | 1% | **0.98** | 4→5 | **Collapse at L4** |

**Killed** after S5 — same collapse pattern as Runs H/J, just at the new L4 (medium distance + obstacles) instead of old L4.

### Key Findings from Run K
1. **6-level curriculum alone doesn't fix the collapse**: The collapse merely shifted from old L4 (2-15m) to new L4 (5-8m). The evader still can't learn to use obstacles with a distance-only reward.
2. **Confirms root cause is reward + speed, not curriculum granularity**: Even at 5-8m (well within FOV), the pursuer dominates with obstacles present because the evader has no incentive to seek obstacles.

## Run L Results & Analysis (COMPLETED 2026-02-23, killed — L4 collapse)

**Config**: Same as Run K + `w_occlusion=0.05` (weak occlusion bonus for evader).

| Phase | SR_P | SR_E | NE Gap | Level | Notes |
|-------|------|------|--------|-------|-------|
| S0 | 33% | 67% | — | 1 | Cold-start |
| S1 | 86% | 14% | 0.72 | 1→2 | |
| S2 | 15% | 85% | 0.70 | 2 | Evader adapts |
| S3 | 83% | 17% | 0.66 | 2→3 | |
| S4 | 76% | 24% | 0.52 | 3→4 | L3 healthy — but advanced to L4 |
| S5 | **99%** | 1% | **0.98** | 4→5 | **Collapse at L4** — health rollback triggered |

**Killed** after S5.

### Key Findings from Run L
1. **Occlusion bonus w=0.05 is too weak**: The per-step bonus (+0.05) is negligible compared to capture bonus (+100) and distance shaping. The evader doesn't learn to prioritize hiding.
2. **Identical collapse pattern**: SR_P=0.99 at L4, same as Runs H/J/K. The occlusion signal didn't change evader behavior.
3. **L3 looked slightly better** (SR_P=0.76 vs 0.72 in Run K) but not enough to prevent L4 collapse.

### Decision: Combine Fix C (speed reduction) + stronger Fix B (occlusion)
- Reduce `pursuer_v_max` from 1.2 to 1.1 (10% advantage instead of 20%)
- Increase `w_occlusion` from 0.05 to 0.2 (4× stronger occlusion signal)
- Hypothesis: smaller speed gap gives evader more time to reach obstacles; stronger occlusion reward makes hiding behind them worthwhile

## Run M Results & Analysis (COMPLETED 2026-02-24, killed — L4 collapse persists)

**Config**: Same as Run K but with `pursuer_v_max=1.1` and `w_occlusion=0.2`.

| Phase | SR_P | SR_E | NE Gap | Level | Notes |
|-------|------|------|--------|-------|-------|
| S0 | 26% | 74% | — | 1 | Lower cold-start (slower pursuer) |
| S1 | 70% | 30% | 0.40 | 1 | Slower improvement (v=1.1 effect) |
| S2 | 64% | 36% | 0.28 | 1 | Best L1 balance yet |
| S3 | 90% | 10% | 0.80 | 1→2 | |
| S4 | 83% | 17% | 0.66 | 2→3 | Rapid advance |
| S5 | **96%** | 4% | **0.92** | 3→4→5 | **4 health rollbacks, still 96%** |

**Killed** after S5 (niro-2 restarted; relaunched as run_m2, same collapse pattern).

### Key Findings from Run M
1. **Reduced speed (1.1 vs 1.2) helps slightly at L1**: NE gap 0.28 at L1 is the best yet (vs 0.40-0.72 in prior runs)
2. **Still collapses at obstacle levels**: SR_P=0.96 despite 4 health rollbacks. The 10% speed gap is still enough for the pursuer to dominate.
3. **Occlusion bonus w=0.2 is still insufficient**: The additive bonus approach fundamentally doesn't work. The bonus (0.2/step) is drowned out by distance shaping (~0.35/step from scale=10) and capture bonus (100).
4. **Root cause confirmed**: The problem is NOT the weight of the bonus — it's that **distance shaping actively teaches the wrong behavior**. The evader's gradient points toward "maximize distance" (flee in straight line), not "get behind obstacle."

### Why Additive Occlusion Bonuses Can't Work

The evader's reward accumulation per episode:
- **Distance shaping**: ~0.35/step × 100 steps = **35 reward** for fleeing
- **Occlusion bonus**: 0.2/step × 50 hidden steps = **10 reward** for hiding
- **Capture penalty**: -100 when caught

The distance shaping signal (35) overwhelms the occlusion signal (10). Even with w_occlusion=1.0, the evader gets 50 for hiding vs 35 for fleeing — not a strong enough differential to overcome the policy's existing bias toward distance maximization. The only solution: **replace distance shaping with visibility-based reward**.

## Deep Analysis: Why the Level 4 Collapse Persists (Literature Review)

After 6 runs (H through M) all showing the same collapse pattern at obstacle levels, we conducted a deep literature review to understand the root cause and find evidence-based solutions.

### The Fundamental Problem

**Distance shaping is anti-correlated with obstacle use.** The evader's reward `r_E = -(d_prev - d_curr)/d_max` teaches: "maximize distance from pursuer." Hiding behind an obstacle *maintains* distance (zero reward) while fleeing in a straight line *increases* distance (positive reward). The policy gradient literally points away from obstacles.

This is diagnosed from the evader's perspective:
- Fleeing in straight line: +0.35/step reward, but caught in 100 steps → **total: +35 - 100 = -65**
- Hiding behind obstacle: +0.0/step (no distance change), survive 600 steps → **total: 0 + 50 (timeout) = +50**

The hiding strategy is *objectively better*, but the per-step gradient from distance shaping is always positive for fleeing and zero for hiding. PPO follows the per-step gradient, not the long-term outcome.

### Key Reference: OpenAI Hide-and-Seek (Baker et al., ICLR 2020)

The most relevant prior work is OpenAI's "Emergent Tool Use From Multi-Agent Autocurricula" (arXiv:1909.07528). Their game is structurally identical to ours: hiders vs seekers with obstacles.

**Critical design choice**: They used a **purely binary, per-timestep, visibility-based reward**:
- Hiders: +1/step if hidden, -1/step if visible
- Seekers: -1/step if hidden, +1/step if visible
- **No distance shaping at all**
- Episode length: 240 steps, 40% preparation phase (seekers frozen)

This reward structure was sufficient to produce 6 phases of emergent strategy: running, fort-building, ramp-climbing, ramp-theft, box-surfing, and ramp-surfing defense. All without any explicit incentive to interact with objects.

**Key insight**: Removing distance shaping forces agents to discover that hiding/seeking is the ONLY way to get reward. The gradient directly points toward obstacle use.

### Solution: Visibility-Based Evader Reward (Fix 1+2)

Implemented as **Mode B** in `RewardComputer`:

**Pursuer reward**: Unchanged — distance shaping + capture bonus + timeout penalty. The pursuer needs to learn "chase and catch."

**Evader reward** (when obstacles present):
- `r_evader = visibility_weight × (+1 if hidden, -1 if visible) + survival_bonus`
- No distance shaping for evader
- Terminal steps (capture/timeout) remain zero-sum with pursuer
- When no obstacles (curriculum L1/L2): falls back to standard zero-sum distance-based

**Why this should work**:
- Over 600 steps, perfectly hidden evader: 600 × (1 + 1) = **+1200 reward**
- Over 600 steps, always-visible evader: 600 × (-1 + 1) = **0 reward**
- Captured at step 200, half hidden: 100 × 2 + 100 × 0 = **+200**, then episode ends
- The gradient directly and strongly points toward hiding behind obstacles
- Early termination (capture) naturally penalizes — fewer steps = less reward

### Additional Fixes Evaluated

| # | Fix | Status | Notes |
|---|-----|--------|-------|
| 1 | **Visibility reward** | **IMPLEMENTED, TESTED (Run N)** | Helps slightly (SR_E 0.04→0.12) but insufficient alone |
| 2 | **Survival bonus** | **IMPLEMENTED, TESTED (Run N)** | +1/step alive, combined with visibility |
| 3 | **Preparation phase** | **IMPLEMENTED, TESTED (Run O)** | 100 steps freeze — evader wastes time without obstacle-seeking gradient |
| 4 | Evader pre-training vs scripted pursuer | Deferred | Bootstrap obstacle skills before self-play |
| 5 | Asymmetric training budgets | Deferred | 2-3× more evader steps at obstacle levels |
| 6 | Subgame curriculum (SACL) | Deferred | Reset to interesting states for practice |
| 7 | RND intrinsic motivation | Deferred | Curiosity-driven obstacle region exploration |
| 8 | AlphaStar league training | Deferred | Too complex for now |
| 9 | Role-conditioned advantage normalization | Deferred | Per-agent advantage normalization in PPO |
| **10** | **Obstacle proximity shaping (prep phase)** | **NEXT** | Reward evader for moving toward nearest obstacle during prep |
| **11** | **Spawn evader near obstacles** | Candidate | Place evader within 1-2m of obstacle at L3+ |

**References**:
- Baker et al., "Emergent Tool Use From Multi-Agent Autocurricula" (ICLR 2020, arXiv:1909.07528)
- Vinyals et al., "AlphaStar: Grandmaster Level in StarCraft II" (Nature 2019)
- Li et al., "Accelerate MARL with Subgame Curriculum Learning" (AAAI 2024, arXiv:2310.04796)
- Burda et al., "Exploration by Random Network Distillation" (ICLR 2019, arXiv:1810.12894)

## Run N Results & Analysis (COMPLETED 2026-02-24, killed — L3+ collapse)

**Config**: Same as Run M but replacing `w_occlusion=0.2` with `--use_visibility_reward --survival_bonus 1.0`.

```bash
./venv/bin/python -u scripts/train_amsdrl.py \
    --max_phases 18 --timesteps_per_phase 300000 \
    --no_dcbf \
    --distance_scale 10.0 --pursuer_v_max 1.1 --fixed_speed \
    --arena_width 10.0 --arena_height 10.0 --max_steps 600 \
    --n_envs 16 --n_steps 2048 --batch_size 512 \
    --curriculum --opponent_pool_size 5 \
    --use_visibility_reward --survival_bonus 1.0 \
    --seed 42 --output results/stage3/run_n
```

| Phase | SR_P | SR_E | NE Gap | Level | Notes |
|-------|------|------|--------|-------|-------|
| S0 | 26% | 74% | — | 1 | Good cold-start baseline |
| S1 | 74% | 26% | 0.48 | 1→2 | Promoted to L2 |
| S2 | 15% | 85% | 0.70 | 2 | Evader fights back |
| S3 | 40% | 60% | 0.20 | 2 | |
| S4 | 53% | 47% | **0.06** | 2 | Near NE at L2! |
| S5 | 96% | 4% | 0.92 | 2→3 | Promoted to L3 (obstacles) |
| S6 | **88%** | **12%** | 0.76 | 3→4 | Evader: 0.04→0.12 (slight improvement only) |
| S7 | 97% | 3% | 0.94 | 4→5 | 1 health rollback, still collapsed |
| S8 | 97% | 3% | 0.94 | 5→6 | Evader learned NOTHING |
| S9 | — | — | — | 6 | Another health rollback — **killed** |

**Killed** after S9. Curriculum blew through L3→L6 with SR_P ≥ 0.88 at every level.

### Key Findings from Run N
1. **Visibility reward alone is insufficient**: At L3, the evader improved from SR_E=0.04 to 0.12 — better than Runs H-M (0.01-0.04) but far from competitive.
2. **Evader can't REACH obstacles**: The evader gets caught before it can get behind an obstacle. The visibility reward only provides gradient once hidden — there's no gradient TOWARD obstacles.
3. **L2 convergence is excellent**: NE gap 0.06 at L2 (no obstacles). Visibility reward + zero-sum fallback at no-obstacle levels works perfectly.
4. **Curriculum promotion too aggressive**: Promotion at 0.70 capture rate means the evader barely adapts before advancing.

### Root Cause: Missing Gradient Toward Obstacles
The visibility reward provides: `+1 if hidden, -1 if visible`. But the evader starts visible and stays visible — so it gets -1/step consistently. To get +1, it must be behind an obstacle, but there's no reward that increases as it approaches an obstacle. The reward landscape is a **plateau at -1** with an unreachable **spike at +1** behind obstacles.

## Run O Results & Analysis (COMPLETED 2026-02-24, killed — L3+ collapse with prep phase)

**Config**: Same as Run N + `--prep_steps 100` (freeze pursuer for first 100 steps = 5 seconds).

```bash
./venv/bin/python -u scripts/train_amsdrl.py \
    --max_phases 18 --timesteps_per_phase 300000 \
    --no_dcbf \
    --distance_scale 10.0 --pursuer_v_max 1.1 --fixed_speed \
    --arena_width 10.0 --arena_height 10.0 --max_steps 600 \
    --n_envs 16 --n_steps 2048 --batch_size 512 \
    --curriculum --opponent_pool_size 5 \
    --use_visibility_reward --survival_bonus 1.0 \
    --prep_steps 100 \
    --seed 42 --output results/stage3/run_o
```

| Phase | SR_P | SR_E | NE Gap | Level | Notes |
|-------|------|------|--------|-------|-------|
| S0 | 41% | 59% | — | 1 | Higher capture rate (prep reduces effective chase time) |
| S1 | 52% | 48% | **0.04** | 1 | Near NE! Prep gives evader real advantage |
| S2 | 45% | 55% | 0.10 | 1 | |
| S3 | 98% | 2% | 0.96 | 1→2 | Pursuer learned to overcome head start |
| S4 | 69% | 31% | 0.38 | 2 | |
| S5 | 92% | 8% | 0.84 | 2→3 | Promoted to obstacles |
| S6 | **90%** | **10%** | 0.80 | 3→4 | Only SR_E=0.10 — same as Run N |
| S7 | 97% | 3% | 0.94 | 4→5 | 3 health rollbacks, still collapsed |
| S8 | — | — | — | 5 | **Killed** |

### Key Findings from Run O
1. **Prep phase slows early curriculum**: L1 took 3 phases instead of 1 (S1-S3 vs just S1 in Run N). The 100-step head start makes L1 harder for the pursuer — this is good for balance but delays reaching obstacle levels.
2. **Same L3+ collapse**: SR_E=0.10 at L3 — virtually identical to Run N (SR_E=0.12). The prep phase didn't help at obstacle levels.
3. **Why prep phase fails**: The evader has 100 steps (5 seconds, ~5m travel) to reach an obstacle, but it **doesn't know to move toward obstacles**. It uses the prep time to flee in a random direction — the same direction it would go without prep. Without a gradient toward obstacles during prep, the extra time is wasted.

### The Deeper Problem: No Obstacle-Seeking Gradient

The evader needs THREE capabilities to use obstacles:
1. **Time to reach obstacles** ← prep phase provides this
2. **Incentive to hide behind obstacles** ← visibility reward provides this
3. **Knowledge of WHERE obstacles are** ← **MISSING**

The evader's observation includes obstacle positions (from PartialObsWrapper), but there's no reward gradient that says "move toward the nearest obstacle." The visibility reward only activates once hidden. The prep phase gives time but no direction.

### Potential Fixes (Priority Order)

**Fix A — Obstacle proximity shaping during prep phase**: During prep steps, add a small reward for moving toward the nearest obstacle. This provides the missing gradient. Example: `r_approach = 0.1 * (d_prev_to_obs - d_curr_to_obs) / d_max`. Only active during prep phase; after prep, pure visibility reward takes over.

**Fix B — Spawn evader near obstacles**: Modify initial state distribution at obstacle levels (L3+) to place the evader within 1-2m of an obstacle. Removes the need to "find" obstacles entirely. Simple but reduces game diversity.

**Fix C — Reduce prep_steps + increase speed ratio**: Use prep_steps=40 (2 seconds) and v_max=1.0 (equal speeds). The evader needs less head start if the pursuer isn't faster. But equal speeds may make the game unsolvable for the pursuer without obstacles.

**Fix D — Explicit obstacle-seeking cold-start**: Pre-train the evader in a NavigationEnv variant where the goal is always behind an obstacle. Bootstraps obstacle-seeking behavior before self-play.

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
