# Evader Strategy Analysis: Breaking the Obstacle-Hugging Monoculture

**Date**: 2026-03-01 (Session 59)
**Context**: SP3 warm-seeded self-play shows cycling — pursuer cracks obstacle-hugging, evader has no fallback

## Problem Statement

The evader learns exactly one strategy: **obstacle-hugging** (orbiting tightly around obstacles). This works brilliantly early on (98% escape vs greedy at variable speed), but once the pursuer learns to predict orbit patterns and cut off, the evader collapses with no alternative. SP3 shows this clearly:

- M1100: Near-convergence (49/51) — both agents competitive
- M1200–M1800: Pursuer dominates 73–98%, evader gets ALL training but can't recover
- The evader has had 700K+ steps of focused training and still can't break through

### Root Causes

1. **Full observability kills deception**: The pursuer always knows the evader's exact position, heading, and velocity. Feinting is pointless — the pursuer reacts to reality, not predictions.

2. **Equal speed + bounded arena = guaranteed capture without obstacles**: Proportional navigation catches the evader given enough time. Obstacles are the ONLY thing preventing capture.

3. **Obstacle-hugging is the dominant pure strategy**: It's not a local optimum — it's genuinely the best strategy given full information. Once cracked, there's nothing else in the strategy space.

4. **PPO learns policies, not mixed strategies**: The Nash equilibrium likely requires randomizing between orbit directions, obstacle switching, and feinting. PPO converges to a single deterministic policy, causing the cycling.

5. **Early success prevents exploration**: 98% escape rate early → policy concentrates all probability mass on orbiting → entropy collapses → alternatives never discovered.

---

## Current Approach: Obstacle-Hugging

**Mechanism**: The evader orbits tightly around obstacles, using them as physical barriers. With variable speed, it can execute tighter turns via v/omega tradeoff.

**Strengths**:
- 98% escape rate vs greedy pursuer (S1v2b)
- Exploits genuine geometric advantage (obstacle blocks pursuit path)
- Easy to learn — clear reward gradient (closer to obstacle = harder to catch)

**Weaknesses**:
- Single strategy — once cracked, no fallback
- Predictable — orbit direction and speed become learnable patterns
- Doesn't exploit information asymmetry (no hiding, no deception)
- Doesn't generalize — fails if obstacle layout changes significantly

---

## Proposed Approaches (Ranked by Impact × Feasibility)

### Approach 1: Partial Observability — LOS Masking ★★★★★ (RECOMMENDED)

**Core idea**: When line-of-sight to the opponent is blocked by an obstacle, mask the opponent's state in the observation (replace with zeros). This creates an **information asymmetry game** where hiding behind obstacles denies the opponent knowledge of your position.

**New strategies this enables**:
- **Stealth repositioning**: Move while hidden, reappear somewhere unexpected
- **Deception**: Show yourself on one side, hide, emerge on the other
- **Information management**: Actively manage what the pursuer can/can't see
- **Ambush**: Wait behind obstacle, move when pursuer commits wrong direction

**Implementation**:
- Modify `ObservationBuilder.build()` to accept `los_blocked` flag
- When blocked, zero out opponent features (position, heading, velocity, distance, bearing)
- Add `los_visible` binary flag to observation so agent knows visibility state
- Compute LOS in `PursuitEvasionEnv._get_obs()` using existing `line_of_sight_blocked()`
- Observation dimension: +1 (for `los_visible` flag) = 15 + 2*n_obstacle_obs

**Infrastructure already built**:
- `line_of_sight_blocked()` in `envs/rewards.py` — fully implemented and tested (12 tests)
- Visibility reward (Mode B) already uses this function

**Complexity**: Medium — modifies observation builder + env, may want LSTM later
**Risk**: Low — principled game-theoretic change, not hacky reward shaping
**This IS Phase 3** — partial observability is the stated research goal

---

### Approach 2: Randomize Obstacle Count (1–3 per episode) ★★★★☆

**Core idea**: Instead of fixed 2 obstacles, randomly sample 1–3 obstacles per episode. Prevents memorizing a single layout, forces generalization.

**New strategies this enables**:
- Adaptation to different obstacle configurations
- Different orbiting patterns for different obstacle counts
- Obstacle switching (more viable with 3 obstacles)

**Implementation**: 1-line change in env creation — `n_obstacles = rng.randint(1, 4)`

**Complexity**: Very low
**Risk**: Low — evader may just learn to orbit whatever's available
**Limitation**: Doesn't create fundamentally new strategy types, just variants of orbiting

---

### Approach 3: Adaptive Entropy in Self-Play ★★★☆☆

**Core idea**: When the evader is losing, boost `ent_coef` from 0.03 to 0.05–0.10. This forces broader action distributions, approximating mixed strategies.

**Mechanism**: Higher entropy → more exploration → better chance of discovering alternatives. Formula: `ent_coef = base * max(1.0, 2.0 * (1.0 - SR_E))`

**Implementation**: Modify AMSDRLSelfPlay `_run_micro_phases()` to dynamically adjust ent_coef based on success rate.

**Complexity**: Low
**Risk**: Medium — too much entropy prevents learning any coherent strategy
**Limitation**: Encourages action-space noise, not strategy-space diversity

---

### Approach 4: Strategy-Conditioned Policy (Latent z) ★★★☆☆

**Core idea**: Add a discrete latent variable z ~ Categorical(K) sampled at episode start. The policy conditions on z, producing different behavioral modes. A diversity loss (mutual information between z and trajectory features) ensures different z values produce genuinely different strategies.

**New strategies this enables**:
- z=0: Orbit obstacle A
- z=1: Orbit obstacle B
- z=2: Switch between obstacles
- z=3: Sprint-and-dodge in open field

**Implementation**:
- Add z to observation space (one-hot encoded)
- Sample z at episode start in wrapper
- Add diversity loss: maximize I(z; trajectory_features)
- Similar to DIAYN (Diversity Is All You Need) framework

**Complexity**: High — requires custom policy architecture + auxiliary loss
**Risk**: Medium — diversity objective may conflict with task reward
**Potential**: Very high if it works — directly creates multiple behavioral modes

---

### Approach 5: Population of Specialist Evaders ★★☆☆☆

**Core idea**: Train multiple evaders in different environment configurations (1 obstacle near wall, 2 spread, 3 clustered, etc.) and seed all into the self-play opponent pool. PFSP then samples from the diverse pool.

**New strategies this enables**:
- Pool-level diversity: different evaders use different strategies
- Pursuer must generalize across multiple opponent types
- Natural curriculum as different specialists challenge different aspects

**Implementation**:
- Train N specialist evaders (each ~3M steps, focused training)
- Seed all into opponent pool at self-play start
- PFSP sampling naturally selects challenging opponents

**Complexity**: Medium — mostly training orchestration, no code changes
**Risk**: All specialists may converge to obstacle-hugging anyway
**Limitation**: Doesn't create within-policy diversity

---

## Recommendation

**Implement Approach 1 (Partial Observability) first.** It's the highest-impact change because it fundamentally transforms the game from a geometry problem into an information management problem. The evader now has a new resource to manage (opponent's knowledge of its position), which naturally creates strategies beyond obstacle-hugging.

Combine with **Approach 2 (randomize obstacles)** for additional robustness.

If partial observability alone is insufficient, add **Approach 3 (adaptive entropy)** next, then consider **Approach 4 (latent z)** for explicit strategy diversity.
